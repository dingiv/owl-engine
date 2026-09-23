# 审查报告:显存管理后端(crates/backends/cuda + iface)

> 审查人:worker(显存管理域)。2026-09-24。
> 范围:pool.rs / device.rs / governor.rs / buffers.rs / audit.rs / lib.rs / ffi.rs / iface(lib+signal)。
> 方法:全文通读 + 与已实锤四案(cat 错位 / cu_seqlens 清零 / safetensors 偶发零读 / rotary 零读)交叉验证。

## 核心结论(先读这个)

后端的账本/相位/租约三层设计是健全的,真正的问题集中在**一个系统性架构缺陷**:

> **P0-内存-1:NULL 流 memcpy vs non-blocking 主流 = 全仓系统性同步缺口。**
> 设备主流是显式 non-blocking 流(T4 判例:legacy NULL 流不可捕获,device.rs:44),
> 但全仓 **31 处** H2D/D2H 用的是 `cuMemcpyHtoD_v2/DtoH_v2`(legacy NULL 流语义,
> 非 Async 变体)。CUDA 规范:non-blocking 流**不与 NULL 流同步**。
> 因此:任何 NULL 流 memcpy 与主流上的 kernel 之间**没有任何顺序保证**。
>
> 已实锤四案全部由此解释:
> - safetensors 装载偶发整块零读(DtoH 先于主流装载 kernel 完成)
> - rotary_truth 零读(同型,已用 synchronize 打补丁)
> - cu_seqlens 清零案(H2D 可见性 vs 主流 kernel 写竞速)
> - cat 案的"脏数据"面(scratch 复用块的旧值可晚于预期可见)
>
> 现存补丁(读前 synchronize)是逐点止血;正确修法是**一次性收口**:
> 在 Device 上提供 `memcpy_h2d/d2h(stream)`(cuMemcpyAsync + 主流)并迁移全部
> 31 处;或最低成本——Device 提供 `sync()` 语义审计 + 全仓排查 NULL 流 memcpy。
>
> 附:31 处分布 = attention.rs 9 / graphplan.rs 4 / layers/mod.rs 4 / device.rs 3 /
> session.rs 2 / rig.rs 2 / 其余各 1。

## 危险清单

### P0(会静默产出错误数值/崩溃,已实锤或机制等同)

| # | 位置 | 违反契约 | 触发条件 | 建议修法 |
|---|---|---|---|---|
| P0-1 | 31 处 NULL 流 memcpy(分布见上) | 流顺序语义:non-blocking 流与 NULL 流互不同步 | 主流 kernel 在飞 + NULL 流 memcpy | 统一收口为 stream-aware memcpy(Async+主流)或封装 `Device::copy_h2d/d2h` |
| P0-2 | pool.rs:151 `alloc_zeros` / 释放路径 vs 消费方裸指针 | 租约纪律:归还后地址失权——但 raw ptr 逃逸无强制点 | 任何 raw ptr/视图存活超过句柄(cat 案、slab 案、cu_seqlens 案的全部根因) | debug 构建下 freed 块填 NaN 哨兵(fill-before-free),让悬空读立刻 NaN 化;中期:指针访问收口到带 lifetime 的 `TensorRef` |
| P0-3 | pool.rs:168-172 `base` 登记时机 | 哨兵①区间索引一致性 | Slice 的 base 在创建时取一次;cudarc 底层若重映射/重分配(理论),索引失效 | 记录为已知约束;审计区间查询加 gen 校验 |

### P1(高概率在并发/边界场景出问题)

| # | 位置 | 问题 | 建议修法 |
|---|---|---|---|
| P1-1 | governor.rs:57-70 + pool.rs:407-424 | 相位翻转(→Idle)drain 延迟释放队列时,**不验证主流已排空**(eager kernel 可能仍在飞);seal 路径若未先 synchronize 即 set_phase(Idle),延迟回收可回收在写缓冲 | set_phase(Idle) 前置 `ctx.synchronize()`(或 drain 前) |
| P1-2 | pool.rs:452-457 `Drop for PoolBufInner` | 多线程下 replay(Weak upgrade → device_ptr)与 drop(Empty 化)竞态:`unreachable!("PoolBuf 已被消费")` 可成真 panic | Empty 分支改结构化报错;或 upgrade 后 validate backing |
| P1-3 | buffers.rs:159-163 `unsafe impl Send for RemoteBuf<T>` | P2P 远端指针的 Send 担保成立前提 = 对端映射存活且本端不 free;目前**无文档、无映射存活跟踪**(M1 P2P 接线时消费,风险后置) | 接线时补:映射来源 Arc 保活 + 文档成文 |
| P1-4 | cublas.rs:106-112 | cublasSetStream 在构造时设置一次;若捕获期走会话流,必须逐调用 set_stream(130 行有 API)——审查所有 cublas 调用点的 stream 参数传递是否一致 | 调用点审计(归算子域交叉验证) |

### P2(正确性小账/卫生)

| # | 位置 | 问题 | 建议修法 |
|---|---|---|---|
| P2-1 | pool.rs:140-146 vs cudarc 内部 | Slice 路径按 bytes 记账,cudarc 内部按分配粒度对齐——账本 vs driver 真实占用漂移(仅影响预算精度) | 记账同乘对齐粒度或注明偏差 |
| P2-2 | device.rs:357 `Arch::Sm86` 硬编码 | TODO(M1) 运行时读取未做 | cuDeviceGetAttribute 到位后回填 |
| P2-3 | pool.rs:365-376 VMM drop 三步失败仅 eprintln | 失败即静默泄漏;至少应计数到 stats | 泄漏计数 + 阈值告警 |

## 安全项(审查过,无需改动,注明理由)

1. **VMM 分配/释放三步链**(pool.rs:244-330):reserve→create→map→setAccess 每步失败路径都完整回滚前序资源;drop 侧 unmap→release→address_free + `_ctx` 保活 + P1-A bind + P1-B VMM 前置 synchronize——**这是全仓质量最高的一段**。
2. **Arc 租约系统**(CudaPoolBuf):Clone=租约、retire 落 Arc 归零点(P1-C 判例注释)、live_bufs 持 Weak 防账本泄漏(2026-09-22 判例)——设计正确且有判例沉淀。
3. **单主流串行纪律**:设备主显存流为显式 non-blocking,池分配/释放/重放/内核发射全部同流(eager 下)——同流顺序保证成立;捕获流为独立流且租约保活覆盖(设计上)。
4. **双账本 + 预算三道闸**(governor charge:校验先行、失败不触账、driver 对账警线)——2026-09-22 双重扣减判例已消化。
5. **UUID 钉卡**(device.rs:248-300):数字序禁用替代完整。

## 设计问题(架构级,需用户裁决)

1. **raw pointer 暴露面过宽**:`device_ptr()` 在 iface trait 上是一等公民,全仓 119 处 unsafe 的 ~70% 都在消费它。契约注释("只许传地址")是君子协定。方向:iface 提供 `PtrOf<'a, T>`(借用 lifetime 绑定句柄)或至少 debug_assert 存活性;否则同类 bug(cat/slab/cu_seqlens)会持续再生。
2. **NULL 流 memcpy 是历史包袱**(candle/cudarc 迁移习惯):收口后建议 lint 级禁止直接调 `cuMemcpy{HtoD,DtoH}_v2`。
3. **相位机与流生命的耦合**:捕获流(capture_session)每会话新建、池流常驻——deferred 队列跨相位的净空语义依赖"Idle 时用户 kernel 已完成"这一隐式前提(P1-1),建议把"sync before Idle"写成机制而非约定。

## 复核证据(命令与原文)

- unsafe 计数:119(不含 tests/);`unsafe impl` 6 处(Send/Sync 担保面)
- legacy NULL 流 memcpy:31 处/13 文件(grep cuMemcpyHtoD_v2|cuMemcpyDtoH_v2,不含 tests)
- 已实锤四案与本报告 P0-1/P0-2 的映射:见"核心结论"

git status(worktree,审查产出外零改动):
```
?? docs/audit/memory-backend.md
```
