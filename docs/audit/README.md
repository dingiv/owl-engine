# owl 显存/算子/图机制审查总清单(2026-09-24)

> 审查范围:packages/owl 全仓(worktree @ audit/memory-review 分支)。
> 三份域报告:memory-backend.md / operator-design.md / cuda-graph-session.md / session-graph.md。
> 校准案例:cat `*mut u8` 错位、varlen scratch 悬空、cu_seqlens 清零、
> safetensors/rotary 偶发零读、Gemma norm offset。
> unsafe 存量:119 处(不含 tests);`unsafe impl Send/Sync` 6 处。

## 系统性根因(一个缺陷解释了四个悬案)

**NULL 流 memcpy vs non-blocking 主流**:全仓 31 处 H2D/D2H 走
`cuMemcpy{HtoD,DtoH}_v2`(legacy NULL 流语义),而设备主流是显式 non-blocking 流。
CUDA 规范:non-blocking 流与 NULL 流**互不同步**。所以任何"同步读回/写入"
与主流 kernel 之间没有顺序保证——safetensors 偶发零读、rotary 零读、
cu_seqlens 清零、cat 脏数据四个悬案全部由此解释。逐点 synchronize 是止血,
**收口 = Device 提供 stream-aware memcpy 并迁移 31 处**。

## 修复总清单(建议波次)

### 第一波|止血(机械修复,预计 1-2 天)

| # | 级 | 项 | 位置 |
|---|---|---|---|
| 1 | P0 | stream-aware memcpy 封装 + 31 处迁移(根除悬案土壤) | 全仓 |
| 2 | P0 | 哨兵③ suspicious 非空 = 拒绝 instantiate(现仅 WARN) | graph.rs ~183 |
| 3 | P0 | 捕获闭包 panic 兜底(catch_unwind 或 CaptureGuard 补 end_capture) | graph.rs ~145 |
| 4 | P0 | replay 租约校验移出 debug_assertions(release 保留表查询) | graph.rs:255 |
| 5 | P1 | 捕获失败路径清空 leases/keepalive(会话复用污染) | graph.rs Err 分支 |
| 6 | P1 | cat 入口 dtype guard(非 f32 结构化报错) | erased.rs:445 |
| 7 | P2 | OwlTensor::reshape 残留 backtrace 清理 + n≠have bail + contiguous 断言 | layers/mod.rs:283 |

### 第二波|机制收口(预计 2-3 天)

| # | 级 | 项 | 位置 |
|---|---|---|---|
| 8 | P0 | **freed 块 NaN 哨兵填树**(debug 构建 fill-before-free,悬空读立刻 NaN 化——cat/slab/cu_seqlens 类 bug 的通用探测器) | pool.rs 归还路径 |
| 9 | P1 | set_phase(Idle) 前置 synchronize(drain 语义写进机制而非约定) | governor.rs:57-70 |
| 10 | P1 | DeviceGraph::Drop 前 stream synchronize(防在飞 replay) | graph.rs:274 |
| 11 | P1 | PoolBufInner drop 竞态:unreachable! 改结构化报错 | pool.rs:452 |
| 12 | P1 | cublas workspace 显式 SetWorkspace(P 阶段池内;图捕获期 cublas 自建分配 = 图外地址) | cublas.rs |
| 13 | P1 | u32 idx 乘 inner 溢出审计(gather/scatter 族;长上下文 KV 前) | erased + .cu |
| 14 | P1 | attention.rs 硬编码 4(元素宽)随 fp8/bf16 KV 化时收敛 | attention.rs:309 |

### 第三波|架构收口(需裁决)

| # | 级 | 项 |
|---|---|---|
| 15 | P0(架构) | **device_ptr() 类型化收口**:iface 提供 `PtrOf<'a,T>`(lifetime 绑句柄)或 debug_assert 存活;契约从"君子协定注释"变机制。119 处 unsafe 的 ~70% 消费面,根治 cat/slab/cu_seqlens 再生 |
| 16 | P2 | graphplan 退役 + Raw 视图变体处置(双轨收敛,挂账 §五·5) |
| 17 | P2 | `cuMemcpy*_v2` 直接调用 lint 禁令(收口后防回流) |
| 18 | P2 | OwlTensor 显式 ctx 化(拆 ctx_scope TLS,与 signal-scope 重构捆排) |
| 19 | P2 | RemoteBuf Send 担保文档化(P2P 接线前置条件) |

## 审查通过(无需改动,留档理由)

- VMM 三步分配/释放链(全仓质量最高段,失败路径完整回滚);
- Arc 租约系统(Clone=租约、retire 判例、live_bufs Weak 防泄漏);
- downcast dtype 检查纪律(无静默位型重解释);
- 双账本 + 预算三道闸(9-22 双重扣减判例已消化);
- UUID 钉卡;signal emit 覆盖面;testkit rig 防悬空模式。

## 架构裁决需求(用户拍板项)

1. 第三波 #15:PtrOf 类型化(改面大、根治)vs debug_assert(轻量、防呆)——建议先后者,PtrOf 随 M-Ⅲ runner 接线一起立项;
2. #8 NaN 哨兵填树是否进 release(建议仅 debug;release 用页保护成本高);
3. unsafe 总量目标:收口后预期 119 → ~40(全部集中在 ffi/发射器边界,业务层零 unsafe)。
