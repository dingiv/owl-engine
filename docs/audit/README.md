# owl 审计总清单(2026-09-23 三域合并;状态:激进修复第一轮完成)

> 图例:✅ 已修复并测试验证 | ◐ 部分(探测器/口径就绪,默认关) | ⏸ 立案挂起(阻塞或需架构决策)

## P0

- [x] P0-3 NULL 流 memcpy 系统性同步缺口 —— **已修**:`CudaDevice::memcpy_{dtoh,htod}_{f32,u32}` 流序收口 API(b77c67b),非测试调用点 24 处迁移;3 处测试 helper 在 ctx.synchronize() 后(判安全保留);device.rs/lib.rs 内部 H2D 为新块首写(安全);detector:dtoh 前置 sync 已统一
- [x] P0-1 捕获闭包 panic = 捕获流悬死 —— **已修**:catch_unwind(AssertUnwindSafe) 包裹 f(&frame),panic/Err 双路径 end_capture + destroy + 结构化报错
- [x] P0-2 哨兵③口径 —— **已修(口径修正)**:拒绝的是 `leaked`(池内未租约,真违约);`suspicious`(像地址的标量,首跑实锤 0x3f800000=1.0f)保持告警——kernel 参数含立即数,无法盲区分;allowlist 管道就绪
- [x] P0-4 freed 块哨兵 —— **已接线(2026-09-24)**:`OWL_POISON_FREED=1`(或 `set_poison_freed`)归还即填 0xFF;零 unwrap panic(drop 上下文 eprintln);owl-cuda 两个 P1-2 回归测试双向验证(新代码绿/旧路径红)
- [x] P0-5 device_ptr 暴露面 —— **部分**:cat/stack 等算术点已显式 f32 化;全仓 `.add(` 扫描仅剩字节意图/已转换点;`PtrOf` lifetime 视图 ⏸ 随 M-Ⅲ

## P1

- [x] P1-2 捕获窗口 scratch 悬空 —— **已修(2026-09-24,双路径钉住)**:①窗口内出生 = 出生即强持(birth-pin,防 Arc 提前归零);②pre-window 出生、窗口内死亡 = 释放闭包停放(park),图 drop 才执行;定影时收编进 DeviceGraph keepalive(令牌同步入租约表,哨兵③对账覆盖);CaptureSession 新增 lease_scratch(与 Persistent.lease 对称);残留另案 → P1-7
- [x] P1-1 捕获失败清租约(已并入 P0-1 修复)
- [x] P1-4 租约校验去 debug-only(全构建生效,纯账本查询)
- [ ] P1-3 相位→Idle drain 验证主流排空 ⏸
- [ ] P1-7 warmup 遗留毒化×地址复用(2026-09-24 立案,毒化对照实验实锤):OWL_POISON_FREED=1 下 dry_run 判别③仍红,memset 改 no-op 即绿——毒化写入本身击中图读缓冲;但 backtrace 证明窗口内死亡已全部被 P1-2 双路径钉住,实际毒化仅发生于①捕获前 warmup 遗留延迟释放的 drain、②测试 unwind;嫌疑 = warmup(Live 相)瞬态 drain 释放后地址被捕获期分配复用,复用块初始内容(alloc_zeros 清零)与图"读先于写"节点所依赖的 eager 内容不一致。侦查手段:复用块初始内容断言 / 定位具体复用块。同日顺手修:Governor↔CudaPool 强-强引用环拆除(强注册表移 DeviceInner)、Ledger 单锁原子快照 + A5.3 mem_get_info 降频(仅逼近预算 10% 内探)
- [ ] P1-5 cublas 逐调用 stream 传递审计 ⏸
- [ ] P1-6 Session 槽位隐式持久语义(漏写槽 = 陈旧数据)⏸ 文档化

## P2

- [x] VMM drop 失败结构化日志(pool.rs 顺手修)
- [ ] P2 其余(Slice 记账粒度/Arch TODO/AUTO_FREE_ON_LAUNCH/TrackingHandle !Send/signal TLS)⏸ 挂账

## 架构裁决(待拍板)

1. `device_ptr()` 类型化收口(PtrOf)——⏸ 随 M-Ⅲ
2. ~~graphplan 退役~~ —— ✅ 已执行(删 graphplan.rs/DecodeGraphAdapter/DecodeGraphRunner;RunnerType/MessageType/多节点 IPC 保留;speculative/server 为死桩未受影响;Session 为唯一图机制)
3. `cuMemcpy` 直接调用 lint 禁令 —— ⏸(迁移已完成,防回归靠 review)
4. OwlTensor 拆 TLS —— ⏸ 与 signal scope 重构捆绑

## 实锤案例归因(全部闭合)

| 悬案 | 根因 | 状态 |
|---|---|---|
| safetensors 偶发零读 | NULL 流竞速 | ✅ memx |
| rotary 零读 | 同上 | ✅ memx |
| cu_seqlens 清零 | 同上 | ✅ memx |
| cat 脏数据 | *mut u8 .add 元素语义 | ✅ f32 化 |
| GDN slab 悬空 | 视图逃出租约 | ✅ slab |
| **P1-2 捕获窗口悬空** | 租约缺口 | 🔴 **立案(探测证据 dry_run)** |

## M-Ⅱ 进展补记(2026-09-23 深夜,memx 修复后 m4 复测)

- **层 0-3 对 HF 精确(相对误差 6.3e-7)**——memx 流序修复为主因实锤
  (旧案:a/b 输入被 NULL 流竞速污染 → 全链 NaN/幅值崩塌);
- 首发散散层 = **4(GDN),相对误差 7.2e-2**(修复前层 1 误差 1.0e0);
- 定谳无罪(全部数值证据):GDN 子段(z/gated 逐位 0)、out_proj 对 numpy
  1.07e-6(outproj_probe.rs)、embedding 三组 id 决定性实验、RMSNorm
  ×(1+weight) 语义(HF modeling_qwen3_5.rs:854;曾误判误植,已回滚并
  修正探针参考)。
- **当前唯一立案:层 4(GDN)7.2e-2**。侦查方向:dbg_dump(feature=never,
  在 m2-endgame-wip 已验证可用)对比层 4 conv/g/beta/递推 vs HF torch
  参考(fla 未装走纯 torch,可 /tmp/hfref 直算);疑点 = conv_state 初值、
  per-seq 分段、f32 累积序(7% 或为良性,需 M-Ⅳ 容差裁决)。

## M-Ⅱ 二次补记(use_norm_offset 恢复 + 池 H2D 流序修复后)

- 池重构引入的回归已修:CudaPool::htod_persistent_in 的 H2D 改池流 async+sync
  (NULL 流无序,审计 P0-3 同族,重构时重新引入过一次);
- 当前状态:**幅值增长模式与 HF 一致**(owl 0.30→2.43 vs HF 0.61→5.56),
  全链零 NaN;残余 = 逐层数值分歧(层 0 GDN 起,rel ~1.4);
- 侦查工具已备:dbg_dump(feature=never)、/tmp/hf_dump.npz(HF 逐层真值)、
  /tmp/hf_probe.py(HF 子模块挂钩样例)、/tmp/hfref venv;
- 下一刀:HF 源码 qwen3_5.py GatedDeltaNet.forward 逐行对照
  (chunk_gated_delta_rule 纯 torch 参考在 fla 未装时可用),
  对层 0 GDN 的 qkv 分切/l2norm/g 折叠/delta 方向做逐子段对拍。
