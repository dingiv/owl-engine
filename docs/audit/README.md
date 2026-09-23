# owl 审计总清单(2026-09-23 三域合并;状态:激进修复第一轮完成)

> 图例:✅ 已修复并测试验证 | ◐ 部分(探测器/口径就绪,默认关) | ⏸ 立案挂起(阻塞或需架构决策)

## P0

- [x] P0-3 NULL 流 memcpy 系统性同步缺口 —— **已修**:`CudaDevice::memcpy_{dtoh,htod}_{f32,u32}` 流序收口 API(b77c67b),非测试调用点 24 处迁移;3 处测试 helper 在 ctx.synchronize() 后(判安全保留);device.rs/lib.rs 内部 H2D 为新块首写(安全);detector:dtoh 前置 sync 已统一
- [x] P0-1 捕获闭包 panic = 捕获流悬死 —— **已修**:catch_unwind(AssertUnwindSafe) 包裹 f(&frame),panic/Err 双路径 end_capture + destroy + 结构化报错
- [x] P0-2 哨兵③口径 —— **已修(口径修正)**:拒绝的是 `leaked`(池内未租约,真违约);`suspicious`(像地址的标量,首跑实锤 0x3f800000=1.0f)保持告警——kernel 参数含立即数,无法盲区分;allowlist 管道就绪
- [x] P0-4 freed 块哨兵 —— **◐ 探测器就绪,默认关**:`OWL_POISON_FREED=1`(debug)归还即填 0xFF;开启即复现 P1-2(见下),修后应转默认开
- [x] P0-5 device_ptr 暴露面 —— **部分**:cat/stack 等算术点已显式 f32 化;全仓 `.add(` 扫描仅剩字节意图/已转换点;`PtrOf` lifetime 视图 ⏸ 随 M-Ⅲ

## P1

- [ ] P1-2 捕获窗口 scratch 悬空(立案,数字证据):**OWL_POISON_FREED=1 时 dry_run 判别③ 红**——捕获后归还的 scratch 块被 0xFF 覆盖,图内存在"读先于写"的引用;修法方向 = 捕获期分配的块 pin 到 DeviceGraph drop;修复后 P0-4 转默认开
- [x] P1-1 捕获失败清租约(已并入 P0-1 修复)
- [x] P1-4 租约校验去 debug-only(全构建生效,纯账本查询)
- [ ] P1-3 相位→Idle drain 验证主流排空 ⏸
- [ ] P1-5 cublas 逐调用 stream 传递审计 ⏸
- [ ] P1-6 Session 槽位隐式持久语义(漏写槽 = 陈旧数据)⏸ 文档化

## P2

- [x] VMM drop 失败结构化日志(pool.rs 顺手修)
- [ ] P2 其余(Slice 记账粒度/Arch TODO/AUTO_FREE_ON_LAUNCH/TrackingHandle !Send/signal TLS)⏸ 挂账

## 架构裁决(待拍板)

1. `device_ptr()` 类型化收口(PtrOf)——⏸ 随 M-Ⅲ
2. graphplan 退役 + runner 迁 Session —— ⏸ 双轨风险已记录
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
