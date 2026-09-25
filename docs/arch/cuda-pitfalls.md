# CUDA / Graph 坑点总账 —— 判例 × 新实现状态

> 2026-09-25 汇总。**本文件已合并并取代**(原文件已删):
> `docs/arch/cuda-death-rules.md`(十姿势判例)、
> `roadmap.local/{cuda-graph-session, session-graph, memory-backend, README}.md`
> (旧世界四份审查报告)。对账对象 = **新声明式 gpu_server**
> (`crates/backends/cuda/src/{server,state,command,launch,gpu_client}.rs`,
> 2026-09-24/25 重写:三固定流 + 命令信封 + host 回调完成通知 + 图捕获护栏)。
>
> 状态图例:✅ 结构性消除(该坑在新架构下不存在) | ✅* 已修(等价机制)
> | ⚠️ 已登记待办(存在残余风险,有缓解) | ⏳ 未覆盖(将来立项)
> | N/A 旧世界专属(机制本身已退役)

## 一、旧架构 vs 新架构:为什么坑点大面积消失

旧世界(CaptureSession/DeviceGraph/Governor/租约/审计/池相位)的核心难题是
**"自动内存管理 × 图冻结地址"的对抗**——租约、哨兵③、POISON_FREED、
birth-pin、park 全是为它发明的。新实现釜底抽薪:

| 新机制 | 消灭的问题族 |
|---|---|
| **块只增不减**(账房无 free/trim/归还 API) | 姿势 2/3(UAF/trim)整族 |
| **捕获 slab 预分配 + 切块**(捕获前 cudaMalloc,窗内零分配) | 捕获期 malloc 非法 / 图内存预算 |
| **dispatch 白名单**(捕获期仅 Launch/Alloc/GraphEnd) | 捕获期隐式同步/回读/其他流任务 |
| **三固定流 + 内部路由**(无默认流、无动态建流) | NULL 流同步缺口 / 跨流捕获边界 |
| **event-tracking 启动即关** | CAPTURE_ISOLATION / 图内事件节点污染 |
| **host 回调完成通知**(cuLaunchHostFunc,派发线程) | 完成语义 guessing / reaper 轮询 |
| **无闭包捕获窗**(捕获 = 客户端多命令序列) | 捕获闭包 panic 悬死(旧 P0-1) |

## 二、十姿势逐条对账(cuda-death-rules.md)

| 姿势 | 状态 | 说明 |
|---|---|---|
| 1 捕获期隐式同步/回读 | ✅ | 捕获期 Dtoh/Sync/Htod 被 dispatch 拒绝;三流全 non-blocking(ctx.new_stream) |
| 2 replay 悬空地址(UAF/Xid 31) | ✅ | 账房无释放 API;slab Arc 随 Carved 块保活;图引用块永不回收 |
| 3 图存活期 free/trim | ✅→⚠️ | 当前无任何 trim/empty_cache 路径(结构性);**将来加块回收时必须立 trim 纪律**(A1.2)——登记为将来立项的强制护栏 |
| 4 库懒分配进图(静默数值损坏) | ✅ 当前 | 无 cublas/cudnn/第三方句柄,kernels 全自写 nvrtc;A5.2 准入铁律(death-rules)对将来引入的库继续生效 ⏳ |
| 5 图内存无预算 | ✅* | slab 固定 64MiB 图前分配、耗尽 = 结构化报错;全局显存规划器(A1.1 allowance 进池计算)暂无 → 立项时带 ⏳ |
| 6 缺 warmup | ⚠️ | warmup 契约写入 DeviceClient trait 文档,graph_capture 示例演示;server 无法强制(客户端侧纪律) |
| 7 多图/多池互踩 | ✅ | 单 slab 单池、无池切换;AUTO_FREE_ON_LAUNCH 只作用图内分配节点(我们无),slab 图外安全(旧 P2-3 结案) |
| 8 形状/档位热更未走 ExecUpdate | ⏳ | 未实现 ExecUpdate;形状变 = 重新捕获。档位预捕获/ExecUpdate 随 runner 立项 |
| 9 跨流捕获边界未 join | ✅ | 捕获恒单流(COMPUTE),其他流命令捕获期全拒(姿势 9 防线"单流先行"落地) |
| 10 capture 期 host 暂存回收 → DMA 悬空 | ✅ | 捕获期 Htod 拒绝;htod 码头随 finish 回调释放(host 回调 = 搬运完成点) |

## 三、roadmap.local 审查发现逐条对账

### cuda-graph-session.md(危险清单)

| 项 | 状态 | 说明 |
|---|---|---|
| P0-1 捕获闭包 panic 悬死 | ✅ | 新实现无捕获闭包;actor 与客户端生命周期解耦(disconnect/Close 收摊) |
| P0-2 哨兵③ suspicious 只告警 | N/A | 无 lease/audit 系统;等价保障 = 参数槽全部由账房解析(无"未知指针入图"通道) |
| P1-1 捕获失败租约污染 | N/A | 无租约 |
| P1-2 scratch 悬空窗口 | ✅ | 块只增不减 + slab Arc 保活 |
| P1-3 租约校验 debug-only | N/A | 无租约 |
| P1-4 Session 槽位持久语义 | N/A | 无 Session 层;**将来 runner 层立项时必须带此教训**(漏写槽 = 陈旧数据) |
| P1-5 双图体系并存 | ✅ | graphplan 已删(架构裁决);新实现单一图注册表 |
| P2-1 TrackingHandle 跨线程 drop | ⏳ | signal 在 iface 未启用;A2.6 多线程化前加 `!Send`(旧修法有效) |
| P2-2 signal TLS 线程退出容忍 | ⏳ | 同上,signal scope 重构捆绑 |
| P2-3 AUTO_FREE_ON_LAUNCH 交互 | ✅ | 只作用图内分配节点;slab 图外预分配(ffi 注释钉明) |
| P2-4 H2D 同步 memcpy 序依赖 | ✅ | htod 全 async + pinned + host 回调完成;无同步 memcpy 面 |

### session-graph.md(S1-S6)

| 项 | 状态 |
|---|---|
| S1 租约校验 debug-only | N/A(无租约) |
| S2 捕获 panic 悬死 | ✅(无闭包) |
| S3 DeviceGraph Drop 未与流同步 | ⚠️ 图无显式销毁原语(graphs 注册表只增),Drop 发生在 server 收摊(此前 sync 栅栏已跑)→ 当前安全;**将来加 graph 销毁命令时须先 stream synchronize** |
| S4 graphplan 双轨 | ✅ 已删 |
| S5 槽位 dtype 定死 | ⏳ runner 层立项带 |
| S6 signal TLS | ⏳ 同 P2-2 |

### memory-backend.md

| 项 | 状态 |
|---|---|
| P0-1 NULL 流 memcpy 系统性缺口(31 处,四实锤案) | ✅ **彻底收口**:新实现全部 `memcpy_*_async` + 三 non-blocking 流 + pinned 码头;旧 31 处随旧世界作废 |
| P0-2 freed 块裸指针逃逸(cat/slab/cu_seqlens 案根因) | ✅ 无 free、无裸指针出口(dtoh 走账房 ptr 即用即弃) |
| P0-3 Slice base 登记时机 | ✅ carve 记录块内偏移,指针即用即算,无跨时点登记 |
| P1-1 相位→Idle drain 不验排空 | ✅ 无相位机 |
| P1-2 PoolBuf drop 竞态 | ✅ 无 PoolBuf |
| P1-3 RemoteBuf P2P Send 担保 | ⏳ P2P 未接;接线时补映射保活 + 文档 |
| P1-4 cublas 逐调用 set_stream | ✅ 当前无 cublas;引入时按 A5.2 审 + 逐调用审计 ⏳ |
| P1-6 槽位持久语义 | N/A(同 P1-4) |

### README.md 审计总清单(跨域)

| 项 | 状态 |
|---|---|
| P0-3 NULL 流 memcpy | ✅(同上,新世界无此面) |
| P0-4 POISON_FREED | N/A(无 free) |
| P0-5 device_ptr 暴露面 / cat 案 f32 化 | ✅ 新实现唯二 ptr 出口 = block_ptr(账房内)/ launch 槽装配(即用即弃);无算术裸指针 |
| P1-7 warmup 遗留毒化 × 地址复用(旧世界立案 🔴) | ⚠️ **新世界有同族面**:carve slab 块不清零,复用块初始内容 = 旧数据。当前安全前提 = "输出块被 kernel 写满"(我们的 kernel 契约);**将来引入部分写的 kernel 时必须:carve 清零(memset async 捕获安全)或块读前写断言** ⚠️ 登记 |
| graphplan 退役裁决 | ✅ 已执行 |

### death-rules 延伸议题

| 议题 | 状态 |
|---|---|
| host pinned 内存账本(attention-rs workspace 教训) | ⚠️ Staging malloc_host 无账本;当前量小(每搬运一码头、用完即还)。搬运高频化(M2)前立 host 账本 ⏳ |
| "库内注释自曝不兼容图" grep 法 | ✅ 制度保留(death-rules 准入清单) |

## 四、新实现自身的新增待办(对账过程中发现)

| # | 项 | 风险 | 触发条件 |
|---|---|---|---|
| N1 | **捕获 slab / 图注册表无回收**:carve 块与图只增不减;反复捕获(多档位)会 slab 耗尽(现 = 结构化报错,不炸) | 显存常驻增长 | 多档位/多次捕获的 runner 立项时 |
| N2 | **图销毁原语缺失**:无 GraphDestroy 命令(旧 S3 同款) | 同上 | 同 N1,一并立项:销毁前 stream synchronize + graphs.remove |
| N3 | **sticky 错误收割点不全**:launch fire-and-forget,错误在下次 sync/dtoh 暴露——sync/dtoh 已覆盖;但纯 launch 序列 + 长期不 sync 的客户端会攒错 | 客户端永不 sync → 错误无限延迟 | 契约已文档化;runner 必有周期 sync,风险低 |
| N4 | carve 块不清零(同 P1-7 ⚠️) | 见上 | 部分写 kernel 引入时 |
| N5 | 事件边(跨流依赖)缺失 | H2D/COMPUTE/D2H 重叠目前靠客户端 sync 粗栅栏,重叠收益打折 | 流水线重叠立项 |
| N6 | graph 未做 upload() 预上载(首 launch 有 setup 开销) | 首次 replay 抖动 | graph_launch 加 upload 可选旗标,小改 |

## 五、结论

十姿势中 **7 个结构性消除、2 个已修等价、1 个(warmup)契约文档化**;
旧世界四份审查的全部 P0/P1(除跨流事件边、P2P、signal 三项挂账)在新架构下
要么不存在、要么有等价机制。残余待办集中在:

1. **资源回收三件套**(N1/N2/S3):slab/图/块的销毁原语 + trim 纪律 —— runner
   多档位捕获立项时第一优先;
2. **carve 清零 or 读前写断言**(N4/P1-7 同族)—— 部分写 kernel 引入前必须;
3. **事件边**(N5)—— 流水线重叠立项;
4. 跨域挂账( signal !Send、P2P RemoteBuf、host 账本、A5.2 准入)随各自域立项。

**使用规则(承 death-rules)**:graph 相关新 bug 先对号本文件第二/三节;
新坑新开条目并带卷宗链接;每条 ✅ 的机制如有改动,本文件同步降级重审。
