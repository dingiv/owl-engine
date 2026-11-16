# owl-store(状态存储)模块约束

> **来源**:`docs/devlog/memory/store.md`(立项卷,唯一权威;本文件只做落地约束的浓缩)。
> **code-master 标记**:本 crate 的约束写在这里与各文件头;行内偏离用 `// CMA:`。

## 这个 crate 是什么

引擎**状态(KV / 线性注意力状态 / 快照)的三层存储与召回**:GPU 池 → host pinned → 磁盘。
它回答三个问题:**东西在哪(repository view)**、**这一步该看到什么(selection)**、
**放不下时丢谁(eviction)** —— 不做前向计算、不认识模型层。

## 结构(裁决②:两腿独立 + 一层关系)

```
Rel(关系层门面)  身份映射 · 召回语义(Exact / ContextOnly)· 换出单合并 · 报告口径
├─ KvStore       attention 块腿 = 槽池(SlotPool)+ 内容身份前缀树(PrefixTree)
│                (迁自 BlockManager/PrefixCache 的 host 账;设备叶子仍归 engine)
└─ StateStore    GDN 快照与行账(snap 身份 = 块边界内容身份)
公共: tier(三层账/预算) · policy(驱逐与准入策略对象) · host(宿主承载) · moves(搬运自由函数)
```

**已立**(S3-3,§10-§19):
```
BlockLeg(engine 侧唯一门面)= KvStore(身份+槽账)+ HostTier(宿主物理)+ 搬运口
                              ↑ 布局由 engine 注册(id 级 + epoch);server 执行搬运(swap 单命令)
Rel   = 关系层门面(S3b 已接线):双腿关联 / 准入 / 搬运计划;组合件字段私有,
        engine 只调语义方法(slots_free / state_locate / kv_disk_* / plan_* / release_*)
```

## 谁在用(2026-10-17,S3-3 全刀之后;交互契约全文见设计卷 §10、钩子 §14、需求驱动 §17)

`owl-engine` 的 `SessionBank` 持 target 链 `Rel` + MTP 块腿(`BlockLeg`);`KvStore` 是 `Rel` 内部件。
engine 侧原来的 `blocks.rs` / `prefix_cache.rs` **已整文件删除**(§9.5)。

已经通的:

- **换出/回灌**(§16/§17.7):链级**整条**下沉(链头段也进宿主 ⇒ 命中面从链头就是宿主格)
  + `onload_blocks` **一条命令搬 N 块**;搬运走 `DeviceClient::swap`(CUDA 单命令 / CPU 池化解码);
- **需求驱动保洁**(§17):engine 钩子 H1/H2 用与 `schedule` **同源算式**算 need;
  `ensure_for_len` 的同步驱逐降为兜底(报告面 `ops.ensure_evicted` 监控);
- **无卡验证**:`tests/store_e2e.rs`(端到端三门口 + 链级粒度,0.00s)
  与 engine 侧 `cpu_mock_smoke`(boot/装载/warmup/图捕获/turn)。

旧口径更正(2026-11-16):

- 盘层**已接线**(S6:`attach_disk` / `adopt_disk_*` / spill / fill),不再是"未接线";
- 历史挂账清单集中在本包 `docs/devlog/memory/store.md` **§20**。

## 律(违反即 `// CMA:` 标记)

1. **分层单向**:本 crate 只依赖 `owl-iface`(契约面)+ `owl-shared`(工具面)+
   **`libc`(仅内存布局提示:宿主块 `madvise(MADV_HUGEPAGE)`,2026-10-19 记注于
   `crates/store/Cargo.toml`)**;
   **禁止**依赖 `owl-cuda` / `owl-engine` / `owl-models` 内部(engine → store 单向)。
   补注(**测试面例外**):dev-dependency 允许 `owl-cpu` —— 它是契约面
   (`owl-iface::DeviceClient`)的一个真实现体,用真 face 测搬运闭环胜过手写假 face;
   `owl-cuda` / `owl-engine` 连测试面也不许拉(那会造成环)。
2. **账与叶子分家**:store 持**身份 + 账 + 宿主物理**;**设备叶子句柄的所有权归 engine**
   (store 只经 `KvLeaves` 口引用,不持有 `Bytes` 所有权 —— 避免双主人)。
3. **搬运只在段边界**:换入/换出只能发生在 piecewise 段接缝 / pump 间隙 / prefill 分块之间
   (捕获窗禁 htod/dtoh);本 crate 的接口**不接受**"随时搬"的调用形状。
4. **保留额不可驱逐**:活跃 turn 的 `dense_floor` 是保留面;驱逐只能动
   ①idle 会话缓存/快照 ②活跃 turn 的弹性余量。
5. **`Strict` 不许静默降级**:零耐受 turn 在显存不足时**响亮拒绝**(或按策略排队),
   绝不允许改走近似召回。
6. **判定口禁副作用**:`probe` 一类的查询必须零分配零设备动作
   (对全量 Want 逐个问的声扫同理 —— 见坑册 §40)。
7. **I/O 走顶层自由函数**:本 crate 的设备/宿主/磁盘 I/O 一律写成自由函数(显式传 `&mut D`),
   结构体方法保持自肃(只改自身字段);不设全局态、不设线程局部态。
8. **身份律**:store 里"是谁"一律用**内容身份**(`ChainHash`),**物理槽号不入身份** ——
   槽可回收复用,拿槽号当键 = 槽复用后假命中(设计要求见 §9.1)。
   槽是**绑定**(`SlotPool` 发号、前缀树持 `Option<u32>` 绑定),可随手沉/换。
9. **引用配对在腿内自洽**:缓存持有的那一份引用由 `KvStore` 自己增/减
   (登记 → `incref`,逐出 → `decref`)—— 调用方不再手工配 `for id in inserted { incref }` 的双写。
10. **指针 / ID 律**(裁决⑫):裸指针只有 server 持;本 crate 与 models/engine 一样**只持
    id + 元数据** —— 不持裸指针,不持 `Bytes` 所有权(布局由 engine **注册**进来,id 级 + `epoch`)。
11. **三分工**(裁决⑪):engine = 决策/调用 · **本 crate = 实现**(换什么/搬到哪层/账怎么走/
    宿主缓冲/策略出序)· server = 执行(地址/DMA/一次 ack)。
12. **载体三层**:`Carrier{Device(SlotId) | Host(HostId) | Disk(FileOff)}` —— 身份与载体**正交**,
    换层只换 `Carrier` + 台账挪一行;`Disk` 位现留、实现归 S6。
13. **批面为搬运唯一规范**:单件 = `n = 1` 的糖(实现共享一条路径)。批的 N 有上限并切块。
14. **段边界独占**：搬运口只接受"段边界"的调用形状;调用点不得持会话借用(先收集工作单,
    再独占执行)—— 与律⑦(自由函数)配套。捕获窗内 server 对 Swap 类命令**响亮拒**。
15. **需求驱动(§12.4/§17;2026-10-16)**：保洁只由**真需求**触发(准入 Δ_j / prefill 增量 /
    decode 上界 / 空转零需求),**禁用常数水位**预搬;需求算式必须与调度决策**单源**
    (engine 的 `step_need` ↔ `reuse_face`/`prefill_step_chunk`);`ensure_for_len` 的同步
    驱逐是**兜底**,触发即计入 `LegOps::ensure_evicted`(非零 = 保洁不足的指纹);
    需求上界宁可**多算**(多腾不丢内容),**少算才丢内容** —— 方向性错误不对称。
16. **组合件字段私有 + 语义方法(2026-11-16)**：`Rel` 的 `leg`/`state`/`snap_host`/`disk`/
    `snap_disk`/`index`/`admission`/`reservations` 不对 engine 暴露，**也不提供
    `leg()`/`state()`/`disk()` 访问器**；engine 只调 `Rel` 的语义方法
    (`slots_free`/`state_locate`/`kv_disk_plan_spill`/`plan_onload`/`release_table` …)。
    新增存储能力一律先给 store 加方法；禁止在 engine 侧重新拼内部账或直接替换组合件。
    2026-11-16 第一刀已落:engine 的 `spill_kv` / `spill_snap` / `fill_kv` /
    `fill_kv_window` / `fill_snap` / `write_chain_index` / `ensure_host_resident`
    已抽到 `engine::mem::disk_runtime::DiskRuntime`(无本地配额副本，接收 `&mut Rel`)；
    store 继续只出计划/记账。盘层淘汰已接 session 感知：
    `Rel::kv_disk_plan_evict_sessions` / `snap_disk_plan_evict_sessions`
    先避开活跃 session，不够才回落到活跃面；`StoreReport.disk_sessions` 给出
    逐 session 的盘上 KV/快照账。三级级联（GPU → Host → Disk → Drop）的候选
    出序也统一带会话档：`order_candidates`（KV/状态）与 `order_snap_victims`
    （快照）都是“冷会话先、会话内再按 LRU/LRFU”。
    2026-11-16 第二刀已落:`SessionBank` 的 `sessions`/`rel`/`leg_mtp`/`mtp_ring`
    已私有化；engine 统一走 `SessionBank::parts()` / `parts_mut()` split-borrow
    或 `reset_session_for_rebuild` / `reopen_session_chain` 这类复合语义口。
    `AgentSession::rebase` / `reopen_chain` 都是消费式（销毁旧账 → 新账放回），
    `reset` 已收为私有。
    2026-11-16 第三刀：报告面拆 `disk_blocks`（`KvStats`/`LegStats`），
    `Rel::report`/`leg_stats` 从 DiskTier 扣除盘上-only；checkpoint 的 store
    侧计划收为 `Rel::plan_checkpoint_kv`（连续前缀 + 真实 carrier + 配额），
    engine 只执行 D2H/宿主 clone/spill；写盘循环再收进
    `DiskRuntime::{spill_host_blocks,spill_device_table}`；死代码
    `checkpoint_tail_snap` 已删。运行期盘满 = 内容留宿主，下一拍压力决定 Drop
    （指标 `store.disk.*_full`）。下一步：配置段收口。
