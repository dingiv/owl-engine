# store = 状态存储(GPU 池 → host pinned → 磁盘)设计与迁移

> 立项:2026-10-16,用户令「KV cache 专项立项」;**落点由用户指定:
> `packages/owl/crates/store`(新 crate `owl-store`)**。
> 方法:code-master 模式二(画现状 → 定痛点 → 定目标 → 分阶段迁移)+
> 工作流四子步(接口 → 链路 → 依赖 → 空间)。
> 需求源:`docs/arch/requirements.md` REQ-PRE-03(P1)+ C4 风险条款
> (「默认关闭三级缓存,需要时按场景开」+「host RAM 预算显式配置」);
> 约束源:`docs/arch/charter.md` A1(图租约)/A5.1(预算分解封闭)/A5.2(一切分配过账本)。
> 相邻卷:`三流事件序-设计与迁移.md`(搬运的设备端序)、`装载提速-定谳轮-2026-10-16.md`
> (宿主层 pinned 池/预算两栏/块账的现成先例)。
> **设计参考(必读)**:`KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer
> GPU`(Di Chai et al.,arXiv:2609.04852v1,2026-09;本机 `kvmem.pdf`)——
> 同题(24GB 卡 + Qwen3.8-27B + MTP 跑 1M 工作区),其四机制 = 本卷骨架的对照系(§3.6)。

## 0. 立项口径(边界,先说清楚)

- **做什么**:把引擎的状态(KV/线性注意力状态/草稿/视觉)从「显存容量的函数」
  变成「存储层的函数」——**GPU 池 → host pinned → 磁盘** 三层,按前缀身份换入换出。
- **交付线**:262k 单卡可服务(现墙 fp8 128k);重灌命中 TTFT 可测改善;
  host RAM 预算可配可断言;开启默认关。
- **不做**(本卷之外,勿混刀):①分布式 store(TP 立项再议,见 §7);
  ②KV 量化新档(REQ-CTX-03 另案;本卷只留 dtype 轴);③前缀命中语义
  (E2c 块级缓存不动,本卷只回答「命中之后从哪一层取」)。

### 0.1 立项裁决(用户,2026-10-16)

| # | 裁决 | 结果 |
|---|---|---|
| ① | BlockManager/PrefixCache 是否迁入 store | **迁入** —— host 账与设备叶子分家,engine 只留叶子与编排 |
| ② | attention 块链 × GDN 快照 的耦合方式 | **二者独立管理 + 上层一个「关系层」关联**(不硬绑、不合并成一个管理器) |
| ③ | 换出/召回粒度 | 见 §3.6.2:定为 **逻辑块 = 32 token = 物理页**(与 KVMem 缺省一致,且天然等于我们 XQA 的 `tokensPerPage`) |
| ④ | 草稿池一期是否入账 | **推后**(S5 单独立刀) |

## 1. 病根(现状:四类状态、两本账、一张策略表)

### 1.1 现状盘点(代码事实,逐条可查)

| # | 状态 | 落点 | 几何/粒度 | dtype | 容量与上限 | 可共享 |
|---|---|---|---|---|---|---|
| ① | **attention KV** | `engine/src/state.rs::StatePool.kvs` | paged classic `kc [nb,Hkv,hd/x,page,x]`、`vc [nb,Hkv,hd,page]`;策略表 `models::module::kv_paged_policy(dtype)` 分发(legacy token-major 为回退) | 随 `spec.dtype` 或 fp8(U32 承载 1B/elem,元素数 ÷4) | `nb=ceil(pool_tokens/page)`(+1 dump page);`pool_tokens` 缺省 = **2× 单会话容量**,再被 `vram_target` 预算夹 | ✅ 块级 ref-count(≥2 = 前缀共享) |
| ② | **GDN/线性注意力状态** | `StatePool.gdns`(每 GDN 层 4 块:`conv_q/conv_k/conv_v/rec`) | 行宽 = `elems/gdn_slots`;**一格一会话** | **恒 f32** | 格数 = `knobs.gdn_slots`(缺省 8)⇒ **并发上限被格数钉死**(`session.rs` 耗尽报错);27B `rec` ≈ 1MiB/格/层 × 18 层 | ❌ 独占;可**快照** |
| ③ | **草稿 KV** | `StatePool.mtp_kvs` / `StatePool.dflash_kvs` | MTP:1 层独立页池;DFlash2:5 层 × (8 头 × hd128) | MTP = 主池 dtype;DFlash2 = BF16 或 fp8(`OWL_DFLASH_KV_FP8`) | 两者都**按 `pool_tokens` 全额预留**(无按需/降级账) | ❌(MTP **不接前缀缓存** —— 依赖 hidden,不可 token 哈希) |
| ④ | **视觉塔** | `models/src/layers/vision.rs` + `models/src/vlm.rs` | 塔权重 `model.visual.*` + **每图一次前向**;输出 `[n_img,D]` 经 `image_scatter` 注入文本序列 | — | 图像 token 的 KV **落进主池**(⇒ 天然可前缀复用);塔自身无驻留 KV | ✅(经文本池) |
| ⑤ | **GDN 快照池** | `StatePool.snaps`(snap_max)+ `spec_snap` | key = **链尾物理块 id**(内容身份);行宽同 ②;D2D 行拷贝 | f32 | `snap_max` 槽,**LRU 换 key 覆盖写**(零 alloc) | —(是"状态续跑"的钥匙) |

### 1.2 两本账(host 侧 vs 设备侧)

- **设备侧** `StatePool`(`state.rs`)= 分配 + 叶子句柄(`kv_caches`/`gdn_caches`/`bt_leaf`)
  + 状态治理(`reset_gdn` / `capture_snap` / `restore_snap` / `write_bt`);
- **host 侧** `BlockManager`(`blocks.rs`)= ref_count(0 空闲/1 独占/≥2 共享)
  + 空闲 FIFO + `ensure_for_len` **事务性增长** + **压力驱逐**(先逐前缀缓存叶补自由度)
  + **E2c 块级前缀缓存**(内容链哈希,`prefix_cache.rs` 持索引/LRU);
- **会话** `SessionTable`(`session.rs`)= `slot_base` + `gdn_slot`;close 释放格号;
- 全局:**没有任何 host/盘层卸载**(现有 `dtoh/htod` 只出现在取证探针里)。

### 1.3 三条结构性痛点(本卷要治的)

1. **无分层**:状态只在显存 ⇒ 池容量 = 显存剩余的函数。实测(27B + draft,
   slots2/snap1):**fp8 池 131072 tok READY @ 23358MiB(95.1%);196608 起 OOM**
   (k-cache 首片 201MB 分配失败);f16 同配置 ≈ 64k。262k 达标线**必须**换层。
2. **两腿之间没有关系层,且都只有一层**:attention 块链(BlockManager)与 GDN 快照
   (`snaps`)本就是两套账、两套驱逐 —— 这本身**不是**病(裁决②:它们几何/共享性/
   复算代价结构性不同,该独立);病在 **①二者之间没有关联层**(没有"这条前缀的
   两条腿在不在"的统一判定与统一换出序 ⇒ 混合架构只能要么全在、要么全无),
   **②各自只有显存一层** ⇒ 谁都换不出去。llama.cpp 侧的完整判例在册
   (`docs/llama.cpp/mtmd-hybrid-缓存失效-排查实录`:中部断前缀 = 全量重灌)。
3. **草稿池无账**:③ 的两个池都按 `pool_tokens` 全额预留,不吃"按需/切分/降级"
   的治理 ⇒ 有 draft 的档位相当于第二个主池。
   附带:**固定面 17.9G live**(权重账面 13.7G,实际 ~17.5G,**~2G 待审计**)
   —— 归因六件套已就位,可先清(见 §6 刀序 S0)。

### 1.4 前作盘点(源码级):三者同族 =「**身份命中 + 时间序淘汰**」

> 本机源码:`packages/vllm`(v1 kv_offload)、`packages/llama.cpp`(本仓 fork)、
> `packages/sglang`(HiCache)。结论先行:**它们都假设"该留的历史 = 前缀 ∪ 最近",
> 命中判定问的是"这段 token 我处理过吗"(身份),不是"哪些历史块与当前查询相关"(相关度)。**

| 系统 | 命中判定(键) | 工作集形状 | 淘汰/准入信号 | 三层 | 复发状态 |
|---|---|---|---|---|---|
| **vLLM**(本仓 v1) | **块哈希精确匹配**(`OffloadKey = block_hash + group_idx`,`kv_offload/base.py`)⇒ 前缀链身份 | 连续(前缀)∪ 活跃请求 | **LRU / ARC / chain-LRU**(`kv_offload/cpu/policies/`);`async_lookup.py` = 按哈希**预取**(预测的是"后续块在不在池里",不是"哪些块相关") | ✅ `cpu/` + `tiering/{fs,obj,p2p}`(盘/对象/跨实例) | —(无混合模型概念) |
| **llama.cpp**(本仓 fork) | **位置对齐 + KV shifting**(`--cache-reuse N` = "min chunk size to reuse from the cache via **KV shifting**",`common.h:626`) | 连续(前缀)∪ 最近 | context shift 丢最老;`--ctx-checkpoints`(缺省 32/槽)= **整态 partial 快照**(带 pos_min/pos_max,含草稿态)做回滚/部分复用,FIFO 淘汰(`server-context.cpp:2363-2416`) | ⚠️ 只有"VRAM ↔ host RAM"(`-nkvo`/`offload_kqv`),**无盘层** | **整态快照**,非分页、非按需 |
| **sglang HiCache** | **页键(字符串/内容哈希)+ 最长前缀**语义(`hicache_storage.py`: `prefix_keys`、"Longest-prefix semantics") | 连续(前缀) | **LRU 文件驱逐器**(`storage/file/lru_file_evictor.py`) | ✅ GPU/host/盘 | — |

**三条推论(定我们对标姿态)**:

1. **三层化本身是工程活,不是研究活**:vLLM/sglang 的形态 = 「内容键 + 策略对象 +
   异步 lookup + 分档后端」,可以照抄骨架;它们的瓶颈不在搬运,而在**选择**。
2. **"相关度"是缺口,也是论文的价值**:KVMem 把命中判定从**身份**换成**相关度**
   (Mean-K 打分 ⇒ 工作集可以是**稀疏非连续**的、每步重选),代价 = 需要重排/位置重建
   (§3.6.3 的 A/B)与质量门(§5 G3)。
   另注:研究界的"相关度"家族分两支 —— **剪枝**(H2O/StreamingLLM:token 级、
   **丢数据、不可逆**)vs **虚拟化**(KVMem:全量留在层里、只选择物化)。
   我们对齐后者:**不分页丢数据**;混合架构下剪枝尤其危险(GDN 递推状态被剪即失真)。
3. **我们的形态三者都没有**:vLLM 无复发状态、llama.cpp 只有整态快照、
   KVMem 明确**不虚拟化复发状态**(*"recurrent model state continues through the
   original execution path"*)—— 「attention 块链 + GDN 快照两腿独立、关系层关联」
   (裁决②)是**本卷相对全部前作的结构增量**。

## 2. 需求 = 端到端链路(触发点 → 终止条件)

| # | 链路 | 触发点 | 终止条件 | 现状 |
|---|---|---|---|---|
| **L1** | **重灌命中** | 会话提交 prompt(turn 开始) | 首 token 产出,且**未重算命中部分的 KV/状态** | 只有显存层可命中;超池即整段重算 |
| **L2** | **压力换出** | 池自由度不足(`ensure_for_len` 判定) | 池可服务(或响亮拒绝) | 只能逐前缀缓存叶;再不够直接报错 |
| **L3** | **turn 收口登记** | turn 完成 | bundle 在 store 可见可命中 | 只登记前缀缓存块 + 快照 key,不构成可换出对象 |
| **L4** | **启动预算** | `engine.run` 装配 | 三层预算全部分解完毕,越界**拒绝启动** | 只有显存一层(A5.1) |

### 2.1 两范式:容量层(A)× 选择层(B) —— **互补,不是同题**(2026-10-16 用户定调)

先把"假设集"摆清楚 —— 两个范式对**"活跃序列要驻留多少 KV"**的假设根本不同:

| | **A 容量/搬运范式**(vLLM / sglang HiCache / llama.cpp) | **B 选择/虚拟化范式**(KVMem) |
|---|---|---|
| 对活跃序列的假设 | **整条活跃序列的 KV 必须能参与注意力**(dense)⇒ 常驻需求 ∝ **上下文长度** | **参与集合可以是任意有界子集**(selected)⇒ 常驻需求 ∝ **窗口**,与地址空间解耦 |
| 解决的问题 | **跨请求/多会话的容量**:前缀复用、抢占换出、多租户经济性 | **单请求超窗**:地址空间 ≫ 原生窗口(1M 工作区) |
| 命中判定 | 身份(块哈希/位置) | 相关度(注意力空间打分) |
| 索引资产 | 块哈希链(小) | **Mean-K 索引**(层×KV头×块) |
| 损失性 | **无损**(只搬家) | **近似**(被选中的集合改变了注意力分布)⇒ 必须配质量门 |
| 谁受益 | 同前缀重复到来的负载(RAG/模板/多轮) | 长轨迹单会话(agent 工作区/长文档) |

**单卡 262k 的算术(本机 27B 实测配置:16 全注意力 + 48 GDN、Hkv=4、hd=256 ⇒ 每 token 64KiB f16 / 32KiB fp8)**:

| 上下文(dense) | KV(fp8) | 单卡(权重 ~17.5G) | TP2(权重 8.75G/卡) |
|---|---|---|---|
| 128k | 4.29 GB | **21.8 G ✓**(= 实测 23358MiB,即 128k 墙) | 10.9 G |
| **262k** | 8.59 GB | **26.1 G ✗(放不下)** | 13.0 G ✓ |

⇒ **定论**:①单卡 262k,dense 路线**无解**(算术,不是调优);要走 A 达到 262k **必须加卡(TP)**;
②单卡 262k **只能**走 B(常驻 = 执行视图窗口;64K 窗口 fp8 ≈ **2.1GB**,舒适;
地址空间 262k/1M 落 host/盘);
③A 与 B **不是二选一** —— **B 决定"这一步该看到什么",A 决定"东西在哪、怎么搬、放不下时丢谁"**。
B 选出的块必须落在 A 的三层里搬运;A 的 LRU/ARC 在"同优先级候选"里做跨会话容量裁决。
**本卷两者都做,但门的归属不同**:A → 多会话/前缀复用/抢占(容量经济性);
B → 单请求超窗(**262k 门在这里**)。

**混合架构的额外账(两范式共同的前提)**:GDN 状态 = **162 MB/会话**(48 层 × 3.38MB,f32;
`gdn_slots=8 ⇒ 1.3GB`)。它**不可共享、不可内容哈希**,但**任何"续跑/复用"都必须带上它**
(否则退化成重算)⇒ 这是裁决②「两腿独立 + 关系层」的定量理由(§3.6.4)。

### 2.2 分界点:三条容量线 + 一条性能交叉线(不是一根常数线)

**预算式(单卡,一切判据的来源)**:

```
V_avail = V_card − B_weights − Σ_live S_gdn − O_runtime − P_draft        ← 可给 KV 的余量
dense 可服务 ⇔ N_req · k ≤ V_avail            ⇒ N* = V_avail / k        ← 容量线①
selection 可服务 ⇔ W · k ≤ V_avail  且  N_req ≤ N_pos                    ← 容量线②(+位置墙)
多会话可服务 ⇔ Σ_i N_i · k ≤ pool 且 并发 ≤ gdn_slots                     ← 容量线③(池墙)
```

其中(本机 27B 实测锚点):`k = 32 KiB/token(fp8)/ 64 KiB(f16)`;
`B_weights ≈ 17.5 GB`(账面 13.7G,**2G 待审计**);`S_gdn = 162 MB/会话`;
`O_runtime` = 图 slab + 激活 + 上下文 + warmup(≈2~3 GB);`P_draft` = MTP/DFlash2 草稿池
(按 `pool_tokens` 全额预留,DFlash2 fp16 约 20 KiB/token ⇒ 128k 档 ≈ 2.6 GB);
`W` = 执行视图预算(KVMem 档位 32K/64K);`N_pos` = 位置墙(原生 262k / YaRN 1M)。

**三条容量线(硬墙,越线只能换范式或拒服务)**:

| 线 | 判据 | 本机 24G × 27B 的实测/估算 | 越线后 |
|---|---|---|---|
| ① 单请求 dense | `N_req ≤ N*` | **fp8:实测 128k 墙(draft on);清 2G 审计面后 ~190k**;f16 ≈ 64k | 强制 selection(或拒/抢占/整段重算) |
| ② selection 常驻 | `W ≤ N*` 且 `N_req ≤ N_pos` | W=64K fp8 ⇒ **2.1 GB**(舒适);N_pos = 262k(原生) | 位置墙:上 YaRN 或"紧凑窗口"方案 B(§3.6.3) |
| ③ 多会话池 | `Σ N_i ≤ pool/k`;`并发 ≤ gdn_slots` | 池缺省 = 2× 单会话容量;`gdn_slots=8` ⇒ GDN 1.3 GB | A 轴(前缀复用/LRU 换出/抢占) |

**一条性能交叉线(不是墙,是"择优"区)**:
`W < N_req ≤ N*` 时**两种都可行** —— dense 每步读全量 KV(带宽 ∝ N,保真无损);
selection 每步只物化 W(+索引打分,开销 ∝ 块数)。⇒ **两可区间内按策略择优**
(默认:dense 优先保真;当 N/W 超过阈值或首 token 延迟吃紧时切 selection)。
**交叉点不是容量问题,是吞吐/保真权衡 —— 归策略对象,不归硬判据。**

**三条工程结论(本卷据此定形态)**:

1. **线是动的**:`V_avail` 随(并发数 × GDN 格、draft 开关、KV dtype、前缀缓存占块)实时变化
   ⇒ 判据必须走**实时账本 + per-request admission**(不是 boot 期常数);
2. **越线要能"就地切"**:一条序列 dense 起步、涨过 N* 后切 selection —— 切换点是一个
   **段边界**,且必须落 **GDN 快照**(否则递推状态丢,回不去);
3. **索引必须在 dense 阶段就建**(prefill 顺手采 Mean-K,边际成本≈0;论文同款 incremental
   indexing)⇒ 任何序列**随时可切**,且历史块天然可被后续会话相关度召回。
   **这一条是"两范式互补"的工程接缝:索引 = 两范式共用的资产。**

### 2.3 活跃度模型:布局是"活跃 turn 数"的函数,不是会话数的函数(2026-10-16 用户定调)

**定义(本卷用词;代码现状与它不一致的地方 = 改造点)**

| 词 | 定义 | 生命周期 | 常驻成本 |
|---|---|---|---|
| **会话(session)** | 客户端身份 + 其历史(块链、快照、索引、元数据);跨多轮存在 | 创建 → `close` / LRU 逐出 | 元数据(µs 级)+ **可驱逐资产的引用** |
| **turn(轮次)** | 一次提交(prefill → decode 到 stop/abort)= 一个请求 | 提交 → `Completed`/`Failed`/`Abort` | — |
| **活跃 turn** | 已提交且**未终结**的 turn | 同上 | **不可驱逐资源**(见下表) |
| **活跃会话** | 当前有活跃 turn 的会话 | — | — |
| **idle 会话** | 曾经存在、但 turn 已终结(历史还在) | — | **只占可驱逐资产**(快照 + 缓存块) |

**资源归属律(改造后的目标形态)**

| 资源 | 归属 | 借 / 还 | 可驱逐 | 现状(证据) |
|---|---|---|---|---|
| KV 工作集(在飞读写) | **活跃 turn** | turn 起 / 止 | ❌ | 现:块由 ref-count 持有,基本正确 |
| KV 历史块(缓存) | 会话(经前缀缓存) | 登记 / 逐出 | ✅ | 现:E2c 前缀缓存 ✓ |
| **GDN 行(162 MB/行)** | **活跃 turn**(← 改造点) | turn 起:`restore_snap(链尾)` 命中则填充,否则零起;turn 止:`capture_snap(链尾)` → **还格** | ❌ | 现:**会话级**(`session.rs:159-171` 建会话即占,`close` 才还)⇒ idle 会话白占 162MB |
| GDN 快照 | 会话(身份 = 链尾块 id) | 登记 / LRU | ✅(且**应可下沉 host/盘**) | 现:`snap_max` 槽 LRU 覆盖写(**不够装 idle 会话**) |
| 会话元数据 | 会话 | 创建 / close | — | ✓ |
| 图 slab / 激活 / 工作区 | 引擎 | boot | ❌ | ✓ |

**三条由此而来的结论**

1. **`gdn_slots` 的语义要正名**:其真实含义 = **并发 turn 上限**(`max_active_turns`),
   **不是"能挂多少会话"**;能挂多少会话只受 **host/盘 + 缓存预算**约束(agent 场景:几十~几百)。
   现状的"会话数超并发上限"报错 = 历史包袱(会话与并发被同一变量表示)。
2. **分界点公式按活跃度修正**(取代 §2.2 里"Σ_live"的粗口径):
   ```
   V_avail(卡)  = V_card − 权重 − 运行底价 − #active_turn · 162MB − 草稿池(开时)
   H_avail(host)= host 预算 − Σ_idle 会话快照(162MB/会话,可下沉) − 缓存块驻留
   准入(turn_i) ⇔ Σ_active N_j·k + N_i·k + 缓存驻留 ≤ pool·k
                 ∧ #active_turn + 1 ≤ gdn_slots
                 ∧ 激活/图槽 有余
   ```
   ⇒ **分界点由三元组决定:(单请求 N, 活跃 turn 数, idle 会话数×快照常驻)**,
   而不是单看 N —— 这正是"活跃会话个数直接影响布局"的形式化。
3. **idle 会话 = 快照(可下沉)+ 缓存块(可驱逐)**:这才使"单卡挂 100 个长尾 agent 会话"
   成为可能(同时只有 2-4 个活跃 turn,吃 KV 的是它们的**在飞工作集**)。
   先例:llama.cpp 的 **slot(`-np`)= per-task**,任务结束时把状态存 prompt cache
   (`llama_state_seq_*`)⇒ 与"行=活跃 turn、idle=快照"同构;vLLM 亦是 per-request KV + radix 持 idle 态。
   我们比二者多的 = 把这三家都**暗含**的活跃度模型**显式化**,并给它三层。

**改造刀(插入 §6 S2 之后)**

| 刀 | 内容 | 验收 |
|---|---|---|
| **S2b** | **GDN 行改"活跃 turn 借还"**:turn 起 `restore_snap(链尾)`(命中即跳过重算)/ turn 止 `capture_snap(链尾)` 后还格;`gdn_slots` 语义正名为并发 turn 上限 | ① idle 会话不再占行(以 `gdn_slot_profile` 证明全零行可复用)② 命中路径与未命中路径**逐 token 恒等**(G3)③ 挂 N 个 idle 会话 + M 个活跃 turn 不冲突(并发门重写) |

> 依赖:**S2b 依赖 S2**(状态腿三层化)—— 否则 idle 会话的快照只能挤 `snap_max` 那几个
> LRU 槽,互相覆盖 ⇒ 还不了格。两刀分先后,不合并。

### 2.4 两个用户旋钮:boot 的"最大并发 turn 数" × per-turn 的"直接显存地板"

用户要能自己定取舍 —— 两个旋钮各管一头,且**语义互为对偶**:

| 旋钮 | 面 | 落点 | 语义 |
|---|---|---|---|
| **`max_active_turns`** | **boot**(flavor/`EngineKnobs`) | 现 `gdn_slots`(`engine.rs:65`,缺省 8)⇒ 正名 | **最多几个 turn 同时在飞**。它是**硬天花板**:同时活跃 turn 数 ≤ 此值(每活跃 turn 要一行 GDN + 一份在飞工作集) |
| **`dense_floor`(直接显存地板)** | **per-turn**(`TurnOpts`,`turn.rs:61`) | 新增字段:字节数 或 `Strict` 枚举 | 本 turn **要求直接驻留显存的最小 KV 量**。它**就是对 KVMem 式损失的耐受度**:`耐受度 = 1 − dense_floor / (N·k)` |

**语义(一条式子说清两者关系)**:

```
dense_floor = N·k        ⇔  Strict(零耐受)⇒ **禁用选择层(B)**:本 turn 必须全量 dense 驻留
dense_floor < N·k        ⇔  容忍 B :差额 (N·k − dense_floor) 落三层的 host/盘,由选择层按需物化
                            它的执行视图预算 W ≈ dense_floor / k(下限即 W 的下限)
```
- `dense_floor` 是**下限而非目标**:引擎给得起就给更多(§2.2 的"两可区间":`W < N ≤ N*` 时默认仍走 dense 保真);
- **`Strict` + 显存不足 = 响亮拒绝**(或按策略排队),**绝不静默降级**成近似 —— 这是"不耐受"的唯一正确行为;
- 一个**保留额不变量**:turn 一旦被准入,其 `dense_floor` 部分**不可被驱逐**(reserved floor);
  A 轴的驱逐只能动 ①idle 会话的缓存块/快照 ②活跃 turn 在 `dense_floor` 之上的**弹性余量**。
  ⇒ 驱逐面 = `缓存 + 弹性余量`,保留面 = `Σ_active dense_floor`。

**准入式(把 §2.3 的式子补全)**:

```
准入(turn_i, floor_i) ⇔
  (a) Σ_active floor_j + floor_i + 缓存驻留 + 弹性余量 + 运行底价 ≤ V_card     ← 保留额 + 非保留
  (b) floor_i = N_i·k(Strict)时:同上即"全量在卡",无弹性项
  (c) #active_turn + 1 ≤ max_active_turns
否则:先按 A 轴驱逐(缓存块/快照下沉)→ 仍不足 ⇒ 拒绝 或 排队(策略对象,见 §3.2)
```

**为什么必须 per-turn 而不是 per-session**:同一个 agent 会话里,不同 turn 的保真需求不同
(读文件摘要的 turn 可耐受、算数/取值的 turn 不可耐受)⇒ 旋钮挂在 turn 上。
服务面口径:OpenAI 兼容扩展字段(或 `TurnOpts` 直通),缺省值由 boot 配置给(如 `dense_floor = 64K`),
**客户端可不指定**(走缺省)、可指定 `Strict`、可指定任意地板值。

**报告面(可观测即可控)**:每 turn 回吐 `dense_floor 请求值 / 实得值 / 实际物化比(materialized vs workspace)`,
并可入 metrics ⇒ 客户端能看见自己买到的"保真"。

## 3. 目标架构(四子步)

### 3.1 入口与空间(crate 边界:两个独立管理器 + 一层关系)

**分工(裁决②落地)**:存储面按**状态种类**分成两个互不知道对方的管理器,
再上叠一层**关系层**把二者按"前缀身份"关联起来 —— 关系层只做映射与配对策略,
不做数据搬运、不持物理缓冲。

```
        ┌─────────────────────── owl-store(crates/store)────────────────────────┐
        │  Rel(关系层):(chain_hash ↔ 块边界 ↔ snap_key) 映射 · 配对策略 ·         │
        │              召回语义(Exact 续跑 / ContextOnly 上下文召回)              │
        │  ┌──────────────────────────┐   ┌──────────────────────────────┐      │
        │  │ KvStore(迁入)            │   │ StateStore(新,与 KvStore 平级) │      │
        │  │  逻辑块账 + 前缀索引      │   │  GDN 快照/格账(行宽视图)        │      │
        │  │  (BlockManager+PrefixCache)│  │  key = 块边界标识(同源)        │      │
        │  │  三层:GPU页表/宿主/盘     │   │  三层:GPU 快照槽/宿主/盘        │      │
        │  └──────────────────────────┘   └──────────────────────────────┘      │
        │  公共面:宿主层物理(pinned 池)· 预算账(TierLedger)· 策略 trait        │
        └──────────────────────────────────────────────────────────────────────┘
                 ▲ 单向依赖(engine → store;store 只依赖 owl-iface + owl-shared)
        ┌────────┴─────────────────────────────────────────────────────────────┐
        │  owl-engine:设备叶子所有权(StatePool 的块句柄)+ 时序编排(段边界/图租约)│
        │  owl-cuda:命令面(htod/dtoh/copy_block_at/alloc_pinned/复合口 —— 已存在)│
        └──────────────────────────────────────────────────────────────────────┘
```

**为什么独立而不是合并**(与请求 B/流水线不同,状态种类差异是结构性的):

| 维度 | attention KV(KvStore) | GDN 状态(StateStore) |
|---|---|---|
| 几何 | paged 页(32 tok/页) | 每层 4 块(conv×3 + rec),行宽 = elems/slots |
| 身份 | **内容可哈希**(token 链)⇒ 可跨会话共享 | **不可内容哈希**(依赖 hidden/递推)⇒ 只能按**块边界**标识 |
| 共享 | ✅ 多会话同块 | ❌ 一格一会话 |
| 换出单位 | 逻辑块(32 tok) | 快照行(整层行宽) |
| 复算代价 | 高(prefill) | 极高(要重跑整段递推) |
| RoPE | K 位置敏感(见 §3.6.3) | 无位置概念 |

⇒ 二者的**命中条件、驱逐序、搬运形状全不同**,合在一个管理器里必然互相污染
(1.3 痛点②)。独立 + 关系层是唯一能同时保住"块可共享"与"状态不可共享"的形态。

```
        ┌───────────────────────────── owl-store(crates/store,叶子 crate)─────────────────────────────┐
        │  对象模型 KvBundle / 身份(链尾块哈希)/ 三层账(TierLedger)/ 换入换出策略 /                │
        │  宿主层物理(pinned 池、磁盘文件)/ 预算报告 / 目标口径                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────┘
                 ▲ 单向依赖(engine → store;store 只依赖 owl-iface 契约面 + owl-shared 工具面)
        ┌────────┴──────────────────────────────────────────────────────────────────────────────────┐
        │  owl-engine:设备叶子所有权(StatePool 的块句柄)+ 时序编排(pump/段边界)+ 图租约           │
        │  owl-cuda:命令面(htod/dtoh/copy_block_at/alloc_pinned/lease 流完成回调 —— 全部已存在)      │
        └───────────────────────────────────────────────────────────────────────────────────────────┘
```

**分工律**(本卷最重要的一条):
- **store 持「身份 + 账 + 宿主/盘物理」**——纯策略与簿记,可在 CpuFace 下全量单测;
- **engine 持「设备叶子句柄」与「时序」**——store 要动设备时,由 engine 把
  `&mut D`(DeviceClient)借给它,**store 自己发命令**,不新增后端原语;
- 数据面 I/O 写成 crate 级**自由函数**(host↔device 的块搬运),store 本体方法保持自肃
  (只改自身字段;不摸别人的变量、不做隐藏在方法里的 I/O)—— 与工作区 AGENTS 纪律同源。

### 3.2 对外接口(伪代码契约;输入/输出/失败/生命周期)

```rust
/// 前缀身份(关系层的键)—— 两个管理器各自认它的一个投影:
///   KvStore 认 chain_hash(内容);StateStore 认 last_block(块边界)
pub struct PrefixId { pub chain_hash: u64, pub last_block: u32 }

/// **召回语义**(关系层产出;二者独立 ⇒ 命中可以只有一半)
pub enum RecallMode {
    /// 块链 + 状态快照都在 ⇒ **精确续跑**(位置不重排、状态直接还原)
    Exact,
    /// 只有块链 ⇒ 作为**注意力上下文**召回(KVMem 路线:复发状态沿原路,
    /// 不重算历史)—— 语义是"上下文召回"而非"状态续跑",必须显式标注
    ContextOnly,
}

pub struct StoreConfig {
    pub host_budget_bytes: usize,   // RAM 上限(0 = 关闭本层)——REQ C4「预算显式配置」
    pub disk_dir: Option<PathBuf>,  // 二期;None = 无磁盘层
    pub policy: EvictPolicy,        // LRU / 命中优先(策略对象,非硬编码)
}

/// 关系层(唯一对外门面;两个管理器在它后面)
impl Rel {
    /// 启动装配:三层预算分解 + 两管理器就绪;分解失败 = **拒绝启动**(A5.1)
    pub fn open(cfg: StoreConfig, budget: &BudgetInputs) -> Result<Rel, StoreError>;

    /// 命中查询(**纯 host,零设备副作用、零分配**)——L1 第一步
    /// 回**分层命中面**:块链在哪层、快照在不在、能到什么语义(Exact/ContextOnly)
    pub fn probe(&self, id: PrefixId) -> RecallHit;

    /// 换入(两腿按需,各自独立;**失败不留半态** —— 回滚已做的那条腿)
    pub fn admit<'a>(&'a mut self, face: &mut impl DeviceClient, hit: &RecallHit)
        -> Result<AdmitLease<'a>, StoreError>;

    /// turn 收口登记(L3):两腿各自登记(块链进 KvStore;快照进 StateStore)
    pub fn register(&mut self, id: PrefixId, blocks: &[BlockId], snap: Option<SnapRef>) -> Result<(), StoreError>;

    /// 压力换出(L2):两管理器各自出候选,关系层合并成一张**统一换出单**
    /// (排序 = 策略;执行 = engine 在段边界做)
    pub fn plan_evict(&mut self, need: EvictNeed, protect: &[PrefixId]) -> EvictPlan;

    /// 预算/命中报告:三层 × 两管理器 + 关系层口径(Exact/ContextOnly 命中数)
    pub fn report(&self) -> StoreReport;
}

/// 设备侧叶子的**唯一入口**:engine 把叶子借给 store,store 只经此登记
/// (store 不持 Bytes 的所有权,只记账 —— 所有权留在 engine,避免双主人)
pub trait KvLeaves {
    fn blocks_of(&self, chain: &[BlockId]) -> Vec<LeafRef>; // 逻辑块 → 设备叶子
    fn snap_rows_of(&self, snap: SnapRef) -> Vec<LeafRef>;
}
```

**关系层的最小职责(避免长成第二个上帝对象)**:
① 映射(`PrefixId` ↔ 两条腿的键);② 语义判定(`Exact` / `ContextOnly`);
③ 换出单合并(把两腿候选按统一策略排序);④ 报告口径。**不含**任何搬运/分配/驱逐实现。

**不变量(写在类型上,不靠注释)**:
- `admit` 的租约未 drop 前,该 bundle 的宿主副本**不可被覆写**(宿主层缓冲 = 带租约的池);
- `plan_evict` 绝不返回 `protect` 里的 bundle,也绝不拆散一个 bundle(§4 裁决②);
- `probe` 无副作用(可以在任意时刻调,包括图捕获期),`admit` 只能在**段边界**调。

### 3.3 调用链(三条端到端链路)

```
L1 重灌命中
  session.submit(prompt) → 块链哈希 → Rel::probe
    ├ 未命中 → 原路(现行为,零变化)
    └ 命中(host 层) → [段边界] Rel::admit
         ├ 自由显存不足 → Rel::plan_evict → engine 执行换出(段边界)→ 再 admit
         ├ 回灌:attention 块链逐块 H2D(复合口;宿主 lease 由流完成回调还池)
         ├ 回灌:GDN 快照 D2D(快照槽 → 会话格;与块链**同一批**)
         └ prefill 只算未命中尾巴 → 首 token
L2 压力换出
  KvStore 自由度不足(原 BlockManager.ensure_for_len 语义)→ Rel::plan_evict(victim 序;两腿候选合并)
    → [段边界] 逐 victim:设备块 → 宿主缓冲(D2H/复合)→ 释放设备块(BlockManager.decref)
L3 turn 收口登记
  turn 完成 → capture_snap(GDN 快照槽 key=链尾块 id)→ Rel::register(块链 + snap)
L4 启动预算
  engine.run → Rel::open(BudgetInputs{ 权重/池/图/激活/通信/host/盘 })→ 越界拒绝启动
```

### 3.4 依赖与驱动时序(为什么必须"只在段边界")

- **硬约束**:图捕获期禁 `htod/dtoh`(现有 A2.2 同源约束;捕获窗内只允许
  D2D/核);所以换入换出**只能**发生在:①piecewise 段的接缝 ②pump 间隙
  (`Idle`/turn 边界)③prefill 分块之间(**不是** decode 步内)。
- **搬运的设备端序**:宿主 lease 复用靠 server 侧"流完成回调才还池";与
  计算流的序走复合口(写等读/读等写),沿用 `三流事件序` 卷的既有机制,**不新造**。
- **单向依赖**:`engine → store → (iface, shared)`,禁止反向;
  store 不认识 `GraphPlan`/`StatePool`,只认识 `DeviceClient` 与块 id。

### 3.5 空间预算(A5.1/A5.2 形态)

```
显存 B = 权重 + KV 池 + 草稿池 + GDN 状态 + 图 slab + 激活峰值 + 通信缓冲 + 底价 + 余量   ← 现有,不变
host  H = pinned 池上限(可配) + 索引/账目开销(µs 级)                              ← 本卷新增,显式
磁盘  D = 缓存目录配额(可配;二期)                                                  ← 本卷预留
```
- 每条项**没有实测数字就拒绝启动**(A5.1 原话);
- host/盘占用入 store 自己的 `TierLedger`(可与后端 Governor 对账),`report()` 可 dump;
- 默认:`host_budget_bytes = 0`(关层)——满足 C4「默认关闭三级缓存」。

### 3.6 与 KVMem(arXiv:2609.04852)的对齐与差异

**3.6.1 直接采纳的四机制**(原样,勿自创):

| 机制 | 内容 | 对我们的落点 |
|---|---|---|
| **块级(非 token 级)检索** | 论文定谳:H2O/StreamingLLM 的 token 级打分在百万 token 持久工作区里**索引容量与查询延迟都不可行**;改为逻辑块级 | 我们本来就只到"页"粒度,天然适配 |
| **Mean-K 索引** | 每 (层, KV 头, 块) 一个**去 RoPE 的位置无关**平均 K 向量;查询同样去 RoPE 后打分,跨块**全局 softmax** 归一,再对层/头取均值 ⇒ 块相关度 \(R_b\) | 索引入 host(我们 host 层已有位置);打分在 GPU 分片流式(KVMem 的 GPU-tiled scoring) |
| **两条视图分离** | *repository view*(每块的有效副本在哪层)× *execution view*(本步选中的块按时间序进紧凑窗口);用**页表别名**而非密拷贝 | 我们**已有页表**(`xqa_pt_of` / 块表)⇒ 这条几乎是白送 |
| **三层策略** | ①proactive stage-out(与 chunked prefill 重叠)②retrieval-aware 分层(host 准入看"近期+累计检索频次",非纯 LRU)③打包+流水再物化 | 我们的 prefill 分块/pump 间隙正是天然的 stage-out 窗口 |

**3.6.2 粒度裁决(原文依据)**:论文实测"top-8 块覆盖 66.5% 历史注意力、
top-16 覆盖 77%",并明确 *"remains efficient at fine granularity. Our implementation
defaults to **32-token blocks** with Mean-K"*;块越小 = 均值保真损失越小 + 召回分辨率越高,
代价是索引与块管理开销 ↑。

⇒ **本卷定案:逻辑块 = 32 token**(检索/选择/搬运单位),**与物理页一一对齐**,
且**天然等于我们 XQA 的 `tokensPerPage=32`**(`docs/bench` 的 xqa 资产自描述行)
—— 不需要新页几何、不改任何 attention 内核契约。

**3.6.3 位置一致性:论文的 re-RoPE 机器我们可能不需要**(待实测裁决,见 §4-③)

论文的难点在于:被召回的块要**重排进紧凑窗口 ⇒ 位置变了 ⇒ RoPE 失效**,
故需"raw K 为真源 + 入块重打 RoPE + 常驻块有界 delta re-RoPE(超限重建)"。
我们的 paged attention 走**页表 + 显式位置向量**(位置由调用方按 token 给,
XQA 路径 = `xqa_seq`/页表),因此有一条更便宜的路:

- **方案 A(原位置稀疏召回)**:召回块**保留原始逻辑位置**(页表里散放,空洞不参与
  注意力;内核按页表遍历 ⇒ 计算量 ∝ 命中块数,不 ∝ 逻辑跨度)⇒ **零 re-RoPE、零数值误差**;
  代价 = 逻辑位置空间不压缩(若位置空间等于原生窗口上限,则召回量受 262k 位置空间约束)。
- **方案 B(KVMem 紧凑窗口)**:位置空间可无限扩张(地址空间 = 存储层),但需整套
  raw-K/re-RoPE 机器 + 数值误差界。

⇒ **建议(⚠️ 已被 §29.3 更正为 B)**:一期走 **A** 是"未读论文前的纸面倾向";
读完全文后改判 **B**(紧凑窗口 + re-RoPE)—— 依据与反例见 **§29.3**。

**3.6.4 复发状态(GDN):论文不虚拟化 —— 这正是我们的增量所在**

论文原话:*"It manages the growing KV state of standard RoPE attention layers, while
**recurrent model state continues through the original execution path**."*
⇒ 它只虚拟化 attention KV;混合模型的递推状态沿用原路(即"上下文召回"而非"状态续跑")。
我们的 27B 是 GDN+attention 混合、且**已有** GDN 快照池 ⇒ 我们比论文多一条腿:
- 用户裁决②给出形态:**两腿独立管理 + 关系层关联**;
- 关系层据此产出 `RecallMode::{Exact, ContextOnly}`:有快照 = 精确续跑(我们的优势),
  无快照 = 论文式上下文召回(退化但可用)。**这一对语义是本卷相对论文的主要增量**。

**3.6.5 我们独有、论文没有的约束**(设计必须额外照顾):

1. **图捕获**:我们的 decode 是整图捕获 + piecewise 分段;论文无图。⇒ 搬运**只能**在段边界
   (捕获窗禁 htod/dtoh),这是 §3.4 的硬约束来源;
2. **投机解码**(MTP/DFlash2 + spec-aware graph):召回会改变 verify/propose 的形状与
   草稿 KV,一期把草稿池排除(裁决④);
3. **量化栈**:marlin W4A16/AWQ + fp8 KV(KVMem 自称不支持 FP8 KV)⇒ 召回路径要
   同时覆盖 fp16/fp8 两种页格式(页 dtype 是 store 的一个轴);
4. **前缀缓存已存在**(E2c 块级):我们不从零做"共享",而是把它的命中面**从显存一层扩到三层**;
5. **宿主层先例已就绪**:装载域的 pinned 池/租约/复合口/预算两栏/块账 —— 直接搬(§0 相邻卷)。

## 4. 裁决台账(录结果;未定项写明待什么)

| # | 裁决 | 状态 | 结果/依据 |
|---|---|---|---|
| ① | BlockManager/PrefixCache 迁入 store | ✅ **定:迁入** | 用户 2026-10-16;host 账集中在 `KvStore`,engine 只留设备叶子 |
| ② | attention × GDN 的耦合方式 | ✅ **定:独立管理 + 关系层** | 用户裁决;落点见 §3.1 分工 + §3.2 `Rel` 最小职责;语义面 = `RecallMode` |
| ③ | 位置一致性方案(A 原位置稀疏召回 / B 紧凑窗口 + re-RoPE) | ⏳ **待实测**(A 为首发建议,见 §3.6.3) | 判据:一期能否在"位置空间 = 262k"内满足门 G1;若必须 >原生窗口 ⇒ 上 B |
| ③' | 换出/召回**粒度** | ✅ **定:逻辑块 = 32 token = 物理页** | 论文缺省 32-token;与我们 XQA `tokensPerPage=32` 天然对齐(§3.6.2) |
| ④ | 草稿池入账 | ✅ **定:推后**(S5) | 用户裁决;牵扯投机路径,单独立刀 |
| ⑤ | 磁盘层格式 | ⏳ 二次裁决(S6 前) | 是否与 llama.cpp checkpoint 形态对齐(跨引擎可读) |
| ⑥ | 目标口径 | ✅ **定:单卡 262k 为目标 ⇒ B(选择层)在关键路径;A 为容量底座**(2026-10-16 用户定调) | 依据 = §2.1 算术;若将来把 TP 立项并接受"262k 靠双卡 dense",A 单独即可达标 |
| ⑦ | 位置一致性 A/B | ⏳ 待实测(§3.6.3) | 一期 A(原位置稀疏召回);若目标改 1M 工作区再上 B |
| ⑩ | **施工序** | ✅ **定:A 轴(vLLM 式)先行,B 轴(KVMem 选择层)后加**(用户 2026-10-16) | "先把 vLLM 那一套实现出来,再在这个上面加"。A 轴 = 身份命中(块哈希/内容链)+ LRU/ARC + 三层搬运;**B 轴不改 A 的接口**,只在其上加选择层 |
| ⑧ | **boot 旋钮:最大并发 turn 数** | ✅ **定**(用户 2026-10-16) | 落 `EngineKnobs`(现 `gdn_slots` 正名 `max_active_turns`);硬天花板 |
| ⑨ | **per-turn 旋钮:直接显存地板 = 损失耐受度** | ✅ **定**(用户 2026-10-16) | 落 `TurnOpts`;`Strict` = 禁 B + 不足即拒/排队;**保留额不可驱逐** |
| **⑪** | **换入换出的执行分工** | ✅ **定**(用户 2026-10-16) | **server 执行**(地址/DMA/一次 ack)· **engine 决策 + 调用**(何时换/哪条链/保护面)· **store 持实现**(换什么/搬到哪层/账怎么走/宿主缓冲/策略) |
| **⑫** | **指针 / ID 律** | ✅ **定**(用户 2026-10-16) | **裸指针只有 server 持有**;models / store / engine 只持 **ID + 元数据**;store 必须区分**显存 / 内存 / 磁盘**三类载体(`Carrier`) |

## 5. 验收门

- **G1 262k 可服务**:27B fp8 + 262k 单请求 READY(现墙 128k);重灌场景
  TTFT 相对现状改善可测(口径:现有 `apps/server/tests/prefix_reuse_probe.py` + std 矩阵);
- **G2 账目封闭**:开启后**设备峰值 ≤ 关闭时**;`host_budget_bytes` 越界 = 响亮拒;
  `StoreReport` 三项(设备/host/盘)可与 nvidia-smi/`/debug/blocks` 对账;
- **G3 正确性恒等(分档,与 per-turn 旋钮对齐)**:
  ① **Strict turn**:命中/未命中两条链路**逐 token 恒等**(现有恒等门范式;spec 臂按纪律用 spec-off);
  ② **地板 < N·k 的 turn**:不与 dense 逐位比(定义上允许近似),改用
  **物化比 + top-k 覆盖率** 与**任务级**判据(G5③),并回吐实得地板(§2.4 报告面);
- **G2b 保留额门**:活跃 turn 的 `dense_floor` 在任何驱逐压力下**不被换出**(工具:
  `/debug/blocks` + 逐 turn 实得地板断言);
- **G4 稳定性**:长跑零崩、零 Xid;命中/换出/回灌/命中延迟入 metrics(默认关,开 <2% 开销);
- **G5 对照靶**(KVMem 实测,供我们定标;同机不同卡,仅作量级参照):
  ① 工作区/执行视图比 ≥ 4×(论文:1M 工作区 / 64K 视图 / 24GB 卡);
  ② 召回延迟亚秒级(论文 1M 工作区 pre-answer 1.81s;对照 Compact+RAG 106s);
  ③ 命中块覆盖率量级:top-8 块 ≈ 66% 历史注意力(top-16 ≈ 77%)——用于校我们的
  召回预算与选择策略是否合理;
  ④ 单会话生成 ~50 tok/s(论文档位;我们的 spec/graph 栈只应更快)。

## 6. 刀序(两轴:底座 / A 容量 / B 选择;每刀独立可验、可回滚)

**底座(两范式共用,先做)**
| 刀 | 内容 | 验收 |
|---|---|---|
| **S0** | 2G 待审计清账(显存归因六件套;固定面 17.9G vs 账面 13.7G) | 固定面逐项有实测数字(可能直接换数万 token 池) |
| **S1** | 立 `crates/store`:`Rel` + `KvStore` + `StateStore` 骨架 + 三层账/预算分解 + 策略 trait + `arch.md` + 注册 workspace(纯 host,CpuFace 单测) | 预算分解、`probe` 语义、换出单合并 —— 单测全绿;零设备依赖 |
| **S1b** | **宿主 pinned 层搬运闭环**(设备⇄host,单块往返;复用装载域 pinned 池/复合口) | 往返后设备内容逐块回读一致;host 预算可断言 |
| **S2** | **GDN 状态腿**:状态三层化 + 关系层 `RecallMode::Exact` 全链(162MB/会话 的账立起来) | 热命中 GDN **零重算**(`gdn_slot_fingerprint` 为证);恒等门 G3 |

**A 轴(容量):跨会话/前缀复用/抢占**
| 刀 | 内容 | 验收 |
|---|---|---|
| **S3** | 迁入 `BlockManager`/`PrefixCache`(裁决①)+ LRU/ARC 策略对象 + 压力换出接真实路径 | 4096 会话压力不报错;命中率/换出入 metrics;TTFT(重复前缀)改善可测 |

**B 轴(选择):单请求超窗——262k 门的唯一路径**
| 刀 | 内容 | 验收 |
|---|---|---|
| **S4b-1** | Mean-K 索引构建(prefill 期顺手采;去 RoPE)+ host 常驻 + GPU 分片打分内核 | 索引开销/显存实测;块相关度与"全量注意力 top-k"覆盖率对齐(G5③) |
| **S4b-2** | 每步/每步点选块 + **页表别名**组执行视图(位置方案按 ⑥/⑦ 裁决) | **G1:单卡 262k 可服务**,恒等/质量门 G3 通过(近似路径需与 dense 对照) |
| **S5** | 草稿池入账(裁决④)+ 视觉塔图像嵌入备忘录(§7-1) | 有 draft 档位池按需;同图重放塔零重跑 |
| **S6** | 磁盘层(冷工作区/跨进程持久) | 冷重灌 TTFT 改善可测;原子写/崩溃恢复 |

> 顺序纪律:**底座 → A → B** 是依赖序(搬运栈是共同前提),但**门不互替**:
> A 的完成**不**兑现 262k(§2.1 算术);262k 只在 B 落刀后成立。

## 7. 挂账/未决(与本卷相邻,勿混刀)

1. **视觉塔**:图像 KV 已在文本池(设计正确),但①图像前向**无备忘录**(同图重放要
   重跑塔)②`apps/server` 的 `image_url` 尚未接线(M5 剩件)③视频/多图 token 类型
   (`token_types`)的缓存键如何进身份 —— S5 一起做;
2. **TP 下的 store**:每 rank 一份还是共享宿主层?TP 立项时定(A2.6 单进程多实例 ⇒
   倾向"每实例一份账,宿主层按 rank 分片");
3. **KV 量化新档**(REQ-CTX-03:int8/fp8/q4-turbo)与 store 的 dtype 轴对齐 —— 另案;
4. **磁盘格式**:是否与 llama.cpp 的 checkpoint 形态对齐(跨引擎可读)—— 二期再议;
5. **草稿池的前缀语义**:MTP 不可 token 哈希(依赖 hidden)⇒ 它的"命中"需要别的身份
   (e.g. hidden 指纹);S5 立项时单独设计。

## 8. 施工单(S0 → S2b;每步 = 交付物 + 门 + 证据)

> 纪律:每步独立可验、可回滚;**不合并刀**;门不过不进下一步。

### S0 · 显存账清账(测量,无代码改动)
- **交付**:固定面分解表(权重 / 图 slab / 激活峰值 / 工作区 / 上下文 / 杂项)+ **2G 差额定位**
  (`docs/` 一张表 + 本卷 §2.2 常数更新:`B_weights`、`O_runtime`、`P_draft` 各是多少实测值)。
- **门**:A5.1 —— 每一项都有实测数字;**没有数字的项禁止进预算式**。
- **工具**:显存归因六件套(已合入 main:`a374343`)、`/debug/blocks`、`vram 治理` 日志、`GraphPlan::slab_bytes()`。

### S1 · 立 crate(纯 host:账 + 策略)—— ✅ **已落(2026-10-16)**
- **交付**:`crates/store`(`owl-store`)+ 注册 workspace + `arch.md`;模块面:
  `id`(身份)/ `tier`(三层账)/ `budget`(预算分解 + 准入判据 + `TurnFloor`)/ `policy`(驱逐与准入策略对象)/
  `kv`(块账 + 前缀索引)/ `state`(GDN 快照与行账)/ `rel`(关系层门面:probe/register/plan_evict/report)。
- **门**:①`cargo test -p owl-store` 全绿;②预算分解与准入判据、`Strict` 拒服务、
  保留额不可驱逐、换出单合并 —— 各有单测;③零 `owl-cuda`/`owl-engine` 依赖(单向律)。
- **实测**:9 文件 / ~1300 行;**26 单测**;依赖面 = `thiserror`(+测试面 `owl-cpu`);
  写测试当场抓出三个真缺陷(共享块会被误逐、Strict 拒绝文案漏标、候选按链去重会双算)。

### S1b · 宿主层与搬运闭环(设备 ⇄ host)—— ✅ **已落(2026-10-16)**
- **交付**:数据面自由函数(host↔device 块搬运)+ 宿主缓冲池租约(复用装载域 pinned 池与复合口);
  `Rel::admit/evict` 接上真实搬运;注入式失败路径(部分失败 = **整单回滚**)。
- **门**(全过,9 单测):①**真 face 往返逐字节一致**(`CpuFace`;门内含"清零后回灌复原"
  与"图样校验"两步);②宿主层预算越界**响亮拒**(含关层 `host_bytes=0`);③长度不符在触设备前拒;
  ④设备侧失败原样上抛;⑤账目迁移平衡(Gpu/KvBlock ⇄ Host/KvBlock)。
- **形态定案(vLLM 同款)**:宿主**内容不 pin**(pageable `Vec<u8>`;RAM 可被内核回收),
  pin 只用于搬运途中的**一次性暂存租约**(`alloc_pinned` → 拷贝 → `upload_pinned`,
  租约所有权移入 server,流完成回调归还其池);`free_cap` 上限保证驱逐后 RAM 真释放。
- **⚠️ 实测教训(入坑册 §42)**:块字节**不能**用 `Bytes` 自述长度(`len × dtype`)——
  CPU 面值块恒 F32、`htod` 回 `len = 0` ⇒ 各 face 元素口径不同。搬运口一律**显式收
  `len_bytes`**,真源 = store 的块几何常量(`KvStore::block_bytes()`)。

### S2 · GDN 状态腿三层化(快照可下沉)+ 关系层 `Exact` 全链
- **门**:①热命中路径 GDN **零重算**(`gdn_slot_fingerprint` 为证);②G3①(Strict 逐 token 恒等);
  ③idle 快照的 host/盘往返往返一致。

### S2b · GDN 行 = 活跃 turn 借还 + `max_active_turns` 正名
- **交付**:格号借还点从会话生命周期挪到 turn 生命周期;`EngineKnobs` 字段正名;
  并发门按 `max_active_turns` 重写(挂 N idle 会话 + M 活跃 turn)。
- **门**:①idle 会话不占行(全零行可复用,`gdn_slot_profile` 证);②命中/未命中**逐 token 恒等**;
  ③N=100 idle + M=`max_active_turns` 活跃不冲突。

### 后续(**A 轴先行** —— 裁决⑩;见 §6 两轴表)
| 序 | 刀 | 内容 | 门 |
|---|---|---|---|
| 1 | **S2** | GDN 状态腿三层化(快照可下沉)+ 关系层 `Exact` 全链 | 热命中 GDN 零重算 + G3① 逐 token 恒等 |
| 2 | **S2b** | GDN 行 = 活跃 turn 借还 + `gdn_slots` 正名 `max_active_turns` | 挂 N idle + M 活跃不冲突;idle 不占行 |
| 3 | **S3** | **迁入 BlockManager/PrefixCache**(裁决①)+ LRU/ARC 策略 + 压力换出接真实路径 | 4096 会话压力不报错;前缀重复 TTFT 改善;命中/换出入 metrics |
| 4 | **S3b** | per-turn `dense_floor` 接 `TurnOpts` + `AdmissionPolicy` 落地 | G2b 保留额门;实得地板回吐 |
| 5 | **S4b-1/2** | **B 轴(后加)**:Mean-K 索引 + 每步选块 + 页表别名 | **单卡 262k 门**;G3② 物化比/覆盖率 |
| 6 | **S5/S6** | 草稿池/视觉备忘录;磁盘层 | 见 §6 |

---

## 9. S3 首刀实录(身份迁移:物理块号 → 内容身份)· 2026-10-16

> 落点:`crates/store`(**未接线到 engine**;engine 接线是 S3-2)。本节的裁决面
> 只补 §4 台账未覆盖的一处:**"身份用什么"** —— 它是 S3 能不能做 host 卸载的前提。

### 9.1 裁决补记:store 里"是谁"一律用**内容身份**(S3 定案)

**物理槽号不得进入身份。** 槽是可回收资源:`decref` 归零 → 回自由队列 →
下一次 `acquire` 从**队首**弹出 = 立刻复用(`blocks.rs` 现役行为)。任何以槽号为键的
账(host 副本键、快照键)在槽复用后都会**指向别人的内容**。

- KV 腿:块身份 = **逐块父链内容哈希**(`ChainHash`;含父链 ⇒ 字面相同但历史不同
  的两段 token 不是同一块)
- 状态腿:快照键 = **边界块的内容哈希**(`SnapKey(ChainHash)`)——不是"快照的内容哈希"
  (递推态不可内容寻址),而是"它对应哪一段内容"的身份(`arch.md` 律②)
- 宿主层键 = 同一内容身份(`HostTier` 的键从 `BlockId` 改为 `ChainHash`)

**引擎侧现存隐患(本刀发现,尚未修;归 S3-2 或独立微刀)**:GDN 快照键在 engine 里 =
`block_table[边界]`(**物理块号**,`exec/spec.rs::capture_gdn_snap`),而快照池
(`state.rs::snaps`)**不随前缀缓存驱逐失效** ⇒ 场景:
①会话 A 的快照 key = 物理块号 `k`;②A 的块被缓存 LRU 逐出 ⇒ A 表释放后引用归零 ⇒
槽 `k` 回池;③新会话 B 的链拿到槽 `k` 且**边界恰好落在它上面** ⇒
`has_snap(k)` 为真(`scheduler.rs:114`)⇒ `restore_snap` 把 **A 的 GDN 态**灌进 B
= **静默错值**(不报错、不崩,只错)。
修法:**快照键全部走内容身份**,~70 行,四处:

| # | 落点 | 改动 |
|---|---|---|
| ① | `prefix_cache.rs` | `hash_block_chain` 提为模块级自由函数(**身份链唯一产地**:前缀缓存建树 / 游标 / 跨会话匹配同一份)+ 新增 **`ChainCursor`**(会话级懒惰推进:只喂新增整块 ⇒ 全 turn O(n),避免每边界从头算的 O(n²)) |
| ② | `session.rs` | `AgentSession.chain: ChainCursor`(生命周期随链:`reset()` 归零;前缀命中由调度层 `seed(m, hashes[m-1])` 播种) |
| ③ | `blocks.rs` / `scheduler.rs` | `match_prefix` 连**内容身份链**一起回;`has_snap(hashes[m-1])` 判定 + `restore_key = hashes[m-1]`(不再碰物理块号) |
| ④ | `state.rs` / `exec/spec.rs` | 快照键类型 `u32 → u64`;`capture_gdn_snap` 改由游标出键(边界 token 段 = `prompt_ids ++ out`;`fed` 单调不减是前提,已核 spec 回滚只前进不后退);同一边界重复调用 = 幂等跳过 |

**✅ 已修(2026-10-16 同日微刀,用户裁决「先修隐患」)**:engine 侧 7 文件 / +246 行;
门 = ①engine lib 测试 44/44(新增 3 条:`chain_cursor_matches_cache_hash_chain`
游标 ≡ 缓存哈希链、`recycled_slots_do_not_forge_snapshot_keys` 槽复用不撞键、
cursor 回退/段不足响亮拒)②clippy 改动面零告警 ③store 46/46 未受影响。
**遗留**:G3① 的 GPU 端到端(`gpu_prefix_cache_hit` 等)待设备空闲时跑
(本机 `OWL_TEST_DEVICE` 未设、GPU2 正被模型服务占用)。

### 9.2 S3 施工分解(每刀独立可验;顺序 = 依赖序)

| 刀 | 内容 | 状态 / 门 |
|---|---|---|
| **S3-1** | **`KvStore` 升级**:`kv/slot.rs`(槽池:分配/引用/回收)+ `kv/tree.rs`(内容身份前缀树:匹配/登记/叶 LRU 驱逐)+ `kv/mod.rs`(引擎级 API:`ensure_for_len`/`match_prefix`/`register_chain`/`release_table`/`unbind`/`rebind`);`Rel` 改门面(跨腿台账在门面维护);`id.rs` 去物理 `BlockId` | ✅ **已落**;46/46 单测;与 `BlockManager`+`PrefixCache` **行为逐条对齐**(FIFO 复用、叶 LRU + 惰性校验、断链即停、压力驱逐 ≤4 轮、失败事务性) |
| **S3-2** | **engine 接线**:`SessionBank` 的 `blocks_m`/`blocks_mtp` 换成 `owl-store` 的块腿(engine 只留 `StatePool` 设备叶子);删 `blocks.rs`/`prefix_cache.rs` 双份 | ✅ **已落**(2026-10-16,见 §9.5);门 = 行为等价(engine lib 36/36 + store 50/50;`prefix_reuse_probe` 待 GPU 空闲补跑) |
| **S3-3** | **host 换出接真实路径**(形态见 §10):立 `BlockLeg`(账 + 宿主物理 + 搬运口)+ **布局注册制**(id 级,`epoch` 校验)+ 新批搬运面(`DtohBatch`/`HtodBatch`);压力先下沉、命中面 `slot = None` 在段边界回灌 | 待做;门 = 往返逐字节一致(真 face)+ 池压力下命中率/TTFT 改善可测;**迁移本身零行为变化**(`host_bytes = 0` 默认关层) |
| **S3-4** | **策略对象 + 报告**:ARC/chain-LRU 并列实现(策略是对象,不是分支);命中/换出入 metrics;`S3b` per-turn `dense_floor` 接 `TurnOpts` | 待做;门 = G2b 保留额门 + 实得地板回吐 |

### 9.3 S3-1 的形态要点(给接线刀的接口约定)

- **分配**:`ensure_for_len(table, target_len)` —— 与 engine 现名同,语义同(只增、幂等、
  事务性、池不足先逐缓存叶);返回被逐条目(内容身份, 槽 `Option`),门面据此维护台账。
- **命中**:`match_prefix(tokens) -> PrefixMatch { matched_blocks, blocks: Vec<MatchedBlock>, tail }`;
  `MatchedBlock { chain, slot: Option<u32> }` —— **`slot = None` 就是"内容在 host 层,
  该回灌"的显式信号**(本刀只立形态,S3-3 接搬运)。
- **登记**:`register_chain(tokens, table)` —— 引用配对(缓存持一份绑定引用)在**方法内自洽**;
  engine 现版"调用方手工 `for id in inserted { incref }`"的耦合被消掉。
- **下沉/回灌**:`unbind`(引用必须为 1,活跃面**响亮拒** —— 律④)/ `rebind(chain, slot)`
  (回灌绑**新槽**;旧槽早被复用 —— 这正是身份必须内容化的现场证明)。
- **台账面**:`TierLedger` 的 `Gpu/Host · KvBlock` 记**store 管的缓存面**
  (`kv.stats().gpu_bytes` 同口径);`Admission` 的保留/弹性面不在 ledger 里(§2.4 面)。

### 9.4 与 S2 的关系(刀序说明)

S3 与 S2 是**两条腿**(裁决②),S3 不依赖 S2;但**跨会话精确复用**(`RecallMode::Exact`)
需要 S2(快照可下沉)才有意义 —— 否则 host 里的 KV 链命中只能落到 `ContextOnly`
(复发态沿原路,不能跳过)。⇒ §8"后续"表的 S2 → S3 顺序**保留为推荐**(收益完整度),
而 S3-1/S3-2 可先行(A 轴容量底座,裁决⑩)。

### 9.5 S3-2 接线实录(engine 块腿迁入 store)· 2026-10-16

**改了什么**

| 面 | 变更 |
|---|---|
| 账房 | `SessionBank.blocks_m` / `blocks_mtp`:`engine::blocks::BlockManager` → **`owl_store::KvStore`**(槽池 + 内容身份前缀树);`engine/src/blocks.rs`、`engine/src/prefix_cache.rs` **整文件删除**(两本账的合并语义已在 S3-1 迁入 store) |
| 几何 | 新增 `session::BankPlan { nb, page, block_bytes, paged, mtp, prefix_cache }`(参数对象,替代 6 个裸参);`block_bytes` 由 `StatePool::block_bytes()` 出(**元素口径真源** —— 坑册 §42:fp8 池 U32 承载 ⇒ 元素数 ÷4、元素宽 ×4) |
| 复合口 | 块腿复合操作做成**自由函数**(`session::ensure_chain` / `session::match_prefix_chain`):借用只落 `&mut KvStore` 一个字段,与"调用点仍持 `&mut AgentSession`"的借用形状对齐(方法口会整借 bank 而冲突) |
| 身份 | 快照键类型 `u64` → **`owl_store::SnapKey`**;链游标 → `owl_store::ChainCursor`(哈希原语唯一产地 = `store::kv::chain`)⇒ **物理槽号在类型上进不了快照键** |
| 未接线面 | 命中/驱逐出现 `slot = None`(宿主承载)= **响亮拒**(`match_prefix_chain` / `ensure_chain` 内),不静默少用几块/丢账 —— 这两处就是 S3-3 的接线点 |

**门**(本机可跑部分全绿):`cargo test -p owl-engine --lib` **36/36**(删掉的 9 条 = 原 `blocks`/`prefix_cache` 单测,语义已在 store 侧以同名用例接管);
`cargo test -p owl-store` **50/50**;engine 改动面 clippy 零告警。
**待补**:`prefix_reuse_probe`(需 GPU 空闲;GPU2 正被模型服务占用)。

**挂账(故意不做,别混刀)**

1. **`Rel` 门面未接线**:engine 现在**直接用 `KvStore`**。走 `Rel` 需要 `BudgetInputs` 全套实测
   (权重/运行底价/草稿池/每 token KV 字节)——那是 **S0 清账**的交付;S0 未做时 `Rel::open`
   会(正确地)拒绝启动。⇒ **S3-3(host 换出)之前先做 S0 的那几个数字**,或让 Rel 支持
   "账面子集 + 明确标注未实测项"的降级装配(后者要在卷内再拍一次)。
2. MTP 链仍是"无前缀缓存"的独立页池(S5 再议)。

### 9.6 S0 实测(0.8B @ RTX 3080,2026-10-16)—— `Rel::open` 的输入齐了

**探针**(engine e2e 场景,按需开):`gpu_08b_budget_probe`
```bash
CUDA_VISIBLE_DEVICES=GPU-<3080> OWL_TEST_DEVICE=0 OWL_BUDGET_PROBE=1 OWL_E2E_ONLY=budget_probe \
  cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1 --nocapture
```
口径 = 真分配账(`vram::live` 五相采样 + `StatePool` 三个**真源口**:
`block_bytes()` / `gdn_row_bytes()` / `snapshot_pool_bytes()` / `draft_pool_bytes()`,
2026-10-16 新立),**不按公式估**。

| 项 | 实测值 | 说明 |
|---|---|---|
| `v_card_bytes` | **20,818,165,760 B**(≈19.39 GiB) | 开卡后 `cuMemGetInfo` free(§2.2 口径);total = 21,082,537,984 |
| `weights_bytes` | **1,538,340,480 B**(1467 MiB) | live(装载后) − live(Engine::new) |
| `runtime_floor_bytes` | **5,835,808 B**(5.6 MiB) | 余项 = 图 slab + 表件;**不含 CUDA ctx**(free 口径已扣,实测 ctx+p排保留 ≈250 MiB) |
| `draft_pool_bytes` | **0** | 本档无投机(关 = 0 是合法实测值) |
| `gdn_row_bytes` | **20,201,472 B**(19.27 MiB/活跃 turn) | 18 层 × 1.07 MiB;池总 = ×`gdn_slots` |
| `kv_bytes_per_token` | **12,288 B**(12 KiB) | 6 全注意力层 × 2(k+v) × hkv2 × hd256 × 2B |
| `host_bytes` / `disk_bytes` | 0 / 0 | 默认关层(REQ C4) |
| 池几何 | `nb=nb_kv=256`(page 32 ⇒ 8192 tok)、`n_full=6`、`n_gdn=18`、`kv_fp8=false`、`gdn_slots=8`、`snap_max=4` | 0.8B 默认档 |

**固定面分解(开档实测,与 `vram::live` 逐项对上)**:
权重 1467.1 + KV 池 96.0 + GDN 格 154.1 + 快照池 77.1 + 草稿池 0 + 运行底价 5.6 ≈ **1800 MiB**;
turn 期 live 峰值 1800.5 MiB(瞬态面 ~1 MiB)。

**容量换算**:`V_avail`(2 活跃 turn)≈ 18,073 MiB ⇒ `N* ≈ 1.54M tok`
—— 0.8B 档的卡上容量**完全不紧张**,门槛在 **GDN 行 × 并发** 与 host/盘面(正是 store 要治的)。

> **27B 的 2G 待审计仍挂**(需 3090 空闲;同一探针换 boot 27B 即可复跑)。
> **S3-3 解锁**:0.8B 档的 `BudgetInputs` 现在齐了 ⇒ 可以在 0.8B + 3080 这个测试台上
> 真开 `Rel`(host 卸载闭环的接线与验证不必等 27B 清账)。

---

## 10. 交互契约(2026-10-16 设计会话定案)—— 与 engine / backend / models 的边界

> **定位**:store 是本引擎里**耦合度最高**的部件 —— 它同时牵住 engine 的时序、backend 的搬运
> 原语、models 的结构知识。它的相邻模块(GPU server、graph、comm)解耦很好,靠的是"契约窄";
> store 的契约要变宽(它必须知道"在哪/多大/丢谁"),所以**不靠约定,靠律**。本节是那套律的产权证。

### 10.1 指针 / ID 律(裁决⑫)

> **裸指针只有 server 持有;server 以上的每一层,只准持"ID + 元数据"。**

| 层 | 允许持有 | 绝不允许 |
|---|---|---|
| **server**(`GpuServer`/actor 线程) | **地址表**(块 id → 设备指针/偏移)、流、图、pinned 租约 | 块语义、身份、layer/session 概念 |
| **models** | `TensorOps` = (块 id, dtype, shape) —— 纯逻辑张量 | 任何地址;任何 tier 概念(它永远只看"当前物化视图") |
| **store** | 身份(内容哈希)、`Carrier`(见下)、宿主物理、三层账、策略 | 裸指针;`Bytes` 所有权;模型类型 |
| **engine** | 设备池的**所有权与生命周期**、会话表(逻辑块 → 槽)、**时序** | 宿主物理内部、搬运动作细节 |

**store 里"东西在哪"的唯一表达(三层必须区分)**:

```rust
pub enum Carrier {
    Device(SlotId),   // 显存:槽号(server 持 槽号 → 每层片段地址)
    Host(HostId),     // 内存:宿主 slab 槽(store 自己的缓冲池)
    Disk(FileOff),    // 磁盘:(文件代, 偏移);S6 落地,enum 位现在就留
}
pub struct BlockState { pub refs: u32, pub fp: u64, pub carrier: Carrier, pub last_use: u64 }
```

身份(`ChainHash`)与载体**正交**:换层 = 只换 `Carrier` + 台账挪一行,身份不变 —— 这正是
"槽会被回收复用、身份不能"那条律的结构化落点(§9.1)。

### 10.2 三分工(裁决⑪)

| 层 | 角色 | 判据 |
|---|---|---|
| **engine** | **决策 + 调用 + 编排** | 何时换(段边界/压力时刻)、哪条链、**保护面**;**搬运的编排**(取计划 → `face.swap` → 回执 → 记账)—— engine 是**唯一发起 I/O** 的一层 |
| **store** | **实现(纯 host;§21)** | 换什么、搬到哪层、账怎么迁、宿主缓冲怎么管、策略怎么出序 —— 出**计划**与**记账**,**不碰 face** |
| **server** | **执行** | 地址计算、DMA/流序、**一次 ack** |

### 10.3 两条硬推论

**推论 A:** **"块 → 设备片段"的映射是注册制**,不是每次调用传叶子。

上一版草案(engine 每次构造 `BlockLeaf` 传进来)被这两条律否掉:换出的**实现**要在 store 里
独立完成 ⇒ store 必须自己知道"这块在第几层的哪个偏移"。故:

```rust
// boot 一次(engine 提供、store 存档;**id 级,不含指针**)
leg.attach_pool(PoolLayout { epoch: u64, pieces: Vec<PieceSpec>, block_tokens: usize })?;
pub struct PieceSpec { pub block: u64 /*句柄 id*/, pub stride_bytes: usize, pub len_bytes: usize }

// 之后所有交换只谈"块"
leg.offload(chain)?;               // store 内:算片段 → 组装批命令 → 账目 → 宿主缓冲
leg.onload(chain, dst_slot)?;
```
收益:①store 的实现完整自洽 ②engine 调用点一行 ③每调用的叶子参数对象消失
④`epoch` 挡陈旧布局(池重建/几何变化 ⇒ 重注册)。

**推论 B:** **一次交换 = server 的一条命令、一次 ack。**

现状:契约面只有单件 `dtoh`(且回 `Vec<u8>`,server 侧分配)与 D2D `CopyBatch`。而一个逻辑块
= **层数 × 2** 个片段(0.8B 12 段、27B 32 段)⇒ 不批化就是 N 次往返/块。故后端增量(唯一一个):

```rust
DtohBatch { pieces: Vec<(Bytes, off_bytes, HostSpan)>, ack }   // D2H:读语义,回执即落地
HtodBatch { pieces: Vec<(HostSpan, Bytes, off_bytes)>, ack }   // H2D:经一次性 pinned 租约
// HostSpan = 调用方(store)的宿主缓冲片(ptr+len);与今天 dtoh 的 out: &mut [u8] 同源
```
**形态律**:批面为**唯一规范**,单件 = n=1 的糖(实现共享一条路径,不设两套规范)。
**server 侧额外加一条断言**:捕获窗内收到 Swap 类命令 ⇒ **响亮拒**(时序判断仍归 engine,但
server 不再"默默照办",焊死"随手搬")。

### 10.4 五条接口律(API 变更的判据)

1. **门面挂叶子字段,不挂组合体**:接口长在 `BlockLeg`/`StateStore` 上,不长在
   `SessionBank`/`Rel` 上 ⇒ 调用点借用天然分裂(`self.bank.leg.ensure_for_len(&mut s.block_table, n)`
   合法;`self.bank.ensure_for_len(...)` = E0499 —— 2026-10-16 现场证过,见坑册 §44)。
2. **控制面同步零 I/O / 数据面自由函数**:判定与记账不得触发设备动作;搬运口显式收
   `&mut D: DeviceClient`(律⑦)。
3. **身份 = 内容哈希,槽 = 绑定,载体 = `Carrier`,叶子 = 注册的 id**:store 不缓存 `Bytes`,
   长度显式传(坑册 §42)。
4. **段边界独占**:搬运调用点**不得持会话借用** —— 先"收集工作单"(短借用),再"独占执行"。
5. **一条腿一个管理器;`Rel` 只在两腿齐备时出现**。只有 KV 一条腿时立 `Rel` = 门面里一半是
   空气(`StateStore`/准入/预算无人用)且把启动绑到未实测数字上 ⇒ **先立 `BlockLeg`,S2 落地
   后再立 `Rel`**(关系层)。半空门面 = 旁路(工作区明令禁止)。

### 10.5 搬运面与"批命令 ⇒ 调用迁移"

**新面**:`DtohBatch` / `HtodBatch`(§10.3 推论 B);`CopyBatch` **已有**。

**清存量**(已有批面、调用点仍在逐件 await;核过源码):

| 位置 | 现状 | 收益/处置 |
|---|---|---|
| `capture_snap` / `restore_snap`(E2c 前缀快照) | **每次 18 层 × 4 = 72 次 `copy_block_at` await**(27B = 192) | **极高**:每个块边界都跑;一次可省 71/191 次往返。D2D `CopyBatch` 已存在,语义等价 |
| `capture_spec_snap`(spec 快照) | **已批化**(192 → 1) | 现成样板,照抄 |
| boot `zero_block` 逐层 alloc+memset、`reset_gdn` 逐层清零 | 逐件 | 中低(启动/每 turn 一次),后置 |
| 采样 `dtoh`(每 token ~1MB logits) | 单件,天生一件一步 | **不批** |

**迁移纪律**:一调用点一刀 + 等价门(逐字节/逐值一致);批的 N 有上限并切块(命令体/ack 负载)。

### 10.6 models ↔ store:这不是"谁依赖谁",是**驻留控制回路**(MoE 是它的正脸)

```
        结构(谁是可分层的块)  ← models
                │
   执行(前向/调度) ← engine(决定:这一批要哪些) → store(实现:驻留/搬运/账) → server(执行)
                ▲
        热信号(router 每层命中:谁热)
```

| 环节 | 归属 | 内容 |
|---|---|---|
| 结构 | models | 专家 = **独立可分层的块**(`ExpertBlock{layer, expert_id, pieces, bytes}`);稠密核(attention/router/embed)标 `DenseCore` = 永不驱逐 |
| 热信号 | models → engine | 每层窗口内专家命中统计(router 输出顺手上报,零额外成本) |
| 驻留决策 | engine(时机)+ store(实现/策略) | 这一批/下一步的工作集;store 按**类**出序(LRU 给 KV,**热度**给专家) |
| 执行 | server | 地址 + DMA + 一次 ack |

**量级论证(为什么决策必须在 engine,不能"缺页现搬")**:专家块是 MB 级(4bit 下 ≈6MB/专家;
fp8 大 MoE 单专家 40MB+),而路由是**每 token 每层**的 ⇒ 40MB/10GB·s⁻¹ ≈ 4ms;一层 8 专家 × 60 层
= 每 token 十几 GB —— 现缺现搬物理不可行。**可行解只有:大批次 + 热集驻留 + 与计算重叠的提前搬**。

**因此的接口形态**(将来 MoE 落地时启用,现在只留形状):

1. **词汇下沉**:`BlockId` / `BlockClass` / `BlockDesc` / `Tier` 四个小类型放 **`owl-iface`**
   (它已经是"块句柄 + 契约"的家,`TensorOps` 也依赖它)⇒ **models ⊥ store 各自依赖词汇,
   互不依赖**,engine 是唯一装配点。避免"store 要吃 models 几何 + models 依赖 store"成环。
2. **类轴 + 每类策略对象**:`BlockClass { KvPage, GdnRow, Expert, DenseCore }`(现在只立枚举与分发;
   第一个非 KV 类真接进来时填策略)。
3. **搬运面留"多块在飞"的形状**:`prefetch(ids) -> Lease` + `lease.wait()`,而不是一次一块 await。
4. **权重预算拆分**:`weights_bytes` → `core_weights_bytes`(常驻不可驱逐)+ `expert_pool_bytes`
   (可分层,驻留量由策略定);per-turn 的"地板"语义将来给专家层一个对应物(`expert_floor`)。
5. **缺页路径**:先做**响亮断言**(前向开始前 engine 保证工作集驻留;缺 = 拒),
   要真做 fault 再引入 `BlockProvider` trait(声明在 iface,实现在 store,engine 注入)——
   fault 会把决策权从 engine 漏回 models,与本分工冲突,故非必要不开。

### 10.7 刀序更新(A 轴;形态按 §10 收敛)

| 刀 | 内容 | 状态/门 |
|---|---|---|
| **S3-3** | 立 `BlockLeg`(账 + 宿主物理 + 搬运口)+ **注册制**(id 级 + `epoch`)+ 新批搬运面 + engine 调用点收敛到 leg 方法 | 待做;**迁移零行为变化**(`host_bytes = 0` 默认关层);门 = engine 36/store 51/0.8B 场景 |
| **S3-3b** | 开宿主面(某个档 `host_bytes > 0`):压力先下沉、命中面 `slot = None` 段边界回灌 | 门 = 往返逐字节一致(真 face)+ 命中率/TTFT 改善可测 |
| **P1**(并行小刀) | E2c 快照迁 `CopyBatch`(72/192 次 await → 1) | 纯收益、零架构改动;门 = 快照恢复值逐字节一致 |
| **S2 / S2b** | 状态腿三层化 + GDN 行借还 | 见 §6;落定后立 `Rel`(关系层) |
| **S5/S6+** | 草稿池 / 权重账统一 / **MoE 专家分层**(§10.6)/ 磁盘层 | 各自独立轴 |

### 10.8 待拍(设计会话遗留)

1. 词汇下沉到 `owl-iface`(`BlockId`/`BlockClass`/`BlockDesc`/`Tier`)—— 认否?
2. `BlockClass` + 每类策略对象:现在就立枚举,还是等第一个非 KV 类落地?(建议现在立)
3. 权重账统一(engine 装配点注册权重块,只为报告/预算对账,不动所有权)—— 现在做否?
4. 批面形状:两条(`DtohBatch`/`HtodBatch`)还是一条带方向?(建议两条)
5. 迁移顺序:P1(E2c 快照批化)与 S3-3 并行,可以吗?
6. `Disk` 载体:现在只留 enum 位 + 接口,还是这轮落最小文件层?(建议留位)

---

## 11. S3-3 第①层实录(BlockLeg + 注册制)· 2026-10-16

> 目标:把 `KvStore` 收成 store 内部件,engine 侧只认 **一条腿**(`BlockLeg`),
> 并把"块 → 设备片段"的映射改成**注册制**。**迁移零行为变化**(`host_bytes = 0` 默认关层)。

### 11.1 落点

| 面 | 内容 |
|---|---|
| `store::leg` | `BlockClass`(类轴)· `Carrier`(三层载体,身份正交)· `PieceSpec`/`PoolLayout`(id 级布局 + `epoch`)· `LegConfig`/`LegOps`/`LegStats` · **`BlockLeg`**(控制面:`match_prefix`→`ChainHit`/`ensure_for_len`/`register_chain`/`release_table`/`incref`/`evict_candidates`/`carrier_of`/`report`)· **数据面自由函数** `offload_block`/`onload_block`(显式收 `&mut D`) |
| `store::moves` | **`download_span` / `upload_span`**(带偏移;老口 `offload_to_host`/`onload_from_host` 改为其特例 —— 一套实现) |
| `store::kv` | `alloc_slot` / `bound_slot` / **`ensure_offloadable`**(换出前置:存在 + 设备绑定 + 引用 = 1) |
| `owl-iface` | **`dtoh_at`**(带偏移 D2H;默认 = 响亮"未实现") |
| `owl-cpu` | `dtoh_at` 实现(store 的搬迁单测跑**真 face**) |
| `engine` | `SessionBank { leg_target, leg_mtp: BlockLeg }`;`BankPlan` 加 `mtp_block_bytes`/`host_bytes`;boot 注册布局(`StatePool::pool_layout`/`mtp_pool_layout`,id 级,`epoch = 0`);`EngineKnobs.host_cache_mb`(`OWL_HOST_CACHE_MB`,默认 **0 = 关层**);删掉 session 侧的两条自由函数包装 |

### 11.2 门(全绿)

- `cargo test -p owl-store`:**55/55**(新增 4 条腿测试:①真 face 上**换出→换回逐字节一致** ②关宿主层第一枪响亮拒 ③活跃面(引用 > 1)响亮拒 ④关层时压力驱逐不碰宿主;老 51 条不动)
- `cargo test -p owl-engine --lib`:**36/36**
- 0.8B 套件(`spec_identity_gate` 之外)**12 场景 OK** —— 与迁移前**同一画像**(唯一红 = §45 那条既存机制,且错误文案现在自带链名:`target 链:…`)
- `apps/server` 二进制照常编译;两侧 clippy 改动面零告警

### 11.3 阻塞点与解消(2026-10-16 同日第二刀)

1. ✅ **`dtoh_at` 已解**(原阻塞①):`Command::DtohAt` + server 侧把偏移并进
   **唯一执行路径**(`try_dtoh(stream, id, off, want)`;老 `Dtoh` = `off = 0` 的特例 ——
   老命令迁到新路径上)+ `GpuClient::dtoh_at` + CPU 面实现。
   门:`crates/backends/cuda/tests/dtoh_at_probe.rs`(3080 实机)——
   前缀读回归 / **中段** / 尾段 / 末字节逐字节一致 + 越界**响亮拒**(错误原文带块 id 与区间)。
2. ✅ **批面已解**(原阻塞②):`Command::DtohBatch`(N 段一次命令,回执按序拼接)+
   契约面 `dtoh_batch`/`htod_batch`(默认 = 逐件循环,语义等价;GPU 面覆写为单命令,
   H2D 侧复用既有的 `HtodChunks`)+ **D2H 序口抽出**(`d2h_ordered_stream`,三命令共享一处
   跨流保序);store 的换出/换回**已切到批命令**(一次往返/块)。
   门:同文件 `dtoh_batch_and_htod_batch_round_trip`(带偏移批量写 → 批量读回逐字节一致;
   与单段读对照;越界段整条命令响亮拒)。
   挂账(不阻塞):批读回执仍是 server 侧 `Vec<u8>`(客户端再拷进宿主 slab)——
   将来可改"**写进调用方缓冲**"省一次拷贝(需命令里带宿主指针,另议)。
3. **`Rel` 里有重复面**(非阻塞,但欠一刀):`Rel` 自持 `KvStore + HostTier + TierLedger` 与
   自己的 `offload/onload`(整块口径),与 `BlockLeg` 的(带偏移、注册制)重复
   ⇒ 下一刀把 `Rel` 收成 `{ leg: BlockLeg, state, admission, budget, policy, evict }`,删掉重复实现
   与重复测试(腿侧已有等价覆盖)。
4. **布局代数目前恒 0**:池在 boot 后不变,够用;若将来 dump 页/量化档在运行期变
   ⇒ 重注册 + `epoch += 1`(API 已备好,调用点要补)。

### 11.4 下一刀顺序(建议;① ② 已解)

**S3-3b 开宿主面真跑换出/回灌**(门 = 往返逐字节 + 命中率/TTFT 改善)→ `Rel` 收编(解③)→
P1(E2c 快照迁 `copy_batch`,纯收益)→ 批读直写宿主缓冲(省一次拷贝)。

---

## 12. 需求 / 供给 / 约束(淘汰机制的公理;A 轴口径 = B 关)· 2026-10-16

> 起因:S3-3b 实测卡在"**谁该被丢**"上 —— 而任何淘汰机制若不先把需求侧说清,就是在猜。
> 本节的立场:**先把"需求方要什么、我们手上有多少"写成式子,再谈分配**。
> 口径:**B(KVMem 选择层)关** ⇒ 不做窗口压缩 ⇒ **活跃 turn 的工作集必须整块 dense 驻留**。

### 12.1 需求侧(四项,各有来源)

| # | 需求项 | 式子 | 现状来源 | 可动性 |
|---|---|---|---|---|
| ① | **活跃 turn 工作集** | `D_active = Σ_{i∈active} ceil(N_i / 32)` 块 | `sessions.block_table.len()`(每活跃会话) | **不可动**(硬) |
| ② | **缓存叶(前缀资产)** | `D_cache = 有内容身份、当前无活跃引用的块数` | `KvStore.cached_blocks − 活跃持有` | **可下沉 → 可驱逐** |
| ③ | **待准入增量** | `Δ_j = ceil(N_j/32) − 命中块数` | 新 turn 的 prompt 与命中面(`match_prefix`) | 可拒/可排队 |
| ④ | **运行固定面** | 权重 / GDN 行 / 快照池 / 草稿池 / 图 slab | §9.6 实测表 + `BudgetInputs` | 不可动 |

**需求的驱动量**(用户点题):并发数(**活跃 turn 数** `#active`)与 §④ 的档位;而
**每 turn 的需求 ∝ 它自己的上下文长度 `N_i`**(B 关 ⇒ 不压缩)。
⇒ 需求不是常数,是 `(#active, {N_i}, 命中率)` 的函数 —— 与 §2.2 的分界点公式同源。

### 12.2 供给侧(三层,显式)

| 层 | 容量 | 现状旋钮 |
|---|---|---|
| 设备 | `S_dev = nb` 槽(槽 = 32 token 页) | `pool_tokens` / `vram_target`(治理夹) |
| 宿主 | `S_host = host_bytes / slot_bytes` | `host_cache_mb`(**0 = 关层**) |
| 盘 | `S_disk`(S6;位留着) | — |

### 12.3 约束层级(A 轴 ⇒ 只有一条硬约束)

```
硬:  D_active ≤ S_dev                              ← 活跃面必须 dense 驻留(B 关)
     #active ≤ max_active_turns(GDN 行数)          ← 资源归属律 §2.3
软:  D_cache 驻留 ≤ S_dev − D_active(其余在宿主/被逐)
     D_cache 总量 ≤ S_host 驻留 + ∞(被逐 = 丢内容)
判据: 准入(turn_j) ⇔ D_active + Δ_j ≤ S_dev(先做 ② 的下沉/驱逐腾位;仍不足 = 拒/排队)
```

**次序(唯一正确顺序)**:
1. **先下沉**(`D_cache` 的叶 → 宿主;`S_host` 有位就搬,内容不丢);
2. **再驱逐**(宿主满 ⇒ 丢最老叶;内容真丢,下次重算);
3. **仍不足 ⇒ 拒/排队**(**不许**动活跃面;抢占 = 把整个活跃 turn 的工作集下沉 —— 需 S2b 的状态腿配合,归 A 轴后续)。

### 12.4 由此定下的实现(替换"固定水位"的临时法)

**按需保洁(demand-driven)**:需求由 `schedule()` 早已算出(`ensure_for_len` 的 target),
所以保洁不该用 `nb/8` 这种常数水位,而应:

```
段边界:need = 本步/本轮要长的槽数(schedule 已知)
        free = S_dev − D_active − D_cache(设备侧)
        if free < need ⇒ 下沉 (need − free) 个缓存叶(引用 = 1 的);
        仍不足 ⇒ 驱逐;仍不足 ⇒ 该 turn 响亮拒(真容量不足)
```
落点:`pump()` 段边界调 `pool_housekeeping(need)`(need 由 schedule 的同一份算式给出),
而不是等 `ensure_for_len` 自己在同步路径里**先丢内容**(实测教训:小池下条目被撕,
回灌永远等不到命中)。

**这也解释了 S3-3b 实测的 `matched=0`**:在位的水位太靠后 ⇒ `ensure_for_len` 的同步驱逐
先一步把整条链**条目**(连宿主副本)撕掉;按需保洁把"下沉"提到"增长"之前即可。

---

## 13. 两本账:物理账(server)× 语义账(store)· 2026-10-16

> 用户定调:**server 的显存账本只按"指针/块"统计 —— 它不管哪块是哪块、是干什么的;
> "哪一块显存是什么、干啥用、归谁"是 store 的事**。store 记不清这个,调度就无从谈起。

### 13.1 分工(同一个字节,两种视角)

| | **物理账(server)** | **语义账(store)** |
|---|---|---|
| 记账单位 | 块(id/指针):`bytes / dtype / 地址` | **语义对象**:内容身份(ChainHash)/ 布局位置(层×槽) |
| 关心 | 池 `used/reserved`(含高水)、驱动 `free/total`、分配成败 | **用途**(BlockClass)、**载体**(设备槽/宿主槽/盘)、**引用**(谁还持着)、**新鲜度**(LRU/热度) |
| 认识 | 只认识"块" | 只认识"东西";**不持指针**(裁决⑫) |
| 现状落点 | `block_ledger` / `pool_ledger_snapshot` / `vram::{live,free,total}` / **硬预算 `vram_budget_guard`** | `KvStore`(身份+槽账+引用+LRU)、`HostTier`(宿主物理)、`BlockLeg`(布局注册 + 载体 + 报告) |

一句话:**server 管"花了多少、还剩多少";store 管"花在谁身上、该不该继续花"。**

### 13.2 store 必须记的六件(逐条对齐现状)

| # | 记什么 | 用途 | 现状 |
|---|---|---|---|
| ① | **身份**:每块的内容身份(ChainHash) | 跨会话命中、快照键 | ✅ `KvStore` 内容身份前缀树 |
| ② | **位置**:它在哪条腿的哪一层哪一槽(`PoolLayout` 注册,id 级) | 搬运时算片段偏移 | ✅ `BlockLeg::attach_pool` + `epoch` |
| ③ | **载体**:设备槽 / 宿主槽 / 盘 | 决定"能不能搬、搬到哪" | ✅ `Carrier`(设备/宿主/盘位) |
| ④ | **用途**:KV 页 / GDN 行 / 专家 / 稠密核 | **策略分流**(LRU vs 热度 vs 永不驱逐) | ⚠️ 枚举已立(`BlockClass`),现在**腿级**;真接非 KV 类时升**块级** |
| ⑤ | **引用**:谁还持着(活跃 turn / 缓存 / 快照) | 判"可不可以动"(`Busy` = 活跃面) | ✅ `SlotPool.refs` + `ensure_offloadable` |
| ⑥ | **新鲜度/热度**:last_use、(将来)检索/命中频次 | 出驱逐序 | ✅ `last_use`;热度面 = MoE 落地时接(§10.6) |

### 13.3 由两本账得到的**可调度性**(接 §12)

有了 §13.2,§12 的两项需求就能**现算**:

```
D_active = 活跃 turn 的工作集     ← engine 侧(会话块表)⊕ store 的引用账(⊕ = 对齐校验)
D_cache  = 有身份、无活跃引用的块 ← store 的条目账(含载体分布)
"丢谁"   = (用途 ④ → 引用 ⑤ → 新鲜度 ⑥)三段判据 —— 不再是"最老的就丢"
```

### 13.4 对账口(G2 账目封闭;新立)

两个视角必须能**互相验证**(不许各说各话):

```
store 报:按类 × 按层的块数/字节(KvPage: n 块 × slot_bytes;host: m 块 …)
server 报:块账房 used/reserved(/debug/blocks)+ 池高水
对账:store 的"设备驻留字节" ≤ server 的"池 used";差额 = 非 store 管的块
      (权重/图 slab/草稿/scratch —— 归 §9.6 固定面表)
不一致 = **响亮报**(打点 + WARN),不许静默
```

**落地清单**:①`LegStats` 加"按类 × 按载体"的明细行;②engine 侧加一条 `/debug/blocks`-风的
对账打印(store 账 × server 账 × 固定面表);③`BlockClass` 升块级(第一个非 KV 类接进来时)。

### 13.5 与 §12 的施工绑定

§12 的"按需保洁"要实现,依赖的正是本节:①**need**(需求)由 engine 的 schedule 给;
②**"下沉谁"**由 store 的语义账给(用途 + 引用 + 新鲜度);③**搬没搬动**由 server 执行并回账。
⇒ 下一刀 = `pool_housekeeping(need)`(需求驱动)+ §13.4 的对账口。

---

## 14. 引擎钩子框架(engine hooks)—— store 的插入语义 · 2026-10-16

> 用户定调:engine 里存在**许多特殊的点位**,在这些时间点上可以触发 store 的调度。
> **store 的插入语义 = "此时我们应该怎么办?给你一个需求,你告诉我:谁应该离开、谁应该进来。"**
>
> ⇒ 钩子框架 = ①把那些点位**显式登记**(不再散落成 if);②每个点位两相——**判(纯)与搬(段边界)**;
> ③store 的产物是一张**决策单**(谁走/谁进/拒),搬运由 server 执行。

### 14.1 每个钩子两相(与三分工严格对齐)

```rust
/// 相一:**判**(纯 host、零 I/O、可单测) —— 在 schedule 的决策面调用
pub fn plan(leg/rel, need: Need, protect: &Protected) -> Verdict;   // "谁走、谁进、还是拒"

/// 相二:**搬**(段边界、async) —— **由 engine 编排**(§21):
///   store(纯 host):plan_offload/plan_onload(算片 + 借台)· commit_*(回账)· abort_*(回滚)
///   engine:         取计划 → `face.swap(dir, &pieces, conn)` → reclaim → commit
```

- **判**可以在任意时刻(包括图捕获窗内)调用 —— 它只动账(§13.2 的六件);
- **搬**只在段边界:piecewise 接缝 / pump 间隙 / prefill 分块之间(律③);
- 两相之间**状态可变**(判完到搬完之间可能有别的 turn 动账)⇒ **取计划时必须重验**
  (`plan_*` 的 `Busy` = 跳过该条,不是错误 —— 见 §21)。

### 14.2 决策单(Verdict)

```rust
pub struct Verdict {
    pub out:  Vec<EvictOut>,   // 谁应该离开:下沉(宿主有位)或驱逐(宿主满)
    pub in_:  Vec<ChainHash>,  // 谁应该进来/留下:需物化的回灌清单 + 建议保留的缓存块
    pub deny: Option<String>,  // 谁都不该走/进:响亮拒(含具体数字,§12.3 次序走完仍不足)
}
```
`out` 的判据链 = **用途 ④ → 引用 ⑤ → 新鲜度 ⑥**(§13.3),不再"最老的就丢";
`in_` 的两个来源:**命中面**(S3-3b 的 `pending_host`)+ **准入建议**(值得留的热资产)。

### 14.3 钩子点(engine 侧的第一版清单)

| 钩子 | 点位(engine 落点) | 触发 | 已知量(需求) | 可动面 | 合法性 |
|---|---|---|---|---|---|
| **H1 turn 准入** | `schedule()` 起手(submit 后) | 新 turn 要进 | `Δ_j = ceil(N_j/32) − 命中块数` | 缓存叶(全部) | 判 = 自由;搬 = 段边界 |
| **H2 块表增长** | `ensure_for_len` 之前(prefill chunk 间 / decode 跨页 / spec verify 行) | 要长 `need` 个槽 | `need`(本步/本轮) | 缓存叶 | 判 = 自由;搬 = 段边界 |
| **H3 turn 收口** | `exec/turn_end`(register_chain 前后) | 链转入缓存资产 | `落账块数`(负需求:供给↑) | 缓存叶 | 段边界 |
| **H4 前缀命中** | `execute_begin`(write_bt 之前) | 命中面含宿主格 | `pending_host.len()` | 该链自身(回灌) | **段边界**(必须在 write_bt 前) |
| **H5 空转整理** | `pump()` 得 `Idle` 时 | 无活跃 turn | 全局(把冷资产下沉、热资产留下) | 缓存叶 + 快照 | 段边界 |
| **H6 引用归零** | `session.close/burn`、LRU 逐出会话 | 可动资产面变化 | — | 该会话链 | 判 = 自由 |
| **H7 状态腿事件** | GDN 行借还 / 快照下沉(S2/S2b) | 状态腿三层化落地后 | 行/快照 | 快照池 | 段边界 |

**第一版要做**:H1/H2/H3/H4(H5 idle 整理是纯收益,可紧随其后;H6 是记账维护;H7 归 S2)。

### 14.4 与 §12/§13 的关系

```
§12 需求侧 → 决定"need"(H1/H2 各给一份)
§13 语义账 → 决定"谁可动、谁该留"(plan 的输入)
§14 钩子   → 决定"何时问、何时搬"(框架)
```

### 14.5 落地清单(第一刀)

1. `store` 出 `plan(need, protect) -> Verdict`(**纯判定**,覆盖 §12.3 次序:下沉 → 驱逐 → 拒)
   与 `plan_offload/plan_onload + commit_*/abort_*`(**纯 host** 的搬运计划与记账);
   `apply` 的**编排在 engine**(§21);
2. engine 侧把现有触发点**改经钩子**(H1/H2 现在各有一处"临时法":`pool_housekeeping(nb/8)` 与
   `ensure_for_len` 的同步驱逐)⇒ 全部走 `plan`/`apply`;
3. 打点:每个钩子的 `need / out.len() / in_.len() / deny / 实际搬了多少`(metrics);
4. 门:演示台(0.8B + 池 8 槽 + 宿主 8MiB)①换出>0 ②**回灌>0** ③命中路径 vs 冷启动 **token 恒等**。

### 14.6 ⚠️ 链级卸载律(2026-10-16 实测定谳;修正 §14.3 的候选粒度)

**现场**:演示台(0.8B + 池 16 槽 + 宿主 8MiB)第 4 轮重复前缀,`matched=0`;钩子诊断显示
H5 每次只停 **1 块**,而宿主里的那 1 块是**某条链的尾块**。

**根因**:前缀树里**只有叶可动**(逐父节点会孤儿化子链)—— 所以"按 LRU 逐叶下沉"停掉的
永远是**链尾**。而**前缀匹配是从链头开始走的**;链头不在设备上 ⇒ 整条前缀不可复用
⇒ 停在宿主的尾块对复用**零贡献**,命中永远落在设备上,回灌永不触发。

**律(A 轴新增)**:
1. **下沉的粒度 = 链(前缀),不是叶**:卸载一条链时必须**从链尾向链头逐块搬完**
   (或至少搬到一个"可复用的前缀边界"),不许停在半截 —— 半截链 = 不可复用的碎片;
2. **驱逐(丢内容)仍可按叶**:丢尾部不动链头,语义上"前缀短了一截",但前缀仍可复用
   (这正是 LRU 逐叶在**驱逐**面上正确的原因);
3. 由此,候选面必须**分两种**:
   - **驱逐候选**:叶(现行 `evict_candidates`)✔
   - **卸载候选**:**链**(取"最冷的链",整条搬;判据 = 链尾的 `last_use`)
4. 推论:**"停在什么状态下可复用"是前缀层的概念** —— 与 §12 的准入、§13 的语义账同源:
   store 记的 ⑥(新鲜度)在链级要按**链尾**取,而不是按块取。

**修法(下一刀)**:`BlockLeg` 加 `chain_offload_candidates(protect, limit) -> Vec<ChainHash>`
(按链尾 LRU 出**链**)+ `offload_chain(face, chain)`(从尾到头整条搬);
H5 用它;H2 的"按需下沉"同样用链(不足一条链时退化按叶,并打点)。

---

## 15. Swap 原子命令 + Connector 契约(iface 声明 / store 实现 / server 使用)· 2026-10-16

> 用户定调:①swap **要在 GPU server 上是一条(原子)命令** —— 直接用命令让 server 去做交换;
> ②server 为此需要一个**接口**,接口**在 iface 里声明**,它接受一个 **`Connector`**
> —— 可以是 **RAM connector**,也可以是 **disk connector**;
> ③一条命令就能把"显存里的东西换出到 RAM/盘、再从那里换回来";
> ④**Connector 的实现者是 store,使用者是 server**。

### 15.1 归属矩阵(这条设计的关键)

| 物 | 声明 | 实现 | 使用 |
|---|---|---|---|
| `Connector` trait(+ `SwapDir` / `SwapPiece` / `ConnSlot`) | **owl-iface** | **owl-store**(RAM = `HostTier`;disk = 文件层) | **owl-cuda 的 server**(搬运时逐片读/写) |
| `DeviceClient::swap(...)`(一条命令的入口) | **owl-iface** | owl-cuda(GpuClient)+ owl-cpu(循环等价) | **owl-store**(`apply` 里调) |
| 交换的**决策**(谁走谁进) | — | owl-store(`plan`/`Verdict`) | owl-engine(钩子) |

**为什么要这样切**:server 是**唯一持裸指针**的一层(裁决⑫)—— 只有它能把设备字节搬到
别处;而"别处是什么"(RAM/盘/未来的 NVMe-oF)**不该让 server 知道** ⇒ 抽象成 `Connector`,
server 只按槽号读写,**层级知识留在 store**。

### 15.2 契约形状(伪代码)

```rust
// ── iface:声明 ──────────────────────────────────────────────
/// 宿主/盘侧的一块存储槽(store 分配,server 只按号读写)
pub struct ConnSlot(pub u64);
pub enum SwapDir { Out, In }                     // Out:设备→connector(换出);In:反向(换回)
pub struct SwapPiece { pub block: u64, pub off_bytes: usize, pub len_bytes: usize, pub slot: ConnSlot }

/// 交换的"另一端"(RAM / 磁盘 / …)
pub trait Connector: Send {
    /// 写入一片(设备→该层;bytes 由 server 从设备侧拷来)
    fn put(&mut self, slot: ConnSlot, off_bytes: usize, bytes: &[u8]) -> Result<(), ModelError>;
    /// 读出一片(该层→设备)
    fn get(&self, slot: ConnSlot, off_bytes: usize, out: &mut [u8]) -> Result<(), ModelError>;
    /// 层名(诊断/报告面)
    fn tier_name(&self) -> &'static str;
}

// ── iface:DeviceClient 上的一条命令面 ───────────────────────
/// **swap = 一条原子命令**:N 片一次提交、一次 ack(server 侧按器件序逐片搬);
/// 捕获窗内 = 结构化拒(时序归 engine,§10.3)。
fn swap(&mut self, dir: SwapDir, pieces: &[SwapPiece], conn: Box<dyn Connector + Send>)
    -> impl Future<Output = Result<Box<dyn Connector + Send>, ModelError>> + Send;
```

- **Connector 随命令进出**:`Box<dyn Connector + Send>` 移入命令、回执时**原箱奉还**
  ⇒ 单主、零锁(store 在两批 swap 之间持着它);
- **原子性**:一条命令 = 一次 ack = 全部片段完成(同流保序);中途失败 = 整条失败,
  由 store 回滚账(引用/载体不变 ⇒ 不留半态);
- **层级无关**:RAM 与 disk 只差 `Connector` 的实现;server 侧代码零分支。

### 15.3 与现有面(dtoh_at / dtoh_batch / htod_batch)的关系

```
现在:S3-3b 用 dtoh_batch(设备→内存缓冲)+ htod_batch(内存→设备)——**两条命令、两趟**,
      且"宿主缓冲"由 client 侧拷贝进 store 的 Vec(多一次 memcpy,§11.3 挂账)
将来:swap(dir, pieces, conn) —— **一条命令一通**,Connector 直接落盘/落 Vec,
      省掉 client 侧的中间缓冲与第二次 memcpy
```
⇒ 迁移纪律(§10.5 同款):**新面先加、老面保留**(`dtoh_at`/`dtoh_batch` 继续可用),
store 的 `apply` 里逐刀切到 `swap`;CPU 面给 `swap` 的**循环等价实现**(任何 face 都能跑)。

### 15.4 磁盘层的现实问题(先记着,别踩)

- RAM connector:`put/get` = memcpy ✔ 无阻塞问题;
- **disk connector 的 I/O 是阻塞的** —— 在 server 命令线程里做文件写会**钉死整条命令管道**
  (与 dtoh 早期"整流排空"同一个病)。⇒ 二选一(磁盘层落地时拍):
  ①**专用 I/O 线程 + 队列**:命令把片段投递给 I/O 线程,ack 语义 = "已提交",
    完成由 store 轮询/回调(交换与计算重叠,符合 §10.6 的 MoE 预热形态);
  ②`io_uring` 等异步面:ack = 真完成。
  **v1 只做 RAM**;disk 层按 S6 立项时按 ① 落(它天然支持"多块在飞")。

### 15.5 落地清单

1. `iface`:`ConnSlot`/`SwapDir`/`SwapPiece`/`Connector` + `DeviceClient::swap`(默认实现 =
   逐片 `dtoh_at`/`upload_pinned` 循环,语义等价 ⇒ 老 face 不改也能跑);
2. `owl-cuda`:`Command::Swap { dir, pieces, conn, ack }` + server 侧一处 handler
   (逐片 `memcpy_*` + `conn.put/get`;捕获窗落 reject 臂)+ `GpuClient::swap`;
3. `owl-store`:`HostTier` 实现 `Connector`(RAM);`apply` 改用 `face.swap` ——
   **引擎侧零改动**(钩子接口不变);
4. 门:store 单测(CpuFace 的循环实现)+ 演示台(`OWL_OFFLOAD_PROBE=1`)。

---

## 16. 第三刀实录:链级卸载 + Swap 单命令(2026-10-16)

> 一刀两事,同一现场:**§14.6 的律落地**(链级下沉)+ **§15 的契约落地**(swap 单命令)。
> 期间翻掉一个**假现场**(诊断口径 bug),修掉一个**静默丢内容**的驱逐 bug。

### 16.0 一句话

```
沉链不沉叶(链头段进宿主 ⇒ 命中面从链头就是 host 格 ⇒ 回灌真触发);
搬一趟不漏(swap 单命令 = N 块 × 每层片段,一次 ack;回执 = 连接器原箱奉还)。
```

### 16.1 现场翻了案:旧诊断口径错(§14.6 的理由要修正)

诊断用 `tok.encode(prompt)` 算命中面,而 `submit_session` 内部是 `tok.encode(chat_wrap(prompt))`
⇒ 哈希链从第 0 块起就不同 ⇒ 打印出来的 `matched=0` 是**假的**(同时刻生产路径命中 2 块)。

修正口径后现场(0.8B + 16 槽池 + 宿主 8MiB,三轮后):

```
[s3-3b] 诊断 链0:tokens=89 matched=2 pending=0 · 0:dev|snap=Y 1:dev|snap=Y
[s3-3b] 诊断 链1:... 全 dev · 链2:... 全 dev
[s3-3b] 三轮后:缓存 9 块 · 宿主 1 块 / 0.38MiB · 换出 1 次
```

⇒ 每条链实际是 **3 块**(2 块前缀 + 1 块生成段);旧 H5 只按 `short=1` 沉 **1 个叶**,
而那个叶是**生成段碎块** —— 不在 89-token 命中面里 ⇒ `pending` 恒 0 ⇒ 回灌永不触发。

**律(§14.6)仍然成立,理由精确化**:

> 命中面**只覆盖完整前缀块**;要让"下沉"对复用有贡献,沉下去的那部分必须**落在命中面里**
> —— 而命中面从链头开始 ⇒ 必须把**链头段**一起沉。逐叶下沉只会沉到链尾(最新生成段),
> 那正是命中面之外的部分。⇒ 下沉粒度 = **链**(尾→头整条搬);驱逐(丢内容)仍可按叶。

### 16.2 §14.6 落地(store 侧)

| 新增 | 作用 |
|---|---|
| `KvStore::chain_down_span(tail) -> Option<usize>` | 整条链可搬评估:逐块查设备引用 = 1;全链在宿主 ⇒ `None`(不出空票) |
| `BlockLeg::chain_offload_candidates(protect, budget)` | 按**链尾 LRU** 出**整条链**(`ChainCand{tail, head, dev_blocks, blocks, last_use}`) |
| `offload_chain(leg, face, tail)` | 先整条校验(`Busy` = 零副作用拒),再**一条 swap**搬完;已在宿主的块幂等跳过 |
| `PrefixTree::evict_device` | **腾槽语义**:只逐设备叶(host 叶跳过 —— 见坑册 §48) |

`hooks::plan` 次序(A 轴,§12.3 不变,粒度细化):

```
① 链级下沉(host 有位;株链长度可能**超出** need —— 粒度就是链,打点照记)
② 叶级下沉(没有可整条沉的链时退化;同律附则)
③ 叶级驱逐(host 满;丢内容)
④ 拒(响亮,含具体数字)
```

`EvictOut` 加 `span`:`span = 1` 单块;`span > 1` = 整条链(`chain` = 链尾)。

### 16.3 §15 落地(三处改动 + 契约补记)

| 层 | 落点 |
|---|---|
| iface | `ConnSlot` / `SwapDir` / `SwapPiece{block,dev_off,conn_off,len,slot}` / `Connector{put,get,tier_name,unbox}` + `DeviceClient::swap`(默认实现 = 逐片 `dtoh_at`/`alloc_pinned+upload_pinned` 循环,**语义等价**) |
| owl-cuda | `Command::Swap` + `handle_swap`(`swap_out`/`swap_in`)+ `GpuClient::swap` + 五处路由(段计时表 / `cmd_stream` / dispatch / closed / **捕获窗 reject**) |
| owl-store | `SlotTable`(Connector 的 RAM 实现)+ `HostTier::{lend_slots, land, recycle_many}`;`offload_planes`(批量一条命令)/`offload_chain`/`onload_block` 全切 `face.swap` —— **store 内部只此一条搬运面** |

**实现补记(对 §15.2 伪代码的修正,详见坑册 §49)**:

1. `SwapPiece` 的偏移**拆成两个**(`dev_off` 设备块内 / `conn_off` 连接器槽内)—— 两侧布局不同构;
2. `Connector::unbox` = 原箱奉还的类型安全取回口(不能给默认实现);
3. **空片段表** = 响亮拒(handler 第一道);
4. 回执语义:`Out` = 拷贝完成 **且** `conn.put` 完成;`In` = 全部落地(H2D 流序 + 末尾单 notify);
5. `swap_in` 的 `conn.get` 在**命令线程同步**做(RAM = memcpy);disk 层按 §15.4 另开 I/O 线程;
6. 错误纪律:任一失败 = 整条失败(未入队 DMA 不投 finish;staging 回池;回执必达)。

### 16.4 门(全绿)

| 门 | 结果 |
|---|---|
| `owl-store` 单测 | **60/60**(新增:链级候选/整条搬逐字节/活跃面零副作用拒/腾槽只逐设备叶/SlotTable 越界) |
| `owl-cuda --test swap_probe` | **2/2**(变长片 + 片序 ≠ 槽序 + 原箱奉还;设备越界/conn 越界/空表三拒) |
| `owl-engine --lib` | **36/36** |
| 演示台 `OWL_OFFLOAD_PROBE=1` | **三门口全过** |

演示台现场(修正后):

```
[s3-3b] 三轮后:自由槽 10/16 · 缓存 9 块 · 宿主 3 块 / 1.12MiB · 换出 3 次 1152KiB
[s3-3b] 诊断 链0:tokens=89 matched=2 pending=2 · 0:host|snap=Y 1:host|snap=Y
[s3-3b] 命中轮回吐:"您提供的文本看起来像是某种**信号"
[s3-3b] 命中轮后:宿主 1 块 / 0.38MiB · 换出 3 · 回灌 2
```

- 门① 换出 > 0:**3 块**(链级整条,一条命令 3 块 × 12 片 = 36 片一次 ack);
- 门② 回灌 > 0:**2 块**(链 0 的前缀段 —— 命中面里,正是"链头段也进宿主"的红利);
- 门③ **命中路径 vs 冷启动逐字节恒等**(断言在演示台里:`text == cold_text`,同 prompt/贪心)。

### 16.5 挂账(下一刀)

1. **回灌批次化**(`onload_chain`:一条命令搬 N 块)—— `execute_begin` 现在逐块 `swap`,
   命中 2 块 = 2 条命令;批面骨架已在(`SlotTable` 多槽 + `in_pieces(base_slot)`),只差 engine 侧收集;
   ✅ **已落(§17.7,同日)**:`onload_blocks` + `LegOps::onload_batches`;演示台
   “回灌 2 块 / 1 命令”。
2. H2 的按需下沉目前仍先走链级、无链可沉时退化叶级(**打点需区分**,现在只记总量);
3. H5 水位 = `nb/2` 仍是临时法(§12.4 的 demand-driven 未落);
   ✅ **已落(§17,同日第三刀)**:改为需求驱动 `step_need`(准入 Δ_j / prefill 增量 /
   decode 上界 / 空转零需求),`deny` 不再整单丢弃,并加 `ops.ensure_evicted` 计数门。
4. `Rel` 收编(去重)+ P1(E2c 快照迁 `copy_batch`,72/192 次 await → 1);
5. disk connector(S6;§15.4 的 I/O 线程形态)。

---

## 17. 需求驱动保洁落地(§12.4 做实)· 2026-10-16

> 上一刀把**搬什么**做对了(链级 + swap 单命令);这一刀把**什么时候搬、搬多少**做对 ——
> 用**真需求**替换常数水位,并把"同步驱逐"从常态降为兜底。

### 17.1 需求的真源(`step_need`,与 `schedule` 同源算式)

保洁发生在 pump 的段边界,需求**不再猜**,而由与调度决策**同一份算式**给出:

| 情形 | need(要长的槽数) | why |
|---|---|---|
| 队列里有待准入 turn | `Δ_j = ceil(N_j/page) − 设备已驻留的命中块数 − 当前表长` | `Admit` |
| 活跃 turn 在 prefill | `ceil(fed + chunk)/page − 表长`(chunk 走 `prefill_step_chunk` 单源) | `Grow` |
| 活跃 turn 在 decode | `ceil(pos + spec 深度上界 + 1)/page − 表长`(**上界是安全方向**) | `Grow` |
| **空转** | **0(不搬)** —— §12.4 明确废弃"总槽/2"这类固定水位 | `Idle` |

**两条关键口径**:

1. **Δ_j 只能减"设备已驻留"的命中块**(§12.1 的修正点):host 命中的块虽然"命中",
   **回灌时照样要设备槽** —— 少算就会在回灌那一刻 `Exhausted`。
   ⇒ `reuse_face(...)` 返回 `(m 可复用块数, dev 其中设备驻留的块数)`,需求侧减 `dev`。
2. **上界安全律**:decode 的 spec 深度用上界(不查降级态)—— 多算只是**多腾几个槽**
   (内容不丢);少算才会退回 `ensure_for_len` 的同步驱逐(**丢内容**)。方向性错误不对称。

### 17.2 算式单源(不许两套规范)

| 算式 | 单源落点 | 消费者 |
|---|---|---|
| prefill 本步块长 | `RunningEngine::prefill_step_chunk(fed, prompt_len, cfg)` | `schedule()` 的 chunk 决策 + `step_need()` |
| 命中面裁剪(末块对齐 / 快照在场)+ 设备驻留计数 | `RunningEngine::reuse_face(pool, hit, prompt_len, page)` | `schedule()` 的复用决策 + `step_need()` |
| 表覆盖目标 | 与 `ensure_for_len(target)` 同一批调用点 | 同上 |

### 17.3 判拒语义修正(deny ≠ 整单丢弃)

旧:plan 的 `deny`(凑不够 need)⇒ `pool_housekeeping` 直接 return(**整单不搬**)。
新:**`deny` = 真容量不足预警** —— 响亮 WARN,但 `out` 里能搬的**照搬**(下沉不丢内容),
真拒绝留给 `ensure_for_len` 的 `Exhausted`(响亮、带数字)。

### 17.4 可见性:兜底驱逐计数器

`LegOps::ensure_evicted`:凡 `ensure_for_len` 内部触发了压力驱逐,就 +N(报告面可查)。
语义 = **"需求驱动保洁不够用"的指纹**;需求驱动到位时它应恒为 0。

### 17.5 演示台重做:冷热交替(真需求)

旧演示台靠"H5 空转水位"造下沉(那正是 §12.4 批评的临时法);新场景**只靠真需求**:

```text
池 = 8 槽(单会话容量下限,page 32)     宿主面 = 8MiB
轮 1  热链 A(89 tok ⇒ 3 块)         need = 3,free 8 ⇒ 不动
轮 2  热链 B(3 块)                  need = 3,free 5 ⇒ 不动
轮 3  长 prompt(7 块)               need = 7,free 2 ⇒ **沉 A+B 整条(6 块)** ⇒ free 8 ⇒ 无驱逐
轮 4  重发 A(命中 2 块,**都在宿主**)  need = 3(dev=0!)⇒ 沉冷链 D 让路 ⇒ **回灌 2 块**
```

现场(0.8B @ RTX 3080):

```
[hook] Admit need=3 free=8/8 判 out=0        ← 轮 1
[hook] Admit need=3 free=5/8 判 out=0        ← 轮 2
[hook] Admit need=7 free=2/8 判 out=2        ← 轮 3:整条沉 A+B
[s3-3b] 轮 3 后:自由槽 2/8 · 缓存 12 块 · 宿主 6 块 / 2.25MiB · 换出 6 次 2304KiB
[s3-3b] 诊断 链A:tokens=89 matched=2 pending=2 · 0:host|snap=Y 1:host|snap=Y
[hook] Admit need=3 free=2/8 判 out=1        ← 轮 4:冷链让路
[s3-3b] 轮 4 后:宿主 10 块 · 换出 12 · 回灌 2 · 兜底驱逐 0
```

### 17.7 回灌批次化(§15 批面收尾;同刀)

`execute_begin` 原本**逐块** `onload_block`(命中 N 块 = N 次 actor 往返);现改
`onload_blocks` **一条 swap 搬 N 块**(单块 = N=1 的糖,同一条路径 —— 律 13)。
`LegOps` 加 `offload_batches` / `onload_batches`(命令次数;块数仍看 `offloads`/`onloads`)。

演示台现场:

```
[s3-3b] 轮 4 后:宿主 10 块 / 3.75MiB · 换出 12 块/3 命令 · 回灌 2 块/1 命令 · 兜底驱逐 0
```

⇒ 换出:每条链一命令(链内 N 块一次 ack);回灌:N 块一命令。门 = 演示台断言
`onload_batches == 1` + store 单测 `onload_blocks_moves_n_blocks_in_one_command`。

### 17.8 门(全绿)

| 门 | 结果 |
|---|---|
| 演示台三门口 | 换出 12 块/3 命令(限整条链)/ 回灌 2 块/**1 命令** / 命中 vs 冷启动**逐字节恒等** |
| **兜底驱逐** | `ensure_evicted == 0` **全程**(需求驱动到位;轮 4 也不靠兜底) |
| `owl-store` 单测 | **61/61**(含批量回灌门) |
| `owl-engine --lib` | 36/36 |
| e2e 场景(0.8B) | `prefix_cache_hit` / `schedule_decision_surface` / `session_continuity` / `multi_session_isolation` 全过 |

### 17.9 挂账

1. **草稿腿的 need 未纳入算式**:`exec/draft.rs` 里 MTP/草稿链的 `ensure_for_len(… + 8)`
   仍是固定余量(它们多为独立池;spec 轮的 target 腿需求已由 decode 上界覆盖);
2. **空转预搬**降级为"不做":要把搬运从关键路径挪走(TTFT 优化)应作为**显式策略**
   (预热/热度预测)立项,而不是恢复常数水位;
3. **B 轴(KVMem 窗口压缩)落地后** `D_active` 会变小 ⇒ 需求算式要跟着改(现在按"活跃面整块 dense 驻留"算);
4. H1(准入)现在只在 pump 段边界触发一次;若将来支持"一次准入多个 turn"(批调度),
   需求要按批合并(算式不变,求和即可)。

---

## 18. CPU mock server 档(无卡 GPU server)· 2026-10-16

> 用户点题:GPU 后端"无非是申请显存、分配、记账"—— **这些动作不需要真动显卡**。
> ⇒ 把 CPU 面升格为"无卡 GPU server":**命令面等价、账目等价、数值可空转**。

### 18.1 落点:不新造 server,补 `CpuFace`

`owl-cpu::CpuFace` 本来就是 `DeviceClient` 的一个**真实现**(store 的 61 项单测跑在它上面),
缺的只是"GPU 的形状"。补齐后它**就是** mock server,而且契约单源(同一个 trait、同一份
捕获/搬运语义)。若另造一套,立刻多出"两套规范"(AGENTS.md 明令禁止)。

### 18.2 补的四件

| # | 能力 | 语义(与 GPU server 对齐) |
|---|---|---|
| 1 | **图捕获/重放** | 捕获窗 = **只录不发**(记 kernel+args+out);`graph_end` 空窗**响亮拒且仍留在窗内**(须 `graph_abort` 收尾);重放 = 依次重算(CPU 无指针烤入);**窗内禁** htod/dtoh/sync/copy/memset/upload(与 GPU 的 reject 表同律 —— mock 不放宽捕获契约) |
| 2 | **命令面补齐** | `upload_and_launch`(装载 cast 写口;按**声明 dtype** 解码 —— F16/BF16 都通)、`block_ledger_dump`(活块投影 + 单调假地址,审计可判定)、`mem_pool_trim` = 0 |
| 3 | **容量墙** | `MockLimits::vram_bytes`:按**声明 dtype** 记账(`alloc(F16,n)` 占 2n —— 不因 mock 的 f32 存储翻倍,否则"卡能装多少"的账失真);越界 = `PoolExhausted`(真卡同一条语义);`free` 回冲 |
| 4 | **计算空转** | `MockLimits::null_kernels`:launch 不路由算子、输出清零后返回 —— warmup / 全前向可"跑得过" |

另修:`Value::from_bytes` 补 **BF16 解码**(bf16 = f32 高 16 位)—— 装载 cast 路径的源头字节。

### 18.3 界线(必须说清,否则会被误用)

- **等价面**:账(分配/释放/容量)、搬运(dtoh/htod/swap/复合上传)、命令序、图捕获契约;
- **不等价面**:数值(空转档输出清零 ⇒ 生成内容是垃圾,但 token 合法不崩);**性能数不可用**。
- ⇒ mock 档的用途 = **结构与压力测试**(boot 拒启 / 容量不足 / 准入拒绝 / 搬运往返 / 钩子行为),
  不是跑分,也不是正确性验证的替代品。

### 18.4 门(全绿)

| 门 | 结果 |
|---|---|
| `owl-cpu` 单测 | **5/5**(图录放幂等 + 空窗/嵌套/窗内搬运三拒;容量墙 + 注入 + 声明口径;BF16 复合上传;账本投影;空转档) |
| **`cpu_mock_smoke`**(engine 跑在 CpuFace 上) | **boot / 装载 / warmup / 图捕获 / turn 全链无卡通过** |
| 容量墙拒启 | `PoolExhausted: cpu-mock(容量墙) 需 508559360B 余 1048576B`(**自带归因**) |
| 回归 | store 61/61、engine lib 36/36 不变 |

跑法:`OWL_CPU_MOCK=1 cargo test -p owl-engine --test cpu_mock_smoke -- --nocapture`
(默认 skip:0.8B 装载在 CPU 面上约 70s,不进默认套件)。

### 18.5 挂账

1. **演示台/offload_probe 的 CPU 档**:e2e 场景体已泛型(`RunningEngine<D>`),但装配入口
   (`Engine::new` → GpuClient)与快照/GDN 门需按 CPU 面补齐 —— 补完即可"无卡跑 store 三门口";
2. **故障注入扩展**:swap/搬运失败注入(现在只有分配注入);
3. 若要当 CI 常驻档:装载 70s 需提速(CPU cast 路径)或做装载产物缓存。

#### 18.4.1 补记:CPU 档装载耗时(实测账,免得后人当 bug 查)

| 口径 | 0.8B 装载耗时 | 数值 |
|---|---|---|
| 改造前(旧 `upload_pinned` 不管声明 dtype,按 f32 位模式解) | ~25s | **错的**(BF16 字节被当 f32 读 ⇒ 权重垃圾;既有 CPU 测试只查层数/tokenizer,故"通过") |
| 改造后(按声明 dtype 解码:BF16 → f32 位补齐) | ~65s | 正确(有 `upload_and_launch_decodes_declared_dtype` 门) |

⇒ 这 40s 是**修正数值的代价**,不是回归。优化挂账:装载产物缓存 / 批量位转换(SIMD)——
若要让 mock 档做 CI 常驻,先做这两件。

#### 18.5.1 挂账①实施细则:CPU 档跑"引擎真前向"的三个缺口(2026-10-16 探路停)

目标(推①)= 让 store 演示台(`offload_probe`)**无卡**跑同一套三门口。
基础设施已就位并入库:场景体与 face 解耦(`offload_probe_scenario<D>`)、
旋钮共用(`offload_probe_config`)、无卡入口(`OWL_CPU_MOCK` ⇒ `e2e_body` 只跑 CPU 档)、
`cpu_mock_offload_probe` 场景。**卡在 engine 侧的三处口径/能力缺口**:

| # | 缺口 | 现象 | 修法(独立刀) |
|---|---|---|---|
| 1 | **回读字节口径**:engine 部分 `dtoh` 调用按**模型 dtype** 算字节,而 CpuFace 的存储/布局是 **f32 口径** | `dtoh: 块 2964 字节 131072 != 收割 65536` | 让"设备字节口径"成为 face 的**自述能力**(如 `DeviceClient::block_byte_scale()`),engine 按它算;或 CPU 档统一 f32 |
| 2 | **分页策略依赖 dtype**:`kv_paged_policy(dtype)` 对 F32 无策略 ⇒ 落 legacy 臂 | 试把 CPU 档 dtype 归一为 F32 ⇒ `attention.rs` 的"v1 臂已退役"panic | 分页策略与执行域 dtype 解耦(F32 也要有 paged 几何) |
| 3 | **attn v2 scratch 注入**以 paged 为前提 | 同上(缺 `ctx.attn_v2` 才触发退役 panic) | 与 ② 同刀 |

**结论**:CPU mock 档当前能力 = ①账/搬运/命令面/图捕获/容量墙/注入(§18.4 门全绿);
"engine 真前向跑在 mock 上"需先做 ①②③ 这一刀 —— 那是 **engine 执行域 dtype 收拾**,
不是 mock 台账的事。**在那之前,无卡档的 store 验证走 store 单测(61 项)+ `cpu_mock_smoke`
(全链 boot/装载/warmup/图捕获/turn)。**

#### 18.4.2 ⚠️ 65s 疑云:已排除清单(未定谳,2026-10-17 凌晨)

> 起因:用户质疑"0.8B 小模型在 CPU 上装载要 65s?"——**质疑成立**:1.5GB BF16 → f32
> 的理论成本是内存带宽级(秒级),65s 差 20 倍以上。此前"这是正确性代价"的说法**已撤回**。

**已排除(全部有实测)**:

| 假设 | 实测 | 判定 |
|---|---|---|
| 解码转换慢 | 换 65536 项 LUT(与 `half` 逐位一致,有对拍门)⇒ 61s→**52.7s** | ❌ 非主因 |
| debug 无优化 | `owl-cpu` + `half` 均开 `opt-level=3`(rustc 命令行已确认) | ❌ 无效 |
| 环境(swap/内存/NUMA/cgroup) | 107GB available、swap 未动、单 NUMA、无 cgroup 限、load 5.4/48 | ❌ 排除 |
| 长尾(个别慢调用) | 307 次调用**均匀** 57–80 ns/元素 | ❌ 非长尾 |
| 分配/page fault | `alloc` 段合计 0.26s(块容量 1.5G 元素) | ❌ 排除 |

**现状**:时间 100% 落在 `decode_loop` 段(52.7s),该段只剩"顺序读 `bytes`(1.5GB)
+ LUT 查表 + 顺序写 `dst`(3.2GB)" —— 按带宽应 ~3s,**实测 68ns/元素(≈200 周期)**。
对照组"纯 f32 写 = 0.00s"**不可信**(疑被编译器当死写消除,缺 `black_box`)。

**下一步(受控微基准,勿再猜)**:在同一块 67M 元素内存上做 4 组带 `black_box` 的对照
(纯读 / 纯写 / 读+写 / 读+转换+写),把"内存 vs 循环体"一刀切开。**在定谳前**:
①不要再用"CPU 固有成本"解释;②mock 档保持"按需开"。

**临时插桩(定位后删)**:`owl_cpu::{mock_stats, mock_segments, mock_slow_calls}` +
`cpu_mock_*` 三个诊断用例;LUT 与 `f16_lut_matches_half_bit_for_bit` 门**保留**(它是净收益)。

#### 18.4.3 定谳:瓶颈在**输入侧 staging**,不在解码循环(2026-10-17)

受控微基准(`bench2`/`bench3`,全部带 `black_box`,Rust 1.97):

| 环境 | 变体 | ns/元素 |
|---|---|---|
| 隔离(10M 元素/40MB) | `chunks_exact+zip+lut` / 索引 / 纯写 | 3.8 / **1.8** / 1.0 |
| 隔离(67M 元素/134MB,超 L3) | `chunks_exact+zip+lut` / 索引 / 纯写 | **2.5** / **1.1** / 0.7 |
| **真实装载(67M 元素/块,同函数内)** | 解码循环 / **紧跟的热写对照** | **67** / **6.1** |

⇒ 结论:
1. **解码循环本身不慢**(隔离 2.5ns/元素;`chunks_exact` 比索引约慢 2×,是次要项);
2. 真实装载里"纯写"已慢 9×(6.1 vs 0.7)⇒ 目标块(6GB 分散 Vec,首次触碰)本就有代价;
3. **剩下 11 倍(67 vs 6.1)= 读输入 `bytes`** —— 它是 `alloc_pinned` **每次新建的 staging 缓冲**
   (默认 `HeapRegion(vec![0u8; n])`,每次 33MB),**不复用、每次都冷**。

**修法方向(下一刀,不属 mock 台账)**:
- ① staging 缓冲**池化复用**(装载路径已有 `LoadStaging`,但它按"臂"分槽 —— 实际每次 `alloc_pinned` 仍新建;
  让 CPU face 的 `alloc_pinned` 走**自有页锁池**是等价 GPU 语义);
- ② 或**零中转**:从 safetensors 直读直解(省掉"填 staging → 再读 staging"的一趟)。

**用户的判断部分被证实**:数组切分(`chunks_exact`)确实比索引慢 **2×**(2.5 vs 1.1 ns/元素),
但**不是** 65s 的主因;主因是输入侧 staging 冷读。

#### 18.4.4 ✅ 定谳与修复:65s → 3.83s(2026-10-17)

**三处叠加**(按贡献排序,全部有对照数据):

| # | 根因 | 证据 | 修法 | 收益 |
|---|---|---|---|---|
| 1 | **staging 每次新建**:默认 `alloc_pinned` 每次 `HeapRegion(vec![0u8; n])`(33MB/次),用完即 drop | 池化后命中 **306/307** | `CpuFace::alloc_pinned` 池化 + `upload_pinned` 末尾归池(与 GPU"完成回调归还池"同语义) | 60.9s → 20.5s |
| 2 | **别名信息丢失**:`bytes` 来自 `Box<dyn PinnedRegion>::as_bytes()`(**虚调用**)⇒ LLVM 无法证明与 `dst` 不别名 ⇒ 每次写后重载源、禁向量化 | 同一循环:独立函数 **0.51ns/元素** vs 函数内 **38ns/元素**(74×);循环前 0.00s / 循环体 29.31s | 抽成自由函数 **`#[inline(never)]`**(签名 `fn decode_into(dtype, src: &[u8], dst: &mut [f32])` ⇒ noalias 交给编译器)。**`#[inline]` 无效** —— 内联会把别名假设传染回去 | 19.23s → 1.83s |
| 3 | **数组切分**:`chunks_exact(2).zip` | 受控微基准 2.5 vs 索引 1.1 ns/元素 | 索引式循环 | ~2s |

**总账**:

```
装载墙钟         65s → 3.83s   (17×)
upload_pinned    60.9s → 1.83s
engine lib 套件  65s → 4.59s   (含两处 CPU 装载)
```

**可复用规律(已入坑册 §50)**:热循环里数据若来自 `dyn`/trait-object 边界,LLVM 会丢别名信息
⇒ 循环失优化;**把循环抽成 `#[inline(never)]` 的独立函数**(参数用切片签名)即恢复。
判据:同一段代码"独立基准 ≪ 原地循环" ⇒ 先查别名与内联,**不要先怀疑内存带宽**。

**保留物**:`bench_write` / `bench_read` / `bench_decode_f16`(优化 crate 内的性能锚)+ `mock_stats`。
**诊断插桩全部删除**(PROBE/SCRATCH/SEGA/PB/池计数)。

---

## 19. store 无卡端到端演示(§19)· 2026-10-17

> 背景:engine 侧的 store 演示台需要**真模型前向**(无卡档受执行域 dtype 约束,§18.5.1);
> 而 store 的**账与搬运**闭环**不需要模型** ⇒ 用 store **自己的公开 API** 复刻同一套三门口。

**落点**:`crates/store/tests/store_e2e.rs`(integration;dev-dep `owl-cpu` 做真搬运,CpuFace 走真字节)。

| 场景 | 内容 | 门 |
|---|---|---|
| ① 冷热交替 | 池 6 槽 → 建 A/B 两条热链(池满)→ 钩子判(need=2)⇒ **链级整条沉 A** → 二次命中 ⇒ **一条命令批量回灌** | 换出>0 / 回灌=块数 / `onload_batches == 1` / `ensure_evicted == 0` / **逐字节恒等** |
| ② 链级粒度 | 整条 `offload_chain` ⇒ 链头段也在宿主 ⇒ 命中面 `pending == 3` | 候选票(dev_blocks=3)+ 载体全 Host |

**耗时 0.00s**(无卡、默认套件内跑)⇒ 这是 store 自己的回归锚,不再依赖 GPU/模型。

**教训(演示脚本也是代码)**:脚本里为取槽号调了 `match_prefix`,它 **touch 使 A 变热** ⇒
`plan` 就**正确地**去沉更冷的 B。链级候选按**链尾 LRU** 出票 —— 脚本必须尊重这个语义
(否则会误判 store 有 bug)。已写进测试注释。

#### 18.5.2 裁决(2026-10-17,用户):①②③ **不追** —— 由 §19 覆盖

"engine 真前向跑在 mock 上"需要先做执行域 dtype 收拾(①②③),那是 **engine 侧**的工程;
**store 需要的无卡验证不需要它** —— store 的账与搬运闭环用**自己的 API** 复刻即可(§19,
`crates/store/tests/store_e2e.rs`,0.00s)。
⇒ ①②③ 保留为**将来做"CPU 执行档"时**的入口清单,不再是 store 刀的阻塞项。

---

## 20. 挂账清单(集中;2026-10-17 收尾)

> 各刀的散落挂账在此汇总(详据见对应章节)。**★ = 建议优先**。

### A. store 刀内(设计/实现)

1. ~~★ **`Rel` 收编去重**:`rel.rs` 仍持一份与 `BlockLeg` 重复的 offload/host 面(§11.4 挂账③)~~
   ✅ **已结(S3b,§24.1)**:`Rel` 只持 `leg`/`state`/`ledger`/预算/准入账;重复面删除。
   **本刀同刀完成 §6 刀序第 4 项(S3b:per-turn 地板 + `AdmissionPolicy` 落地)** —— 见 §23/§24。
2. ~~★ **P1**:E2c 快照 capture/restore 迁 `copy_batch`(72/192 次 await → 1)~~
   ✅ **已结(§27)**:capture/restore 各一条 `copy_batch`(与 `capture_spec_snap` 同构)。
3. H2 的"按需下沉"退化路径(无整条可沉时按叶)**打点未区分**(现在只记总量)。
4. `upload_pinned` 的 F16 LUT:定谳后 LUT 与位运算**都已优化** ⇒ LUT 的净收益待重测;
   若持平可删(简化)。
5. `CpuFace::mock_stats`(通用诊断)保留;其余探针已删(§18.4.4)。

### B. 状态腿(S2/S2b)

6. ~~**S2**:GDN 状态腿三层化(快照可下沉)+ `Rel::Exact` 全链~~
   ✅ **已结(S2,§25/§26)**:快照池矩阵化 + `StateStore` 为账 + 状态腿搬运口 +
   `reuse_face` 查 store 层账 + 段边界回灌;**同刀完成引用语义拆分**(§24.5② 收口:
   idle 会话不持设备表 ⇒ `session_held = 0`、`reserved` = 真·在飞工作集)。
7. ~~★ **S2b**:GDN 行 = 活跃 turn 借还 + `max_active_turns` 正名~~
   ✅ **已结(§28)**:行随 turn 借还(`StateStore::borrow_row/return_row`),
   会话不再绑行、不再有"格耗尽/LRU 逐出会话";旋钮正名 `max_active_turns`
   (`OWL_MAX_ACTIVE_TURNS`;旧名 `OWL_GDN_SLOTS` 兼容)。
8. 草稿腿(MTP/draft)的 `ensure_for_len(… + 8)` 固定余量**未纳入需求算式**(§17.9)。
9. 空转预搬(TTFT 优化)若要做 ⇒ 作为**显式策略**立项(**禁回常数水位**,§12.4/§17)。

### C. 更远

10. **S6**:disk connector(§15.4 的 I/O 线程形态;`Carrier::Disk` 位已留)。
11. B 轴(KVMem 窗口压缩)落地后 `D_active` 变小 ⇒ 需求算式要跟着改(§17.9)。
12. engine"CPU 执行档"(真前向无卡跑):三缺口清单见 §18.5.1;裁决**不追**(§18.5.2);
    store 所需的无卡验证已由 §19 覆盖。
13. **对账口进服务面**(§13.4 落地②):store 账 × server 账 × 固定面表的对账打印 ——
    `store_report()`/`leg_stats()` 已公开(S3b),但还没接 `/debug` 与告警面。
14. ~~**`Rel` 门面全链**:engine 分两处查、门面没人用~~
    ✅ **已结(§27)**:两处调用点都走 `Rel::match_prefix → (ChainHit, RecallHit)`;
    `reuse_face` 用 `RecallHit.snap_tier` 做**尾边界在场**的快路(`RecallMode::Exact`
    在调度面的落点);逐边界回查仍走 store 的账(同源,不是旁路)。
15. **状态腿驱逐序接策略对象**:目前"最冷设备槽"= 槽级 LRU(`snap_last_use`);
    store 的 `EvictPolicy` 只服务 KV 腿(§26.5③)。
16. **快照密度/按需补拍**(§26.5②):`prefill_chunk > page` 时中途边界无快照 ⇒
    复用粒度被 chunk 卡住;要更细就得"按需补拍"(27B 一条 ≈ 162MB ⇒ 需显式策略)。
17. **状态腿盘层**(§26.5④):`Carrier::Disk` 位已留;宿主分片满了仍是丢(S6 内)。
18. **准入面的收尾**(§24.5①④):`Queue` 无定时器(重判时机 = 下一次事件)**仍挂**;
    ~~`OWL_ADMIT_ON_FULL` 坏值 = panic~~ ✅ 已结(§27:坏值 = 响亮拒 + exit(2))。

### D. 已结(留档以免重开)

- 装载 65s → **3.83s**:staging 池化 + `#[inline(never)]` 别名修 + 数组切分(§18.4.4 + 坑册 §50)
- 需求驱动保洁(§17)/ 回灌批次化(§17.7)/ 链级卸载律(§16.2)
- CPU mock 档四件(§18)/ store 无卡端到端演示(§19)
- **S3b:`Rel` 接线 + 准入预算**(§23/§24):收编去重 + 固定池档池墙 + H1 判决
  (Admit/EvictThen/Queue/Reject)+ 四面对账;门 = 无卡准入门 3 + GPU 三门口 + 0.8B 全场景


---

## 21. 裁决:store **不做 I/O**(plan/commit/abort),搬运由 engine 编排 · 2026-10-17

> 收敛 §10.2 与 §14.1 的表述张力:§10.2 写"engine = 决策 + 调用 · store = 实现 · server = 执行",
> 而 §14.1 的伪代码把 `apply(face, …)` 放在了 store。**本裁决以前者为准** ——
> **store 是纯逻辑状态机,连 `face` 都不认**。

```rust
// ── store(纯 host;零 I/O 依赖)────────────────────────────
pub fn plan(leg, need, protect) -> Verdict;                        // 判:谁走谁进/拒
pub fn plan_offload(leg, tail) -> StoreResult<OutPlan>;            // 算片 + 借宿主台(整条链)
pub fn plan_onload(leg, chains) -> StoreResult<InPlan>;            // 取内容 + 分配槽 + 组片
pub fn commit_offload(leg, &blocks, table) -> StoreResult<usize>;  // 记账(落宿主 + 摘绑定)
pub fn commit_onload(leg, &chains, &slots, table) -> StoreResult<Vec<u32>>;
pub fn abort_offload(leg, &blocks);  pub fn abort_onload(leg, &chains, &slots);   // 回滚
pub fn reclaim_slot_table(conn) -> StoreResult<SlotTable>;         // 回执取台(原箱奉还)

// ── engine(编排;唯一发起 I/O 的一层)─────────────────────
let (blocks, pieces, table) = { let p = plan_offload(leg, tail)?; (p.blocks, p.pieces, p.table) };
let conn = face.swap(SwapDir::Out, &pieces, Box::new(table)).await?;   // ← engine 发起
let table = reclaim_slot_table(conn)?;
commit_offload(leg, &blocks, table)?;
```

**为什么**:
- store 因此可**当纯状态机穷举测试**(无 face / 无异步 / 无设备面);`plan_*` 任意时刻可调(含捕获窗内);
- 搬运编排的形态(顺序、并发、多块在飞、盘层 I/O 线程)归 engine —— 那正是它"决策 + 调用"的职责;
- 代价:重验/回滚/计数的**判定**仍在 store(`plan_*` 的 `Busy`),但**执行序**在 engine。

**失败语义(HARD)**:搬运失败 ⇒ `abort_*` —— **内容已随命令消失 ⇒ 条目一并删**(不留假账);
槽台随失败的 connector 一起释放。

**落点**:`engine::running::apply_verdict`(决策单逐条)· `engine::exec::phases::execute_begin`(段边界回灌);
样板 = `crates/store/tests/store_e2e.rs` 的 `engine_offload` / `engine_onload`
(store 侧测试也有一份同构替身,只为在无 engine 下测搬运闭环)。

**门(全绿)**:store **61 + 2**(单测 + 无卡 e2e)· engine lib 36 · `cpu_mock_smoke` 3 ·
GPU 演示台(换出 12 块 / 3 命令 · 回灌 2 块 / 1 命令 · 兜底驱逐 0)。

---

## 22. GPU 生产形态实测:换出/回灌/命中三门口(2026-10-17)

> 目标:store 的搬运闭环在**真前向 + 真 GPU + owl server(OpenAI 面)**下验证。
> 结论:**三门口全绿**;并定谳两条真实机制约束。

### 22.1 现场(3080;owl server;0.8B)

配置:`OWL_HOST_CACHE_MB=512 OWL_MAX_SEQ=384 OWL_POOL_TOKENS=384 OWL_SNAP_MAX=64 OWL_PREFILL_CHUNK=32`
负载:三轮(长 A → 长 B 不同前缀 → **重复 A**),ephemeral(无 session ⇒ 焚链 ⇒ 缓存独占)。

```
轮 1  match=0 · 拍 6 张边界快照
轮 2  Grow: free=0 ⇒ 钩子判 out=1 ⇒ 下沉 6 块(整条链 A 进宿主)
轮 3  match=5 · chain=[占位符 ×5](全在宿主) · 边界快照在场=[true,true,true,true]
      [store] 段边界回灌 5 块(session=2;宿主→设备;一条命令)
      cached_tokens = 160
```

⇒ ①换出 ✓ ②回灌 ✓(一条命令搬 5 块)③命中复用 160 token ✓;Xid 0、无崩。

### 22.2 定谳①:可复用性 = **块边界快照**密度;快照只在 **prefill chunk 边界**拍

- 首实验(prompt 732、chunk 缺省 512)⇒ `cached=512` = **恰好一个 chunk**;
- 块边界快照缺失时(match=5 而 `边界快照在场=[false,…]`)⇒ m 被裁到 **0**
  ⇒ **块全程在设备也不复用**(§17 的快照裁剪律在生产形态显形);
- `prefill_chunk=32`(= 一页)⇒ 每块边界有快照 ⇒ `cached=160`(5 块)✓

**规律**:跨轮前缀复用的最小粒度 = `min(prefill_chunk, page)`(现即 chunk)。
生产档若要**块粒度**复用,`prefill_chunk` 必须 = `page`。

### 22.3 定谳②:池压力与命中率互相抵消

命中率高 ⇒ 准入 `need = 总块 − 设备命中` 小 ⇒ **不必腾位** ⇒ 看不到换出。
要观察换出/回灌,池必须**刚好满**(本例池 12 槽、两轮各 6 块 ⇒ 第三轮 `need=1` 也触发腾位);
且腾位是**整条链**粒度(§14.6)⇒ **为 1 个槽搬 6 块**(代价,已在挂账口径内)。

### 22.4 净产出(入库)

- **四个 env 旋钮**(实测必需;`OWL_HOST_CACHE_MB` 原有):`OWL_POOL_TOKENS` / `OWL_MAX_SEQ` /
  `OWL_SNAP_MAX` / `OWL_PREFILL_CHUNK`;`probes.debug` 走既有 **`OWL_DEBUG`**(不新造第二套命名);
- **诊断**:scheduler 的 `dbg prefix` 打印现在带 `snap_used/cap` 与**前 4 个边界快照在场性**
  —— 排查"命中面被裁到 0"的第一现场。

### 22.5 事故记录(操作纪律)

清理"残留测试服务"时用 `pgrep -x server`,而进程名 `server` 被**另一个 owl 服务**(单卡 3090Ti
22GB)占用 ⇒ **误杀**(2026-10-17,已报)。
**纪律**:杀进程只用 **PID 文件**或精确校验 `cmdline`;**禁止**按进程名 `-x` 匹配杀
(除 AGENTS 的"pkill -f 匹配自身"之外,新增同类:进程名撞车)。

---

## 23. S3b:`Rel` 接线 + 准入预算(收编去重 × 固定池档的池墙)· 2026-10-17

> **本刀 = §20★A1(`Rel` 收编去重)+ §6 刀序第 4 项(per-turn `dense_floor` 接 `TurnOpts` +
> `AdmissionPolicy` 落地)** —— 两件事共用同一条接缝(`Rel` 门面),并刀做,一次把
> §13.5 的施工绑定兑现:**建账 → 判据 → 判决 → 执行** 全链在场。
>
> 落定后 engine **经 `Rel` 走**(不再直用 `BlockLeg`),H1 准入从"占槽后 `Exhausted`"
> 变成"**判在前**":够就进、不够先腾、腾不够就拒/排队(带数字)。

### 23.1 收编形态(store 侧;§11.4 挂账③ 的兑现)

```rust
pub struct Rel {
    pub leg: BlockLeg,       // KV 腿:账 + 宿主物理 + 搬运口(唯一状态持有者)
    pub state: StateStore,   // GDN 腿(账;搬运待 S2)
    pub ledger: TierLedger,  // 语义账(状态腿 / 保留面 / 弹性面;KV 块面从腿报出,不重复记)
    pub policy: AdmissionPolicy,
    pub admission: Admission,
    budget: VramBudget,
    evict: Box<dyn EvictPolicy>,
}
```

| 改动 | 内容 | 理由 |
|---|---|---|
| **删重复状态** | `Rel::{kv, host}` 两个字段消失(腿已持同样的两件) | 两本账同源才可能对不上;§13.4 的对账口要求单一真源 |
| **删重复实现** | `Rel::offload_block` / `onload_block`(整块口径)、`Rel::evict_cached`、`account_evicted` | `leg::{plan_offload,commit_offload,plan_onload,commit_onload}` + `leg.drop_cached` 已取代(§15/§16/§21) |
| **保留门面方法** | `probe` / `match_prefix` / `register_chain` / `ensure_for_len` / `release_table` / `plan_evict` / `report` | 关系层职责(①身份映射 ②召回语义 ③换出单合并 ④报告)—— 委托腿 + 两腿配对 |
| **KV 面报告** | `report().kv` ← `leg.report()`;`tiers` 只报 store 自管面(状态腿/保留面/弹性面) | KV 块面的真源 = 腿(槽/绑定/宿主计数),不另记一份 |
| **`Rel::open`** | `(inputs, policy, LegConfig)` —— 几何与每 token 字节从 `LegConfig` 推(`block_bytes / block_tokens`),与 `inputs.kv_bytes_per_token` **交叉校验**(矛盾 = 拒) | 同一件事只有一个入口(旧签名要调用方手抄一遍几何) |

### 23.2 固定池档的准入面:`BudgetInputs.kv_face_bytes`

§2.4 的准入式按**卡级余量**(`V_avail`)判,但那适用于"KV 面随请求增长"的动态池档。
**本引擎是固定池档**:设备 KV 面在 boot 就预分配好(`nb × block_bytes`),请求来了
不会再去向驱动要显存 ⇒ 准入的真判据是 **§2.3 的池墙**(`Σ N_j·k + N_i·k + 缓存驻留 ≤ pool·k`),
卡级 `v_avail` 只回答"池**该有多大**"(容量线①)。

⇒ `BudgetInputs` 加一项、`VramBudget` 加一口:

```rust
pub struct BudgetInputs {
    /* …原有八项… */
    /// **预分配的设备 KV 面**(池字节)。> 0 = 固定池档 ⇒ 准入按它判(池墙);
    /// `0` = 未预分配(动态池档)⇒ 按 `v_avail` 判(卡级余量)。
    pub kv_face_bytes: usize,
}

impl VramBudget {
    /// 准入判据面(唯一产地):固定池档 = 池;动态档 = 卡级余量。
    pub fn face_bytes(&self) -> usize {
        if self.inputs.kv_face_bytes > 0 { self.inputs.kv_face_bytes } else { self.v_avail_bytes() }
    }
}
```

`check_admission` 的阈值一律改用 `face_bytes()`(判据单源)。

**装配纪律(A5.1 的分档)**:固定池档的**准入**不依赖卡级三项(`v_card`/`weights`/
`runtime_floor`),故 `decompose` 在 `kv_face_bytes > 0` 时允许它们缺项(0)—— 但
`VramBudget::card_measured()` 必须**可查**(未实测 = 报告面标出来,日志 WARN,
不是静默)。动态池档仍是全项必需(旧行为不变)。

**保留额账(`Admission`)的拆分规则(curl 本刀定案)**:

```
reserved_bytes = Σ_active turn 的**设备实占**(引擎每段边界同步;= 块表长 × 块字节)
cache_bytes    = 池实占 − reserved(可动面 = 可下沉/可驱逐的缓存块)
elastic_bytes  = 0(留 B 轴:`dense_floor` 之上的灵活面)
⇒ committed = reserved + cache = **池实占**(精确,不双算)
⇒ evictable = cache + elastic = 可动面(判 `EvictThen` 用)
```

`Admission` 加一口 `set_floor(turn, bytes)`(活跃 turn 的实占地板更新 —— 保留额的唯一更新口,
与 `set_elastic` 同形)。**"保留额不可驱逐" 在本引擎由引用计数兜底**(腿里 `refs > 1` =
活跃面,`ensure_offloadable`/`evict_candidates` 天然排除),字节账只做**判据与报告**。

### 23.3 engine 侧接线(三处)

| 处 | 改动 |
|---|---|
| **boot 装配** | `Engine::{new,on}` 落两个快照:`v_card_bytes`(开卡后 `vram::free_bytes`)、`live_at_new`(`vram::live_bytes`);`run()` **在池与图都落位之后**组装 `BudgetInputs`(表见下)并 `Rel::open` —— 失败 = **响亮拒启动**;`SessionBank` 的构造**移到图装配之后**(它现在要收预算) |
| **账房换人** | `SessionBank.leg_target: BlockLeg` → `SessionBank.rel: Rel`;engine 全部调用点 `bank.leg_target.X()` → `bank.rel.leg.X()`(数据面 `plan_*`/`commit_*` 的自由函数仍收 `&mut BlockLeg`,形状不变);新增一个私有的 `sync_pool_account()`(段边界/准入/释放后把池实占与活跃实占喂给 `Admission`) |
| **H1 判决** | `pump()` 的段边界:待准入 turn 的 `need` 先过 `Rel::check(floor, ctx_tokens)`;`Admit` 照旧、`EvictThen` 抬需求(先腾位)、`Queue` 留在队首计时、`Reject` 出 `TurnEvent::Failed`(文案带数字) |

**boot 装配表(逐项真源;实测口径 = §9.6 的探针算术)**:

| 项 | 真源 |
|---|---|
| `v_card_bytes` | `Engine::on` 后的 `vram::free_bytes()`(CUDA ctx 已在 free 里扣掉) |
| `weights_bytes` | `live(装载后) − live(Engine::new)` |
| `runtime_floor_bytes` | `live(run 末) − live(装载后) − (KV 池 + GDN 格 + 快照池 + 草稿池)` |
| `kv_face_bytes` | `nb_kv × StatePool::block_bytes()` |
| `gdn_row_bytes` | `StatePool::gdn_row_bytes()` |
| `draft_pool_bytes` | `StatePool::draft_pool_bytes()` |
| `kv_bytes_per_token` | `block_bytes / page` |
| `host_bytes` / `disk_bytes` | `knobs.host_cache_mb << 20` / 0 |
| `AdmissionPolicy` | `max_active_turns = knobs.gdn_slots`(正名待 S2b)、`on_full = knobs.admission_on_full` |

**无设备账档**(CPU mock:全 `vram` 账为 0)⇒ 固定池档放宽规则接管:池墙照判,卡级三项记 0
并在报告面标"未实测";有设备账却算出 `weights = 0` = 测量病 ⇒ 响亮 WARN(不是静默)。

### 23.4 判决映射(§2.4 的四种结果 → engine 行为)

| 判决 | engine 行为 |
|---|---|
| `Admit { floor }` | `admission.admit(turn, floor, 0)` → 继续(照旧入队首→schedule 弹) |
| `EvictThen { floor, need_bytes }` | 把保洁需求抬到 `max(Δ_j, ceil(need_bytes/块字节))`(同一次段边界),搬完**重判**;仍不足 ⇒ 按 `on_full` |
| `Queue { reason }` | 留队首(不弹),记等待起点;等待 > `max_wait_ms` ⇒ 按 `Reject`(文案带原因与等待时长) |
| `Reject { reason }` | 弹掉该 turn + `TurnEvent::Failed { turn, err }`(队列继续;文案含**具体数字**) |

**文案纪律**:准入拒**不带** "超预算" 字样 —— 那是"prompt+max_tokens > max_seq"的专用标记
(server 据此归一化成 `context_length_exceeded` 触发客户端自动 compact)。池墙拒是
**另一类**(客户端该做的动作是降并发/重试,不是压缩上下文),故用 `准入拒绝:` 前缀,
server 侧照原样 400 回吐。

### 23.5 per-turn 地板(`TurnOpts.dense_floor`)

`TurnOpts` 加 `dense_floor: Option<TurnFloor>`(请求可指定),缺省走 `knobs.dense_floor`。
**本刀只实做 `Strict`**:`TurnFloor::Bytes(_)` ⇒ **响亮拒**(文案点明"选择层(B 轴)未落地,
地板语义无从兑现")—— 宁拒不许静默降级(§2.4 的红线)。
B 轴落地后放开(那时 `Bytes` 才有物质基础:窗口压缩 + 按需物化)。

### 23.6 门

| 门 | 判据 |
|---|---|
| store 单测 | 全绿(收编后用例改写;新增:池墙面、`set_floor` 语义、固定池档卡级缺项装配) |
| engine lib | 36/36(行为零变化路径)+ 新:准入判决(Admit/EvictThen/Reject/Queue)|
| 0.8B e2e | 旧场景不回退(§11.2 同一画像) |
| GPU 三门口(§22) | 换出/回灌/命中 三门口重跑(池压小 + `EvictThen` 主动腾位) |
| **新门** | ①**池墙拒**:需求 > 池面 ⇒ `Failed` 带数字(不落 `Exhausted`)②**先腾后进**:池满 + 冷链 ⇒ `EvictThen` 下沉后准入成功 ③`Strict` 路径**逐 token 恒等**不回退 |

---

## 24. S3b 实录:`Rel` 接线 + 准入预算落地 · 2026-10-17

> §23 是拍定的设计;本节是**实现后**的实录 —— 含三处**实现期修正**(设计里没写对的地方)+ 门证据。
> 结论:**engine 经 `Rel` 走**,H1 准入判决在真路径上生效(无卡门 3 + GPU 门回归全绿)。

### 24.1 实现期修正(三条;都是"真源选错会把话说满")

| # | §23 原设计 | 实测发现 | 定案 |
|---|---|---|---|
| ① | `Admission` 三个面(reserved/cache/elastic) | idle 会话**持着的历史**既不是活跃地板也不是可动缓存 ⇒ 归 `cache` 会让准入先出 `EvictThen`(允诺腾得出)而搬家时 `Busy`(搬不动),最后仍退回 `Exhausted` | 加**第四面** `session_held_bytes`(非 turn 归属的不可动面);`committed = reserved + session_held + cache` |
| ② | 池实占取 `leg.report().gpu_bytes`(树绑定面) | 尾部**不满一块**的前缀(`tokens.len()/page` 截断)不进树,但槽已经占了 —— 实测 8 会话占满 8 槽,绑定面只报 6 块 ⇒ 按绑定面判**多放一个 turn 进来**(然后撞 `Exhausted`) | 加 `BlockLeg::occupied_bytes()`(**槽账真源**:已用槽 × 块字节),准入按它算 |
| ③ | 可动面 = `池实占 − 活跃实占` | 字节相减会把 idle 会话持着的历史算成可动(同①的病) | 加 `BlockLeg::movable_bytes()` = 设备绑定且**引用 = 1** 的块 —— 与 `chain_down_span`/`ensure_offloadable` **同一个判据**(引用账),不是两套口径 |

**两条派生律(实现期新增,已写进代码注释)**:

1. **H1 判决只在"无活跃 turn"时判**。有活跃 turn 时队首 turn 只是在**等位** ——
   此刻"池被活跃面占着"是常态(它马上会退成缓存),照判会把**正常排队的第二个请求误拒**。
   准入 = "谁能**开始**"。
2. **`schedule()` 不许弹已被判排队的 turn**(`admit_wait` 命中队首 ⇒ 回 Idle)。
   否则"排队"语义形同虚设:弹了就直接撞 `ensure_for_len`。`waited >= max_wait_ms`
   (上限 0 = 立刻按拒处理,不许"等 0 毫秒"再等一轮)。

### 24.2 落点(与 §23.1/§23.3 一致的部分从略)

| 面 | 变更 |
|---|---|
| `store::rel` | `Rel { leg: BlockLeg, state, ledger, policy, admission, budget, evict }` —— 删 `kv`/`host` 两份重复状态与 `offload_block`/`onload_block`/`evict_cached`/`account_evicted`;门面方法一律委托腿(`probe`/`match_prefix`/`register_chain`/`ensure_for_len`/`release_table`/`plan_evict`/`report`);`Rel::open(inputs, policy, LegConfig)` 并从腿几何**交叉校验**每 token 字节与 host 分片(≤ 总账) |
| `store::budget` | `BudgetInputs.kv_face_bytes`(固定池档的池墙)+ `face_bytes()`(准入判据唯一产地)+ `card_measured()`(固定池档允许卡级缺项,但**必须可查**)+ `Admission::{set_floor, set_session_held}` |
| `store::leg` | 新增 `occupied_bytes()` / `movable_bytes()` / `note_probe()` / `chain_len()` / `tail_hash()` / `kv_stats()` / `host_free_bytes()`;`LegStats` 加 `block_bytes` / `block_tokens` / `occupied_bytes` / `movable_bytes`(§13.4 对账口) |
| `engine::engine` | `Engine.boot: BootVram{v_card_bytes, live_at_new}`(构造时快照)+ `assemble_budget()`(实测真源表见 §23.3);**`SessionBank` 构造移到图装配之后**(它现在要收全量固定面) |
| `engine::session` | `SessionBank { sessions, rel: Rel, leg_mtp }`;`BankPlan` 不变(host 预算与预算入参同源 ⇒ 腿不超总账) |
| `engine::running` | `sync_pool_account()`(四面对账)+ `admission_gate()`(H1 判决)+ `reject_queued_turn()`(出队 + 焚临时会话 + `Failed` 事件)+ `store_report()`/`leg_stats()`/`queue_len()`(报告面)+ `pump` 的 Idle 分支先回吐 `pending_events`(被判拒的 turn 才发得出去) |
| `engine::exec::turn_end` | `close_admission(turn_id)`(turn 生死 = 保留额生死;`complete` 与 `abandon` 共用) |
| `engine::turn` | `TurnOpts.dense_floor`(非 `Strict` 在 **submit 边界响亮拒**:B 轴未落地,地板语义无从兑现) |
| `engine::scheduler` | 弹 turn 时 `admission.admit(id, 0, 0)`;已排队 turn 不弹 |
| `apps/server` | `TurnOpts { dense_floor: None }`(OpenAI 面暂无扩展字段;缺省走引擎旋钮) |

**旋钮**:`EngineKnobs::{admission_on_full(OWL_ADMIT_ON_FULL=reject|queue[:ms]), dense_floor(缺省 Strict)}`。

### 24.3 门(全绿)

| 门 | 结果 |
|---|---|
| `owl-store` 单测 + e2e | **63 + 2**(新增:池墙面 / `set_floor` 语义 / `session_held` 不变量 / 固定池档卡级缺项) |
| `owl-engine --lib` | **36/36**(行为零变化路径) |
| CPU mock 冒烟 | **3/3**(无卡档照旧可 boot —— 固定池档放宽规则接管) |
| **新准入门**(`crates/engine/tests/store_admission_gate.rs`,无卡可跑) | **4/4**:①**先腾后进**(池 8 槽被可动缓存占满 ⇒ `EvictThen` ⇒ 段边界**下沉 7 块** ⇒ 准入成功;兜底驱逐 0)②**池墙拒**(8 个键控会话占满 8 槽、可动面 0 ⇒ `Failed` 带数字且**不含 "超预算"**)③**排队姿态**(`Queue{1ms}`:首轮 pump 回 Idle + turn 留队 + 零兜底驱逐;超时后 `Failed`「排队 5ms 仍无法腾位」)④**地板语义**(非 `Strict` 在 knobs(boot)与 `TurnOpts`(submit)两处**响亮拒**) |
| **GPU 三门口**(§22 场景重跑;3080 ordinal 0) | 轮 3 判 `EvictThen` ⇒ **换出 6 块**/自由槽 2/8/宿主 6 块 · 轮 4 **回灌 2 块 1 命令** · 兜底驱逐 **0** |
| 0.8B 场景回归(GPU) | `prefix_cache_hit` / `two_turns_lifecycle` / `session_continuity` / `multi_session_isolation` / `schedule_decision_surface`(max_seq=64 小档)/ `gdn_slot_cross_probe` / **`spec_identity_gate`(逐 token 恒等)** 全 OK |
| clippy | 改动面**零告警**(既有告警在 sampler/graph_plan/e4_api 等未触面) |

### 24.4 操作期发现(两条给后人)

1. **CPU mock 档的包络 = 单 prefill chunk**(prompt ≤ `prefill_chunk`)。多 chunk 会炸在
   `exec/prefill_chunk.rs` 中间块收割的 dtype 口径(`d.hidden × esz` 按**模型 dtype** 算,
   而 CPU 面**值块恒 f32**)—— 即 §18.5.1 的**既有缺口**,非本刀引入。
   写无卡门时把负载压在单 chunk 内即可(锚:**n=4 个数字词 ⇒ prompt 27 tok**;n=6 ⇒ 35 tok 已越界)。
2. **`apps/server` 的 e2e 判据卡在装载暂存区**(`暂存区 raw.in: 本键 635.7MB > 槽 134.2MB`,
   需 `OWL_LOAD_STAGING_MB` 调大)—— **既有环境问题**(与本刀无关:错误发生在装载相,
   早于 `run()` 的预算装配)。27B 档要在本机跑 server e2e 需先给该旋钮。

### 24.5 挂账(本刀新增)

1. **`Queue` 无定时器**:重判时机 = 下一次 pump(通常由"活跃 turn 终了"或新提交驱动)——
   若引擎转 Idle 且无人再提交,排队的 turn 会一直等(客户端自身超时会断开)。
   要真定时器 = 服务面加轮询/事件源(属于 agent 服务面 A 类债)。
2. **`reserved` 目前记的是"活跃会话整条链"**(含历史),不是严格意义的"在飞工作集" ——
   因为二者在引用账上同命(都是 refs > 1)。要拆开 ⇒ 需"会话引用 ≠ 活跃引用"的引用语义刀
   (与 §2.3 目标模型的差,见 §23.2 注)。
3. **`gdn_slots` 仍未正名**为 `max_active_turns`(S2b;本刀先按现名取值)。
4. `OWL_ADMIT_ON_FULL` 的**坏值 = panic**(响亮拒启动)—— 若要改成结构化 boot 错误,随
   §18.5 的"配置错误静默 exit 1"那条红线一起收。

---

## 25. S2 设计:状态腿三层化 + 引用语义拆分 · 2026-10-17

> 两条目标(用户拍定;§24.5② 的直接承接):
> **①引用语义拆分**(会话引用 ≠ 活跃引用)⇒ `reserved` 才是真正的"在飞工作集";
> **②状态腿三层化**(GDN 快照设备 ⇄ 宿主)⇒ 冷快照可下沉,`snap_max` 不再是 idle 会话的容量墙;
> 两条合成的产物 = **`Rel::Exact` 全链**(块链 + 快照都在,且快照可在宿主机 → 段边界回灌后精确续跑)。

### 25.1① 引用语义拆分:定案 =「**idle 会话不持设备表**」

**现状的病**(§24.5②):会话在 `SessionTable.block_table` 里**长期持着**它整段历史的设备槽引用
⇒ 那些块 `refs ≥ 2` ⇒ `chain_down_span`/`ensure_offloadable` 一律 `Busy` ⇒
「idle 会话的历史」既不可动也不可下沉,`reserved` 里混着历史(不是纯在飞工作集)。

**定案(与 §2.3 资源归属律逐字对齐)**:§2.3 表早就写明
「KV **工作集**(在飞读写)= 活跃 turn 持;KV **历史块**(缓存)= 会话经**前缀缓存**持」
—— 会话对历史的持有**本来就该走缓存**,不该走设备表。故:

```
turn 起:块表**按内容身份重建**(leg.match_prefix(tokens) → 槽 + 待回灌单)
turn 中:会话表 = 活跃 turn 的设备工作集(唯一的设备持有者)
turn 止:register_chain(历史进前缀缓存)→ **释放设备表**(表清空)
        ⇒ idle 会话对设备零持有(只有 tokens + cached_len 两个纯 host 字段)
```

由此**不需要**新的引用类型(`active/idle` 双计数):idle 会话根本不持引用 ⇒
所有块在 turn 外都是 `refs = 1`(纯缓存) ⇒ **现成的 `chain_down_span`/`movable_bytes` 判据直接成立**
(少一套账 = 少一处可能对不上的地方)。

**连带裁决(三条)**:

1. **删「会话守卫(hit)」快路**(`scheduler.rs` 的 GuardHit 分支):它的角色被**内容寻址**天然取代
   ——客户端改写历史 ⇒ 命中自然变短(自回退全量),不需要守卫拦。守卫只留作
   `cached_tokens` 的**报告**用(`submit` 时给初值;AGT-12 校正帧随后覆盖)。
2. **turn 止的快照是复用钥匙**:块表重建走 `match_prefix` ⇒ 复用长度被 `reuse_face` 裁到
   **有快照在场的块边界** ⇒ 关键会话(agent 流量)的跨 turn 复用**依赖 turn 末边界有快照**。
   现网已在 **decode 每页边界**拍快照(`phases.rs` `fed % page == 0`)⇒ turn 末的最后一个
   页边界必有快照 ✓(缺口 ≤ 1 页 = 32 token 重算,可忽略)。
3. **`abandon` 与 `complete` 同规**:两者都释放设备表(ephemeral 焚会话;键控会话保 `tokens`)。
   半途弃置的链**不进缓存**(没有 `register_chain`)⇒ 下次内容寻址自然不命中 ✓(安全)。

**报告面的收益**:`Admission.reserved_bytes` = 活跃 turn 的表长 × 块字节(**真保留额**);
`session_held_bytes` ⇒ **恒 0**(不再有"非 turn 归属的不可动面");
`cache_bytes` = 池里全部纯缓存(含各 idle 会话的历史)—— 它们**可动、可驱、可下沉** ✓。

### 25.2② 状态腿三层化:定案 =「快照池矩阵化 + 独立搬运口 + `StateStore` 为账」

| 面 | 现状 | 定案 |
|---|---|---|
| 设备物理 | `snaps: Vec<GdnSnapSlot>`(**每槽**一套 (层×4) 缓冲) | **矩阵化**:每 (层,quad) **一条** 缓冲,长 = `snap_max × 行宽` ⇒ 快照 = 「一种块(槽 × 片)」,直接套 KV 腿的 `SwapPiece`/`SlotTable` 机制 |
| 账 | 只在引擎(`key`/`last_use` LRU 覆盖写) | **`StateStore` 为真源**:`register_snap(key, bytes)` / `set_tier(key, tier)` / `drop_snap` / `evict_candidates`;引擎的槽位表退化为"设备槽 → key"的索引 |
| 宿主层 | 无 | `HostTier<SnapKey>`(键泛型化;预算 = store host 总量里分给状态腿的份额) |
| 搬运口 | 无(快照只在设备) | `Rel` 的**状态腿搬运口**:`plan_snap_offload/commit/abort` + `plan_snap_onload/commit/abort`(与 KV 腿同形:**store 出计划与账,engine 发 `face.swap`**) |
| 复用判据 | `pool.has_snap(key)`(只看设备) | `Rel::match_prefix(...) → RecallHit.snap_tier`(`Some(Gpu)`/`Some(Host)` 都可复用 ⇒ `RecallMode::Exact`);宿主层快照在 `execute_begin` **段边界回灌** |
| 槽位不足 | LRU 直接覆盖写(**静默丢状态**) | **先下沉**(最冷的设备快照 → 宿主;内容留),宿主也满 ⇒ 才丢(打点) |

**快照搬运的粒度律(与 KV 腿同构)**:一条快照 = 一个"块"(槽 × (层×4) 片);
下沉/回灌 = **一条 swap 命令搬 N 片**(不是 N 次往返)。

### 25.3 与 S2b 的分界(不在本刀)

- **本刀**:设备表在 turn 止释放(§25.1①);**GDN 行仍按会话绑**(现状)。
- **S2b(下一刀)**:行改 turn 借还 + `gdn_slots` 正名 `max_active_turns` ——
  那时 idle 会话连"行"也不占,跨 turn 状态**只**走快照(本刀已把它三层化)。

### 25.4 门

| 门 | 判据 |
|---|---|
| 引用拆分 | ①turn 止后 `session_held_bytes == 0` 且历史块进**可动面**(`movable_bytes` 含之)②idle 会话占满池 ⇒ 新 turn **先腾后进**(不再拒)③键控会话跨 turn **仍命中**(块对齐),答案内容不受影响 |
| 状态腿 | ④快照 host 往返**逐字节一致**(CPU 档可跑)⑤冷快照下沉后仍能 `Exact` 命中(段边界回灌)⑥`StateStore` 账与设备槽数/宿主字节对账 |
| 回归 | 0.8B 全场景(session_continuity / multi_session_isolation / prefix_cache_hit / spec 恒等)+ GPU 三门口 + `owl-store`/`owl-engine` 单测 |

---

## 26. S2 实录:引用语义拆分 + 状态腿三层化 · 2026-10-17

> §25 是设计;本节记**实现**与实现期发现。结论:两条目标都落地,**门全绿**
> (含 GPU 真机:§22 三门口 / 0.8B 全场景 / 新状态腿探针)。

### 26.1 S2-1 引用语义拆分(§25.1①)

| 落点 | 变更 |
|---|---|
| `exec/turn_end.rs` | `complete`:登记缓存后 **释放设备表**(`release_turn_table`;`cached_len` 归**块边界**);`abandon` 同规(弃置的链**不进缓存** ⇒ 下次内容寻址自然不命中) |
| `scheduler.rs` | **删 GuardHit 快路** ⇒ 复用一律走**内容寻址**(会话 = 身份 + token 历史;块表按内容重建) |
| 效果(实测) | idle 会话占满池后:`session_held_bytes = 0`、`movable_bytes = gpu_bytes`(**全池可动**)⇒ 新 turn 从"池墙拒"变成"**先腾后进**" |

### 26.2 S2-2 状态腿三层化(§25.2)

| 落点 | 变更 |
|---|---|
| `store::host` | `HostTier<K = ChainHash>` **键泛型**(状态腿用 `SnapKey`) |
| `store::state` | `register_snap` / `snap_bytes` / `snaps_in_tier`(分层统计进 `StateStats`) |
| `store::rel` | `snap_host: HostTier<SnapKey>` + `attach_snap_layout` + **状态腿搬运口**(`plan_snap_offload`/`commit`/`abort` + `plan_snap_onload`/`commit`/`abort` + `drop_snap_host`);`Rel::open` 按"总账 − KV 分片"切状态腿宿主预算 |
| `engine::state` | 快照池**矩阵化**(每 (层,quad) 一条 `snap_max × 行宽` 缓冲)⇒ 快照 = 「槽 × 片」的一种块;`snap_pick_slot`(原地/空槽/**最冷**) + `snap_pool_layout` + `snap_place/clear_slot/slot_of` |
| `engine::exec` | `capture_gdn_snap`:槽满 ⇒ **先下沉最冷**(宿主满/关才丢) + `StateStore::register_snap`;`execute_begin`:宿主层快照**段边界回灌**(`snap_ensure_resident`)再 D2D 恢复;`snap_offload`/`snap_onload` 编排(store 出计划 → `face.swap` → 记账) |
| `engine::scheduler` | `reuse_face` 的"边界快照在场"判据改查 **store 的层账**(设备/宿主都算在场)—— 旧实现只看设备池 ⇒ 下沉后的快照被误裁 |

### 26.3 实现期发现(四条)

1. **宿主预算是两条腿的分片,不是一根旋钮**。S3b 里 `host_cache_mb` 全给了 KV 块腿
   ⇒ `Rel::open` 算出的状态腿分片 = 0 ⇒ 快照下沉**无位**(静默走"丢"分支)。
   ⇒ 新增 `EngineKnobs.snap_host_mb`(`OWL_SNAP_HOST_MB`),**总账 = KV 分片 + 状态腿分片**
   (`Rel::open` 交叉校验分片不超总账)。教训:两个热区完全不同的腿共用一个预算数,
   只会让一边饿死 —— 预算要**显式分片**。
2. **一条快照的体量**(布局注册后才看得见):0.8B = **19.3MiB**(18 层 × 4 quad 行宽之和),
   27B ≈ **162MB** ⇒ 宿主分片的量级感直接决定"能攒多少 idle 会话状态"(27B:1GB ⇒ 6 条)。
3. **CPU mock 的单 chunk 包络**(§18.5.1 的 dtype 缺口)**挡住了"中途边界快照"**:
   单 chunk ⇒ 只有 prefill 末(需 32 对齐)与 decode 页边界出快照 ⇒ "同会话跨 turn
   复用 + 宿主回灌"在无卡档构造不出来。⇒ 引擎级 S2 门改成 **crate 内场景**
   (`cpu_mock_state_leg_probe`/`gpu_08b_state_leg_probe`;能直接驱动 `snap_ensure_resident`),
   真机复用由 GPU 场景覆盖。
4. **三个既有场景的判据要跟着语义升级**(旧判据测的是"会话持设备表"这个已消失的物)——
   见 26.4。

### 26.4 门(全绿)

| 门 | 结果 |
|---|---|
| `owl-store` | **63 + 2(snap_leg) + 2(store_e2e)**;新增 `snap_leg.rs`:快照 host 往返**逐字节一致** / 层账跟着走 / 宿主满响亮拒(零副作用)/ 回灌失败条目删(不留假账)/ 两腿宿主互相独立 |
| `owl-engine --lib` | 36/36 |
| CPU 档 | `cpu_mock_smoke` 3/3 · `store_admission_gate` **3/3**(②改判为"**idle 占满池 ⇒ 先腾后进**"+`session_held=0`+全池可动)· `cpu_mock_state_leg_probe` ✓(快照 churn ⇒ 下沉 2 条 / 回灌就位 / 层账回设备) |
| **GPU 真机**(3080 ordinal 0) | §22 三门口 ✓(换出 6 块/回灌 2 块/兜底 0)· S0 预算探针 ✓ · **S2 状态腿探针 ✓**(设备 1 条 / 宿主 2 条 ⇒ 回灌 ⇒ 恢复)· `spec_identity_gate` ✓(逐 token 恒等不回退) |
| 0.8B 场景(判据升级后) | `two_turns_lifecycle` ✓ · `session_continuity` ✓(同会话跨 turn 答对)· **`prefix_cache_hit`**(物理共享 → **内容身份共享 9 块**;B 0 chunk/TTFT 129ms vs A 9 chunk/1322ms)✓ · **`multi_session_isolation`**(块表 → 内容链;A 3 块身份不变 + A/B 身份零相交)✓ · `schedule_decision_surface`(GuardHit → **内容寻址 Fresh**)✓ · `gdn_slot_cross_probe` ✓ · `longctx_4k_prefix_qa` ✓(账本块对齐口径) |

### 26.5 挂账(本刀新增)

1. **S2b**(下一刀):GDN 行改 turn 借还 + `gdn_slots` 正名 `max_active_turns` ——
   本刀已把它的前置(快照三层化 + 会话不钉设备)做完;S2b 之后 idle 会话连"行"也不占。
2. **快照密度**:prefill 期只在**块边界**(32 对齐)与 decode 页边界出快照 ⇒
   `prefill_chunk > page` 时中途边界无快照(§22 定谳①)。要更细的复用粒度 ⇒ 要么
   降 chunk,要么"按需补拍"(代价 = 一条快照的 D2D,27B ≈ 162MB ⇒ 需显式策略)。
3. **状态腿的驱逐序**:目前"最冷设备槽"= 槽级 LRU(`snap_last_use`);store 的
   `EvictPolicy` 还没接状态腿(两条腿的策略对象统一是 S2b/后续的事)。
4. **快照的盘层**:`Carrier::Disk` 位已留;宿主分片满了仍是"丢"(响亮打点)。

---

## 27. A 轴收尾(四小件)· 2026-10-17

> 用户拍板:**先把 A 面做完**(A 轴 = 容量层:身份命中/三层搬运/需求驱动/准入),
> 再上 B 轴(KVMem 式动态选块)。本节 = A 轴的**边角收尾**(4 件,全部零架构改动)。

| # | 件 | 落点 | 证据 |
|---|---|---|---|
| 1 | **P1:快照搬运批命令化** | `StatePool::{capture_snap, restore_snap}` 逐 (层,quad) `copy_block_at` → **一条 `copy_batch`**(0.8B 72 次 / 27B 192 次 actor 往返 ⇒ 1) | `owl-engine --lib` 36 · CPU/GPU 状态腿探针 ✓(拍摄/回灌/恢复全链) |
| 2 | **状态腿 metrics** | `store.snap.{out,in,drop}` + `.{out,in}.bytes`(KV 腿同族口径;G4 要求命中/换出/回灌可观测) | 编译期接线;计数在 `exec/spec.rs` 的三条路各一处 |
| 3 | **`Rel` 门面全链** | `schedule`/`step_need` 都改走 `Rel::match_prefix → (ChainHit, RecallHit)`;`reuse_face` 收 `RecallHit`,用 `snap_tier.is_some()` 做**尾边界在场**快路(`RecallMode::Exact` 在调度面的落点) | 同上 + §22 三门口(命中面判定不变) |
| 4 | **准入旋钮坏值结构化** | `OWL_ADMIT_ON_FULL` 坏值:panic → **响亮拒 + `exit(2)`**(§24.5④ 收口;与"配置错误静默 exit 1"红线同族) | `parse_admission_on_full() -> Result<_, String>` |

**门(全绿)**:`owl-store` 63+2+2 · `owl-engine --lib` 36 · CPU 档(准入门 3 + 状态腿探针)·
GPU 真机(状态腿探针 + §22 三门口 + `session_continuity`)。

**A 轴剩余**:**S2b**(GDN 行 = 活跃 turn 借还 + `gdn_slots` 正名 `max_active_turns`)——
唯一还欠的 A 轴主刀;前置(快照三层化、会话不钉设备、`StateStore::borrow_row`/`return_row`)
已全部就位。**B 轴(S4b)待 S2b 之后立项。**

---

## 28. S2b:GDN 行 = 活跃 turn 借还 + 正名 · 2026-10-17

> A 轴的**最后一刀**。目标(§2.3/§6):行(27B 约 162MB/行)从"会话资源"改成
> "**turn 资源**" ⇒ idle 会话对设备**零占用**(KV 表不占、行也不占),
> "能挂多少会话"与"能并发几个 turn"彻底解耦。

### 28.1 落点

| 面 | 变更 |
|---|---|
| `store::StateStore` | **账即池**:`borrow_row()` 返回的行号**就是物理行号**(一本账,不两份);`return_row(row, None)` 归还。S1 就埋好的两口,本刀才接上 |
| `engine::turn` | `ActiveTurn.gdn_slot`(turn 域资源;生命周期随 turn —— 与 spec 态同纪律) |
| `engine::scheduler` | turn 起:`state.borrow_row()?`(耗尽 = 响亮 `Exhausted`;正常路径由准入面 `max_active_turns` 天花板先挡);行号进 `BeginPlan` + `ActiveTurn` |
| `engine::exec::turn_end` | `complete`:还行(状态已由边界快照承担);`abandon`:还行且**不落快照**(弃置的 turn 账面不可信) |
| `engine::session` | `AgentSession.gdn_slot` **删除**;`SessionTable::get_or_create(id)` 不再分配行 ⇒ **"GDN 格耗尽"与"LRU 逐出会话"两条路消失**(`get_or_create_evicting` 退役) |
| 读行点 | `capture_gdn_snap` / draft / prefill_chunk / decode 相位一律从**活跃 turn**读(`self.turns.active.gdn_slot`) |
| 正名 | `gdn_slots` → **`max_active_turns`**(knobs / `PoolPlan` / `StatePool` / `OwlConfig.pool` / server config);env `OWL_MAX_ACTIVE_TURNS`(旧名 `OWL_GDN_SLOTS` 兼容 alias) |

### 28.2 为什么现在做是安全的(S2 换来的三件红利)

1. **复用面被裁到"有快照在场的块边界"**(`reuse_face`,§22 定谳①)⇒ 只要复用了前缀,
   **必然**有快照可恢复;没有快照 ⇒ `m = 0` ⇒ 全量重算 + 零行 ⇒ 状态自洽。
   —— 行可换,是因为"状态跟**内容身份**走,不跟格号走"。
2. **快照三层化**(S2)⇒ 行归还后状态不丢(设备槽满 ⇒ 下沉宿主;宿主满 ⇒ 丢并打点)。
3. **快照拍摄点已覆盖 turn 末**(解码页边界 + prefill 块边界)⇒ 无需新增拍摄点。

### 28.3 门(全绿)

| 门 | 结果 |
|---|---|
| **新门⑤**(`store_admission_gate.rs`):idle 会话不占行 | 行池只给 **2 行**、挂 **6 个键控会话**:每轮终了 `rows_free == rows_total`(**2/2**)✓;六会话的状态留在**快照账**(设备 4 条达 cap 后 churn)✓ |
| `owl-engine --lib` | 36/36(会话表用例改写:不再有"格耗尽"断言) |
| CPU 档 | `cpu_mock_smoke` 3/3 · `store_admission_gate` **4/4** · `cpu_mock_state_leg_probe` ✓ |
| GPU 真机(3080) | `gdn_slot_cross_probe` ✓(`gdn_slot == 0` 判据仍成立:借行返回最小空闲号)· `session_continuity` ✓(跨 turn 答对)· `spec_identity_gate` ✓(逐 token 恒等)· `prefix_cache_hit` ✓ · §22 三门口 ✓ · 状态腿探针 ✓ |

### 28.4 挂账(本刀新增)

1. **会话级 LRU 现在没人调**(`SessionTable::lru_victim` 保留但无调用点)—— 留给将来
   "显式会话上限"策略(§2.3:会话数只该受 host/盘预算约束,故默认不设硬上限)。
2. **行的借还粒度 = turn**:若将来做"一次准入多个 turn"(批调度),行池要按批借
   (算式不变:求和)。
3. **`gdn_slot_profile`/`gdn_slot_fingerprint` 两个探针**仍按行号取样 —— 现在行号是
   turn 域的,取样语义要跟着改(仅诊断面,不影响生产)。

---

## 29. 论文精读:KVMem / ActKV —— 与 §3.6 的更正 · 2026-10-17

> 触发:用户指令("这个是需要看论文的")。读的是全文(`kvmem.pdf` 1087 行 / `actkv.pdf` 1147 行,
> `pdftotext -layout` 抽取)。本节 = 逐条事实 + **对本卷 §3.6/§4 的更正**;
> B 轴的落地设计独立成卷:`docs/arch/kvmem-选择层-设计与施工.md`。

### 29.1 KVMem(arXiv:2609.04852)事实面(逐条可核)

| 面 | 论文事实 |
|---|---|
| 目标 | 24GB 笔记本 5090 + Qwen3.6/3.8-27B NVFP4-MTP:虚拟化 **1M token** 工作区(原生窗 256K 的 4×),单会话 ~50 tok/s;执行视图固定 80K |
| **调度** | step-level:每 step **prefill 后、decode 前**更新一次工作集,decode 全程固定;实测(8 条 OpenHands SWE-bench Lite,128-tok 滑窗)**同 step 内 KL 0.070 bits vs 跨 step 边界 2.59 bits(37.3×)** |
| **索引** | 每块 × 每层 × 每 KV head 一个**去 RoPE** 的 K 均值 `k̄ = (1/|b|)Σk̃`;缺省 **32-token 块** |
| **打分** | `R_b = Σ_m mean_{l,h} softmax_{b'∈C}(q·k̄_{b'}/√d)`(**跨候选块全局 softmax**);sink + recent 保底;选择后**按时间序重排**再物化 |
| **稀疏度** | **top-8 块 = 66.5%** 历史注意力质量、**top-16 = 77.0%**(G5③ 的 66%/77% 逐字对上) |
| **位置一致性** | raw K(**位置无关**)是**重建权威**;活跃 GPU 页里的 K = 当前紧凑位置编码;**页表别名零拷贝装配**;位置变化用**有界 delta re-RoPE**(超阈值从 raw K 重建一次);V 原样搬 |
| **三层管理** | ①**主动 stage-out**(与 chunked prefill 重叠,后台 worker 继续落盘)②**检索感知分层**(GPU 命中保物理页;host 准入按**近期 + 累积检索频率**,非纯 LRU)③**packed + pipelined 重物化**(host gather raw K → 一条 H2D → GPU scatter+re-RoPE;三级流水 + 双缓冲 + CUDA event) |
| 索引放置 | 全量在 **host**,GPU 只 staged **bounded tile** 打分(0.25GiB@256K → 9.5GiB@10M) |
| 实现 | 自研 **QW3**(C++/CUDA):因为要在 prefill 期拿 Q/K、控制页表/位置/搬运 |
| 结果 | LongMemEval-S 85.6%(Full 86.6%)· AgentLongBench **60.87% 反超 Full 59.54%** · >256K 档 40.99% vs Compact+RAG 34.80% · pre-answer 延迟 0.38–1.81s vs Compact+RAG 10.6–39.2s(**11.4–53.8×**)· TTFT 0.43s@256K→1.60s@10M · prefill ~2.6k tok/s · decode 78–85 tok/s · DeepSWE Pass@1 48.4 vs 43.8 / **Pass@4 93.8 vs 81.3** / prefill 时间 −55.1% |
| **自认局限** | ①虚拟化的是**可寻址工作区**,不是单次注意力范围(每步 ≤ min(模型窗, GPU KV 预算));②**KV 复用非数学等价**(历史 KV 在原因果上下文下算出;靠 re-RoPE + 对当前 query 重新 prefill 缓解;检索可能漏块);③存储开销 KV ≫ 文本(10M ⇒ NVMe 324GB) |

**与我们已有的对齐(白送的三件)**:页表别名 ↔ 我们的 `PoolLayout`+槽池;三层账 ↔
`TierLedger`;packed 的 host 连续槽 ↔ `SlotTable`/`Connector`。**论文为此自研了整个引擎,
我们的 A 轴已经把这三件做完了** —— 这正是"A 轴先行"的回报。

### 29.2 ActKV(arXiv:2609.31395)事实面

| 面 | 论文事实 |
|---|---|
| 定位 | **KV 压缩/淘汰**(不是虚拟化):agentic 推理的显存与吞吐 |
| **淘汰** | action-critical 条目与 observation/reasoning 偏好**不同**且跨轮**稳定**(AAV:top-p 90% 命中向量);**attention-aware LRFU** `KS_t = λ·KS_{t-1} + Attn_t·{hit}`,每 ORA 轮末 `TopK(KS,B)` |
| **预算** | token 置信 = top-k 平均 log-prob → 滑窗 **min-pooling** + bottom-b% 分位 → 一阶拟合斜率 α;**α<0(置信下滑)⇒ 预算放大** `B' = min(B·scale_up, B_upper)`;证据:预算过紧 ⇒ 模型出现 "wait / I guess / let me check" 式**低置信补上下文** |
| **page-aware 原语** | 统一 `CalAttn / Eviction / Compaction`;两核:①**recovery-based 注意力**(用 softmax LSE 反推 `attn=exp(logits−LSE)`)②**conflict-free in-place compaction**(保留条目压到尾部 `X=⌈B/block_size⌉` 块,源/目的槽不相交 ⇒ 一遍并行拷贝) |
| 结果 | 长轨迹 **98.53% FullKV 准确率 @ 25.98% 峰值显存**;token/task 吞吐 3.97×/3.58×;paged 压缩核平均 **10.41×** 快于 gather 式;块 16 / kv_heads 8 / head_dim 128 |

**归位**:ActKV 是**另一根轴**(丢内容),我们三层是**下沉**(内容保留)⇒ 严格更强;
但它的 **LRFU** 与**置信度预算**两件可直接摘(接 `EvictPolicy` 与 B 轴的预算在线调节),
见 B 轴卷 §6。

### 29.3 对本卷的更正(两条;其余 §3.6 经核对**成立**)

1. **§3.6.3 位置方案:A → B**(改判)。原建议"一期走 A(原位置稀疏召回,零 re-RoPE)"
   是未读全文时的纸面倾向;论文的做法与理由是:
   - 论文用**紧凑窗口 + re-RoPE**(= 我们的方案 B),且给出完整实现律:raw K 真源 +
     入块重打 RoPE + **有界 delta re-RoPE**(超阈值重建,防低精度累加);
   - **页表别名**让装配零拷贝(与我们 A 轴的槽池/页表同构);
   - 我们的方案 A 要在**原位置**留洞 ⇒ 页表稀疏 + 位置空间 = 原生窗(262k 位置墙)
     ⇒ 与"工作区 > 原生窗"的目标直接冲突;
   - B 的紧凑位置与我们 **XQA `tokensPerPage=32` 的连续页表**天然对齐。
   ⇒ **改判 B**;A 仅作"位置空间足够时的零误差退路"保留。
2. **§3.6.1 第 4 行"三层策略"补细**:host 准入 = **近期 + 累积检索频率**(非纯 LRU);
   stage-out **与 chunked prefill 重叠**(后台 worker);重物化 = **packed + pipelined**
   (host gather → H2D bulk → GPU scatter+re-RoPE,双缓冲 + event)。这三条要写进 B 轴施工序。

**其余 §3.6 条目经全文核对成立**(块级检索、Mean-K、两视图分离、32-token 块、top-8/16 覆盖率、
"论文不虚拟化复发状态 = 我们的增量")。

---

## 30. 单一调度器:判据归一(容量调度的唯一产地)· 2026-10-19

> 用户拍板(本会话):"继续做引擎最重要的容量调度管理 —— 先**收敛判据**为单一调度器。"
> 起因是可 grep 证的三处分裂(§30.0)。本刀只做**判据归一 + 两腿同堂**,
> 不碰并发(单活跃 turn 模型不变)、不碰盘层。

### 30.0 病根(接线前的代码事实,逐条可查)

| # | 事实 | 证据 |
|---|---|---|
| ① | `Rel::plan_evict`(两腿合并 + `EvictPolicy` 出序)**零调用点** | `grep plan_evict` 只有定义与自有测试 |
| ② | 生产路径的"谁该走"由 `hooks::plan` 决定,而它的签名只吃 `&BlockLeg` | `hooks.rs`(旧)`plan(leg: &BlockLeg, …)` |
| ③ | 因此 `EvictPolicy::{Lru,Lrfu}` / `note_recall` 只影响死代码 —— **策略对象装了但没上生产** | boot `set_evict_policy` + ① |
| ④ | 状态腿(GDN 快照)不在 H1/H2/H5/N6 的保洁计划里;只能等 capture 撞满时被 engine 池被动逐 | `exec/spec.rs::snap_pick_slot`(engine 内 rank + LRU = 第二套判据) |
| ⑤ | 同步兜底 `ensure_for_len` 自带第三套逐叶循环 | `kv/mod.rs::ensure_for_len` |

⇒ 同一件事(腾设备)三套规范 = AGENTS.md 的"接口不唯一"红线;而且**策略对象是死的**。

### 30.1 接口(唯一入口)

```rust
// store::hooks —— 类型
pub struct Need { pub slots: usize, pub snap_bytes: usize, pub why: NeedWhy }
pub enum  EvictOut { Kv { chain, sink, span }, Snap { key: SnapKey, sink } }   // 两腿同一种票
pub struct HostRoom { pub kv_slots: usize, pub snap_slots: usize }

// store::Rel —— 生产唯一入口
pub fn plan_need(&self, need: Need, protect: &Protected) -> Verdict
```

- **两腿同一种票**:KV 链/叶与快照进同一张 `Verdict.out`;执行仍归 engine(§21 不破)。
- **双目标**:`slots`(KV 设备槽)与 `snap_bytes`(状态腿设备字节)**各自独立满足** ——
  KV 缺槽不能靠丢快照补,反之亦然。`deny` 文案带两腿数字。
- **策略对象第一次上生产**:`plan_need` 内部用 `self.evict`(`Lru`/`Lrfu(attention)`)
  给候选出序;`note_recall` 的召回信号从此真的影响"谁先让位"。

### 30.2 算法(两轮 + 事实档 + 策略档)

```text
轮一 先下沉(内容不丢):KV 链级候选(§14.6)+ 状态腿快照候选 → 同一候选表
      ├ 事实档:Transient 快照(生成段,逐出即丢)排最前
      ├ 策略档:同档内 EvictPolicy::order(LRU / LRFU)
      └ 出票条件:sink = Host,且**该腿宿主装得下**(KV 整链 span ≤ 剩余槽;快照 ≤ 剩余张)
轮二 再驱逐(内容真丢):未满足的目标 → KV 叶级(不动链头)+ 状态腿候选 → 同序 → sink = Drop
③   仍不足 ⇒ deny(两腿分别报"短多少",不许动活跃面)
```

**关键裁决(沿用旧律,不新造)**:
- KV 下沉粒度 = **整链**(§14.6;半截不出票),驱逐粒度 = **叶**(丢尾部不动链头);
- `Transient` 只 Drop 不下沉(下沉一条 159 MiB 换不回共享前缀价值,§9.1(6)a);
- `protect`(活跃 turn 的链/点名快照)绝不入 `out`。

### 30.3 删了什么 / 保留什么

| 项 | 裁决 |
|---|---|
| `hooks::plan`(只吃 KV 腿) | **删除**;能力并入 `plan_capacity` |
| `Rel::plan_evict` + `EvictItem`/`EvictPlan` | **删除**(零调用点;其"两腿合并 + 策略出序"能力并入 `plan_need`) |
| `KvStore::ensure_for_len` 同步驱逐 | **保留为兜底**,不是第二套判据:报告面 `ops.ensure_evicted > 0` = 调度器失职的**告警信号**(正常路径应恒 0) |

### 30.4 门(全绿)

- store 单测 **91**(含新增 2:`plan_need_moves_durable_snapshot_to_host_when_room`、
  `plan_need_never_offloads_transient_snapshot`)+ snap_leg 3 + store_e2e 4;
- engine lib **55**(含新增 `idle_node_offloads_device_snapshot_via_store_plan`:
  store 判 Host ⇒ engine 执行臂真沉、设备槽释放);
- 回归:old `hooks::plan` 的次序门改写为 `plan_kv`(走 `plan_capacity`),
  语义断言不变(压力/空转零缺口也搬、优先 Host、整链凑满目标)。

### 30.5 行为变化(逐条)

1. **Idle 节点现在带状态腿目标**:`snap_bytes = 一条快照`,且**仅当状态腿宿主有位**时给
   (宿主关/满 ⇒ 0,不产生永远凑不够的 deny)。空转每 `IDLE_HK_EVERY=16` 拍整理一次。
2. **两腿宿主独立开关**:KV 宿主关 ⇒ 该腿目标清零(旧行为:宿主关不换出,**也不无谓驱逐**);
   状态腿宿主关同理。不会因为一腿开就把另一腿 Drop 掉。
3. **Pressure 节点暂不带状态腿目标**(`snap_bytes = 0`):压力档的目标是"把驱动 free 拉回",
   两腿字节如何分摊需显式策略 ⇒ 挂账 §30.6-①。
4. `EvictOut` 枚举化:engine `apply_verdict` 对 `Snap{Host}` 走 `snap_offload` + 清槽,
   `Snap{Drop}` 走 `drop_snap_host + state.drop_snap + 清槽`;新增 `Applied.snap_offloaded/dropped`
   与 metrics `store.hook.snap.out|drop`。

### 30.6 挂账(本刀新增)

1. **Pressure 的状态腿分摊**:N6 目标是"驱动字节",两腿都能腾;如何分摊(KV 优先?按可搬量比例?)
   需显式策略 —— 本刀先保持 KV-only(行为不变)。
2. ~~**engine 池的 victim 判据**:`snap_pick_slot` 仍用 engine 内的 importance rank + LRU
   (第二套判据)~~ ✅ **已结(§30.7 同会话第二刀)**:`snap_pick_slot` 只做 key→slot 映射,
   判据走 `Rel::snap_victim_order`(与 `plan_need` 同一份排序)。
3. **`ensure_for_len` 关兜底**:等调度器在真机长跑把 `ops.ensure_evicted` 压到 0 后,
   把兜底改成 `debug_assert` + 告警(消灭第三套判据的最后一块)。
4. 状态腿池缩容 / 盘层 / 准入 Queue 定时器 / 对账口进服务面 —— 沿用 §20 挂账。

### 30.7 同会话第二刀:状态腿出序归一(§30.6-② 收口)· 2026-10-19

**病**:`exec/spec.rs::snap_ensure_slot` 的 victim 选择在 engine 池里自带
`rank(Transient/Durable) + 池内 last_use` —— 与 store 的容量调度各一套规范;
且 store 的 `StateStore.last_use` 只在 capture 登记时更新,**回灌/复用不触达**
⇒ 池与账的"冷热"会漂。

**改法(判据仍在 store,物理映射留 engine)**:

```text
Rel::snap_victim_order(protect) -> Vec<SnapKey>
  = state.evict_candidates(protect).filter(tier == Gpu)
  → hooks::order_candidates(同 plan_capacity 的:事实档 + EvictPolicy)
  → 快照键序

StatePool::snap_pick_slot(key, order)   // engine 只做映射
  Hit(同 key 在设备) → Free(空槽) → order 中第一个仍在槽的键 → Victim
  兜底:order 用尽(判与选之间被移走) ⇒ 池内 LRU(物理退化,不是语义档)
```

**触达同步**(两处,消灭漂移):
- `snap_onload` 成功后 `state.touch(key)`(回灌 = 使用);
- `snap_ensure_resident` 的设备层快路 `state.touch(key)`。

**门**:
- store `snap_victim_order_is_the_plan_order`:三张快照(老 Durable / Transient / 新 Durable)
  ⇒ 序 = `[Transient, 老 Durable, 新 Durable]`,且与 `plan_need` 的 `Snap` 票**逐条相同**;
- engine `snapshot_victim_follows_store_order`:池内放置序**故意与 store 序相反**
  (旧实现会选池 LRU 的 k_b)⇒ 新实现选 store 序第一的 k_a,槽号仍是 k_a 的物理槽;
- 回归:engine lib 56 / store lib 92 / workspace check 干净。

### 30.8 真机验证(27B AWQ INT4 / 3090 Ti 对 / 3080 无关)· 2026-10-19

**档**:`/tmp/owl-sched-27b.toml`(临时)—— `pool_tokens=16896` / `snap_max=4` /
`host_cache_mb=2048` / `snap_host_mb=1024` / `gdn_slots=8` / `reserve=512MiB` /
`prefill_chunk=512` / `prefill_bucket=off`。22 个随机 prompt 请求(96 tok 输出)+ 1 个 600 词长 prompt。

**通过项(判据归一在真机成立)**:

| 观测 | 值 | 说明 |
|---|---|---|
| 请求成功率 | **22/22**(2.63–2.88s) | 无 500、无崩 |
| `store.node.idle` | **29** | H5 空转整理在跑 |
| `store.snap.out` / `.bytes` | **19 / 3.03 GB** | 状态腿执行臂真沉(本刀新增能力) |
| `state.snap_gpu` / `snap_host` | 4 槽 637MB / 6 条 955MB | 池满 ⇒ 溢出到宿主 |
| `store.hook.out` / `kv.host_blocks` | 258 / 258 | KV 链级下沉(既有路径) |
| `store.node.pressure` | 197 | N6 在跑(见下 ⚠️A) |
| `pool` | 45 块 / 256 MiB(贴上限)、evicted 17636 | 池上限与顶出生效 |
| `snap_victim_order` | (单测门) | 池听 store 序 |

**修正:`hook_fail` 分类(本刀副产,已落)**

真机首次开宿主层 ⇒ `hook_fail=4352`,**全部**是同一条日志:
`部分换层(片和 ≠ 登记整块) … name=kv.k.7 片和=720896 登记=34603008`。
根因:KV 池**一个物理块 = 多层片**(`kv.k.<层>` × 每页字节),而 `swap` 逐片发
`MoveEvent` ⇒ "片和 < 登记整块"是**常态口径**,不是记账失败(设计卷 §12.13 的
两本账粒度差)。判据修正:

- `LedgerBook` 新增 `partial_moves` 计数(片级换层**跳过**;层账以腿自报
  `kv.host_blocks` 为准);`hook_fail` **恢复为真失败信号**;
- 日志限流(首条 + 每 1024 条);报告面 `/debug/vram → events.partial_moves`;
- 复验:同一档 22 请求后 **`hook_fail=0` / `partial_moves=4352`** ✅。

**⚠️ 暴露的存量问题(本刀不修;逐条立案)**

- **A. `pressure_probe` 阈值导致常态压力**:`target = max(2×reserve, 1GiB)`,
  而 27B 常态 `free ≈ 847MiB < 1GiB` ⇒ **每拍 N6**(197 次)+ `store.degrade=184`(凑不够)。
  建议判据改为"`free < reserve` 才报压力"(reserve = 硬预算线,才是真危险);
  或给 N6 加冷却。
- **B. 「腾设备槽」对驱动 free 零收益(结构性洞见)**:设备 KV 池与快照池都是
  **固定分配、不缩容** ⇒ Idle/N6 把缓存/快照腾走后,`reserved` 不变、驱动 free 不变
  (实测 `kv_free=528/528` 全空但 free 仍 847MiB)。收益只在"池内/快照池内让位",
  代价是下次命中全要回灌。⇒ Idle 整理 KV 面应重新裁决(要么池可缩容,要么只在
  有明确溢出压力时才搬)。
- **C. 快照宿主抖动**:`snap_max=4` + 每请求 3–4 条快照 ⇒ capture 频繁选 victim;
  `snap_host_mb=1024`(≈6 条)满了以后每次下沉逐一条 ⇒ `snap_host_evict_bytes=2.07GB`
  (同一内容反复搬运)。⇒ 参数需与快照产量匹配(snap_host 预算或 snap_decode_stride)。
- **D. 失败 turn 的未归还清单**:长 prompt 撞硬预算 500 后,
  `N4 未归还清单:turn 22 名下仍有 622 块 / 114037 KiB`(turn 23:295 块/246MB)——
  失败路径的相位清理有缺口(与本刀无关,但值得立案)。
- **E. 卡配贴顶**:`live 22.9G / 可用 ≈23.1G`,长 prompt 撞 `avail=14MiB` ⇒
  硬预算如实拒(`PoolExhausted`,引擎不死、丢本 turn)。与 Idle/判据归一无关
  (池 reserved 是整卡口径),属档位配置问题。

---

## 31. 第三层:磁盘(S6)· 2026-10-19

> 用户裁决(本会话):"既然有这种丢失的问题,就先把第三层也接进来 —— 启动时可以在配置里
> 指定能用多少磁盘和内存;**KV 和 GDN 快照不要那么着急地扔掉**,先移到内存、再移到磁盘,
> 然后去管理它;磁盘的话启动时可能要加载一下,从磁盘里回加载进来。"

### 31.1 配置面(唯一旋钮)

```toml
[pool]
host_cache_mb = 2048   # 宿主 KV 分片(MiB;0 = 关层)
snap_host_mb  = 1024   # 宿主状态腿分片
disk_cache_mb = 4096   # 磁盘 KV 分片(S6;0 = 关层)
snap_disk_mb  = 2048   # 磁盘状态腿分片
disk_dir      = "/var/lib/owl/kv"   # 内容目录(空 = 关层;两腿子目录 kv/ 与 snap/)
```

**唯一算式仍在 store**(`tier_budget_of(host_mb, snap_host_mb, disk_mb, snap_disk_mb)`)——
boot 造账与 `Rel::open` 的交叉校验同源(§12.13);分片之和超总账 = **响亮拒**。

### 31.2 `DiskTier`(**store 零 I/O**)

```text
plan_spill(key) -> SpillPlanOf { key, tmp, path, bytes }   调用方写 tmp + rename(原子)
commit_spill(&plan)    落账(内容已写定);幂等(同键覆盖只触达)
plan_fill(key)  -> FillPlanOf { key, path, bytes }         调用方读盘
commit_fill(key)       触达;**不删条目**(磁盘 = 权威副本,宿主 = 缓存)
touch(key)             命中/回填 = 触达(驱逐序输入)
plan_evict(need, protect) -> { victims: [(K, Path)], freed_bytes }   只出计划(最老先)
commit_evict(keys)     出账(文件删除由调用方)
adopt(entries)         启动扫描后**回填账**(大小不符/超预算/重复 ⇒ 隔离并逐条列出)
```

- **文件名 = 内容身份**(`kv-<hex>.bin` / `snap-<hex>.bin`)⇒ 重启可逆解析;
  `.tmp` 残件在扫描时被清理(崩在 rename 前的半块不会被当成好块);
- **有界**:`budget_bytes = 0` = 关层;越界 = 响亮拒(调用方先 `plan_evict`);
- **零 I/O**:store 不碰文件系统(§21);真读写由 engine 在 `spawn_blocking` 里做
  (`engine/src/disk_io.rs`:原子写 / 读回校验大小 / 删文件 / 扫目录)。

### 31.3 层级律:**逐层,不跨层**(设备 → 宿主 → 磁盘)

`plan_capacity` 现在是三段(§30.2 的两轮 → 三轮):

```text
轮一  设备 ⇒ 宿主(Host 票;宿主有位且装得下)
轮1.5 宿主 ⇒ 磁盘(Disk 票;宿主满时,把**宿主里最冷**的挪到盘 —— 腾宿主位,内容不丢)
轮二  仍不足 ⇒ Drop(KV 叶级 / 快照;内容真丢)
```

**为什么设备不直连磁盘**:层级是"贵→廉"的缓存栈,跨层搬要临时缓冲且打乱"内容最近在哪"
的一致语义;逐层的代价只是**一拍延迟**(需求/压力节点每拍都跑),换来内容永不先丢。

**`host_relief`**:本轮若出过「宿主 → 磁盘」票 ⇒ **不出 Drop、不 deny** ——
设备缺口留一拍(下一拍宿主有位,轮一继续接)。真丢弃只在"宿主+磁盘都动不了"时发生。

**`Transient` 快照**(生成段,逐出即丢)**两级都不下沉**(写 159 MiB 换不回共享价值)。

### 31.4 启动回加载(账先回来,内容惰性读)

```text
boot: attach_disk(disk_dir, kv_bytes, snap_bytes)   ← 两腿分片(超总账 = 拒)
      scan_dir_sync::<ChainHash>(kv/) + ::<SnapKey>(snap/)
      adopt_disk(kv_entries, snap_entries)          ← 回填账(不搬内容)
      状态腿:set_tier(key, Tier::Disk)(层账精确到盘)
日志: 磁盘层:配额 X MiB(KV / 快照)· 目录 · 回填 KV a/n 快照 b/m 条 · 隔离 …
```

**为什么惰性**:27B 一条快照 159 MiB、KV 块 2 MiB;全量预读会把启动拖垮。
账先回来(命中判定/调度立刻可用),内容在**第一次需要时**读回宿主
(`fill_kv_from_disk` / `fill_snap_from_disk`;回灌口前置 + `snap_ensure_resident` 的 Disk 分支)。

### 31.5 门(全绿)

- store **100**(+4:盘层容器 6 项含 adopt 隔离/文件名往返/预算/驱逐序;+2:第三层出票 +
  `Transient` 不写盘)
- engine **57**(+`disk_spill_and_fill_roundtrip`:CPU mock 真写文件、层账到 Disk、回填回 Host、
  文件大小 = 一条快照)
- iface 11 / cpu 11 / shared 82 / workspace check 干净

### 31.6 挂账(S6 之后)

1. **设备直连磁盘**(省一拍延迟):需临时槽台(不经宿主预算)+ store 侧的"两跳计划";
   实测一拍延迟在压力场景是否可忽略后再定。
2. **盘层驱逐序接 `EvictPolicy`**:现在盘内是纯 LRU(`last_use`);与 KV/快照同策略可后续统一。
3. **盘层的启动审计**:`adopt` 只校验大小;内容哈希/校验和(可选,代价 = 读全盘)。
4. **跨进程复用**:目录独占(同一目录给两个引擎 = 互相扫到对方块)—— 需要锁文件/租约。
5. KV 腿的盘层在层账里仍记 `Tier::Host`(`locate` 二值);磁盘条目数在 `DiskStats` 单独报。
   `Entry.slot` 升三值(加 `on_disk`)可让层账更精确 —— 改动面大,按需再做。

### 31.7 跨重启复用的语义裁决(2026-10-19 同日实测后)

**状态腿 ✅ 可跨重启复用**:`SnapKey` = 边界块的**内容身份**,独立可恢复 ⇒
`adopt_disk` 为盘上快照**重建 `StateStore` 账**(tier = Disk;`durability` 盘上没存 ⇒
保守记 `Durable`)⇒ 重启后 `probe` 直接看得见(`RecallMode::Exact`),第一次需要时
`fill_snap_from_disk` 读回。门:`disk_adopt_on_boot_restores_ledger`
(第一次进程落盘 → drop → 第二次进程同目录 ⇒ 账回填、层账 = Disk、**内容未预读**)。

**KV 腿 ❌ 不跨重启复用(本版)**:块的身份是**父链哈希** —— 盘上文件名只有块身份,
恢复不出"哪些 token 对应哪些块"(树需要父链与逐块指纹)。⇒ boot 见 KV 盘目录残留 =
**清掉并 WARN**(占配额还骗自己"有缓存"比删掉更糟)。

**⚠️ 单腿不可达(2026-10-19 真机定谳)**:`RecallHit::mode()` 要求 **KV 腿在场**才能
给出 `ContextOnly`/`Exact` —— 只有快照腿(`None, Some(_)`)是 `Miss`。而索引重建需要
KV 树(见下)⇒ **本版的快照回填"账上可见、实际选不中"**(内容不丢,但复用要等 S6.1)。
真机实测(shutdown/restart 同目录):`回填 KV 0/0 快照 6/6 条` + `state.snaps = 6` +
层账 Disk;KV 侧 `清掉 14 个残留块`(WARN)。

**挂账(S6.1,KV 跨重启复用;含让快照回填真正生效)**:盘上为每条链存一份**索引文件**
`kv-<tail>.idx`(整链 token 序列 + 逐块 `(hash, bytes)`)。boot 用它**重算块身份**
(`hash_block(parent, tokens)`,指纹自然一致)并**重建树条目为"非设备层"**
(`slot = None`;内容在盘上 ⇒ `locate` 归 `Tier::Host` ⇒ 命中后 `fill_kv_from_disk`
自动把它读回宿主 —— 该链路已打通)。代价 = 索引写放大(链级一次,不是逐块)+ 恢复期树插入;
收益 = 重启后**前缀命中率不归零**且**快照回填同时生效**(长上下文 agent 场景价值大)。

**写入点**:索引必须由**链级**动作写(`spill` 是逐块的,没有 tokens)⇒ engine 在
"链整体离开设备"或 turn 收口时落索引(`sessions`/`turns` 有 token 序列);store 只出
"该链现在在盘上"的计划,索引内容由 engine 提供(与 `register_chain` 同源)。

### 31.8 S6.1:跨重启复用落地(2026-10-19 同会话)

> 用户当日裁决:"你为什么会用双卡呢?用单卡加载小模型 + **硬预算显存约束**,
> 就可以**快速触发回收**" —— 验证方法论(见 §31.9);并开工 S6.1。

**交付**:

| 件 | 落点 |
|---|---|
| `ChainIndex`(魔数 + 页宽 + token 序列;编解码/截断;**只增不删**) | `store/src/disk_index.rs`(新) |
| 树的无槽登记(`insert_offloaded`;`TreeUpdate.inserted` 升 `Option<u32>`) | `kv/tree.rs`、`kv/mod.rs` |
| `KvStore::host_chain_candidates`(叶为锚 + **整链在宿主**) | `kv/mod.rs` |
| `Rel::adopt_disk_chains`(逐块重算身份 + 按内容在场截断 + 记 `kept`) | `store/src/rel.rs` |
| 索引写(turn 收口一次,`kv-<tail>.idx`,同 tail 幂等) | `engine/src/{running.rs,exec/turn_end.rs}` |
| boot:内容入账 → 索引解码 → 重建树 → **孤儿内容清理** | `engine/src/engine.rs` |
| 票**链级**化:轮 1.5 出 span=整链,engine 按 `chain_blocks` 逐块搬盘 | `hooks.rs`、`running.rs` |

**整链律(S6.1 真机定谳)**:盘上要能"从根连续"重建,所以

1. 轮 1.5 的锚必须是**叶**(以任意宿主块为锚会退化成只搬根块 ⇒ 重启只能重建 1–2 块);
2. **整链都在宿主**才出票(半链 = 重启不可用,与 §14.6 同一律);
3. 同拍可以"设备→宿主→磁盘"两跳(`apply_verdict` 按 Host 票在前、Disk 票在后执行),
   所以轮 1.5 **不再排除轮一刚沉的链**(早期版本用 `picked` 排除 ⇒ 磁盘票恒零)。

**配置律(新)**:`host_cache_mb / disk_cache_mb` 必须分别**容纳最长链**
(`ceil(ctx/32) × 块字节`),否则整链下沉/整链搬盘永不发生(0.8B 真机两次踩中:
宿主 1 MiB / 盘 1 MiB 都装不下 5–6 块链)。配得太小 = 退回逐叶 Drop(内容丢)。

**真机结果(0.8B / 单卡 3080 / 池 16 页 / host 16 MiB / disk 16 MiB)**:

```
运行期:store.disk.index.write=13 · store.disk.spill=24 · disk.kv=24 块 · disk.snap=2 条 · host=45
        (bin=24 idx=17;P warm cached=160/173)
重启  :回填内容 KV 24/24 快照 2/2 条
       跨重启复用:索引 17 条 ⇒ 重建 5 链 / 24 块 · 跳过 12(索引孤儿) · 截断 0
```

**跳过 12 条 = 索引孤儿**(turn 收口写索引,但该链的内容从未溢写到盘,或早已被逐)——
boot 以"内容在场"为准跳过并**清掉未被引用的内容**,索引文件保留(只增不删,~32 KB/链)。

**门**:store **106**(+ 索引往返/拒坏魔数/截断 · 叶锚整链候选 · 重建树/截断)·
engine **59**(+ `disk_chain_index_enables_cross_restart_reuse`:turn 写索引 → 溢写 →
重启重建 → **同 prompt 命中**)· workspace check 干净。

### 31.9 验证方法论(用户 2026-10-19):单卡小模型 + 硬预算,快速触发回收

26–27B 双卡档"装载 20 s + 每请求 2.7 s + 卡常被同侪占",不适合频繁验证存储层。改用:

```text
0.8B + 单卡(ordinal 3, 3080 20G) + 池 16 页 + host/disk 各 16 MiB + prefix_cache_share 0.9
⇒ 每请求 0.2 s,几次请求就把"设备→宿主→磁盘"三级全部触发,重启也在 1 s 级
```

**纪律**:①配额要容得下最长链(见 §31.8 配置律);②`prefix_cache_share` 拿高
(0.9)可让设备池更快紧;③同 prompt 发两次看 `usage.prompt_tokens_details.cached_tokens`
(第一次 0、第二次 >0 即命中);④停服务用 `ss -ltnp | grep <port>` 拿**真 pid**
(`setsid bash -c '…server…' &` 的 `$!` 是 bash,不是 server)。

---

## 32. 会话感知淘汰(per-session tail-first)· 2026-10-19

> 用户裁决:"每个 session 块能感知它属于哪个会话;**淘汰 KV 块和 GDN 快照时,
> 优先淘汰同一 session 的末端**,而不是从头开始淘汰 —— 从头淘汰,后面的全变垃圾。"

### 32.1 病根(两处)

1. **快照的 LRU 是"时间序",不是"结构序"**:同一会话的多个边界快照(`depth` 递增),
   最后拍的 `last_use` 最新 ⇒ 纯 LRU 会**先逐链头**(浅边界),而链头一丢,
   该边界之后的所有状态都无法 `Exact` 续跑 —— 正是"后面全变垃圾"。
2. **块/快照没有会话归属**:跨会话只有一个全局 LRU,无法回答"这个块属于谁",
   也无法按会话做"从末端削"。

### 32.2 交付

| 件 | 落点 |
|---|---|
| `EvictCandidate` 加 `session` / `depth`(KV 叶 = 链深;快照 = 边界块号) | `policy.rs` |
| `KvStore.owners`(链 → 会话)+ `session_of`/`depth_of`/`register_chain_owned`/`set_owner` | `kv/mod.rs` |
| `StateStore`:`SnapEntry` 加 `session`/`depth`;`register_snap_owned` + `snap_session` | `state.rs` |
| **会话档**排序(事实档 Transient 之后):**全局深度降序** → 同层按会话冷热 → 同 (会话,深度) 内才交给 `EvictPolicy` | `hooks.rs::order_candidates` |
| engine 写入口:turn 收口 `register_chain_owned(..., sid)`;prefill 边界 `register_snap_owned(..., sid, fed/page)`;增量登记 `set_owner` | `exec/{turn_end.rs,spec.rs}` |
| 诊断面:`/debug/vram → state.evict_order`(前 8 条:key/session/depth/transient) | `rel.rs::snap_evict_preview` + `debug.rs` |

**`NO_SESSION = u64::MAX`**:会话 id 0 是**合法 id**(临时会话从 0 起号)⇒ 不能用 0 表示
"无主";无归属的候选排在所有会话之后。

### 32.3 语义(两条规则,用户 2026-10-19 精修)

```text
① 全局深度降序:每个会话都先从**自己的最深边界**(末端)削 —— 浅的(前缀)留到最后
   ⇒ "优先留存已计算内容"对**每个会话**成立,而不是"最冷会话被一路削光";
② 同深度层内:会话冷热(该会话最冷候选的 last_use 小者先)→ 再交给 EvictPolicy
   (LRU / LRFU / 注意力召回信号只在同会话同深度内决胜负)。
③ 事实档在前:Transient(生成段,逐出即丢)永远最先,深度规则只管 Durable 与 KV。
```

**为什么深度优先于策略**:新鲜度是"跨会话的时间维",而"从末端削"是**单会话内的结构约束**;
结构约束先满足,时间维只在同层同深度兜底。

### 32.4 门(全绿)

- store **108**(+`session_tail_first_eviction_order`:同会话 depth 3/2/1 即使 depth3 **最新**
  也最先逐;+`kv_register_chain_owned_records_session_and_depth`)
- engine **60**(+`turn_records_session_ownership_for_kv_and_snaps`:turn 收口后
  KV 尾块与快照都归本会话,快照带边界深度)
- workspace check 干净

**真机(0.8B/单卡/snap_max=4,两会话各 128 tok 输出)**:

```text
/debug/vram → state.evict_order:
  depth 10 session 1 transient   ← Transient 档(生成段)先
  depth  5 session 0 durable
  depth  4 session 0 durable     ← 同层(depth 4)冷会话先
  depth  4 session 1 durable
```

**残余(挂账)**:①`oldest_first` 的 KV 叶按"链深"参与全局深度序(长链先削尾)——
语义正确但"公平性"可按需再加(按会话配额);②`evict_order` 只是**候选序**,
一次淘汰多条时仍会连续取同一会话的多层(浅层排在后面,实际很少轮到);
③会话归属在**跨重启**后不恢复(NV 盘上索引不含 sid;需要则加进 `.idx` 头)。
