# KVMem 选择层(B 轴)—— 设计与施工

> 立项 2026-10-17。**A 轴(容量层)已收官**(store 设计卷 §20-§28):内容身份命中 +
> 三层搬运(设备/host/盘位留)+ 需求驱动保洁 + 准入判决 + 状态腿三层化 + 行借还。
> **本卷 = B 轴(选择层)**:单卡 262k 的**唯一路径**(store 设计卷 §2.1 算术 / 裁决⑥)。
>
> **依据(论文精读,不凭印象)**:
> - **KVMem**(arXiv:2609.04852,1M 工作区 @ 24GB 笔记本,~50 tok/s)—— 本卷主体蓝图;
> - **ActKV**(arXiv:2609.31395,98.53% 准确率 @ 25.98% 峰值 KV)—— 摘两件(见 §7);
> - 判决与更正的逐条依据 = store 设计卷 **§29**(论文精读 + §3.6 更正)。
>
> **三条铁律**:①**不改 A 的接口**(裁决⑩:B 只在 A 之上加选择层);
> ②块 = **32 token = 物理页 = XQA `tokensPerPage=32`**(§3.6.2 定案);
> ③**非数学等价必须留对照门**(论文自认 fidelity 局限)。

## 1. 需求 = 端到端链路

**链路 = 一个 agent step 的一次模型调用**(触发点 → 终止条件):

```text
① prefill 当前 query 段          顺手采 Mean-K 索引(去 RoPE;零额外扫描)
② 召回点(prefill 后 / decode 前)  打分 → 选块 → 装配执行视图(页表别名 + 紧凑位置 + re-RoPE)
③ decode                        工作集**固定**(不重选;新生成 KV 追加在视图尾部)
④ step 收口                     完成块 stage-out(与 prefill 重叠)/ 索引增量落 host / GDN 快照
```

**为什么召回点只在 step 边界**:论文实测(8 条 OpenHands SWE-bench Lite rollout,
128-token 滑窗)**同 step 内相邻窗 KL = 0.070 bits,跨 step 边界 = 2.59 bits(37.3×)**
—— 历史相关性在 step 内几乎不变、跨 step 才跳变。**我们的"段边界"就是 step 边界**,
钩子框架(store 卷 §14)已在位,本轴只需加"选择"这一步。

**⚠️ 时间预算(2026-10-18 实测;防误读:这是「每 step 一次」,不是「每 token 一次」)**:

| 项 | 262k ctx(8192 块) | 频率 |
|---|---|---|
| 召回点**打分**(快路,纯 host) | **165ms**(基线构建 323ms;见 §8.10) | **每 step 一次** |
| **选择**(top-k + sink/recent + 时间序重排) | 0.1ms | 每 step 一次 |
| decode 单 token | — | 每 token;**工作集 = 选中块**(2k tok ⇒ 每 token 只读原 KV 的 ~1/128) |

摊薄:一个 agent step 通常几百~几千 token ⇒ 召回摊到 **~0.3ms/token**;且是**纯 host**
的活,可与 prefill/decode **重叠**(④ 的 stage-out 预取同一窗口)。
**反面对照(为什么要花这笔钱)**:不做选择 ⇒ 262k 的 KV ≈ **17GB(f16)/8.6GB(fp8)**
装不进 24G 卡,且每 token 要流全部 KV(≈10ms 的带宽下限)。
⇒ 召回开销 = 「跑得起的门票」,不是「每 token 的税」。规模档:16k=19ms / 64k=78ms / 262k=165ms。

**什么情况才真该担心**:要**每 token 重选**的用法(每 token 都跨 step)—— 那种场景
论文同样实测过:同 step 内重选无收益(相关性没变)。真要更频繁召回,对策是
**异步预取 + 增量打分**(只给新块补分),不是硬扛全量打分。

## 2. 接口(伪代码;落点逐个标注)

```rust
// ── 索引(采集在 prefill;账进 store 的三层)────────────────────────────
struct MeanKEntry { block: ChainHash, layer: u16, kv_head: u16, k_bar: Vec<f16> }  // 去 RoPE 的 K 均值
// 采集点:prefill 的 K 写完处(块边界)→ 逐 (层, KV 头) 求均值 → host
fn collect_index(k_block: &[K], n_layers, n_kv_heads) -> Vec<MeanKEntry>;

// ── 打分(host 索引 + GPU staging tile;论文同构)────────────────────────
fn score_blocks(q_tilde: &[Q], tiles: &[MeanKEntry], lse_scale: f32) -> Vec<f32>;  // R_b
// 规则:q 去 RoPE 后与 k̄ 点积 /√d;跨候选块**全局 softmax**(tile 化要维护全局归一);
//       再对 (层,头) 取均值(论文 Eq.10)

// ── 选择(纯 host;策略对象)──────────────────────────────────────────
fn select(score: &[f32], budget_blocks: usize, sink: usize, recent: usize) -> BlockSet;

// ── 装配(页表别名;零拷贝)───────────────────────────────────────────
struct ViewPlan { page_table: Vec<u32>, pos_frames: Vec<(ChainHash, u32)> }
// 位置:raw K 是**重建权威**;活跃页里的 K = 当前紧凑位置编码;
//      位置变化 ⇒ 有界 delta re-RoPE(超阈值从 raw K 重建一次)

// ── 搬运(packed + pipelined)─────────────────────────────────────────
fn gather(host: &[ChainHash]) -> PinnedBuf;             // host 侧连续槽(= 我们的 SlotTable)
fn h2d_bulk(buf) -> StagingBuf;                         // 一条大传输
fn scatter_and_rerope(staging, dst_pages, pos_frames);  // GPU 核(新写)
```

**A 轴已有的接口一件不改**:页表真源 = `PoolLayout` + 槽池;搬运 = `swap`(需扩展为
gather/scatter 形态,见 §5-B3);账 = `TierLedger`(`Index` 作新 `AssetKind`);
准入 = `Admission`(执行视图预算 W 进 `TurnFloor`/`elastic`)。

## 3. 空间预算(与 A 轴账对接)

| 项 | 公式 | 27B 量级(待实测) |
|---|---|---|
| 索引 | `块数 × 层数 × KV 头数 × d × 2B` | 262k/32 = 8192 块 × 48 层 × 8 头 × 128 × 2B ≈ **0.8 GiB** |
| host 驻留 | 索引 + warm 块(论文:10M 工作区 ⇒ 索引 9.5GiB / host 64GiB) | 索引全量进 host ✓ |
| device | 执行视图 W(≤ 模型窗)+ 生成 reserve + **staging tile**(有界) | W = 64K ⇒ 2.1GiB(fp8) |
| 盘 | 冷块(论文:10M ⇒ NVMe 324GiB) | 我们一期不做盘层(S6) |

## 4. 施工序(B-0 → B-5;每步独立可验、可回滚)

| 步 | 内容 | 门 |
|---|---|---|
| **B-0** | **索引采集**(prefill 顺手采,去 RoPE 的 Mean-K 落 host)+ 索引账 | 采集开销/体量实测 + **top-k 覆盖率 vs 全量注意力**(对齐论文 top-8≈66.5% / top-16≈77%) |
| **B-1** | **选择器**(纯 host:打分 + 全局 softmax + sink/recent + top-k) | 覆盖率 + 选择延迟(论文 174ms@256K 量级) |
| **B-2** | **执行视图装配**:页表别名 + 紧凑位置 + **re-RoPE**(raw K 真源 + 有界 delta) | **G3② 物化比 + 覆盖率 + dense 对照**(论文自认非数学等价) |
| **B-3** | **packed/pipelined 搬运**:host gather → H2D bulk → GPU scatter+re-RoPE(双缓冲 + event) | 恢复延迟(论文 174ms@256K / 1.31s@10M 量级) |
| **B-4** | **策略面**:预算在线调节(§7 ActKV 置信度)+ 下沉/保留序(§7 LRFU) | 长跑零崩 + 质量不降 + 命中/换出 metrics |
| **B-5** | **终局** | **G1:单卡 262k 可服务** + G3①(选择层关闭 ⇒ Strict 逐 token 恒等) |

## 5. 风险与对照门

| 风险 | 依据 | 我们的对策 |
|---|---|---|
| **KV 复用非数学等价**(历史 KV 在原因果上下文下算出) | 论文 §7 明说 | B-2 起**每个门都带 dense 对照**;Strict 档(选择层关)保持逐 token 恒等 |
| **检索漏块**(task-critical 块可能没被选中) | 论文 §7 | 覆盖率门(B-0/B-1)+ sink/recent 保底 + 预算可调(B-4) |
| 位置重排的数值误差 | 论文:低精度页内反复 delta re-RoPE 会累积 | **有界 delta**(超阈值从 raw K 重建)+ f16/fp8 档分别定阈值 |
| 我们与论文的**模型差异**:混合注意力(GDN 线性层 + full) | 论文只虚拟化 RoPE 注意力层 | 我们的 **GDN 快照腿**(A 轴已三层化)承担复发状态 ⇒ **完整状态虚拟化**(论文没有) |
| 我们的**栈差异**:图捕获 + 投机解码 | 论文 QW3 无 spec | 图重捕获门(页表是图输入槽 ⇒ 每步换页表要重放不重捕;A 轴 XQA 页表已走图输入槽 ✓) |

## 6. 与 ActKV 的分工(摘两件,归位不同)

| ActKV 的件 | 我们的归位 |
|---|---|
| **attention-aware LRFU** `KS_t = λ·KS_{t-1} + Attn_t·{hit}` | 接到 A 轴已有的 **`EvictPolicy` 策略对象**:决定"**谁下沉 / 谁留 host**"(我们三层是**内容保留**,强于 ActKV 的丢弃) |
| **置信度驱动预算**(token 置信 → 滑窗 min-pooling + bottom-b% → 斜率 α<0 ⇒ 扩预算) | 接到 B 轴的**执行视图预算 W 在线调节**(比固定水位更合 store 卷 §12.4 的"需求驱动"哲学) |
| recovery-based 注意力计算(用 LSE 反推) | **不需要**(我们打分用 query·Mean-K 自算,不依赖后端暴露 attention) |
| conflict-free in-place compaction(保留条目压尾部块) | 备选:优先**页表别名**(零拷贝);仅当某核要求连续块时用 |

## 7. 待拍 / 挂账

1. **位置方案**:A 轴设计卷 §3.6.3 原建议 A(原位置稀疏召回),**本卷改判 B**
   (紧凑窗口 + re-RoPE)—— 依据 §29.3(论文实证 + 页表别名零拷贝 + 与 XQA 连续页表对齐)。
2. **索引是否上盘**(S6 之后):一期 host 足够。
3. **块大小**:32 ✓ 已定(§3.6.2);若 B-0 实测覆盖率不足,再评估 16(ActKV 用 16)。
4. **与投机解码的交互**:verify 行的 KV 是否进选择视图(论文无 spec,无先例)⇒ B-3 时拍。
5. **GDN 行与选择视图的时序**:每 step 借行 + 恢复快照(A 轴 S2b 已就位)⇒ B-2 时对齐。

---

## 8. 架子落点(2026-10-17;B-0 前的基础件已落)

**已落**:`crates/store/src/select.rs`(纯逻辑核心,零 I/O、零设备面;与 store 的
「账 + 身份 + 策略」纪律一致)+ `owl_store::select` 的公开导出。

| 论文对应 | 落点 | 已验(单测 6/6) |
|---|---|---|
| Mean-K 索引(每块 × 层 × KV 头,去 RoPE) | `MeanKIndex` / `MeanKEntry`(+ `f32_to_f16`/`f16_to_f32` 自算,不引依赖) | 登记/形状校验/触达/**淘汰序 = 近期 + 累积频率**(非纯 LRU)/体量 |
| 块相关度 `R_b = Σ_m mean_{l,h} softmax(q·k̄/√d)` | `score_blocks`(逐 token 做 softmax —— 非线性不可先求和)+ `QueryFace` | **手算对拍**(2 块 × 1 层 × 1 头:0.668/0.332,归一到 1) |
| sink + recent 保底 → 按分 top-k → **时间序重排** | `select` / `SelectBudget` / `BlockSet` | 保底+补足+重排逐项断言;预算 0 = 选择层关 |
| 两视图分离 `R_t/L_t/E_t` + 有界 delta re-RoPE | `assemble_view` / `ViewPlan` / `PosFrame` / `PosAction{Keep,DeltaReRope,Rebuild,StageIn}` | 三态判定 + 紧凑跨度 + `stage_in`/`rebuild` 清单 |

**接口边界(裁决⑩)**:本模块只出**计划**;执行(页表写 / re-RoPE / gather-scatter 搬运)
仍归 engine —— 与 store 卷 §21「store 不做 I/O,engine 编排」同律。
**A 轴的接口一件未动**(`Rel`/`BlockLeg`/槽池/`Admission` 零改动)。

**下一步(按 B 轴卷 §4)**:

1. **B-0 采集接线**:模型面暴露「块边界处的 K 段(含 RoPE 前 raw K 与 RoPE 后 K)」⇒
   prefill 写完 K 处顺手求均值 + `MeanKIndex::register`;索引账挂 `TierLedger`(新 `AssetKind::Index`)。
   门 = 采集开销/体量 + **top-k 覆盖率 vs 全量注意力**(对齐 top-8≈66.5%)。
2. **B-1 打分接线**:query 侧去 RoPE 面(`QueryFace`)+ host 索引 → **GPU staged tile** 打分
   (host 全量 + 有界 staging;`score_blocks` 的 tile 化版本,全局 softmax 归一跨 tile 维护)。
3. **B-2 装配执行**:页表别名(复用 A 轴 `PoolLayout`)+ `PosFrame` → re-RoPE 核(raw K 真源 +
   有界 delta);**门带 dense 对照**(论文自认非数学等价)。
4. **B-3 搬运**:`ViewPlan.stage_in` → packed/pipelined(host gather → H2D bulk → scatter+re-RoPE)。

### 8.1 B-0 第一件已落:采集器 + 索引账(2026-10-17)

| 件 | 落点 | 门 |
|---|---|---|
| **Mean-K 采集器**(纯 host) | `MeanKCollector`(`select.rs`):`push_pre_rope`(RoPE 前 K,天然位置无关)/ `push_post_rope`(**逆 RoPE**:`x'=x·cos+y·sin; y'=−x·sin+y·cos`,半维形式)/ `finish()` → f16 条目;f64 累加(f16 只用于排序) | 单测:**逆 RoPE 往返**(打 RoPE 再逆 ⇒ 回原值,逐维 ≤1e-2)+ 均值 + 形状/越界/空块**响亮拒** |
| **索引账** | `Rel::attach_index(dim, layers, kv_heads)` / `index_register` / `index_drop` / `index_bytes`;`AssetKind::Index`(host 层;可驱逐/可下沉) | 单测:登记入账(host/Index 行)+ 形状错**不进账** + 出账归零 |

`owl-store` 全量 **71 + 2 + 2**。

**B-0 剩余(下一步;一条真设计岔路)**:**K 源**。两条路,代价不同,需拍:

| 路 | 做法 | 代价 | 适配 |
|---|---|---|---|
| **A. 回读**(不改核) | 块写完 ⇒ `dtoh_at` 读该页 K(已打 RoPE)→ `push_post_rope` | 每块 ~3MB D2H(27B:32 tok × 48 层 × 8 头 × 128 × 2B);262k 工作区累计 ~24GB 流量(一次性建索引) | **今天就能接**(用 A 轴现成的 `dtoh_at`/`swap`) |
| **B. 核内累加**(新核) | 在 `qknorm_rope_kv_insert` 后加"块边界归约"臂:核内去 RoPE + 累加,直出索引 | 新 CUDA 核 + 归约缓冲;但零额外显存流量(论文做法) | 索引采集"顺手"(论文:incremental indexing,边际成本≈0) |

⇒ 建议:**先走 A 打通门(覆盖率)**,拿到真实覆盖率后若开销不可接受再上 B(核内累加)。
**覆盖率门**(B-0 的硬门)需要"全量注意力权重"作为对照 —— 落在 engine 的 e2e 场景里
(env 门控:同一 prompt 跑 dense 路径,取注意力 → 与索引 top-k 覆盖对比)。

### 8.2 B-0 接线侦察结论(2026-10-17;三条硬事实 + 两个待定)

**硬事实(源码级,接线必须照此)**:

1. **K 池布局 = kNHD**:`models::env::knhd_offset(slot, h, d, hkv, hd) = (slot·hkv + h)·hd + d`
   —— 一个 token 的 K 是 `[hkv][hd]` **连续**;整页 = `page` 个 token 顺排。
   ⇒ 单页 K 字节 = `per_page_bytes(dtype, k_cache.1)` = `page · hkv · hd · elem_bytes`。
   ⇒ 回读一段 = 对每层 `k_cache` 一次 `dtoh_at(page_off, page_bytes)`,再按上式切 (token, head, dim)。
2. **RoPE 表是设备张量**(`RopeTables { cos, sin }` = `TensorOps`)⇒ **host 侧拿不到现成表**。
   两条路:①按 `loaded.rope` 的配置在 host **重算** cos/sin(零设备流量,推荐)②`dtoh` 表(多一次搬运)。
3. **去 RoPE 的配对约定是唯一的静默错值风险**:interleaved `(2i, 2i+1)` vs
   rotate-half `(i, i+d/2)`。采集器**两种都实现了**(`push_post_rope` /
   `push_post_rope_halves`,各有往返单测)⇒ 接线现场只需**核对写入核的约定并二选一**,
   不必凭记忆猜。

**两个待定(接线时定,均有一句话判据)**:

| # | 待定 | 判据 |
|---|---|---|
| 1 | 配对约定(interleaved / rotate-half) | 读 `qknorm_rope_kv_insert` 的旋转代码;或**位置不变性自检**:同一段内容放在两个不同起始位置跑两次,去 RoPE 后的 `k̄` 应一致(不一致 = 约定选错) |
| 2 | K 源粒度(整页一次回读 / 分块) | 单页 ~3MB(27B)⇒ 一次 `dtoh_at` 搬一页即可;若开销咬人再上核内累加(§8.1 的路 B) |

**接线落点(下一步的精确位置)**:
- 调用点 = `exec/spec.rs::capture_gdn_snap` 的**同一处**(同为"块边界 + 链尾身份"事件;
  一次算 key、一次推进游标 ⇒ 索引与 GDN 快照共享边界身份,不重复推进游标);
- 键 = `owl_store::SnapKey::at_boundary(tail)` 的 `ChainHash`(`SnapKey` 内层);
- 账 = `Rel::index_register`(已落,§8.1);
- 开关 = 新旋钮 `OWL_INDEX_PROBE`(缺省关;与 `OWL_OFFLOAD_PROBE` 同族)。

### 8.3 B-0 接线已落(回读形态;2026-10-17)—— 采集链路真机跑通

**约定定谳(读核,不再猜)**:`crates/kernels/cu/fused/qknorm_insert.cu` 的 K 头旋转 =
**rotate-half + partial 直通**:
```
d < half:      out[d]      = n[d]·cos[d] − n[d+half]·sin[d]
d ∈ [half,2h): out[d]      = n[d]·cos[d−half] + n[d−half]·sin[d−half]
d ≥ 2h(rotary_dim 之外): 直通(不旋转)
```
⇒ 采集器用 **`push_post_rope_halves`**(逆变换 `x[i]=k[i]·c+k[i+h]·s; x[i+h]=−k[i]·s+k[i+h]·c`),
并按 `rotary_dim` **partial 感知**(尾维直通,与核内同律)✓。

**另两条事实(接线依据)**:
- **布局单源 = `models::env::knhd_offset`**(P6-A 后单布局)⇒ 页内 `[token][head][dim]` 连续,
  一次 `dtoh_at` 读整页(每页 K 字节 = `page·hkv·hd·2`);
- **cos/sin 无需回读**:`Rope` 里就有 host 侧 `cos_data`/`sin_data`(设备表的同源)⇒
  新增 `Rope::cos_sin_at(pos)` / `rotary_dim()`(与设备表**同一份数据**,零重算风险)。

**落点**:
| 件 | 位置 |
|---|---|
| 旋钮 | `EngineKnobs.index_probe`(`OWL_INDEX_PROBE`,缺省关) |
| boot | `rel.attach_index(hd, n_full_layers, hkv)`(几何与池交叉校验) |
| 采集 | `exec/spec.rs::collect_index_for_block(key, full_blocks)` —— 挂在 **`capture_gdn_snap` 同一处**(同一边界事件、同一链尾身份;`contains` 早退保幂等) |
| 账 | `Rel::index_register` → `AssetKind::Index`(host 层) |

**门(已跑)**:
| 档 | 结果 |
|---|---|
| CPU mock(`OWL_CPU_MOCK=1 OWL_INDEX_PROBE=1`) | 链路通:3 块 / 18432B = 3 × 6144B · host 层账与 **Index 行**一致 ✓(K 恒零 ⇒ 数值无意义) |
| **GPU 真机(3080 ordinal 0)** | 同上 ✓ **3.75s**(0.8B 装载 + 一轮 + 采集)· 无崩无 Xid |

**0.8B 的索引体量**:每块 **6144B** = 6 全注意力层 × 2 KV 头 × 128 dim × 2B(f16)。
⇒ 外推 27B(48 层 × 8 头 × 128 × 2B)= **98KB/块**;262k 工作区(8192 块)≈ **0.8GiB**
(与 §3 的预算表一致 ✓);**回读开销**同量级:每块 ~3MB D2H ⇒ 8192 块 ≈ **24GB 一次性**
(这正是 §8.1 里"路 B 核内累加"的动机)。

**剩余(B-0 收口的两件)**:
1. **位置不变性自检**(配对约定的**实证**,不是推理):同一段内容放在两个不同起始位置
   跑两次 ⇒ 去 RoPE 后的 `k̄` 应一致。键是**内容身份**(位置无关)⇒ 可直接比对两次的条目。
   建议作为 `gpu_08b_index_probe` 的第二段(需两轮 + 一次比较;CPU mock 的 K 恒零验不了)。
2. **覆盖率门**(B-0 的硬门):dense 路径取全量注意力 ⇒ 与索引 top-k 覆盖对比
   (对齐论文 top-8≈66.5%);需要 attention 权重的读出口,落在 engine e2e(env 门控)。

### 8.4 B-0 收口(2026-10-17)

**量具已落**(`select.rs`,纯 host;单测 9/9):

| 件 | 语义 | 门 |
|---|---|---|
| `block_mass(logits, block_of_token, n_blocks)` | 一个 query 对全部历史 token 的 logits ⇒ **逐块注意力质量**(softmax 归一,Σ=1;全 -inf/空 = 零,不 panic) | 手算归一 + 单调性 |
| `coverage_curve(attn_mass, scores, ks)` | 按索引打分排序 ⇒ **top-k 覆盖率曲线**(论文 G5③ 的口径) | 完美排序 = 前缀和;top-4 = 1.0;**反序 ⇒ 只捞到最差块**(排序错一眼可辨) |

**一条合并裁决(把两件并成一件)**:原计划的"位置不变性自检"与"覆盖率门"**本质是同一个测量**
—— 去 RoPE 的**配对约定选错** ⇒ 索引打分退化为噪声 ⇒ 覆盖率必然掉到 ≈ `k/块数` 的随机水平。
⇒ **不需要**另做"同内容两位置"的对照实验(那个实验还有致命缺陷:前插 token 会改变
隐藏态 ⇒ K 本身变了,不可比);**覆盖率门就是约定的实证**。

**B-0 剩余的唯一一件(属于 B-1 的前置)**:给 coverage 供料的 **attention 读出口** ——
`block_mass` 需要"一个 query 对全部历史 token 的 logits(或等价的注意力质量)"。
两条取法:
| 路 | 做法 | 代价 |
|---|---|---|
| **A. host 重算**(推荐) | 回读该 query 的 Q(一次)+ 历史 K(采集时已读)+ 逐块点积 ⇒ logits ⇒ `block_mass` | 一次 D2H(Q)+ host 点积(与采集同量级);**零新核** |
| B. 核内 tap | attention 核加输出臂(每 query 的逐 token 概率) | 新核 + 输出槽;但可复用现成 slab |

⇒ 按 A 走:`coverage` 的供料 = "回读一个 query 的 Q + 已有 K 回读" ⇒ 与 B-1 的
"打分器"共用同一条回读通道。**B-0 的架子/采集/量具到此收口**;下一步进 **B-1 打分器**
(host 索引 + GPU staged tile 打分 ⇒ 接 `coverage_curve` 出第一条覆盖率曲线)。

### 8.5 B-1 第一步的阻塞点定谳:Q 没有稳定出口(2026-10-17)

**实测侦察(源码级)**:覆盖率门的供料要"某个 query 的 Q"。

| 事实 | 依据 |
|---|---|
| 融合核**确实**产出 Q(`q_out` 参数;norm+rope 后的 Q 直接落该缓冲) | `crates/kernels/cu/fused/qknorm_insert.cu`(q 头段写 `q_out`) |
| 但 **`q_out` 不在模型面暴露**:`models::module` 的 attention 层把它当**内部 scratch**(chunk 结束即随 arena 归池) | `grep q_out crates/models/src/module.rs` = 空 |
| 引擎现有的读出口只有三类:GDN 画像(`probe_gdn_dump`)/ KV 页 dump(`dump_kv_page`)/ **DFlash2 taps**(`model::tapped_hidden` 返回 hidden + 每层 tap) | `exec/phases.rs:414/520`、`models/src/model.rs:275` |

⇒ **下一步 = 给 attention 层加一个 Q 输出槽**,形态**照抄 DFlash2 taps**(已有先例、已有图输出槽机制):

```text
attention 层:q_out 从"内部 scratch"升级为"可选输出槽"(声明期挂,与 taps 同族)
  ├ 声明:每层一个 [t, hq, hd] 输出(仅覆盖率探针档开 ⇒ 生产零改动)
  ├ 图:输出槽进图 slab(与 taps 同律 —— 图捕获后仍是稳定句柄)
  └ 读:探针 `dtoh` 该槽 → host 点积(K 已有回读通道)→ block_mass → coverage_curve
```

**为什么不用"host 重算 Q"**:Q = qk-norm(×w) + rope 的产物,重算要复刻 norm 权重与
rope 表(第二套实现 = 静默错值温床);而 taps 路线**零重算、零新核**(只多一个输出槽)。

**B-0 收口清单(全绿)**:约定定谳(rotate-half + partial)· 采集链路(GPU 3.75s 跑通)·
体量实测(0.8B 6144B/块 ⇒ 27B 98KB/块)· 量具(`block_mass`/`coverage_curve`)·
**唯一开口 = Q 输出槽**(属 B-1 第一步,形态已定谳)。

### 8.6 Q 输出槽:施工单(下一刀的精确工单;2026-10-17 侦察定稿)

> 用户已批"开工";本轮**只做侦察定稿**(发现这不是"加个字段"而是**声明面改动**,
> 故按纪律先出工单,不在预算末段动模型面)。

**侦察补充(比 §8.5 更进一步)**:

| 事实 | 依据 |
|---|---|
| DFlash2 taps 的本质 = **host 侧 TensorOps 句柄的克隆**(`xs.clone().tag(...)`)⇒ 它要求被 tap 的东西**本身就是声明树里的一个节点** | `model.rs::tapped_hidden`:逐层 `layer.forward(&xs,&sub)` 后按 `tap_ids` 克隆 `xs` |
| 而 attention 层的 Q **不是节点**:融合核把 Q 写进内部 scratch(`q_out`),attention 直接消费 —— 声明树里没有"Q"这个值 | `layers/attention.rs`(融合分派臂;B 臂 = `owl_qknorm_rope_kv_insert`) |
| ⇒ 因此 **不能"照抄 taps"**:taps 是"已有节点加个 tag",Q 槽是"**造一个新输出节点**"(要分配缓冲、改核的写目标、进图 slab、加读出口) | 同上 |

**工单(四步;每步独立可验)**:

1. **声明面**:`AttentionLayer::forward` 的融合臂里,把 `q_out` 从内部 scratch 升级为
   **可选声明输出**(`ctx.tap_q: bool` 或 `ForwardCtx` 上一位;缺省关 ⇒ 生产路径逐字不变)。
   产出 = 一个 `[t, hq, hd]` 的 `TensorOps` 节点(标 `.tag("q")`)。
2. **图/slab**:该节点进图输出槽(与 taps 同律:捕获后句柄稳定);
   开档时多一块 slab(`t×hq×hd×2B`;t = chunk,0.8B 档 ~64KB 量级)。
3. **读出口**:engine 探针(`OWL_COVERAGE_PROBE`)在段边界 `dtoh` 该槽 + 用 §8.3 的
   K 回读通道 ⇒ host 点积(`q·k`,**post-RoPE 对 post-RoPE** = 真实 logits)⇒
   `block_mass` ⇒ `coverage_curve` ⇒ **第一条覆盖率曲线**(对齐 top-8≈66.5%)。
4. **索引侧对齐**:同一 query 的**去 RoPE** 面(`q̃`)喂 `score_blocks` ⇒ 两条曲线的
   对照 = "索引能否复现真实注意力"的**唯一判据**(同时是去 RoPE 配对约定的实证,§8.4)。

**验收门**:覆盖率曲线落在论文量级(top-8 ≈ 0.66 ± 0.1)⇒ B-1 打分器可开工;
若 ≈ k/块数(随机水平)⇒ 先查 §8.2 的配对约定与 §8.3 的布局单源(两处都已有判据)。

**风险提示(给下一位)**:第 1 步动的是**声明面**(模型层),生产路径必须**逐字不变**
(缺省关档);改完先跑 `owl-models` 单测 + `cargo test -p owl-engine --lib` +
0.8B 共享装载组(§四.1)确认零回归,再开探针档。

### 8.7 Q 槽工单**修正**(用户口径 + 源码复核;2026-10-17)

> 用户口径:*"KVMem 开 ⇒ 就把它选择开这个点;没开 ⇒ 不需要,继续用融合核。"*
> 源码复核后:这条口径**比 §8.6 的工单更省** —— 因为 **Q 早就是声明节点**。

**修正依据(源码级,一句话)**:

```rust
// crates/models/src/layers/attention.rs:185
let use_fused_insert = dt_raw == F16 && !force_naive && ctx.env.attn.qkv_fuse && page_ok;
if use_fused_insert {
    let q = TensorOps::call(insert_op).arg(&q_raw)…      // ← **Q 就是这条调用的返回值**
    …
}
```

⇒ 融合核**已经**把 `q_out` 作为声明树的输出节点交回(`let q = …`)。
⇒ §8.6 的"造新输出节点(分配缓冲 + 改核写目标)"**全部不需要**;真正缺的只有一件事:
**把这个已存在的节点挂出去**(tap),让 engine 能在执行后读到它。

**修正后的工单(两步,比 §8.6 少两步)**:

| 步 | 内容 | 关键约束 |
|---|---|---|
| **Q-1** | `ForwardCtx` 加**可选 Q 收集口**(探针档才非空;缺省 `None`);融合臂把 `q` 节点存进去(每全注意力层一份)。生产档 `qkv_fuse = true` 且收集口为 `None` ⇒ **路径逐字不变** | **不切臂** —— KVMem 开也照用融合核(同一条数学、同一个节点);只是多挂一个引用 |
| **Q-2** | 引擎侧:探针档的入口(与 `tapped_hidden` 同族)把收集到的 Q 节点**并进执行根**(或作为附加输出),执行后 `dtoh` 该槽 ⇒ host 点积(post-RoPE × post-RoPE = 真实 logits)⇒ `block_mass` ⇒ `coverage_curve` | 图槽/捕获后句柄稳定(与 taps 同律);K 用 §8.3 已有回读通道 |

**与用户口径的对应**:

| KVMem | `qkv_fuse` | Q 收集口 | 生产路径 |
|---|---|---|---|
| 关(缺省) | true | `None` | **逐字不变**(融合核 + 无额外节点) |
| 开 | true(**仍融合**) | `Some` | 融合核照跑 + Q 节点被挂出(多一份引用,零新核) |

⇒ 结论:**不需要"KVMem 开就走非融合臂"** —— 融合臂本身就能给出 Q;
"非融合臂"只在 `qkv_fuse = false`(诊断档)时出现,与 KVMem 无关。

**剩余不确定(给下一位)**:`ForwardCtx` 能否挂"收集口"取决于它的类型约束
(`Copy`?是否允许 `Rc/RefCell`?)⇒ 落地前先读 `module.rs::ForwardCtx` 的字段面;
若不允许,退路 = 在 `Model` 上加一个 `q_sink: RefCell<Vec<Option<TensorOps>>>`,
由层在 `forward` 里写(层已持有 `&self`,写法同 `taps` 的"调用方收集")。

### 8.8 Q-1 落地前侦察(类型面已核实;2026-10-17)

**核实结果(两点,都在源码里)**:

1. **`ForwardCtx<'a>` 是"引用袋"结构**(`tokens/kind/pos/kv/rope/gdn/kvs/gdns/kv_slots/kv_lens/gdn_slot/ctx_base/fi/fi_kvi/seq_cu/gdn_tap…`),
   逐层派生经 **`Model::layer_ctx(ctx, kvi, gi) -> ForwardCtx<'a>`**(`model.rs:147`)——
   即"根 ctx → 单层子 ctx"的**唯一派生点**。
2. **全 `crates/models/src/` 里 `ForwardCtx { … }` 字面量只有 1 处**(就是 `layer_ctx`)
   ⇒ 加一个字段的**改动面 = 一处字面量 + 根构造处**;子 ctx 派生天然继承(照抄 `gdn_tap` 的写法:
   引用型字段 `Option<&'a …>` 或 `clone()` 过的共享句柄)。

**Q-1 的字段形状(照 `gdn_tap` 先例)**:

```rust
// module.rs::ForwardCtx 新增(缺省 None ⇒ 生产路径逐字不变)
/// **Q 收集口**(B 轴覆盖率探针;`Some` = 逐全注意力层把 Q 节点写进去)
pub q_sink: Option<&'a std::cell::RefCell<Vec<Option<TensorOps>>>>,
```

- 融合臂(`attention.rs:185` 的 `use_fused_insert` 分支)在 `let q = TensorOps::call(insert_op)…`
  **之后**加一句"若 `ctx.q_sink` 非空 ⇒ 按 `fi_kvi` 序号写入" —— **零新核、零额外节点**;
- 非融合臂(诊断档)同法(它的 Q 也是节点);
- `layer_ctx` 里 `q_sink: ctx.q_sink`(引用直接透传,与 `gdn_tap` 同型);
- **生产档** = `q_sink: None` ⇒ 编译器优化掉整句,行为逐字不变。

**Q-2(读出口)**:探针档在引擎侧构造 `RefCell<Vec<Option<TensorOps>>>` 作为根 ctx 的
`q_sink`;执行完 chunk 后,把 sink 里的 Q 节点**并进执行根**(或作为附加输出)⇒ `dtoh` ⇒
host 点积 ⇒ `block_mass` ⇒ `coverage_curve`。

### 8.9 Q-1 / Q-2 落地 + **覆盖率门全绿**(2026-10-18)

> **一句话**:Q 槽接上 ⇒ 覆盖率曲线出来了,**top-8 = 87.5%(上界 88.7%,随机 1.0%)**
> —— 索引打分与真实注意力**同序**(top-8 重合 7/8)。B-1 打分器开工。

**Q-1(模型面,Q 输出槽;§8.7 修正案的落地)**

| 件 | 落点 |
|---|---|
| 收集口 | `ForwardCtx.q_sink: Option<&'a RefCell<Vec<Option<TensorOps>>>>`(`module.rs`;7 处构造字面量 `None` + `with_q_sink()` 构造器;`layer_ctx` 引用透传) |
| 写入 | `Attention::tap_q(ctx, &q)`(三处臂:**decode 融合臂**(`let q = TensorOps::call(insert_op)…` 之后)、**decode 非融合/f32 臂**、**prefill f16/f32 臂**,`(q, gate, k)` 之后);槽按 **`ctx.fi_kvi`** 落位(不是 push 序) |
| 不成新物 | Q 本就是声明节点(融合核的返回值 / NormRope 的返回)⇒ **零新核、零额外节点、不改写目标**;`q_sink = None`(生产缺省)⇒ 一处 `is_some()` 判即返 |
| 门(CPU 声明级) | `attention::tests::q_sink_taps_node_per_layer_slot`(槽位/形状/dtype/哨兵格不动/decode 同律) |

**Q-2(引擎面,探针三段式;`crates/engine/src/exec/coverage.rs`)**

```text
① eval_root_and_q   主根 + Q 节点**同一遍** `eval_ops_multi_scoped_env`
                    (前向带 GDN 状态副作用 ⇒ 绝不双执行;与 pf_bisect 同律)
② coverage_take_q   块边界 chunk 末 Q 面 dtoh → host f32([层][t·hq·hd]);设备块当场 free
③ coverage_analyze  边界事件**之后**(索引/快照已落)读池 K(会话真页链)+ 索引账 ⇒ 曲线
```

- **旋钮**:`OWL_COVERAGE_PROBE=1`(**与 `OWL_INDEX_PROBE=1` 同开** —— 打分要有索引);
  诊断档 ⇒ prefill 桶图**让位 eager**(与 `pf_bisect` 同律,响亮告知一次);
- **query 面** = 本 chunk **末 W=8 行**(论文 step 级 query;单 token 注意力太局部 —
  末 token 常只盯前几个字)⇒ 逐行去 RoPE + 逐行真实质量,再取均(与 Eq.10 的 `Σ_m` 同构);
- **真实质量** = post-RoPE `q` × 池内 post-RoPE `K` 逐 (层,查询头) 因果 softmax
  → `block_mass` → (层,头) 均;GQA 映射 = 核内同律 `g = h/(hq/hkv)`;
- **索引面** = 同一批 query 的**去 RoPE** `q̃` → `score_blocks`(Eq.10)→ `coverage_curve`;
- **逆 RoPE 单源**:新增 `owl_store::derope_pair` / `derope_halves_inplace`
  (`MeanKCollector::push_post_rope_halves` 改为调它 —— 约定只有一份实现)。

**B-0 采集口两处修正(Q-2 顺带暴露的实证缺陷)**

| 缺陷 | 现象 | 修法 |
|---|---|---|
| **只登记最后一块** | 块边界事件按 **chunk 末**触发(chunk 512 ⇒ 每次事件跨 16 块)⇒ 只有每 16 块中的最后一块进索引;**实测 112 块只有 7 块在场**,打分全 0 ⇒ 曲线被打崩 | `capture_gdn_snap` 内**逐块**登记 `start..full_blocks`(身份用 `hash_block` 从事件前的链尾重放一小段;`debug_assert` 重放尾 ≡ 游标尾 = 身份单源硬断) |
| **逻辑块号当物理页号** | `dtoh_at(k_cache, block·page…)` —— 单会话且顺序分配才成立;多会话共享池 = 读到别家 K | 采集口改收 `(block_hash, logical_block, phys_page)`;物理页由**会话块表** `bt[logical]` 定,位置仍用逻辑块号 |

**实测(3080 单卡,0.8B,ctx 3072 tok / 96 块;chunk 1024;prompt = 6 针埋 520 条随机底噪 + 尾部点名 query)**

```text
覆盖率:  top1=19.9% top2=45.5% top4=78.7% top8=87.5% top16=92.5% top32=94.8%
上界(按真实质量自排):top1=26.6% top2=52.2% top4=78.7% top8=88.7% top16=92.5% top32=95.0%
top-8 重合 7/8;随机基线 1.0%
逐块:针块 0=0.199 / 17=0.256 / 36=0.042 / 56=0.066 / 76=0.022 / 95=0.266(索引分 0.132/0.129/0.107/0.113/0.101/0.114 = **全场前六**)
```

⇒ 三件一起被钉住:**①Q 槽取到的确实是 post-RoPE Q(槽位/层序对)②去 RoPE 配对约定对
(约定错 ⇒ 打分退化噪声 ⇒ 贴随机线)③K 回读布局对(否则 Mean-K 与 q 不同空间)**。
块 0 的 19.9%(注意力 sink)与论文一致,额外旁证。

**门定义(写进 e2e,`gpu_08b_coverage_probe`)**:①`indexed == blocks`(索引无缝)
②`top8 ≥ 0.6 × 上界`(选择器质量)③`top8 > 2×随机`(约定)④`上界 > 0.35`(防"空过":
数据太平坦时上界本身也低,前三门就失去意义)。

**值得记下的两条读数(给 B-1 打分器)**

1. **softmax 后取均值的动态范围压缩**:`score_blocks` 逐 (层,头) 先 softmax 再均 ⇒
   候选块多时每个分量 ≈ `1/块数`(**96 块实测分值全在 0.073~0.132 之间**,"平局"成片)。
   **排序**(取 top-k)不受影响(softmax 单调 ⇒ 序 = logit 序),但**任何按分值取阈值/
   按分值做预算的用法都会失效**(B-4 的在线预算调节需改用 logit 面或做 per-head 归一)。
2. **上界必须作为门的另一半**:覆盖率绝对值被数据平坦度支配(第一版随机词池 prompt
   的注意力质量极平,归档 24.1%@top8 **连上界也只有 31.5%**)。⇒ "选择器差"与
   "数据平"必须分开判 —— 上界曲线进报告(`CoverageReport::oracle`)就是这个分离器。

### 8.10 B-1 选择器落地(打分口径 / 性能 / 召回点读数;2026-10-18)

> **一句话**:打分与选择都通了 —— **真实 27B/262k 规模 host 打分 165ms**(基线构建 323ms,
> 论文 174ms@256K 同量级),召回点真实选择覆盖 **97.9%**(同预算上界 98.1%)。

**(1)打分之辩定谳(`ScoreMode`)**

| 口径 | 语义 | 实测(96 块) |
|---|---|---|
| `SoftmaxMean`(**论文 Eq.10 原样**) | 逐 (层,头) 先对候选块 softmax 再取均 | 分值 0.073~0.132(**≈1/块数**,平局成片) |
| `LogitMean`(**排序/阈值用 ≈2 块**) | 原始 logit 直接取均(softmax 单调 ⇒ **同序**) | 0.5~2.85(自然展开) |

⇒ 定谳:**排序两口径等价**(实测同样选中 64 块、覆盖同为 97.9%),但**凡按分值取阈值/
做预算的用法必须用 `LogitMean`**(B-4 在线预算、报表)。`SoftmaxMean` 保留为保真对照。

**(2)两层线性折叠(让打分降 48×,且不是近似)**

```text
mean_{l,h} softmax≈(q_{l,h}·k̄_{l,g}/√d)   ← 打分主体是「窗 W × 头 hq × 层 × KV 头」次行扫描
 ① query 窗折叠:Σ_m q_m·k̄ = (Σ_m q_m)·k̄                (logit 对 q 线性 ⇒ 精确)
 ② GQA 头折叠:mean_{h∈G(g)} q_h·k̄ = (mean_{h∈G(g)} q_h)·k̄  (对头线性 ⇒ 精确)
 ⇒ 单次「层 × KV 头」行扫描(本档 8×6 = 48× 更少;仅 FP 重排)
```
落点:`QueryFace::collapsed_sum()` + `score_blocks` 的 `LogitMean` 热路(按 KV 头分组、
`dot8` 8 路展开、`TILE_ROWS=256` 分块解码)。

**(3)索引存储改扁平(消除指针追逐)**

原实现每 (层, KV 头) 一条 `MeanKEntry`,各自一个 `Vec<u16>` ⇒ 打分时每行一次堆指针追逐;
27B/262k(8192 块)实测 **52 万次分散读**,把打分推到超线性(2048 块 97ms → 8192 块 550ms)。
改为**每块一段扁平 `[layer][kv_head][dim]`** + `HashMap`(`entry_at` 返回段内定长切片)⇒
查找从 52 万降到 8K,且是将来 GPU staged tile 上传的天然单位。

**(4)延迟门(纯 host;`crates/store` release bench)**

真实 27B 几何(**16 全注意力层** / hq24 / hkv4 / hd256;`models/cyankiwi/Qwen3.8-27B-AWQ-INT4`),
query 窗 8,候选块 = 全 ctx 块:

| 块数(≈ctx) | 论文口径(softmax 均值,逐 token) | **快路(折叠 + logit 面)** |
|---|---|---|
| 512(16k) | 816 ms | **19 ms** |
| 2048(64k) | 3.35 s | **78 ms** |
| 8192(262k) | 11.7 s | **323 ms**(`-C target-cpu=native` ⇒ **164.9 ms**) |

⇒ 门(strict ≤400ms 两构建都过;优化史 7745 → 1183 → 473 → 323ms = 24×);**结论:
host 打分够用,设计卷 §2 的「GPU staged bounded tile」可降级为备选**(W 或多会话并发
上来时再启用)。

**(5)召回点真实读数(0.8B / ctx 3072 tok = 96 块 / 预算 64 = sink2 + recent4)**

```text
选中 64 块 · 挤掉 32 · 保底占位 6
覆盖 97.9% | 同预算上界 98.1% | 论文口径 97.9% | 打分 5.7ms + 选择 0.1ms(debug 构建)
```

门(`gpu_08b_coverage_probe` 新增四条):预算/满额/保底计数/被挤掉数、快路与论文口径
**都 ≥ 同预算上界的 85%**、单次打分延迟上限。**B-1 完成**。

**(6)挂账**
① `-C target-cpu=native` 是否全局开(165 vs 323ms;单机项目可考虑 `.cargo/config.toml`);
② 索引上盘(S6);③ fp8 池 K 回读解码;④ 选择→**物化**仍缺(下一刀 = B-2 视图装配)。

### 8.11 B-2 执行视图装配落地(**位置方案 A**;2026-10-18)

> **决策(本轮试一个)**:先做 **A = 原位置稀疏**(K 一字不改,只换页表),
> **零新核、零数值漂移**;方案 B(紧凑窗口 + re-RoPE)留 B-3 当**对照实验**
> —— 两条路跑同一套门,用数字定。

**(1)机制硬门(models,`gpu_paged_view_aliases_selected_pages`;GPU)**

同一批 `(xs,pos)` 写**两处**(内容区 A 页 3/5/7 与副本区 B 页 0/1/2),然后:

| 对照 | 断言 | 实测 |
|---|---|---|
| 视图表 `[3,5,7]` vs 搬移表 `[0,1,2]` | 逐位相同 | **A−B = 0.000000** ✓ |
| **改写副本区**(未选中页 0/1/2)后再跑视图 | 视图输出**逐位不变** | **A1−A2 = 0.000000** ✓ |
| 同一改写,直读那批页的表 | 必须变(鉴别力) | B1−B2 = 0.304932 ✓ |
| 视图表 vs 恒等表(改写后) | 必可辨(防"忘了传视图") | A−D = 0.304932 ✓ |

⇒ 两件事被钉住:**①页表别名 ≡ 物理搬移(逐位)②视图真的只读页表列出的页**。
⚠️ 记一笔弯路:原先设计的对照组是"页序置换必须改变输出"—— **单 query 的注意力对页序
本来就不敏感**(`Σp_iV_i` 置换不变),那条断言是错的(已改写成上面的"未选中页"对照)。

**(2)视图面落引擎**

- `StatePool` 新增 **视图页表槽** `bt_view`(`PoolPlan::select_view`;与 `bt` 并存:
  全量链给 dense/未选中块,视图给选择层),`write_select_view`(超容量 = 响亮拒)、
  `select_view_leaf()`(喂 `KvBuffers::block_tables`);
- 召回点(`coverage_analyze`)在选块后:选中块(时间序)⇒ 会话真链的**物理页** ⇒ 写视图表;
- 读数:`RecallReport { view_pages, view_tokens, materialize_ratio, pos_scheme }`。

**(3)真实 run 读数(0.8B / ctx 3072 tok / 96 块 / 预算 64)**

```text
视图:64 页 / 2048 tok · **物化比 66.7%** · 位置方案 A(原位置稀疏)
(选择覆盖 97.9% / 同预算上界 98.1%;打分 6.0ms + 选择 0.1ms)
```

262k 时同预算 ⇒ 物化比 ≈ 2048/262144 = **0.78%**(G3② 的目标形态)。

**(4)门(B-2 部分)**:机制门(models,上面四条)· 视图页数 = 选中块数 · 视图 token =
选中 × 页宽 · 物化比 < 0.75(3072 tok 档)· 视图表容量不溢出(响亮拒路径)。
**"输出级 dense 对照"顺延到 B-3**:同一状态跑两遍注意力必须先在 decode 路径上把视图
接上(否则只能重放整模 = 带 GDN 副作用),那是 B-3 的接入动作。

### 8.12 B-3 施工单(搬运 + 紧凑位置;2026-10-18 侦察定稿)

> **关键发现(省一枚新核)**:re-RoPE **不需要写 CUDA 核** —— 现有原语够:
> `copy_block_at/copy_batch`(设备内块间按字节偏移拷贝)+ `SemanticKernel::Rope`
> (按 `pos` 表逐 token 旋转,单源约定已经在用)。

**为什么够**:把选中页拷进**连续的 staging 页**(页内已是 token-major ⇒ 拼起来就是
`[W tokens][hkv][hd]` 连续),再对 staging 调一次 `Rope::forward_k(staging, pos_compact, W, hkv)`
⇒ **紧凑位置 + 精确重建**(不需要 delta 近似,也不需要 host 端 raw K 备份)。V 不用旋。

**步骤(每步可独立验)**

| 步 | 内容 | 门 |
|---|---|---|
| B-3a | **staging 池**:`W_max` 页 × 层 × (K+V),持久块(boot 分配;`W_max` = `OWL_SELECT_BLOCKS`) | 显存账进 `store_report`;`W_max=2048 tok/16 层` K+V ≈ 134MB(27B) |
| B-3b | **gather**:`copy_batch` 逐 (层, 选中页) → staging 连续页 | 内容对拍(逐页 dtoh 比对) |
| B-3c | **紧凑位置 + re-RoPE**:`pos_compact[i] = i`(按视图序),对 staging K 逐层跑 `Rope` | **与方案 A 的覆盖/输出对照**(同一 query:紧凑 vs 原位置) |
| B-3d | **接入**:decode 的 `kvs_step.block_tables = staging 视图页表`、`kv_lens = W`、query `pos = W`(紧凑语义) | **输出级 dense 对照**(logits top-1/KL)+ **G3①**(选择层关 ⇒ 逐 token 恒等) |
| B-3e | **流水**:gather+H2D 与 prefill 重叠(双缓冲 + event);host 未驻留块走 `upload_span` | 恢复延迟(论文 174ms@256K / 1.31s@10M 量级) |

**待拍(此刀内)**:
1. 方案 A vs B 的最终定谳 —— 由 B-3c/B-3d 的**同一门**数字定(覆盖 + 输出对照 + 延迟);
2. 与投机解码 verify 行的交互(B-3d 时拍:verify 行是否也走视图);
3. staging 池与 KV 池的显存分配顺序(A5 预算账)。


### 8.13 B-4 / B-5 骨架(依赖 B-3)

- **B-4**:①预算在线调节(ActKV:token 置信 → 滑窗 min-pooling → bottom-b% → 斜率 α<0 ⇒ 扩预算),
  落点 = 召回点的 `budget` 由 `RecallReport`+采样面统计驱动(现在是固定 `OWL_SELECT_BLOCKS`);
  ②下沉/保留序 = **attention-aware LRFU** 接到 `owl_store::EvictPolicy`(策略对象已有口),
  决定"谁留 host / 谁下沉"(我们只下沉不丢);
  ③门:长跑零崩 + 质量不降 + 命中/换出 metrics。
- **B-5**:G1(单卡 262k 可服务)+ **G3①(选择层关 ⇒ Strict 逐 token 恒等)**。
  G3① 现在就能证(选择层尚未进生产 decode 路径 ⇒ 关档等价),B-3d 之后需重证。

### 8.14 **方案 A vs B 判谳**(2026-10-18;B-3c-pre:用 host 侧同口径对照,免建整套 staging)

> **做法(为什么先做这个)**:B-3c 的原始门是"A/B 与 dense 的输出对照",而拿这个数字
> **不必**先建 staging 池 + gather + 设备侧 re-RoPE —— 用探针已有的 K 回读 + 一次 V
> 回读,在 host 侧按**同一份数学**(同 q、同 softmax、同 V 加权)算三种方案的注意力
> 输出即可:同口径可比、几分钟出结果;若 B 不占优,那套设备面就**不必建**。
> re-RoPE 走同一原语(`owl_store::derope_pair`;正向 = 传 `-sin`),相对角取样
> `cos_sin_at(|Δ|)`(`θ(a)−θ(b) = θ(a−b)`)⇒ 无第二套约定。

**(1)两档实测(0.8B;预算 64 块;与 dense 的注意力输出比)**

| 档 | ctx / 候选块 | A(原位置稀疏) | B(紧凑 + re-RoPE) | 集中度 dense/A/B |
|---|---|---|---|---|
| 短 | 3072 tok / 96 块 | **cos 0.9999 · relL2 1.3%** | cos 0.9984 · relL2 4.2% | 0.255 / 0.259 / 0.266 |
| 长 | 14336 tok / **448 块** | cos 0.9937 · relL2 9.5% | cos 0.9938 · relL2 9.7% | 0.066 / **0.097** / 0.079 |

**(2)判谳**:**A 与 B 在输出贴近度上等价** —— 两档各胜一次,差距 ≤ 噪声级
(长档 Δcos = 1e-4)。而 B 要多付三笔:staging 池(27B 档 ≈ 134MB 持久显存)+ 每召回点
一遍 K 搬运与旋转 + 复杂度(§8.12 五步 vs A 的零步)。
⇒ **选 A(原位置稀疏)**;B 的施工单保留为"若将来长档/真机发现质量不够"的备选。
旁证:A 的"稀疏相对距离"并未把注意力搞乱(短档 A 的集中度几乎等于 dense);
B 在两档都**更锐化**(紧凑化把老块搬到近处 ⇒ 相对距离变小 ⇒ 峰值变高)。

⚠️ **本判谳的边界(诚实记)**:①指标是**注意力输出**的 cos/relL2,不是端任务质量
(logits/困惑度);②只跑了 0.8B 与两个 ctx 档;③每档单 prompt 单 query。
⇒ 真机 27B 长档复测仍是 B-5 的一部分;若那时 A 翻车,B 的施工单随时可提。

### 8.15 规模档(14k ctx / 448 块)读数与**门口径修正**(2026-10-18)

**读数**(`OWL_COV_POOL=16384 OWL_COV_CHUNK=2048 OWL_COV_ITEMS=1800`,0.8B):

```text
覆盖率:top1=1.0% top2=2.7% top4=8.0% top8=11.7% …… 上界 top8=21.9%(随机基线 0.2%)
选择器:预算 64 ⇒ 覆盖 54.9%(**同预算上界 56.5% = 97%**)· 打分 24.8ms + 选择 0.7ms(debug)
索引:448/448 块无缝;物化比 = 2048/14336 = 14.3%
```

**两条门口径修正**(都是被这一档实测逼出来的)

1. **主门 = 预算口径**:`选择覆盖 ≥ 0.9 × 同预算上界`(生产预算是 64 块 ⇒ 这才是
   有意义的判据;实测 97% 过得很稳)。**top-8 降为参考门**(≥ 0.4 × 上界 + > 2× 随机):
   448 块时上界 top-8 本身只有 21.9%(6 根针被稀释),拿它当主门会**误读**成"选择器不能用"。
2. **"数据有峰"门改相对口径**(≥ 5× 随机基线):绝对阈值随块数漂(96 块时上界 88.7% →
   448 块时 21.9%)—— 针的**占比**降了,不是数据变坏。实测 10.6× / 12.2× 两档都过。

**新挂账(B-1 补强,不阻塞)**:小预算(top-8 级)的排序区分度不足(只到上界的 53%)。
候选改法:①对 (层,头) 用 **z-score 归一**后再聚(压掉头的量纲差)②max 聚合替 mean
③两段式(先粗筛 1/8 候选再精排)。⚠️ 但**按预算口径已够用**,故不进关键路径。

### 8.16 B-3d 落地:视图接进 decode + **输出级 dense 对照**(2026-10-18)

> **一句话**:选择层第一次真正驱动生产形态的 decode —— **物化比 50% 时输出与 dense
> 逐字一致**;物化比 25% 时措辞变但**答案仍切题**;G3①(探针全关)逐字恒等 ✓。

**(1)接线形态(方案 A,零新核)**

```text
召回点(coverage_analyze)  ⇒ SelectView{ pages(选中页,时间序), tokens, recall_pos }
decode 每步(execute_decode) ⇒ select_view_step(sid, pos) ⇒ 图输入槽
        sel_pt  = 选中页 ++ 尾部页(recall_pos/page ..= pos/page;会话真链,天然连续)
        sel_len = 选中 token + (pos+1 − recall_pos)
        (无视图态 ⇒ 写全量链 + pos+1 = 与不开视图**逐字等价**,单一路径不切图)
```

- 图侧:decode 图新增 `sel_pt`/`sel_len` **输入槽**(缺省关 ⇒ 图拓扑与历史逐字不变);
  页表编码 = **数值 f32**(与 `write_bt` 同源 —— XQA 的位型编码是另一套,别混);
- 尾部**必须带**(否则模型看不见自己刚写的 token);召回点落在块边界 ⇒ 尾部从整页起;
- 视图态是 **turn 域资源**(turn 收口清空;不清 = 拿旧视图读新会话的块);
- ⚠️ 只在 `select_decode` 档落视图态 —— 否则 dense 对照臂也会用上视图(踩过);
- 前置:XQA 裸 decode 臂与视图**不共戴**(页表语义两套)。

**(2)三臂对照(0.8B;prompt ~1.0k tok;chunk 512;贪心 24 token)**

| 臂 | 视图步 | 召回视图 | 质量覆盖 | 物化比 | 输出 |
|---|---|---|---|---|---|
| **dense**(探针开、选择关) | 0 | 16 页 | 81.0% | 50.0% | 基准(42 字) |
| **view**(选择接 decode) | 23 | 16 页 | 81.0% | 50.0% | **逐字一致(42/42)** |
| **plain**(探针全关) | — | — | — | — | **≡ dense(逐字)⇒ G3① ✓** |

**(3)"能丢多少才开始变"探索曲线**(同一 prompt,只改预算)

| 预算 | 视图页 | 质量覆盖 | 物化比 | 与 dense 的前缀一致 | 输出性质 |
|---|---|---|---|---|---|
| 16 | 16 | 81.0% | 50% | **100%** | 逐字相同 |
| 8 | 8 | 44.9% | 25% | 20% | 措辞变,**答案仍切题**(汇总校准值/编号/页签) |
| 4 / 2 | 6(下限) | 33.4% | 18.8% | 20%(与预算 8 同输出) | 同上 |

📌 预算下限 = `sink 2 + recent 4 = 6` 块(`select` 的保底恒保留)⇒ 预算 < 6 不改变结果。

**(4)门(e2e `gpu_08b_select_decode_probe`,按需开 `OWL_SELECT_DECODE=1`)**
①视图臂真的走了视图(`select_view_steps > 0` 且回退 0)②dense 臂一步都没走(否则不成对照)
③视图臂物化比 < 95%(防"预算设太大 ⇒ 视图=全量 ⇒ 对照空过" —— 踩过)
④**G3①**:探针全关的裸跑 ≡ dense 臂逐字 ⑤视图臂与 dense 前缀一致 ≥ 60%(首刀口径;
更严的"端任务质量"门留给 27B 长档,见 §8.14 的边界声明)。

**结论**:方案 A 在**输出级**站住了 —— 50% 物化率零代价,25% 仍切题。下一刀(B-3e)是把
host 未驻留块的 `upload_span` 与 prefill 重叠(流水),之后进 B-4/B-5。

### 8.17 B-3e:stage-in(宿主 → 设备回灌)实测 + **恢复延迟定档**(2026-10-18)

> **先摸清前置(重要修正)**:KV 腿的下沉/回灌**早已全接线**(`plan_offload`/`swap`/
> `commit_offload`/`plan_onload`/`commit_onload`,由 store 的 `Verdict` 与 prefill 段
> 边界驱动)⇒ B-3e 不是"从零修通道",而是 **①量它的延迟定档 ②找出 G1(262k 单卡)的真前置**。

**(1)实测(0.8B;小池 16 页 + 宿主层 32MiB;造压序列 = 会话1 → 会话2 压 → 会话3 原 prompt 召回)**

```text
轮 1 后:缓存 8 块(宿主承载 0)· 空闲槽 8/16
轮 2 后:缓存 20 块(宿主承载 8)· 宿主驻留 3.00MiB · **下沉 8 次/3072KiB** · 空闲槽 4
stage-in:**6 块 / 2.2MiB / 0.6ms(每块 384KiB · 3575MiB/s;全部走 prefill 段边界前置)**
轮 3 后:回灌 6 次/2304KiB(命中面复用)
```

**27B 工作集外推**(W=64 块 × 每块 2.1MiB ≈ 128MiB;按实测 3575MiB/s)= **≈ 36ms**
⇒ **优于论文 174ms@256K**,且这项工作还是**同步**做的(流水化后还能再藏)。
⇒ 结论:**恢复延迟不是瓶颈**(G1 的卡点另在别处,见 (2))。

**(2)本刀最有价值的发现:两条腿在召回路径上**耦合**

第一次跑 stage-in 账恒为 0:原因是**状态腿(快照)没开宿主层** ⇒ 压力把边界快照挤掉 ⇒
`reuse_face` 找不到边界快照 ⇒ 复用面被裁到 0 ⇒ **整条前缀复用失败**(不是"回灌慢",
而是"根本不回灌",退化成全量重算)。补上 `snap_host_mb` 后立刻正常。
📌 **律**:KV 块能下沉 ≠ 这条路能用 —— **边界快照的在场性是复用的通行证**;
配置里 `host_cache_mb` 与 `snap_host_mb` 必须成对开(否则 KV 落地了也没人认)。

**(3)G1(单卡 262k)的真前置(立案):那是我们自己的策略粗化**

**先纠正一处早先的误述**:本卷初稿把 ①"会话块表升级为载体查询"列为前置 —— **错的**。
逐块载体查询**早已存在**:

- 树条目本就是 `slot: Option<u32>`(`None` = 内容只在宿主层;`store/src/kv/tree.rs`);
- `pending_host()` / `unbind` / `rebind` / `alloc_slot` / `incref` / `decref` 全在;
- 引擎侧"占位(`PENDING_LOAD_PLACEHOLDER`)→ `pending_loads` → 段边界回灌 → 写回块表"
  整套流程也在 —— **只是目前只在 turn 起点用**,没在召回点用。

真正挡住 262k 的,是**两处刻意的粗化**(都是我们自己的设计选择,不是物理约束):

| 处 | 代码 | 粗化 |
|---|---|---|
| **store 出票粒度** | `KvStore::chain_down_span`(`kv/mod.rs`):`if pool.refs(slot) != 1 { return None }` | ①下沉单位 = **整条链**(`plan_offload(leg, tail)` 走 root→tail,all-or-nothing)②**一个活跃引用否决整条链**(哪怕只 1 块被会话表持有) |
| **engine 取数面** | `prefill_chunk.rs`:`block_tables = bt_leaf_flat()`(整条会话链)+ 每 token `kv_lens` | prefill 的"过去"只能表达**全量链**,不能表达"选中子集" |

⇒ **B-3e-2 = 三处小改**(两处是在现成机制上开口子,不是重构):

| # | 改哪 | 大小 | 说明 |
|---|---|---|---|
| 1 | store 加**逐块**下沉计划(`plan_offload_blocks(&[ChainHash])`:逐块判 `refs==1`、跳过活跃块,不再整链否决) | 小 | 账早已逐块(`slot: Option<u32>`),只是出票粒度是整链 |
| 2 | engine 在**召回点**复用现成的"占位 + 回灌"流程(被踢出工作集的块 ⇒ 释放会话引用 + 下沉;需要的块 ⇒ 回灌 + `rebind` + 写回块表) | 中 | 与 turn 起点的流程同构,主要是搬位置 |
| 3 | prefill 的"过去"也走**子集页表**(= B-3d 视图那套构造;核签名已是 `block_tables` + `context_lens` 页表语义,只差窗口算术那一处核内确认) | 中 | 唯一有"新东西"的一项 |

搬运速度**不是**瓶颈(实测 3575MiB/s;27B W=64 ≈36ms)⇒ §8.12 的流水/双缓冲(第 4 项)可以最后做。

**(4)门(e2e `gpu_08b_stage_in_probe`,按需开 `OWL_B3E_PROBE=1`)**
①真下沉(`offloads > 0`)②**真回灌**(`stage.blocks > 0` 且 `onloads > 0`)
③吞吐/字节/墙钟有样本 + 打印 27B 外推。⚠️ 旧的下沉探针**只下沉不回灌**(它的轮 3
命中的块仍在设备)⇒ 那边的硬门已撤,只报数(首跑就踩:把回灌门挂在错误的场景上)。

### 8.18 B-3e-2 第 1 项:store **逐块下沉票**(2026-10-18)

> 前置见 §8.17(3):挡住 262k 的是**我们自己的两处粗化**(出票粒度 + 取数面),不是物理约束。
> 本项只动**出票粒度**这一处,小而独立、可单测(不依赖 GPU / engine)。

**(1)新增 API(`crates/store/src/leg.rs`,与链级票并列)**

```rust
pub fn plan_offload_blocks(leg: &mut BlockLeg, chains: &[ChainHash])
    -> StoreResult<BlockOutPlan>          // 纯 host:算片 + 借宿主台(全无票 ⇒ 不借)
pub struct BlockOutPlan {                 // blocks / pieces / table 与 OutPlan 同形
    pub skipped: Vec<(ChainHash, SkipWhy)>,
}
pub enum SkipWhy { Unknown, AlreadyHost, LiveRefs }
```

**两处粗化的正反面**(这就是本函数存在的理由):

| 维度 | 链级 `plan_offload(tail)` | **逐块 `plan_offload_blocks(chains)`** |
|---|---|---|
| 单位 | 整条链(root→tail,**all-or-nothing**) | **逐块**(给哪些办哪些) |
| 活跃引用 | **一块活跃 ⇒ 整链 `Busy`** | 活跃块**跳过并记 `LiveRefs`**,不连坐 |
| 全无票 | `Busy`(零副作用) | `blocks` 为空(零副作用,不借宿主台) |

**调用方职责(写进 doc)**:要让某块可沉,必须先 `decref` 释放会话对它的引用 ——
这正是"占位 + 回灌"流程里会话表改写成占位符那一步在账上的对应动作
(第 2 项会在召回点接上;本项只把票开出来)。

**(2)单测(`crates/store/tests/store_e2e.rs:per_block_offload_skips_live_block_without_vetoing_others`,CpuFace 真搬运)**

```text
池 6 槽/自由 3 · 链 A(槽 0/1/2)其中**槽 1 被会话持有**
对照:链级票 ⇒ 整链 Busy          ← 粗化的现场(本测试要证的反面)
逐块票 ⇒ 下沉 2 块 · 跳过 1 块(LiveRefs)
  ⇒ 槽 1 仍 Device(活跃块不沉,但**不连坐**);槽 0/2 变 Host
回灌 2 块 ⇒ **逐字节恒等**(新槽 vs 沉前内容)
幂等:趁在宿主复询 ⇒ 全 AlreadyHost(零搬运);未知身份 ⇒ Unknown(不静默当成功)
回灌后又回设备 ⇒ 载体查询可证(幂等只对"已在宿主"成立 —— 首跑踩过)
```

**(3)状态**:第 1 项完成(store 侧 75+3 全绿;engine 未接线故零回归面)。
**下一项 = 第 2 项**:召回点复用现成的"占位 → `pending_loads` → 段边界回灌 → 写回块表"
流程(把 turn 起点的代码搬到召回点),届时本 API 才有生产调用者;第 3 项 = prefill 子集页表。

### 8.19 B-3e-2 第 2/3 项:召回点部分驻留 + prefill 子集表(2026-10-18)

> 第 1 项(逐块下沉票)见 §8.18。本刀把票**用起来**:召回点做下沉/回灌,
> 并让 prefill 的"过去"能表达**子集**(= 链上出洞时的唯一正确取数面)。

**(1)接线(三处,全在 B 轴自留地;A 轴只加了一个 `decref` 门面 + 占位符判定)**

| 件 | 位置 | 要点 |
|---|---|---|
| **召回点驻留调和** | `exec/coverage.rs::reconcile_residency` | 被踢出工作集的块:①**可沉性前置**(只认"缓存树 + 会话双持" `refs == 2`)②`decref` → `plan_offload_blocks` → 一条 `swap(Out)` → `commit_offload` → 表写**占位符**;选中但内容在宿主的块:一条 `swap(In)` → 回灌 → 写回新槽。**先沉后升**(沉腾出的槽正是升要用的) |
| **prefill 子集表** | `exec/prefill_chunk.rs`(`ChunkTables::build_view` + `past_view_of`) | **只在链上真出现洞时**才换表(没洞用全量链更保真);换表 = 页链 = 选中 ++ 尾部、`context_lens[t'] = 视图 token + (p+1−recall)`;FI/XQA 的 `indices/page_table` 同表 |
| **占位符守卫** | `coverage.rs::select_view_step` + `store::KvStore::release_table` | ①视图缺位但链上有占位符 = **响亮打哨**(不许当页号读)②`release_table` **跳过占位符**(宿主块没有设备槽可还 —— 实测踩出 `槽 4294967295 越界`) |

**单一定义**:占位符哨兵 = `owl_store::HOST_PLACEHOLDER`(引擎侧 `PENDING_LOAD_PLACEHOLDER`
改为引用它 —— 两侧同源)。

**(2)实测(0.8B;两轮:轮 A 填缓存树 ⇒ 轮 B 前缀命中 + 新尾巴;预算 32 块 / 48 块)**

```text
dense  :32 页 · 覆盖 91.7% · 物化比 66.7% ⇒ 39c
view   :26 视图步 · 回退 0 ⇒ **逐字一致 39/39 = 100%**
partial:26 视图步 · 回退 0 · **下沉 15 块** · 不可沉 1 块 · **prefill 子集表 1 次** ⇒ 40c
G3①    :plain ≡ dense **逐字**(探针全关零扰动)✓
```

**(3)口径裁决:partial 臂**不能**拿"与 dense 前缀一致"当门**

部分驻留的语义就是"被踢出工作集的块**在 prefill 阶段也不参与**"(它们已沉到宿主)⇒
稀疏 prefill 必然改变首批 token(实测前缀 28%,但输出完整、切题)。故 partial 臂的门 =
**机制三条**:①真下沉 ②prefill 子集表真生效 ③输出非退化;质量门归 B-5(27B 长档)。

**(4)新发现 / 立案**
1. **宿主层按缓存树条目键控** ⇒ **本轮刚写、还没进树的块不可沉**(`refs == 1`,硬沉会让槽
   变自由槽 —— 实测踩 "槽 2 是自由槽,不能加引用")。链注册在 turn 收口 ⇒
   **第一轮内无法下沉**,第二轮起才可以。要让"本轮内的块也能沉"需**增量注册**(O(n) 总代价,
   需 store 侧配合)⇒ **第 2b 项**(G1 的真前置之一;设计卷 §8.17(3) 的三处小改之外的第 4 条)。
2. **召回点回灌**(selected-but-host)在本场景未触发(选中集没落在已沉块上)⇒ 该方向仅有
   构造保证(与 turn 起点的回灌同原语),**未实证**。挂账:造一个"轮内两次召回、第二次选中
   首次沉掉的块"的场景。

### 8.20 B-3e-2 第 2b 项:增量链登记(**轮内可沉**)(2026-10-18)

> 前置问题(§8.19(4)):宿主层按**缓存树条目**键控 ⇒ 本轮刚写、还没进树的块
> `refs == 1`,硬沉会把槽变成自由槽("槽 2 是自由槽,不能加引用")⇒ **第一轮内沉不了**。

**(1)两处新增(store 侧,链身份语义**零改动**)**

| 件 | 位置 | 要点 |
|---|---|---|
| `PrefixTree::insert_inc(parent0, blocks)` | `kv/tree.rs` | 从**已注册链尾**续挂新块:每块一次 `hash_block`(全 turn **O(n)**);与 `insert` 共用同一个 `insert_one` + `trim`(**单块插入语义只有一份实现**) |
| `KvStore/BlockLeg::register_chain_inc` | `kv/mod.rs` / `leg.rs` | 与 `register_chain` 同记账(新块 +1 引用、被逐条目 -1 引用) |

**(2)engine 接线**:`capture_gdn_snap`(块边界事件,与索引采集/GDN 快照同处)在
`select_partial` 档调 `register_chain_inc(parent0 = 推进前链尾, 本次新增整块 + 真槽)`
—— `parent0` 就是上次登记的尾(逐块父链连续)⇒ 全 turn 只哈希一遍。
只登记**真槽**块(占位块本就在树里);`store.chain.inc.{blocks,evicted}` 计数入 metrics。

**(3)实测(同一 B-3e-2 场景)**

| | 下沉 | 不可沉(未入树) | prefill 子集表 | 输出 |
|---|---|---|---|---|
| 2b 之前 | 15 块 | **22 块** | 1 次 | 40c(与 dense 前缀 28%,稀疏 prefill 语义) |
| **2b 之后** | **16 块** | **0 块** | 1 次 | 40c(同上,**不变**) |

⇒ 轮内所有被踢出工作集的块现在**都可沉**;输出与之前一致(语义未变,只是可沉面全开)。

**(4)单测**(`store/kv/mod.rs`):增量登记(**两段续挂**)≡ 全量登记 —— 命中**块数 /
逐块身份 / 槽序**三件逐项相等;重复登记同一段 = 只触达不重复插入。

**(5)风险与边界**:**活跃面保护不受影响** —— `chain_down_span`(链级票)与
`drop_cached` 都对 `refs != 1` 响亮拒 ⇒ 轮内登记的块不会被常规保洁误逐/误沉,
只有**选择层的召回点**(先 `decref`)能动它们 ✓。缓存容量变紧是代价(登记面变大 ⇒
`up.evicted` 可能逐掉别的叶)—— 计数已入 metrics。

**至此 B-3e-2 四项全落**(逐块票 / 召回点下沉与回灌 / prefill 子集表 / 增量登记);
G1(单卡 262k)的**结构性前置已齐**,剩的是 27B 真机验证(B-5)。

### 8.21 B-4 第 ② 项:attention-aware LRFU 接策略对象(2026-10-18)

> 设计卷 §6 的落点:"ActKV 的 attention-aware LRFU 接到 A 轴已有的 `EvictPolicy` ——
> 决定谁下沉/谁留设备";我们**只下沉不丢内容** ⇒ 它定的是"谁的设备副本先让位"。

**(1)新增(store;纯逻辑 + 一个缺省空操作的口子)**

| 件 | 语义 |
|---|---|
| `EvictPolicy::note_recall(&[ChainHash])` | **召回信号面**(缺省 = 无操作 ⇒ 旧策略行为**逐字不变**) |
| `LrfuPolicy(lambda)` | `KS(b) ← λ·KS(b) + (1−λ)·hit(b)`,`hit` = 被召回点选中;出序 = KS 升序 |
| `EvictPolicyKind{Lru, Lrfu(λ)}` | boot 参数对象(§A2"策略是对象,不是分支");`build()` 造策略 |
| `Rel::note_recall / evict_policy_name` | 门面(引擎只经门面碰策略面) |

**懒衰减**(关键实现点):不逐步扫全表,按 `KS_last·λ^(now−t_last)` 惰性折算 ⇒ 每操作
O(1)、排序时按需折算;时间戳**只读**、**步进由 `order` 推进**(每次出序 = 一步)——
否则"命中越频繁、时间戳越新"会自相矛盾(首版就踩:命中时 `fetch_add` 导致时间戳比自己读到的晚一拍)。

**(2)引擎接线**:`EngineKnobs.evict_policy`(`OWL_EVICT_POLICY=lru|lrfu|lrfu:λ`;
**坏值 = 响亮拒 + exit 2**,与准入姿态同律);boot 装配一次
(`b.rel.set_evict_policy(kind.build())`);召回点选完块即 `rel.note_recall(&set.blocks)`
(LRU 下是空操作 ⇒ 默认路径零影响)。

**(3)单测(store,纯逻辑)**
1. **B-4 门**:"被注意力选中 20 次的冷块" vs "从没被选中的温块" ⇒ LRFU 把温块排前面
   (先沉它);**同场景 LRU 的序正相反** —— 这就是要修的病,一句话可证;
2. **懒衰减 ≡ 逐步衰减**(λ=0.5 空转 4 步:惰性值 == `λ^4·(1−λ)`,1e-12);
3. 无 `chain` 的候选(状态腿)→ 按 `last_use` 兜底;λ 越界取缺省且不 panic;
   `LruPolicy.note_recall` 编译期证明是空操作。

**(4)第 ① 项(预算在线调节)立案 —— 附一条**节奏**洞察**:ActKV 的置信度驱动预算需要
"**decode 期间也会召回**"才有用武之地;而当前召回**只在 prefill 块边界**发生
(§1 的"召回点 = step 边界"),那时**还没有任何生成 token** ⇒ 置信度信号无从谈起。
⇒ ① 的正确序是:先拍"召回节奏"(是否允许 decode 中重选),再接置信度;
否则就是给一个一年只跑一次的旋钮装传感器。**挂账**(设计卷 §7 待拍 4 之一)。

### 8.22 **B-5 Stage 2:27B 单卡 + KVMem 虚拟空间 —— 实测跑通**(2026-10-18)

> **一句话**:单张 3090 Ti(权重 19.6GiB,装配后驱动空闲 **3927MiB**),**设备 KV 池只给
> 2048 token 却服务了 8192 token 的上下文(4× 超卖)** —— 下沉 369 块 / 回灌 161 块 /
> prefill 子集表 52 次 / decode 23 步零回退,输出切题。
> ⇒ **这就是"KVMem 虚拟空间能力"的实证**:设备上只留工作集,全量 KV 在宿主。
> (同池尺寸下 vLLM 这类"全量常驻"引擎必然 OOM —— 它没有"只留工作集"的能力。)

**(1)前置澄清(用户点出、我早先写错)**
- **G1 不需要双卡**;owl 也**没有** TP2(`crates/comm` 只有策略签名,"M2 接 CUDA 实现"未接线)。
- 我此前那句"需 3090 对空闲"是把 **vLLM 的 27B 生产档(TP2)** 误当成 owl 前提 —— 已纠正。

**(2)单卡算术(实测)**

| 项 | 值 |
|---|---|
| 27B AWQ-INT4 权重(磁盘 5 分片) | 19.6 GiB |
| 装配后 live 账 / 驱动空闲 | 19819 MiB / **3927 MiB** |
| 262k 全量 KV(fp8,16 层×hkv4×hd256) | 8.0 GiB ⇒ **全量必然装不下**(需 27.6GiB > 24GiB) |
| 工作集 W=64 块(2048 tok) | 64 MiB ⇒ **只留工作集绰绰有余** |

**(3)Stage 2 实测(27B / ctx 8192 / 池 2048 = 4× 超卖 / chunk 128 / 单 3090 Ti)**

```text
[b5-27b] 用时 625.5s · 视图步 23 · 回退 0 · 召回 32 页 · 物化比 13.3%
[b5-27b] 部分驻留:下沉 369 块 · 回灌 161 块 · 不可沉 0 块 · prefill 子集表 52 次
[b5-27b] 池 223/256 槽自由 · 宿主 KV 驻留 0.41GiB
[b5-27b] 输出:用户要求我从六份记录中汇总校准值、编号、页签。
```

📌 **`回灌 161 块` 顺便把 §8.19(4) 的"召回点回灌未实证"挂账关掉** —— 轮内"选中但已沉的块"
确实被回灌回来用了。

**(4)27B 首跑挖出的 5 个真 bug(全是"洞 × 消费方"这一类)**

| # | 症状 | 真因 | 修法 |
|---|---|---|---|
| 1 | `dtoh` 偏移 `2.8e14` | **索引采集**拿占位符当页号 | 跳过占位块(内容身份键控 ⇒ 沉前已采过) |
| 2 | 同上 | **覆盖率探针 K/V 回读**同病 | 跳过 + 记数(读数口径收窄到设备驻留子集) |
| 3 | `ILLEGAL_ADDRESS` | **XQA 裸 decode 臂**的页表来自会话链(含洞) | 档位级互斥:部分驻留档**关 XQA** + 响亮告知(视图×XQA 对接挂账) |
| 4 | 同上 | **视图在回灌之前装配** ⇒ 页表里是占位符 | 顺序改为**先调和驻留、再装配视图**,装配处加占位符哨兵 |
| 5 | 同上 | **`write_bt`** 把带洞的表写进设备 | 守卫(响亮拒)+ **视图在场时跳过 `write_bt`**(解码走 `sel_pt`) |

**(5)两个性能陷阱(27B 才暴露)**
- **逐页同步回读**:原实现每 (层,页) 一次 `dtoh_at` ⇒ 27B/8k 每边界 4096 次往返 ≈ 20s+,
  一次 prefill 拖到 20 分钟("像卡死")⇒ 改 **`dtoh_batch` 单命令**(索引采集 + 探针两处);
- **诊断采样**:重诊断(K/V 全量回读 + host 逐元素 f16→f32)在 debug 下每边界秒级
  ⇒ 加 `OWL_COV_ANALYZE_EVERY=k`(选择/视图/驻留**每边界都做**,只采样重读数)。

**(6)环境教训**:`CUDA_VISIBLE_DEVICES=n` 按 **CUDA 序**(FASTEST_FIRST)解释:
**0/1 = 双 3090 Ti,2/3 = 双 3080**(≠ nvidia-smi 序;我为此吃了一次"选到 3080 就 OOM")。

### 8.23 B 轴**耗时构成实测** + 两刀纯性能修正(2026-10-19)

> **一句话**:先立相位账再动刀 —— 实测把三个猜测全打翻:**①逆 RoPE 只占 0.2ms(不是瓶颈)
> ②真凶是"诊断面混在生产路径里",单次真实质量扫描 = 6.1s(27B/8k,占墙钟 **31%**)
> ③生产面 B 轴本身只占 7.4%**。

**(1)相位账(`BAxisProfile`,纯计时零语义)**

落点:`crates/engine/src/running.rs`(`BAxisProfile::line()`)、`exec/coverage.rs`
(`coverage_analyze` / `coverage_take_q` 分段)、`exec/spec.rs`(`collect_index_for_block`)。
八相位:索引 / 查询面(设备往返 · host 解码 · 逆 RoPE · 真实质量)/ 重诊断 / 打分 /
论文口径 / 选择 / 视图 / 调和。读数口与 `PartialStats` 同款(`[cov]` / `[b5-27b]`)。

**(2)两刀修正(零语义)**

| 刀 | 症状 | 修法 |
|---|---|---|
| **① 论文口径打分移进采样门** | `paper_scores`(softmax 口径)在**每个**块边界无条件重算 —— store release 延迟门实测它是快路的 **53×**(8192 块 11905ms vs 222ms);而它报的 `paper_coverage` 是**相对 `mass`** 的覆盖,`mass` 不在场时恒 0 ⇒ 采样外重算 = 纯白烧 + 脏读数 | 与 `mass` 同门(代码里"重诊断采样"注释本来就写的是这个意图) |
| **② 重诊断默认只在"末个边界"** | K/V 全量回读 + **`true_block_mass`**(逐 query 行 × (层,头) 扫**全 ctx**)在每边界都跑 ⇒ 全军 **O(ctx²)** | `analyze_now` 三档:`未设` = 末边界一次 · `=k` = 每 k 块 · `=0` = 关。**末边界**由调度器**同一份**块长函数(`Self::prefill_step_chunk`)预测下一条边界会不会来 —— 而非另写一套分块算术 |

> ⚠️ 口径提醒:`is_last`(末个 **chunk**)≠ **末个边界**:prompt 末尾不页对齐时,
> 最后一个边界之后还有 chunk,但没有边界了。用 `is_last` 判会导致**一次诊断都不做**。

**(3)0.8B 实测(ctx 3072 / 96 块 / chunk 1024;release)**

| 相位 | 修前(×3 边界) | 修后 |
|---|---:|---:|
| **合计** | **793.2ms** | **405.6ms** |
| 索引 | 74.4 | 71.4 |
| 查询面 | 751.8 | 431.7 |
| ├ 设备往返 | — | 91.4(**30ms/边界**) |
| ├ host 解码 | — | 0.9 |
| ├ 逆 RoPE | — | **0.2** |
| └ 真实质量 | — | 339.1 |
| 重诊断 | 78.7 | 42.9 |
| 打分(快路) | 1.4 | 1.4 |
| 论文口径 | 36.7 | 18.4 |

**(4)27B 实测(ctx 8192 / 池 4096(2× 超卖)/ chunk 128;release;墙钟 19.9s)**

```text
B 轴相位 ×54 边界:合计 7044.6ms
  真实质量(诊断) 6138.4 ← **单次 6.1s = 墙钟 31%**
  索引 754.5(14ms/边界) · 打分 220.7(4ms/边界) · 调和 288.3 · 重诊断 105.2
  论文口径 245.6 · 查询面(非诊断) 202.0 · 选择 0.8 · 视图 4.6
  ⇒ **生产面合计 ≈ 1470ms = 墙钟 7.4%**
```

对照 §8.22(同参,debug):625.5s ⇒ **release 19.9s**(debug 的 host 热点 ≈ 30×)。
门不变:下沉 350 块 / 回灌 166 / 子集表 46 / 视图步 23 / **回退 0** / 输出同形。

**(5)三条反直觉读数(给后续调优)**

1. **逆 RoPE 不是瓶颈**(0.2ms@0.8B / 13.3ms@27B 全军)。原以为"M1 图内取 pre-RoPE K
   能省掉逆 RoPE"⇒ **收益点判断错了**;图内 tap 的价值在**别的**地方(fp8 池可用、
   已沉块可采、口径与核内一致),不在算力。
2. **`dtoh` 的固定成本随 chunk 计算量走,不随字节走**:"读语义"要**排空 COMPUTE** ⇒
   单次往返 ≈ 该 chunk 的 GPU 时间(0.8B/chunk1024 = **30ms/次**;27B/chunk128 = **1.7ms/次**)。
   ⇒ 少一次往返 > 少一半字节。
3. **诊断面 vs 生产面必须分开记账**:在 27B 档,诊断是生产的 **4×**。凡"探针开着测性能"
   得到的读数,都要先把诊断摘出去看(否则会把诊断的账算到选择层头上)。

**(6)当前挂账(按收益排序)**

| # | 项 | 预期收益(27B/8k 口径) |
|---|---|---|
| 1 | **真实质量搬到设备**(或只在小 ctx 采) | **6.1s → ~0**(占墙钟 31%) |
| 2 | **两次 D2H 并成一条批命令**(索引 K + Q 面) | ~50%(0.8B 档 30ms→15ms/边界) |
| 3 | **召回节奏**(每边界 → 每 N 块/压力驱动) | 线性(边界数 ↓ k 倍) |
| 4 | 图内 pre-RoPE K tap(M1) | 非算力收益:解 **fp8 池**与**已沉块**两个结构性限制 |

**(7)第三刀:召回节奏(`OWL_RECALL_EVERY`,2026-10-19 同日)**

**病灶**:单次打分是 O(块数) **线性**,但"每块边界都打"⇒ 一次 prefill 累计 **O(块数²)**。
参考实现把这条曲线压回线性,靠的是**每请求只召回一次**(中间边界只走不打分的压力路径)。

**两版设计的取舍(第一版实测吃亏,已推翻)**:

| 廉价档语义 | 打分 | 调和 | 下沉/回灌 | 末边界覆盖率读数 | 判决 |
|---|---:|---:|---:|---:|---|
| v1:**近期选择 + 调和** | 220.6→60.5ms | **269→545.6ms ↑** | 350/166 → **650/466 ↑** | 74.7% → **41.8%** | ❌ **净亏** |
| v2:**只采索引,不干预驻留** | 220.6→**63.0ms** | 269→**149.7ms ↓44%** | 350/166 → **231/47** | 74.7% → **70.4%** | ✅ |

- v1 为什么亏:近期名单与上一次的打分名单**互相覆盖** ⇒ 每次完整召回都要把上一次
  廉价档沉掉的块再回灌一遍;"过了多少边界"成了搬运量的自变量。
- v2 的语义:**两次完整召回之间不改变选择** —— 驻留交给 A 轴(池满时本来就在按策略淘汰),
  下一次完整召回时一次性对齐 ⇒ 搬运量由"**选择变了多少**"决定。
- 41.8% 那条是**读数失真**而非质量下降:驻留子集变了 ⇒ `mass` 的可读面变了
  (`host_skipped` 块按零计)⇒ **跨配置的覆盖率读数不可直接比**。这条要写进读数口径纪律。

**采样步进用边界序号,不用块号**:块号是页宽的整数倍(边界只在页对齐处),拿它取模会退化
—— 当页宽本身是 k 的倍数时**每个**边界都命中(实测 chunk 与页对齐时 k=4 仍 15/15 全完整召回)。

**27B/8k 实测(k=1 → k=4;release)**:打分 221→63ms · 调和 269→150ms · 下沉 231/回灌 47 ·
视图步 23 **回退 0** · 输出同形。⇒ 生产面 B 轴 **1.47s → 1.2s(占墙钟 ~6%)**。

**(8)顺带修掉的 O(B²)**:`store::select()` 的 `score_of` 原是对 `scored` 的**线性查找**,
建 `rest` 时逐元素调一次 ⇒ **O(B²)**(B=8192 时 3300 万次比对/边界,纯 latent 开销)。
改一次建表 ⇒ O(B)。新增 `select_recency()`(无打分无排序,参考的 prefill 压力路径同构)。

### 8.24 第四刀:**生产面与诊断面拆分**(低显存档可开)+ 配置面落地(2026-10-19)

> **一句话**:拆之前 B 轴整个挂在 `OWL_COVERAGE_PROBE` 下 ⇒ **低显存档位根本开不了
> 这个功能**;拆之后 27B/8k/2× 超卖 的墙钟 **19.6s → 13.7s**,生产面 B 轴 =
> **1480ms(10.8%)**,其中诊断面 0(不再背账)。

**(1)病根:三处门共用同一个诊断旋钮**

| 件 | 旧门 | 新门 |
|---|---|---|
| Q 槽接入(`prefill_chunk.rs`) | `coverage_probe` | **`coverage_probe \|\| select_decode`** |
| 设备视图面(`engine.rs` `select_view`) | `coverage_probe` | **`coverage_probe \|\| select_decode`** |
| 召回入口(`phases.rs` 边界事件) | `coverage_probe` | **`coverage_probe \|\| select_decode`** |
| 真重诊断(质量/曲线/上界/论文口径/方案对照) | (无独立门) | **`coverage_probe`**(`diag`) |

**为什么必须拆**:诊断面单边界成本**含 `ctx` 因子**(逐 query 行 × (层,头) 扫全 ctx)
⇒ 27B/8k 实测单次 **6.3s = 墙钟 32%**,而那时生产面只有 6%。
"想开功能就先背诊断的账"是不能交付的形态。

**(2)新发现并接住的一处静默风险**

拆分后 `select_decode` 可以在 `coverage_probe` 关闭时生效 —— 而 **prefill 桶图路
不支持 Q 槽**(Q 槽只在 eager 路接线)。若不处理,生产档会走桶图 ⇒ **Q 面静默为空**
⇒ 打分在空 query 上算(不报错,只是选错)。⇒ 把 `select_decode` 加入桶图让位条件
(与诊断档同律,响亮告知一次),并挂账"桶图路学会 Q 槽后生产档就不必让位"。

**(3)27B 实测(纯生产档:`OWL_COVERAGE_PROBE=0`;ctx 8192 / 池 4096(2× 超卖)/ chunk 128)**

| 召回节奏 | 墙钟 | B 轴合计 | 索引 | 查询面 | 打分 | 调和 | 下沉/回灌 |
|---|---:|---:|---:|---:|---:|---:|---:|
| k=1 | 13.7s | 1481ms | 766 | 207 | 223 | 279 | 350/166 |
| k=4 | 13.3s | **1136ms** | 723 | 210 | 63 | 139 | 231/47 |
| k=8 | 12.9s | 1184ms | 749 | 273 | 33 | 130 | 215/31 |

全档 **视图步 23 / 回退 0**;门(下沉>0 · 视图驱动 · 输出非退化)全绿。
⇒ 生产面 B 轴 ≈ **8.5%**(k≥4);剩下的大头是**索引采集的 D2H 往返**(~750ms = 52%)。

**(4)配置面(`shared/config.rs` + `assets/flavors/`)**

新增 `[kvmem]` 段(`KvmemCfg`):`select_decode` / `select_partial` / `index` /
`select_blocks` / `select_sink` / `select_recent` / `recall_every`。
读取优先级同 `pool_tokens` 的既有律:**env 覆盖 > 配置对象 > 缺省**;
`EngineKnobs::from_config` 是唯一搬运点。

- `assets/flavors/awq27b.toml`(高显存档):显式写 `[kvmem]` **全关** ——
  池 ≥ ctx 时每个召回点都是纯开销;
- `assets/flavors/awq27b-kvmem.toml`(**新的低显存档档**):派生自 awq27b +
  `select_decode/select_partial/index = true`、`select_blocks = 32`、`recall_every = 4`。

**(5)顺手修的既存回归(非本次引入,已如实标注)**

`7c5627f`(A 轴收官)把 `PoolCfg.gdn_slots` 改名 `max_active_turns`,但
`assets/flavors/{awq27b,spec-dflash}.toml` 的文本没跟上 ⇒ `deny_unknown_fields`
**直接拒解析**(`owl-shared` 两条测试红)。已改名为新字段名(语义不变:单会话 = 1)。

⚠️ **未修、待你拍**:`awq27b` flavor 的白名单列了 `owl_cast_bf16_f16`,而
`PROD_AWQ27B_2026_10_13` 台账没有它 ⇒ `server/ops_whitelist` 的
`flavor_whitelists_match_audited_tables` 红(同为既存)。两个方向(从 flavor 删,
或补进台账)会改变生产档放行集,**不由我单方面定**。

### 8.25 第五刀:**锁①/锁② 条件解禁 + fp8 池支持**(2026-10-19)

> **一句话**:先纠正一个我此前给出的**错误结论** —— 我说过"fp8 无分页 ⇒ 选择层结构上
> 不存在",**那是错的**:`kv_paged_policy(dims.dtype)` 收的是**模型 dtype**(F16),
> `kv_fp8` 是独立开关 ⇒ **fp8 池本来就是分页的**。真正的堵点只有一处:
> **索引采集把 fp8 页 K 读回来解码**。本刀把它接上,并给两道锁装了**配置门**。

**(1)fp8 池的字节事实(单源 = 池写核)**

`owl_qknorm_rope_kv_insert_f16_{knhd,fp8kv_knhd}`(`kernels/cu/fused/qknorm_insert.cu`):

```text
key/value_cache: [nb, page, Hkv, hd]      kk = ((block·page + off)·Hkv + kh)·hd + d
f16 : __half          2B/elem
fp8 : unsigned char   1B/elem,值 = __nv_fp8_e4m3(n).__x —— **裸 e4m3,无 scale**
```
两侧**寻址式逐式相同** ⇒ 分支点只有「每页字节」与「字节→f32」两处。

**(2)落地:单源解码面 `exec/kvpage.rs`**

`KvPageFormat::{F16, Fp8E4M3}`,`of_pool(kv_fp8)` = **从池实况定格式**(不从第二个配置项猜)。
e4m3 → f32 走 `owl_models::kernel::e4m3_decode`(re-export 自 kernels 的**独立手写、
已被 GPU 逐字节验证**的参考)⇒ **引擎侧零第二份 e4m3 实现**;
单测 `fp8行解码与kernels参考同源` 逐位比对,将来若有人写第二份即红。
索引采集(`spec.rs`)与诊断回读(`coverage.rs`)两处改写,原有 fp8 硬拒全部撤除。

**实测(0.8B / ctx3072 / 96 块 / 池 KV = fp8 e4m3)**:

| 指标 | f16 | **fp8** |
|---|---:|---:|
| top8 覆盖率 | 87.5% | **87.5%** |
| 上界 | 88.7% | 88.8% |
| 选择器覆盖 / 论文口径 | 97.9% / 97.9% | **97.9% / 97.9%** |
| 索引在场块 | 96/96 | **96/96** |

⇒ 与 f16 **语义等价**(差异只在 e4m3 量化误差,读数落在千分位)。

**(3)锁①:池下限按配置门控放开**

原 `nb = max(ceil(pool_tokens/page), ceil(seq_tokens/page))` **无条件**把池钉在
"单会话全量上下文" ⇒ **选择层只省访问面、不省显存**。改:

```
nb = ceil(pool_tokens/page)                       若 PoolPlan.pool_overcommit
nb = max(上述, ceil(seq_tokens/page))             否则(缺省 = 旧行为逐位不变)
```
`pool_overcommit` 的唯一产地 = `EngineKnobs::kvmem_virtualization`
= `[kvmem].select_decode && select_partial`。配套 **boot 硬校验**:
`select_blocks + sink + recent ≤ nb`,装不下直接拒启并报数(不许跑起来才 `Exhausted`),
同时**响亮告知**池/上下文/超卖倍率/宿主预算。

**(4)锁②:`TurnFloor::Bytes` 条件解禁**

旧护栏"非 Strict 一律拒启"(`engine.rs`)是因为当时 B 轴未落地。现改**条件放行**:
只有 `[kvmem].select_decode + select_partial` 同时为真才接受非 Strict;
配套 `[kvmem].dense_floor_mib`(`None` + 虚拟化开 ⇒ `Bytes(0)` = "工作集装得下就准入";
`Some(n)` ⇒ `n` MiB)。映射归引擎(配置层不依赖 store —— 分层)。

**两侧方向都由 e2e 钉住**(`store_admission_gate`,CPU mock):

| 门 | 场景 | 期望 |
|---|---|---|
| `non_strict_dense_floor_is_refused_loudly` | 虚拟化**关** + Bytes 地板 | **拒启**(文案指明虚拟化/select_decode) |
| `non_strict_dense_floor_is_allowed_when_virtualization_is_on` | 虚拟化**开** + Bytes(0) | **放行** |

后者的第一版**被新校验正确拦下**(池 16 页 < 默认工作集 70 页)⇒ 反过来证明 boot 校验在干活。

**(5)配置面 + flavor**

`[kvmem]` 新增 `dense_floor_mib` / `pool_overcommit`;`KvmemCfg::virtualization_enabled()`
= **所有解锁类护栏的唯一判据**。

`assets/flavors/awq27b-kvmem.toml` 改成**真正的低显存档配方**(27B/24G 为例):

```text
上下文 32768 tok = 1024 页        (27B f16: 2 GiB KV)
设备池  2048 tok =   64 页        (        128 MiB)   ⇒ 超卖 16×
宿主   非驻留 960 页 ≈ 1.9 GiB    ⇒ OWL_HOST_CACHE_MB=3072
```

**(6)锁①的**另一半**:块表必须按 chunk 增量长(2026-10-19 实测暴露)**

`nb` 放开之后 27B 首跑仍然挂,原文:

```text
store: 资源耗尽:target 链:KV 块池耗尽:需 218 块,空闲 64,table=0 —— 缓存压力驱逐后仍不足
```

**根因不在池,在块表的槽预留模型**:turn 起点 `scheduler.rs` 一次
`ensure_for_len(block_table, prompt_len)` —— 把**整个 prompt** 的页数一次性要到位。
池 ≥ ctx 时这一步从不暴露(反正装得下);池 < ctx 时它要几千页槽 ⇒ 必 `Exhausted`。

修法(同样**只在虚拟化档**生效):**用到哪长到哪** ——

| 位置 | 非虚拟化档(缺省) | 虚拟化档 |
|---|---|---|
| turn 起点(`scheduler.rs`) | 长到 `prompt_len`(旧行为) | 只长到**首个 chunk** |
| 每 chunk(`execute_prefill` 前) | 不动 | 长到 `base + t`(增量 `chunk/page` 页) |

腾位归召回点的驻留调和(A/B 两轴)——**这正是超卖成立的方式**。
配套 boot 硬校验加一项:`select_blocks + sink + recent + chunk/page ≤ nb`。

**(7)27B 低显存档实测:通过(4× 超卖)**

```text
[b5-27b] ctx=8192 tok · 设备池=2048 tok(超卖 4.0×)· 装配后驱动空闲 4311MiB · live 账 19433MiB
[b5-27b] 用时 12.6s · 视图步 23 · 回退 0 · 召回 32 页(物化 14.8%)
[b5-27b] 部分驻留:下沉 268 块 · 回灌 84 块 · prefill 子集表 43 次
[b5-27b] 池 31/64 槽自由          ← 池真的是 64 页(128 MiB),不是 256 页
[b5-27b] B 轴相位 ×54 边界:合计 251.2ms | 索引 501 查询面 312 打分 60 调和 168
         ‖ 召回档:完整 14 · 廉价 40
```

| 指标 | 池被钉在 ctx(旧) | **池 64 页(本刀)** |
|---|---:|---:|
| 设备 KV 面(27B f16) | 512 MiB(256 页) | **128 MiB(64 页)** |
| 装配后驱动空闲 | 3927 MiB | **4311 MiB(+384 MiB)** |
| 链长上限 | = 池 | **4× 池**(超卖) |
| 门(零崩/下沉/视图驱动/回退/输出) | 全绿 | **全绿** |

⇒ **"设备只留工作集、全量 KV 在宿主"第一次在 27B 上成立并省下了显存**(此前只省访问面)。

**(8)一条新读数:超卖档把压力推给了**状态腿**

低显存档下 GDN 快照腿会频繁撞自己的分片上限,响亮打点(不崩):

```text
DBG-SNAP 下沉失败: 资源耗尽:宿主层越界:637009920 > 536870912(关层请置 0,放量请调大 host_bytes)
```

⇒ 超卖档必须同时给 **`OWL_SNAP_HOST_MB`** 留量(27B 快照 607 MiB > 场景给的 512 MiB 分片)。
flavor 注释里已写明;这是**配置配平**问题,不是缺陷(行为 = 沉不下的快照被丢,该边界不可复用)。

### 8.26 **"8K prefill 12s"定谳** + 池下限/并发算式 + 一个既存真 bug(2026-10-19)

> **一句话**:12s 不是 KVMem 造成的 —— 是**三个协议因素叠加**,其中一个是**既存真 bug**
> (生产主力 prefill 核在本分支上根本开不起来)。同协议对照下,低显存档代价
> = **+18%**(bare 6.8s vs partial 8.0s),且这 1.2s **全是可优化的回读**。

**(1)三方对质(同一场景/同一 prompt/同一卡,顺序跑)**

| 档 | chunk | 核 | 用时 | t/s |
|---|---:|---|---:|---:|
| 裸档基线 `std-bare.csv`(server;f16 池 296 页) | 512 | FI+split | **6366ms** | **1287** |
| 我最初的跑(KVMem 全开) | 128 | 无 FI、split 撞门 | 12.6s | 553 |
| **bare 臂**(KVMem 全关) | 128 | 无 FI、split 撞门 | 15.2s | 458 |
| bare | 512 | FI+split(修后) | **6.1s** | 1143 |
| **partial 臂**(池 64 页 = 4× 超卖) | 256 | FI+split | **8.0s** | 871 |
| partial | 512 | — | 差 **2.5 MB** 显存(体量硬墙) | — |

⇒ **三条原因,与 KVMem 无关**:

| # | 原因 | 代价 |
|---|---|---|
| 1 | 场景把 `chunk` 压到 **128**(为省显存;512 的工作区 ~3.9GB) | 12.5s → 6.1s(**2×**) |
| 2 | 测试档 `EnvProvider::default()` ⇒ **`fi = false`**,而生产"FI 恒启用"(`from_dispatch` 原话);生产 prefill 走 FlashInfer | ~2s |
| 3 | **`prefill_split` 契约 rank bug**(见下) | 挡住 1+2 的生效路径 |

**(2)既存真 bug:`prefill_split` 的 scratch 声明 rank 错(修了)**

```text
[算子准入] contract owl_prefill_split_f16_knhd_hd256:
  scr_out 期望 rank = 2(声明) 实得 1(线格式事实)
```
- 契约(`kernels/ops/attn.rs` 的 PARAMS):`scr_out` F16 **r2**、`scr_stat` F32 **r2**;
- `attention.rs` 的 **paged 分块 prefill 路**却把两者声明成 **r1**(一维扁平);
- 同一内核在 `attention.rs` 另一条调用路声明的是 r2(注释还写着"契约:scr_out/scr_stat 皆 r2")
  ⇒ **两处规范不一致**,而准入门站在 r2 那边。
- 后果:**本分支上开 `prefill_split` 必然拒启** ⇒ 所有 27B prefill 读数都是"未开分块核"档。
- **`main` 上同样存在**(已核:main 的 941 行 r1 / 2680 行 r2)⇒ **既存,非本刀引入**。
- 修法:**只改声明、零新节点**(`with_shape` 就是那个口子,元素数不变只改 rank)。

**(3)池下限与并发的算式(boot 硬校验,已落地)**

```text
池页数 ≥ N_active × (select_blocks + sink + recent + ceil(chunk/page))
         └─ N_active = max_active_turns(每个活跃 turn 都要自己一套)
准入地板(每 turn) = (select_blocks + sink + recent) 页 × 每页字节   ← 缺省自动填
```
**27B f16 算例**(页 = 32 tok × 4 头 × 256 维 × 2(K+V) × 2B × 16 层 = **2 MiB/页**;fp8 = 1 MiB):

| 并发 | 每 turn 页 | 池下限 | 对照 dense |
|---:|---:|---:|---:|
| 1 | 32+2+4+8 = 46 页 = 92 MiB | 92 MiB | 512 MiB |
| 2 | 46 页 | 184 MiB | 1 GiB |
| **8** | 46 页 | **736 MiB** | **4 GiB** ⇒ **省 5.6×** |

**地板为什么不能用 0**(我上一版是错的):N 个 turn 若都"无条件准入",就会互抢池 ⇒ 抖动。
缺省自动填 = **本 turn 工作集字节**,与池下限算式**同源**(同一套页数,乘字节而非并发数);
显式 `[kvmem].dense_floor_mib` 则尊重显式值(`dense_floor_explicit` 标记)。

**校验四层**:

| 层 | 判据 | 时机 | 我的实测 |
|---|---|---|---|
| boot | 池 ≥ N_active × 每 turn 页 | 拒启 | 池 64 页 + 2 并发 ⇒ **正确拒启**(报数 92 页) |
| boot | 池 < ctx 仅虚拟化档 | 拒启 | ✓ |
| boot | 非 Strict 地板需虚拟化档 | 拒启 | ✓(双向 e2e 门) |
| 准入 | committed + 本 turn 地板 ≤ 准入面(store 四态) | 每 turn | ✓ |
| 运行 | slot 不足 ⇒ `Exhausted` 响亮拒 | 每 chunk | ✓ |

**(4)剩下那 1.2s 全是可优化的回读(下一步的靶面)**

partial/chunk256 的相位账:`索引 515ms + 查询面 804ms(其中**设备往返 742ms**)+ 调和 125 + 打分 34`
⇒ **回读往返 = 1260ms 是全部开销**。两刀已定位:

| 刀 | 做法 | 预期 |
|---|---|---|
| **Q 面只在完整召回边界采** | 廉价档根本不需要 query(不打分)⇒ 4 次里省 3 次回读 | −550ms |
| **索引 GPU 累加** | 参考实现 `kvmem_meank_add` 的做法:设备上累加、块满才 flush | −450ms |

⇒ 两刀后 partial ≈ **7.0s**,与 bare 6.8s 齐平(即**超卖档不再有可测代价**)。

### 8.27 **配置面统一**(2026-10-19):单一来源 + 三个 flavor + 每次跑自证条件

> **病因**(用户点出,成立):此前**条件散在多个入口** —— B 轴的关键开关
> (`index_probe`/`coverage_probe`/`select_decode`)**只在 `tests.rs` 里读 env**,
> 根本不在配置面上;其余是 `env > config` 覆盖;探针档自己拼 `EngineKnobs`;
> 每个数字都换条件 ⇒ **数字对不上**。

**(1)配置成为唯一来源**

| 件 | 之前 | 现在 |
|---|---|---|
| `[kvmem]` | 只有 7 项(`select_*`/`index`/`recall_every`/`dense_floor_mib`/`pool_overcommit`) | **11 项**:补 `coverage_probe` / `evict_policy` / `analyze_at_ctx` |
| `[pool]` | 无宿主两账(靠 `OWL_HOST_CACHE_MB`/`OWL_SNAP_HOST_MB`) | 补 `host_cache_mb` / `snap_host_mb` |
| `EngineKnobs::from_config` | `env > config > 缺省` | **只读配置,零 env** |
| 探针档 | 自己拼 knobs(散落 `OWL_SELECT_*`/`OWL_COV_*`) | **走同一条 `OwlConfig::load`**;`OWL_PROBE_FLAVOR` + `OWL_PROBE_OVERLAY` |
| 已废 env 读取器 | `parse_evict_policy` / `recall_every_from_env` | **删** |

**(2)每次跑自证条件**:`EngineKnobs::digest()` 单行摘要(boot 与探针都打)

```text
[probe-conf] flavor=awq27b-kvmem overlay=- ⇒
  pool=Some(3072) turns=1 snap_max=4 host=3072M snap_host=1024M |
  kvmem=ON[idx=1 cov=0 view=1 part=1 budget=32+2+4 every=4 evict=lru floor=Bytes(0)] |
  fi=1 split=1 gdn_chunked=1 kv=fp8 graph_off=0
```
凡影响行为的旋钮都在这行里;**不在这行里的就不影响行为**。

**(3)flavor 收敛:5 → 3**(按引用计数定,不是按口味)

| flavor | 引用 | 处置 |
|---|---:|---|
| `awq27b` | 15 处 | **留**(27B dense 生产档;**`[kvmem]` 显式全关** = 高显存档) |
| `spec-dflash` | 21 处 | **留**(27B fp8 + 投机 生产档;测试/testdata/文档都在用) |
| `awq27b-kvmem` | 新增 | **留**(低显存档;重写为自洽配方,见下) |
| `dev-08b` | 0 → 探针基档 | **留**(0.8B 冒烟 + 0.8B 探针基档) |
| `eager-debug` / `fi-kv8` | **0 处** | **删** |

**(4)数字必须自洽 —— boot 硬校验就是"统一"的执行者**

`awq27b-kvmem.toml` 现在把三层账写在一个算式里:
```text
池页数 ≥ max_active_turns × (select_blocks + sink + recent + ceil(chunk/page))
96 ≥ 1 × (32 + 2 + 4 + 8) = 46 ✓         页 = 1 MiB(fp8)/2 MiB(f16)
```
改数只动三行(`max_seq` / `pool_tokens` / `select_blocks`);不自洽 **boot 直接拒启并报数**。
`dev-08b` 同样(192 ≥ 2×86 ✓)。实测拦下过两次真实不自洽(池 64 页 vs 2 并发 × 46 页;
池 16 页 vs 8 并发 × 11 页)—— 这正是它该干的事。

**(5)两处口径修正(读数可比)**

- **`[kvmem].analyze_at_ctx`**(新):**钉住分析边界**。边界只在"chunk 末且页对齐"
  处发生 ⇒ "最后一个边界"的位置**随 chunk 漂移**(同一 prompt:1024→3072 / 512→3584 /
  256→3840);钉住 ctx 后 query 面 `[ctx−8, ctx)` 与块数 `ctx/page` **都与 chunk 无关**。
- **`cov_last` 只由"真跑了诊断"的边界更新**(原实现每边界覆盖 ⇒ 目标边界的报告被冲掉)。
- **保真档不开视图**:`select_decode` 开了 ⇒ 召回点之后的 prefill 只扫选中块 ⇒
  之后写下的 K **被自己扰动** ⇒ 质量读数漂。⇒ 保真读数与视图**分开量**。

**(6)平反与挂账**

- ✅ **"8K prefill 12s"定谳(§8.26 的三因)**:chunk 128 + FI 缺省关 + `prefill_split`
  契约 rank bug(既存,含 `main`)。修后同协议对照:bare 6.8s vs partial 8.0s(+18%)。
- ⚠️ **挂账(未定谳)**:同一 prompt、同一 `analyze_at_ctx=3072`、同一批块,
  **只改 chunk** 仍改读数:`1024→87.5%/88.7%` · `512→18.5%/24.7%` · `256→16.4%/22.4%`。
  已排除:分析边界(已钉)、`select_decode`(已关)、`select_partial`(已关)、预算/节奏。
  ⇒ **跨 chunk 的读数一律不可比**;`dev-08b` 的 chunk 已**冻结 1024** 并写死注释。
- ⚠️ **廉价边界的驻留调和**:两版都试过 —— ①"近期选择 + 调和"= 名单互相覆盖(净亏);
  ②"什么都不做"= `ensure_for_len` **驱逐不了会话自己持有的页** ⇒ 链长不动 ⇒ 池耗尽
  (实测池 96 页、链要 104 页)。现版 = **冻结上次选择 + sink + 新尾部**,仍会在
  2.7× 超卖 + chunk 256 下撞"表长 > 池"的墙 ⇒ **下一刀:链增长与调和的时序**
  (增长前先腾/或把"未入会话引用的页"纳入同步驱逐面)。

### 8.28 **"剩下 4 GB 到哪了"定谳** + 池容量的正确算式(2026-10-19)

> **用户判据成立**:27B 加载后占 ~19.2 GiB,卡 24 GiB,余下 ~4 GiB —— 按 fp8
> **32 KiB/tok** 算确实该有 ~100K tok 的 KV 空间。而我把池手写成 96 MiB(过小)/
> 8 GiB(过大)两头错,**根因是没人算这个数**。

**(1)一行数字的真相:`total_set` 其实调了,治理器是活的**

`[vram-gov]` 打出治理器的三输入一产出(本轮新增,取代"猜"):

```text
[vram-gov] total=23.60G target=97% reserve=128M fixed_after=1.12G live=17.94G
           per_tok=32768B ⇒ budget=3.71G **cap=121446 tok**(manual=32768)
```

| 项 | 27B fp8 实数 |
|---|---|
| `per_tok_bytes` | 16 层 × 2(K+V) × 4 头 × 256 维 × **1B** = **32 KiB/tok**(f16 翻倍) |
| 治理器 cap | **121446 tok = 3.71 GiB**(= 用户说的"~100K") |
| 池页宽 | 32 tok ⇒ 1 MiB/页(fp8) |

⇒ **算式在代码里(`StatePool::alloc` 的 B6.5 治理),不在配置里**;
配置只该给**安全阀**(`[pool].pool_tokens`),不该手写猜数。已改:`awq27b-kvmem`
留一行注释 + 一个安全阀值。

**(2)"4 GB"的真实分解(27B / 24 GiB / fp8 / chunk 256 实测)**

```text
23.60 GiB 卡总(= total)
 19.2     权重 + 运行底价(装载后实称 live)
  1.1     fixed_after:GDN 行 + 快照池 + 图 slab + CUDA ctx
  3.4     **prefill 中间件:每 chunk ≈128 MiB,27 个 chunk 一路涨,不回收** ← 真凶
  0.1     图捕获 slab(实测 29 MiB/图)
  1.0     KV 池(32768 tok = 1024 页)
```
⇒ 到第 27 个 chunk:`free = 110 MiB`,守卫(`free − reserve = 0`)拒发 3 MiB。
**这不是配置错,是运行期的空间累积**(eager 路每 chunk 的中间件没有回收面)。

**(3)新增三行取证(常驻,供以后不再猜)**

| 行 | 位置 | 作用 |
|---|---|---|
| `[vram-gov]` | `StatePool::alloc` | 治理器的 total/target/reserve/fixed_after/live/per_tok ⇒ cap(唯一产地) |
| `[vram-guard]` | `vram_budget_guard` 失败路径 | 要多少 / free / reserve / avail / live 账 |
| `[vram-big]`(env `OWL_VRAM_DBG`) | server alloc ≥64 MiB | 大额分配归因(实测只有装载期的两笔 2425 MiB) |

**(4)结论与下一步**

- **池容量的正确姿势**:`pool_tokens` = 安全阀(显式),上限由治理器算;
  它必须 ≥ 工作集下限算式(`N_active × (blocks+sink+recent+chunk/page)`)。
- **低显存档在 27B/24G 的可达规模**:扣掉 prefill 中间件的 3.4 GiB 后,
  长期可用池 ≈ **1 GiB(32K tok)** ⇒ 配 `max_seq = 131072` = **4× 超卖**。
  要"池吃满 ~100K",必须先解决第 (2) 条的中间件累积。
- **下一刀 = prefill 中间件回收面**(每 chunk 128 MiB × N chunk):查 eager 路的
  块释放链(`free_blocks` / arena carve)为何不把中间件还给分配器。

### 8.29 **超卖实现**(2026-10-19):4× 超卖跑通 + A0/A1 两条挂账定谳

> **一句话**:27B / fp8 / **ctx 131072** / **设备池 32768 tok(1 GiB)= 4× 超卖** ——
> 下沉 242 块 · 回灌 58 块 · **不可沉 0** · 子集表 49 次 · 视图步 23 回退 0 · 输出切题。
> 两个"阻塞点"其实都是**配置面缺一条约束**,不是算法问题。

**(1)A0 定谳:"剩下 4 GB 到哪了" —— 不是泄漏,是 prefill 工作区 ∝ chunk**

逐 chunk 显存轨迹(`OWL_VRAM_DBG=1`,27B / ctx 131072 / 池 1 GiB):

| chunk | live 账 | 驱动 free | ⇒ **工作区** |
|---:|---:|---:|---:|
| 64 | 20188 MiB(**恒定**) | 2510 MiB(**恒定**) | **1065 MiB** |
| 128 | 20188(恒定) | 1454(恒定) | **2121 MiB** |
| 256 | 20188(恒定) | 110 ⇒ 守卫拒发 | **3465 MiB** |

⇒ **每 chunk 零净增**(与代码注释 "E2a:中间块归池(每 chunk 零净增)" 相符),
`live` 一动不动 —— **我先前的"每 chunk 漏 128 MiB"是误判,收回**。
真相是 **单 chunk 的激活面 ∝ chunk**(≈16.5 MiB/token),它是**峰值**不是累积。

**"那 4 GB"的完整分解(shipped 形态)**:

```text
23.60 GiB 卡总 = 19.2 权重/运行底价 + 1.1 fixed_after(GDN/快照/图/CUDA ctx)
               + W(chunk 工作区:64→1.07G / 128→2.12G / 256→3.47G)
               + 池(1.0 GiB)+ 守卫余量(128 MiB)
```
⇒ **池容量与 chunk 直接争显存**:装配后 free 3575 MiB 里,chunk 256 时
`3575 − 3465 − 128 < 0` ⇒ 第 27 个 chunk 起分配必被守卫拒发(实测)。
本档取 **chunk 128**(工作区 2.12 GiB,运行余量 ~1.4 GiB)。

**(2)A1 定谳:`prefix_cache` 关着 ⇒ 缓存树 cap=0 ⇒ 块结构性不可沉**

`session.rs`:
```rust
let cache_cap = if plan.paged && plan.prefix_cache { (plan.nb/2 + host_slots).max(1) } else { 0 };
```
而下沉机制 = **缓存树 + 会话双持(refs==2) → decref → 缓存独占 → 下沉**
(`reconcile_residency` 的可沉性前置)。⇒ `prefix_cache = false` 时:
每 chunk 送 4 片 → **入树 0**(`insert_inc` 在 `!enabled()` 处直接返空)→
块永远 `refs==1` → **一块都沉不下去**(实测:下沉 0 / 不可沉 4208)。

`awq27b` 系的 `prefix_cache = false` 是"24G 贴顶省 300MB"的**旧**取舍 —— 与
虚拟化档**结构性冲突**。⇒ 低显存档必须开它;并在 boot 加**响亮拒**
(`select_partial = true` 要求 `prefix_cache = true`),不许静默白跑一趟。

**(3)取证面沉淀(三轮 debug 不白做)**

| 行/口 | 位置 | 用途 |
|---|---|---|
| `[vram-gov]` | `StatePool::alloc` | 治理器五输入→cap(**池容量唯一产地**:27B fp8 实测 cap=121446 tok) |
| `[vram-guard]` + 池账 | `vram_budget_guard` | 要多少/free/reserve/**池内可复用**/live |
| `[vram-c]` | `execute_prefill` 尾(`OWL_VRAM_DBG`) | **逐 chunk 轨迹** —— "累积 vs 峰值"的唯一判别器 |
| `[vram-big]` / `[graph-slab]` | server alloc / 图捕获 | 大额归因 |
| `[recon]` refs 直方图 | `reconcile_residency` | **refs(1/2/3+)** —— "能不能沉"的直接读数 |
| `[chain-inc]` | `capture_gdn_snap` | 送 N 片 / **入树 M** / 被逐 K —— 断链处 |
| `OWL_VRAM_DUMP` + 块账 dump | `dump_vram_on_oom`(守卫失败也调) | **块名级**归因(`owl_*` 中间件 / `gdn.*` 工作区 / `wt.*` 权重) |

**(4)当前能力与代价(27B / 24G / fp8)**

| 项 | 值 |
|---|---|
| 上下文 | **131072 tok(128K)** |
| 设备 KV 池 | 32768 tok = 1024 页 × 1 MiB = **1 GiB** |
| 超卖 | **4×** |
| 宿主 KV 驻留 | 0.18 GiB(下沉 242 块) |
| 装配后驱动空闲 | 3575 MiB(容得下 chunk 128 的 2.12 GiB 工作区) |
| B 轴生产面 | 索引 1941ms + 查询面 150 + 打分 63 + 调和 93 ≈ **2.2s / 14.9s(15%)** ⇒ A3 的靶面 |
