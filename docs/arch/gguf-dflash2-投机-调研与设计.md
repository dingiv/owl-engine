# GGUF 线 DFlash2 投机采样 —— 调研与设计（2026-10-20）

> 目标：让 **GGUF 主干**吃 **GGUF 草稿**（`z-lab/Qwen3.8-27B-DFlash2-gguf`），
> 与现役 AWQ/HF 主干 + HF 草稿的 DFlash2 生产路**共用同一套引擎机制**。
> 本文只做调研与设计，不含实现。

---

## 0. 结论摘要（TL;DR）

- **引擎侧不需要新机制**：DFlash2 的草稿池 / 8 行 propose / 码本选择器 /
  verify / 降级窗口**全部已生产**（AWQ 主干 + HF 草稿已在跑）。
- **缺的全在“装载面”**：`formats/gguf/meta.rs` 现在**响亮拒** `arch=dflash`
  的 GGUF（`rejects_other_arch` 门），且没有“草稿 GGUF → `DFlash2Draft`”的
  键约定 / 逐张量量化计划 / 形状（rank-3）通路。
- **产物齐备**：`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（1.14 GB）与 `-Q8_0.gguf`
  （2.06 GB）都是 **81 张量、Q4_K/Q6_K/F32 全在已移植集**，且 `dflash.*`
  KV 把家族常数全部自带。
- **一个必须先钉死的坑**：GGUF 的 `dflash.target_layers = [6,20,34,48,62]`
  比 HF 的 `target_layer_ids = [5,19,33,47,61]` **大 1** ⇒ GGUF 侧是
  **1-based**。引擎现在在 **5 处**硬编码 0-based `[5,19,33,47,61]`；
  照抄 GGUF 值会使 taps 整体错一层 —— **AL 静默塌陷**（不报错、不崩）。
- 工作量估计：**2–3 天**（P0 装载门 / P1 数值门 / P2 引擎接线 / P3 e2e+调优）。

---

## 1. 现状盘点

| 面 | 已有 | 缺 |
|---|---|---|
| 草稿架构 | `layers/dflash2.rs`（5 层 backbone + fc 五 tap 投影 + 码本 selector，sglang 逐式同构） | — |
| 草稿装载 | `qwen35::dflash2_draft(dir)` / `load_dflash2_draft`（HF safetensors，BF16 执行域） | **GGUF 源** |
| 草稿 meta | `ckpt_config`（`architectures = DFlash2DraftModel`） | **`dflash.*` GGUF KV → DraftSpec** |
| spec 运行时 | `engine/spec_runtime.rs`：mode 裁决（DFlash2/MTP/Dumb/Off）、草稿池、草稿 rope、taps=5 | **草稿源枚举**（目录 \| GGUF） |
| taps | `tapped_hidden(..., &[5,19,33,47,61])` × **5 处**（`prefill_plan.rs`、`graph_fleet.rs`×2、`prefill_chunk.rs`×2） | **由草稿声明驱动**（禁硬编码） |
| 配置 | `model.dflash2_dir`、`[spec] depth/draft_kv_fp8/notaps/tapdecl` | **`model.dflash2_gguf`** |
| 门 | AL 实测口径、4-bit 草稿先例（syvai W4A16：AL 2.86 / 5.42 vs BF16 3.10 / 4.96） | GGUF 草稿的 G0–G3 门 |

**关键结论：4-bit 草稿不是瓶颈**（上面 W4A16 与 BF16 同量级）⇒ Q4_K 草稿
在数值上**预期可用**，但必须用门证明，不能靠推断。

---

## 2. 产物事实（本机实测）

### 2.1 `dflash.*` KV（Q4_K_M 文件；Q8_0 同）

| KV | 值 | 用途 |
|---|---|---|
| `general.architecture` | `dflash` | 现在被 `meta.rs` 拒的判据 |
| `dflash.block_count` | 5 | 草稿层数 |
| `dflash.attention.head_count / head_count_kv` | 32 / 8 | GQA（hd 128） |
| `dflash.attention.key_length / value_length` | 128 / 128 | `DraftGeom.hd` |
| `dflash.attention.causal` | **false** | 非因果块（ENCODER_ONLY） |
| `dflash.attention.sliding_window` | 2048（pattern 全 true） | 草稿注意力窗 |
| `dflash.block_size` | 8 | 1 锚 + 7 草稿 |
| `dflash.conv_kernel_size / conv_group_size` | 2 / 16 | GroupedConv（= 层内常量 TAPS/GROUP） |
| `dflash.selector_rank / selector_top_k` | 256 / 16 | 码本选择器（= RANK/TOP_K） |
| `dflash.embedding_length / feed_forward_length` | 5120 / 17408 | 与 27B 主干一致 |
| `dflash.rope.freq_base / dimension_sections` | 1e7 / `[64,0,0,0]` | θ 与主干同；sections 是文本 mrope 声明 |
| `dflash.target_layers` | **[6,20,34,48,62]** | ⚠️ **1-based**（见 §4-D1） |

**提取策略**：`DraftSpec` 逐字段来自 `dflash.*`，**不与 27B 家族常量做等值绑定**
（家族常量只作“已知家族”的交叉校验，见 G0）。

### 2.2 81 张量（Q4_K 45 / Q6_K 4 / F32 32）

| GGUF 键 | ggml 形状（ne，内维在前） | owl 本地键 | torch 形状 | 备注 |
|---|---|---|---|---|
| `fc.weight` | [25600, 5120] | `fc.weight` | [5120, 25600] | 零拷贝转置消费（同 HF） |
| `enc.output_norm.weight` | [5120] | `hidden_norm.weight` | [5120] | 名字不同 |
| `output_norm.weight` | [5120] | `norm.weight` | [5120] | 名字不同 |
| `selector_hidden.weight` | [5120, 256] | `candidate_selector.hidden_projection.weight` | [256, 5120] | |
| `selector_predecessor.weight` | [256, 248320] | `candidate_selector.predecessor_codebook` | [248320, 256] | **码本**（非 GEMM） |
| `selector_successor.weight` | [256, 248320] | `candidate_selector.successor_codebook` | [248320, 256] | 同上 |
| `blk.N.attn_norm.weight` | [5120] | `layers.N.input_layernorm.weight` | [5120] | |
| `blk.N.ffn_norm.weight` | [5120] | `layers.N.post_attention_layernorm.weight` | [5120] | |
| `blk.N.attn_q.weight` | [5120, 4096] | `layers.N.self_attn.q_proj.weight` | [4096, 5120] | |
| `blk.N.attn_k / attn_v.weight` | [5120, 1024] | `...k_proj / v_proj` | [1024, 5120] | |
| `blk.N.attn_output.weight` | [4096, 5120] | `...o_proj.weight` | [5120, 4096] | |
| `blk.N.attn_q_norm / attn_k_norm.weight` | [128] | `...q_norm / k_norm.weight` | [128] | |
| `blk.N.attn_conv_base` | **[5120, 2, 2]** | `layers.N.attention_conv.base_kernel` | **[2, 2, 5120]** | rank-3；ggml ne 反转 ⇒ 扁平序相同 |
| `blk.N.attn_conv_proj.weight` | [5120, 1280] | `layers.N.attention_conv.kernel_projection.weight` | [1280, 5120] | |
| `blk.N.ffn_gate / ffn_up.weight` | [5120, 17408] | `layers.N.mlp.{gate,up}_proj.weight` | [17408, 5120] | |
| `blk.N.ffn_down.weight` | [17408, 5120] | `layers.N.mlp.down_proj.weight` | [5120, 17408] | |
| `blk.N.ffn_conv_base` | [5120, 2, 2] | `layers.N.mlp_conv.base_kernel` | [2, 2, 5120] | 同上 |
| `blk.N.ffn_conv_proj.weight` | [5120, 1280] | `layers.N.mlp_conv.kernel_projection.weight` | [1280, 5120] | |

**四个命名特例**：`enc.output_norm→hidden_norm`、`output_norm→norm`、
`selector_*→candidate_selector.*_codebook`、`attn_conv_*→attention_conv.{base_kernel,kernel_projection}`。
其余 76 键是机械映射（`blk.N.x` ↔ `layers.N.y`）。

**转换器没有做任何布局变换**：草稿只有全注意力层（无 GDN / 无 v-head tiled /
无 `ssm_*`）⇒ GGUF 线**不需要** `transform::xf_for` 那套逆变换；唯一要处理的是
**rank-3 的 conv base**（`ne=[5120,2,2]` ↔ torch `[2,2,5120]`，扁平字节一致，
只需源按 torch 序报 shape）。

---

## 3. 执行路径事实（决定“能不能直接复用”）

| 事实 | 值 | 影响 |
|---|---|---|
| 草稿执行 dtype | **BF16**（单源 `draft_exec_dt`；F16 在 propose 残差流溢出，2026-10-19 定谳） | GGUF 权重 + **BF16 激活** |
| GGUF Linear 臂 | MMVQ 与复用臂吃 **q8_1**（`GgufQuantActBf16` 可 ⇒ BF16 激活可用）；**cuBLAS/range 与 smem 臂要求 F16 激活** | 草稿 propose（m=8）走 MMVQ/tile 都行 ⇒ **够用**；不需要为草稿加 BF16 prefill 臂 |
| 草稿 propose 形状 | 8 行（1 锚 + 7 草稿）× 5120 | 每次 verify 一发；权重读一遍（1.1 GB @ 4bit） |
| 选择器输入 | **target lm_head** 的 top-16（`ensure_dt(hidden, F16)` 后走 target 的 GGUF lm_head） | 与草稿量化无关；主干任何档都可用 |
| 草稿池 | 独立池，BF16（`draft_kv_fp8` 才 e4m3），sliding window 2048 | 与现役 DFlash2 相同，无新增 |

---

## 4. 设计决策

### D1 — 草稿 meta：`draft_spec_from_gguf`（含 **1-based 归一**）

新增 `formats/gguf/draft_meta.rs`：

```text
DraftSpec {
  layers: 5, hidden: 5120, inter: 17408, vocab: <target 词表>,   // 无 embed/lm_head（共享 target）
  hq: 32, hkv: 8, hd: 128,
  block: 8, taps: 2, group: 16, depth: 7,
  rank: 256, top_k: 16,
  mask_token: 248070,                        // ⚠️ GGUF 未导出！见下
  tap_ids: Vec<usize>,                       // 0-based
  rope: (max_pos 262144, dim 128, theta 1e7),
  sliding_window: Some(2048),
}
```

- **`target_layers` 归一**：GGUF 值 `[6,20,34,48,62]` 按 **1-based** 解释
  （`-1` ⇒ `[5,19,33,47,61]`，与 HF/引擎现役完全一致）。**依据**：
  该 GGUF 由 HF 检查点转换而来，HF 声明是 0-based `[5,19,33,47,61]`；
  转换器做 `+1`（ggml 侧层号习惯），这是唯一自洽解释。
- **门 G0-b**（防再次踩）：若 `tap_ids` 与 27B 家族常量不一致 ⇒ **拒启**并
  打印两边的值（这层错是 AL 静默塌陷，必须硬门）。
- **`mask_token_id` 缺 KV**：GGUF 没导出它（HF config 有 248070）。两个选项：
  ①按家族常量兜底 + **响亮点名**（“GGUF 未声明 mask token，取 27B 家族常量 248070”）；
  ②从 target 词表按 `"<|MASK|>"` 查（若词表含）。v1 用 ① + 记录，等转换器补 KV。
- `arch=dflash` 的**响亮拒**改成“进草稿分支”（`spec_from_gguf` 仍拒 —— 草稿
  文件绝不走主干路径，两条入口各自点名）。

### D2 — 键约定：`GgufDraftConvention`（81 键，闭区间校验）

- 复用 `KeyConvention` 的 `map_keys` 机制，新增草稿实现（`blk.N.*` ↔ `layers.N.*`），
  四个特例见 §2.2。
- **覆盖面门**：装载后 `unresolved` 必须为空，且“文件里没被问过”的键也必须为空
  （81/81 双向），否则拒启（仿 `GgufPlan::take_unresolved`）。
- 形状门：rank-2 用 `ne` 反转后的 `[out, in]`；rank-3（conv base）报 **torch 序**
  `[2,2,5120]`（= `reverse(ne)`），并逐键比对 `Weight` 声明的 shape。

### D3 — 源与装载：`GgufDraftSource`（不是 `GgufSource`）

`GgufSource::open` 现在会 `spec_from_gguf`（qwen35 主干），草稿必须走**独立源**：

```rust
pub struct GgufDraftSource { g: Gguf, dims: GgufDims }   // 只做「键 → 字节」
impl WeightSource for GgufDraftSource {
    fn take(&mut self, key, off, len) -> Taken<WeightBlob>;  // 量化块字节原样；原生(F32)元素原样
    fn facts(&self) -> SourceFacts;                          // norm_baked_plus_one=false(草稿 norm 未 +1)
}
```

- **零解量化驻留**（同主干纪律 I-3）：Q4_K/Q6_K 字节原样上卡，槽 = `U8`；
  F32（norm/conv base）走元素路径。
- rank-3 支持是**唯一需要扩的点**（当前源按 `rows × inner` 报形状）。

### D4 — 逐张量量化计划：复用 `GgufPlan` + `QuantPlan::Gguf(ty)`

- 草稿是**混型**（Q4_K 45 / Q6_K 4 / F32 32）⇒ 必须逐键给计划，不能是单一
  `QuantPlan`。现成机制：`PlanTable`/`GgufPlan`（主干已在用）。
- `DFlash2Draft::from_spec(spec, plan, dt, kv_fp8)` 的 `plan` 参数升级为
  `PlanSource`（`QuantPlan` 只是它的 trivial 实现）—— 与主干构造同形；
  `from_spec` 的既有调用点（HF 档）零改动。
- **未移植型 = 构造期拒**（同主干）；Q8_0 草稿（2.06 GB）同一条路。

### D5 — 执行 dtype：**BF16 不变**，MMVQ-only

- 草稿仍恒 BF16（数值域要求，不因量化而改）。
- 权重 GGUF 块 + 激活 BF16 ⇒ Linear 走 **MMVQ**（m<8）或 **tile 复用臂**（m≥8，吃 q8_1、与 MMVQ 位级相同）。
  propose 是 8 行的小 GEMM，这两条臂就是正路；**不为草稿加 BF16 cuBLAS/smem 臂**
  （那是 27B 主干的 prefill 优化，草稿用不上）。
- cast 边界不变：`ensure_dt(hidden, F16)` 后才进 target lm_head。

### D6 — selector 码本：v1 **装载期一次反量化为 BF16**

- 两张码本 `[248320, 256]`，Q4_K 各 ≈ **71 MB**（合计 143 MB）；选择器核
  （`owl_topk16_f16` + `owl_selector_walk`）**只认 f16/bf16 行**。
- v1：装载时用 `dequant_row(Q4_K)`（已与 gguf-py 位级对拍）把两张码本反量化成
  **BF16 常驻**（2 × 127 MB = **254 MB**）；代价可接受（见预算），零核改动。
- v2（挂账）：给 selector walk 加 GGUF-gather 变体（省 254 MB，但多一条内核腿）。
- **预算**（单 3090 Ti 24G，27B GGUF 主干 + 草稿）：
  `13.3 GiB 主干 + 1.07 GiB 草稿 + 0.25 GiB 码本 + 0.75 GiB KV 池(12k)
  + 0.6 GiB slab + 1.2 GiB GDN 行 ≈ 17.2 GiB` ✓（余 ~6 GiB）。

### D7 — 引擎接线：草稿源枚举 + taps 由声明驱动

```text
config:  model.dflash2_dir  (HF, 现有)
         model.dflash2_gguf (新; 二选一，同时给 = 拒启)

spec_runtime::SpecBootInput {
    draft_src: Option<DraftSrc>,       // enum { Dir(PathBuf), Gguf(PathBuf) }
    ...
}
mode 裁决:  depth>0 && draft_src.is_some() → DFlash2
装载:       Dir  → dflash2_draft(dir)            （现状）
            Gguf → dflash2_draft_gguf(path)      （新：D2+D3+D4）
几何:       Dir  → dflash2_draft_geom(dir, tp)   （现状）
            Gguf → dflash2_draft_geom_gguf(path, tp)
taps:       spec.tap_ids（来自草稿声明）—— 替换 4 处硬编码 [5,19,33,47,61]
```

- **taps 单源化是硬要求**：现在 **5 处** `&[5,19,33,47,61]` 硬编码（`prefill_plan.rs`、
  `graph_fleet.rs`×2、`prefill_chunk.rs`×2）⇒ 改为 `boot.spec.tap_ids`；
  HF 档由 `dflash2_draft` 一并给出同一组值（逐字节不变）。
- 草稿池 / rope / verify / 降级窗口 / accept：**零改动**（同一机制）。
- TP：v1 只支持单卡草稿（`dflash2_dir` 在 TP 档已有 `tp_draft` 闸门；GGUF 草稿
  先挂“TP 档响亮拒”，P4 再谈分片）。

---

## 5. 施工分解（P0–P4，每步一个门）

| 步 | 内容 | 门（先跑独立最小用例） | 估 |
|---|---|---|---|
| **P0** | `draft_meta.rs`（`dflash.*` → DraftSpec，1-based 归一）+ `GgufDraftConvention` + `GgufDraftSource`（含 rank-3）+ 把 `arch=dflash` 从“拒”改成“进草稿分支” | **G0-a** 81/81 键双向覆盖、形状逐键相等、unresolved 空；**G0-b** `tap_ids == [5,19,33,47,61]`（否则拒启并打印）；**G0-c** 显存账（≤1.3 GiB 驻留） | 0.5–1 d |
| **P1** | 逐张量 `GgufPlan` 接到草稿构造；BF16+MMVQ 路径；码本反量化（D6） | **G1** 同一组 taps 下，`fc` / layer0 / final hidden 与 HF 草稿参考 **cos ≥ 0.999**、max|Δ| ≤ 2e-2；**G1-b** selector top-16 候选集合与 HF 草稿重合 ≥ 99% | 0.5–1 d |
| **P2** | 引擎接线（配置键、mode 裁决、几何、taps 单源化） | **G2** propose 对拍：同一 anchor/hidden 下 7 草稿 **top-1 一致率 ≥ 95%**（vs HF W4A16 草稿）；**G2-b** 关 spec 路径逐字节回归 | 0.5 d |
| **P3** | 27B GGUF 主干 + GGUF 草稿 e2e；AL 与吞吐 | **G3** AL ≥ HF W4A16 草稿 × 0.9（先例 2.86/5.42）；decode tok/s ≥ 无 spec × 1.3；首 token 正确性门不回归 | 0.5 d |
| **P4**（挂账） | TP2 草稿分片；码本 GGUF-gather（省 254 MB）；Q8_0 草稿档对比 | — | — |

**回退姿势**：`spec.depth = 0` 或删 `dflash2_gguf` ⇒ 回到现役无 spec 路径；
GGUF 草稿装载失败必须**拒启**（不静默降级成无 spec，避免“以为在投机其实没投”）。

---

## 6. 风险 / 未决

1. **taps 1-based 误读**（最高危）：错一层 = AL 静默塌陷。G0-b 硬门 + G1
   逐层 cos 门双保险。
2. **mask_token 缺 KV**：GGUF 未导出 `mask_token_id`（HF 有 248070）。v1 家族常量
   兜底 + 响亮点名；长期等转换器补 `dflash.mask_token_id`。
3. **码本量化噪声**：selector 的 predecessor/successor 是 63.6M×2 参数的大表，
   Q4_K 的逐块量化会直接影响候选排序 —— G1-b/G2 必须看“候选集合重合”，
   不能只看最终 AL（AL 是二阶量）。
4. **`sliding_window_pattern` 全 true**：现役 HF 草稿同一配置，行为一致；
   若引擎的窗口语义与 sglang 有差，两档会同时偏（不是 GGUF 新问题，记录在案）。
5. **Q4_K 草稿 vs Q8_0 草稿**：若 G3 的 AL 掉超过 10%，用 2.06 GB 的 Q8_0 档
   （同一条装载路，只换文件）。
6. **GGUF 转换器版本**：`target_layers` 的 `+1` 是否在所有 z-lab GGUF 里一致？
   G0-b 会在每个文件上验证（不假设）。

---

## 7. 明确不做（本轮）

- **不做** GGUF 草稿的 TP2 分片（P4）。
- **不做** 码本 GGUF-gather 内核（v1 装载期反量化，254 MB 换零核改动）。
- **不做** DSpark/其他 draft 家族的 GGUF 线（`DimInfer/…Dspark…` 另案）。
- **不改** verify / accept / 降级窗口语义（复用现役，零风险）。

---

## 8. 施工记与实测(P0–P3,2026-10-20 当日结案)

### 8.1 落了什么

| 刀 | 文件 | 内容 |
|---|---|---|
| P0-a | `formats/gguf/draft_meta.rs`(新) | `dflash.*` KV → `DraftSpec`;**1-based 归一** + 家族对账(G0-b)+ 跨文件一致性(hidden/taps 越界/block/rank/conv)+ `mask_token` 兜底点 |
| P0-b | `formats/gguf/draft.rs`(新) | `DraftGgufConvention`(计划键)+ `DraftGgufPlan`(PlanTable)+ `GgufDraftSource`(译表 + fc tap 重排 + 码本反量化)+ `GgufDraftBoot::{open,draft,load}`(装载 + G0-a 覆盖门) |
| P1 | `layers/dflash2.rs` | `DraftFc::G`(五拆 GGUF 臂)+ `from_dims_keyed`(逐张量计划)+ `DfAttn/DfLayer` 改吃 `PlanSource` + conv 投影走 `PlanSource::or_default` |
| — | `layers/linear.rs` / `mlp.rs` | `Linear::with_out_dtype`(输出 dtype 钉定)+ **GGUF 动态范围护栏**(见 8.3) |
| — | `kernels/cu/elems/scale.cu`(新) | `owl_scale_{f16,bf16}`(逐元素乘常数;成对指数移位的执行体) |
| P2 | `engine/spec_runtime.rs` / `engine.rs` / 4 个调用点 | 草稿源按**路径形态**判别(目录=HF / 文件=GGUF)+ `dflash_tap_ids`/`dflash_mask_token`/`dflash_rope` **单源化** + 5 处硬编码 taps 全替 |
| P3 | `profiles/qwen38_27b-3090-iq4xs-dflash.toml`(新) | 27B UD-IQ4_XS + GGUF 草稿 d7 生产靶位 |

**实证到的三条事实(不再靠推断)**
1. **草稿 norm 未烘焙 `+1`**:GGUF 与 HF BF16 检查点逐元素 **max|Δ| = 0**(5 类 norm + rank-3 conv base 全 0)⇒ 草稿源事实恒 `SourceFacts::HF`,且 `DFlash2Draft` 的 `RmsNorm::new`(plain ×w)两族通用;
2. **`target_layers` 确为 1-based**:`[6,20,34,48,62] - 1 = [5,19,33,47,61]` 与 HF 逐值相等;
3. **`mask_token_id` 确实不在 GGUF**(48 个 KV 里没有)⇒ v1 家族兜底 + `log_warn` 点名。

### 8.2 计划面:混型不只是"注意力/MLP"

GGUF 草稿的量化件**比设计预期多两个**(HF 检查点里它们是原生 BF16):
- `attn_conv_proj.weight` / `ffn_conv_proj.weight`(Q4_K)!⇒ `GroupedConv::proj` 必须吃计划;
- `selector_hidden.weight`(Q4_K)⇒ `CandidateSelector::proj` 必须吃计划。

两者在 HF 档必须**保持 F16**(改它 = 静默改变 W4A16/BF16 草稿数值)⇒ 引入
`PlanSource::or_default(local, out, in, default)`:**Uniform 臂取 default,Keyed 臂逐键查表**。
这条是"混型源"与"单一方案源"的语义分界,后续新格式接入照抄。

`fc` 是唯一走**源侧重排**的件:`fc.weight` 的五个 tap 沿**内维**排布,量化块沿内维走
⇒ 列切跨 `out` 行,扁平字节槽无法表达列窗。源在装载期按行搬 5 份
(`fc_i.weight`,纯 memcpy;**设备驻留总量 = 一份 fc**),5 个 GEMM 的调用面与 W4A16 臂完全同形。

### 8.3 数值面:GGUF GEMM 的 **F16 输出上限**是 BF16 草稿的结构约束

GGUF 块型 GEMM 的所有臂(MMVQ/tile/MMQ/smem/cuBLAS)落点都是 `__half*`,
而草稿执行域是 **BF16**(F16 装不下残差流,2026-10-19 定谳)。实测:
Q4_K 草稿的 `mlp_out` / `fc` 可达 **6e4** —— 贴 F16 上限 65504,随机输入下直接出
`inf`(静默 NaN 塌陷)。

处置(不新增 GEMM 变体,零精度损失):
- `Linear::with_out_dtype(dt)` 让 GGUF 臂**成对移位**:`x·2⁻⁶ → GEMM → f16→bf16 → ·2⁶`;
- 2 的幂在 f16/bf16 上**精确**;相对精度仍由 F16 尾数(10-bit)决定 ⇒ 与"BF16 输出 GEMM"等价;
- 2⁶ = 64× 上限余量(1.7e7 以下不饱和);F16 次正规临界 ×2⁶ 仍远细于 BF16 自身粒度;
- 新核只有 20 行(`owl_scale_{f16,bf16}`),门 = 成对移位逐位复原(见 8.4);
- `out_dt` 缺省 `None` ⇒ 主干 GGUF 路径(F16 执行域)**逐字节不变**。

### 8.4 门(全部绿)

| 门 | 判据 | 实测 |
|---|---|---|
| **G0** 声明/计划 | `dflash.*` → DraftSpec;7 个 Linear/层 + fc + selector 全部 `QuantPlan::Gguf` | 绿(taps `[5,19,33,47,61]`) |
| **G0-a** 覆盖 | 声明请求集 ⇔ 文件张量集(双向) | **85 wants ↔ 81 张量**双向相等 |
| **G0-c** 预算 | 草稿驻留 ≤ 1.3 GiB | **1253 MiB**(文件 1079 + 码本反量化净增 173) |
| **G1-a** fc 重排 | 5 个 tap vs HF `fc` 列块(含**非行对齐窗口** ≡ 整段切片) | max rel **4.2e-2 / 3.9e-2** |
| **G1-b** 原生件 | norm / rank-3 conv base vs HF | **逐位相等(max\|Δ\| = 0)** |
| **G1-c** 码本 | 两本码本抽样行(0 / 1234 / 248319) | rel < 1e-1(量化噪声) |
| **P1/P2-GPU** | GGUF 草稿 ≡ **HF BF16 草稿**(同 taps、同 BF16 域):`project_memory` / layer0 `conv_in` / `conv_delta` / `mlp_out` / `conv_finish` | 4.9e-2 / 6.6e-2 / 6.4e-2 / 3.1e-2 / 9.6e-2(全在 Q4_K 噪声域) |
| 核门 | `owl_scale_bf16` 成对移位 | 逐位复原 |
| 全量回归 | `cargo test -p owl-models --lib` / `-p owl-engine --lib` | **291 / 76 全绿** |

### 8.5 P3 e2e(单 3090 Ti;27B UD-IQ4_XS 主干;256 tok,greedy)

| 档 | tok/s | AL(窗口) | 轮成本 |
|---|---:|---:|---:|
| 无 spec(`spec.depth=0`) | **32.4** | — | 27–30 ms/步 |
| **GGUF 草稿 d7**(本轮) | 23.5 | **2.50–3.25** | 167–174 ms |
| HF W4A16 草稿 d7(对照) | 24.6 | **2.50–3.25** | 165–170 ms |
| GGUF 草稿 d3 | 23.9 | 1.38 | 87 ms |

**判决**
1. **草稿本身完全达标**:AL 与 HF W4A16 草稿**逐窗口相同**,轮成本相同(propose 37–39 ms)⇒
   Q4_K 草稿在数值上可用,W4A16 先例(2.86/5.42)得到复现;
2. **spec 在本配置下净亏**(23.5 vs 32.4 tok/s),但**与草稿无关**——
   `spec-prof` 定谳:`verify` 的**图内 GPU 时间 = 175 ms / 8 行**(裸 decode 27 ms/1 行,
   即 m=8 的 GGUF GEMM 只有 m=1 的 ~1/10 效率)。HF 草稿在同主模型上数字一致 ⇒
   这是 **GGUF 主干在 m=8(verify 形状)上的既有性能缺口**,不是草稿线引入的;
   关 tile 臂(`OWL_GGUF_TILE=0`)无改善 ⇒ 缺口在 m=8 的形状适配层,另案。

### 8.6 发现的新缺口(挂账)

1. **GGUF 主干 m=8(verify)GEMM 效率**:175 ms/8 行 vs 27 ms/1 行。spec 线的
   吞吐门(≥ 无 spec × 1.3)要等它;候选修法 = m∈[2,32) 的专用臂(或 MMVQ 多行复用),
   与 `GGUF_TILE_MIN_M=8` 的门槛重定;
2. **`mask_token_id` KV 缺失**:需向转换器(或 z-lab)提 issue 补 `dflash.mask_token_id`;
3. P4(原设计):GGUF 草稿 TP2 分片、码本 GGUF-gather(省 254 MB)、Q8_0 草稿档对比;
4. 顺手修:上一轮 IQ 落地后 `dequant::tests::rejects_bad_geometry` 拿 IQ2_XXS 当
   "未接臂"(陈旧断言)→ 改用 IQ1_M。

---

## 9. 投机性能专项(2026-10-21 同日):m=8 的 8 倍权重流量

### 9.1 定谳:verify 慢的根因是 **MMVQ 网格 = (权重行, token)**

`nsys` + `--cuda-graph-trace=node` 拿到逐核数据后,证据链闭合:

- 27B UD-IQ4_XS 的 verify(depth=7 ⇒ 8 行)图内 GPU 时间 **175 ms**,裸 decode 27 ms;
- 逐深度扫描 **完美线性**:2 行 88 ms / 4 行 172 ms / 8 行 322 ms(≈43 ms/行);
- 单张量微基准(nsys 逐例):29 MB 的 IQ4_XS 张量
  **m=1 → 32 µs(900 GB/s,带宽地板)/ m=8 → 238 µs / m=64 → 1885 µs**;
- 读 `gguf_gemm.cu` 的发射契约:`grid = (ceil(n/n_warps), m)` ⇒ **每个 (权重行, token)
  独立读一遍权重行** ⇒ 权重流量 = `m × 权重字节数`。

即:**decode(m=1)本来就是对的;verify(m=8)与小块 prefill 一直在读 8 遍权重**。
27B 权重 13.3 GB × 8 = 106 GB ⇒ 175 ms(≈600 GB/s 有效)正是这条公式。

### 9.2 两刀(都可回退,零语义变化)

**① 复用臂补 10 型**(`gguf_tile.cu`;UD 混型档的 IQ/K-quant 全族)
`owl_gguf_mmvq_tile_body` 本是逐型模板,只缺实例 ⇒ 加 wrapper + 入口名 + 声明:
Q2_K/Q3_K/IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS(MT=16)。
门 = `tile_gemm_matches_mmvq_bitwise` 的 IQ 版:**12 型 × m∈{8,17} 全 24 组逐位相同**。
效果:verify 175 → **127 ms**。

**② MMQ2(int8-MMA)小 m 下限 64 → 4**(`family::gguf::MMQ2_SMALL_M`)
实测交叉表(29 MB IQ4_XS 张量,nsys 逐例):

| m | MMVQ | 复用臂 | **MMQ2(J=64)** |
|---|---:|---:|---:|
| 2 | 66 µs | — | 77 µs |
| 4 | ~130 µs | — | ~78 µs |
| 8 | 238 µs | 153 µs | **81 µs** |
| 32 | 941 µs | ~400 µs | **84 µs** |
| 64 | 1885 µs | 450 µs | **83 µs** |

MMQ2 对 m **近乎恒定**(J=64 分块 + 权重读一遍),交叉点 ≈ 4;m=1(decode)仍选 MMVQ。
门 = `gpu_gguf_mmq2_small_m`:7 型 × m∈{2,8,17,32,64},MMQ2 vs MMVQ 相对差
**最大 1.1e-2**(int8 量化噪声域;判据 5e-2)。
效果:verify 127 → **71 ms**,propose 33 → **23 ms**。

### 9.3 结果(单 3090 Ti,27B UD-IQ4_XS,256 tok greedy)

| 阶段 | verify(GPU) | propose | 轮 | tok/s(degrade 开) | 纯 spec(degrade 关) |
|---|---:|---:|---:|---:|---:|
| 起点(MMVQ) | 175 ms | 37 ms | 167 ms | 23.5 | — |
| + ① 复用臂补型 | 127 ms | 33 ms | ~145 ms | 27.0 | — |
| + ② MMQ2 小 m | **71 ms** | **23 ms** | **73 ms** | **31.8** | **18.6** |
| 无 spec 基线 | — | — | 26 ms | **32.4** | — |

**结论(诚实的边界)**
1. 两刀合计 **2.5× verify / 1.6× propose / 2.3× 轮**;草稿侧已无低垂果实。
2. 纯 spec 仍**亏**(18.6 vs 32.4):验收分布**双峰**(m=0×12 / m=1×4 / m=2×2 /
   m=3×1 / m=5×1 / m=7×6 ⇒ 均值 ≈ 2.2),而 verify 8 行仍 ≈ 2× 裸步。
   盈亏线 = 轮/裸 = 73/26 = **2.8** ⇒ AL 均值 2.2 在线下。
3. 因此**降级机制在位时 ≈ 平手(31.8 vs 32.4)**;关降级反而更差(18.6)——
   该档的正确姿势就是"让降级机制自己选臂"。
4. 剩余杠杆(按性价比):
   - **verify 的 m=8 GEMM 有效带宽 400 vs decode 610 GB/s** ⇒ 需小 m 专用
     tile/split-k(把 CTA 数从 n/128 提上来;IQ 族的 vec_dot 是算术瓶颈,
     MMQ2 的 J=64 在 n=5120 时只有 40 CTA);
   - **propose 23 ms vs 理论 ~9 ms**(lm_head 2.5 GB + 草稿 1.1 GB):逐节点归因
     未做完(图形回放与 decode/verify 交错,窗口切分待做);
   - **AL 是模型上限**(同族 W4A16 草稿 2.86)—— 草稿侧优化抬不动它;
     若要 spec 真赚,只能在**更大 batch 的端侧**(多会话合批)或**更强草稿**上做。

### 9.4 第三刀:propose 瘦身(MMQ2 的 J=16 档 + BF16 激活)

**归因**(nsys 按 CUDA graph 分图;`[dflash-prof]` 的 drain/first/pure 三段):

```
propose 图 GPU = 11.5 ms(engine 的 23 ms 是 profiling 双回放的假象)
  tile_q4k   4.73 ms(46.8 发/次)← 草稿的 Q4_K 全在**复用臂**
  tile_q6k   2.13 ms( 5.2 发/次,412 µs/发)
  mmq2_q5k   1.71 ms(target lm_head 874 MB)
  其余(selector/topk/conv/norm/scale/cast)≈ 0.6 ms
```

**两个堵点,各一刀:**

1. **草稿的 GEMM 根本进不了 MMQ2** —— MMQ2 的激活量化手只吃 F16,而草稿执行域是
   BF16 ⇒ 全部退回复用臂(≈290 GB/s)。
   刀 = `owl_mmq2_quant_q8_1_{ds4,d4}_bf16`(量化体按输入 dtype 模板化,4 行改动)
   + `mmq2_quant_entry_for_act(ty, bf16)` + `Linear` 放行 BF16。
2. **J=64 在 m=8 上有 4× 的 MMA 浪费** —— 上游 `mul_mat_q` 的 J 表本就含 16/24/32;
   `ntx = ceil(m/J)` 在 J=16 与 J=64 下**都是 1** ⇒ 权重流量不变,只砍无效算力。
   刀 = 7 型 × `J16` 实例 + `mmq2_j(m)` 三段(m≤16 → 16 / <128 → 64 / ≥128 → 128)。

**门**:`gpu_gguf_mmq2_small_m` 增 BF16 通道(7 型 × m∈{2,8,17,32,64} 相对差 ≤7e-3);
草稿对拍门 `gpu_gguf_draft_matches_hf_draft` 的 taps 从 3 行改 **8 行**(让 fc 走新路径)。

**实测(单 3090 Ti / 27B UD-IQ4_XS):**

| 面 | 前 | 后 |
|---|---:|---:|
| propose(图 GPU) | 11.5 ms | **6.1 ms** |
| verify(图 GPU,d7) | 71 ms | **52.6 ms** |
| round(engine EWMA) | 83 ms | **62 ms** |

**std 五点三轮(spec=d7,greedy,`pool.prefix_cache=false`):**

| 点 | spec 前(§9.3) | **spec 后** | 无 spec 基线 | 相对基线 |
|---|---:|---:|---:|---:|
| 512 | 52.3 / 54.4 / 48.0 | **72.4 / 73.8 / 66.2** | 40.5 | **+78%** |
| 1024 | 51.3 / 48.9 / 48.0 | **69.2 / 66.3 / 63.7** | 40.7 | **+63%** |
| 2048 | 37.4 / 35.8 / 35.6 | **48.7 / 45.9 / 44.5** | 40.7 | **+14%** |
| 4096 | 33.4 / 33.3 / 33.6 | 34.6 / 34.2 / 34.7 | 40.1 | −14% |
| 8192 | 32.4 / 31.3 / 32.3 | 32.3 / 33.0 / 33.3 | 38.8 | −15% |

⇒ **投机在中短 ctx 正式转正**(512/1024 +60~80%,2048 +14%);4096+ 仍亏,
根因 = **AL 随生成位置衰减**(≈4.0 早期 → ≈1.7 后期)+ verify 随 ctx 涨,
降级机制会在窗口 AL 跌破盈亏线(62/28≈2.2)时自动回落裸 decode。
