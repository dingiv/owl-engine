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
