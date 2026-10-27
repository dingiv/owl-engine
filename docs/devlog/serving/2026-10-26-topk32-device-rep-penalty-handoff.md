# top-32 候选路径支持 rep_penalty —— 设计与交接

> 2026-10-26。状态：**已实现并真机验收**（P0–P4）。
> 分支：`feat/topk32-rep-penalty`（从 `feat/store-hook-p2` tip `898bf77e`
> 起，含 sampling/abort 修复；main 合入顺序见文末）。
> 现场服务：3080 TP2，profile `qwen38_27b-3080_tp2.toml`，
> `sampling.rep_penalty = 1.05`，日志 `/tmp/owl-8135-topk32-final2.log`。

---

## 0. 一句话

把 repetition penalty 放到**设备端 top-32 选候选之前**，用固定 256 token
的 committed history 作为输入槽；这样 `rep_penalty > 1` 仍然走候选快路，
不再回退整行 logits，也**不增加任何桶图**。

---

## 实现结果（2026-10-26）

### 与设计的一处必要偏差

`rep_penalty` 在核里是 **`const float*` 设备指针**，不是 `float` 标量：

- 图输入槽的内容必须逐轮可换；标量会在 CUDA graph 捕获时烘死；
- 因此 kernel 参数声明为 `Param::t_in("rep_penalty", Dtype::F32, 1)`，
  核内 `const float rp = rep_penalty[0]`；
- `hist` / `rep_penalty` 都是固定槽，内容变化不触发重捕获；
- 其余（H=256、hist 升序去重、尾部哨兵、`vocab_offset`、先惩罚后 top-32）
  与设计一致。

### 代码落点

```text
crates/engine/src/sampler.rs       REP_HISTORY_WINDOW/HISTORY_SENTINEL + penalty_history
crates/kernels/cu/elems/topk32.cu  hist 载入 shared + 二分命中 + 惩罚 + top-32
crates/kernels/src/ops/elems.rs    Topk32Split 参数表（hist/rep_penalty/hist_cap/offset）
crates/models/src/ops.rs           topk32_of_logits 接 hist/rep/offset
crates/engine/src/graph_fleet.rs   session/verify 图固定槽 rep_hist + rep_penalty
crates/engine/src/exec/phases.rs   decode 每步写 committed history
crates/engine/src/exec/spec.rs     verify 每轮写 committed history + 打开 use_cand
crates/kernels/tests/topk32_penalty.rs  5 类 kernel 对拍
profiles/qwen38_27b-3080_tp2.toml  rep_penalty = 1.05
```

### 验收数字（3080 TP2，同 prompt/seed，256 tok）

| 配置 | decode | spec AL |
|---|---:|---:|
| `rep=1.0` | 56.0–57.3 t/s | 1.57 |
| `rep=1.05` | 57.6–58.6 t/s | 1.71 |

另一条 thinking 长 prompt 上 `rep=1.05` 实测 85–94 t/s（旧回退路径
同档约 52 t/s），证明没有关闭候选快路。`spec.verify.pre.lgread` 在
真机 metrics 中 **0 行**（未读整行 logits），`spec.verify.pre.cand.r0`
≈0.33 ms/轮。

图：boot 前后均为 session×2 / verify×2 / fold×2 / propose×2；运行期
（多轮、不同 history 内容）**无重捕获**。

测试：

```text
topk32_penalty（GPU）        1 passed（5 cases，值与 id 逐位对 host）
sampler unit                 12 passed
op_tables                    10 passed
owl-engine --lib             171 passed
gpu_spec_identity_gate       0.8B + 27B 均 PASS
gpu_image_evict_spec_identity_gate  PASS
```

### 合并注意

本分支基于 `feat/store-hook-p2`（尚未合入 main）。合并顺序建议：
先合 store-hook-p2，再合本分支；`profiles/qwen38_27b-3080_tp2.toml`
的 sampling 段和 `docs/README.md` 索引按冲突双方保留。

---

## 1. 现场与动机

当前分布式采样快路：

```text
rank0/1 logits shard
→ owl_topk32_split_f16 / merge
→ host 合并 ≤64 候选
→ sample_from_candidates
```

准入在：

- `crates/engine/src/exec/spec.rs::use_cand`
- `crates/engine/src/exec/phases.rs::use_cand`

两处都有：

```rust
sampler_cfg.rep_penalty <= 1.0
```

一旦 `rep_penalty > 1.0`：

```text
候选快路关闭
→ 回读整行 logits（半表 ~2MB/rank）
→ host 全表采样
```

2026-10-26 实测（3080 TP2，同 prompt/seed，256 tok，spec d7）：

| 配置 | decode |
|---|---:|
| `temp=0` greedy | 78.7–79.2 t/s |
| `temp=0.7, rep_penalty=1.05` | ~52.2 t/s |
| `temp=0.7, rep_penalty=1.0` | 81.8–82.4 t/s |

所以：**不是采样慢，是 `rep_penalty>1` 把候选快路关了**。

---

## 2. 目标 / 非目标

### 目标

1. `rep_penalty > 1.0` 时仍走 `cand` top-32 快路；
2. 惩罚发生在 top-32 **选择之前**；
3. 图数量不变：复用现有 session / verify 图，只加固定输入槽；
4. `rep_penalty=1.0` 时核内直接跳过，当前性能零回归；
5. decode 相对当前采样快路损失 ≤5%，最好 ≤2%。

### 非目标

- 不新增 `rep_penalty=1.05/1.1/...` 的图桶；
- 不新增 history 长度 32/64/128 的图桶；
- 不改 batch 图的采样路径（当前 batch 仍走 logits 回退）；
- 不把 rep_penalty 做成 per-request OpenAI 字段（第一版只走
  `[sampling] rep_penalty`；要加请求级字段另开一项）；
- 不改 DFlash2 草稿自己的采样。

---

## 3. 语义定义（必须先钉死）

### 3.1 penalty set

第一版：

```text
history = 最近 H 个「已提交」token
          = prompt 尾部 + 已生成 out
H = 256（编译期常量，例如 REP_HISTORY_WINDOW）
```

- 按 token id **去重**；
- 排序后上传（kernel 内二分查找）；
- 未使用槽填 `u32::MAX` 哨兵；
- 空 history = 全哨兵。

**v1 所有 verify 行共用同一份 history**：草稿 token 是 speculative，不进入
penalty set；只惩罚已经提交的 prompt/out。这样：

- 不需要 `[rows,H]` 的 per-row 历史；
- 图槽只有 `[H]` + 一个 scalar；
- 不引入行间历史差异。

如果后续发现复读来自“草稿行内部”，再评估 per-row history；那是第二阶段，
不在本刀。

### 3.2 惩罚公式

与 host sampler 一致：

```text
if rep_penalty > 1.0:
    v = logit
    if v > 0: v / rep_penalty
    else:     v * rep_penalty
```

只对 history 命中的 token 做一次。

### 3.3 top-k 顺序

必须是：

```text
penalty → top-32
```

不能是：

```text
top-32 → penalty
```

因为被惩罚的重复 token 可能从第 33 名掉下去，原第 33 名必须补进 top-32；
只在 64 个候选上惩罚不等价。

### 3.4 rank 分片

`topk32` 吃的是**本 rank 的半表**：

```text
local index i  ↔  global token = vocab_offset + i
```

`history` 统一存**全局 token id**。kernel 内：

```text
gid = vocab_offset + lo + i
if hist_contains(gid): penalty
```

`vocab_offset`：

- rank0 = 0；
- rank1 = `logits_vocab`（半表宽度）；
- 单卡/未切词表 = 0。

候选输出索引仍是 local；host 合并时照旧加 `vocab_offset`。

---

## 4. 接口设计

### 4.1 kernel

`crates/kernels/cu/elems/topk32.cu`

`owl_topk32_split_f16` 增加参数：

```c
const uint32_t* __restrict__ hist,  // [H] 全局 id，升序去重，尾部 u32::MAX
int hist_cap,                       // H；固定 256
int vocab_offset,                   // 本 rank 全局偏移
float rep_penalty                   // <=1.0 时跳过
```

`owl_topk32_merge_f16` **不动**。

核内建议：

1. 每个 block 首次把 `hist[0..H]` 载入 shared（1 KiB）；
2. 对每个扫描到的 logit：
   - `gid = vocab_offset + local_index`；
   - 若 `rep_penalty > 1.0`，在 shared 排序表里二分；
   - 命中后按 §3.2 变换；
3. 再走现有 `topk32_push`。

`rep_penalty <= 1.0` 时整段跳过，保持当前路径。

> H=256 时二分约 8 次比较/元素。若实测开销 >5%，再考虑
> 128 槽开放寻址 hash（仍固定 H，不建桶）。

### 4.2 ops / driver

- `crates/kernels/src/ops/elems.rs::Topk32SplitF16`
  增加 `Param::t_in("hist", Dtype::U32, 1)`、
  `Param::i32("hist_cap")`、`Param::i32("vocab_offset")`、
  `Param::f32("rep_penalty")`。
- `crates/kernels/src/driver.rs::topk32_split` 的 Shape 不变，只透传新标量。
- `crates/models/src/ops.rs::topk32_of_logits` 增加
  `hist: &TensorOps, rep_penalty: f32, vocab_offset: usize`；
  split 节点接 hist，merge 节点不变。

### 4.3 图槽

只给现有图加固定输入槽，**不新增图**。

Session/bare decode 图（`graph_fleet.rs`）：

```text
InputSlot::f32("rep_hist", H)        // u32 位型，f32::from_bits
InputSlot::f32("rep_penalty", 1)     // 运行期 scalar
```

Verify 图（每个 depth 一张，已存在）：

```text
InputSlot::f32("rep_hist", H)
InputSlot::f32("rep_penalty", 1)
```

- `H` 是编译期常量，不随 depth 变；
- `rep_hist` 的 shape 恒为 `[H]`，内容可变；
- `rep_penalty` 是内容 scalar，不是图常量；
- 不按 `rep_penalty` 值分图；
- 不按 history 长度分图。

`vocab_offset` 建议**烘焙进 per-rank 图**（rank0/1 各自的图本来就分开）；
如果图构建面是 rank 共享的，则再加一个 `[1]` 输入槽，仍然不建新图。

### 4.4 host 接线

新增唯一 helper（建议住 `sampler.rs` 或 `img_evict` 同级）：

```rust
pub const REP_HISTORY_WINDOW: usize = 256;

pub fn penalty_history(
    prompt_ids: &[u32],
    generated: &[u32],
) -> Vec<u32>;  // 最近 256，去重排序，尾部 u32::MAX
```

- **bare decode**：`prompt_ids` + `act.out`；
- **verify**：`prompt_ids` + `act.out`（v1 不加入 block/draft）；
- 两者使用同一 helper；
- 每步/每轮写 `rep_hist` 和 `rep_penalty` 输入槽；
- `rep_penalty<=1.0` 时也写（哨兵 + 1.0），核内零成本跳过。

### 4.5 `use_cand` 门

两处 `use_cand` 删除：

```rust
sampler_cfg.rep_penalty <= 1.0
```

仍保留：

```text
sampler_on
&& head_split
&& topk > 0 && topk <= 32
&& proposal_q.is_none()
```

`sample_from_candidates` 不再接收 history；它只对**已经惩罚过**的候选做
temperature/top-k/top-p/抽样。

### 4.6 非候选路径

以下仍走旧 logits：

- 未切词表（`!head_split`）；
- `topk > 32`；
- `proposal_q`；
- batch 图。

它们的 rep_penalty 语义保持现状；不与候选路径做逐 token parity，
只各自对 host 参考做门。

---

## 5. 为什么不能只惩罚候选

反例：

```text
原始 logits:  A(重复) 排名 1, B 排名 33
rep_penalty 后: A 掉到 34, B 应进入 top-32
```

只在 ≤64 候选上惩罚：

- B 不在候选里；
- top-32 集合少了 B；
- top-p 归一化也少一块质量。

因此必须**惩罚在 top-k 之前**。

---

## 6. 实施顺序

### P0：常量与纯函数

- 加 `REP_HISTORY_WINDOW=256`；
- 加 `penalty_history`；
- 单测：去重、排序、尾部截断、空 history、哨兵。

### P1：kernel 与 host 参考

- 改 `topk32_split` 签名/参数；
- 加 kernel 单测：
  - `rep_penalty=1.0` 与旧 topk32 逐位一致；
  - 惩罚后第 33 名补位；
  - rank1 `vocab_offset` 命中/不命中；
  - sentinel/空 history。
- 用 host f32 参考对拍 top-32 集合与顺序。

### P2：图槽（先保持 rep=1.0）

- session / verify 图加 `rep_hist` + `rep_penalty` 槽；
- `rep_penalty=1.0` 跑全量回归，确认图数量不变、性能不变；
- 检查 graph 输入写入不需要每步同步。

### P3：打开 rep_penalty > 1

- 删除 `use_cand` 的 `rep_penalty<=1` 条件；
- `sample_from_candidates` 明确“candidates 已惩罚”；
- 单测：候选路径与全表参考（同 history）逐 token 等价。

### P4：profile / 联调

- 3080 档从 `rep_penalty=1.0` 试到 `1.05`；
- A/B decode、spec AL、GUI 复读率；
- 若性能或工具调用质量变差，回退到 1.0 即可（核内跳过）。

---

## 7. 验收门

### 功能

```text
rep_penalty=1.0：topk32 输出与旧实现逐位一致
rep_penalty=1.05：top-32 = host 参考（penalty 后）的 top-32
rank0/rank1 的 global id 映射正确
空 history / 哨兵 / 重复 id 正确
```

### 图

```text
boot 前后图数量完全一致
没有新增 depth 桶 / rep_penalty 桶 / H 桶
rep_hist / rep_penalty 内容变化不触发重捕获
```

### 性能

```text
temp=0.7, rep=1.0  vs  rep=1.05
decode 损失 ≤5%（目标 ≤2%）
spec AL 不因惩罚 lookup 明显下降
```

### e2e

- `OWL_E2E_ONLY=gpu_spec_identity_gate`（0.8B DFlash2）
- `OWL_E2E_ONLY=gpu_image_evict_spec_identity_gate`
- 3080 真实 GUI 会话：连续删图 + 中断 + 下一轮
- `ops.ensure_evicted == 0` canary 保持

---

## 8. 性能预算

H=256：

- 图槽：每图 256×4B = 1 KiB + scalar；
- 每 block 载入 1 KiB shared；
- 每元素二分约 8 次比较；
- `rep_penalty=1.0` 全跳过。

如果 topk 核成本上涨 >5%，优化顺序：

1. shared 内 hash（128/256 槽）；
2. 只对 `gid` 落在 `[hist[0], hist[last]]` 的元素做二分；
3. 把 H 降到 128（需同步改 host 窗口和门）。

不要用 Bloom/bitmap 近似：会引入假阳性，破坏 parity。

---

## 9. 风险 / 待决

1. **verify 的 penalty 历史语义**：v1 用 committed history，不加入 draft。
   如果后续发现草稿行内部复读，再评估 per-row `[rows,H]`；那会增加槽和
   host 构造成本，但仍不需要新图桶（固定 `[max_rows,H]`）。
2. **f16 logits**：惩罚作用在 f16→f32 后，与 host 全表 f32 有量化差；
   门用相对误差/候选集合，不做逐位。
3. **图输入写入成本**：每步 1 KiB 上传，必须走无同步 input slot；
   不要每步新建 host 缓冲或触发 capture。
4. **batch**：本刀不改；batch 采样仍走全表。
5. **rep_penalty 质量**：`sampler.rs` 历史注记显示 1.15 在 27B 上曾导致
   长输出崩坏；1.05 也只作为 opt-in，需 GUI 长跑。
6. **请求级 rep_penalty**：OpenAI 面当前没有该字段；第一版只用 profile。

---

## 10. 文件地图

```text
crates/kernels/cu/elems/topk32.cu
crates/kernels/src/ops/elems.rs
crates/kernels/src/driver.rs
crates/models/src/ops.rs
crates/engine/src/graph_fleet.rs
crates/engine/src/exec/phases.rs
crates/engine/src/exec/spec.rs
crates/engine/src/sampler.rs
crates/kernels/tests/          # topk32 parity / penalty
crates/engine/src/tests.rs     # candidate parity / graph slots
profiles/qwen38_27b-3080_tp2.toml
```

### 快速验证命令

```bash
# host / kernel
cargo test -p owl-kernels --lib topk32
cargo test -p owl-engine  --lib candidate

# 0.8B spec 恒等门
OWL_TEST_DEVICE=42:00.0 OWL_E2E_ONLY=gpu_spec_identity_gate \
  cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1

# 图像 + spec
OWL_TEST_DEVICE=42:00.0 OWL_E2E_ONLY=gpu_image_evict_spec_identity_gate \
  cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1
```

### 交接起手

1. 从 main 新建 `feat/topk32-rep-penalty`；
2. 先做 P0/P1，保持 `rep_penalty=1.0` 全绿；
3. P2 只加槽，确认图数量/性能不变；
4. P3 再打开 1.05，跑 3080 GUI A/B；
5. 不在本刀引入 per-row history / batch / 请求级 rep_penalty。
