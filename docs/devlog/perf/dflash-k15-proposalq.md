# DFlash2 K=1..15 / proposal_q / 32K 质量门（2026-10-23 续）

> 接 `dflash-optimization-handoff.md`。本轮把 NInfer 的三件事落进 owl：
> 结构化行容量 16、DFlash2 运行时 W=K+1、proposal_q + 稀疏拒绝采样。
> 数据 = 2×3080（CUDA ordinal 2/3），Qwen3.8-27B AWQ + DFlash2 W4A16。

## 0. 结论先行

- **K=1 与 K=15 都能跑通**：verify 图 T=K+1、concat 16 槽、fold 记录 8 件/层已解耦。
- **K=15 不划算**：同 prompt 16 token，K15 round 52.7ms / AL 1.81；
  K7 约 35ms / AL 1.98 ⇒ K15 的 verify 税（T=16）远大于 AL 增益。
- **masked-compute 已实现，但实测无显著 verify 收益**：K15 N8
  verify 42.19→41.65ms（噪声），K7 N8 vs N4 27.66→27.70ms；verify 仍由
  权重带宽主导。真正省下的是末轮 propose（K15 N8 round 63.48→59.58ms，
  propose 次数 14→11）。
- **default 保持 `[spec] depth=7`**。K=15 只作为配置能力/实验档。
- **proposal_q + 拒绝采样已接图档**：K7 采样实测 `q=11 / none=6 / reject=10`；
  K15 采样 `q=8 / none=6 / reject=8`。首轮/无图档仍走旧的
  “目标抽样 == draft” 等价式（分布正确，接受率较低）。
- **32K fixed-tail W8A16 TC 质量门**：f16 vs i8 **64/64 逐 token 全等**；
  round 82.03 vs 80.66ms，verify 69.21 vs 68.79ms。
- **采样档**：temp=0.1 时 f16 vs i8 **384/384 全等**（6 prompt×64）；
  temp=0.7 时 **322/384 = 0.8385**（4/6 逐字全等）—— 这是采样放大微小
  logits 差异的预期现象，不作为 0.95 等价门；低温度门已过。

## 1. 代码变更

### 1.1 行容量 16（结构）

- `SPEC_VERIFY_ROWS_MAX = 16`、`SPEC_DEPTH_MAX = 15`。
- 新增 `SPEC_FOLD_RECORDS_PER_LAYER = 8`：GDN fold 记录（conv 3 raw +
  递推 5 件）与 verify 行数**解耦**。此前两者都是 8，K>7 会去找不存在的
  `r16..r31` 槽。
- `CONCAT_ROWS_SLOTS = 16`；`concat.cu` 的 f16/f32 两核从 8 参扩到 16 参。
- 所有 `owl_concat_rows_*` 调用点（verify 首轮、batch decode、prefill
  taps、`concat_rows` helper）都按 16 槽补齐。
- `kv_slots` 从 `[u32; 9]` 改为 `[u32; SPEC_VERIFY_ROWS_MAX + 1]`。
- `verify` 图仍为 T=K+1；`fold`/`n_rec` 仍按每层 8 件。

### 1.2 DFlash2 运行时 W=K+1

- `DFlash2Draft` 新增 `block` / `depth` 字段与 `with_block_size()`；
  `DfLayer`/`GroupedConv` 增加运行时 `set_block_size()`。
- `forward_hidden` 的 embed 行数、selector 切片行数改用 `self.block/self.depth`。
- `spec_runtime` 装载后调用 `.with_block_size(depth + 1)`。
- `graph_fleet::plan_dflash` 的输入槽/输出槽/tap 形状全部按 `d2.block()/depth()`；
  dflash 图仍单图，不再固定 8 行。
- `dflash_graph_round` / `dflash_propose` / `dflash_encode` 的 8/7 字面量
  改为 `block/depth`；ring 槽上界改为 `BLOCK_MAX=16`（页对齐仍 2080）。
- `SPEC_DEPTH_MAX=15` 允许配置 K=15；默认 profile 不变。

### 1.3 target live P+1（提交 + masked compute）

- `scheduler`：`remaining_out <= 1` 时直接裸 decode；否则仍发**满 K**
  spec 轮（图 T=K+1 定形）。
- `execute_spec_round`：`live = min(K, remaining-1)`；只接受/提交前 P+1 行。
- `verify_round(..., target_rows=live+1)`：
  - `gdn_cu = [0, target_rows]` 作为运行时图输入 ⇒ GDN conv/递推只走 live 行；
  - `fi_q_cu = [0, target_rows]`、`ctx_total=pos+target_rows`、FI plan 的
    `t=target_rows` ⇒ FI attention 只处理 live query；
  - XQA seq_len 同步为 `pos+target_rows`；
  - 图仍 T=K+1 容量，额外行的输出槽不提交、不 fold。
- 预算尽时**跳过末轮 propose**（`spec.propose.skip`），避免 tail 轮的
  未算 tap 行被下一轮满 W 草稿读取。
- 语义验证：K15 N16（live=15，无 mask）与 N8（live=7，mask）前 8 token
  逐位相同；K7 N8 与 N4 前 4 token 逐位相同。

### 1.4 proposal_q + sparse rejection

新增 `crates/engine/src/exec/proposal_q.rs`：

- `DraftQ::from_scores(scores[K,16,16], cand[K,16], temp, seed, pos)`：
  按实际前驱做 conditional walk，每步 `softmax(score/temp)`，从 q 抽
  draft；保留每步 q 与候选 id。
- `accept_rejection()`：`min(1,p(d)/q(d))` 接受；首次拒绝从
  `max(p-q,0)` 归一化抽 correction；全接受从最后一行 p 抽 bonus。
  RNG 用途分离：proposal / accept / correction 三个 purpose，避免
  “拒绝 gate 与 correction 用同一个均匀数”的相关性 bug。
- `lookup` 改写草稿后 `DraftQ::reconcile/force_one_hot`：lookup 是 host
  确定性改写，不是选择器分布 ⇒ q 退化为该 token 的 one-hot。
- 图档：dflash 图新增 `sel`（drafts‖scores 单缓冲）与 `topk_all`
  （值区‖索引区）输出；采样档读两槽建 q，greedy 档不回读。
- eager 首轮/无图档：**不在草稿树外二次求值**（树含 RowParallel AR，
  单边重跑会与 peer 失步挂住，2026-10-23 实测）⇒ 回退旧等价式。
- 分档（`tp_vocab_split`）暂未接 q，回退旧等价式；生产默认
  `tp_vocab_split=false`，图档 q 生效。

## 2. 实测

### 2.1 K 边界（greedy，TP2，3080 对，N=16/4）

| 档 | rounds | m 直方图 | AL | round | verify | propose |
|---|---:|---|---:|---:|---:|---:|
| K=1 | 6 | `[0,6]` | 1.00 | 33.42ms | 21.83ms | 10.45ms |
| K=7（历史档） | — | — | ~1.98 | ~35ms | ~27.5ms | ~6.0ms |
| **K=15** | 31 | `[7,11,5,3,2,1,1,1]` | **1.81** | **52.70ms** | **39.95ms** | 10.78ms |

K=15 的 round 比 K7 高 ~50%，AL 反而略低（更长 block 改变 proposal，
并非短 block 的前缀）。**不转默认**。

### 2.2 K=7 采样 / q 路径（full head，temp=0.7，N=8）

```text
rounds=17  hist [6,3,1,2,1,2,1,1]  q=11 none=6 reject=10
round 40.56ms / verify 28.00ms / propose 8.41ms
```

`q=11` 证明图档 proposal_q 生效；`none=6` 是 6 个 prompt 的首轮
（eager 草稿，回退旧式）；`reject=10` 证明拒绝采样在跑。

### 2.3 K=15 采样（full head，temp=0.7，N=8，live cap 生效）

```text
rounds=14  hist [4,5,1,1,2,0,1]  q=8 none=6 reject=8
round 63.48ms / verify 42.19ms / propose 17.41ms
```

N=8 时 scheduler 把 live 压到 7（保留 bonus），verify 仍 T=16；
只提交 8 token，预算不越界。

### 2.4 32K fixed-tail W8A16 TC 质量门

prompt = `tools/fixedtail_prompt.py --length 32768`（34,240 tok），
N=64，TP2 3080，DFlash2 d7，head split。

| 臂 | rounds | AL | round | verify | propose | decode |
|---|---:|---:|---:|---:|---:|---:|
| f16 | 30 | 1.10 | 82.03ms | 69.21ms | 10.81ms | 39.50ms/tok |
| **i8 (W8A16 TC)** | 30 | 1.10 | 80.66ms | 68.79ms | 9.82ms | **38.84ms/tok** |

token 一致率 **64/64 = 1.0000**，逐字全等 1/1。32K 上 W8A16 TC
没有质量回归；round 小幅快 1.7%。

### 2.5 masked-compute A/B

K15 N8（live=7，target_rows=8，图容量 16）：

| 项 | mask 前 | mask 后 |
|---|---:|---:|
| round | 63.48ms | **59.58ms** |
| verify | 42.19ms | 41.65ms |
| propose 次数 | 14 | 11（末轮 skip） |

K7 N8（live=7，target_rows=8）vs K7 N4（live=3，target_rows=4）：

| 档 | verify |
|---|---:|
| N8（满 live） | 27.66ms |
| N4（mask 一半） | 27.70ms |

结论：**attention/GDN 的 mask 对 verify 无可测收益**（权重带宽主导）；
round 的 3.9ms 改善主要来自“预算尽跳过末轮 propose”。语义 A/B：

- K15 N16（无 mask）vs N8（mask）：6 prompt 前 8 token **逐位相同**；
- K7 N8（无 mask）vs N4（mask）：6 prompt 前 4 token **逐位相同**。

### 2.6 采样档质量

6 prompt × 64 tok，temp=0.7 / topk20 / topp0.95 / seed 12345：

- f16 vs i8：**322/384 = 0.8385**，4/6 逐字全等；首个分歧在
  prompt 0 的第 38 token、prompt 5 的第 24 token。
- 同档 temp=0.1：**384/384 = 1.0000**，6/6 逐字全等。

结论：低温采样门（0.1）全过；0.7 的 token 差异是采样对微小 head
数值差的放大，不代表 32K/贪心语义回归。生产若要求 0.7 采样的
分布级判据，下一步应比较 top-1/概率质量，而不是要求 token 逐位全等。

## 3. 测试命令

```bash
# K=1 / K=15 冒烟
OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_TP_HEAD_ARM=i8 \
OWL_TP_HEAD_SPEC=1 OWL_TP_HEAD_DEPTH=15 OWL_TP_HEAD_N=16 \
OWL_TP_HEAD_CTX=1024 cargo test --release -p owl-engine \
  --test tp_head_i8_gate -- --nocapture

# 32K fixed-tail（两臂 + compare）
python3 tools/fixedtail_prompt.py --length 32768 --out /tmp/fixedtail32k.txt
OWL_TP_HEAD_PROMPT_FILE=/tmp/fixedtail32k.txt OWL_TP_HEAD_CTX=35000 \
OWL_TP_HEAD_N=64 ... OWL_TP_HEAD_ARM=f16 ...
# i8 同参；然后 cp 到 /tmp/owl-tp-head-{f16,i8}.txt 跑 compare

# 采样档
OWL_TP_HEAD_SAMPLE=1 OWL_TP_HEAD_TEMP=0.1 OWL_TP_HEAD_SEED=12345 ...
```

`OWL_TP_HEAD_VSPLIT=0` 才走 full-head 图档 q；默认 `vsplit=1` 走
半表合并路径（q 暂回退旧式）。

## 4. 遗留

1. **masked-compute 已做但无性能收益**：GDN/FI 只 mask 了 attention/GDN
   小头，权重带宽仍在；若继续追性能，应做 row-sliced projections/MLP
   或直接转 verify 算子优化，而不是继续扩 mask。
2. **分档 q**：`tp_vocab_split` 的合并路径还没有 proposal_q；默认关，
   但若要开半表生产档需补。
3. **K sweep**：K=8..14 未逐档测；K=15 已证不划算，K7 仍最优候选。
4. **首轮 q**：eager 草稿没有 q（避免二次求值挂住）；首轮用旧等价式，
   分布正确但接受率低于 q。
5. **32K 采样门**：只跑了 512/1024；32K 的 temp0.7 采样未跑。
