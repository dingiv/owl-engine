# 词表切档 × greedy：EOS 永不命中（2026-11-04 立案）

> **一句话**：`parallel.tp_vocab_split=true` 且请求走 **greedy** 时，模型永不输出
> EOS，一直生成到 `max_tokens`；同一 prompt 关掉词表切档即正常 `stop`。
> **影响面**：3080 生产档（`qwen38_27b-3080_tp2.toml`，`tp_vocab_split=true`）的
> **greedy 客户端**（`temperature=0`）；sampling 档（`cand` 候选路径）未见症状。
>
> 发现路径：GGUF TP2 prefill 排查（`gguf-tp2-kquant-设计.md` §13.5）的数值门副产品。

---

## §0 复现（2026-11-04，双 3080 20G，GGUF IQ4_XS TP2，spec off）

同一 prompt、`temperature=0`、`enable_thinking=false`、`max_tokens=64`：

| `tp_vocab_split` | finish_reason | completion_tokens | content |
|---|---|---:|---|
| **true** | **length** | **64** | `391\n\n（注：17 × 23 = 391。虽然您要求"只给数字"…` |
| false | **stop** | 3 | `391` |

两次运行重复 3 次，结果逐字相同（确定性，非采样抖动）。

32 prompt 数值门的 token 数一致性：
- 切档 **24/32**（8 条跑满 16 token 不 stop）
- 不切档 **32/32**

## §1 关键事实

- 词表切档：rank0 = 低半 `[0, vocab/2)`，rank1 = 高半；
  Qwen3.8-27B 的 `vocab = 248320` ⇒ 半表界 = **124160**。
- EOS id ≈ **151645 / 151643**（模型 `eos_id` 族）⇒ **恒在 rank1 半表**。
- greedy 路径（`phases.rs::execute_decode` 的 `else` 分支）在
  `head_split_active()` 时走 `read_full_logits("logits")`：
  本 rank 半表 + peer 半表 → `concat_head_rows(a, b, half)`（rank0 低半 + rank1 高半）。
- prefill 首 token（`harvest_prefill`）同样走 `read_full_logits`，**首 token 是对的**
  （L1 30–31/32 与单卡一致）⇒ 拼接/偏移至少在低半侧成立。
- 症状形态 = "低半 argmax 一直赢"：EOS（高半）永远进不了 argmax。

## §2 嫌疑（按可能性）

1. **peer 的 `logits` 输出槽未写入/恒零**（最可能）：`read_full_logits` 对 peer 槽
   **不查 `has_output`**，直接 `read_output_f32`；若该槽在 peer 图上缺失/未写，
   读到的可能是零块 —— 半表 logits 全 0 ⇒ 低半任何正值都赢，EOS 不可达。
   首 token 仍然正确（真 argmax 恰在低半的概率高）。
2. `concat_head_rows` 的 `half` 与 peer 槽宽不一致（会在形状门响亮拒，实测未拒
   ⇒ 宽度是对的；若成立只能是内容错）。
3. decode 图在切档时的 `logits` 输出被 **device argmax 的 token 槽** 覆盖/混用。

## §3 快速判据（建议的下一步）

- 在 `read_full_logits` 里对 peer 槽做一次**内容自证**（诊断门）：比较
  `peer_logits` 的 max/min 与零块；或加 `has_output("logits")` 的响亮拒。
- 单测级：`tp_head_i8_gate.rs` 已有两进程门 —— 扩一条"切档 greedy 32 步 token 与
  单卡一致（含 EOS 步）"。
- 修复方向（取决于判据）：peer 槽补齐 / 读侧回落"半表未在场 ⇒ 响亮拒" /
  greedy 也走 `cand` 合并路径（`read_session_cands` 的 `half` 偏移已验证用于采样）。

## §4 临时规避

- 切档档位要跑 greedy：加 `--set parallel.tp_vocab_split=false`（或客户端用采样）；
- 生产 3080 档（sampling 0.7）不受影响的前提是 `cand` 路径确实带 peer 半表 ——
  建议顺手用 `/debug/agent` 复核一次 stop 行为。
