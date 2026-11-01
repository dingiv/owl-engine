# W4A8 M-split 激活尺度未随切片推进（立案与修复，2026-10-31）

## 结论

**根因**：`marlin_mm`（`crates/kernels/cu/marlin/marlin_host.cu`）在 M 方向
分段发射时，只推进了 `A_ptr` / `C_ptr`，**没有推进 `a_s_ptr`**。W4A8 的
`a_scales` 是 `[M,1]` 的 **per-token** 尺度；`m > 1024` 时 M 被拆成多段
（2048 → 1024 + 1024），**第 2 段起沿用第 1 段的尺度**做权重反量化 →
每个 prefill 块的后半段激活被错误缩放 → 长上下文后段乱码/丢针。

**修复**：`is_a_8bit` 时每段 `a_s_ptr += prob_m_split`。

**验证**：20k prompt、位置 2k/6k/10k/15k/19.5k × 2 = **10/10** 命中；
a8 + DFlash d7 12k 四位置 **4/4**；w4a16 路径不受影响。

## 症状（用户报告）

- 27B / 3080 双卡 / `quant=w4a8`：短问答正常，长上下文"胡言乱语、看到乱码、
  读到的信息不全"。
- 两款模型（cyankiwi AWQ、pi AWQ）都复现 → 与 artifact 无关。

## 复现

- owl 3080 TP2，`spec.depth=0`，其余 profile 默认（`kv_fp8=true`，
  `prefill_chunk=2048`）。
- 12k/20k 随机词池，针 `K7-ZEBRA-4713` 埋 head/mid/tail，greedy、
  `enable_thinking=false`。
- 结果（修复前）：

| 配置 | 12k head | 12k mid | 12k tail | 20k |
|---|---|---|---|---|
| w4a8（cyankiwi / pi） | ✅ | ❌ | ❌ | ❌ |
| w4a16（pi） | ✅ | ✅ | ✅ | ✅ |
| w4a8 + bf16 KV | ✅ | ❌ | ❌ | — |
| w4a16 + DFlash d7 | ✅ | ✅ | ✅ | — |
| **w4a8 + `prefill_chunk=1024`** | ✅ | ✅ | ✅ | ✅ |

- 位置扫描（12k，chunk=2048）：500/1000/3000 命中；2000/4000/6000/8000/
  10000/11500 输出 `K7-Z` / `K7` / 幻觉——**不是完全读不到，而是读到一半
  后被错误 logits 截断**。
- chunk 1536/2048/2560/3072 的失败位置不固定 → 与切片边界/针落点有关，
  不是硬阈值。

## 根因

`marlin_mm` 的 M-split 循环（修复前）：

```cpp
A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
C_ptr += prob_m_split * (prob_n / 8);
rest_m -= prob_m_split;
```

`prob_m_split` 的计算：`par_count = rest_m / 64`，cap 到 `max_par`；
`prob_n > 4096` 时 `max_par=16` → `prob_m_split = 16*64 = 1024`。

- `prefill_chunk=1024`：m=1024 → **单段** → `a_s_ptr` 基址正确 → 正常。
- `prefill_chunk=2048`：m=2048 → **两段**（1024+1024）→ 第二段仍用
  `a_scales[0..1024]`，而 A 已经是第 1025..2048 行 → 反量化尺度错位。
- 2048 是 2026-10-25 起 3080 TP2 档的 prefill 块长；10-27 起 a8 成为
  默认 → 两者叠加把这个**潜伏 bug** 引爆。
- 解码 m=1 只有一段，所以短问答/GSM8K 看不出问题。

## 修复

```cpp
A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
C_ptr += prob_m_split * (prob_n / 8);
// per-token scales [M,1] 必须随 M 切片推进
if (is_a_8bit) a_s_ptr += prob_m_split;
rest_m -= prob_m_split;
```

覆盖 `marlin_gemm_v2_w4a8_ffi` 与 `marlin_gemm_v2_awq_w4a8_ffi`（两者都走
`marlin_mm`，`a_is_s8=true`）。

## 验证

- 修复后（debug 构建，chunk=2048，20k prompt，唯一 nonce 冷启动）：
  位置 2000/6000/10000/15000/19500 × 2 = **10/10** 精确命中。
- a8 + DFlash d7（syvai 草稿）12k：3000/6000/9000/11500 = **4/4**。
- w4a16 分支 `a_is_s8=false`，`a_s_ptr` 不推进，行为零变化。
- 短 prompt / decode（m≤1024）行为不变。

## 影响面

- 所有 W4A8 × Marlin 的 **prefill**（m>1024）；decode 与短 prefill 不受影响。
- 不只 27B：任何 `quant=w4a8` + `prefill_chunk>1024` 的模型都受影响。
- 与 AWQ artifact、DFlash、fp8 KV 无关（已逐一排除）。

## 预防

1. **回归测试**：在 `crates/models/src/layers/linear.rs` 的
   `gpu_w4a8_matches_w4a16_on_real_layer` 旁补 `m=2048` 用例；构造
   **两段不同 per-token scale** 的输入，断言第二段输出与 w4a16 对拍。
2. **契约**：M-split 循环里凡是 `[M,...]` 行主序的附属张量（a_scales、
   任何 per-row bias/mask）都必须与 A/C 同步偏移；代码注释已写明。
3. **验收口径**：a8 档的长上下文门（12k/20k needle，head/mid/tail）应
   纳入 profile 变更检查，不能只跑短问答。
