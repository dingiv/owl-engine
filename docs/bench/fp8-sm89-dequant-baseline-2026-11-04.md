# FP8 KV SM89 基线：反量化税（2026-11-04）

> **后续**：原生 FP8 MMA 已落地，实测与对比见
> [`fp8-sm89-native-mma-2026-11-04.md`](fp8-sm89-native-mma-2026-11-04.md)。
> 本文件保留为旧 dequant / shadow 路径的基线。

## 目的

在开发 SM89 原生 FP8 attention 前，先量当前 **FP8 KV 反量化路径**的性能，
作为后续 native FP8 MMA 的对照基线。

## 环境

- GPU：RTX 4080 SUPER，CC 8.9，32G（`0000:81:00.0`）
- 二进制：`target-sm89/release/server`（`arch-sm89`，基于 `bbd4a367`）
- profile：`profiles/qwen38_27b-4080.toml`
- 起机覆盖：`spec.depth=0`、`spec.lookup=false`，文本基准不启投机
- 工具：`packages/llm_speedtest`，`flavor std`，`rounds=1`，输出 256；
  长上下文 `--lengths 32768,65536`，输出 16
- prompt 固定 nonce（std / long 各自固定）；每档 fresh boot

## 结果

### 当前路径：FI FP8 KV，核内反量化（`prefill_shadow=false`）

| 长度 | prefill t/s | decode t/s | TTFT ms |
|---:|---:|---:|---:|
| 512 | 1699.81 | 37.78 | 302.9 |
| 1024 | 2977.66 | 37.67 | 345.8 |
| 2048 | 3173.99 | 37.56 | 647.3 |
| 4096 | 2863.25 | 37.04 | 1432.5 |
| 8192 | 3104.33 | 37.34 | 2640.9 |
| **std avg** | **2763.81** | **37.48** | **1073.89** |
| 32768 | 2379.26 | 37.38 | 13774.0 |
| 65536 | 2099.38 | 35.42 | 31218.7 |
| **long avg** | **2239.32** | **36.40** | **22496.34** |

### 对照：prefill-only f16 shadow（`prefill_shadow=true`）

每层 attention 前把 FP8 K/V 一次性反量化到 dense f16，再走 FI f16 臂。

| 长度 | prefill t/s | decode t/s | TTFT ms |
|---:|---:|---:|---:|
| 512 | 1699.80 | 38.04 | 302.8 |
| 1024 | 2952.70 | 38.19 | 348.6 |
| 2048 | 3158.74 | 38.04 | 650.3 |
| 4096 | 3106.69 | 37.88 | 1320.4 |
| 8192 | 3142.95 | 37.37 | 2608.5 |
| **std avg** | **2812.18** | **37.90** | **1046.11** |
| 32768 | 2705.61 | 37.33 | 12112.8 |
| 65536 | 2274.13 | 35.54 | 28819.9 |
| **long avg** | **2489.87** | **36.44** | **20466.36** |

## 判读

- 512 点是首次请求/图 warmup，明显偏低；std avg 被它拉低。看 1024–8192
  趋势更稳。
- 当前**核内反量化**在 32k/65k 的 prefill 分别为 2379/2099 t/s；
  一次性 shadow 提到 **2706/2274 t/s**（+13.7% / +8.3%）。
  ⇒ 反量化税真实存在，且在长 ctx 更明显。
- 4080 的 FP8 attention 目前仍是 **half MMA + e4m3→f16 dequant**；
  Ada FP8 tensor core 没有被这条路径使用。

## 下一步

在 `packages/owl-worktrees/fp8-native-sm89`（branch
`feat/fp8-native-sm89`）里接 `repos/flashinfer` 0.7.3 的
`USE_FP8_MMA` 路径：

- `include/flashinfer/attention/prefill.cuh`：e4m3 Q/K + e4m3 KV
  的 `m16n8k32` FP8 MMA（`compute_qk_fp8` / `vosplit_compute_pv`）；
- `include/flashinfer/mma.cuh`：`__CUDA_ARCH__ >= 890` 的 FP8 MMA；
- 需要补 Q→e4m3 的量化/尺度路径，并验证 paged KV 入口。

目标：以本文件的 **in-kernel dequant** 与 **shadow** 两列作为基线，
比较 native FP8 MMA 在 std / 32k / 65k 上的 prefill/decode。
