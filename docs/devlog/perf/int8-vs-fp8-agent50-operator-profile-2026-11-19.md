# INT8 KV vs FP8 KV：50k agent 追问的算子级占比（2026-11-19）

> 口径：2×3080 20G TP2，Qwen3.8-27B W4A8，DFlash2 d7；
> 固定 50k 前缀 + 两轮追问（cached ≈99.8%），CUPTI `perf,cupti` +
> `--perf-level gpu`，取 `/debug/perf/epoch/json` 的每轮 epoch。
> 同 prompt/salt；冷轮只出 8 token 建前缀，两轮追问各 128 token。

## 1. 家族占比（GPU 图叶节点 self 之和）

分母 = `gpu.graph.<gid>.*` 的 verify + propose 两族之和；不含 fold、
select、prefill 等旁路。CUPTI 单 tag 有 4096 样本封顶，绝对值是下限，
**同口径横向比较可用，不要当完整总量**。

| 轮 | KV | verify | propose |
|---|---|---:|---:|
| T1 | INT8 | **81.8%** | 18.2% |
| T2 | INT8 | **82.4%** | 17.6% |
| T1 | FP8 | **77.4%** | 22.6% |
| T2 | FP8 | **78.9%** | 21.1% |

图 id（本轮）：INT8 verify = 9+11，propose = 21+23；FP8 verify = 9+14，
propose = 21+23。fold 图另计（INT8 14+17，FP8 12+17）。

## 2. verify 族内部（T1 / T2）

### INT8 KV

| 核 | T1 | T2 | 占 verify |
|---|---:|---:|---:|
| target Marlin | 332.6 ms | 332.4 ms | 29.5% / 31.7% |
| `owl_int8_decode_pth2_pc_split` | **260.4 ms** | **231.0 ms** | **23.1% / 22.1%** |
| `owl_comm_ar_barrier_f16` | 172.9 ms | 162.8 ms | 15.3% / 15.6% |
| GDN recurrence | 86.1 ms | 73.8 ms | 7.6% / 7.0% |
| `owl_gemm_i8w_a16_tc_f16` | 65.9 ms | 56.5 ms | 5.8% / 5.4% |
| GDN conv | 31.3 ms | 31.3 ms | 2.8% / 3.0% |
| quant_int8_row | 20.4 ms | 20.3 ms | 1.8% / 1.9% |

### FP8 KV

| 核 | T1 | T2 | 占 verify |
|---|---:|---:|---:|
| target Marlin | 332.9 ms | 332.9 ms | 31.2% / 33.3% |
| `owl_comm_ar_barrier_f16` | 175.6 ms | 176.0 ms | 16.5% / 17.6% |
| `BatchPrefillWithPagedKVCacheKernel`(FI) | **158.2 ms** | **136.7 ms** | **14.8% / 13.7%** |
| GDN recurrence | 103.6 ms | 90.0 ms | 9.7% / 9.0% |
| `owl_gemm_i8w_a16_tc_f16` | 79.0 ms | 68.7 ms | 7.4% / 6.9% |
| GDN conv | 31.3 ms | 31.3 ms | 2.9% / 3.1% |
| quant_int8_row | 20.3 ms | 20.3 ms | 1.9% / 2.0% |

**读法**：两边 verify 的第一大头都是 target Marlin（~30–33%），
第二大头分叉——INT8 是 PTH attention（22–23%），FP8 是 FI attention
（14–15%）+ AR barrier（16–18%）。INT8 的 attention 比 FP8 多约
**8–9 个百分点 / 1.5× 时间**，但 PTH 不是 verify 的绝对大头。

## 3. propose 族内部（T1 / T2）

| 核 | INT8 T1 | INT8 T2 | FP8 T1 | FP8 T2 |
|---|---:|---:|---:|---:|
| `owl_naive_attn_nc_tiled_bf16`(draft attn) | 73.6 ms / 29.3% | 65.0 ms / 29.1% | 90.7 ms / 29.1% | 77.7 ms / 29.1% |
| draft Marlin | 67.3 ms / 26.8% | 59.4 ms / 26.6% | 83.2 ms / 26.7% | 71.3 ms / 26.7% |
| `owl_gemm_i8w_a16_tc_f16` | 63.9 ms / 25.5% | 56.4 ms / 25.3% | 78.8 ms / 25.3% | 67.6 ms / 25.3% |
| bf16 gemm | 18.4 ms / 7.3% | 16.2 ms / 7.3% | 22.7 ms / 7.3% | 19.5 ms / 7.3% |
| `owl_comm_ar_barrier_bf16` | 15.2 ms / 6.1% | 15.2 ms / 6.8% | 20.7 ms / 6.7% | 18.2 ms / 6.8% |
| dflash_conv | 2.4 ms / 1.0% | 2.1 ms / 0.9% | 3.0 ms / 0.9% | 2.5 ms / 0.9% |

**读法**：propose 的内部结构在 INT8/FP8 下几乎同形——draft attention /
draft Marlin / i8w GEMM 三份各 ~25–29%，合计 ~81%。KV dtype 只改变
verify 的 attention 项，不改变 propose 的形状。

## 4. 口径警告

1. **`gpu.graph` 是设备 self 时间，`spec.verify`/`spec.propose` 是 host
   墙时间**；两者不要相加。fold 的 host 账（`spec.fold` 6–8ms/次）明显
   大于图内 GDN fold（~2.6ms/次），说明 fold 还有图外搬运，不在本表。
2. CUPTI 单 tag 4096 样本封顶：Marlin/GDN 这类每轮几千发的 tag 会低估；
   表中百分比是**下限 + 同口径可比**，不是绝对 GPU 总时间。
3. 本轮 INT8 用当前 main 的 PTH pair verify（实验臂），FP8 用 FI；
   若最终上线 INT8 是 GQA TC 或另一变体，需要按该族重跑同一探针。
4. `decode_gen` 用 `completion_tokens / (total − TTFT)`；本轮 INT8
   90.9/110.2 t/s，FP8 78.3/87.7 t/s，但两臂 AL/轮数不同，**不能**据此
   直接下速度结论；速度 A/B 仍用固定输出/固定 AL 的独立实验。
