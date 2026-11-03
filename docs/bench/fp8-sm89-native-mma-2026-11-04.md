# 40/50 系原生 FP8 MMA prefill — 落地与实测（2026-11-04）

> 目标：40/50 系（Ada SM89 / Blackwell SM120）的 FP8 KV prefill **不再把 K/V 反量化成 half**，
> 直接走 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`。
>
> 基线（同一张卡、旧 dequant 路径 / f16 shadow）：见
> [`fp8-sm89-dequant-baseline-2026-11-04.md`](fp8-sm89-dequant-baseline-2026-11-04.md)。

## 1. 结论

- 原生路径已接通；**`KernelPick` 按运行期 `OpEnv.hw.arch` 选臂**：
  Ada/SM89 或 Blackwell/SM120 + FP8 主池 → `PREFILL_FI_FP8MMA`，其余代际
  仍走旧 `PREFILL_FI_FP8KV`（e4m3→half 反量化）。不再由编译 feature 自行猜。
  **SM120 共用同一核**（上游 FP8 MMA 指令门 >=890；实测数据仅 4080，50 系
  待真机验收）。
- 原生路径 **长上下文 +16~20%**：32k **2844.5 vs 2379.3 t/s**（+19.6%），
  65k **2434.6 vs 2099.4**（+16.0%）。
- std 去掉首个冷启动 512 点后（1024–8192）平均 **3113.4 vs 3029.8 t/s**（+2.8%）；
  512 首点与基线同为冷启动（1698 vs 1699）。
- 同时超过 f16 shadow 基线：32k +5.1%、65k +7.1%。
- decode 不变（原生只改 prefill）：std 37.92 t/s，32k/65k 37.97/35.61 t/s。
- 质量：1k/8k/65k needle 全 RETRIEVED；视觉 grounding + `image_evict`
  36/36、hit@16px 100%。

## 2. 实现

### 2.1 核路径

上游 `repos/flashinfer` 0.7.3（`8826c9fa3`）的 FP8 MMA 增量已最小移植到
vendored `prefill.cuh`：

- `KernelTraits::USE_FP8_MMA` / `INVALID_FP8_MMA`
- `compute_qk_fp8`：Q/K 直接 `ldmatrix` + `m16n8k32` e4m3 MMA
- `vosplit_softmax_store_p` 的 e4m3 P 分支（P×256）
- `vosplit_compute_pv_fp8`：e4m3 P × e4m3 V
- `vosplit_write_o`：归一化除 256 + 4 维连续写
- paged 主循环 / smem 预算 / dispatcher 的 VO-split 判别

新增独立 TU：
`crates/kernels/cu/flashinfer/owl_fi_attention_fp8mma.cu`

- `owl_fi_quant_f16_to_fp8`：Q f16 → e4m3（scale=1.0）
- `owl_fi_prefill_run_fp8mma`：
  `owl_fi_run_impl<__nv_fp8_e4m3, __nv_fp8_e4m3>`

选核面（`crates/kernels/src/driver.rs::attn::fi_prefill`）：

- `env.hw.arch ∈ {Sm89, Sm120}` 且 `fp8kv=true` → `PREFILL_FI_FP8MMA`；
- SM86/其他代际 → 旧 `PREFILL_FI_FP8KV`；非 FP8 池 → `PREFILL_FI`；
- 新增 `FiPrefillFp8mma` 声明（同参数/池/形状族），`foreign_sig` 同 SIG。

Rust 侧（`crates/kernels/src/server/flashinfer.rs`）：

- `parse_prefill` 从入口名解析 `fp8mma`；`run` 按它写 Q8 scratch 再调原生入口；
- `ensure_q8` 复用/增长 scratch，捕获窗内未预留则结构化拒；
- `plan_sync` 用引擎投影的 `FlashinferCfg.fp8_native`（同样来自 `env.hw.arch`）
  选择原生 plan；旧入口保留。

### 2.2 为什么固定 CTA_TILE_Q=16

上游 `INVALID_FP8_MMA` 要求 `NUM_WARPS_Q == 1`。实测中放开到
`NUM_WARPS_Q=4`（CTA64）会在随机 Q/K 上产生 NaN/零行；固定 CTA16 后
`cos=0.999976`、`mean|Δ|=2.6e-3`，故保留上游约束。

计划侧新增：

- `scheduler.cuh`：`force_cta16` 参数贯穿
  `PrefillPlanWorkspaceSize` / `PrefillPlan` / `PrefillSplitQOKVIndptr`；
- `owl_fi_prefill_plan_fp8mma`：`kv_bytes=1` + `force_cta16=true`；
- split-kv 沿用上游默认（CTA16 下 T=1/3072 已对拍）。

### 2.3 构建门

- 新 TU 只在 `single_arch() >= Arch::Sm89` 时编入 FI `.a`（sm89/sm120 都编）；
- 运行期选臂由 `KernelPick` 决定；`#[cfg(any(arch-sm89, arch-sm120))]` 只守
  **设备符号可用性**（选到 native 而二进制未编 = 结构化拒，不静默回退）；
- `fi_pick_uses_runtime_arch` 单测锁死：SM86 → 旧臂，SM89/SM120 → 原生臂；
- sm86 构建不编新 TU，仍走旧 dequant 路径（已构建验证）。
- 50 系（sm120）不在本次打开范围；`mma.cuh` 指令门虽然是 `>=890`，
  但 profile/质量门未覆盖 sm120。

构建：

```bash
GDN_TRITON_PY=/home/div/Documents/codes/packages/owl/crates/kernels/.venv/bin/python \
CARGO_TARGET_DIR=target-sm89 cargo build --release -p server \
  --no-default-features --features arch-sm89
# 50 系换成 --features arch-sm120
```

## 3. 单算子对拍

`fi_prefill_probe`，4080，T=128/ctx=512：

| 量 | 原生 FP8 MMA |
|---|---:|
| `max|FI−host|` | 0.0277 |
| `cos` | 0.9999758 |
| `mean|Δ|` | 2.61e-3 |
| 零值 | 0 / 786432 |

误差来源 = Q 与 P 各多一次 e4m3 量化；probe 对 arch-sm89/120 + fp8 KV
使用 `3e-2 / cos 0.9999 / mean 5e-3` 的口径（f16 池仍用历史 1e-2 门）。

## 4. 整模质量

`tools/fixture/int8_text_gate.py`（greedy，spec off）：

| 目标长度 | 结果 | 耗时 |
|---:|---|---:|
| 1k | RETRIEVED | 0.6s |
| 8k | RETRIEVED | 2.7s |
| 32k | RETRIEVED | 11.4s |
| 65k | RETRIEVED | 27.0s |

视觉（`tools/vision_grounding.py`，1280×720，12 方块）：

- 无 evict：36/36，hit@16px 100%，slope 0.9974/0.9991；
- 带 `owl_evict`：36/36，hit@16px 100%，slope 0.9990/0.9997。

## 5. 性能（4080 SUPER，`llm_speedtest`，单并发，spec off）

### 5.1 std（512/1024/2048/4096/8192，out 256）

| 臂 | prefill avg t/s | decode avg t/s | TTFT avg ms |
|---|---:|---:|---:|
| 旧核内 dequant（含冷 512） | 2763.81 | 37.48 | 1073.89 |
| f16 shadow（含冷 512） | 2812.18 | 37.90 | — |
| **原生 FP8 MMA（含冷 512）** | **2830.32** | **37.92** | **1038.87** |
| **原生 FP8（1024–8192）** | **3113.4** | **37.90** | — |

分点（原生，含冷 512）：1698.2 / 2973.8 / 3184.0 / 3128.5 / 3167.1 t/s；
TTFT 303.1 / 346.1 / 645.2 / 1311.3 / 2588.6 ms。512 为首次请求/图 warmup，
与基线同注，不用于均值判读。

### 5.2 长上下文（out 16）

| 长度 | 旧 dequant | f16 shadow | **原生 FP8** | vs 旧 | vs shadow |
|---:|---:|---:|---:|---:|---:|
| 32768 | 2379.26 | 2705.61 | **2844.45** | +19.6% | +5.1% |
| 65536 | 2099.38 | 2274.13 | **2434.56** | +16.0% | +7.1% |

TTFT：32k **11.52s**（旧 13.77s），65k **26.92s**（旧 31.22s）。
decode：37.97 / 35.61 t/s，与旧路径同级。

## 6. 约束与后续

1. **CTA16 是当前验证边界**：multi-Q-warp（CTA64/128）在 e4m3 Q 上
   尚未跑通；若将来要提速，优先查 `vosplit_*_fp8` 的 `warp_q_idx`
   布局/`p_smem` 行映射，别直接删 `INVALID_FP8_MMA`。
2. **split-kv** 默认保留；已对拍 T=1/ctx=3072，大 T 形状通常不触发拆分。
3. **Q scale = 1.0**：qk-norm 后 Q 的 RMS≈1，e4m3 上限 448；若将来换
   模型/层出现离群值，应把 q_scale 折进 `sm_scale`，不要静默 clamp。
4. **50 系另案**：sm120 的 FP8 MMA 指令形式/质量门未验证，不要复制 profile。
5. **30 系不变**：arch-sm86 构建仍走旧 dequant 路径，INT8 默认与本次无关。

## 7. 文件地图

| 文件 | 变更 |
|---|---|
| `crates/kernels/cu/flashinfer/include/flashinfer/attention/prefill.cuh` | FP8 MMA 增量 + CTA_TILE_KV/occupancy 适配 |
| `crates/kernels/cu/flashinfer/include/flashinfer/attention/scheduler.cuh` | `force_cta16` 贯穿 plan |
| `crates/kernels/cu/flashinfer/owl_fi_attention.cu` | `owl_fi_prefill_plan_fp8mma` |
| `crates/kernels/cu/flashinfer/owl_fi_attention_common.h` | `DTypeQ` 参数化 |
| `crates/kernels/cu/flashinfer/owl_fi_attention_fp8mma.cu` | 新 TU：Q 量化 + 原生 run |
| `crates/kernels/build.rs` | arch>=sm89 编入新 TU |
| `crates/kernels/src/family/flashinfer.rs` | FFI 声明/入口 |
| `crates/kernels/src/server/flashinfer.rs` | Q8 scratch + native 选择 + plan 选择 |
| `crates/kernels/tests/fi_prefill_probe.rs` | arch-sm89/120 误差门 |
| `profiles/qwen38_27b-4080.toml` | 注释更新（native 路径） |
| `docs/bench/fp8-sm89-dequant-baseline-2026-11-04.md` | 旧路径基线（对照） |
