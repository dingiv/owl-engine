# `cu/cutlass/` —— **裁剪过的** vendored CUTLASS（4.8.0 快照）

> 本目录**不是** NVIDIA 的完整发行版。它是"我们这个仓库实际用到的闭包"，
> 由 `crates/kernels/build.rs` 的两条 `-I` 消费（nvfp4 + flash_attn）。
> 全量覆盖/升级 CUTLASS 后，**必须按本文重裁一次**，否则体积会回到 28 MB。

## 1. 我们到底用它做什么（唯二消费者）

| 消费者 | 用了 CUTLASS 的什么 | 说明 |
|---|---|---|
| `cu/nvfp4/nvfp4_gemm.cu` | **sm120 blockscaled FP4 mainloop** + TMA epilogue（`Tile 128×128×256`） | 全仓唯一的 CUTLASS 计算路径。30/40 系走 Marlin/GGUF 的自写核，**不碰这里** |
| `cu/flash_attn/owl_fa2.cu`（vLLM FA2 fork） | 只用 `cutlass/layout/*`、`cutlass/array.h`、`cutlass/numeric_types.h`、`cutlass/numeric_conversion.h` 的**基础类型** | 它自带 MMA 调度；对 CUTLASS 无计算依赖 |

因此裁剪口径 = **sm120 稠密 blockscaled 一条路 + FA2 的类型面**。sm70/75/80/89/90 的
计算路径、sm100/103/107 的 UMMA 全家、全部 sparse/array/emulated/interleaved/
planar/mixed-input 变体，都不在本仓的端到端链路里。

## 2. 裁剪点（改的是"include 半径"，不是删死文件）

当前 517 个文件里曾经 513 个都在预处理器闭包里 —— 单纯删文件等于零收益。
真正的收益来自把 **umbrella 头**换成精确清单（每处都留了 `owl 裁剪点` 注释）：

| 文件 | 原版 include | 现留 |
|---|---|---|
| `cutlass/gemm/collective/collective_mma.hpp` | 40+ mainloop（sm70/80/90 GMMA 全族 + sm100 UMMA 全族 + sm103/sm107/sm120） | `sm120_blockscaled_mma_tma.hpp` |
| `cutlass/epilogue/collective/collective_epilogue.hpp` | sm70/sm90/sm100 全部 epilogue collective | `default_epilogue(_array)` + `sm90_epilogue_tma_warpspecialized(_bias_elementwise)` |
| `cutlass/gemm/collective/collective_builder.hpp` | 30 个 builder `.inl` | `sm100_blockscaled_umma_builder.inl`（只为 `sm100_compute_stage_count_or_override_blockscaled`）+ `sm120_blockscaled_mma_builder.inl` |
| `cutlass/epilogue/collective/collective_builder.hpp` | sm90/sm100/sm103/sm107/sm120 五份 | `gemm/.../sm90_common.inl`（`ss_smem_selector`）+ `builders/sm90_builder.inl`（`sm90_get_epilogue_smem_swizzle_layout_atom`）+ `builders/sm120_builder.inl` |
| `cutlass/gemm/collective/builders/sm90_common.inl` | 含 sparse GMMA traits（→ 1 MB 的 `mma_sm90_gmma_sparse.hpp`） | 去掉 sparse 那一行 |
| `cute/atom/mma_atom.hpp` | sm61/70/75/80/89/90/gmma/100/107/120/120_sparse 全量 traits | sm75（**FA2 要用**）+ sm80 + sm90 + sm90_gmma + sm120 |
| `nvfp4_gemm.cu` 自身 | `cutlass/gemm/device/gemm_universal_adapter.h`（2.x device 层，连带 sm70/sm100/sm107 kernel 头） | **自写启动器**（`Kernel::initialize_workspace` → `to_underlying_arguments` → 自己 `<<<>>>`，`__launch_bounds__` 与 CUTLASS 版逐字一致） |

保留 `sm100_blockscaled_umma_builder.inl` / `sm100_tile_scheduler.hpp` 这类
"只为几个函数/声明"的文件，是**有意的**：那两个名字在模板定义处是**非依赖名**
（缺了直接编译错），而它们各自引的 `copy_sm100` / UMMA trait 本来就在
sm120 的块缩放布局链上。

## 3. 重裁步骤（升级 CUTLASS 后照做）

```bash
cd crates/kernels
# ① 两个消费者的预处理器闭包（并集）
nvcc -M -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_120a \
     -Icu/cutlass/include -Icu cu/nvfp4/nvfp4_gemm.cu > /tmp/nv.mk
nvcc -M -O3 -std=c++17 -arch=sm_86 \
     -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_SOFTCAP \
     -DFLASHATTENTION_DISABLE_LOCAL -DFLASHATTENTION_DISABLE_ALIBI \
     -Icu/flash_attn/compat -Icu/flash_attn/vendor/flash_attn/src \
     -Icu/cutlass/include -Icu/flashinfer/include -Icu \
     cu/flash_attn/owl_fa2.cu > /tmp/fa2.mk
# ② 取并集 → 删闭包外
# ③ **重新应用 §2 的 include 裁剪**（这几处不是自动的）
# ④ `-M` 只跑预处理器，**不查符号** —— 必须真编译两个 TU：
#    cargo build --release -p server --no-default-features --features arch-sm120
#    缺哪个头就补哪个（FA2 曾因丢了 `mma_traits_sm75.hpp` 在
#    `SM75_16x8x8_F32F16F16F32_TN` 上炸，而 `-M` 完全看不出来）
# ⑤ 门：装配期 oracle `max|err|=0`、4B NVFP4 一发对话、9B GGUF 输出逐字节一致
```

## 4. 当前规模与实测（2026-10-28，RTX 5070 Ti）

| 口径 | 裁剪前 | 裁剪后 |
|---|---:|---:|
| 文件数 | 517 | **243** |
| 体积 | 17 MB | **8.7 MB** |
| nvfp4 TU 闭包 | 513 头 / 15.7 MB | 238 头 / 8.4 MB |

性能（同一 4B NVFP4 档，裁剪前后同机同档）：

| 面 | 前 | 后 | 判据 |
|---|---:|---:|---|
| decode | 8.487 ms/step | **8.492 ms/step** | ±3% ✓ |
| prefill（1374 tok） | 8.8~11.1k tok/s（中位 9849） | 10.5~13.8k tok/s（中位 10644） | 不劣化 ✓ |
| 装配期 oracle | `max|err|=0` ×5 档 | `max|err|=0` ×5 档 | 逐档相等 ✓ |
| 文本 | `17*23→391`、3.81 B/tok | `391`、3.76 B/tok | 正常 ✓ |
| 9B GGUF | 137/94/757 B | 137/94/757 B | 逐字节一致 ✓ |

一句话：**代码生成没变**（同一个 kernel、同一份 `__launch_bounds__`、同一条
`mma.sync.kind::mxf4nvf4`），变的只是"编译器要读多少头"。
