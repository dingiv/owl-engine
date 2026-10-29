# CUTLASS 虚拟依赖裁剪（2026-10-28）

> 结论先行：`crates/kernels/cu/cutlass/` 从 **517 文件 / 17 MB** 砍到
> **243 文件 / 8.7 MB**（nvfp4 TU 的闭包 513 → 238 头），**代码生成零变化**
> —— 同一个 kernel、同一份 `__launch_bounds__`、同一条
> `mma.sync.kind::mxf4nvf4.block_scale`，decode 8.487 → 8.492 ms/step。
> 口径与重裁步骤写在 `crates/kernels/cu/cutlass/TRIMMED.md`。

## 一、为什么可以砍

全仓只有 **两个** TU 消费 CUTLASS：

1. `cu/nvfp4/nvfp4_gemm.cu` —— sm120 blockscaled FP4 mainloop + TMA epilogue；
2. `cu/flash_attn/owl_fa2.cu`（vLLM FA2 fork）—— 只用 `cutlass/layout/*`、
   `array.h`、`numeric_types.h`、`numeric_conversion.h` 的基础类型。

**架构裁决**（用户）：30/40 系走 Marlin/GGUF 的自写核，根本不走 CUTLASS；
50 系（sm120）唯一需要的 CUTLASS 计算路径就是块缩放 FP4。所以
sm70/75/80/89/90 的计算面、sm100/103/107 的 UMMA 全家、以及所有
sparse/array/emulated/interleaved/planar/mixed-input 变体都是死重。

## 二、关键认识：删死文件 = 零收益

裁剪前实测：517 个文件里 **513 个在预处理器闭包里** —— 上一轮已经按
`nvcc -M` 裁到"最小可达闭包"，再删文件就会编译失败。

真正的收益来自**改 include 半径**（7 处，每处都在源码里留了 `owl 裁剪点`
注释）。按收益排序：

| 裁剪点 | 收益 |
|---|---|
| `gemm/collective/collective_mma.hpp`：40+ mainloop → 只留 sm120 blockscaled | **约 60 个头 / 2.6 MB**（sm100 UMMA 全家 + sm103/sm107） |
| `nvfp4_gemm.cu`：去掉 `gemm_universal_adapter.h`（2.x device 层），自写启动器 | **约 160 个头 / 3.8 MB**（2.x threadblock/warp/epilogue 树 + sm70/sm100/sm107 kernel 头） |
| `epilogue/collective/collective_epilogue.hpp`：五份 → sm90 TMA warp-specialized | 约 10 头 |
| `gemm/collective/collective_builder.hpp`：30 个 builder → 2 | 约 40 头 |
| `epilogue/collective/collective_builder.hpp`：五份 → sm120（+ sm90 glue） | 约 20 头 |
| `gemm/collective/builders/sm90_common.inl`：去 sparse GMMA traits | 1 MB（`mma_sm90_gmma_sparse.hpp`） |
| `cute/atom/mma_atom.hpp`：全代际 traits → sm75（FA2 要）+ sm80/90/gmma/120 | 约 20 头 |

自写启动器只有 ~30 行：`Kernel::initialize_workspace` →
`to_underlying_arguments` → 自己 `<<<>>>`。`__launch_bounds__` 与 CUTLASS
的 `device_kernel<>` **逐字对齐**（它参与寄存器分配），否则"只换启动器"就
会悄悄改变代码生成。

## 三、两个坑

1. **`nvcc -M` 看不到符号级依赖**。FA2 的 `kernel_traits.h` 用
   `SM75_16x8x8_F32F16F16F32_TN` 当 MMA atom；把它从 traits umbrella 里
   摘掉后 `-M` 依旧干净，真编译才炸。**闭包必须用真编译验证**，`-M`
   只用来算候选集。
2. **"只为几个函数"的整文件保留是有意的**：
   `sm100_compute_stage_count_or_override_blockscaled`（在
   `sm100_blockscaled_umma_builder.inl`）与 `PersistentTileSchedulerSm100`
   （在 `sm100_tile_scheduler.hpp`）在模板定义处是**非依赖名**，缺了直接
   编译错；而它们引的 UMMA trait 本来就在 sm120 的块缩放布局链上 ⇒ 保留
   成本接近 0。

## 四、验收（同机同档，裁剪前后）

| 面 | 前 | 后 |
|---|---:|---:|
| decode | 8.487 ms/step（118.3 tok/s） | **8.492 ms/step（117.8 tok/s）** |
| prefill 1374 tok | 8.8~11.1k tok/s（中位 9849） | 10.5~13.8k tok/s（中位 10644） |
| 装配期 oracle（5 档几何） | `max|err| = 0` | `max|err| = 0`（逐档相同） |
| 4B 文本 | `17*23 → 391`，3.81 B/tok | `391`，3.76 B/tok |
| 9B GGUF 生产档 | 137/94/757 B | **137/94/757 B（逐字节一致）** |

## 五、下一步（还没做）

- 若要把最后两个"整文件 glue"也去掉：把 `ss_smem_selector` /
  `sm90_get_epilogue_smem_swizzle_layout_atom` / stage-count 启发式搬成
  owl 侧 shim（约 150 行），收益再 ~1.2 MB（sm90 GMMA 稠密链）。
- sm100 UMMA trait 的 sparse 分支（`mma_sm90_gmma_sparse.hpp` 1 MB）目前
  被 `mma_traits_sm100.hpp` 的 `Sp` 布局 atom 需要 —— 想砍得连那批 atom
  一起自己写，风险/收益不划算（这些布局必须与硬件逐位一致）。
