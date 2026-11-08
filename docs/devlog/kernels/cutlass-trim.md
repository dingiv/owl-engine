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

## 五、第二轮（2026-10-29）：把"只用到几个符号"的大头换成 owl shim

第一轮之后重新量：**闭包外已零文件**（243/243 都在闭包里），只能继续改
include 半径。最大一块是 **sm90 GMMA 家族 2.85 MB（33%）**，而我们对它的
需求量是：**8 个稠密 smem 布局原子 + 8 个稀疏变体 + 3 个小函数**
（`ss_smem_selector` / `sm90_get_epilogue_smem_swizzle_layout_atom` /
`sm90_cluster_shape_to_tma_atom`）+ `Major` 枚举 + `DescriptorIterator`。

于是把这一小撮逐字搬进 owl 侧 `cu/nvfp4/owl_cutlass_gmma_shim.hpp`（纯类型
别名 + 纯 `constexpr`，零代码生成），让 **12 个文件 / 3.04 MB** 从树里消失。

| 口径 | 原始 | 第一轮 | **第二轮** |
|---|---:|---:|---:|
| 文件数 | 517 | 246 | **235** |
| 体积 | 17 MB | 8.8 MB | **5.9 MB** |
| nvfp4 闭包 | 513 头 / 15.7 MB | 238 / 8.4 MB | **228 / 5.41 MB** |
| fa2 闭包 | — | 129 / 5.34 MB | **127 / 2.75 MB** |

### 5.1 两条纪律（不然会静默错）

1. **shim 必须先于任何 `cute/…` 头**：上游 `copy_traits_sm90_tma.hpp` 在
   *定义处* 引用 `GMMA::Layout_MN_SW128_Atom`（非依赖名）⇒ `nvfp4_gemm.cu` 与
   `owl_fa2.cu` 两个 TU 都在最前面 include 它。
2. 上游靠 `namespace cute { using namespace SM90; }` 才能写 `cute::GMMA::…`；
   原头被裁后这条可见性要在 shim 里补回来。

### 5.2 验收：用 PTX 证明"代码生成没变"

跑分会被卡上其它负载干扰（本轮 prefill 实测 10.6~16.3k tok/s 波动，
decode 8.347 ms 稳定），所以这轮的硬判据是**编译产物比对**：

- 两个 TU 各跑 `nvcc -ptx`，去掉 `//` 与 `_INTERNAL_<hash>_`（随 include 集合
  变化的 TU id）后 **diff = 0 行**；
- `.o` 大小逐字节相同（236,056 B），差异字节只有那些 TU-id 符号串。

同时：装配期 oracle 5 档仍 `max|err| = 0`；4B 文本 `391` / 3.81 B/tok；
9B GGUF 1、2 号探针逐字节一致（137/94），3 号 757 → 745 字节且**同二进制可重复**
—— 两个 CUDA TU 已证明 PTX 相同，该差异来自这一轮之间上游合入的引擎改动
（scheduler/vision/并发），与本裁剪无关。

## 六、第三轮（2026-10-29）："用到什么就抄什么" —— 2.x arch 整支 + StreamK 下树

第二轮之后剩下的体积**全是真被编译的代码**，于是换了个问法：不再问"哪个文件
没被 include"，而是问 **"这个文件里我们到底用到了哪几个符号"** —— 用到的
抄成 owl 侧小头，没用到的那一支整批删。这是用户定的"正向增量"路线：
**从"我们需要什么"出发，把要的东西复刻过来**，而不是猜哪个文件可以删。

### 6.1 目标：`cutlass/arch/mma.h` 那条链

```
sm90_tile_scheduler_stream_k.hpp  (StreamK；本仓只用 PersistentScheduler)
  → block_striped.h → wmma_array.h → arch/mma.h
      → arch/mma_sm{50,60,61,70,75,80,89,90}.h
      → arch/mma_sparse_sm{80,89}.h · arch/wmma{,_sm70,_sm75}.h
      → mma_sm50.h → quaternion.h → cutlass/matrix.h   ← 355 KB
```

我们从 `arch/mma.h` 真正用到的只有**两个空标签**：`arch::OpClassTensorOp` /
`arch::OpClassBlockScaledTensorOp`（epilogue 与 blockscaled 主循环的算子类别）。
它们进了 **`cu/nvfp4/owl_cutlass_arch_tags.hpp`**（纯 struct，零代码生成）。
`arch/mma_sm90.h` 只被 kernel 头引进来、内容是 2.x 的 `arch::Mma<SM90_*>`
—— 我们用的是 **cute atom**（`cute/arch/mma_sm120.hpp`），一个没用到 ⇒
连同那行 include 一起删。

删掉 20 个文件 / ≈ 668 KB（名单见 `TRIMMED.md` §2 第 3 轮），另改 2 个文件：
`gemm/kernel/tile_scheduler.hpp`（去掉 2 行 StreamK include + 5 个 StreamK 选择器
特化）、`gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`（去掉
`arch/mma_sm90.h`）。

### 6.2 规模与验收（本轮全部实测）

| 口径 | 第 2 轮 | **第 3 轮** | **第 4 轮（现在）** |
|---|---:|---:|---:|
| 文件数 | 235 | **211** | **208** |
| 体积 | 5.9 MB | **5.15 MB** | **4.74 MB** |
| nvfp4 TU 闭包 | 228 头 / 5.41 MB | **205 头 / 4.69 MB** | **202 头 / 4.65 MB** |
| fa2 TU 闭包 | 127 头 / 2.75 MB | **127 头 / 2.75 MB**（未动） | **127 头 / 2.75 MB** |
| 两 TU 并集 | 231 头 / 5.48 MB | **209 头 / 4.76 MB** | **206 头 / 4.72 MB** |

第 4 轮（同门刀法的最后一勺）：`GroupScheduler` / `StaticPersistentScheduler`
三支选择器（≈ 45 KB）—— 本仓只用 `PersistentScheduler`，选择器壳子保留、
`using Scheduler = …` 换成注释，永不会实例化。

- **PTX 逐行相同**：`nvfp4_gemm.cu` 前后各出一次 `-ptx`，去 `//` 与
  `_INTERNAL_<hash>_` 后 **diff = 0 行**，文件同为 **173,768 B**。
- 装配期 oracle 5 档 **`max|err| = 0`**；4B NVFP4 `391`、3.81 B/tok、
  **decode 8.258 ms/step**（第 2 轮 8.347，判据 ≤ 8.49 ✓）；
  9B GGUF 探针 **137 / 94 / 745** 字节（与第 2 轮逐字节一致）。

### 6.3 同轮的一个**否掉的方案**（记录以免重走）

“正向”的另一读法是**自研整个主循环**（裸 PTX atom + 自写 cp.async 流水 +
自有 SF 布局），曾完整实现并数值全绿（`nvfp4_owl.cu` + `nvfp4_mma_sm120.cuh`）。
但**性能差 1.3~2.1 倍**（单核微基准：prefill 272 vs CUTLASS 554 TFLOP/s；
decode 权重流 200~366 vs 305~546 GB/s）—— 差在 CUTLASS 的 TMA 多级流水与
warp-specialization，补齐它就是“重新研发”。**用户裁决：不重写，尊重现有实现**
⇒ **两个文件已删除**（它们与 `nvfp4_gemm.cu` **FFI 符号同名**，留在树里是
“被误加进 `build.rs` 就重复定义”的隐患）。事实与数据集保留：
`sm120-fp4-mma-facts.md` + `cu/probe/fp4_mma_sm120{,_sf}_probe.cu`（裸 PTX 探针，
不进构建、不引 CUTLASS）。

## 七、下一步（还可以砍什么）

- 剩下 4.7 MB 的**大头都是真被实例化的代码**：`copy_sm100.hpp`(356K)、
  `numeric_conversion.h`(287K)、`mma_traits_sm100.hpp`(172K)、
  `copy_traits_sm100.hpp`(146K)、`sm90_callbacks_tma_warpspecialized.hpp`(112K)。
  砍它们要**逐个特化做符号级外科**（拿掉文件前必须先把引用它的
  `Copy_Traits<…>` / `NumericArrayConverter<…>` 特化一并剪掉，否则解析就报错），
  每切一刀都要一次真编译 + PTX 比对。
- 机械可做的还有一档：`GroupScheduler` / `StaticPersistentScheduler` /
  `DynamicPersistentScheduler` 三支选择器（≈45 KB），与 StreamK 同律。
- **不建议**动 `cute/atom/mma_traits_sm100.hpp` 的 UMMA 布局代数 —— 那是
  "重写硬件布局描述"，风险与收益不匹配。
