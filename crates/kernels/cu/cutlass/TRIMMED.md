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

每一轮的实测都指向同一件事：**树上每个文件都在闭包里** —— 单纯删文件等于零收益，
收益全部来自把 **umbrella 头**换成精确清单 / 把"只用到几个符号"的大头换成 owl shim。
每处都留了 `owl 裁剪点` 注释。

### 第 1 轮（2026-10-28）

| 文件 | 原版 include | 现留 |
|---|---|---|
| `cutlass/gemm/collective/collective_mma.hpp` | 40+ mainloop（sm70/80/90 GMMA 全族 + sm100 UMMA 全族 + sm103/sm107/sm120） | `sm120_blockscaled_mma_tma.hpp` |
| `cutlass/epilogue/collective/collective_epilogue.hpp` | sm70/sm90/sm100 全部 epilogue collective | `default_epilogue(_array)` + `sm90_epilogue_tma_warpspecialized(_bias_elementwise)` |
| `cutlass/gemm/collective/collective_builder.hpp` | 30 个 builder `.inl` | `sm100_blockscaled_umma_builder.inl`（只为 `sm100_compute_stage_count_or_override_blockscaled`）+ `sm120_blockscaled_mma_builder.inl` |
| `cutlass/epilogue/collective/collective_builder.hpp` | sm90/sm100/sm103/sm107/sm120 五份 | 第 2 轮起只剩 `builders/sm120_builder.inl` |
| `cute/atom/mma_atom.hpp` | sm61/70/75/80/89/90/gmma/100/107/120/120_sparse 全量 traits | sm75（**FA2 要用**）+ sm80 + sm120（第 2 轮把 sm90/gmma 也移除了） |
| `nvfp4_gemm.cu` 自身 | `cutlass/gemm/device/gemm_universal_adapter.h`（2.x device 层，连带 sm70/sm100/sm107 kernel 头） | **自写启动器**（`Kernel::initialize_workspace` → `to_underlying_arguments` → 自己 `<<<>>>`，`__launch_bounds__` 与 CUTLASS 版逐字一致） |

保留 `sm100_blockscaled_umma_builder.inl` / `sm100_tile_scheduler.hpp` 这类
"只为几个函数/声明"的文件，是**有意的**：那两个名字在模板定义处是**非依赖名**
（缺了直接编译错），而它们各自引的 `copy_sm100` / UMMA trait 本来就在
sm120 的块缩放布局链上。

### 第 2 轮（2026-10-29）：把"只用到几个符号"的大头换成 owl shim

第 1 轮之后重新量：**闭包外已零文件**（243/243 全在闭包里），于是只能继续动
include 半径。最大的两块是 **sm90 GMMA 家族 2.85 MB（占全树 33%）** —— 而我们
对它的需求量小到可笑：**8 个稠密 smem 布局原子 + 8 个稀疏变体 + 3 个小函数**。

把它们逐字搬进 **owl 侧 shim** `cu/nvfp4/owl_cutlass_gmma_shim.hpp`（纯类型别名 +
纯 `constexpr`，零代码生成），然后让这 12 个文件从树里消失（合计 3.04 MB）：

`cute/arch/mma_sm90.hpp` · `cute/arch/mma_sm90_gmma.hpp` · `cute/arch/mma_sm90_gmma_sparse.hpp` ·
`cute/atom/mma_traits_sm90.hpp` · `cute/atom/mma_traits_sm90_gmma.hpp` · `cute/atom/mma_traits_sm90_gmma_sparse.hpp` ·
`gemm/collective/builders/sm90_common.inl`（`ss_smem_selector` / `sm90_cluster_shape_to_tma_atom`）·
`epilogue/collective/builders/sm90_builder.inl`（`sm90_get_epilogue_smem_swizzle_layout_atom`）·
`epilogue/thread/linear_combination*.h` ×4（原来只被 sm90_builder 拉进来）

**两条纪律**（不然会静默错）：

1. **shim 必须先于任何 `cute/…` 头被 include** —— 上游
   `cute/atom/copy_traits_sm90_tma.hpp` 在**定义处**就引用 `GMMA::Layout_MN_SW128_Atom`
   （非依赖名）。所以 `nvfp4_gemm.cu` **与** `owl_fa2.cu` 都在文件最前面 include 它。
2. 上游在 `namespace cute` 里靠 `using namespace SM90;` 才能写 `cute::GMMA::…` ——
   原头被裁后，shim 里必须把这条可见性补回来。

### 第 3 轮（2026-10-29）："用到什么就抄什么" —— 把 2.x arch 整支与 StreamK 从树上摘掉

第 2 轮后剩下的体积**全是真被编译的代码**，于是换了个问法：不再问
“哪个文件没被 include”，而是问 **“这个文件里我们到底用到了哪几个符号”** ——
用到的抄成 owl 侧小头，没用到的那一支整批删。

本轮的对象是 `cutlass/arch/mma.h` 那条链：

```
sm90_tile_scheduler_stream_k.hpp  (StreamK，本仓只用 PersistentScheduler)
  → block_striped.h → wmma_array.h → arch/mma.h
      → arch/mma_sm{50,60,61,70,75,80,89,90}.h
      → arch/mma_sparse_sm{80,89}.h · arch/wmma{,_sm70,_sm75}.h
      → mma_sm50.h → quaternion.h → cutlass/matrix.h   ← 355 KB
```

我们从 `arch/mma.h` 真正用到的只有**两个空标签**：`arch::OpClassTensorOp` /
`arch::OpClassBlockScaledTensorOp`（epilogue 与 blockscaled 主循环的算子类别）。
按“抄过来”的规矩它们进了 **`cu/nvfp4/owl_cutlass_arch_tags.hpp`**（纯 struct，
零代码生成，**必须先于任何 collective/epilogue builder 头 include**）。
`arch/mma_sm90.h` 则只被 kernel 头引进来、内容是 2.x 的 `arch::Mma<SM90_*>`
（我们用的是 **cute atom**，一个没用到）⇒ 连同那行 include 一起删。

删掉的 20 个文件（合计 **≈ 668 KB**）：

`cutlass/matrix.h` · `cutlass/quaternion.h` · `cutlass/block_striped.h` ·
`cutlass/wmma_array.h` · `cutlass/arch/mma.h` · `cutlass/arch/mma_sm{50,60,61,70,75,80,89,90}.h` ·
`cutlass/arch/mma_sparse_sm{80,89}.h` · `cutlass/arch/wmma{,_sm70,_sm75}.h` ·
`cutlass/gemm/kernel/sm9{0,00}_tile_scheduler_stream_k.hpp`

同轮还改了 2 个文件：`gemm/kernel/tile_scheduler.hpp`（去掉 2 行 StreamK include
+ 5 个 StreamK 选择器特化）、`gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`
（去掉 `arch/mma_sm90.h`）。**判据照旧：两个 TU 的 PTX 逐行相同
（diff = 0 行，文件同为 173,768 B）。**

### 第 4 轮（2026-10-29）：Group / Static 持久调度器下树（同一把刀接着切）

与 StreamK 同律：本仓只用 `PersistentScheduler`。`GroupScheduler`（分组 GEMM）与
`StaticPersistentScheduler` 的 7 个选择器壳子保留（只把 `using Scheduler = …`
换成注释，永不实例化），三个头删掉（≈ 45 KB）：
`gemm/kernel/sm90_tile_scheduler_group.hpp` · `sm100_tile_scheduler_group.hpp` ·
`sm100_static_tile_scheduler.hpp`。**PTX 仍逐行相同（diff = 0 行，173,768 B）。**

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
#    第 2 轮后还要验“代码生成未变”：对两个 TU 各跑一次 `nvcc -ptx`，
#    去掉 `//` 行与 `_INTERNAL_<hash>_` 后 `diff` 应为 0 行。
# ⑤ 门：装配期 oracle `max|err|=0`、4B NVFP4 一发对话、9B GGUF 文本探针
```

## 4. 当前规模与实测（2026-10-29 第 3/4 轮后，RTX 5070 Ti）

| 口径 | 原始 | 第 1 轮 | 第 2 轮 | 第 3 轮 | **第 4 轮（现在）** |
|---|---:|---:|---:|---:|---:|
| 文件数 | 517 | 246 | 235 | 211 | **208** |
| 体积 | 17 MB | 8.8 MB | 5.9 MB | 5.15 MB | **4.74 MB** |
| nvfp4 TU 闭包 | 513 头 / 15.7 MB | 238 头 / 8.4 MB | 228 头 / 5.41 MB | 205 头 / 4.69 MB | **202 头 / 4.65 MB** |
| fa2 TU 闭包 | — | 129 头 / 5.34 MB | 127 头 / 2.75 MB | 127 头 / 2.75 MB | **127 头 / 2.75 MB** |
| 两 TU 并集 | — | 243 头 / 8.52 MB | 231 头 / 5.48 MB | 209 头 / 4.76 MB | **206 头 / 4.72 MB** |

性能（同一 4B NVFP4 档，第 1 轮 → 第 2 轮）：

| 面 | 第 1 轮 | 第 2 轮 | 判据 |
|---|---:|---:|---|
| decode | 8.492 ms/step | **8.347 ms/step** | ±3% ✓（反而略快） |
| prefill（1374 tok） | 中位 10644 tok/s | 中位 13827（区间 10.6~16.3k） | 同区间波动，不劣化 ✓ |
| 装配期 oracle 5 档 | `max|err|=0` | `max|err|=0` | 逐档相等 ✓ |
| 文本 | `391`、3.76 B/tok | `391`、3.81 B/tok | 正常 ✓ |
| 9B GGUF | 137/94/757 B | 137/94/**745** B | 见下 |

### 4.1 “代码生成没变”是怎么证明的（比跑分硬）

第 2 轮前后，对**两个 CUDA TU** 各做一次 `nvcc -ptx` 并逐行比对
（`diff <(grep -v '^//' a) <(grep -v '^//' b)`，再去掉 `_INTERNAL_<hash>_` 这个
随 include 集合变化的 TU id）：**差异行数 = 0**；`.o` 大小逐字节相同（236,056 B），
唯一不同的字节就是那些 TU-id 符号串。

也就是说：**同一个 kernel、同一条 `mma.sync.kind::mxf4nvf4`、同一份
`__launch_bounds__`，连寄存器分配都没动** —— 变的只是“编译器要读多少头”。

（9B 第 3 号探针 757 → 745 字节**与 CUTLASS 无关**：两个 CUDA TU 的 PTX 已证明相同，
而这一轮之间上游还合进了 scheduler/vision/并发调度的引擎改动；1、2 号探针仍逐字节
一致，且结果在同一二进制上可重复。）

## 5. 工具：`crates/kernels/tools/cu_trim/cu_trim.py`（试删 + “用了什么”）

第 3/4 轮那种人肉探索已固化成脚本（在 `crates/kernels/` 下跑）：

```bash
source ../../.local-state/build-env.sh
./tools/cu_trim/cu_trim.py files -n 12                     # 全量：每头一份树副本「清空→编译→PTX 比对」
./tools/cu_trim/cu_trim.py symbols 'arch/copy_sm100.hpp'    # 单头：清空后编译错误里的符号名
```

- 判据 = 本档硬判据：**PTX 去注释行 / TU-id / 副本路径后逐行相同**。
- `files` 写 `.local-state/cu_trim_report.csv`：`path,size,verdict,detail`，
  verdict ∈ **可删**（编过且 PTX 不变）/ **需要**（编不过；detail 给前 6 个缺名）/
  **变了**（编过但 PTX 变）。

**全量结果（2026-10-29，208 头）：`可删` = 0**，169 `需要` + 39 `变了`。
⇒ 第 1~4 轮之后**文件粒度已经削到底**：每个文件要么参与解析、要么影响代码生成。
（`变了` 那 39 个不是噪声：例如拿掉 `cutlass/arch/config.h`，架构特性宏一变就改变
dispatch，PTX 真变。）

**下一刀只剩符号粒度**（`symbols` 模式 = “清空后编译器叫缺的名字”，即最小需求证据）：

| 文件 | 体积 | 清空后缺的名字 | 解读 |
|---|---:|---|---|
| `epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` | 112 KB | **7 个**（`Sm90LinearCombination` 为中心） | 真实现只 1 个，其余是 metafunction 的解析期名字引用 |
| `cute/arch/copy_sm100.hpp` | 356 KB | **63 个**（全是 `SM100_TMEM_LOAD_*`） | 全部来自 `copy_traits_sm100.hpp` 里 `Copy_Traits<>` 特化的解析期引用 |

于是套路是：**先删掉无用的 `Copy_Traits<>` / metafunction 特化（把需求降下来），
再把剩下真正需要的实现抄成 owl 头**，然后原文件即可整块删。每刀仍以 PTX 逐行相同收口。

## 6. 第 5 刀（2026-10-29）：`copy_traits_sm100.hpp` 只留“名字”，删掉 TMEM 特化

`copy_traits_sm100.hpp` 原 3,897 行 / 146 KB，其中 3,443 行是 **`Copy_Traits<>`
对 Blackwell `SM100_TMEM_LOAD_* / TMEM_STORE_*` 的特化**。我们的 epilogue 是
**SM90 TMA warp-specialized**（`sm90_epilogue_tma_warpspecialized.hpp`），TMEM
一个都不实例化 ⇒ 整段删除，文件降到 **454 行 / 18 KB**。

唯一要补的不是实现，而是**两个名字**：`gemm/dispatch_policy.hpp` 在模板实参里
直接写了 `SM100_TMEM_LOAD_16dp256b1x` / `SM100_TMEM_LOAD_32dp32b32x`
（非依赖名 ⇒ 解析期必须可见），但那支 Blackwell epilogue 从未实例化。所以在文件
末尾补两行 `using SM100::TMEM::LOAD::…;` —— 不补实现。

**判据：PTX 逐行相同（diff = 0 行，173,768 B）。** 规模：include 4.74 → **4.60 MB**，
nvfp4 闭包 202 头 / **4.51 MB**，并集 206 头 / **4.58 MB**。

### 6.1 下一刀：`cute/arch/copy_sm100.hpp`（356 KB → 17 个 atom）

删完 TMEM 特化后，`symbols` 模式的需求从 **63 个降到 17 个名字**：
`SM100_{LOAD,STORE}_256bit_CACHE_NOALLOCATION`、`SM100_U8x{8,16}_LDSM_T`、
`SM100_SU{4,6}_DU8x16_x{1,2,4}_LDSM_N`、`SM100_U8x{4,8,16}_STSM_T` 与上面那
2 个 TMEM atom。**把这 17 个 struct 抄成 owl 头 → 删 356 KB 原文件 → PTX 比对**
就是下一刀（同律已验证三遍）。

## 7. 第 6/7 刀（2026-10-29）：SM100 原子与 UMMA traits —— 356 KB + 172 KB → 0.24 KB 级

两刀同一套路（`symbols` 模式问需求 → 只抄“非它不可”的部分 → PTX 比对）：

| 刀 | 文件 | 前 | 后 | 做法 |
|---|---|---:|---:|---|
| 6 | `cute/atom/copy_traits_sm100.hpp` + `cute/arch/copy_sm100.hpp` | 146 KB + **356 KB** | 10 KB + **删除** | 先删 14 个无用的 `Copy_Traits<>`（LDSM/STSM/256bit）→ 再问需求，只剩 4 个 **fp4 LDSM 原子**（`SM100_SU{4,6}_DU8x16_*_LDSM_N`，主循环真的实例化）+ 2 个 **TMEM 原子名**（`dispatch_policy.hpp` 的模板实参，从不实例化）⇒ 逐字抄进 `cu/nvfp4/owl_cutlass_sm100_tmem_atoms.hpp` |
| 7 | `cute/atom/mma_traits_sm100.hpp` | 172 KB | **8 KB** | 3,717 行是 48 个 `MMA_Traits<SM100_MMA_*>` 特化（Blackwell tcgen05）—— 我们的 atom 是 sm120，一个不实例化 ⇒ 只留 `UMMA::Layout_K_{SW128,SW64,SW32,INTER}_Atom` 与 `get_utccp_smem_desc_tensor` |

**判据：两次 PTX 均逐行相同（0 行差异，173,768 B）；fa2 TU 预处理闭包也验过。**
规模：include 4.60 → **4.07 MB**（207 文件），nvfp4 闭包 202 头 / **3.98 MB**，
并集 207 头 / **4.05 MB**。**距原始 17 MB 已 −76%**；四门仍全绿
（oracle 5 档 `max|err|=0`、`391`、decode 8.27 ms）。

## 8. 下一刀候选（`symbols` 现成清单）

- `cutlass/numeric_conversion.h`（287 KB）：缺名只 `FloatRoundStyle` 等，但要小心
  —— 里面 f32↔f16 的 `NumericArrayConverter` 是**真被实例化**的（epilogue 用），
  抄的时候必须逐字带上 sm80+ 的 `cvt.rn.f16x2.f32` asm，否则数值会变（PTX 判据会抓到）。
- `cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp`（112 KB）：缺名 7 个，
  真实现只 `Sm90LinearCombination`（alpha_ptr 面）。
- `cute/atom/copy_traits_sm90_tma.hpp`(66 KB) / `cute/arch/copy_sm90_tma.hpp`(56 KB)：
  TMA 原子面，主循环真用，需先问 `symbols` 再判断。

### 8.1 `numeric_conversion.h` 的**需求已经量出来了**（第 8 刀待做）

两次“清空 + 编译”实验（都已还原，树未变）：

1. **完全清空** → 只有 62× `identifier "FloatRoundStyle" is undefined` + 38 个级联
   （`FloatRoundStyle` 就在本文件第 63 行）⇒ 说明**先拿掉它，编译能推进到实例化阶段**；
2. **只留 `FloatRoundStyle` 枚举** → 报错的都是 `NumericArrayConverter<…, …, FragmentSize, RoundStyle>`
   的**主模板缺失**，即我们需要的是：两个主模板（`NumericConverter` / `NumericArrayConverter`）
   + 本 TU 真正实例化的那几对（`half_t↔float`，`round_to_nearest`）的特化。

⇒ 下一刀的正确开法：把这 4 段抄进 `cu/nvfp4/owl_cutlass_numeric_convert.hpp`（枚举 + 两个主模板 +
`(half_t,float)` 与 `(float,half_t)` 的 sm80+ 特化，**asm 必须逐字带上**
`cvt.rn.f16x2.f32`），然后整文件下树（287 KB），PTX 判据抓任何数值漂移。
