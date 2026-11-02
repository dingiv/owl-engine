# NVFP4 探针(sm_120)—— 独立可执行,不进构建链

> **归档定位(2026-11-03 迁入)**:这批文件是 NVFP4 臂**立项期**的一次性判定工具
> (算力天花板 / 布局校准 / 数值门 / 手写 GEMM 对照)。
>
> **生产算子早已落地**:`crates/kernels/cu/nvfp4/{nvfp4_gemm,nvfp4_quant_act}.cu`
> 由 `crates/kernels/build.rs`(`build_nvfp4`)编成 `libnvfp4.a`
> (`-gencode arch=compute_120a,code=sm_120a`);落地记录、布局事实与代际准入见
> `docs/devlog/kernels/nvfp4-arm-and-graph.md`。
>
> 保留理由:下表里的**唯一一手读数**(713 TFLOPS 天花板、M≈128 分水岭、手写 GEMM
> 66 TFLOPS、适配器 459/536 GFLOPS)不在别处记录 —— 下次换硬件(sm_120 真机 /
> 下一代 FP4)或重估"值不值得做"时的基线。

## 为什么不放 `crates/kernels/cu/`

`cu/` 是**会被构建期编进 .a 的源**;这里每个文件都带 `main()`,是**手工 nvcc
编跑**的独立程序 ⇒ 归 `tools/fixture/`(与 `ar-*.cu` / `p2p-*.cu` /
`vmm-*.cu` / `gdn-rec-probe.cu` 同类)。

**别把这批文件改回去当生产核**:生产核在 `crates/kernels/cu/nvfp4/`
(没有 `main()`,走注册表 + plan 发射)。

> 历史:原路径 `crates/kernels/probes/nvfp4/`(那个目录挨着 `cu/`,容易被误认为
> 构建面;`build.rs` 从来没扫过它)。2026-11-03 按仓内约定迁到
> `tools/fixture/nvfp4/`,与其它 GPU 探针同屋。

## 编译(统一姿势;在 owl 仓根执行)

```bash
CUDA=${CUDA_HOME:-/usr/local/cuda}            # 任何自带 nvcc 的 CUDA ≥ 12.8 都行
INC=crates/kernels/cu/cutlass/include

$CUDA/bin/nvcc -std=c++17 -O3 --expt-relaxed-constexpr \
  -gencode arch=compute_120a,code=sm_120a -I$INC \
  -o /tmp/p tools/fixture/nvfp4/<file>.cu
LD_LIBRARY_PATH=$CUDA/lib64 /tmp/p
```

> **`-gencode arch=compute_120a,code=sm_120a` 两个都要带 `a`**：只写 `-arch=sm_120a`
> 时 nvcc 会把 PTX 目标退成 `compute_120`，ptxas 直接拒 FP4 指令
> （`Feature '.kind::mxf4nvf4' not supported on .target 'sm_120'`）。

## 各文件是什么

| 文件 | 判据 / 结论 |
|---|---|
| `nvfp4_probe.cu` | **算力天花板**：纯寄存器 MMA。实测 5070 Ti：`mxf4nvf4.block_scale`(m16n8k64) **713 TFLOPS**；`f8f6f4`(m16n8k32,无缩放) 192；现役 MMQ2 int8 ≈145 ⇒ k=64 块缩放是唯一值得做的形态 |
| `calib2.cu` | **缩放映射校准**：A 的 SF `m = 8*(lane%2)+lane/4`；B 的 SF `n = lane/4`（t0 维广播）；每 u32 = 4 个 e4m3（k 组 0..3） |
| `g2.cu` | **手写 GEMM v1/v2**（无 smem / 加 smem）：数值正确但对拍后只有 **66 TFLOPS**，证明没有 TMA 多级流水线打不过 MMQ2 —— 保留作对照 |
| `nvfp4_gate_datapath.cu` | **数值门①·数据通路**：全 1.0 缩放，A/B nibble 布局 + MMA + epilogue + stride + 输出布局。实测 max\|Δ\| = 1 个 f16 ULP，0/131072 超差 |
| `nvfp4_gate_sf.cu` | **数值门②·SF 落位**：`make_tensor(ptr, tile_atom_to_shape_SFA(shape(M,N,K,1)))` 后按 `t(m,16g,0)` 写即可；解析期望 960 = 实测 960 |
| `nvfp4_gemm.cu` | **早期适配器原型**(对拍用,带 `main()`)：CUTLASS SM120 块缩放 mainloop + 默认 epilogue + 本仓 packer（SF 交错布局）+ 4×4 切片对拍。实测 4B 真权重：M=512/N=9216/K=2560 **459 GFLOPS**、M=2048 **536**（现役 MMQ2 145 ⇒ 3.2~3.7×）；真权重相对误差 **2.35e-04**。⚠️ 生产版是 `crates/kernels/cu/nvfp4/nvfp4_gemm.cu` |

## 与 M n 的分水岭

NVFP4 在 **M≈128** 才开始赢（M=64 时 0.83×，128 → 1.79×，512 → 3.17×，2048 → 3.69×）。
接进引擎时必须按 `m ≥ 128 && caps.fp4_blockscale` 分流，否则小 m 倒挂。
