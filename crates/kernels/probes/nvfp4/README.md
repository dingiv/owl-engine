# NVFP4 probes (sm_120) —— 独立可执行，不进构建链

这一目录里是**探针/原型**，不是产物：每个文件都带 `main()`，直接用 `nvcc` 编跑，
**不会被 `crates/kernels/build.rs` 的 `include_str!` 闭包扫到**（那条闭包只收
`src/**/*.rs` 里 `include_str!` 引用的 `.cu`）⇒ 对生产构建零影响。

为什么放这里而不是 `cu/`：`cu/` 是**会被构建期编成 cubin 的源**，而这些都是
一次性判定工具（布局校准 / 算力天花板 / 数值门）。等算子正式接入时，只有
`nvfp4_gemm.cu` 的内核部分会挪回 `cu/nvfp4/` 并由 `build.rs` 编成
`libowl_nvfp4.a`（`-gencode arch=compute_120a,code=sm_120a`）。

## 编译（统一姿势）

```bash
CUDA=/home/ubuntu/gui_agent/owl-cuda-13.2      # 任何自带 nvcc 的 CUDA ≥ 12.8 都行
INC=crates/kernels/cu/cutlass/include

$CUDA/bin/nvcc -std=c++17 -O3 --expt-relaxed-constexpr \
  -gencode arch=compute_120a,code=sm_120a -I$INC \
  -o /tmp/p crates/kernels/probes/nvfp4/<file>.cu
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
| `nvfp4_gemm.cu` | **能跑的适配器**：CUTLASS SM120 块缩放 mainloop + 默认 epilogue + 本仓 packer（SF 交错布局）+ 4×4 切片对拍。实测 4B 真权重：M=512/N=9216/K=2560 **459 GFLOPS**、M=2048 **536**（现役 MMQ2 145 ⇒ 3.2~3.7×）；真权重相对误差 **2.35e-04** |

## 与 M n 的分水岭

NVFP4 在 **M≈128** 才开始赢（M=64 时 0.83×，128 → 1.79×，512 → 3.17×，2048 → 3.69×）。
接进引擎时必须按 `m ≥ 128 && caps.fp4_blockscale` 分流，否则小 m 倒挂。
