# vendored:s**head 版 int8-MMA MMQ**(W-2f;设计 §二十二)

> 本目录是第三方代码(MIT,见 `LICENSE`)。上游 =
> 本机 `packages/llama.cpp` fork(即 §21 剖析的那个二进制之源码树)。
>
> **三处登记补丁**(2026-10-19 W-2f 接线;判据 = 门全部绿):
>
> | # | 文件 | 改动 | 原因 |
> |---|---|---|---|
> | P1 | `mmq.cuh` | `mul_mat_q`:`__launch_bounds__(…) static __global__` → `static __device__ __forceinline__` | fatbin 只出**设备码**(宿主函数永不执行)⇒ 入口必须由本仓 `extern "C" __global__` wrapper 提供;`launch_bounds(256,1)` 搬到 wrapper |
> | P2 | `mmq.cuh` | 内核 + `mul_mat_q_process_tile` + `write_back_{mma,dp4a}` 的 `dst`:`float*` → `__half*`(+`__float2half`),fixup 分支强转 | 本仓输出 f16 `[m,n]` 行主序,与上游 `dst[col*stride+row]` 布局逐位相同 ⇒ 省 f32 中间缓冲 |
> | P3 | (无) | StreamK 取上游"tiles 效率 ≥90%"形态(一维 `grid=ntiles`、`tmp_fixup=nullptr`) | 零 fixup 核、零新机制 |
>
> 其余逐字未改。

| 项 | 值 |
|---|---|
| 仓库 | `packages/llama.cpp`(llama.cpp fork) |
| commit | `dfcf18193`(2026-09-13,"G7-③-capture …") |
| 二进制对照 | `packages/llama.cpp/build-master2/bin/`(`libggml-cuda.so.0.23.0`) |

## 为什么是这一版而不复用 `../vendor/`(§十八)

`../vendor/mmq.cu` = `repos/llama.cpp@1b1e27cb49`(2024-05)**同代**版:它的
`vecdotq.cuh` **一条 `mma` 指令都没有**(所谓 tensor cores 只改 tile 配置),
dp4a 档 27B prefill 实测 557 t/s。本目录这版才走 `mma.sync.m16n8k16.s8`
(sm80+ int8 张量核)⇒ 同机 103 TFLOPS 等效(§21.1),**1.53× 于本仓 cuBLAS 臂**。

## 文件

| 文件 | 说明 |
|---|---|
| `mmq.cu` | 唯一入口 TU(`ggml_cuda_mul_mat_q_switch_type` + `ggml_cuda_op_mul_mat_q`) |
| `mmq.cuh` | 内核 `mul_mat_q<type, J, fallback>` + 配置表(设备侧 constexpr)+ `mul_mat_q_case` |
| `mmq-load-tiles.cuh` / `mmq-vec-dot.cuh` / `vecdotq.cuh` / `mma.cuh` | 逐型装载 / 点积 / int8-mma 原语 |
| `mmq-config-*.cuh` | 逐架构配置表(我们只用 `ampere`) |
| `common.cuh` + `ggml-*.h` + `gguf.h` + `ggml-common.h` | 依赖头(vendored 副本,勿与仓内 `../vendor/ggml-common.h` 混用:两代块布局同,但 `ggml_type` 词表不同) |
| `vendors/` | 上游的 cuda/hip/musa 小头 |
| `mmid.cuh` / `quantize.cuh` | `mmq.cu` 的直系依赖(`quantize_mmq_q8_1` 的声明;实现在上游 `quantize.cu`,**尚未 vendoring** —— 接线时要补) |

## 已验证(2026-10-19,**零补丁编过**)

```bash
# ① 独立 TU(只需一个 -I):
nvcc -DGGML_CUDA_USE_GRAPHS -I<vendor2> -O3 -std=c++17 \
     "--generate-code=arch=compute_86,code=[sm_86]" -use_fast_math -extended-lambda \
     -x cu -c mmq.cu -o /tmp/mmq_v2.o            # ⇒ 214 KB .o

# ② 本仓 build.rs 的旗标(经 `../gguf_mmq2.cu` 的 `#include "vendor2/mmq.cu"`):
nvcc -fatbin -O3 -std=c++17 --expt-relaxed-constexpr \
     "-gencode" "arch=compute_86,code=sm_86" ../gguf_mmq2.cu -o /tmp/mmq2.fatbin  # ⇒ OK
```

## 接线清单(设计 §22.3;**接线前本目录对现役路径零影响**)

1. `quantize.cu` 的 `quantize_mmq_q8_1`(→ `block_q8_1_mmq`,144 B)+ owl 侧新激活
   量化核(DS 布局按型:Q4_K/Q5_K = DS4,Q6_K/Q8_0 = D4);
2. `gguf_mmq2.cu` 里的 `extern "C" owl_gguf_mmq2_<ty>`(填 `mmq_args` + 选
   `J = min(m,512)` 向下对齐 8 + `fallback = (n%128 != 0)`),`stream_k` 强制 false;
3. `cudaFuncSetAttribute`(动态 smem)**boot 期**一次性预置(H2:捕获期禁);
4. 分派 + 门 + std 验收(见 §22.3 的 7 件表)。
