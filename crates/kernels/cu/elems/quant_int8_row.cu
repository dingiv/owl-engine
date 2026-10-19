// W4A8 前置:激活 **per-token int8 量化**(S1c;2026-10-16)。
//
//   owl_quant_int8_row_f16 / owl_quant_int8_row_bf16
//     入 x[rows, cols]        f16/bf16 行主序(激活)
//     入 rows / cols          i32
//     入 mul                  **设备标量块** f32[1] —— 权重侧全局因子
//                             (= `pack_marlin_s_a8` 第二返回值 `max/4096`;
//                             无需折叠时传常 1.0 的块)。设备标量而非 host 标量:
//                             该值是**装载期由权重尺度数据算出**的,走 host 标量
//                             要给装载器开一条「数据 → host 图标量」的旁路;
//                             设备块与 vLLM 的 `input_global_scale` 同构
//                             (层声明 want,装载器填)。
//     写 scales[rows]         f32 = absmax/127 · mul(供 marlin a8 GEMM 的
//                             `a_scales` 槽)
//     出 q[rows, cols]        int8(**按本仓惯例以字节承载**,语义在核侧)
//
// 语义与上游 `dynamic_scaled_int8_quant_kernel` **逐点对齐**
// (vLLM `csrc/libtorch_stable/quantization/w8a8/int8/scaled_quant.cu`):
//   ① 每 token 一块求 |x| 行最大值 —— 上游用 cub::BlockReduce,此处用
//      **warp shuffle + 共享内存两级归约**(本仓零外部依赖纪律,不引 cub);
//   ② `scale = absmax/127`;`absmax == 0 ⇒ inv_s = 0`(整行输出 0);
//   ③ 量化 = `cvt.rni.sat.s8.f32`(**round-nearest-even + 饱和**,上游同款 asm);
//   ④ 全局因子**只乘进 scales 出口**,不进量化逆尺度 —— 代数上与量化互不干扰
//      (`a_int * (a_scale·factor) = a_f32·factor`;见设计卷 §9.1/§9.2③)。
//
// blockDim 由发射侧定(256 = 上游同款);cols 任意(核内 strided 循环兜底)。
// 发射几何:grid = (rows, 1, 1)(driver 按 rows 给;见 `driver::elems::quant_int8_row`)。
#ifdef __CUDACC_RTC__
typedef unsigned short ushort;
#endif
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/// `cvt.rni.sat.s8.f32`(round-nearest-even + 饱和)—— 上游
/// `float_to_int8_rn` 的 CUDA 分支逐字同款(`asm` 写 32 位寄存器,取低字节)。
static __device__ __forceinline__ signed char owl_rn_i8(float x) {
    unsigned int d;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(d) : "f"(x));
    return (signed char)(d & 0xFFu);
}

static __device__ __forceinline__ float owl_to_f32(const __half v) { return __half2float(v); }
static __device__ __forceinline__ float owl_to_f32(const __nv_bfloat16 v) { return __bfloat162float(v); }

/// 一行的量化:两级归约求 absmax → 出 scale → strided 量化写回。
///
/// **2026-10-19 W4A8 decode 治理(§P0.1 逐核账)**:生产 nsys 实测本核
/// **6.46 µs/次 × 345 次/步 = 2.23 ms/步**(W4A8 的 decode −12.4% 全在此)。
/// 病根与 §58 刀③ 的 norm 同款:**每线程最多 20 次 2 字节标量依赖装载**
/// (5120 宽、256 线程)—— 数据只 15KB,时间全在装载延迟/启动。
/// ⇒ 两处结构改动(照 `cu/ln/rmsnorm.cu` / `fused_add_rmsnorm.cu` 的配方):
///   ① **16B 向量装载**(`uint4` = 8 half)⇒ 每线程 2-3 次迭代(原 20);
///   ② 量化趟**8B 向量写**(`long long`,对齐前提 `cols % 8 == 0`)。
/// **数值逐位不变**(与旧版比):absmax = `fmaxf` 归约 —— max 可交换/结合,
///   归约序变**不引入**浮点差;量化本身 = 逐元素 `cvt.rni.sat.s8`,与顺序无关。
///   ⇒ 本刀**不需要新容差**,门按逐位比对即可(见 `quant_int8_row_probe`)。
template <typename T>
__device__ __forceinline__ void owl_quant_row(
    const T* __restrict__ x, int rows, int cols, const float* __restrict__ mul,
    float* __restrict__ scales, signed char* __restrict__ q) {
    __shared__ float s_absmax;
    __shared__ float s_warp[32];

    const int row = blockIdx.x;
    if (row >= rows) return;
    const T* __restrict__ row_in = x + (size_t)row * (size_t)cols;
    signed char* __restrict__ row_out = q + (size_t)row * (size_t)cols;

    // ① 行内 absmax(warp 内 shuffle → 每 warp 一把手 → 共享内存收口)
    //    向量支:16B 装载(8 元素);对齐前提 = 行基址 16B 对齐(
    //    `row·cols·sizeof(T)`:T = 2B 且 `cols % 8 == 0` ⇒ 16B ✓)
    const bool vec_ok = ((cols % 8) == 0);
    float m = 0.f;
    if (vec_ok) {
        for (int c = (int)threadIdx.x * 8; c < cols; c += (int)blockDim.x * 8) {
            const uint4 xr = *reinterpret_cast<const uint4*>(row_in + c);
            const T* xh = reinterpret_cast<const T*>(&xr);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                m = fmaxf(m, fabsf(owl_to_f32(xh[e])));
            }
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            m = fmaxf(m, fabsf(owl_to_f32(row_in[i])));
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    }
    const int nwarps = (int)((blockDim.x + 31u) >> 5);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.f;
        for (int i = 0; i < nwarps; i++) v = fmaxf(v, s_warp[i]);
        s_absmax = v;
        // ④ 全局因子只进出口(量化本身与因子无关)
        scales[row] = v / 127.f * mul[0];
    }
    __syncthreads();

    // ②③ 量化(零行:inv = 0 ⇒ 整行 0);向量支同 ① 的装载 + 8B 写
    const float inv = (s_absmax == 0.f) ? 0.f : 127.f / s_absmax;
    if (vec_ok) {
        for (int c = (int)threadIdx.x * 8; c < cols; c += (int)blockDim.x * 8) {
            const uint4 xr = *reinterpret_cast<const uint4*>(row_in + c);
            const T* xh = reinterpret_cast<const T*>(&xr);
            signed char o8[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                o8[e] = owl_rn_i8(owl_to_f32(xh[e]) * inv);
            }
            *reinterpret_cast<long long*>(row_out + c) = *reinterpret_cast<const long long*>(o8);
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            row_out[i] = owl_rn_i8(owl_to_f32(row_in[i]) * inv);
        }
    }
}

extern "C" __global__ void owl_quant_int8_row_f16(
    const __half* __restrict__ x, int rows, int cols, const float* __restrict__ mul,
    float* __restrict__ scales, signed char* __restrict__ q) {
    owl_quant_row<__half>(x, rows, cols, mul, scales, q);
}

extern "C" __global__ void owl_quant_int8_row_bf16(
    const __nv_bfloat16* __restrict__ x, int rows, int cols, const float* __restrict__ mul,
    float* __restrict__ scales, signed char* __restrict__ q) {
    owl_quant_row<__nv_bfloat16>(x, rows, cols, mul, scales, q);
}
