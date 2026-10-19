// ============================================================================
// owl 装载域 —— **lm_head 逐行对称 int8 量化**(P1.5-a;装载期一次,非轮内)
//
// # 为什么是"两发"而不是"一发两输出"
//
// 单发要同时写 `q[rows,cols] i8` 与 `scale[rows] f32` 两个输出块,而本仓的
// 非族算子契约是 **Out 槽必须在最后一个张量位**(`op_tables::non_family_out_slot_is_last_tensor`)
// ⇒ 两个输出只能拆成两发(装载期一次,代价 ~ms 级,不值得为省一发破坏契约)。
//
// # 语义(与 host 参考 `rtn_i8_row_sym` **逐元素相同**,判据在台子上)
//
//   s_r   = (max_i |w[r,i]| > 0) ? max/127 : 1.0        // 全零行 s=1,q 全 0
//   q[r,i] = clamp( round_to_nearest_even( f32(w) / s_r ), -127, 127 )
//
// ⚠️ **除法不换倒数**:两侧都写 `x / s`(IEEE 除法确定性),避免 `x * (1/s)`
// 与 `x / s` 在舍入上分家 —— 那是"host 与设备逐元素对不上"最常见的来源。
// ⚠️ **舍入取 RN(最近偶数)**,与 host 的 `f32::round_ties_even()` 对齐
// (`__float2int_rn` 同为 RN-even;`round()` 是 half-away-from-zero,不可用)。
//
// # 几何
//
// **一 block 一行**(grid = rows,block 256):行内 absmax 走 warp 归约 + 32 槽
// shared 归约;量化再扫一遍该行(第二遍天然 L2 命中)。
// ============================================================================

#include <cuda_bf16.h>

__device__ __forceinline__ float owl_q8_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }

/// 行内 absmax → `scale[row]`(block 归约;threadIdx.x 必须能整除 cols 的尾)
// 槽序契约(与 ops/load.rs 的 PARAMS **逐字对位**):w, rows, cols, scale(输出末参)
extern "C" __global__ void owl_quant_i8_row_w_scale_bf16(
    const __nv_bfloat16* __restrict__ w,  // [rows, cols]
    int rows,
    int cols,
    float* __restrict__ scale) {           // [rows](末参 = 输出)
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* wrow = w + (size_t)row * cols;
    float mx = 0.0f;
    for (int i = threadIdx.x; i < cols; i += (int)blockDim.x) {
        mx = fmaxf(mx, fabsf(owl_q8_to_f(wrow[i])));
    }
    __shared__ float red[32];
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = ((int)blockDim.x + 31) >> 5;
        float m = 0.0f;
        for (int i = 0; i < nw; ++i) m = fmaxf(m, red[i]);
        // 全零行:尺度取 1(避免 0/0),量化结果仍为 0
        scale[row] = (m > 0.0f) ? (m / 127.0f) : 1.0f;
    }
}

/// 逐元素 RTN(吃上一步的 `scale`;**不重算 max** —— 两发之间不插别的写者)
// 槽序契约:w, scale, rows, cols, q(输出末参)
extern "C" __global__ void owl_quant_i8_row_w_q_bf16(
    const __nv_bfloat16* __restrict__ w,  // [rows, cols]
    const float* __restrict__ scale,      // [rows]
    int rows,
    int cols,
    signed char* __restrict__ q) {        // [rows, cols](末参 = 输出)
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* wrow = w + (size_t)row * cols;
    signed char* qrow = q + (size_t)row * cols;
    const float s = scale[row];
    for (int i = threadIdx.x; i < cols; i += (int)blockDim.x) {
        const float v = owl_q8_to_f(wrow[i]) / s;
        int k = __float2int_rn(v);  // RN(最近偶数)= host round_ties_even
        k = k > 127 ? 127 : (k < -127 ? -127 : k);
        qrow[i] = (signed char)k;
    }
}
