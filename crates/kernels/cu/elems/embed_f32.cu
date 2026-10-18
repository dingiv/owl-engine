// embedding 查表(models layers/embedding 使用;port dry_kernels.cu @ owl_embed_f32)
// ids 以 f32 数值形态过线(<2^24 精度无损;S4 u32 dtype 扩展挂账)。
// 标量形参 = size_t(与 Arg::U64/arg_usize 8 字节严格对位);
// 输出块固定末参(槽序契约 4)。
//
// ============================================================================
// 2026-10-19 §58 刀④(**decode 小核延迟治理**;读数出处 TP2-实录卷 §58.5/§58.7)
//
// 旧发射几何 = `grid (tokens,1,1) × block (1,1,1)`(**一 token 一线程**)⇒
// decode(1 token)时**单线程**串行拷 5120 个 half:单块核实测 **264µs**
// (生产 nsys 288µs/decode 步,占 1.2%)。写它的原意是"一 token 一 block",
// 但 block 一起被写成 1 —— 于是 blockIdx 维度铺开了、threadIdx 维度没铺。
//
// 两处改动:
//   ① **核内 16B 向量拷贝**(八 half / 四 float 一次;VEC = 16/sizeof(T));
//   ② 几何 block 从 1 起(几何在 `driver::elems::embed`,**不在本文件**)。
// 判据(`crates/kernels/tests/ln_elems_bench.rs`):逐位相等 + µs/发。
// 对齐前提:向量支只在 `d_dim % VEC == 0` 时走(表行/输出行基址 = idx·d_dim
//   ⇒ d_dim%VEC==0 即保证每行 16B 对齐);否则退回标量支(逐元素同式)。
// 本核**纯查表拷贝,无数值转换,无归约** ⇒ 改动不影响任何一位。
// ============================================================================
#include <cuda_fp16.h>

extern "C" __global__ void owl_embed_f32(
    const float* w,           // [vocab, D]
    const float* ids,         // [tokens](f32 数值形态;kernel 内 cast)
    size_t d_dim,
    float* out) {             // [tokens, D]
    const unsigned int t = blockIdx.x;
    const unsigned int id = (unsigned int)ids[t];
    const float* src = w + (unsigned long long)id * d_dim;
    float* dst = out + (unsigned long long)t * d_dim;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if ((d_dim & 3) == 0) {
        for (size_t d = (size_t)tid * 4; d < d_dim; d += (size_t)stride * 4) {
            *reinterpret_cast<uint4*>(dst + d) = *reinterpret_cast<const uint4*>(src + d);
        }
    } else {
        for (size_t d = tid; d < d_dim; d += stride) {
            dst[d] = src[d];
        }
    }
}

// f16 基线变体(F3):表/出 half;ids 恒 f32 数值过线(契约 5)。
// 纯查表拷贝,无数值转换。
extern "C" __global__ void owl_embed_f16(
    const __half* w,          // [vocab, D]
    const float* ids,         // [tokens](f32 数值形态;kernel 内 cast)
    size_t d_dim,
    __half* out) {            // [tokens, D]
    const unsigned int t = blockIdx.x;
    const unsigned int id = (unsigned int)ids[t];
    const __half* src = w + (unsigned long long)id * d_dim;
    __half* dst = out + (unsigned long long)t * d_dim;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if ((d_dim & 7) == 0) {
        for (size_t d = (size_t)tid * 8; d < d_dim; d += (size_t)stride * 8) {
            *reinterpret_cast<uint4*>(dst + d) = *reinterpret_cast<const uint4*>(src + d);
        }
    } else {
        for (size_t d = tid; d < d_dim; d += stride) {
            dst[d] = src[d];
        }
    }
}
