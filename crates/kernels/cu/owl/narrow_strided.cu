// ============================================================================
// owl 窄切物化拷贝(2026-10-13 C 轮自 cu/owl/attention.cu 按**算子族**拆出 ——
// 一算子一 `.cu`,与 ops/elems.rs 的 `owl_narrow_strided_*` 声明对望)。
//
// port 自旧世界 dry_kernels.cu + 新世界契约改造。
// dst[idx] = src[r * src_dim + start + d]:attn_output_gate 的 per-head
// [value|gate] 切分用(两次发射,start=0 / head_dim,outer = tokens*heads,
// src_dim = 2*head_dim);GDN conv 输出三段切分同款复用。
// 标量形参 = size_t(与 Arg::U64 8 字节严格对位);输出块固定末参(契约 4)。
// f16 变体 = 纯 gather 拷贝,无数值转换(half 位型直搬)。
// ============================================================================
#include <cuda_fp16.h>

// 非连续窄切物化拷贝(dst[idx] = src[r * src_dim + start + d]):
// attn_output_gate 的 per-head [value|gate] 切分用(两次发射,start=0 / head_dim,
// outer = tokens*heads,src_dim = 2*head_dim);GDN conv 输出三段切分同款复用。
// 标量形参 = size_t(与 Arg::U64/arg_usize 8 字节严格对位);
// 输出块固定末参(槽序契约 4)。
extern "C" __global__ void owl_narrow_strided_f32(
    const float* src,
    size_t outer, size_t src_dim, size_t start, size_t out_dim,
    float* dst) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = outer * out_dim;
    if (idx >= total) return;
    unsigned long long r = idx / out_dim;
    unsigned long long d = idx % out_dim;
    dst[idx] = src[r * src_dim + start + d];
}

// f16 基线变体(F3):纯 gather 拷贝,无数值转换(half 位型直搬)。
extern "C" __global__ void owl_narrow_strided_f16(
    const __half* src,
    size_t outer, size_t src_dim, size_t start, size_t out_dim,
    __half* dst) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = outer * out_dim;
    if (idx >= total) return;
    unsigned long long r = idx / out_dim;
    unsigned long long d = idx % out_dim;
    dst[idx] = src[r * src_dim + start + d];
}
