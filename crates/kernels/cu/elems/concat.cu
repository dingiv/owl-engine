#include <cuda_fp16.h>

// PF1a 栈核(concat_rows;pf1-port-design §三批P1):
// n 份同形 [r, d] 行块 → [n·r, d] 纯行栈。
//
// arity 16 封顶(2026-10-23 DFlash2 K=1..15 单 block:verify 行数 = K+1
// 最大 16;n < 16 时多余指针位传任一实块,核内 in >= n 不读)。生产
// prefill 仍走 PF1b 批核 varlen,大 T 不经此核;f16/f32 两变体。
//
// 槽序契约(注册表 args 同序;输出块固定末参):
//   T×16 输入块, sz n, sz r, sz d, T out

extern "C" __global__ void owl_concat_rows_f16(
    const __half* i0, const __half* i1, const __half* i2, const __half* i3,
    const __half* i4, const __half* i5, const __half* i6, const __half* i7,
    const __half* i8, const __half* i9, const __half* i10, const __half* i11,
    const __half* i12, const __half* i13, const __half* i14, const __half* i15,
    size_t n, size_t r, size_t d,
    __half* out) {              // [n·r, d](末参 = 输出)
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = n * r * d;
    if (idx >= total) return;
    const __half* ptrs[16] = {i0, i1, i2, i3, i4, i5, i6, i7,
                              i8, i9, i10, i11, i12, i13, i14, i15};
    const size_t rd = r * d;
    const size_t in = idx / rd;
    if (in >= n) return;        // n < 16:多余指针位防读
    out[idx] = ptrs[in][idx % rd];
}

// f32 锚变体(注册表纪律:f32 核保留 = 单元测试锚;fixture 模型 prefill
// 对拍走 f32 链)
extern "C" __global__ void owl_concat_rows_f32(
    const float* i0, const float* i1, const float* i2, const float* i3,
    const float* i4, const float* i5, const float* i6, const float* i7,
    const float* i8, const float* i9, const float* i10, const float* i11,
    const float* i12, const float* i13, const float* i14, const float* i15,
    size_t n, size_t r, size_t d,
    float* out) {               // [n·r, d](末参 = 输出)
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = n * r * d;
    if (idx >= total) return;
    const float* ptrs[16] = {i0, i1, i2, i3, i4, i5, i6, i7,
                             i8, i9, i10, i11, i12, i13, i14, i15};
    const size_t rd = r * d;
    const size_t in = idx / rd;
    if (in >= n) return;
    out[idx] = ptrs[in][idx % rd];
}
