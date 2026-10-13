// embedding 查表(models layers/embedding 使用;port dry_kernels.cu @ owl_embed_f32)
// ids 以 f32 数值形态过线(<2^24 精度无损;S4 u32 dtype 扩展挂账)。
// 标量形参 = size_t(与 Arg::U64/arg_usize 8 字节严格对位);
// 输出块固定末参(槽序契约 4)。
#include <cuda_fp16.h>

extern "C" __global__ void owl_embed_f32(
    const float* w,           // [vocab, D]
    const float* ids,         // [tokens](f32 数值形态;kernel 内 cast)
    size_t d_dim,
    float* out) {             // [tokens, D]
    unsigned int t = blockIdx.x;
    unsigned int id = (unsigned int)ids[t];
    for (size_t d = threadIdx.x; d < d_dim; d += blockDim.x) {
        out[(unsigned long long)t * d_dim + d] = w[(unsigned long long)id * d_dim + d];
    }
}

// f16 基线变体(F3):表/出 half;ids 恒 f32 数值过线(契约 5)。
// 纯查表拷贝,无数值转换。
extern "C" __global__ void owl_embed_f16(
    const __half* w,          // [vocab, D]
    const float* ids,         // [tokens](f32 数值形态;kernel 内 cast)
    size_t d_dim,
    __half* out) {            // [tokens, D]
    unsigned int t = blockIdx.x;
    unsigned int id = (unsigned int)ids[t];
    for (size_t d = threadIdx.x; d < d_dim; d += blockDim.x) {
        out[(unsigned long long)t * d_dim + d] = w[(unsigned long long)id * d_dim + d];
    }
}
