// embedding 查表(models layers/embedding 使用;port dry_kernels.cu @ owl_embed_f32)
// ids 以 f32 数值形态过线(<2^24 精度无损;S4 u32 dtype 扩展挂账)。
extern "C" __global__ void owl_embed_f32(
    const float* w,           // [vocab, D]
    const float* ids,         // [tokens](f32 数值形态;kernel 内 cast)
    float* out,               // [tokens, D]
    unsigned int d_dim) {
    unsigned int t = blockIdx.x;
    unsigned int id = (unsigned int)ids[t];
    for (unsigned int d = threadIdx.x; d < d_dim; d += blockDim.x) {
        out[t * d_dim + d] = w[(unsigned long long)id * d_dim + d];
    }
}
