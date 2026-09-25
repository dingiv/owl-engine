// rope:interleaved partial(Qwen3.5:rotary_dim 64/256,相邻对 2i/2i+1)
// fused 不做(q/k 独立发射,单输出契约);一 block 一 token。
// 标量形参 = size_t(与 Arg::U64/arg_usize 8 字节严格对位);
// 输出块固定末参(槽序契约 4)。
extern "C" __global__ void owl_rope_interleaved_partial_f32(
    const float* x,            // [tokens, heads, head_dim]
    const float* cos_t,        // [max_pos, half]
    const float* sin_t,        // [max_pos, half]
    const float* pos,          // [tokens](f32 数值形态)
    size_t heads, size_t head_dim, size_t half,
    float* out) {              // [tokens, heads, head_dim]
    unsigned int t = blockIdx.x;
    unsigned int p = (unsigned int)pos[t];
    const float* c = cos_t + (unsigned long long)p * half;
    const float* s = sin_t + (unsigned long long)p * half;
    for (unsigned int h = 0; h < heads; ++h) {
        const float* xs = x + ((unsigned long long)t * heads + h) * head_dim;
        float* od = out + ((unsigned long long)t * heads + h) * head_dim;
        for (unsigned int i = 0; i < half; ++i) {
            float a = xs[2 * i], b = xs[2 * i + 1];
            od[2 * i]     = a * c[i] - b * s[i];
            od[2 * i + 1] = a * s[i] + b * c[i];
        }
        for (unsigned int d = 2 * half + threadIdx.x; d < head_dim; d += blockDim.x) {
            od[d] = xs[d];   // partial:旋转维之外直通
        }
    }
}
