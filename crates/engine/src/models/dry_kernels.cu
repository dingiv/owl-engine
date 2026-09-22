// owl dry-run kernels(K1 真 kernel 落地前的 naive 通路;捕获安全契约同 .cu 全家)
// 约定:全部 f32;naive attention = 非分页 slot 直排(真分页归 K1);
// MAX_KV = 256(dry-run 规模上限,编译期常量)。

extern "C" __global__ void owl_sin_f32(const float* x, float* out, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sinf(x[i]);
}

extern "C" __global__ void owl_cos_f32(const float* x, float* out, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cosf(x[i]);
}

extern "C" __global__ void owl_embed_f32(
    const float* w,           // [vocab, D]
    const unsigned int* ids,  // [tokens]
    float* out,               // [tokens, D]
    unsigned int d_dim) {
    unsigned int t = blockIdx.x;
    for (unsigned int d = threadIdx.x; d < d_dim; d += blockDim.x) {
        out[t * d_dim + d] = w[(unsigned long long)ids[t] * d_dim + d];
    }
}

// rotate-half:对每个 head 的前半/后半做旋转(q_out[i] = q[i]*cos[i] - q[i+H]*sin[i];
// q_out[i+H] = q[i]*sin[i] + q[i+H]*cos[i])。k 同理(kv_heads 行)。
extern "C" __global__ void owl_rope_half_f32(
    const float* q,            // [tokens, q_heads, D]
    const float* k,            // [tokens, kv_heads, D]
    float* q_out,
    float* k_out,
    const float* cos_t,        // [max_pos, D/2]
    const float* sin_t,
    const unsigned int* pos,   // [tokens]
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int half) {
    unsigned int t = blockIdx.x;
    unsigned int p = pos[t];
    const float* c = cos_t + (unsigned long long)p * half;
    const float* s = sin_t + (unsigned long long)p * half;
    for (unsigned int h = 0; h < q_heads; ++h) {
        const float* qs = q + ((unsigned long long)t * q_heads + h) * half * 2;
        float* qd = q_out + ((unsigned long long)t * q_heads + h) * half * 2;
        for (unsigned int i = threadIdx.x; i < half; i += blockDim.x) {
            float a = qs[i], b = qs[i + half];
            qd[i] = a * c[i] - b * s[i];
            qd[i + half] = a * s[i] + b * c[i];
        }
    }
    for (unsigned int h = 0; h < kv_heads; ++h) {
        const float* ks = k + ((unsigned long long)t * kv_heads + h) * half * 2;
        float* kd = k_out + ((unsigned long long)t * kv_heads + h) * half * 2;
        for (unsigned int i = threadIdx.x; i < half; i += blockDim.x) {
            float a = ks[i], b = ks[i + half];
            kd[i] = a * c[i] - b * s[i];
            kd[i + half] = a * s[i] + b * c[i];
        }
    }
}

// decode(seq=1)naive attention:一线程一 (t, q_head)。
// 写 cache(slot>=0)→ 对 cache[0..kv_len] 全行打分 softmax → 加权和出 out。
#define OWL_DRY_MAX_KV 256

extern "C" __global__ void owl_naive_decode_attn_f32(
    const float* q,            // [bs, Hq, D]
    const float* k,            // [bs, Hkv, D]
    const float* v,            // [bs, Hkv, D]
    float* kc,                 // [max_slots, Hkv, D](slot 直排)
    float* vc,                 // [max_slots, Hkv, D]
    const int* slots,          // [bs](u32 位型直读;负数 = padding)
    const int* kv_lens,        // [bs]
    float* out,                // [bs, Hq*D]
    unsigned int q_heads,
    unsigned int kv_heads,
    unsigned int d_dim) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = gridDim.x * blockDim.x;
    // 线程一 (t,h):由 1D 展平反解(调用方 grid = bs*Hq)
    (void)total;
    unsigned int t = idx / q_heads;
    unsigned int h = idx % q_heads;
    unsigned int kvh = h / (q_heads / kv_heads);
    const int slot = slots[t];
    int kv_len = kv_lens[t];
    if (kv_len > OWL_DRY_MAX_KV) kv_len = OWL_DRY_MAX_KV;

    const float* qs = q + ((unsigned long long)t * q_heads + h) * d_dim;

    // 1) 写 cache(slot >= 0)
    if (slot >= 0) {
        float* kcd = kc + (unsigned long long)slot * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        float* vcd = vc + (unsigned long long)slot * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        const float* ks = k + ((unsigned long long)t * kv_heads + kvh) * d_dim;
        const float* vs = v + ((unsigned long long)t * kv_heads + kvh) * d_dim;
        for (unsigned int d = 0; d < d_dim; ++d) {
            kcd[d] = ks[d];
            vcd[d] = vs[d];
        }
    }

    // 2) 打分(全行扫描;scale = 1/sqrt(D))
    float scores[OWL_DRY_MAX_KV];
    float maxs = -3.0e38f;
    float scale = rsqrtf((float)d_dim);
    for (int s = 0; s < kv_len; ++s) {
        const float* kc_row = kc + (unsigned long long)s * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        float acc = 0.0f;
        for (unsigned int d = 0; d < d_dim; ++d) acc += qs[d] * kc_row[d];
        acc *= scale;
        scores[s] = acc;
        if (acc > maxs) maxs = acc;
    }

    // 3) softmax(kv_len 掩码 = 循环上界本身)
    float denom = 0.0f;
    for (int s = 0; s < kv_len; ++s) {
        scores[s] = expf(scores[s] - maxs);
        denom += scores[s];
    }

    // 4) 加权和
    float* od = out + ((unsigned long long)t * q_heads + h) * d_dim;
    for (unsigned int d = 0; d < d_dim; ++d) od[d] = 0.0f;
    for (int s = 0; s < kv_len; ++s) {
        float wgt = scores[s] / denom;
        const float* vc_row = vc + (unsigned long long)s * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        for (unsigned int d = 0; d < d_dim; ++d) od[d] += wgt * vc_row[d];
    }
}

// 单维窄切的物化拷贝(非连续;元数据视图做不了的维度走这里):
// dst[idx] = src[r * src_dim + start + d],idx = r*out_dim + d
extern "C" __global__ void owl_narrow_strided_f32(
    const float* src, float* dst,
    unsigned int outer, unsigned int src_dim,
    unsigned int start, unsigned int out_dim) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)outer * out_dim;
    if (idx >= total) return;
    unsigned int r = (unsigned int)(idx / out_dim);
    unsigned int d = (unsigned int)(idx % out_dim);
    dst[idx] = src[(unsigned long long)r * src_dim + start + d];
}

extern "C" __global__ void owl_fill_f32(float* out, float v, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v;
}

extern "C" __global__ void owl_recip_f32(const float* x, float* out, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / x[i];
}

// silu_and_mul:x = [rows, 2*cols](gate | up 横排),out = [rows, cols]
// (candle_nn::ops::silu_and_mul 直译;silu(gate) * up)
extern "C" __global__ void owl_silu_and_mul_f32(
    const float* x, float* out, unsigned long long cols) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = gridDim.x * blockDim.x;
    if (i >= total) return;
    unsigned long long row = i / cols;
    unsigned long long col = i % cols;
    const float* g = x + row * 2 * cols + col;
    const float* u = g + cols;
    float v = *g;
    float silu = v / (1.0f + expf(-v));
    out[i] = silu * (*u);
}

// u32 → f32 数值转换(非位型;dry-run 装载/位置面)
extern "C" __global__ void owl_u32_to_f32(
    const unsigned int* x, float* out, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)x[i];
}

// 2D 转置物化拷贝(out [cols, rows];dry-run 参考/拼接面)
extern "C" __global__ void owl_transpose2d_f32(
    const float* src, float* dst, unsigned long long rows, unsigned long long cols) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = rows * cols;
    if (i >= total) return;
    unsigned long long r = i / cols;
    unsigned long long c = i % cols;
    dst[c * rows + r] = src[r * cols + c];
}
