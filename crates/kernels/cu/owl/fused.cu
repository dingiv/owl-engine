// ============================================================================
// owl 融合核(C1;2026-10-01 立项)
//
// Ampere-first 准则(driver 立项同步确立):
//   ① 最小发射数优先于最小流量 —— 一次 DRAM 往返,中间值全寄存器;
//   ② 128-bit 向量化访问(half2/uint4);
//   ③ grid 铺满 SM(按元素/按 (t,head) 开,不整行串行);
//   ④ nvrtc compute_86 + float 中间精度(与被替核同式,口径可对拍)。
//
// 本文件核族(全部单输出,SSA 契约友好):
//   owl_norm_rope_f16     —— qk-norm(×(1+w) / ×w)+ rotate-half partial rope
//                            三发合一(narrow+norm+rope;strided 读 q_raw 的
//                            per-head [value|gate] 半段,narrow 视图消失)
//   owl_silu_and_mul_f16  —— SwiGLU 门控 silu(g)⊙u 双输入单输出
//                            (port 语义 = vLLM silu_and_mul;dry_kernels
//                            挂账兑现,mlp.rs 头注 2026-09-26 立项)
//   owl_fused_add_rmsnorm_f16 —— residual 原地 += mixed + rmsnorm·w(vLLM port)
//
// BF16 变体(E5-DF3 同日十四;DFlash2 草稿路径 BF16 化,sglang 对齐):
//   owl_norm_rope_bf16          x/w/out bf16;cos/sin 保持 f16 指针
//                               (rope 表与 target 共享,免双表)
//   owl_silu_and_mul_bf16       全 bf16
//   owl_fused_add_rmsnorm_bf16  全 bf16(含 gamma;检查点原生 BF16)
// ============================================================================

// ---- nvrtc 序言(同 ops_pair 族:cuda_fp16 提供 __half;float 中间)----
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ----------------------------------------------------------------------------
// owl_norm_rope_f16:out[t,h,d] = rope( rmsnorm(x[t,h,·]) ×(1+w)^{w_off} )[d]
//
// x 寻址(strided;base + t·row_stride + h·head_stride + d)—— q 链:
//   base = q_raw [T, Hq·2HD](per-head [value|gate] 交错),row_stride =
//   Hq·2HD,head_stride = 2HD,value 半段 [0,HD);k 链:连续池
//   row_stride = Hkv·HD,head_stride = HD。
// 归一:per-(t,h) 行内 hd 元素 float 累计(与 owl_rmsnorm 同式);
// rope:rotate-half partial,配对 (d, d+half),[2·half, hd) 直通
//   (与 owl_rope_half_partial 逐式同源;float 中间精度同口径)。
// grid (tokens, heads, 1);block (hd, 1, 1)(d = threadIdx.x);
// smem = hd·4B(块内平方和归约)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_norm_rope_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ cos_t,   // [max_pos, half]
    const __half* __restrict__ sin_t,   // [max_pos, half]
    const float* __restrict__ pos,      // [tokens]
    float eps,
    size_t row_stride,
    size_t head_stride,
    size_t half,
    int w_off,
    __half* __restrict__ out)
{
    const size_t t = blockIdx.x;
    const size_t h = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const __half* xs = x + t * row_stride + h * head_stride;

    // pass 1:平方和(块内归约;float 累计,owl_rmsnorm 同式)
    extern __shared__ float smem[];
    const float xf = __half2float(xs[d]);
    smem[d] = xf * xf;
    __syncthreads();
    for (size_t s = hd / 2; s > 0; s >>= 1) {
        if (d < s) smem[d] += smem[d + s];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)hd + eps);

    // pass 2:归一 + rope 配对
    const float wf = __half2float(w[d]);
    const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    const size_t base = (t * gridDim.y + h) * hd;

    if (d < half) {
        // 配对 (d, d+half):伙伴侧 n 值就地重算(strided 读已入 L2)
        const float xb = __half2float(xs[d + half]);
        const float wb = __half2float(w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        out[base + d] = __float2half(n * cf - nb * sf);
        out[base + d + half] = __float2half(nb * cf + n * sf);
    } else if (d >= 2 * half) {
        out[base + d] = __float2half(n);   // partial 维直通
    }
}

// ----------------------------------------------------------------------------
// owl_silu_and_mul_f16:out = silu(g) ⊙ u(逐元素;SwiGLU 门控)。
// half2 向量化(n 偶数;inter 全偶保证);float 中间 —— 被替两发核
// (silu 写 half → mul 读 half)的中间量化在此消除(精度红利)。
// grid = ceil(n / 512);block 256(每线程 2 元素)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_silu_and_mul_f16(
    const __half* __restrict__ g,
    const __half* __restrict__ u,
    size_t n,
    __half* __restrict__ out)
{
    const size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < n) {
        const __half2 gv = *reinterpret_cast<const __half2*>(g + i);
        const __half2 uv = *reinterpret_cast<const __half2*>(u + i);
        const float2 gf = __half22float2(gv);
        const float2 uf = __half22float2(uv);
        float2 r;
        r.x = (gf.x / (1.0f + expf(-gf.x))) * uf.x;
        r.y = (gf.y / (1.0f + expf(-gf.y))) * uf.y;
        *reinterpret_cast<__half2*>(out + i) = __floats2half2_rn(r.x, r.y);
    }
}

// ----------------------------------------------------------------------------
// owl_fused_add_rmsnorm_f16(vLLM layernorm_kernels.cu fused_add_rms_norm port;
// 双原地语义 —— residual 块原地 += mixed(已知副作用,GDN conv_upd 同款
// 单流保序律),out = rmsnorm(residual)·w^{w_off} 单输出)。
// 替 decoder 层 [残差 add + post_ln rmsnorm] 两发;grid (rows,1,1),
// block (256,1,1),smem 256·4B(行归约;n = hidden 5120,行内分段循环)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_fused_add_rmsnorm_f16(
    const __half* __restrict__ mixed,       // mixer 输出(只读)
    __half* __restrict__ residual,          // in/out:+= mixed(原地副作用)
    const __half* __restrict__ w,           // [n]
    float eps,
    size_t n,
    int w_off,
    __half* __restrict__ out)               // [rows, n] = rmsnorm(residual)·w
{
    const size_t row = blockIdx.x;
    const size_t base = row * n;
    __half* r = residual + base;
    const __half* m = mixed + base;
    __shared__ float smem[256];

    float local = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(m[i]) + __half2float(r[i]);
        r[i] = __float2half(v);            // 原地写(回收前根已收割,单流序)
        local += v * v;
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (size_t s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) smem[threadIdx.x] += smem[threadIdx.x + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)n + eps);
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(r[i]);
        const float wf = __half2float(w[i]);
        out[base + i] = __float2half(v * inv * (w_off ? (wf + 1.0f) : wf));
    }
}

// ----------------------------------------------------------------------------
// owl_qknorm_rope_kv_insert_f16(Wave-2 头号;port 源 = vLLM
// fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu 的 (token, head-slot)
// warp 结构,适配 owl gated 布局与 classic cache 寻址):
//
//   q 头:blockIdx.y ∈ [0, Hq)     —— q_raw 的 value 半段 strided 读 →
//                                    qk-norm(×(1+w)^{w_off})+ rope → q_out
//   k 头:blockIdx.y ∈ [Hq, Hq+Hkv) —— k 的 qk-norm + rope → **key_cache
//                                    slot 散写**;同块捎带 v → value_cache
//
// 替三发:norm_rope(q)+ norm_rope(k)+ reshape_and_cache(k,v)。
// q_out = 尾参(v1 注意力输入;cache 写 = 原地副作用,单流保序 ——
// v1 为本核输出块的消费者,树序天然后置)。
// cache 寻址 = reshape_and_cache 逐字(classic:kc [nb,Hkv,D/x,page,x],
// vc [nb,Hkv,D,page];slot<0 = padding 跳写,k 头仍出 q…… k 头 return
// 前 q_out 与本核无关,直接 return 安全)。
// grid (T, Hq+Hkv, 1);block (hd, 1, 1);smem = hd·4B(行归约)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_qknorm_rope_kv_insert_f16(
    const __half* __restrict__ q_raw,   // [T, Hq·2HD](per-head [value|gate])
    const __half* __restrict__ k,       // [T, Hkv·HD]
    const __half* __restrict__ v,       // [T, Hkv·HD]
    __half* __restrict__ key_cache,     // [nb, Hkv, hd/x, page, x]
    __half* __restrict__ value_cache,   // [nb, Hkv, hd, page]
    const float* __restrict__ slots,    // [T]
    const __half* __restrict__ q_w,     // [hd]
    const __half* __restrict__ k_w,     // [hd]
    const __half* __restrict__ cos_t,   // [max_pos, half]
    const __half* __restrict__ sin_t,   // [max_pos, half]
    const float* __restrict__ pos,      // [T]
    float eps,
    int hkv, int half, int page, int w_off,
    __half* __restrict__ q_out)         // [T, Hq·hd](尾参契约 4)
{
    const size_t t = blockIdx.x;
    const size_t head = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const size_t hq = gridDim.y - (size_t)hkv;
    const long long slot = (long long)slots[t];

    if (head < hq) {
        // ---- q:value 半段 strided 读 + norm + rope → q_out ----
        const size_t row_stride = hq * 2 * hd;
        const __half* xs = q_raw + t * row_stride + head * 2 * hd;
        extern __shared__ float smem[];
        const float xf = __half2float(xs[d]);
        smem[d] = xf * xf;
        __syncthreads();
        for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) smem[d] += smem[d + s2];
            __syncthreads();
        }
        const float inv = rsqrtf(smem[0] / (float)hd + eps);
        const float wf = __half2float(q_w[d]);
        const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
        const size_t p = (size_t)pos[t];
        const size_t base = (t * hq + head) * hd;
        if (d < (size_t)half) {
            const float xb = __half2float(xs[d + half]);
            const float wb = __half2float(q_w[d + half]);
            const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
            const float cf = __half2float(cos_t[p * half + d]);
            const float sf = __half2float(sin_t[p * half + d]);
            q_out[base + d] = __float2half(n * cf - nb * sf);
            q_out[base + d + half] = __float2half(nb * cf + n * sf);
        } else if (d >= 2 * (size_t)half) {
            q_out[base + d] = __float2half(n);
        }
            return;
    }

    // ---- k 头:qk-norm + rope → key_cache;捎带 v → value_cache ----
    if (slot < 0) return;   // padding 跳写(K0 契约)
    const size_t kh = head - hq;
    const __half* ks = k + t * (size_t)hkv * hd + kh * hd;
    extern __shared__ float smem[];
    const float kf = __half2float(ks[d]);
    smem[d] = kf * kf;
    __syncthreads();
    for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
        if (d < s2) smem[d] += smem[d + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)hd + eps);
    const float wf = __half2float(k_w[d]);
    float n = kf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    if (d < (size_t)half) {
        // 配对 (d, d+half):每 thread 只写自己的 d 位(伙伴位由 d+half 写)
        const float xb = __half2float(ks[d + half]);
        const float wb = __half2float(k_w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        n = n * cf - nb * sf;
    } else if (d >= 2 * (size_t)half) {
        // partial 维直通(n 已是 norm 值)
    } else {
        // d ∈ [half, 2half):伙伴 = d - half(基值 = 自己 n[d],旋转伙伴 n[d-half]
        // —— 2026-10-02 转置案修复:曾写 nb*cf + n*sf(基/伙伴互换,pos=0 时
        // 直接写 n[d-half],引擎 kv-hex 3-4% 偏差真凶)
        const float xb = __half2float(ks[d - half]);
        const float wb = __half2float(k_w[d - half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d - half]);
        const float sf = __half2float(sin_t[p * half + d - half]);
        n = n * cf + nb * sf;
    }
    // 诊断(隔离 hd128 档 slot 2 取证):分段值打印
    const int block_idx = (int)(slot / page);
    const int off = (int)(slot % page);
    const size_t kk = ((block_idx * (size_t)hkv + kh) * (hd / 8) + d / 8) * page * 8
                    + off * 8 + d % 8;
    key_cache[kk] = __float2half(n);
    // v 捎带拷贝(线性寻址)
    const size_t vi = ((block_idx * (size_t)hkv + kh) * hd + d) * page + off;
    value_cache[vi] = v[t * (size_t)hkv * hd + kh * hd + d];
}

// ----------------------------------------------------------------------------
// owl_qknorm_rope_kv_insert_f16_fp8kv(B6.3 收口;2026-10-10):fp8 e4m3
// 主池变体。数学逐式同 f16 版(同一 norm/rope);差异仅池写 ——
//   key_cache/value_cache = e4m3 1B/elem(u8 寻址,元素序不变),
//   池写 = __nv_fp8_e4m3(value) SATFINITE(与 K0 写核同转换)。
// 背景:fp8 池下 f16 版 2B 存储落在 2× 字节偏移 —— 真槽区永不落笔
// (kv-dump 实证 decode 槽全零)+ f16 位型毒液洒 2s 槽区 = decode 崩坏
// 真凶(B6.2 曾覆盖 K0/chunked/v2 读三路,独漏此融合插池路)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_qknorm_rope_kv_insert_f16_fp8kv(
    const __half* __restrict__ q_raw,   // [T, Hq·2HD](per-head [value|gate])
    const __half* __restrict__ k,       // [T, Hkv·HD]
    const __half* __restrict__ v,       // [T, Hkv·HD]
    unsigned char* __restrict__ key_cache,   // e4m3 [nb, Hkv, hd/x, page, x]
    unsigned char* __restrict__ value_cache, // e4m3 [nb, Hkv, hd, page]
    const float* __restrict__ slots,    // [T]
    const __half* __restrict__ q_w,     // [hd]
    const __half* __restrict__ k_w,     // [hd]
    const __half* __restrict__ cos_t,   // [max_pos, half]
    const __half* __restrict__ sin_t,   // [max_pos, half]
    const float* __restrict__ pos,      // [T]
    float eps,
    int hkv, int half, int page, int w_off,
    __half* __restrict__ q_out)         // [T, Hq·hd](尾参契约 4)
{
    const size_t t = blockIdx.x;
    const size_t head = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const size_t hq = gridDim.y - (size_t)hkv;
    const long long slot = (long long)slots[t];

    if (head < hq) {
        // ---- q:value 半段 strided 读 + norm + rope → q_out ----
        const size_t row_stride = hq * 2 * hd;
        const __half* xs = q_raw + t * row_stride + head * 2 * hd;
        extern __shared__ float smem[];
        const float xf = __half2float(xs[d]);
        smem[d] = xf * xf;
        __syncthreads();
        for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) smem[d] += smem[d + s2];
            __syncthreads();
        }
        const float inv = rsqrtf(smem[0] / (float)hd + eps);
        const float wf = __half2float(q_w[d]);
        const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
        const size_t p = (size_t)pos[t];
        const size_t base = (t * hq + head) * hd;
        if (d < (size_t)half) {
            const float xb = __half2float(xs[d + half]);
            const float wb = __half2float(q_w[d + half]);
            const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
            const float cf = __half2float(cos_t[p * half + d]);
            const float sf = __half2float(sin_t[p * half + d]);
            q_out[base + d] = __float2half(n * cf - nb * sf);
            q_out[base + d + half] = __float2half(nb * cf + n * sf);
        } else if (d >= 2 * (size_t)half) {
            q_out[base + d] = __float2half(n);
        }
            return;
    }

    // ---- k 头:qk-norm + rope → key_cache;捎带 v → value_cache ----
    if (slot < 0) return;   // padding 跳写(K0 契约)
    const size_t kh = head - hq;
    const __half* ks = k + t * (size_t)hkv * hd + kh * hd;
    extern __shared__ float smem[];
    const float kf = __half2float(ks[d]);
    smem[d] = kf * kf;
    __syncthreads();
    for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
        if (d < s2) smem[d] += smem[d + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)hd + eps);
    const float wf = __half2float(k_w[d]);
    float n = kf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    if (d < (size_t)half) {
        // 配对 (d, d+half):每 thread 只写自己的 d 位(伙伴位由 d+half 写)
        const float xb = __half2float(ks[d + half]);
        const float wb = __half2float(k_w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        n = n * cf - nb * sf;
    } else if (d >= 2 * (size_t)half) {
        // partial 维直通(n 已是 norm 值)
    } else {
        // d ∈ [half, 2half):伙伴 = d - half(基值 = 自己 n[d],旋转伙伴 n[d-half]
        // —— 2026-10-02 转置案修复:曾写 nb*cf + n*sf(基/伙伴互换,pos=0 时
        // 直接写 n[d-half],引擎 kv-hex 3-4% 偏差真凶)
        const float xb = __half2float(ks[d - half]);
        const float wb = __half2float(k_w[d - half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d - half]);
        const float sf = __half2float(sin_t[p * half + d - half]);
        n = n * cf + nb * sf;
    }
    const int block_idx = (int)(slot / page);
    const int off = (int)(slot % page);
    const size_t kk = ((block_idx * (size_t)hkv + kh) * (hd / 8) + d / 8) * page * 8
                    + off * 8 + d % 8;
    key_cache[kk] = __nv_fp8_e4m3(n).__x;
    // v 捎带拷贝(线性寻址;e4m3 转换)
    const size_t vi = ((block_idx * (size_t)hkv + kh) * hd + d) * page + off;
    value_cache[vi] = __nv_fp8_e4m3(__half2float(v[t * (size_t)hkv * hd + kh * hd + d])).__x;
}

// ----------------------------------------------------------------------------
// owl_norm_rope_bf16(E5-DF3 同日十四):x/w/out = BF16;cos/sin 保持 f16
// 指针(rope 表与 target 共享,免双表/免拷贝)。数学逐式同 f16 版。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_norm_rope_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __half* __restrict__ cos_t,   // [max_pos, half](f16 共享表)
    const __half* __restrict__ sin_t,   // [max_pos, half]
    const float* __restrict__ pos,      // [tokens]
    float eps,
    size_t row_stride,
    size_t head_stride,
    size_t half,
    int w_off,
    __nv_bfloat16* __restrict__ out)
{
    const size_t t = blockIdx.x;
    const size_t h = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const __nv_bfloat16* xs = x + t * row_stride + h * head_stride;

    extern __shared__ float smem[];
    const float xf = __bfloat162float(xs[d]);
    smem[d] = xf * xf;
    __syncthreads();
    for (size_t s = hd / 2; s > 0; s >>= 1) {
        if (d < s) smem[d] += smem[d + s];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)hd + eps);

    const float wf = __bfloat162float(w[d]);
    const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    const size_t base = (t * gridDim.y + h) * hd;

    if (d < half) {
        const float xb = __bfloat162float(xs[d + half]);
        const float wb = __bfloat162float(w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        out[base + d] = __float2bfloat16(n * cf - nb * sf);
        out[base + d + half] = __float2bfloat16(nb * cf + n * sf);
    } else if (d >= 2 * half) {
        out[base + d] = __float2bfloat16(n);   // partial 维直通
    }
}

// ----------------------------------------------------------------------------
// owl_silu_and_mul_bf16:全 bf16,逐式同 f16 版(float 中间;bf16 无 half2
// 便捷对 —— 逐元素标量处理,带宽非瓶颈位)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_silu_and_mul_bf16(
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ u,
    size_t n,
    __nv_bfloat16* __restrict__ out)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gf = __bfloat162float(g[i]);
        const float uf = __bfloat162float(u[i]);
        out[i] = __float2bfloat16((gf / (1.0f + expf(-gf))) * uf);
    }
}

// ----------------------------------------------------------------------------
// owl_fused_add_rmsnorm_bf16:全 bf16(含 gamma;检查点原生 BF16)。
// residual 原地 += mixed + rmsnorm·w,逐式同 f16 版。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_fused_add_rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ mixed,       // mixer 输出(只读)
    __nv_bfloat16* __restrict__ residual,          // in/out:+= mixed(原地副作用)
    const __nv_bfloat16* __restrict__ w,           // [n]
    float eps,
    size_t n,
    int w_off,
    __nv_bfloat16* __restrict__ out)               // [rows, n] = rmsnorm(residual)·w
{
    const size_t row = blockIdx.x;
    const size_t base = row * n;
    __nv_bfloat16* r = residual + base;
    const __nv_bfloat16* m = mixed + base;
    __shared__ float smem[256];

    float local = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(m[i]) + __bfloat162float(r[i]);
        r[i] = __float2bfloat16(v);            // 原地写(回收前根已收割,单流序)
        local += v * v;
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (size_t s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) smem[threadIdx.x] += smem[threadIdx.x + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)n + eps);
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(r[i]);
        const float wf = __bfloat162float(w[i]);
        out[base + i] = __float2bfloat16(v * inv * (w_off ? (wf + 1.0f) : wf));
    }
}
