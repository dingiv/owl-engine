// ============================================================================
// DFlash2 草稿核族:非因果块 attention(v1 流式 + v2 split-K;classic/kNHD × f16/bf16/fp8)(2026-10-13 C 轮自 cu/owl/dflash2.cu 按**算子族**
// 拆出 —— 一算子一 `.cu`,与 ops/attn.rs 的声明对望)。
//
// 上游语义锚 = packages/sglang python/sglang/srt/models/dflash.py +
// kernels/ops/speculative/dflash.py(sglang-dflash-patch 分支,逐式对拍)。
//
// 数值律:输入 f16/bf16,累计 f32,输出随槽(契约 5:索引量 f32 过线;词表
// id < 2²⁴ 精确)。平局取最小索引(与 owl_argmax / torch.topk sorted 语义
// 一致,host 对拍同律)。输出块固定末参(槽序契约 4)。
//
//   q [T, Hq, D];keys/values = 前缀池行 + **自块核输入直读** k/v_self
//   [T, Hkv, D](写-读同核有跨 block 竞态,池回读免谈,直读 = 零 launch 内依赖)。
//   每查询行可见 kv_len = prefix+T 行(ENCODER_ONLY,is_causal=false 语义)。
//
// ⭐ 草稿自注意力**滑窗**(2026-10-13 D2 修正):检查点 config.json
//   `sliding_window = 2048` + `use_sliding_window = true`(5 层全滑窗)
//   → 非因果块注意力的**前缀**只取最近 2048 行(自块 8 行恒全可见)。
//   仅当前缀 > 2048 时生效(短 ctx 逐位不变)。
//
// ⚠️ v2 数值序变化(点积改 warp butterfly、softmax 改 split-K 归并)⇒
//   必须过 dflash2_nc_probe 的 host 参考门(1e-2)与引擎恒等门/AL 红线。
// ⚠️ v2 约束:d_dim 必须 32 整除且 DPL ∈ {1..8}。
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

#define OWL_NC_SLIDING_WINDOW 2048

// ---------------------------------------------------------------------------
// 4) 非因果块 attention(naive;测试 + eager 回退臂)
//
//   q [T, Hq, D];keys/values = 前缀池行 [0, prefix_len)(classic 布局,
//   寻址与 reshape_and_cache 同式)+ **自块核输入直读** k/v_self
//   [T, Hkv, D](行 s-prefix;写-读同核有跨 block 竞态,池回读免谈,
//   直读 = 零 launch 内依赖)。每查询行可见全部 kv_len = prefix+T 行
//   (ENCODER_ONLY,sglang dflash is_causal=false 语义)。
//
// 在线 softmax 单趟流式(零分数组,零窗上限);f16 读,f32 算,f16 写。
// grid = ceil(T*Hq/256),block 256。
// ---------------------------------------------------------------------------
// v2(E5-DF4 性能:block-per-head flash 式在线 softmax):
//   grid (T, Hq) × block (hd) —— 一 block 一 (t,h) 查询行,thread d 独占
//   一维(acc 单寄存器)。v1 一 thread 一行 + acc[512] 溢出 local memory +
//   全程单 block(82 SM 占用 1.2%),实测 804µs/次。每 kv 行一次块内
//   归约(q·k);m/l/p 行标量住 shared,thread-0 独写。
// ---- B6 偷显存:fp8kv 变体(前缀池 e4m3 读入即转;自块直读保持原 dtype)----
// B6.3 收口:恢复 f16 前缀版(fb1e453 曾直接改名删体 → (F16,false)
// 悬空 = spec+f16 草稿池 propose 首轮 CUDA_ERROR_NOT_FOUND;本函数 =
// pre-fb1e453 原体逐字回植,前缀池 f16 读)
extern "C" __global__ void owl_naive_attn_nc_f16(
    const __half* __restrict__ q,      // [T, Hq, D]
    const __half* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __half* __restrict__ v_self, // 同上
    const __half* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x]
    const __half* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page]
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __half* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __half2float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            kv = __half2float(kc[((long long)b * kv_heads + kvh) * hd_x * page * x
                                 + (long long)(d / x) * page * x + off * x + d % x]);
        } else {
            const int st = srow - prefix_len;
            kv = __half2float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            p *= __half2float(vc[((long long)b * kv_heads + kvh) * d_dim * page
                                 + (long long)d * page + off]);
        } else {
            const int st = srow - prefix_len;
            p *= __half2float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2half(acc / s_l);
}

extern "C" __global__ void owl_naive_attn_nc_fp8kv_f16(
    const __half* __restrict__ q,      // [T, Hq, D]
    const __half* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __half* __restrict__ v_self, // 同上
    const unsigned char* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x](e4m3)
    const unsigned char* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page](e4m3)
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __half* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __half2float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            // B6 偷显存:前缀池 e4m3 读入即转(元素序 = 字节序)
            __nv_fp8_e4m3 ke; ke.__x = kc[((long long)b * kv_heads + kvh) * hd_x * page * x
                                 + (long long)(d / x) * page * x + off * x + d % x];
            kv = __half2float(__half(ke));
        } else {
            const int st = srow - prefix_len;
            kv = __half2float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            __nv_fp8_e4m3 ve; ve.__x = vc[((long long)b * kv_heads + kvh) * d_dim * page
                                 + (long long)d * page + off];
            p *= __half2float(__half(ve));
        } else {
            const int st = srow - prefix_len;
            p *= __half2float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2half(acc / s_l);
}

// ---- BF16 v2(同款 block-per-head flash 式;池/出 bf16)----
extern "C" __global__ void owl_naive_attn_nc_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_self,
    const __nv_bfloat16* __restrict__ v_self,
    const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc,
    const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __nv_bfloat16* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __bfloat162float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            kv = __bfloat162float(kc[((long long)b * kv_heads + kvh) * hd_x * page * x
                                     + (long long)(d / x) * page * x + off * x + d % x]);
        } else {
            const int st = srow - prefix_len;
            kv = __bfloat162float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            p *= __bfloat162float(vc[((long long)b * kv_heads + kvh) * d_dim * page
                                     + (long long)d * page + off]);
        } else {
            const int st = srow - prefix_len;
            p *= __bfloat162float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2bfloat16(acc / s_l);
}

extern "C" __global__ void owl_naive_attn_nc_fp8kv_bf16(
    const __nv_bfloat16* __restrict__ q,      // [T, Hq, D]
    const __nv_bfloat16* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __nv_bfloat16* __restrict__ v_self, // 同上
    const unsigned char* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x](e4m3)
    const unsigned char* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page](e4m3)
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __nv_bfloat16* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __nv_bfloat16* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __bfloat162float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            // B6 偷显存:前缀池 e4m3 读入即转(元素序 = 字节序)
            __nv_fp8_e4m3 ke; ke.__x = kc[((long long)b * kv_heads + kvh) * hd_x * page * x
                                 + (long long)(d / x) * page * x + off * x + d % x];
            kv = __bfloat162float(__float2bfloat16_rn(float(ke)));
        } else {
            const int st = srow - prefix_len;
            kv = __bfloat162float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            __nv_fp8_e4m3 ve; ve.__x = vc[((long long)b * kv_heads + kvh) * d_dim * page
                                 + (long long)d * page + off];
            p *= __bfloat162float(__float2bfloat16_rn(float(ve)));
        } else {
            const int st = srow - prefix_len;
            p *= __bfloat162float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2bfloat16_rn(acc / s_l);
}

// ---- kNHD 变体(kv布局统一契约 P2):前缀池 [nb,page,hkv,hd],K/V 同式;自块直读不变 ----
extern "C" __global__ void owl_naive_attn_nc_f16_knhd(
    const __half* __restrict__ q,      // [T, Hq, D]
    const __half* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __half* __restrict__ v_self, // 同上
    const __half* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x]
    const __half* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page]
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __half* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __half2float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            kv = __half2float(kc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d]);
        } else {
            const int st = srow - prefix_len;
            kv = __half2float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            p *= __half2float(vc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d]);
        } else {
            const int st = srow - prefix_len;
            p *= __half2float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2half(acc / s_l);
}


extern "C" __global__ void owl_naive_attn_nc_fp8kv_f16_knhd(
    const __half* __restrict__ q,      // [T, Hq, D]
    const __half* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __half* __restrict__ v_self, // 同上
    const unsigned char* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x](e4m3)
    const unsigned char* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page](e4m3)
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __half* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __half2float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            // B6 偷显存:前缀池 e4m3 读入即转(元素序 = 字节序)
            __nv_fp8_e4m3 ke; ke.__x = kc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d];
            kv = __half2float(__half(ke));
        } else {
            const int st = srow - prefix_len;
            kv = __half2float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            __nv_fp8_e4m3 ve; ve.__x = vc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d];
            p *= __half2float(__half(ve));
        } else {
            const int st = srow - prefix_len;
            p *= __half2float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2half(acc / s_l);
}

// ---- BF16 v2(同款 block-per-head flash 式;池/出 bf16)----

extern "C" __global__ void owl_naive_attn_nc_bf16_knhd(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_self,
    const __nv_bfloat16* __restrict__ v_self,
    const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc,
    const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __nv_bfloat16* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __bfloat162float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            kv = __bfloat162float(kc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d]);
        } else {
            const int st = srow - prefix_len;
            kv = __bfloat162float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            p *= __bfloat162float(vc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d]);
        } else {
            const int st = srow - prefix_len;
            p *= __bfloat162float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2bfloat16(acc / s_l);
}


extern "C" __global__ void owl_naive_attn_nc_fp8kv_bf16_knhd(
    const __nv_bfloat16* __restrict__ q,      // [T, Hq, D]
    const __nv_bfloat16* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __nv_bfloat16* __restrict__ v_self, // 同上
    const unsigned char* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x](e4m3)
    const unsigned char* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page](e4m3)
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __nv_bfloat16* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __nv_bfloat16* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __bfloat162float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            // B6 偷显存:前缀池 e4m3 读入即转(元素序 = 字节序)
            __nv_fp8_e4m3 ke; ke.__x = kc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d];
            kv = __bfloat162float(__float2bfloat16_rn(float(ke)));
        } else {
            const int st = srow - prefix_len;
            kv = __bfloat162float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            __nv_fp8_e4m3 ve; ve.__x = vc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d];
            p *= __bfloat162float(__float2bfloat16_rn(float(ve)));
        } else {
            const int st = srow - prefix_len;
            p *= __bfloat162float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2bfloat16_rn(acc / s_l);
}

// ============================================================================
// NC 注意力 **v2 核**(2026-10-13 ③;只换并行结构,**语义与上面 `owl_naive_attn_nc_*`
// 逐字同一套**:非因果自块 + 前缀滑窗 2048 + 经典池布局 + kv_len_p 设备读)。
//
// 为什么:旧核每 KV 行一次 128 宽 block 归约(7 步树 + 共 11 次 __syncthreads)。
// 拆分实测(prefix 2048/窗口 2048,测试文件里的诊断核 diag_*):
//   纯同步链+softmax = **1.295ms** / +K 读 0.48 / +V 读 0.36 / 合计 2.237ms
//   ⇒ **同步链占 58%**,是唯一值得砍的大项(设计稿 D2 判断正确;V 的 sector
//   放大只值 0.36ms —— L2 把它兜住了,原估的"16× 放大"不成立)。
//
// 结构(lane→连续 dim + warp butterfly + split-K + 收尾一次归并):
//   ① 每 lane 持 DPL = d_dim/32 个**连续** dim(sr 与 q 同布局);
//   ② 逐行:lane 内点积 → **warp butterfly(__shfl_xor_sync × 5)** ⇒ 每个 lane
//      都拿到完整点积 ⇒ online softmax 的 m/l/resc 在 lane 间**逐位一致**
//      (无需共享、无需同步);
//   ③ 行的遍历区间按 warp 切成 split-K(floating 均分),各 warp 独立
//      online softmax;
//   ④ **每 block 只归并一次**(flash-decoding 式:smem 存各 warp 的 m/l/acc,
//      1 次 __syncthreads,再按本 lane 的 dim 合并 + 归一)。
//   ⇒ 内层循环零同步;全局同步从"每行 11 次"降到"每 block 1 次"。
//
// ⚠️ 数值序变化(点积改 butterfly、softmax 改 split-K 归并)⇒ 必须过
//   `dflash2_nc_probe` 的 host 参考门(1e-2)与引擎恒等门/AL 红线。
//
// ⚠️ 约束:d_dim 必须 32 整除且 DPL ∈ {1..8};x 取 dtype 的向量宽(bf16/f16 → 8)。
//   池 K 的 4/8 连续 dim 落在同一个 x 组内(对齐前提见工单 §P1)。
// ============================================================================

#define OWL_NC_DPL_MAX 8

/// 4 个连续元素 → float[4](8B 合曲读;bf16 = 2×bfloat162 / f16 = 2×half2)
template <typename T>
struct NcVec4;
template <>
struct NcVec4<__nv_bfloat16> {
    static __device__ __forceinline__ void load(const __nv_bfloat16* p, float* o) {
        uint2 v = *reinterpret_cast<const uint2*>(p);
        const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&v.x);
        const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
        o[0] = __bfloat162float(a.x);
        o[1] = __bfloat162float(a.y);
        o[2] = __bfloat162float(b.x);
        o[3] = __bfloat162float(b.y);
    }
};
template <>
struct NcVec4<__half> {
    static __device__ __forceinline__ void load(const __half* p, float* o) {
        uint2 v = *reinterpret_cast<const uint2*>(p);
        const __half2 a = *reinterpret_cast<const __half2*>(&v.x);
        const __half2 b = *reinterpret_cast<const __half2*>(&v.y);
        o[0] = __low2float(a);
        o[1] = __high2float(a);
        o[2] = __low2float(b);
        o[3] = __high2float(b);
    }
};

template <typename T>
struct NcConv;
template <>
struct NcConv<__nv_bfloat16> {
    static __device__ __forceinline__ float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __device__ __forceinline__ __nv_bfloat16 fromf(float v) { return __float2bfloat16(v); }
};
template <>
struct NcConv<__half> {
    static __device__ __forceinline__ float tof(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half fromf(float v) { return __float2half(v); }
};

template <typename T, int DPL>
__device__ __forceinline__ void nc_attn_loop(
    const T* __restrict__ q,
    const T* __restrict__ k_self,
    const T* __restrict__ v_self,
    const T* __restrict__ kc,
    const T* __restrict__ vc,
    const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    T* __restrict__ out) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarp = blockDim.x >> 5;
    // DPL>0 = 编译期定死(可矢量读);DPL==0 = 运行时 dpl(标量兜底)
    const int dpl = (DPL > 0) ? DPL : (d_dim >> 5);
    const int d0 = lane * dpl;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    const float scale = rsqrtf((float)d_dim);

    const T* qs = q + ((long long)t * q_heads + h) * d_dim;
    float qd[OWL_NC_DPL_MAX];
#pragma unroll
    for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
        qd[j] = (j < dpl) ? NcConv<T>::tof(qs[d0 + j]) : 0.0f;
    }

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    // 本 warp 的行切片(连续均分;余数归前几片)
    const int nrow = kv_len - srow_begin;
    const int per = (nrow + nwarp - 1) / nwarp;
    const int lo = srow_begin + warp * per;
    const int hi = (lo + per < kv_len) ? (lo + per) : kv_len;

    float m = -3.0e38f;
    float l = 0.0f;
    float acc[OWL_NC_DPL_MAX];
#pragma unroll
    for (int j = 0; j < OWL_NC_DPL_MAX; ++j) acc[j] = 0.0f;

    for (int srow = lo; srow < hi; ++srow) {
        const bool pre = (srow < prefix_len);
        const int b = pre ? (srow / page) : 0;
        const int off = pre ? (srow % page) : 0;
        float dot = 0.0f;
        if (DPL == 4 && (x % 4) == 0) {
            // 合曲路径:4 个连续 dim 落在同一 x 组(池)/ 本身就连续(自块)
            float kvf[4];
            if (pre) {
                const long long base = ((long long)b * kv_heads + kvh) * hd_x * page * x
                                       + (long long)off * x
                                       + (long long)(d0 / x) * page * x + (d0 % x);
                NcVec4<T>::load(kc + base, kvf);
            } else {
                const int st = srow - prefix_len;
                NcVec4<T>::load(k_self + ((long long)st * kv_heads + kvh) * d_dim + d0, kvf);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) dot += qd[j] * kvf[j];
        } else if (pre) {
            const long long base = ((long long)b * kv_heads + kvh) * hd_x * page * x
                                   + (long long)off * x;
#pragma unroll
            for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
                if (j < dpl) {
                    const int dim = d0 + j;
                    dot += qd[j]
                           * NcConv<T>::tof(kc[base + (long long)(dim / x) * page * x
                                               + (dim % x)]);
                }
            }
        } else {
            const int st = srow - prefix_len;
            const T* ks = k_self + ((long long)st * kv_heads + kvh) * d_dim;
#pragma unroll
            for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
                if (j < dpl) dot += qd[j] * NcConv<T>::tof(ks[d0 + j]);
            }
        }
        // warp butterfly:5 步后**每个 lane 都持完整点积**(故 softmax 态逐位一致)
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, o);
        }
        const float dot_s = dot * scale;
        const float m_new = fmaxf(m, dot_s);
        const float resc = __expf(m - m_new);
        const float p = __expf(dot_s - m_new);
        l = l * resc + p;
        m = m_new;
        if (pre) {
            const long long base = ((long long)b * kv_heads + kvh) * (long long)d_dim * page;
#pragma unroll
            for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
                if (j < dpl) {
                    acc[j] = acc[j] * resc
                             + p * NcConv<T>::tof(vc[base + (long long)(d0 + j) * page + off]);
                }
            }
        } else {
            const int st = srow - prefix_len;
            const T* vs = v_self + ((long long)st * kv_heads + kvh) * d_dim;
#pragma unroll
            for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
                if (j < dpl) acc[j] = acc[j] * resc + p * NcConv<T>::tof(vs[d0 + j]);
            }
        }
    }

    // ---- 收尾:每 block 一次跨 warp 归并(flash-decoding 式;1 次 __syncthreads)----
    __shared__ float s_acc[8][256];
    __shared__ float s_m[8];
    __shared__ float s_l[8];
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
#pragma unroll
    for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
        if (j < dpl) s_acc[warp][d0 + j] = acc[j];
    }
    __syncthreads();
    float mg = -3.0e38f;
    for (int w = 0; w < nwarp; ++w) mg = fmaxf(mg, s_m[w]);
    float lg = 0.0f;
    float ag[OWL_NC_DPL_MAX];
#pragma unroll
    for (int j = 0; j < OWL_NC_DPL_MAX; ++j) ag[j] = 0.0f;
    for (int w = 0; w < nwarp; ++w) {
        const float e = __expf(s_m[w] - mg);
        lg += s_l[w] * e;
#pragma unroll
        for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
            if (j < dpl) ag[j] += s_acc[w][d0 + j] * e;
        }
    }
    T* o = out + ((long long)t * q_heads + h) * d_dim;
#pragma unroll
    for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
        if (j < dpl) o[d0 + j] = NcConv<T>::fromf(ag[j] / lg);
    }
}

extern "C" __global__ void owl_naive_attn_nc_bf16_v2(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_self,
    const __nv_bfloat16* __restrict__ v_self, const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc, const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim, int page, int x,
    __nv_bfloat16* __restrict__ out) {
    const int dpl = d_dim >> 5;
    if (dpl == 4) {
        nc_attn_loop<__nv_bfloat16, 4>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,
                                       d_dim, page, x, out);
    } else if (dpl == 8) {
        nc_attn_loop<__nv_bfloat16, 8>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,
                                       d_dim, page, x, out);
    } else {
        nc_attn_loop<__nv_bfloat16, 0>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,
                                       d_dim, page, x, out);
    }
}
