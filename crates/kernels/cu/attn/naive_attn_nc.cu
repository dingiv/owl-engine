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

// **草稿 ring 槽映射**(2026-10-22):0 = 线性(既有全池路径);>0 = 逻辑位对 ring
// 取模。ring_cap 必须 >= 滑窗 + 自块(T),否则不同逻辑位会撞槽(核内 __trap)。
__device__ __forceinline__ int nc_ring_slot(int srow, int ring_cap) {
    return ring_cap > 0 ? (srow % ring_cap) : srow;
}

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
    int page, int x, int ring_cap,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    // ring 门:窗口 + 自块必须互不覆盖(违反 = 硬错,不静默错槽)
    if (ring_cap != 0 && ring_cap < OWL_NC_SLIDING_WINDOW + (int)gridDim.x) { __trap(); }

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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
    int page, int x, int ring_cap,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    // ring 门:窗口 + 自块必须互不覆盖(违反 = 硬错,不静默错槽)
    if (ring_cap != 0 && ring_cap < OWL_NC_SLIDING_WINDOW + (int)gridDim.x) { __trap(); }

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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
    int page, int x, int ring_cap,
    __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    // ring 门:窗口 + 自块必须互不覆盖(违反 = 硬错,不静默错槽)
    if (ring_cap != 0 && ring_cap < OWL_NC_SLIDING_WINDOW + (int)gridDim.x) { __trap(); }

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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
    int page, int x, int ring_cap,
    __nv_bfloat16* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    // ring 门:窗口 + 自块必须互不覆盖(违反 = 硬错,不静默错槽)
    if (ring_cap != 0 && ring_cap < OWL_NC_SLIDING_WINDOW + (int)gridDim.x) { __trap(); }

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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
    int page, int x, int ring_cap,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    // ring 门:窗口 + 自块必须互不覆盖(违反 = 硬错,不静默错槽)
    if (ring_cap != 0 && ring_cap < OWL_NC_SLIDING_WINDOW + (int)gridDim.x) { __trap(); }

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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
    int page, int x, int ring_cap,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    // ring 门:窗口 + 自块必须互不覆盖(违反 = 硬错,不静默错槽)
    if (ring_cap != 0 && ring_cap < OWL_NC_SLIDING_WINDOW + (int)gridDim.x) { __trap(); }

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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
    int page, int x, int ring_cap,
    __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    // ring 门:窗口 + 自块必须互不覆盖(违反 = 硬错,不静默错槽)
    if (ring_cap != 0 && ring_cap < OWL_NC_SLIDING_WINDOW + (int)gridDim.x) { __trap(); }

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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
    int page, int x, int ring_cap,
    __nv_bfloat16* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    // ring 门:窗口 + 自块必须互不覆盖(违反 = 硬错,不静默错槽)
    if (ring_cap != 0 && ring_cap < OWL_NC_SLIDING_WINDOW + (int)gridDim.x) { __trap(); }

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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
            const int slot = nc_ring_slot(srow, ring_cap);
            const int b = slot / page;
            const int off = slot % page;
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
// **GQA 感知(2026-10-18,propose 立案;grid 由 (T,Hq) 改 (T,Hkv))**:
//   同一 kv-head 的 `group = q_heads/kv_heads` 个 q-head 曾各起一个 CTA
//   ⇒ 同一批 K/V 行被读 group 遍(草稿档 group=4:实测 268 MB/发、~225 GB/s
//   ⇒ 1.19 ms,占 propose 图 66%)。现改:**一个 CTA 覆盖一个 kv-head 的全部
//   q-head** —— CTA 内 warp 映射 `wq = warp % group`(组内 q-head)、
//   `sub = warp / group`(行切片,`nsub = nwarp/group`),同组 warp 走 L1 共享
//   同一行 ⇒ HBM 流量 ÷ group,且行遍历同序 ⇒ 数值序不变(除 split-K 归并宽度)。
//   结构门:仅当 `nwarp ≥ group` 且 `nwarp % group == 0` 时可用(驱动件选核;
//   越界 = `__trap()` 硬错,不静默)。
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
    int page, int x, int ring_cap,
    T* __restrict__ out) {
    const int t = blockIdx.x;
    const int kvh = blockIdx.y;   // ← GQA:一个 CTA = 一个 kv-head(其全部 q-head)
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarp = blockDim.x >> 5;
    const int group = q_heads / kv_heads;   // GQA 组宽
    // **结构门**:驱动件只在本几何下选本核;越界 = 硬错(不静默、不半跑)
    if (group <= 0 || nwarp < group || (nwarp % group) != 0) {
        __trap();
    }
    const int wq = warp % group;            // 组内 q-head 序号
    const int sub = warp / group;           // 行切片序号
    const int nsub = nwarp / group;         // 行切片数
    const int h = kvh * group + wq;         // 全局 q-head
    // DPL>0 = 编译期定死(可矢量读);DPL==0 = 运行时 dpl(标量兜底)
    const int dpl = (DPL > 0) ? DPL : (d_dim >> 5);
    const int d0 = lane * dpl;
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    // ring 门:窗口 + 自块必须互不覆盖(违反 = 硬错,不静默错槽)
    if (ring_cap != 0 && ring_cap < OWL_NC_SLIDING_WINDOW + (int)gridDim.x) { __trap(); }
    const float scale = rsqrtf((float)d_dim);

    const T* qs = q + ((long long)t * q_heads + h) * d_dim;
    constexpr int DPLN = (DPL > 0) ? DPL : OWL_NC_DPL_MAX;
    float qd[DPLN];
#pragma unroll
    for (int j = 0; j < DPLN; ++j) {
        qd[j] = (j < dpl) ? NcConv<T>::tof(qs[d0 + j]) : 0.0f;
    }

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    // 本 warp 的行切片(同 q-head 的 nsub 个 warp 连续均分;余数归前几片)
    const int nrow = kv_len - srow_begin;
    const int per = (nrow + nsub - 1) / nsub;
    const int lo = srow_begin + sub * per;
    const int hi = (lo + per < kv_len) ? (lo + per) : kv_len;

    float m = -3.0e38f;
    float l = 0.0f;
    float acc[DPLN];
#pragma unroll
    for (int j = 0; j < DPLN; ++j) acc[j] = 0.0f;

    // ⚠️ **不展开行循环**(2026-10-18 实测):`#pragma unroll 4` 会让编译器把
    // online-softmax 的累加重排 ⇒ 装配期 oracle 门(2e-2)当场红 ⇒ 保持串行序。
    for (int srow = lo; srow < hi; ++srow) {
        const bool pre = (srow < prefix_len);
        const int slot = pre ? nc_ring_slot(srow, ring_cap) : srow;
        const int b = pre ? (slot / page) : 0;
        const int off = pre ? (slot % page) : 0;
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
            for (int j = 0; j < DPLN; ++j) {
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
            for (int j = 0; j < DPLN; ++j) {
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
            for (int j = 0; j < DPLN; ++j) {
                if (j < dpl) {
                    acc[j] = acc[j] * resc
                             + p * NcConv<T>::tof(vc[base + (long long)(d0 + j) * page + off]);
                }
            }
        } else {
            const int st = srow - prefix_len;
            const T* vs = v_self + ((long long)st * kv_heads + kvh) * d_dim;
#pragma unroll
            for (int j = 0; j < DPLN; ++j) {
                if (j < dpl) acc[j] = acc[j] * resc + p * NcConv<T>::tof(vs[d0 + j]);
            }
        }
    }

    // ---- 收尾:每 block 一次归并(**只并同 q-head 的 nsub 个 warp**;
    //      warp 下标 = wq + s2*group;1 次 __syncthreads)----
    // ⚠️ **动态 smem**(不是 `__shared__` 静态):本函数被 DPL=4/8/0 三个实例化
    // 共用,静态声明会**各自占一份**(3×16KB ⇒ 超 48KB 静态上限,ptxas 报
    // "too much shared data")。布局由驱动件按 `nwarp*(d_dim+2)*4` 申报:
    //   [nwarp][d_dim] 归并 acc ‖ [nwarp] m ‖ [nwarp] l
    extern __shared__ float s_nc[];
    float* s_acc = s_nc;                        // nwarp × d_dim
    float* s_m = s_nc + nwarp * d_dim;
    float* s_l = s_m + nwarp;
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
#pragma unroll
    for (int j = 0; j < DPLN; ++j) {
        if (j < dpl) s_acc[warp * d_dim + d0 + j] = acc[j];
    }
    __syncthreads();
    float mg = -3.0e38f;
    for (int s2 = 0; s2 < nsub; ++s2) mg = fmaxf(mg, s_m[wq + s2 * group]);
    float lg = 0.0f;
    float ag[DPLN];
#pragma unroll
    for (int j = 0; j < DPLN; ++j) ag[j] = 0.0f;
    for (int s2 = 0; s2 < nsub; ++s2) {
        const int w = wq + s2 * group;
        const float e = __expf(s_m[w] - mg);
        lg += s_l[w] * e;
#pragma unroll
        for (int j = 0; j < DPLN; ++j) {
            if (j < dpl) ag[j] += s_acc[w * d_dim + d0 + j] * e;
        }
    }
    // 落盘只由切片 0 做(同组各 warp 的归并结果逐位相同 ⇒ 冗余写无必要)
    if (sub == 0) {
        T* o = out + ((long long)t * q_heads + h) * d_dim;
#pragma unroll
        for (int j = 0; j < DPLN; ++j) {
            if (j < dpl) o[d0 + j] = NcConv<T>::fromf(ag[j] / lg);
        }
    }
}

extern "C" __global__ void owl_naive_attn_nc_bf16_v2(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_self,
    const __nv_bfloat16* __restrict__ v_self, const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc, const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim, int page, int x, int ring_cap,
    __nv_bfloat16* __restrict__ out) {
    const int dpl = d_dim >> 5;
    if (dpl == 4) {
        nc_attn_loop<__nv_bfloat16, 4>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,
                                       d_dim, page, x, ring_cap, out);
    } else if (dpl == 8) {
        nc_attn_loop<__nv_bfloat16, 8>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,
                                       d_dim, page, x, ring_cap, out);
    } else {
        nc_attn_loop<__nv_bfloat16, 0>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,
                                       d_dim, page, x, ring_cap, out);
    }
}

// ============================================================================
// v3(2026-10-26 propose 刀1):**页粒度共享 K/V** + 键内零同步
// ---------------------------------------------------------------------------
// v2(上)在 propose 形状下的两处浪费(probe 实测:prefix 2048 / T=8 / Hq=16 /
// Hkv=4 / 单发 ≈ 0.5ms/层,5 层占 propose 图 ~55%):
//   ① V 是**标量散读**:池 V 布局 [nb,hkv,D,page] 下,一个 key 的 128 维散落在
//      128 个 32B 扇区(每扇区只用 2B)⇒ 单 key 单 warp 128 次事务;
//   ② nsub 个 warp 各切**连续** key 段 ⇒ 同 CTA 内 warp 读不同页,shared 无从
//      复用(只能吃 L1/L2 的整页时间局部性)。
// v3 把 key 循环改成**页(池块)粒度**:
//   - 一页的 K 块((D/x)×page×x)与 V 块(D×page)在池里都是**连续 8KB**,按 16B
//     整块搬进 shared(全 CTA 共享)⇒ 事务数 = 元素数/8,且整页只读一遍;
//   - key 分片改成**页内条带**(sub 取 j ∈ [sub·page/nsub, (sub+1)·page/nsub)),
//     全 CTA 同页 ⇒ shared 复用成立;跨条带归并与 v2 同式(s_m/s_l/s_acc);
//   - 键内**零 __syncthreads**(dot 的 5 步 butterfly 让每 lane 持完整点积 ⇒
//     softmax 态逐 lane 一致,与 v2 同律)。
// 语义/槽序/平局律与 v2 逐字同一套(只有 f32 累加顺序不同 ⇒ 门 = host 参考 1e-2)。
// 几何门(驱动件 + 核内 __trap 双保险):32|d_dim、group ≤ nwarp 且整除、
// page ≤ 64、ring_cap ≥ 窗口+自块。
// smem = 2·page·d_dim·sizeof(KVT) + nwarp·(d_dim+2)·4(≤ 48KB 免 opt-in)。
// ============================================================================
#define OWL_NC_TILE_MAX_PAGE 64

template <typename KVT>
struct NcTileConv;
template <>
struct NcTileConv<__nv_bfloat16> {
    static __device__ __forceinline__ float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
};
template <>
struct NcTileConv<__half> {
    static __device__ __forceinline__ float tof(__half v) { return __half2float(v); }
};
template <>
struct NcTileConv<unsigned char> {
    static __device__ __forceinline__ float tof(unsigned char v) {
        __nv_fp8_e4m3 e;
        e.__x = v;
        return __half2float(__half(e));
    }
};

// 一页前缀 → shared([pos][dim] 行主序;pos-major 让 key 内 16B 读与转置写都无冲突)
template <typename KVT>
__device__ __forceinline__ void nc_tile_load(
    const KVT* __restrict__ kc, const KVT* __restrict__ vc,
    KVT* s_k, KVT* s_v,
    int b, int kvh, int kv_heads, int d_dim, int page, int x, int ring_cap,
    int tid, int nthreads) {
    const int hd_x = d_dim / x;
    const int esz = (int)sizeof(KVT);
    // ring 页对齐(`ring_cap % page == 0`,生产档 2080/32 恒成立)⇒ 一个逻辑页
    // 整页映到另一个**槽页**,页内 32 个逻辑位的槽号仍连续 ⇒ 快速路照走。
    // 非页对齐 ring 走慢速路(正确但逐元素;不静默错槽)。
    const bool ring_paged = (ring_cap == 0) || (ring_cap % page) == 0;
    if (ring_paged && x * esz == 16 && (page % 8) == 0) {
        // 快速路:池页块连续 ⇒ 16B 整块搬(每 thread 1~2 块)
        const int bp = (ring_cap == 0)
                           ? b
                           : (int)((((long long)b * page) % ring_cap) / page);
        const long long kb = ((long long)bp * kv_heads + kvh) * (long long)hd_x * page * x;
        const int nck = hd_x * page;
        for (int c = tid; c < nck; c += nthreads) {
            const int g = c / page;
            const int off = c - g * page;
            const uint4 v = *reinterpret_cast<const uint4*>(kc + kb + (long long)c * x);
            *reinterpret_cast<uint4*>(s_k + (long long)off * d_dim + g * x) = v;
        }
        const long long vb = ((long long)bp * kv_heads + kvh) * (long long)d_dim * page;
        const int per = page / 8;
        const int ncv = d_dim * per;
        for (int c = tid; c < ncv; c += nthreads) {
            const int d = c / per;
            const int q8 = c - d * per;
            const uint4 v = *reinterpret_cast<const uint4*>(vc + vb + (long long)d * page + q8 * 8);
            const KVT* tmp = reinterpret_cast<const KVT*>(&v);
#pragma unroll
            for (int j = 0; j < 8; ++j) s_v[(long long)(q8 * 8 + j) * d_dim + d] = tmp[j];
        }
    } else {
        // 慢速路(ring 槽映射 / 非 16B 组):逐元素,槽按逻辑位现算
        for (int j = 0; j < page; ++j) {
            const int srow = b * page + j;
            const int slot = nc_ring_slot(srow, ring_cap);
            const int bb = slot / page;
            const int off = slot - bb * page;
            const long long kb = ((long long)bb * kv_heads + kvh) * (long long)hd_x * page * x;
            const long long vb = ((long long)bb * kv_heads + kvh) * (long long)d_dim * page;
            for (int d = tid; d < d_dim; d += nthreads) {
                s_k[(long long)j * d_dim + d] =
                    kc[kb + (long long)(d / x) * page * x + (long long)off * x + (d % x)];
                s_v[(long long)j * d_dim + d] = vc[vb + (long long)d * page + off];
            }
        }
    }
}

// 单 key 的 online-softmax 一步(dot → 5 步 butterfly → 更新 → PV)
template <typename KVT, int DPL>
__device__ __forceinline__ void nc_tile_key(
    const KVT* __restrict__ s_k, const KVT* __restrict__ s_v, long long off, int d0,
    const float* qd, float scale, float& m, float& l, float* acc) {
    KVT raw[DPL];
    const KVT* kp = s_k + off + d0;
    if (DPL * (int)sizeof(KVT) == 8) {
        *reinterpret_cast<uint2*>(raw) = *reinterpret_cast<const uint2*>(kp);
    } else if (DPL * (int)sizeof(KVT) == 16) {
        *reinterpret_cast<uint4*>(raw) = *reinterpret_cast<const uint4*>(kp);
    } else {
#pragma unroll
        for (int i = 0; i < DPL; ++i) raw[i] = kp[i];
    }
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) dot += qd[i] * NcTileConv<KVT>::tof(raw[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
    const float ds = dot * scale;
    const float m_new = fmaxf(m, ds);
    const float resc = __expf(m - m_new);
    const float p = __expf(ds - m_new);
    l = l * resc + p;
    m = m_new;
    KVT vraw[DPL];
    const KVT* vp = s_v + off + d0;
    if (DPL * (int)sizeof(KVT) == 8) {
        *reinterpret_cast<uint2*>(vraw) = *reinterpret_cast<const uint2*>(vp);
    } else if (DPL * (int)sizeof(KVT) == 16) {
        *reinterpret_cast<uint4*>(vraw) = *reinterpret_cast<const uint4*>(vp);
    } else {
#pragma unroll
        for (int i = 0; i < DPL; ++i) vraw[i] = vp[i];
    }
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = acc[i] * resc + p * NcTileConv<KVT>::tof(vraw[i]);
}

template <typename T, typename KVT, int DPL>
__device__ __forceinline__ void nc_attn_tiled(
    const T* __restrict__ q, const T* __restrict__ k_self, const T* __restrict__ v_self,
    const KVT* __restrict__ kc, const KVT* __restrict__ vc,
    const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim, int page, int x, int ring_cap,
    T* __restrict__ out) {
    const int t = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = (int)blockDim.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarp = nthreads >> 5;
    const int group = q_heads / kv_heads;
    // 结构门(驱动件同判;越界 = 硬错,不静默、不半跑)
    if (group <= 0 || nwarp < group || (nwarp % group) != 0) { __trap(); }
    if (page <= 0 || page > OWL_NC_TILE_MAX_PAGE) { __trap(); }
    if (ring_cap != 0 && ring_cap < OWL_NC_SLIDING_WINDOW + (int)gridDim.x) { __trap(); }
    const int nsub = nwarp / group;
    const int wq = warp % group;
    const int sub = warp / group;
    const int h = kvh * group + wq;
    const int d0 = lane * DPL;
    const int kv_len = (int)kv_len_p[0];
    const int t_rows = (int)gridDim.x;   // 自块行数(= propose 的 8;gridDim.x)
    const int prefix_len = kv_len - t_rows;
    const float scale = rsqrtf((float)d_dim);

    // ---- shared:[2][page][d_dim] tile ‖ 归并区(nwarp·d_dim + 2·nwarp)----
    extern __shared__ __align__(16) char s_raw[];
    KVT* s_k = reinterpret_cast<KVT*>(s_raw);
    KVT* s_v = s_k + (long long)page * d_dim;
    float* s_merge = reinterpret_cast<float*>(s_v + (long long)page * d_dim);
    float* s_acc = s_merge;                       // nwarp × d_dim
    float* s_m = s_merge + (long long)nwarp * d_dim;
    float* s_l = s_m + nwarp;

    const T* qs = q + ((long long)t * q_heads + h) * d_dim;
    float qd[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) qd[i] = NcConv<T>::tof(qs[d0 + i]);

    float m = -3.0e38f;
    float l = 0.0f;
    float acc[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) acc[i] = 0.0f;

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    const int p0 = srow_begin / page;
    const int p1 = (prefix_len + page - 1) / page;   // 末页可含越界行(池内安全)
    const int jlo = sub * page / nsub;
    const int jhi = (sub + 1) * page / nsub;
    for (int b = p0; b < p1; ++b) {
        __syncthreads();
        nc_tile_load<KVT>(kc, vc, s_k, s_v, b, kvh, kv_heads, d_dim, page, x, ring_cap, tid,
                          nthreads);
        __syncthreads();
        for (int j = jlo; j < jhi; ++j) {
            const int srow = b * page + j;
            if (srow < srow_begin || srow >= prefix_len) continue;
            nc_tile_key<KVT, DPL>(s_k, s_v, (long long)j * d_dim, d0, qd, scale, m, l, acc);
        }
    }

    // ---- 自块 T 行(直读 k_self/v_self;同一 tile 循环,位置 0..T-1)----
    {
        __syncthreads();
        for (int e = tid; e < t_rows * d_dim; e += nthreads) {
            const int st = e / d_dim;
            const int d = e - st * d_dim;
            s_k[e] = k_self[((long long)st * kv_heads + kvh) * d_dim + d];
            s_v[e] = v_self[((long long)st * kv_heads + kvh) * d_dim + d];
        }
        __syncthreads();
        const int slo = sub * t_rows / nsub;
        const int shi = (sub + 1) * t_rows / nsub;
        for (int j = slo; j < shi; ++j) {
            nc_tile_key<KVT, DPL>(s_k, s_v, (long long)j * d_dim, d0, qd, scale, m, l, acc);
        }
    }

    // ---- 收尾:跨条带归并(v2 同式;1 次 __syncthreads)----
    __syncthreads();
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
#pragma unroll
    for (int i = 0; i < DPL; ++i) s_acc[(long long)warp * d_dim + d0 + i] = acc[i];
    __syncthreads();
    float mg = -3.0e38f;
    for (int s2 = 0; s2 < nsub; ++s2) mg = fmaxf(mg, s_m[wq + s2 * group]);
    float lg = 0.0f;
    float ag[DPL];
#pragma unroll
    for (int i = 0; i < DPL; ++i) ag[i] = 0.0f;
    for (int s2 = 0; s2 < nsub; ++s2) {
        const int w = wq + s2 * group;
        const float e = __expf(s_m[w] - mg);
        lg += s_l[w] * e;
#pragma unroll
        for (int i = 0; i < DPL; ++i) ag[i] += s_acc[(long long)w * d_dim + d0 + i] * e;
    }
    // 落盘只由条带 0 做(同组各 warp 的归并结果逐位相同 ⇒ 冗余写无必要)
    if (sub == 0) {
        T* o = out + ((long long)t * q_heads + h) * d_dim;
#pragma unroll
        for (int i = 0; i < DPL; ++i) o[d0 + i] = NcConv<T>::fromf(ag[i] / lg);
    }
}

// ---- 四个入口(与 v1/v2 同签名:PARAMS 一字不动)----
#define OWL_NC_TILED_ENTRY(NAME, T, KVT)                                                       \
    extern "C" __global__ void NAME(                                                           \
        const T* __restrict__ q, const T* __restrict__ k_self, const T* __restrict__ v_self,    \
        const KVT* __restrict__ kc, const KVT* __restrict__ vc,                                \
        const float* __restrict__ kv_len_p, int q_heads, int kv_heads, int d_dim, int page,     \
        int x, int ring_cap, T* __restrict__ out) {                                            \
        const int dpl = d_dim >> 5;                                                            \
        if (dpl == 4) {                                                                        \
            nc_attn_tiled<T, KVT, 4>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,    \
                                     d_dim, page, x, ring_cap, out);                            \
        } else if (dpl == 8) {                                                                 \
            nc_attn_tiled<T, KVT, 8>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,    \
                                     d_dim, page, x, ring_cap, out);                            \
        } else {                                                                               \
            __trap();                                                                          \
        }                                                                                      \
    }

OWL_NC_TILED_ENTRY(owl_naive_attn_nc_tiled_bf16, __nv_bfloat16, __nv_bfloat16)
OWL_NC_TILED_ENTRY(owl_naive_attn_nc_tiled_f16, __half, __half)
OWL_NC_TILED_ENTRY(owl_naive_attn_nc_tiled_fp8kv_bf16, __nv_bfloat16, unsigned char)
OWL_NC_TILED_ENTRY(owl_naive_attn_nc_tiled_fp8kv_f16, __half, unsigned char)
