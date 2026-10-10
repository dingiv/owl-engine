// ============================================================================
// owl prefill split attention(flash-decoding 式 context 分块;C1 自研序,
// 2026-10-02 立案:旧 chunked_prefill_paged_attn_opt(标量,vendor port)
// 33.9ms/发 ~6 TFLOPS —— 病灶 = 1 thread/query 全 context 串行扫,
// 总线程 = T×Hq ≈ 24.5k,每 SM ~9 warp(需 48+),延迟 hiding 缺失;
// 且 THREAD_GROUP_SIZE=1 → acc[256] 寄存器必溢出。
//
// 结构(flash-decoding,同 vLLM v2 decode 的 partition 思想搬到 prefill):
//   K1 owl_prefill_split_f16 —— grid (Hq/Hkv, Hkv, qchunks × nparts);
//      每 block = 64 query token(THREAD_GROUP_SIZE=4,每线程 64 dims,
//      acc[64] 寄存器无溢出)× 一个 context partition;在线 softmax 出
//      **未归一化** partial(acc)+ (m, l) 分区统计,写 scratch;
//      空/越界 partition 写中性值(m=-INF, l=0, tmp=0)。
//   K2 owl_prefill_split_reduce_f16 —— 每 thread 一 (token, head):
//      w_p = exp(m_p − M);out = Σ tmp_p·w_p / Σ l_p·w_p。
//
// 语义锚 = chunked_prefill_paged_attn_opt_f16(单 partition 时逐位等价域,
// 由隔离矩阵/真权重 prefill 对照锚定)。简化面(owl prefill 实际不用):
// 无 alibi/sinks/sliding_window/softscapping/跨序列 boundary(单序列)。
// causal = 每 token attend [0, abs_pos](abs = ctx_base + local)。
//
// 布局契约(同 pagedattention_f16.cu classic):q/out [T, Hq, hd];
// kc [nb, Hkv, hd/x, page, x](x=8,page=32);vc [nb, Hkv, hd, page];
// block_tables f32 [nb];scratch:tmp_out f16 [T, Hq, nparts, hd],
// stat f32 [T, Hq, nparts, 2](m 前 l 后)。
// grid/block/smem 契约(发射方 driver 单源):
//   K1 grid (Hq/Hkv, Hkv, qchunks×nparts) block 256
//      smem = 64 tok × hd × 2B × 2(K+V)= 64KB(>48KB 走 opt-in 通道)
//   K2 grid ceil(T×Hq/256) block 256 smem 0
//
// ⚠️ 结案(2026-10-02,W3):真凶 = 尾部 store 的隐式转换 ——
// `o[u16] = __float2half(acc)` 经 operator float() 向 uint16_t 截断(值 ≥1
// 全变 1、负数全变 0;位型指纹 scr_out 0x0001 @ acc≈1.14/1.33/1.52 处),
// scr_stat(f32)不受累 → 「核数学对、store 写零」假象。修复 = vLLM
// from_float 位精确 store; Marshalling/liveness/核数学全部无罪。
// **翻默认前置件已就绪,门 OWL_PREFILL_SPLIT 待引擎实测后裁决。**
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// nvrtc 极简头集需自备 typedef;离线 nvcc 检查走 <cstdint>(带 RTC 守卫)
#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#else
#include <cstdint>
#endif
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu
#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

// ---- K1:split 主核 -------------------------------------------------------
// 模板体(2026-10-12 fp8 直读变体):KV_FP8=true 时 K/V 从 e4m3 字节池
// 读入,smem 前转 f16(chunked 核 B6.3 同式 —— smem 宽度不变,QK/PV
// 数学零改动);smem 恒 f16。classic 寻址 index 数学同式(fp8 = 1B 单位)。
// q [T, Hq, hd];scr_out f16 [T, Hq, nparts, hd];scr_stat f32 [T, Hq, nparts, 2]
// dummy 尾参 = 契约 4(真输出 = reduce 核;本核纯副作用写 scratch)
template <bool KV_FP8, bool KNHD = false>
__device__ void split_k1_body(
    const uint16_t* __restrict__ q,            // [T, Hq, hd]
    const uint16_t* __restrict__ k_cache,      // classic(f16 视角;fp8 = 字节池 reinterpret)
    const uint16_t* __restrict__ v_cache,      // classic
    const float* __restrict__ block_tables,    // [nb](单序列恒等/块链)
    uint16_t* __restrict__ scr_out,            // [T, Hq, nparts, hd]
    float* __restrict__ scr_stat,              // [T, Hq, nparts, 2](m,l)
    const uint16_t* __restrict__ k0_alibi,     // 树序槽(K0 写池先于本核;不解引用)
    float scale,
    int32_t hkv, int32_t T, int32_t ctx_base, int32_t nparts,
    int32_t kv_block_stride, int32_t kv_head_stride, int32_t page,
    int32_t hq,
    uint16_t* __restrict__ dummy)              // 哑输出(契约 4:输出 = 末位)
{
    (void)dummy;
    (void)k0_alibi;
    constexpr int HD = 256;          // hd256 特化(hd128 另实例化)
    constexpr int TG = 4;            // thread group:4 线程合作 1 query
    constexpr int DIM_PER_TH = HD / TG;   // 64
    constexpr int VEC = 8;           // uint4 = 8 halves
    constexpr int VQ = DIM_PER_TH / VEC;  // 8 vec/thread
    constexpr int QUERIES_PER_BLOCK = 256 / TG;  // 64
    constexpr int TILE = QUERIES_PER_BLOCK;      // K tile = 64 tokens(与 query 数对齐)

    const int tid = threadIdx.x;
    const int qh_base = blockIdx.x;          // kv 组内 query 头序
    const int kv_head = blockIdx.y;
    const int qchunk = blockIdx.z / nparts;
    const int part = blockIdx.z % nparts;

    const int tok = qchunk * QUERIES_PER_BLOCK + tid / TG;   // query token 局部序
    const int g = tid % TG;                  // 组内 dim 段序

    const int head = kv_head * (hq / hkv) + qh_base;
    const int q_abs = ctx_base + tok;        // 绝对 pos;context = q_abs + 1
    const int ctx_end = q_abs + 1;
    const int max_ctx = ctx_base + T;
    const int P = (max_ctx + nparts - 1) / nparts;
    const int ps = part * P;
    const int pe = MIN((part + 1) * P, max_ctx);

    const bool active = (tok < T);

    // ---- 分区空处理(2026-10-12 探针 ⑤ 定谳的 UB 雷)----
    // 旧版:if (ps >= ctx_end || !active) return; —— ctx_end 依赖 tok,
    // 同 block 内部分线程早退、部分继续 → 后续 __syncthreads UB(违反
    // 本文件 2026-10-02 块统一律;np>1 且 token 跨分区边界时输出错,
    // f16/fp8 同病;生产 t0 恰为 64 倍数掩盖多年)。
    // 修法:①块统一早退仅保留 ps ≥ max_ctx(整个分区对所有 token 空);
    // ②per-token 中性(ps ≥ ctx_end)不再早退,走循环由 in_ctx 掩码
    //   自然产出中性值(M=-INF, L=0, acc=0),写 partial 用 active 门。
    if (ps >= max_ctx) {
        if (active) {
            const long long b = ((long long)tok * hq + head) * nparts + part;
            scr_stat[b * 2 + 0] = -INFINITY;
            scr_stat[b * 2 + 1] = 0.0f;
            uint16_t* o = scr_out + b * HD;
            for (int d = 0; d < HD; d += VEC) {
                *reinterpret_cast<uint4*>(o + d) = make_uint4(0, 0, 0, 0);
            }
        }
        return;
    }

    // ---- q 装载(本线程 64 dims;f16→f32 寄存器常驻)----
    const long long q_off = (long long)tok * hq * HD + (long long)head * HD
                          + g * DIM_PER_TH;
    float qv[VQ][VEC];
    #pragma unroll
    for (int v = 0; v < VQ; ++v) {
        const uint4 raw = *reinterpret_cast<const uint4*>(
            q + q_off + v * VEC);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
        #pragma unroll
        for (int e = 0; e < VEC; ++e) qv[v][e] = __half2float(
            *reinterpret_cast<const __half*>(&h[e]));
    }

    float acc[DIM_PER_TH];
    #pragma unroll
    for (int d = 0; d < DIM_PER_TH; ++d) acc[d] = 0.0f;
    float M = -INFINITY;
    float L = 0.0f;

    // ---- smem tile:64 K tokens × HD f16(K)+ 同 V ----
    extern __shared__ uint16_t smem[];
    uint16_t* k_smem = smem;                  // [TILE, HD]
    uint16_t* v_smem = k_smem + TILE * HD;    // [TILE, HD]

    // classic 寻址:K = [d/x][page][x] 转置;V = [d][page] 直排
    // kNHD(统一契约 P2):(phys·page + off)·hkv·hd + head·hd + d,
    // K/V 同式;host 传 kv_block_stride = page·hkv·hd、kv_head_stride = hd
    auto kc_index = [&](int kt, int d) -> long long {
        const int phys = (int)block_tables[kt / page];
        const int off = kt % page;
        if constexpr (KNHD) {
            return (long long)phys * kv_block_stride
                 + (long long)kv_head * kv_head_stride
                 + (long long)off * (kv_block_stride / page) + d;
        }
        return (long long)phys * kv_block_stride
             + (long long)kv_head * kv_head_stride
             + (long long)(d / VEC) * page * VEC + off * VEC + (d % VEC);
    };
    auto vc_index = [&](int kt, int d) -> long long {
        const int phys = (int)block_tables[kt / page];
        const int off = kt % page;
        if constexpr (KNHD) {
            return (long long)phys * kv_block_stride
                 + (long long)kv_head * kv_head_stride
                 + (long long)off * (kv_block_stride / page) + d;
        }
        return (long long)phys * kv_block_stride
             + (long long)kv_head * kv_head_stride
             + (long long)d * page + off;
    };

    // ⚠️ 循环上界必须块统一(ps..pe;每线程 causal 差 ≤64 由 in_ctx 掩码)——
    // 曾用每线程 ctx_hi = 发散 __syncthreads = UB/trap(2026-10-02)
    // 装载步长 = 恒 blockDim(2026-10-12:旧 nth=MIN(256, 活跃tok×TG) 是
    // 「tok≥T 早退」时代的配套;UB 修复后全线程走完全程,恒 256 全覆盖,
    // 尾块/小 T(verify T=8)装载并发不再随活跃 tok 数坍缩 —— 后者曾致
    // 20k ctx verify 592ms/轮(与 chunked 同烂,flash-decoding 形态失效))
    const int nth = 256;
    for (int t0 = ps; t0 < pe; t0 += TILE) {
        const int tlen = MIN(TILE, pe - t0);
        __syncthreads();   // 上一轮消费完毕
        // 协作装 tile:K uint4(转置布局 [d/x][page][x],同 vg 连续 kt 合并);
        // V 标量(直排 [d][page],warp 内连续 kt = 连续地址,合并 ✓)——
        // V 逐 token 的 8 连 dim 不构成 uint4(off 任意 → 2B 对齐,曾致
        // MISALIGNED_ADDRESS,2026-10-02);smem 内 V 已转 [token][dim]
        for (int e = tid; e < TILE * HD / VEC; e += nth) {
            const int kt = e % TILE;            // tile 内 token(连续)
            const int vg = e / TILE;            // vec 组(d/VEC)
            const int d = vg * VEC;
            const int gt = t0 + kt;
            if (gt < max_ctx) {
                if constexpr (KV_FP8) {
                    // e4m3 字节池(1B/elem):index 数学同 classic,单位 1B。
                    // K 的 x 段 8 连字节(d=8 倍数 → off·8 字节,8B 对齐)
                    // → uint2 一次读;V 标量逐字节。smem 恒 f16。
                    const uint8_t* k8 = reinterpret_cast<const uint8_t*>(k_cache);
                    const uint8_t* v8 = reinterpret_cast<const uint8_t*>(v_cache);
                    const uint2 kraw = *reinterpret_cast<const uint2*>(
                        k8 + kc_index(gt, d));
                    const __nv_fp8_e4m3* kb = reinterpret_cast<const __nv_fp8_e4m3*>(&kraw);
                    uint4 kh;
                    __half* kh16 = reinterpret_cast<__half*>(&kh);
                    #pragma unroll
                    for (int e2 = 0; e2 < VEC; ++e2)
                        kh16[e2] = __half(__nv_cvt_fp8_to_halfraw(kb[e2].__x, __NV_E4M3));
                    *reinterpret_cast<uint4*>(k_smem + kt * HD + d) = kh;
                    for (int vv = 0; vv < VEC; ++vv) {
                        __nv_fp8_e4m3 ve;
                        ve.__x = v8[vc_index(gt, d + vv)];
                        v_smem[kt * HD + d + vv] = __half_as_ushort(
                            __half(__nv_cvt_fp8_to_halfraw(ve.__x, __NV_E4M3)));
                    }
                } else {
                    *reinterpret_cast<uint4*>(k_smem + kt * HD + d) =
                        *reinterpret_cast<const uint4*>(k_cache + kc_index(gt, d));
                    for (int vv = 0; vv < VEC; ++vv) {
                        v_smem[kt * HD + d + vv] = v_cache[vc_index(gt, d + vv)];
                    }
                }
            } else {
                *reinterpret_cast<uint4*>(k_smem + kt * HD + d) = make_uint4(0, 0, 0, 0);
                for (int vv = 0; vv < VEC; ++vv) v_smem[kt * HD + d + vv] = 0;
            }
        }
        __syncthreads();
        // ---- 本线程的 query:64 dims(= 全头 256 dims 的 g 段)----
        // 注意:qk 需全头点积 → 组内 4 线程各持 64 dims 部分和,shfl 归约。
        for (int kt = 0; kt < tlen; ++kt) {
            const int gt = t0 + kt;
            const bool in_ctx = (gt <= q_abs);
            // QK 部分和(64 dims)
            float part_dot = 0.0f;
            if (in_ctx) {
                #pragma unroll
                for (int v = 0; v < VQ; ++v) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(
                        k_smem + kt * HD + g * DIM_PER_TH + v * VEC);
                    const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
                    #pragma unroll
                    for (int e = 0; e < VEC; ++e) {
                        part_dot += qv[v][e] * __half2float(
                            *reinterpret_cast<const __half*>(&h[e]));
                    }
                }
            }
            // TG 归约(shfl_xor 1,2)+ 乘 scale
            part_dot += __shfl_xor_sync(FULL_MASK, part_dot, 1);
            part_dot += __shfl_xor_sync(FULL_MASK, part_dot, 2);
            part_dot *= scale;

            // 组内广播(qk/m 对 4 lanes 同值 —— shfl_xor 全归约后天然一致)
            const float m_new = fmaxf(M, in_ctx ? part_dot : -INFINITY);
            const float alpha = (m_new == -INFINITY) ? 1.0f : __expf(M - m_new);
            const float p = in_ctx ? __expf(part_dot - m_new) : 0.0f;

            // PV(64 dims;p 组内一致;uint4 向量化)
            // ⚠️ 在线 softmax 必须同步重缩放 acc(alpha 只作用 L 是本立案
            // 第二雷:kt0 权重系统性虚高,偏差向量拟合 = δ×v[0] 逐维全中)
            #pragma unroll
            for (int v = 0; v < VQ; ++v) {
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    v_smem + kt * HD + g * DIM_PER_TH + v * VEC);
                const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
                #pragma unroll
                for (int e = 0; e < VEC; ++e) {
                    acc[v * VEC + e] = acc[v * VEC + e] * alpha + p * __half2float(
                        *reinterpret_cast<const __half*>(&h[e]));
                }
            }
            M = m_new;
            L = L * alpha + p;
        }
        __syncthreads();   // acc 消费完毕再覆写 tile(保守;load 前亦有屏障)
    }

    // ---- 写 partial(未归一化 acc + (M, L);active 门:尾块 tok≥T 不越界)----

    if (active) {
        const long long b = ((long long)tok * hq + head) * nparts + part;
        scr_stat[b * 2 + 0] = M;
        scr_stat[b * 2 + 1] = L;
// ⚠️ 定谳(2026-10-02 结案):初版 scr_out 全零的真凶不是 nvrtc —— 是本行
// 隐式转换:`o[u16] = __float2half(acc)` 走 __half::operator float() 再向
// uint16_t 截断(位型指纹 = 值 ≥1 的 acc 全变 1、负数全变 0)。修复 =
// vLLM from_float/float_to_half 同款位精确 store(reinterpret __half*,本
// 文件 reduce 核与 fused.cu 全族同式;fused 族 out 形参本就是 __half* 无罪)。
// 另撤除 scr_stat[16..32) 诊断转储(T>4 时污染真实 M/L)。
        uint16_t* o = scr_out + b * HD;
        #pragma unroll
        for (int d = 0; d < DIM_PER_TH; ++d) {
            reinterpret_cast<__half*>(o)[g * DIM_PER_TH + d] = __float2half(acc[d]);
        }
    }
}

// ---- K1 双入口(f16 / fp8kv;签名逐字同,fp8 的 K/V 字节池载体)----------
extern "C" __global__ void owl_prefill_split_f16_hd256(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    uint16_t* __restrict__ scr_out,
    float* __restrict__ scr_stat,
    const uint16_t* __restrict__ k0_alibi,
    float scale,
    int32_t hkv, int32_t T, int32_t ctx_base, int32_t nparts,
    int32_t kv_block_stride, int32_t kv_head_stride, int32_t page,
    int32_t hq,
    uint16_t* __restrict__ dummy)
{
    split_k1_body<false>(q, k_cache, v_cache, block_tables, scr_out, scr_stat,
                         k0_alibi, scale, hkv, T, ctx_base, nparts,
                         kv_block_stride, kv_head_stride, page, hq, dummy);
}

extern "C" __global__ void owl_prefill_split_fp8kv_hd256(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,      // e4m3 字节池(1B/elem)
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    uint16_t* __restrict__ scr_out,
    float* __restrict__ scr_stat,
    const uint16_t* __restrict__ k0_alibi,
    float scale,
    int32_t hkv, int32_t T, int32_t ctx_base, int32_t nparts,
    int32_t kv_block_stride, int32_t kv_head_stride, int32_t page,
    int32_t hq,
    uint16_t* __restrict__ dummy)
{
    split_k1_body<true>(q, k_cache, v_cache, block_tables, scr_out, scr_stat,
                        k0_alibi, scale, hkv, T, ctx_base, nparts,
                        kv_block_stride, kv_head_stride, page, hq, dummy);
}

// ---- kNHD 变体(kv布局统一契约 P2):页内 [page,hkv,hd] 连续,K/V 同址;
// host 传 kv_block_stride = page·hkv·hd、kv_head_stride = hd ----
extern "C" __global__ void owl_prefill_split_f16_knhd_hd256(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    uint16_t* __restrict__ scr_out,
    float* __restrict__ scr_stat,
    const uint16_t* __restrict__ k0_alibi,
    float scale,
    int32_t hkv, int32_t T, int32_t ctx_base, int32_t nparts,
    int32_t kv_block_stride, int32_t kv_head_stride, int32_t page,
    int32_t hq,
    uint16_t* __restrict__ dummy)
{
    split_k1_body<false, true>(q, k_cache, v_cache, block_tables, scr_out, scr_stat,
                               k0_alibi, scale, hkv, T, ctx_base, nparts,
                               kv_block_stride, kv_head_stride, page, hq, dummy);
}

extern "C" __global__ void owl_prefill_split_fp8kv_knhd_hd256(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,      // e4m3 字节池(1B/elem)
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    uint16_t* __restrict__ scr_out,
    float* __restrict__ scr_stat,
    const uint16_t* __restrict__ k0_alibi,
    float scale,
    int32_t hkv, int32_t T, int32_t ctx_base, int32_t nparts,
    int32_t kv_block_stride, int32_t kv_head_stride, int32_t page,
    int32_t hq,
    uint16_t* __restrict__ dummy)
{
    split_k1_body<true, true>(q, k_cache, v_cache, block_tables, scr_out, scr_stat,
                              k0_alibi, scale, hkv, T, ctx_base, nparts,
                              kv_block_stride, kv_head_stride, page, hq, dummy);
}

// ---- K2:partition 归一化合并 --------------------------------------------
// 每 thread 一 (token, head):w_p = exp(m_p − M);out = Σ tmp_p·w_p / Ltot。
extern "C" __global__ void owl_prefill_split_reduce_f16_hd256(
    const uint16_t* __restrict__ split_alibi, // 树序槽(split 先于本核;不解引用)
    const uint16_t* __restrict__ scr_out, // [T, Hq, nparts, hd]
    const float* __restrict__ scr_stat,   // [T, Hq, nparts, 2](m,l)
    int32_t nparts, int32_t hq, int32_t hd, int32_t T,
    uint16_t* __restrict__ out)           // [T, Hq, hd](契约 4:输出 = 末位)
{
    (void)split_alibi;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= T * hq) return;
    const int tok = id / hq;
    const int head = id % hq;
    const long long b = ((long long)tok * hq + head) * nparts;

    float M = -INFINITY;
    for (int p = 0; p < nparts; ++p) M = fmaxf(M, scr_stat[(b + p) * 2 + 0]);

    float Ltot = 0.0f;
    float w[64];
    for (int p = 0; p < nparts && p < 64; ++p) {
        const float mp = scr_stat[(b + p) * 2 + 0];
        const float lp = scr_stat[(b + p) * 2 + 1];
        const float wp = (mp == -INFINITY) ? 0.0f : __expf(mp - M);
        w[p] = wp;
        Ltot += lp * wp;
    }
    const float inv = 1.0f / (Ltot + 1e-6f);

    uint16_t* o = out + (long long)id * hd;
    const uint16_t* s = scr_out + b * hd;
    for (int d = 0; d < hd; ++d) {
        float acc = 0.0f;
        for (int p = 0; p < nparts && p < 64; ++p) {
            acc += __half2float(*reinterpret_cast<const __half*>(
                s + (long long)p * hd + d)) * w[p];
        }
        *reinterpret_cast<__half*>(o + d) = __float2half(acc * inv);
    }
}
