// ============================================================================
// owl GGUF 族:prefill **smem 复用 GEMM**(W-2b;设计 §十七)
//
// # 为什么再有这一臂(2026-10-19 nsys 定谳)
//
// W-2a 的复用臂已把权重流量 ÷16,但 nsys 显示 prefill **93% 时间仍在 GEMM**
// 且与 FLOPs 成比例(4B:Q6_K 45.6% / Q4_K 40.5% / Q8_0 4.8%)⇒ 有效算力只有
// **≈3.7-4.4 TFLOPS**。根因:`vec_dot` 对**每个 token** 重新解量化同一块
// (解量化算术 ∝ m × 参数量)。
//
// # 本核做的事
//
// 把"每 token 解量化"换成"**每窗解量化一次**":
//
//   ① 权重 `[NT, KC]` 块 → **smem 里的 f16 行**(每 warp 一行,协作式解量化);
//   ② 激活 `[MT, KC]` 直接拷 f16(不再经 q8_1 量化 —— 省一发核,且更准);
//   ③ 寄存器 4×4 分块的点积(smem 读 : FMA = 1 : 2,避免 smem 带宽墙)。
//
// # 几何(与 `family::gguf::smem_geom` 是同一契约)
//
//   grid  = (ceil(n / NT), ceil(m / MT));block = (32, 8) = 256 线程
//   smem  = 2 × MT|NT × (KC + 2) × 2B(MT = NT = 64,KC = 256 ⇒ 66 048 B,opt-in)
//   每线程 = 4 token × 4 行 = 16 个 f32 累加器
//
// # 数值口径(与 MMVQ **不再位级相同**,这是有意的)
//
// MMVQ 的激活先量化成 q8_1(±0.4% 相对误差);本核用原始 f16 激活 ⇒ 本核
// **更准**。门:`kernel_gate::gpu_gguf_smem_gemm_matches_mmvq`(allclose 1e-2,
// 参考 = CPU f64;两臂差异应落在 q8_1 量化噪声量级)。
//
// # 纪律(与 `gguf_gemm.cu` / `gguf_tile.cu` 同律)
//
// H1 零分配(f16 激活/输出由调用方给)· H2 零运行期设备查询 · H3 逐型入口。
// ============================================================================
#include "gguf.cuh"

namespace owl_gguf_smem {

// ---- 几何常量(host 侧 `family::gguf` 的镜像;两边由单测互锁)----
constexpr int MT = 64;    // 每 CTA 的 token 窗
constexpr int NT = 64;    // 每 CTA 的权重行窗
constexpr int KC = 256;   // 每次 k 块(= 一个 K-quant 超块;Q8_0 = 8 块)
constexpr int PAD = 2;    // smem 行 padding(halfs):stride = 258 半字 = 129 字 ⇒ 2-way 内
constexpr int STRIDE = KC + PAD;
constexpr int NW = 16;    // warp 数(512 线程 ⇒ 1 CTA/SM 也有 16 warp 在飞)
constexpr int THREADS = 32 * NW;
constexpr int NR = 2;     // 每线程负责的**行**数(4 token × NR 行 = 16 个累加器以内)
constexpr int GR = NT / NR;               // 行组数
constexpr int GT = MT / 4;                // token 组数
static_assert(GR * GT == THREADS, "线程映射必须恰好铺满 [MT, NT] 窗");

/// **Q8_0 行解量化**(快照里没有 `dequantize_block_q8_0` —— 数学 = `d · int8`;
/// 由数值门兜底)
__device__ __forceinline__ void row_q8_0(const void * __restrict__ vx,
                                         __half * __restrict__ yy, int tid) {
    const block_q8_0 * b = reinterpret_cast<const block_q8_0 *>(vx);
    const float d = __half2float(b->d);
    for (int i = tid; i < QK8_0; i += 32) {
        yy[i] = __float2half(d * (float) b->qs[i]);
    }
}

/// **Q6_K 行解量化**(转写自 `gguf.cuh::dequantize_block_q6_K`;唯一改动 =
/// `tid` 由参数给 —— 原实现假定 **64 线程**,此处由同一 warp 用
/// `tid = lane` 与 `tid = lane + 32` 两趟覆盖。数学逐式未动)
__device__ __forceinline__ void row_q6_K(const void * __restrict__ vx,
                                         __half * __restrict__ yy, int tid) {
    const block_q6_K * x = reinterpret_cast<const block_q6_K *>(vx);
    const int ip = tid / 32;      // 0 或 1
    const int il = tid - 32 * ip; // 0..31
    const int is = 8 * ip + il / 16;
    __half * y = yy + 128 * ip + il;
    const half d = x->d;
    const uint8_t * ql = x->ql + 64 * ip + il;
    const uint8_t qh = x->qh[32 * ip + il];
    const int8_t * sc = x->scales + is;
    y[0] = __hmul(d, __int2half_rn(sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32)));
    y[32] = __hmul(d, __int2half_rn(sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32)));
    y[64] = __hmul(d, __int2half_rn(sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32)));
    y[96] = __hmul(d, __int2half_rn(sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32)));
}

/// **Q4_K / Q5_K**:快照函数按 `threadIdx.x` 分工(需 32 线程)。本核 block =
/// `(32, NW)` ⇒ **`threadIdx.x` ≡ lane** ⇒ 任意单个 warp 调用语义正确
/// (这是 W-2b 立项时核过的第一条事实,别再重写一遍数学)。
__device__ __forceinline__ void row_q4_K(const void * __restrict__ vx,
                                         __half * __restrict__ yy, int) {
    dequantize_block_q4_K<__half>(vx, yy);
}

/// **Q5_K 行解量化**(转写自 `gguf.cuh::dequantize_block_q5_K`;原实现同样假定
/// **64 线程**(`il = tid/16` 需 0..3)⇒ 按 Q6_K 同法用 `tid = lane, lane+32`
/// 两趟覆盖,数学逐式未动)
__device__ __forceinline__ void row_q5_K(const void * __restrict__ vx,
                                         __half * __restrict__ yy, int tid) {
    const block_q5_K * x = reinterpret_cast<const block_q5_K *>(vx);
    const int il = tid / 16;    // 0..3
    const int ir = tid % 16;    // 0..15
    const int is = 2 * il;      // 0..6
    __half * y = yy + 64 * il + 2 * ir;
    const half dall = __low2half(x->dm);
    const half dmin = __high2half(x->dm);
    const uint8_t * ql = x->qs + 32 * il + 2 * ir;
    const uint8_t * qh = x->qh + 2 * ir;
    uint8_t sc, m;
    get_scale_min_k4(is + 0, x->scales, sc, m);
    const half d1 = __hmul(dall, __int2half_rn(sc));
    const half m1 = __hmul(dmin, __int2half_rn(m));
    get_scale_min_k4(is + 1, x->scales, sc, m);
    const half d2 = __hmul(dall, __int2half_rn(sc));
    const half m2 = __hmul(dmin, __int2half_rn(m));
    uint8_t hm = 1 << (2 * il);
    y[0] = __hsub(__hmul(d1, __int2half_rn((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0))), m1);
    y[1] = __hsub(__hmul(d1, __int2half_rn((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0))), m1);
    hm <<= 1;
    y[32] = __hsub(__hmul(d2, __int2half_rn((ql[0] >> 4) + (qh[0] & hm ? 16 : 0))), m2);
    y[33] = __hsub(__hmul(d2, __int2half_rn((ql[1] >> 4) + (qh[1] & hm ? 16 : 0))), m2);
}

/// 逐型行解量化(静态分发;`two_pass = true` 的型需要两趟 `tid = lane, lane+32`)
template <typename block_q_t, int qk, void (*ROW)(const void *, __half *, int)>
struct RowDeq {
    static constexpr bool two_pass = false;
    static __device__ __forceinline__ void run(const void * blk, __half * dst, int lane) {
        ROW(blk, dst, lane);
    }
};
template <>
struct RowDeq<block_q6_K, QK_K, row_q6_K> {
    static constexpr bool two_pass = true;
    static __device__ __forceinline__ void run(const void * blk, __half * dst, int lane) {
        row_q6_K(blk, dst, lane);
        row_q6_K(blk, dst, lane + 32);
    }
};
// ⚠️ **Q5_K 也是 64 线程型**(`il = tid/16`):必须两趟,否则只写半块(实测
// NaN —— 未写的一半是上一个 chunk 的残留)。Q4_K 是 32 线程型 ⇒ 单趟即满。
template <>
struct RowDeq<block_q5_K, QK_K, row_q5_K> {
    static constexpr bool two_pass = true;
    static __device__ __forceinline__ void run(const void * blk, __half * dst, int lane) {
        row_q5_K(blk, dst, lane);
        row_q5_K(blk, dst, lane + 32);
    }
};

/// 共享体:一个 CTA 出 `[MT, NT]` 一块
template <typename block_q_t, int qk, void (*ROW)(const void *, __half *, int)>
__device__ __forceinline__ void body(
    const void * __restrict__ weights, const __half * __restrict__ x,
    __half * __restrict__ out, int m, int n, int k, int out_off, int out_ld) {
    extern __shared__ __half smem[];
    __half * sx = smem;                       // [MT, STRIDE]
    __half * sw = smem + MT * STRIDE;         // [NT, STRIDE]

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * MT;
    const int n0 = blockIdx.x * NT;
    const int nb_per_row = k / qk;            // 每行块数

    // 分块:token 基 = (tid/GR)*4,行 基 = (tid%GR)*NR
    const int tm = (tid / GR) * 4;
    const int rn = (tid % GR) * NR;
    float acc[NR][4];
#pragma unroll
    for (int i = 0; i < NR; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < k; k0 += KC) {
        const int kc = min(KC, k - k0);
        // ---- ① 激活窗:直接拷 f16(合并访存;越界 token 填 0)----
        for (int idx = tid; idx < MT * kc; idx += THREADS) {
            const int t = idx / kc;
            const int c = idx - t * kc;
            const int gm = m0 + t;
            sx[t * STRIDE + c] = (gm < m) ? x[(size_t) gm * k + k0 + c] : __float2half(0.0f);
        }
        // ---- ② 权重窗:每 warp 解一行,`warp` 轮转覆盖 NT 行 ----
        for (int r = warp; r < NT; r += NW) {
            const int gr = n0 + r;
            __half * dst = sw + r * STRIDE;
            if (gr >= n) {
                for (int c = lane; c < kc; c += 32) {
                    dst[c] = __float2half(0.0f);
                }
                continue;
            }
            // ⚠️ 必须带 `k0 / qk` 偏移:窗只解 k ∈ [k0, k0+kc),块索引 = k0/qk 起
            const block_q_t * rowp = reinterpret_cast<const block_q_t *>(weights)
                                     + (size_t) gr * nb_per_row + (size_t) (k0 / qk);
            for (int b = 0; b < kc / qk; ++b) {
                RowDeq<block_q_t, qk, ROW>::run(reinterpret_cast<const void *>(rowp + b),
                                                dst + b * qk, lane);
            }
        }
        __syncthreads();
        // ---- ③ 4×4 寄存器分块点积 ----
        for (int c = 0; c < kc; c += 2) {
            const __half2 xa = *reinterpret_cast<const __half2 *>(&sx[(tm + 0) * STRIDE + c]);
            const __half2 xb = *reinterpret_cast<const __half2 *>(&sx[(tm + 1) * STRIDE + c]);
            const __half2 xc = *reinterpret_cast<const __half2 *>(&sx[(tm + 2) * STRIDE + c]);
            const __half2 xd = *reinterpret_cast<const __half2 *>(&sx[(tm + 3) * STRIDE + c]);
            const float2 fx[4] = {__half22float2(xa), __half22float2(xb),
                                  __half22float2(xc), __half22float2(xd)};
#pragma unroll
            for (int i = 0; i < NR; ++i) {
                const __half2 wv = *reinterpret_cast<const __half2 *>(&sw[(rn + i) * STRIDE + c]);
                const float2 fw = __half22float2(wv);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += fx[j].x * fw.x + fx[j].y * fw.y;
                }
            }
        }
        __syncthreads();
    }

    // ---- ④ 写回(带边界)----
#pragma unroll
    for (int i = 0; i < NR; ++i) {
        const int gr = rn + i;
        if (n0 + gr >= n) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int gm = m0 + tm + j;
            if (gm < m) {
                out[(size_t) gm * out_ld + out_off + n0 + gr] = __float2half(acc[i][j]);
            }
        }
    }
}

} // namespace owl_gguf_smem

// ---- M1 实例集(四型)----
extern "C" __global__ void owl_gguf_gemm_smem_q8_0(
    const void * __restrict__ weights, const __half * __restrict__ x,
    __half * __restrict__ out, int m, int n, int k, int out_off, int out_ld) {
    owl_gguf_smem::body<block_q8_0, QK8_0, owl_gguf_smem::row_q8_0>(
        weights, x, out, m, n, k, out_off, out_ld);
}

extern "C" __global__ void owl_gguf_gemm_smem_q4k(
    const void * __restrict__ weights, const __half * __restrict__ x,
    __half * __restrict__ out, int m, int n, int k, int out_off, int out_ld) {
    owl_gguf_smem::body<block_q4_K, QK_K, owl_gguf_smem::row_q4_K>(
        weights, x, out, m, n, k, out_off, out_ld);
}

extern "C" __global__ void owl_gguf_gemm_smem_q5k(
    const void * __restrict__ weights, const __half * __restrict__ x,
    __half * __restrict__ out, int m, int n, int k, int out_off, int out_ld) {
    owl_gguf_smem::body<block_q5_K, QK_K, owl_gguf_smem::row_q5_K>(
        weights, x, out, m, n, k, out_off, out_ld);
}

extern "C" __global__ void owl_gguf_gemm_smem_q6k(
    const void * __restrict__ weights, const __half * __restrict__ x,
    __half * __restrict__ out, int m, int n, int k, int out_off, int out_ld) {
    owl_gguf_smem::body<block_q6_K, QK_K, owl_gguf_smem::row_q6_K>(
        weights, x, out, m, n, k, out_off, out_ld);
}
