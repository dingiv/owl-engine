// ============================================================================
// owl_argmax_f32idx_f16 —— 设备侧贪心采样(E3;REQ-DEC-04 每步零大 D2H)
//
// 语义:x[offset .. offset+n) 的 argmax 索引,以 f32 数值写 out[0]
// (owl 契约 5:索引/位置量 f32 过线;词表 id < 2²⁴ 内 f32 精确)。
// 平局取**最小索引**(与 host argmax 的 fold 语义一致,保证路径等价)。
//
// 结构:单块两段规约 —— 线程内步进扫描保 (val, idx) → shared 树形跨线程
// 归约(不依赖 shfl:nvrtc 对 sync 蝶形展开有未验证风险,§五风险清单在案,
// 实测确有错值)。
//
// C2 性能修(2026-10-03):单线程标量 2B 步进对 248320 词表 = ~970 次
// 依赖装载,单 SM 延迟受限(实测 315µs/发,应 ~10µs)。改 uint4 向量化
// (线程 OwN=8 连续 half,16B 对齐 —— 词表 V%8==0 ⇒ 行基址恒对齐;尾
// n%8 标量兜底)+ #pragma unroll 4 路独立装载 ILP。平局取小索引:线程内
// 按全局索引升序扫(chunk 序 = tid 起步 +stride),chunk 内 e 升序,严格 >
// 保留首见最大;跨线程树形同律。发射契约零变化:grid (1,1,1) × block
// (256,1,1),动态 smem 0,registry args = "T,i32,i32,T"。
// ============================================================================
#include <cuda_fp16.h>

// nvrtc 极简头集需自备 typedef(同 prefill_split_f16.cu 契约)
#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

constexpr int ARGMAX_BLOCK = 256;
constexpr int ARGMAX_OWN = 8;   // 每线程每 chunk 连续 half 数(uint4)

extern "C" __global__ void owl_argmax_f32idx_f16(
    const __half* __restrict__ x,   // [n](logits 末行窄视图或整行)
    int n,
    int offset,                     // 起始元素(末行 = (T-1)·V)
    float* __restrict__ out) {      // [1] = argmax 索引(f32 数值)
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const __half* row = x + offset;

    float best = -3.4e38f;
    int bi = -1;

    // 对齐前奏:offset%8 != 0 时先标量扫到 16B 边界(词表 V%8==0 时
    // offset 恒对齐,此支为空;测试任意 offset 兑底)
    const int pre = (int)(((8 - ((size_t)(row - x) & 7)) & 7));
    const int pre_n = (n < pre) ? n : pre;
    for (int i = tid; i < pre_n; i += stride) {
        const float v = __half2float(row[i]);
        if (v > best) { best = v; bi = i; }
    }

    // 向量体:从 pre 起的 n8 个 8-half chunk,chunk c = 全局 [pre + c*8, …+8)。
    // 线程扫描序 c = tid, tid+stride, …(各线程内部全局升序 ✓)。
    const __half* vrow = row + pre;
    const int n8 = (n - pre_n) / ARGMAX_OWN;
    const int tail0 = pre + n8 * ARGMAX_OWN;
    for (int c0 = tid; c0 < n8; c0 += stride * 4) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int c = c0 + u * stride;
            if (c < n8) {
                const uint4 raw = *reinterpret_cast<const uint4*>(vrow + c * ARGMAX_OWN);
                const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
                const int base = pre + c * ARGMAX_OWN;
                #pragma unroll
                for (int e = 0; e < ARGMAX_OWN; ++e) {
                    const float v = __half2float(
                        *reinterpret_cast<const __half*>(&h[e]));
                    if (v > best) { best = v; bi = base + e; }
                }
            }
        }
    }
    // 标量尾
    for (int i = tail0 + tid; i < n; i += stride) {
        const float v = __half2float(row[i]);
        if (v > best) { best = v; bi = i; }
    }

    // 归约 = shared 树形;平局取小索引贯穿全程。
    __shared__ float s_val[ARGMAX_BLOCK];
    __shared__ int s_idx[ARGMAX_BLOCK];
    s_val[tid] = best;
    s_idx[tid] = bi;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float ov = s_val[tid + off];
            int oi = s_idx[tid + off];
            if (ov > s_val[tid] || (ov == s_val[tid] && oi < s_idx[tid])) {
                s_val[tid] = ov;
                s_idx[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = (float)s_idx[0];
    }
}
