// ============================================================================
// owl_argmax_f32idx_f16 —— 设备侧贪心采样(E3;REQ-DEC-04 每步零大 D2H)
//
// 语义:x[offset .. offset+n) 的 argmax 索引,以 f32 数值写 out[0]
// (owl 契约 5:索引/位置量 f32 过线;词表 id < 2²⁴ 内 f32 精确)。
// 平局取**最小索引**(与 host argmax 的 fold 语义一致,保证路径等价)。
//
// 结构:单块两段规约 —— 线程内步进扫描保 (val, idx) → warp shuffle
// 下三角归约 → shared 跨 warp 归约。词表 ~152k / 256 线程 ≈ 594 元素/
// 线程,显存带宽为主(~300KB f16 读)。
//
// 发射契约:grid (1,1,1) × block (256,1,1),动态 smem 0(静态
// s_val/s_idx 在核内)。注册表 args = "T,i32,i32,T"(输出固定末参)。
// ============================================================================
#include <cuda_fp16.h>

extern "C" __global__ void owl_argmax_f32idx_f16(
    const __half* __restrict__ x,   // [n](logits 末行窄视图或整行)
    int n,
    int offset,                     // 起始元素(末行 = (T-1)·V)
    float* __restrict__ out) {      // [1] = argmax 索引(f32 数值)
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float best = -3.4e38f;
    int bi = -1;
    for (int i = tid; i < n; i += stride) {
        float v = __half2float(x[offset + i]);
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    }

    // 归约 = shared 树形(不依赖 shfl:nvrtc 对 sync 蝶形展开有未验证
    // 风险,attention-kernel-port.md §五风险清单在案;实测确有错值)。
    // 平局取小索引贯穿全程。
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
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
