// ============================================================================
// owl GGUF 族:dequant-in-kernel GEMM 的**权重复用臂**(prefill;设计 §十六 W-2)
//
// # 为什么需要它(2026-10-19 实测定式)
//
// `owl_gguf_gemm_*`(MMVQ)的网格是 `(n / n_warps, m)` —— **每个 (权重行, token)
// 对**独立读一遍该权重行 ⇒ 一次前向的权重流量 = `m × 权重字节数`。
// 两个靶位实测完美符合这条公式:
//
//   4B  : 936 GB/s ÷ 2.81 GB = 333 t/s(实测 prefill 327.7)
//   27B : 936 GB/s ÷ 17.7 GB =  53 t/s(实测 ~50)
//
// 即 **decode 是对的(m = 1 时本来就该读一遍),prefill 是纯带宽浪费**:
// m = 1024 时同一行权重被读了 1024 次。
//
// # 本核做的事(一句话)
//
// 同一 warp 一次拿住**一行权重**,连续算 `MT` 个 token 的点积 ⇒ 权重流量 ÷ MT。
// `vec_dot` 的归约序**逐位不变**(每 token 仍是同一串 k 块、同一 warp reduce)
// ⇒ 输出与 MMVQ **位级相同**(门:`tile_gemm_matches_mmvq_bitwise`)。
//
// # 几何(声明面 `ops/gguf.rs` / `family::gguf::gemm_tile_geom`)
//
//   grid  = (ceil(n / n_warps), ceil(m / MT));block = (32, n_warps, 1)
//   lane = threadIdx.x;warp = threadIdx.y;row = blockIdx.x * n_warps + warp
//   token 窗 = [blockIdx.y * MT, +MT)
//
// # 为什么 MT 是编译期常量
//
// 累加器住**寄存器**(`float acc[MT]`),`#pragma unroll` 让每 token 的归约体
// 独立展开;动态长度会逼出局部内存(register spill)⇒ 慢。故 MT 是模板参数,
// 由声明面的**逐档入口**表达(`..._tile16`),不是运行期标量。
//
// # 纪律(与 `gguf_gemm.cu` 同律)
//
// H1 零分配(激活 q8_1 缓冲由调用方传入)· H2 零运行期设备查询 · H3 逐型入口。
// `out_off`/`out_ld` 语义与 MMVQ 一致(分段合并写)。
// ============================================================================
#include "gguf.cuh"

/// 复用体:一行权重 × MT 个 token(权重块在同一 warp 的 MT 次 `vec_dot` 间复用)
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda,
          int MT>
__device__ __forceinline__ void owl_gguf_mmvq_tile_body(
    const void * __restrict__ weights, const void * __restrict__ xq,
    __half * __restrict__ out, int m, int n, int k, int k_padded, int out_off,
    int out_ld) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int n_warps = blockDim.y;
    const int row = blockIdx.x * n_warps + warp;
    const int m0 = blockIdx.y * MT;
    if (row >= n || m0 >= m) {
        return;
    }
    const int nt = min(MT, m - m0);
    const int weight_blocks_per_row = k / qk;
    const int input_blocks_per_row = k_padded / QK8_1;
    const block_q_t * __restrict__ weight_row =
        reinterpret_cast<const block_q_t *>(weights) + (size_t) row * weight_blocks_per_row;
    const block_q8_1 * __restrict__ input = reinterpret_cast<const block_q8_1 *>(xq);

    // 每 lane 的 MT 个部分和(**寄存器**;unroll 展开 ⇒ 无局部内存)
    float acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        acc[t] = 0.0f;
    }

    const int blocks_per_iter = vdr * WARP_SIZE / qi;
    const int quant_index = vdr * (lane % (qi / vdr));
    for (int k_block = lane / (qi / vdr); k_block < weight_blocks_per_row;
         k_block += blocks_per_iter) {
        const int input_block = k_block * (qk / QK8_1);
        const block_q_t * wblk = &weight_row[k_block];
        const block_q8_1 * xbase = input + (size_t) m0 * input_blocks_per_row + input_block;
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t < nt) {
                acc[t] += vec_dot_q_cuda(wblk, xbase + (size_t) t * input_blocks_per_row,
                                         quant_index);
            }
        }
    }
    // warp 归约 + 写回(`warp_reduce_sum` 是 shuffle 集合操作 ⇒ 全 lane 必须参与;
    // 非活动 token 参与归约但**不写**)
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        const float value = warp_reduce_sum(acc[t]);
        if (t < nt && lane == 0) {
            out[(size_t) (m0 + t) * out_ld + out_off + row] = __float2half(value);
        }
    }
}

// ---- 实例集(M1 四型;加型 = 加一个 wrapper + `family::gguf::tile_entry_for`
//      加一项 + `ops/gguf.rs` 加一条声明)----
//
// MT = 16:权重流量 ÷16(27B prefill 的带宽上限 53 → 850 t/s 量级),
// 累加器 16 个 float/lane + vec_dot 的临时量仍在 64 寄存器档内(实测无 spill)。
#define OWL_GGUF_TILE_MT 16

extern "C" __global__ void owl_gguf_gemm_tile_q8_0(
    const void * __restrict__ weights, const void * __restrict__ xq,
    __half * __restrict__ out, int m, int n, int k, int k_padded, int out_off,
    int out_ld) {
    owl_gguf_mmvq_tile_body<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                            vec_dot_q8_0_q8_1, OWL_GGUF_TILE_MT>(
        weights, xq, out, m, n, k, k_padded, out_off, out_ld);
}

extern "C" __global__ void owl_gguf_gemm_tile_q4k(
    const void * __restrict__ weights, const void * __restrict__ xq,
    __half * __restrict__ out, int m, int n, int k, int k_padded, int out_off,
    int out_ld) {
    owl_gguf_mmvq_tile_body<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                            vec_dot_q4_K_q8_1, OWL_GGUF_TILE_MT>(
        weights, xq, out, m, n, k, k_padded, out_off, out_ld);
}

extern "C" __global__ void owl_gguf_gemm_tile_q5k(
    const void * __restrict__ weights, const void * __restrict__ xq,
    __half * __restrict__ out, int m, int n, int k, int k_padded, int out_off,
    int out_ld) {
    owl_gguf_mmvq_tile_body<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                            vec_dot_q5_K_q8_1, OWL_GGUF_TILE_MT>(
        weights, xq, out, m, n, k, k_padded, out_off, out_ld);
}

extern "C" __global__ void owl_gguf_gemm_tile_q6k(
    const void * __restrict__ weights, const void * __restrict__ xq,
    __half * __restrict__ out, int m, int n, int k, int k_padded, int out_off,
    int out_ld) {
    owl_gguf_mmvq_tile_body<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                            vec_dot_q6_K_q8_1, OWL_GGUF_TILE_MT>(
        weights, xq, out, m, n, k, k_padded, out_off, out_ld);
}
