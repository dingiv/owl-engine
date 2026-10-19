// ============================================================================
// owl GGUF 族:激活量化(f16/bf16 → q8_1 块)**与 GEMM 解耦的发**。
//
// 为什么拆出来(设计 §四 L3 硬律 **H1**):上游快照的 `gguf_gemm()` 在核内
// `cudaMallocAsync` 装 q8_1 中间缓冲 —— 违反 A5.2「一切分配过账本」(账外
// 分配)且**图捕获期禁 malloc**(捕获白名单只放 Launch/Alloc)。本仓把它拆成
// 两发:①本核(激活 → q8_1,缓冲由池供给、调用方传入);②gemm(纯读)。
//
// 语义与上游 `quantize_q8_1` **逐字一致**(warp 内恰好一个 32 元素块):
//   d = amax/127(块内最大绝对值 warp 归约);q = round(x/d);ds.y = 块内求和
//   —— scale 与 sum **同一次归约**产出(分两次归约 = 与上游语义漂移)。
//
// 发射契约(声明面 `ops/gguf.rs::GgufQuantActF16` 逐参对位):
//   grid = (ceil(kx_padded / 32), rows);block = (32, 1, 1)
//   每 warp 恰好覆盖一个 QK8_1 块 ⇒ **归约域 == 块域**(不可改成多块/线程)。
// 越界读:`ix >= kx` 的尾元素按 0 计入(与上游同;`kx_padded` 须为 32 倍数)。
//
// ============================================================================
// **tiled 变体**(2026-10-19;GGUF `ssm_out` 的 v-head 列序号)
//
// 为什么要它(设计 §11.4):GGUF 转换器把 `ssm_out` 的**内维**(v-head 轴)写成
// ggml 的 tiled 序(`conversion/qwen.py::_reorder_v_heads`),而 owl 的 GDN 核
// 产出 grouped 序。该置换**跨量化块**(`hv = 128` < `QK_K = 256`)
// ⇒ 权重字节不可逆 ⇒ 在激活侧做等价置换。落点选在**本核**(而非独立 gather):
// 本核已经在逐元素读激活、按 32 元素块写 q8_1 —— 把“读哪个块”换成查表只是
// **换一个下标**,零额外发射、零额外趟数(与先 gather 再量化**逐位等价**:
// 置换单元 = 32 元素块,块内容不变 ⇒ 量化结果不变)。
//
// 表(`tbl`,U32 元素数 = kx/32)由 host 侧 `transform::vhead_q8_blocks` 生成
// (单一公式源),以**常驻显存**形态由权重槽供给(源侧虚拟键 `blk.N.ssm_out.perm`)。
// 语义:目标块 b 的 32 个元素取自源块 `tbl[b]`;表项越出 `[0, kx/32)` ⇒
// 该元素按 0 计(确定性、响亮可见;表是本仓生成,越界即上游 bug)。
// ============================================================================
#include "gguf.cuh"

template <typename T, float (*to_float)(T), bool TILED>
__device__ __forceinline__ void owl_gguf_quant_act_body(
    const T * __restrict__ x, const unsigned int * __restrict__ tbl,
    void * __restrict__ xq, int kx, int kx_padded) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) {
        return;
    }
    const int iy = blockIdx.y;
    const int i_padded = iy * kx_padded + ix;
    block_q8_1 * y = (block_q8_1 *) xq;
    const int ib = i_padded / QK8_1;
    const int iqs = i_padded % QK8_1;
    float xi = 0.0f;
    if (ix < kx) {
        if (TILED) {
            // 目标元素 ix 的块号/块内偏移 → 源元素(表项越界 ⇒ 0)
            const unsigned int sb = tbl[ix / QK8_1];
            const int src = (int) sb * QK8_1 + (ix % QK8_1);
            xi = src < kx ? to_float(x[iy * kx + src]) : 0.0f;
        } else {
            xi = to_float(x[iy * kx + ix]);
        }
    }
    float amax = fabsf(xi);
    float sum = xi;
    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);
    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    y[ib].qs[iqs] = q;
    if (iqs > 0) {
        return;
    }
    reinterpret_cast<half &>(y[ib].ds.x) = __float2half(d);
    reinterpret_cast<half &>(y[ib].ds.y) = __float2half(sum);
}

extern "C" __global__ void owl_gguf_quant_act_f16(
    const __half * __restrict__ x, void * __restrict__ xq, int kx, int kx_padded) {
    owl_gguf_quant_act_body<__half, __half2float, false>(x, nullptr, xq, kx, kx_padded);
}

extern "C" __global__ void owl_gguf_quant_act_bf16(
    const __nv_bfloat16 * __restrict__ x, void * __restrict__ xq, int kx, int kx_padded) {
    owl_gguf_quant_act_body<__nv_bfloat16, __bfloat162float, false>(x, nullptr, xq, kx, kx_padded);
}

extern "C" __global__ void owl_gguf_quant_act_f16_tiled(
    const __half * __restrict__ x, const unsigned int * __restrict__ tbl,
    void * __restrict__ xq, int kx, int kx_padded) {
    owl_gguf_quant_act_body<__half, __half2float, true>(x, tbl, xq, kx, kx_padded);
}

extern "C" __global__ void owl_gguf_quant_act_bf16_tiled(
    const __nv_bfloat16 * __restrict__ x, const unsigned int * __restrict__ tbl,
    void * __restrict__ xq, int kx, int kx_padded) {
    owl_gguf_quant_act_body<__nv_bfloat16, __bfloat162float, true>(x, tbl, xq, kx, kx_padded);
}
