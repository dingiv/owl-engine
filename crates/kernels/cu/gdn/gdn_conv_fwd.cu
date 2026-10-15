// ============================================================================
// GDN 核族:causal conv1d prefill varlen(k=4)(2026-10-13 C 轮自 cu/owl/gdn.cu 按**算子族**拆出 ——
// 一算子一 `.cu`,与 ops/gdn.rs 的声明对望)。
//
// 溯源:repos/attention.rs/src/kernels/src/gdn.cu 模板体显式展开(nvrtc 禁
// host launcher;旧世界 GDN_GATING_KERNEL 宏模式同款)。
//
// dtype 律:激活 half → float 计算 → half 写;**rec/conv state 恒 f32**
// (上游同款 + HF/vLLM/xinfer 三方先例);slots 恒 f32 数值过线(契约 5,
// 上游 i64 已适配);输出块固定末参(槽序契约 4);标量 i32 与 Arg::I32
// 严格对位(禁止 unsigned int)。数学与 f32 版逐式同源(移植不失真由
// 金标对拍背书)。
//
// ⚠️ 2026-10-14 N1(t 维并行)改写 —— 语义不变,发射几何变:
//   旧版:`grid=(batch, ceil(d/256))`,`t` 循环核内顺序扫描 ⇒ 批 1 时
//         q/k/v 各只 8/8/24 个 CTA(84 SM 上),延迟绑死(实测 122-170 µs/发,
//         144 发/chunk = 17.6 ms = prefill 的 4.6%;nsys 定谳见
//         docs/arch/prefill-graph-施工.md §12.4b)。
//   新版:`grid=(ceil(d/256), tiles, batch)`,`tiles = ceil(len/32)`(picker),
//         **核内 ts = ceil(len/gridDim.y) 自算**(tile 行数不新增参数 ⇒
//         picker 与核零常量耦合),每 tile 只走自己那段 t。
//   为什么能并行:**conv state 携带的只有"前 3 个输入"**(state 槽 = 前 3 个
//         x,不是 conv 输出)⇒ t>=start+3 的行是纯滑窗,x 就在全局内存里;
//         只有 tile 0 需要读旧 state。
//   竞争面封死:**只有 blockIdx.y==0 读旧 state、也只有它写新 state**
//         (同一线程读写同一 cell ⇒ 程序序足够,无需 syncthreads);
//         tile>=1 的 prologue 读 x(t0-3 >= start 由 ts>=3 保证)⇒ 跨块零竞争。
//         **前提:ts>=3 或 tiles==1** —— picker 用 `tiles=ceil(len/32)` 保证
//         (len>=32 ⇒ ts>=32;len<32 ⇒ tiles==1)。
//   数值:**与旧版逐位一致**(同一 fma 序:先 x_t*w3,再 w0/w1/w2;同一 __expf
//         与 `/`;state 写回公式 `hist_for_row(end)` 对 len<3 退化为移位寄存器)。
// ============================================================================

#include <cuda_fp16.h>

__device__ __forceinline__ float gdn16_to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ __half gdn16_from_float(float x) { return __float2half(x); }
__device__ __forceinline__ float gdn16_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

// ---- causal conv1d prefill varlen f16(上游 causal_conv1d_fwd_varlen_
// kernel<T,4> 同体;owl 适配:state 槽寻址(上游 seq 直索,免 gather/scatter
// 垫)、去 state_snapshots(MTP 域外)、cu_seqlens [batch+1] u32 保真)----
//
// grid = (ceil(d/256), tiles, batch) × (256,1,1);t 维度按 tile 并行
// (tiles = ceil(len/32);核内 ts = ceil(len/gridDim.y))。
//
// 行 `t` 的第 `i` 个历史输入(hist[0]=最老 = x[t-3]):位置 pos = t-3+i;
// pos 落在本序列内 ⇒ 直接读 x(pos 在全局内存里,滑窗纯函数);
// pos < start(只有前 3 行会)⇒ 读 conv_state 槽 `pos-(start-3)`。
// slot < 0 ⇒ 无 state,历史补 0(旧版同款语义)。
__device__ __forceinline__ float gdn_hist(const __half *__restrict__ x,
                                          const float *__restrict__ st,
                                          long long start, int d, int chan,
                                          long long t, int i) {
    const long long pos = t - 3 + i;
    if (pos >= start) return gdn16_to_float(x[(size_t)pos * d + chan]);
    if (st != nullptr) return st[(size_t)(pos - start + 3)];
    return 0.0f;
}

extern "C" __global__ void owl_gdn_conv_fwd_f16(
    const __half *__restrict__ x,            // [total_tokens, d]
    const __half *__restrict__ w,            // [d, 4]
    float *__restrict__ conv_state,          // [max_slots, d, 3](in/out 恒 f32)
    const float *__restrict__ slots,         // [batch](f32 数值;负 = 全跳)
    const float *__restrict__ cu_seqlens,    // [batch + 1](f32 数值过线,契约 5)
    int batch, int d, int silu,
    __half *__restrict__ out) {              // [total_tokens, d](末参 = 输出)
    const int seq_idx = blockIdx.z;
    const int channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= batch || channel_idx >= d) return;

    const int slot = (int)slots[seq_idx];
    const long long start = (long long)cu_seqlens[seq_idx];
    const long long end = (long long)cu_seqlens[seq_idx + 1];
    const long long seq_len = end - start;
    if (seq_len <= 0) return;

    // 本 tile 的行区间(tile 行数核内自算;均衡分块)
    const long long ts_raw = (seq_len + (long long)gridDim.y - 1) / (long long)gridDim.y;
    // 防线(2026-10-14 焊死):"只有 tile 0 碰 state" 的前提是 ts>=3
    // (tile>=1 的 prologue 读 x 而非旧 state)。picker 按 TILE_ROWS=32 给网格
    // 时恒 ts>=32;但 **batch>1 的不均长序列** 会算出 ts<3(那条序列短),
    // 此时若照常分块 ⇒ tile>=1 读旧 state 而 tile 0 正在写 ⇒ 跨块竞争
    // (静默错值,且随调度时快时慢)。⇒ ts<3 时退化为**单 tile 串行**
    // (等价旧版行为):只有 tile 0 有活,竞争窗口物理消失。
    const bool degenerate = (ts_raw < 3) && (gridDim.y > 1);
    if (degenerate && blockIdx.y != 0) return;
    const long long ts = degenerate ? seq_len : ts_raw;
    const long long t0 = start + (long long)blockIdx.y * ts;
    const long long t1 = (t0 + ts < end) ? (t0 + ts) : end;
    if (t0 >= t1) return;

    const long long st_base = (long long)slot * d + channel_idx;
    const float *st = (slot >= 0) ? (conv_state + (size_t)st_base * 3) : nullptr;

    const __half *w_ptr = w + (size_t)channel_idx * 4;
    float w_reg[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) w_reg[k] = gdn16_to_float(w_ptr[k]);

    // 旧 state 三槽(tile 0 的写回需要;tile>=1 用不到 ⇒ 只在 tile 0 读)
    float st_old[3] = {0.0f, 0.0f, 0.0f};
    if (blockIdx.y == 0 && st != nullptr) {
#pragma unroll
        for (int i = 0; i < 3; ++i) st_old[i] = st[i];
    }

    // tile 起点前 3 个输入(hist[0] = 最老)
    float hist[3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
        hist[i] = gdn_hist(x, st, start, d, channel_idx, t0, i);

    for (long long t = t0; t < t1; ++t) {
        const float x_t = gdn16_to_float(x[(size_t)t * d + channel_idx]);
        float sum = x_t * w_reg[3];
#pragma unroll
        for (int k = 0; k < 3; ++k) sum = __fmaf_rn(hist[k], w_reg[k], sum);
        if (silu) sum /= (1.0f + __expf(-sum));
        out[(size_t)t * d + channel_idx] = gdn16_from_float(sum);
        hist[0] = hist[1];
        hist[1] = hist[2];
        hist[2] = x_t;
    }

    // 新 state = 本序列末 3 个输入 = `历史输入(end, i)`。
    // 只有 tile 0 写(它的 st_old 已读;同线程读写同 cell ⇒ 程序序足够)。
    // 对 seq_len < 3 该式自动退化为移位寄存器(旧版逐位同源)。
    if (blockIdx.y == 0 && st != nullptr) {
        float *st_mut = conv_state + (size_t)st_base * 3;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const long long pos = end - 3 + i;
            st_mut[i] = (pos >= start)
                            ? gdn16_to_float(x[(size_t)pos * d + channel_idx])
                            : st_old[pos - start + 3];
        }
    }
}
