#include <cuda_fp16.h>

// **图像窗口旋转**:池内 K 的 Δ-rope(原地方向,单行增量表)。
//
// 语义:把一组物理页里的 K 整体乘 R(Δ) —— rotate-half,配对 (i, i+half),
// 单行 cos/sin 表广播到全部分页。**V 不在此核**(V 无位置)。
//
// 为什么不能复用 `owl_rope_half_partial_f16`:那个核吃的是 token-major
// `[tokens, heads, head_dim]` 连续布局;主池 K 是 kNHD 分页
// `[nb_kv, hkv, hd/x, page, x]`(见 `StatePool::k_shape_kv`)—— token 在
// 倒数第二维,**拿不到 token-major 视图**(设计卷 §5.3 的"页内 32 tok 连续"
// 是错的,2026-10-20 施工核对翻案)。硬拆/搬运只会把 40 MiB 变成四趟。
//
// 几何:
//   * 单元 = (物理页, kv 头, 半维组 dg ∈ [0, half/x)) —— 一单元 page·x 个元素,
//   * 线程 = 单元内一个 (t, j);每线程处理**一对**(dg, dg + half/x),
//   * 配对的两半落在不同 dg ⇒ 同线程读/写同一 token 的两个 8 元素段,
//     **无竞争**,故可原地(in-place)且不需要 scratch。
//
// 约束(装配面已对契约,这里只是最后一道):`half % x == 0`、
// `2·half ≤ hd`、`page·x ≤ 1024`(block 尺寸)。
//
// ⚠️ `blocks` = **f32 过线**(契约 5;与 `slots` 同律:算子准入解析器
// 禁 `unsigned int` 形参,索引数组一律 float —— 页号 ≤ 2^24 故精确)。
extern "C" __global__ void owl_kv_rope_k_shift_f16(
    const __half* cos1,           // [half]   Δ 表(单行;与 K 同 dtype)
    const __half* sin1,           // [half]
    const float* blocks,          // [nblocks] 物理页号(逻辑序无关)
    size_t hkv, size_t hd, size_t page, size_t x, size_t half,
    __half* k)                    // [nb_kv, hkv, hd/x, page, x](原地读写)
{
    const size_t dg_per_half = half / x;               // 半维里的 dg 组数
    const size_t units_per_block = hkv * dg_per_half;  // 每页的单元数
    unsigned int u = blockIdx.x;
    const size_t b = u / units_per_block;
    const size_t rem = u % units_per_block;
    const size_t h = rem / dg_per_half;
    const size_t dg = rem % dg_per_half;

    const size_t blk = (size_t)blocks[b];
    const size_t tid = threadIdx.x;      // ∈ [0, page*x)
    const size_t t = tid / x;
    const size_t j = tid % x;

    // 低半:d = dg·x + j;高半:d + half ⇒ dg + half/x
    const size_t lo_off = blk * (hkv * hd * page)
                        + h * (hd * page)
                        + dg * (page * x)
                        + t * x + j;
    const size_t hi_off = lo_off + (half / x) * (page * x);

    const float c = __half2float(cos1[dg * x + j]);
    const float s = __half2float(sin1[dg * x + j]);
    const float a = __half2float(k[lo_off]);
    const float bb = __half2float(k[hi_off]);
    k[lo_off] = __float2half(a * c - bb * s);
    k[hi_off] = __float2half(bb * c + a * s);
}
