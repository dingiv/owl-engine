#include <cuda_fp16.h>
#include <cuda_fp8.h>

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


// ---- FP8 e4m3 主池变体(生产 FP8 KV 档;同一几何/同一配对律)----
//
// 与 f16 版的唯一差别 = 读写时的 e4m3 量化:
//   load:  e4m3 字节 → half → float
//   store: float → e4m3(RNE, SATFINITE)字节
// 这样 FP8 生产档也能走“删除 = 后缀 K 乘 R(−S)”,不需要临时 f16 副本。
// 代价 = 每轮一次量化舍入(见文档 §FP8 漂移;超过 delta_limit 走整段重建)。
extern "C" __global__ void owl_kv_rope_k_shift_fp8(
    const __half* cos1,           // [half]
    const __half* sin1,           // [half]
    const float* blocks,          // [nblocks]
    size_t hkv, size_t hd, size_t page, size_t x, size_t half,
    unsigned char* k)             // e4m3 [nb_kv, hkv, hd/x, page, x]
{
    const size_t dg_per_half = half / x;
    const size_t units_per_block = hkv * dg_per_half;
    unsigned int u = blockIdx.x;
    const size_t b = u / units_per_block;
    const size_t rem = u % units_per_block;
    const size_t h = rem / dg_per_half;
    const size_t dg = rem % dg_per_half;

    const size_t blk = (size_t)blocks[b];
    const size_t tid = threadIdx.x;
    const size_t t = tid / x;
    const size_t j = tid % x;

    const size_t lo_off = blk * (hkv * hd * page)
                        + h * (hd * page)
                        + dg * (page * x)
                        + t * x + j;
    const size_t hi_off = lo_off + (half / x) * (page * x);

    __nv_fp8_e4m3 le;
    le.__x = k[lo_off];
    __nv_fp8_e4m3 he;
    he.__x = k[hi_off];
    const float a = __half2float(__half(le));
    const float bb = __half2float(__half(he));
    const float c = __half2float(cos1[dg * x + j]);
    const float s = __half2float(sin1[dg * x + j]);
    const __nv_fp8_e4m3 lo(a * c - bb * s);
    const __nv_fp8_e4m3 hi(bb * c + a * s);
    k[lo_off] = lo.__x;
    k[hi_off] = hi.__x;
}
