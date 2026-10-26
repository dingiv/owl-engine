// ============================================================================
// **两半表候选合并**(DFlash2 切档 propose 的 rank0 收割面;2026-10-26 propose 刀3)
//
// 语义 = host `merge_topk_two_halves`(crates/engine/src/running.rs)**逐位同律**:
//   ① 每行取 A 的 K 个候选(全表域,id 原样)与 B 的 K 个(半表域,id += half);
//   ② 丢弃 **id 非有限**的哨兵槽(未填满的候选位;f32 载 id 是契约 5);
//   ③ 按 **(值降, id 升)** 严格全序取前 K —— 只比值会让同值不同 id 乱序,
//      与 topk16/topk32 的全序律同一条(平局取小索引);
//   ④ 输出**单缓冲** [2·rows·K]:前段值区 ‖ 后段索引区(与 topk16 同布局,
//      便于直接喂 `dflash.select` 的两条 slice_view)。
//
// 并行:grid = rows,block = 32(一 warp 一行;lane 0 串行选取)。
// 为什么单 lane:候选只有 2K ≤ 32 个,插入式选取 ~K·2K = 512 次比较 ≈ 1µs,
// 远低于一次发射的固定开销;多 lane 只会把全序律搅成两套规范。
//
// NaN:值域出现 NaN 时,host `sort_by(partial_cmp.unwrap_or(Equal))` 的序是
// 未定义的;本核**显式**把 NaN 当 −∞(排最后),即"响亮地把垃圾沉底"。
// ============================================================================
#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
#endif

#define OWL_MERGE_HALVES_KMAX 32

extern "C" __global__ void owl_topk_merge_halves_f32(
    const float* __restrict__ vals_a,   // [rows, K] 全表域候选值
    const float* __restrict__ ids_a,    // [rows, K] 全表域候选 id(f32 载整数)
    const float* __restrict__ vals_b,   // [rows, K] 半表域候选值
    const float* __restrict__ ids_b,    // [rows, K] 半表域候选 id
    int half,                           // rank1 的 id 偏移(= 半表宽)
    int rows, int k,
    float* __restrict__ out) {          // [2·rows·K]:值区 ‖ 索引区
    const int r = blockIdx.x;
    if (threadIdx.x != 0 || r >= rows || k <= 0 || k > OWL_MERGE_HALVES_KMAX) return;

    float bv[2 * OWL_MERGE_HALVES_KMAX];
    int bi[2 * OWL_MERGE_HALVES_KMAX];
    int m = 0;
    for (int j = 0; j < k; ++j) {
        const float ia = ids_a[r * k + j];
        if (isfinite(ia)) {
            const float v = vals_a[r * k + j];
            bv[m] = isnan(v) ? -3.0e38f : v;
            bi[m] = (int)ia;
            ++m;
        }
        const float ib = ids_b[r * k + j];
        if (isfinite(ib)) {
            const float v = vals_b[r * k + j];
            bv[m] = isnan(v) ? -3.0e38f : v;
            bi[m] = (int)ib + half;
            ++m;
        }
    }

    // 选取前 k:每轮在剩余里挑全序最大,交换到位置 o(选择排序)
    const int take = (m < k) ? m : k;
    for (int o = 0; o < take; ++o) {
        int best = o;
        for (int j = o + 1; j < m; ++j) {
            const bool better =
                (bv[j] > bv[best]) || (bv[j] == bv[best] && bi[j] < bi[best]);
            if (better) best = j;
        }
        if (best != o) {
            const float tv = bv[o];
            bv[o] = bv[best];
            bv[best] = tv;
            const int ti = bi[o];
            bi[o] = bi[best];
            bi[best] = ti;
        }
    }
    // 输出单缓冲;候选不足 k 时余位填 0(下游 select 读满 k 宽 —— 与 host
    // 只会给出 k 个(2k ≥ k 恒成立)不同,这里只防御 rows/k 声明失配)
    for (int o = 0; o < k; ++o) {
        out[r * k + o] = (o < take) ? bv[o] : 0.0f;
        out[(long long)rows * k + r * k + o] = (o < take) ? (float)bi[o] : 0.0f;
    }
}
