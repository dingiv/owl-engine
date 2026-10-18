//! TP 切分计划(TP2 S4 第一刀)—— 张量并行的**形状与切分单源**。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §4(S4 接线表)、§14(S4 侦察与分期)。
//!
//! # 三种切分(Megatron 形态;矩阵视角 [out, in])
//!
//! | 形态 | 切哪一维 | 每 rank 权重 | AR? | 典型层 |
//! |---|---|---|---|---|
//! > **通用性注记(2026-10-16 用户裁决)**:本结构就是**通用分片协议** ——
//! > 规则是**确定性纯函数**,每个 rank 独立用 `(slice_id, n_slices)` 算出自己的
//! > 区间,**零协调、零通信、可并行装载**("分布式共识"在这里不需要:没有分歧
//! > 要消解)。TP 是第一消费者;PP(按层切)/EP(按专家切)将来复用同一函数。
//!
//! | [`TpSplit::ColParallel`] | **out**(行) | `[out/world, in]` | 无(下游消费分片) | gate/up/q/k/v(输出按头/按块分) |
//! | [`TpSplit::RowParallel`] | **in**(列) | `[out, in/world]` | **有**(输出 = 部分和) | down/o_proj/out_proj |
//! | [`TpSplit::Replicate`] | 不切 | 全量 | 无 | norm/embedding(v1 复制) |
//!
//! # 纪律(为什么这么写)
//!
//! 1. **整除硬门**:被切维度必须被 `world` 整除 ⇒ 否则**结构化拒**
//!    (不许 pad、不许静默降级 —— 与项目 "响亮拒绝" 同律);
//! 2. **切分是纯函数**:不碰设备、不碰权重格式 —— 量化权重(packed
//!    qweight/scales)的切分由装载面按同款区间规则做(见 §14 分期;group
//!    对齐另立硬门,不在本层)。
//! 3. **两个 rank 的分片必须恰好覆盖全域、互不重叠** —— 由单测锁死。

use core::ops::Range;

/// 张量并行的切分计划(rank/world;v1 只支持 world = 1 或 2,但本层不限)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpPlan {
    rank: usize,
    world: usize,
}

/// 切分形态(见模块头表)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpSplit {
    /// 不切(每 rank 全量;v1 的 norm / embedding / lm_head)
    Replicate,
    /// 切 out 维(权重行;每 rank 输出 `[out/world, …]`)
    ColParallel,
    /// 切 in 维(权重列;每 rank 输出 `[out, …]` 的**部分和** ⇒ 需 AR)
    RowParallel,
}

impl TpPlan {
    /// 构造(非法 rank/world **响亮拒**;world = 1 是合法的单卡退化)。
    pub fn new(rank: usize, world: usize) -> Result<Self, String> {
        if world == 0 {
            return Err("TpPlan: world = 0(非法)".to_string());
        }
        if rank >= world {
            return Err(format!("TpPlan: rank {rank} ≥ world {world}(非法)"));
        }
        Ok(Self { rank, world })
    }

    /// 单卡退化(等价于不开 TP)。
    pub const fn single() -> Self {
        Self { rank: 0, world: 1 }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world(&self) -> usize {
        self.world
    }

    pub fn is_single(&self) -> bool {
        self.world == 1
    }

    /// 某维度上本 rank 的区间(整除硬门:不整除 ⇒ 拒)。
    ///
    /// 区间 = `[rank × dim/world, (rank+1) × dim/world)` —— 等分块
    /// (Megatron 默认;不平衡切分/序列维切分是将来立项)。
    pub fn shard_range(&self, dim: usize) -> Result<Range<usize>, String> {
        if self.world == 1 {
            return Ok(0..dim);
        }
        if dim % self.world != 0 {
            return Err(format!(
                "TpPlan: 维度 {dim} 不被 world {} 整除(拒绝 —— 不许 pad/静默降级)",
                self.world
            ));
        }
        let per = dim / self.world;
        let start = self.rank * per;
        Ok(start..start + per)
    }

    /// 本 rank 分片长度 = `shard_range().len()`(便捷口)。
    pub fn shard_len(&self, dim: usize) -> Result<usize, String> {
        Ok(self.shard_range(dim)?.len())
    }

    /// 权重 `[out_dim, in_dim]` 的**行区间**(ColParallel 用;其余形态 = 全量)。
    pub fn weight_rows(
        &self,
        split: TpSplit,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Range<usize>, String> {
        match split {
            TpSplit::ColParallel => self.shard_range(out_dim),
            TpSplit::RowParallel | TpSplit::Replicate => {
                let _ = in_dim; // in 维只对 RowParallel 有意义(见 weight_cols)
                Ok(0..out_dim)
            }
        }
    }

    /// 权重 `[out_dim, in_dim]` 的**列区间**(RowParallel 用;其余形态 = 全量)。
    pub fn weight_cols(
        &self,
        split: TpSplit,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Range<usize>, String> {
        match split {
            TpSplit::RowParallel => self.shard_range(in_dim),
            TpSplit::ColParallel | TpSplit::Replicate => {
                let _ = out_dim;
                Ok(0..in_dim)
            }
        }
    }

    /// **分段切**(融合权重:q_proj = q ⊕ gate 两段、in_proj_qkv = q ⊕ k ⊕ v 三段)。
    ///
    /// 输入 = 各段长度(源坐标系);输出 = 本 rank 在**每段内**的子区间
    /// (每段各自等分)。要求每段长度都被 world 整除 ⇒ 否则响亮拒。
    ///
    /// 为什么必须支持:27B 的 `q_proj [12288, 5120]` 是 `q(24×256) ⊕ gate(24×256)`
    /// 融合,按头切后本 rank 要的是"两段各前 3072 行"—— **不是一段连续行区间**。
    pub fn segmented_ranges(
        &self,
        segments: &[usize],
    ) -> Result<Vec<Range<usize>>, String> {
        segments.iter().map(|&len| self.shard_range(len)).collect()
    }

    /// 分段切的**拼接视图**:各段区间 → 源坐标系的行号列表(装载/校验用)
    pub fn segmented_rows(&self, segments: &[usize]) -> Result<Vec<usize>, String> {
        let mut rows = Vec::new();
        let mut off = 0usize;
        for (i, r) in self.segmented_ranges(segments)?.iter().enumerate() {
            for j in r.clone() {
                rows.push(off + j);
            }
            off += segments[i];
        }
        Ok(rows)
    }

    /// 本 rank 的权重形状(按切分形态;装载面据此声明 Want 尺寸)。
    pub fn weight_shape(
        &self,
        split: TpSplit,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<(usize, usize), String> {
        let rows = self.weight_rows(split, out_dim, in_dim)?;
        let cols = self.weight_cols(split, out_dim, in_dim)?;
        Ok((rows.len(), cols.len()))
    }
}

// ============================================================================
// 切分原语(纯 host;f32/f16 同构 —— 按元素宽泛化由调用方换算字节)
// ============================================================================

/// 取行区间子矩阵:`[out, in]` → `rows.range × in`(行主序,整行拷贝)。
///
/// 越界/不完整 = **响亮拒**(`Err`)—— 不静默截断。
pub fn take_rows_f32(w: &[f32], in_dim: usize, rows: Range<usize>) -> Result<Vec<f32>, String> {
    if in_dim == 0 || w.len() % in_dim != 0 {
        return Err(format!("take_rows: 权重长 {} 不是 in_dim {in_dim} 的整数倍", w.len()));
    }
    let out_dim = w.len() / in_dim;
    if rows.start > rows.end || rows.end > out_dim {
        return Err(format!(
            "take_rows: 行区间 {:?} 越界(out_dim={out_dim})",
            rows
        ));
    }
    Ok(w[rows.start * in_dim..rows.end * in_dim].to_vec())
}

/// 取列区间子矩阵:`[out, in]` → `out × cols.range`(逐行切片拷贝)。
pub fn take_cols_f32(w: &[f32], in_dim: usize, cols: Range<usize>) -> Result<Vec<f32>, String> {
    if in_dim == 0 || w.len() % in_dim != 0 {
        return Err(format!("take_cols: 权重长 {} 不是 in_dim {in_dim} 的整数倍", w.len()));
    }
    let out_dim = w.len() / in_dim;
    if cols.start > cols.end || cols.end > in_dim {
        return Err(format!(
            "take_cols: 列区间 {:?} 越界(in_dim={in_dim}, out_dim={out_dim})",
            cols
        ));
    }
    let n = cols.len();
    let mut out = Vec::with_capacity(out_dim * n);
    for r in 0..out_dim {
        let base = r * in_dim + cols.start;
        out.extend_from_slice(&w[base..base + n]);
    }
    Ok(out)
}

/// 激活向量的末维切片(`x [.., in_dim]` → `x [.., cols]`;逐行拷贝)。
pub fn slice_last_dim_f32(x: &[f32], in_dim: usize, cols: Range<usize>) -> Result<Vec<f32>, String> {
    if in_dim == 0 || x.len() % in_dim != 0 {
        return Err(format!("slice_last_dim: 输入长 {} 不是 in_dim {in_dim} 的整数倍", x.len()));
    }
    if cols.start > cols.end || cols.end > in_dim {
        return Err(format!("slice_last_dim: 列区间 {cols:?} 越界(in_dim={in_dim})"));
    }
    let rows = x.len() / in_dim;
    let n = cols.len();
    let mut out = Vec::with_capacity(rows * n);
    for r in 0..rows {
        let base = r * in_dim + cols.start;
        out.extend_from_slice(&x[base..base + n]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(rank: usize, world: usize) -> TpPlan {
        TpPlan::new(rank, world).expect("合法计划")
    }

    /// 区间性质:world 个分片**恰好覆盖**、**互不重叠**、首尾相接
    #[test]
    fn shard_ranges_tile_the_dimension() {
        for world in [1usize, 2, 4] {
            let mut covered: Vec<(usize, usize)> = Vec::new();
            for rank in 0..world {
                let r = plan(rank, world).shard_range(12).expect("整除");
                covered.push((r.start, r.end));
            }
            assert_eq!(covered[0].0, 0, "首片起点 = 0");
            assert_eq!(covered[world - 1].1, 12, "末片终点 = dim");
            for w in covered.windows(2) {
                assert_eq!(w[0].1, w[1].0, "相邻分片必须首尾相接(无缝隙/无重叠)");
            }
            assert_eq!(plan(0, world).shard_len(12).unwrap(), 12 / world);
        }
    }

    /// 整除硬门:不整除 = 响亮拒(不许 pad)
    #[test]
    fn shard_rejects_non_divisible() {
        assert!(plan(0, 2).shard_range(7).is_err());
        assert!(plan(1, 3).shard_range(10).is_err());
        // 单卡退化:任意维度都合法
        assert_eq!(TpPlan::single().shard_range(7).unwrap(), 0..7);
    }

    /// 非法计划响亮拒
    #[test]
    fn plan_rejects_illegal() {
        assert!(TpPlan::new(0, 0).is_err());
        assert!(TpPlan::new(2, 2).is_err());
        assert!(TpPlan::new(1, 2).is_ok());
    }

    /// take_rows / take_cols 与手算参考一致(小矩阵逐元素)
    #[test]
    fn take_primitives_match_hand_reference() {
        // W = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]](out=4, in=3)
        let w: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        let rows = take_rows_f32(&w, 3, 1..3).unwrap();
        assert_eq!(rows, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let cols = take_cols_f32(&w, 3, 1..3).unwrap();
        assert_eq!(cols, vec![2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0]);
        // 越界 = 拒
        assert!(take_rows_f32(&w, 3, 0..5).is_err());
        assert!(take_cols_f32(&w, 3, 0..4).is_err());
        // 区间不完整 = 拒(权重长度与 in_dim 不匹配)
        assert!(take_rows_f32(&w[..10], 3, 0..2).is_err());
    }

    /// 激活末维切片(两行)
    #[test]
    fn slice_last_dim_rows() {
        let x: Vec<f32> = (1..=6).map(|v| v as f32).collect(); // [2, 3]
        let s = slice_last_dim_f32(&x, 3, 1..3).unwrap();
        assert_eq!(s, vec![2.0, 3.0, 5.0, 6.0]);
        assert!(slice_last_dim_f32(&x, 3, 2..4).is_err());
    }

    // ── 切分数学硬证据(纯 host;RowParallel 的 AR 语义在此钉死)─────────

    /// host 参考:y = W @ x(W [out, in] 行主序,x [in] 列向量)
    fn matvec(w: &[f32], out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
        (0..out_dim)
            .map(|r| (0..in_dim).map(|c| w[r * in_dim + c] * x[c]).sum())
            .collect()
    }

    /// **RowParallel(切 in + AR)== 全量**:两个 rank 各算部分和,相加 = W@x
    #[test]
    fn row_parallel_partial_sums_equal_full_product() {
        let (out_dim, in_dim, world) = (4usize, 6usize, 2usize);
        // 确定性伪随机权重/输入(避免 rand 依赖)
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| ((i * 37) % 19) as f32 * 0.25 - 2.0).collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i * 11) % 13) as f32 * 0.5 - 3.0).collect();
        let full = matvec(&w, out_dim, in_dim, &x);

        let mut acc = vec![0.0f32; out_dim];
        for rank in 0..world {
            let p = plan(rank, world);
            let cols = p.weight_cols(TpSplit::RowParallel, out_dim, in_dim).unwrap();
            let n_cols = cols.len();
            let w_shard = take_cols_f32(&w, in_dim, cols.clone()).unwrap();
            let x_shard = slice_last_dim_f32(&x, in_dim, cols).unwrap();
            let partial = matvec(&w_shard, out_dim, n_cols, &x_shard);
            for (a, b) in acc.iter_mut().zip(partial) {
                *a += b;
            }
        }
        for (i, (a, b)) in acc.iter().zip(full.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "RowParallel 部分和 #{i}: {a} vs 全量 {b}"
            );
        }
    }

    /// **ColParallel(切 out)== 全量输出的分块拼接**:各 rank 输出行块,拼起来 = 全量
    #[test]
    fn col_parallel_row_blocks_concatenate_to_full() {
        let (out_dim, in_dim, world) = (6usize, 4usize, 3usize);
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| ((i * 29) % 17) as f32 * 0.1 - 1.0).collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i * 7) % 11) as f32 * 0.3 - 1.5).collect();
        let full = matvec(&w, out_dim, in_dim, &x);

        let mut cat: Vec<f32> = Vec::with_capacity(out_dim);
        for rank in 0..world {
            let p = plan(rank, world);
            let rows = p.weight_rows(TpSplit::ColParallel, out_dim, in_dim).unwrap();
            let w_shard = take_rows_f32(&w, in_dim, rows.clone()).unwrap();
            // ColParallel:输入不切(全量);输出 = 该 rank 的行块
            let part = matvec(&w_shard, rows.len(), in_dim, &x);
            cat.extend_from_slice(&part);
        }
        for (i, (a, b)) in cat.iter().zip(full.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "ColParallel 拼接 #{i}: {a} vs 全量 {b}"
            );
        }
    }

    /// Replicate:两 rank 拿到的权重/输出都与全量一致(零切分)
    #[test]
    fn replicate_is_identity() {
        let (out_dim, in_dim) = (3usize, 5usize);
        let w: Vec<f32> = (1..=15).map(|v| v as f32).collect();
        assert_eq!(w.len(), out_dim * in_dim);
        for rank in 0..2 {
            let p = plan(rank, 2);
            let rows = p.weight_rows(TpSplit::Replicate, out_dim, in_dim).unwrap();
            let cols = p.weight_cols(TpSplit::Replicate, out_dim, in_dim).unwrap();
            assert_eq!((rows, cols), (0..out_dim, 0..in_dim));
            assert_eq!(p.weight_shape(TpSplit::Replicate, out_dim, in_dim).unwrap(), (out_dim, in_dim));
        }
    }

    /// **分段切**(融合权重):q_proj 两段 / in_proj_qkv 三段,各段独立等分
    #[test]
    fn segmented_ranges_split_each_segment() {
        // q_proj:12288 = q(6144) ⊕ gate(6144);world=2 ⇒ 每段取 3072
        let p0 = plan(0, 2);
        assert_eq!(p0.segmented_ranges(&[6144, 6144]).unwrap(), vec![0..3072, 0..3072]);
        let p1 = plan(1, 2);
        assert_eq!(p1.segmented_ranges(&[6144, 6144]).unwrap(), vec![3072..6144, 3072..6144]);
        // 行号拼接视图 = 源坐标系(rank0 = 两段各前半)
        assert_eq!(p0.segmented_rows(&[6144, 6144]).unwrap()[..3], [0, 1, 2]);
        assert_eq!(&p0.segmented_rows(&[6144, 6144]).unwrap()[3072..3075], [6144, 6145, 6146]);
        // in_proj_qkv:10240 = q(2048) ⊕ k(2048) ⊕ v(6144);world=2
        assert_eq!(
            p0.segmented_ranges(&[2048, 2048, 6144]).unwrap(),
            vec![0..1024, 0..1024, 0..3072]
        );
        // 段长不整除 = 响亮拒(不许 pad)
        assert!(p0.segmented_ranges(&[6144, 6145]).is_err(), "6145 为奇数 ⇒ 拒");
        assert!(p0.segmented_ranges(&[2049, 6144]).is_err(), "段 2049 不被 2 整除 ⇒ 拒");
    }

    /// 权重形状面:两形态的切分形状正确
    #[test]
    fn weight_shapes_by_split() {
        let p = plan(1, 2);
        assert_eq!(p.weight_shape(TpSplit::ColParallel, 8, 6).unwrap(), (4, 6));
        assert_eq!(p.weight_shape(TpSplit::RowParallel, 8, 6).unwrap(), (8, 3));
        // 不整除 = 拒
        assert!(p.weight_shape(TpSplit::ColParallel, 7, 6).is_err());
        assert!(p.weight_shape(TpSplit::RowParallel, 8, 5).is_err());
    }
}
