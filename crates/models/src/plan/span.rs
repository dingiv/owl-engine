//! **源读取计划**(S4 ② 的装载面核心)—— 分片后的读请求 → 源坐标系的读序列。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §15.3 / §15.3b。
//!
//! # 为什么要有它
//!
//! 装载执行面(6 条路径:direct/cast/packed/device_rearrange/transposed/…)都
//! 只做一件事:把"某个键的 [off, len) 元素"搬进目标块。切分后**目标坐标系
//! 变了**(形状变成分片),同一个 `[off, len)` 在源坐标系里可能是:
//!
//! | 形态 | 源读法 |
//! |---|---|
//! | [`SourceSpan::Rows`] | **连续**区间(out 切:行窗口) |
//! | [`SourceSpan::Cols`] | 每行取窗口(in 切:stride 读) |
//! | [`SourceSpan::RowsSeg`] | 多段行窗口(融合权重:q⊕gate / q⊕k⊕v) |
//!
//! 本模块是**纯算术**(零 IO、零依赖)→ 有单测;执行由装载面的源包装器
//! (`ShardedSource`)按 [`SourceSpan::map`] 的读序列做。
//!
//! # 纪律
//!
//! - **确定性纯函数**:同一 `(rule, shape, plan)` 在任何 rank/任何进程上给出
//!   同一结果 ⇒ 零协调并行装载(用户裁决 2026-10-16);
//! - 越界/不整除 = **响亮拒**(不静默截断)。

use crate::plan::rules::{SegKind, WantAxis, WeightRule};
use crate::plan::tp::TpPlan;

/// 本 slice 的源读取计划(目标坐标系 → 源坐标系的映射)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceSpan {
    /// 行窗口(源行 `[row_start, row_start+rows)`,**连续**)
    Rows { row_start: usize, rows: usize },
    /// 列窗口(每行取 `[col_start, col_start+cols)`)
    Cols { col_start: usize, cols: usize },
    /// 分段行窗口(融合权重;`(段起点, 段内起点, 段内行数)`)
    RowsSeg { segs: Vec<(usize, usize, usize)> },
    /// **分段列窗口**(融合权重 × **k-major 派生键**:每行取多段列窗口;
    /// `(段起点, 段内起点, 段内列数)`)。
    ///
    /// 为何需要(2026-10-17):`q_proj.scales` / `in_proj_qkvz.scales` 这类
    /// **分段 + marlin 布局**的键,段沿 **n 侧**、而 n 在 k-major 序里是第 1 维
    /// ⇒ 每行要取"各段各半"的多个窗口,`Cols`(单窗)与 `RowsSeg`(行段)
    /// 都表达不了。
    ColsSeg { segs: Vec<(usize, usize, usize)> },
}

impl SourceSpan {
    /// **纯算术**:目标坐标系的 `[off, len)` 元素区间 → 源坐标系的读序列
    /// `[(src_off, len), …]`。
    ///
    /// - `tgt_row_width`:目标行宽(元素;= 分片 shape 的首行元素数)
    /// - `src_row_width`:源行宽(元素;= 全量 shape 的行元素数)
    ///
    /// 目标区间跨行时自动拆成每行一段(Rows 形态下相邻行连续,**合并**成一段
    /// —— 少读一次)。
    pub fn map(
        &self,
        off: usize,
        len: usize,
        tgt_row_width: usize,
        src_row_width: usize,
    ) -> Result<Vec<(usize, usize)>, String> {
        if tgt_row_width == 0 || src_row_width == 0 {
            return Err("SourceSpan::map: 行宽为 0(非法)".into());
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        // 行窗口:目标行 → 源行(连续情形可整段合并)
        let row_of = |r: usize| -> Result<usize, String> {
            match self {
                SourceSpan::Rows { row_start, rows } => {
                    if r >= *rows {
                        return Err(format!("行 {r} 越界(窗口 {rows} 行)"));
                    }
                    Ok(row_start + r)
                }
                SourceSpan::Cols { .. } | SourceSpan::ColsSeg { .. } => Ok(r),
                SourceSpan::RowsSeg { segs } => {
                    let mut acc = 0usize;
                    for (seg_off, in_off, in_rows) in segs {
                        if r < acc + in_rows {
                            return Ok(seg_off + in_off + (r - acc));
                        }
                        acc += in_rows;
                    }
                    Err(format!("行 {r} 越界(分段共 {acc} 行)"))
                }
            }
        };
        // 目标列 → **(源列, 本段剩余列数)**
        //
        // ⚠️ 第二项不可省(2026-10-17 单测当场抓到):`take` 原先按"整行连续"
        // 算(`tgt_row_width - c`),而 **`ColsSeg` 是分段连续** —— 目标列跨段时
        // 源列会跳段 ⇒ 不夹住就会**静默少读/错读**(实测把跨段的两个元素合并成
        // 一段连续读)。`Rows`/`Cols` 恒连续 ⇒ 返回 `usize::MAX` 等价旧行为。
        let col_run = |c: usize| -> Result<(usize, usize), String> {
            match self {
                SourceSpan::Cols { col_start, cols } => {
                    if c >= *cols {
                        return Err(format!("列 {c} 越界(窗口 {cols} 列)"));
                    }
                    Ok((col_start + c, cols - c))
                }
                SourceSpan::ColsSeg { segs } => {
                    let mut acc = 0usize;
                    for (seg_off, in_off, in_cols) in segs {
                        if c < acc + in_cols {
                            let within = c - acc;
                            return Ok((seg_off + in_off + within, in_cols - within));
                        }
                        acc += in_cols;
                    }
                    Err(format!("列 {c} 越界(分段共 {acc} 列)"))
                }
                _ => Ok((c, usize::MAX)),
            }
        };

        let mut out: Vec<(usize, usize)> = Vec::new();
        let mut o = off;
        let mut remaining = len;
        while remaining > 0 {
            let r = o / tgt_row_width;
            let c = o % tgt_row_width;
            let (src_col, run) = col_run(c)?;
            let take = (tgt_row_width - c).min(run).min(remaining);
            let src_row = row_of(r)?;
            let src_off = src_row * src_row_width + src_col;
            if let Some(last) = out.last_mut() {
                // 与上一段连续 ⇒ 合并(仅 Rows/Cols 的顺行情形)
                if last.0 + last.1 == src_off {
                    last.1 += take;
                    o += take;
                    remaining -= take;
                    continue;
                }
            }
            out.push((src_off, take));
            o += take;
            remaining -= take;
        }
        Ok(out)
    }
}

/// **计划器**:`(规则, 全量 shape, plan, 段长, 坐标序) → (读计划, 分片 shape)`。
///
/// - 返回 `None` = 不切(`Replicate`;调用方按全量直通);
/// - `segs`:分段切的段长(**源坐标系**;由调用方按 `SegKind::segments(spec)` 展开);
/// - `axis`:[`WantAxis`] —— want 形状的坐标序。规则表按**逻辑** `[out, k]`
///   写;`KMajor` 的键(量化 marlin 产物)两个维**对调** ⇒
///   Col(out) → 第 1 维、Row(k) → 第 0 维、分段沿 n ⇒ **列段**。
///   漏给/给错 = 切错维(分片形状与层侧声明不符 ⇒ `build_shard_pass` 当场拒)。
pub fn plan_span(
    rule: WeightRule,
    shape: &[usize],
    plan: TpPlan,
    segs: &[usize],
    axis: WantAxis,
) -> Result<Option<(SourceSpan, Vec<usize>)>, String> {
    if plan.is_single() || rule == WeightRule::Replicate {
        return Ok(None);
    }
    if shape.is_empty() {
        return Err("plan_span: 空 shape".into());
    }
    // 规则 → want 自身坐标系的切法
    #[derive(Clone, Copy)]
    enum Split {
        Dim0,
        Dim1,
        Dim0Seg,
        Dim1Seg,
    }
    let split = match (rule, axis) {
        (WeightRule::Replicate, _) => return Ok(None),
        // A_log / dt_bias 是一维小向量(只有 Logical 档)
        (WeightRule::PerHead, _) => Split::Dim0,
        (WeightRule::UniformCol, WantAxis::Logical)
        | (WeightRule::UniformRow, WantAxis::KMajor) => Split::Dim0,
        (WeightRule::UniformRow, WantAxis::Logical)
        | (WeightRule::UniformCol, WantAxis::KMajor) => Split::Dim1,
        (WeightRule::SegmentedCol(_), WantAxis::Logical) => Split::Dim0Seg,
        (WeightRule::SegmentedCol(_), WantAxis::KMajor) => Split::Dim1Seg,
    };
    let mut out_shape = shape.to_vec();
    let seg_split = |segs: &[usize], dim: usize| -> Result<(Vec<(usize, usize, usize)>, usize), String> {
        if segs.is_empty() {
            return Err("plan_span: 分段切的段长为空(调用方须按 SegKind 展开)".into());
        }
        let total: usize = segs.iter().sum();
        if total != shape[dim] {
            return Err(format!(
                "plan_span: 段长合计 {total} ≠ 第 {dim} 维 {}(shape {shape:?})",
                shape[dim]
            ));
        }
        let mut spans = Vec::with_capacity(segs.len());
        let (mut off, mut shard) = (0usize, 0usize);
        for &seg in segs {
            let r = plan.shard_range(seg)?;
            spans.push((off, r.start, r.len()));
            off += seg;
            shard += r.len();
        }
        Ok((spans, shard))
    };
    match split {
        Split::Dim0 => {
            let r = plan.shard_range(shape[0])?;
            let span = SourceSpan::Rows { row_start: r.start, rows: r.len() };
            out_shape[0] = r.len();
            Ok(Some((span, out_shape)))
        }
        Split::Dim1 => {
            if shape.len() < 2 {
                return Err(format!(
                    "plan_span: 切第 1 维需要 ≥2 维(得 {shape:?};一维权重请用 PerHead)"
                ));
            }
            let r = plan.shard_range(shape[1])?;
            let span = SourceSpan::Cols { col_start: r.start, cols: r.len() };
            out_shape[1] = r.len();
            Ok(Some((span, out_shape)))
        }
        Split::Dim0Seg => {
            let (segs_span, shard_rows) = seg_split(segs, 0)?;
            let span = SourceSpan::RowsSeg { segs: segs_span };
            out_shape[0] = shard_rows;
            Ok(Some((span, out_shape)))
        }
        Split::Dim1Seg => {
            if shape.len() < 2 {
                return Err(format!(
                    "plan_span: 列段切需要 ≥2 维(得 {shape:?})"
                ));
            }
            let (segs_span, shard_cols) = seg_split(segs, 1)?;
            let span = SourceSpan::ColsSeg { segs: segs_span };
            out_shape[1] = shard_cols;
            Ok(Some((span, out_shape)))
        }
    }
}

/// **段长单位换算**(层侧**逻辑单位** → 键自身维单位)。
///
/// 为什么需要(2026-10-17):层侧声明的是语义事实(`q_gate` / `k_dim` / `v_dim`),
/// 而打包派生键的 n 侧单位不同 —— `zeros` = `n_pack/8`、marlin B 视图 = `2·n_pack`
/// ⇒ 段长必须等比换到该键的第 `dim` 维口径上。
///
/// 换算比 = `dim_len / Σsegs`(有理,**两个方向都认**);不成整数比 / 某段换算后
/// 非整数 / 合计不符 ⇒ **响亮拒**(不许猜)。**唯一实现** —— 装载面
/// (`build_shard_pass`)与门/参考实现共用,免两套规范。
pub fn segments_in_key_units(
    segs_full: &[usize],
    dim_len: usize,
    key: &str,
    dim: usize,
) -> Result<Vec<usize>, String> {
    let declared: usize = segs_full.iter().sum();
    if declared == dim_len {
        return Ok(segs_full.to_vec());
    }
    let (num, den) = if declared != 0 && dim_len % declared == 0 {
        (dim_len / declared, 1usize)
    } else if dim_len != 0 && declared % dim_len == 0 {
        (1usize, declared / dim_len)
    } else {
        return Err(format!(
            "段长合计 {declared} 与第 {dim} 维 {dim_len} 不成整数比({key};声明须为逻辑单位)"
        ));
    };
    let mut scaled = Vec::with_capacity(segs_full.len());
    for s in segs_full {
        let v = s * num;
        if v % den != 0 {
            return Err(format!("{key}: 段 {s} 换算后非整数(比 {num}/{den})"));
        }
        scaled.push(v / den);
    }
    let total: usize = scaled.iter().sum();
    if total != dim_len {
        return Err(format!(
            "{key}: 段长换算后合计 {total} ≠ 第 {dim} 维 {dim_len}"
        ));
    }
    Ok(scaled)
}

/// 段长展开(按 `SegKind` + `ModelSpec`;装载面的便捷口)
pub fn segments_of(kind: SegKind, spec: &crate::model::ModelSpec) -> Result<Vec<usize>, String> {
    kind.segments(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::rules::WeightRule;

    fn plan(r: usize, w: usize) -> TpPlan {
        TpPlan::new(r, w).unwrap()
    }

    /// **KMajor 序(量化 marlin 产物)**:切 out 落到第 1 维(列窗)、切 k 落到第 0 维。
    /// 用例取真形状:gate_proj(out=17408, k=5120)的 `scales` = [k/32, out] = [160, 17408]
    #[test]
    fn kmajor_axis_swaps_dims() {
        let (span, shape) = plan_span(
            WeightRule::UniformCol,
            &[160, 17408],
            plan(1, 2),
            &[],
            WantAxis::KMajor,
        )
        .unwrap()
        .unwrap();
        assert_eq!(span, SourceSpan::Cols { col_start: 8704, cols: 8704 });
        assert_eq!(shape, vec![160, 8704], "KMajor 切 out = 切第 1 维");

        // 同一键若是 Row 规则(down_proj):切第 0 维(k 侧)
        let (span, shape) = plan_span(
            WeightRule::UniformRow,
            &[544, 5120],
            plan(0, 2),
            &[],
            WantAxis::KMajor,
        )
        .unwrap()
        .unwrap();
        assert_eq!(span, SourceSpan::Rows { row_start: 0, rows: 272 });
        assert_eq!(shape, vec![272, 5120], "KMajor 切 k = 切第 0 维");
    }

    /// **列段切**(分段融合权重 × KMajor):q_proj.scales = [k/32, 2·q_gate],
    /// 本 slice 取每段各半 ⇒ 每行两个列窗口
    #[test]
    fn kmajor_segmented_is_cols_seg() {
        let (q_gate, k) = (3072usize, 5120usize);
        let shape = vec![k / 32, 2 * q_gate];
        let segs = vec![q_gate, q_gate];
        let (span, out_shape) = plan_span(
            WeightRule::SegmentedCol(SegKind::QGate),
            &shape,
            plan(1, 2),
            &segs,
            WantAxis::KMajor,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            span,
            SourceSpan::ColsSeg { segs: vec![(0, q_gate / 2, q_gate / 2), (q_gate, q_gate / 2, q_gate / 2)] }
        );
        assert_eq!(out_shape, vec![k / 32, q_gate], "列段:第 1 维 = 段各半之和");
        // 逐行映射自证(rank1 ⇒ 段内窗口起点 = 段长/2 = 1536):
        // 目标列 0,1 → 源列 1536,1537(段内连续 ⇒ 合并成一段)
        let m = span.map(0, 2, out_shape[1], shape[1]).unwrap();
        assert_eq!(m, vec![(1536, 2)]);
        // 跨段:目标列 1535(第一段末)→ 源 3071;目标列 1536(第二段首)→ 源 4608
        // ⇒ 源不连续 ⇒ 拆两段(这正是 Cols 单窗表达不了的形状)
        let m2 = span.map(1535, 2, out_shape[1], shape[1]).unwrap();
        assert_eq!(m2, vec![(3071, 1), (4608, 1)]);
    }

    /// 行切(out):连续区间 + 分片 shape;跨行合并成一段
    #[test]
    fn rows_span_is_contiguous() {
        // 源 [8, 6](out=8, in=6);world=2 ⇒ rank1 取行 4..8
        let (span, shape) = plan_span(WeightRule::UniformCol, &[8, 6], plan(1, 2), &[], WantAxis::Logical)
            .unwrap()
            .unwrap();
        assert_eq!(shape, vec![4, 6]);
        // 目标 [4, 6] 的第 0..12 元素(前两行)→ 源 4*6=24 .. 36(连续!)
        assert_eq!(span.map(0, 12, 6, 6).unwrap(), vec![(24, 12)]);
        // 跨行边界:目标 [0,8) → 源 [24, 32)(仍连续)
        assert_eq!(span.map(0, 8, 6, 6).unwrap(), vec![(24, 8)]);
        // 越界(目标 4 行,请求到第 5 行)
        assert!(span.map(30, 6, 6, 6).is_err());
    }

    /// 列切(in):stride 读(逐行窗口)
    #[test]
    fn cols_span_is_strided_per_row() {
        // 源 [4, 8](in=8);world=2 ⇒ rank1 取列 4..8
        let (span, shape) = plan_span(WeightRule::UniformRow, &[4, 8], plan(1, 2), &[], WantAxis::Logical)
            .unwrap()
            .unwrap();
        assert_eq!(shape, vec![4, 4]);
        // 目标 [4, 4] 的第 0..4(第 0 行)→ 源行 0 的列 4..8 = [4, 4)
        assert_eq!(span.map(0, 4, 4, 8).unwrap(), vec![(4, 4)]);
        // 两行:目标 [0, 8) → 源 (4,4) + (12,4)
        assert_eq!(span.map(0, 8, 4, 8).unwrap(), vec![(4, 4), (12, 4)]);
        // 跨行中点:目标 [2, 6) → 源 (6,2) + (12,4)
        assert_eq!(span.map(2, 6, 4, 8).unwrap(), vec![(6, 2), (12, 4)]);
    }

    /// 分段切(融合权重):q_proj [12288, 5120],两段各 6144,world=2
    #[test]
    fn segmented_span_covers_each_segment() {
        let shape = [12288usize, 5120];
        let segs = [6144usize, 6144];
        let (span, out_shape) = plan_span(WeightRule::SegmentedCol(SegKind::QGate), &shape, plan(1, 2), &segs, WantAxis::Logical)
            .unwrap()
            .unwrap();
        assert_eq!(out_shape, vec![6144, 5120], "两段各取半 ⇒ 总行数 = 12288/2");
        // 目标行 0(段 0 的第 0 行)→ 源行 3072;目标行 3072(段 1 的第 0 行)→ 源行 6144+3072
        let m = span.map(0, 4, 5120, 5120).unwrap();
        assert_eq!(m, vec![(3072 * 5120, 4)]);
        let m2 = span.map(3072 * 5120, 4, 5120, 5120).unwrap();
        assert_eq!(m2, vec![(9216 * 5120, 4)], "第二段起点 = 6144 + 3072");
        // 段长合计不符 = 拒
        assert!(plan_span(WeightRule::SegmentedCol(SegKind::QGate), &shape, plan(0, 2), &[6144, 6143], WantAxis::Logical).is_err());
    }

    /// 一维权重(PerHead:A_log [16])按行切
    #[test]
    fn per_head_1d_is_rows() {
        let (span, shape) = plan_span(WeightRule::PerHead, &[16], plan(1, 2), &[], WantAxis::Logical).unwrap().unwrap();
        assert_eq!(shape, vec![8]);
        // 行宽 = 1(一维)
        assert_eq!(span.map(0, 8, 1, 1).unwrap(), vec![(8, 8)]);
    }

    /// world=1 / Replicate ⇒ 不切(None)
    #[test]
    fn single_and_replicate_are_none() {
        assert!(plan_span(WeightRule::UniformCol, &[8, 6], TpPlan::single(), &[], WantAxis::Logical).unwrap().is_none());
        assert!(plan_span(WeightRule::Replicate, &[8, 6], plan(1, 2), &[], WantAxis::Logical).unwrap().is_none());
    }

    /// 分片 rows 与"全量后切"等价(纯算术自洽:两 rank 映射的行集拼接 = 全域)
    #[test]
    fn spans_tile_the_source_rows() {
        let shape = [8usize, 6];
        let mut got: Vec<usize> = Vec::new();
        for r in 0..2 {
            let (span, _) = plan_span(WeightRule::UniformCol, &shape, plan(r, 2), &[], WantAxis::Logical).unwrap().unwrap();
            // 目标 4 行 × 行宽 6 → 源读序列展开成源行号
            for (src_off, len) in span.map(0, 4 * 6, 6, 6).unwrap() {
                for i in 0..len {
                    got.push((src_off + i) / 6);
                }
            }
        }
        got.sort_unstable();
        got.dedup();
        assert_eq!(got, (0..8).collect::<Vec<_>>(), "两片行集拼接 = 全量行");
    }
}
