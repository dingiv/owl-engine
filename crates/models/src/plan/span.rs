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

use crate::plan::rules::{SegKind, WeightRule};
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
                SourceSpan::Cols { .. } => Ok(r),
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
        let col_start = match self {
            SourceSpan::Cols { col_start, cols } => {
                let _ = cols;
                *col_start
            }
            _ => 0,
        };

        let mut out: Vec<(usize, usize)> = Vec::new();
        let mut o = off;
        let mut remaining = len;
        while remaining > 0 {
            let r = o / tgt_row_width;
            let c = o % tgt_row_width;
            let take = (tgt_row_width - c).min(remaining);
            let src_row = row_of(r)?;
            let src_off = src_row * src_row_width + col_start + c;
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

/// **计划器**:`(规则, 全量 shape, plan, 段长) → (读计划, 分片 shape)`。
///
/// - 返回 `None` = 不切(`Replicate`;调用方按全量直通);
/// - `segs`:分段切的段长(源坐标系;由调用方按 `SegKind::segments(spec)` 展开);
/// - 非分段形态忽略 `segs`。
pub fn plan_span(
    rule: WeightRule,
    shape: &[usize],
    plan: TpPlan,
    segs: &[usize],
) -> Result<Option<(SourceSpan, Vec<usize>)>, String> {
    if plan.is_single() || rule == WeightRule::Replicate {
        return Ok(None);
    }
    if shape.is_empty() {
        return Err("plan_span: 空 shape".into());
    }
    let mut out_shape = shape.to_vec();
    match rule {
        WeightRule::Replicate => Ok(None),
        WeightRule::UniformCol | WeightRule::PerHead => {
            let r = plan.shard_range(shape[0])?;
            let span = SourceSpan::Rows { row_start: r.start, rows: r.len() };
            out_shape[0] = r.len();
            Ok(Some((span, out_shape)))
        }
        WeightRule::UniformRow => {
            if shape.len() < 2 {
                return Err(format!(
                    "plan_span: UniformRow 需要 ≥2 维(得 {shape:?};一维权重请用 PerHead)"
                ));
            }
            let r = plan.shard_range(shape[1])?;
            let span = SourceSpan::Cols { col_start: r.start, cols: r.len() };
            out_shape[1] = r.len();
            Ok(Some((span, out_shape)))
        }
        WeightRule::SegmentedCol(kind) => {
            let _ = kind;
            if segs.is_empty() {
                return Err("plan_span: 分段切的段长为空(调用方须按 SegKind 展开)".into());
            }
            let total: usize = segs.iter().sum();
            if total != shape[0] {
                return Err(format!(
                    "plan_span: 段长合计 {total} ≠ out 维 {}(shape {shape:?})",
                    shape[0]
                ));
            }
            let mut rows_span = Vec::with_capacity(segs.len());
            let mut off = 0usize;
            let mut shard_rows = 0usize;
            for &seg in segs {
                let r = plan.shard_range(seg)?;
                rows_span.push((off, r.start, r.len()));
                off += seg;
                shard_rows += r.len();
            }
            let span = SourceSpan::RowsSeg { segs: rows_span };
            out_shape[0] = shard_rows;
            Ok(Some((span, out_shape)))
        }
    }
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

    /// 行切(out):连续区间 + 分片 shape;跨行合并成一段
    #[test]
    fn rows_span_is_contiguous() {
        // 源 [8, 6](out=8, in=6);world=2 ⇒ rank1 取行 4..8
        let (span, shape) = plan_span(WeightRule::UniformCol, &[8, 6], plan(1, 2), &[])
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
        let (span, shape) = plan_span(WeightRule::UniformRow, &[4, 8], plan(1, 2), &[])
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
        let (span, out_shape) = plan_span(WeightRule::SegmentedCol(SegKind::QGate), &shape, plan(1, 2), &segs)
            .unwrap()
            .unwrap();
        assert_eq!(out_shape, vec![6144, 5120], "两段各取半 ⇒ 总行数 = 12288/2");
        // 目标行 0(段 0 的第 0 行)→ 源行 3072;目标行 3072(段 1 的第 0 行)→ 源行 6144+3072
        let m = span.map(0, 4, 5120, 5120).unwrap();
        assert_eq!(m, vec![(3072 * 5120, 4)]);
        let m2 = span.map(3072 * 5120, 4, 5120, 5120).unwrap();
        assert_eq!(m2, vec![(9216 * 5120, 4)], "第二段起点 = 6144 + 3072");
        // 段长合计不符 = 拒
        assert!(plan_span(WeightRule::SegmentedCol(SegKind::QGate), &shape, plan(0, 2), &[6144, 6143]).is_err());
    }

    /// 一维权重(PerHead:A_log [16])按行切
    #[test]
    fn per_head_1d_is_rows() {
        let (span, shape) = plan_span(WeightRule::PerHead, &[16], plan(1, 2), &[]).unwrap().unwrap();
        assert_eq!(shape, vec![8]);
        // 行宽 = 1(一维)
        assert_eq!(span.map(0, 8, 1, 1).unwrap(), vec![(8, 8)]);
    }

    /// world=1 / Replicate ⇒ 不切(None)
    #[test]
    fn single_and_replicate_are_none() {
        assert!(plan_span(WeightRule::UniformCol, &[8, 6], TpPlan::single(), &[]).unwrap().is_none());
        assert!(plan_span(WeightRule::Replicate, &[8, 6], plan(1, 2), &[]).unwrap().is_none());
    }

    /// 分片 rows 与"全量后切"等价(纯算术自洽:两 rank 映射的行集拼接 = 全域)
    #[test]
    fn spans_tile_the_source_rows() {
        let shape = [8usize, 6];
        let mut got: Vec<usize> = Vec::new();
        for r in 0..2 {
            let (span, _) = plan_span(WeightRule::UniformCol, &shape, plan(r, 2), &[]).unwrap().unwrap();
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
