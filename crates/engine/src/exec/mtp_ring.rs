//! MTP 草稿 KV 的**物理环**（T5，2026-11-04）。
//!
//! Windowed-MTP（[`super::mtp_window`]）只压缩**读视图**：attention 只读
//! `sink + 最近 W`，但物理页仍按整条上下文线性分配。本模块把同一个读集
//! 映射到固定大小的物理环，**不改变读集**，只回收永远不会再被读的旧槽。
//!
//! 无精度损失的两个容量条件：
//!
//! 1. `capacity >= min_ctx`：短 ctx（窗口未生效）仍读完整历史，环在
//!    `pos < min_ctx` 内不得回卷；
//! 2. `capacity >= sink_eff + window + depth + page`：窗口生效后，
//!    为第 p 轮草稿保住的读集是 `sink + [p+1-W, p]`；环至少多出
//!    `depth + page` 个槽，保证本轮/后续草稿写入（`p+1..p+depth`）
//!    不会覆盖本轮仍要读的槽。
//!
//! `sink_eff` 页对齐并单独钉住；环形区只覆盖 `pos >= sink_eff`。
//! 逻辑位置、RoPE 位置、`visible` 长度仍是**绝对口径**；只有物理页号
//! 变成 `slot_of(pos) / page`。
//!
//! 本模块先落纯映射 + 无窃读证明；池分配/写核/读表接线在下一阶段。

use owl_iface::contract::ModelError;

/// MTP 环形草稿 KV 的纯映射计划。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MtpRingPlan {
    page: usize,
    /// 页对齐后的 sink token 数；这些位置固定，不参与环形覆盖。
    sink_tokens: usize,
    /// 环形区 token 数（`capacity - sink_tokens`，页对齐）。
    ring_tokens: usize,
    /// 总物理容量 token 数（页对齐）。
    capacity_tokens: usize,
    /// 窗口未生效的绝对 ctx 上限；用于短 ctx 的 full-view 口径。
    min_ctx: usize,
    /// 草稿块/深度余量（至少 1）。
    depth: usize,
}

impl MtpRingPlan {
    /// 构造计划。
    ///
    /// `page` = 池页宽（通常 32）；`cfg` = MTP window 配置；
    /// `depth` = 草稿最大深度/块行数；`min_ctx` = 窗口生效阈值。
    pub(crate) fn new(
        page: usize,
        cfg: super::mtp_window::MtpWindowCfg,
        depth: usize,
        min_ctx: usize,
    ) -> Result<Self, ModelError> {
        if page == 0 {
            return Err(ModelError::EngineInvariant {
                what: "MTP ring: page = 0（池几何非法）".into(),
            });
        }
        if !cfg.active() {
            return Err(ModelError::EngineInvariant {
                what: "MTP ring: 需要 mtp_window > 0（ring 只服务窗口读视图）".into(),
            });
        }
        if cfg.sink >= cfg.window {
            return Err(ModelError::EngineInvariant {
                what: format!(
                    "MTP ring: sink={} 必须小于 window={}",
                    cfg.sink, cfg.window
                ),
            });
        }
        let sink_tokens = align_up(cfg.sink, page)?;
        // 窗口生效后读集 = sink + 最近 window；再留 depth（本轮草稿写）
        // 与 page（读视图按页向下取整的余量）。最后再页对齐。
        let need = min_ctx
            .max(sink_tokens)
            .max(
                sink_tokens
                    .checked_add(cfg.window)
                    .and_then(|v| v.checked_add(depth.max(1)))
                    .and_then(|v| v.checked_add(page))
                    .ok_or_else(|| ModelError::EngineInvariant {
                        what: "MTP ring: 容量计算溢出".into(),
                    })?,
            );
        let capacity_tokens = align_up(need, page)?;
        let ring_tokens = capacity_tokens.checked_sub(sink_tokens).ok_or_else(|| {
            ModelError::EngineInvariant {
                what: format!(
                    "MTP ring: capacity={} < sink={}",
                    capacity_tokens, sink_tokens
                ),
            }
        })?;
        if ring_tokens == 0 || ring_tokens % page != 0 {
            return Err(ModelError::EngineInvariant {
                what: format!(
                    "MTP ring: ring_tokens={} 非页对齐/为零（page={}）",
                    ring_tokens, page
                ),
            });
        }
        Ok(Self {
            page,
            sink_tokens,
            ring_tokens,
            capacity_tokens,
            min_ctx,
            depth: depth.max(1),
        })
    }

    pub(crate) fn sink_tokens(&self) -> usize {
        self.sink_tokens
    }
    pub(crate) fn ring_tokens(&self) -> usize {
        self.ring_tokens
    }
    pub(crate) fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }
    pub(crate) fn capacity_pages(&self) -> usize {
        self.capacity_tokens / self.page
    }
    pub(crate) fn min_ctx(&self) -> usize {
        self.min_ctx
    }
    pub(crate) fn depth(&self) -> usize {
        self.depth
    }

    /// 绝对 token 位置 → 物理 slot。
    ///
    /// `pos < sink_tokens` 固定在 sink 区；其余在环形区按
    /// `(pos - sink) % ring_tokens` 回卷。
    pub(crate) fn slot_of(&self, pos: usize) -> usize {
        if pos < self.sink_tokens {
            pos
        } else {
            self.sink_tokens + (pos - self.sink_tokens) % self.ring_tokens
        }
    }

    /// 绝对 token 位置 → 物理页号。
    ///
    /// 因为 sink_tokens 与 ring_tokens 都是 page 的倍数，页内偏移
    /// `pos % page` 在映射后保持不变。
    pub(crate) fn page_of(&self, pos: usize) -> usize {
        self.slot_of(pos) / self.page
    }

    /// 逻辑页表（长度 = `pages`，值 = 物理页号）。
    ///
    /// 给现有 `MtpWindowPlan` 当 `full_bt` 用：窗口视图仍按逻辑页选
    /// sink/recent，选出来的值已经是物理环页。`pages` 可以小于
    /// `capacity_pages`（短 ctx），也可以大于（超过容量时值按环回卷）。
    pub(crate) fn table(&self, pages: usize) -> Vec<u32> {
        (0..pages)
            .map(|q| self.page_of(q * self.page) as u32)
            .collect()
    }
}

fn align_up(v: usize, align: usize) -> Result<usize, ModelError> {
    if align == 0 {
        return Err(ModelError::EngineInvariant {
            what: "MTP ring: align = 0".into(),
        });
    }
    v.checked_add(align - 1)
        .map(|x| x / align * align)
        .ok_or_else(|| ModelError::EngineInvariant {
            what: format!("MTP ring: align_up({v}, {align}) 溢出"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::mtp_window::MtpWindowCfg;
    use std::collections::BTreeSet;

    fn cfg() -> MtpWindowCfg {
        MtpWindowCfg {
            window: 4032,
            sink: 64,
            min_ctx: 12288,
        }
    }

    #[test]
    fn capacity_covers_min_ctx_and_window() {
        let p = MtpRingPlan::new(32, cfg(), 7, 12288).unwrap();
        assert_eq!(p.sink_tokens(), 64);
        assert_eq!(p.capacity_tokens(), 12288);
        assert_eq!(p.ring_tokens(), 12224);
        assert_eq!(p.capacity_pages(), 384);
    }

    #[test]
    fn short_ctx_full_region_has_no_alias() {
        let p = MtpRingPlan::new(32, cfg(), 7, 12288).unwrap();
        let mut seen = BTreeSet::new();
        for pos in 0..p.min_ctx() {
            assert!(seen.insert(p.slot_of(pos)), "短 ctx slot 重复: pos={pos}");
        }
        assert_eq!(seen.len(), p.min_ctx());
    }

    #[test]
    fn sink_is_never_overwritten_by_ring() {
        let p = MtpRingPlan::new(32, cfg(), 7, 12288).unwrap();
        for pos in 0..p.sink_tokens() {
            assert_eq!(p.slot_of(pos), pos);
        }
        for pos in p.sink_tokens()..p.sink_tokens() + 4 * p.ring_tokens() {
            assert!(
                p.slot_of(pos) >= p.sink_tokens(),
                "环映射侵入 sink: pos={pos}"
            );
        }
    }

    #[test]
    fn window_read_set_survives_current_and_next_draft_writes() {
        let p = MtpRingPlan::new(32, cfg(), 7, 12288).unwrap();
        // 窗口刚生效、以及几个长 ctx 位置
        for pos in [p.min_ctx(), p.min_ctx() + 1, 16_000, 57_343, 65_535] {
            let tokens = pos + 1;
            let recent_start = tokens
                .saturating_sub(cfg().window)
                .max(p.sink_tokens());
            let aligned_start = recent_start / p.page * p.page;
            let mut read = BTreeSet::new();
            // sink 固定页 + 最近窗口（与 MtpWindowPlan 的读集同口径）
            for q in 0..p.sink_tokens() {
                assert!(read.insert(p.slot_of(q)), "sink 读集重复 pos={pos}");
            }
            for q in aligned_start..=pos {
                assert!(
                    read.insert(p.slot_of(q)),
                    "窗口读集重复 pos={pos} q={q}"
                );
            }
            // 本轮/后续草稿写入位置不能落在本轮读集里
            for w in (pos + 1)..=(pos + p.depth()) {
                assert!(
                    !read.contains(&p.slot_of(w)),
                    "草稿写窃读 pos={pos} write={w} slot={}",
                    p.slot_of(w)
                );
            }
        }
    }

    #[test]
    fn page_table_preserves_page_offset_and_ring_pages() {
        let p = MtpRingPlan::new(32, cfg(), 7, 12288).unwrap();
        let pages = 65_536 / p.page + 4;
        let table = p.table(pages);
        assert_eq!(table.len(), pages);
        for q in 0..p.capacity_pages() {
            assert!(table[q] < p.capacity_pages() as u32);
        }
        for pos in [0usize, 63, 64, 95, 12288, 12289, 65535] {
            let slot = p.slot_of(pos);
            let page = p.page_of(pos);
            assert_eq!(page, table[pos / p.page] as usize);
            assert_eq!(slot, page * p.page + pos % p.page);
        }
    }

    #[test]
    fn capacity_formula_keeps_depth_slack() {
        let p = MtpRingPlan::new(32, cfg(), 7, 12288).unwrap();
        assert!(p.capacity_tokens() >= p.sink_tokens() + 4032 + 7 + 32);
        let p0 = MtpRingPlan::new(32, MtpWindowCfg { window: 8, sink: 4, min_ctx: 0 }, 8, 0).unwrap();
        assert!(p0.capacity_tokens() >= 4 + 8 + 8 + 32);
    }
}
