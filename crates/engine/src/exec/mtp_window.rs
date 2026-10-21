//! MTP 草稿注意力的读视图（Windowed-MTP，2026-10-21）。
//!
//! 只做**页表视图**：把完整 MTP 链的 `full_bt` 压成
//! `[最老 sink 页] ‖ [最近 window 页]`，并给出 v2 注意力要的
//! `visible`（压缩后的逻辑长度）。不碰写入、不改池、不重新旋转 RoPE。
//!
//! 语义见 `docs/arch/mtp-长上下文抗衰减-设计与施工.md` §4.2；这里保持
//! 纯函数 + 显式错误，热路径调用方负责把 `blocks` / `visible` 装进
//! `KvBuffers` 或图输入槽。

use owl_iface::contract::ModelError;

/// MTP 窗口配置（token 语义；页粒度在 `MtpWindowPlan` 内向上取整）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MtpWindowCfg {
    /// 最近窗口 token 数；0 = 关闭（full-context）。
    pub window: usize,
    /// 最老前缀 token 数；仅 `window > 0` 生效。
    pub sink: usize,
    /// 窗口生效的最小绝对上下文（token）。0 = 从 `sink+window` 之外立即生效。
    /// 短 ctx 下草稿尾部仍需要完整历史；这是 8k d7 回归的门。
    pub min_ctx: usize,
}

impl MtpWindowCfg {
    pub(crate) fn active(self) -> bool {
        self.window > 0
    }
}

/// 单行的压缩页表视图。
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MtpRowView {
    /// 物理块号，sink 页在前、recent 页在后。
    pub blocks: Vec<u32>,
    /// v2 的每行可见长度（压缩后的逻辑长度，不是绝对 pos）。
    pub visible: u32,
}

/// 完整 MTP 链 → 每行读视图。
#[derive(Debug)]
pub(crate) struct MtpWindowPlan<'a> {
    full_bt: &'a [u32],
    page: usize,
    cfg: MtpWindowCfg,
    sink_eff: usize,
}

impl<'a> MtpWindowPlan<'a> {
    pub(crate) fn new(
        full_bt: &'a [u32],
        page: usize,
        cfg: MtpWindowCfg,
    ) -> Result<Self, ModelError> {
        if page == 0 {
            return Err(ModelError::EngineInvariant {
                what: "MTP window: page = 0（池几何非法）".into(),
            });
        }
        if cfg.active() && cfg.sink >= cfg.window {
            return Err(ModelError::EngineInvariant {
                what: format!(
                    "MTP window: sink={} 必须小于 window={}",
                    cfg.sink, cfg.window
                ),
            });
        }
        let sink_eff = if cfg.active() {
            cfg.sink
                .div_ceil(page)
                .checked_mul(page)
                .ok_or_else(|| ModelError::EngineInvariant {
                    what: format!("MTP window: sink={} 页对齐溢出", cfg.sink),
                })?
        } else {
            0
        };
        Ok(Self { full_bt, page, cfg, sink_eff })
    }

    /// 构造 `[base, base+n)` 的行视图。
    pub(crate) fn rows(&self, base: usize, n: usize) -> Result<Vec<MtpRowView>, ModelError> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = base.checked_add(i).ok_or_else(|| ModelError::EngineInvariant {
                what: format!("MTP window: row 溢出 base={base} i={i}"),
            })?;
            out.push(self.row(row)?);
        }
        Ok(out)
    }

    /// 单行：`row` 是当前 query 的绝对位置（0-based，已提交 token 的位置）。
    pub(crate) fn row(&self, row: usize) -> Result<MtpRowView, ModelError> {
        let tokens = row
            .checked_add(1)
            .ok_or_else(|| ModelError::EngineInvariant {
                what: format!("MTP window: tokens 溢出 row={row}"),
            })?;
        let cur_page = row / self.page;
        if self.full_bt.get(cur_page).is_none() {
            return Err(ModelError::EngineInvariant {
                what: format!(
                    "MTP window: row={row} 需要页 {cur_page}，但 MTP 链表只有 {} 页",
                    self.full_bt.len()
                ),
            });
        }

        // window=0，或上下文短到 sink+window 已经覆盖全部：保留完整表，
        // 与旧路径逐字节同物（包括链表尾部尚未读到的页）。
        if !self.cfg.active()
            || tokens <= self.cfg.min_ctx
            || tokens <= self.sink_eff + self.cfg.window
        {
            return self.full_view(tokens);
        }

        let sink_pages = self.sink_eff / self.page;
        let recent_start = tokens.saturating_sub(self.cfg.window).max(self.sink_eff);
        let recent_page = recent_start / self.page;
        let aligned_start = recent_page * self.page;

        // 最近窗口与 sink 页在同一页内（或更早）：保留完整表。
        if aligned_start <= self.sink_eff {
            return self.full_view(tokens);
        }

        let mut blocks = Vec::with_capacity(sink_pages + (cur_page - recent_page + 1));
        blocks.extend_from_slice(&self.full_bt[..sink_pages]);
        blocks.extend_from_slice(&self.full_bt[recent_page..=cur_page]);
        let visible = self
            .sink_eff
            .checked_add(tokens - aligned_start)
            .ok_or_else(|| ModelError::EngineInvariant {
                what: format!("MTP window: visible 溢出 row={row}"),
            })?;
        let visible = u32::try_from(visible).map_err(|_| ModelError::EngineInvariant {
            what: format!("MTP window: visible={visible} 超 u32（row={row}）"),
        })?;
        Ok(MtpRowView { blocks, visible })
    }

    fn full_view(&self, tokens: usize) -> Result<MtpRowView, ModelError> {
        let visible = u32::try_from(tokens).map_err(|_| ModelError::EngineInvariant {
            what: format!("MTP window: full visible={tokens} 超 u32"),
        })?;
        Ok(MtpRowView {
            blocks: self.full_bt.to_vec(),
            visible,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bt(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| 100 + i).collect()
    }

    #[test]
    fn window_off_is_full_even_for_long_ctx() {
        let full = bt(16);
        let plan = MtpWindowPlan::new(
            &full,
            4,
            MtpWindowCfg { window: 0, sink: 64, min_ctx: 0 },
        )
        .unwrap();
        let v = plan.row(39).unwrap();
        assert_eq!(v.blocks, full);
        assert_eq!(v.visible, 40);
    }

    #[test]
    fn short_ctx_is_full() {
        let full = bt(16);
        let plan = MtpWindowPlan::new(
            &full,
            4,
            MtpWindowCfg { window: 8, sink: 4, min_ctx: 0 },
        )
        .unwrap();
        // tokens=6 <= sink_eff(4)+window(8) => full
        let v = plan.row(5).unwrap();
        assert_eq!(v.blocks, full);
        assert_eq!(v.visible, 6);
    }

    #[test]
    fn sink_plus_recent_is_compressed() {
        let full = bt(16);
        let plan = MtpWindowPlan::new(
            &full,
            4,
            MtpWindowCfg { window: 8, sink: 4, min_ctx: 0 },
        )
        .unwrap();
        // row=39 => tokens=40; sink_eff=4; recent_start=32; aligned=32;
        // cur_page=9; blocks=[100] + [108,109]; visible=4+8=12
        let v = plan.row(39).unwrap();
        assert_eq!(v.blocks, vec![100, 108, 109]);
        assert_eq!(v.visible, 12);
    }

    #[test]
    fn sink_zero_is_recent_only() {
        let full = bt(16);
        let plan = MtpWindowPlan::new(
            &full,
            4,
            MtpWindowCfg { window: 8, sink: 0, min_ctx: 0 },
        )
        .unwrap();
        let v = plan.row(39).unwrap();
        assert_eq!(v.blocks, vec![108, 109]);
        assert_eq!(v.visible, 8);
    }

    #[test]
    fn sink_is_page_rounded_up() {
        let full = bt(16);
        let plan = MtpWindowPlan::new(
            &full,
            4,
            MtpWindowCfg { window: 12, sink: 5, min_ctx: 0 },
        )
        .unwrap();
        // sink_eff=8; row=39 => tokens=40; recent_start=28 => aligned=28;
        // sink_pages=2; recent_page=7; blocks=[100,101,107,108,109];
        // visible=8+(40-28)=20
        let v = plan.row(39).unwrap();
        assert_eq!(v.blocks, vec![100, 101, 107, 108, 109]);
        assert_eq!(v.visible, 20);
    }

    #[test]
    fn recent_and_sink_overlap_falls_back_full() {
        let full = bt(16);
        let plan = MtpWindowPlan::new(
            &full,
            4,
            MtpWindowCfg { window: 8, sink: 4, min_ctx: 0 },
        )
        .unwrap();
        // tokens=12 => sink_eff+window=12 => full branch
        let v = plan.row(11).unwrap();
        assert_eq!(v.blocks, full);
        assert_eq!(v.visible, 12);
    }

    #[test]
    fn missing_page_is_explicit_error() {
        let full = bt(2);
        let plan = MtpWindowPlan::new(
            &full,
            4,
            MtpWindowCfg { window: 8, sink: 4, min_ctx: 0 },
        )
        .unwrap();
        let e = plan.row(20).unwrap_err();
        assert!(format!("{e}").contains("只有 2 页"), "{e}");
    }

    #[test]
    fn min_ctx_keeps_full() {
        let full = bt(16);
        let plan = MtpWindowPlan::new(
            &full,
            4,
            MtpWindowCfg { window: 8, sink: 4, min_ctx: 20 },
        )
        .unwrap();
        // tokens=20 <= min_ctx => full；tokens=21 才压缩
        assert_eq!(plan.row(19).unwrap().blocks, full);
        assert_eq!(plan.row(20).unwrap().visible, 13);
        assert!(plan.row(20).unwrap().blocks.len() < full.len());
    }

    #[test]
    fn invalid_window_is_refused() {
        let full = bt(2);
        let e = MtpWindowPlan::new(
            &full,
            4,
            MtpWindowCfg { window: 4, sink: 4, min_ctx: 0 },
        )
        .unwrap_err();
        assert!(format!("{e}").contains("sink=4"), "{e}");
    }

    #[test]
    fn rows_preserve_order_and_length() {
        let full = bt(16);
        let plan = MtpWindowPlan::new(
            &full,
            4,
            MtpWindowCfg { window: 8, sink: 4, min_ctx: 0 },
        )
        .unwrap();
        let v = plan.rows(36, 4).unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(v[0].visible, 13);
        assert_eq!(v[3].visible, 12);
    }
}
