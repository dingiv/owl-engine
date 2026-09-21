//! 图运行时 —— 公理 A1( charter.md §二)的代码落点。
//!
//! 本 crate 的存在理由:图内存在 xinfer 时代是"捕获时的副作用",在这里是
//! "内存规划时的输入"。所有池计算发生前,图预算必须已经登记(A1.1);
//! 所有治理动作(trim/销毁/重捕)必须在合法窗口(A1.2)。

/// 图预算 —— 内存规划器的一等输入,不是事后余量。
///
/// port 语义参考:ninfer `DecodeGraphProfile.graph_allowance_bytes`、
/// vLLM `CudagraphManager` 的 capture sizing。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphAllowance {
    /// decode 全图的实例化后常驻预算(bytes/卡)
    pub decode_bytes: u64,
    /// 投机 verify 图(未来含 piecewise 段)预算
    pub verify_bytes: u64,
}

impl GraphAllowance {
    pub fn total(&self) -> u64 {
        self.decode_bytes.saturating_add(self.verify_bytes)
    }
}

/// 图档位:形状在此固定,动态量一律走 device 张量(A1.5 kernel 契约)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphProfile {
    /// 捕获 batch(该档全量,不足者 pad)
    pub batch: u32,
    /// 允许的本档最长 KV frontier(谓词化上界)
    pub max_frontier: u32,
}

/// 捕获治理状态机 —— A1.2 生命周期律的执行者。
///
/// 合法迁移:
///   Idle → Capturing(净空窗口:无任何存活图)→ Ready(逐档)
///   Ready → Idle(全部销毁;唯一合法 trim 窗口之一)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphPhase {
    /// 无图存活:唯一允许 trim / 池属性变更的窗口
    Idle,
    /// 捕获中:禁止一切治理动作
    Capturing,
    /// 图存活:禁止 trim / empty_cache / 池属性变更(A1.2)
    Live,
}

/// 捕获治理器。持有 phase 与档位表;内存规划器读 allowance,
/// runner 调 capture/replay,所有非法操作在此层被 debug_assert + 记录。
pub struct GraphGovernor {
    phase: GraphPhase,
    allowance: GraphAllowance,
    profiles: Vec<GraphProfile>,
    /// 已实例化图计数(Live 的判据;M1 接 CUDA exec 后成为真实句柄表)
    live_graphs: u32,
}

impl GraphGovernor {
    pub fn new(allowance: GraphAllowance, profiles: Vec<GraphProfile>) -> Self {
        assert!(!profiles.is_empty(), "A1.1: 档位表不得为空");
        Self {
            phase: GraphPhase::Idle,
            allowance,
            profiles,
            live_graphs: 0,
        }
    }

    pub fn allowance(&self) -> &GraphAllowance {
        &self.allowance
    }

    pub fn phase(&self) -> GraphPhase {
        self.phase
    }

    /// 内存规划器入口:池容量必须扣除 `allowance.total()`。
    /// (A1.1 的断言面;真正的池计算在 core/planner。)
    pub fn reserve_from(&self, free_bytes: u64) -> u64 {
        free_bytes.saturating_sub(self.allowance.total())
    }

    /// 进入捕获。仅 Idle 合法(A1.2:图存活期重捕 = 泄露案路径,直接拒绝)。
    pub fn begin_capture(&mut self) -> Result<(), GraphError> {
        match self.phase {
            GraphPhase::Idle => {
                self.phase = GraphPhase::Capturing;
                Ok(())
            }
            _ => Err(GraphError::IllegalTransition {
                from: self.phase,
                op: "begin_capture",
                law: "A1.2: 捕获仅允许净空窗口(无存活图)",
            }),
        }
    }

    /// 结束捕获,登记一档图。仅 Capturing 合法。
    pub fn end_capture(&mut self, profile: GraphProfile) -> Result<(), GraphError> {
        if self.phase != GraphPhase::Capturing {
            return Err(GraphError::IllegalTransition {
                from: self.phase,
                op: "end_capture",
                law: "A1.2",
            });
        }
        debug_assert!(
            self.profiles.contains(&profile),
            "捕获档必须来自规划档位表"
        );
        self.live_graphs += 1;
        self.phase = GraphPhase::Live;
        Ok(())
    }

    /// 捕获预检(A1.4):free 不足则要求收窄档位表或降级。
    /// 返回 Ok(降级后档位表) 或 Err(整体关图回 eager)。
    pub fn preflight(
        &self,
        free_bytes: u64,
        per_profile_bytes: impl Fn(&GraphProfile) -> u64,
    ) -> Result<Vec<GraphProfile>, GraphError> {
        let mut kept = Vec::new();
        let mut budget = free_bytes;
        // 档位从大到小:牺牲小 bs 档保大档(图峰值由最大档主导,A1.3)
        for p in self.profiles.iter().rev() {
            let cost = per_profile_bytes(p);
            if cost <= budget {
                budget -= cost;
                kept.push(p.clone());
            }
        }
        if kept.is_empty() {
            Err(GraphError::DegradedToEager {
                free_bytes,
                needed: self.profiles.iter().map(per_profile_bytes).sum(),
            })
        } else {
            kept.reverse(); // 恢复升序
            Ok(kept)
        }
    }

    /// 销毁全部图。Live → Idle 的唯一出口;返回的窗口内允许 trim(A1.2)。
    pub fn destroy_all(&mut self) -> Result<GraphPhase, GraphError> {
        match self.phase {
            GraphPhase::Live | GraphPhase::Idle => {
                self.live_graphs = 0;
                self.phase = GraphPhase::Idle;
                Ok(self.phase)
            }
            GraphPhase::Capturing => Err(GraphError::IllegalTransition {
                from: self.phase,
                op: "destroy_all",
                law: "A1.2: 捕获中禁止销毁",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gov() -> GraphGovernor {
        GraphGovernor::new(
            GraphAllowance { decode_bytes: 1 << 30, verify_bytes: 1 << 30 },
            vec![
                GraphProfile { batch: 1, max_frontier: 4096 },
                GraphProfile { batch: 4, max_frontier: 4096 },
            ],
        )
    }

    /// A1.1:图预算必须从池容量中前置扣除
    #[test]
    fn allowance_reserved_before_pool() {
        let g = gov();
        assert_eq!(g.reserve_from(10_000_000_000), 10_000_000_000 - 2 * (1 << 30));
    }

    /// A1.2:图存活期重捕 = 泄露案路径,必须被拒
    #[test]
    fn recapture_while_live_rejected() {
        let mut g = gov();
        g.begin_capture().unwrap();
        g.end_capture(GraphProfile { batch: 1, max_frontier: 4096 }).unwrap();
        assert!(matches!(
            g.begin_capture(),
            Err(GraphError::IllegalTransition { law: "A1.2: 捕获仅允许净空窗口(无存活图)", .. })
        ));
    }

    /// A1.2:Live → trim 窗口的唯一出口是 destroy_all
    #[test]
    fn destroy_all_reopens_trim_window() {
        let mut g = gov();
        g.begin_capture().unwrap();
        g.end_capture(GraphProfile { batch: 1, max_frontier: 4096 }).unwrap();
        assert_eq!(g.destroy_all().unwrap(), GraphPhase::Idle);
        g.begin_capture().unwrap(); // 净空窗口重开合法
    }

    /// A1.4:预算不足时档位收窄,全不足则降级 eager(不撞死)
    #[test]
    fn preflight_shrinks_then_degrades() {
        let g = gov();
        let cost = |p: &GraphProfile| (p.batch as u64) * 1_000_000_000;
        // 只够 bs=1 一档 → 收窄到 1 档
        let kept = g.preflight(1_500_000_000, cost).unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].batch, 1);
        // 一档都不够 → 整体回 eager
        assert!(matches!(
            g.preflight(100, cost),
            Err(GraphError::DegradedToEager { .. })
        ));
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    /// 生命周期律违反 —— 这是架构错误,不是运行时故障
    #[error("非法状态迁移 {from:?} → {op}: {law}")]
    IllegalTransition {
        from: GraphPhase,
        op: &'static str,
        law: &'static str,
    },
    /// A1.4 优雅降级:预算不足以支撑任何档位,整体回 eager
    #[error("图预算不足,降级 eager:free {free_bytes} < 最低档需求 {needed}")]
    DegradedToEager { free_bytes: u64, needed: u64 },
}
