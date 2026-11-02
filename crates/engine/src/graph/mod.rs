//! **图治理域**:`GraphPlan`(图三态编排面)+ `graph_fleet`(全部图实例的族账)。
//!
//! 进这里的东西 = "图的形状、捕获三态、图族账";**执行**归 `exec/`,
//! **决策**归 `sched/`。执行路径的接缝纪律见
//! `docs/devlog/graph/graph-model-seam.md`。

pub(super) mod graph_fleet;
pub mod graph_plan;
