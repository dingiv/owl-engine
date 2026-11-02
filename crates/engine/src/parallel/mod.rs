//! **并行域**(TP):world = 2 的双 rank 装配面(staging 布局 / 双 actor 形态 /
//! 装配期架构与对称性门)。通信核在 `owl-cuda`(`crates/backends/cuda/src/comm/`)。
//!
//! 设计见 `docs/devlog/comm/tp2-design.md`;全账见 `comm/tp2-log.md`。

pub(super) mod tp;
