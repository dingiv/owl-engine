//! 算子对拍测试框架(设计见 crates/shared/README.md)。

pub mod case;
pub mod checks;
pub mod rig;
pub mod rng;

pub use case::{Bound, Case, Gen, GenU32};
pub use checks::{allclose, assert_finite, expect_shape, DiffReport};
pub use rig::{DevBuf, Rig};
pub use rng::Rng;
