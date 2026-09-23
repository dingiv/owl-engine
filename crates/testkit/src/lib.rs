//! owl-testkit:单算子对拍测试框架(设计见 README.md)。
//!
//! 三件套:
//! - [`rng`]:seed 确定性随机(splitmix64;同 seed 同序列,跨机可复现);
//! - [`rig`]:进程级设备/池 OnceLock + htod/dtoh/sync 一条龙(自动 synchronize);
//! - [`checks`]:allclose(带 DiffReport)/形状契约/有限性;
//! - [`case`]:声明式对拍用例(输入声明 → 设备闭包 → host 参考 → 自动对比)。
//!
//! 纪律:设备序号读 `OWL_TEST_DEVICE`(回落 `owl_cuda::test_device_ordinal`);
//! 参考实现一律纯 Rust f32 直译公式(注释标出处);框架不感知发射器内部,
//! 只管「指针进 / 指针出 + stream」。

pub mod case;
pub mod checks;
pub mod rig;
pub mod rng;

pub use case::{Case, Gen, GenU32};
pub use checks::{allclose, assert_finite, expect_shape, DiffReport};
pub use case::Bound;
pub use rig::{DevBuf, Rig};
pub use rng::Rng;
