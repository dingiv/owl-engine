//! 算子对拍测试框架(设计见 crates/shared/README.md)。

pub mod checks;
pub mod rng;

// 归档(.rsx:旧治理世界 CudaDevice/CudaPool 的对拍框架,随声明式重构退役;
// 新世界等价物 = GpuClient 直接作测试执行器,待 A4 落地):
// - case.rsx:对拍用例(Rig 注入式)
// - rig.rsx:进程级测试装置
pub use checks::{allclose, assert_finite, expect_shape, DiffReport};
pub use rng::Rng;
