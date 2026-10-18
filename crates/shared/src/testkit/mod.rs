//! 算子对拍测试框架(设计见 crates/shared/README.md)。
//!
//! - `case` = `#[owl_test]` 用例配置注入(唯一 env `OWL_TEST_CONF` + 约定 overlay;
//!   设计 `docs/arch/owl-test-macro.md`);
//! - `res` = **测试资源清单**(`testdata/resources.toml`;取代 `OWL_*_DIR` 一族 env)。

pub mod case;
pub mod checks;
pub mod res;
pub mod rng;

pub use case::{case_config, TEST_CONF_ENV};
pub use res::{missing_reason, resource, resource_flag, resource_ok, resource_or};

/// **HF 对拍门**(`OWL_HF_PARITY=1`;`#[owl_test(hf_parity)]` 的判据 —— 与
/// `models::testkit::parity_enabled` 同语义)
pub fn hf_parity_enabled() -> bool {
    crate::env_reader::str("OWL_HF_PARITY")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// HF 对拍门未开的 skip 行(与 `models::testkit::skip_note` **逐字同** ——
/// 旧文案 grep 兼容;宏 wrapper 直接打印此常量)
pub const HF_PARITY_SKIP: &str =
    "skip: OWL_HF_PARITY=1 未开(首次会同步 torch CPU 轮)";

// 归档(.rsx:旧治理世界 CudaDevice/CudaPool 的对拍框架,随声明式重构退役;
// 新世界等价物 = GpuClient 直接作测试执行器,待 A4 落地):
// - case.rsx:对拍用例(Rig 注入式)
// - rig.rsx:进程级测试装置
pub use checks::{allclose, assert_finite, expect_shape, DiffReport};
pub use rng::Rng;
