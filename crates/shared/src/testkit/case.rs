//! `#[owl_test]` 用例配置注入(过程宏 `owl-test-macros` 的运行期伴生)。
//!
//! **隐藏环境变量**(设计:`docs/arch/owl-test-macro.md` §2.2):每用例
//! 一个键 `OWL_TEST_CONFIG_<用例函数名大写>` = 该用例专属配置文件路径
//! (TOML,引擎同款 `OwlConfig` 格式)。未设 ⇒ `OwlConfig::default()`
//! (零配置可跑);设了 ⇒ 走**唯一 loader**(deny_unknown/枚举严格,
//! 测试配置错键 = panic,与引擎拒启同语义)。
//!
//! 与 `OWL_E2E_*`/`OWL_GATE_*` 同族:账外键(env 不进正式配置链,
//! 测试域豁免),登记见 `config.rs` 账外键清单。

use crate::config::OwlConfig;

/// 隐藏键:`OWL_TEST_CONFIG_<NAME>`(用例 fn 名 → 大写)
fn case_key(name: &str) -> String {
    format!("OWL_TEST_CONFIG_{}", name.to_uppercase())
}

/// **用例配置注入**(宏 wrapper 唯一调用点):隐藏 env → 唯一 loader →
/// [`OwlConfig`];`runtime.test_device` 无条件回填 `OWL_TEST_DEVICE`
/// (serde(skip) 不入 toml,设备序只能来自 env —— GPU 门控据此判跳)。
///
/// # Panics
/// - env 指向的文件缺失/解析失败/含账外键 ⇒ **响亮 panic**(带键名与
///   路径;测试配置错误不静默降级)。
pub fn case_config(name: &str) -> OwlConfig {
    let key = case_key(name);
    let mut cfg = match crate::env_reader::str(&key) {
        Some(path) => {
            let p = std::path::Path::new(&path);
            OwlConfig::load(None, Some(p)).unwrap_or_else(|e| {
                panic!("[owl_test] 用例 {name} 的配置文件 {path} 加载失败: {e}")
            })
        }
        None => OwlConfig::default(),
    };
    // 设备序回填(隐藏 env 的另一半;GPU 门控判据)
    cfg.runtime.test_device = crate::env_reader::parse::<usize>("OWL_TEST_DEVICE");
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// std::env 进程全局 —— 触碰本族键的用例经此互斥(env_reader 同款)
    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn 键名派生_大写() {
        assert_eq!(case_key("cpu_case"), "OWL_TEST_CONFIG_CPU_CASE");
        assert_eq!(
            case_key("gpu_capture_matches_eager"),
            "OWL_TEST_CONFIG_GPU_CAPTURE_MATCHES_EAGER"
        );
    }

    /// 双臂实证(文档 §四验收 2):env 未设 ⇒ default;设 ⇒ 文件值真的到手
    /// 且设备序回填生效。
    #[test]
    fn 注入双臂_文件值与缺省() {
        let _g = env_lock();
        // 臂 1:文件(临时 toml;引擎同款格式,deny_unknown 生效面)
        let path = std::env::temp_dir().join(format!("owl_case_cfg_{}.toml", std::process::id()));
        std::fs::write(&path, "[runtime]\nmax_seq = 128\nprefill_chunk = 32\n")
            .expect("写临时配置");
        let _guard = crate::env_reader::EnvGuard::set("OWL_TEST_CONFIG_INJECT_ARM", path.to_str().unwrap());
        let cfg = case_config("inject_arm");
        assert_eq!(cfg.runtime.max_seq, 128, "文件值注入");
        assert_eq!(cfg.runtime.prefill_chunk, 32, "文件值注入②");
        assert_eq!(cfg, OwlConfig::load(None, Some(&path)).expect("同 loader 复算"), "与唯一 loader 逐字段同");
        drop(_guard);
        std::fs::remove_file(&path).ok();

        // 臂 2:未设 ⇒ default(另一用例名,键不存在)
        let cfg2 = case_config("definitely_unset_case_xyz");
        assert_eq!(cfg2, OwlConfig::default(), "未设 = 引擎缺省配置");

        // 臂 3:设备序回填(隐藏 env 下半场;GPU 门控判据)
        let _d = crate::env_reader::EnvGuard::set("OWL_TEST_DEVICE", "2");
        let cfg3 = case_config("inject_arm_dev");
        assert_eq!(cfg3.runtime.test_device, Some(2), "OWL_TEST_DEVICE 回填");
    }

    /// 错误配置响亮 panic(不静默降级;§四验收 2 的失败臂)
    #[test]
    fn 坏配置_响亮拒绝() {
        let _g = env_lock();
        let path = std::env::temp_dir().join(format!("owl_case_cfg_bad_{}.toml", std::process::id()));
        std::fs::write(&path, "[runtime]\n不存在的键 = 1\n").expect("写临时配置");
        let _guard = crate::env_reader::EnvGuard::set("OWL_TEST_CONFIG_BAD_ARM", path.to_str().unwrap());
        let r = std::panic::catch_unwind(|| case_config("bad_arm"));
        assert!(r.is_err(), "账外键必须 panic,不许静默取缺省");
        drop(_guard);
        std::fs::remove_file(&path).ok();
    }
}
