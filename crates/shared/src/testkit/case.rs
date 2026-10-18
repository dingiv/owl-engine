//! `#[owl_test]` 用例配置注入(过程宏 `owl-macros` 的运行期伴生)。
//!
//! **单一环境入口 `OWL_TEST_CONF`**(2026-10-18):指向一份"测试环境档"
//! (TOML,引擎同款 `OwlConfig` 格式;可含 `flavor = "<预设名>"` 自带基准)。
//! 它给整轮测试定共同口径 —— 例如 `flavor = "spec-dflash"` + 本机几何 ⇒
//! 各会话/各 agent 测出来的配置基本一致,不再各写一套。
//!
//! **用例 overlay 走约定路径(零 env)**:`testdata/cases/<crate>/<模块路径>/<用例名>.toml`
//! (由宏传入的 `file!()` 推导)。要临时改某个用例的配置 = 丢一个文件,
//! 不留 env 键;不想要了删掉即可。
//!
//! 链(**后者覆盖前者**):`flavor(环境档/用例档各自可带,见 config::FLAVOR_KEY)`
//! → `OWL_TEST_CONF` 档 → 用例 overlay。三层都不在 = `OwlConfig::default()`。
//!
//! 旧键 `OWL_TEST_CONFIG_<用例名大写>` **退役**:设了 = 响亮 panic 并给出
//! 迁移指引(不静默忽略)。

use std::path::PathBuf;

use crate::config::OwlConfig;

/// 唯一测试环境入口键(指向一份测试环境档;可含 `flavor` + 覆盖项)
pub const TEST_CONF_ENV: &str = "OWL_TEST_CONF";
/// 退役的每用例键前缀(设了 = 响亮拒,附迁移指引)
const RETIRED_PER_CASE_PREFIX: &str = "OWL_TEST_CONFIG_";

/// **用例 overlay 约定路径**(纯函数;由 `file!()` + 用例名推导)。
///
/// `file!()` 形如 `crates/<crate>/src/<模块路径>.rs`(相对 workspace 根)⇒
/// `testdata/cases/<crate>/<模块路径>/<用例名>.toml`。
/// 集成测试(`crates/<crate>/tests/*.rs`)无 `/src/` 段 ⇒ `None`(无 overlay;
/// 它们仍可用 `OWL_TEST_CONF` 定口径)。
fn case_overlay_path(file: &str, name: &str) -> Option<PathBuf> {
    let rel = file.strip_prefix("crates/")?;
    let (crate_name, rest) = rel.split_once("/src/")?;
    let module = rest.strip_suffix(".rs")?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    Some(
        repo_root
            .join("testdata/cases")
            .join(crate_name)
            .join(module)
            .join(format!("{name}.toml")),
    )
}

/// **用例配置注入**(宏 wrapper 唯一调用点)。
///
/// `name` = 用例函数名;`file` = 宏调用点的 `file!()`(推导 overlay 路径)。
/// `runtime.test_device` 无条件回填 `OWL_TEST_DEVICE`(serde(skip) 不入 toml,
/// 设备序只能来自 env —— GPU 门控据此判跳)。
///
/// # Panics
/// - `OWL_TEST_CONF` 指向的文件缺失 / 解析失败 / 含账外键 ⇒ **响亮 panic**;
/// - 设了退役键 `OWL_TEST_CONFIG_<NAME>` ⇒ **响亮 panic**(附迁移指引)。
pub fn case_config(name: &str, file: &str) -> OwlConfig {
    // ① 退役键:响亮拒(不静默忽略)
    let retired = format!("{RETIRED_PER_CASE_PREFIX}{}", name.to_uppercase());
    if crate::env_reader::str(&retired).is_some() {
        let hint = case_overlay_path(file, name)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<仓库>/testdata/cases/<crate>/<模块路径>/<用例名>.toml".into());
        panic!(
            "[owl_test] 用例 {name} 设了退役键 {retired}:每用例 env 键已退役 —— \
             改用单一入口 {TEST_CONF_ENV}(测试环境档)+ 约定 overlay 文件 {hint}"
        );
    }

    // ② 层链:OWL_TEST_CONF → 用例 overlay(存在才入链)
    let mut layers: Vec<PathBuf> = Vec::new();
    if let Some(p) = crate::env_reader::str(TEST_CONF_ENV) {
        let pb = PathBuf::from(&p);
        if !pb.is_file() {
            panic!("[owl_test] {TEST_CONF_ENV}={p} 指向的文件不存在(测试环境档必须显式存在)");
        }
        layers.push(pb);
    }
    if let Some(o) = case_overlay_path(file, name) {
        if o.is_file() {
            layers.push(o);
        }
    }

    let refs: Vec<&std::path::Path> = layers.iter().map(|p| p.as_path()).collect();
    let mut cfg = OwlConfig::load_chain(None, &refs).unwrap_or_else(|e| {
        panic!("[owl_test] 用例 {name} 配置加载失败(层:{layers:?}): {e}")
    });
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

    /// overlay 路径派生:`crates/<crate>/src/**` ⇒ `testdata/cases/<crate>/<模块>/<用例>.toml`;
    /// 集成测试无 `/src/` ⇒ None。
    #[test]
    fn overlay_路径派生() {
        let p = case_overlay_path("crates/models/src/layers/gdn.rs", "gpu_l2norm_matches_host")
            .expect("应有 overlay 路径");
        let s = p.to_string_lossy().replace('\\', "/");
        assert!(s.ends_with("testdata/cases/models/layers/gdn/gpu_l2norm_matches_host.toml"), "{s}");
        assert!(case_overlay_path("crates/models/tests/e2e.rs", "x").is_none(), "集成测试无 overlay");
        assert!(case_overlay_path("apps/server/src/main.rs", "x").is_none(), "非 crates/ 前缀不猜");
    }

    /// 环境档双臂:未设 ⇒ default;设 ⇒ `flavor` 基准 + 覆盖项都到手,且与
    /// 独立复算(`load_chain` 直调)逐字段同。
    #[test]
    fn 环境档_双臂() {
        let _g = env_lock();
        let _no_dev = crate::env_reader::EnvGuard::unset("OWL_TEST_DEVICE");
        let _no_conf = crate::env_reader::EnvGuard::unset(TEST_CONF_ENV);
        // 臂 1:未设 + 无 overlay ⇒ 引擎缺省
        let cfg = case_config("definitely_unset_case_xyz", "crates/shared/src/testkit/case.rs");
        assert_eq!(cfg, OwlConfig::default(), "未设 = 引擎缺省配置");
        // 臂 2:设环境档(flavor + 覆盖)
        let path = std::env::temp_dir().join(format!("owl_test_conf_{}.toml", std::process::id()));
        std::fs::write(&path, "flavor = \"spec-dflash\"\n[runtime]\nmax_seq = 2048\n").expect("写");
        let _c = crate::env_reader::EnvGuard::set(TEST_CONF_ENV, path.to_str().unwrap());
        let cfg2 = case_config("env_conf_arm", "crates/shared/src/testkit/case.rs");
        assert_eq!(cfg2.runtime.max_seq, 2048, "覆盖项到手");
        assert_eq!(cfg2.sampling.mode, crate::config::SamplerMode::Greedy, "flavor 基准在手");
        let direct = OwlConfig::load_chain(None, &[path.as_path()]).expect("直调复算");
        assert_eq!(cfg2, direct, "与 load_chain 直调逐字段同");
        drop(_c);
        std::fs::remove_file(&path).ok();
        // 臂 3:设备序回填
        let _d = crate::env_reader::EnvGuard::set("OWL_TEST_DEVICE", "2");
        let cfg3 = case_config("env_conf_dev", "crates/shared/src/testkit/case.rs");
        assert_eq!(cfg3.runtime.test_device, Some(2), "OWL_TEST_DEVICE 回填");
    }

    /// 退役键 = 响亮 panic(带迁移指引)。
    #[test]
    #[should_panic(expected = "已退役")]
    fn 退役键_响亮拒() {
        let _g = env_lock();
        let _k = crate::env_reader::EnvGuard::set("OWL_TEST_CONFIG_RETIRED_PROBE", "/tmp/x.toml");
        let _ = case_config("retired_probe", "crates/shared/src/testkit/case.rs");
    }

    /// 用例 overlay 真的被吃到(约定路径;测试自清场)。
    #[test]
    fn 用例_overlay_生效() {
        let _g = env_lock();
        let _no_conf = crate::env_reader::EnvGuard::unset(TEST_CONF_ENV);
        let file = "crates/shared/src/testkit/case.rs";
        let p = case_overlay_path(file, "overlay_probe").expect("路径");
        crate::file_loader::create_dir_all(p.parent().unwrap()).expect("mkdir");
        crate::file_loader::write(&p, "[runtime]\nmax_seq = 4321\n").expect("写 overlay");
        let cfg = case_config("overlay_probe", file);
        assert_eq!(cfg.runtime.max_seq, 4321, "overlay 文件被吃到");
        std::fs::remove_file(&p).ok();
    }
}
