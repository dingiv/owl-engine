//! **profiles 面**(2026-10-20 统一):配置档的**唯一**发现/路径模块。
//!
//! 定位:profile 就是"预先写好的一份配置文件";配置层
//! (`crate::config::OwlConfig::load`)只认"基准文件 + 覆盖文件"。
//! 用户入口只传**路径**:`server -p profiles/xxx.toml` —— 本模块**不做**
//! 名字查找;`profile_path`/`legacy_path` 只服务内部测试/清单。
//!
//! 目录:
//! - `profiles/` = 常用、可直接复制修改的档;
//! - `profiles/legacy/` = 历史低层档(保留回归/取证用,不再用户主入口)。
//!
//! 根目录发现(按序取首个存在者):二进制同级 → `target/{debug,release}/../..`
//! → 源码树 `CARGO_MANIFEST_DIR/../..`。

use std::path::PathBuf;

/// 配置档根目录名(相对仓库根)
pub const PROFILE_DIR_NAME: &str = "profiles";
/// 历史档子目录(相对 `profiles/`)
pub const LEGACY_DIR_NAME: &str = "legacy";

/// 在仓库/部署的候选位置找某个目录(按序取首个存在者)
fn find_dir(name: &str) -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            cands.push(d.join(name)); // ① 与二进制同级(部署)
            cands.push(d.join("../..").join(name)); // ② target/<profile>/../..
            cands.push(d.join("../../..").join(name)); // ③ 兜底再上一层
        }
    }
    cands.push(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(name),
    ); // ④ 源码树
    cands.into_iter().find(|p| p.is_dir())
}

/// 目录下全部 `.toml` 名(去扩展;排序;子目录忽略)
fn names_in(dir: &PathBuf) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return v };
    for e in rd.flatten() {
        if !e.path().is_file() {
            continue;
        }
        if let Some(stem) = e.file_name().to_str().and_then(|f| f.strip_suffix(".toml")) {
            v.push(stem.to_string());
        }
    }
    v.sort();
    v
}

/// 常用配置档目录(`profiles/`)
pub fn profile_dir() -> Option<PathBuf> {
    find_dir(PROFILE_DIR_NAME)
}

/// 历史档目录(`profiles/legacy/`)
pub fn legacy_dir() -> Option<PathBuf> {
    find_dir(&format!("{PROFILE_DIR_NAME}/{LEGACY_DIR_NAME}"))
}

/// 常用档名(顶层 `.toml`)
pub fn profile_names() -> Vec<String> {
    profile_dir().map(|d| names_in(&d)).unwrap_or_default()
}

/// 历史档名(`profiles/legacy/*.toml`)
pub fn legacy_names() -> Vec<String> {
    legacy_dir().map(|d| names_in(&d)).unwrap_or_default()
}

/// 常用档名 → 文件路径(内部测试/清单;用户 CLI 直接传路径)
pub fn profile_path(name: &str) -> Result<PathBuf, String> {
    let dir = profile_dir().ok_or_else(|| {
        "profiles 目录未找到(需要它;找过二进制同级与源码树)".to_string()
    })?;
    let p = dir.join(format!("{name}.toml"));
    if !p.is_file() {
        return Err(format!("未知 profile {name:?}(可用:{:?})", profile_names()));
    }
    Ok(p.canonicalize().unwrap_or(p))
}

/// 历史档名 → 文件路径(测试/回归)
pub fn legacy_path(name: &str) -> Result<PathBuf, String> {
    let dir = legacy_dir().ok_or_else(|| {
        format!("profiles/{LEGACY_DIR_NAME} 目录未找到(历史档需要它)")
    })?;
    let p = dir.join(format!("{name}.toml"));
    if !p.is_file() {
        return Err(format!("未知历史档 {name:?}(可用:{:?})", legacy_names()));
    }
    Ok(p.canonicalize().unwrap_or(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 历史档清单非空且全部可解析(扩展路径走文件 loader)
    #[test]
    fn 历史档清单非空且全部可解析() {
        let names = legacy_names();
        assert!(names.iter().any(|n| n == "spec-dflash"), "应有 spec-dflash:{names:?}");
        for n in &names {
            let path = legacy_path(n).expect("历史档路径");
            crate::config::OwlConfig::load(None, Some(&path))
                .unwrap_or_else(|e| panic!("历史档 {n} 应可解析: {e}"));
        }
        assert!(legacy_path("no-such-preset").is_err(), "未知档应 Err");
    }

    /// 常用档清单:至少含一个 `[intent]` 档,且都能经完整 loader 可加载。
    #[test]
    fn 常用档清单非空且全部可解析() {
        let names = profile_names();
        assert!(
            names.iter().any(|n| n == "awq27b-3090-max"),
            "应有 awq27b-3090-max:{names:?}"
        );
        let mut intent_seen = false;
        for n in &names {
            let p = profile_path(n).expect("profile 路径");
            let raw = std::fs::read_to_string(&p).expect("读");
            if !raw.contains("[intent]") {
                // 顶层也允许放 legacy 低层档(如 262k-vlm);历史清单测试覆盖它们。
                continue;
            }
            intent_seen = true;
            let cli = vec![
                (
                    "model.dflash2_dir".to_string(),
                    toml::Value::String("/tmp/owl-test-draft".into()),
                ),
                (
                    "runtime.tp_ordinals".to_string(),
                    toml::Value::Array(vec![
                        toml::Value::Integer(2),
                        toml::Value::Integer(3),
                    ]),
                ),
            ];
            let r = crate::config::OwlConfig::load_resolved(Some(&p), &[], &cli)
                .unwrap_or_else(|e| panic!("profile {n} 应可加载: {e}"));
            assert!(r.intent().is_some(), "profile {n} 必须含 [intent]");
        }
        assert!(intent_seen, "至少应有一个 [intent] profile");
        assert!(profile_path("no-such-profile").is_err(), "未知档应 Err");
    }
}
