//! **flavors 面**(2026-10-18:自 `flavors/` **提到仓库顶层 `flavors/`**):
//! 预设档(flavor)名字 ↔ 文件发现。
//!
//! 定位:**flavor 不是配置层的概念** —— 它就是"预先写好的一份配置文件"。
//! 配置层(`crate::config::OwlConfig::load`)只认"基准文件 + 覆盖文件";
//! 把 `--flavor <name>` / 配置里的 `flavor = "<name>"` 变成路径的活在本模块。
//!
//! 根目录发现(按序取首个存在者):
//! ① 二进制同级 `flavors/`(部署:与可执行文件一起分发);
//! ② `target/{debug,release}/../../flavors`(dev 构建:exe 在 target/<profile>/);
//! ③ `CARGO_MANIFEST_DIR/../../flavors`(源码树:`crates/shared/../../`)。

use std::path::PathBuf;

/// 预设档根目录名(相对仓库根)
pub const PRESET_DIR_NAME: &str = "flavors";

/// **意图档**根目录名(相对仓库根;`profiles/*.toml`,2026-10-20 配置面 P0)
pub const PROFILE_DIR_NAME: &str = "profiles";

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

/// 目录下全部 `.toml` 名(去扩展;排序)
fn names_in(dir: &PathBuf) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return v };
    for e in rd.flatten() {
        if let Some(stem) = e.file_name().to_str().and_then(|f| f.strip_suffix(".toml")) {
            v.push(stem.to_string());
        }
    }
    v.sort();
    v
}

/// flavors 根目录(找不到 = None)
pub fn dir() -> Option<PathBuf> {
    find_dir(PRESET_DIR_NAME)
}

/// 预设档目录(= flavors 根;保留旧名,调用点零改)
pub fn preset_dir() -> Option<PathBuf> {
    dir()
}

/// **意图档目录**(`profiles/`;找不到 = None)
pub fn profile_dir() -> Option<PathBuf> {
    find_dir(PROFILE_DIR_NAME)
}

/// 可用预设档名(= 目录下文件名去 `.toml`;`--list-flavors` 消费面)
pub fn preset_names() -> Vec<String> {
    preset_dir().map(|d| names_in(&d)).unwrap_or_default()
}

/// **可用意图档名**(`--list-profiles` 消费面)
pub fn profile_names() -> Vec<String> {
    profile_dir().map(|d| names_in(&d)).unwrap_or_default()
}

/// 预设档名 → 文件路径(不存在/无 flavors 目录 ⇒ Err 带可用清单)
pub fn preset_path(name: &str) -> Result<PathBuf, String> {
    let dir = preset_dir().ok_or_else(|| {
        format!("flavors 目录未找到(预设档需要它;找过二进制同级与源码树)")
    })?;
    let p = dir.join(format!("{name}.toml"));
    if !p.is_file() {
        return Err(format!("未知预设档 {name:?}(可用:{:?})", preset_names()));
    }
    Ok(p.canonicalize().unwrap_or(p))
}

/// **意图档名 → 文件路径**(不存在/无 profiles 目录 ⇒ Err 带可用清单)
pub fn profile_path(name: &str) -> Result<PathBuf, String> {
    let dir = profile_dir().ok_or_else(|| {
        format!("profiles 目录未找到(意图档需要它;找过二进制同级与源码树)")
    })?;
    let p = dir.join(format!("{name}.toml"));
    if !p.is_file() {
        return Err(format!("未知意图档 {name:?}(可用:{:?})", profile_names()));
    }
    Ok(p.canonicalize().unwrap_or(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 预设档清单非空且全部可解析(清单来自目录;解析走配置层同一 serde 路径)
    #[test]
    fn 预设档清单非空且全部可解析() {
        let names = preset_names();
        assert!(names.iter().any(|n| n == "spec-dflash"), "应有 spec-dflash:{names:?}");
        for n in &names {
            let raw = std::fs::read_to_string(preset_path(n).expect("预设档路径")).expect("读");
            crate::config::OwlConfig::from_toml_str(&raw)
                .unwrap_or_else(|e| panic!("预设档 {n} 应可解析: {e}"));
        }
        assert!(preset_path("no-such-preset").is_err(), "未知档应 Err");
    }

    /// **意图档清单**:非空、全部经完整 loader(`[intent]` 派生 + 内存账)可加载
    #[test]
    fn 意图档清单非空且全部可解析() {
        let names = profile_names();
        assert!(
            names.iter().any(|n| n == "awq27b-3090-max"),
            "应有 awq27b-3090-max:{names:?}"
        );
        for n in &names {
            let p = profile_path(n).expect("意图档路径");
            let r = crate::config::OwlConfig::load_resolved(Some(&p), &[], &[])
                .unwrap_or_else(|e| panic!("意图档 {n} 应可加载: {e}"));
            assert!(r.intent().is_some(), "意图档 {n} 必须含 [intent]");
        }
        assert!(profile_path("no-such-profile").is_err(), "未知意图档应 Err");
    }
}
