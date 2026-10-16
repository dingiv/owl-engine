//! **assets 面**(2026-10-16):`assets/` 根发现 + **预设档(flavor)名字 ↔ 文件**。
//!
//! 定位:**flavor 不是配置层的概念** —— 它就是"预先写好的一份配置文件"。
//! 配置层(`crate::config::OwlConfig::load`)只认"基准文件 + 覆盖文件";
//! 把 `--flavor <name>` 变成路径的活在本模块。
//! 根目录发现:① 可执行文件同级 `assets/` ② 源码树 `crates/shared/../../assets`。

use std::path::PathBuf;

/// 预设档子目录名(相对 assets 根)
pub const PRESET_SUBDIR: &str = "flavors";

/// assets 根目录(找不到 = None)
pub fn dir() -> Option<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            let p = d.join("assets");
            if p.is_dir() { return Some(p); }
        }
    }
    let dev = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
    if dev.is_dir() { return Some(dev); }
    None
}

/// 预设档目录(`assets/flavors`)
pub fn preset_dir() -> Option<PathBuf> { dir().map(|d| d.join(PRESET_SUBDIR)) }

/// 可用预设档名(= 目录下文件名去 `.toml`;`--list-flavors` 消费面)
pub fn preset_names() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let Some(dir) = preset_dir() else { return v };
    let Ok(rd) = std::fs::read_dir(&dir) else { return v };
    for e in rd.flatten() {
        if let Some(stem) = e.file_name().to_str().and_then(|f| f.strip_suffix(".toml")) {
            v.push(stem.to_string());
        }
    }
    v.sort();
    v
}

/// 预设档名 → 文件路径(不存在/无 assets ⇒ Err 带可用清单)
pub fn preset_path(name: &str) -> Result<PathBuf, String> {
    let dir = preset_dir().ok_or_else(|| "assets 目录未找到(预设档需要它)".to_string())?;
    let p = dir.join(format!("{name}.toml"));
    if !p.is_file() { return Err(format!("未知预设档 {name:?}(可用:{:?})", preset_names())); }
    Ok(p)
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
}
