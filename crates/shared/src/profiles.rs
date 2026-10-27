//! **profiles 面**(2026-10-20 统一):配置档的**唯一**发现/路径模块。
//!
//! 定位:profile 就是"预先写好的一份配置文件";配置层
//! (`crate::config::OwlConfig::load`)只认"基准文件 + 覆盖文件"。
//! 用户入口只传**路径**:`server -p profiles/xxx.toml` —— 本模块**不做**
//! 名字查找;`profile_path` 只服务内部测试/清单。
//!
//! 目录:`profiles/` = **全量档**(2026-10-21 起每档必须全字段显式;
//! `profiles/legacy/` 已随该裁决删除,不再有历史低层档)。
//!
//! 根目录发现(按序取首个存在者):二进制同级 → `target/{debug,release}/../..`
//! → 源码树 `CARGO_MANIFEST_DIR/../..`。

use std::path::PathBuf;

/// 配置档根目录名(相对仓库根)
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

/// 常用档名(顶层 `.toml`)
pub fn profile_names() -> Vec<String> {
    profile_dir().map(|d| names_in(&d)).unwrap_or_default()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// **常驻档门**(2026-10-24 用户裁决"只留唯二硬件档";2026-10-27 增补第三档):
    /// `profiles/` 顶层入口档只允许三个 ——
    ///   ① 单 3090 / 27B **AWQ-INT4**(视觉 + image_evict 全特性);
    ///   ② 双 3080 TP2 / 27B AWQ-INT4(长上下文 + 大图窗口);
    ///   ③ 单 3090 / 27B **GGUF Q4_K_M**(文本档:prefill/decode 都比①快 25%+,
    ///      代价 = 上下文 56k + 无视觉;2026-10-27 实测见该档头注)。
    /// 其余形状一律用 `--set` 临时覆盖,不再落成新文件(历史 28 个入口档的
    /// 收敛记录见 `roadmap.local/profiles-consolidation-2026-10-24.md`)。
    /// `full.toml` = 全量模板(构成底座,不是档位)、`parts/**` = 组件,不计入。
    #[test]
    fn 入口档面只允许三个常驻档() {
        let names = profile_names();
        let mut got: Vec<&str> = names
            .iter()
            .map(|s| s.as_str())
            .filter(|n| *n != "full")
            .collect();
        got.sort_unstable();
        assert_eq!(
            got,
            vec!["qwen38_27b-3080_tp2", "qwen38_27b-3090", "qwen38_27b-gguf_3090"],
            "档面膨胀了:要新形状请用 `--set` 覆盖,不要新增入口档"
        );
    }

    /// **全量档门**:`profiles/` 下每一档都必须能经 `Completeness::Enforce`
    /// 加载(缺字段 = 拒启)。这是"档可启动"的最低保证,替代旧的历史档清单。
    #[test]
    fn 全部档均可经完整度门加载() {
        let names = profile_names();
        assert!(names.iter().any(|n| n == "qwen38_27b-3090"), "应有单 3090 常驻档:{names:?}");
        assert!(names.iter().any(|n| n == "qwen38_27b-3080_tp2"), "应有双 3080 常驻档:{names:?}");
        for n in &names {
            let p = profile_path(n).expect("profile 路径");
            crate::config::OwlConfig::load_resolved(
                Some(&p), &[], &[], crate::config::Completeness::Enforce,
            )
            .unwrap_or_else(|e| panic!("profile {n} 必须可加载(全字段显式): {e}"));
        }
        assert!(profile_path("no-such-profile").is_err(), "未知档应 Err");
    }

    /// 目录发现:同一份找法对两种入口一致(清单与单档路径同源)
    #[test]
    fn 目录发现与单档路径同源() {
        let dir = profile_dir().expect("profiles 目录");
        for n in profile_names().iter().take(3) {
            let p = profile_path(n).expect("路径");
            assert_eq!(p.parent().expect("父目录"), dir.canonicalize().unwrap_or(dir.clone()));
        }
    }
}
