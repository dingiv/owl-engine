//! **测试资源清单**(`testdata/resources.toml`;2026-10-18) —— 取代
//! `OWL_AWQ27B_DIR` / `OWL_CKPT_*` / `OWL_DFLASH2_DIR` / `OWL_VLM_CKPT` 一族 env。
//!
//! 形态(TOML;`key = "路径"` 或 `"flag:x" = true`):
//!
//! ```toml
//! "ckpt:27b-awq" = "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4"
//! "ckpt:dflash2" = "/home/div/Documents/codes/models/syvai/Qwen3.8-27B-DFlash2-W4A16"
//! "flag:hf-parity" = true
//! ```
//!
//! 读取面:[`resource`] / [`resource_or`] / [`resource_flag`] / [`missing_reason`];
//! 清单路径固定 = 仓库根 `testdata/resources.toml`(不存在 = 空表 ⇒ 相关用例 skip,
//! 零 env 也能跑)。**旧 env 仍被读到 = 响亮 panic**(迁移指引:改用清单里的哪个 key)。
//!
//! 单一来源:只有本模块读环境(且只为**拒绝**旧键);资源值一律走清单。

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// 清单路径(仓库根 `testdata/resources.toml`;不存在 = 空表)
pub fn manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/resources.toml")
}

/// **退役 env → 清单 key**(设了 = 响亮拒,附迁移指引)
const RETIRED_ENV: &[(&str, &str)] = &[
    // 27B 主干 / 草稿
    ("OWL_AWQ27B_DIR", "ckpt:27b-awq"),
    ("OWL_CKPT_27B_AWQ", "ckpt:27b-awq"),
    ("OWL_DFLASH2_DIR", "ckpt:dflash2"),
    ("OWL_CKPT_DFLASH2", "ckpt:dflash2-bf16"),
    ("OWL_DFLASH2_BF16_DIR", "ckpt:dflash2-bf16"),
    // 小档 / 视觉(4B 塔与 VLM 同源)
    ("OWL_CKPT_0_8B", "ckpt:0.8b"),
    ("OWL_CKPT_4B_VL", "ckpt:qwen35-4b"),
    ("OWL_VLM_CKPT", "ckpt:qwen35-4b"),
    ("OWL_VISION_CKPT", "ckpt:qwen35-4b"),
    // 模板 / 新档
    ("OWL_TPL_DIR", "tpl:default"),
    ("OWL_TPL_DIR_QWEN35", "tpl:qwen35"),
    ("OWL_NEW_CKPT", "ckpt:new"),
    // 开关
    ("OWL_NEW_CKPT_LOAD", "flag:new-ckpt-load"),
    ("OWL_VLM_REAL", "flag:vlm-real"),
    ("OWL_VLM_BENCH", "flag:vlm-bench"),
    ("OWL_VLM_LOAD_DEBUG", "flag:vlm-load-debug"),
];

/// 清单条目
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resource {
    /// `key = "路径"`
    Path(String),
    /// `"flag:x" = true/false`
    Flag(bool),
}

fn table() -> &'static BTreeMap<String, Resource> {
    static T: OnceLock<BTreeMap<String, Resource>> = OnceLock::new();
    T.get_or_init(|| {
        // ① 退役 env:响亮拒(不静默忽略 —— 旧脚本设了 env 却走清单会静默换资源)
        if let Some(msg) = retired_env_error(|k| std::env::var_os(k).is_some()) {
            panic!("{msg}");
        }
        // ② 清单(缺 = 空表)
        let p = manifest_path();
        let Ok(raw) = std::fs::read_to_string(&p) else {
            return BTreeMap::new();
        };
        let parsed: BTreeMap<String, toml::Value> = toml::from_str(&raw)
            .unwrap_or_else(|e| panic!("[测试资源] 清单 {} 解析失败: {e}", p.display()));
        let mut out = BTreeMap::new();
        for (k, v) in parsed {
            let r = match v {
                toml::Value::String(s) => Resource::Path(s),
                toml::Value::Boolean(b) => Resource::Flag(b),
                other => panic!(
                    "[测试资源] 清单 {} 的 \"{k}\" 类型非法(要路径字符串或布尔):{other:?}",
                    p.display()
                ),
            };
            out.insert(k, r);
        }
        out
    })
}

/// 退役 env 检查(纯函数;可单测):命中任一 = Some(迁移指引)
fn retired_env_error(has: impl Fn(&str) -> bool) -> Option<String> {
    RETIRED_ENV.iter().find(|(env, _)| has(env)).map(|(env, key)| {
        format!(
            "[测试资源] {env} 已退役:改用资源清单 {} 里的 \"{key}\"(env 不再参与测试资源选择)",
            manifest_path().display()
        )
    })
}

/// 取资源**路径**(非路径条目/未列 = None)
pub fn resource(key: &str) -> Option<String> {
    match table().get(key) {
        Some(Resource::Path(s)) => Some(s.clone()),
        _ => None,
    }
}

/// 取资源路径,**缺则回退 `default`**(兼容"仓库内约定路径"的调用点)
pub fn resource_or(key: &str, default: &str) -> String {
    resource(key).unwrap_or_else(|| default.to_string())
}

/// 取布尔开关(未列/非布尔 = false)
pub fn resource_flag(key: &str) -> bool {
    matches!(table().get(key), Some(Resource::Flag(true)))
}

/// 资源是否**可用**(路径条目 ⇒ 路径存在;布尔条目 ⇒ true)
pub fn resource_ok(key: &str) -> bool {
    match table().get(key) {
        Some(Resource::Path(p)) => std::path::Path::new(p).exists(),
        Some(Resource::Flag(b)) => *b,
        None => false,
    }
}

/// skip 文案:`needs = "<key>"` 门未过的可读原因(点名 key + 清单路径)
pub fn missing_reason(key: &str) -> String {
    let p = manifest_path();
    match table().get(key) {
        None => format!("skip: 资源清单缺 \"{key}\"({} 未列或不存在)", p.display()),
        Some(Resource::Path(path)) => {
            format!("skip: 资源 \"{key}\" 的路径不在:{path}(清单 {})", p.display())
        }
        Some(Resource::Flag(false)) => {
            format!("skip: 资源开关 \"{key}\" = false(清单 {})", p.display())
        }
        Some(Resource::Flag(true)) => format!("资源 \"{key}\" 可用"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 清单缺失 = 空表语义:`resource` = None、`resource_flag` = false、
    /// `missing_reason` 点名 key 与清单路径(不 panic)。
    #[test]
    fn missing_manifest_reads_as_empty() {
        assert_eq!(resource("ckpt:definitely-not-listed"), None);
        assert!(!resource_flag("flag:definitely-not-listed"));
        let r = missing_reason("ckpt:nope");
        assert!(r.contains("ckpt:nope") && r.contains("resources.toml"), "{r}");
    }

    /// 回退语义:`resource_or` 缺时给默认;有则清单优先(本机清单可能列了该 key,
    /// 故只断言"要么清单值要么默认值")。
    #[test]
    fn resource_or_falls_back() {
        let v = resource_or("ckpt:definitely-not-listed", "/tmp/fallback");
        assert_eq!(v, "/tmp/fallback");
    }

    /// 退役 env = 响亮(纯函数面):命中即给迁移指引,未命中 = None。
    #[test]
    fn retired_env_is_loud() {
        let msg = retired_env_error(|k| k == "OWL_AWQ27B_DIR").expect("应命中");
        assert!(msg.contains("已退役") && msg.contains("ckpt:27b-awq"), "{msg}");
        assert!(retired_env_error(|_| false).is_none(), "无退役键 = 无话");
        // 全表覆盖:每个退役键都能给出指引
        for (env, key) in RETIRED_ENV {
            let m = retired_env_error(|k| k == *env).expect(env);
            assert!(m.contains(key), "{env} → {key}: {m}");
        }
    }
}
