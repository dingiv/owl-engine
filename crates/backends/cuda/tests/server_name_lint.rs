//! server 侧名字面量门(P0 收官;operator-contract 设计 §4.1)。
//!
//! **server 主循环零算子知识**:`server/` 子树不得含任何算子名字面量
//! —— 名字分派在 kernels::registry(knows/execute),家族名经**族身份常量**
//! 引用。违例 = 本测试红。
//!
//! # 住址纪律(2026-10-15;用户裁决 + 审查补强)
//!
//! 族名/算子名的**唯一定义处 = 各族自己的 `kernels/src/family/<族>.rs`**
//! (身份面零门,声明面 `ops/` 与 models 可引用)。本门是住址表的**两侧互证**:
//!
//! 1. **负侧**(`server_shall_not_know_operator_names`):`backends/cuda/src/server/**`
//!    一个名字都不许出现(要引就引族常量);
//! 2. **正侧**(`names_really_live_in_family_files`,2026-10-15 审查补):每个名字
//!    必须**真的以字面量住在** `kernels/src/family/` 下某文件里 —— 旧实现只有
//!    负侧,住址表被改坏(名字搬到别处/被删)时门**不会红**。
//!
//! 清单**不再手写**:由 `owl_kernels::ops` 的族算子声明 + `KernelSpec::FAMILY`
//! 派生 ⇒ 新增族/族算子自动进门(旧手写清单对新名字零覆盖)。

use std::path::{Path, PathBuf};

/// **派生清单**:全部族算子 entry ∪ 全部族 id(主名)。
fn family_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for m in owl_kernels::ops::all().filter(|m| m.is_family()) {
        if !v.contains(&m.entry) {
            v.push(m.entry);
        }
        if !v.contains(&m.family.0) {
            v.push(m.family.0);
        }
    }
    v.sort_unstable();
    v
}

fn walk_rs(dir: &Path, hits: &mut Vec<(PathBuf, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_rs(&p, hits);
        } else if p.extension().is_some_and(|x| x == "rs") {
            let Ok(src) = std::fs::read_to_string(&p) else { continue };
            for bad in family_names() {
                if src.contains(bad) {
                    hits.push((p.clone(), bad.to_string()));
                }
            }
        }
    }
}

fn kernels_family_dir() -> PathBuf {
    // 本 crate = <root>/crates/backends/cuda ⇒ kernels = <root>/crates/kernels
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/src/family")
}

#[test]
fn server_shall_not_know_operator_names() {
    let names = family_names();
    assert!(names.len() >= 11, "派生清单异常小({names:?})—— 声明面/门坏了?");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server");
    let mut hits = Vec::new();
    walk_rs(&src, &mut hits);
    assert!(
        hits.is_empty(),
        "server 触碰算子名字面量(应经 registry/族身份常量):\n  {}",
        hits.iter()
            .map(|(p, n)| format!("{}: `{n}`", p.display()))
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// **正侧**:名字必须真的住在 `kernels/src/family/` 的字面量里(住址表互证)。
#[test]
fn names_really_live_in_family_files() {
    let dir = kernels_family_dir();
    let mut src_all = String::new();
    let mut files = Vec::new();
    walk_rs(&dir, &mut files);
    assert!(!files.is_empty(), "未扫到 family 源({})—— 路径推导错了?", dir.display());
    for (p, _) in &files {
        if let Ok(t) = std::fs::read_to_string(p) {
            src_all.push_str(&t);
        }
    }
    let missing: Vec<&str> = family_names()
        .into_iter()
        .filter(|n| !src_all.contains(&format!("\"{n}\"")))
        .collect();
    assert!(
        missing.is_empty(),
        "名字未以字面量住在 `kernels/src/family/`(住址纪律破了;正侧半边):\n  {missing:?}"
    );
}
