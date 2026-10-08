//! 引用方向 lint(operator-contract 设计 v0.4 §4.1 P0/P2 机器门)。
//!
//! 分层纪律:**models(客户端)禁触服务端面** —— device(执行引擎)/
//! registry(FamilyRuntime/OpRegistry)是 server 侧专属;models 只许
//! 走客户端面(contract 的 Call builder / OpId 常量)。
//! 违例 = 本测试红,不等 code review 人眼。
//!
//! (server 侧反向门——backends 零算子名字面量——随 M4 切换时启用,
//!  当前 foreign.rs 尚存旧分派表,M4 退役后加 backends 侧同款测试。)

use std::path::Path;

const FORBIDDEN_IN_MODELS: &[&str] = &[
    "owl_kernels::device",
    "owl_kernels::registry",
    "owl_kernels::device::",
    "owl_kernels::registry::",
];

fn walk_rs(dir: &Path, hits: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_rs(&p, hits);
        } else if p.extension().is_some_and(|x| x == "rs") {
            let Ok(src) = std::fs::read_to_string(&p) else { continue };
            for bad in FORBIDDEN_IN_MODELS {
                if src.contains(bad) {
                    hits.push(format!("{}: 引用服务端面 `{bad}`", p.display()));
                }
            }
        }
    }
}

#[test]
fn models_shall_not_touch_server_face() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let src = Path::new(manifest).join("../models/src");
    let mut hits = Vec::new();
    walk_rs(&src, &mut hits);
    assert!(
        hits.is_empty(),
        "models 触碰服务端面(应走客户端面 contract/OpId):\n{}",
        hits.join("\n")
    );
}

/// 用户律(2026-10-12):**层不越解释层直触 kernels** —— layers/ 子树
/// 不得含任何 kernels 直引(face builder/名字/签名/谓词知识一律收编
/// crate::ops §6 动作表;crate::kernel 垫子与 ops::kernel_call 为层侧
/// 唯一合法声明面)。违例 = 本测试红,不等 code review 人眼。
#[test]
fn layers_shall_not_touch_kernels() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let src = Path::new(manifest).join("../models/src/layers");
    let mut hits = Vec::new();
    walk_literal(&src, "owl_kernels", &mut hits);
    assert!(
        hits.is_empty(),
        "layers 直触 kernels(应经 crate::ops §6 / crate::kernel 垫子):\n{}",
        hits.join("\n")
    );
}

fn walk_literal(dir: &Path, literal: &str, hits: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_literal(&p, literal, hits);
        } else if p.extension().is_some_and(|x| x == "rs") {
            let Ok(src) = std::fs::read_to_string(&p) else { continue };
            if src.contains(literal) {
                hits.push(format!("{}: 含 `{literal}`", p.display()));
            }
        }
    }
}
