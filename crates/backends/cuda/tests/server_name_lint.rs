//! server 侧名字面量门(P0 收官;operator-contract 设计 §4.1)。
//!
//! **server 主循环零算子知识**:`server/` 子树不得含任何算子名字字面量
//! —— 名字分派在 kernels::registry(knows/execute),家族名经**族身份常量**
//! 引用。违例 = 本测试红。
//!
//! **住址纪律(2026-10-15)**:族名/算子名的**唯一定义处 = 各族自己的
//! `kernels::family/<族>.rs`**(身份面零门,声明面 `ops/` 与 models 可引用);
//! 本清单 = 该住址表的**两侧互证**(清单里的名字必须在代码里以字面量存在于
//! 那个文件,而 server 子树里一个都不许出现)。

use std::path::Path;

const OP_NAME_LITERALS: &[&str] = &[
    "gdn_chunked_delta_rule_fwd",
    "gdn_scalar_delta_rule_fwd",
    "cublas_gemm_f16",
    "cublas_gemm_bf16",
    "marlin_gemm_w4a16",
    "marlin_gemm_w4a16_awq",
    "marlin_gemm_w4a16_bf16",
    "flashinfer_prefill_paged_f16",
    "flashinfer_prefill_paged_fp8kv",
    // 2026-10-15 补:此前漏了 XQA 两名(门有覆盖缺口 ⇒ 清单不完整)
    "owl_xqa_verify",
    "owl_xqa_decode",
];

fn walk_rs(dir: &Path, hits: &mut Vec<String>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_rs(&p, hits);
        } else if p.extension().is_some_and(|x| x == "rs") {
            let Ok(src) = std::fs::read_to_string(&p) else { continue };
            for bad in OP_NAME_LITERALS {
                if src.contains(bad) {
                    hits.push(format!("{}: 算子名字面量 `{bad}`", p.display()));
                }
            }
        }
    }
}

#[test]
fn server_shall_not_know_operator_names() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let src = Path::new(manifest).join("src/server");
    let mut hits = Vec::new();
    walk_rs(&src, &mut hits);
    assert!(
        hits.is_empty(),
        "server 触碰算子名字面量(应经 registry/face 常量):\n{}",
        hits.join("\n")
    );
}
