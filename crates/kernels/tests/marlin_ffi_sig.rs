//! **marlin foreign 通道:`.cu` 薄壳 ⇔ Rust `extern` 声明**形参序互证
//! (2026-10-16 W4A8 线;§12 施工单的"槽序 ⇔ C 形参序"门)。
//!
//! 为什么需要第三道门:`op_tables.rs` 的两道门(token 级 [`parse_cu_sig`] /
//! 名字级 [`parse_cu_param_names`])只覆盖 **native** 核(`extern "C"
//! __global__ void …`)。marlin 是 **foreign**(`SRC` 为空)⇒ 两道门都不看它,
//! 而它的 `extern "C"` 薄壳与 Rust 绑定是**两份手抄的签名**,且形参序还跨语言
//! 不同(线槽序 `m,k,n` vs C 形参序 `prob_m,prob_n,prob_k`)—— 写反一次 =
//! n/k 静默对调(两者常同为 2 的幂,形状检查看不出,结果全错而不报错)。
//!
//! 判据:**两侧形参名逐位相等**(忽略大小写 —— `.cu` 用 `A/B/C` 大写)。
//! 名字改了/序改了/条数改了 ⇒ 红,逼作者同时审两边。
//!
//! [`parse_cu_sig`]: owl_kernels::cu_sig::parse_cu_sig
//! [`parse_cu_param_names`]: owl_kernels::cu_sig::parse_cu_param_names

use std::path::Path;

/// 参与互证的 (C 函数名, 该名在两侧都必须存在)。
/// 只列**有 Rust 绑定**的四个薄壳(`marlin_gemm_v2_w4a8_ffi` 的对称变体
/// 目前在 Rust 侧未引,故不在此表 —— 新接一个就加一行)。
const ENTRIES: &[&str] = &[
    "marlin_gemm_v2_ffi",
    "marlin_gemm_v2_bf16_ffi",
    "marlin_gemm_v2_awq_ffi",
    "marlin_gemm_v2_awq_w4a8_ffi",
];

fn read(rel: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("读 {} 失败: {e}", p.display()))
}

/// 取 `anchor` 之后第一对括号里的形参串(无嵌套括号;薄壳签名如此)
fn parens_after(src: &str, anchor: &str) -> Option<String> {
    let start = src.find(anchor)? + anchor.len();
    let end = src[start..].find(')')? + start;
    Some(src[start..end].to_string())
}

/// C 形参名(末标识符:`const void* A` → `A`)
fn c_param_names(src: &str, name: &str) -> Vec<String> {
    let raw = parens_after(src, &format!("int {name}("))
        .unwrap_or_else(|| panic!("{name}: .cu 里找不到 `int {name}(` 薄壳"));
    raw.split(',')
        .map(|p| {
            let p = p.replace("const ", "").replace('*', " ");
            p.split_whitespace()
                .last()
                .unwrap_or_else(|| panic!("{name}: 无法解析的形参 `{p}`"))
                .to_string()
        })
        .collect()
}

/// Rust 形参名(`a: *const c_void` → `a`)
fn rs_param_names(src: &str, name: &str) -> Vec<String> {
    let raw = parens_after(src, &format!("fn {name}("))
        .unwrap_or_else(|| panic!("{name}: Rust 侧找不到 extern 声明"));
    raw.split(',')
        .filter_map(|p| {
            let n = p.split(':').next().unwrap_or("").trim();
            (!n.is_empty()).then(|| n.to_string())
        })
        .collect()
}

/// **形参序逐位互证**(名相同、序相同、条数相同)
#[test]
fn marlin_ffi_param_order_matches_c_shell() {
    let cu = read("cu/marlin/marlin_host.cu");
    let rs = read("src/family/marlin/mod.rs");
    for name in ENTRIES {
        let c = c_param_names(&cu, name);
        let r = rs_param_names(&rs, name);
        assert!(!c.is_empty() && !r.is_empty(), "{name}: 空形参表(解析器/锚点失配?)");
        let c_lc: Vec<String> = c.iter().map(|s| s.to_ascii_lowercase()).collect();
        let r_lc: Vec<String> = r.iter().map(|s| s.to_ascii_lowercase()).collect();
        assert_eq!(
            c_lc, r_lc,
            "{name}: C 薄壳与 Rust extern 的形参序不一致(跨语言手抄两份 ⇒ \
             写反 = 静默错值)。\n  .cu : {c:?}\n  rust: {r:?}"
        );
    }
    eprintln!(
        "[marlin-ffi] {} 个薄壳的 C ⇔ Rust 形参序逐位一致 ✓",
        ENTRIES.len()
    );
}

/// **换位雷面**:C 形参序里 `prob_m, prob_n, prob_k` 是**连着的三连**,
/// 而线槽序是 `m, k, n` —— 本门锁住"两侧都是 m,n,k",换位只可能发生在
/// 中间的接线点(由 `client::marlin::MarlinCall::ffi_mnk` 单源 + 其单测锁)。
#[test]
fn mnk_triple_is_adjacent_and_ordered_in_both_sides() {
    let cu = read("cu/marlin/marlin_host.cu");
    let rs = read("src/family/marlin/mod.rs");
    for name in ENTRIES {
        let c = c_param_names(&cu, name);
        let at = c
            .iter()
            .position(|s| s == "prob_m")
            .unwrap_or_else(|| panic!("{name}: C 薄壳缺 prob_m"));
        assert_eq!(
            &c[at..at + 3],
            &["prob_m", "prob_n", "prob_k"],
            "{name}: C 侧 (prob_m,prob_n,prob_k) 必须连续且按此序"
        );
        let r = rs_param_names(&rs, name);
        let rt = r
            .iter()
            .position(|s| s == "prob_m")
            .unwrap_or_else(|| panic!("{name}: Rust extern 缺 prob_m"));
        assert_eq!(
            &r[rt..rt + 3],
            &["prob_m", "prob_n", "prob_k"],
            "{name}: Rust 侧 (prob_m,prob_n,prob_k) 必须连续且按此序"
        );
    }
}
