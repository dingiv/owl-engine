//! **零运行期编译**门(2026-10-13 用户裁决;设计 §15.3/§15.5)。
//!
//! 裁决原文:设备码一律 cubin(构建期产出)+ 主机侧一律 Rust + **零运行期编译**。
//! 这条律此前只有"自觉"保证(`load_nvrtc` 删了但可以被重新加回来,`compile_ptx`
//! 可以在新调用点复活)。本测试把它变成**源码级机器门**:
//!
//! 1. 生产路径(`crates/**/src/**`、`apps/**/src/**`)零 `compile_ptx*` /
//!    `nvrtcCompileProgram` 调用 —— 设备码只允许来自构建期资产;
//! 2. `cudarc::nvrtc` 的**唯一合法用法**是 `Ptx::from_binary`(纯数据类型
//!    包装,`load_module` 入参),不许出现 `compile_ptx`/`CompileOptions`;
//! 3. 例外白名单 = 测试域(`tests/`、`#[cfg(test)]` 同文件)+ `examples/`
//!    (探针/文档用途,不进生产二进制)。
//!
//! 为什么是源码扫描而不是运行期断言:运行期断言只能证明"本轮没编译",
//! 证明不了"代码里没有编译入口"(未被走到的分支 / 新加的调用点都漏)。
//! 扫描是**结构性**判据,与 `op_tables.rs` 的 `.cu` 互证同款纪律。

use std::path::{Path, PathBuf};

/// 生产源根(相对 workspace 根);按此顺序扫描
const ROOTS: &[&str] = &["crates", "apps"];

/// 禁词(生产路径出现即红)
fn banned(line: &str) -> Option<&'static str> {
    // 白名单:纯数据类型包装(不含编译语义)
    if line.contains("Ptx::from_binary") || line.contains("PtxKind") {
        return None;
    }
    [
        "compile_ptx",
        "nvrtcCompileProgram",
        "CompileOptions",
        "nvrtcCreateProgram",
        "nvrtcGetCUBIN",
        "nvrtcGetPTX",
    ]
    .into_iter()
    .find(|pat| line.contains(pat))
}

/// 递归收集 `*.rs`(跳过 `tests/`、`examples/`、`benches/`、`target/`)
fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if p.is_dir() {
            // tests/ 与 examples/ 是**例外白名单**(探针自由编译源)
            if matches!(name, "tests" | "examples" | "benches" | "target") {
                continue;
            }
            collect(&p, out);
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

#[test]
fn production_paths_never_compile_cuda_at_runtime() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace 根")
        .to_path_buf();
    let mut files = Vec::new();
    for r in ROOTS {
        collect(&root.join(r), &mut files);
    }
    assert!(files.len() > 50, "扫描面异常小({} 个 .rs)—— 路径推导错了?", files.len());

    let mut hits: Vec<String> = Vec::new();
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        // 同文件 `#[cfg(test)]` 段:豁免(测试域允许自己编译源做对拍)
        let prod = match text.find("#[cfg(test)]") {
            Some(i) => &text[..i],
            None => &text[..],
        };
        for (ln, line) in prod.lines().enumerate() {
            // 注释行也报:文档里的"曾经这么干"要显式改写,不留误导
            if let Some(pat) = banned(line) {
                let rel = f.strip_prefix(&root).unwrap_or(f).display().to_string();
                hits.push(format!("{rel}:{} [{pat}] {}", ln + 1, line.trim()));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "零运行期编译门被破:生产路径出现 NVRTC 编译调用(设备码只能来自构建期 cubin 资产,\
         见设计 §15.3/§15.5;测试/示例请放 tests/ 或 examples/):\n  {}",
        hits.join("\n  ")
    );
}

/// 反证面:门的**判据本身**要能抓到违规(防"扫描器坏了 ⇒ 永远绿")
#[test]
fn banned_matcher_is_sensitive() {
    assert_eq!(banned("let ptx = compile_ptx_with_opts(src, opts)?;"), Some("compile_ptx"));
    assert_eq!(banned("let _ = nvrtcCreateProgram(&mut p, src, ..);"), Some("nvrtcCreateProgram"));
    assert_eq!(banned("cudarc::nvrtc::CompileOptions::default()"), Some("CompileOptions"));
    // 合法用法不得误伤
    assert_eq!(banned("ctx.load_module(Ptx::from_binary(bytes.to_vec()))"), None);
    assert_eq!(banned("// cubin 载入走 driver API,零编译"), None);
}
