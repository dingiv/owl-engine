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

/// 生产源根(相对 workspace 根);按此顺序扫描。
///
/// ⚠️ **新增顶层代码根必须登记在此**(2026-10-15 审查补):旧实现只扫
/// `crates`/`apps` ⇒ 新加的顶层 crate(如 `tools/`)自动落在门外面。
/// 配套自检见 `production_paths_never_compile_cuda_at_runtime` 的 ROOTS 覆盖断言。
const ROOTS: &[&str] = &["crates", "apps", "tools"];

/// 禁词(生产路径出现即红)
///
/// ⚠️ **不再有"白名单短路"**(2026-10-15 审查定谳):旧实现先查白名单
/// (`Ptx::from_binary` / `PtxKind`)再查禁词 ⇒ 一行**同时**含两者即被豁免
/// (`let p = Ptx::from_binary(compile_ptx(..)?)` 这类组合),等于一行级绕过。
/// 现在:禁词命中即报 —— 白名单本就多余(`Ptx::from_binary` 不含任何禁词),
/// 它唯一的实际效果就是开这个口子。
fn banned(line: &str) -> Option<&'static str> {
    BANNED.iter().copied().find(|pat| line.contains(pat))
}

/// 禁词表(生产路径出现即红)
const BANNED: &[&str] = &[
    "compile_ptx",
    "nvrtcCompileProgram",
    "CompileOptions",
    "nvrtcCreateProgram",
    "nvrtcGetCUBIN",
    "nvrtcGetPTX",
];

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
    // 自检:顶层任何带 Cargo.toml 的目录都必须在 ROOTS 里(否则门自动失明)
    for e in std::fs::read_dir(&root).expect("读 workspace 根").flatten() {
        let p = e.path();
        if !p.is_dir() || !p.join("Cargo.toml").is_file() {
            continue;
        }
        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        assert!(
            ROOTS.contains(&name),
            "顶层代码根 {name}/ 不在 ROOTS ⇒ 零运行期编译门对它失明(请登记)"
        );
    }
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
    // **混合行必报**(旧白名单短路的绕过口;2026-10-15 审查定谳)
    assert_eq!(
        banned("let p = Ptx::from_binary(compile_ptx(src, opts)?);"),
        Some("compile_ptx"),
        "一行同时含白名单词与禁词时不得被豁免"
    );
}

// ============================================================================
// §2 **GGUF 族的源码级 I-4 门**(2026-10-19 gguf 轮;设计 §六 I-4 / REQ-GGUF-06)
//
// I-4 原文:`cu/gguf/*.cu` 无 `cudaMalloc*` / `cudaMemset*` /
// `cudaGetDeviceProperties`。为什么必须是**源码级**门(而不是运行期断言):
// ① 账外分配在捕获期才炸(A5.2:一切分配过账本),而运行期断言只证明"本轮没踩";
// ② 运行期设备查询(H2)是"只在别的卡/别的 CC 才走到"的隐蔽分支 ——
//    正是 `mistral.rs flashinfer` 与上游 `gguf_gemm()` 的原始形态。
//
// ⚠️ **注释剥离是刻意的**:本族 `.cu` 的**头注逐字引用了这些禁词**去讲"上游这么干、
// 本仓改掉了"(H1/H2 条目)—— 引文不是违规。故本门只看**代码行**
// (`//` 行注释与 `/* */` 块注释先剔),与 §1 的 `.rs` 门(`注释行也报`)口径不同,
// 差别及理由写在这里,不许两处各说各话。
// ============================================================================

/// GGUF 族源码根(相对 workspace 根)
const GGUF_CU_ROOT: &str = "crates/kernels/cu/gguf";

/// I-4 禁词(设备源码里出现即红)
const GGUF_BANNED: &[&str] = &[
    "cudaMalloc",
    "cudaFree",
    "cudaMemset",
    "cudaGetDeviceProperties",
    "cudaMemGetInfo",
];

/// 剥掉 C/C++ 注释(块注释先、行注释后;字符串里的 `//` 本族源不存在,不做词法级处理)
fn strip_c_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let (mut i, mut in_block) = (0usize, false);
    let mut in_line = false;
    while i < bytes.len() {
        let two = |k: usize| -> &[u8] { &bytes[k..(k + 2).min(bytes.len())] };
        if in_line {
            if bytes[i] == b'\n' {
                in_line = false;
                out.push('\n');
            } else {
                out.push(' ');
            }
            i += 1;
        } else if in_block {
            if two(i) == b"*/" {
                in_block = false;
                out.push_str("  ");
                i += 2;
            } else {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
        } else if two(i) == b"/*" {
            in_block = true;
            out.push_str("  ");
            i += 2;
        } else if two(i) == b"//" {
            in_line = true;
            out.push_str("  ");
            i += 2;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

#[test]
fn gguf_kernels_never_allocate_or_query_device() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace 根")
        .to_path_buf();
    let dir = root.join(GGUF_CU_ROOT);
    let mut files: Vec<PathBuf> = Vec::new();
    collect_cu(&dir, &mut files);
    assert!(
        files.len() >= 3,
        "扫描面异常小({} 个 .cu/.cuh,期望 ≥3:gemm/quant/rows)—— 路径推导错了?",
        files.len()
    );
    let mut hits: Vec<String> = Vec::new();
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        let code = strip_c_comments(&text);
        for (ln, line) in code.lines().enumerate() {
            for pat in GGUF_BANNED {
                if line.contains(pat) {
                    let rel = f.strip_prefix(&root).unwrap_or(f).display().to_string();
                    hits.push(format!("{rel}:{} [{pat}] {}", ln + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "I-4 被破:GGUF 族设备源码出现分配/设备查询调用(设计 §六 I-4 / REQ-GGUF-06;\n  \
         A5.2 一切分配过账本 + 热路径禁 host 查询):\n  {}",
        hits.join("\n  ")
    );
    eprintln!("[I-4] GGUF 族 {} 个源文件零 cudaMalloc*/Free/Memset*/GetDeviceProperties ✓", files.len());
}

/// 递归收集 `.cu`/`.cuh`(与 [`collect`] 同律的姊妹;源码门专用)
fn collect_cu(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_cu(&p, out);
        } else if matches!(p.extension().and_then(|s| s.to_str()), Some("cu" | "cuh")) {
            out.push(p);
        }
    }
}

/// 反证面:注释剥离与禁词匹配都要**能抓到违规**(防"扫描器坏了 ⇒ 永远绿")
#[test]
fn gguf_i4_matcher_is_sensitive() {
    let viol = "static __device__ void f() { void * p; cudaMalloc(&p, 64); }";
    let code = strip_c_comments(viol);
    assert!(GGUF_BANNED.iter().any(|p| code.contains(p)), "禁词匹配失灵");
    // 引文(注释里提到禁词)必须**不算违规** —— 否则头注的 H1/H2 条目会误红
    let doc = "// 上游 gguf_gemm() 用 cudaGetDeviceProperties 查 CC\n/* cudaMallocAsync 装中间缓冲 */\n";
    let code = strip_c_comments(doc);
    assert!(
        !GGUF_BANNED.iter().any(|p| code.contains(p)),
        "注释剥离漏了(引文被当成代码)"
    );
    assert!(code.matches('\n').count() >= 1, "行号保真丢了");
}
