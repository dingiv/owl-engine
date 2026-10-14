//! **NULL 流同步原语收口门**(A13 追加,2026-10-15)
//!
//! # 判据
//!
//! 全仓源码里操作 **NULL 流**的 raw driver 同步原语
//! (`memcpy_htod_sync` / `memcpy_dtoh_sync` / `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2`)
//! 只允许出现在**白名单文件**;且 `ffi.rs` 的再导出必须是 **`pub(crate)`**
//! (不许对 crate 外开 NULL 流后门)。
//!
//! # 为什么要这条门
//!
//! A13(27B chunked GDN 末行 hidden 确定性算错 → pi 场景"不发工具")的病根就是
//! **NULL 流同步拷贝写流序异步分配**:`DeviceRes::upload` 用 `cuMemcpyHtoD`
//! (NULL 流)写 `stream.alloc`(流序 `cudaMallocAsync`)刚给的地址 ⇒ 两者零序,
//! 写被**静默丢弃**(compute-sanitizer:`Use-before-alloc`)。
//!
//! 同族事故在本仓**已经踩过三次**(2026-10-09 “meta 表读回全零”、`xqa.rs` 的
//! consts 回读、`gdn_chunked.rs:453` 的 diag 回读),每次都是**本地补 sync 绕过**;
//! 没有机器判据时,第四处迟早会出现。本门把"再犯"变成编译后立刻可见的红。

use std::path::{Path, PathBuf};

/// 白名单:(相对仓根路径, 理由)。命中白名单外 = 红。
const ALLOW: &[(&str, &str)] = &[
    ("crates/backends/cuda/src/ffi.rs", "受控再导出(必须 pub(crate))"),
    (
        "crates/backends/cuda/src/server/mod.rs",
        "GpuRes 三处:upload(legacy 位)/ upload_block(legacy 位)/ download(已内建排空)",
    ),
    (
        "crates/backends/cuda/tests/raw_sync_primitives_gate.rs",
        "本门自身(token 表 + 断言;非调用点 —— 自证豁免)",
    ),
];

/// 被门钉住的 token
const TOKENS: &[&str] = &[
    "memcpy_htod_sync",
    "memcpy_dtoh_sync",
    "cuMemcpyHtoD_v2",
    "cuMemcpyDtoH_v2",
];

fn workspace_root() -> PathBuf {
    // 本 crate = <root>/crates/backends/cuda ⇒ 上三级 = 仓根
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("仓根 canonicalize")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if p.is_dir() {
            if matches!(name.as_str(), "target" | ".git" | "node_modules" | ".venv") {
                continue;
            }
            walk(&p, out);
        } else if name.ends_with(".rs") {
            out.push(p);
        }
    }
}

/// 去行注释(`//` 之后);本门只关心**代码面**命中,注释里提名字是允许的
/// (A13 的三处注释都是"病史记载",不该被自己的门绊倒)
fn code_of(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

#[test]
fn null_stream_sync_primitives_are_fenced() {
    let root = workspace_root();
    let mut files = Vec::new();
    walk(&root.join("crates"), &mut files);
    assert!(
        files.len() > 50,
        "扫描面异常小({} 个 .rs)—— 仓根判定错了?",
        files.len()
    );

    // 1) 逐行扫代码面(token 命中)
    let mut hits: Vec<(String, String, usize)> = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(&root).unwrap_or(f).to_string_lossy().to_string();
        let Ok(txt) = std::fs::read_to_string(f) else { continue };
        for (ln, raw) in txt.lines().enumerate() {
            let code = code_of(raw);
            if let Some(t) = TOKENS.iter().find(|t| code.contains(**t)) {
                hits.push((rel.clone(), (*t).to_string(), ln + 1));
            }
        }
    }

    // 2) 白名单外命中 = 红
    let allowed: Vec<&str> = ALLOW.iter().map(|(p, _)| *p).collect();
    let bad: Vec<&(String, String, usize)> = hits
        .iter()
        .filter(|(p, _, _)| !allowed.contains(&p.as_str()))
        .collect();
    assert!(
        bad.is_empty(),
        "★ NULL 流同步原语出现在白名单之外(A13 病根,禁再犯):\n{}\n\
         修法:同流 `memcpy_htod_async(..., COMPUTE)`(上传)或拷贝前排空目标流(回读);\n\
         确需新增白名单走 backends/README 准入审核并在此登记理由。",
        bad.iter()
            .map(|(p, t, l)| format!("  {p}:{l}  {t}"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    // 3) 导出面必须 pub(crate)(不对 crate 外开放 NULL 流后门)
    let ffi_path = root.join("crates/backends/cuda/src/ffi.rs");
    let ffi = std::fs::read_to_string(&ffi_path).expect("ffi.rs 可读");
    let export = ffi
        .lines()
        .find(|l| code_of(l).contains("memcpy_htod_sync"))
        .expect("ffi.rs 必有一条 NULL 流同步原语的再导出(白名单项)");
    assert!(
        export.trim_start().starts_with("pub(crate)"),
        "ffi.rs 的 NULL 流同步原语必须是 pub(crate)(外部走 DeviceRes::upload/download):\n{export}"
    );

    // 4) 白名单项必须真实命中(防空白名单漂移 —— 白名单自己也会腐坏)
    for (p, why) in ALLOW {
        assert!(
            hits.iter().any(|(h, _, _)| h == p),
            "白名单项 {p} 已无命中({why})—— 请同步收敛白名单"
        );
    }

    eprintln!(
        "[gate] NULL 流同步原语:命中 {} 处,全部落在白名单({});导出面 pub(crate) ✓",
        hits.len(),
        hits.iter()
            .map(|(p, _, l)| format!("{p}:{l}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
}
