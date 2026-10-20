//! 构建期预编译:marlin W4A16 → .a(nvcc,feature "marlin")。
//! (旧 cu/ops.cu → PTX 预编链随 cuda_ops 退役删除,2026-10-12;
//!  现役内核编译 = server nvrtc 懒编译 + AOT cubin 资产,零 build.rs PTX 面。)

use std::env;
use std::path::PathBuf;
use std::process::Command;


// ============================================================================
// native 核**构建期**编 cubin(2026-10-13 用户裁决"零运行期编译,设备码一律 cubin")
//
// 用户定调(2026-10-13):"编译的时候就要很明确地感知显卡架构;编译选项里肯定
// 会有这种东西;针对不同的架构,编出来的东西不一样;可以在编译时选择支持多少种架构"。
//
// # 源清单 = `include_str!` 闭包(零第二份清单)
//
// 编译哪些 `.cu` **不另立清单**:扫描 `src/**/*.rs` 里所有
// `include_str!("…/*.cu")`,解析成真实路径去重 —— 命中的正是
// `KernelSpec::SOURCE`(就地 include_str,R5)+ `list.rs` §0(未建表“仅引入”算子)。
// 新增算子 = 加表(自带 include_str)⇒ 构建期自动纳入,物理上不可能
// "加了表但忘了编"("两处对不上"的 E3/E15 根因在构建面也堵死)。
// 家族自管源(marlin 设备码 / xqa / flashinfer)不在此闭包内,天然排除。
//
// # 产物
//
// - `$OUT_DIR/cubins/<asset>.fatbin`(多代际容器;见下 -fatbin 注)
// - `$OUT_DIR/cubin_assets.rs` —— 生成的资产登记表(**源哈希 → 资产字节**),
//   经 `kernels::family::cubins` 编进 lib;运行期按源文本查资产,零路径知识。
//
// # 旋钮
//
// - `OWL_CUBIN_ARCHS` = 逗号分隔目标架构(缺省 `sm_86`);多档 ⇒ 一个 fatbin 含多代际
// - `OWL_NVCC`        = nvcc 路径(缺省 PATH)
// - `OWL_CUBIN_SKIP`  = 跳过构建期编 cubin(排障:纯 NVRTC 基线对照)
// ============================================================================
fn cubin_archs() -> Vec<String> {
    env::var("OWL_CUBIN_ARCHS")
        .unwrap_or_else(|_| "sm_86".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// FNV-1a 64 —— **必须**与 `kernels::family::cubins::hash_source` 逐位一致
/// (两处一份算法:自洽性由 `cubins::tests` 对全资产回算锁死)。
fn hash_source(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// 递归列 `dir` 下的 `.rs`(`src/` 扫描用;零 walkdir 依赖)
fn rs_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// 提取 `include_str!("X")` 里以 `.cu` 结尾的路径,按 rs 文件所在目录解析。
///
/// 解析器刻意**严格**(裸字符串字面量 + 单行):解析不了又看着像 `.cu`
/// 的一律 panic —— 静默漏编会退化成运行期 NVRTC,正是本机制要消灭的东西。
fn scan_cu_sources(root: &std::path::Path) -> Vec<PathBuf> {
    let mut rs = Vec::new();
    rs_files(&root.join("src"), &mut rs);
    println!("cargo:rerun-if-changed=src");
    for f in &rs {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    // E 轮(2026-10-13):源文本不再进二进制 ⇒ 发现规则从"扫 include_str! 闭包"
    // 改成"扫**资产 id 字面量**"(`const SRC: &'static str = "cu/…/x.cu";`,
    // `KernelSpec::SRC` 与 `list.rs` §0 常量两处同形)。规则仍是**零第二份清单**:
    // 加表即自动纳入;路径不存在 = 构建期直接失败(fail-closed)。
    let mut found: Vec<PathBuf> = Vec::new();
    let mut ids: Vec<(String, PathBuf)> = Vec::new();
    for f in &rs {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        for line in text.lines() {
            if !line.contains("const ") { continue; }
            let Some(pos) = line.find("\"cu/") else { continue };
            let Some(end) = line[pos + 1..].find('"') else { continue };
            let id = &line[pos + 1..pos + 1 + end];
            if !id.ends_with(".cu") { continue; }
            let abs = root.join(id);
            let canon = std::fs::canonicalize(&abs).unwrap_or_else(|e| {
                panic!("资产 id {id:?}(来自 {})解析失败: {e} —— 路径写错或文件已挪", f.display())
            });
            if !found.contains(&canon) { found.push(canon.clone()); }
            if !ids.iter().any(|(i, _)| i == id) { ids.push((id.to_string(), canon)); }
        }
    }
    // .cu 自身也要进 rerun 集合(内容变了要重编)
    for c in &found { println!("cargo:rerun-if-changed={}", c.display()); }
    found.sort();
    found
}

/// `.cu` 源里"有定义"的核符号集合(字面核 + 单层宏实例化;判据与运行期
/// 旧 `source_defines` 同源,现改为**构建期一次算好写进清单**)。
fn symbols_of(text: &str) -> Vec<String> {
    let clean: String = text
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("\\\n", " ");
    let mut out: Vec<String> = Vec::new();
    // 1) 字面核:`__global__ void NAME(`
    let mut rest = clean.as_str();
    while let Some(p) = rest.find("__global__ void ") {
        rest = &rest[p + "__global__ void ".len()..];
        let name: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        // 全大写 = 宏形参(宏体里 `void NAME(...)` 形态),不是核名 —— 真核名含小写
        if !name.is_empty()
            && rest[name.len()..].trim_start().starts_with('(')
            && name.chars().any(|c| c.is_ascii_lowercase())
        {
            out.push(name);
        }
    }
    // 2) 宏实例化:先找"体里有 void"的命名宏,再取其调用行的首参
    let mut macros: Vec<String> = Vec::new();
    for line in clean.lines() {
        let t = line.trim_start();
        let Some(body) = t.strip_prefix("#define ") else { continue };
        let Some((mac, _)) = body.split_once('(') else { continue };
        let mac = mac.trim();
        if mac.is_empty() || !mac.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') { continue; }
        if body.contains("void ") { macros.push(mac.to_string()); }
    }
    for line in clean.lines() {
        let t = line.trim_start();
        let Some((head, rest)) = t.split_once('(') else { continue };
        if !macros.iter().any(|m| m == head) { continue; }
        let first = rest.split(',').next().unwrap_or("").trim();
        // 全大写 = 宏形参(NAME/PARAMS/…),不是核名 —— 真核名一律含小写
        if !first.is_empty()
            && first.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && first.chars().any(|c| c.is_ascii_lowercase())
        {
            out.push(first.to_string());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// **增量闸**:fatbin 与 `.stamp` 都在且指纹一致 ⇒ 跳过 nvcc。
///
/// 为什么必须有:build.rs 的 rerun 面包含 `src/**/*.rs`(源清单 = include_str!
/// 闭包),**改任何一个 Rust 文件都会重跑 build.rs** ⇒ 没有本闸的话每次 Rust
/// 编辑都要把 20 个 `.cu` 重编一遍(实测数十秒)。指纹 = 源 mtime+长度 +
/// 架构集 + 额外编译选项;任一变 ⇒ 重编。
/// **头文件依赖**(2026-10-19 定谳):fatbin 指纹原先只看 `.cu` 本身 ⇒ 改
/// `#include` 进来的头(如 `cu/gguf/vendor/common.cuh`)**不触发重编**,
/// 现场表现为"改了开关但核行为没变"(MMQ 张量核↔dp4a 切换时踩到:几何在 Rust
/// 侧变了、cubin 还是旧的 ⇒ 数值门给 1.0)。
///
/// 规则:凡 `.cu` 文本里出现 `#include "vendor/`(vendored 第三方树),
/// 就把该目录下**所有文件**的(相对路径, 长度, mtime)混进指纹。
fn vendor_stamp(src: &std::path::Path) -> String {
    let Ok(text) = std::fs::read_to_string(src) else { return String::new() };
    if !text.contains("#include \"vendor/") {
        return String::new();
    }
    let dir = src.parent().map(|p| p.join("vendor")).unwrap_or_default();
    let mut out = String::from(" vendor=");
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let mut stack = vec![dir];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                files.push(p);
            }
        }
    }
    files.sort();
    for p in files {
        let md = std::fs::metadata(&p).expect("stat vendor 文件");
        let mtime = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        out.push_str(&format!(
            "{}:{}:{};",
            p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
            md.len(),
            mtime
        ));
    }
    out
}

fn fingerprint(src: &std::path::Path, archs: &[String], extra: &[&str]) -> String {
    let md = std::fs::metadata(src).expect("stat .cu");
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!(
        "src={} len={} mtime={} archs={} flags={}{}",
        src.display(),
        md.len(),
        mtime,
        archs.join(","),
        extra.join(" "),
        vendor_stamp(src)
    )
}

/// 编一个 `.cu` → fatbin(带增量闸);返回是否真的调了 nvcc
fn compile_fatbin(
    nvcc: &str,
    src: &std::path::Path,
    dst: &std::path::Path,
    archs: &[String],
    extra: &[&str],
) -> bool {
    let fp = fingerprint(src, archs, extra);
    let stamp = dst.with_extension("stamp");
    if dst.exists() && std::fs::read_to_string(&stamp).ok().as_deref() == Some(fp.as_str()) {
        return false;
    }
    let mut cmd = Command::new(nvcc);
    cmd.args(["-fatbin", "-O3", "-std=c++17", "--expt-relaxed-constexpr"]);
    for a in archs {
        let cc = a.replace("sm_", "compute_");
        cmd.arg("-gencode").arg(format!("arch={cc},code={a}"));
    }
    cmd.args(extra).arg(src).arg("-o").arg(dst);
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("cubin: failed to spawn {nvcc}: {e}"));
    assert!(status.success(), "cubin: nvcc failed for {}", src.display());
    std::fs::write(&stamp, fp).expect("write stamp");
    true
}

fn build_native_cubins(
    out_dir: &std::path::Path,
    xqa_rows: &[(&'static str, PathBuf)],
    gdn_rows: &[(&'static str, PathBuf)],
) {
    let manifest_path = out_dir.join("cubin_assets.rs");
    // ⚠️ 必须在**两条分支之外**发这条指令:指令只在 build.rs 真的跑过那次才被
    // cargo 记住 —— 上一轮没发,这一轮改 OWL_CUBIN_SKIP 就不会触发重跑
    // (实测踩坑:负例验证拿到的还是上一轮的资产表)
    println!("cargo:rerun-if-env-changed=OWL_CUBIN_SKIP");
    if env::var_os("OWL_CUBIN_SKIP").is_some() {
        println!("cargo:warning=owl-kernels: OWL_CUBIN_SKIP —— 跳过构建期 cubin(纯 NVRTC 基线)");
        std::fs::write(&manifest_path, EMPTY_ASSETS_RS).expect("write cubin_assets.rs");
        return;
    }
    let nvcc = env::var("OWL_NVCC").unwrap_or_else(|_| "nvcc".into());
    let archs = cubin_archs();
    assert!(!archs.is_empty(), "OWL_CUBIN_ARCHS 为空");
    let cuda_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let sources = scan_cu_sources(&cuda_root);
    assert!(
        sources.len() >= 15,
        "构建期 cubin 源清单只有 {} 个 —— include_str! 扫描器失效(应 ≥15 个 native 源)",
        sources.len()
    );
    let cu_root = cuda_root.join("cu");
    let bin_dir = out_dir.join("cubins");
    std::fs::create_dir_all(&bin_dir).expect("mkdir cubins");

    // (module 名, 源哈希, 源长度, 源路径, fatbin 路径, 符号清单)
    let mut rows: Vec<(String, u64, usize, PathBuf, PathBuf, Vec<String>)> = Vec::new();
    let mut rebuilt = 0usize;
    let mut src_ids: Vec<String> = Vec::new();
    for src in &sources {
        println!("cargo:rerun-if-changed={}", src.display());
        let rel = src.strip_prefix(&cu_root).unwrap_or(src);
        let asset = rel
            .with_extension("")
            .to_string_lossy()
            .replace(['/', '\\'], "__")
            .trim_end_matches('.')
            .to_string();
        let dst = bin_dir.join(format!("{asset}.fatbin"));
        // ⚠️ nvcc 规则:`--cubin` 不允许"多个 GPU code instance"(实测
        // `Option '--cubin' is not allowed when compiling for multiple GPU code instances`)
        // ⇒ 多代际必须用 `-fatbin`(内含多份 cubin;driver 载入 API 完全相同,
        // 运行期由驱动按设备挑最匹配那份)。单代际两者等价 ⇒ 统一用 fatbin。
        if compile_fatbin(&nvcc, src, &dst, &archs, &[]) {
            rebuilt += 1;
        }
        let text = std::fs::read_to_string(src).expect("read .cu");
        rows.push((asset, hash_source(&text), text.len(), src.clone(), dst, symbols_of(&text)));
        // 资产 id = cu/ 相对路径(与 KernelSpec::SRC 同形)
        src_ids.push(format!("cu/{}", rel.to_string_lossy().replace('\\', "/")));
    }

    // ---- 生成资产登记表(源哈希 → 资产;运行期零路径知识)----
    let mut out = String::from(
        "// @generated by crates/kernels/build.rs —— **勿手改**(改 build.rs 或加算子表即可)。\n\
         //\n\
         // native 核的构建期 fatbin 资产登记表:源文本哈希 → 资产字节。\n\
         // 查表键 = 源**内容**,不是名字 —— 与 `KernelSpec::SOURCE`(就地\n\
         // include_str,R5)天然一一对应,不存在第二份名字→资产映射。\n\n",
    );
    out.push_str("/// 构建期编出的代际集(设备 sm ∉ 本集 ⇒ 装配期拒启;见 §15.4 架构门)\n");
    out.push_str("pub const CUBIN_ARCHS: &[&str] = &[");
    out.push_str(&archs.iter().map(|a| format!("\"{a}\"")).collect::<Vec<_>>().join(", "));
    out.push_str("];\n\n");
    out.push_str("/// 一个 `.cu` 文件产出的 fatbin 资产(E 轮:不再携带源文本 —— 运行期只要身份 + 字节)\npub struct CubinAsset {\n    /// **资产 id** = `cu/` 相对路径(与 `KernelSpec::SRC` 同形;查表键)\n    pub id: &'static str,\n    /// 模块缓存键/诊断名(`cu/` 相对路径,`/` → `__`)\n    pub module: &'static str,\n    /// 源内容 FNV-1a 64(**构建期指纹**;自洽门回算比对用)\n    pub src_hash: u64,\n    /// fatbin 字节(`include_bytes!`;多代际容器)\n    pub bytes: &'static [u8],\n}\n\n");
    out.push_str("pub const CUBIN_ASSETS: &[CubinAsset] = &[\n");
    let mut syms = String::from("// @generated by crates/kernels/build.rs —— **勿手改**。\n//\n// 资产符号清单(构建期从 .cu 解析;非 device 门 —— boot 期池清单门要用)。\npub const ASSET_SYMBOLS: &[(&str, &[&str])] = &[\n");
    for ((asset, hash, _len, _src, bin, _symbols), src_id) in rows.iter().zip(src_ids.iter()) {
        out.push_str(&format!(
            "    CubinAsset {{ id: \"{id}\", module: \"{asset}\", src_hash: 0x{hash:016x}, bytes: include_bytes!(\"{bin}\") }},\n",
            id = src_id,
            bin = bin.display()
        ));
    }
    out.push_str("];\n\n");
    for (_, _, _, _, _, symbols) in &rows {
        let list = symbols.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(", ");
        syms.push_str(&format!("    (\"@@ID@@\", &[{list}]),\n"));
    }
    syms.push_str("];\n");
    for ((_, _, _, _, _, _), id) in rows.iter().zip(src_ids.iter()) {
        syms = syms.replacen("\"@@ID@@\"", &format!("\"{id}\""), 1);
    }

    out.push_str(&xqa_table_src(xqa_rows));
    out.push_str(&gdn_table_src(gdn_rows));
    // ---- 资产指纹(见下方注释:这是**强制 crate 重编**的唯一可靠手段)----
    let mut fp: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |bytes: &[u8]| {
        for b in bytes.iter().step_by(7) {
            fp ^= *b as u64;
            fp = fp.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (_, _, _, _, bin, _) in &rows {
        if let Ok(b) = std::fs::read(bin) {
            mix(&b);
        }
    }
    for (_, bin) in xqa_rows.iter().chain(gdn_rows.iter()) {
        if let Ok(b) = std::fs::read(bin) {
            mix(&b);
        }
    }
    println!("cargo:rustc-env=OWL_ASSET_FP={fp:016x}");
    std::fs::write(&manifest_path, out).expect("write cubin_assets.rs");
    std::fs::write(out_dir.join("cubin_symbols.rs"), syms).expect("write cubin_symbols.rs");
    println!(
        "cargo:warning=owl-kernels: 构建期 cubin {} 个源文件 ({}) —— {} 个源可零 NVRTC 发射{}",
        rows.len(),
        archs.join(","),
        rows.len(),
        if rebuilt == 0 { "(全增量命中,零 nvcc)".to_string() } else { format!("(本次重编 {rebuilt} 个)") }
    );
    println!("cargo:rerun-if-env-changed=OWL_CUBIN_ARCHS");
}

// ============================================================================
// XQA 资产(2026-10-13 C3a:`.a` → 构建期 fatbin + **Rust host 发射器**)
//
// 与 native 面的差别:**同一份源、不同编译选项**(宏变体),所以键是**变体名**
// 而不是源哈希(源哈希区分不了两份实例化)。两个变体:
//   verify → `-DSPEC_DEC=1`                (`M_TILESIZE` 缺省 32;T=8 伪序列)
//   decode → `-DSPEC_DEC=1 -DM_TILESIZE=16`(q=1 只需 6 行,收窄 tile 减浪费)
//
// `-DGENERATE_CUBIN` = 上游自带的"只出设备码"模式:`CUBIN_EXPORT` 变
// `extern "C"` ⇒ `kernel_mha` / `owl_xqa_launch_consts` 落成 **STB_GLOBAL**
// (不加它,单 TU cubin 会把 `__global__` 之外的符号全内部化,`cuModuleGetFunction`
// 取不到;实测详录 §15.8)。host 发射器已搬 Rust(`server/xqa.rs`)⇒ 无 `.a`、
// 无 `libstdc++` 链接、无符号改名 hack(两份实例化是**两个模块**,同名不撞)。
// ============================================================================
fn build_xqa_assets(out_dir: &std::path::Path) -> Vec<(&'static str, PathBuf)> {
    if env::var_os("CARGO_FEATURE_XQA").is_none() {
        return Vec::new();
    }
    let nvcc = env::var("OWL_NVCC").unwrap_or_else(|_| "nvcc".into());
    let archs = cubin_archs();
    // 头文件全量追踪(任一改动 ⇒ 重编)
    for h in [
        "cu/xqa/mha.cu", "cu/xqa/mha.h", "cu/xqa/defines.h", "cu/xqa/utils.h",
        "cu/xqa/utils.cuh", "cu/xqa/mhaUtils.cuh", "cu/xqa/mha_components.cuh",
        "cu/xqa/mha_stdheaders.cuh", "cu/xqa/mma.cuh", "cu/xqa/barriers.cuh",
        "cu/xqa/ldgsts.cuh", "cu/xqa/cuda_hint.cuh", "cu/xqa/specDec.h",
        "cu/xqa/platform.h", "cu/xqa/hostUtils.h",
    ] {
        println!("cargo:rerun-if-changed={h}");
    }
    // 通用实例化参数 = owl 生产形状(见 cu/xqa/README.md 头注)
    const COMMON: &[&str] = &[
        "-DNDEBUG=1",
        "-DBEAM_WIDTH=1",
        "-DUSE_INPUT_KV=0",
        "-DUSE_CUSTOM_BARRIER=1",
        "-DTOKENS_PER_PAGE=32",
        "-DHEAD_ELEMS=256",
        "-DHEAD_GRP_SIZE=6",
        "-DINPUT_FP16=1",
        "-DDTYPE=__half",
        "-DCACHE_ELEM_ENUM=2",
        "-DSLIDING_WINDOW=0",
        "-DLOW_PREC_OUTPUT=0",
        "-DMLA_WRAPPER=0",
        "-DUSE_SM90_MHA=0",
        "-DGENERATE_CUBIN",
    ];
    // ⚠️ M_TILESIZE=64 一档**不可行**(2026-10-15 实测):内核 `mha.cu:85`
    //    `static_assert(nbValidRows <= 32u)` ⇒ verify 的 48 打包单元
    //    (q=8 × headGrp 6)**结构上**必须分 2 个 token block ⇒ 同 kv head 的
    //    KV 被读两遍是内核设计约束,不是几何选择。
    let specs: [(&'static str, &[&str]); 2] = [
        ("verify", &["-DSPEC_DEC=1"]),
        ("decode", &["-DSPEC_DEC=1", "-DM_TILESIZE=16"]),
    ];
    let dir = out_dir.join("xqa");
    std::fs::create_dir_all(&dir).expect("mkdir xqa assets");
    let mut rows = Vec::new();
    for (name, extra) in specs {
        let dst = dir.join(format!("xqa_{name}.fatbin"));
        let mut flags: Vec<&str> = COMMON.to_vec();
        flags.extend_from_slice(extra);
        let built = compile_fatbin(&nvcc, std::path::Path::new("cu/xqa/mha.cu"), &dst, &archs, &flags);
        println!(
            "cargo:warning=owl-kernels: xqa_{name}.fatbin [{}] {} B{}",
            archs.join(","),
            std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0),
            if built { "" } else { "(增量命中,未重编)" }
        );
        rows.push((name, dst));
    }
    rows
}

// ============================================================================
// GDN chunked 资产(2026-10-13 C4b-β:**opt-in 构建期 AOT 自编**)
//
// 调 `triton/gdn_aot_compile.py`(需 Python+triton 环境),把六核从
// **树内 vendor 的 FLA 源**(`triton/fla/`)+ **配置表**(`triton/gdn_chunked_aot.json`)
// 编成 cubin;产物与入库采集件**逐字节一致**(实测,见设计 §15.11)。
//
// 缺省(不设该 env)= 用入库采集产物 ⇒ 默认构建**不需要** Python/triton。
//
// 旋钮:`GDN_TRITON_PY`(解释器;缺省探工作区 `vllm-test/.venv`)、
//      `GDN_TRITON_DEVICE`(warmup 取 target 的设备,缺省 cuda:0)。
// ============================================================================
fn build_gdn_assets(out_dir: &std::path::Path) -> Vec<(&'static str, PathBuf)> {
    const NAMES: &[&str] = &["cumsum", "kkt", "merge", "wu", "h", "o"];
    // ⚠️ 三条 env 指令必须**无条件**发(cargo 只在 build.rs 真跑过那次才记住它):
    // 写在 early-return 之后 = 缺省构建从不登记 ⇒ 之后设 env 不会触发重跑
    // (实测踩过 OWL_CUBIN_SKIP 同款坑)
    println!("cargo:rerun-if-env-changed=GDN_TRITON_BUILD");
    println!("cargo:rerun-if-env-changed=GDN_TRITON_PY");
    println!("cargo:rerun-if-env-changed=GDN_TRITON_DEVICE");
    // **默认自编**(2026-10-13 用户裁决:既然构建期已引入 Python+triton,
    // 就不留入库采集件)。`GDN_TRITON_BUILD=0` = 显式关(无 triton 环境仍能
    // **编过**这个 crate;代价 = 该族的资产表为空 ⇒ 装配期拒启,见 family 侧)。
    if env::var("GDN_TRITON_BUILD").ok().as_deref() == Some("0") {
        println!("cargo:warning=owl-kernels: GDN_TRITON_BUILD=0 —— 跳过 gdn_chunked 六核自编(资产表为空,该族装配期拒启)");
        return Vec::new();
    }
    let script = "triton/gdn_aot_compile.py";
    println!("cargo:rerun-if-changed={script}");
    let mut deps: Vec<String> = vec![script.to_string()];
    deps.push("triton/gdn_chunked_aot.json".to_string());
    all_files("triton", &mut deps);
    deps.sort();

    let dir = out_dir.join("gdn");
    std::fs::create_dir_all(&dir).expect("mkdir gdn assets");
    let flags = format!(
        "gdn-aot py={:?} dev={:?}",
        env::var("GDN_TRITON_PY").ok(),
        env::var("GDN_TRITON_DEVICE").ok()
    );
    let probe = dir.join("cumsum.cubin");
    if !stamp_hit(&probe, &deps, &flags) {
        let py = env::var("GDN_TRITON_PY").unwrap_or_else(|_| {
            // 本 crate 自己的 uv 环境(`uv sync --project crates/kernels`;声明见
            // crates/kernels/pyproject.toml:triton==3.7.1 + CPU torch)
            // → 退到工作区现成的 venv(vllm-test,见 cu/xqa/README 先例)
            let cands = [".venv/bin/python", "../../../vllm-test/.venv/bin/python"];
            match cands.iter().map(PathBuf::from).find(|p| p.exists()) {
                Some(p) => p.to_string_lossy().into_owned(),
                None => panic!(
                    "gdn_chunked 六核 AOT 需要 Python+triton 环境:先 `uv sync --project crates/kernels` \
                     生成 crates/kernels/.venv,或设 `GDN_TRITON_PY=<venv>/bin/python`;\
                     也可 `GDN_TRITON_BUILD=0` 显式跳过(该族装配期拒启)"
                ),
            }
        });
        // 缺省 `none` = 零 GPU(target 桩);`GDN_TRITON_DEVICE=cuda:N` 切真实设备
        let dev = env::var("GDN_TRITON_DEVICE").unwrap_or_else(|_| "none".into());
        let status = Command::new(&py)
            .arg(script)
            .arg("--out")
            .arg(&dir)
            .arg("--device")
            .arg(&dev)
            .status()
            .unwrap_or_else(|e| panic!("gdn-aot: failed to spawn {py}: {e}"));
        assert!(status.success(), "gdn-aot: 自编失败({script})");
        for n in NAMES {
            let p = dir.join(format!("{n}.cubin"));
            assert!(p.exists(), "gdn-aot: 自编未产出 {n}.cubin");
        }
        stamp_write(&probe, &deps, &flags);
        println!(
            "cargo:warning=owl-kernels: GDN chunked 六核 **AOT 自编**完成(树内 FLA 源 triton/fla/ + 配置表 triton/gdn_chunked_aot.json)"
        );
    } else {
        println!("cargo:warning=owl-kernels: GDN chunked 六核 AOT 增量命中(零 triton)");
    }
    NAMES.iter().map(|n| (*n, dir.join(format!("{n}.cubin")))).collect()
}

/// `OWL_CUBIN_SKIP` 档的空登记表(纯 NVRTC 基线对照;同一文件形态免分支)
const EMPTY_ASSETS_RS: &str = "// @generated (OWL_CUBIN_SKIP)\npub const CUBIN_ARCHS: &[&str] = &[];\n\npub struct CubinAsset {\n    pub id: &'static str,\n    pub module: &'static str,\n    pub src_hash: u64,\n    pub bytes: &'static [u8],\n}\n\npub const CUBIN_ASSETS: &[CubinAsset] = &[];\n\npub const XQA_ASSETS: &[(&str, &[u8])] = &[];\n\npub fn xqa_asset(_name: &str) -> Option<&'static [u8]> {\n    None\n}\n\npub const GDN_ASSETS: &[(&str, &[u8])] = &[];\n\npub fn gdn_asset(_name: &str) -> Option<&'static [u8]> {\n    None\n}\n";

/// 生成 GDN chunked 资产表(opt-in 自编;空表 = 用入库采集件)
fn gdn_table_src(rows: &[(&'static str, PathBuf)]) -> String {
    let mut t = String::from(
        "/// GDN chunked 六核(opt-in `GDN_TRITON_BUILD=1` 构建期 AOT 自编;空 = 用入库采集件)\n\
         pub const GDN_ASSETS: &[(&str, &[u8])] = &[\n",
    );
    for (name, path) in rows {
        t.push_str(&format!("    (\"{name}\", include_bytes!(\"{}\")),\n", path.display()));
    }
    t.push_str("];\n\n/// GDN 资产查表(核名 → cubin 字节;空表返回 None)\npub fn gdn_asset(name: &str) -> Option<&'static [u8]> {\n    GDN_ASSETS.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)\n}\n");
    t
}

/// 生成 XQA 资产表(宏变体:键 = 变体名;见 `build_xqa_assets` 头注)
fn xqa_table_src(rows: &[(&'static str, PathBuf)]) -> String {
    let mut t = String::from(
        "/// XQA 资产(宏变体;键 = 变体名 —— 源同、编译选项不同,源哈希区分不了)\n\
         pub const XQA_ASSETS: &[(&str, &[u8])] = &[\n",
    );
    for (name, path) in rows {
        t.push_str(&format!("    (\"{name}\", include_bytes!(\"{}\")),\n", path.display()));
    }
    t.push_str("];\n\n/// XQA 资产查表(变体名 → fatbin 字节)\npub fn xqa_asset(name: &str) -> Option<&'static [u8]> {\n    XQA_ASSETS.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)\n}\n");
    t
}

/// 多源/多头构建的**增量闸**(与 `compile_fatbin` 同款思路)。
///
/// 指纹 = 每个依赖(源 + 头)的 `len:mtime` + 编译旗子;存 `<dst>.stamp`。
/// 为什么必须有:这些 `.a` 的冷编是分钟级(marlin 4 个模板重 TU ≈ 3.5 min、
/// FI ≈ 1.5 min),而 build.rs 的 rerun 面很宽 —— 没有闸就没有可用的编辑回路。
fn deps_fingerprint(deps: &[String], flags: &str) -> String {
    let mut s = String::new();
    for d in deps {
        let (len, mtime) = match std::fs::metadata(d) {
            Ok(m) => (
                m.len(),
                m.modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|x| x.as_nanos())
                    .unwrap_or(0),
            ),
            Err(_) => (0, 0),
        };
        s.push_str(&format!("{d}:{len}:{mtime}\n"));
    }
    s.push_str("flags=");
    s.push_str(flags);
    s.push('\n');
    s
}

fn stamp_hit(dst: &std::path::Path, deps: &[String], flags: &str) -> bool {
    let stamp = dst.with_extension("stamp");
    dst.exists()
        && std::fs::read_to_string(&stamp).ok().as_deref() == Some(deps_fingerprint(deps, flags).as_str())
}

fn stamp_write(dst: &std::path::Path, deps: &[String], flags: &str) {
    std::fs::write(dst.with_extension("stamp"), deps_fingerprint(deps, flags)).expect("write stamp");
}

/// tag 词表 → `KernelTag` / `KERNEL_TAGS` 生成(2026-10-20)。
///
/// 单一真源 = `tags.toml`;算子侧 `#[kernel_tags("dom.attn", ...)]` 由宏
/// 展开成 `KernelTag::DOM_ATTN` 这类关联常量 —— 写错 tag = 编译期红。
/// build.rs 只生成**词表/位集**,不解析每个算子(算子行由 `build_ops_table`
/// 的 static 提升点负责;tag 随 `KernelMeta` 走)。
fn build_tags(out_dir: &std::path::Path) {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let path = root.join("tags.toml");
    println!("cargo:rerun-if-changed={}", path.display());
    #[derive(serde::Deserialize)]
    struct RawTag {
        id: String,
        desc: String,
        /// **互斥 facet**(空 = 纯描述/非约束)。同一 facet 内是“单选”语义:
        /// 算子只允许带本 facet 中被配置选中的值;无本 facet tag = 通用算子。
        #[serde(default)]
        facet: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct Raw { tag: Vec<RawTag> }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read tags.toml: {e}"));
    let raw: Raw = toml::from_str(&text).expect("parse tags.toml");
    assert!(!raw.tag.is_empty(), "tags.toml 为空");
    assert!(raw.tag.len() <= 64, "tag 数 {} > 64(位集是 u64;扩位集须改 KernelTag::bit)", raw.tag.len());

    let mut ids = std::collections::BTreeSet::new();
    let mut cnames = std::collections::BTreeSet::new();
    for t in &raw.tag {
        assert!(!t.id.is_empty(), "tag id 为空");
        assert!(ids.insert(t.id.clone()), "tag id 重复:{}", t.id);
        let cname: String = t.id.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
            .collect();
        assert!(cnames.insert(cname.clone()), "tag `{}` 的常量名 `{cname}` 与别的 tag 冲突", t.id);
        assert!(
            t.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_' || c == '-'),
            "tag id 只允许小写字母/数字/._-:{}", t.id
        );
        if let Some(f) = &t.facet {
            assert!(
                !f.is_empty() && f.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_' || c == '-'),
                "tag `{}` 的 facet 名非法:{f:?}", t.id
            );
        }
    }

    fn cname(id: &str) -> String {
        id.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
            .collect()
    }

    let mut out = String::from(
        "// @generated by crates/kernels/build.rs from tags.toml —— **勿手改**。\n\
         //\n\
         // 分面 tag 词表:KernelTag(位集) + KERNEL_TAGS(名称/描述)。\n\n",
    );
    out.push_str("/// 算子 tag(分面命名空间;值 = 位下标,`bit()` = u64 位掩码)\n");
    out.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]\n");
    out.push_str("pub struct KernelTag(pub u8);\n\nimpl KernelTag {\n");
    for (i, t) in raw.tag.iter().enumerate() {
        out.push_str(&format!("    /// `{}` — {}\n    pub const {}: Self = Self({i});\n", t.id, t.desc, cname(&t.id)));
    }
    out.push_str("\n    /// 位掩码(u64;最多 64 个 tag)\n    pub const fn bit(self) -> u64 { 1u64 << self.0 }\n");
    out.push_str("    /// 词表下标\n    pub const fn index(self) -> usize { self.0 as usize }\n");
    out.push_str("    /// 词表 id(如 `dom.attn`)\n    pub fn as_str(self) -> &'static str { KERNEL_TAGS[self.0 as usize].id }\n");
    out.push_str("}\n\n");
    out.push_str("/// 一条 tag 的静态描述\n#[derive(Clone, Copy, Debug)]\npub struct KernelTagInfo {\n    pub tag: KernelTag,\n    pub id: &'static str,\n    pub desc: &'static str,\n}\n\n");
    out.push_str("/// 全部 tag(顺序 = tags.toml)\npub const KERNEL_TAGS: &[KernelTagInfo] = &[\n");
    for (i, t) in raw.tag.iter().enumerate() {
        out.push_str(&format!("    KernelTagInfo {{ tag: KernelTag({i}), id: \"{}\", desc: \"{}\" }},\n", t.id, t.desc));
    }
    out.push_str("];\n\n");
    out.push_str(&format!("/// tag 总数\npub const KERNEL_TAG_COUNT: usize = {};\n\n", raw.tag.len()));
    // 互斥 facet 组:同组 tag 是“单选”语义;无组 tag = 非约束。
    let mut facet_order: Vec<String> = Vec::new();
    let mut facet_masks: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for (i, t) in raw.tag.iter().enumerate() {
        if let Some(f) = &t.facet {
            if !facet_masks.contains_key(f) {
                facet_order.push(f.clone());
            }
            *facet_masks.entry(f.clone()).or_insert(0) |= 1u64 << i;
        }
    }
    out.push_str("/// 一个互斥 tag facet(同组 tag 单选;算子无该组 tag = 通用)\n");
    out.push_str("#[derive(Clone, Copy, Debug)]\npub struct KernelTagFacet {\n    pub name: &'static str,\n    pub mask: u64,\n}\n\n");
    out.push_str("/// 参与过滤的互斥 facet(顺序 = 首次出现序)\npub const EXCLUSIVE_TAG_FACETS: &[KernelTagFacet] = &[\n");
    for f in &facet_order {
        out.push_str(&format!("    KernelTagFacet {{ name: \"{f}\", mask: 0x{:016x} }},\n", facet_masks[f]));
    }
    out.push_str("];\n\n");
    out.push_str("/// id → tag(线性;boot/诊断面用)\npub fn tag_by_id(id: &str) -> Option<KernelTag> {\n    KERNEL_TAGS.iter().find(|t| t.id == id).map(|t| t.tag)\n}\n\n");
    out.push_str("/// tag 集 → 位掩码\npub fn tag_mask(tags: &[KernelTag]) -> u64 {\n    let mut m = 0u64;\n    for t in tags { m |= t.bit(); }\n    m\n}\n");
    std::fs::write(out_dir.join("kernel_tags.rs"), out).expect("write kernel_tags.rs");
    println!("cargo:warning=owl-kernels: tag 词表 {} 项(建期生成 KernelTag/KERNEL_TAGS)", raw.tag.len());
}

/// **算子总表**(2026-10-20;用户裁决:构建期统计,不手写)。
///
/// 扫描口径 = `src/ops/*.rs` 里的
/// `pub static NAME: KernelMeta = meta_of::<T>();`
/// —— 这是**每算子一行、宏展开后仍原样存在的源级锚点**。
///
/// 为什么不扫 `impl KernelSpec for T`:
/// - GGUF / paged-attention 等 **43 个算子由 `macro_rules!` 生成**,源里只有
///   `impl KernelSpec for $t` / `const ENTRY = $entry`;build.rs 看不到宏展开,
///   也解析不了 `$entry` 指向的常量。
/// - static 提升点没有这个问题:每个生产算子都显式写一行,且 `KernelMeta`
///   在编译期已经携带 `TAGS`。因此本表是**算子全量清单**,tag 行不需要在
///   build.rs 再复刻一遍。
///
/// 生成物 `$OUT_DIR/ops_table.rs` 被 `ops/mod.rs` include;手写 `TABLES` 与
/// 各域 `METAS` 数组全部退役(单一真源 = static 提升点)。
fn build_ops_table(out_dir: &std::path::Path) {
    const DOMAIN_ORDER: &[&str] = &[
        "attn", "gdn", "gguf", "gdn_chunked", "gdn_scalar", "comm", "elems",
        "fused", "ln", "dflash", "load", "marlin", "cublas", "flashinfer",
        "xqa", "vis",
    ];
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let ops_dir = root.join("src").join("ops");
    println!("cargo:rerun-if-changed={}", ops_dir.display());

    // 文件清单门:新增 ops/<域>.rs 必须显式进 DOMAIN_ORDER,否则总表会漏域
    let mut present: Vec<String> = std::fs::read_dir(&ops_dir)
        .expect("read src/ops")
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension().is_some_and(|x| x == "rs")
                && p.file_name().is_some_and(|n| n != "mod.rs"))
                .then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    present.sort();
    let mut ordered = DOMAIN_ORDER.to_vec();
    ordered.sort_unstable();
    assert_eq!(
        present, ordered,
        "src/ops/*.rs 与 build.rs DOMAIN_ORDER 不一致 —— 新增域必须登记,\
         否则算子总表静默漏域"
    );

    let mut domains: Vec<(&str, Vec<String>)> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for d in DOMAIN_ORDER {
        let path = ops_dir.join(format!("{d}.rs"));
        println!("cargo:rerun-if-changed={}", path.display());
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        // 测试模块里的假算子不是生产表成员;按 `#[cfg(test)]` 截断是
        // 本项目的稳定分段约定(所有 ops 文件都把测试放文件尾)。
        let prod = text.split("#[cfg(test)]").next().unwrap_or(&text);
        let mut names = Vec::new();
        for (lineno, line) in prod.lines().enumerate() {
            let t = line.trim();
            let Some(rest) = t.strip_prefix("pub static ") else { continue };
            if !rest.contains(": KernelMeta = meta_of::<") {
                continue;
            }
            let name = rest.split(':').next().unwrap().trim();
            assert!(
                !name.is_empty()
                    && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'),
                "{d}.rs:{}: 无法解析 static 名 {name:?}",
                lineno + 1
            );
            assert!(
                seen.insert(name.to_string()),
                "{d}.rs:{}: static `{name}` 重复(跨域同名会让 total 表歧义)",
                lineno + 1
            );
            names.push(name.to_string());
        }
        assert!(!names.is_empty(), "{d}.rs: 未扫到任何 `pub static ... KernelMeta = meta_of::<...>()`");
        // 字面 impl 必须有对应的 static 提升点 —— 否则总表会静默漏掉它。
        // 宏模板(`impl KernelSpec for $t`)不在此列:它的实例由 static 行覆盖。
        for line in prod.lines() {
            let t = line.trim();
            let Some(rest) = t.strip_prefix("impl KernelSpec for ") else { continue };
            let Some(ty) = rest.split_whitespace().next() else { continue };
            let ty = ty.trim_end_matches('{');
            if ty.starts_with('$') || !ty.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                continue;
            }
            assert!(
                prod.contains(&format!("meta_of::<{ty}>()")),
                "{d}.rs: `impl KernelSpec for {ty}` 没有对应 `pub static ... meta_of::<{ty}>()` \
                 —— 算子总表会漏项(补 static 提升点)"
            );
        }
        domains.push((d, names));
    }

    let total: usize = domains.iter().map(|(_, n)| n.len()).sum();
    assert!(total >= 200, "算子总表只有 {total} 张 —— 扫描器失效(应 ≥200)");

    let mut out = String::from(
        "// @generated by crates/kernels/build.rs —— **勿手改**。\n\
         //\n\
         // 算子总表:扫描 `pub static NAME: KernelMeta = meta_of::<T>();` 生成。\n\
         // 手写 TABLES / 各域 METAS 已退役;tag 由 KernelMeta 自带(宏注入)。\n\n",
    );
    out.push_str("/// 按域分组的算子元数据(顺序 = build.rs DOMAIN_ORDER)\npub static TABLES: &[&[&KernelMeta]] = &[\n");
    for (d, names) in &domains {
        out.push_str(&format!("    // {d}\n    &[\n"));
        for n in names {
            out.push_str(&format!("        &crate::ops::{d}::{n},\n"));
        }
        out.push_str("    ],\n");
    }
    out.push_str("];\n\n");
    out.push_str(&format!(
        "/// 构建期扫到的生产算子总数(唯一真源;测试/诊断用)\npub const OP_COUNT: usize = {total};\n"
    ));
    std::fs::write(out_dir.join("ops_table.rs"), out).expect("write ops_table.rs");
    println!("cargo:warning=owl-kernels: 算子总表 {total} 张 / {} 域(构建期扫描 static 提升点)", domains.len());
}

fn main() {
    // native cubin 构建期产出(终态 = 零运行期编译)
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    build_ops_table(&out_dir);
    build_tags(&out_dir);
    let xqa_rows = build_xqa_assets(&out_dir);
    let gdn_rows = build_gdn_assets(&out_dir);
    build_native_cubins(&out_dir, &xqa_rows, &gdn_rows);
    // ⚠️ 2026-10-13 修连坐:原 main() 在 `CARGO_FEATURE_MARLIN` 缺席时整段
    // return —— 而 FI/XQA 两段(.a 复制 + rustc-link-lib 指令)当时都写在
    // build_marlin() 体内 ⇒ **单独开 `--features flashinfer`(或 xqa)会静默
    // 不链接**,症状 = 运行期 undefined symbol。现改为三 feature 各自成段、
    // 各自门控(marlin 段本体原有自门控,原 early-return 是冗余且有害)。
    build_marlin();
    build_flashinfer();
}

/// marlin 编译期依赖的头(E5-DF3 同日十二:selector 改动不触发重编 →
/// host `.o` 落后一代 → 新旧分支混链 undefined)
const MARLIN_HEADERS: &[&str] = &[
    "cu/marlin/vllm_marlin/kernel.h",
    "cu/marlin/vllm_marlin/kernel_selector.h",
    "cu/marlin/vllm_marlin/kernel_selector_full.upstream.h",
    "cu/marlin/vllm_marlin/marlin_template.h",
    "cu/marlin/vllm_marlin/marlin_dtypes.cuh",
    "cu/marlin/vllm_marlin/marlin_mma.h",
    "cu/marlin/vllm_marlin/dequant.h",
];

/// **selector ⊆ 实例化**门(2026-10-13)。
///
/// `marlin_host.cu` 的 `get_marlin_kernel` 靠 `#include "kernel_selector.h"`
/// 选核,而该头是**生成的**(从实例化文件 1:1 生成)。一旦两者漂移,后果是
/// **链接期** undefined hidden symbol(实测踩过;E5-DF3 同款"新旧分支混链")。
/// 本门在**构建期**把两边的 `Marlin<...>` 变体集抽出来比对,selector 引用
/// 而未实例化 ⇒ 直接拒构建(带首个缺失变体)。
fn check_marlin_selector(sources: &[PathBuf]) {
    fn variants(text: &str) -> std::collections::BTreeSet<String> {
        let mut set = std::collections::BTreeSet::new();
        let bytes = text.as_bytes();
        let needle = b"Marlin<";
        let mut i = 0usize;
        while let Some(pos) = text[i..].find("Marlin<") {
            let start = i + pos + needle.len();
            if let Some(end) = text[start..].find('>') {
                set.insert(text[start..start + end].split_whitespace().collect::<String>());
                i = start + end;
            } else {
                break;
            }
            let _ = bytes;
        }
        set
    }
    let sel = std::fs::read_to_string("cu/marlin/vllm_marlin/kernel_selector.h").expect("selector");
    let want = variants(&sel);
    let mut have = std::collections::BTreeSet::new();
    for src in sources {
        if let Ok(t) = std::fs::read_to_string(src) {
            have.extend(variants(&t));
        }
    }
    let missing: Vec<&String> = want.difference(&have).collect();
    assert!(
        missing.is_empty(),
        "marlin selector 引用了 {} 个**未实例化**的 Marlin<...> 变体(链接期 undefined hidden symbol 的根因);\
         首个: {:?} —— 要么把对应 sm80_kernel_*.cu 纳入 sources,要么从 selector 删掉该分支(见 kernel_selector.h 头注)",
        missing.len(),
        missing.first()
    );
    println!(
        "cargo:warning=owl-kernels: marlin selector ⊆ 实例化 ✓({} 变体引用 / {} 个已实例化)",
        want.len(),
        have.len()
    );
}

fn build_marlin() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    // ------------------------------------------------------------------
    // marlin W4A16(f16 激活 × u4 权重 × f16 scales;foreign-kernel 通道)
    //
    // **2026-10-13 C3-α**:由"预编 `.a` 入库(19 MB)"改为**构建期从源编**
    // —— 源本就在树(`cu/marlin/*.cu` + `vllm_marlin/`),入库二进制只是
    // "省一次 nvcc"的代价,却让 19 MB 二进制留在 git 且无法 review。
    // 冷编 ≈ 3.5 min,由 [`stamp_hit`] 增量闸吸收(只在源/头变化时付)。
    // 旗子纪律(照 xinfer marlin-ffi 先例,禁 fast_math;vLLM≥12.8 的
    // -static-global-template-stub=false 必带,否则跨 TU 拒链)。
    // ------------------------------------------------------------------
    if std::env::var_os("CARGO_FEATURE_MARLIN").is_some() {
        // .h 追踪(E5-DF3 同日十二:selector 改动不触发重编 → host .o 落后
        // 一代 → 新旧分支混链 undefined)
        for h in MARLIN_HEADERS {
            println!("cargo:rerun-if-changed={}", h);
        }
        println!("cargo:rerun-if-env-changed=MARLIN_FORCE_BUILD");
        println!("cargo:rerun-if-env-changed=MARLIN_CUDA_ARCH");
        println!("cargo:rerun-if-env-changed=MARLIN_NVCC");
        let force = env::var("MARLIN_FORCE_BUILD").ok().as_deref() == Some("1");
        let obj_dir = out_dir.join("marlin");
        std::fs::create_dir_all(&obj_dir).unwrap();
        let lib_out = out_dir.join("libmarlin.a");
        {
            let arch = env::var("MARLIN_CUDA_ARCH").unwrap_or_else(|_| "86".into());
            let nvcc_m = env::var("MARLIN_NVCC").unwrap_or_else(|_| "nvcc".into());
            let base = PathBuf::from("cu/marlin");
            let sources = [
                base.join("marlin_host.cu"),
                base.join("vllm_marlin/sm80_kernel_float16_u4_float16.cu"),   // AWQ 非对称(kU4)
                base.join("vllm_marlin/sm80_kernel_float16_u4b8_float16.cu"), // GPTQ 对称(kU4B8)
                base.join("vllm_marlin/sm80_kernel_bfloat16_u4b8_bfloat16.cu"), // BF16 激活变体(E5-DF3 草稿 BF16 化)
                base.join("vllm_marlin/sm80_kernel_s8_u4b8_float16.cu"),      // W4A8 s8→f16(S1;selector 引用之)
                base.join("vllm_marlin/sm80_kernel_s8_u4_float16.cu"),        // W4A8 s8→f16 × AWQ 非对称(kU4;生产 27B 走这条)
                base.join("vllm_marlin/sm80_kernel_s8_u4_bfloat16.cu"),       // W4A8 s8→bf16 × kU4(BF16 激活档)
            ];
            let includes = [base.join("vllm_marlin"), base.join("vllm_marlin/shim")];
            let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
            // ⚠️ `force` 不进指纹(一次性旁路,不是构建输入;进了会导致
            // "用 FORCE_BUILD 跑过一次之后,普通构建反而重编")
            check_marlin_selector(&sources);
            let flags = format!("marlin arch=sm_{arch} nvcc={nvcc_m} ar={ar}");
            let mut deps: Vec<String> = sources.iter().map(|p| p.display().to_string()).collect();
            deps.extend(MARLIN_HEADERS.iter().map(|h| h.to_string()));
            if !force && stamp_hit(&lib_out, &deps, &flags) {
                println!("cargo:warning=owl-kernels: libmarlin.a 增量命中(零 nvcc)");
            } else {
            let mut objs = Vec::new();
            for src in &sources {
                println!("cargo:rerun-if-changed={}", src.display());
                let obj = obj_dir.join(src.file_name().unwrap()).with_extension("o");
                let mut args = vec![
                    "-c".to_string(),
                    src.to_str().unwrap().to_string(),
                    "-o".to_string(),
                    obj.to_str().unwrap().to_string(),
                ];
                for inc in &includes {
                    args.push("-I".into());
                    args.push(inc.to_str().unwrap().to_string());
                }
                args.extend([
                    format!("-arch=compute_{arch}"),
                    format!("-code=sm_{arch}"),
                    "-O3".into(),
                    "-std=c++17".into(),
                    "-Xcompiler".into(),
                    "-fPIC".into(),
                    // 禁 -fvisibility=hidden:__global__ 模板实例跨 TU 拒链(xinfer 坑)
                    "--expt-relaxed-constexpr".into(),
                    "-static-global-template-stub=false".into(),
                    "-Xcudafe".into(),
                    "--diag_suppress=20236".into(),
                ]);
                let status = Command::new(&nvcc_m)
                    .args(&args)
                    .status()
                    .unwrap_or_else(|e| panic!("marlin: failed to spawn {nvcc_m}: {e}"));
                assert!(status.success(), "marlin: nvcc failed for {}", src.display());
                objs.push(obj);
            }
                let _ = std::fs::remove_file(&lib_out); // ar rcs 是按名替换:旧成员会留存
                let mut cmd = Command::new(&ar);
                cmd.arg("rcs").arg(lib_out.to_str().unwrap());
                for o in &objs {
                    cmd.arg(o.to_str().unwrap());
                }
                let status = cmd.status().expect("marlin: failed to spawn ar");
                assert!(status.success(), "marlin: ar failed");
                stamp_write(&lib_out, &deps, &flags);
                println!("cargo:warning=owl-kernels: built libmarlin.a (sm_{arch}) from source");
            }
        }
        let cuda_lib = env::var("CUDA_HOME")
            .map(|h| PathBuf::from(h).join("lib64"))
            .unwrap_or_else(|_| PathBuf::from("/usr/local/cuda/lib64"));
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        println!("cargo:rustc-link-lib=static=marlin");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib.display());
    }

}

/// 递归列目录下所有文件(指纹/追踪用;零 walkdir 依赖)
fn all_files(dir: &str, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            all_files(&p.to_string_lossy(), out);
        } else {
            out.push(p.to_string_lossy().to_string());
        }
    }
}

fn build_flashinfer() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    // ------------------------------------------------------------------
    // FlashInfer prefill(foreign-kernel 通道;E1.5 工作包)
    //
    // **2026-10-13 C3-α**:由"预编 `.a` 入库(8.2 MB)"改为**构建期从源编**;
    // 头闭包已 **vendor 进树**(`cu/flashinfer/include/`,**上游 0.6.18 闭包**
    // 22 文件 ~700 KB;闭包复现法见其 README)—— 原先缺省读 `~/.cudaforge/...` 的**树外**检出,
    // 违 §15.1.1 律 1"所有族的源必须在树"。冷编 ≈ 1.5 min,增量闸吸收。
    // ------------------------------------------------------------------
    if std::env::var_os("CARGO_FEATURE_FLASHINFER").is_some() {
        // 2026-10-15 **拆 TU**:上游 0.6.18 单 TU 里 f16+fp8 双实例化 FA2 模板
        // ⇒ ptxas 病态卡住。按 KV dtype 臂拆:plan+f16 住主文件,fp8 臂单列
        // (共享模板头 `owl_fi_attention_common.h`);两个 .o 归同一个 `.a`,
        // **ABI 零改动**。
        let src_f16 = "cu/flashinfer/owl_fi_attention.cu";
        let src_fp8 = "cu/flashinfer/owl_fi_attention_fp8.cu";
        // 视觉塔注意力(2026-10-15):**同仓 FlashInfer ragged 非因果臂**
        // (复用 vendored FA2 家族核,零自研核;见该 .cu 头注)
        let src_ragged = "cu/flashinfer/owl_fi_attn_ragged.cu";
        let src_hdr = "cu/flashinfer/owl_fi_attention_common.h";
        let fi_include = env::var("FLASHINFER_INCLUDE")
            .unwrap_or_else(|_| "cu/flashinfer/include".into());
        for f in [src_f16, src_fp8, src_ragged, src_hdr] {
            println!("cargo:rerun-if-changed={f}");
        }
        println!("cargo:rerun-if-changed={fi_include}");
        println!("cargo:rerun-if-env-changed=FLASHINFER_FORCE_BUILD");
        println!("cargo:rerun-if-env-changed=FLASHINFER_INCLUDE");
        println!("cargo:rerun-if-env-changed=FLASHINFER_NVCC");
        let force = env::var("FLASHINFER_FORCE_BUILD").ok().as_deref() == Some("1");
        let nvcc_m = env::var("FLASHINFER_NVCC").unwrap_or_else(|_| "nvcc".into());
        let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
        let fi_dir = out_dir.join("flashinfer");
        std::fs::create_dir_all(&fi_dir).expect("mkdir flashinfer out");
        let lib = fi_dir.join("libowl_flashinfer.a");
        let flags = format!(
            "fi arch=sm_86 nvcc={nvcc_m} ar={ar} inc={fi_include} src=upstream-0.6.18 split=2tu"
        );
        let mut deps: Vec<String> = vec![
            src_f16.to_string(),
            src_fp8.to_string(),
            src_ragged.to_string(),
            src_hdr.to_string(),
        ];
        all_files(&fi_include, &mut deps);
        deps.sort();
        if !force && stamp_hit(&lib, &deps, &flags) {
            println!("cargo:warning=owl-kernels: libowl_flashinfer.a 增量命中(零 nvcc)");
        } else {
            // 2026-10-14 换源(上游 0.6.18):`-DFLASHINFER_ENABLE_FP8_E4M3`
            // **已不需要** —— 旧 guoqingbao fork 用该宏圈住 fp8 入口,上游核内
            // 自带 fp8 反量化(`fast_dequant_f8f16x4`)。
            // (历史坑留档:旧预编 `.a` 带该宏、从源路径起初没带 ⇒ 链接期
            //  undefined symbol;换源后这类"配方不一致"面自然消失)
            let mut objs: Vec<PathBuf> = Vec::new();
            for src in [src_f16, src_fp8, src_ragged] {
                let obj = fi_dir.join(format!(
                    "{}.o",
                    PathBuf::from(src).file_stem().unwrap().to_string_lossy()
                ));
                let status = std::process::Command::new(&nvcc_m)
                    .args(["-O3", "-std=c++17", "-arch=sm_86"])
                    .arg(format!("-I{fi_include}"))
                    .arg("-c").arg(src).arg("-o").arg(&obj)
                    .status()
                    .expect("flashinfer: failed to spawn nvcc");
                assert!(status.success(), "flashinfer: nvcc failed on {src}");
                objs.push(obj);
            }
            let _ = std::fs::remove_file(&lib);
            let mut ar_cmd = std::process::Command::new(&ar);
            ar_cmd.args(["rcs"]).arg(&lib);
            for o in &objs {
                ar_cmd.arg(o);
            }
            let status = ar_cmd.status().expect("flashinfer: failed to spawn ar");
            assert!(status.success(), "flashinfer: ar failed");
            stamp_write(&lib, &deps, &flags);
            println!(
                "cargo:warning=owl-kernels: built libowl_flashinfer.a from source (in-tree headers, 3 TU)"
            );
        }
        let cuda_lib = env::var("CUDA_HOME")
            .map(|h| PathBuf::from(h).join("lib64"))
            .unwrap_or_else(|_| PathBuf::from("/usr/local/cuda/lib64"));
        println!("cargo:rustc-link-search=native={}", fi_dir.display());
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        println!("cargo:rustc-link-lib=static=owl_flashinfer");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib.display());
    }
}
