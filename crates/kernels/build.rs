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
// `KernelSpec::SOURCE`(就地 include_str,R5)+ `sources.rs` 的那一份集合。
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
    // src/**/*.rs 任一改动都可能新增/删除 include_str ⇒ 构建期重扫
    println!("cargo:rerun-if-changed=src");
    for f in &rs {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    let mut found: Vec<PathBuf> = Vec::new();
    for f in &rs {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        let base = f.parent().expect("parent");
        let mut rest = text.as_str();
        while let Some(pos) = rest.find("include_str!(") {
            rest = &rest[pos + "include_str!(".len()..];
            let trimmed = rest.trim_start();
            if trimmed.starts_with('r') {
                panic!("include_str! 用了原始字符串({}):旧解析器只认裸字面量,请改写或扩解析器", f.display());
            }
            let Some(hit) = trimmed.strip_prefix('"') else { continue };
            let Some(end) = hit.find('"') else {
                panic!("include_str! 字面量未闭合: {}", f.display());
            };
            let rel = &hit[..end];
            if !rel.ends_with(".cu") {
                continue;
            }
            let abs = base.join(rel);
            let canon = std::fs::canonicalize(&abs)
                .unwrap_or_else(|e| panic!("include_str!({rel}) 解析失败({}): {e}", f.display()));
            if !found.contains(&canon) {
                found.push(canon);
            }
        }
    }
    found.sort();
    found
}

/// **增量闸**:fatbin 与 `.stamp` 都在且指纹一致 ⇒ 跳过 nvcc。
///
/// 为什么必须有:build.rs 的 rerun 面包含 `src/**/*.rs`(源清单 = include_str!
/// 闭包),**改任何一个 Rust 文件都会重跑 build.rs** ⇒ 没有本闸的话每次 Rust
/// 编辑都要把 20 个 `.cu` 重编一遍(实测数十秒)。指纹 = 源 mtime+长度 +
/// 架构集 + 额外编译选项;任一变 ⇒ 重编。
fn fingerprint(src: &std::path::Path, archs: &[String], extra: &[&str]) -> String {
    let md = std::fs::metadata(src).expect("stat .cu");
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!(
        "src={} len={} mtime={} archs={} flags={}",
        src.display(),
        md.len(),
        mtime,
        archs.join(","),
        extra.join(" ")
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

fn build_native_cubins(out_dir: &std::path::Path, xqa_rows: &[(&'static str, PathBuf)]) {
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

    // (asset, 源哈希, 源长度, 源路径, fatbin 路径)
    let mut rows: Vec<(String, u64, usize, PathBuf, PathBuf)> = Vec::new();
    let mut rebuilt = 0usize;
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
        rows.push((asset, hash_source(&text), text.len(), src.clone(), dst));
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
    out.push_str("/// 一个 `.cu` 文件产出的 fatbin 资产\npub struct CubinAsset {\n    /// 资产名(`cu/` 相对路径,`/` → `__`;= 模块缓存键)\n    pub asset: &'static str,\n    /// 该资产的源文本(`include_str!` —— 反查/自洽锁/诊断用)\n    pub src: &'static str,\n    /// 源文本 FNV-1a 64(与 `hash_source` 一致;查表键)\n    pub src_hash: u64,\n    /// 源文本字节数(哈希碰撞时的第二道判据 —— 双标识才敢直接装配)\n    pub src_len: usize,\n    /// fatbin 字节(`include_bytes!`;多代际容器)\n    pub bytes: &'static [u8],\n}\n\n");
    out.push_str("pub const CUBIN_ASSETS: &[CubinAsset] = &[\n");
    for (asset, hash, len, src, bin) in &rows {
        out.push_str(&format!(
            "    CubinAsset {{ asset: \"{asset}\", src: include_str!(\"{src}\"), src_hash: 0x{hash:016x}, src_len: {len}, bytes: include_bytes!(\"{bin}\") }},\n",
            src = src.display(),
            bin = bin.display()
        ));
    }
    out.push_str("];\n\n");
    out.push_str(&xqa_table_src(xqa_rows));
    std::fs::write(&manifest_path, out).expect("write cubin_assets.rs");
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

/// `OWL_CUBIN_SKIP` 档的空登记表(纯 NVRTC 基线对照;同一文件形态免分支)
const EMPTY_ASSETS_RS: &str = "// @generated (OWL_CUBIN_SKIP)\npub const CUBIN_ARCHS: &[&str] = &[];\n\npub struct CubinAsset {\n    pub asset: &'static str,\n    pub src: &'static str,\n    pub src_hash: u64,\n    pub src_len: usize,\n    pub bytes: &'static [u8],\n}\n\npub const CUBIN_ASSETS: &[CubinAsset] = &[];\n\npub const XQA_ASSETS: &[(&str, &[u8])] = &[];\n\npub fn xqa_asset(_name: &str) -> Option<&'static [u8]> {\n    None\n}\n";

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

fn main() {
    // native cubin 构建期产出(终态 = 零运行期编译)
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let xqa_rows = build_xqa_assets(&out_dir);
    build_native_cubins(&out_dir, &xqa_rows);
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
    // 头闭包已 **vendor 进树**(`cu/flashinfer/include/`,22 文件 ~700 KB,
    // 闭包复现法见其 README)—— 原先缺省读 `~/.cudaforge/...` 的**树外**检出,
    // 违 §15.1.1 律 1"所有族的源必须在树"。冷编 ≈ 1.5 min,增量闸吸收。
    // ------------------------------------------------------------------
    if std::env::var_os("CARGO_FEATURE_FLASHINFER").is_some() {
        let src = "cu/flashinfer/owl_fi_attention.cu";
        let fi_include = env::var("FLASHINFER_INCLUDE")
            .unwrap_or_else(|_| "cu/flashinfer/include".into());
        println!("cargo:rerun-if-changed={src}");
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
        let flags = format!("fi arch=sm_86 nvcc={nvcc_m} ar={ar} inc={fi_include} fp8=1");
        let mut deps: Vec<String> = vec![src.to_string()];
        all_files(&fi_include, &mut deps);
        deps.sort();
        if !force && stamp_hit(&lib, &deps, &flags) {
            println!("cargo:warning=owl-kernels: libowl_flashinfer.a 增量命中(零 nvcc)");
        } else {
            let obj = fi_dir.join("owl_fi_attention.o");
            // ⚠️ `-DFLASHINFER_ENABLE_FP8_E4M3` 是**必需**的:fp8 KV 入口
            // `owl_fi_prefill_run_fp8kv` 被该宏圈住(源注释:"编译门"),而
            // owl 生产档 quant=Fp8E4M3 + fi=true ⇒ 缺它 = 链接期 undefined
            // symbol(实测:旧预编 `.a` 是带该宏编的,从源路径原先没带 ——
            // 又一例"预编配方与树内配方不一致")
            let status = std::process::Command::new(&nvcc_m)
                .args(["-O3", "-std=c++17", "-arch=sm_86", "-DFLASHINFER_ENABLE_FP8_E4M3"])
                .arg(format!("-I{fi_include}"))
                .arg("-c").arg(src).arg("-o").arg(&obj)
                .status()
                .expect("flashinfer: failed to spawn nvcc");
            assert!(status.success(), "flashinfer: nvcc failed");
            let _ = std::fs::remove_file(&lib);
            let status = std::process::Command::new(&ar)
                .args(["rcs"]).arg(&lib).arg(&obj)
                .status()
                .expect("flashinfer: failed to spawn ar");
            assert!(status.success(), "flashinfer: ar failed");
            stamp_write(&lib, &deps, &flags);
            println!("cargo:warning=owl-kernels: built libowl_flashinfer.a from source (in-tree headers)");
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
