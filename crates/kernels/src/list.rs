//! 登记表(list)+ Kernel 值(2026-10-12 自 models::kernel 迁入;
//! 用户裁决:登记表是 kernels 语义,models 留 re-export 垫子)。
//! 更名注记:native.rs → list.rs —— 本模块本体 = **封闭登记表**
//! (名字 → 源/签名/dtype 的唯一权威表),"native" 只是表中条目的
//! 属性之一(native/foreign 之别住 driver 分派与 registry 路由)。
//!
//! ```text
//! ops/<域>.rs(算子自托管声明:名字/源/参数表 —— 已建表算子源的唯一住址)
//!        │  未建表「仅引入」算子退到 ↓(R1 建表补齐后本表整体退役)
//!        ▼
//! list.rs(本模块:§0 就地 .cu 源常量 + Kernel/LaunchShape 值 + 封闭登记表)
//!        ↓ kernel / kernel_with / source(经 models::kernel 垫子)
//! layers + TensorOps::of(Kernel)(组合时只出现名字与发射配置)
//! ```
//!
//! - `name`:kernel 入口名(server 编译缓存键);
//! - `source`:.cu 源码(只经注册表导出;上层零源码感知 —— 2026-09-25
//!   用户裁决);
//! - `launch`:发射配置(grid = (0,0,0) 哨兵 = 解释层按输出元素数自动
//!   1D ceil/256;行核用 `with_launch` 显式);
//! - 后端懒编译:lower 为 LaunchMsg,server 按 (name, source) 懒编译
//!   (nvrtc → PTX → module),缓存后逐次直发。
//!
//! 登记表是**编译期封闭常量**:新增 kernel = owl-kernels 加源 + 本表加
//! 一条;查不到名字 = 编程错误(拼错名),`source` 直接 panic(带登记
//! 表提示)—— 这发生在组合构造期,不是运行期可恢复错误。
//!
//! 对位契约(C2 schema 化挂账,见 roadmap.local/api-stabilize-plan.md):
//! 标量形参宽度必须与 Arg 严格对位(arg_usize ↔ `size_t` 8B;arg_f32 ↔
//! `float`;arg_i32 ↔ `int`),**输出块固定末参**。
//!
//! driver.rs 挂账清(2026-10-12):"登记表暂留 models" 已了结 —— 登记表
//! 与 driver 同 crate,耦合测试(driver_picks_are_registered)随迁。

// 构建期生成的**资产符号清单**(`build.rs` 产出;非 device 门 ——
// `models::kernel::kv_manifest_gate` 在 boot 期要用)
include!(concat!(env!("OUT_DIR"), "/cubin_symbols.rs"));

/// 发射配置(grid 哨兵 (0,0,0) = 自动 1D)
#[derive(Clone, Debug)]
pub struct LaunchShape {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

impl Default for LaunchShape {
    fn default() -> Self {
        Self { grid: (0, 0, 0), block: (256, 1, 1), shared_mem: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: &'static str,
    pub src: &'static str,
    pub launch: LaunchShape,
    /// 逃生舱签名(非注册 kernel 必带;格式同 Entry.args,含末位输出 T)。
    /// 注册 kernel 此字段留空 —— 槽序权威取注册表/算子元数据。
    pub sig: &'static str,
    /// **算子准入声明**(2026-10-13 P3):已建表算子由
    /// [`with_pick`](crate::list::with_pick) 从 `KernelPick` 带入;
    /// 未建表(仅引入)与逃生舱核为 `None`。
    pub meta: Option<&'static crate::spec::KernelMeta>,
}

impl Kernel {
    pub fn new(name: &'static str, src: &'static str) -> Self {
        // F 轮:准入声明是槽序/几何的唯一真源 ⇒ 构造期**自动挂载**
        // (名字已在声明面就带上;逃生舱核自然为 None,走 `sig`)。
        Self { name, src, launch: LaunchShape::default(), sig: "", meta: crate::ops::meta(name) }
    }

    /// 显式发射配置(覆盖自动 1D)
    pub fn with_launch(mut self, grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: u32) -> Self {
        self.launch = LaunchShape { grid, block, shared_mem };
        self
    }

    /// 逃生舱签名(非注册 kernel 的槽序权威;注册 kernel 勿用)
    pub fn with_sig(mut self, sig: &'static str) -> Self {
        self.sig = sig;
        self
    }

    /// 挂算子准入声明(槽序/检查函数随之而来)
    pub fn with_meta(mut self, meta: &'static crate::spec::KernelMeta) -> Self {
        self.meta = Some(meta);
        self
    }

    /// **槽序权威**(F 轮后优先序):算子元数据 → 逃生舱签名。
    /// 两条都是 `'static`,零分配;有声明时它就是唯一真源
    /// (登记表已退役 —— 槽序不再有第三条落点)。
    pub fn slot_tokens(&self) -> Vec<&'static str> {
        if let Some(m) = self.meta {
            return m.slot_tokens();
        }
        // 逃生舱核(无声明,如测试/诊断核):槽序必须显式给签名
        assert!(
            !self.sig.is_empty(),
            "kernel::slot_tokens({}): 既无准入声明也无逃生舱槽序 —— 拒发",
            self.name
        );
        self.sig.split(',').collect()
    }
}

/// **算子准入查询**(原"登记表查名字"入口;F 轮 2026-10-13 起**唯一真源
/// = 算子自托管声明**,登记表已退役)。
pub fn lookup(name: &str) -> Option<&'static crate::spec::KernelMeta> {
    crate::ops::meta(name)
}

/// 资产 id → `.cu` 源文本(**仅测试/工具面**;运行期已不再需要源文本 ——
/// E 轮后二进制里不内联任何 `.cu`)。
pub fn cu_text(src_id: &str) -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(src_id))
        .unwrap_or_else(|e| panic!("cu_text({src_id}): {e}"))
}

/// **资产符号清单查表**(E 轮:源文本退役 ⇒ 判据从"正则扫文本"升级为
/// "构建期实际编出的符号清单";符号由 `build.rs` 从 `.cu` 解析)。
/// `models::kernel::kv_manifest_gate` 的池写者/读者核对走本函数。
pub fn source_defines(src_id: &str, name: &str) -> bool {
    ASSET_SYMBOLS
        .iter()
        .find(|(id, _)| *id == src_id)
        .is_some_and(|(_, syms)| syms.contains(&name))
}

/// 算子名 → **资产 id**(`KernelSpec::SRC` 的唯一读取口;F 轮后不再有
/// 登记表兜底 —— 未建表 = 未准入 = 编程错误)。
pub fn source(name: &str) -> &'static str {
    crate::ops::source(name)
}

/// 构造 Kernel 值(默认发射配置:grid 哨兵 = 自动 1D ceil/256)
pub fn kernel(name: &'static str) -> Kernel {
    Kernel::new(name, source(name))
}

/// 构造 Kernel 值(显式发射配置;行核 embed/rope/attn 等非 ceil/256 网格用)
pub fn kernel_with(name: &'static str, grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: u32) -> Kernel {
    Kernel::new(name, source(name)).with_launch(grid, block, shared_mem)
}

/// 构造 Kernel 值(driver 拾取形态;**新调用点一律走此入口**,名字/参数
/// 由 owl-kernels::driver 单源推导 —— 字符串核名与手抄 grid/block/smem
/// 在 models 层属遗留,迁移完归零,grep 审计入验收)。
///
/// **已建表算子**(`pick.meta()` 非空):名字/源码/槽序全部取算子自托管的
/// 元数据(foreign 家族 `SOURCE` 为空 ⇒ 源码为空串,server 走家族 FFI);
/// **未建表**(仅引入):回落登记表取源。
pub fn with_pick(p: crate::driver::KernelPick) -> Kernel {
    let launch = LaunchShape { grid: p.shape.grid, block: p.shape.block, shared_mem: p.shape.smem };
    match p.meta() {
        Some(m) => Kernel { name: m.entry, src: m.src, launch, sig: "", meta: Some(m) },
        None => Kernel { name: p.name, src: source(p.name), launch, sig: "", meta: None },
    }
}

/// 发射配置便捷(与 kernel_with 同参,LaunchShape 形态)
pub fn launch_shape(grid: (u32, u32, u32), block: (u32, u32, u32), shared_mem: u32) -> LaunchShape {
    LaunchShape { grid, block, shared_mem }
}


// ============================================================================
// §3 `.cu` 签名解析(机器互证;唯一解析器 —— 测试门与算子准入门共用,
//    两道门不许各写一遍,否则会各说各话)
// ============================================================================

/// 解析 .cu 源里 `extern "C" __global__ void <name>(...)` 的形参序列,
/// 映射为 Entry.args 同构字符串(无 regex,朴素扫描)。
/// 形参序列 → Entry.args 词表字符串(T/sz/i32/f32;三种核形态共用)
pub fn map_params(params: &str, name: &str, kind: &str) -> String {
    params
        .split(',')
        .map(|p| {
            let p = p
                .replace("const ", "")
                .replace("__restrict__ ", "")
                .replace('\\', " ")
                .trim()
                .to_string();
            if p.contains('*') {
                "T".to_string()
            } else if p.contains("size_t") || p.contains("unsigned long long") {
                "sz".to_string()
            } else if p.contains("float") {
                "f32".to_string()
            } else if p.contains("int") {
                // 含 unsigned int:4 字节无符号 —— Arg 无此宽度,登记表禁止
                assert!(!p.contains("unsigned"), "{name}: 禁止 unsigned int 形参(用 size_t;Arg::U64 8 字节对位)");
                "i32".to_string()
            } else {
                panic!("{name}: 无法识别的{kind}形参 `{p}`(args 词表:T/sz/i32/f32)");
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// 从 `text[start..]` 截到首个 `)` 的形参串 → 词表
pub fn params_at(text: &str, start: usize, name: &str, kind: &str) -> Option<String> {
    let end = text[start..].find(')')? + start;
    Some(map_params(&text[start..end], name, kind))
}

pub fn parse_cu_sig(src: &str, name: &str) -> Option<String> {
    // 剥行注释(形参注记含逗号/括号,会干扰边界与切分)+ **折叠宏续行**
    // (`\` + 换行 → 空格:宏体形参住在续行里,不折叠则形参串被反斜杠污染)
    let clean: String = src
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("\\\n", " ");
    // 形态 1:宏展开核(F5 ops_pair):实例行 `OWL_<FAMILY>_KERNEL(<name>, <T>, PARAMS)`
    // —— PARAMS 自带类型化形参,机器对账以实例为准(C2 不弱化)
    for line in clean.lines() {
        let t = line.trim();
        if t.starts_with("OWL_") && t.contains("KERNEL(") {
            if let Some(rest) = t.split_once('(').map(|x| x.1) {
                let mut parts = rest.splitn(3, ',');
                if parts.next().map(|x| x.trim()) == Some(name) {
                    let params = parts
                        .nth(1)
                        .and_then(|p| p.trim().strip_prefix('('))
                        .and_then(|p| p.strip_suffix(')'))
                        .unwrap_or("");
                    return Some(map_params(params, name, "宏"));
                }
            }
        }
    }
    // 形态 2:单层宏实例化核(chunked KNHD 族:`CHUNKED_KNHD_ENTRY(<name>, <hd>, <kv_fp8>)`)
    // — 实例行只带实参,形参住在 `#define <MACRO>(<首形参>, …)` 的定义体
    //   `void <首形参>(…)` 里:按定义体取参,与字面核同口径(名字门不弱化)
    for line in clean.lines() {
        let t = line.trim();
        let Some((mac, rest)) = t.split_once('(') else { continue };
        if mac.is_empty() || !mac.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        if rest.split(',').next().map(str::trim) != Some(name) {
            continue;
        }
        let def_anchor = format!("#define {mac}(");
        let Some(d) = clean.find(&def_anchor) else { continue };
        let formal = clean[d + def_anchor.len()..].split(',').next()?.trim().to_string();
        let body_anchor = format!("void {formal}(");
        let Some(s) = clean[d..].find(&body_anchor) else { continue };
        return params_at(&clean[d..], s + body_anchor.len(), name, "宏");
    }
    // 形态 3:字面核 `extern "C" __global__ void <name>(…)`
    let anchor = format!("__global__ void {name}(");
    let start = clean.find(&anchor)? + anchor.len();
    params_at(&clean, start, name, "")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;
    

    /// **每张原生表的资产与符号清单在场**(原"登记表有源"门的等价升级)。
    #[test]
    fn every_declared_op_has_asset_and_symbols() {
        let mut n = 0;
        for m in ops::all() {
            if m.src.is_empty() { continue; }   // foreign(FFI):无 `.cu`
            assert!(m.src.starts_with("cu/") && m.src.ends_with(".cu"),
                    "{}: 资产 id 非法 {}", m.entry, m.src);
            assert!(cu_text(m.src).contains("extern \"C\" __global__"),
                    "{}: 资产 {} 不是核源码", m.entry, m.src);
            assert!(source_defines(m.src, m.entry),
                    "{}: 不在资产 {} 的构建期符号清单里", m.entry, m.src);
            n += 1;
        }
        eprintln!("[list] {n} 张原生表的资产 + 符号清单在场 ✓");
    }

    /// **R1(F 轮换源):构建期编出的符号 ⊆ 声明集** —— 凡编出来的核都必须有
    /// `impl KernelSpec`。比原"登记表差额"更贴 R1 本意:能抓"新加 `.cu` 但没
    /// 建表"(登记表差额抓不到,因为那条路要人先去登记)。
    #[test]
    fn every_compiled_symbol_is_declared() {
        let und = ops::undeclared_symbols();
        assert!(und.is_empty(), "编出但未声明的核符号 {} 条:{und:?}", und.len());
        eprintln!("[R1] 编出符号集 ⊆ 声明集 ✓(登记表已退役,判据换源)");
    }

    /// **每张原生表都有输出槽** ⇒ 声明口径 dtype 可派生(F 轮删 `Entry.dtype`
    /// 的依据;`op_tables` 的 O 指针门是它的机器面)。
    #[test]
    fn every_native_table_declares_out_dtype() {
        for m in ops::all() {
            if m.src.is_empty() { continue; }
            assert!(m.out_dtype().is_some(), "{}: 无输出槽 ⇒ 声明口径 dtype 不可派生", m.entry);
        }
    }

    /// kNHD 翻转不变量(2026-10-13,P1 迁移收尾):每个登记的 `*_knhd*` 核
    /// 必须有 classic 伙伴(去 `_knhd` 标记仍能查到),且 **args 与源逐位相同**。
    /// 锁住“翻转 = 只换核名,不改发射实参表”这条全部接线的前提 ——
    /// 一旦 kNHD 变体换了签名,本门先红,不让翻转在 GPU 上爆。
    #[test]
    fn knhd_variants_mirror_classic_signature() {
        // kNHD 侧**独有**条目(classic 无对偶;纯新增)。每加一条必须登
        // 在此并写理由 —— 否则下面循环会把它当“换签名”报红,谁也别想混过去。
        const KNHD_ONLY: &[(&str, &str)] = &[
            (
                "vllm_paged_attention_v2_f16_knhd_hd256",
                "KNHD 侧新增:f16 无 bs32 以外的 classic v2 对偶(fp8 档才有)",
            ),
            (
                "vllm_paged_attention_v2_f16_knhd_hd128",
                "同上",
            ),
        ];
        let mut paired = 0usize;
        for e in ops::all() {
            if !e.entry.contains("_knhd") {
                continue;
            }
            if KNHD_ONLY.iter().any(|(n, _)| *n == e.entry) {
                continue;
            }
            let classic = e.entry.replace("_knhd", "");
            let Some(c) = ops::all().find(|x| x.entry == classic.as_str()) else {
                panic!(
                    "{}: 无 classic 伙伴 `{classic}` —— 要么补对偶,要么登进 KNHD_ONLY 并写理由",
                    e.entry
                );
            };
            assert_eq!(
                c.args_str(), e.args_str(),
                "{}: 槽序与 classic 伙伴不符 —— 翻转须只换名(实参表同一)",
                e.entry
            );
            assert_eq!(c.src, e.src, "{}: 与 classic 伙伴不同源", e.entry);
            paired += 1;
        }
        // 覆盖锁:22 = 2026-10-13 F 轮实况对数(新增 kNHD 核时同步上调)
        // (C 轮拆文件后 attn 域补入 4 个 knhd NC 变体:18 → 22)
        assert_eq!(paired, 22, "配对 kNHD 变体数异常(实 {paired})—— 漏登记或误删?");
        for (n, why) in KNHD_ONLY {
            assert!(
                ops::all().any(|e| e.entry == *n),
                "KNHD_ONLY 条目 `{n}` 已不在登记表({why}),请清理白名单"
            );
        }
        eprintln!("[knhd] {paired} 个 kNHD 变体与 classic 伙伴签名逐位相同 ✓(另有 {} 条 kNHD 独有,已登记理由)", KNHD_ONLY.len());
    }

    #[test]
    fn lookup_known_and_unknown() {
        assert!(lookup("owl_add_f32").is_some());
        assert!(lookup("owl_not_registered").is_none());
    }

    /// 登记名全库唯一(2026-10-12 review F 案:chunked prefill 四条目
    /// 曾重复登记两次,lookup 首中掩盖;封闭登记表的唯一性不变量机器锁)
    #[test]
    fn registry_names_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for e in ops::all() {
            assert!(seen.insert(e.entry), "{} 重复声明", e.entry);
        }
    }

    /// driver ↔ 登记表耦合测试(漂移拦截):driver 产出的每个名字必须
    /// ∈ 登记表 —— driver 改名/登记表删条,红灯在 CI,不在 GPU 上
    #[test]
    fn driver_picks_are_registered() {
        use crate::driver::{self, DType};
        let names = [
            driver::gdn::gating_g(DType::F16).name,
            driver::gdn::gating_g(DType::F32).name,
            driver::gdn::l2norm(DType::F16, 1).name,
            driver::gdn::l2norm(DType::F32, 1).name,
            driver::gdn::conv_upd(DType::F16).name,
            driver::gdn::conv_upd(DType::F32).name,
            driver::gdn::delta_dec(DType::F16, 128, 1, 16, 128).name,
            driver::gdn::delta_dec(DType::F32, 128, 1, 16, 128).name,
            driver::gdn::conv_fwd(DType::F16, 6144).name,
            driver::gdn::recurrence_varlen_gqa(DType::F16, 128, 16, 128).name,
            driver::gdn::norm_act(DType::F16, 1, 2048, 128).name,
            driver::gdn::norm_act(DType::F32, 1, 2048, 128).name,
            driver::ops::sigmoid(DType::F16).name,
            driver::ops::sigmoid(DType::F32).name,
        ];
        for n in names {
            assert!(lookup(n).is_some(), "driver 拾取名未登记:{n}");
        }
    }



}
