//! **`.cu` 取证面**(G 轮 2026-10-13 自 `list.rs` 拆出;**纯测试/工具面**)。
//!
//! 三件事,消费者都不在热路径上:
//! 1. [`cu_text`] —— 资产 id → 磁盘上的 `.cu` 源文本(E 轮起运行期不再内联源);
//! 2. [`source_defines`] —— 构建期生成的**资产符号清单**查表
//!    (`models::kernel::kv_manifest_gate` 的池写者/读者核对走它);
//! 3. [`parse_cu_sig`] —— `.cu` 形参序列解析器(**唯一**实现:表⇔签名互证门
//!    与准入门共用它 —— 两道门各写一份 = 各说各话)。
//!
//! 为什么与算子值分开:`Kernel`/构造面服务**发射**,本模块服务**取证**
//! (测试门 + boot 清单门),生命周期与依赖面都不同。

include!(concat!(env!("OUT_DIR"), "/cubin_symbols.rs"));

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

// ⚠️ **不是死代码,勿删**(2026-10-15 审查定谳):`parse_cu_sig` 是**唯一**的
// `.cu` 形参解析器 —— `tests/op_tables.rs` 的"声明表 ⇔ 签名"互证门靠它,
// `models::kernel` 也再导出它。删掉 = 互证门失明(原 FIXME 误导已撤)。
pub fn parse_cu_sig(src: &str, name: &str) -> Option<String> {
    let (params, kind) = cu_params_of(src, name)?;
    Some(map_params(&params, name, kind))
}

/// **核形参串取证**(三形态统一;**唯一实现** —— token 级互证
/// [`parse_cu_sig`] 与名字级互证 [`parse_cu_param_names`] 共用,
/// 两道门不许各写一遍解析器)。
///
/// 返回 `(形参串, 形态标签)`。三形态:
/// 1. 宏展开核实例行 `OWL_<FAMILY>_KERNEL(<name>, <T>, PARAMS)`;
/// 2. 单层宏实例化核 `CHUNKED_KNHD_ENTRY(<name>, <hd>, <kv_fp8>)`
///    (形参住 `#define` 定义体);
/// 3. 字面核 `extern "C" __global__ void <name>(…)`。
pub fn cu_params_of(src: &str, name: &str) -> Option<(String, &'static str)> {
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
                    return Some((params.to_string(), "宏"));
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
        let start = d + s + body_anchor.len();
        let end = clean[start..].find(')')? + start;
        return Some((clean[start..end].to_string(), "宏"));
    }
    // 形态 3:字面核 `extern "C" __global__ void <name>(…)`
    let anchor = format!("__global__ void {name}(");
    let start = clean.find(&anchor)? + anchor.len();
    let end = clean[start..].find(')')? + start;
    Some((clean[start..end].to_string(), ""))
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

    /// **每张原生表的资产与符号清单在场**(原"登记表有源"门的等价升级)。
    #[test]
    fn every_compiled_symbol_is_declared() {
        let und = ops::undeclared_symbols();
        assert!(und.is_empty(), "编出但未声明的核符号 {} 条:{und:?}", und.len());
        eprintln!("[R1] 编出符号集 ⊆ 声明集 ✓(登记表已退役,判据换源)");
    }

}

// ============================================================================
// §4 `.cu` 形参**名**解析(名字级互证;2026-10-15 L3 事实面补)
// ============================================================================
//
// 为什么需要它:token 级互证(`parse_cu_sig`)只比 T/sz/i32/f32 的**形状**,
// 看不见"同形状、不同语义槽位"的漂移 —— 2026-10-15 实测到活体:
// `owl_qknorm_rope_kv_insert_f16{,_knhd}` 的声明表是**旧签名**
// (`q, q_raw, kw, beta, cos, sin, k, v, kc, vc, slots, …`),
// 而 .cu 早已是 `q_raw, k, v, key_cache, value_cache, slots, q_w, k_w, …`
// —— 两者 token 串**逐字相同**(11×T + f32 + 4×i32 + T),门全绿,
// 而语义槽位全错位(调用点按新序喂 ⇒ f16 档融合插池读写全错)。

/// **核形参名序列**(名字级互证;与 token 级共用 [`cu_params_of`] 取证口)。
///
/// 为什么需要它:token 级只比 T/sz/i32/f32 的**形状**,看不见"同形状、不同
/// 语义槽位"的漂移 —— 2026-10-15 实测活体 `owl_qknorm_rope_kv_insert_f16{,_knhd}`
/// (声明旧签名 / .cu 新签名,token 串逐字相同)。
pub fn parse_cu_param_names(src: &str, name: &str) -> Option<Vec<String>> {
    let (params, _kind) = cu_params_of(src, name)?;
    Some(
        params
            .split(',')
            .filter_map(|p| {
                let p = p
                    .replace("const ", "")
                    .replace("__restrict__ ", "")
                    .replace('*', " ")
                    .replace('&', " ");
                let last = p.split_whitespace().last()?.trim_matches(|c: char| {
                    matches!(c, ')' | '(' | ';' | '{' | '}' | ',')
                }).to_string();
                if last.is_empty() { None } else { Some(last) }
            })
            .collect(),
    )
}

