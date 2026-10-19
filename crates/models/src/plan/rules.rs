//! **权重切分规则表**(S4 ②)—— 键模式 → 切分形态的**唯一一张表**。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §15.3。
//!
//! # 为什么是"一张表"(而不是每层各写一份)
//!
//! 切分形态是**权重键的属性**,不是层的属性:同一个 `Linear` 结构体既可能
//! 装 `gate_proj`(切 out)又可能装 `down_proj`(切 in)。把规则写成**数据**
//! (这张表)⇒ 装载面与 AR 注入面**同读一张表**(单源);
//! 层结构体不新增任何字段(用户裁决:切分不进结构体)。
//!
//! # 消费者(两个,一张表)
//!
//! 1. **装载面**:按本表算"本 slice 要读的源区间"与分片形状(§15.3 的矩形切);
//! 2. **AR 注入面**:`Linear::forward` 查 [`needs_ar`](self::needs_ar) 决定
//!    是否追加 allreduce(行并行输出 = 部分和)。
//!
//! # 纪律
//!
//! - **默认 `Replicate`**(大多数键不切);**新模型的融合权重必须入表** ——
//!   漏入不会静默:装载面按分片形状与实读字节对账,不符即拒;
//! - 段长(QGate/QKV/ConvQKV)**不写在表里**(那是 spec 的数值),
//!   表只给语义标记,由 [`SegKind::segments`] 按 `ModelSpec` 展开 ——
//!   这样表是静态数据,模型换规格不用改表。

use crate::model::ModelSpec;

/// 切分形态(规则词汇;与 `plan::tp::TpSplit` 是"同义不同层":
/// 这里描述**权重键**,那里描述**张量维**)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightRule {
    /// 不切(每 slice 全量;norm/embed/lm_head/小向量以外的杂项)
    Replicate,
    /// out 维等分(行切;下游消费分片,不需要 AR)
    UniformCol,
    /// in 维等分(列切;**输出 = 部分和 ⇒ 需要 AR**)
    UniformRow,
    /// out 维**分段**切(融合权重;段长按 spec 展开)
    SegmentedCol(SegKind),
    /// 按头切的小向量(A_log/dt_bias;等价 out 等分,单列以示语义)
    PerHead,
}

/// 分段语义(表只给标记;段长按 [`SegKind::segments`] 展开)。
///
/// 段结构 = **层代码的 narrow 顺序**(单源:GDN/Attention 的 forward)——
/// 键名与段数已按真实 want 键逐条核实(2026-10-17):
/// - `q_proj` 是 `Linear::new("q_proj", hq*hd*2, …)`(value|gate 融合);
/// - `in_proj_qkvz` 是 `[2K+2V, hidden]` 四段(源侧 `in_proj_qkv`+`in_proj_z`
///   由装载源**虚拟合成**为这一个 want);
/// - `conv1d` 是 `[2K+V, 4]` 三段(`conv_dim = 2·key_dim + value_dim`)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegKind {
    /// `q_proj = q ⊕ gate`(Qwen3Next 门控注意力:两段各 hq 头 × hd)
    QGate,
    /// `in_proj_qkvz = q ⊕ k ⊕ v ⊕ z`(GDN 四段:q/k 各 nk 头、v/z 各 nv 头)
    QKVZ,
    /// `conv1d` 的通道 `= q ⊕ k ⊕ v`(三段;`conv_dim = 2K + V`)
    ConvQKV,
}

impl SegKind {
    /// 段长展开(源坐标系;按 `ModelSpec` 的**全量**维度 —— 不是分片后的)
    pub fn segments(self, spec: &ModelSpec) -> Result<Vec<usize>, String> {
        let (hq, _hkv, hd) = spec.full_heads;
        let (nk, hk, nv, hv) = spec.gdn_heads;
        let (k_dim, v_dim) = (nk * hk, nv * hv);
        match self {
            SegKind::QGate => Ok(vec![hq * hd, hq * hd]),
            SegKind::QKVZ => Ok(vec![k_dim, k_dim, v_dim, v_dim]),
            SegKind::ConvQKV => Ok(vec![k_dim, k_dim, v_dim]),
        }
    }
}

/// 尾巴匹配:`key == pat` 或 `key` 以 `.pat` 结尾(兼容"层内短键"与
/// "checkpoint 全键"两种形态)。**唯一实现**(规则表与段长定向标注共用)
pub(crate) fn tail(key: &str, pat: &str) -> bool {
    key.len() >= pat.len() && (key == pat || key.ends_with(pat)) && {
        let cut = key.len() - pat.len();
        cut == 0 || key.as_bytes()[cut - 1] == b'.'
    }
}

/// **派生键尾巴**(量化族:同一份源数据的**布局产物**,不是语义键)。
/// 规则表按**语义键**匹配 ⇒ 查规则前必须先剥掉它们(2026-10-17 定谳:
/// 不剥 ⇒ 回落 `Replicate` ⇒ 分片区间不翻译 ⇒ 两 rank 读同一段)。
pub const DERIVED_SUFFIXES: &[&str] = &[
    ".qweight",
    ".qweight_a8",
    ".packed_raw",
    ".scales",
    ".scales_a8",
    ".zeros",
    ".zeros_a8",
    ".marlin_ws",
    ".marlin_ctmp",
    ".a8_factor",
];

/// 剥派生后缀 → 语义基键(未命中则原样返回)。
pub fn base_key(key: &str) -> &str {
    for s in DERIVED_SUFFIXES {
        if let Some(b) = key.strip_suffix(s) {
            return b;
        }
    }
    key
}

/// **非数据派生件**(scratch / 标量):形状由层按分片 spec 直接声明,
/// **不参与区间翻译** —— 规则一律 [`WeightRule::Replicate`](内容无意义:
/// `marlin_ws` 是内核暂存区、`marlin_ctmp`/`a8_factor` 是单元素/标量)。
/// 不排除它们 = 会被当成融合权重的段切目标而形状不符 ⇒ 无谓拒启。
const SCRATCH_SUFFIXES: &[&str] = &[".marlin_ws", ".marlin_ctmp", ".a8_factor"];

/// 是否非数据派生件(见 [`SCRATCH_SUFFIXES`])
pub fn is_scratch_key(key: &str) -> bool {
    SCRATCH_SUFFIXES.iter().any(|s| key.ends_with(s))
}

/// **want 形状的坐标序**(规则表按**逻辑** `[out, k]` 写)。
///
/// 量化族的 marlin 布局产物是 **k-major**(`[k 侧, n 侧]`;物理序由打包函数
/// 定死,改序会让内核读错):`scales`/`zeros` 的打包索引就是 `g*out + o`
/// (`pack_marlin_z` 注释"(out, groups) → 转置 (groups, n=out)"),
/// `qweight` 的 marlin B 视图同理。
/// `packed_raw`(原始 AWQ packed)反而是 **n-major** `[out, k/8]` ⇒ Logical。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WantAxis {
    /// `[out, k…]`(规则表原生坐标):`.weight` / `.packed_raw` / `A_log` …
    Logical,
    /// `[k…, out…]`(marlin 布局产物):`.qweight` / `.scales` / `.zeros`(含 a8 族)
    KMajor,
}

/// 键 → 坐标序(唯一映射点;span 层据此把规则的"切哪一维"落到真实维度)。
pub fn want_axis(key: &str) -> WantAxis {
    const KMAJOR: &[&str] = &[
        ".qweight",
        ".qweight_a8",
        ".scales",
        ".scales_a8",
        ".zeros",
        ".zeros_a8",
    ];
    if KMAJOR.iter().any(|s| key.ends_with(s)) {
        WantAxis::KMajor
    } else {
        WantAxis::Logical
    }
}

/// **键 → 规则**(唯一入口;探测顺序从具体到一般)。
///
/// ⚠️ **只认 owl 语义名**(`gate_proj`/`out_proj`/`in_proj_qkvz`…),**不认**
/// GGUF 的文件名(`blk.N.ffn_gate` 等):AR 标记在**层构造期**查表(那时用的
/// 就是短键),GGUF 的分片几何又走"源自担"路线(`formats/gguf/shard.rs` 的
/// 角色表)—— 所以全键永远不会到这里。将来若有代码拿**全键**查本表,必须先
/// 经 `GgufConvention` 反译;否则会静默落 `Replicate`(漏 AR)。
pub fn rule_of(key: &str) -> WeightRule {
    // 非数据派生件(scratch/标量):一律不切(形状已按分片 spec 声明)
    if is_scratch_key(key) {
        return WeightRule::Replicate;
    }
    // 派生键(量化布局产物)与语义键**同规则** ⇒ 先剥后缀(2026-10-17)
    let key = base_key(key);
    // 去掉 `.weight`/`.bias` 尾巴(规则按语义 token 匹配)
    let k = key
        .strip_suffix(".weight")
        .or_else(|| key.strip_suffix(".bias"))
        .unwrap_or(key);

    // 匹配用**最短唯一尾巴**(want 键是层内短键、改键后是全局键 —— 两种形态通吃;
    // 键名已按层代码逐条核实:见 `gdn.rs` / `attention.rs` 的 `Linear::new` 实参)
    // ── MLP ──
    if tail(k, "gate_proj") || tail(k, "up_proj") {
        return WeightRule::UniformCol;
    }
    if tail(k, "down_proj") {
        return WeightRule::UniformRow;
    }
    // ── full attention(Qwen3Next 门控)──
    if tail(k, "q_proj") {
        return WeightRule::SegmentedCol(SegKind::QGate);
    }
    if tail(k, "k_proj") || tail(k, "v_proj") {
        return WeightRule::UniformCol;
    }
    if tail(k, "o_proj") {
        return WeightRule::UniformRow;
    }
    // ── GDN(linear_attn)──
    // ⚠️ want 键是 `in_proj_qkvz`(四段,源侧由 in_proj_qkv + in_proj_z 虚拟合成);
    // 检查点的 `in_proj_qkv` / `in_proj_z` 是**源的内部实现**,不在此表。
    if tail(k, "in_proj_qkvz") {
        return WeightRule::SegmentedCol(SegKind::QKVZ);
    }
    if tail(k, "in_proj_a") || tail(k, "in_proj_b") {
        return WeightRule::PerHead;
    }
    if tail(k, "out_proj") {
        return WeightRule::UniformRow;
    }
    if tail(k, "conv1d") {
        return WeightRule::SegmentedCol(SegKind::ConvQKV);
    }
    if tail(k, "A_log") || tail(k, "dt_bias") {
        return WeightRule::PerHead;
    }

    // ── 其余(spec 里的小权重 + embed/lm_head/norm)──
    WeightRule::Replicate
}

/// **AR 注入判据**:该键所在层的输出是否为**部分和**(需要 allreduce)。
///
/// 行并行(`UniformRow`)= 输入被切 ⇒ 输出是部分和 ⇒ 需要 AR。
/// 分段/等分的 out 切法输出的是**分片**(下游按头/按块消费)⇒ 不需要。
pub fn needs_ar(key: &str) -> bool {
    matches!(rule_of(key), WeightRule::UniformRow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::tp::TpPlan;

    fn spec27b() -> ModelSpec {
        crate::formats::qwen35::qwen3_8_27b()
    }

    /// 27B 的键(两种形态:checkpoint 全键 + 层内短键)→ 规则归类
    #[test]
    fn rules_cover_27b_keys() {
        let cases: &[(&str, WeightRule)] = &[
            ("model.layers.0.mlp.gate_proj.weight", WeightRule::UniformCol),
            ("mlp.up_proj.weight", WeightRule::UniformCol),
            ("model.layers.3.mlp.down_proj.weight", WeightRule::UniformRow),
            ("model.layers.3.self_attn.q_proj.weight", WeightRule::SegmentedCol(SegKind::QGate)),
            ("self_attn.k_proj.weight", WeightRule::UniformCol),
            ("model.layers.3.self_attn.v_proj.weight", WeightRule::UniformCol),
            ("model.layers.3.self_attn.o_proj.weight", WeightRule::UniformRow),
            // want 键(层内短键)→ 四段融合
            ("in_proj_qkvz.weight", WeightRule::SegmentedCol(SegKind::QKVZ)),
            ("model.layers.0.linear_attn.in_proj_qkvz.weight", WeightRule::SegmentedCol(SegKind::QKVZ)),
            ("linear_attn.in_proj_a.weight", WeightRule::PerHead),
            ("model.layers.0.linear_attn.in_proj_b.weight", WeightRule::PerHead),
            ("model.layers.0.linear_attn.out_proj.weight", WeightRule::UniformRow),
            ("model.layers.0.linear_attn.conv1d.weight", WeightRule::SegmentedCol(SegKind::ConvQKV)),
            ("model.layers.0.linear_attn.A_log", WeightRule::PerHead),
            ("model.layers.0.linear_attn.dt_bias", WeightRule::PerHead),
            // 其余一律 Replicate
            ("model.layers.0.input_layernorm.weight", WeightRule::Replicate),
            ("model.layers.0.post_attention_layernorm.weight", WeightRule::Replicate),
            ("model.layers.0.linear_attn.norm.weight", WeightRule::Replicate),
            ("model.layers.3.self_attn.q_norm.weight", WeightRule::Replicate),
            ("model.norm.weight", WeightRule::Replicate),
            ("model.embed_tokens.weight", WeightRule::Replicate),
            ("lm_head.weight", WeightRule::Replicate),
        ];
        for (key, want) in cases {
            assert_eq!(rule_of(key), *want, "键 {key} 规则错");
        }
    }

    /// AR 判据:只有行并行(in 切)的输出需要 AR
    #[test]
    fn only_row_parallel_needs_ar() {
        let need = [
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.3.self_attn.o_proj.weight",
            "model.layers.0.linear_attn.out_proj.weight",
        ];
        let no_need = [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
            "in_proj_qkvz.weight",
            "model.layers.0.input_layernorm.weight",
            "model.embed_tokens.weight",
        ];
        for k in need {
            assert!(needs_ar(k), "{k} 应需要 AR");
        }
        for k in no_need {
            assert!(!needs_ar(k), "{k} 不应需要 AR");
        }
    }

    /// 段长展开(27B 全量维度)× 分段切 ⇒ 每 rank 的分片形状
    #[test]
    fn segment_lengths_match_27b_shapes() {
        let spec = spec27b();
        assert_eq!(SegKind::QGate.segments(&spec).unwrap(), vec![6144, 6144], "q ⊕ gate");
        assert_eq!(
            SegKind::QKVZ.segments(&spec).unwrap(),
            vec![2048, 2048, 6144, 6144],
            "q ⊕ k ⊕ v ⊕ z(27B:2K=4096 + 2V=12288 = 16384)"
        );
        assert_eq!(SegKind::ConvQKV.segments(&spec).unwrap(), vec![2048, 2048, 6144], "conv q ⊕ k ⊕ v");
        // world=2 下,每段取一半(分段切的实际区间)
        let p = TpPlan::new(1, 2).unwrap();
        let qkv = SegKind::ConvQKV.segments(&spec).unwrap();
        assert_eq!(
            p.segmented_ranges(&qkv).unwrap(),
            vec![1024..2048, 1024..2048, 3072..6144],
            "rank1 的三段后一半"
        );
    }
}
