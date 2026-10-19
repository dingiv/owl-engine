//! **GGUF 读侧变换表**(设计 §十一 六类变换的 per-key 描述符;2026-10-19 刀 4b)。
//!
//! 写侧(转换器 `llama.cpp/conversion/qwen.py`)做了六类变换;读侧必须逆变换。
//! 本模块 = **纯函数**:`(键, 维度) → 变换描述符`,零 I/O 零设备 ⇒ 可单测。
//!
//! # 代价分层(哪些能字节级做)
//!
//! 关键几何:量化块沿 **`ne[0]`(内维)**、`QK_K = 256`,而 `hv = 128`
//! ⇒ **一个 v-head = 半个块**。于是:
//!
//! | 变换 | 粒度 | 实现 |
//! |---|---|---|
//! | `attn_qkv` 的 V 段 / `in_proj_z` 行 | **行**(行 = 内维的完整块链) | **字节级**(块组搬移)✓ |
//! | `in_proj_a`/`in_proj_b` 行(head_dim=1) | 行 | 字节级 ✓ |
//! | `ssm_out` **列**(内维) | 跨块 ✗ | **激活侧等价置换**(`owl_gguf_gather_cols`) |
//! | `conv1d` V 通道段 | 通道(F32) | host 重排 ✓ |
//! | `ssm_a` → `ln(-x)` | 逐元素(F32) | host ✓ |
//! | norm `+1` | 逐元素 | **语义法**(`w_off=false`;零变换)✓ |
//!
//! # V-head 序公式(实测反解 + 转换器源码互证)
//!
//! 转换器把 HF 的 **grouped** 序换成 ggml 的 **tiled** 序
//! (`_reorder_v_heads`:`[nk, vpr, hd] →permute→ [vpr, nk, hd]`)。
//!
//! ```text
//! vpr = nv / nk
//! 写(HF → GGUF): g = (h % vpr) * nk + (h / vpr)
//! 读(GGUF → HF): h = (g % nk) * vpr + (g / nk)      ← 本模块的 `vhead_perm`
//! ```
//!
//! 实测证据(4B layer 0,逐行精确匹配):`ssm_alpha`/`ssm_beta`/`ssm_dt.bias`
//! 逆变换后**精确 0**、`ssm_a`→`ln(-x)` maxΔ 6e-8、`attn_qkv` V 段 rel 1.6e-2
//! (= 量化噪声)✓

use crate::contract::ModelError;
use owl_kernels::family::gguf::GgmlTy;

/// GGUF 侧维度事实(变换表需要的全部输入)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufDims {
    /// key 头数(GDN)
    pub nk: usize,
    /// value 头数(GDN)
    pub nv: usize,
    /// key 头维
    pub hk: usize,
    /// value 头维(**v-head 的元素宽度**)
    pub hv: usize,
    /// full-attn 的 q/kv 头数(判定"是否需要 v-head 重排":`nkv == nq` ⇒ 不需要)
    pub nq: usize,
    pub nkv: usize,
}

impl GgufDims {
    /// v-head 重排是否生效(转换器只在 `nk != nv` 时做)
    pub fn vhead_active(&self) -> bool {
        self.nk > 0 && self.nv > 0 && self.nk != self.nv
    }
    /// 每 k 头挂几个 v 头
    pub fn vpr(&self) -> usize {
        if self.nk == 0 {
            0
        } else {
            self.nv / self.nk
        }
    }
    /// GDN 段几何(q|k|v 三段;`conv_dim = 2·nk·hk + nv·hv`)
    pub fn seg(&self) -> (usize, usize, usize) {
        (self.nk * self.hk, self.nk * self.hk, self.nv * self.hv)
    }
}

/// 逐键的**值变换**(逐元素;小件走 host)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueXf {
    /// 原值直读
    None,
    /// `x → ln(-x)`(GGUF `ssm_a` 存的是 `A = -exp(A_log)`)
    LnNeg,
    /// `x → x - 1`(GGUF norm = HF 值 + 1)—— **本仓不采用**(改用语义法
    /// `w_off=false`,零变换);保留变体供诊断/对照
    MinusOne,
}

/// 行置换目标(行粒度;**字节级可做** —— 行 = 内维的完整块链)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowPerm {
    /// 不置换
    No,
    /// **整张量按 v-head 重排**(行 = v-head 连续段)
    VHead { nk: usize, nv: usize, hd: usize },
    /// **仅 V 段**重排(q/k 段直拷;`attn_qkv`)
    VSegOnly { nk: usize, nv: usize, hd: usize, seg_off: usize },
    /// **仅 V 通道段**重排(`conv1d`;F32 host)
    VChanOnly { nk: usize, nv: usize, hd: usize, seg_off: usize },
    /// **列**(内维)重排 —— 跨块 ⇒ 由**激活侧**负责(`ssm_out`);源不动字节
    ColVHead { nk: usize, nv: usize, hd: usize },
}

/// 逐键变换描述符
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyXf {
    pub value: ValueXf,
    pub row_perm: RowPerm,
}

impl KeyXf {
    pub const IDENTITY: KeyXf = KeyXf { value: ValueXf::None, row_perm: RowPerm::No };
    /// 源是否需要动字节(行置换)
    pub fn touches_bytes(&self) -> bool {
        !matches!(self.row_perm, RowPerm::No | RowPerm::ColVHead { .. })
    }
}

/// 键分类(层内局部名 / 顶层名;与 `GgufConvention` 同一词表)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyClass {
    /// 顶层:词表 / 头 / 最终 norm
    Embed,
    Head,
    Norm,
    /// 层内:GDN 的哪一件
    Gdn(GdnPart),
    /// 层内:full-attn / ffn / 两个 norm
    Plain,
    /// 不认识(调用方决定拒还是忽略)
    Unknown,
}

/// GDN 层内的可变换件
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnPart {
    /// `attn_qkv.weight`(= in_proj_qkv ⊕ z 的第一段)
    Qkv,
    /// `attn_gate.weight`(= in_proj_z)
    Z,
    /// `ssm_out.weight`(列 = v-head)
    Out,
    /// `ssm_alpha` / `ssm_beta`(in_proj_a/b;行 = v-head,head_dim=1)
    Ab,
    /// `ssm_dt.bias`(行 = v-head)
    Dt,
    /// `ssm_a`(A_log;**行 = v-head** + `ln(-x)`)
    A,
    /// `ssm_conv1d`(V 通道段)
    Conv,
    /// `ssm_norm`(**不 +1**,不重排)
    Norm,
}

/// 键名 → 类(纯字符串判定;不识别的层内名 = `Unknown`)
pub fn classify(name: &str) -> KeyClass {
    if name == "token_embd.weight" {
        return KeyClass::Embed;
    }
    if name == "output.weight" {
        return KeyClass::Head;
    }
    if name == "output_norm.weight" {
        return KeyClass::Norm;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return KeyClass::Unknown;
    };
    let Some((_, local)) = rest.split_once('.') else {
        return KeyClass::Unknown;
    };
    match local {
        "attn_qkv.weight" => KeyClass::Gdn(GdnPart::Qkv),
        "attn_gate.weight" => KeyClass::Gdn(GdnPart::Z),
        "ssm_out.weight" => KeyClass::Gdn(GdnPart::Out),
        "ssm_alpha.weight" | "ssm_beta.weight" => KeyClass::Gdn(GdnPart::Ab),
        "ssm_dt.bias" => KeyClass::Gdn(GdnPart::Dt),
        "ssm_a" => KeyClass::Gdn(GdnPart::A),
        "ssm_conv1d.weight" => KeyClass::Gdn(GdnPart::Conv),
        "ssm_norm.weight" => KeyClass::Gdn(GdnPart::Norm),
        _ => KeyClass::Plain,
    }
}

/// **逐键变换描述符**(唯一入口;纯函数)
///
/// `ty` 参与判定吗?不参与 —— 变换是**布局**事实,与块型无关(块型只决定
/// "字节级能不能做",那由 `RowPerm` 的粒度回答)。
pub fn xf_for(name: &str, d: &GgufDims) -> Result<KeyXf, ModelError> {
    let active = d.vhead_active();
    let (_, _, v) = d.seg();
    let ok = |p: RowPerm| KeyXf { value: ValueXf::None, row_perm: p };
    let out = match classify(name) {
        KeyClass::Embed | KeyClass::Head | KeyClass::Norm | KeyClass::Unknown => KeyXf::IDENTITY,
        KeyClass::Plain => KeyXf::IDENTITY,
        KeyClass::Gdn(GdnPart::Norm) => KeyXf::IDENTITY,
        KeyClass::Gdn(GdnPart::A) => {
            let row = if active {
                RowPerm::VHead { nk: d.nk, nv: d.nv, hd: 1 }
            } else {
                RowPerm::No
            };
            KeyXf { value: ValueXf::LnNeg, row_perm: row }
        }
        KeyClass::Gdn(GdnPart::Dt) | KeyClass::Gdn(GdnPart::Ab) => {
            if active {
                ok(RowPerm::VHead { nk: d.nk, nv: d.nv, hd: 1 })
            } else {
                KeyXf::IDENTITY
            }
        }
        KeyClass::Gdn(GdnPart::Z) => {
            if active {
                ok(RowPerm::VHead { nk: d.nk, nv: d.nv, hd: d.hv })
            } else {
                KeyXf::IDENTITY
            }
        }
        KeyClass::Gdn(GdnPart::Qkv) => {
            if active {
                ok(RowPerm::VSegOnly { nk: d.nk, nv: d.nv, hd: d.hv, seg_off: d.nk * d.hk * 2 })
            } else {
                KeyXf::IDENTITY
            }
        }
        KeyClass::Gdn(GdnPart::Conv) => {
            if active {
                ok(RowPerm::VChanOnly { nk: d.nk, nv: d.nv, hd: d.hv, seg_off: d.nk * d.hk * 2 })
            } else {
                KeyXf::IDENTITY
            }
        }
        KeyClass::Gdn(GdnPart::Out) => {
            // 列重排:源不动字节,**调用方**在激活侧做(`owl_gguf_gather_cols`)
            let _ = v;
            if active {
                ok(RowPerm::ColVHead { nk: d.nk, nv: d.nv, hd: d.hv })
            } else {
                KeyXf::IDENTITY
            }
        }
    };
    Ok(out)
}

/// **v-head 逆重排索引**(GGUF 位置 g → HF/owl 位置 h;`perm[h] = g`)
///
/// 用于:①行置换(取源行 `perm[i]` 放到目标行 `i`);②激活侧列置换(同样取
/// 源 `perm[j]` 放到目标 `j` —— 两个方向在**置换表**上同形,方向由用例表钉定)。
pub fn vhead_deperm(nk: usize, nv: usize, hd: usize) -> Vec<u32> {
    let vpr = nv / nk;
    let mut p = Vec::with_capacity(nv * hd);
    for h in 0..nv {
        let g = (h % vpr) * nk + (h / vpr);
        for e in 0..hd {
            p.push((g * hd + e) as u32);
        }
    }
    p
}

/// **内维(列)置换**(原生件的装载期扭回;设计 §11.4 落点 = `ColVHead`)
///
/// 仅对**无块约束**的原生件有效(`F32`/`F16`/`BF16`):逐元素搬,零精度损失。
/// 量化件的同一位置换**跨量化块** ⇒ 不可在字节/元素面做,由**激活侧**
/// `owl_gguf_quant_act_*_tiled` 承担(同一张表,同一公式)。
///
/// 形态门(不成立即 `Err` —— 不静默错值):
/// · `inner == nv·hd`(内维就是 v-head 轴);
/// · `map.len() == nv·hd`(表与内维同宽)。
pub fn col_deperm_in_place(
    v: &mut [f32],
    rows: usize,
    inner: usize,
    nk: usize,
    nv: usize,
    hd: usize,
) -> Result<(), ModelError> {
    if inner != nv * hd || rows == 0 || v.len() != rows * inner {
        return Err(ModelError::Msg(format!(
            "列置换形态不符:张量 [{rows}×{inner}] vs nv·hd = {nv}·{hd}(或缓冲尺寸不符)"
        )));
    }
    let map = vhead_deperm(nk, nv, hd);
    if map.len() != inner {
        return Err(ModelError::Msg(format!(
            "列置换表长 {} ≠ 内维 {inner}",
            map.len()
        )));
    }
    let mut out = vec![0f32; v.len()];
    for r in 0..rows {
        let base = r * inner;
        for (j, s) in map.iter().enumerate() {
            out[base + j] = v[base + *s as usize];
        }
    }
    v.copy_from_slice(&out);
    Ok(())
}

/// **q8_1 块粒度的 v-head 列序表**(`ssm_out` 的**激活侧**置换用;= 行置换表的**逆表**)
///
/// # 方向(★ 最容易反的一处:两个方向互为**逆**,不是同一个数组)
///
/// 记号:HF/owl 列(按 v-head)= `h`,文件列(ggml tiled)= `p`
/// (`h(p) = (p % nk) * vpr + p / nk` = 文件位置 p 上坐的是哪个 HF 头)。
///
/// | 用途 | 数组 | 取数式 | 消费者 |
/// |---|---|---|---|
/// | 权重侧(行/元素):文件 → HF | `vhead_deperm`,即 `g(j) = (j%vpr)*nk + j/vpr` | `out_hf[j] = in_file[g(j)]` | `GgufSource` 行置换 / 原生件列置换 |
/// | **激活侧**:HF 分组序 → 文件列序 | **本函数**,即 `h(p)` | `out_file[p] = in_grouped[h(p)]` | `owl_gguf_quant_act_*_tiled` 的表 |
///
/// 两者复合 = 恒等(`vhead_perm_is_inverse_of_deperm` 单测钉死)。**写反了不会报错、
/// 只会整数列错位** —— 所以本函数与 `vhead_deperm` 分名、分注释、并有互逆门。
///
/// # 为什么能按块做
///
/// `hv` 被 q8_1 块宽(`QK_32` = 32)整除 ⇒ **一个 v-head = 整数块**(否则响亮拒:
/// 置换单元与量化块错位 = 静默错值)。表长 = `nv * hd / 32`。
pub fn vhead_q8_blocks(nk: usize, nv: usize, hd: usize) -> Result<Vec<u32>, ModelError> {
    const Q8: usize = owl_kernels::family::gguf::QK_32;
    if hd % Q8 != 0 || nk == 0 || nv % nk != 0 {
        return Err(ModelError::Msg(format!(
            "v-head 列置换:头维 {hd} 不是 q8_1 块大小 {Q8} 的整数倍(或 nk={nk} 不整除 nv={nv})⇒ 块粒度置换不成立"
        )));
    }
    let bph = hd / Q8; // 每个 v-head 的 q8_1 块数
    let vpr = nv / nk;
    let mut p = Vec::with_capacity(nv * bph);
    for g in 0..nv {
        // ★ 方向:h(g) = (g % nk) * vpr + g / nk(文件位置 g 上的 HF 头)
        let h = (g % nk) * vpr + (g / nk);
        for b in 0..bph {
            p.push((h * bph + b) as u32);
        }
    }
    Ok(p)
}

/// 全模型逐键变换清单(诊断/门:列出所有**非恒等**键)
pub fn non_identity_keys(g: &crate::formats::gguf::reader::Gguf, d: &GgufDims) -> Vec<(String, KeyXf)> {
    let mut out = Vec::new();
    for t in &g.tensors {
        if let Ok(xf) = xf_for(&t.name, d) {
            if xf != KeyXf::IDENTITY {
                out.push((t.name.clone(), xf));
            }
        }
    }
    out
}

/// 块型 → 该型是否"字节级行置换可行"(诊断;行粒度恒可行,列粒度恒不可行)
pub const fn row_perm_is_byte_level(_ty: GgmlTy) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims_4b() -> GgufDims {
        GgufDims { nk: 16, nv: 32, hk: 128, hv: 128, nq: 16, nkv: 4 }
    }
    fn dims_27b() -> GgufDims {
        GgufDims { nk: 16, nv: 48, hk: 128, hv: 128, nq: 24, nkv: 4 }
    }

    /// **逆重排公式自证**:`perm` 与写侧公式互逆(往返 = 恒等)
    #[test]
    fn vhead_perm_round_trips() {
        for (nk, nv, hd) in [(16, 32, 128), (16, 48, 128), (2, 4, 1)] {
            let vpr = nv / nk;
            let p = vhead_deperm(nk, nv, hd);
            // 写:g = (h%vpr)*nk + h/vpr;读:perm[h] = g
            for h in 0..nv {
                let g = (h % vpr) * nk + (h / vpr);
                assert_eq!(p[h * hd], (g * hd) as u32, "h={h}");
                // 逆:给定 g,反解 h
                let hh = (g % nk) * vpr + (g / nk);
                assert_eq!(hh, h, "公式不互逆 h={h} g={g}");
            }
            // 是置换(0..nv*hd 全覆盖)
            let mut s = p.clone();
            s.sort_unstable();
            assert_eq!(s, (0..(nv * hd) as u32).collect::<Vec<_>>());
        }
    }

    /// 变换表逐类(4B:6 类非恒等键各就其位)
    #[test]
    fn xf_table_4b() {
        let d = dims_4b();
        let cases: &[(&str, KeyXf)] = &[
            ("blk.0.attn_qkv.weight", KeyXf { value: ValueXf::None, row_perm: RowPerm::VSegOnly { nk: 16, nv: 32, hd: 128, seg_off: 4096 } }),
            ("blk.0.attn_gate.weight", KeyXf { value: ValueXf::None, row_perm: RowPerm::VHead { nk: 16, nv: 32, hd: 128 } }),
            ("blk.0.ssm_out.weight", KeyXf { value: ValueXf::None, row_perm: RowPerm::ColVHead { nk: 16, nv: 32, hd: 128 } }),
            ("blk.0.ssm_alpha.weight", KeyXf { value: ValueXf::None, row_perm: RowPerm::VHead { nk: 16, nv: 32, hd: 1 } }),
            ("blk.0.ssm_beta.weight", KeyXf { value: ValueXf::None, row_perm: RowPerm::VHead { nk: 16, nv: 32, hd: 1 } }),
            ("blk.0.ssm_dt.bias", KeyXf { value: ValueXf::None, row_perm: RowPerm::VHead { nk: 16, nv: 32, hd: 1 } }),
            ("blk.0.ssm_a", KeyXf { value: ValueXf::LnNeg, row_perm: RowPerm::VHead { nk: 16, nv: 32, hd: 1 } }),
            ("blk.0.ssm_conv1d.weight", KeyXf { value: ValueXf::None, row_perm: RowPerm::VChanOnly { nk: 16, nv: 32, hd: 128, seg_off: 4096 } }),
            // 恒等:norm 族(ssm_norm **不 +1**;两个 layernorm 走语义法)/ ffn / full-attn / 顶层
            ("blk.0.ssm_norm.weight", KeyXf::IDENTITY),
            ("blk.0.attn_norm.weight", KeyXf::IDENTITY),
            ("blk.0.post_attention_norm.weight", KeyXf::IDENTITY),
            ("blk.0.ffn_gate.weight", KeyXf::IDENTITY),
            ("blk.0.ffn_up.weight", KeyXf::IDENTITY),
            ("blk.0.ffn_down.weight", KeyXf::IDENTITY),
            ("blk.3.attn_q.weight", KeyXf::IDENTITY),
            ("blk.3.attn_k.weight", KeyXf::IDENTITY),
            ("blk.3.attn_v.weight", KeyXf::IDENTITY),
            ("blk.3.attn_output.weight", KeyXf::IDENTITY),
            ("blk.3.attn_q_norm.weight", KeyXf::IDENTITY),
            ("blk.3.attn_k_norm.weight", KeyXf::IDENTITY),
            ("token_embd.weight", KeyXf::IDENTITY),
            ("output.weight", KeyXf::IDENTITY),
            ("output_norm.weight", KeyXf::IDENTITY),
        ];
        for (name, want) in cases {
            let got = xf_for(name, &d).expect("xf");
            assert_eq!(got, *want, "键 {name}");
        }
        // 27B:同一张表(只是 nv=48 进置换公式)
        let d27 = dims_27b();
        assert_eq!(
            xf_for("blk.1.attn_gate.weight", &d27).unwrap().row_perm,
            RowPerm::VHead { nk: 16, nv: 48, hd: 128 }
        );
        assert_eq!(
            xf_for("blk.1.attn_qkv.weight", &d27).unwrap().row_perm,
            RowPerm::VSegOnly { nk: 16, nv: 48, hd: 128, seg_off: 4096 }
        );
    }

    /// `vhead_active` 门:full-attn 层(nkv == nq)**无** v-head 概念 ⇒ 恒等;
    /// 且 GDN 的 `nk == nv` 档(如 0.8B nv=16=nk)也恒等(转换器 early-return)
    #[test]
    fn no_perm_when_not_needed() {
        let d = GgufDims { nk: 16, nv: 16, hk: 128, hv: 128, nq: 8, nkv: 2 };
        assert!(!d.vhead_active());
        assert_eq!(xf_for("blk.0.attn_gate.weight", &d).unwrap(), KeyXf::IDENTITY);
        assert_eq!(xf_for("blk.0.ssm_out.weight", &d).unwrap(), KeyXf::IDENTITY);
        assert_eq!(xf_for("blk.0.ssm_a", &d).unwrap(), KeyXf { value: ValueXf::LnNeg, row_perm: RowPerm::No });
    }

    /// **列(内维)置换**(原生件装载期扭回;设计 §11.4 落点 `ColVHead`)
    ///
    /// 两条判据:① **手算小例**(head 序从文件 tiled 序扭回 grouped 序的确切顺序);
    /// ② **互逆往返**(用逆表扭回 ⇒ 恒等)+ 形态门(内维 ≠ nv·hd ⇒ 拒)。
    #[test]
    fn col_deperm_exact_and_round_trip() {
        // nk=4, nv=8, hd=2(vpr=2):
        //   gagg(j) = (j%2)*4 + j/2 = [0,4,1,5,2,6,3,7]  ← 行置换表(文件 → HF)
        //   hagg(p) = (p%4)*2 + p/4 = [0,2,4,6,1,3,5,7]  ← 激活侧表(逆表)
        let (nk, nv, hd, rows) = (4usize, 8usize, 2usize, 1usize);
        let inner = nv * hd;
        let mut v: Vec<f32> = (0..rows * inner).map(|i| i as f32).collect();
        col_deperm_in_place(&mut v, rows, inner, nk, nv, hd).expect("列置换");
        assert_eq!(
            v,
            vec![
                0.0, 1.0, 8.0, 9.0, 2.0, 3.0, 10.0, 11.0, 4.0, 5.0, 12.0, 13.0, 6.0, 7.0,
                14.0, 15.0
            ],
            "列置换顺序不符(文件 head 0,4,1,5,… 应落到 HF head 0,1,2,3,…)"
        );
        // ② 互逆往返(逆表 = 激活侧表 h;两表都是置换 ⇒ 复合 = 恒等)
        let fwd = vhead_deperm(nk, nv, hd);
        let vpr = nv / nk;
        let inv: Vec<u32> = (0..nv)
            .map(|p| ((p % nk) * vpr + (p / nk)) as u32)
            .flat_map(|h| (0..hd as u32).map(move |e| h * hd as u32 + e))
            .collect();
        assert_ne!(fwd, inv, "两个方向得同一数组 ⇒ 本测形同虚设(nk,nv 太退化)");
        let mut back = v.clone();
        let src = back.clone();
        for (j, s) in inv.iter().enumerate() {
            back[j] = src[*s as usize];
        }
        assert_eq!(back, (0..inner).map(|i| i as f32).collect::<Vec<_>>(), "往返 ≠ 恒等");
        // 多行同表(行间独立)
        let mut v2: Vec<f32> = (0..2 * inner).map(|i| i as f32).collect();
        col_deperm_in_place(&mut v2, 2, inner, nk, nv, hd).expect("两行");
        assert_eq!(&v2[..inner], &v[..inner]);
        let v2_want: Vec<f32> = v.iter().map(|x| x + inner as f32).collect();
        assert_eq!(&v2[inner..], &v2_want[..], "第二行未按同表置换(行间应独立)");
        // 形态门:内维不是 nv·hd ⇒ 响亮拒
        let mut bad = vec![0f32; rows * inner];
        let e = col_deperm_in_place(&mut bad, rows, inner - 1, nk, nv, hd).unwrap_err().to_string();
        assert!(e.contains("形态不符"), "{e}");
    }

    /// q8 块粒度置换:hv 必须被 32 整除(否则响亮拒)
    #[test]
    fn q8_block_perm_requires_alignment() {
        assert!(vhead_q8_blocks(16, 32, 128).is_ok());
        assert!(vhead_q8_blocks(16, 32, 96).is_ok());
        let e = vhead_q8_blocks(16, 32, 48).unwrap_err().to_string();
        assert!(e.contains("整数倍"), "{e}");
        // 128 元素 / 32 = 4 块 ⇒ 表长 = nv * 4
        assert_eq!(vhead_q8_blocks(16, 32, 128).unwrap().len(), 32 * 4);
    }

    /// ★ **两个方向互逆**(行置换表 `g` 与本函数 `h`):
    /// `h[g[j]] == j` 且 `g[h[j]] == j` —— 这条门钉死“激活侧列序表不是行表”。
    /// 反了不会报错,只会整数列错位 ⇒ 必须机器判据。
    #[test]
    fn vhead_perm_is_inverse_of_deperm() {
        for (nk, nv, hd) in [(16, 32, 128), (16, 48, 128), (4, 8, 64)] {
            let g = vhead_deperm(nk, nv, hd); // 元素级:out_hf[j] = in_file[g[j]]
            let h = vhead_q8_blocks(nk, nv, hd).expect("块表");
            // 块表 → 元素级展开(每块 QK_32 元素;hd 被 32 整除 ⇒ 头内块对齐)
            let mut he = Vec::with_capacity(h.len() * 32);
            for b in &h {
                for e in 0..32 {
                    he.push((*b as usize * 32 + e) as u32);
                }
            }
            assert_eq!(he.len(), g.len(), "两表长度不等");
            assert_ne!(he, g, "({nk},{nv},{hd}) 两方向得同一数组 ⇒ 本门失效");
            for j in 0..g.len() {
                assert_eq!(he[g[j] as usize] as usize, j, "h∘g ≠ id @ {j}");
                assert_eq!(g[he[j] as usize] as usize, j, "g∘h ≠ id @ {j}");
            }
        }
    }

    /// 值变换/字节触碰标记(源侧分派依据)
    #[test]
    fn xf_flags() {
        let d = dims_4b();
        assert!(xf_for("blk.0.attn_gate.weight", &d).unwrap().touches_bytes());
        assert!(xf_for("blk.0.ssm_alpha.weight", &d).unwrap().touches_bytes());
        // 列置换**不碰字节**(激活侧做)
        assert!(!xf_for("blk.0.ssm_out.weight", &d).unwrap().touches_bytes());
        // 恒等
        assert!(!xf_for("blk.0.ffn_up.weight", &d).unwrap().touches_bytes());
    }
}
