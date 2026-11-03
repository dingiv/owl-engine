//! **GGUF + TP 分片几何**(W-1;2026-10-19;设计 §14.6 的落地)。
//!
//! # 为什么 GGUF 的分片不能复用 `formats::shard` 的 span 包装器
//!
//! 通用 pass 的模型是"**矩形切一个连续的源区间**"(`Rows`/`Cols`/`RowsSeg`/
//! `ColsSeg`,坐标 = 源的元素序)。GGUF 有两条事实让它不成立:
//!
//! 1. **量化块的列切是块算术**(切点必须落在 `be` 元素边界,目标行宽 = 块数 ×
//!    块字节)⇒ 单位不是元素而是**原子**(一个块 / 一个原生元素);
//! 2. **v-head 的全局置换**:转换器把 HF 的 grouped 序写成了 ggml 的 tiled 序
//!    (`g = (h % vpr) * nk + h / vpr`)。单卡时靠 A2 表在**激活侧**扭回;
//!    TP 下"各切一半文件序"会拿到**非 grouped 连续**的头集 ⇒ 必须在
//!    **装载期按 grouped 序聚合**(每 rank 拿 grouped 序的连续窗口)。
//!
//! 所以分片几何**住在源里**(源知道块型、v-head 公式、行置换),由
//! [`GgufSource`](super::source) 自己承担([`WeightSource::owns_shard`]
//! (crate::module::WeightSource::owns_shard)),通用 pass 对该源直通。
//!
//! # 一条贯穿全部角色的不变式(★ 测试的锚)
//!
//! > **每一 rank 装载后的张量 == 全量装载(已逆变换 = grouped 序)张量的
//! > 连续块**:行 = grouped 行窗口,列 = grouped 原子窗口。
//!
//! 分段件(`attn_qkv` / `ssm_conv1d`)是"每段各取一个窗口再拼接"——段内仍是
//! grouped 序的连续块。`attn_q` 不是分段件：它是 per-head `[q_h, gate_h]` 交错，
//! 只能按整行窗口切（2026-10-24 定谳；源布局与 HF `q_proj` 同序）。
//!
//! **2026-11-03 补注(列序不再是 grouped 序的一支)**：`ssm_out` 在
//! `atom > hv`(K/I 超块 256 vs 头宽 128)时,一个原子 = 文件里两个连续头 ⇒
//! "逐头取整原子"不存在,只能按**文件头升序的整原子段**切。此时本 rank 的
//! **列序是原子序**,不是 grouped 连续窗口 ⇒ 激活侧需要一张 per-rank 置换表
//! ([`transform::vhead_q8_blocks_rank`](super::transform::vhead_q8_blocks_rank))。
//! 不变式因此扩写为:
//!
//! > 每一 rank 的列序 == grouped 头集上按
//! > [`rank_head_order`](super::transform::rank_head_order) 指定的头序窗口;
//! > `atom | hv` 时该头序 = grouped 连续序(退化为旧不变式),`atom > hv` 时 = 原子序。
//!
//! 单卡仍是一张全量 tiled→grouped 表;TP 的 `atom | hv` 档**不需要**表
//! (源事实 `tp_rank_shard` + 逐层原子宽判据),而 `atom > hv` 档**必须**要(per-rank)。
//!
//! # 角色表(GGUF 名 → 切法;**唯一点**)
//!
//! | GGUF 键 | 角色 | 切法 |
//! |---|---|---|
//! | `ffn_gate` / `ffn_up` | `Out` | 行窗口(外维 = inter) |
//! | `attn_k` / `attn_v` | `Out` | 行窗口(外维 = kv 头) |
//! | `attn_q`(= per-head `[q_h, gate_h]`) | `Out` | 行窗口(每 rank 完整头对) |
//! | `attn_qkv`(= q ⊕ k ⊕ v) | `Segs[QKV]` | 三段各取行窗口(V 段 grouped) |
//! | `ssm_conv1d`(通道 = q ⊕ k ⊕ v) | `Segs[QKV]` | 同上(原生 F32,行置换在源侧) |
//! | `attn_gate`(= in_proj_z) | `OutHead` | v-head grouped 行窗口 |
//! | `ssm_alpha` / `ssm_beta` / `ssm_a` / `ssm_dt` | `OutHead` | 同上(head_dim = 1) |
//! | `ffn_down` / `attn_output` | `In` | 每行的块对齐列窗口 |
//! | `ssm_out` | `InHead` | 每行的 **v-head 列窗**:`atom \| hv` 逐头一段;`atom > hv` 整原子段(文件头升序) |
//! | `output`(仅 `ShardSel.head_split`) | `Out` | 行窗口(lm_head 半表) |
//! | 其余(norm / embed / q_norm / k_norm / ssm_norm) | `Replicate` | 全量 |
//!
//! # 三条响亮拒(不静默出错值)
//!
//! 1. 切分维不被 `world` 整除;
//! 2. 量化件的列切点不落块边界(`inner / world % be != 0`,或 `hv % be != 0`);
//! 3. 混合形态(1-D 却要切行 / 2-D 却要切元素)。

use crate::contract::ModelError;
use owl_kernels::family::gguf::GgmlTy;

use super::transform::{classify, GdnPart, GgufDims, KeyClass};

/// 分片选择(`None` = 单卡全量)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardSel {
    pub id: usize,
    pub world: usize,
    /// 是否切 lm_head 词表(GGUF `output.weight` 按行窗口;`false` = 整表复制)
    pub head_split: bool,
}

impl ShardSel {
    /// 切分维在本 rank 的窗口(起点, 长度);不整除 = **响亮拒**
    fn window(&self, total: usize, what: &str, key: &str) -> Result<(usize, usize), ModelError> {
        if self.world == 0 {
            return Err(ModelError::Msg(format!("[{key}] world = 0 非法")));
        }
        if !total.is_multiple_of(self.world) {
            return Err(ModelError::Msg(format!(
                "[{key}] {what} = {total} 不被 world = {} 整除(GGUF 等分切分要求整除;拒绝 pad)",
                self.world
            )));
        }
        let n = total / self.world;
        Ok((self.id * n, n))
    }
}

/// 目标侧的**列计划**(单位 = 原子 = 一个量化块 / 一个原生元素)。
///
/// 目标坐标 = 本 rank 的**列序**(`atom | hv` 时是 grouped 连续序;`atom > hv`
/// 时是文件头升序的原子序 —— 见表;[`transform::rank_head_order`]
/// (super::transform::rank_head_order) 与这里的 `Segs` **同源**),`src_*` 是
/// **文件**序的原子偏移。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cols {
    /// 整行(不切列)
    Full,
    /// 连续窗口(`ffn_down` / `attn_output`)
    Window { src0: usize, n: usize },
    /// 逐段拼接(`ssm_out`:每段 = 一个目标**头段/原子段**,文件里同段连续)
    Segs(Vec<(usize, usize)>),
}

/// 目标侧**行窗口表**(grouped 序;按目标拼接顺序,每段一窗)
pub type RowWins = Vec<(usize, usize)>;

/// 键的切分角色(纯名字 + 维度事实)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    /// 不切(每 rank 全量)
    Replicate,
    /// 外维等分(行窗口;无 v-head 置换)
    Out,
    /// 外维**分段**等分(段长 = `lens`;`tail_perm` = 末段(V)按 v-head 分组)
    Segs { lens: Vec<usize>, tail_perm: bool },
    /// 外维按 v-head 分组等分(`attn_gate` / `ssm_a*` / `ssm_dt`)
    OutHead,
    /// 内维等分(块对齐列窗口)
    In,
    /// 内维按 v-head 分组等分(`ssm_out`)
    InHead,
}

/// 键 → 角色(**唯一映射点**;未知层内键 = `Replicate`,由首 token 门兜底)。
/// 段长来自维度事实 `d`;`attn_q` 的 per-head 交错布局不需要张量几何输入。
pub fn role_of(key: &str, d: &GgufDims, head_split: bool) -> Result<Role, ModelError> {
    let (k, v) = (d.nk * d.hk, d.nv * d.hv);
    Ok(match classify(key) {
        KeyClass::Gdn(GdnPart::Qkv) => Role::Segs { lens: vec![k, k, v], tail_perm: true },
        KeyClass::Gdn(GdnPart::Z) => Role::OutHead,
        KeyClass::Gdn(GdnPart::Ab) | KeyClass::Gdn(GdnPart::Dt) | KeyClass::Gdn(GdnPart::A) => {
            Role::OutHead
        }
        KeyClass::Gdn(GdnPart::Conv) => Role::Segs { lens: vec![k, k, v], tail_perm: true },
        KeyClass::Gdn(GdnPart::Out) => Role::InHead,
        KeyClass::Gdn(GdnPart::Norm) => Role::Replicate,
        // 词表切档:GGUF 的 `output.weight` 是普通 2-D 行表(untied head),
        // 与 v-head 无关 ⇒ 走 `Out` 的行窗口;未开切时仍整表复制。
        KeyClass::Head if head_split => Role::Out,
        KeyClass::Embed | KeyClass::Head | KeyClass::Norm | KeyClass::Unknown => Role::Replicate,
        KeyClass::Plain => {
            // 层内局部名(`blk.N.` 之后、`.weight`/`.bias` 之前)
            let local = key
                .strip_prefix("blk.")
                .and_then(|r| r.split_once('.').map(|(_, l)| l))
                .unwrap_or(key)
                .rsplit_once('.')
                .map(|(l, _)| l)
                .unwrap_or(key);
            match local {
                // full attention:q 是 per-head `[q_h, gate_h]` 交错，
                // 不能按 q ⊕ gate 两个全局段切；整行窗口天然落在完整头对上。
                "attn_q" | "attn_k" | "attn_v" | "ffn_gate" | "ffn_up" => Role::Out,
                "attn_output" | "ffn_down" => Role::In,
                // 其余 plain 键 = 各 norm(attn_norm / post_attention_norm / q_norm / k_norm)
                _ => Role::Replicate,
            }
        }
    })
}

/// 目标张量的**单位几何**
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Shape {
    /// 2-D:`rows` 行 × `row_atoms` 原子(行 = grouped 行窗口表)
    Mat { wins: RowWins, rows: usize },
    /// 1-D 原生:grouped 序的元素窗口
    Elem { first: usize, len: usize },
}

/// 一个键的分片几何(**单卡的逆变换之后**的坐标;见模块头的不变式)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyGeom {
    pub shape: Shape,
    pub cols: Cols,
    /// 目标行宽(原子;量化 = 块数,原生 = 元素数)
    pub row_atoms: usize,
    /// 每原子字节(量化 = 块字节;原生 = 元素字节)
    pub atom_bytes: usize,
    /// 每原子元素数(量化 = `be`;原生 = 1)
    pub elem_per_atom: usize,
    /// 目标列的 **grouped 元素起点**(原生路径切列用;原生件无块 ⇒ 连续窗口)
    pub col_first: usize,
    /// 量化块键(交付单位 = **字节**;原生键交付单位 = 元素)
    pub quantized: bool,
}

impl KeyGeom {
    /// 目标行数(2-D)或元素数(1-D)
    pub fn rows(&self) -> usize {
        match &self.shape {
            Shape::Mat { rows, .. } => *rows,
            Shape::Elem { .. } => 1,
        }
    }

    /// **交付长度**(量化键 = 字节;原生键 = 元素;与 `WeightSource::elem_len` 同口径)
    /// 是否为空(与 `len` 成对;clippy 的长度接口约定)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        match &self.shape {
            Shape::Mat { rows, .. } => {
                rows * self.row_atoms * if self.quantized { self.atom_bytes } else { 1 }
            }
            Shape::Elem { len, .. } => *len,
        }
    }

    /// 目标行宽(元素;原生路径的列窗口用它)
    pub fn row_elems(&self) -> usize {
        self.row_atoms * self.elem_per_atom
    }
}

/// 张量事实(几何推导的输入)
#[derive(Debug, Clone, Copy)]
pub struct TensorFacts {
    pub ty: GgmlTy,
    /// 外维行数(`ne[1..]` 之积;1-D 时 = 1)
    pub rows: usize,
    /// 最内维(`ne[0]`,块轴)
    pub inner: usize,
}

/// **推导一个键的分片几何**(纯函数;`sel = None` ⇒ 不切,返回 `None`)
///
/// `dims` 给 v-head 事实,`f` 给张量几何。全部失败路径 = `Err`(响亮)。
pub fn geom_of(
    key: &str,
    d: &GgufDims,
    f: &TensorFacts,
    sel: Option<ShardSel>,
) -> Result<Option<KeyGeom>, ModelError> {
    let Some(sel) = sel else { return Ok(None) };
    let role = role_of(key, d, sel.head_split)?;
    if role == Role::Replicate {
        return Ok(None);
    }
    let (be, bb) = f.ty.block();
    let native = f.ty.native_dtype().is_some();
    // 原生件:原子 = 元素(`block()` 对原生型 = (1, 元素字节));量化件:原子 = 块
    let (atom_bytes, elem_per_atom) = if native { (bb, 1) } else { (bb, be) };
    let one_d = f.rows <= 1;
    let quantized = !native;

    let geom = match (role.clone(), one_d) {
        // ── 1-D 原生(A_log / dt_bias):grouped 元素窗口 ────────────────
        (Role::OutHead, true) => {
            if !native {
                return Err(ModelError::Msg(format!(
                    "[{key}] 1-D 量化件不在 M1 支持集(ssm_a/ssm_dt 恒 F32)"
                )));
            }
            let (first, len) = sel.window(f.inner, "v-head 数", key)?;
            KeyGeom {
                shape: Shape::Elem { first, len },
                cols: Cols::Full,
                col_first: 0,
                row_atoms: len,
                atom_bytes,
                elem_per_atom,
                quantized,
            }
        }
        (Role::Replicate, _) => unreachable!("上面已提前返回"),
        // ── 2-D 行窗口类(Out / OutHead)──────────────────────────────
        (Role::Out | Role::OutHead, false) => {
            if !f.inner.is_multiple_of(elem_per_atom) {
                return Err(ModelError::Msg(format!(
                    "[{key}] 内维 {} 不是原子 {elem_per_atom} 的整数倍",
                    f.inner
                )));
            }
            let (g0, gn) = sel.window(f.rows, "外维(行)", key)?;
            KeyGeom {
                shape: Shape::Mat { wins: vec![(g0, gn)], rows: gn },
                cols: Cols::Full,
                col_first: 0,
                row_atoms: f.inner / elem_per_atom,
                atom_bytes,
                elem_per_atom,
                quantized,
            }
        }
        (Role::Segs { lens, tail_perm }, false) => {
            let total: usize = lens.iter().sum();
            if total != f.rows {
                return Err(ModelError::Msg(format!(
                    "[{key}] 分段和 {total} ≠ 张量行数 {rows}(维度事实与文件不符)",
                    rows = f.rows
                )));
            }
            let mut wins = Vec::with_capacity(lens.len());
            let mut acc = 0usize;
            for &len in lens.iter() {
                let (o, n) = sel.window(len, "段长", key)?;
                // 窗口 = **grouped 序**坐标;末段(V)在文件里是 tiled 序 ⇒ 由源侧
                // 的行置换表(`row_map`)把它翻成文件行,前段恒等(`tail_perm`
                // 只是把这条事实写进角色表,算术由源的置换表统一承担)
                let _ = tail_perm;
                wins.push((acc + o, n));
                acc += len;
            }
            let rows: usize = wins.iter().map(|(_, n)| n).sum();
            KeyGeom {
                shape: Shape::Mat { wins, rows },
                cols: Cols::Full,
                col_first: 0,
                row_atoms: f.inner / elem_per_atom,
                atom_bytes,
                elem_per_atom,
                quantized,
            }
        }
        // ── 2-D 列窗口类(In / InHead)─────────────────────────────────
        (Role::In, false) => {
            let (c0, cn) = sel.window(f.inner, "内维(列)", key)?;
            if !native && (c0 % be != 0 || cn % be != 0) {
                return Err(ModelError::Msg(format!(
                    "[{key}] 列切不落块边界:内维 {} / world {} = {cn} 元素,块宽 {be} \
                     (量化块被撕开 ⇒ TP 拒;设计 §14.6 ②)",
                    f.inner, sel.world
                )));
            }
            KeyGeom {
                shape: Shape::Mat { wins: vec![(0, f.rows)], rows: f.rows },
                cols: Cols::Window { src0: c0 / elem_per_atom, n: cn / elem_per_atom },
                col_first: c0,
                row_atoms: cn / elem_per_atom,
                atom_bytes,
                elem_per_atom,
                quantized,
            }
        }
        (Role::InHead, false) => {
            // ssm_out:内维 = v-head(hd = hv)。rank 取 grouped 序的连续头集。
            //
            // 列几何有两种形态(2026-11-03;见 `transform::rank_head_order`):
            // ① **原子不跨头**(`hv % atom == 0`,如 32 块量子 Q8_0 / 原生):
            //    文件里每头一段(hv 元素 ⇒ hv/atom 个原子)⇒ 逐头一段;
            // ② **原子跨头**(`atom = hpa·hv`,K/I 超块 256 vs hv 128):
            //    一个原子 = 文件里两个**连续**头 ⇒ 逐头切必然为 0。改按
            //    "文件头升序的整原子段"切(rank 的 grouped 头窗在文件里本就是
            //    3 段对齐窗口)。代价:dest 列序不再是 grouped 序 ⇒ 激活侧要
            //    一张 per-rank 置换表(`transform::vhead_q8_blocks_rank`)。
            let hv = d.hv;
            if hv == 0 || f.inner != d.nv * hv {
                return Err(ModelError::Msg(format!(
                    "[{key}] 内维 {} ≠ nv·hv = {}·{hv}(列 = v-head 轴的判据不成立)",
                    f.inner, d.nv
                )));
            }
            if d.nk == 0 || !d.nv.is_multiple_of(d.nk) {
                return Err(ModelError::Msg(format!(
                    "[{key}] v-head 事实不自洽:nv = {} 不被 nk = {} 整除(无法推出 vpr)",
                    d.nv, d.nk
                )));
            }
            let (h0, hn) = sel.window(d.nv, "v-head 数", key)?;
            let vpr = d.nv / d.nk;
            let file_head = |h: usize| (h % vpr) * d.nk + h / vpr; // 文件头(`vhead_deperm` 同式)
            let (segs, row_atoms) = if hv.is_multiple_of(elem_per_atom) {
                // 旧路:每头整数个原子 ⇒ 逐头一段(不变)
                let bph = hv / elem_per_atom;
                let segs = (0..hn)
                    .map(|t| {
                        let g = file_head(h0 + t);
                        (g * bph, bph)
                    })
                    .collect::<Vec<_>>();
                (segs, hn * bph)
            } else {
                // 新路:原子跨头 ⇒ 按文件头升序切**整原子段**
                if !elem_per_atom.is_multiple_of(hv) {
                    return Err(ModelError::Msg(format!(
                        "[{key}] v-head 宽 {hv} 与量化原子 {elem_per_atom} 互不整除 ⇒ 列序无定义\
                         (该型按头不可分 ⇒ TP 拒;设计 §14.6 ②)"
                    )));
                }
                let hpa = elem_per_atom / hv; // 一原子跨 hpa 个文件头
                let mut gs: Vec<usize> = (h0..h0 + hn).map(file_head).collect();
                gs.sort_unstable();
                let mut segs = Vec::with_capacity(hn / hpa.max(1));
                let mut i = 0usize;
                while i < gs.len() {
                    let mut j = i + 1;
                    while j < gs.len() && gs[j] == gs[j - 1] + 1 {
                        j += 1;
                    }
                    let run = j - i;
                    if !gs[i].is_multiple_of(hpa) || !run.is_multiple_of(hpa) {
                        return Err(ModelError::Msg(format!(
                            "[{key}] rank{} 的文件头窗 [{}, {}) 不落整原子(原子 = {elem_per_atom} \
                             = {hpa} 头)⇒ world = {} 下窗口跨原子 ⇒ TP 拒",
                            sel.id,
                            gs[i],
                            gs[j - 1] + 1,
                            sel.world
                        )));
                    }
                    segs.push((gs[i] / hpa, run / hpa));
                    i = j;
                }
                (segs, hn * hv / elem_per_atom)
            };
            KeyGeom {
                shape: Shape::Mat { wins: vec![(0, f.rows)], rows: f.rows },
                cols: Cols::Segs(segs),
                col_first: h0 * hv,
                row_atoms,
                atom_bytes,
                elem_per_atom,
                quantized,
            }
        }
        // ── 形态不自洽 ───────────────────────────────────────────────
        (Role::Out | Role::Segs { .. } | Role::In | Role::InHead, true) => {
            return Err(ModelError::Msg(format!(
                "[{key}] 角色 {role:?} 要求 2-D 张量,实得 1-D(ne 只有一维)"
            )))
        }
    };
    Ok(Some(geom))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::gguf::reader::Gguf;
    use owl_shared::testkit::{resource, resource_ok};
    use std::path::PathBuf;

    fn open(key: &str) -> Option<Gguf> {
        resource_ok(key).then(|| Gguf::open(&PathBuf::from(resource(key).unwrap())).expect("open"))
    }

    fn facts(g: &Gguf) -> GgufDims {
        let (spec, _r) = crate::formats::gguf::spec_from_gguf(g).expect("spec");
        let (nk, hk, nv, hv) = spec.gdn_heads;
        let (nq, nkv, _hd) = spec.full_heads;
        GgufDims { nk, nv, hk, hv, nq, nkv }
    }

    fn tf(t: &crate::formats::gguf::TensorInfo) -> TensorFacts {
        TensorFacts { ty: t.ty, rows: t.rows() as usize, inner: t.inner() as usize }
    }

    /// **27B 全键角色/几何表**(手算钉值;零设备)
    #[test]
    fn key_geometry_27b_is_pinned() {
        let Some(g) = open("gguf:27b") else {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:27b\"");
            return;
        };
        let d = facts(&g);
        assert_eq!((d.nk, d.hk, d.nv, d.hv), (16, 128, 48, 128), "27B v-head 事实");
        let sel = ShardSel { id: 0, world: 2, head_split: false };
        let sel1 = ShardSel { id: 1, world: 2, head_split: false };
        let case = |k: &str, want: KeyGeom| {
            let t = g.get_tensor(k).unwrap_or_else(|| panic!("缺键 {k}"));
            let got = geom_of(k, &d, &tf(t), Some(sel)).unwrap().unwrap_or_else(|| panic!("{k} 未切"));
            assert_eq!(got, want, "[{k}] rank0 几何");
        };
        // ── ssm_out(Q8_0,列 = v-head;每头 4 块 = 136 B)──────────────
        // grouped 头 0..23 → 文件头 g = (h%3)*16 + h/3 ⇒ 0,16,32, 1,17,33, …
        case(
            "blk.0.ssm_out.weight",
            KeyGeom {
                shape: Shape::Mat { wins: vec![(0, 5120)], rows: 5120 },
                cols: Cols::Segs(
                    (0..24)
                        .map(|h| {
                            let g = (h % 3) * 16 + h / 3;
                            (g * 4, 4)
                        })
                        .collect(),
                ),
                col_first: 0,
                row_atoms: 96,    // 24 头 × 4 块
                atom_bytes: 34,   // Q8_0 块字节
                elem_per_atom: 32,
                quantized: true,
            },
        );
        // rank1 的头 = grouped 24..47 ⇒ 文件头从 g=8 起
        let t = g.get_tensor("blk.0.ssm_out.weight").unwrap();
        let g1 = geom_of("blk.0.ssm_out.weight", &d, &tf(t), Some(sel1)).unwrap().unwrap();
        match &g1.cols {
            Cols::Segs(s) => {
                assert_eq!(s.len(), 24);
                // h = 24 ⇒ g = (24 % 3) * 16 + 24 / 3 = 8 ⇒ 文件块 8×4
                assert_eq!(s[0], (8 * 4, 4), "rank1 首头文件块");
            }
            other => panic!("ssm_out rank1 应是 Segs:{other:?}"),
        }
        // ── ffn_down(Q4_K,内维 17408 = 68 块 ⇒ 各 34 块)─────────────
        case(
            "blk.0.ffn_down.weight",
            KeyGeom {
                shape: Shape::Mat { wins: vec![(0, 5120)], rows: 5120 },
                cols: Cols::Window { src0: 0, n: 34 },
                col_first: 0,
                row_atoms: 34,
                atom_bytes: 144,
                elem_per_atom: 256,
                quantized: true,
            },
        );
        // ── attn_output(Q6_K,内维 6144 = 24 块 ⇒ 各 12 块)──────────
        case(
            "blk.3.attn_output.weight",
            KeyGeom {
                shape: Shape::Mat { wins: vec![(0, 5120)], rows: 5120 },
                cols: Cols::Window { src0: 0, n: 12 },
                col_first: 0,
                row_atoms: 12,
                atom_bytes: 210,
                elem_per_atom: 256,
                quantized: true,
            },
        );
        // ── attn_qkv(Q8_0 三段 2048|2048|6144 ⇒ 各半)──────────────
        case(
            "blk.0.attn_qkv.weight",
            KeyGeom {
                shape: Shape::Mat { wins: vec![(0, 1024), (2048, 1024), (4096, 3072)], rows: 5120 },
                cols: Cols::Full,
                col_first: 0,
                row_atoms: 160, // 5120/32
                atom_bytes: 34,
                elem_per_atom: 32,
                quantized: true,
            },
        );
        // ── attn_q(full 层:per-head [q_h, gate_h],整行窗口 6144 行)──
        case(
            "blk.3.attn_q.weight",
            KeyGeom {
                shape: Shape::Mat { wins: vec![(0, 6144)], rows: 6144 },
                cols: Cols::Full,
                col_first: 0,
                row_atoms: 160,
                atom_bytes: 34,
                elem_per_atom: 32,
                quantized: true,
            },
        );
        // rank1 必须从 head 12 开始（文件 row 6144），而不是 q/gate 两段各半
        let t = g.get_tensor("blk.3.attn_q.weight").unwrap();
        let q1 = geom_of("blk.3.attn_q.weight", &d, &tf(t), Some(sel1)).unwrap().unwrap();
        assert_eq!(
            q1.shape,
            Shape::Mat { wins: vec![(6144, 6144)], rows: 6144 },
            "attn_q rank1 行窗口"
        );
        // ── ffn_gate(外维 17408 ⇒ 各 8704 行)───────────────────────
        case(
            "blk.0.ffn_gate.weight",
            KeyGeom {
                shape: Shape::Mat { wins: vec![(0, 8704)], rows: 8704 },
                cols: Cols::Full,
                col_first: 0,
                row_atoms: 20, // 内维 = hidden 5120 / 256
                atom_bytes: 144,
                elem_per_atom: 256,
                quantized: true,
            },
        );
        // ── attn_gate(in_proj_z,行 = v-head;各 24 头 = 3072 行)─────
        case(
            "blk.0.attn_gate.weight",
            KeyGeom {
                shape: Shape::Mat { wins: vec![(0, 3072)], rows: 3072 },
                cols: Cols::Full,
                col_first: 0,
                row_atoms: 160,
                atom_bytes: 34,
                elem_per_atom: 32,
                quantized: true,
            },
        );
        // ── ssm_alpha / ssm_a(行 = v-head 24;hd = 1)────────────────
        case(
            "blk.0.ssm_alpha.weight",
            KeyGeom {
                shape: Shape::Mat { wins: vec![(0, 24)], rows: 24 },
                cols: Cols::Full,
                col_first: 0,
                row_atoms: 160,
                atom_bytes: 34,
                elem_per_atom: 32,
                quantized: true,
            },
        );
        // 1-D 原生 F32:元素窗口(24 个)
        let t = g.get_tensor("blk.0.ssm_a").unwrap();
        let ga = geom_of("blk.0.ssm_a", &d, &tf(t), Some(sel)).unwrap().unwrap();
        assert_eq!(
            ga,
            KeyGeom {
                shape: Shape::Elem { first: 0, len: 24 },
                cols: Cols::Full,
                col_first: 0,
                row_atoms: 24,
                atom_bytes: 4,
                elem_per_atom: 1,
                quantized: false,
            }
        );
        // ── Replicate:norm / embed / output(未开词表切)────────────
        for k in ["blk.0.attn_norm.weight", "blk.0.ssm_norm.weight", "token_embd.weight", "output.weight"] {
            let t = g.get_tensor(k).unwrap();
            assert!(
                geom_of(k, &d, &tf(t), Some(sel)).unwrap().is_none(),
                "[{k}] 应不切(Replicate)"
            );
        }
    }

    /// **GGUF 词表切几何**(2026-10-28):`output.weight` 是普通 2-D 行表,
    /// 开 `head_split` 后每 rank 取连续半行;未开切仍是整表复制。
    #[test]
    fn head_split_geometry_27b_is_half_rows() {
        let Some(g) = open("gguf:27b") else {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:27b\"");
            return;
        };
        let d = facts(&g);
        let t = g.get_tensor("output.weight").expect("output.weight");
        let f = tf(t);
        let g0 = geom_of(
            "output.weight",
            &d,
            &f,
            Some(ShardSel { id: 0, world: 2, head_split: true }),
        )
        .expect("geom")
        .expect("head_split 应切");
        let g1 = geom_of(
            "output.weight",
            &d,
            &f,
            Some(ShardSel { id: 1, world: 2, head_split: true }),
        )
        .expect("geom")
        .expect("head_split 应切");
        assert_eq!(g0.shape, Shape::Mat { wins: vec![(0, 124160)], rows: 124160 });
        assert_eq!(g1.shape, Shape::Mat { wins: vec![(124160, 124160)], rows: 124160 });
        assert_eq!(g0.cols, Cols::Full);
        assert_eq!(g0.row_atoms, 20, "Q6_K 内维 5120 / 256");
        assert_eq!(g0.atom_bytes, 210, "Q6_K 块字节");
        assert_eq!(g0.len() + g1.len(), f.rows * (f.inner / 256) * 210);
        assert!(
            geom_of(
                "output.weight",
                &d,
                &f,
                Some(ShardSel { id: 0, world: 2, head_split: false }),
            )
            .expect("geom")
            .is_none(),
            "未开词表切时 output.weight 应整表复制"
        );
    }

    /// **几何自洽门**:对 27B 每个键,
    /// ① 两 rank 的交付长度之和 == 全量张量的交付长度(不重不漏的**量**);
    /// ② 切行类:两 rank 行数之和 == 全量行数(列恒 Full);
    /// ③ 切列类:两侧列原子之和 == 全量原子数(行恒全量)。
    #[test]
    fn geometry_tiles_full_tensor_27b() {
        let Some(g) = open("gguf:27b") else {
            owl_shared::testkit::report::skip("资源清单缺 \"gguf:27b\"");
            return;
        };
        let d = facts(&g);
        let mut n_split = 0usize;
        for t in &g.tensors {
            let key = &t.name;
            let f = tf(t);
            let (be, _bb) = t.ty.block();
            let native = t.ty.native_dtype().is_some();
            let geom = |id: usize| {
                geom_of(key, &d, &f, Some(ShardSel { id, world: 2, head_split: false }))
                    .unwrap_or_else(|e| panic!("[{key}] rank{id}: {e}"))
            };
            let (Some(g0), Some(g1)) = (geom(0), geom(1)) else {
                continue; // Replicate
            };
            n_split += 1;
            assert_eq!(g0.row_atoms, g1.row_atoms, "[{key}] 两侧行宽应相等");
            let full_len = if native { f.rows * f.inner } else { f.rows * (f.inner / be) * _bb };
            assert_eq!(g0.len() + g1.len(), full_len, "[{key}] 两 rank 交付量之和 ≠ 全量");
            // 1-D 原生:元素窗口(行/列概念不适用)
            if matches!(g0.shape, Shape::Elem { .. }) {
                assert_eq!(g0.len() + g1.len(), f.inner, "[{key}] 1-D 元素窗口不覆盖全量");
                continue;
            }
            let rows_of = |s: &Shape| match s {
                Shape::Mat { rows, .. } => *rows,
                Shape::Elem { len, .. } => *len,
            };
            let atoms_of = |c: &Cols| match c {
                Cols::Full => f.inner / g0.elem_per_atom,
                Cols::Window { n, .. } => *n,
                Cols::Segs(s) => s.iter().map(|(_, n)| *n).sum(),
            };
            if rows_of(&g0.shape) + rows_of(&g1.shape) == f.rows {
                // 切行:列必须是全量
                for (id, gg) in [&g0, &g1].into_iter().enumerate() {
                    assert_eq!(gg.cols, Cols::Full, "[{key}] rank{id} 切行类应列全量");
                }
                assert_eq!(g0.row_atoms * g0.elem_per_atom, f.inner, "[{key}] 行宽 ≠ 全量内维");
            } else {
                // 切列:行必须是全量,列原子两侧之和 == 全量
                assert_eq!(rows_of(&g0.shape), f.rows, "[{key}] 切列类应行全量");
                let full_atoms = f.inner / g0.elem_per_atom;
                assert_eq!(
                    atoms_of(&g0.cols) + atoms_of(&g1.cols),
                    full_atoms,
                    "[{key}] 列原子不覆盖全量"
                );
            }
        }
        eprintln!("[gguf-tp] 27B 几何自洽:{n_split} 个键被切(其余 Replicate)");
        // 27B:48 GDN × (qkv/gate/out/alpha/beta/conv/a/dt) + 16 full × 4 + 64 × 3 = 640
        assert_eq!(n_split, 640, "被切的键数不符 —— 角色表漏臂/多臂?");
    }

    /// 不整除 / 块不对齐 ⇒ **响亮拒**(不许静默 pad)
    #[test]
    fn misalignment_is_loud() {
        let d = GgufDims { nk: 16, nv: 48, hk: 128, hv: 128, nq: 24, nkv: 4 };
        let f = TensorFacts { ty: GgmlTy::Q4K, rows: 100, inner: 5120 };
        let e = geom_of("blk.0.ffn_gate.weight", &d, &f, Some(ShardSel { id: 0, world: 8, head_split: false }))
            .unwrap_err()
            .to_string();
        assert!(e.contains("不被 world"), "{e}");
        // 列切不落块边界:inner 5120 / world 3(量化块 256)⇒ 响亮拒
        let e = geom_of("blk.0.ffn_down.weight", &d, &f, Some(ShardSel { id: 0, world: 3, head_split: false }))
            .unwrap_err()
            .to_string();
        assert!(e.contains("不落块边界") || e.contains("不被 world"), "{e}");
        // ssm_out 的 hv 与原子互不整除 ⇒ 按头不可分 ⇒ TP 拒
        let d2 = GgufDims { nk: 16, nv: 48, hk: 128, hv: 100, nq: 24, nkv: 4 };
        let f2 = TensorFacts { ty: GgmlTy::Q8_0, rows: 5120, inner: 4800 };
        let e = geom_of("blk.0.ssm_out.weight", &d2, &f2, Some(ShardSel { id: 0, world: 2, head_split: false }))
            .unwrap_err()
            .to_string();
        assert!(e.contains("互不整除"), "{e}");
    }

    /// **原子跨头条的几何 × 表 互证**(2026-11-03;零设备)。
    ///
    /// 27B 的 K/I 量子(`ssm_out` = Q4_K,atom 256 > hv 128):
    /// ① `Cols::Segs` 必须是三个整原子段(rank0 `[0,4),(8,4),(16,4)`);
    /// ② 按 Segs 把 dest 块映回文件块 ⇒ 该 rank 的文件块集 = 12 个原子;
    /// ③ 表值 == 该文件块坐的 grouped 头的局部块号(**同一张头序的两面**);
    /// ④ 两 rank 的文件块并集 = 全量 24 原子(不重不漏 —— 拼回门)。
    #[test]
    fn ssm_out_atom_tiled_geometry_matches_rank_table() {
        use crate::formats::gguf::transform::{rank_head_order, vhead_q8_blocks_rank};
        let d = GgufDims { nk: 16, nv: 48, hk: 128, hv: 128, nq: 24, nkv: 4 };
        let f = TensorFacts { ty: GgmlTy::Q4K, rows: 5120, inner: 6144 };
        let (atom, atom_bytes) = GgmlTy::Q4K.block();
        assert_eq!((atom, atom_bytes), (256, 144), "Q4_K 原子事实");
        let bph = d.hv / 32; // 每头 q8_1 块数 = 4
        let abl = atom / 32; // 每原子 q8_1 块数 = 8
        let mut all_file_blocks: Vec<usize> = Vec::new();
        for rank in 0..2usize {
            let sel = ShardSel { id: rank, world: 2, head_split: false };
            let g = geom_of("blk.0.ssm_out.weight", &d, &f, Some(sel))
                .expect("geom")
                .expect("ssm_out 应切");
            let Cols::Segs(segs) = &g.cols else {
                panic!("ssm_out 应是 Segs,实得 {:?}", g.cols)
            };
            // ① 段 = 整原子(27B rank0/1 各 3 段×4 原子)
            let want: Vec<(usize, usize)> = if rank == 0 {
                vec![(0, 4), (8, 4), (16, 4)]
            } else {
                vec![(4, 4), (12, 4), (20, 4)]
            };
            assert_eq!(segs, &want, "rank{rank} 原子段");
            assert_eq!(g.row_atoms, 12, "rank{rank} 行宽 = 12 原子");
            assert_eq!(g.len(), f.rows * 12 * atom_bytes, "rank{rank} 交付量");
            // ② dest 块 → 文件块
            let dest_file: Vec<usize> = segs
                .iter()
                .flat_map(|&(src0, n)| (0..n * abl).map(move |b| src0 * abl + b))
                .collect();
            assert_eq!(dest_file.len(), 96, "rank{rank} 列块数");
            // ③ 表的每一块 = 该文件块坐的 grouped 头的局部块号
            let table = vhead_q8_blocks_rank(d.nk, d.nv, d.hv, atom, 2, rank).expect("table");
            let order = rank_head_order(d.nk, d.nv, d.hv, atom, 2, rank).expect("order");
            assert_eq!(table.len(), dest_file.len());
            let (h0, hn) = (rank * (d.nv / 2), d.nv / 2);
            assert_eq!(order.len(), hn);
            for (dc, fblk) in dest_file.iter().enumerate() {
                let fh = fblk / bph; // 文件头
                let h = (fh % d.nk) * (d.nv / d.nk) + fh / d.nk; // grouped 头
                assert_eq!(h, order[dc / bph], "rank{rank} dest 头 {} 的 grouped 头", dc / bph);
                assert_eq!(
                    table[dc],
                    ((h - h0) * bph + fblk % bph) as u32,
                    "rank{rank} dest 块 {dc}(文件块 {fblk})"
                );
            }
            all_file_blocks.extend(dest_file);
        }
        // ④ 两 rank 文件块并集 = 全量(拼回门:不重不漏)
        all_file_blocks.sort_unstable();
        assert_eq!(
            all_file_blocks,
            (0..d.nv * bph).collect::<Vec<_>>(),
            "两 rank 的文件块不重不漏"
        );
    }
}
