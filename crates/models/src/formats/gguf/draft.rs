//! **GGUF 草稿装载面**(起点族第五件;2026-10-20 DFlash2 线 P0/P1)。
//!
//! 契约与其它源同:**「键 → 字节」**。本模块只解决 DFlash2 草稿文件的三处
//! 特殊性,其余交给既有装载面(计划表 / 槽定形 / 上卡):
//!
//! 1. **命名**:草稿的 owl 本地键是 HF 形态(`layers.N.self_attn.q_proj.weight`),
//!    而 GGUF 叫 `blk.N.attn_q.weight` —— 译表见 [`translate`](GgufDraftSource::translate)
//!    (闭区间;未知键 = 响亮拒,禁兜底);
//! 2. **fc 五拆**:GVUF `fc.weight` 是 `[out=hidden, in=5·hidden]`,五个 tap 沿
//!    **内维**;量化块沿内维 ⇒ 列切跨 `out` 行,槽(扁平字节)表达不了列窗。源侧
//!    在**装载期**把 tap 列块重排成 `fc_i.weight`(纯 memcpy,零解量化;设备
//!    驻留总量 = 一份 fc);
//! 3. **码本反量化**:`selector_predecessor/successor` 是 63.6M×2 的 Q4_K 大表,
//!    而选择器核只看 f16/bf16 行 ⇒ 装载期按行反量化为 BF16(D6;v1 用 254 MB
//!    常驻换零核改动)。
//!
//! # 与主干源的分野(为什么不复用 [`GgufSource`](super::GgufSource))
//!
//! 主干源会跑 `spec_from_gguf`(qwen35)+ `transform::xf_for`(GDN 六类逆变换)
//! + v-head 分片 —— 草稿文件**没有 GDN**,且维度事实是 `dflash.*`。共用会引入
//! 两套规范(AGENTS:接口不唯一);故草稿源是**独立实现**,但只做"键 → 字节"
//! 这一件事,零设备知识。
//!
//! # 零解量化驻留(I-3)的**唯一例外**
//!
//! 量化权重(Q4_K/Q6_K)字节原样上卡;唯一例外是两本 selector 码本
//! (核面只认 f16/bf16 行)。例外**按名单封闭**(`codebook_allowed`):其它量化
//! 键请求元素面 = 响亮拒,不许把例外扩成通用路。

use std::collections::BTreeSet;
use std::sync::Mutex;

use crate::contract::{Dtype, ModelError};
use crate::module::{KeyConvention, QuantPlan, SourceFacts, Taken, WeightSource};
use owl_kernels::family::gguf::GgmlTy;

use crate::layers::dflash2::DFlash2Draft;

use super::dequant::{dequant_row, dequant_all};
use super::draft_meta::{DraftSpec, TargetFacts};
use super::reader::Gguf;

/// **1-based 归一 + 家族对账**(草稿 meta 的深门;此处只做命名层的键名约定)
///
/// 键约定(**只覆盖草稿声明面**;末 3 个方法草稿不用 —— 草稿共享 target 的
/// embed/lm_head/final norm,`Model::new` 那条路不走这里):
#[derive(Debug, Clone, Copy, Default)]
pub struct DraftGgufConvention;

impl DraftGgufConvention {
    /// 层内局部键 → `blk.N.<gguf 名>`。
    ///
    /// **只列计划表会问的局部键**(`DfAttn`/`Mlp` 的七个 Linear)。conv / norm /
    /// base_kernel 由固定 F16 计划构造,不经本约定;列进来反而会两套规范
    /// (conv 的 `base_kernel` 在 attn_conv 与 mlp_conv 里同名、GGUF 键不同)。
    fn layer_name(local: &str) -> Option<&'static str> {
        Some(match local {
            "q_proj" => "attn_q.weight",
            "k_proj" => "attn_k.weight",
            "v_proj" => "attn_v.weight",
            "o_proj" => "attn_output.weight",
            "gate_proj" => "ffn_gate.weight",
            "up_proj" => "ffn_up.weight",
            "down_proj" => "ffn_down.weight",
            // conv 投影(GGUF 里是量化块;HF 检查点里是原生 BF16 ⇒ 只有
            // Keyed 臂会问到这里,见 `PlanSource::or_default`)
            "attn_conv_proj" => "attn_conv_proj.weight",
            "mlp_conv_proj" => "ffn_conv_proj.weight",
            _ => return None,
        })
    }
}

impl KeyConvention for DraftGgufConvention {
    fn embed_key(&self, _local: &str) -> String {
        // 草稿不持有 embed(共享 target);真被问到 = 上游接错 ⇒ 返回的键
        // 在索引里不存在,计划表/源会以"缺键"响亮拒(不静默造值)
        "draft.shared_embed".to_string()
    }

    fn layer_key(&self, i: usize, local: &str) -> String {
        match DraftGgufConvention::layer_name(local) {
            Some(name) => format!("blk.{i}.{name}"),
            None => format!("blk.{i}.{local}"),
        }
    }

    fn norm_key(&self, _local: &str) -> String {
        "output_norm.weight".to_string()
    }
}

/// **草稿逐键计划表**(PlanTable;`plan_at` 吃 **GGUF 键**)
///
/// 与主干 [`super::GgufPlan`] 同律(缺键/形状不符/型未移植 ⇒ 记 unresolved,
/// 由调用方一次取走并响亮拒),差别只有"不吃 TP 角色换算"(v1 草稿单卡;
/// TP 档在装载入口就拒 —— 见 [`GgufDraftBoot::draft`])。
pub struct DraftGgufPlan<'g> {
    g: &'g Gguf,
    unresolved: Mutex<std::collections::BTreeMap<String, String>>,
}

impl<'g> DraftGgufPlan<'g> {
    pub fn new(g: &'g Gguf) -> Self {
        Self { g, unresolved: Mutex::new(Default::default()) }
    }

    fn note(&self, key: &str, why: &str) {
        let mut m = self.unresolved.lock().unwrap_or_else(|e| e.into_inner());
        m.entry(key.to_string()).or_insert_with(|| why.to_string());
    }
}

impl crate::module::PlanTable for DraftGgufPlan<'_> {
    fn plan_at(&self, key: &str, out_dim: usize, in_dim: usize) -> QuantPlan {
        let Some(t) = self.g.get_tensor(key) else {
            self.note(key, "GGUF 草稿索引里没有该键(译表漏臂 / 文件不含该张量)");
            return QuantPlan::F16;
        };
        if t.rows() as usize != out_dim || t.inner() as usize != in_dim {
            self.note(
                key,
                &format!(
                    "形状不符:索引 [rows {} × inner {}](ne {:?})vs 声明 [{out_dim} × {in_dim}]",
                    t.rows(),
                    t.inner(),
                    t.ne
                ),
            );
            return QuantPlan::F16;
        }
        if t.ty.native_dtype().is_some() {
            return QuantPlan::F16;
        }
        if owl_kernels::family::gguf::entry_for(t.ty).is_none() {
            self.note(
                key,
                &format!("型 {} 未移植(内核族无入口;见 family::gguf PORTED)", t.ty.name()),
            );
            return QuantPlan::F16;
        }
        QuantPlan::Gguf(t.ty)
    }

    fn take_unresolved(&self) -> Vec<String> {
        let mut m = self.unresolved.lock().unwrap_or_else(|e| e.into_inner());
        let out = m.iter().map(|(k, w)| format!("{k} ({w})")).collect();
        m.clear();
        out
    }
}

/// 译表命中结果
#[derive(Clone, Debug, PartialEq, Eq)]
enum Target {
    /// 普通 GGUF 键
    Key(String),
    /// fc 的第 i 个 tap(源侧重排;文件键恒 `fc.weight`)
    FcTap(usize),
}

/// **GGUF 草稿源**:`WeightSource` 实现(键 = owl 草稿的全键)
pub struct GgufDraftSource<'g> {
    g: &'g Gguf,
    spec: DraftSpec,
    /// 已问过的 **GGUF 键**(覆盖门:G0-a 81/81 双向)
    asked: Mutex<BTreeSet<String>>,
}

impl<'g> GgufDraftSource<'g> {
    /// 由已打开的容器建立(借用 —— 装载期不重开 mmap)
    pub fn new(g: &'g Gguf, spec: DraftSpec) -> Self {
        Self { g, spec, asked: Mutex::new(BTreeSet::new()) }
    }

    /// 本轮**未被请求**的文件键(装载后必须为空 —— G0-a 双向覆盖)
    pub fn unrequested(&self) -> Vec<String> {
        let asked = self.asked.lock().unwrap_or_else(|e| e.into_inner());
        self.g
            .tensors
            .iter()
            .filter(|t| !asked.contains(&t.name))
            .map(|t| t.name.clone())
            .collect()
    }

    /// (已请求的 distinct GGUF 键, 文件张量数)
    pub fn coverage(&self) -> (usize, usize) {
        let asked = self.asked.lock().unwrap_or_else(|e| e.into_inner());
        (asked.len(), self.g.tensors.len())
    }

    /// owl 草稿全键 → GGUF 键(闭表;未知 = 响亮拒)
    fn translate(key: &str) -> Result<Target, ModelError> {
        if let Some(rest) = key.strip_prefix("layers.") {
            let (i, tail) = rest.split_once('.').ok_or_else(|| {
                ModelError::Msg(format!("GGUF 草稿键 `{key}` 形态不符(缺层内路径)"))
            })?;
            let i: usize = i.parse().map_err(|_| {
                ModelError::Msg(format!("GGUF 草稿键 `{key}` 的层号不是整数"))
            })?;
            let name = match tail {
                "attention_conv.base_kernel" => "attn_conv_base",
                "attention_conv.kernel_projection.weight" => "attn_conv_proj.weight",
                "mlp_conv.base_kernel" => "ffn_conv_base",
                "mlp_conv.kernel_projection.weight" => "ffn_conv_proj.weight",
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                other => {
                    return Err(ModelError::Msg(format!(
                        "GGUF 草稿译表:未知层内键 `{other}`(键 `{key}`;禁止静默兜底)"
                    )))
                }
            };
            return Ok(Target::Key(format!("blk.{i}.{name}")));
        }
        if let Some(rest) = key.strip_prefix("fc_") {
            let i: usize = rest
                .strip_suffix(".weight")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    ModelError::Msg(format!("GGUF 草稿译表:fc 键 `{key}` 形态不符(应 fc_N.weight)"))
                })?;
            return Ok(Target::FcTap(i));
        }
        let gguf = match key {
            "hidden_norm.weight" => "enc.output_norm.weight",
            "norm.weight" => "output_norm.weight",
            "candidate_selector.hidden_projection.weight" => "selector_hidden.weight",
            "candidate_selector.predecessor_codebook" => "selector_predecessor.weight",
            "candidate_selector.successor_codebook" => "selector_successor.weight",
            other => {
                return Err(ModelError::Msg(format!(
                    "GGUF 草稿译表:未知顶层键 `{other}`(禁止静默兜底)"
                )))
            }
        };
        Ok(Target::Key(gguf.to_string()))
    }

    /// 译表 → (gguf 键, fc tap) + 记账
    fn resolve(&self, key: &str) -> Result<(String, Option<usize>), ModelError> {
        let t = Self::translate(key)?;
        let (gkey, tap) = match t {
            Target::Key(k) => (k, None),
            Target::FcTap(i) => ("fc.weight".to_string(), Some(i)),
        };
        self.asked
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(gkey.clone());
        Ok((gkey, tap))
    }

    /// 码本名单(唯一允许在 host 端反量化的量化键)
    fn codebook_allowed(gkey: &str) -> bool {
        matches!(gkey, "selector_predecessor.weight" | "selector_successor.weight")
    }

    /// fc 张量几何(开箱验一次)
    fn fc_geom(&self) -> Result<(GgmlTy, usize, usize, usize), ModelError> {
        let t = self.g.get_tensor("fc.weight").ok_or_else(|| {
            ModelError::Msg("GGUF 草稿缺 `fc.weight`(memory 投影;没有它草稿不成立)".into())
        })?;
        let (be, bb) = t.ty.block();
        let (rows, inner) = (t.rows() as usize, t.inner() as usize);
        let want_inner = self.spec.layers * self.spec.hidden;
        if inner != want_inner || rows != self.spec.hidden {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿 `fc.weight` 形状 [{rows} × {inner}] ≠ [{hidden} × {layers}·{hidden}={want_inner}]",
                hidden = self.spec.hidden,
                layers = self.spec.layers,
            )));
        }
        if inner % be != 0 || inner / self.spec.layers % be != 0 {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿 `fc.weight` 内维 {inner} / 块 {be} 不整除(tap 列块无法按块重排)"
            )));
        }
        let row_bytes = inner / be * bb;
        let tap_row_bytes = self.spec.hidden / be * bb;
        Ok((t.ty, rows, row_bytes, tap_row_bytes))
    }

    /// 按行窗口搬字节(量化键的 U8 通道;`offset/len` = 字节)
    fn copy_rows(
        &self,
        gkey: &str,
        src_row_bytes: usize,
        tap: Option<usize>,
        rows: usize,
        offset: usize,
        len: usize,
        dst: &mut [u8],
    ) -> Taken<()> {
        if dst.len() < len {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿源:键 `{gkey}` 目标缓冲 {}B < 需 {len}B",
                dst.len()
            )));
        }
        let ix = self.g.index_of(gkey).ok_or_else(|| {
            ModelError::Msg(format!("GGUF 草稿源:索引里没有 `{gkey}`"))
        })?;
        let src = self.g.bytes(ix);
        let dst_row = match tap {
            Some(_) => src_row_bytes / self.spec.layers,
            None => src_row_bytes,
        };
        if dst_row == 0 || rows == 0 {
            return Err(ModelError::Msg(format!("GGUF 草稿源:键 `{gkey}` 行几何为 0")));
        }
        if src.len() < rows * src_row_bytes {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿源:键 `{gkey}` 源字节 {} < 行 {rows} × {src_row_bytes}B",
                src.len()
            )));
        }
        if offset + len > rows * dst_row {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿源:键 `{gkey}` 请求 [{offset},{}) 越界(共 {}B)",
                offset + len,
                rows * dst_row
            )));
        }
        let seg = tap.map(|i| i * dst_row).unwrap_or(0);
        let mut done = 0usize;
        while done < len {
            let abs = offset + done;
            let r = abs / dst_row;
            let c = abs % dst_row;
            let n = (dst_row - c).min(len - done);
            let s = r * src_row_bytes + seg + c;
            dst[done..done + n].copy_from_slice(&src[s..s + n]);
            done += n;
        }
        Ok(Some(()))
    }

    /// 反量化窗口(码本专用):按行解 → 目标 dtype
    fn dequant_window(
        &self,
        gkey: &str,
        offset_elems: usize,
        len_elems: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Taken<()> {
        let t = self.g.get_tensor(gkey).ok_or_else(|| {
            ModelError::Msg(format!("GGUF 草稿源:索引里没有 `{gkey}`"))
        })?;
        let (rows, inner) = (t.rows() as usize, t.inner() as usize);
        let (be, bb) = t.ty.block();
        let row_bytes = inner / be * bb;
        let ix = self.g.index_of(gkey).ok_or_else(|| ModelError::Msg(format!("GGUF 草稿源:索引里没有 `{gkey}`")))?;
        let src = self.g.bytes(ix);
        let esz = dtype.size_bytes();
        if dst.len() < len_elems * esz {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿源:键 `{gkey}` 目标缓冲 {}B < 需 {}B",
                dst.len(),
                len_elems * esz
            )));
        }
        if offset_elems + len_elems > rows * inner {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿源:键 `{gkey}` 元素窗口 [{offset_elems},{}) 越界({} 元素)",
                offset_elems + len_elems,
                rows * inner
            )));
        }
        let mut row_buf = vec![0f32; inner];
        let mut done = 0usize;
        while done < len_elems {
            let abs = offset_elems + done;
            let r = abs / inner;
            let c = abs % inner;
            let n = (inner - c).min(len_elems - done);
            dequant_row(t.ty, &src[r * row_bytes..(r + 1) * row_bytes], &mut row_buf)
                .map_err(|e| ModelError::Msg(format!("GGUF 草稿源:键 `{gkey}` 第 {r} 行解量化: {e}")))?;
            let out = &mut dst[done * esz..(done + n) * esz];
            for (k, v) in row_buf[c..c + n].iter().enumerate() {
                match dtype {
                    Dtype::BF16 => out[k * 2..k * 2 + 2]
                        .copy_from_slice(&half::bf16::from_f32(*v).to_le_bytes()),
                    Dtype::F16 => out[k * 2..k * 2 + 2]
                        .copy_from_slice(&half::f16::from_f32(*v).to_le_bytes()),
                    Dtype::F32 => out[k * 4..k * 4 + 4].copy_from_slice(&v.to_le_bytes()),
                    other => {
                        return Err(ModelError::Msg(format!(
                            "GGUF 草稿源:键 `{gkey}` 反量化目标 dtype {other:?} 不支持(需 f32/f16/bf16)"
                        )))
                    }
                }
            }
            done += n;
        }
        Ok(Some(()))
    }

    /// 原生(F32)元素窗口 → 目标 dtype
    fn native_window(
        &self,
        gkey: &str,
        offset_elems: usize,
        len_elems: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Taken<()> {
        let t = self.g.get_tensor(gkey).ok_or_else(|| {
            ModelError::Msg(format!("GGUF 草稿源:索引里没有 `{gkey}`"))
        })?;
        let (rows, inner) = (t.rows() as usize, t.inner() as usize);
        let ix = self.g.index_of(gkey).ok_or_else(|| ModelError::Msg(format!("GGUF 草稿源:索引里没有 `{gkey}`")))?;
        let v = dequant_all(t.ty, self.g.bytes(ix), rows, inner)
            .map_err(|e| ModelError::Msg(format!("GGUF 草稿源:键 `{gkey}` 原生解包: {e}")))?;
        let Some(slice) = v.get(offset_elems..offset_elems + len_elems) else {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿源:键 `{gkey}` 元素窗口 [{offset_elems},{}) 越界({} 元素)",
                offset_elems + len_elems,
                v.len()
            )));
        };
        let esz = dtype.size_bytes();
        if dst.len() < len_elems * esz {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿源:键 `{gkey}` 目标缓冲 {}B < 需 {}B",
                dst.len(),
                len_elems * esz
            )));
        }
        for (k, f) in slice.iter().enumerate() {
            match dtype {
                Dtype::F32 => dst[k * 4..k * 4 + 4].copy_from_slice(&f.to_le_bytes()),
                Dtype::F16 => dst[k * 2..k * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(*f).to_le_bytes()),
                Dtype::BF16 => dst[k * 2..k * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(*f).to_le_bytes()),
                other => {
                    return Err(ModelError::Msg(format!(
                        "GGUF 草稿源:键 `{gkey}` 元素通道目标 dtype {other:?} 不支持(HF 档同律:原生件只出 f32/f16/bf16)"
                    )))
                }
            }
        }
        Ok(Some(()))
    }
}

impl WeightSource for GgufDraftSource<'_> {
    /// v1 草稿单卡(TP 分片挂 P4;`GgufDraftBoot::draft` 在入口就拒)
    fn owns_shard(&self) -> bool {
        false
    }

    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        let (gkey, tap) = self.resolve(key)?;
        if tap.is_some() {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿源:键 `{key}` 是 fc 的量化 tap 块(只走 U8 字节通道)"
            )));
        }
        let Some(t) = self.g.get_tensor(&gkey) else {
            return Ok(None);
        };
        if t.ty.native_dtype().is_some() {
            let ix = self.g.index_of(&gkey).ok_or_else(|| {
                ModelError::Msg(format!("GGUF 草稿源:索引里没有 `{gkey}`"))
            })?;
            let v = dequant_all(t.ty, self.g.bytes(ix), t.rows() as usize, t.inner() as usize)
                .map_err(|e| ModelError::Msg(format!("GGUF 草稿源:键 `{gkey}`: {e}")))?;
            return Ok(Some(v));
        }
        if Self::codebook_allowed(&gkey) {
            let ix = self.g.index_of(&gkey).ok_or_else(|| {
                ModelError::Msg(format!("GGUF 草稿源:索引里没有 `{gkey}`"))
            })?;
            let v = dequant_all(t.ty, self.g.bytes(ix), t.rows() as usize, t.inner() as usize)
                .map_err(|e| ModelError::Msg(format!("GGUF 草稿源:键 `{gkey}` 码本反量化: {e}")))?;
            return Ok(Some(v));
        }
        Err(ModelError::Msg(format!(
            "GGUF 草稿源:量化键 `{gkey}` 不接受元素读取(压缩驻留 I-3;唯一例外 = 两本 selector 码本)"
        )))
    }

    fn elem_len(&self, key: &str) -> Taken<usize> {
        let (gkey, tap) = self.resolve(key)?;
        Ok(match (self.g.get_tensor(&gkey), tap) {
            (None, _) => None,
            // fc tap:目标行字节 × 行(hidden)
            (Some(t), Some(_)) => {
                let (be, bb) = t.ty.block();
                Some(self.spec.hidden / be * bb * self.spec.hidden)
            }
            (Some(t), None) if t.ty.native_dtype().is_some() => {
                Some((t.rows() * t.inner()) as usize)
            }
            (Some(_t), None) => Some(self.g.bytes(self.g.index_of(&gkey).ok_or_else(|| {
                ModelError::Msg(format!("GGUF 草稿源:索引里没有 `{gkey}`"))
            })?).len()),
        })
    }

    fn native_dtype(&self, key: &str) -> Taken<Dtype> {
        let (gkey, tap) = self.resolve(key)?;
        let _ = tap;
        Ok(match self.g.get_tensor(&gkey) {
            // 量化块 ⇒ U8(**唯一**声称原生字节的情形;码本例外在
            // `convert_chunk_into_bytes` 里按名单处理)
            Some(t) => Some(t.ty.native_dtype().unwrap_or(Dtype::U8)),
            None => None,
        })
    }

    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Taken<()> {
        let (gkey, tap) = self.resolve(key)?;
        let Some(t) = self.g.get_tensor(&gkey) else {
            return Ok(None);
        };
        // ---- 量化件(fc tap / 普通块 / 码本)----
        if t.ty.native_dtype().is_none() {
            if let Some(i) = tap {
                // fc tap:U8 字节(源侧按行重排)
                if dtype != Dtype::U8 {
                    return Err(ModelError::Msg(format!(
                        "GGUF 草稿源:fc tap `{key}` 只以 U8 交付(实得 {dtype:?})"
                    )));
                }
                let (_, rows, row_bytes, _) = self.fc_geom()?;
                if i >= self.spec.layers {
                    return Err(ModelError::Msg(format!(
                        "GGUF 草稿源:fc tap {i} 越界(层数 {})",
                        self.spec.layers
                    )));
                }
                return self.copy_rows(&gkey, row_bytes, Some(i), rows, offset, len, dst);
            }
            if dtype == Dtype::U8 {
                let (be, bb) = t.ty.block();
                let rows = t.rows() as usize;
                let rows_bytes = t.inner() as usize / be * bb;
                return self.copy_rows(&gkey, rows_bytes, None, rows, offset, len, dst);
            }
            if Self::codebook_allowed(&gkey) {
                return self.dequant_window(&gkey, offset, len, dst, dtype);
            }
            return Err(ModelError::Msg(format!(
                "GGUF 草稿源:量化键 `{gkey}` 只能以 U8 交付(实得 {dtype:?})—— \
                 解量化请走内核(唯一例外 = 两本 selector 码本)"
            )));
        }
        // ---- 原生件(F32 norm / conv base)----
        self.native_window(&gkey, offset, len, dst, dtype)
    }
}

/// **草稿装载入口**(声明 + 加载一次到位;引擎侧唯一调用点)
///
/// 两步是分开的(`draft()` 只声明、`load()` 才上卡),因为引擎需要先拿到
/// 几何(池分配)再装权重 —— 与 HF 草稿档同序。
pub struct GgufDraftBoot {
    pub gguf: Gguf,
    pub spec: DraftSpec,
    /// target 事实(词表/层数/hidden;`draft()` 要用词表定 selector 码本行数)
    pub target: TargetFacts,
}

impl GgufDraftBoot {
    /// 打开容器 + 取声明(G0-a 的前半:跨文件一致性在 [`DraftSpec::from_gguf`])
    pub fn open(path: &std::path::Path, target: TargetFacts) -> Result<Self, ModelError> {
        if !path.is_file() {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿:期望 **.gguf 单文件**(实得 {})",
                path.display()
            )));
        }
        let gguf = Gguf::open(path)?;
        let spec = DraftSpec::from_gguf(&gguf, target)?;
        spec.assert_family_constants()?;
        Ok(Self { gguf, spec, target })
    }

    /// 声明草稿容器(逐张量计划;构造期取走 unresolved ⇒ 有未解析 = 响亮拒)
    pub fn draft(&self, dt: Dtype, kv_fp8: bool) -> Result<DFlash2Draft, ModelError> {
        let plan = DraftGgufPlan::new(&self.gguf);
        let keys = DraftGgufConvention;
        // 草稿 norm **未烘焙 +1**(实测 max|Δ|=0);且 DFlash2Draft 的 RmsNorm
        // 恒 plain ×w ⇒ 两族同档。`SourceFacts::HF` 只影响 norm 语义与
        // GDN 融合键(草稿无 GDN)⇒ 这里选 HF 是**事实**而非习惯。
        let ps = crate::module::PlanSource::keyed(&plan, &keys, SourceFacts::HF);
        // fc 计划:按源张量全形查表(五拆 Linear 的每片都是同一型/同一源张量)
        let fc_plan = crate::module::PlanTable::plan_at(
            &plan,
            "fc.weight",
            self.spec.hidden,
            self.spec.layers * self.spec.hidden,
        );
        // selector 投影(`selector_hidden.weight`,[rank, hidden])也是量化块
        let selector_plan = crate::module::PlanTable::plan_at(
            &plan,
            "selector_hidden.weight",
            self.spec.rank,
            self.spec.hidden,
        );
        let draft = DFlash2Draft::from_dims_keyed(
            self.spec.hidden,
            self.spec.inter,
            self.spec.hq,
            self.spec.hkv,
            self.spec.hd,
            self.spec.layers,
            self.spec.eps,
            // 词表来自 target(selector 码本行数 = target 词表)
            self.target.vocab,
            ps,
            fc_plan,
            selector_plan,
            dt,
            kv_fp8,
        );
        let unresolved = crate::module::PlanTable::take_unresolved(&plan);
        if !unresolved.is_empty() {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿计划表有 {} 个未解析键(构造期响亮拒):{}",
                unresolved.len(),
                unresolved.iter().take(6).cloned().collect::<Vec<_>>().join(" | ")
            )));
        }
        Ok(draft)
    }

    /// **权重上卡**(唯一入口:`GgufDraftSource` 建在本函数内,装载后立刻过
    /// G0-a 覆盖门 —— 文件里每个张量都被问过,且问过的键都存在)
    pub async fn load<D: crate::contract::DeviceClient>(
        &self,
        face: &mut D,
        draft: &DFlash2Draft,
        opts: crate::formats::load::LoadOpts,
    ) -> Result<crate::module::LoadManifest, ModelError> {
        let src = GgufDraftSource::new(&self.gguf, self.spec.clone());
        let manifest = crate::formats::load::load_component(face, draft, &src, opts).await?;
        let (asked, total) = src.coverage();
        let missed = src.unrequested();
        if !missed.is_empty() {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿覆盖门:文件有 {} 个张量未被声明请求(声明 {} / 文件 {}):{}{}",
                missed.len(),
                asked,
                total,
                missed.iter().take(6).cloned().collect::<Vec<_>>().join(", "),
                if missed.len() > 6 { ", …" } else { "" }
            )));
        }
        owl_shared::log_info!(
            "dflash",
            "GGUF 草稿装载:{} 个文件张量全部覆盖(声明请求 {} 项;fc 五拆共享同一源张量)",
            total,
            asked
        );
        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use owl_macros::owl_test;
    use owl_shared::testkit::resource;
    use crate::module::Module as _;
    use crate::{ForwardCtx, TensorOps};

    /// 27B target 事实(草稿共享 target 的词表/层数/hidden)
    fn target() -> TargetFacts {
        TargetFacts { vocab: 248_320, layers: 64, hidden: 5120 }
    }

    fn boot_key(key: &str) -> Option<GgufDraftBoot> {
        let p = resource(key)?;
        Some(GgufDraftBoot::open(std::path::Path::new(&p), target()).expect("开箱草稿"))
    }

    fn boot() -> Option<GgufDraftBoot> {
        boot_key("gguf:draft-q4")
    }

    /// 两档草稿(Q4_K_M / Q8_0)共用的声明门:同一套键面/形状/覆盖
    fn spec_gate_for(key: &str) {
        let Some(b) = boot_key(key) else {
            owl_shared::testkit::report::skip(format_args!("资源清单缺 {key}"));
            return;
        };
        let s = &b.spec;
        assert_eq!((s.layers, s.hidden, s.inter), (5, 5120, 17408));
        assert_eq!(s.tap_ids, vec![5, 19, 33, 47, 61], "[{key}] 1-based 归一");
        let plan = DraftGgufPlan::new(&b.gguf);
        let keys = DraftGgufConvention;
        use crate::module::PlanTable;
        let (hq, hkv, hd, hid, inter) = (s.hq, s.hkv, s.hd, s.hidden, s.inter);
        let cases: [(&str, usize, usize); 9] = [
            ("q_proj", hq * hd, hid),
            ("k_proj", hkv * hd, hid),
            ("v_proj", hkv * hd, hid),
            ("o_proj", hid, hq * hd),
            ("gate_proj", inter, hid),
            ("up_proj", inter, hid),
            ("down_proj", hid, inter),
            ("attn_conv_proj", 2 * 2 * (hid / 16), hid),
            ("mlp_conv_proj", 2 * 2 * (hid / 16), hid),
        ];
        for i in 0..s.layers {
            for (local, out, inp) in cases {
                let k = keys.layer_key(i, local);
                match plan.plan_at(&k, out, inp) {
                    QuantPlan::Gguf(_) => {}
                    other => panic!("[{key}] {k} 计划 {other:?}"),
                }
            }
        }
        assert!(matches!(plan.plan_at("fc.weight", hid, s.layers * hid), QuantPlan::Gguf(_)));
        assert!(matches!(plan.plan_at("selector_hidden.weight", s.rank, hid), QuantPlan::Gguf(_)));
        assert!(plan.take_unresolved().is_empty(), "[{key}] 计划表未解析键非空");

        // 覆盖双向(85 wants ↔ 81 张量)
        use crate::module::Loadable;
        let d = b.draft(Dtype::BF16, false).expect("声明草稿");
        let ctx = crate::module::LoaderCtx { dtype: Dtype::BF16, ..Default::default() };
        let ops = d.layout(&ctx);
        let mut wanted = std::collections::BTreeSet::new();
        for w in ops.wants() {
            let gkey = match GgufDraftSource::translate(&w.key)
                .unwrap_or_else(|e| panic!("[{}] {e}", w.key))
            {
                Target::Key(k) => k,
                Target::FcTap(_) => "fc.weight".to_string(),
            };
            assert!(b.gguf.get_tensor(&gkey).is_some(), "[{key}] 译表产出不存在的键 {gkey}");
            wanted.insert(gkey);
        }
        let file: std::collections::BTreeSet<String> =
            b.gguf.tensors.iter().map(|t| t.name.clone()).collect();
        assert!(file.difference(&wanted).next().is_none(), "[{key}] 有未被请求的张量");
        assert_eq!(file.len(), 81, "[{key}] 张量数");
        eprintln!("[{key}] 声明/计划/覆盖绿({} 项 want → {} 张量)", ops.wants().len(), file.len());
    }

    #[test]
    fn spec_gate_both_quantizations() {
        for key in ["gguf:draft-q4", "gguf:draft-q8"] {
            spec_gate_for(key);
        }
    }

    fn hf() -> Option<SafeHf> {
        let p = resource("ckpt:dflash2-bf16")?;
        Some(SafeHf::open(&std::path::Path::new(&p).join("model.safetensors")).expect("HF 草稿"))
    }

    /// HF 单文件 safetensors 直读(BF16/F32 → f32)
    struct SafeHf {
        data: Vec<u8>,
        idx: std::collections::HashMap<String, (usize, usize, String, Vec<usize>)>,
    }

    impl SafeHf {
        fn open(p: &std::path::Path) -> Result<Self, ModelError> {
            let data = std::fs::read(p)
                .map_err(|e| ModelError::Msg(format!("读 HF {}: {e}", p.display())))?;
            let st = safetensors::SafeTensors::deserialize(&data)
                .map_err(|e| ModelError::Msg(format!("HF safetensors 解析: {e}")))?;
            let mut idx = std::collections::HashMap::new();
            for name in st.names() {
                let v = st.tensor(name).map_err(|e| ModelError::Msg(format!("HF {name}: {e}")))?;
                // 偏移 = 视图指针 − 文件基址(同进程内一次 mmap/读,指针可用)
                let start = v.data().as_ptr() as usize - data.as_ptr() as usize;
                idx.insert(
                    name.to_string(),
                    (start, v.data().len(), format!("{:?}", v.dtype()), v.shape().to_vec()),
                );
            }
            Ok(Self { data, idx })
        }

        fn f32(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>), ModelError> {
            let (start, len, dt, shape) = self
                .idx
                .get(key)
                .ok_or_else(|| ModelError::Msg(format!("HF 缺键 `{key}`")))?
                .clone();
            let b = &self.data[start..start + len];
            let v = match dt.as_str() {
                "BF16" => b
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                "F32" => b
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                other => return Err(ModelError::Msg(format!("HF `{key}` dtype {other} 未接"))),
            };
            Ok((v, shape))
        }
    }

    /// 相对误差 + cos(量化噪声判据;错布局会给 O(0.5–1))
    fn rel_max(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "元素数");
        let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
    }

    /// **G0**:声明 + 计划表覆盖(零设备)
    #[test]
    fn spec_and_plan_cover_the_file() {
        let Some(b) = boot() else {
            owl_shared::testkit::report::skip("资源清单缺 gguf:draft-q4");
            return;
        };
        let s = &b.spec;
        assert_eq!((s.layers, s.hidden, s.inter), (5, 5120, 17408));
        assert_eq!((s.hq, s.hkv, s.hd), (32, 8, 128));
        assert_eq!((s.block, s.depth), (8, 7));
        assert_eq!((s.rank, s.top_k), (256, 16));
        assert_eq!(s.tap_ids, vec![5, 19, 33, 47, 61], "1-based 归一");
        assert_eq!(s.mask_token, DraftSpec::FAMILY_27B_MASK_TOKEN);
        assert_eq!(s.sliding_window, Some(2048));

        let plan = DraftGgufPlan::new(&b.gguf);
        let keys = DraftGgufConvention;
        use crate::module::PlanTable;
        // 七个 Linear 逐层:键在场 + 形状逐键相等 + 型已移植
        let (hq, hkv, hd, hid, inter) = (s.hq, s.hkv, s.hd, s.hidden, s.inter);
        let cases: [(&str, usize, usize); 7] = [
            ("q_proj", hq * hd, hid),
            ("k_proj", hkv * hd, hid),
            ("v_proj", hkv * hd, hid),
            ("o_proj", hid, hq * hd),
            ("gate_proj", inter, hid),
            ("up_proj", inter, hid),
            ("down_proj", hid, inter),
        ];
        for i in 0..s.layers {
            for (local, out, inp) in cases {
                let k = keys.layer_key(i, local);
                assert!(b.gguf.get_tensor(&k).is_some(), "索引缺 {k}");
                match plan.plan_at(&k, out, inp) {
                    QuantPlan::Gguf(_) => {}
                    other => panic!("{k} 计划 {other:?}(应为 GGUF 块型)"),
                }
            }
        }
        // fc:源张量全形 [hidden, layers·hidden]
        assert!(matches!(
            plan.plan_at("fc.weight", hid, s.layers * hid),
            QuantPlan::Gguf(_)
        ));
        assert!(plan.take_unresolved().is_empty(), "计划表未解析键非空");
        eprintln!("[G0] 草稿声明/计划表绿:{:?} taps", s.tap_ids);
    }

    /// **G0-a**:声明请求集 ⇔ 文件张量集(85 wants / 81 文件键;双向覆盖)
    #[test]
    fn want_list_matches_file_tensors() {
        let Some(b) = boot() else {
            owl_shared::testkit::report::skip("资源清单缺 gguf:draft-q4");
            return;
        };
        let d = b.draft(Dtype::BF16, false).expect("声明草稿");
        let ctx = crate::module::LoaderCtx {
            dtype: Dtype::BF16,
            ..Default::default()
        };
        use crate::module::Loadable;
        let ops = d.layout(&ctx);
        let mut wanted = std::collections::BTreeSet::new();
        for w in ops.wants() {
            let gkey = match GgufDraftSource::translate(&w.key)
                .unwrap_or_else(|e| panic!("[{}] {e}", w.key))
            {
                Target::Key(k) => k,
                Target::FcTap(_) => "fc.weight".to_string(),
            };
            assert!(b.gguf.get_tensor(&gkey).is_some(), "译表产出了不存在的键 {gkey}(源 {})", w.key);
            wanted.insert(gkey);
        }
        let file: std::collections::BTreeSet<String> =
            b.gguf.tensors.iter().map(|t| t.name.clone()).collect();
        let missed: Vec<&String> = file.difference(&wanted).collect();
        let extra: Vec<&String> = wanted.difference(&file).collect();
        assert!(missed.is_empty(), "文件有未被请求的张量:{missed:?}");
        assert!(extra.is_empty(), "译表有文件里没有的键:{extra:?}");
        eprintln!(
            "[G0-a] 覆盖绿:声明 {} 项 → 文件 {} 张量双向相等",
            ops.wants().len(),
            file.len()
        );
    }

    /// **G1-a**:fc 五拆重排(含非行对齐窗口)与 HF 一致(量化噪声)
    #[test]
    fn fc_tap_repack_matches_hf() {
        let (Some(b), Some(hf)) = (boot(), hf()) else {
            owl_shared::testkit::report::skip("缺 gguf:draft-q4 / ckpt:dflash2-bf16");
            return;
        };
        let s = b.spec.clone();
        let src = GgufDraftSource::new(&b.gguf, s.clone());
        let (fc, shape) = hf.f32("fc.weight").expect("HF fc");
        assert_eq!(shape, vec![s.hidden, s.layers * s.hidden], "HF fc [out, in]");
        let (ty, rows, _row_bytes, tap_row) = src.fc_geom().expect("fc 几何");
        assert_eq!(rows, s.hidden);
        let mut full = vec![0u8; rows * tap_row];
        for tap in [0usize, 4] {
            let key = format!("fc_{tap}.weight");
            // ① 整段
            src.convert_chunk_into_bytes(&key, 0, full.len(), &mut full, Dtype::U8)
                .expect("字节通道")
                .expect("键在场");
            // ② 非行对齐窗口 ≡ 整段切片(chunk 口径回归)
            let (off, len) = (100usize, 5_000usize);
            let mut part = vec![0u8; len];
            src.convert_chunk_into_bytes(&key, off, len, &mut part, Dtype::U8)
                .expect("窗口")
                .expect("键在场");
            assert_eq!(&part[..], &full[off..off + len], "tap{tap} 非行对齐窗口");

            let got = dequant_all(ty, &full, s.hidden, s.hidden).expect("fc tap 解量化");
            // HF:列块 [tap*hidden, (tap+1)*hidden)
            let mut want = Vec::with_capacity(s.hidden * s.hidden);
            for r in 0..s.hidden {
                let base = r * s.layers * s.hidden + tap * s.hidden;
                want.extend_from_slice(&fc[base..base + s.hidden]);
            }
            let e = rel_max(&got, &want);
            assert!(e < 1e-1, "fc tap{tap} 相对误差 {e}(Q4_K 噪声应 ≤1e-1)");
            eprintln!("[G1-a] fc tap{tap}:max rel {e:.3e}");
        }
    }

    /// **G1-b**:norm / conv base(原生 F32)与 HF **逐元素相等**(max|Δ| = 0)
    #[test]
    fn norms_and_conv_bases_match_hf() {
        let (Some(b), Some(hf)) = (boot(), hf()) else {
            owl_shared::testkit::report::skip("缺 gguf:draft-q4 / ckpt:dflash2-bf16");
            return;
        };
        let src = GgufDraftSource::new(&b.gguf, b.spec.clone());
        let cases = [
            ("hidden_norm.weight", "hidden_norm.weight"),
            ("norm.weight", "norm.weight"),
            ("layers.0.input_layernorm.weight", "layers.0.input_layernorm.weight"),
            ("layers.3.post_attention_layernorm.weight", "layers.3.post_attention_layernorm.weight"),
            ("layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_norm.weight"),
            ("layers.0.attention_conv.base_kernel", "layers.0.attention_conv.base_kernel"),
            ("layers.4.mlp_conv.base_kernel", "layers.4.mlp_conv.base_kernel"),
        ];
        for (only, hk) in cases {
            let (want, shape) = hf.f32(hk).expect("HF 张量");
            let n: usize = shape.iter().product();
            let mut buf = vec![0u8; n * 2];
            src.convert_chunk_into_bytes(only, 0, n, &mut buf, Dtype::BF16)
                .expect("元素通道")
                .expect("键在场");
            let got: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();
            assert_eq!(got.len(), want.len(), "[{only}] 元素数");
            let d = got
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert_eq!(d, 0.0, "[{only}] 与 HF 不逐位相等(max|Δ|={d})");
        }
        eprintln!("[G1-b] 7 件原生张量逐位相等(norm 未 +1 · rank-3 conv base 扁平序一致)");
    }

    /// **G1-c**:码本反量化(行窗口)与 HF 一致(量化噪声)
    #[test]
    fn codebook_rows_match_hf() {
        let (Some(b), Some(hf)) = (boot(), hf()) else {
            owl_shared::testkit::report::skip("缺 gguf:draft-q4 / ckpt:dflash2-bf16");
            return;
        };
        let src = GgufDraftSource::new(&b.gguf, b.spec.clone());
        let rank = b.spec.rank;
        for (owl, hk) in [
            ("candidate_selector.predecessor_codebook", "candidate_selector.predecessor_codebook"),
            ("candidate_selector.successor_codebook", "candidate_selector.successor_codebook"),
        ] {
            let (want, shape) = hf.f32(hk).expect("HF 码本");
            assert_eq!(shape, vec![248_320, rank]);
            for r in [0usize, 1234, 248_319] {
                let mut buf = vec![0u8; rank * 4];
                src.convert_chunk_into_bytes(owl, r * rank, rank, &mut buf, Dtype::F32)
                    .expect("码本窗口")
                    .expect("键在场");
                let got: Vec<f32> = buf
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let e = rel_max(&got, &want[r * rank..(r + 1) * rank]);
                assert!(e < 1e-1, "[{hk}] 行 {r} 相对误差 {e}");
            }
        }
        eprintln!("[G1-c] 两本码本抽样行绿(装载期反量化路径)");
    }

    /// **G0-c 预算**:草稿驻留 ≤ 1.3 GiB(Q4_K 1.14 GB + 码本反量化 0.25 GB)
    #[test]
    fn budget_is_within_cap() {
        let Some(b) = boot() else {
            owl_shared::testkit::report::skip("资源清单缺 gguf:draft-q4");
            return;
        };
        let bytes: u64 = b.gguf.tensors.iter().map(|t| t.nbytes).sum();
        // 两本码本 GGUF 里是 Q4_K(1 block/行),反量化后 2 × 248320 × 256 × 2B
        let codebooks_gguf: u64 = b
            .gguf
            .tensors
            .iter()
            .filter(|t| t.name.contains("selector_"))
            .map(|t| t.nbytes)
            .sum();
        let codebooks_bf16 = 2 * 248_320u64 * b.spec.rank as u64 * 2;
        let resident = bytes - codebooks_gguf + codebooks_bf16;
        assert!(
            resident <= 1_300 * 1024 * 1024,
            "草稿驻留 {} MiB 超 1.3 GiB 门",
            resident >> 20
        );
        eprintln!(
            "[G0-c] 草稿驻留 {} MiB(文件 {} MiB + 码本反量化净增 {} MiB)",
            resident >> 20,
            bytes >> 20,
            (codebooks_bf16 - codebooks_gguf) >> 20
        );
    }
    // ========================================================================
    // GPU 门(P1/P2 的装载与数值面;OWL_TEST_DEVICE 门控)
    // ========================================================================

    fn bf16b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| half::bf16::from_f32(*f).to_le_bytes()).collect()
    }

    fn gen(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i as f32 + seed) * 0.017).sin() * 0.8).clamp(-1.0, 1.0))
            .collect()
    }

    /// 复合激活(conv/mlp 输出)的容差单列:`conv_finish` 把 Q4_K 的
    /// `kernel_projection` 噪声再乘一次 base 并跨 tap 相加 ⇒ 随机测试输入下
    /// 实测 9.6e-2(贴 1e-1)。布局错位会给 O(0.5+) ⇒ 2e-1 仍判得开。
    fn assert_close_tol(a: &[f32], b: &[f32], tol: f32, tag: &str) {
        assert_eq!(a.len(), b.len(), "[{tag}] 长度");
        let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
        let d = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale;
        assert!(d <= tol, "[{tag}] 相对最大差 {d} > {tol}");
        eprintln!("[{tag}] rel {d:.3e}");
    }

    /// **P1/P2 门(GPU)**:GGUF 草稿装载 → `project_memory` + 层 0 的
    /// conv/mlp 与 **HF BF16 草稿**逐一对拍(量化噪声 ≤ 1e-1;cos 隐式由
    /// 相对差锁住)。同一份 taps、同一执行域(BF16)。
    #[owl_test(gpu)]
    async fn gpu_gguf_draft_matches_hf_draft() {
        use crate::module::LoaderCtx;
        let (Some(b), Some(hf_dir)) = (boot(), resource("ckpt:dflash2-bf16")) else {
            owl_shared::testkit::report::skip("缺 gguf:draft-q4 / ckpt:dflash2-bf16");
            return;
        };
        let mut gpu = crate::testkit::gpu_client().await;

        // ① 两侧装载(同一执行 dtype = 草稿单源 BF16)
        let g = b.draft(Dtype::BF16, false).expect("GGUF 草稿声明");
        let opts = crate::formats::load::LoadOpts::new(&gpu, Dtype::BF16);
        let man = b.load(&mut gpu, &g, opts).await.expect("GGUF 草稿装载");
        crate::formats::load::manifest_mem_ledger(&man, "gguf-draft-mem");
        let hf = crate::formats::qwen35::dflash2_draft(std::path::Path::new(&hf_dir), false)
            .expect("HF 草稿声明");
        let opts = crate::formats::load::LoadOpts::new(&gpu, Dtype::BF16);
        let src = crate::formats::safetensors::SafeTensorsSource::open_dir(std::path::Path::new(
            &hf_dir,
        ))
        .expect("HF 源");
        crate::interpreters::eval_load(&hf, &mut gpu, &src, &opts.ctx())
            .await
            .expect("HF 草稿装载");

        // ② memory 投影(fc 五拆 + hidden_norm)
        let t = 3usize;
        let hid = b.spec.hidden;
        let taps: Vec<TensorOps> = (0..5)
            .map(|i| {
                TensorOps::from_host(Dtype::BF16, vec![t, hid], &bf16b(&gen(t * hid, 3.0 + i as f32)))
            })
            .collect();
        let ctx = ForwardCtx::minimal(t);
        let got = crate::testkit::harvest_bf16(&mut gpu, &g.project_memory(&taps, &ctx)).await;
        let want = crate::testkit::harvest_bf16(&mut gpu, &hf.project_memory(&taps, &ctx)).await;
        assert_close_tol(&got, &want, 1e-1, "project_memory(gguf vs hf)");

        // ③ 层 0 的 conv 前包裹 + mlp(gate/up/down + silu)+ conv 后包裹
        let x = TensorOps::from_host(
            Dtype::BF16,
            vec![t, hid],
            &bf16b(&gen(t * hid, 77.0)),
        );
        for tag in ["l0"] {
            let (gin, gdel) = g.layers()[0].attn_conv().prepare(&x, &ctx);
            let (hin, hdel) = hf.layers()[0].attn_conv().prepare(&x, &ctx);
            let (a, b2) = (
                crate::testkit::harvest_bf16(&mut gpu, &gin).await,
                crate::testkit::harvest_bf16(&mut gpu, &hin).await,
            );
            assert_close_tol(&a, &b2, 1e-1, &format!("{tag}.conv_in"));
            let (a, b2) = (
                crate::testkit::harvest_bf16(&mut gpu, &gdel).await,
                crate::testkit::harvest_bf16(&mut gpu, &hdel).await,
            );
            assert_close_tol(&a, &b2, 1e-1, &format!("{tag}.conv_delta"));

            let gy = g.layers()[0].mlp().forward(&gin, &ctx);
            let hy = hf.layers()[0].mlp().forward(&hin, &ctx);
            let a = crate::testkit::harvest_bf16(&mut gpu, &gy).await;
            let b2 = crate::testkit::harvest_bf16(&mut gpu, &hy).await;
            let stat = |v: &[f32]| {
                let nan = v.iter().filter(|x| !x.is_finite()).count();
                let mx = v.iter().filter(|x| x.is_finite()).fold(0f32, |m, x| m.max(x.abs()));
                format!("{nan} 非有限 / max|·| {mx:.3e} / 首值 {:?}", &v[..3.min(v.len())])
            };
            eprintln!("[l0.mlp_out] gguf={} hf={}", stat(&a), stat(&b2));
            assert_close_tol(&a, &b2, 2e-1, &format!("{tag}.mlp_out"));

            let gfin = g.layers()[0].attn_conv().finish(&gy, &gdel, &ctx);
            let hfin = hf.layers()[0].attn_conv().finish(&hy, &hdel, &ctx);
            let a = crate::testkit::harvest_bf16(&mut gpu, &gfin).await;
            let b2 = crate::testkit::harvest_bf16(&mut gpu, &hfin).await;
            assert_close_tol(&a, &b2, 2e-1, &format!("{tag}.conv_finish"));
        }
        // ④ 成对指数移位自洽(`owl_scale_bf16` 的核门):x·2⁻⁶ → ·2⁶ ≡ x
        //    (2 的幂在 bf16 上精确;布局错/常量错会给 4096 倍级偏差)
        let raw: Vec<f32> = gen(64, 5.0);
        let x0 = TensorOps::from_host(Dtype::BF16, vec![4, 16], &bf16b(&raw));
        let pair = crate::layers::scale16(
            &crate::layers::scale16(&x0, 1.0 / 64.0),
            64.0,
        );
        let a = crate::testkit::harvest_bf16(&mut gpu, &pair).await;
        let b = crate::testkit::harvest_bf16(&mut gpu, &x0).await;
        assert_eq!(a, b, "[scale] 成对移位必须逐位复原");
        eprintln!("[P1/P2-gpu] owl_scale_bf16 成对移位逐位复原 ✓");

        let _ = LoaderCtx::default();
        gpu.close().await.expect("关机");
        eprintln!("[P1/P2-gpu] GGUF 草稿 ≡ HF 草稿(memory / conv / mlp;Q4_K 噪声域内)✓");
    }
}
