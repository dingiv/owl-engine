//! **GGUF 权重源**(`WeightSource` 第四族;2026-10-19 刀 4c)。
//!
//! 契约与既有源同:**源只管「键 → 字节」**(不懂设备、不懂目标布局);GGUF 的
//! 特殊性有两条,都在本模块内解决:
//!
//! 1. **压缩驻留**:量化键的字节**原样**交给装载面(槽 = `U8`、形状
//!    `[out, 块行字节]`),永不在 host 解量化(设计 I-3 / REQ-GGUF-05);
//! 2. **读侧变换**(设计 §十一):转换器做了六类变换,本源按
//!    [`transform::xf_for`] 的描述符逆变换 ——
//!    · **行粒度**置换(`attn_qkv` 的 V 段 / `attn_gate` / `alpha` / `beta`):
//!      在 **字节面**做(按行搬块,**零重量化**);
//!    · **值变换**(`ssm_a` = `ln(-x)`;小件 F32):在元素面做;
//!    · **列置换**(`ssm_out` 的内维):跨块 ⇒ **源不动字节**,由调用方在
//!      激活侧做(`owl_gguf_col_gather`;描述符 `RowPerm::ColVHead` 就是它的
//!      声明 —— 消费者据此知道"这层的 out_proj 需要先置换输入")。
//!
//! # 为什么行置换能在字节面做(设计 §11.4 的几何)
//!
//! 量化块沿 **`ne[0]`(内维)**,而置换发生在**外维**(行)⇒ 一行 = 内维的
//! 完整块链(如 `attn_gate` Q4_K:2560/256×144 = 1440 B/行)⇒ **搬整行即可**。
//! 反之列置换(内维)会跨块(`hv=128` < `QK_K=256`)⇒ 必须走激活侧。
//!
//! # 1-D 张量(`ssm_a` / `ssm_dt.bias`)的置换轴
//!
//! 这两个是 1-D(`ne=[nv]`),v-head 轴 = **元素轴**(`hd=1`)⇒ 走**元素面**
//! (它们是 F32、几十字节 —— 无性能问题)。`rows > 1` 的才走字节面。

use crate::contract::{Dtype, ModelError};
use crate::formats::mmap::Mmap;
use crate::module::WeightSource;
use owl_kernels::family::gguf::GgmlTy;

use super::dequant::dequant_all;
use super::meta::spec_from_gguf;
use super::reader::Gguf;
use super::shard::{geom_of, Cols, KeyGeom, ShardSel, Shape, TensorFacts};
use super::transform::{self, GgufDims, KeyXf, RowPerm, ValueXf};

/// 源读取错误里的键上下文(装载面按 `Taken` 收割;错误自带键名便于归因)
pub type Taken<T> = Result<Option<T>, ModelError>;

/// **虚拟键词尾**(源合成,文件里没有):GDN `out_proj` 的**激活侧列置换表**
/// (v-head 反重排;32 元素块粒度 = q8_1 块)。
///
/// 为什么它必须走"键 → 字节"这条道:置换表要**常驻显存**(轮内零 host 工作
/// ⇒ 可入捕获图),而“上卡”的唯一通道 = 权重槽 ⇒ 它像权重一样有键。
/// 名字由 [`GgufConvention`](super::convention) 的 `ssm_out_perm` 局部键给。
pub const PERM_TAIL: &str = ".ssm_out.perm";

/// GGUF 权重源
pub struct GgufSource {
    g: Gguf,
    /// 维度事实(开箱时从同一索引推出;**装载面与转渡面同源** —— 不让调用方
    /// 再开一次容器推 spec,避免两处推法漂移)
    spec: crate::model::ModelSpec,
    dims: GgufDims,
    /// 行映射缓存(键 → 目标行←源行;懒算 —— 同键多槽只算一次)
    perms: std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<Vec<u32>>>>,
    /// **TP 分片选择**(`None` = 单卡全量;几何见 [`super::shard`])
    shard: Option<ShardSel>,
    /// 逐键分片几何缓存(键 → 几何或"不切";懒算)
    geoms: std::sync::Mutex<std::collections::HashMap<String, Option<std::sync::Arc<KeyGeom>>>>,
}

impl GgufSource {
    /// 打开(容器已解析;维度事实经 `spec_from_gguf` 取 —— 同一条事实链)
    pub fn open(path: &std::path::Path) -> Result<Self, ModelError> {
        Self::open_shard(path, None)
    }

    /// **分片打开**(TP 档;`sel = None` 与 [`Self::open`] 等价 = 单卡全量)。
    ///
    /// 开箱即**逐键验一遍可分片性**(不整除 / 块不对齐 / 按头不可分 ⇒ 响亮拒),
    /// 而不是等装载到某个键才炸 —— 启动期一次判死(与计划表同律)。
    pub fn open_shard(
        path: &std::path::Path,
        shard: Option<ShardSel>,
    ) -> Result<Self, ModelError> {
        let g = Gguf::open(path)?;
        let (spec, _rep) = spec_from_gguf(&g)?;
        let (nk, hk, nv, hv) = spec.gdn_heads;
        let (nq, nkv, _hd) = spec.full_heads;
        let me = Self {
            g,
            spec,
            dims: GgufDims { nk, nv, hk, hv, nq, nkv },
            perms: std::sync::Mutex::new(Default::default()),
            shard,
            geoms: std::sync::Mutex::new(Default::default()),
        };
        me.validate_shard()?;
        Ok(me)
    }

    /// 分片选择(诊断/门用)
    pub fn shard(&self) -> Option<ShardSel> {
        self.shard
    }

    /// **逐键分片几何**(懒算 + 缓存;`None` = 不切 / 单卡)
    fn geom(&self, key: &str) -> Result<Option<std::sync::Arc<KeyGeom>>, ModelError> {
        let Some(sel) = self.shard else { return Ok(None) };
        if let Some(hit) = self
            .geoms
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .cloned()
        {
            return Ok(hit);
        }
        let Some(t) = self.g.get_tensor(key) else { return Ok(None) };
        let f = TensorFacts { ty: t.ty, rows: t.rows() as usize, inner: t.inner() as usize };
        let g = geom_of(key, &self.dims, &f, Some(sel))?.map(std::sync::Arc::new);
        self.geoms
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.to_string(), g.clone());
        Ok(g)
    }

    /// **开箱验分片**:索引里每个键都过一遍几何推导(失败 = 响亮拒 + 键名)
    fn validate_shard(&self) -> Result<(), ModelError> {
        if self.shard.is_none() {
            return Ok(());
        }
        for t in &self.g.tensors {
            let f = TensorFacts { ty: t.ty, rows: t.rows() as usize, inner: t.inner() as usize };
            geom_of(&t.name, &self.dims, &f, self.shard)
                .map_err(|e| ModelError::Msg(format!("GGUF 分片几何: {e}")))?;
        }
        Ok(())
    }

    /// 维度事实(引擎接线面:`Model::new` 的 spec 从这里取 —— 与 `dims` 同源)
    pub fn spec(&self) -> &crate::model::ModelSpec {
        &self.spec
    }

    /// 容器(诊断/上层推导用;词表/模板同源)
    pub fn gguf(&self) -> &Gguf {
        &self.g
    }

    /// 维度事实(变换表输入)
    pub fn dims(&self) -> &GgufDims {
        &self.dims
    }

    /// 逐键变换描述符(未知键 = 恒等)
    pub fn xf(&self, key: &str) -> KeyXf {
        transform::xf_for(key, &self.dims).unwrap_or(KeyXf::IDENTITY)
    }

    /// 该键是否量化块(需要走字节通道)
    fn ty_of(&self, key: &str) -> Option<GgmlTy> {
        self.g.get_tensor(key).map(|t| t.ty)
    }

    /// **虚拟键 → 块序表**(`None` = 不是虚拟键;
    /// 是虚拟键但形态不成立 ⇒ **响亮拒**,不静默给恒等表)
    fn virtual_perm(&self, key: &str) -> Taken<Vec<u32>> {
        if !key.ends_with(PERM_TAIL) {
            return Ok(None);
        }
        if self.shard.is_some() {
            return Err(ModelError::Msg(format!(
                "GgufSource: 虚拟键 `{key}`(单卡的 A2 列序表)在 TP 档**不存在**                 —— 分片装载已按 v-head grouped 序聚合 ⇒ 激活侧无需置换                 (`SourceFacts::GGUF_TP.vhead_tiled_columns = false`)"
            )));
        }
        if !self.dims.vhead_active() {
            return Err(ModelError::Msg(format!(
                "GgufSource: 虚拟键 `{key}` 请求列置换表,但本模型 nk == nv({} == {})\
                 —— 转换器对该模型不做 v-head 重排(设计 §11.1),表无意义",
                self.dims.nk, self.dims.nv
            )));
        }
        let t = transform::vhead_q8_blocks(self.dims.nk, self.dims.nv, self.dims.hv)
            .map_err(|e| ModelError::Msg(format!("GgufSource: 虚拟键 `{key}`: {e}")))?;
        Ok(Some(t))
    }

    fn is_quantized(&self, key: &str) -> bool {
        self.ty_of(key).is_some_and(|t| t.native_dtype().is_none())
    }

    /// **行映射**(目标行 i ← 源行 map[i];`None` = 恒等)
    ///
    /// 只在 `rows > 1` 时有意义;1-D 张量走元素面(见模块头 §1-D)。
    fn row_map(&self, key: &str, rows: usize) -> Result<Option<std::sync::Arc<Vec<u32>>>, ModelError> {
        let xf = self.xf(key);
        let (nk, nv, hd) = match xf.row_perm {
            RowPerm::No | RowPerm::ColVHead { .. } => return Ok(None),
            RowPerm::VHead { nk, nv, hd } => (nk, nv, hd),
            RowPerm::VSegOnly { nk, nv, hd, .. } => (nk, nv, hd),
            RowPerm::VChanOnly { nk, nv, hd, .. } => (nk, nv, hd),
        };
        if rows <= 1 {
            return Ok(None); // 1-D:元素面处理
        }
        // 中毒恢复:锁内只有纯查表/建表(无 I/O 无设备),中毒 = 别处 panic 的副作用;
        // 取回内部数据继续(禁 unwrap 吞没)
        let mut c = self.perms.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = c.get(key) {
            return Ok(Some(m.clone()));
        }
        // 段偏移(仅 V 段变体的前缀直拷)
        let seg_off = match xf.row_perm {
            RowPerm::VSegOnly { seg_off, .. } | RowPerm::VChanOnly { seg_off, .. } => seg_off,
            _ => 0,
        };
        let tail = rows - seg_off;
        if tail != nv * hd {
            // 形态不自洽 = 变换描述符与张量不符 ⇒ 响亮拒(不许静默搬错)
            drop(c);
            return Err(ModelError::Msg(format!(
                "GgufSource: 键 `{key}` 尾段行数 {tail} ≠ nv·hd = {}·{hd}(变换描述符与张量形态不符)",
                nv
            )));
        }
        let sub = transform::vhead_deperm(nk, nv, hd);
        let mut map: Vec<u32> = (0..seg_off as u32).collect();
        map.extend(sub.iter().map(|g| g + seg_off as u32));
        let arc = std::sync::Arc::new(map);
        c.insert(key.to_string(), arc.clone());
        Ok(Some(arc))
    }

    /// 元素面读取(原生型;含行/元素置换与值变换)
    fn read_elements(&self, key: &str) -> Taken<Vec<f32>> {
        if self.virtual_perm(key)?.is_some() {
            // 虚拟键是 U32 索引表,元素面（f32）载不动它 —— 响亮拒（走字节通道）
            return Err(ModelError::Msg(format!(
                "GgufSource: 虚拟键 `{key}` 是 U32 置换表,不接受元素读取(装载面应走 U32 字节通道)"
            )));
        }
        let Some(t) = self.g.get_tensor(key) else {
            return Ok(None);
        };
        if t.ty.native_dtype().is_none() {
            // 量化键不走元素面(调用方应走字节通道;此处响亮拒而非静默解量化)
            return Err(ModelError::Msg(format!(
                "GgufSource: 键 `{key}` 是 {} 量化块,不接受元素读取(压缩驻留:I-3)",
                t.ty.name()
            )));
        }
        let ix = self.g.index_of(key).expect("get_tensor 已过");
        let mut v = dequant_all(t.ty, self.g.bytes(ix), t.rows() as usize, t.inner() as usize)
            .map_err(|e| ModelError::Msg(format!("GgufSource: 键 `{key}`: {e}")))?;
        let rows = t.rows() as usize;
        let inner = t.inner() as usize;
        let xf = self.xf(key);
        // ① 置换:rows > 1 ⇒ 行;rows == 1(1-D)⇒ 元素
        if rows > 1 {
            if let Some(map) = self.row_map(key, rows)? {
                if map.len() != rows {
                    return Err(ModelError::Msg(format!(
                        "GgufSource: 键 `{key}` 行置换表长 {} ≠ 行数 {rows}(变换描述符与张量形态不符)",
                        map.len()
                    )));
                }
                let mut out = vec![0f32; v.len()];
                for (i, s) in map.iter().enumerate() {
                    let s = *s as usize;
                    out[i * inner..(i + 1) * inner].copy_from_slice(&v[s * inner..(s + 1) * inner]);
                }
                v = out;
            }
        } else if let RowPerm::VHead { nk, nv, hd } | RowPerm::VSegOnly { nk, nv, hd, .. } = xf.row_perm {
            // 1-D:元素置换(hd 必须为 1,否则该键的形态判断有误)
            if hd != 1 || v.len() != nv {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 键 `{key}` 是 1-D 却带 hd={hd} 的置换(形态不符)"
                )));
            }
            let map = transform::vhead_deperm(nk, nv, 1);
            let src = v.clone();
            for (i, s) in map.iter().enumerate() {
                v[i] = src[*s as usize];
            }
            // 值变换在置换后应用(两者可交换;此处固定顺序便于对拍)
        } else if let RowPerm::VChanOnly { nk, nv, hd, .. } = xf.row_perm {
            if hd != 1 || v.len() != nv {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 键 `{key}` 是 1-D 却带 VChanOnly(hd={hd})(形态不符)"
                )));
            }
            let map = transform::vhead_deperm(nk, nv, 1);
            let src = v.clone();
            for (i, s) in map.iter().enumerate() {
                v[i] = src[*s as usize];
            }
        }
        // ③ **列(内维)置换**(设计 §11.4 `ColVHead`):量化件跨块不可做
        // (走激活侧 tiled 量化核),**原生件**在此扭回 —— 零精度损失。
        if let RowPerm::ColVHead { nk, nv, hd } = xf.row_perm {
            if rows == 0 {
                return Err(ModelError::Msg(format!("GgufSource: 键 `{key}` 空张量")));
            }
            transform::col_deperm_in_place(&mut v, rows, inner, nk, nv, hd).map_err(|e| {
                ModelError::Msg(format!("GgufSource: 键 `{key}` 列置换: {e}"))
            })?;
        }
        // ② 值变换
        match xf.value {
            ValueXf::None => {}
            ValueXf::LnNeg => {
                for x in v.iter_mut() {
                    let a = -*x;
                    *x = if a > 0.0 { a.ln() } else { f32::NAN };
                }
            }
            ValueXf::MinusOne => {
                for x in v.iter_mut() {
                    *x -= 1.0;
                }
            }
        }
        // ③ **TP 切片**(原生件):全量变换后已是 grouped 序 ⇒ 目标 = grouped 的
        //    连续块(行窗口 × 列窗口;见 `shard` 模块头的不变式)
        if let Some(g) = self.geom(key)? {
            return Ok(Some(slice_grouped(&v, inner, &g, key)?));
        }
        Ok(Some(v))
    }
}


/// **grouped 序切片**(原生件的 TP 目标;见 `shard` 模块头的不变式)
///
/// 输入 = 全量逆变换后的 f32 数组(grouped 行序、grouped 列序),
/// 输出 = 本 rank 的连续块:行 = `wins` 的行窗口,列 = `[col_first, col_first+row_elems)`。
fn slice_grouped(
    v: &[f32],
    inner_full: usize,
    g: &KeyGeom,
    key: &str,
) -> Result<Vec<f32>, ModelError> {
    match &g.shape {
        Shape::Elem { first, len } => v
            .get(*first..first + len)
            .map(<[f32]>::to_vec)
            .ok_or_else(|| {
                ModelError::Msg(format!(
                    "[{key}] 1-D 元素窗口 [{first}, {}) 越界(全量 {})",
                    first + len,
                    v.len()
                ))
            }),
        Shape::Mat { wins, rows } => {
            let rw = g.row_elems();
            let c0 = g.col_first;
            if inner_full == 0 || c0 + rw > inner_full {
                return Err(ModelError::Msg(format!(
                    "[{key}] 列窗口 [{c0}, {}) 越界(全量内维 {inner_full})",
                    c0 + rw
                )));
            }
            let have_rows = v.len() / inner_full;
            let mut out = Vec::with_capacity(rows * rw);
            for (g0, n) in wins {
                let end = g0 + n;
                if end > have_rows {
                    return Err(ModelError::Msg(format!(
                        "[{key}] 行窗口 [{g0}, {end}) 越界(全量行数 {have_rows})"
                    )));
                }
                for r in *g0..end {
                    let b = r * inner_full + c0;
                    out.extend_from_slice(&v[b..b + rw]);
                }
            }
            if out.len() != rows * rw {
                return Err(ModelError::Msg(format!(
                    "[{key}] 分片元素数 {} ≠ 行 {} × 行宽 {rw}",
                    out.len(),
                    rows
                )));
            }
            Ok(out)
        }
    }
}

/// **列段表**(目标字节段 → 源字节偏移;单位 = 字节,段内连续)
///
/// `Full` = 整行;`Window` = 连续窗口;`Segs` = 逐段拼接(`ssm_out` 的
/// v-head grouped 列)。目标行的字节宽 = `row_tgt`。
fn col_segs(g: &KeyGeom, row_tgt: usize) -> Result<Vec<(usize, usize, usize)>, ModelError> {
    let segs: Vec<(usize, usize, usize)> = match &g.cols {
        Cols::Full => vec![(0, 0, row_tgt)],
        Cols::Window { src0, n } => {
            vec![(0, src0 * g.atom_bytes, n * g.atom_bytes)]
        }
        Cols::Segs(list) => {
            let mut out = Vec::with_capacity(list.len());
            let mut t = 0usize;
            for (src0, n) in list {
                out.push((t, src0 * g.atom_bytes, n * g.atom_bytes));
                t += n * g.atom_bytes;
            }
            out
        }
    };
    let sum: usize = segs.iter().map(|(_, _, n)| *n).sum();
    if sum != row_tgt {
        return Err(ModelError::Msg(format!(
            "列段字节和 {sum} ≠ 目标行宽 {row_tgt}(几何不自洽)"
        )));
    }
    Ok(segs)
}

/// 目标行 → grouped 行(行窗口表按目标拼接顺序,Windows 顺序即目标序)
fn grouped_row(wins: &[(usize, usize)], r: usize) -> Option<usize> {
    let mut acc = 0usize;
    for (g0, n) in wins {
        if r < acc + n {
            return Some(g0 + (r - acc));
        }
        acc += n;
    }
    None
}

impl GgufSource {
    /// **量化件的分片读**(单条源字节路径;见模块头 + `shard` 子模块)
    ///
    /// 目标字节 `[offset, offset+len)` 按 (行, 列段) 分解:行 = grouped 行窗口
    /// (`wins`),文件行 = 行置换表(若该键有 v-head 行序),列 = `col_segs`。
    /// 一次只搬"段内连续"的一段 ⇒ 与单卡路径同样零解量化。
    fn copy_quant_shard(
        &self,
        key: &str,
        g: &KeyGeom,
        offset: usize,
        len: usize,
        dst: &mut [u8],
    ) -> Taken<()> {
        let Shape::Mat { wins, rows } = &g.shape else {
            return Err(ModelError::Msg(format!(
                "GgufSource: 键 `{key}` 是量化件却拿到 1-D 分片几何(形态不符)"
            )));
        };
        let Some(t) = self.g.get_tensor(key) else { return Ok(None) };
        let (be, bb) = t.ty.block();
        if t.ty.native_dtype().is_some() {
            return Err(ModelError::Msg(format!(
                "GgufSource: 键 `{key}` 是原生型却走量化分片读(分派错)"
            )));
        }
        let inner = t.inner() as usize;
        let frows = t.rows() as usize;
        if inner % be != 0 {
            return Err(ModelError::Msg(format!(
                "GgufSource: 键 `{key}` 内维 {inner} 不是块宽 {be} 的整数倍"
            )));
        }
        let row_file = inner / be * bb;
        let row_tgt = g.row_atoms * g.atom_bytes;
        let total = rows * row_tgt;
        match offset.checked_add(len) {
            Some(end) if end <= total => {}
            _ => {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 键 `{key}` 请求 [{offset}, {}) 越界(本 rank 共 {total}B)",
                    offset + len
                )))
            }
        }
        if dst.len() < len {
            return Err(ModelError::Msg(format!(
                "GgufSource: 键 `{key}` 目标缓冲 {}B < 需 {len}B",
                dst.len()
            )));
        }
        let ix = self.g.index_of(key).unwrap_or_else(|| {
            unreachable!("get_tensor 已过 ⇒ 索引必在(单源)")
        });
        let src = self.g.bytes(ix);
        if src.len() != frows * row_file {
            return Err(ModelError::Msg(format!(
                "GgufSource: 键 `{key}` 源字节 {} ≠ 行 {frows} × 行字节 {row_file}",
                src.len()
            )));
        }
        let map = self.row_map(key, frows)?;
        let segs = col_segs(g, row_tgt)
            .map_err(|e| ModelError::Msg(format!("GgufSource: 键 `{key}`: {e}")))?;
        let mut abs = offset;
        let mut left = len;
        let mut out = 0usize;
        while left > 0 {
            let r = abs / row_tgt;
            let c = abs % row_tgt;
            let Some(grow) = grouped_row(wins, r) else {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 键 `{key}` 目标行 {r} 越出行窗口表(共 {rows})"
                )));
            };
            let frow = match &map {
                Some(m) => *m.get(grow).ok_or_else(|| {
                    ModelError::Msg(format!(
                        "GgufSource: 键 `{key}` grouped 行 {grow} 越出行置换表(共 {})",
                        m.len()
                    ))
                })? as usize,
                None => grow,
            };
            if frow >= frows {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 键 `{key}` 文件行 {frow} 越界(共 {frows})"
                )));
            }
            // 找到 c 所在的列段(段内连续;块边界由几何保证)
            let mut hit = None;
            for (t0, s0, n) in &segs {
                if c >= *t0 && c < *t0 + *n {
                    hit = Some((*t0, *s0, *n));
                    break;
                }
            }
            let Some((t0, s0, n)) = hit else {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 键 `{key}` 行内偏移 {c} 不在任何列段(几何不自洽)"
                )));
            };
            let within = c - t0;
            let take = (n - within).min(left);
            let s = frow * row_file + s0 + within;
            dst[out..out + take].copy_from_slice(&src[s..s + take]);
            abs += take;
            left -= take;
            out += take;
        }
        Ok(Some(()))
    }
}

impl WeightSource for GgufSource {
    /// ★ **源自己承担分片**(GGUF 臂):量化块的列切 = 块算术,`ssm_out` 的列
    /// 还要按 v-head grouped 序聚合 —— 通用 span 包装器的"连续源区间"模型
    /// 表达不了(见 [`super::shard`] 模块头)
    fn owns_shard(&self) -> bool {
        self.shard.is_some()
    }

    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        self.read_elements(key)
    }

    fn elem_len(&self, key: &str) -> Taken<usize> {
        // 虚拟键:块序表长度(= c/32;与 want 的 shape.product 同口径)
        if let Some(t) = self.virtual_perm(key)? {
            return Ok(Some(t.len()));
        }
        // TP:本 rank 的交付量(量化 = 字节,原生 = 元素;装载面按 shape.product 对账)
        if let Some(g) = self.geom(key)? {
            return Ok(Some(g.len()));
        }
        // 量化键的"元素"= **字节**(槽 dtype=U8)⇒ 报告字节数(与 want 的
        // shape.product 同口径,错误归因才准)
        Ok(match self.g.get_tensor(key) {
            None => None,
            Some(t) if t.ty.native_dtype().is_some() => Some((t.rows() * t.inner()) as usize),
            Some(_) => Some(self.g.bytes(self.g.index_of(key).unwrap()).len()),
        })
    }

    fn native_dtype(&self, key: &str) -> Taken<Dtype> {
        // 虚拟键:U32 索引表(槽 dtype = U32)
        if self.virtual_perm(key)?.is_some() {
            return Ok(Some(Dtype::U32));
        }
        Ok(match self.ty_of(key) {
            // 量化块 ⇒ **U8**(不透明字节;装载面走 esz=1 的元素通道,零转换)
            Some(t) => Some(t.native_dtype().unwrap_or(Dtype::U8)),
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
        // 虚拟键(列置换表):合成 U32 元素 → LE 字节(唯一交付形态)
        if let Some(t) = self.virtual_perm(key)? {
            if dtype != Dtype::U32 {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 虚拟键 `{key}` 只以 U32 交付(实得 {dtype:?})"
                )));
            }
            let Some(slice) = t.get(offset..offset + len) else {
                return Ok(None);
            };
            if dst.len() < len * 4 {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 虚拟键 `{key}` 目标缓冲 {}B < 需 {}B",
                    dst.len(),
                    len * 4
                )));
            }
            for (i, v) in slice.iter().enumerate() {
                dst[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            return Ok(Some(()));
        }
        if !self.is_quantized(key) {
            // 原生型:走缺省元素路径(dequant_all + 元素转换)
            return self.default_convert(key, offset, len, dst, dtype);
        }
        if dtype != Dtype::U8 {
            return Err(ModelError::Msg(format!(
                "GgufSource: 量化键 `{key}` 只能以 U8 字节交付(实得 {dtype:?})—— 解量化请走内核"
            )));
        }
        // **TP 分片读**(量化件):按 (grouped 行窗口, 列原子段) 从 mmap 搬块;
        // 行置换(文件行 ← grouped 行)与块宽都在这里算 —— 通用 span 包装器
        // 表达不了"行置换 × 块对齐列切"的组合(见 `shard` 模块头)
        if let Some(g) = self.geom(key)? {
            return self.copy_quant_shard(key, &g, offset, len, dst);
        }
        let Some(t) = self.g.get_tensor(key) else {
            return Ok(None);
        };
        let ix = self.g.index_of(key).unwrap();
        let src = self.g.bytes(ix);
        let (rows, inner) = (t.rows() as usize, t.inner() as usize);
        let (be, bb) = t.ty.block();
        let row_bytes = inner / be * bb;
        if dst.len() < len {
            return Err(ModelError::Msg(format!(
                "GgufSource: 键 `{key}` 目标缓冲 {}B < 需 {len}B",
                dst.len()
            )));
        }
        let map = self.row_map(key, rows)?;
        // 逐**目标行**搬块(行 = 内维完整块链 ⇒ 行内可整段 memcpy)
        let mut done = 0usize;
        while done < len {
            let abs = offset + done;
            let r = abs / row_bytes;
            let c = abs % row_bytes;
            if r >= rows {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 键 `{key}` 请求偏移 {abs} 越界(行 {r} ≥ {rows})"
                )));
            }
            let n = (row_bytes - c).min(len - done);
            let srow = match &map {
                Some(m) if !m.is_empty() => m[r] as usize,
                _ => r,
            };
            if srow >= rows {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 键 `{key}` 行映射越界({srow} ≥ {rows})"
                )));
            }
            let s = srow * row_bytes + c;
            dst[done..done + n].copy_from_slice(&src[s..s + n]);
            done += n;
        }
        Ok(Some(()))
    }
}

impl GgufSource {
    /// 原生型的缺省元素转换(mmap → f32 → 目标 dtype)
    fn default_convert(
        &self,
        key: &str,
        offset: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Taken<()> {
        let Some(v) = self.read_elements(key)? else {
            return Ok(None);
        };
        let Some(slice) = v.get(offset..offset + len) else {
            return Ok(None);
        };
        let need = match dtype {
            Dtype::F32 => len * 4,
            Dtype::F16 | Dtype::BF16 => len * 2,
            other => {
                return Err(ModelError::Msg(format!(
                    "GgufSource: 键 `{key}` 原生通路不支持目标 dtype {other:?}"
                )))
            }
        };
        if dst.len() < need {
            return Err(ModelError::Msg(format!(
                "GgufSource: 键 `{key}` dst {}B < 需 {need}B",
                dst.len()
            )));
        }
        for (i, f) in slice.iter().enumerate() {
            match dtype {
                Dtype::F32 => dst[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes()),
                Dtype::F16 => {
                    dst[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(*f).to_le_bytes())
                }
                Dtype::BF16 => {
                    dst[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(*f).to_le_bytes())
                }
                _ => unreachable!("上面已筛过"),
            }
        }
        Ok(Some(()))
    }
}

/// 未用:Mmap 由 `Gguf` 持有(lifetime 说明;留此以便将来"源独立持映射")
#[allow(dead_code)]
type _MmapAlias = Mmap;


#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::WeightSource;
    use owl_shared::testkit::{resource, resource_ok};
    use std::path::PathBuf;
    use std::process::Command;

    /// 列变换档(比较侧;见用例表 `ssm_out` 一条)
    #[derive(PartialEq, Eq, Clone, Copy)]
    enum ColXf {
        None,
        /// 内维按 v-head 反重排(与行同表)
        VHead,
    }

    /// **HF 快照对拍门**(刀 4c 的判据;设计 §十一 11.2/11.5)    ///
    /// 判据 = `源(键) == HF 快照同张量`(解量化 + 逆变换后**逐元素**相等,
    /// 容差 = 量化噪声)。这一门同时锁住三件事:
    /// ① 块解量化(与 `gguf-py` 位级门互补:那门锁算法,这门锁**端到端用途**);
    /// ② **逆变换表**(V-head / V 段 / V 通道 / `ln(-x)`);③ 键 → HF 键的对应关系。
    ///
    /// ⚠️ **容差口径**:GGUF 是量化件,HF 是 bf16 源 ⇒ 逐元素差 ≈ 量化误差
    /// (相对 max|ref| 约 1e-2);**错变换**会给 O(1)。
    #[test]
    fn source_matches_hf_snapshot_after_inverse_transforms() {
        let (Some(gguf), Some(hf)) = (
            resource_ok("gguf:4b").then(|| resource("gguf:4b").unwrap()),
            resource_ok("tpl:hf4b").then(|| resource("tpl:hf4b").unwrap()),
        ) else {
            eprintln!("skip: 资源清单缺 gguf:4b / tpl:hf4b");
            return;
        };
        let src = GgufSource::open(std::path::Path::new(&gguf)).expect("open 源");
        let base_l0 = "model.language_model.layers.0.";
        let base_l3 = "model.language_model.layers.3.";
        // (GGUF 键, HF 局部键, 容差, **norm 语义**:true = GGUF 值已烘焙 +1)
        //
        // ⚠️ **容差定档依据(实测,2026-10-19)**:量化张量的最大单元素相对误差
        // 实测 1.5e-2(Q6_K)/ 4.1e-2(Q4_K)/ **6.3e-2(Q5_K)** —— 这是**量化噪声**
        // (小幅度元素落在粗子块里),与"错置换"的签名不同:错置换/错段偏移会给
        // **O(0.5–1.0)**(整行内容换了)。原生/小件(F32)门保持 1e-5 级(实测
        // `ssm_alpha/beta/dt_bias/conv1d/ssm_norm` 与四个 norm **精确 0**)。
        // 阈值取 1e-1:容得下量化噪声,仍远离 O(0.5)。
        //
        // 为什么 norm 要单列一栏:转换器给 norm **加 1**(llama.cpp 的 `1+w` 约定),
        // 而本仓选择**语义法**(`w_off=false`,即直接 `×w`)—— 零变换、零新算子。
        // 因此源侧**原样**交出 `w+1`,对拍时减回来。⚠️ 若将来改回"源侧减 1",
        // 本条与 `RmsNorm::new_add_one` 的选档必须同步(单源 = 设计 §11.3)。
        let cases: &[(&str, &str, f32, bool, ColXf)] = &[
            ("blk.0.attn_qkv.weight", "linear_attn.in_proj_qkv.weight", 1e-1, false, ColXf::None),
            ("blk.0.attn_gate.weight", "linear_attn.in_proj_z.weight", 1e-1, false, ColXf::None),
            ("blk.0.ssm_alpha.weight", "linear_attn.in_proj_a.weight", 1e-2, false, ColXf::None),
            ("blk.0.ssm_beta.weight", "linear_attn.in_proj_b.weight", 1e-2, false, ColXf::None),
            ("blk.0.ssm_dt.bias", "linear_attn.dt_bias", 1e-3, false, ColXf::None),
            ("blk.0.ssm_a", "linear_attn.A_log", 1e-5, false, ColXf::None),
            ("blk.0.ssm_conv1d.weight", "linear_attn.conv1d.weight", 1e-5, false, ColXf::None),
            // `linear_attn.norm` **不 +1**(转换器明文例外)⇒ 直比
            ("blk.0.ssm_norm.weight", "linear_attn.norm.weight", 1e-5, false, ColXf::None),
            ("blk.0.attn_norm.weight", "input_layernorm.weight", 1e-5, true, ColXf::None),
            ("blk.0.post_attention_norm.weight", "post_attention_layernorm.weight", 1e-5, true, ColXf::None),
            ("blk.3.attn_q_norm.weight", "self_attn.q_norm.weight", 1e-5, true, ColXf::None),
            ("blk.3.attn_k_norm.weight", "self_attn.k_norm.weight", 1e-5, true, ColXf::None),
            ("output_norm.weight", "norm.weight", 1e-5, true, ColXf::None),
            ("blk.3.attn_q.weight", "self_attn.q_proj.weight", 1e-1, false, ColXf::None),
            ("blk.3.attn_v.weight", "self_attn.v_proj.weight", 1e-1, false, ColXf::None),
            ("blk.3.attn_output.weight", "self_attn.o_proj.weight", 1e-1, false, ColXf::None),
            // ★ **列(内维)反重排的**方向**钉定**(2026-10-19 刀 7f)**:
            // `ssm_out` 的内维是 v-head(ggml tiled 序)⇒ 与 HF 比较时在内维上
            // 走**与行同表**(`vhead_deperm` = g):`W_hf[:, h] = W_gguf[:, g(h)]`。
            // 这是**独立于转换器源码阅读**的实验证据:方向若反 ⇒ O(1) 相对误差
            // (整列换了),若对 ⇒ 量化噪声(1e-1 阈内)。
            ("blk.0.ssm_out.weight", "linear_attn.out_proj.weight", 1e-1, false, ColXf::VHead),
        ];
        let script = crate::testkit::manifest_dir().join("src/formats/hf_probe.py");
        assert!(script.is_file(), "缺 {}", script.display());
        let mut checked = 0usize;
        for (gk, hkl, tol, norm_plus_one, col) in cases {
            // 顶层键(embed/head/norm)不带 layers.N 前缀
            let hf_key = if gk.starts_with("blk.3") {
                format!("{base_l3}{hkl}")
            } else if gk.starts_with("blk.") {
                format!("{base_l0}{hkl}")
            } else {
                format!("model.language_model.{hkl}")
            };
            // 量化键走**字节通道**(I-3:元素面按纪律拒);原生键走元素面。
            // 两条通道的产物都是"经全部逆变换后的逻辑值" ⇒ 与 HF 可比。
            let t = src.gguf().get_tensor(gk).unwrap_or_else(|| panic!("源缺键 {gk}"));
            let mut v: Vec<f32> = if t.ty.native_dtype().is_some() {
                src.take(gk).expect("take").unwrap_or_else(|| panic!("源缺键 {gk}"))
            } else {
                let (rows, inner) = (t.rows() as usize, t.inner() as usize);
                let (be, bb) = t.ty.block();
                let row_bytes = inner / be * bb;
                let mut buf = vec![0u8; rows * row_bytes];
                src.convert_chunk_into_bytes(gk, 0, buf.len(), &mut buf, Dtype::U8)
                    .expect("字节通道")
                    .unwrap_or_else(|| panic!("源缺键 {gk}"));
                dequant_all(t.ty, &buf, rows, inner).expect("dequant_all")
            };
            let outp = std::env::temp_dir().join(format!("owl_hf_{}.bin", gk.replace(['.', '_'], "-")));
            let out = Command::new("uv")
                .arg("run")
                .arg("python")
                .arg(&script)
                .arg(&hf)
                .arg(&hf_key)
                .arg(&outp)
                .current_dir(crate::testkit::manifest_dir())
                .output()
                .expect("spawn uv(python 环境未就绪?先在 crates/models 下 `uv sync`)");
            assert!(
                out.status.success(),
                "[{gk}] HF 导出失败:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let raw = std::fs::read(&outp).expect("读 HF 导出");
            let want: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(v.len(), want.len(), "[{gk}] 元素数(源 {} vs HF {})", v.len(), want.len());
            // ★ 列反重排(比较侧:与行同表;见用例表注释与设计 §11.4)
            if *col == ColXf::VHead {
                let d = src.dims();
                let t = src.gguf().get_tensor(gk).expect("张量在场");
                let (rows, inner) = (t.rows() as usize, t.inner() as usize);
                assert_eq!(inner, d.nv * d.hv, "[{gk}] 内维 {inner} ≠ nv·hv = {}·{}", d.nv, d.hv);
                let map = transform::vhead_deperm(d.nk, d.nv, d.hv);
                let prev = v.clone();
                for r in 0..rows {
                    for j in 0..inner {
                        v[r * inner + j] = prev[r * inner + map[j] as usize];
                    }
                }
            }
            let scale = want.iter().fold(0f32, |a, x| a.max(x.abs())).max(1e-6);
            let mut maxd = 0f32;
            let mut worst = 0usize;
            for (i, (a, b)) in v.iter().zip(&want).enumerate() {
                // 语义法:GGUF 的 norm 值 = HF + 1 ⇒ 对拍时减回
                let a = if *norm_plus_one { &(*a - 1.0) } else { a };
                let d = match (a.is_finite(), b.is_finite()) {
                    (false, false) => 0.0,
                    (true, true) => (a - b).abs() / scale,
                    _ => f32::INFINITY,
                };
                if d > maxd {
                    maxd = d;
                    worst = i;
                }
            }
            eprintln!("[{gk}] vs HF {hkl}: max|Δ|/max|ref| = {maxd:.3e} @ {worst}");
            assert!(
                maxd <= *tol,
                "[{gk}] 超容差 {tol:e}:{maxd:.3e} @ {worst}(源 {} vs HF {};错变换会给 O(1))",
                v[worst],
                want[worst]
            );
            checked += 1;
            let _ = std::fs::remove_file(&outp);
        }
        assert!(checked >= 8, "对拍用例太少({checked})");
        eprintln!("[gguf-src] HF 快照对拍绿({checked} 张量)");
    }

    /// 响亮拒 + 两条通道的自洽
    #[test]
    fn loud_rejections_and_channel_agreement() {
        if !resource_ok("gguf:4b") {
            eprintln!("skip: 资源清单缺 gguf:4b");
            return;
        }
        let src = GgufSource::open(&PathBuf::from(resource("gguf:4b").unwrap())).expect("open");
        // 量化键:元素面必须拒(I-3)
        let e = src.take("blk.0.attn_gate.weight").unwrap_err().to_string();
        assert!(e.contains("量化块"), "{e}");
        // 量化键:字节面只收 U8
        let mut buf = vec![0u8; 64];
        let e = src
            .convert_chunk_into_bytes("blk.0.attn_gate.weight", 0, 64, &mut buf, Dtype::F16)
            .unwrap_err()
            .to_string();
        assert!(e.contains("只能以 U8"), "{e}");
        // 原生键:byte 通道可用(走元素路径)
        let mut buf2 = vec![0u8; 64];
        assert!(src
            .convert_chunk_into_bytes("blk.0.ssm_norm.weight", 0, 32, &mut buf2, Dtype::F16)
            .unwrap()
            .is_some());
        // 未知键 = None(装载面收成缺键)
        assert!(src.take("blk.999.nope").unwrap().is_none());
        assert!(src.elem_len("blk.999.nope").unwrap().is_none());

        // 字节面 vs 文件:恒等键逐字节同;行置换键按表搬;列置换键原样
        for key in ["blk.0.attn_gate.weight", "blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "blk.0.ffn_up.weight"] {
            let t = src.gguf().get_tensor(key).unwrap();
            let (rows, inner) = (t.rows() as usize, t.inner() as usize);
            let (be, bb) = t.ty.block();
            let row_bytes = inner / be * bb;
            let mut got = vec![0u8; rows * row_bytes];
            src.convert_chunk_into_bytes(key, 0, got.len(), &mut got, Dtype::U8)
                .unwrap()
                .expect("字节通道");
            let file = src.gguf().bytes(src.gguf().index_of(key).unwrap());
            let xf = src.xf(key);
            let map = src.row_map(key, rows).expect("row_map");
            match (&xf.row_perm, &map) {
                (RowPerm::ColVHead { .. }, _) | (_, None) => {
                    assert_eq!(&got, file, "[{key}] 列置换/恒等键应逐字节同源文件")
                }
                (_, Some(m)) => {
                    let s0 = m[0] as usize * row_bytes;
                    assert_eq!(&got[..row_bytes], &file[s0..s0 + row_bytes], "[{key}] 首行未按表搬");
                    // 表是置换 ⇒ 逐行内容集合 = 文件行内容集合
                    if m[0] == 0 {
                        let s_last = m[rows - 1] as usize * row_bytes;
                        assert_eq!(
                            &got[(rows - 1) * row_bytes..],
                            &file[s_last..s_last + row_bytes],
                            "[{key}] 末行未按表搬"
                        );
                    }
                }
            }
            eprintln!("[{key}] 字节通道 {:?} ✓", xf.row_perm);
        }
    }
}
