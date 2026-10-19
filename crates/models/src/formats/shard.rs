//! **装载面的分片化**(S4 ②)—— 确定性分片协议在装载侧的落地。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §15.3 / §15.3b。
//!
//! # 一个 pass + 一个包装器(零改 6 条装载路径)
//!
//! ```text
//! layer.layout(ctx) ──▶ LoaderOps(全量 Want)
//!        │
//!   [build_shard_pass]        ← 规则表 + (slice_id, shard) + Want.segments
//!        │  · 就地改写:want.shape → 分片形状(DeviceRearrange 的 rows/cols 同步)
//!        │  · 产出:key → SourceSpan(读映射)+ 行宽
//!        ▼
//!   ShardedSource(inner, spans) ← 包装源:把"目标坐标系的 [off,len)"
//!        │                        翻译成源坐标系的读序列(行窗口/列窗口/分段)
//!        ▼
//!   eval_want(…, &sharded, …)  ← 6 条路径**零改动**(它们只调 WeightSource)
//! ```
//!
//! # 为什么是"包装源"而不是改 6 条路径
//!
//! 装载执行面(direct/cast/packed/device_rearrange/transposed)各自实现细节
//! 不同,但**都只通过 `WeightSource` 取数** ⇒ 在源的边界上做翻译,路径全部
//! 复用;每条路径各自的"分块/暂存/上卡"逻辑不用知道切分存在。
//!
//! # 纪律
//!
//! - `shard = 1` ⇒ **零开销**(pass 直接返回空表,源不包装);
//! - 包装器对**未进表**的键直通(不是所有键都被切);
//! - 越界/映射失败 = `None`(由调用方归因为"缺键/形状错",不 panic)。

use std::collections::HashMap;

use crate::contract::{Dtype, ModelError};
use crate::module::Taken;
use crate::module::{Layout, LoaderCtx, LoaderOps, WeightSource};
use crate::plan::rules::{rule_of_want, want_axis, WantAxis, WeightRule};
use crate::plan::span::segments_in_key_units;
use crate::plan::span::{plan_span, SourceSpan};
use crate::plan::tp::TpPlan;

/// span 表项:读映射 + 行宽(元素口径;`SourceSpan::map` 的入参)
#[derive(Debug, Clone)]
pub struct SpanEntry {
    pub span: SourceSpan,
    /// 源行宽(元素;全量 `shape[1..]` 之积,一维 = 1)
    pub src_row_width: usize,
    /// 目标行宽(元素;分片 `shape[1..]` 之积)
    pub tgt_row_width: usize,
    /// 源元素总数(校验/诊断)
    pub src_elems: usize,
    /// 分片元素总数(=`elem_len` 的答案)
    pub tgt_elems: usize,
}

/// **分片形状 → 源全量形状**(等分切分的逆;§15.3b 的"形状贯通")。
///
/// 为什么需要:层的 `Want.shape` 已经是**按分片 spec 建的分片形状**
/// (`ModelSpec::sharded` 先行)—— 而 [`SourceSpan`] 的映射基准是**源的全量
/// 形状**。等分切分的逆是唯一的:全量 = 分片 × shard(分段切逐段同律)。
///
/// `axis`(2026-10-17):量化 marlin 产物是 **k-major** ⇒ "切 out" 落在第 1 维。
fn inflate(
    shape: &[usize],
    rule: WeightRule,
    plan: TpPlan,
    segs: &[usize],
    axis: WantAxis,
) -> Result<Vec<usize>, String> {
    let mut full = shape.to_vec();
    let w = plan.world();
    let mul = |v: usize| -> Result<usize, String> {
        v.checked_mul(w).ok_or_else(|| "inflate: 维溢出".to_string())
    };
    // 规则 → want 自身坐标系的维度(与 `plan_span` 同一套映射)
    let dim_of_out = if axis == WantAxis::KMajor { 1 } else { 0 };
    let dim_of_k = if axis == WantAxis::KMajor { 0 } else { 1 };
    match rule {
        WeightRule::Replicate => {}
        WeightRule::PerHead => {
            full[0] = mul(shape[0])?;
        }
        WeightRule::UniformCol => {
            full[dim_of_out] = mul(shape[dim_of_out])?;
        }
        WeightRule::UniformRow => {
            if shape.len() < 2 {
                return Err(format!("inflate: UniformRow 需要 ≥2 维(得 {shape:?})"));
            }
            full[dim_of_k] = mul(shape[dim_of_k])?;
        }
        WeightRule::SegmentedCol(_) => {
            // 全量维 = 分片维 × shard(**键自身维口径**)。
            // ⚠️ 不能用"段长合计 × shard":段长声明是**逻辑单位**(q_gate/k_dim/
            //    v_dim),而打包派生键的 n 侧单位不同(`zeros` = n_pack/8)——
            //    2026-10-17 首跑就栽在这(算出 [160, 8192] vs 层侧 [160, 1024])。
            //    段长→键单位的换算在下面 ①b 单独做。
            let _ = segs;
            full[dim_of_out] = mul(shape[dim_of_out])?;
        }
    }
    Ok(full)
}

/// **分片化 pass**(纯逻辑,就地改写):
///
/// 1. 从 `want.shape`(层侧的**分片形状**)**反推源全量形状**;
/// 2. `plan_span(全量形状, plan, 全量段长)` → 读计划 + 分片形状;
/// 3. **强校验**:反推的分片形状必须逐字等于 `want.shape` —— 层侧
///    (`ModelSpec::sharded`)与装载侧(规则表)的切分约定**必须一致**,
///    不一致 ⇒ 响亮拒(胜过"装载出形状不对的块,forward 时才炸");
/// 4. 同步 `Layout::DeviceRearrange{rows,cols}`;返回 `key → SpanEntry`。
///
/// `ctx.shard = 1` ⇒ 空表(零开销退化;`Replicate` 的键不进表)。
pub fn build_shard_pass(
    ops: &mut LoaderOps,
    ctx: &LoaderCtx,
) -> Result<HashMap<String, SpanEntry>, ModelError> {
    let mut spans = HashMap::new();
    if ctx.shard <= 1 {
        return Ok(spans);
    }
    let plan = TpPlan::new(ctx.slice_id, ctx.shard).map_err(ModelError::Msg)?;
    for w in ops.wants_mut() {
        // **带段长事实**的规则推导(P2.1-a):同名键在「融合/不融合」两个模型里
        // 语义不同 —— 段长由层声明,缺声明 ⇒ 不按段切(见 `rule_of_want`)。
        // **层声明的规则覆盖优先**(P1.1,2026-10-19):"这个张量切不切"是装配期
        // 事实(如 lm_head 的词表切随档位变),键名里没有它 ⇒ 层是声明者,
        // 装载面是消费者(want.rule;仅真覆盖时在场)。
        let rule = w
            .rule
            .unwrap_or_else(|| rule_of_want(&w.key, w.segments.as_deref()));
        let axis = want_axis(&w.key);
        let shard_segs: &[usize] = w.segments.as_deref().unwrap_or(&[]);
        if rule == WeightRule::Replicate {
            continue; // 不切:直通
        }
        // ⚠️ **CPU marlin 物化键不可矩形切**(2026-10-17):`.qweight`/`.qweight_a8`
        //    的 flat 序是 marlin B 的 **tile 排列**,分片的 marlin B **不是**全量的
        //    子矩形(换序也不成)⇒ 分片档必须走 `device_repack`(源侧取原始
        //    packed 矩形,上卡后由 `owl_ct_repack` 出本 rank 的 marlin B)。
        //    响亮拒,不静默出错值。
        if ctx.shard > 1 && (w.key.ends_with(".qweight") || w.key.ends_with(".qweight_a8")) {
            return Err(ModelError::Msg(format!(
                "build_shard_pass: {} 是 CPU marlin 物化键,不可矩形切(shard={})—— \
                 分片档请开 device_repack(源侧取 packed_raw 矩形 + 设备重排)",
                w.key, ctx.shard
            )));
        }
        // ① 分片形状 → 源全量形状(段长同为分片口径 ⇒ 每段 × shard)
        let full_shape = inflate(&w.shape, rule, plan, shard_segs, axis).map_err(ModelError::Msg)?;
        let mut full_segs: Vec<usize> = shard_segs.iter().map(|s| s * plan.world()).collect();
        // ①b **段长单位换算**(层侧逻辑单位 → 键自身维单位;唯一实现见
        //    `plan::span::segments_in_key_units`;2026-10-17 定谳)
        if matches!(rule, WeightRule::SegmentedCol(_)) {
            let dim = if axis == WantAxis::KMajor { 1 } else { 0 };
            full_segs = segments_in_key_units(
                &full_segs,
                full_shape[dim],
                &w.key,
                dim,
            )
            .map_err(ModelError::Msg)?;
        }
        // ② 全量形状 → 读计划 + 期望分片形状
        let (span, expect_shard) = plan_span(rule, &full_shape, plan, &full_segs, axis)
            .map_err(ModelError::Msg)?
            .ok_or_else(|| {
                ModelError::Msg(format!(
                    "build_shard_pass: {}({rule:?})在 shard>1 下应被切分,却得到 Replicate 计划",
                    w.key
                ))
            })?;
        // ③ **约定一致性硬门**(层侧 spec 切分 == 装载侧规则切分)
        if expect_shard != w.shape {
            return Err(ModelError::Msg(format!(
                "build_shard_pass: {} 的分片形状不一致 —— 层侧(spec)给 {:?},                 装载侧(规则表+plan)算得 {:?}(切分约定漂移;拒绝)",
                w.key, w.shape, expect_shard
            )));
        }
        let prod = |s: &[usize]| -> usize { s[1..].iter().product::<usize>().max(1) };
        let (src_rw, tgt_rw) = (prod(&full_shape), prod(&w.shape));
        let src_elems: usize = full_shape.iter().product();
        let tgt_elems: usize = w.shape.iter().product();

        // 布局同步(**2026-10-17 定谳修正**):`DeviceRearrange{rows,cols}` 由**层**
        // 按**分片 spec** 声明(`rows = out_shard`、`cols = k_shard/8`)—— 与
        // `w.shape` **同尺**。故分片 pass **不得再切一次**:
        //   原实现在此又 `plan.shard_len` 了一道 ⇒ **双切** ⇒ staged 矩形
        //   (`rows*cols`)与发射 grid 都只剩**一半** ⇒ GPU 重排只写了 marlin B
        //   的前半,**后半是未初始化内存** ⇒ 27B TP2 前向爆炸(NaN)。
        //   实测证据:门 `tp_load_gate_27b::tp_repack_27b` 的对照臂 —— `shard=1`
        //   22 键全绿(不走本 pass)、`shard=2` 44 键全红;装载器自带
        //   `OWL_LOAD_VERIFY` 现场账 `rows=4096 cols=640 n=5242880`(n 对、
        //   rows 只有一半)一锤定音。
        // **不变量**(这里显式钉死):rows/cols == `w.shape`(分片形状)。
        if let Layout::DeviceRearrange { op, .. } = w.layout {
            if axis != WantAxis::Logical {
                return Err(ModelError::Msg(format!(
                    "build_shard_pass: {} 是 KMajor 序却挂了 DeviceRearrange(rows/cols 语义不明;拒绝)",
                    w.key
                )));
            }
            if w.shape.len() != 2 {
                return Err(ModelError::Msg(format!(
                    "build_shard_pass: {} 挂 DeviceRearrange 但形状非 2 维({:?};拒绝)",
                    w.key, w.shape
                )));
            }
            w.layout = Layout::DeviceRearrange { op, rows: w.shape[0], cols: w.shape[1] };
        }
        spans.insert(
            w.key.clone(),
            SpanEntry { span, src_row_width: src_rw, tgt_row_width: tgt_rw, src_elems, tgt_elems },
        );
    }
    Ok(spans)
}

/// **分片源**:把"目标坐标系的读请求"翻译成源坐标系的读序列。
///
/// 未进表的键**直通**内层源(零分支开销的常见情形)。
pub struct ShardedSource<'a, S: WeightSource + ?Sized> {
    inner: &'a S,
    spans: HashMap<String, SpanEntry>,
    /// **全键读缓存**(k 切键专治;2026-10-19 定谳):
    ///
    /// 病灶现场:草稿分片装载 `layers.0.mlp.down_proj.packed_raw`(UniformRow = k 切)
    /// 的映射是 **5120 段**(每目标行一段,源步长 2176、段长 1088);逐段调内层源时,
    /// **每段都重做一次懒物化**(`[w4a16] 物化 … @ 13.5ms`)⇒ 5120 × ~88ms =
    /// **分钟级**"挂住"假象(实测 300s 只走到 2048/5120 段;不是死锁,是病态慢)。
    ///
    /// 治法:**整键读一次**进缓存,再在内存里按映射段拷 —— 内层源只被调一次,
    /// 与单卡档同价。守卫:超 [`FULL_CACHE_CAP`] 的大键(embed/lm_head 2.5GB 级)
    /// **不进缓存**(逐段直调:慢但内存不爆;而那类键是**行切**=段数极少,无此病态)。
    full: std::sync::Mutex<Option<(String, Dtype, Vec<u8>)>>,
}

/// 全键缓存上限(字节;超出 ⇒ 退回逐段直调)
const FULL_CACHE_CAP: usize = 256 << 20;

impl<'a, S: WeightSource + ?Sized> ShardedSource<'a, S> {
    pub fn new(inner: &'a S, spans: HashMap<String, SpanEntry>) -> Self {
        Self { inner, spans, full: std::sync::Mutex::new(None) }
    }

    /// 取全键字节(缓存未命中则整读一次;`dtype` 变了要重读)
    fn full_bytes(&self, key: &str, dtype: Dtype, src_elems: usize) -> Taken<()> {
        let esz = dtype.size_bytes();
        let mut g = self
            .full
            .lock()
            .map_err(|_| ModelError::Msg("ShardedSource 全键缓存锁中毒".into()))?;
        let hit = matches!(&*g, Some((k, d, _)) if k == key && *d == dtype);
        if !hit {
            let mut buf = vec![0u8; src_elems * esz];
            let Some(()) = self
                .inner
                .convert_chunk_into_bytes(key, 0, src_elems, &mut buf, dtype)?
            else {
                return Ok(None);
            };
            *g = Some((key.to_string(), dtype, buf));
        }
        Ok(Some(()))
    }

    /// 目标 `[off, len)` → 源读序列(`None` = 越界/不在表)
    fn map(&self, key: &str, off: usize, len: usize) -> Option<Vec<(usize, usize)>> {
        let e = self.spans.get(key)?;
        e.span.map(off, len, e.tgt_row_width, e.src_row_width).ok()
    }
}

impl<S: WeightSource + ?Sized> WeightSource for ShardedSource<'_, S> {
    fn take(&self, key: &str) -> Taken<Vec<f32>> {
        match self.spans.get(key) {
            None => self.inner.take(key),
            // ⚠️ 不能"整取再切"(那会把全量读进主机 —— 违背分片装载本意)
            Some(e) => self.take_range(key, 0, e.tgt_elems),
        }
    }

    fn take_range(&self, key: &str, offset_elems: usize, len: usize) -> Taken<Vec<f32>> {
        if !self.spans.contains_key(key) {
            return self.inner.take_range(key, offset_elems, len);
        }
        let Some(seq) = self.map(key, offset_elems, len) else {
            return Ok(None); // 越界/不在表(与内层 take_range 同语义:Ok(None) = 缺)
        };
        let mut out = Vec::with_capacity(len);
        for (src_off, n) in seq {
            let Some(chunk) = self.inner.take_range(key, src_off, n)? else {
                return Ok(None);
            };
            out.extend(chunk);
        }
        Ok(Some(out))
    }

    fn elem_len(&self, key: &str) -> Taken<usize> {
        match self.spans.get(key) {
            Some(e) => Ok(Some(e.tgt_elems)),
            None => self.inner.elem_len(key),
        }
    }

    fn native_dtype(&self, key: &str) -> Taken<Dtype> {
        self.inner.native_dtype(key)
    }

    fn copy_native_into(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
    ) -> Taken<()> {
        if !self.spans.contains_key(key) {
            return self.inner.copy_native_into(key, offset_elems, len, dst);
        }
        // 原始字节直拷:元素宽 = **源**的 native 宽(与内层语义一致)。
        // 内层无原始通道(`Ok(None)`)= 本层也回退(调用方转 convert 路径)。
        let Some(src_dt) = self.inner.native_dtype(key)? else {
            return Ok(None);
        };
        let esz = src_dt.size_bytes();
        let Some(seq) = self.map(key, offset_elems, len) else {
            return Ok(None);
        };
        let mut tgt_elem = 0usize;
        for (src_off, n) in seq {
            let b0 = tgt_elem * esz;
            let b1 = b0 + n * esz;
            if b1 > dst.len() {
                return Ok(None);
            }
            self.inner.copy_native_into(key, src_off, n, &mut dst[b0..b1])?;
            tgt_elem += n;
        }
        Ok(Some(()))
    }

    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Taken<()> {
        if !self.spans.contains_key(key) {
            return self
                .inner
                .convert_chunk_into_bytes(key, offset_elems, len, dst, dtype);
        }
        // 目标字节布局:元素宽 = **目标 dtype**
        let esz = dtype.size_bytes();
        let Some(seq) = self.map(key, offset_elems, len) else {
            return Ok(None);
        };
        // **k 切键专治**(见 `full` 字段注):整键读一次 ⇒ 段数再多也只调内层源一次
        let ent = &self.spans[key];
        if ent.src_elems * esz <= FULL_CACHE_CAP {
            self.full_bytes(key, dtype, ent.src_elems)?;
            let g = self
                .full
                .lock()
                .map_err(|_| ModelError::Msg("ShardedSource 全键缓存锁中毒".into()))?;
            let buf = &g.as_ref().expect("刚填过").2;
            let mut tgt_elem = 0usize;
            for (src_off, n) in seq {
                let b0 = tgt_elem * esz;
                let b1 = b0 + n * esz;
                let s0 = src_off * esz;
                let s1 = s0 + n * esz;
                if b1 > dst.len() || s1 > buf.len() {
                    return Ok(None);
                }
                dst[b0..b1].copy_from_slice(&buf[s0..s1]);
                tgt_elem += n;
            }
            return Ok(Some(()));
        }
        let mut tgt_elem = 0usize;
        let n_seg = seq.len();
        owl_shared::log_debug!(
            "load-rearrange",
            "{key}: 映射 off={offset_elems} len={len} ⇒ {n_seg} 段(目标行宽 {} / 源行宽 {})",
            self.spans[key].tgt_row_width,
            self.spans[key].src_row_width
        );
        for (si, (src_off, n)) in seq.into_iter().enumerate() {
            let b0 = tgt_elem * esz;
            let b1 = b0 + n * esz;
            if b1 > dst.len() {
                return Ok(None);
            }
            // 进度账:段数可能上千(k 切键 = 每目标行一段)⇒ 每 512 段打一行,
            // 用来区分"**慢**"与"**卡**"(判据:同一行是否推进)
            if si % 512 == 0 {
                owl_shared::log_debug!(
                    "load-rearrange",
                    "{key}: 段 {si}/{n_seg} src_off={src_off} n={n}"
                );
            }
            self.inner
                .convert_chunk_into_bytes(key, src_off, n, &mut dst[b0..b1], dtype)?;
            tgt_elem += n;
        }
        owl_shared::log_debug!("load-rearrange", "{key}: {n_seg} 段读完");
        Ok(Some(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Want;
    use crate::module::Taken;
    use crate::plan::rules::rule_of; // 键表默认对照（声明规则覆盖的判据面）
use crate::plan::rules::{SegKind, WeightRule};

    /// 内存源(全量 f32;`Vec` 值域即"源坐标系")
    struct MemSource {
        data: HashMap<String, Vec<f32>>,
    }

    /// 测试小助手:`Taken` 两层展开(读取错 = panic;缺键 = panic)
    fn t<T>(r: Taken<T>) -> T {
        r.expect("读取失败").expect("键不在源")
    }

    impl WeightSource for MemSource {
        fn take(&self, key: &str) -> Taken<Vec<f32>> {
            Ok(self.data.get(key).cloned())
        }
        fn take_range(&self, key: &str, off: usize, len: usize) -> Taken<Vec<f32>> {
            Ok(self.data.get(key).and_then(|v| v.get(off..off + len)).map(<[f32]>::to_vec))
        }
        fn elem_len(&self, key: &str) -> Taken<usize> {
            Ok(self.data.get(key).map(Vec::len))
        }
        fn native_dtype(&self, _key: &str) -> Taken<Dtype> {
            Ok(Some(Dtype::F32))
        }
    }

    fn want(key: &str, shape: Vec<usize>) -> Want {
        Want {
            key: key.to_string(),
            dtype: Dtype::F32,
            shape,
            layout: Layout::Direct,
            segments: None,
            rule: None,
            sink: crate::module::Sink::for_test(),
        }
    }

    /// 逐元素对拍:分片源读出的全部元素 == "全量读后按 span 取"。
    ///
    /// 模拟**真实用法**:层侧已按分片 spec 建 Want(shape = **分片形状**、
    /// segments = **分片段长**)⇒ pass 反推全量形状并强校验一致性。
    fn assert_bytes_equal(
        key: &str,
        full_shape: Vec<usize>,
        full_segs: Option<Vec<usize>>,
        full: &[f32],
        slice_id: usize,
        shard: usize,
    ) {
        let rule = rule_of(key);
        // 层侧(分片 spec)会建出的形状/段长
        let shard_shape: Vec<usize> = {
            let mut s = full_shape.clone();
            match rule {
                WeightRule::UniformCol | WeightRule::PerHead => s[0] /= shard,
                WeightRule::UniformRow => s[1] /= shard,
                WeightRule::SegmentedCol(_) => {
                    s[0] = full_segs.as_ref().expect("融合权重需段长").iter().sum::<usize>() / shard
                }
                WeightRule::Replicate => {}
            }
            s
        };
        let shard_segs: Option<Vec<usize>> =
            full_segs.as_ref().map(|v| v.iter().map(|s| s / shard).collect());
        let mut w = want(key, shard_shape.clone());
        w.segments = shard_segs;
        let mut ops = LoaderOps::want(w);
        let mut src = MemSource { data: HashMap::new() };
        src.data.insert(key.to_string(), full.to_vec());
        let ctx = LoaderCtx { dtype: Dtype::F32, shard, slice_id, ..Default::default() };
        let spans = build_shard_pass(&mut ops, &ctx).expect("pass");
        let sharded = ShardedSource::new(&src, spans);
        // 分片后的目标:逐元素读回(经 take_range 分块,验证跨行/跨段拼接)
        let n = t(sharded.elem_len(key));
        let mut got = Vec::with_capacity(n);
        for chunk in 0..n.div_ceil(7) {
            let off = chunk * 7;
            let len = 7.min(n - off);
            got.extend(t(sharded.take_range(key, off, len)));
        }
        assert_eq!(got.len(), n, "{key}: 读回元素数");
        // 参考:按规则"先全量后切"(plan/tp 的切分原语)
        let plan = TpPlan::new(slice_id, shard).unwrap();
        let want_ref: Vec<f32> = match rule {
            WeightRule::UniformCol | WeightRule::PerHead => {
                let r = plan.shard_range(full_shape[0]).unwrap();
                crate::plan::tp::take_rows_f32(full, full_shape[1], r).unwrap()
            }
            WeightRule::UniformRow => {
                let r = plan.shard_range(full_shape[1]).unwrap();
                crate::plan::tp::take_cols_f32(full, full_shape[1], r).unwrap()
            }
            WeightRule::SegmentedCol(_) => {
                let segs = full_segs.expect("分段切必须给段长");
                let ranges = plan.segmented_ranges(&segs).unwrap();
                let mut rows: Vec<usize> = Vec::new();
                let mut base = 0usize;
                for (i, r) in ranges.iter().enumerate() {
                    for j in r.clone() {
                        rows.push(base + j);
                    }
                    base += segs[i];
                }
                let in_w = full_shape[1];
                let mut out = Vec::with_capacity(rows.len() * in_w);
                for r in rows {
                    out.extend_from_slice(&full[r * in_w..(r + 1) * in_w]);
                }
                out
            }
            WeightRule::Replicate => full.to_vec(),
        };
        assert_eq!(got, want_ref, "{key}(slice {slice_id}/{shard}): 分片读 ≠ 先全量后切");
    }

    /// **P1.5-a 派生键 × 层声明规则**:`lm_head.i8q`/`.i8s` 的切分完全靠
    /// `Want.rule = UniformCol`(键表对 `lm_head` 的默认是 `Replicate`)。
    ///
    /// 回归判例(2026-10-20,TP2 实测):①派生后缀表漏 `.i8q/.i8s` ②层侧漏
    /// `.with_rule` ⇒ 两 rank 读同一段 ⇒ concat 后 top-2 是 `x` 与 `x+vocab/2`
    /// **同值**(高半 = 低半的复制,看起来"头没坏",实则词表后半全错)。
    /// 本门在 CPU 上锁死"两 rank 读不同行段、且各等于该 rank 的行区间"。
    #[test]
    fn derived_i8_head_rows_split_per_rank() {
        let (vocab, k) = (8usize, 4usize); // 词表 8 ⇒ 每 rank 4 行
        let q_full: Vec<f32> = (0..vocab * k).map(|i| i as f32).collect();
        let s_full: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.25).collect();
        for (key, full_shape, fullv, row_w) in [
            ("lm_head.i8q", vec![vocab, k], q_full, k),
            ("lm_head.i8s", vec![vocab], s_full, 1),
        ] {
            let mut seen: Vec<Vec<f32>> = Vec::new();
            for slice_id in 0..2usize {
                let mut shard_shape = full_shape.clone();
                shard_shape[0] /= 2;
                let mut w = want(key, shard_shape);
                // 层侧覆盖(生产唯一入口 = `Embedding::layout_lm_head`)
                w.rule = Some(WeightRule::UniformCol);
                let mut ops = LoaderOps::want(w);
                let ctx =
                    LoaderCtx { dtype: Dtype::F32, shard: 2, slice_id, ..Default::default() };
                let spans = build_shard_pass(&mut ops, &ctx).expect("pass");
                let mut src = MemSource { data: HashMap::new() };
                src.data.insert(key.to_string(), fullv.clone());
                let sharded = ShardedSource::new(&src, spans);
                let n = t(sharded.elem_len(key));
                let got = t(sharded.take_range(key, 0, n));
                let rows = vocab / 2;
                let wantv: Vec<f32> =
                    fullv[slice_id * rows * row_w..(slice_id + 1) * rows * row_w].to_vec();
                assert_eq!(got, wantv, "{key} slice{slice_id}: 分片读 ≠ 本 rank 的行区间");
                seen.push(got);
            }
            assert_ne!(seen[0], seen[1], "{key}: 两 rank 读到同一段(高半 = 低半复制)");
        }
        // 剥后缀表必须认识这两件(否则 `rule_of` 回落 Replicate ⇒ 上面的覆盖失效)
        assert_eq!(crate::plan::rules::base_key("lm_head.i8q"), "lm_head");
        assert_eq!(crate::plan::rules::base_key("lm_head.i8s"), "lm_head");
    }

    /// 行切(out)逐元素对拍
    #[test]
    fn rows_shard_matches_full_then_slice() {
        let shape = vec![8usize, 6];
        let full: Vec<f32> = (0..48).map(|i| i as f32 * 0.5).collect();
        for slice in 0..2 {
            assert_bytes_equal("model.layers.0.mlp.gate_proj.weight", shape.clone(), None, &full, slice, 2);
        }
    }

    /// 列切(in;stride 读)逐元素对拍
    #[test]
    fn cols_shard_matches_full_then_slice() {
        let shape = vec![4usize, 8];
        let full: Vec<f32> = (0..32).map(|i| i as f32).collect();
        for slice in 0..2 {
            assert_bytes_equal("model.layers.0.mlp.down_proj.weight", shape.clone(), None, &full, slice, 2);
        }
    }

    /// 分段切(融合权重)逐元素对拍
    #[test]
    fn segmented_shard_matches_full_then_slice() {
        // 8 行 = 段 4 + 段 4;in = 5
        let shape = vec![8usize, 5];
        let full: Vec<f32> = (0..40).map(|i| i as f32 + 1.0).collect();
        let segs = Some(vec![4usize, 4]);
        for slice in 0..2 {
            assert_bytes_equal(
                "in_proj_qkvz.weight",
                shape.clone(),
                segs.clone(),
                &full,
                slice,
                2,
            );
        }
    }

    /// Replicate 键不受影响(直通)
    #[test]
    fn replicate_passes_through() {
        let key = "model.layers.0.input_layernorm.weight";
        let mut ops = LoaderOps::want(want(key, vec![8]));
        let src = MemSource { data: HashMap::from([(key.to_string(), (0..8).map(|i| i as f32).collect())]) };
        let ctx = LoaderCtx { dtype: Dtype::F32, shard: 2, slice_id: 1, ..Default::default() };
        let spans = build_shard_pass(&mut ops, &ctx).unwrap();
        assert!(spans.is_empty(), "Replicate 不进表");
        let sharded = ShardedSource::new(&src, spans);
        assert_eq!(t(sharded.take(key)), (0..8).map(|i| i as f32).collect::<Vec<_>>());
        assert_eq!(t(sharded.elem_len(key)), 8, "全量元素数不变");
    }

    /// shard = 1 ⇒ 零开销(空表;源直通)
    #[test]
    fn single_slice_is_free() {
        let mut ops = LoaderOps::want(want("model.layers.0.mlp.down_proj.weight", vec![4, 8]));
        let ctx = LoaderCtx { dtype: Dtype::F32, shard: 1, slice_id: 0, ..Default::default() };
        let spans = build_shard_pass(&mut ops, &ctx).unwrap();
        assert!(spans.is_empty());
    }

    /// **层声明的规则覆盖**(P1.1,2026-10-19):`lm_head.weight` 键表默认
    /// `Replicate`,但词表切档的 want 自带 `UniformCol` ⇒ 分片 pass 按声明切。
    /// 同键**无声明** ⇒ 直通(未切档零行为变化)。
    #[test]
    fn declared_rule_overrides_keytable() {
        let key = "lm_head.weight";
        assert_eq!(rule_of(key), WeightRule::Replicate, "键表默认 = Replicate");
        let full: Vec<f32> = (0..40).map(|i| i as f32 + 1.0).collect(); // [8, 5]
        // 对照臂:无声明 ⇒ 直通
        {
            let mut ops = LoaderOps::want(want(key, vec![8, 5]));
            let ctx = LoaderCtx { dtype: Dtype::F32, shard: 2, slice_id: 0, ..Default::default() };
            let spans = build_shard_pass(&mut ops, &ctx).unwrap();
            assert!(spans.is_empty(), "无声明 + 键表 Replicate ⇒ 不进表");
        }
        // 覆盖臂:声明 UniformCol ⇒ 按 out 维切(rank0 = 低半,rank1 = 高半)
        for slice in 0..2usize {
            let mut ops = LoaderOps::want(want(key, vec![4, 5])).with_rule(WeightRule::UniformCol);
            let mut src = MemSource { data: HashMap::new() };
            src.data.insert(key.to_string(), full.clone());
            let ctx = LoaderCtx { dtype: Dtype::F32, shard: 2, slice_id: slice, ..Default::default() };
            let spans = build_shard_pass(&mut ops, &ctx).expect("声明规则档应切");
            let sharded = ShardedSource::new(&src, spans);
            let n = t(sharded.elem_len(key));
            let mut got = Vec::with_capacity(n);
            for chunk in 0..n.div_ceil(7) {
                let off = chunk * 7;
                let len = 7.min(n - off);
                got.extend(t(sharded.take_range(key, off, len)));
            }
            let base = slice * 4 * 5;
            assert_eq!(got, full[base..base + 20].to_vec(), "lm_head 分片读(slice {slice})");
        }
    }
}
