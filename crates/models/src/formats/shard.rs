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
use crate::module::{Layout, LoaderCtx, LoaderOps, WeightSource};
use crate::plan::rules::{rule_of, want_axis, WantAxis, WeightRule};
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
        let rule = rule_of(&w.key);
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
}

impl<'a, S: WeightSource + ?Sized> ShardedSource<'a, S> {
    pub fn new(inner: &'a S, spans: HashMap<String, SpanEntry>) -> Self {
        Self { inner, spans }
    }

    /// 目标 `[off, len)` → 源读序列(`None` = 越界/不在表)
    fn map(&self, key: &str, off: usize, len: usize) -> Option<Vec<(usize, usize)>> {
        let e = self.spans.get(key)?;
        e.span.map(off, len, e.tgt_row_width, e.src_row_width).ok()
    }
}

impl<S: WeightSource + ?Sized> WeightSource for ShardedSource<'_, S> {
    fn take(&self, key: &str) -> Option<Vec<f32>> {
        match self.spans.get(key) {
            None => self.inner.take(key),
            // ⚠️ 不能"整取再切"(那会把全量读进主机 —— 违背分片装载本意)
            Some(e) => self.take_range(key, 0, e.tgt_elems),
        }
    }

    fn take_range(&self, key: &str, offset_elems: usize, len: usize) -> Option<Vec<f32>> {
        if !self.spans.contains_key(key) {
            return self.inner.take_range(key, offset_elems, len);
        }
        let mut out = Vec::with_capacity(len);
        for (src_off, n) in self.map(key, offset_elems, len)? {
            out.extend(self.inner.take_range(key, src_off, n)?);
        }
        Some(out)
    }

    fn elem_len(&self, key: &str) -> Option<usize> {
        match self.spans.get(key) {
            Some(e) => Some(e.tgt_elems),
            None => self.inner.elem_len(key),
        }
    }

    fn native_dtype(&self, key: &str) -> Option<Dtype> {
        self.inner.native_dtype(key)
    }

    fn copy_native_into(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
    ) -> Option<()> {
        if !self.spans.contains_key(key) {
            return self.inner.copy_native_into(key, offset_elems, len, dst);
        }
        // 原始字节直拷:元素宽 = **源**的 native 宽(与内层语义一致)
        let esz = self.inner.native_dtype(key)?.size_bytes();
        let mut tgt_elem = 0usize;
        for (src_off, n) in self.map(key, offset_elems, len)? {
            let b0 = tgt_elem * esz;
            let b1 = b0 + n * esz;
            if b1 > dst.len() {
                return None;
            }
            self.inner.copy_native_into(key, src_off, n, &mut dst[b0..b1])?;
            tgt_elem += n;
        }
        Some(())
    }

    fn convert_chunk_into_bytes(
        &self,
        key: &str,
        offset_elems: usize,
        len: usize,
        dst: &mut [u8],
        dtype: Dtype,
    ) -> Option<()> {
        if !self.spans.contains_key(key) {
            return self
                .inner
                .convert_chunk_into_bytes(key, offset_elems, len, dst, dtype);
        }
        // 目标字节布局:元素宽 = **目标 dtype**
        let esz = dtype.size_bytes();
        let mut tgt_elem = 0usize;
        for (src_off, n) in self.map(key, offset_elems, len)? {
            let b0 = tgt_elem * esz;
            let b1 = b0 + n * esz;
            if b1 > dst.len() {
                return None;
            }
            self.inner
                .convert_chunk_into_bytes(key, src_off, n, &mut dst[b0..b1], dtype)?;
            tgt_elem += n;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Want;
    use crate::plan::rules::{SegKind, WeightRule};

    /// 内存源(全量 f32;`Vec` 值域即"源坐标系")
    struct MemSource {
        data: HashMap<String, Vec<f32>>,
    }

    impl WeightSource for MemSource {
        fn take(&self, key: &str) -> Option<Vec<f32>> {
            self.data.get(key).cloned()
        }
        fn take_range(&self, key: &str, off: usize, len: usize) -> Option<Vec<f32>> {
            self.data.get(key)?.get(off..off + len).map(<[f32]>::to_vec)
        }
        fn elem_len(&self, key: &str) -> Option<usize> {
            self.data.get(key).map(Vec::len)
        }
        fn native_dtype(&self, _key: &str) -> Option<Dtype> {
            Some(Dtype::F32)
        }
    }

    fn want(key: &str, shape: Vec<usize>) -> Want {
        Want {
            key: key.to_string(),
            dtype: Dtype::F32,
            shape,
            layout: Layout::Direct,
            segments: None,
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
        let n = sharded.elem_len(key).expect("elem_len");
        let mut got = Vec::with_capacity(n);
        for chunk in 0..n.div_ceil(7) {
            let off = chunk * 7;
            let len = 7.min(n - off);
            got.extend(sharded.take_range(key, off, len).expect("take_range"));
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
        assert_eq!(sharded.take(key).unwrap(), (0..8).map(|i| i as f32).collect::<Vec<_>>());
        assert_eq!(sharded.elem_len(key), Some(8), "全量元素数不变");
    }

    /// shard = 1 ⇒ 零开销(空表;源直通)
    #[test]
    fn single_slice_is_free() {
        let mut ops = LoaderOps::want(want("model.layers.0.mlp.down_proj.weight", vec![4, 8]));
        let ctx = LoaderCtx { dtype: Dtype::F32, shard: 1, slice_id: 0, ..Default::default() };
        let spans = build_shard_pass(&mut ops, &ctx).unwrap();
        assert!(spans.is_empty());
    }
}
