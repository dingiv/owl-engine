//! 装载域解释器(与计算域 [`super::eval`] 对偶,互不依赖)。
//!
//! 三件套同构 eval 域:
//! - **声明**:层侧 `Loadable::layout(ctx)` 产出 `LoaderOps`(Want 清单,
//!   各带 sink 回填口;层零执行);
//! - **执行**:本解释器按 Want 取数 → 布局变换 → 物化进设备块 → sink 回填
//!   (直接/转置两臂,见 §3);
//! - **观测**:[`LoadTap`](装载域 Tap 律:只观测,零执行权 —— 键边界回调,
//!   结构上不可能改变装载;OWL_LOAD_DEBUG 挂 StderrTap,见 §5)。
//!
//! 数据单副本流动律:take_range / convert_chunk_into_bytes 查到才实体化,
//! pinned 租约 move 进消息,DMA 直读,用毕即弃(主机驻留 = 在途份)。
//!
//! 性能定谳(2026-09-26,F5-4):转换走 [`owl_f16c`](F16C 单 pass)+
//! 微 crate profile 覆盖后,装载已 **DMA-bound**(0.8B 模型 1.75GB ≈
//! 0.72s;server 全程仅 ~355ms,详见 roadmap.local/f5-switch-workorder.md)。

use crate::contract::{Dtype, DeviceClient, ModelError};
use crate::module::{Layout, LoadEntry, LoadManifest, LoaderOps, WeightSource};
use futures_util::future::join_all;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ============================================================================
// §1 入口:eval_load(声明 → 求值 → 上载栅栏)
// ============================================================================

/// 装载并发度(桶数;每桶一束键组。GpuClient 的 faces = 同管道句柄克隆,
/// server 单线程自然串行 —— 并发面只用于客户端转换/DMA 流水重叠)。
const LOAD_WORKERS: usize = 4;

/// 装载执行(层级入口):驱动层的 `Loadable::layout` 需求并物化。
/// 解释器自己调用层钩子并为它传递 LoaderCtx —— 使用者只给层与源。
///
/// ```rust,ignore
/// eval_load(&model, face, &src, &LoaderCtx { dtype: Dtype::F16, shard: 1 }).await?;
/// ```
pub async fn eval_load<M, D, S>(
    layer: &M,
    face: &mut D,
    src: &S,
    ctx: &crate::module::LoaderCtx,
) -> Result<LoadManifest, ModelError>
where
    M: crate::module::Loadable,
    D: DeviceClient,
    S: WeightSource + ?Sized,
{
    let want = layer.layout(ctx); // 声明:层产出需求清单(ctx 引用透传)
    let tap = load_tap(); // 观测:OWL_LOAD_DEBUG 门控,缺省零开销
    let manifest = eval_want(&want, face, src, tap.as_ref()).await?;
    // 上载栅栏:全部 DMA 落定后方可进入计算域(upload_pinned 入队即回执,
    // 完成语义由本栅栏一次总代价兜底)
    face.sync().await?;
    Ok(manifest)
}

// ============================================================================
// §2 调度:键分组 + LPT 分桶 + 顺序/并发两路径
// ============================================================================

/// 需求清单求值:face 具备并发句柄且 Want 多于一条 → 分桶流水;否则顺序。
async fn eval_want<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
) -> Result<LoadManifest, ModelError> {
    let handles = face.loader_faces(LOAD_WORKERS);
    let manifest = match handles {
        Some(handles) if handles.len() > 1 && want.wants().len() > 1 => {
            eval_want_parallel(want, handles, src, tap).await?
        }
        _ => eval_want_sequential(want, face, src, tap).await?,
    };
    Ok(manifest)
}

/// 键分组:同键多 Want(tied 双槽 = w 直读 + w_t 转置读)原子成组,
/// take 一次供全组;组序 = 清单首次出现序。
fn key_groups<'a>(wants: &'a [crate::module::Want]) -> Vec<Vec<&'a crate::module::Want>> {
    let mut order: Vec<&'a str> = Vec::new();
    let mut map: std::collections::HashMap<&'a str, Vec<&'a crate::module::Want>> =
        std::collections::HashMap::new();
    for w in wants {
        if !map.contains_key(w.key.as_str()) {
            order.push(w.key.as_str());
            map.insert(w.key.as_str(), Vec::new());
        }
        map.get_mut(w.key.as_str()).unwrap().push(w);
    }
    order.into_iter().map(|k| map.remove(k).unwrap()).collect()
}

/// 顺序路径(CpuFace / 单 Want / 无并发能力)
async fn eval_want_sequential<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
) -> Result<LoadManifest, ModelError> {
    let mut manifest = LoadManifest::default();
    for group in key_groups(want.wants()) {
        manifest.0.extend(load_group(face, &group, src, tap).await?);
    }
    Ok(manifest)
}

/// 并发流水(键组按字节 LPT 装给最轻桶;join_all 协作驱动):
/// 桶 await DMA 期间,其他桶的取数/转换在同一线程推进 —— F5-4 定谳
/// 转换已非墙,协作交错即可吃满 DMA,无需 OS 线程。
/// 交付序无关(Want 各带各的 sink;分配无跨 Want 定序)。
async fn eval_want_parallel<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    handles: Vec<D>,
    src: &S,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
) -> Result<LoadManifest, ModelError> {
    let mut groups = key_groups(want.wants());
    groups.sort_by_key(|g| std::cmp::Reverse(g.iter().map(|w| want_bytes(w)).sum::<usize>()));
    let k = handles.len().min(groups.len()).max(1);
    let mut buckets: Vec<(D, Vec<Vec<&crate::module::Want>>, usize)> = handles
        .into_iter()
        .take(k)
        .map(|f| (f, Vec::new(), 0usize))
        .collect();
    for g in groups {
        let bytes: usize = g.iter().map(|w| want_bytes(w)).sum();
        let b = buckets.iter_mut().min_by_key(|(_, _, total)| *total).unwrap();
        b.1.push(g);
        b.2 += bytes;
    }
    let futs = buckets.into_iter().map(|(mut face, bundles, _)| async move {
        let mut entries = Vec::new();
        for bundle in &bundles {
            entries.extend(load_group(&mut face, bundle, src, tap).await?);
        }
        Ok::<_, ModelError>(entries)
    });
    let results = join_all(futs).await;
    let all = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(all.into_iter().flatten().collect())
}

// ============================================================================
// §3 执行臂:load_group 分派 → load_direct / load_transposed
// ============================================================================

/// 流式分块预算(≥64MB 的键自动多块,判据 = 分块循环;主机在途 = 单块)
const CHUNK_BYTES: usize = 128 << 20;

/// 单键组装载:同键全 Want 原子成组,逐键分派执行臂。
async fn load_group<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    group: &[&crate::module::Want],
    src: &S,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
) -> Result<Vec<LoadEntry>, ModelError> {
    let mut manifest = Vec::new();
    for w in group {
        let n: usize = w.shape.iter().product();
        let entry = match w.layout {
            Layout::Transposed => load_transposed(face, w, src, n, tap).await?,
            Layout::Direct => load_direct(face, w, src, n, tap).await?,
        };
        w.sink.deliver(crate::contract::Bytes::new(entry.block.id, n));
        manifest.push(entry);
    }
    Ok(manifest)
}

/// 直接臂(主路径):设备块 → [pinned 租约 → 源直写字节 → DMA] × N 块。
/// 大张量自动多块,每 4 块排空一次(租约回池,码头零 churn)。
async fn load_direct<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    w: &crate::module::Want,
    src: &S,
    n: usize,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
) -> Result<LoadEntry, ModelError> {
    let esz = w.dtype.size_bytes();
    let mut probe = KeyProbe::start(tap, &w.key, n * esz);
    let b = face
        .alloc(w.dtype, n)
        .await
        .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
    let chunk_elems = (CHUNK_BYTES / esz).max(1);
    let mut off = 0usize;
    while off < n {
        let len = chunk_elems.min(n - off);
        let mut lease = {
            let _s = probe.span("pinned_alloc");
            face.alloc_pinned(len * esz)
                .await
                .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?
        };
        {
            let _s = probe.span("convert");
            src.convert_chunk_into_bytes(&w.key, off, len, lease.slice_bytes_mut(), w.dtype)
                .ok_or_else(|| attribution(src, &w.key, n))?;
        }
        {
            let _s = probe.span("htod");
            face.upload_pinned(lease, &b, off * esz, len * esz)
                .await
                .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
        }
        off += len;
        if off % (chunk_elems * 4) == 0 || off >= n {
            face.sync().await?;
        }
    }
    probe.done();
    Ok(LoadEntry {
        key: w.key.clone(),
        dtype: w.dtype,
        shape: w.shape.clone(),
        layout: Layout::Direct,
        block: crate::contract::Bytes::new(b.id, n),
    })
}

/// 转置臂(tied 头等转置槽):整取 f32 → TILE² 转置 → 按 Want dtype 写
/// 字节 → htod 单发。转置需全键 f32 工作副本,流式化挂账(量小:仅 tied)。
async fn load_transposed<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    w: &crate::module::Want,
    src: &S,
    n: usize,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
) -> Result<LoadEntry, ModelError> {
    let mut probe = KeyProbe::start(tap, &w.key, n * w.dtype.size_bytes());
    let data = {
        let _s = probe.span("decode");
        src.take_range(&w.key, 0, n)
            .ok_or_else(|| attribution(src, &w.key, n))?
    };
    let t = {
        let _s = probe.span("transpose");
        transpose_into_vec(&data, w, n)?
    };
    let (dtype, bytes) = {
        let _s = probe.span("convert");
        match w.dtype {
            Dtype::F16 => (Dtype::F16, to_f16_bytes(&t)),
            _ => (
                Dtype::F32,
                t.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(),
            ),
        }
    };
    let b = {
        let _s = probe.span("htod");
        face.htod(dtype, &w.shape, &bytes)
            .await
            .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?
    };
    probe.done();
    Ok(LoadEntry {
        key: w.key.clone(),
        dtype: w.dtype,
        shape: w.shape.clone(),
        layout: Layout::Transposed,
        block: crate::contract::Bytes::new(b.id, n),
    })
}

// ============================================================================
// §4 host 变换(转置臂专用;主路径转换在源侧 owl-f16c 直写租约)
// ============================================================================

/// TILE² 分块转置 → 新 Vec(源 [rows, cols] → 目标 [cols, rows] f32)。
/// 大矩阵按输出行分 4 带并行(out.chunks_mut(rows) 天然不相交;小矩阵串行)。
fn transpose_into_vec(data: &[f32], w: &crate::module::Want, n: usize) -> Result<Vec<f32>, ModelError> {
    const TILE: usize = 64;
    let (cols, rows) = (w.shape[0], w.shape[1]);
    if data.len() != n {
        return Err(ModelError::Msg(format!(
            "Weight '{}': 元素 {} != 源形状 {rows}×{cols}",
            w.key,
            data.len()
        )));
    }
    let mut out = vec![0f32; n];
    const PAR_THRESHOLD: usize = 4 << 20;
    if n >= PAR_THRESHOLD {
        let threads = 4usize;
        let c_band = cols.div_ceil(threads);
        let mut band_slices: Vec<Vec<&mut [f32]>> = (0..threads).map(|_| Vec::new()).collect();
        for (ci, s) in out.chunks_mut(rows).enumerate() {
            band_slices[ci / c_band].push(s);
        }
        std::thread::scope(|scope| {
            for (t, band) in band_slices.into_iter().enumerate() {
                let c_lo = t * c_band;
                scope.spawn(move || {
                    for (ci, s) in band.into_iter().enumerate() {
                        let c = c_lo + ci;
                        for r in 0..rows {
                            s[r] = data[r * cols + c];
                        }
                    }
                });
            }
        });
        return Ok(out);
    }
    for r0 in (0..rows).step_by(TILE) {
        let r1 = (r0 + TILE).min(rows);
        for c0 in (0..cols).step_by(TILE) {
            let c1 = (c0 + TILE).min(cols);
            for r in r0..r1 {
                for c in c0..c1 {
                    out[c * rows + r] = data[r * cols + c];
                }
            }
        }
    }
    Ok(out)
}

/// f32 切片 → f16 字节(F16C 单 pass;舍入与 from_f32 一致,checksum 门可证)
fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut dst = vec![0u8; v.len() * 2];
    owl_f16c::f32_slice_to_f16_bytes(v, &mut dst);
    dst
}

// ============================================================================
// §5 观测:LoadTap(装载域 Tap;只观测,零执行权)
// ============================================================================

/// 单键装载账(键边界交付 —— 解释器保证无跨键交错,tap 无需自行归并)
pub struct KeyRecord {
    pub key: String,
    /// Want 声明字节量(elem × dtype 宽)
    pub bytes: usize,
    pub total: Duration,
    /// 分相累计(直接臂:pinned_alloc/convert/htod;转置臂:
    /// decode/transpose/convert/htod)
    pub phases: Vec<(&'static str, Duration)>,
}

/// 装载观测面。与 eval 域 Tap 同律:解释器代执行计时并在键边界回调,
/// tap 无取数/上传/分配权 —— 观测在结构上不可能改变装载。
pub trait LoadTap: Send {
    fn on_key(&mut self, _rec: &KeyRecord) {}
}

/// stderr 逐键打印(OWL_LOAD_DEBUG=1 挂载;分相排查/瓶颈定位用)
struct StderrTap;

impl LoadTap for StderrTap {
    fn on_key(&mut self, rec: &KeyRecord) {
        let detail: Vec<String> = rec
            .phases
            .iter()
            .map(|(n, d)| format!("{n}={:.1}ms", d.as_secs_f64() * 1e3))
            .collect();
        eprintln!(
            "[load] {:<46} {:>9}B total={:>7.1}ms [{}]",
            rec.key,
            rec.bytes,
            rec.total.as_secs_f64() * 1e3,
            detail.join(" ")
        );
    }
}

/// 观测装配:OWL_LOAD_DEBUG 门控;缺省 None = 零开销
fn load_tap() -> Option<Mutex<Box<dyn LoadTap>>> {
    if std::env::var_os("OWL_LOAD_DEBUG").is_some() {
        Some(Mutex::new(Box::new(StderrTap)))
    } else {
        None
    }
}

/// 单键分相账本(RAII span;恒累计 —— 成本每键数次 Duration 记账,
/// 交付时 tap 缺省即弃)
struct KeyProbe<'a> {
    tap: Option<&'a Mutex<Box<dyn LoadTap>>>,
    rec: KeyRecord,
}

impl<'a> KeyProbe<'a> {
    fn start(tap: Option<&'a Mutex<Box<dyn LoadTap>>>, key: &str, bytes: usize) -> Self {
        Self {
            tap,
            rec: KeyRecord {
                key: key.to_string(),
                bytes,
                total: Duration::ZERO,
                phases: Vec::new(),
            },
        }
    }

    /// 记一段相(span 作用域结束即入账;同相多段自动累加,如分块循环)
    fn span(&mut self, name: &'static str) -> SpanGuard<'_, 'a> {
        SpanGuard {
            probe: self,
            name,
            t0: Instant::now(),
        }
    }

    /// 键边界交付(总耗时 = 各相之和)
    fn done(mut self) {
        self.rec.total = self.rec.phases.iter().map(|(_, d)| *d).sum();
        if let Some(t) = self.tap {
            t.lock().unwrap().on_key(&self.rec);
        }
    }
}

/// span 守卫:Drop 即入账(早退错误路径也记账,键未 done 不交付)
struct SpanGuard<'p, 'a> {
    probe: &'p mut KeyProbe<'a>,
    name: &'static str,
    t0: Instant,
}

impl Drop for SpanGuard<'_, '_> {
    fn drop(&mut self) {
        let d = self.t0.elapsed();
        match self.probe.rec.phases.iter_mut().find(|(n, _)| *n == self.name) {
            Some(e) => e.1 += d,
            None => self.probe.rec.phases.push((self.name, d)),
        }
    }
}

// ============================================================================
// §6 杂项:归因 / 字节量
// ============================================================================

/// 装载取数失败归因:键不在 = 缺键;在但长度不符 = 元素数错误
fn attribution<S: WeightSource + ?Sized>(src: &S, key: &str, n: usize) -> ModelError {
    match src.elem_len(key) {
        Some(l) => ModelError::Msg(format!(
            "Weight '{key}': 元素 {l} != 期望 {n}"
        )),
        None => ModelError::Msg(format!("Weight '{key}': 数据源缺键")),
    }
}

/// Want 的目标字节量(元素数 × dtype 宽;LPT 分桶权重)
fn want_bytes(w: &crate::module::Want) -> usize {
    w.shape.iter().product::<usize>() * w.dtype.size_bytes()
}
