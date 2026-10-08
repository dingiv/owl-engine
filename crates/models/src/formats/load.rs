//! 装载域解释器(与计算域 [`crate::interpreters::eval`] 对偶,互不依赖;
//! 2026-09-30 自 interpreters/ 迁入 formats/ —— 格式源与装载执行同域)。
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
//! 性能定谳(2026-09-26,F5-4):转换走 [`crate::f16c`](F16C 单 pass)+
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
/// eval_load(&model, face, &src, &LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false, verify: false, debug_tap: false }).await?;
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
    let tap = if ctx.debug_tap { Some(std::sync::Mutex::new(Box::new(StderrTap) as Box<dyn LoadTap>)) } else { None }; // 观测:ctx.debug_tap 门控,缺省零开销
    // 设备重排的原始块延迟回收清单(E2a 契约:回收前数据须已收割 ——
    // repack 核异步读 raw,火后即 free 会踩;栅栏后统一回收)
    let deferred: std::sync::Arc<Mutex<Vec<u64>>> = std::sync::Arc::default();
    let manifest = eval_want(&want, face, src, tap.as_ref(), &deferred, ctx.verify).await?;
    // 上载栅栏:全部 DMA 落定后方可进入计算域(upload_pinned 入队即回执,
    // 完成语义由本栅栏一次总代价兜底)
    face.sync().await?;
    // 装载校验(OWL_LOAD_VERIFY 门控):整块回读 vs staged 校验和
    verify_loaded(&manifest, face, ctx.verify).await?;
    let ids = std::mem::take(&mut *deferred.lock().unwrap());
    if !ids.is_empty() {
        face.free(&ids).await?;
    }
    Ok(manifest)
}

// ============================================================================
// §2 调度:键分组 + LPT 分桶 + 顺序/并发两路径
// ============================================================================

/// 需求清单求值:face 具备并发句柄且 Want 多于一条 → 分桶流水;否则顺序。
type Deferred = std::sync::Arc<Mutex<Vec<u64>>>;

async fn eval_want<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
    deferred: &Deferred,
    verify: bool,
) -> Result<LoadManifest, ModelError> {
    let handles = face.loader_faces(LOAD_WORKERS);
    let manifest = match handles {
        Some(handles) if handles.len() > 1 && want.wants().len() > 1 => {
            eval_want_parallel(want, handles, src, tap, deferred, verify).await?
        }
        _ => eval_want_sequential(want, face, src, tap, deferred, verify).await?,
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
const DEFERRED_FREE_COUNT: usize = 16; // raw 滞留 16 块批量回收一次(~1-2GB)

async fn eval_want_sequential<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
    deferred: &Deferred,
    verify: bool,
) -> Result<LoadManifest, ModelError> {
    let mut manifest = LoadManifest::default();
    for group in key_groups(want.wants()) {
        manifest.0.extend(load_group(face, &group, src, tap, deferred, verify).await?);
        // 批量回收:滞留 raw 超阈值 → sync(COMPUTE 排空,E2a 契约)+ free
        if deferred.lock().unwrap().len() >= DEFERRED_FREE_COUNT {
            face.sync().await?;
            let ids = std::mem::take(&mut *deferred.lock().unwrap());
            if !ids.is_empty() {
                face.free(&ids).await?;
            }
        }
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
    deferred: &Deferred,
    verify: bool,
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
            entries.extend(load_group(&mut face, bundle, src, tap, deferred, verify).await?);
            // 批量回收(并行桶;sync 任一句柄 = 全设备排空)
            if deferred.lock().unwrap().len() >= DEFERRED_FREE_COUNT {
                face.sync().await?;
                let ids = std::mem::take(&mut *deferred.lock().unwrap());
                if !ids.is_empty() {
                    face.free(&ids).await?;
                }
            }
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
    deferred: &Deferred,
    verify: bool,
) -> Result<Vec<LoadEntry>, ModelError> {
    let mut manifest = Vec::new();
    for w in group {
        let n: usize = w.shape.iter().product();
        let entry = match w.layout {
            Layout::Transposed => load_transposed(face, w, src, n, tap, verify).await?,
            Layout::Direct => load_direct(face, w, src, n, tap, verify).await?,
            Layout::DeviceRearrange { op, rows, cols } => {
                load_device_rearrange(face, w, src, n, op, rows, cols, tap, deferred, verify).await?
            }
        };
        w.sink.deliver(crate::contract::Bytes::new(entry.block.id, n));
        manifest.push(entry);
    }
    Ok(manifest)
}

// ============================================================================
// §3.1 装载校验(E5-DF4;OWL_LOAD_VERIFY 门控):staged FNV-1a 64 →
// 装载后整块回读对比。抓:DMA 半成品/池污染/装载后覆写;metrics 标签
// `loadv.{key}` = 校验和(双 boot 对比定位被写坏的具体键)。
// ============================================================================

/// FNV-1a 64(字批量混入)
fn fnv1a(data: &[u8], init: u64) -> u64 {
    let mut h = init;
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h = (h ^ w).wrapping_mul(0x100000001b3);
    }
    for b in chunks.remainder() {
        h = (h ^ (*b as u64)).wrapping_mul(0x100000001b3);
    }
    h
}

/// 装载后回读校验(eval_load 栅栏后调用;OWL_LOAD_VERIFY 门控):
/// 每键整块 dtoh 重哈希对比 staged 校验和;metrics 标签 `loadv.{key}`;
/// 失配 = 结构化错误(列出键)。DeviceRearrange 臂(csum=0)跳读。
pub async fn verify_loaded<D: DeviceClient>(
    manifest: &LoadManifest,
    face: &mut D,
    verify: bool,
) -> Result<(), ModelError> {
    if !verify {
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for e in manifest.entries() {
        if e.csum == 0 {
            continue; // 重排臂/未校验
        }
        let n: usize = e.shape.iter().product();
        let esz = e.dtype.size_bytes();
        let total = n * esz;
        let mut h = 0xcbf29ce484222325u64;
        // dtoh 整块契约:缓冲必须等大(分块回读会被拒)
        let mut buf = vec![0u8; total];
        face.dtoh(&e.block, &mut buf)
            .await
            .map_err(|err| ModelError::Msg(format!("verify '{}': {err}", e.key)))?;
        h = fnv1a(&buf, h);
        owl_shared::metrics::with_metrics_store(|s| {
            s.counter_add(&format!("loadv.{}", e.key), h & 0x7fff_ffff_ffff_ffff, file!(), line!());
        });
        if h != e.csum {
            bad.push(e.key.clone());
        }
        checked += 1;
    }
    let skipped = manifest.entries().len() - checked;
    eprintln!(
        "[load-verify] checked={checked} skipped(rearrange)={skipped} mismatch={} {:.2}s",
        bad.len(),
        t0.elapsed().as_secs_f64()
    );
    if !bad.is_empty() {
        return Err(ModelError::Msg(format!(
            "装载校验失配 {} 键:{}",
            bad.len(),
            bad.iter().take(8).cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    Ok(())
}

/// 直接臂(主路径):设备块 → [pinned 租约 → 源直写字节 → DMA] × N 块。
/// 大张量自动多块,每 4 块排空一次(租约回池,码头零 churn)。
async fn load_direct<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    w: &crate::module::Want,
    src: &S,
    n: usize,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
    verify: bool,
) -> Result<LoadEntry, ModelError> {
    let esz = w.dtype.size_bytes();
    let mut probe = KeyProbe::start(tap, &w.key, n * esz);
    let b = face
        .alloc(w.dtype, n)
        .await
        .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
    let chunk_elems = (CHUNK_BYTES / esz).max(1);
    let mut off = 0usize;
    
    let mut csum = 0xcbf29ce484222325u64;
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
        if verify {
            csum = fnv1a(&lease.slice_bytes_mut()[..len * esz], csum);
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
        csum: if verify { csum } else { 0 },
    })
}

/// 设备重排臂(2026-10-01 AWQ 装载提速):源供**原始 packed** 字节
/// (CPU 零重排)→ DMA 上卡 → registry 重排核就地生成 marlin 布局块
/// → 原始块回收(Free 通道)。CPU 33s 重量排出热路径,重排在 GPU
/// (显存带宽 ~900GB/s,哑 gather)。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
async fn load_device_rearrange<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    w: &crate::module::Want,
    src: &S,
    n: usize,
    op: owl_kernels::driver::OpId,
    rows: usize,
    cols: usize,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
    deferred: &Deferred,
    verify: bool,
) -> Result<LoadEntry, ModelError> {
    let raw_elems = rows * cols;
    let mut probe = KeyProbe::start(tap, &w.key, raw_elems * 4);
    let mut packed_host = if verify {
        Some(vec![0u8; raw_elems * 4])
    } else {
        None
    };

    // 1) 原始块上卡(经 pinned;raw > 128MB 自动分块循环)
    let raw = face
        .alloc_uninit(crate::contract::Dtype::U32, raw_elems)
        .await
        .map_err(|e| ModelError::Msg(format!("Weight '{}': raw alloc {e}", w.key)))?;
    {
        let chunk_elems = (CHUNK_BYTES / 4).max(1);
        let mut off = 0usize;
        while off < raw_elems {
            let len = chunk_elems.min(raw_elems - off);
            let mut lease = {
                let _s = probe.span("pinned_alloc");
                face.alloc_pinned(len * 4)
                    .await
                    .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?
            };
            {
                let _s = probe.span("convert");
                src.convert_chunk_into_bytes(w.key.as_str(), off, len, lease.slice_bytes_mut(), crate::contract::Dtype::U32)
                    .ok_or_else(|| attribution(src, &w.key, raw_elems))?;
            }
            if let Some(ph) = &mut packed_host {
                ph[off * 4..(off + len) * 4].copy_from_slice(&lease.slice_bytes_mut()[..len * 4]);
            }
            {
                let _s = probe.span("htod");
                face.upload_pinned(lease, &raw, off * 4, len * 4)
                    .await
                    .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
            }
            off += len;
            // upload_pinned 异步语义契约:计算读前必 sync(load_direct 同款
            // 纪律:每 4 块 + 尾;repack 核读 raw 前无栅栏 = 竞态读半截)
            if off % (chunk_elems * 4) == 0 || off >= raw_elems {
                face.sync().await?;
            }
        }
    }
    // 2) 重排核(语义拾取:driver::resolve(OpEnv 必传,被动律);
    //    grid/block 契约公式住 driver(load::ct_repack))
    let out_block = face
        .alloc_uninit(crate::contract::Dtype::U32, n)
        .await
        .map_err(|e| ModelError::Msg(format!("Weight '{}': out alloc {e}", w.key)))?;
    // OWL_LOAD_VERIFY=1:逐层端到端数据校验(核输出 vs CPU fused 逐位;
    // host 留 packed 副本 + dtoh 对拍 —— 装载慢,仅排障用)
    {
        let _s = probe.span("repack");
        let env = face.op_env().ok_or_else(|| {
            ModelError::Msg(format!("Weight '{}': DeviceRearrange 需 GPU 环境", w.key))
        })?;
        let pick = owl_kernels::driver::resolve(owl_kernels::driver::OpReq {
            op,
            env: &env,
            dt: owl_kernels::driver::DType::U32,
            shapes: &[vec![rows, cols]],
            aux: &[],
            scalars: &[],
        });
        face.launch(crate::contract::LaunchMsg {
            kernel: crate::contract::KernelSpec {
                name: pick.name.to_string(),
                source: crate::kernel::source(pick.name).to_string(),
            },
            args: vec![
                crate::contract::Arg::Block { id: raw.id },
                crate::contract::Arg::U64(rows as u64),
                crate::contract::Arg::U64(cols as u64),
                crate::contract::Arg::Block { id: out_block.id },
            ],
            grid: pick.shape.grid,
            block: pick.shape.block,
            shared_mem: pick.shape.smem,
            out_elems: n,
        })
        .await
        .map_err(|e| ModelError::Msg(format!("Weight '{}': repack launch {e}", w.key)))?;
        // 端到端校验(OWL_LOAD_VERIFY;前 4 层):GPU 重排输出 vs CPU fused 逐位
        if let Some(ph) = &packed_host {
            static VERIFY_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let vi = VERIFY_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if vi < 9999 {
                face.sync().await?;
                let mut gbuf = vec![0u8; n * 4];
                face.dtoh(&out_block, &mut gbuf).await?;
                let k = cols * 8;
                let idx = owl_kernels::family::marlin::repack::marlin_fused_indices(k, rows);
                let p32: Vec<i32> = ph
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let mut b_ref: Vec<i32> = Vec::new();
                owl_kernels::family::marlin::repack::pack_marlin_b_fused(&p32, &idx, rows * k / 8, &mut b_ref);
                let mut bad = 0usize;
                for (i, w) in gbuf.chunks_exact(4).enumerate() {
                    let got = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                    if got != b_ref[i] { bad += 1; }
                }
                eprintln!("[load-verify] {vi} {} 坏字 {bad}/{}", w.key, b_ref.len());
                assert_eq!(bad, 0, "GPU repack 端到端校验失败: {}", w.key);
            }
        }
    }
    // 3) 原始块回收(E2a Free 通道;数据已进重排输出块)
    face.free(&[raw.id]).await;
    probe.done();
    Ok(LoadEntry {
        key: w.key.clone(),
        dtype: w.dtype,
        shape: w.shape.clone(),
        layout: Layout::DeviceRearrange { op, rows, cols },
        block: crate::contract::Bytes::new(out_block.id, n),
        csum: 0, // staged = raw 而块 = 重排后:回读不可比,深校验挂账
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
    verify: bool,
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
        csum: if verify { fnv1a(&bytes, 0xcbf29ce484222325) } else { 0 },
    })
}

// ============================================================================
// §4 host 变换(转置臂专用;主路径转换在源侧 crate::f16c 直写租约)
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
    crate::f16c::f32_slice_to_f16_bytes(v, &mut dst);
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
        owl_shared::metrics::with_metrics_store(|s| {
            s.timer_record_tag("load.key", self.rec.total, file!(), line!())
        });
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
        // metrics 直用 API(非宏;装载一次性路径,release 恒开 —— 分相
        // 瓶颈数据面,tag = load.{phase};开销 ~µs/段,对 134s 量级无感)
        owl_shared::metrics::with_metrics_store(|s| {
            s.timer_record_tag(&format!("load.{}", self.name), d, file!(), line!())
        });
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
