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
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ============================================================================
// §1 入口:eval_load(声明 → 求值 → 上载栅栏)
// ============================================================================

/// **装载旋钮**(与模型无关;解释器化设计 §3 — 调用点不再拼 LoaderCtx)
#[derive(Debug, Clone, Copy)]
pub struct LoadOpts {
    /// 目标 dtype(计算档事实;`None` = 由调用方/spec 决定,必给)
    pub dtype: crate::contract::Dtype,
    /// 设备重排(量化族;**已裁决结果**;`None` = 能力自述裁决)
    pub device_repack: Option<bool>,
    /// 装载校验(整块回读对账)
    pub verify: bool,
    /// 逐键分相探针(stderr)
    pub debug_tap: bool,
}

impl LoadOpts {
    /// 常规档:能力自述裁决 repack(生产入口调用点用它,别手拼 LoaderCtx)
    pub fn new<D: DeviceClient>(face: &D, dtype: crate::contract::Dtype) -> Self {
        Self {
            dtype,
            device_repack: Some(
                crate::module::RepackPath::resolve(None, face.device_kernels()).is_device(),
            ),
            verify: false,
            debug_tap: false,
        }
    }

    /// 非量化/固定档(无设备重排 + 零探针):裸 safetensors f16 直拷、
    /// W4A16-CPU 重排等"不需要能力裁决"的装载点。
    pub fn plain(dtype: crate::contract::Dtype) -> Self {
        Self { dtype, device_repack: Some(false), verify: false, debug_tap: false }
    }

    /// 装载 ctx(旋钮 → LoaderCtx;唯一产地)
    pub fn ctx(&self) -> crate::module::LoaderCtx {
        crate::module::LoaderCtx {
            dtype: self.dtype,
            shard: 1,
            device_repack: self.device_repack.unwrap_or(false),
            verify: self.verify,
            debug_tap: self.debug_tap,
        }
    }
}

/// **通用装载入口**(解释器化 L2 第一段):`Loadable::layout` 声明 × 起点源
/// ⇒ 物化。真正的"起点侧"通用化(按 [`CheckpointKind`] 自动开源)见
/// [`load_from_dir`];本函数是纯加法(旧具名函数仍走 `eval_load`)。
pub async fn load_component<T, D>(
    face: &mut D,
    comp: &T,
    src: &dyn WeightSource,
    opts: LoadOpts,
) -> Result<crate::module::LoadManifest, ModelError>
where
    T: crate::module::Loadable,
    D: DeviceClient,
{
    // `&dyn WeightSource`(trait 全程对象安全)⇒ 起点族可在同一条通用路上换源
    eval_load(comp, face, src, &opts.ctx()).await
}

/// 量化**方案**判据:`config.json:quantization_config.config_groups.group_0.weights`
/// 的 `(group_size, symmetric)` —— **不是 `quant_method`**(实测:cyankiwi
/// AWQ-INT4 与 llm-compressor W4A16 产物**同为** `compressed-tensors`,
/// 分族靠方案:g32-asym+zp ⇒ awq.rs;g128-sym ⇒ w4a16.rs)。
///
/// 纯读零猜测;未接方案**响亮拒**(带着实测到的 group_size/symmetric)。
fn quant_family(dir: &Path) -> Result<crate::formats::kind::QuantScheme, ModelError> {
    use crate::formats::kind::{QUANT_G32_ASYM, QUANT_G128_SYM};
    let raw = owl_shared::file_loader::read(&dir.join("config.json"))
        .map_err(|e| ModelError::Msg(format!("load_from_dir: 读 config.json: {e}")))?;
    let v: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|e| ModelError::Msg(format!("load_from_dir: config.json 解析: {e}")))?;
    let w = v
        .get("quantization_config")
        .and_then(|q| q.get("config_groups"))
        .and_then(|g| g.get("group_0"))
        .and_then(|g| g.get("weights"))
        .ok_or_else(|| {
            ModelError::Msg(
                "load_from_dir: 量化检查点缺 config_groups.group_0.weights(不许猜方案)".into(),
            )
        })?;
    let group_size = w.get("group_size").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let symmetric = w.get("symmetric").and_then(|x| x.as_bool()).unwrap_or(false);
    let num_bits = w.get("num_bits").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    // 三轴同判:**位数也必须在位**(W8/FP8 家族与 W4 同 g128-sym ⇒ 只盯
    // group_size/symmetric 会把 8bit 当 4bit 解,静默错值)
    match (num_bits, group_size, symmetric) {
        (4, 32, false) => Ok(QUANT_G32_ASYM),
        (4, 128, true) => Ok(QUANT_G128_SYM),
        _ => Err(ModelError::Msg(format!(
            "load_from_dir: 量化方案 (num_bits={num_bits}, group_size={group_size}, symmetric={symmetric}) 未接源族(响亮拒)"
        ))),
    }
}

/// **起点开箱**:目录 → (探测格式, 源) —— 起点轴收口(设计 §2.2)。
///
/// 三族全接:
/// - `SafeTensors` → [`SafeTensorsSource`];
/// - `PackQuantized` / `AwqGemm` → 按 `quant_method` 选族
///   (`compressed-tensors` → [`W4A16Source`];`awq` → [`AwqSource`];其余响亮拒)。
pub async fn load_from_dir<T, D>(
    dir: &Path,
    face: &mut D,
    comp: &T,
    opts: LoadOpts,
) -> Result<crate::module::LoadManifest, ModelError>
where
    T: crate::module::Loadable,
    D: DeviceClient,
{
    use crate::formats::kind::CheckpointKind;
    let det = crate::formats::kind::detect(dir)?;
    match det.kind {
        CheckpointKind::SafeTensors => {
            let src = crate::formats::safetensors::SafeTensorsSource::open_dir(dir)?;
            load_component(face, comp, &src, opts).await
        }
        CheckpointKind::PackQuantized | CheckpointKind::AwqGemm => {
            match quant_family(dir)? {
                crate::formats::kind::QUANT_G32_ASYM => {
                    let src = crate::formats::awq::AwqSource::open_dir(dir)?;
                    // 源侧重排模式注入(起点开箱的一环:量化大键的 marlin B
                    // 由 CPU 还是 GPU repack 产出;**须在首键物化前**)
                    src.set_device_repack(opts.device_repack.unwrap_or(false));
                    load_component(face, comp, &src, opts).await
                }
                crate::formats::kind::QUANT_G128_SYM => {
                    let src = crate::formats::w4a16::W4A16Source::open_dir(dir)?;
                    src.set_device_repack(opts.device_repack.unwrap_or(false));
                    load_component(face, comp, &src, opts).await
                }
                other => Err(ModelError::Msg(format!(
                    "load_from_dir: 量化方案 {other} 未接源族(响亮拒,不许猜)"
                ))),
            }
        }
    }
}

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
    let manifest = eval_want(&want, face, src, tap.as_ref(), ctx.verify).await?;
    // 上载栅栏:全部 DMA 落定后方可进入计算域(upload_pinned 入队即回执,
    // 完成语义由本栅栏一次总代价兜底)
    face.sync().await?;
    // 装载校验(OWL_LOAD_VERIFY 门控):整块回读 vs staged 校验和
    verify_loaded(&manifest, face, ctx.verify).await?;
    Ok(manifest)
}

// ============================================================================
// §2 调度:键分组 + LPT 分桶 + 顺序/并发两路径
// ============================================================================

/// 需求清单求值:face 具备并发句柄且 Want 多于一条 → 分桶流水;否则顺序。
///
/// **装载批**(2026-10-16 装载提速):设备端转换臂(原始字节上卡 → cast 核)
/// 把"逐键 sync"降成"批末 sync" —— 未冲的 cast 挂在批里,DMA 不逐键排空;
/// 同时收容待回收原始块(DeviceRearrange / 设备转换的 raw;launch 异步读,
/// 必须在栅栏后才能 free)。**每桶/每路独占一个**(跨账本禁混:块 id 只在本
/// face 的账本里可解析)。
#[derive(Default)]
pub(crate) struct LoadBatch {
    /// 待发 cast(顶序 = 上传顺序;flush 时一次性 launch)
    casts: Vec<PendingCast>,
    /// 上一轮 cast 的 raw(本轮 flush 的栅栏后方可回收 —— 两相回收:
    /// **本批 cast 的 raw 必须等下一轮栅栏**,否则 free 出的地址被复用
    /// 而 cast 还在读 = UAF)
    stale_ids: Vec<u64>,
    /// 待发批占用的原始字节(raw;定批阈值用)
    raw_bytes: usize,
}

/// 一件待发设备 cast(bf16 raw → f16 目标)
struct PendingCast {
    raw: u64,
    out: crate::contract::Bytes,
    elems: usize,
    key: String,
}

/// 批阈值:同时约束"滞留原始字节"(显存)与"件数"(sync 次数)
// ⚠️ **批 >1 未定谳**(2026-10-16 实测:batch 8/512MB ⇒ 概率性错值
// ——首图 argmax 79 vs 280、top-10 全不交,重跑又全绿;batch 32/512MB
// 确定复现;batch 4/256MB 单次通过;batch 1 = 逐键 sync 稳定全绿且
// **墙时同档(1.8s)** ⇒ 收益为零、风险确凿 ⇒ 默认批 1。
// 待专项排查:批内多键 raw 与 cast 的生命周期/池复用面(见设计文档 §14 挂账)。
const CAST_BATCH_BYTES: usize = 1;
const CAST_BATCH_KEYS: usize = 1;

impl LoadBatch {
    /// 入批(超阈即冲)
    async fn push_cast<D: DeviceClient>(
        &mut self,
        face: &mut D,
        c: PendingCast,
  ) -> Result<(), ModelError> {
        self.raw_bytes += c.elems * 2;
        self.casts.push(c);
        if self.raw_bytes >= CAST_BATCH_BYTES || self.casts.len() >= CAST_BATCH_KEYS {
            self.flush(face).await?;
        }
        Ok(())
    }

    /// 原始块入**待回收**(调用方保证:该块的所有读者 kernel 已入队;
    /// 回收推迟到下一轮 flush 的栅栏之后)
    fn push_free(&mut self, id: u64) {
        self.stale_ids.push(id);
    }

    /// 冲批:一次栅栏(H2D 落定)→ 逐件 launch cast → 回收原始块。
    /// 顺序不可换:cast 读 raw 必须在 DMA 之后;free 必须在 cast 之后。
    async fn flush<D: DeviceClient>(&mut self, face: &mut D) -> Result<(), ModelError> {
        if self.casts.is_empty() {
            return Ok(());
        }
        // 栅栏:H2D 落定(cast 读 raw 的前提)。
        // ⚠️ 2026-10-16 试验记录:把 DMA 换到 COMPUTE 流("同流免栅栏")
        // 在**去掉本栅栏**时**确定性错值**(四图 top-1 全错、top-10 不交),
        // 带栅栏则全绿 ⇒ "同流即序"的朴素推理不成立(根因未定,
        // 记设计文档 §14 挂账);故栅栏保留 = 正确性优先。
        face.sync().await?;
        self.reclaim(face).await?;
        // 本轮 cast 的 raw 记入下一轮(两相回收)
        for c in self.casts.drain(..) {
            launch_bf16_to_f16(face, c.raw, &c.out, c.elems, &c.key).await?;
            self.stale_ids.push(c.raw);
        }
        self.raw_bytes = 0;
        Ok(())
    }

    /// 收尾:最后一批 raw 也要回收(调用前须已有栅栏覆盖其读者)
    async fn finish<D: DeviceClient>(&mut self, face: &mut D) -> Result<(), ModelError> {
        self.flush(face).await?;
        face.sync().await?;
        self.reclaim(face).await?;
        Ok(())
    }

    async fn reclaim<D: DeviceClient>(&mut self, face: &mut D) -> Result<(), ModelError> {
        let ids = std::mem::take(&mut self.stale_ids);
        if !ids.is_empty() {
            face.free(&ids).await?;
        }
        Ok(())
    }
}

/// 设备端 bf16→f16 cast(语义拾取:driver::resolve;被动律同 DeviceRearrange)
async fn launch_bf16_to_f16<D: DeviceClient>(
    face: &mut D,
    raw: u64,
    out: &crate::contract::Bytes,
    elems: usize,
    key: &str,
) -> Result<(), ModelError> {
    use crate::contract::{Arg, Dtype, KernelSource, LaunchMsg};
    let env = face
        .op_env()
        .ok_or_else(|| ModelError::Msg(format!("Weight '{key}': 设备 cast 需 GPU 环境")))?;
    let op = crate::ops::SemanticKernel::CastBf16F16;
    let pick = owl_kernels::driver::resolve(owl_kernels::driver::OpReq {
        op: op.op_id(),
        env: &env,
        dt: owl_kernels::driver::DType::F16,
        shapes: &[],
        aux: &[],
        scalars: &[],
    });
    let msg = LaunchMsg {
        kernel: KernelSource {
            name: pick.name.to_string(),
            source: crate::kernel::source(pick.name).to_string(),
        },
        // 参数序 = elems.cast_bf16_f16 线契约:(x: BF16, n: i32, out: F16)
        args: vec![
            Arg::block(raw, Dtype::BF16, 1),
            Arg::I32(
                i32::try_from(elems)
                    .map_err(|_| ModelError::Msg(format!("Weight '{key}': 元素数 {elems} 超 i32")))?,
            ),
            Arg::block(out.id, Dtype::F16, 1),
        ],
        // 哨兵展开(全零 grid = runtime 自持;native 路展开规则 =
        // `models::ops::auto_grid` —— 手发 LaunchMsg 必须自己展开)
        grid: if pick.shape.grid == (0, 0, 0) {
            crate::ops::auto_grid(elems)
        } else {
            pick.shape.grid
        },
        block: pick.shape.block,
        shared_mem: pick.shape.smem,
        out_elems: elems,
    };
    face.launch(msg)
        .await
        .map_err(|e| ModelError::Msg(format!("Weight '{key}': cast launch {e}")))?;
    Ok(())
}

async fn eval_want<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
    verify: bool,
) -> Result<LoadManifest, ModelError> {
    // 单路径(2026-10-16 用户裁决:别用并行 DMA;实测并行桶 ≈1.2GB/s 反劣)
    eval_want_sequential(want, face, src, tap, verify).await
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
    verify: bool,
) -> Result<LoadManifest, ModelError> {
    let mut manifest = LoadManifest::default();
    let mut batch = LoadBatch::default();
    for group in key_groups(want.wants()) {
        manifest.0.extend(load_group(face, &group, src, tap, &mut batch, verify).await?);
    }
    batch.finish(face).await?;
    Ok(manifest)
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
    batch: &mut LoadBatch,
    verify: bool,
) -> Result<Vec<LoadEntry>, ModelError> {
    let mut manifest = Vec::new();
    for w in group {
        let n: usize = w.shape.iter().product();
        let entry = match w.layout {
            Layout::Transposed => load_transposed(face, w, src, n, tap, verify).await?,
            Layout::Direct => load_direct(face, w, src, n, tap, batch, verify).await?,
            Layout::DeviceRearrange { op, rows, cols } => {
                load_device_rearrange(face, w, src, n, op, rows, cols, tap, batch, verify).await?
            }
        };
        w.sink.deliver(crate::contract::Bytes::new(
            entry.block.id,
            n,
            w.dtype,
            w.shape.len() as u8,
        ));
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
    batch: &mut LoadBatch,
    verify: bool,
) -> Result<LoadEntry, ModelError> {
    // **原始字节 + 设备端转换臂**(2026-10-16 装载提速):源天然 bf16、
    // 目标 f16 ⇒ CPU 只做 memcpy(零元素转换),转换由设备 cast 核完成
    // —— 实测 CPU 转换 bf16→f16 仅 ~2.2GB/s(debug),是全载最大单相。
    // ⚠️ 设备转换臂是 **GPU-only**(要 BF16 设备块 + cast 核)⇒ 必须看面能力;
    // CpuFace 的 alloc 只认 F32/F16(2026-10-16 由 L2 等价门当场抓出)
    if face.device_kernels()
        && w.dtype == crate::contract::Dtype::F16
        && src.native_dtype(&w.key) == Some(crate::contract::Dtype::BF16)
    {
        return load_direct_device_cast(face, w, src, n, tap, batch, verify).await;
    }
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
        if off.is_multiple_of(chunk_elems * 4) || off >= n {
            face.sync().await?;
        }
    }
    probe.done();
    Ok(LoadEntry {
        key: w.key.clone(),
        dtype: w.dtype,
        shape: w.shape.clone(),
        layout: Layout::Direct,
        block: crate::contract::Bytes::new(b.id, n, w.dtype, w.shape.len() as u8),
        csum: if verify { csum } else { 0 },
    })
}

/// **原始字节 + 设备 cast 臂**(2026-10-16):CPU 只 memcpy,转换在 GPU。
///
/// 为什么:`load_direct` 的元素转换(mmap → F16C → pinned)在整载里是最大
/// 单相(bf16→f16 实测 ~2.2GB/s debug;9G 权重 ⇒ 数秒),而 DMA 可达
/// 5-9GB/s。本臂把元素转换换成"原始字节 memcpy + 设备 cast 核"(显存带宽
/// 量级),CPU 侧只余一次 memcpy ⇒ CPU 与 DMA 都在 GB/s 级且可流水重叠。
///
/// 三件:
/// 1. 原始块上卡(`Dtype::BF16`,元素数 = n ⇒ 字节数与 f16 目标等量);
/// 2. 设备 cast 核(`SemanticKernel::CastBf16F16`)→ 目标 f16 块;
/// 3. 原始块**延迟回收**(与 DeviceRearrange 同纪律:launch 异步读 raw,
///    火后即 free 会踩 ⇒ 入装载批,批末栅栏后统一回收)。
///
/// `verify` 校验和按**原始字节**计(raw 即数据真身;目标块 = 变换后,
/// 与 staged 不可比 —— 与 DeviceRearrange 同款挂账)。
async fn load_direct_device_cast<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    w: &crate::module::Want,
    src: &S,
    n: usize,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
    batch: &mut LoadBatch,
    verify: bool,
) -> Result<LoadEntry, ModelError> {
    let esz = Dtype::BF16.size_bytes();
    let mut probe = KeyProbe::start(tap, &w.key, n * esz);
    // 1) 原始块上卡(memcpy 直写 pinned → DMA;≥64MB 自动分块)
    let raw = face
        .alloc_uninit(Dtype::BF16, n)
        .await
        .map_err(|e| ModelError::Msg(format!("Weight '{}': raw alloc {e}", w.key)))?;
    let chunk_elems = (CHUNK_BYTES / esz).max(1);
    let mut csum = 0xcbf29ce484222325u64;
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
            let _s = probe.span("raw_copy");
            src.copy_native_into(&w.key, off, len, lease.slice_bytes_mut())
                .ok_or_else(|| attribution(src, &w.key, n))?;
        }
        if verify {
            csum = fnv1a(&lease.slice_bytes_mut()[..len * esz], csum);
        }
        {
            let _s = probe.span("htod");
            face.upload_pinned(lease, &raw, off * esz, len * esz)
                .await
                .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
        }
        off += len;
        if off.is_multiple_of(chunk_elems * 4) || off >= n {
            face.sync().await?;
        }
    }
    // 2) 目标块 + cast **入批**(批末一次 sync 后统一 launch —— 逐键 sync
    //    会打断 DMA 流水:699 键 ≈ 699 次全流排空)
    let b = face
        .alloc(w.dtype, n)
        .await
        .map_err(|e| ModelError::Msg(format!("Weight '{}': {e}", w.key)))?;
    {
        let _s = probe.span("cast_enqueue");
        batch
            .push_cast(
                face,
                PendingCast {
                    raw: raw.id,
                    out: crate::contract::Bytes::new(b.id, n, w.dtype, w.shape.len() as u8),
                    elems: n,
                    key: w.key.clone(),
                },
            )
            .await?;
    }
    probe.done();
    Ok(LoadEntry {
        key: w.key.clone(),
        dtype: w.dtype,
        shape: w.shape.clone(),
        layout: Layout::Direct,
        block: crate::contract::Bytes::new(b.id, n, w.dtype, w.shape.len() as u8),
        // 目标块 = 设备变换产物,与 staged(raw)不可比 ⇒ csum 挂 0
        // (同 DeviceRearrange;深校验留专用通道)
        csum: 0,
    })
}

/// 设备重排臂(2026-10-01 AWQ 装载提速):源供**原始 packed** 字节
/// (CPU 零重排)→ DMA 上卡 → registry 重排核就地生成 marlin 布局块
/// → 原始块回收(Free 通道)。CPU 33s 重量排出热路径,重排在 GPU
/// (显存带宽 ~900GB/s,哑 gather)。
#[allow(clippy::too_many_arguments)]
async fn load_device_rearrange<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    w: &crate::module::Want,
    src: &S,
    n: usize,
    op: crate::ops::SemanticKernel,
    rows: usize,
    cols: usize,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
    batch: &mut LoadBatch,
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
            if off.is_multiple_of(chunk_elems * 4) || off >= raw_elems {
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
            op: op.op_id(),
            env: &env,
            dt: owl_kernels::driver::DType::U32,
            shapes: &[vec![rows, cols]],
            aux: &[],
            scalars: &[],
        });
        let msg = crate::contract::LaunchMsg {
            kernel: crate::contract::KernelSource {
                name: pick.name.to_string(),
                source: crate::kernel::source(pick.name).to_string(),
            },
            args: vec![
                crate::contract::Arg::block(raw.id, crate::contract::Dtype::U32, 2), // ct packed [rows, cols]
                crate::contract::Arg::U64(rows as u64),
                crate::contract::Arg::U64(cols as u64),
                crate::contract::Arg::block(out_block.id, crate::contract::Dtype::U32, 2), // marlin B [k/16, rows*2]
            ],
            grid: pick.shape.grid,
            block: pick.shape.block,
            shared_mem: pick.shape.smem,
            out_elems: n,
        };
        face.launch(msg)
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
    // 3) 原始块回收(E2a Free 通道;**入批**统一回收 —— 阈值到达即冲批,
    //    批冲含一次全流栅栏 ⇒ repack 核读毕后方可 free。)
    batch.push_free(raw.id);
    probe.done();
    Ok(LoadEntry {
        key: w.key.clone(),
        dtype: w.dtype,
        shape: w.shape.clone(),
        layout: Layout::DeviceRearrange { op, rows, cols },
        block: crate::contract::Bytes::new(out_block.id, n, Dtype::U32, 2),
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
        block: crate::contract::Bytes::new(b.id, n, w.dtype, w.shape.len() as u8),
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

/// 装载分类账(dtype×layout 聚合:件数 + 字节)—— 组件档启动诊断用。
/// 通用工具(零模型知识):草稿头/塔/头部的装载账都走它,免得各写一份。
pub fn manifest_mem_ledger(manifest: &crate::module::LoadManifest, tag: &str) {
    let mut agg: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
    for e in manifest.entries() {
        let bytes: u64 = e.shape.iter().map(|&d| d as u64).product::<u64>()
            * match e.dtype {
                Dtype::F16 | Dtype::BF16 => 2,
                Dtype::F32 | Dtype::U32 => 4,
                Dtype::U8 => 1,
            };
        let key = format!("{:?}/{:?}", e.dtype, e.layout);
        let e2 = agg.entry(key).or_insert((0, 0));
        e2.0 += 1;
        e2.1 += bytes;
    }
    for (k, (n, b)) in agg {
        eprintln!("[{tag}] {k}: {n} 项 {b} ({:.2} GB)", b as f64 / 1e9);
    }
}

/// 装载取数失败归因:键不在 = 缺键;在但长度不符 = 元素数错误
fn attribution<S: WeightSource + ?Sized>(src: &S, key: &str, n: usize) -> ModelError {
    match src.elem_len(key) {
        Some(l) => ModelError::Msg(format!(
            "Weight '{key}': 元素 {l} != 期望 {n}"
        )),
        None => ModelError::Msg(format!("Weight '{key}': 数据源缺键")),
    }
}

#[cfg(test)]
mod generic_entry_tests {
    use super::*;
    use crate::formats::qwen35::{qwen3_5_0_8b, Qwen35Convention};

    /// **起点开箱不改清单门**(L2 等价门;**L4 后对照臂 = 手拼 eval_load**,
    /// 旧具名装载函数已删):`load_from_dir`(起点探测 + 开源 + 求值)与
    /// "同一声明直接 `eval_load`"产出**同一份清单**(键/dtype/形状逐条相等)。
    /// CPU face 即可(与 `cpu_real_weights_load` 同档),无需 GPU 门。
    #[tokio::test]
    async fn generic_entry_open_preserves_manifest() {
        let dir = crate::testkit::manifest_dir().join("assets/Qwen3.5-0.8B");
        if !dir.is_dir() {
            eprintln!("skip: {} 不在(模型库软链)", dir.display());
            return;
        }
        let spec = qwen3_5_0_8b();
        let mk = || {
            crate::model::Model::new(
                &spec,
                Qwen35Convention::new("model.language_model"),
                crate::module::QuantPlan::F16,
                crate::module::QuantPlan::F16,
            )
        };
        let mut face = owl_cpu::CpuFace::new();
        let opts = LoadOpts {
            dtype: spec.dtype,
            device_repack: Some(false),
            verify: false,
            debug_tap: false,
        };

        // 对照臂:同一声明 + 同一源,直接 eval_load(不经起点探测/开箱)
        let src = crate::formats::safetensors::SafeTensorsSource::open_dir(&dir).expect("源开箱");
        let old = eval_load(&mk(), &mut face, &src, &opts.ctx()).await.expect("eval_load");

        // 新路(起点探测 + 通用入口)
        let new = load_from_dir(&dir, &mut face, &mk(), opts)
            .await
            .expect("load_from_dir");

        let norm = |m: &LoadManifest| {
            let mut v: Vec<(String, String, Vec<usize>)> = m
                .entries()
                .iter()
                .map(|e| (e.key.clone(), format!("{:?}", e.dtype), e.shape.clone()))
                .collect();
            v.sort();
            v
        };
        let (a, b) = (norm(&old), norm(&new));
        assert_eq!(a.len(), b.len(), "清单条目数");
        assert_eq!(a, b, "键/dtype/形状逐条相等");
        eprintln!("[L2] 通用入口 ≡ 具名装载:{} 条逐项相等 ✓", a.len());
    }

    /// **新模型零代码门**(解释器化设计 §6 = 立案判据):model 库里**代码从未
    /// 见过**的档(Qwen3.5-9B:32 层 3:1 hybrid / hidden 4096)—— 不改一行
    /// .rs:起点探测([`detect`](crate::formats::kind::detect))→ 维度事实
    /// (`from_ckpt`)→ 通用入口(`load_from_dir`)全链数据驱动。
    ///
    /// 默认只跑起点侧(探测 + spec,秒级);`OWL_NEW_CKPT_LOAD=1` 追加
    /// **CPU face 整装**(9B f16 ≈18GB host —— GPU 门不参与:换 GPU 验装
    /// 是另一个门的事)。
    #[tokio::test]
    async fn unseen_size_loads_without_code() {
        let dir = std::path::PathBuf::from(
            std::env::var("OWL_NEW_CKPT")
                .unwrap_or_else(|_| "/home/div/Documents/codes/models/Qwen/Qwen3.5-9B".into()),
        );
        if !dir.is_dir() {
            eprintln!("skip: 新档检查点不在({};OWL_NEW_CKPT 可指路)", dir.display());
            return;
        }
        // ① 起点探测(纯索引嗅探)
        let det = crate::formats::kind::detect(&dir).expect("detect(新档)");
        // ② 维度事实(族预设只供 tokenizer/dtype 两项非维度事实)
        let preset = qwen3_5_0_8b();
        let (spec, rep) = crate::model::ModelSpec::from_ckpt(
            &dir,
            crate::formats::spec::SpecSource::family(&preset),
        )
        .expect("from_ckpt(新档)");
        eprintln!(
            "[new-ckpt] kind={:?} keys_base={:?} hidden={} inter={} layers={} full={:?} gdn={:?} vocab={} tied={} 失配={:?}",
            det.kind,
            rep.keys_base,
            spec.hidden,
            spec.inter,
            spec.layer_types.len(),
            spec.full_heads,
            spec.gdn_heads,
            spec.vocab,
            spec.tied,
            rep.mismatches
        );
        assert_eq!(det.kind, crate::formats::kind::CheckpointKind::SafeTensors);
        assert!(
            !rep.keys_base.is_empty() && rep.keys_base.ends_with('.'),
            "键基座应由索引实测定位"
        );
        assert!(rep.mismatches.is_empty(), "config↔形状应一致: {:?}", rep.mismatches);
        assert!(spec.layer_types.len() >= 8, "层数应成规模");
        for (i, full) in spec.layer_types.iter().enumerate() {
            assert_eq!(*full, i % 4 == 3, "层 {i} 层型不符 3:1 周期");
        }
        assert_eq!(spec.full_heads.0 * spec.full_heads.2 % spec.hidden, 0, "FA 头维守恒");
        if std::env::var("OWL_NEW_CKPT_LOAD").ok().as_deref() != Some("1") {
            eprintln!("[new-ckpt] 起点侧全绿(整装门 OWL_NEW_CKPT_LOAD=1 开启)");
            return;
        }
        // ③ 通用入口整装(CPU face;9B f16 ≈18GB host)
        let model = crate::model::Model::new(
            &spec,
            Qwen35Convention::new(rep.keys_base.trim_end_matches('.')),
            crate::module::QuantPlan::F16,
            crate::module::QuantPlan::F16,
        );
        let mut face = owl_cpu::CpuFace::new();
        let t0 = std::time::Instant::now();
        let manifest = load_from_dir(&dir, &mut face, &model, LoadOpts::plain(spec.dtype))
            .await
            .expect("整装(新档)");
        eprintln!(
            "[new-ckpt] 整装 {} 项 @ {:?}",
            manifest.entries().len(),
            t0.elapsed()
        );
        assert!(manifest.entries().len() > 300, "清单条目数应成规模");
    }

    /// **量化族通用入口**:27B AWQ 经 `load_from_dir` 与 `AwqSource` 直开同清单
    /// (门控真检查点存在;CPU face 不开设备重排 ⇒ `device_repack = Some(false)`)
    #[tokio::test]
    async fn generic_entry_covers_quantized_family() {
        let dir = std::path::PathBuf::from(
            std::env::var("OWL_CKPT_27B_AWQ")
                .unwrap_or_else(|_| "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4".into()),
        );
        if !dir.is_dir() {
            eprintln!("skip: 27B AWQ 检查点不在");
            return;
        }
        // AWQ 源产 U32 marlin 块 ⇒ **必须 GPU face**(CpuFace 只认 F32/F16);
        // 且 27B AWQ 整装 ≈10G+ ⇒ 需 ≥24G 卡(20G 实测 layer40 OOM)。
        // 独立门 `OWL_L2_AWQ=1`(本机跑法:`OWL_L2_AWQ=1 OWL_TEST_DEVICE=<3090>`)。
        if std::env::var("OWL_L2_AWQ").ok().as_deref() != Some("1") {
            eprintln!("skip: OWL_L2_AWQ=1 未开(27B AWQ 整装需 ≥24G 卡)");
            return;
        }
        if !crate::testkit::gpu_enabled() {
            crate::testkit::skip_note();
            return;
        }
        let spec = crate::formats::qwen35::qwen3_8_27b();
        let mk = || {
            crate::model::Model::new(
                &spec,
                Qwen35Convention::new("model.language_model"),
                crate::module::QuantPlan::W4A16,
                crate::module::QuantPlan::F16,
            )
        };
        let mut face = crate::testkit::gpu_client().await;
        let opts = LoadOpts {
            dtype: crate::contract::Dtype::F16,
            device_repack: Some(true), // 生产档(设备 marlin 重排)
            verify: false,
            debug_tap: false,
        };
        let src = crate::formats::awq::AwqSource::open_dir(&dir).expect("awq 源");
        let old = eval_load(&mk(), &mut face, &src, &opts.ctx()).await.expect("eval_load(awq)");
        let new = load_from_dir(&dir, &mut face, &mk(), opts).await.expect("load_from_dir(awq)");
        let norm = |m: &LoadManifest| {
            let mut v: Vec<(String, String, Vec<usize>)> = m
                .entries()
                .iter()
                .map(|e| (e.key.clone(), format!("{:?}", e.dtype), e.shape.clone()))
                .collect();
            v.sort();
            v
        };
        let (a, b) = (norm(&old), norm(&new));
        assert_eq!(a.len(), b.len(), "量化族清单条目数");
        assert_eq!(a, b, "量化族键/dtype/形状逐条相等");
        eprintln!("[L2] 量化族通用入口 ≡ AwqSource 直开:{} 条逐项相等 ✓", a.len());
    }
}
