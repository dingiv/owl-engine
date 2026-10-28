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
//! **装载期暂存预算**(2026-10-16 收尾):一切设备暂存走 [`LoadStaging`]
//! (声明驱动的固定 2 槽环,`OWL_LOAD_STAGING_MB` 上限,超槽键按段流式)
//! —— 装载前一行账、装载后零驻留;禁逐键 alloc/free(27B AWQ 曾因此攒
//! 下 12.15GB raw,把单卡 24G 撑爆在 layer48)。
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
    /// **切片总数**(S4 ②;1 = 单卡。与 `slice_id` 配对 = 确定性分片协议)
    pub shard: usize,
    /// **本切片 id**(S4 ②;`0..shard`)
    pub slice_id: usize,
    /// 设备重排(量化族;**已裁决结果**;`None` = 能力自述裁决)
    pub device_repack: Option<bool>,
    /// 装载校验(整块回读对账)
    pub verify: bool,
    /// 逐键分相探针(stderr)
    pub debug_tap: bool,
    /// **暂存槽上限 MiB**(原 `OWL_LOAD_STAGING_MB`;缺省 128 = 现行为)
    pub staging_mb: usize,
    /// 逐键字节 dump(原 `OWL_LOAD_DUMP`;诊断面)
    pub dump: bool,
    /// **GGUF host 解量化装载**(2026-10-21;MTP 头首用)。
    /// 缺省 false = 生产零解量化通道(`GgufSource`);true = 小件/异型头
    /// 走 `GgufDequantSource`(f32 元素面 → 目标 F16)。
    pub gguf_dequant: bool,
}

impl LoadOpts {
    /// 常规档:能力自述裁决 repack(生产入口调用点用它,别手拼 LoaderCtx)
    pub fn new<D: DeviceClient>(face: &D, dtype: crate::contract::Dtype) -> Self {
        Self {
            dtype,
            shard: 1,
            slice_id: 0,
            device_repack: Some(
                crate::module::RepackPath::resolve(None, face.device_kernels()).is_device(),
            ),
            verify: false,
            debug_tap: false,
            staging_mb: 128,
            dump: false,
            gguf_dequant: false,
        }
    }

    /// 非量化/固定档(无设备重排 + 零探针):裸 safetensors f16 直拷、
    /// W4A16-CPU 重排等"不需要能力裁决"的装载点。
    pub fn plain(dtype: crate::contract::Dtype) -> Self {
        Self { dtype, shard: 1, slice_id: 0, device_repack: Some(false), verify: false, debug_tap: false, staging_mb: 128, dump: false, gguf_dequant: false }
    }

    /// 装载 ctx(旋钮 → LoaderCtx;唯一产地)
    pub fn ctx(&self) -> crate::module::LoaderCtx {
        crate::module::LoaderCtx {
            dtype: self.dtype,
            shard: self.shard,
            slice_id: self.slice_id,
            device_repack: self.device_repack.unwrap_or(false),
            verify: self.verify,
            debug_tap: self.debug_tap,
            staging_mb: self.staging_mb,
            dump: self.dump,
        }
    }
}

/// **通用装载入口**(解释器化 L2 第一段):`Loadable::layout` 声明 × 起点源
/// ⇒ 物化。真正的"起点侧"通用化(按 [`CheckpointKind`] 自动开源)见
/// 装载期设备/池错误的**上下文包裹**:致命错误必须保住结构化类别
/// (`ModelError::is_fatal()` 是服务面"优雅关停 vs 单请求失败"的判据)。
/// 2026-10-16 实证:硬预算拒发(`PoolExhausted`)被包成 `Msg` ⇒ `is_fatal=false`
/// ⇒ 服务面不会走优雅关停 —— 故这里"致命直通 + 记一条带键名的 WARN",其余照包。
fn key_context(key: &str, e: ModelError) -> ModelError {
    if e.is_fatal() {
        owl_shared::log_warn!("load", "装载键 {key} 遇致命错误(保结构化直通):{e}");
        e
    } else {
        ModelError::Msg(format!("Weight '{key}': {e}"))
    }
}

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
    path: &Path,
    face: &mut D,
    comp: &T,
    opts: LoadOpts,
) -> Result<crate::module::LoadManifest, ModelError>
where
    T: crate::module::Loadable,
    D: DeviceClient,
{
    use crate::formats::kind::CheckpointKind;
    let det = crate::formats::kind::detect(path)?;
    match det.kind {
        CheckpointKind::SafeTensors => {
            let src = crate::formats::safetensors::SafeTensorsSource::open_dir(path)?;
            load_component(face, comp, &src, opts).await
        }
        CheckpointKind::PackQuantized | CheckpointKind::AwqGemm => {
            match quant_family(path)? {
                crate::formats::kind::QUANT_G32_ASYM => {
                    let src = crate::formats::awq::AwqSource::open_dir(path)?;
                    // 源侧重排模式注入(起点开箱的一环:量化大键的 marlin B
                    // 由 CPU 还是 GPU repack 产出;**须在首键物化前**)
                    src.set_device_repack(opts.device_repack.unwrap_or(false));
                    load_component(face, comp, &src, opts).await
                }
                crate::formats::kind::QUANT_G128_SYM => {
                    let src = crate::formats::w4a16::W4A16Source::open_dir(path)?;
                    src.set_device_repack(opts.device_repack.unwrap_or(false));
                    load_component(face, comp, &src, opts).await
                }
                other => Err(ModelError::Msg(format!(
                    "load_from_dir: 量化方案 {other} 未接源族(响亮拒,不许猜)"
                ))),
            }
        }
        // **GGUF 臂**(2026-10-19 刀 7c 接线):容器解析(刀 1)+ 内嵌超参(刀 3)
        // + 反变换权重源(刀 4c)。源的契约与其他三族**一致**(键 → 字节):
        // 量化键交给装载面的是**压缩块字节**(槽 dtype = `U8`),永不在 host
        // 解量化(设计 I-3);行置换/`ln(-x)` 在源侧完成(逐键描述符)。
        // 起点的**事实面**(超参 / 词表 / 模板)不在这里 —— 它们由引擎侧从
        // 同一索引取(刀 2/刀 3):本函数只管"键 → 字节"。
        CheckpointKind::Gguf => {
            if opts.gguf_dequant {
                // 小件/异型头(MTP)专用:host 解成 f32 → 目标 F16
                let src = crate::formats::gguf::GgufDequantSource::open(path)?;
                load_component(face, comp, &src, opts).await
            } else {
                let sel = (opts.shard > 1).then_some(crate::formats::gguf::ShardSel {
                    id: opts.slice_id,
                    world: opts.shard,
                });
                let src = crate::formats::gguf::GgufSource::open_shard(path, sel)?;
                load_component(face, comp, &src, opts).await
            }
        }
    }
}

/// 装载执行(层级入口):驱动层的 `Loadable::layout` 需求并物化。
/// 解释器自己调用层钩子并为它传递 LoaderCtx —— 使用者只给层与源。
///
/// ```rust,ignore
/// eval_load(&model, face, &src, &LoaderCtx { dtype: Dtype::F16, shard: 1, slice_id: 0, device_repack: false, verify: false, debug_tap: false, staging_mb: 128, dump: false }).await?;
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
    let mut want = layer.layout(ctx); // 声明:层产出需求清单(ctx 引用透传)
    // ★ S4 ② **确定性分片**(用户裁决 2026-10-16):按 `(slice_id, shard)` 就地
    // 改写 want 形状(→ 分片形状;DeviceRearrange 的 rows/cols 同步),并把源
    // 包成"只读本片"的视图 —— 6 条装载路径**零改动**(它们只经 WeightSource 取数)。
    // `shard = 1` 时 pass 返回空表、源直通(零开销)。
    // **源自担分片**的源(GGUF):跳过通用 pass —— 源已按 `(slice_id, world)`
    // 交付本 rank 的数据(`WeightSource::owns_shard`)。
    let spans = if src.owns_shard() {
        debug_assert!(ctx.shard > 1, "owns_shard 却单卡装载:源会按 world=1 交付");
        std::collections::HashMap::new()
    } else {
        crate::formats::shard::build_shard_pass(&mut want, ctx)?
    };
    // 冷盘预取(2026-10-24):单卡档把全部将读源区间交给源的预取口,
    // 让页缓存 I/O 与后续 CPU 物化重叠。分片档先不预取(会读到别的 rank
    // 的区间);源默认 no-op,只有 mmap 源覆写。
    // 单卡(1) 或 TP2 的 rank0(0):只让 rank0 预取**全量**源区间
    // 一次(两 rank 的切片并集 = 全量),rank1 直接吃 rank0 填的页缓存;
    // 若两 rank 都预取会读 2× 数据,反而更慢。
    if ctx.shard == 1 || ctx.slice_id == 0 {
        let keys: Vec<&str> = want.wants().iter().map(|w| w.key.as_str()).collect();
        src.prefetch(&keys);
    }
    let sharded = crate::formats::shard::ShardedSource::new(src, spans);
    let tap = if ctx.debug_tap { Some(std::sync::Mutex::new(Box::new(StderrTap) as Box<dyn LoadTap>)) } else { None }; // 观测:ctx.debug_tap 门控,缺省零开销
    let t0 = std::time::Instant::now();
    let load_result =
        eval_want(&want, face, &sharded, tap.as_ref(), ctx.verify, ctx.staging_mb).await;
    // 预取线程必须在装载返回前收口(装载后不得继续占盘/占线程);
    // 错误也 join,避免后台线程在失败路径上泄漏。
    src.prefetch_finish();
    let manifest = load_result?;
    // 上载栅栏:全部 DMA 落定后方可进入计算域(upload_pinned 入队即回执,
    // 完成语义由本栅栏一次总代价兜底)
    let t_fence = std::time::Instant::now();
    face.sync().await?;
    // 装载分相上屏(host 侧键循环 vs 设备追平栅栏;装载测速/瓶颈定谳用)
    owl_shared::log_info!(
        "load",
        "eval_load 分相:键循环(host) {:.2}s + 总栅栏(设备追平) {:.2}s",
        t_fence.duration_since(t0).as_secs_f64(),
        t_fence.elapsed().as_secs_f64()
    );
    // 装载校验(OWL_LOAD_VERIFY 门控):整块回读 vs staged 校验和
    verify_loaded(&manifest, face, ctx.verify).await?;
    Ok(manifest)
}

// ============================================================================
// §2 调度:键分组 + LPT 分桶 + 顺序/并发两路径
// ============================================================================

// ============================================================================
// §2 调度 + 装载暂存区(固定槽环;2026-10-16 收尾)
// ============================================================================

/// 暂存臂(**按语义分环**;槽规以**字节**计 —— 两臂元素宽不同,统一到字节)
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum StageArm {
    /// **原始字节入槽**:cast 臂的 bf16 输入 + 重排臂的 u32 输入 ——
    /// 同一语义(「源原始字节上卡,设备侧消费者读它」)⇒ **共用一环**
    /// (2026-10-16:合并前两臂各自开环,27B 档暂存 2×33.6 + 2×44.6 = 223MB
    /// 常驻整载期,在 24G 卡上贴顶;合一后 = 2×44.6MB,与提速前同额)
    RawIn,
    /// cast 臂的**分段输出**槽(超槽键流式用;F16 元素,与输入同元素数)
    CastSegOut,
}

impl StageArm {
    /// 开环时的块 dtype(仅用于 `alloc_uninit` 的元素计数;字节量 = 槽规)
    fn slot_dtype(self) -> Dtype {
        Dtype::U8
    }

    fn tag(self) -> &'static str {
        match self {
            StageArm::RawIn => "raw.in",
            StageArm::CastSegOut => "cast.seg.out",
        }
    }

    /// **本臂槽规上限**(与全局旋钮取小)。
    ///
    /// - `RawIn`:重排臂**无流式路** ⇒ 必须容下最大原始键(取全局旋钮,
    ///   缺省 128MB);cast 臂输入侧也走它(超槽时流式按段借同一环);
    /// - `CastSegOut`:只有流式路用它 ⇒ 槽规贴到 **32MB**(段规取与入槽的小者)。
    ///   32MB 是实测拐点:16.8MB 段时 embed/lm_head 的 host 拷贝只有 5.6GB/s
    ///   (段频 × 每段固定开销),33.6MB 段回到 10GB/s;而大段只多 34MB 常驻。
    fn cap_bytes(self, global_cap: usize) -> usize {
        const SEG_SLOT_CAP: usize = 32 << 20;
        match self {
            StageArm::RawIn => global_cap,
            StageArm::CastSegOut => global_cap.min(SEG_SLOT_CAP),
        }
    }
}

/// **装载暂存区**(装载期唯一设备暂存来源;固定预算)。
///
/// 病根(2026-10-16 收尾立案,实测账):原实现逐键 `alloc_uninit` + "两相
/// 回收",而回收口只在 `LoadBatch::flush`(有待发 cast 时)与装载收尾 ——
/// **纯量化检查点一条 cast 都没有** ⇒ 399 个 raw 块(27B AWQ = **12.15GB**)
/// 全攒到装载结束 ⇒ 24G 单卡在 layer48 崩(报错那一下 40MB 只是最后一根
/// 稻草)。改为:
/// - **声明驱动尺寸**:装载开始扫一遍 `LoaderOps`,按臂算出"本档最大暂存键"
///   ⇒ 槽 = `min(不可流式栏, 上限)`(上限 = `OWL_LOAD_STAGING_MB`,0 = 不设;
///   可流式的 cast 栏**不撑槽** —— 否则 embed/lm_head 的 2.54GB 会把整载期
///   暂存撑到上限)⇒ 装载期暂存预算 = `2 × Σ槽`,**装载前可打印、可断言**;
/// - **固定 2 槽环**:逐键借还,不再 alloc/free ⇒ churn 归零、峰值恒定;
/// - **槽复用时序**(2026-10-16 提速后):次序由**设备端事件**兜住 —— 写口
///   (`upload_and_launch` 复合)在拷贝前等"该槽消费者所在的 COMPUTE 尾事件"
///   (写等读),发射在 COMPUTE 同流(天然保序);由 2 槽轮转保证两次借同一槽
///   之间至少隔了另一键的那对等待。**不再**靠逐键 `face.sync()` 兜;
/// - **超槽键流式**:单键 > 槽 ⇒ 按段搬(输入切片 + 小整块输出 + D2D 贴回),
///   槽不随键长大(见 `load_direct_device_cast` 分段路)。
pub(crate) struct LoadStaging {
    rings: std::collections::BTreeMap<StageArm, SlotRing>,
    /// **上传在飞字节账**(跨键累计;排空判据的唯一状态)
    inflight_bytes: usize,
}

/// 排空阈值:H2D 在飞字节上界(= 4 × CHUNK_BYTES,与原"每 4 块排空一次"同额)。
/// 它同时是 **pinned 码头在飞上界**(租约由 server 侧流完成回调还池)。
const DRAIN_BYTES: usize = 4 * CHUNK_BYTES;

struct SlotRing {
    /// 槽规(plan 期定:min(声明峰值, 上限);字节)
    slot_bytes: usize,
    /// 环实体(首借时开:2 块)
    slots: Vec<Slot>,
    next: usize,
}

#[derive(Clone, Copy)]
struct Slot {
    id: u64,
    /// 槽容量(字节;借出时按需 > 本值 = 超槽 ⇒ 调用方走流式)
    bytes: usize,
}

/// 计划账(一行上屏:声明峰值 → 槽规 = 装载期暂存预算的静态证明)
pub(crate) struct StagingPlan {
    per_arm: Vec<(StageArm, usize, usize)>,
    cap_bytes: usize,
}

impl LoadStaging {
    /// **声明扫描 → 槽规**(纯数据,零设备分配)。
    ///
    /// 两栏账:**不可流式**(重排臂 —— 槽必须容下最大原始键,超槽 = 响亮拒)
    /// 与 **可流式**(cast 臂 —— 超槽按键分段过同一环)。槽规按两栏合成:
    /// `slot = min(不可流式峰值, 上限)`;可流式那栏只供日志与段规,
    /// **不把槽撑大**(否则 embed/lm_head 的 2.54GB 会把整载期暂存拉到
    /// 上限 —— 2026-10-16 实测:共享环被 128MB 上限撑成 2×134MB)。
    fn plan<S: WeightSource + ?Sized>(
        want: &LoaderOps,
        src: &S,
        cap_bytes: usize,
    ) -> Result<(Self, StagingPlan), ModelError> {
        use std::collections::BTreeMap;
        // 不可流式(重排臂;单一入槽 raw.in)
        let mut nostream: BTreeMap<StageArm, usize> = BTreeMap::new();
        // 可流式(cast 臂;raw.in 共用 + 自己的段出槽)
        let mut stream: BTreeMap<StageArm, usize> = BTreeMap::new();
        for w in want.wants() {
            let n: usize = w.shape.iter().product();
            // **探测必须在 match 之前**:`native_dtype` 现在返回 `Taken`
            // (2026-10-17 审查 §③),而 match guard 里写不了 `?`(类型层不允许)
            let native = src.native_dtype(&w.key)?;
            let arm = match w.layout {
                // 设备重排臂:源 packed 整键上卡(rows × cols × 4B)
                Layout::DeviceRearrange { rows, cols, .. } => Some((&mut nostream, StageArm::RawIn, rows * cols * 4)),
                // 设备 cast 臂:want F16 而源天然 BF16 ⇒ 原始字节上卡落槽(2B/元素)
                Layout::Direct if w.dtype == Dtype::F16 && native == Some(Dtype::BF16) => {
                    Some((&mut stream, StageArm::RawIn, n * 2))
                }
                _ => None,
            };
            if let Some((m, a, bytes)) = arm {
                let e = m.entry(a).or_insert(0);
                *e = (*e).max(bytes);
            }
        }
        // 段出槽:容量 = 可流式那栏的最大原始键(段规再取与入槽的小者)
        let cast_peak = stream.get(&StageArm::RawIn).copied().unwrap_or(0);
        if cast_peak > 0 {
            stream.insert(StageArm::CastSegOut, cast_peak);
        }
        let mut rings = BTreeMap::new();
        let mut per_arm = Vec::new();
        for arm in [StageArm::RawIn, StageArm::CastSegOut] {
            let ns = nostream.get(&arm).copied().unwrap_or(0);
            let st = stream.get(&arm).copied().unwrap_or(0);
            let declared = ns.max(st);
            if declared == 0 {
                continue; // 本档不用该臂
            }
            let cap = if cap_bytes == 0 { declared } else { arm.cap_bytes(cap_bytes) };
            // **槽规**(2026-10-16 第二修):
            //  - 有"不可流式"键(重排臂)⇒ 槽**必须**容下它 —— 该臂没有分段流式路,
            //    超槽 = 装不进(响亮拒);⇒ 这里**不能**用旋钮削它。
            //    现场:生产档 `head_quant = rtn-int4-g32` 的 `lm_head`
            //    (248320×5120 BF16 → u4 打包 635.7MB)被旋钮缺省 128MB 削到
            //    134MB ⇒ 启动期 `引擎装配失败`(而计划期已算得出 635.7MB)。
            //  - 只有可流式键 ⇒ `min(声明峰值, 上限)`,超槽键走段流(槽不随键长)。
            let slot_bytes = if ns > 0 { ns } else { st.min(cap) }.max(1);
            per_arm.push((arm, declared, slot_bytes));
            rings.insert(arm, SlotRing { slot_bytes, slots: Vec::new(), next: 0 });
        }
        Ok((Self { rings, inflight_bytes: 0 }, StagingPlan { per_arm, cap_bytes }))
    }

    /// **在飞字节入账**(上传口调用;返回 true = 该排空了)。
    ///
    /// 2026-10-16 release 提速定谳:原直接臂为"每 4 块 **或**键尾"排空
    /// (`off.is_multiple_of(chunk_elems*4) || off >= n`)⇒ **每键一次全设备
    /// sync**(27B 档 ≈1050 个小键),键循环 host 8.3s 中 5.9s 是等设备追平
    /// (单键账单只占 2.4s)—— DMA/重排管线被逐键斩断。改为只按**累计在飞
    /// 字节**排空:小键不再各排一次,大键照旧分段;pinned 码头在飞上界不变
    /// (阈值仍是 4 块),总栅栏仍在 `eval_load` 末尾一次性兜底。
    fn note_inflight(&mut self, bytes: usize) -> bool {
        self.inflight_bytes = self.inflight_bytes.saturating_add(bytes);
        if self.inflight_bytes >= DRAIN_BYTES {
            self.inflight_bytes = 0;
            true
        } else {
            false
        }
    }

    /// 一行上屏(声明峰值 → 槽规;装载期暂存预算 = 2 × Σ槽)
    fn log_plan(plan: &StagingPlan) {
        let mut total = 0usize;
        let mut rows = Vec::new();
        for (arm, declared, slot) in &plan.per_arm {
            total += 2 * slot;
            rows.push(format!(
                "{}: 声明峰值 {:.1}MB → 槽 {:.1}MB ×2",
                arm.tag(),
                *declared as f64 / 1e6,
                *slot as f64 / 1e6
            ));
            // 槽超过旋钮 = 不可流式栏(重排臂)抬上来的 —— 必须可见:
            // 这是 VRAM 代价,不是错误,但读者要知道钱花在哪
            if plan.cap_bytes > 0 && *slot > plan.cap_bytes {
                owl_shared::log_warn!(
                    "load",
                    "⚠ {} 槽 {:.1}MB > 旋钮 {}MB —— 该臂**不可流式**(重排整键上卡)\n        ⇒ 槽必须容下最大重排键(嘬平 = 装不进);要降本:换档/关 head_quant/或减小该键",
                    arm.tag(),
                    *slot as f64 / 1e6,
                    plan.cap_bytes >> 20
                );
            }
        }
        owl_shared::log_info!(
            "load",
            "暂存区(固定预算):{};合计 {:.1}MB(上限 {})",
            if rows.is_empty() { "(本档无暂存臂)".to_string() } else { rows.join(" | ") },
            total as f64 / 1e6,
            if plan.cap_bytes == 0 { "无(取声明峰值)".to_string() } else { format!("{}MB", plan.cap_bytes >> 20) }
        );
    }

    /// 上限旋钮(唯一 env 读取点:**缺省 128MB** = 单键超此就流式;0 = 不设上限)
    /// 缺省值的理由:128MB 是 CHUNK_BYTES(上传分块口径),槽=块 ⇒ 整键单发
    /// 时不会多切一刀;而绝大多数键(27B AWQ 最大 44.6MB)都在槽内 ⇒ 常规路
    /// 零额外栅栏,只有 embed/lm_head 这类巨键(0.5~2GB)走分段。
    /// 上限字节数(**从配置来**:`[load] staging_mb`;0/1 = 最小 1MB 兜底)
    fn cap_bytes(mb: usize) -> usize {
        mb.max(1) << 20
    }

    /// 借槽(固定环轮转;首借开环 = 2 块 × 槽规)。
    ///
    /// **复用正确性**:写口(复合 `upload_and_launch`)在搬运前等"消费者读毕"
    /// 的设备端事件、发射在 COMPUTE 同流 ⇒ 槽复写不会抢跑读者;2 槽轮转
    /// 保证两次借同一槽之间至少隔了另一键的那对等待。返回 `None` = 本键
    /// 超槽(调用方走流式;**禁静默截断**)。
    async fn acquire<D: DeviceClient>(
        &mut self,
        face: &mut D,
        arm: StageArm,
        need: usize,
    ) -> Result<Option<Slot>, ModelError> {
        let Some(ring) = self.rings.get_mut(&arm) else {
            return Ok(None);
        };
        if ring.slots.is_empty() {
            for _ in 0..2 {
                // 记名(全量):暂存区名 + 臂 tag(如 `staging.cast_bf16_in`)
                let nm = format!("staging.{}", arm.tag());
                let id = face
                    .alloc_uninit(arm.slot_dtype(), ring.slot_bytes, &nm)
                    .await
                    .map_err(|e| ModelError::Msg(format!("暂存区 {}: alloc {e}", arm.tag())))?
                    .id;
                ring.slots.push(Slot { id, bytes: ring.slot_bytes });
            }
        }
        let slot = ring.slots[ring.next % ring.slots.len()];
        ring.next = ring.next.wrapping_add(1);
        if need > slot.bytes {
            return Ok(None);
        }
        Ok(Some(slot))
    }

    /// 本臂槽规(字节;0 = 本档不需要该臂)
    fn slot_bytes(&self, arm: StageArm) -> usize {
        self.rings.get(&arm).map(|r| r.slot_bytes).unwrap_or(0)
    }

    /// **收尾:槽全数归还**(暂存只在装载期 —— 装载后零驻留;
    /// 2026-10-16 实测教训:槽不还 = 常驻 2×Σ槽,4B e2e 第四图被挤 OOM)
    async fn finish<D: DeviceClient>(&mut self, face: &mut D) -> Result<(), ModelError> {
        face.sync().await?;
        let ids: Vec<u64> = self
            .rings
            .values_mut()
            .flat_map(|r| r.slots.drain(..).map(|s| s.id))
            .collect();
        if !ids.is_empty() {
            face.free(&ids).await?;
        }
        // **池回收**(2026-10-16):暂存块 free 后仍被 cudaMallocAsync 池缓存
        // (池属性 RELEASE_THRESHOLD=UINT64_MAX ⇒ 永不自动还 OS),会把跑图期
        // 余量吃掉(27B 单卡实测:峰值处 1.7MiB 分配 CUDA_ERROR_OUT_OF_MEMORY)。
        // 装载收尾是唯一的"暂存全空"时刻 ⇒ 此时还给 OS 最干净。
        // 门:仅设备面(GPU 池存在;CPU face 不支持 = 预期,不刷告警)。
        // 失败 = 可见降级(内存面不回退语义;不许静默吞)。
        if !face.device_kernels() {
            return Ok(());
        }
        match face.mem_pool_trim().await {
            Ok(0) => {}
            Ok(n) => owl_shared::log_info!(
                "load",
                "装载收尾池回收:还 OS {:.1}MB", n as f64 / 1e6),
            Err(e) => owl_shared::log_warn!("load", "装载收尾池回收失败(显存面降级): {e}"),
        }
        Ok(())
    }

    /// 超槽文案(响亮:带臂/需求/槽规/旋钮)
    fn over_cap_msg(arm: StageArm, need_bytes: usize, slot_bytes: usize) -> String {
        format!(
            "暂存区 {}: 本键 {:.1}MB > 槽 {:.1}MB(固定预算;OWL_LOAD_STAGING_MB 可调;本臂流式未开)",
            arm.tag(),
            need_bytes as f64 / 1e6,
            slot_bytes as f64 / 1e6
        )
    }
}

/// 设备端 bf16→f16 cast(语义拾取:driver::resolve;被动律同 DeviceRearrange)
/// 设备 cast 的 **LaunchMsg**(不发射):参数序 = `elems.cast_bf16_f16` 线契约
/// `(x: BF16, n: i32, out: F16)`。E3 起由复合口 `upload_and_launch` 一次带走
/// (搬运 + 发射之间的序 = 设备端事件)。
fn cast_msg<D: DeviceClient>(
    face: &D,
    raw: u64,
    out: &crate::contract::Bytes,
    elems: usize,
    key: &str,
) -> Result<crate::contract::LaunchMsg, ModelError> {
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
    Ok(msg)
}

async fn eval_want<D: DeviceClient, S: WeightSource + ?Sized>(
    want: &LoaderOps,
    face: &mut D,
    src: &S,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
    verify: bool,
    staging_mb: usize,
) -> Result<LoadManifest, ModelError> {
    // 单路径(2026-10-16 用户裁决:别用并行 DMA;实测并行桶 ≈1.2GB/s 反劣)
    // 暂存区 = 声明驱动(装载前一行账;固定预算,2026-10-16 收尾)
    let (mut staging, plan) = LoadStaging::plan(want, src, LoadStaging::cap_bytes(staging_mb))?;
    LoadStaging::log_plan(&plan);
    eval_want_sequential(want, face, src, tap, verify, &mut staging).await
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
    staging: &mut LoadStaging,
) -> Result<LoadManifest, ModelError> {
    let mut manifest = LoadManifest::default();
    let groups = key_groups(want.wants());
    let t0 = std::time::Instant::now();
    let mut t_group = std::time::Duration::ZERO;
    for group in &groups {
        let tg = std::time::Instant::now();
        manifest.0.extend(load_group(face, group, src, tap, staging, verify).await?);
        t_group += tg.elapsed();
    }
    // 键循环分账(≈0.5s 时差 = 分组/分派壳成本;大时说明有未入账段)
    if tap.is_some() {
        owl_shared::log_info!(
            "load",
            "键循环分账:{} 组 / 总 {:.2}s = load_group 累计 {:.2}s + 壳成本 {:.2}s",
            groups.len(),
            t0.elapsed().as_secs_f64(),
            t_group.as_secs_f64(),
            (t0.elapsed() - t_group).as_secs_f64()
        );
    }
    staging.finish(face).await?; // 收尾栅栏 + 槽归还(装载后零驻留)
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
    staging: &mut LoadStaging,
    verify: bool,
) -> Result<Vec<LoadEntry>, ModelError> {
    let mut manifest = Vec::new();
    for w in group {
        let n: usize = w.shape.iter().product();
        // **NVFP4 打包臂**(2026-10-28,P2):一源二槽 + host 标量 —— 产出与
        // 交付都在臂内(它一次产两个 `LoadEntry`,套不进下面"单 entry"的形状)
        if let Layout::NvFp4Weight { rows, cols } = w.layout {
            manifest.extend(
                load_nvfp4(face, w, src, rows, cols, tap, verify)
                    .await
                    .map_err(|e| key_context(&w.key, e))?,
            );
            continue;
        }
        let entry = match w.layout {
            Layout::Transposed => load_transposed(face, w, src, n, tap, verify).await?,
            Layout::Direct => load_direct(face, w, src, n, tap, staging, verify).await?,
            Layout::DeviceRearrange { op, rows, cols } => {
                load_device_rearrange(face, w, src, n, op, rows, cols, tap, staging, verify).await?
            }
            // 上一块已消费;此处只围护穷尽匹配(真走到 = 逻辑漏)
            Layout::NvFp4Weight { .. } => unreachable!("NVFP4 臂已在上面分流"),
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
    owl_shared::log_info!(
        "load-verify",
        "checked={checked} skipped(rearrange)={skipped} mismatch={} {:.2}s",
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
    staging: &mut LoadStaging,
    verify: bool,
) -> Result<LoadEntry, ModelError> {
    // **原始字节 + 设备端转换臂**(2026-10-16 装载提速):源天然 bf16、
    // 目标 f16 ⇒ CPU 只做 memcpy(零元素转换),转换由设备 cast 核完成
    // —— 实测 CPU 转换 bf16→f16 仅 ~2.2GB/s(debug),是全载最大单相。
    // ⚠️ 设备转换臂是 **GPU-only**(要 BF16 设备块 + cast 核)⇒ 必须看面能力;
    // CpuFace 的 alloc 只认 F32/F16(2026-10-16 由 L2 等价门当场抓出)
    if face.device_kernels()
        && w.dtype == crate::contract::Dtype::F16
        && src.native_dtype(&w.key)? == Some(crate::contract::Dtype::BF16)
    {
        return load_direct_device_cast(face, w, src, n, tap, staging, verify).await;
    }
    let esz = w.dtype.size_bytes();
    let mut probe = KeyProbe::start(tap, &w.key, n * esz);
    // 记名(全量):`wt.<键名>` —— 装载期最大的一块块账消费者,名字就是来源键
    let nm = format!("wt.{}", w.key);
    let b = {
        let _s = probe.span("out_alloc");
        face.alloc(w.dtype, n, &nm)
            .await
            .map_err(|e| key_context(&w.key, e))?
    };
    let chunk_elems = (CHUNK_BYTES / esz).max(1);
    let mut off = 0usize;
    
    let mut csum = 0xcbf29ce484222325u64;
    while off < n {
        let len = chunk_elems.min(n - off);
        let mut lease = {
            let _s = probe.span("pinned_alloc");
            face.alloc_pinned(len * esz, &format!("pin.{}", w.key))
                .await
                .map_err(|e| key_context(&w.key, e))?
        };
        {
            let _s = probe.span("convert");
            src.convert_chunk_into_bytes(&w.key, off, len, lease.slice_bytes_mut(), w.dtype)?
                .ok_or_else(|| attribution(src, &w.key, n))?;
        }
        if let Some(pat) = owl_shared::env_reader::str("OWL_LOAD_DUMP") {
            if w.key.contains(&pat) {
                let sb = lease.slice_bytes_mut();
                let head: Vec<String> = sb[..8.min(sb.len())]
                    .chunks_exact(2)
                    .map(|c| {
                        let b = u16::from_le_bytes([c[0], c[1]]);
                        format!("{:04X}(f16={:.5},bf16={:.5})", b, half::f16::from_bits(b).to_f32(), half::bf16::from_bits(b).to_f32())
                    })
                    .collect();
                owl_shared::log_debug!(
                    "dump",
                    "key={} want_dtype={:?} head={:?}", w.key, w.dtype, head);
            }
        }
        if verify {
            csum = fnv1a(&lease.slice_bytes_mut()[..len * esz], csum);
        }
        {
            let _s = probe.span("htod");
            face.upload_pinned(lease, &b, off * esz, len * esz)
                .await
                .map_err(|e| key_context(&w.key, e))?;
        }
        off += len;
        if staging.note_inflight(len * esz) {
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
///    火后即 free 会踩 ⇒ 交**固定暂存区**(槽逐键借还,栅栏后复写)。
///
/// `verify` 校验和按**原始字节**计(raw 即数据真身;目标块 = 变换后,
/// 与 staged 不可比 —— 与 DeviceRearrange 同款挂账)。
async fn load_direct_device_cast<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    w: &crate::module::Want,
    src: &S,
    n: usize,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
    staging: &mut LoadStaging,
    verify: bool,
) -> Result<LoadEntry, ModelError> {
    let esz = Dtype::BF16.size_bytes();
    let mut probe = KeyProbe::start(tap, &w.key, n * esz);
    let mut csum = 0xcbf29ce484222325u64;
    // 目标块(永久产物)。记名:`wt.<键名>`(与 cpu-cast 臂同律)。
    let nm = format!("wt.{}", w.key);
    let b = {
        let _s = probe.span("out_alloc");
        face.alloc(w.dtype, n, &nm)
            .await
            .map_err(|e| key_context(&w.key, e))?
    };
    let out_bytes = crate::contract::Bytes::new(b.id, n, w.dtype, w.shape.len() as u8);

    match staging.acquire(face, StageArm::RawIn, n * esz).await? {
        // ① 整键入槽(常见路:键 ≤ 槽)
        Some(slot) => {
            let chunks =
                fill_native_chunks(face, src, &w.key, 0, n, &mut probe, verify, &mut csum).await?;
            let in_bytes = crate::contract::Bytes::new(slot.id, n, Dtype::BF16, 1);
            let msg = cast_msg(face, slot.id, &out_bytes, n, &w.key)?;
            {
                let _s = probe.span("cast_enqueue");
                // **E3 复合口**:搬运 + 发射一次走 —— 两段之间的序 = 设备端事件
                // (H2D 尾事件 → COMPUTE waitEvent),**零 host 排空**
                // (原 `face.sync()` 全排空 ≈1.9ms/键)
                face.upload_and_launch(&in_bytes, chunks, msg)
                    .await
                    .map_err(|e| ModelError::Msg(format!("Weight '{}': 复合上传 + 发射 {e}", w.key)))?;
            }
        }
        // ② **超槽键流式**:段 ≤ 槽;输入切片 → 段输出槽 → D2D 贴回目标块中段
        //    (输出槽只接受整块 ⇒ 走"小整块 + copy_block_at"两跳,不碰内核契约)
        None => {
            // 段规 = min(原始入槽, 段出槽) 的元素数(两环容量都按字节声明)
            let in_cap = staging.slot_bytes(StageArm::RawIn);
            let out_cap = staging.slot_bytes(StageArm::CastSegOut);
            let seg_cap = in_cap.min(out_cap) / esz;
            let in_slot = staging
                .acquire(face, StageArm::RawIn, seg_cap * esz)
                .await?
                .ok_or_else(|| ModelError::Msg(LoadStaging::over_cap_msg(StageArm::RawIn, seg_cap * esz, in_cap)))?;
            let seg_slot = staging
                .acquire(face, StageArm::CastSegOut, seg_cap * w.dtype.size_bytes())
                .await?
                .ok_or_else(|| ModelError::Msg(LoadStaging::over_cap_msg(StageArm::CastSegOut, seg_cap * w.dtype.size_bytes(), out_cap)))?;
            owl_shared::log_info!(
                "load",
                "超槽键流式:{} {:.1}MB > 槽 {:.1}MB ⇒ {} 段",
                w.key,
                (n * esz) as f64 / 1e6,
                (seg_cap * esz) as f64 / 1e6,
                n.div_ceil(seg_cap)
            );
            let mut off = 0usize;
            while off < n {
                let seg = seg_cap.min(n - off);
                let chunks =
                    fill_native_chunks(face, src, &w.key, off, seg, &mut probe, verify, &mut csum)
                        .await?;
                let seg_out = crate::contract::Bytes::new(seg_slot.id, seg, w.dtype, 1);
                let in_bytes = crate::contract::Bytes::new(in_slot.id, seg, Dtype::BF16, 1);
                let msg = cast_msg(face, in_slot.id, &seg_out, seg, &w.key)?;
                // 复合:段搬运 + cast 一次走(设备端序);贴回在 COMPUTE 同流,
                // 与 cast 天然有序;下一段/下一键的输入槽复写由复合口的
                // **写等读前置等待**(H2D ← COMPUTE 尾)罩住 —— 全程零 host 排空
                face.upload_and_launch(&in_bytes, chunks, msg)
                    .await
                    .map_err(|e| ModelError::Msg(format!("Weight '{}': 段复合 {e}", w.key)))?;
                {
                    let _s = probe.span("copyback");
                    face.copy_block_at(&seg_out, 0, &out_bytes, off * w.dtype.size_bytes(), seg * w.dtype.size_bytes())
                        .await
                        .map_err(|e| ModelError::Msg(format!("Weight '{}': 段贴回 {e}", w.key)))?;
                }
                off += seg;
            }
        }
    }
    // 槽复用的"消费者读毕"交给了下一键复合口的写等读前置等待;
    // 装载末 `eval_load` 的上载栅栏负责"全部落定"(host 化读取/校验前)
    probe.done();
    Ok(LoadEntry {
        key: w.key.clone(),
        dtype: w.dtype,
        shape: w.shape.clone(),
        layout: Layout::Direct,
        block: out_bytes,
        // 目标块 = 设备变换产物,与 staged(raw)不可比 ⇒ csum 挂 0
        // (同 DeviceRearrange;深校验留专用通道)
        csum: 0,
    })
}

/// 原始字节分块上卡(源元素区间 `[src_off, src_off+len)` → 目标块**槽内**
/// 首元素;≥64MB 自动分块 + 每 4 块/尾一次栅栏 —— cast 臂与重排臂共用的
/// 上传口(槽是固定块,故目标偏移恒 0;流式的段偏移在源侧)。
#[allow(clippy::too_many_arguments)]
/// 原始 bf16 字节分块**填充** pinned 租约(不上传 —— 块表交给复合口一次带走)。
/// 源元素区间 `[src_off, src_off+len_total)`;每块 = `CHUNK_BYTES` 粒。
#[allow(clippy::too_many_arguments)]
async fn fill_native_chunks<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    src: &S,
    key: &str,
    src_off: usize,
    len_total: usize,
    probe: &mut KeyProbe<'_>,
    verify: bool,
    csum: &mut u64,
) -> Result<Vec<crate::contract::UpChunk>, ModelError> {
    let esz = Dtype::BF16.size_bytes();
    let chunk_elems = (CHUNK_BYTES / esz).max(1);
    let mut chunks = Vec::new();
    let mut off = 0usize;
    while off < len_total {
        let len = chunk_elems.min(len_total - off);
        let mut lease = {
            let _s = probe.span("pinned_alloc");
            face.alloc_pinned(len * esz, &format!("pin.{key}"))
                .await
                .map_err(|e| ModelError::Msg(format!("Weight '{key}': {e}")))?
        };
        {
            let _s = probe.span("raw_copy");
            src.copy_native_into(key, src_off + off, len, lease.slice_bytes_mut())?
                .ok_or_else(|| attribution(src, key, src_off + len_total))?;
        }
        if verify {
            *csum = fnv1a(&lease.slice_bytes_mut()[..len * esz], *csum);
        }
        chunks.push(crate::contract::UpChunk {
            offset_bytes: off * esz,
            len_bytes: len * esz,
            lease,
        });
        off += len;
    }
    owl_shared::log_debug!("load-rearrange", "{key}: fill 完成 {} 段", chunks.len());
    Ok(chunks)
}

/// 源 packed 字节分块**填充** pinned 租约(重排臂专用:源是 U32 packed;
/// `packed_host` = OWL_LOAD_VERIFY 的 host 副本,顺路拷走)。
async fn fill_packed_chunks<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    src: &S,
    key: &str,
    raw_elems: usize,
    probe: &mut KeyProbe<'_>,
    packed_host: &mut Option<Vec<u8>>,
) -> Result<Vec<crate::contract::UpChunk>, ModelError> {
    let chunk_elems = (CHUNK_BYTES / 4).max(1);
    let mut chunks = Vec::new();
    let mut off = 0usize;
    let n_chunks = raw_elems.div_ceil(chunk_elems);
    owl_shared::log_debug!(
        "load-rearrange",
        "{key}: fill 开始 raw_elems={raw_elems} chunk_elems={chunk_elems} ⇒ {n_chunks} 段"
    );
    while off < raw_elems {
        let len = chunk_elems.min(raw_elems - off);
        owl_shared::log_debug!(
            "load-rearrange",
            "{key}: 段 {} 借租约前 off={off} len={len}",
            chunks.len()
        );
        let mut lease = {
            let _s = probe.span("pinned_alloc");
            // 记名(全量):与另一条 repack 路同律(`pin.<键名>`)
            face.alloc_pinned(len * 4, &format!("pin.{key}"))
                .await
                .map_err(|e| ModelError::Msg(format!("Weight '{key}': {e}")))?
        };
        owl_shared::log_debug!("load-rearrange", "{key}: 段 {} 租约到手", chunks.len());
        {
            let _s = probe.span("convert");
            src.convert_chunk_into_bytes(
                key,
                off,
                len,
                lease.slice_bytes_mut(),
                crate::contract::Dtype::U32,
            )?
            .ok_or_else(|| attribution(src, key, raw_elems))?;
        }
        owl_shared::log_debug!("load-rearrange", "{key}: 段 {} 取数完成", chunks.len());
        if let Some(ph) = packed_host.as_mut() {
            ph[off * 4..(off + len) * 4].copy_from_slice(&lease.slice_bytes_mut()[..len * 4]);
        }
        chunks.push(crate::contract::UpChunk {
            offset_bytes: off * 4,
            len_bytes: len * 4,
            lease,
        });
        off += len;
    }
    owl_shared::log_debug!("load-rearrange", "{key}: fill 完成 {} 段", chunks.len());
    Ok(chunks)
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
    staging: &mut LoadStaging,
    verify: bool,
) -> Result<LoadEntry, ModelError> {
    let raw_elems = rows * cols;
    let mut probe = KeyProbe::start(tap, &w.key, raw_elems * 4);
    let mut packed_host = if verify {
        Some(vec![0u8; raw_elems * 4])
    } else {
        None
    };

    // 1) 原始块 = **固定槽**(借还;不再逐键 alloc/free —— 2026-10-16 收尾:
    //    原实现把 raw 交给两相回收,而纯量化档一次 flush 都不发生 ⇒
    //    27B AWQ 的 12.15GB raw 全攒到装载结束 = 单卡 24G OOM 根因)
    let raw_slot = staging
        .acquire(face, StageArm::RawIn, raw_elems * 4)
        .await?
        .ok_or_else(|| {
            // 带**键名**(2026-10-16:原文案只说"本键多大" —— 现场靠算
            // `raw_elems*4` 反推是哪个键,等于让人重建账)
            ModelError::Msg(format!(
                "Weight '{}': {}",
                w.key,
                LoadStaging::over_cap_msg(
                    StageArm::RawIn,
                    raw_elems * 4,
                    staging.slot_bytes(StageArm::RawIn),
                )
            ))
        })?;
    let raw = crate::contract::Bytes::new(raw_slot.id, raw_elems, crate::contract::Dtype::U32, 2);
    // 源 packed 分块填租约(不上传;序交给复合口 —— E3 起不再逐键 host 排空)
    let chunks = fill_packed_chunks(face, src, &w.key, raw_elems, &mut probe, &mut packed_host).await?;
    // 2) 重排核(语义拾取:driver::resolve(OpEnv 必传,被动律);
    //    grid/block 契约公式住 driver(load::ct_repack))
    let out_block = {
        let _s = probe.span("out_alloc");
        face.alloc_uninit(crate::contract::Dtype::U32, n, &format!("repack.{}", w.key))
            .await
            .map_err(|e| ModelError::Msg(format!("Weight '{}': out alloc {e}", w.key)))?
    };
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
        // 排障面(加载器与解释器同级;P2.1-a 草稿分片挂住立案):发射前把**几何账**
        // 打全 —— 键/核名/rows/cols/字数/槽,分片档下第一处不等就在这里现形
        owl_shared::log_debug!(
            "load-rearrange",
            "{}: op={:?} 核={} rows={} cols={} n={} raw_elems={}",
            w.key,
            op,
            pick.name,
            rows,
            cols,
            n,
            raw_elems
        );
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
        // **E3 复合口**:搬运 + 重排一次走(写等读 + 读等写都是设备端事件)
        // (入账在 `repack` 段:@resolve/msg 构建 + 复合往返)
        face.upload_and_launch(&raw, chunks, msg)
            .await
            .map_err(|e| ModelError::Msg(format!("Weight '{}': 复合重排 {e}", w.key)))?;
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
                // 现场账(2026-10-17):GPU 块字数 n vs CPU 参考字数 b_ref.len()
                // —— 分片档实测两者不等(⇒ 索引表/字数口径在分片维上错位)
                owl_shared::log_info!(
                    "load-verify",
                    "{} rows={rows} cols={cols} k={k} n={n} p32={} idx={} b_ref={}",
                    w.key, p32.len(), idx.len(), b_ref.len()
                );
                let mut bad = 0usize;
                for (i, w) in gbuf.chunks_exact(4).enumerate() {
                    let got = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                    if got != b_ref[i] { bad += 1; }
                }
                owl_shared::log_info!(
                    "load-verify",
                    "{vi} {} 坏字 {bad}/{}", w.key, b_ref.len());
                assert_eq!(bad, 0, "GPU repack 端到端校验失败: {}", w.key);
            }
        }
    }
    // 3) 收位:槽复用交给下一键复合口的写等读前置等待;全部落定由装载末
    //    上载栅栏负责(host 读取/校验/交付前)
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

/// **NVFP4 权重打包臂**(2026-10-28 装载管线 P2;设计 §4/§7)。
///
/// 一源二槽:源 = 全精度权重(`[rows, cols]` 行主序)→ `q`(e2m1 对)
/// + `sf`(e4m3/16 块缩放,CUTLASS 交错)+ `gb`(host 标量)。
///
/// **为什么在 host 算**:`pack_weight_b` 本就是纯主机函数,而源字节流式读
/// 文件时天然在 host ⇒ 不新增算子(否则量化器还要再走一遍 client/server/ops
/// 三件套 —— 那才是校验面真正的成本)。将来若要提速,就换一个
/// `Place::Device` 实现,**接口不变**。
///
/// **不暂存设备环**(与其余三臂的关键区别):设备环 / `LoadStaging` 是给
/// "上卡后再重排"用的;本臂在主机上一次性产出最终字节,直接 `htod` ⇒
/// 不进 `StagingPlan` 的账(峰值 = 一个键的 f16 副本 + 两块产物)。
///
/// **落账**:两个槽各一条 `LoadEntry`(`sf` 用派生键 `{key}.sf`),
/// `csum` 照常算(块内容 == 主机字节,回读可比;与重排臂不同)。
async fn load_nvfp4<D: DeviceClient, S: WeightSource + ?Sized>(
    face: &mut D,
    w: &crate::module::Want,
    src: &S,
    rows: usize,
    cols: usize,
    tap: Option<&Mutex<Box<dyn LoadTap>>>,
    verify: bool,
) -> Result<Vec<LoadEntry>, ModelError> {
    use crate::formats::transform::{NvFp4Pack, Transform, TransformCtx};

    let elems = rows * cols;
    let mut probe = KeyProbe::start(tap, &w.key, elems * 2);

    // 1) 取源字节 + 源精度事实(`SrcDtype` 只从源的元数据来)
    let (src_bytes, src_dt) = {
        let _s = probe.span("fetch");
        fetch_source_bytes(src, &w.key, elems)?
    };

    // 2) 变换声明 → 产物缓冲 → 执行(host)
    let xf = NvFp4Pack { rows, cols };
    let slots = xf.slots()?;
    if xf.place() != crate::formats::transform::Place::Host {
        return Err(ModelError::Msg(
            "NVFP4 打包臂只实现了 Host 位(Device 版另立实现,接口不变)".into(),
        ));
    }
    // 产物槽必须与层声明的槽位一一对上(主槽 = q,附加槽 = sf)
    verify_slot_agreement(w, &slots)?;
    let mut outs: Vec<Vec<u8>> = slots.iter().map(|s| vec![0u8; s.bytes()]).collect();
    // **标量口契约**(2026-10-28 review):`gb` 是**数值的一部分** —— 掉了它,
    // 层上会继续用 `ScalarSink` 的缺省 1.0,输出就静默被 `1/gb` 缩放。
    // 所以:初值取 NaN(不可当真值),跑完仍是 NaN = 变换没写;而层没留口
    // (`want.scalar = None`)= 装了也用不上 —— 两个方向都响亮拒。
    if w.scalar.is_none() {
        return Err(ModelError::Msg(
            "NVFP4 声明缺 `scalar` 口:变换会产出 per-tensor 全局缩放 `gb`,\
             没有回填口就只能用缺省 1.0(读数静默偏移)—— 请用 `Weight::layout_as_nvfp4` 声明"
                .into(),
        ));
    }
    let mut scalars = [f32::NAN];
    {
        let _s = probe.span("pack");
        xf.run(&mut TransformCtx {
            src_bytes: &src_bytes,
            src_dt,
            rows,
            cols,
            outs: &mut outs,
            scalars: &mut scalars,
        })?;
    }
    if !scalars[0].is_finite() {
        return Err(ModelError::Msg(format!(
            "NVFP4 打包未产出 `gb`(scalar 仍是 {})—— 变换实现与声明不一致",
            scalars[0]
        )));
    }
    // 源字节用完即弃(下个键前释放;大键 f16 副本不攒留)
    drop(src_bytes);

    // 3) 逐槽上卡 + 交付(主槽 = `w.sink`;其余按声明序走 `w.extra`)
    let mut entries = Vec::with_capacity(slots.len());
    for (i, (slot, bytes)) in slots.iter().zip(outs.iter()).enumerate() {
        let _s = probe.span("htod");
        let nm = if i == 0 {
            format!("wt.{}", w.key)
        } else {
            format!("wt.{}.{}", w.key, slot.name)
        };
        let b = face
            .htod(slot.dtype, &slot.shape, bytes, &nm)
            .await
            .map_err(|e| key_context(&w.key, e))?;
        let numel: usize = slot.shape.iter().product();
        let block = crate::contract::Bytes::new(b.id, numel, slot.dtype, slot.shape.len() as u8);
        // 交付(主槽与附加槽同一种口;`Sink` 自己管单元格)
        let sink = if i == 0 {
            &w.sink
        } else {
            &w.extra[i - 1].sink
        };
        sink.deliver(block.clone());
        entries.push(LoadEntry {
            key: if i == 0 {
                w.key.clone()
            } else {
                format!("{}.{}", w.key, slot.name)
            },
            dtype: slot.dtype,
            shape: slot.shape.clone(),
            layout: w.layout,
            block,
            csum: if verify {
                fnv1a(bytes, 0xcbf29ce484222325)
            } else {
                0
            },
        });
    }

    // 4) host 标量回填(层发射时作 `gb` —— 与本处字节**同源**,不同源 = 静默错值)。
    // 口与值两边都已在上面硬校过(缺口/未写均拒),这里的 `expect` 是**结构保证**:
    // `w.scalar = Some(..)` 恒成立,`scalars[0]` 恒为有限值。
    w.scalar
        .as_ref()
        .expect("上面已校: NVFP4 声明必有 scalar 口")
        .store(scalars[0]);
    probe.done();
    Ok(entries)
}

/// **槽位一致性门**(声明面 ⇔ 变换面):数目、dtype、形状逐项对齐。
///
/// 为什么在装载期硬校:目标形状在两边各算一次(`Linear::quantize_nvfp4` 建槽 /
/// `Transform::slots` 报产物)。两处口径一旦漂移,上卡后的块会**静静地**小一号
/// (设备端按声明读越界),必须在写入显存前就阻住。
fn verify_slot_agreement(
    w: &crate::module::Want,
    slots: &[crate::formats::transform::SlotDecl],
) -> Result<(), ModelError> {
    let mut declared: Vec<(&'static str, Dtype, &[usize])> = vec![("q", w.dtype, &w.shape)];
    for s in &w.extra {
        declared.push((s.name, s.dtype, &s.shape));
    }
    if declared.len() != slots.len() {
        return Err(ModelError::Msg(format!(
            "NVFP4 槽数不符:声明 {} 件,变换产 {} 件",
            declared.len(),
            slots.len()
        )));
    }
    for ((name, dt, shape), s) in declared.iter().zip(slots.iter()) {
        if *name != s.name || *dt != s.dtype || *shape != s.shape.as_slice() {
            return Err(ModelError::Msg(format!(
                "NVFP4 槽位漂移:声明 ({name}, {dt:?}, {shape:?}) vs 变换 ({}, {:?}, {:?})",
                s.name, s.dtype, s.shape
            )));
        }
    }
    Ok(())
}

/// **取源字节 + 源精度事实**(设计 §3.1/§3.3 的唯一入口)。
///
/// 两条路:
/// 1. **原始字节通道**(mmap 源常态):`copy_native_into` 直拷(2B/元素),
///    精度取文件头 `native_dtype` —— 零解码,不吃 f32 中间量;
/// 2. **元素通道**(无可直拷字节的源:虚拟合并键 / 解量化源):逐**行带**
///    `take_range` 取 f32 → 写 LE 字节。分带是为了不让"整键 f32 + 整键字节"
///    同时在场(4B 的 `gate_up` 级键 = 4× 上百 MB)。
///
/// 两条路的产物形式统一 = `(字节, SrcDtype)`,下游只走 `decode_to_f32`。
fn fetch_source_bytes<S: WeightSource + ?Sized>(
    src: &S,
    key: &str,
    elems: usize,
) -> Result<(Vec<u8>, crate::formats::source::SrcDtype), ModelError> {
    use crate::formats::source::SrcDtype;

    if let Some(dt) = src
        .native_dtype(key)?
        .and_then(SrcDtype::from_dtype)
    {
        let esz = dt.size_bytes();
        let mut buf = vec![0u8; elems * esz];
        if src.copy_native_into(key, 0, elems, &mut buf)?.is_some() {
            return Ok((buf, dt));
        }
    }
    // 元素通道(分带;带宽已知 = `elems`)
    let mut buf = vec![0u8; elems * 4];
    // 带宽 = **32MB 的 f32**(不是 32M 个元素 —— 后者是 128MB,
    // 大键上会让“整键 f32 + 整键字节”同时在场的目标白白多占一倍)
    let band = 8usize << 20;
    let mut off = 0usize;
    while off < elems {
        let len = (elems - off).min(band);
        let v = src
            .take_range(key, off, len)?
            .ok_or_else(|| attribution(src, key, elems))?;
        if v.len() != len {
            return Err(ModelError::Msg(format!(
                "Weight '{key}': 区间 [{off},{}）取回 {} 个元素",
                off + len,
                v.len()
            )));
        }
        for (i, x) in v.iter().enumerate() {
            buf[(off + i) * 4..(off + i) * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
        off += len;
    }
    Ok((buf, SrcDtype::F32))
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
        src.take_range(&w.key, 0, n)?
            .ok_or_else(|| attribution(src, &w.key, n))?
    };
    let t = {
        let _s = probe.span("transpose");
        transpose_into_vec(&data, w, n)?
    };
    let (dtype, bytes) = {
        let _s = probe.span("convert");
        // **按声明 dtype 交付**(2026-10-19 定谳):原实现对非 F16 走 F32 分支 ——
        // `htod(F32, shape, 4n 字节)` 后 `LoadEntry.dtype` 却报 `w.dtype`,块内容
        // 是 **F32 字节冒充 f16/bf16** ⇒ 读者按 2B 解释 ⇒ **静默垃圾**。
        // 命中面 = DFlash2 未量化草稿的 `fc`(`Weight::new_transposed` 唯一用户,
        // 2026-10-10 悬案的真因之二:memory 与参考实现 cos ≈ 0)。
        // 其余 dtype = 转置臂未覆盖 ⇒ 响亮拒(不许再走“默认 F32”的静默路)。
        match w.dtype {
            Dtype::F16 => (Dtype::F16, to_f16_bytes(&t)),
            Dtype::BF16 => (Dtype::BF16, to_bf16_bytes(&t)),
            Dtype::F32 => (
                Dtype::F32,
                t.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(),
            ),
            other => {
                return Err(ModelError::Msg(format!(
                    "Weight '{}': 转置臂不支持 dtype {other:?}(只有 f16/bf16/f32;\
                     新增 dtype 必须先补字节转换 —— 不许回落 F32 冒充)",
                    w.key
                )))
            }
        }
    };
    let b = {
        let _s = probe.span("htod");
        face.htod(dtype, &w.shape, &bytes, &format!("wt.{}", w.key))
            .await
            .map_err(|e| key_context(&w.key, e))?
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

/// f32 切片 → bf16 字节(RNE,与 `half::bf16::from_f32` 同律)
fn to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut dst = Vec::with_capacity(v.len() * 2);
    for &x in v {
        dst.extend_from_slice(&half::bf16::from_f32(x).to_le_bytes());
    }
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
    /// 各 span 之和(**不等于**墙钟;`wall - total` = 未入账时间:
    /// 分配/复合往返/排空等未开 span 的段)
    pub total: Duration,
    /// 键墙钟(臂函数入口到交付;诊断未入账时间用)
    pub wall: Duration,
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
        owl_shared::log_info!(
            "load",
            "{:<46} {:>9}B wall={:>7.1}ms spans={:>7.1}ms [{}]",
            rec.key,
            rec.bytes,
            rec.wall.as_secs_f64() * 1e3,
            rec.total.as_secs_f64() * 1e3,
            detail.join(" ")
        );
    }
}

/// 单键分相账本(RAII span;恒累计 —— 成本每键数次 Duration 记账,
/// 交付时 tap 缺省即弃)
struct KeyProbe<'a> {
    tap: Option<&'a Mutex<Box<dyn LoadTap>>>,
    t_start: Instant,
    rec: KeyRecord,
}

impl<'a> KeyProbe<'a> {
    fn start(tap: Option<&'a Mutex<Box<dyn LoadTap>>>, key: &str, bytes: usize) -> Self {
        Self {
            tap,
            t_start: Instant::now(),
            rec: KeyRecord {
                key: key.to_string(),
                bytes,
                total: Duration::ZERO,
                wall: Duration::ZERO,
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
        self.rec.wall = self.t_start.elapsed();
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
                Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::I8 | Dtype::U8 => 1,
            };
        let key = format!("{:?}/{:?}", e.dtype, e.layout);
        let e2 = agg.entry(key).or_insert((0, 0));
        e2.0 += 1;
        e2.1 += bytes;
    }
    for (k, (n, b)) in agg {
        // tag 字面量契约:组件名进正文(调用方给 "draft-mem"/"gguf-draft-mem")
        owl_shared::log_info!("load-ledger", "[{tag}] {k}: {n} 项 {b} ({:.2} GB)", b as f64 / 1e9);
    }
}

/// 装载取数失败归因:键不在 = 缺键;在但长度不符 = 元素数错误
fn attribution<S: WeightSource + ?Sized>(src: &S, key: &str, n: usize) -> ModelError {
    // **错误报告期不许再失败**(2026-10-17 审查 §③-N1):`elem_len` 现返回 `Taken`,
    // 但这里已在造错误消息 ⇒ 探测失败退化为"归因不可用",绝不叠第二个错误
    match src.elem_len(key).unwrap_or(None) {
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
            owl_shared::testkit::report::skip(format_args!("{} 不在(模型库软链)", dir.display()));
            return;
        }
        let spec = qwen3_5_0_8b();
        let mk = || {
            crate::model::Model::new(
                &spec,
                Qwen35Convention::new("model.language_model"),
                crate::module::QuantPlan::F16,
                crate::module::QuantPlan::F16,
            ).expect("Model::new")
        };
        let mut face = owl_cpu::CpuFace::new();
        let opts = LoadOpts {
            dtype: spec.dtype,
            shard: 1,
            slice_id: 0,
            device_repack: Some(false),
            verify: false,
            debug_tap: false,
            staging_mb: 128,
            dump: false,
            gguf_dequant: false,
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
    /// 默认只跑起点侧(探测 + spec,秒级);`ckpt:new_LOAD=1` 追加
    /// **CPU face 整装**(9B f16 ≈18GB host —— GPU 门不参与:换 GPU 验装
    /// 是另一个门的事)。
    #[tokio::test]
    async fn unseen_size_loads_without_code() {
        let dir = std::path::PathBuf::from(
            owl_shared::testkit::resource("ckpt:new")
                .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/Qwen/Qwen3.5-9B").into()),
        );
        if !dir.is_dir() {
            owl_shared::testkit::report::skip(format_args!("新档检查点不在({};ckpt:new 可指路)", dir.display()));
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
        if !owl_shared::testkit::resource_flag("flag:new-ckpt-load") {
            eprintln!("[new-ckpt] 起点侧全绿(整装门 ckpt:new_LOAD=1 开启)");
            return;
        }
        // ③ 通用入口整装(CPU face;9B f16 ≈18GB host)
        let model = crate::model::Model::new(
            &spec,
            Qwen35Convention::new(rep.keys_base.trim_end_matches('.')),
            crate::module::QuantPlan::F16,
            crate::module::QuantPlan::F16,
        ).expect("Model::new");
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
            owl_shared::testkit::resource("ckpt:27b-awq")
                .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/cyankiwi/Qwen3.8-27B-AWQ-INT4").into()),
        );
        if !dir.is_dir() {
            owl_shared::testkit::report::skip("27B AWQ 检查点不在");
            return;
        }
        // AWQ 源产 U32 marlin 块 ⇒ **必须 GPU face**(CpuFace 只认 F32/F16);
        // 且 27B AWQ 整装 ≈10G+ ⇒ 需 ≥24G 卡(20G 实测 layer40 OOM)。
        // 独立门 `OWL_L2_AWQ=1`(本机跑法:`OWL_L2_AWQ=1 OWL_TEST_DEVICE=<3090>`)。
        if std::env::var("OWL_L2_AWQ").ok().as_deref() != Some("1") {
            owl_shared::testkit::report::skip("OWL_L2_AWQ=1 未开(27B AWQ 整装需 ≥24G 卡)");
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
            ).expect("Model::new")
        };
        let mut face = crate::testkit::gpu_client().await;
        let opts = LoadOpts {
            dtype: crate::contract::Dtype::F16,
            shard: 1,
            slice_id: 0,
            device_repack: Some(true), // 生产档(设备 marlin 重排)
            verify: false,
            debug_tap: false,
            staging_mb: 128,
            dump: false,
            gguf_dequant: false,
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

    /// **转置臂按声明 dtype 交付**(2026-10-19 立案):原实现对非 F16 走 F32 分支,
    /// 但 `LoadEntry.dtype` 仍报 `w.dtype` ⇒ **F32 字节冒充 bf16/f16**。
    /// 命中面 = DFlash2 未量化草稿的 `fc`(`Weight::new_transposed` 唯一用户):
    /// memory 与参考实现 cos ≈ 0 ⇒ z-lab 草稿 AL 恒 1.0(10-10 悬案的真因之二)。
    ///
    /// 本门:源 [row=2, col=3] → 声明转置 [3, 2],**BF16** 交付 ⇒ 回读必须
    /// = 源转置且 dtype = BF16(CPU face;无卡可跑)。
    #[tokio::test]
    async fn transposed_arm_delivers_declared_dtype() {
        use crate::module::{Loadable, LoaderCtx, LoaderOps, Weight, WeightSource};

        struct One(Weight);
        impl Loadable for One {
            fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
                self.0.layout(ctx)
            }
        }
        struct Mem(Vec<f32>);
        impl WeightSource for Mem {
            fn take(&self, _key: &str) -> crate::module::Taken<Vec<f32>> {
                Ok(Some(self.0.clone()))
            }
            fn take_range(&self, _key: &str, off: usize, len: usize) -> crate::module::Taken<Vec<f32>> {
                Ok(self.0.get(off..off + len).map(<[f32]>::to_vec))
            }
            fn elem_len(&self, _key: &str) -> crate::module::Taken<usize> {
                Ok(Some(self.0.len()))
            }
            fn native_dtype(&self, _key: &str) -> crate::module::Taken<crate::contract::Dtype> {
                Ok(Some(crate::contract::Dtype::F32))
            }
        }

        // 源 [rows=2, cols=3];声明转置 ⇒ 设备形状 [3, 2]
        let src: Vec<f32> = (0..6).map(|i| i as f32 + 0.5).collect();
        let mut face = owl_cpu::CpuFace::new();
        for dt in [crate::contract::Dtype::BF16, crate::contract::Dtype::F16] {
            // ⚠️ dtype 单元在 Weight 内是 `Cell`(clone **不共享**)⇒ 必须读
            // 装载层持有的那一份,不能拿手建的副本对账
            let layer = One(Weight::new_transposed("fc", 2, 3));
            let ctx = LoaderCtx {
                dtype: dt,
                ..LoaderCtx::default()
            };
            eval_load(&layer, &mut face, &Mem(src.clone()), &ctx)
                .await
                .expect("eval_load(transposed)");
            assert_eq!(layer.0.dtype(), dt, "声明 dtype 必须落为装载 dtype");
            let got = match dt {
                crate::contract::Dtype::BF16 => crate::testkit::harvest_bf16(&mut face, &layer.0.decl()).await,
                crate::contract::Dtype::F16 => crate::testkit::harvest_f16(&mut face, &layer.0.decl()).await,
                _ => unreachable!(),
            };
            // 期望:目标 [c*rows + r] = 源 [r*cols + c]
            let mut want = vec![0f32; 6];
            for r in 0..2 {
                for c in 0..3 {
                    want[c * 2 + r] = src[r * 3 + c];
                }
            }
            assert_eq!(got.len(), 6, "元素数");
            for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-3,
                    "{dt:?} 转置臂第 {i} 项错:得 {g} 期 {e}(F32 字节冒充?)"
                );
            }
        }
        eprintln!("[loader] 转置臂 dtype 交付门:bf16/f16 逐元素 ✓");
    }
}
