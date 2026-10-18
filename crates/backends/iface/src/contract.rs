//! 契约面:对 server 的**能力期望 + 线格式**(前后端共同依赖的唯一权威)。
//!
//! 2026-09-25 自 owl-models 下沉(裁决:后端不依赖前端——owl-cuda 此前
//! 反向依赖 owl-models 取线格式,依赖方向倒挂)。现状:
//!
//! ```text
//! owl-iface::contract   ← 类型权威(Dtype/Shape/线格式/DeviceClient 契约)
//!      ↑            ↑
//! owl-models(前端)  owl-cuda(后端)
//!   张量语义层          哑执行器
//! ```
//!
//! 分工:
//! - 本模块只放**跨 crate 线格式与能力契约**——server 照此实现,
//!   上层照此调用;零张量语义(TensorOps/Op/Kernel 值归 owl-models);
//! - 错误分两族:HAL 池/相位族 = 本 crate [`crate::BackendError`];
//!   server 回执族 = 本模块 [`ModelError`]。

use std::future::Future;

// ============================================================================
// 分配点宏:位置自动带入(`alloc!` / `htod!` 等五臂)
// ============================================================================
//
// **它解决什么**:全量记名要求每个分配点给名字。生产调用点的名字是**语义**
// (核名/层名/装载键名 —— 那才是 OOM 现场要看的"谁占着"),但**测试/探针/
// 例子**里没有语义可得:那些地方真需要的只是"这块是哪一处分配的"。
// 手写 `"文件名::函数#N.方法"` 一是不便,二是**行号会漂**(改一行就指错地方,
// 比没有名字更坏 —— 它会误导排障)。
//
// 宏在**调用点**展开 ⇒ `file!()`/`line!()`/`column!()` 由编译器填真实位置,
// 永远不会漂;名字 = `方法@相对路径:行:列`,天然唯一(无需编号)。
//
// **纪律**(与 §全量记名 同源;docs/arch/全量记名-与-OOM-dump.md §5):
// - 生产代码**仍写语义名**(核名/层名/键名)—— 位置名对"谁占着"没有信息量;
// - 本宏供**测试/探针/example** 使用(它们的诊断口径就是"哪一行");
// - 两条路都落进同一个 `BlockEntry.name`,口径不分裂。
#[macro_export]
macro_rules! alloc {
    ($f:expr, $dtype:expr, $elems:expr) => {
        $f.alloc($dtype, $elems, concat!("alloc@", file!(), ":", line!(), ":", column!()))
    };
}
#[macro_export]
macro_rules! alloc_uninit {
    ($f:expr, $dtype:expr, $elems:expr) => {
        $f.alloc_uninit($dtype, $elems, concat!("alloc_uninit@", file!(), ":", line!(), ":", column!()))
    };
}
#[macro_export]
macro_rules! htod {
    ($f:expr, $dtype:expr, $shape:expr, $bytes:expr) => {
        $f.htod($dtype, $shape, $bytes, concat!("htod@", file!(), ":", line!(), ":", column!()))
    };
}
#[macro_export]
macro_rules! htod_f32 {
    ($f:expr, $shape:expr, $data:expr) => {
        $f.htod_f32($shape, $data, concat!("htod_f32@", file!(), ":", line!(), ":", column!()))
    };
}
#[macro_export]
macro_rules! alloc_pinned {
    ($f:expr, $bytes:expr) => {
        $f.alloc_pinned($bytes, concat!("pin@", file!(), ":", line!(), ":", column!()))
    };
}

// ============================================================================
// 标注词汇:Dtype + Shape(线格式的元数据维)
// —— 2026-10-12 M2 迁往 owl-kernels::contract(契约老家;iface 依赖
//    kernels 方向不变)。下方 re-export 保旧路径全量可用,零破坏。
// ============================================================================

pub use owl_kernels::contract::{numel, Arg, Bytes, Dtype, GraphId, KernelSource, LaunchMsg, Shape};

// ============================================================================
// 错误面:server 回执族(词汇权威在 iface;后端必须能表达)
// ============================================================================

/// 边界错误:装载期(new)与执行期(interpret/to_host)的结构化报错。
#[derive(Debug, Clone)]
pub enum ModelError {
    // ---- 装载期(new;P 阶段)----
    /// 权重键缺失(vb 路径 + 键名;键名双约定问题的结构化出口)
    MissingKey { path: String, key: String },
    /// 形状不符(期望 vs 实际)
    ShapeMismatch { path: String, expected: Vec<usize>, got: Vec<usize> },
    /// dtype 不符
    DtypeMismatch { path: String, expected: Dtype, got: Dtype },

    // ---- 执行期(interpret/to_host;server 回执)----
    /// 池容量不足
    PoolExhausted { pool: String, needed: u64, available: u64 },
    /// 池块句柄死亡后被引用(令牌世代校验失败;哨兵①词汇)
    DeadBlock { id: u64 },
    /// 跨账本解析:块 id 被拿到别家账本解析(pitfall §27 撞号案;
    /// 把六步定位压成一行)
    CrossLedger { id: u64, owner: u32, reader: u32 },
    /// 捕获事务违约(空窗/窗内非法操作/审计不一致)
    CaptureViolation { detail: String },
    /// server 不可达/已关闭
    ServerClosed,
    /// **引擎内部状态不变量失守**(无活跃 turn / 会话账缺失 / 草稿池句柄缺席……)。
    ///
    /// 语义(2026-10-17 用户裁决):这类失守是**引擎状态机已经不一致**,
    /// 但**不许 panic** —— 用显式 `Result` 一路上抛到泵循环,由它
    /// ①打印 ②丢下该 turn ③`shutdown.graceful()`(释放卡锁 + 有序退出)。
    /// `is_fatal` = true(继续跑只会连环错值/泄漏)。
    EngineInvariant { what: String },
    /// 兜底(server 侧透传的结构化细节)
    Msg(String),
}

impl ModelError {
    /// **不可恢复?**(2026-10-16) —— 与"单个请求失败"分开:
    ///
    /// - `PoolExhausted`:池/显存**硬预算拒绝**或真耗尽 ⇒ 继续跑只会连环失败;
    /// - `ServerClosed`:设备 server 已关闭 ⇒ 引擎已无设备面。
    ///
    /// 服务面据此走**优雅关停**(释放卡锁 + 有序退出),而不是把错误塞回单个 turn。
    /// 其余变体(形状/键缺失/形状不符/兜底)`= 可归因到具体请求/装载,不致命。
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            ModelError::PoolExhausted { .. }
                | ModelError::ServerClosed
                // 引擎状态不变量失守:状态机已不一致 ⇒ 与"硬预算拒发"同级处理
                | ModelError::EngineInvariant { .. }
        )
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::MissingKey { path, key } => write!(f, "MissingKey: {path}::{key}"),
            ModelError::ShapeMismatch { path, expected, got } => {
                write!(f, "ShapeMismatch: {path} 期望 {expected:?} 实得 {got:?}")
            }
            ModelError::DtypeMismatch { path, expected, got } => {
                write!(f, "DtypeMismatch: {path} 期望 {expected:?} 实得 {got:?}")
            }
            ModelError::PoolExhausted { pool, needed, available } => {
                write!(f, "PoolExhausted: {pool} 需 {needed}B 余 {available}B")
            }
            ModelError::DeadBlock { id } => write!(f, "DeadBlock: {id}"),
            ModelError::CrossLedger { id, owner, reader } => {
                write!(f, "跨账本块: 块 id={id} 属 ledger {owner},被 ledger {reader} 解析")
            }
            ModelError::CaptureViolation { detail } => write!(f, "CaptureViolation: {detail}"),
            ModelError::ServerClosed => write!(f, "ServerClosed"),
            ModelError::EngineInvariant { what } => {
                write!(f, "EngineInvariant: 引擎状态不变量失守 —— {what}")
            }
            ModelError::Msg(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ModelError {}

// ============================================================================
// 线格式:发射消息 + 池块句柄
// ============================================================================

/// pinned 主机缓冲(DMA 源;流式装载租约的统一视图)。
/// **字节口径**(2026-09-26 f16 基线尾批:装载流水恢复)—— 租约不绑
/// 元素位宽,f16/f32 装载路径同一条流水;位宽语义归数据源写入侧。
pub trait PinnedRegion: Send {
    fn slice_bytes_mut(&mut self) -> &mut [u8];
    fn as_bytes(&self) -> &[u8];
}

/// 堆实现(无 pinned 能力的后端;拷贝语义同旧路径)
pub struct HeapRegion(pub Vec<u8>);

impl PinnedRegion for HeapRegion {
    fn slice_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}


// ============================================================================
// 能力契约:DeviceClient(五原语 + 图三原语;全异步)
// ============================================================================

/// 设备客户端能力契约(server 照此实现;客户端照此调用)。
///
/// 流语义:server 按业界三流模型固定路由(H2D/COMPUTE/D2H,客户端不可
/// 自创也不可见;多并发靠算子维 batching,不靠多流)——
/// htod→H2D,launch/alloc/graph→COMPUTE,dtoh→D2H,sync 排空全部。
/// 流内保序(依赖维),流间并发(传输/计算重叠维)。
///
/// **顺序语义**(2026-09-26 塔零案定谳后成文;此前 dtoh 跨流读序未定义):
/// - 三流之间**无隐式依赖**。跨流读写序由两个且仅两个途径建立:
///   ① 数据搬运命令的**回执时序**(H2D 族回执即落地;dtoh 回执即读毕);
///   ② 显式 [`DeviceClient::sync`] 栅栏。
/// - **dtoh = 读语义**:issue 相排空 COMPUTE(处理本命令时,该流此前
///   提交的全部 kernel 落定),随后 D2H 异步拷贝 —— 回执时 host 拿到的
///   是块在**全部先前已回执命令**之后的值。无序拷贝(收割与独立计算
///   重叠)列为将来显式 opt-in 原语,今日无需求不立项。
/// - **launch 只同流保序**:kernel×kernel 先后(同流)硬件保证;
///   kernel 与数据搬运的先后必须经上述 ①② 建立 —— 客户端顺序 await
///   纪律下自动成立(每个回执都是时序锚);流水线姿势(发后不等回执)
///   不在保序范围,将来需要时加显式 event 边(挂账)。
/// - H2D 族(htod/HtodChunk/UploadPinned)2026-09-26 已核:ack 全部走
///   finish 回调 —— **回执即数据落地**,后续 launch 跨流读安全。
///
/// **回执语义分级**(Future resolve 时机):
/// - `alloc` / `launch`:Ok = 提交成功 + 账房登记(fire-and-forget;
///   数据正确性由同流保序保证,GPU 侧错误 sticky 延迟暴露,在下次
///   `dtoh`/`sync` 收割)
/// - `htod` / `dtoh`:Ok = GPU 真完成(pinned 码头生命周期/数据收割
///   要求真实完成点;完成通知走 host 回调 cuLaunchHostFunc)
/// - `sync`:栅栏,三条流此前全部工作落定(全设备语义屏障)
/// **块账本一行**(调试投影;`GET /debug/blocks`)。
///
/// 为什么需要它:地址复用/区间重叠是"跨流 UAF"与"写丢失"类缺陷的**唯一钥匙**
/// (仓内 2026-10-08 E5 跨流回收律 / 2026-10-03 FI 表渐进腐败案都栽在这里)。
/// 发射账(`owl_cuda::trace`)给的是"每次命令的块 id + 地址";本投影给的是
/// **当前账本的权威快照**(id → 地址区间/字节/dtype)⇒ 可直接判
/// 「同 id 两地址」与「两活块区间相交」。
#[derive(Debug, Clone)]
pub struct BlockLedgerRow {
    pub id: u64,
    /// 账长(字节口径;块账本形态为字节,故不带 dtype)
    pub bytes: usize,
    /// 设备地址(解析当刻)
    pub addr: u64,
    /// 所属账本(actor 序号)
    pub ledger: u32,
}

/// **复合命令 `UploadAndLaunch` 的一块搬运**(字节口径;租约所有权随命令
/// 移入,server 在 DMA 完成后归还 pinned 池)。
/// 定为契约面类型:既是线格式,也是 [`DeviceClient`] 复合口的参数。
pub struct UpChunk {
    pub offset_bytes: usize,
    pub len_bytes: usize,
    pub lease: Box<dyn PinnedRegion + Send>,
}

// ============================================================================
// 状态存储的交换面(设计卷 §15;iface 声明 / store 实现 / server 使用)
// ============================================================================

/// 连接器侧的一块存储槽(store 分配号;server 只按号读写)。
///
/// **层级知识留在 store**:RAM / 磁盘 / NVMe-oF 的差别不外泄给 server ——
/// server 是唯一持裸指针的一层(裁决⑫),只负责"把字节从这里搬到那里",
/// "那里是什么"由 [`Connector`] 决定。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConnSlot(pub u64);

/// 交换方向。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapDir {
    /// 设备 → 连接器(换出/下沉)。
    Out,
    /// 连接器 → 设备(换回/回灌)。
    In,
}

/// **交换的一片**。
///
/// 一个逻辑块 = 每层片段各一片(0.8B 12 片 / 27B 32 片):设备上块是**分层**
/// 存储的,一层一片;而连接器侧一个逻辑块占**一个槽**(连续 `slot_bytes`),
/// 故两侧偏移各说各话(`dev_off` vs `conn_off`)。
#[derive(Clone, Copy, Debug)]
pub struct SwapPiece {
    /// 设备块句柄 id(server 解析地址;沿用 `dtoh_at` 的"只取 id"口径)。
    pub block: u64,
    /// 设备块内字节偏移(= 槽 × 每页字节)。
    pub dev_off: usize,
    /// 连接器槽内字节偏移(该层片段在逻辑块内的起点)。
    pub conn_off: usize,
    /// 字节长度(= 该层每页字节)。
    pub len: usize,
    /// 本片所属逻辑块的连接器槽。
    pub slot: ConnSlot,
}

/// **交换的"另一端"**(RAM / 磁盘 / …)—— 实现者是 store,使用者是 server。
///
/// 契约:
/// - `put`/`get` 只谈**槽号 + 槽内偏移 + 字节**,不得假设设备侧布局;
/// - 错误**响亮**(不许静默丢片:片错了 = 该逻辑块内容错,必须上抛让
///   调用方回滚账);
/// - 实现必须 `Send`(随命令在命令线程与本层之间转移,单主零锁)。
pub trait Connector: Send + 'static {
    /// 写入一片(server 已从设备侧拷出 `bytes`;out 语义 = 落定)。
    fn put(&mut self, slot: ConnSlot, off: usize, bytes: &[u8]) -> Result<(), ModelError>;
    /// 读出一片(server 随后上传到设备)。
    fn get(&self, slot: ConnSlot, off: usize, out: &mut [u8]) -> Result<(), ModelError>;
    /// 层名(诊断/报告面:`ram` / `disk` / …)。
    fn tier_name(&self) -> &'static str;
    /// **解箱取回**(单主语义的回收口):命令回执把 connector 原箱奉还,
    /// 调用方据此 `downcast` 回自己的具体类型(每块只是移动一个 `Vec`,零拷贝)。
    ///
    /// 实现侧写一行 `fn unbox(self: Box<Self>) -> Box<dyn Any + Send> { self }`
    /// (不能给默认实现:trait object 上 `Self` 无尺寸,默认体的 coercion 编不过;
    /// 具体实现里 `Self: Sized` 恒成立)。
    fn unbox(self: Box<Self>) -> Box<dyn std::any::Any + Send>;
}

/// 上传落点流(**参数面 vs 批量面**;见 [`DeviceClient::write_block_f32_on`])。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum UpStream {
    /// H2D 流:批量面默认(装载/换页/材料);跨流 ⇒ 消费方需事件序
    #[default]
    H2d,
    /// COMPUTE 流:**参数面**(图输入槽/小表)—— 与消费它的发射同流 FIFO
    Compute,
}

/// **FI plan 表同步请求**(W3;见 [`DeviceClient::fi_plan_sync`])。
///
/// 为什么要它:FI 核进捕获图后,**回放期不再经过 FI 的宿主处理器** ⇒ 那张
/// 调度表(`int_ws`)没人刷。而 split 档的表必须与**当步真实 ctx** 对应
/// (运行期 `num_kv_chunks = ceil_div(kv_len, chunk_size)` 与表里的
/// `o_indptr/merge_indptr` 不同步 ⇒ partial 行 stride 错;实测症状 = 输出
/// 64 个空 token)。⇒ 引擎在每次图 launch **之前**显式请求一次刷新
/// (同一 COMPUTE 流 ⇒ FIFO 有序)。
#[derive(Clone, Copy, Debug)]
pub struct FiPlanSpec {
    /// 本轮 q 行数(decode = 1;prefill 桶 = 桶 t)
    pub t: usize,
    /// 本轮 KV 总长(plan 键的 ctx 维;**必须给真值**)
    pub ctx_total: usize,
    pub hq: usize,
    pub hkv: usize,
    pub hd: usize,
    pub page: usize,
}

pub trait DeviceClient: Send {
    /// 执行环境提供(硬件感知;None = 无 GPU 环境,语义调用结构化拒绝)。
    /// 感知发生在引擎/boot 侧(实现者可缓存);kernels/解释器零探测
    /// (被动律)。page 项由解释器从页策略单源补齐,实现者可置 0。
    fn op_env(&self) -> Option<owl_kernels::driver::OpEnv> {
        None
    }

    /// 能力自述:能否发射设备侧 kernel(Launch / DeviceRearrange 装载臂)。
    /// 默认 false(纯 host face);GPU 后端覆写 true。装载域 repack
    /// 路径裁决([`crate::module::RepackPath::resolve`] 在 models 侧)的
    /// 能力否决项由此驱动:false 时量化键自动回退 CPU 打包,无需 env。
    fn device_kernels(&self) -> bool {
        false
    }

    /// 图捕获开始:进入捕获模式(此后 Launch 进图;
    /// Alloc 从捕获 slab 切块,零 cudaMalloc)。护栏:
    /// - 不可嵌套/并发捕获;捕获期 Htod/Dtoh/Sync/GraphLaunch 拒绝
    /// - 数据须在图外先物化(Block 叶子)—— warmup 契约:先 eager 跑
    ///   同一声明树,验证逻辑正确后再捕获
    fn graph_begin(&mut self) -> impl Future<Output = Result<(), ModelError>> + Send;
    /// 图捕获结束:实例化并登记,返回图 id(空捕获窗拒绝 —— 发射数须 > 0)
    fn graph_end(&mut self) -> impl Future<Output = Result<GraphId, ModelError>> + Send;
    /// 图重放(流序异步提交;写入捕获时的同一批池块 —— 块只增不减,
    /// 指针稳定,由 server 账房保证)
    fn graph_launch(
        &mut self,
        graph: GraphId,
    ) -> impl Future<Output = Result<(), ModelError>> + Send;
    /// **FI plan 表同步**(W3;仅 CUDA/FI 后端有意义):确保 `spec.ctx_total`
    /// 对应的调度表在场,并把它的表内容刷进 FI 的 `int_ws`。
    ///
    /// 用途 = "FI 核在捕获图内 + 图不烘 plan restore"的档
    /// (`[dispatch] fi_plan_bake`):表刷新必须由图**外面**的一步完成
    /// (图回放不会经过宿主处理器)。缺省实现 = **结构化拒**(不静默)。
    fn fi_plan_sync(
        &mut self,
        spec: FiPlanSpec,
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        let _ = spec;
        async {
            Err(ModelError::Msg(
                "fi_plan_sync: 本后端无 FI plan 面(仅 CUDA + flashinfer 族)".into(),
            ))
        }
    }

    /// **捕获窗作废**(窗内出错/空窗的收尾):退出捕获态 + 销毁半成品图 +
    /// **窗内自产块整批作废**(slab 随之释放)。
    ///
    /// 为什么必须有:`GraphPlan` 的捕获窗一旦出错(声明违约/launch 失败/
    /// slab 耗尽),窗内已切出的块就处于"无图可回放但已在账房"的孤儿态 ⇒
    /// 不做 abort 则**整条 slab 钉到进程结束**(最坏 GB 级,见设计 §4.3c)。
    /// 缺省实现 = 结构化拒(无捕获能力的 face 永不进入此路径)。
    fn graph_abort(&mut self) -> impl Future<Output = Result<(), ModelError>> + Send {
        async { Err(ModelError::Msg("graph_abort: 该 face 无图捕获能力".into())) }
    }
    /// **销毁已登记图**(单图退休:LRU 驱逐 / plan `close()` 的底层)。
    /// 语义 = 排空回放 → 解除本图块租约 → 本图 slab 块移出账房。
    /// 缺省实现 = 幂等空操作(无图后端语义上"无可销毁")。
    fn graph_destroy(
        &mut self,
        _graph: GraphId,
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        async { Ok(()) }
    }
    /// 清零分配:块登记 + COMPUTE 流序 memset 入队(异步;捕获期 slab
    /// 切块同款清零,Op::Zeros 语义的设备侧保证)。f16 基线(2026-09-26):
    /// dtype + elems 双参,字节宽由客户端按 size_bytes 换算 —— 传输面
    /// 不再假设 4 字节元素(旧 alloc(n_bytes)/4 在 f16 下静默减半,雷已拆)。
    /// 返回 Bytes.len = **元素数**口径(C1 断言/账本不变)。
    ///
    /// **`name` = 分配点点名(必填;全量记名,2026-10-16)**:核名/层名/
    /// 装载键名 —— 传"谁要这块"的**语义名**,不传尺寸/相位(那些账上有)。
    /// 为什么必填不默认:OOM dump 的"按名清单"(块账房按名聚合)就是
    /// 靠它;默认值会让漏报静默(空名块在 dump 里单独计 "未命名!",
    /// 那是漏网信号,不是合法状态)。
    fn alloc(&mut self, dtype: Dtype, elems: usize, name: &str)
        -> impl Future<Output = Result<Bytes, ModelError>> + Send;

    /// 未初始化分配(S1,2026-09-30):scratch 输出块 = 随后 kernel
    /// **全量覆写**,不需要零化。GPU 后端覆写为无 memset 分配 ——
    /// 捕获窗内不烙 memset 节点(S1 实测定谳:decode 图 1207 memset
    /// 节点 = launch 税 + ~2ms GPU 时间双税),图外免流序清零。
    /// 默认 = 委托 [`DeviceClient::alloc`](零化语义,保守正确)。
    /// **契约**:调用方保证 kernel 全量覆写后才读;Zeros 语义走
    /// [`DeviceClient::alloc`](零化)。
    fn alloc_uninit(
        &mut self,
        dtype: Dtype,
        elems: usize,
        name: &str,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send
    where
        Self: Sized,
    {
        async move { self.alloc(dtype, elems, name).await }
    }
    /// host 数据入块(H2D 流异步拷贝 + finish 回调携 ack):**回执即数据
    /// 落地**,后续 launch 跨流读安全(顺序语义,见 trait 级文档)。
    /// `name` 同 [`DeviceClient::alloc`](h2d 落地也是一次块出生)。
    fn htod(
        &mut self,
        dtype: Dtype,
        shape: &Shape,
        src: &[u8],
        name: &str,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send;

    /// **带流选择的整块上传**(2026-10-17):参数面(解释器 `from_host` 叶子 ——
    /// 小张量、马上被发射消费)走 [`UpStream::Compute`] ⇒ 同流 FIFO 天然有序;
    /// 批量面(权重/装载)走 H2D 以免占住计算流。默认退化为 [`Self::htod`]。
    fn htod_on(
        &mut self,
        _stream: UpStream,
        dtype: Dtype,
        shape: &Shape,
        src: &[u8],
        name: &str,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send
    where
        Self: Sized,
    {
        self.htod(dtype, shape, src, name)
    }

    /// 块读出(**读语义**,2026-09-26 成文):issue 相排空 COMPUTE 后
    /// D2H 异步拷贝 —— 回执时 out = 块在全部先前已回执命令之后的值。
    /// 与 [`DeviceClient::sync`] 的分工:本命令为读数据;sync 为不读
    /// 数据的显式栅栏(计时/步边界/捕获前净空)
    fn dtoh(&mut self, b: &Bytes, out: &mut [u8])
        -> impl Future<Output = Result<(), ModelError>> + Send;
    /// **带偏移的设备→主机读**(状态存储的搬出面,2026-10-16):
    /// `dtoh` 只能读块前缀,而"逻辑块 = 层缓冲内的一个页"必须按**槽偏移**读
    /// (`off_bytes = slot × 每页字节`)⇒ 补这条。语义与 [`DeviceClient::dtoh`]
    /// 同(读语义:回执时 out = 块在全部先前已回执命令之后的那一段)。
    ///
    /// 默认 = **响亮"未实现"**(后端逐个补齐;CPU 面已给实现供 store 单测)。
    /// 批面(`DtohBatch`,一次命令搬一个块的全部层片段)是它的下一步 —— 见
    /// `docs/arch/store-状态存储-设计与迁移.md` §10.3 推论 B。
    fn dtoh_at(
        &mut self,
        _b: &Bytes,
        _off_bytes: usize,
        _out: &mut [u8],
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        async { Err(ModelError::Msg("dtoh_at: 此后端未实现".into())) }
    }
    /// **批量分块读**(状态存储"整块搬出",2026-10-16):一次命令搬 N 段
    /// (`(块, 偏移字节, 长度字节)`),回执 = 各段按**入参顺序**拼接的字节。
    ///
    /// 为什么必须批:一个逻辑块 = 层数 × 2 个片段(0.8B 12 段 / 27B 32 段),
    /// 不批则每次搬出 12/32 次 actor 往返,收益被往返吃光(设计卷 §10.3 推论 B)。
    /// 默认实现 = 逐段 [`DeviceClient::dtoh_at`] 拼接(语义等价;GPU 面覆写为
    /// 单命令 `DtohBatch`)。
    fn dtoh_batch(
        &mut self,
        pieces: &[(&Bytes, usize, usize)],
    ) -> impl Future<Output = Result<Vec<u8>, ModelError>> + Send
    where
        Self: Sized,
    {
        #[allow(clippy::unnecessary_to_owned)]
        let owned: Vec<(Bytes, usize, usize)> =
            pieces.iter().map(|(b, o, l)| ((*b).clone(), *o, *l)).collect();
        async move {
            let mut out = Vec::new();
            for (b, off, len) in &owned {
                let at = out.len();
                out.resize(at + *len, 0);
                self.dtoh_at(b, *off, &mut out[at..]).await?;
            }
            Ok(out)
        }
    }

    /// **批量分块写**(host → device,带偏移;状态存储"整块搬回")。
    /// 默认 = 逐段 pinned 租约上传(语义等价;GPU 面覆写为单命令 `HtodChunks`)。
    fn htod_batch(
        &mut self,
        writes: &[(&Bytes, usize, &[u8])],
    ) -> impl Future<Output = Result<(), ModelError>> + Send
    where
        Self: Sized,
    {
        #[allow(clippy::unnecessary_to_owned)]
        let owned: Vec<(Bytes, usize, Vec<u8>)> = writes
            .iter()
            .map(|(b, off, data)| ((*b).clone(), *off, data.to_vec()))
            .collect();
        async move {
            for (b, off, data) in &owned {
                let mut lease = self.alloc_pinned(data.len(), "iface:htod_batch").await?;
                lease.slice_bytes_mut()[..data.len()].copy_from_slice(data);
                self.upload_pinned_on(UpStream::Compute, lease, b, *off, data.len())
                    .await?;
            }
            Ok(())
        }
    }

    /// **swap = 一条原子命令**(设计卷 §15;2026-10-16):N 片一次提交、一次 ack。
    ///
    /// 语义:
    /// - **一次 ack = 全部片段完成**(同流保序)⇒ store 侧"落账"只需一次 await;
    /// - `conn` 随命令**进出**(移入、回执时原箱奉还)⇒ 单主、零锁;
    /// - 中途失败 = **整条失败**(不得留半态;调用方负责回滚账 / 响亮上抛);
    /// - 捕获窗内 = 结构化拒(时序归 engine,§10.3)。
    ///
    /// 层级无关:RAM 与 disk 只差 [`Connector`] 实现,server 侧代码零分支。
    /// 默认实现 = 逐片 `dtoh_at` / `alloc_pinned + upload_pinned` 循环
    /// (**语义等价**,任何 face 都能跑);GPU 面覆写为单命令 `Swap`。
    ///
    /// 回执奉还的 connector 用 [`Connector::unbox`] 取回具体类型。
    fn swap(
        &mut self,
        dir: SwapDir,
        pieces: &[SwapPiece],
        conn: Box<dyn Connector + Send>,
    ) -> impl Future<Output = Result<Box<dyn Connector + Send>, ModelError>> + Send
    where
        Self: Sized,
    {
        let owned: Vec<SwapPiece> = pieces.to_vec();
        async move {
            let mut conn = conn;
            // 句柄只取 id(与 `dtoh_at` 同口径 —— 字节级搬运不参与 dtype 校验)
            let handle = |p: &SwapPiece| Bytes::new(p.block, 0, Dtype::U8, 0);
            match dir {
                SwapDir::Out => {
                    for p in &owned {
                        let mut buf = vec![0u8; p.len];
                        self.dtoh_at(&handle(p), p.dev_off, &mut buf).await?;
                        conn.put(p.slot, p.conn_off, &buf)?;
                    }
                }
                SwapDir::In => {
                    for p in &owned {
                        let mut buf = vec![0u8; p.len];
                        conn.get(p.slot, p.conn_off, &mut buf)?;
                        let mut lease = self.alloc_pinned(p.len, "iface:swap_in").await?;
                        lease.slice_bytes_mut()[..p.len].copy_from_slice(&buf);
                        self.upload_pinned_on(UpStream::Compute, lease, &handle(p), p.dev_off, p.len)
                            .await?;
                    }
                }
            }
            Ok(conn)
        }
    }

    /// 算子发射(COMPUTE 流,火后不理):Ok = 提交成功;同流保序,
    /// 与数据搬运的跨流先后靠回执时序/sync 建立(顺序语义,见 trait 级文档)
    fn launch(&mut self, msg: LaunchMsg)
        -> impl Future<Output = Result<Bytes, ModelError>> + Send;

    /// 全设备栅栏:三条流此前全部工作落定方回执(计时/步边界/捕获前净空)
    fn sync(&mut self) -> impl Future<Output = Result<(), ModelError>> + Send;

    /// 中间块回收(E2a,2026-09-27):调用方声明这批块此后不再引用。
    /// 安全前提:回收前已收割所需数据(dtoh 回执即 COMPUTE 已排空);
    /// 图捕获期禁止(捕获块 = slab 切片,生命周期随图)。未知 id 容错
    /// 跳过(幂等)。默认 = 无操作(无块账房的后端,如 CPU 面)。
    fn free(&mut self, _ids: &[u64]) -> impl Future<Output = Result<(), ModelError>> + Send {
        async move { Ok(()) }
    }

    /// f32 直传(流式装载,2026-09-26):owned Vec<f32> **所有权移入**
    /// —— 消费端(装载域)的数据免 f32→LE→f32 字节往返,move 进消息
    /// 后随 server 消费消亡。默认 = LE 编码走 [`DeviceClient::htod`]
    /// (字节世界后端零改动);eval 域的 Htod 声明仍走旧 htod。
    fn htod_f32(
        &mut self,
        shape: &Shape,
        data: Vec<f32>,
        name: &str,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send
    where
        Self: Sized,
    {
        async move {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            self.htod_on(UpStream::Compute, Dtype::F32, shape, &bytes, name).await
        }
    }

    /// pinned 主机缓冲租约(流式装载的 DMA 源;转换直写 → move 上传)。
    /// `_name` = 租约点名(全量记名纪律;本缺省实现是 host 占位面,不建账
    /// ⇒ 有意不存;GPU 面由 `GpuClient` 把名字带进 `Command::AllocPinned`)
    fn alloc_pinned(
        &mut self,
        bytes: usize,
        _name: &str,
    ) -> impl Future<Output = Result<Box<dyn PinnedRegion + Send>, ModelError>> + Send
    where
        Self: Sized,
    {
        async move { Ok(Box::new(HeapRegion(vec![0u8; bytes])) as _) }
    }

    /// pinned 租约上载(**异步语义**,F5 尾批):提交即回执,DMA 完成由
    /// 调用方显式 sync 栅栏兜底(计算读前必 sync);租约所有权移入 server,
    /// 完成回调归还池。在途块数调用方自限(防码头膨胀)。
    /// 上传租约:buf 所有权移入,DMA/拷贝到 dst+offset_elems;
    /// buf 由后端回收(页锁池/释放)。默认 = 不支持。
    /// [`Self::upload_pinned`] 的**流可选**版(2026-10-17;参数面走 Compute)
    fn upload_pinned_on(
        &mut self,
        _stream: UpStream,
        buf: Box<dyn PinnedRegion + Send>,
        dst: &Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> impl Future<Output = Result<(), ModelError>> + Send
    where
        Self: Sized,
    {
        self.upload_pinned(buf, dst, offset_bytes, len_bytes)
    }

    fn upload_pinned(
        &mut self,
        _buf: Box<dyn PinnedRegion + Send>,
        _dst: &Bytes,
        _offset_bytes: usize,
        _len_bytes: usize,
    ) -> impl Future<Output = Result<(), ModelError>> + Send
    where
        Self: Sized,
    {
        async { Err(ModelError::Msg("upload_pinned: 此后端未实现".into())) }
    }

    /// 块内分块写入(流式装载,2026-09-26):向已 alloc 的块在
    /// offset_elems 处写入 data —— 大张量 128MB 分块流式上传的执行面,
    /// 装载域主机在途只余单块。默认 = 不支持。
    fn write_block_f32(
        &mut self,
        _dst: &Bytes,
        _offset_elems: usize,
        _data: &[f32],
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        async {
            Err(ModelError::Msg("write_block_f32: 此后端未实现".into()))
        }
    }

    /// **带流选择的块内写入**(2026-10-17;用户裁决"参数面必须与消费方同流"):
    ///
    /// 上传落在哪条流 = **顺序义务的归属**。默认口走 H2D 流(批量面:装载/
    /// 换页,跨流 ⇒ 消费方需事件序);**参数面**(图输入槽/小表 —— 只有几 KB、
    /// 且紧接着就被同一次发射消费)应走 [`UpStream::Compute`]:与 kernel/图发射
    /// **同流 FIFO** ⇒ 结构上不可能缺序,同时省掉"等回执"的主机往返。
    ///
    /// 默认实现退化到老口(其他后端/mock 零改动)。
    fn write_block_f32_on(
        &mut self,
        _stream: UpStream,
        dst: &Bytes,
        offset_elems: usize,
        data: &[f32],
    ) -> impl Future<Output = Result<(), ModelError>> + Send
    where
        Self: Sized,
    {
        self.write_block_f32(dst, offset_elems, data)
    }

    /// [`Self::write_block_f32_on`] 的批量版(同律)
    fn write_blocks_f32_on(
        &mut self,
        _stream: UpStream,
        writes: &[(&Bytes, usize, &[f32])],
    ) -> impl Future<Output = Result<(), ModelError>> + Send
    where
        Self: Sized,
    {
        self.write_blocks_f32(writes)
    }

    /// 批量块内写入(S1,2026-09-30):decode 每步多槽装填合一次设备
    /// 往返(单 ack 保序协议不变 —— ack = 全部写完)。默认 = 逐块循环
    /// write_block_f32;GPU 后端覆写为批量命令(省 per-write 回执税)。
    fn write_blocks_f32(
        &mut self,
        writes: &[(&Bytes, usize, &[f32])],
    ) -> impl Future<Output = Result<(), ModelError>> + Send
    where
        Self: Sized,
    {
        async move {
            // 参数面(槽/小表)批量写 ⇒ Compute(同流;H2D 面留给显式批量口)
            for (dst, off, data) in writes {
                self.write_block_f32_on(UpStream::Compute, dst, *off, data).await?;
            }
            Ok(())
        }
    }

    /// 块清零(设备侧 memset;2026-09-26 引擎 turn-open 定谳:GDN 状态
    /// 1.2GB 走 host 往返清零 = 14s,memset 后 ~ms 级)。默认 = 不支持。
    fn memset_zero(
        &mut self,
        _dst: &Bytes,
        _len_bytes: usize,
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        async { Err(ModelError::Msg("memset_zero: 此后端未实现".into())) }
    }

    /// **复合口(E3):上传并计算** —— 一次(可分块)搬运 + 一次发射,两者之间的
    /// 序由实现方保证(**设备端事件序**,零 host 阻塞)。需要跨流序的调用点
    /// 只用本口,不自己拼"上传 + 栅栏 + 发射"。
    ///
    /// 默认不支持(只有能跨流定序的 face 实现;天然同步的后端无需它)。
    fn upload_and_launch(
        &mut self,
        _dst: &Bytes,
        _chunks: Vec<UpChunk>,
        _msg: LaunchMsg,
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send {
        async { Err(ModelError::Msg("upload_and_launch: 此后端未实现".into())) }
    }

    /// **复合口(E3):计算并回读** —— 一次发射 + 一次回读;序同样由实现方保证
    /// (发射 → 等该流落定 → 回读),**禁退化成 host 全排空**。默认不支持。
    fn launch_and_harvest(
        &mut self,
        _msg: LaunchMsg,
        _id: u64,
        _want_bytes: usize,
    ) -> impl Future<Output = Result<Vec<u8>, ModelError>> + Send {
        async { Err(ModelError::Msg("launch_and_harvest: 此后端未实现".into())) }
    }

    /// **块账本投影**(调试面;默认不支持):全活块 (id, 字节, dtype, 地址, 账本)。
    fn block_ledger_dump(&mut self) -> impl Future<Output = Result<Vec<BlockLedgerRow>, ModelError>> + Send {
        async { Err(ModelError::Msg("block_ledger_dump: 此后端未实现".into())) }
    }

    /// **设备池回收**(内存面;默认不支持):把 `cudaMallocAsync` 池里
    /// **已 free 但仍被池缓存**的块还给 OS,回释放字节数。
    ///
    /// 为什么需要:池属性 `RELEASE_THRESHOLD=UINT64_MAX`(性能面:不还 OS)
    /// 意味着**装载期暂存/临时块 free 之后仍占卡**(nvidia-smi 计入 used)。
    /// 27B AWQ 单卡实测:装载收尾(暂存 free)后跑图期峰值贴顶,0.6MiB 分配
    /// `CUDA_ERROR_OUT_OF_MEMORY`(2026-10-16 组测确定性复现)⇒ 装载收尾
    /// 调一次本口,把 156MB 级"死账"还给 OS。
    fn mem_pool_trim(&mut self) -> impl Future<Output = Result<u64, ModelError>> + Send {
        async { Err(ModelError::Msg("mem_pool_trim: 此后端未实现".into())) }
    }

    /// 设备内块→块拷贝(E2c:GDN 快照拍摄/恢复 —— 会话格 ↔ 快照缓冲
    /// D2D,免 PCIe 往返;双方偏移字节口径,长度 = 拷贝字节数)。
    /// 默认 = 不支持。
    fn copy_block_at(
        &mut self,
        _src: &Bytes,
        _src_off_bytes: usize,
        _dst: &Bytes,
        _dst_off_bytes: usize,
        _len_bytes: usize,
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        async {
            Err(ModelError::Msg("copy_block_at: 此后端未实现".into()))
        }
    }

    /// 面句柄克隆(E5-M4:verify 图第二个 GraphPlan 需要独立 face 句柄;
    /// 仅可安全克隆的句柄后端覆写 —— GpuClient = actor 发送端 Arc,CPU
    /// 面不可克隆)。默认 = None(引擎侧回落 eager verify 路)。
    fn face_clone(&self) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

    /// **设备 SM(多处理器)数** —— 计算单元真值。
    ///
    /// 用途:XQA 的 KV-split 启发式参数 `sm_count`(原为引擎里硬编码
    /// `XQA_SM_COUNT = 84` = sm86 双卡;3080 = 68 ⇒ 启发式系统性偏大)。
    /// 语义:同步小查询(后端可缓存);**默认 = 不支持** ⇒ 结构化拒(不猜数字),
    /// 调用方自行决定回退(引擎面回退到既有常量并**响亮告警**)。
    fn sm_count(&mut self) -> impl Future<Output = Result<usize, ModelError>> + Send
    where
        Self: Sized,
    {
        async { Err(ModelError::Msg("sm_count: 该后端未提供设备属性查询".into())) }
    }

    /// 设备内批量块→块拷贝(E5-M4:spec 快照 capture/restore 的 192 次
    /// 逐块往返收敛为一次命令;单 ack = 全部已入流)。默认 = 逐条
    /// copy_block_at(CPU face / 未覆写后端语义等价)。
    fn copy_batch(
        &mut self,
        copies: &[(Bytes, usize, Bytes, usize, usize)],
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        // FIXME(clippy::unnecessary_to_owned):此处不能改 `copies.iter().cloned()` ——
        // 返回的 future 在 2021 edition RPIT 下不捕获 `copies` 的生命周期,
        // 必须先 clone 成 owned 迭代器才能移入 async 块。allow 而非改写。
        #[allow(clippy::unnecessary_to_owned)]
        let mut it: std::vec::IntoIter<(Bytes, usize, Bytes, usize, usize)> =
            copies.to_vec().into_iter();
        async move {
            for (src, so, dst, doff, len) in it.by_ref() {
                self.copy_block_at(&src, so, &dst, doff, len).await?;
            }
            Ok(())
        }
    }

    /// 带偏移的块清零(E2b:GDN 状态按格重置 —— 格 = 块内等分行,
    /// offset_bytes = 格号 × 行字节;多会话各清各格,互不踩)。默认 =
    /// 委托整块 memset_zero(offset = 0 时语义等价)
    fn memset_zero_at(
        &mut self,
        dst: &Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> impl Future<Output = Result<(), ModelError>> + Send {
        async move {
            if offset_bytes == 0 {
                return self.memset_zero(dst, len_bytes).await;
            }
            Err(ModelError::Msg("memset_zero_at: 此后端未实现".into()))
        }
    }

    /// 装载并发句柄(可选能力,2026-09-26 M-e loader 性能):返回 k 个
    /// 可独立驱动的 client 句柄克隆 —— 多协程各持一个,host 侧布局变换
    /// 与设备拷贝在 server 线程上流水重叠。默认 None = 顺序装载(单 face)。
    fn loader_faces(&self, _k: usize) -> Option<Vec<Self>>
    where
        Self: Sized,
    {
        None
    }
}
