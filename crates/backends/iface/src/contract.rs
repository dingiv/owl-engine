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
// 标注词汇:Dtype + Shape(线格式的元数据维)
// ============================================================================

/// 数据类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    BF16,
    F16,
    U32,
}

impl Dtype {
    /// 字节宽(server 的分配只认字节;宽是 client 侧换算用的)
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::U32 => 4,
        }
    }
}

/// 形状(行主序;一维 = vec![n])
pub type Shape = Vec<usize>;

/// 元素总数
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

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
    /// 捕获事务违约(空窗/窗内非法操作/审计不一致)
    CaptureViolation { detail: String },
    /// server 不可达/已关闭
    ServerClosed,
    /// 兜底(server 侧透传的结构化细节)
    Msg(String),
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
            ModelError::CaptureViolation { detail } => write!(f, "CaptureViolation: {detail}"),
            ModelError::ServerClosed => write!(f, "ServerClosed"),
            ModelError::Msg(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ModelError {}

// ============================================================================
// 线格式:发射消息 + 池块句柄
// ============================================================================

/// 图身份证(server 签发;graph_end 成功后可 graph_launch 重放)
pub type GraphId = u64;

/// 池块句柄:server 签发的身份证(id → server 账房 → 显存)。
#[derive(Clone, Debug)]
pub struct Bytes {
    pub id: u64,
    /// 元素数(f32)
    pub len: usize,
}

impl Bytes {
    pub fn new(id: u64, len: usize) -> Self {
        Self { id, len }
    }
}

/// kernel 描述:入口名 + 源码。后端按 (源码哈希, 名) 懒编译缓存。
#[derive(Clone, Debug)]
pub struct KernelSpec {
    pub name: String,
    pub source: String,
}

/// 发射参数槽(有序;与 kernel 签名严格对位)
/// 类型化:标量按 kernel 形参宽度入槽(CUDA 参数空间自然对齐,
/// 8 字节槽顶 4 字节形参会错位读参 —— 坑 I)
#[derive(Clone, Debug)]
pub enum Arg {
    Block { id: u64 },
    U64(u64),
    I32(i32),
    F32(f32),
}

/// 发射消息
pub struct LaunchMsg {
    pub kernel: KernelSpec,
    pub args: Vec<Arg>,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
    pub out_elems: usize,
}

// ============================================================================
// 能力契约:DeviceClient(五原语 + 图三原语;全异步)
// ============================================================================

/// 设备客户端能力契约(server 照此实现;客户端照此调用)。
///
/// 流语义:server 按业界三流模型固定路由(H2D/COMPUTE/D2H,客户端不可
/// 自创也不可见;多并发靠算子维 batching,不靠多流)——
/// htod→H2D,launch/alloc/graph→COMPUTE,dtoh→D2H,sync 排空全部。
/// 流内保序(依赖维),流间并发(传输/计算重叠维)。跨流依赖(事件边)
/// 待服务端扩展。
///
/// **回执语义分级**(Future resolve 时机):
/// - `alloc` / `launch`:Ok = 提交成功 + 账房登记(fire-and-forget;
///   数据正确性由同流保序保证,GPU 侧错误 sticky 延迟暴露,在下次
///   `dtoh`/`sync` 收割)
/// - `htod` / `dtoh`:Ok = GPU 真完成(pinned 码头生命周期/数据收割
///   要求真实完成点;完成通知走 host 回调 cuLaunchHostFunc)
/// - `sync`:栅栏,该流此前所有工作全部落定
pub trait DeviceClient: Send {
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
    fn alloc(&mut self, n_bytes: usize)
        -> impl Future<Output = Result<Bytes, ModelError>> + Send;
    fn htod(
        &mut self,
        dtype: Dtype,
        shape: &Shape,
        src: &[u8],
    ) -> impl Future<Output = Result<Bytes, ModelError>> + Send;
    fn dtoh(&mut self, b: &Bytes, out: &mut [u8])
        -> impl Future<Output = Result<(), ModelError>> + Send;
    fn launch(&mut self, msg: LaunchMsg)
        -> impl Future<Output = Result<Bytes, ModelError>> + Send;
    fn sync(&mut self) -> impl Future<Output = Result<(), ModelError>> + Send;
}
