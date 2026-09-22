//! owl-iface —— 后端硬件抽象层(HAL)。
//!
//! 分层契约(charter §四 + REQ-HW-01 的后端级推广):
//!
//! ```text
//! core / scheduler      ← 只认识 iface 的 trait 与词汇类型
//!        ↓ 静态分发(泛型,零虚表)
//! backends/cuda         ← impl Backend(CUDA 厂商栈)
//! backends/rocm ...     ← 未来厂商栈,注册即可,上层零改动
//!
//! Backend(厂商栈) 1 : N Device(卡)
//! Device = 账本 + 池组 + 相位机的唯一宿主(设备隔离律)
//! ```
//!
//! 三元关系(2026-09-22 裁决,修正版):
//! - **Backend = 厂商硬件操作 API 家族**(CUDA 栈/ROCm 栈):负责枚举
//!   设备、按 UUID 打开设备;不记账;
//! - **Device = 某厂商栈下的一个具体硬件实例**:一卡 = 一 Device =
//!   一个独立账本 + 一组池 + 一个相位机;一切分配原语在 Device 上;
//! - **Pool = Device 账本里的容量承诺**(A5.1 分解表一行),Buf ∈ Pool。
//!
//! 设计律:
//! - **选后端是配置期决策**(REQ-CODE-01):主路径静态分发,关联类型
//!   `Persistent<T>/Scratch<T>` 由各后端具体化;不做热路径 dyn。
//! - **词汇类型归 iface**:MemPhase/MemStats/BackendError 定义在此,
//!   后端实现只许实现语义,不许另造词汇(否则上层就要写 match 适配)。
//! - **buffer 契约最小面**:DevBuf 只暴露 len/device_ptr/域标记;
//!   后端特有的能力(如 cuda 的 stream 句柄)走下沉扩展 trait,
//!   不污染通用契约。

/// 后端 Arch 标识(与 kernels 的 arch 分发表共用一套词汇)
pub use owl_kernels::Arch;

/// 缓冲令牌:(id, generation) 二元组。词汇权威在 owl-signal
/// (`owl_signal::Token`,后端无关;rocnn 复用同一定义),
/// 此处做类型别名保持 iface 词汇名稳定。
/// 捕获期记录于 CaptureRecord(哨兵①),replay 前可校验存活
/// (世代校验;死亡令牌 = 结构化报错而非 Xid 盲死)。
pub type BufToken = owl_signal::Token;

/// 厂商后端栈家族
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFamily {
    Cuda,
    // Rocm(未来)
}

/// 设备描述:一卡一身份。UUID 是唯一合法标识(数字序/枚举序禁,
/// 09-19 混插对事故判例)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDesc {
    /// PCI 总线 UUID(如 "GPU-e565c505-…"),与卡物理身份绑定
    pub uuid: String,
    pub arch: Arch,
    /// 设备总显存(bytes)
    pub total_bytes: u64,
}

/// 设备隔离律(A5 的多卡推广,2026-09-22 裁决):
///
/// 1. **一个后端实例 = 一张卡 = 一个独立账本**。后端实现必须把全部
///    记账(预算/池/延迟释放/审计)绑定到自己的唯一设备上;
/// 2. Device 实例**禁止**在初始化后再触碰第二张卡(连查询都不行);
///    多卡编排是 ProcessGroup 层的事,Device 内不存在"跨卡"概念;
/// 3. 跨卡通信 = 每卡的 comm 出口互相握手(ProcessGroup 层),
///    通信缓冲计入各自卡的账本;
/// 4. 账本外分配(包括集合通信库的内部缓冲)按
///    backends/README.md 准入铁律处理:**NCCL 因此除名**。

/// 内存域状态机(与 graph 治理的 GraphPhase 同律;词汇权威在此)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemPhase {
    /// 无图存活:延迟释放在此窗口统一归还
    #[default]
    Idle,
    /// 捕获中
    Capturing,
    /// 已实例化未定影(懒提交窗口;禁 replay,A1.2 定影协议)
    Captured,
    /// 图存活
    Live,
}

/// 内存审计计数(可归因的内存漂移)
#[derive(Debug, Default, Clone, Copy)]
pub struct MemStats {
    pub persistent_allocs: u64,
    pub scratch_allocs: u64,
    /// 图存活期被延迟释放的缓冲数
    pub deferred_frees: u64,
    /// 已归还的延迟释放数
    pub drained_frees: u64,
    /// A2.6 窄口:跨卡远端映射笔数与字节(对端显存的本地映射,本卡记账)
    pub peer_mappings: u64,
    pub peer_mapped_bytes: u64,
}

/// 可上卡值类型的 iface 侧标记(后端实现自行映射到其 driver 要求,
/// 如 cuda 的 DeviceRepr+ValidAsZeroBits)
pub trait MemValue: Send + 'static {}
impl<T: Send + 'static> MemValue for T {}

/// 设备缓冲通用契约:上层能问大小与设备指针,足够喂 kernel 与记账。
pub trait DevBuf<T>: Send {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// 设备侧裸指针(kernel 发射/图捕获参数用;解引用归后端语义)
    fn device_ptr(&self) -> *mut T;
}

// ---- 显存池(A5.1 预算分解项的实体化)----

/// 池句柄(后端分配,进程内唯一)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolId(pub u64);

/// 池用途:对应 A5.1 预算分解的每一行——预算表里的每一笔账
/// 都必须落到一个具名池上,没有"通用池"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    /// 模型权重驻留
    Weights,
    /// KV cache 页
    KvCache,
    /// 图捕获/replay(绑定 A1.1 graph_allowance)
    Graph,
    /// kernel 暂存(峰值按 batched_tokens 实测)
    Scratch,
    /// 通信缓冲 / cuBLAS workspace 等预登记件
    Workspace,
    /// 跨卡共享缓冲(A2.8:必须 VMM 分配,2MiB 粒度,BAR1 窗口预算内)
    PeerShared,
}

/// 建池配置。bytes 即该分解项的预算数字,建池 = 预留承诺。
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub name: String,
    pub kind: PoolKind,
    /// 池容量上限(bytes)。池内分配超此限即 A5.4 违约。
    pub bytes: u64,
}

/// 池用量快照(审计/对账)
#[derive(Debug, Clone, Copy)]
pub struct PoolUsage {
    pub capacity: u64,
    pub used: u64,
    /// 生命周期峰值(漂移归因用)
    pub peak: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// A5.4:目标池容量不足——分解项超支,fail-fast
    #[error("A5.4 池 '{pool}' 耗尽:需 {needed}B,余 {available}B(容量 {capacity}B)")]
    PoolExhausted {
        pool: String,
        needed: u64,
        available: u64,
        capacity: u64,
    },
    #[error("未知池 id={0}")]
    UnknownPool(u64),
    #[error("设备初始化失败: {0}")]
    Init(String),
    #[error("分配失败({kind}, {len} 元素): {detail}")]
    AllocFailed {
        kind: &'static str,
        len: usize,
        detail: String,
    },
    #[error("拷贝失败({dir}): {detail}")]
    CopyFailed { dir: &'static str, detail: String },
    /// 生命周期律违反(charter A1.2)——架构错误
    #[error("{0}")]
    LawViolation(&'static str),
}

/// 厂商后端栈:一个硬件操作 API 家族(CUDA / ROCm / …)。
///
/// 职责仅两件:**枚举**与**打开**。不记账,不持池——账本在 Device 上。
pub trait Backend: Send + Sync + 'static {
    /// 该栈打开的设备类型(实现 Device 契约)
    type Device: Device;

    fn family(&self) -> BackendFamily;

    /// 枚举本机全部设备(按 UUID;枚举序不作为标识)
    fn enumerate(&self) -> Result<Vec<DeviceDesc>, BackendError>;

    /// 按 UUID 打开一张卡(唯一合法的打开方式)
    fn open(&self, uuid: &str) -> Result<Self::Device, BackendError>;
}

/// 设备实例:一张具体卡。账本 + 池组 + 相位机的唯一宿主(设备隔离律)。
///
/// 生命周期律(A1.2,实现者必须遵守):
/// - `Persistent` 在非 Idle 相 drop → 延迟归还,禁止立即释放;
/// - `Scratch` 允许在 Capturing 相创建(捕获安全分配),非 Idle 相
///   drop 同样延迟;
/// - 实现是否真的安全由各自测试兜底(iface 层提供验收用例模板)。
/// 池内字节缓冲(P 阶段原语产物)。
/// 不透明句柄:drop 时由后端自动归还池账 + 全局账本(含延迟语义)。
pub struct PoolBuf {
    inner: Box<dyn OpaqueDevBuf + Send>,
}

impl PoolBuf {
    pub fn wrap(inner: Box<dyn OpaqueDevBuf + Send>) -> Self {
        Self { inner }
    }

    /// 出生令牌(哨兵①)
    pub fn token(&self) -> BufToken {
        self.inner.token()
    }
}

impl DevBuf<u8> for PoolBuf {
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn device_ptr(&self) -> *mut u8 {
        self.inner.device_ptr()
    }
}

/// 后端不透明字节缓冲的 iface 视图(blanket impl,零 dyn 开销于热路径
/// 之外的 P 阶段对象上)
pub trait OpaqueDevBuf: DevBuf<u8> {
    /// 出生令牌(哨兵①:捕获期依赖记录的钥匙)
    fn token(&self) -> BufToken;
}

/// 显存池:Device 账本里的一条容量承诺(A5.1 分解表一行)。
/// **池是分配者**:`malloc` 是 P 阶段唯一分配入口;Device 只提供
/// driver 原语。上层持有池对象 = 持有"从这笔预算出账"的资格。
pub trait Pool: Send + Sync {
    /// 本池所属的设备(池↔设备归属关系类型化)
    type Dev: Device;

    /// 设备句柄(张量创建入口经池直达设备原语,调用点不再出现 Device)
    fn device(&self) -> Self::Dev;

    fn id(&self) -> PoolId;
    fn name(&self) -> &str;
    fn kind(&self) -> PoolKind;
    fn capacity(&self) -> u64;

    /// 用量快照(审计/对账;A5.3 周期调用)
    fn usage(&self) -> PoolUsage;

    // ---- 分配原语(P 阶段;按性质分立)----
    // 校验顺序:kind 语义匹配 → 池余量(A5.4)→ 全局预算(A5.4)→ 物理。
    // 池类型与分配性质不匹配 = LawViolation(语义错配是架构错误)。
    // 返回的不透明缓冲 drop 时自动归还池账 + 全局账(非 Idle 相按 A1.2 延迟)。

    /// 暂存分配:stream-ordered、清零、允许 Capturing 相创建。
    /// 仅接受 PoolKind::Scratch。
    fn malloc_scratch(&self, bytes: u64) -> Result<PoolBuf, BackendError>;

    /// 持久分配(权重/KV/workspace):地址稳定、清零、非 Idle 相 drop
    /// 延迟。接受 Weights/KvCache/Workspace。
    fn malloc_persistent(&self, bytes: u64) -> Result<PoolBuf, BackendError>;

    /// A2.8 跨卡共享分配:VMM(cuMemCreate,粒度取设备最小值,向上
    /// 取整),本地 RW 映射。仅接受 PoolKind::PeerShared。
    /// 注意:VMM 背面**不清零**,调用方负责 memset 或整块覆盖。
    fn malloc_peer_shared(&self, bytes: u64) -> Result<PoolBuf, BackendError>;
}

/// 设备实例:一张具体卡。账本 + 池组 + 相位机的唯一宿主(设备隔离律)。
///
/// 生命周期律(A1.2,实现者必须遵守):
/// - `Persistent` 在非 Idle 相 drop → 延迟归还,禁止立即释放;
/// - `Scratch` 允许在 Capturing 相创建(捕获安全分配),非 Idle 相
///   drop 同样延迟;
/// - 实现是否真的安全由各自测试兜底(iface 层提供验收用例模板)。
pub trait Device: Clone + Send + Sync + 'static {
    /// 持久域缓冲类型(权重/KV/图缓冲)
    type Persistent<T: MemValue>: DevBuf<T>;
    /// 暂存域缓冲类型(kernel 中间结果)
    type Scratch<T: MemValue>: DevBuf<T>;
    /// 跨卡远端映射视图(A2.6 例外通道的窄口产物;本卡账本记账)
    type Remote<T: MemValue>: DevBuf<T>;
    /// 池对象类型
    type Pool: Pool;

    /// 本实例绑定的卡(设备隔离律:Device : 卡 = 1:1)
    fn desc(&self) -> &DeviceDesc;

    fn arch(&self) -> Arch {
        self.desc().arch
    }

    // ---- 哨兵①词汇:缓冲令牌(默认 None;后端有账本则覆写)----

    fn persistent_token<T: MemValue>(&self, _p: &Self::Persistent<T>) -> Option<BufToken> {
        None
    }
    fn scratch_token<T: MemValue>(&self, _s: &Self::Scratch<T>) -> Option<BufToken> {
        None
    }

    // ---- 跨卡窄口(A2.6 唯一例外通道;一切映射入本卡账本)----
    /// 开启对本卡的 P2P 访问授权(peer = 对端 Device 实例,同进程)。
    /// 对称:两卡各调一次、互为 peer。授权本身不分配显存,仅记审计。
    fn enable_peer_access(&self, peer: &Self) -> Result<(), BackendError>;

    /// 把对端 Device 上某缓冲的指针映射成本地可读写的视图。
    /// 映射字节计入本卡 `peer_mapped_bytes` 账本;返回的 Remote 在
    /// drop 时销账。ptr 的来源必须是 peer 的 DevBuf::device_ptr。
    fn map_remote<T: MemValue>(
        &self,
        peer: &DeviceDesc,
        ptr: *mut T,
        len: usize,
    ) -> Result<Self::Remote<T>, BackendError>;

    // ---- 相位(A1.2 状态机镜像;真实状态机在 graph 治理层) ----
    fn phase(&self) -> MemPhase;
    fn set_phase(&self, phase: MemPhase);

    // ---- 审计 ----
    fn stats(&self) -> MemStats;

    // ---- 显存池(A5.2:上层一切申请必须经池校验,无池外分配)----

    /// 建池:A5.1 分解表的一行实体化。重复名字返回错误(分解表不该有
    /// 两行同名账)。
    fn create_pool(&self, cfg: PoolConfig) -> Result<Self::Pool, BackendError>;

    /// 按 id 取回池对象(重拿句柄用;正常路径持有 create_pool 的返回值)
    fn pool(&self, id: PoolId) -> Result<Self::Pool, BackendError>;

    /// 从指定池做持久分配(清零)。任何相位合法。
    /// 校验顺序:池余量(A5.4)→ 全局预算(A5.4)→ 分配。
    fn alloc_persistent_in<T: MemValue>(
        &self,
        pool: &Self::Pool,
        len: usize,
    ) -> Result<Self::Persistent<T>, BackendError>;

    /// host → device 持久写入(权重装载路径),计入指定池。
    fn htod_persistent_in<T: MemValue>(
        &self,
        pool: &Self::Pool,
        src: Vec<T>,
    ) -> Result<Self::Persistent<T>, BackendError>;

    /// 从指定池做暂存分配。允许 Capturing 相(捕获安全分配)。
    fn alloc_scratch_in<T: MemValue>(
        &self,
        pool: &Self::Pool,
        len: usize,
    ) -> Result<Self::Scratch<T>, BackendError>;
}
