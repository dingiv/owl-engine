//! 契约面 —— 算子双面基础设施的**单一权威**(2026-10-12 M2,自
//! owl-iface::contract 迁入;用户裁决:iface 依赖 kernels,kernels 不依赖
//! iface,则契约老家 = kernels,iface 反向 re-export 续用旧路径)。
//!
//! ```text
//! owl-kernels::contract   ← 类型权威(Dtype/线格式/Bytes/LaunchMsg/OpError)
//! owl-kernels::device     ← 服务端面原语(DeviceRes trait + Exec,feature=cuda)
//! owl-kernels::registry   ← OpId/注册宏(名字字面量全库唯一住址)
//!      ↑                    ↑
//! owl-iface(re-export)  owl-models(客户端面:Call builder)
//! owl-backends(服务端面:DeviceRes 实现 + 算子 runtime)
//! ```
//!
//! 迁入物 = 跨 crate 线格式(Dtype/Shape/GraphId/Bytes/KernelSpec/Arg/
//! LaunchMsg);留守 iface = 能力契约族(ModelError/PinnedRegion/
//! DeviceClient —— 经 re-export 引用本模块,零破坏)。
//!
//! 新增词汇(operator-contract 施工设计 v0.4):[`OpError`](Result 全链的
//! 错误模型)/[`FieldStats`]/[`Law`]/[`Stage`]/[`Linkage`]/[`OpId`]/
//! [`InvariantBox`]。

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

/// 图身份证(server 签发;graph_end 成功后可 graph_launch 重放)
pub type GraphId = u64;

// ============================================================================
// 线格式:发射消息 + 池块句柄
// ============================================================================

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
    /// 刀1:块内连续切片视图(发射 ptr = block_ptr(id) + byte_offset;
    /// elems = 元素数,供 CPU 面切片与校验;kernel ABI 不感知 —— 尺寸
    /// 标量已由槽序携带)。输出槽恒为全块 Block,BlockSlice 仅入参。
    BlockSlice { id: u64, byte_offset: u64, elems: u64 },
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
// 算子域错误模型(P3 Result 全链;检测与处置分离)
// ============================================================================

/// 算子执行阶段(诊断定位用;Invariant/Launch 错误携带)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Init,
    Cast,
    Load,
    Store,
    Compute,
    Epilogue,
}

/// 数值不变量律(§4.8 闸门的机器可读名;违例 = `OpError::Invariant`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Law {
    /// cumsum 产物全负律:g_cum ≤ 0+ε(g 语义恒负,累计只会更负)
    GateCumsumNonPositive,
    /// 中间产物有限律(如 solve_tril 后的 A)
    IntermediateFinite,
    /// 算子出口有限律(防染毒下游)
    OutputFinite,
}

/// 体检统计(违例证据;下一个"+142"出现时账面自带数值)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldStats {
    pub min: f32,
    pub max: f32,
    pub nan_count: u64,
    pub inf_count: u64,
}

/// 算子域统一错误(P3:检测在此,处置在引擎策略层)。
#[derive(Debug, Clone)]
pub enum OpError {
    /// 契约违反:槽位/数量/dtype/shape(Call::parse 期)。
    /// op = String(错误路径非热;未注册名字等动态面也能入账)
    Contract { op: String, field: &'static str, expect: String, got: String },
    /// 装配违反:符号缺失 / 资产指纹失配(init 期;P4 装配门)
    Asset { op: String, detail: String },
    /// 发射失败:cast / launch / memcpy(cudarc 错误原样保链)
    Launch { op: String, stage: Stage, detail: String },
    /// 不变量违反:数学律被破(run 期;策略表裁决处置)
    Invariant { op: String, law: Law, stats: FieldStats },
}

impl std::fmt::Display for OpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpError::Contract { op, field, expect, got } => {
                write!(f, "contract {op}: {field} 期望 {expect} 实得 {got}")
            }
            OpError::Asset { op, detail } => write!(f, "asset {op}: {detail}"),
            OpError::Launch { op, stage, detail } => write!(f, "launch {op}/{stage:?}: {detail}"),
            OpError::Invariant { op, law, stats } => {
                write!(f, "invariant {op}/{law:?}: {stats:?}")
            }
        }
    }
}

impl std::error::Error for OpError {}

// ============================================================================
// 注册表词汇:OpId(名字字面量全库唯一住址 = registry.rs)
// ============================================================================

/// 算子身份证(注册表签发;models 只拿这个,不拼名字字符串)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub &'static str);

impl std::fmt::Display for OpId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

// ============================================================================
// 链路形态 + 暂存盒(服务端面词汇;device.rs 消费)
// ============================================================================

/// 链路形态(§4.5;决定装载时机与校验门)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// 构建期静态链接(.a / 系统库,extern "C"):链接期符号解析免费,
    /// init 期只做句柄/workspace 预钉(ensure_blas 律)
    StaticLib,
    /// AOT cubin 资产(include_bytes!,gitignore + manifest):boot 期
    /// load + 符号逐一校验 × manifest sha256 双门
    AotCubin,
    /// NVRTC 动态编译(源码在 git,首用编译 + (source hash, name) 缓存)
    Nvrtc,
}

/// 不变盒:htod staging 的不变源(soff 案纪律)—— 捕获节点引用的
/// host 缓冲必须与发射时序无关(每 T 一份永不互改);构造后不可变。
#[derive(Debug, Clone)]
pub struct InvariantBox {
    bytes: Vec<u8>,
}

impl InvariantBox {
    /// 构造即冻结(此后只读;捕获窗内引用安全)
    pub fn freeze(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}
