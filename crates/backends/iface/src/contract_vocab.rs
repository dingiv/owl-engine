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
//! 迁入物 = 跨 crate 线格式(Dtype/Shape/GraphId/Bytes/KernelSource/Arg/
//! LaunchMsg);留守 iface = 能力契约族(ModelError/PinnedRegion/
//! DeviceClient —— 经 re-export 引用本模块,零破坏)。
//!
//! 新增词汇(operator-contract 施工设计 v0.4):[`OpError`](Result 全链的
//! 错误模型)/[`FieldStats`]/[`Law`]/[`Stage`]/[`OpId`]/
//! [`InvariantBox`]。

// ============================================================================
// 标注词汇:Dtype + Shape(线格式的元数据维)
// ============================================================================

/// 数据类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dtype {
    F32,
    BF16,
    F16,
    /// **FP8 · e4m3**(1B;KV 池/激活的 fp8 格式 —— 与 e5m2 **并列**,不是"FP8"一个东西)
    F8E4M3,
    /// **FP8 · e5m2**(1B;动态范围更大、精度更低;与 e4m3 不同格式,禁混称)
    F8E5M2,
    /// **INT8**(1B;**有符号**整型 —— 与 fp8 不同族!W4A8 的 kS8 激活)
    I8,
    U32,
    /// 字节承载(**仅限真正不透明**的字节块:暂存/码头/布局无关池;
    /// 2026-10-13 P3 引入,2026-10-16 收窄 —— **fp8/int8 必须点名具体格式**,
    /// 不许再用 U8 兜:那等于让"声明"与"事实"脱钩(与 F16/BF16 同案)
    U8,
}

impl Dtype {
    /// 字节宽(server 的分配只认字节;宽是 client 侧换算用的)
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::I8 | Dtype::U8 => 1,
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
// 长度律哨兵(§5.3 ExtentLaw 的 L3 面)
// ============================================================================

/// **全局硬律:numel > 0**。零长块/槽 = 上游形状 bug(锁页分配空指针 /
/// 静默空写,如 `xqa_sem` 零长案、pool dtype 改造漏写者案)。
///
/// 构造期即结构化拒,错误带**槽名**(`slot`)+ **块 id**(定位一步到位):
/// `contract <op>: <slot> 期望 numel ≥ 1 实得 块 id=N numel=0`。
/// 对应 `graph_plan::fill_slots`(槽声明为 0 元素)与
/// `state::Staging::alloc(n, label)` 同类守卫在 kernels 契约面的收口。
pub fn require_numel(numel: usize, slot: &'static str, op: &str, id: u64) -> Result<(), OpError> {
    if numel == 0 {
        return Err(OpError::Contract {
            op: op.to_string(),
            field: slot,
            expect: "numel ≥ 1(零长块/槽 = 上游形状 bug)".into(),
            got: format!("块 id={id} numel=0"),
        });
    }
    Ok(())
}

/// **形状互证:numel(shape) == len 且 numel > 0**(两律合一)。
/// 有 shape 数组的解析点用(声明账长必须等于形状元素总数)。
pub fn require_shape_len(
    shape: &[usize],
    len: usize,
    slot: &'static str,
    op: &str,
    id: u64,
) -> Result<(), OpError> {
    let n = numel(shape);
    require_numel(n, slot, op, id)?;
    if n != len {
        return Err(OpError::Contract {
            op: op.to_string(),
            field: slot,
            expect: format!("numel(shape)={n}"),
            got: format!("块 id={id} len={len}"),
        });
    }
    Ok(())
}

/// 图身份证(server 签发;graph_end 成功后可 graph_launch 重放)
pub type GraphId = u64;

// ============================================================================
// 线格式:发射消息 + 池块句柄
// ============================================================================

/// 池块句柄:server 签发的身份证(id → server 账房 → 显存)。
///
/// **自描述**(2026-10-15 用户裁决):句柄自带 `dtype` + `rank` —— 它们是
/// 客户端(SSA 图)才有的标注,随线格式发到服务端后由 L3 权威校验
/// (客户端不再自查)。`rank = 0` = **尚未定形**(`alloc` 只知元素数、不知形状;
/// 定形时用 [`view`](Self::view) 或直接由张量视图给 `Arg`)。
#[derive(Clone, Debug)]
pub struct Bytes {
    pub id: u64,
    /// 元素数(元素口径;`len × dtype.size_bytes()` = 字节数)
    pub len: usize,
    pub dtype: Dtype,
    pub rank: u8,
}

impl Bytes {
    pub const fn new(id: u64, len: usize, dtype: Dtype, rank: u8) -> Self {
        Self { id, len, dtype, rank }
    }

    /// 视图重声明(只改标注,不动 id/len):`alloc` 后按形状定形、切片重解释用。
    pub const fn view(self, dtype: Dtype, rank: u8) -> Self {
        Self { dtype, rank, ..self }
    }

    /// 事实面(块句柄自带的 dtype + 阶数)
    pub const fn facts(&self) -> (Dtype, u8) {
        (self.dtype, self.rank)
    }
}

/// kernel 描述:入口名 + 源码。后端按 (源码哈希, 名) 懒编译缓存。
#[derive(Clone, Debug)]
pub struct KernelSource {
    pub name: String,
    pub source: String,
}

/// 发射参数槽(有序;与 kernel 签名严格对位)
/// 类型化:标量按 kernel 形参宽度入槽(CUDA 参数空间自然对齐,
/// 8 字节槽顶 4 字节形参会错位读参 —— 坑 I)
///
/// **块参携带事实面**(`dtype` + `rank`;2026-10-15 用户裁决):
/// dtype/rank 只存在于客户端(SSA 图的标注),服务端物理上看不见 ⇒
/// **随线格式发过去**,由服务端 L3(`spec::check_from_params`)权威校验
/// (客户端不再自查 —— 单一咽喉,覆盖所有发射路)。
/// `rank` = 该张量的阶数(`shape.len()`);**必填**(无"未知"档 ——
/// 不静默)。声明侧 `rank = 0` 才是"阶数无关"的显式标记。
#[derive(Clone, Debug)]
pub enum Arg {
    Block { id: u64, dtype: Dtype, rank: u8 },
    /// 刀1:块内连续切片视图(发射 ptr = block_ptr(id) + byte_offset;
    /// elems = 元素数,供 CPU 面切片与校验;kernel ABI 不感知 —— 尺寸
    /// 标量已由槽序携带)。输出槽恒为全块 Block,BlockSlice 仅入参。
    /// `dtype`/`rank` = **切片视图**的事实(与全块可能不同)。
    BlockSlice { id: u64, byte_offset: u64, elems: u64, dtype: Dtype, rank: u8 },
    U64(u64),
    I32(i32),
    F32(f32),
}

impl Arg {
    /// 块参(事实随行)
    pub const fn block(id: u64, dtype: Dtype, rank: u8) -> Self {
        Arg::Block { id, dtype, rank }
    }

    /// 切片视图(事实随行)
    pub const fn block_slice(id: u64, byte_offset: u64, elems: u64, dtype: Dtype, rank: u8) -> Self {
        Arg::BlockSlice { id, byte_offset, elems, dtype, rank }
    }

    /// 块参的事实面(标量槽 = `None`)—— L3 事实面的**唯一读取口**
    pub const fn facts(&self) -> Option<(Dtype, u8)> {
        match self {
            Arg::Block { dtype, rank, .. } | Arg::BlockSlice { dtype, rank, .. } => {
                Some((*dtype, *rank))
            }
            _ => None,
        }
    }

    /// 块 id(标量槽 = `None`)
    pub const fn block_id(&self) -> Option<u64> {
        match self {
            Arg::Block { id, .. } | Arg::BlockSlice { id, .. } => Some(*id),
            _ => None,
        }
    }
}

// **族名/算子名不再集中于此**(2026-10-15 用户裁决:字面量住各族自己的定义处)。
//
// 住址表(每族的 `family/<族>.rs`,身份面零门 ⇒ 声明面 `ops/` 与 models 可引用):
//   cuBLAS      → `family::cublas::{GEMM_F16, GEMM_BF16}`
//   Marlin      → `family::marlin::{GEMM_W4A16, GEMM_W4A16_AWQ, GEMM_W4A16_BF16}`
//   GDN chunked → `family::gdn_chunked::GDN_CHUNKED_FWD`
//   GDN scalar  → `family::gdn_scalar::GDN_SCALAR_FWD`
//   FlashInfer  → `family::flashinfer::{PREFILL_FI, PREFILL_FI_FP8KV}`
//   XQA         → `family::xqa::{VERIFY_XQA, DECODE_XQA}`
// 族身份(`OpId`)挂在**族 runtime 结构体**上(`server/<族>::ID`);声明面
// `KernelSpec::FAMILY` 与执行面 `id()` 引用**同一字面量**。
// "原生哨兵" = `crate::spec::NATIVE`(它不是族,是"无族"标记)。

/// 发射参数值(owned 值表元素;指针参数 = u64 位型直入指针槽)。
///
/// 住 contract(而非 device)的原因:**自检面零 feature 门** —— 装配期 oracle
/// 要构造值表,而 device 面是 `feature = "device"` 门控的。
#[derive(Debug, Clone, Copy)]
pub enum LaunchVal {
    /// 设备指针(块解析产物/scratch;8 字节指针槽)
    Ptr(u64),
    /// 8 字节整数(`size_t`/`long long`)
    U64(u64),
    /// **4 字节无符号**(`uint32_t`;vendor 核常用,如 XQA 的 `qSeqLen`/
    /// `nbKHeads`/strides)。与 [`I32`](Self::I32) 同宽不同语义 —— 宽度必须
    /// 显式,否则 8 字节槽会**错位吞掉**后续参数(实测教训类)。
    U32(u32),
    /// 4 字节有符号(`int`)
    I32(i32),
    /// 4 字节浮点(`float`)
    F32(f32),
    /// **by-value 结构体参数**(内联镜像 ≤ 64 字节)。
    ///
    /// 为什么需要它:`cuLaunchKernel` 的 `kernelParams` 是"**每参数一个指针**",
    /// 驱动按**内核参数元数据里的 `paramSize`** 从该指针拷字节 —— 所以结构体参数
    /// 必须给一个指向 **完整镜像** 的指针(把 40 字节结构体拆成 5 个 u64 槽是
    /// **错的**:驱动会从第一个 u64 的地址拷 40 字节 = 读到邻居栈)。
    /// 内联存储 ⇒ 值随 `LaunchVal` 一起 owned,压槽时取地址天然安全(无悬垂)。
    /// 调用方保证 `#[repr(C)]` 镜像与内核形参**逐字节一致**(size 断言 + 资产
    /// 自描述可双重锁死,见 `family::xqa`)。
    Blob {
        len: u32,
        bytes: [u8; BLOB_MAX],
    },
}

/// [`LaunchVal::Blob`] 内联容量(够 XQA `KVCacheList`=40B 等小结构体)
pub const BLOB_MAX: usize = 64;

impl LaunchVal {
    /// 结构体镜像 → 参数槽(镜像 ≤ [`BLOB_MAX`];超长 = 调用方错误,截断前即断言)
    pub fn blob(bytes: &[u8]) -> Self {
        assert!(
            bytes.len() <= BLOB_MAX,
            "LaunchVal::blob: 镜像 {} B 超上限 {BLOB_MAX} B(扩容 BLOB_MAX 或改走指针)",
            bytes.len()
        );
        let mut buf = [0u8; BLOB_MAX];
        buf[..bytes.len()].copy_from_slice(bytes);
        LaunchVal::Blob { len: bytes.len() as u32, bytes: buf }
    }
}

/// 发射消息
pub struct LaunchMsg {
    pub kernel: KernelSource,
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
// 暂存盒(服务端面词汇;device.rs 消费)
// ============================================================================

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


// ============================================================================
// 胖算子契约(历史注记,2026-10-12 用户律:三形态归一 Call 后退役)
// ============================================================================
// 原 OpSpec trait(name/validate/out/wire,struct 即算子)随 Op::Spec 节点
// 一并退役 —— 值域校验由 driver 臂(契约公式进代码)+ 登记表 O 标记 +
// family face parse 三层承接;样例 NarrowStrided 与 SemanticKernel::Narrow
// 重复,已删。E1-E15 台账的代码级防线 = KernelCall 时代遗产 + 现役
// lower_kernel C2 对拍 + 家族 face 校验,不受影响。

#[cfg(test)]
mod tests {
    use super::*;

    /// 零长块/槽 ⇒ 结构化 Contract 错(带槽名 + 块 id),不是 panic/静默。
    #[test]
    fn numel_guard_rejects_zero_with_name_and_id() {
        let e = require_numel(0, "q", "demo_op", 42).unwrap_err();
        match e {
            OpError::Contract { field, ref got, .. } => {
                assert_eq!(field, "q", "错误未带槽名");
                assert!(got.contains("42"), "错误未带块 id: {got}");
                assert!(got.contains("numel=0"), "错误未带 numel: {got}");
            }
            other => panic!("期望 Contract 错,实得 {other}"),
        }
        assert!(require_numel(6, "q", "demo_op", 1).is_ok(), "非零应放行");
    }

    /// 形状互证:numel(shape) == len 且 numel > 0(两律合一)。
    #[test]
    fn shape_len_guard_cross_checks() {
        assert!(require_shape_len(&[2, 3], 6, "x", "demo_op", 1).is_ok());
        let e = require_shape_len(&[2, 3], 5, "x", "demo_op", 9).unwrap_err();
        assert!(format!("{e}").contains("numel(shape)=6"), "{e}");
        let z = require_shape_len(&[0, 4], 0, "x", "demo_op", 9).unwrap_err();
        assert!(format!("{z}").contains("numel=0"), "{z}");
    }
}

// ============================================================================
// §E 执行环境词汇(Hw / OpEnv)—— 2026-10-29 自 `owl-kernels::driver` 搬入
// ============================================================================
//
// 为什么属于契约面:它们是**引擎装配时传给算子层的被动事实**(硬件代际 /
// KV 页宽 / 头维 / GQA 比),kernels 零探测;接口层定义词汇,实现层消费。
// `owl_kernels::driver::{Hw, OpEnv}` 保留为 re-export(旧路径不变)。

/// 硬件事实的**最小面**(只收会被选核消费的;SM 数/频率/显存一律不在此)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hw {
    /// 计算能力代际(选核键)
    pub arch: crate::arch::Arch,
}

/// 执行环境(引擎/解释器装配,**必传参**;被动律:kernels 零探测)
#[derive(Clone, Copy, Debug)]
pub struct OpEnv {
    pub hw: Hw,
    /// KV 页大小(tokens/block;0 = 非 paged/legacy)。页配对律的 env 侧
    /// 单源 —— wrapper 变体选择在此吃页,调用点无页可错配
    pub page: usize,
    /// 注意力头维(0 = 未知 ⇒ 几何门跳过该项)。模型级几何事实,
    /// 由引擎 boot 从 `ModelDims` 合入(与 hw 同律:被动律)
    pub head_dim: usize,
    /// GQA 分组比 hq/hkv(0 = 未知/不适用)
    pub grp: usize,
}
