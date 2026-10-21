//! 算子准入契约 —— **单算子的声明式元数据**(2026-10-13 P3;设计见
//! `docs/arch/kernel-准入门槛-施工设计.md` §五)。
//!
//! ```text
//! KernelSpec(trait,算子自己实现)   ← 关联常量:入口名/源码/参数表/形状族/池/律/自检
//!         │ meta_of::<T>()(编译期擦除)
//!         ▼
//! KernelMeta(纯数据,零虚表)        ← driver 拾取、发射期 check、文档生成共用同一份
//! ```
//!
//! # 三条纪律(为什么这么设计)
//!
//! 1. **元数据住在算子身上**(关联常量),不在集中登记表里 —— 名字/源码/槽序/
//!    dtype 只有一个住址,"两处对不上"(E3/E15 根因)物理不可能;
//! 2. **槽序是派生的,不是手抄的**:`PARAMS` 是唯一真源,`T/O/sz/i32/f32`
//!    词表由 [`token_of`] 在 const 上下文算出(见 [`args_len`]/[`args_bytes`]
//!    与 `spec::tests::args_derive_from_params`);
//! 3. **准入 = 编译期 + 装配期 + 运行期三层**(设计 §三):
//!    L1 没有 `impl KernelSpec` 就发不出去;L2 族构造函数(装配门,见
//!    `registry::KernelFamily::assemble`);L3 [`KernelMeta::check`](运行期哨兵,
//!    零成本、能指名)。
//!
//! # 与既有资产的关系
//!
//! - **登记表已退役**(F 轮):声明面是唯一真源,未建表 = 仅引入(无准入声明),
//!   槽序/源码改从本模块的元数据取(未建表的旧入口继续走登记表字符串);
//! - [`crate::driver`] 的 `KernelPick` 携带 `&'static KernelMeta`(算子身份),
//!   不再携带裸名字字符串。

use crate::contract::{Arg, Dtype, Law, OpError, OpId};
use crate::bootcheck::{BootCheck, CheckCtx};

// 构建期 tag 词表(`tags.toml` → `$OUT_DIR/kernel_tags.rs`;2026-10-20)。
// 生成物定义 `KernelTag` / `KERNEL_TAGS` / `tag_mask`;本模块只消费。
include!(concat!(env!("OUT_DIR"), "/kernel_tags.rs"));

// ============================================================================
// §1 参数表词汇
// ============================================================================

/// 参数类别(由 [`ParamType`] 派生 —— 不设独立字段,消灭双真源)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    Tensor,
    Scalar,
}

/// 参数角色。**输出/原地/临时必须显式**(E3 起槽序里 `O` 的语义来源)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamRole {
    /// 只读入参
    In,
    /// 输出块(线槽序词表 = `O`)
    Out,
    /// 原地读写(副作用律;如 fused_add_rmsnorm 的 residual)
    InOut,
    /// 核内临时/工作区
    Scratch,
    /// 编译期常量(如烘焙的 group size)
    Const,
}

/// 标量类型。`*Bits` = **位型走私显式化**(设计 §5.6):`f32::from_bits(kv_len)`
/// 这类"注释约定"在表里是显式 token,发射端与内核解析端必须同型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarTy {
    /// 8 字节整数(`size_t`;线槽 `Arg::U64`,词表 `sz`)
    U64,
    /// 4 字节有符号(`int`;`Arg::I32`,词表 `i32`)
    I32,
    /// 4 字节浮点(`float`;`Arg::F32`,词表 `f32`)
    F32,
    /// f32 位型过 8 字节槽(词表 `sz`)
    F32Bits,
    /// i32 位型过 8 字节槽(词表 `sz`)
    I32Bits,
}

/// 参数类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    Tensor { dtype: Dtype, rank: u8 },
    Scalar(ScalarTy),
}

impl ParamType {
    pub const fn kind(self) -> ParamKind {
        match self {
            ParamType::Tensor { .. } => ParamKind::Tensor,
            ParamType::Scalar(_) => ParamKind::Scalar,
        }
    }
}

/// 维间关系(符号维名见 [`ShapeRule::dims`])。能表达的自动机检,表达不了的
/// 由算子覆写 [`KernelSpec::check`]。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rel {
    /// 两维相等(如 `D == Hkv*HD`)
    Eq(&'static str, &'static str),
    /// 该维必须被整除(如 `numel % 4 == 0`)
    DivisibleBy(&'static str, u64),
    /// 该维必须 ≥ 给定值
    AtLeast(&'static str, u64),
}

/// 形状规则(符号维 + 维间关系;空 = 不约束)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShapeRule {
    pub dims: &'static [&'static str],
    pub rels: &'static [Rel],
}

impl ShapeRule {
    /// 不约束(未建形状族的算子显式用这个,而不是留白 —— 留白无法区分
    /// "还没写"和"确实无约束")
    pub const NONE: ShapeRule = ShapeRule { dims: &[], rels: &[] };
}

/// 数值/长度范围规则
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RangeRule {
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// 枚举集(优先于 min/max)
    pub set: Option<&'static [i64]>,
}

impl RangeRule {
    pub const NONE: RangeRule = RangeRule { min: None, max: None, set: None };
    /// `[1, +∞)` —— 零值即上游形状 bug(标量槽用;如页数/序列长)
    pub const POSITIVE: RangeRule = RangeRule { min: Some(1.0), max: None, set: None };
}

/// 单个参数(表的一行;**全部 const 可构造**,进 `static` 表零运行期初始化)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Param {
    pub name: &'static str,
    pub role: ParamRole,
    pub ty: ParamType,
    pub shape: ShapeRule,
    pub range: RangeRule,
    /// 向量化访存前提(8B/16B;`None` = 无要求)
    pub align: Option<usize>,
}

impl Param {
    /// 张量入参(最常用形态)。注意:**张量的"非空"由张量分支的账长检查
    /// 负责**(`RangeRule` 只作用于标量槽)—— 这里不填 range,避免"填了
    /// 却不生效"的假声明。
    ///
    /// **`rank` 是契约**(2026-10-15 用户裁决,能多严就多严):它声明该槽
    /// **必须喂进来的张量阶数**,由客户端面 L3(`spec::check_facts`)逐发射
    /// 严格比对(线格式不承载 rank ⇒ 服务端查不到)。`rank = 0` = **显式
    /// 阶数无关**(核按计数扁平索引:逐元素核/铸核/argmax)。
    pub const fn t_in(name: &'static str, dtype: Dtype, rank: u8) -> Self {
        Self {
            name,
            role: ParamRole::In,
            ty: ParamType::Tensor { dtype, rank },
            shape: ShapeRule::NONE,
            range: RangeRule::NONE,
            align: None,
        }
    }

    /// 张量输出(线槽序 `O`)
    ///
    /// ⚠️ **不填 range**(2026-10-15 审查补):张量槽的 range 分支**从不被求值**
    /// (`arg_scalar` 对块参返回 `None`)⇒ 填了就是假声明。这里原为
    /// `RangeRule::POSITIVE`,现已改回 [`RangeRule::NONE`] —— 零长由**账长面**
    /// (服务端 `check_from_params` 的块非零长)与**事实面**负责。
    pub const fn t_out(name: &'static str, dtype: Dtype, rank: u8) -> Self {
        Self {
            name,
            role: ParamRole::Out,
            ty: ParamType::Tensor { dtype, rank },
            shape: ShapeRule::NONE,
            range: RangeRule::NONE,
            align: None,
        }
    }

    /// 张量原地读写
    pub const fn t_inout(name: &'static str, dtype: Dtype, rank: u8) -> Self {
        Self { role: ParamRole::InOut, ..Self::t_in(name, dtype, rank) }
    }

    /// 标量入参
    pub const fn s(name: &'static str, ty: ScalarTy) -> Self {
        Self {
            name,
            role: ParamRole::In,
            ty: ParamType::Scalar(ty),
            shape: ShapeRule::NONE,
            range: RangeRule::NONE,
            align: None,
        }
    }

    /// 8 字节长度标量(`sz`)
    pub const fn sz(name: &'static str) -> Self {
        Self::s(name, ScalarTy::U64)
    }

    /// 4 字节整数标量(`i32`)
    pub const fn i32(name: &'static str) -> Self {
        Self::s(name, ScalarTy::I32)
    }

    /// 4 字节浮点标量(`f32`)
    pub const fn f32(name: &'static str) -> Self {
        Self::s(name, ScalarTy::F32)
    }

    pub const fn kind(&self) -> ParamKind {
        self.ty.kind()
    }

    /// 带范围约束(链式;const 上下文可用)
    pub const fn range(mut self, r: RangeRule) -> Self {
        self.range = r;
        self
    }

    /// 带形状规则(链式;const 上下文可用)
    pub const fn shape(mut self, s: ShapeRule) -> Self {
        self.shape = s;
        self
    }

    /// 带对齐要求(链式;const 上下文可用)
    pub const fn align(mut self, a: usize) -> Self {
        self.align = Some(a);
        self
    }
}

// ============================================================================
// §2 槽序派生(唯一真源 = PARAMS;const fn,编译期可用)
// ============================================================================

/// 参数 → 线槽序词表 token(`T`/`O`/`sz`/`i32`/`f32`)。
///
/// `ParamRole::Out` ⇒ `O`(输出槽);其余按 [`ParamType`] 定 token。
/// **const fn** —— 可在 const 上下文里建表(见 [`args_bytes`]),也可运行期调用。
pub const fn token_of(p: &Param) -> &'static str {
    match p.role {
        ParamRole::Out => "O",
        _ => match p.ty {
            ParamType::Tensor { .. } => "T",
            ParamType::Scalar(ScalarTy::U64) => "sz",
            ParamType::Scalar(ScalarTy::I32) => "i32",
            ParamType::Scalar(ScalarTy::F32) => "f32",
            // 位型走私:线槽仍是 8 字节(Arg::U64)
            ParamType::Scalar(ScalarTy::F32Bits) | ParamType::Scalar(ScalarTy::I32Bits) => "sz",
        },
    }
}

/// 参数 → **线槽类型** token(`T`/`sz`/`i32`/`f32`)—— 输出槽在线格式里
/// 也是 `Arg::Block`(其 `O` 是**角色**标注,不是另一种槽型)。
pub const fn wire_token_of(p: &Param) -> &'static str {
    match p.ty {
        ParamType::Tensor { .. } => "T",
        ParamType::Scalar(ScalarTy::U64) => "sz",
        ParamType::Scalar(ScalarTy::I32) => "i32",
        ParamType::Scalar(ScalarTy::F32) => "f32",
        ParamType::Scalar(ScalarTy::F32Bits) | ParamType::Scalar(ScalarTy::I32Bits) => "sz",
    }
}

/// 逗号连接的槽序串长度(空表 = 0)
pub const fn args_len(params: &[Param]) -> usize {
    let mut n = 0usize;
    let mut i = 0usize;
    while i < params.len() {
        n += token_of(&params[i]).len();
        i += 1;
    }
    if params.is_empty() {
        0
    } else {
        n + params.len() - 1
    }
}

/// 编译期构造槽序字节串(长度 = [`args_len`];供 const 表使用)
pub const fn args_bytes<const N: usize>(params: &[Param]) -> [u8; N] {
    let mut b = [b','; N];
    let mut o = 0usize;
    let mut i = 0usize;
    while i < params.len() {
        let t = token_of(&params[i]).as_bytes();
        let mut j = 0usize;
        while j < t.len() {
            b[o] = t[j];
            o += 1;
            j += 1;
        }
        if i + 1 < params.len() {
            o += 1;
        }
        i += 1;
    }
    b
}

// ============================================================================
// §3 形状族 / 池访问 / 捕获与确定性(设计 §5.3-5.5;P4 填充值,P3 先立词汇)
// ============================================================================

/// KV 池布局标签(类型化,取代隐式协定 —— E14 双寻址/E15 契约腐烂的机器面)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LayoutTag {
    /// `[page, head, dim]`(vLLM classic)
    Classic,
    /// `[page, dim, head]`(owl kNHD)
    Knhd,
    /// 草稿池(自块直读)
    Draft,
}

/// 形状族:算子能吃哪些几何。**装配期**对池块逐项对表(设计 §6.3 L2)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeFamily {
    /// 池 dtype(**声明性**;一致性由 `PoolAccess` 审计负责 —— 布局/池 dtype
    /// 的单一住址是 `PoolAccess`,这里只声明"本核吃哪种池")
    pub kv_dtype: Option<Dtype>,
    pub head_dim: &'static [u32],
    pub page: &'static [u32],
    /// hq/hkv(GQA 分组比)
    pub grp: Option<u32>,
    /// 核实例化的行数上界(如 XQA `M_TILESIZE`)
    pub tile: Option<u32>,
}

impl ShapeFamily {
    /// 未建形状族(显式)
    pub const NONE: ShapeFamily = ShapeFamily {
        kv_dtype: None,
        head_dim: &[],
        page: &[],
        grp: None,
        tile: None,
    };

    /// **几何接受性检查**(声明驱动:只查**已声明**的项;空/`None` = 不约束)。
    ///
    /// 调用点 = `driver::resolve`(拾取即验,compose 期;错配 = 编程/配置错,
    /// 与 `aux 越界`/`无配对 wrapper` 同类处置 —— 结构化 panic,在 boot
    /// warmup 阶段即暴露 ⇒ 等价于拒启)。
    pub fn accepts(&self, q: &ShapeQuery) -> Result<(), String> {
        // 池 dtype/布局**不在此**:它们属于"哪个池"的身份,不是模型的全局几何
        // (主池 fp8 + 草稿池 bf16 并存 ⇒ 全局事实无法表达)——
        // 由 `PoolAccess` 审计负责(写者/读者声明互证 + 与 ShapeFamily.kv_dtype 自洽)。
        if !self.head_dim.is_empty() && q.head_dim != 0 && !self.head_dim.contains(&(q.head_dim as u32)) {
            return Err(format!("head_dim 要求 ∈ {:?},实得 {}", self.head_dim, q.head_dim));
        }
        if !self.page.is_empty() && q.page != 0 && !self.page.contains(&(q.page as u32)) {
            return Err(format!("page 要求 ∈ {:?},实得 {}", self.page, q.page));
        }
        if let Some(g) = self.grp {
            if q.grp != 0 && q.grp as u32 != g {
                return Err(format!("GQA 分组比要求 {g},实得 {}", q.grp));
            }
        }
        if let Some(t) = self.tile {
            if q.tile != 0 && q.tile > t {
                return Err(format!("实例化行数上界 {t},实得 {}", q.tile));
            }
        }
        Ok(())
    }
}

/// 拾取时的几何事实(引擎侧;`driver` 被动律 —— kernels 零探测,一律必传参)。
/// `0` / `None` = **未知 ⇒ 该项不检查**(诊断/单测环境不必凑齐全部事实)。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShapeQuery {
    pub head_dim: usize,
    pub page: usize,
    /// hq/hkv(0 = 未知/不适用)
    pub grp: usize,
    /// 核实例化行数(0 = 未知)
    pub tile: u32,
}

/// 池访问模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

/// 池访问声明(写集穷举审计的依据 —— E10"漏写者 = 槽区全零"的机器面)
///
/// **池身份 = (名字, 布局, dtype) 三元组**:同一个名字的池可以有两种
/// 布局/两种 dtype 并存(主池 f16 vs fp8、草稿池 bf16 vs fp8 都是配置选择),
/// 因此审计按三元组分组,而不是只按名字。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoolAccess {
    pub pool: &'static str,
    pub mode: AccessMode,
    pub layout: Option<LayoutTag>,
    pub dtype: Option<Dtype>,
}

impl PoolAccess {
    /// 读池(不含写)
    pub const fn read(pool: &'static str, layout: LayoutTag, dtype: Dtype) -> Self {
        Self { pool, mode: AccessMode::Read, layout: Some(layout), dtype: Some(dtype) }
    }
    /// 写池
    pub const fn write(pool: &'static str, layout: LayoutTag, dtype: Dtype) -> Self {
        Self { pool, mode: AccessMode::Write, layout: Some(layout), dtype: Some(dtype) }
    }
    /// 原地读写(状态池:GDN 状态格)
    pub const fn rw(pool: &'static str, dtype: Dtype) -> Self {
        Self { pool, mode: AccessMode::ReadWrite, layout: None, dtype: Some(dtype) }
    }

    pub const fn is_reader(&self) -> bool {
        matches!(self.mode, AccessMode::Read | AccessMode::ReadWrite)
    }
    pub const fn is_writer(&self) -> bool {
        matches!(self.mode, AccessMode::Write | AccessMode::ReadWrite)
    }
}

/// 池名(全库唯一字面量住址;禁止在算子表里写裸字符串)
pub mod pools {
    /// 主 KV 池(注意力 K/V;fp8 或 f16)
    pub const KV_MAIN: &str = "kv.main";
    /// DFlash2 草稿池(自块直读)
    pub const KV_DRAFT: &str = "kv.draft";
    /// GDN 递推状态格([slots, HV, KD, VD] f32)
    pub const GDN_STATE: &str = "gdn.state";
}

/// 捕获证据(A1.5 可捕获 kernel 契约的声明面)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureEvidence {
    /// 无 host 工作、无同步、无 D2H(可进图)
    NoHostWork,
    /// 需 host 侧准备(不可进图,或需前置拷贝)
    NeedsHostPrep,
    /// 未评估
    Unassessed,
}

/// 确定性
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Determinism {
    /// 同输入同输出(位一致)
    Deterministic,
    /// 原子累加序不定 ⇒ 位不一致(数值等价)
    NonDeterministic,
    Unassessed,
}

// ============================================================================
// §4 算子契约 trait(关联常量;编译期擦除)
// ============================================================================

/// **原生哨兵**(非族):`KernelSpec::FAMILY` 的缺省值 —— AOT cubin 直发。
///
/// 与任何族 id 都不相等;boot 期用它判"这个算子归谁执行"。它不是族字面量,
/// 而是"**无族**"标记,故住 spec 面(族名字面量住各族 `family/<族>.rs`)。
pub const NATIVE: &str = "native";

/// 单算子契约。**只有关联常量 + 一个可默认的静态方法**(无 `&self`):
/// 带关联常量的 trait 不能做 trait 对象(rustc E0038 实证),这不是代价 ——
/// 它把算子侧彻底静态化,元数据在编译期擦成 [`KernelMeta`] 纯数据表。
pub trait KernelSpec: 'static {
    /// **执行家族**(2026-10-15;声明驱动路由)。
    ///
    /// 缺省 = [`NATIVE`](crate::spec::NATIVE)(原生 AOT 直发);
    /// 家族算子填**族 id**(= `KernelFamily::id()` 的 `OpId`)。
    ///
    /// 为什么必须有这一维(旧设计只有反向的 `KernelFamily::ops()`):
    /// 1. **归属在声明处可见** —— 读 `ops/marlin.rs` 就能看出它归 `MarlinRuntime`,
    ///    不必翻到族侧;
    /// 2. 族 `ops()` 可以**派生**(`ops::of_family(id)`)而不是手写一张表 ⇒
    ///    消灭 `names() = 族主名 ∪ ops()` 那个"部分建表"陷阱
    ///    (2026-10-13 实测事故:FI f16 主名 + gdn_scalar 静默从路由表消失);
    /// 3. boot 期建 **名字 → 族** 索引 ⇒ `execute` 从"线性扫族×名"变 O(1);
    /// 4. 新增一条**声明驱动**的硬门:非 Native 的算子,其家族必须真实存在
    ///    (否则那份元数据永远发不出去 —— 纯粹的摆设)。
    const FAMILY: OpId = OpId(crate::spec::NATIVE);
    /// 线格式入口名(与某个 `.cu` 入口一对一)
    const ENTRY: &'static str;
    /// `.cu` 源码(`include_str!`;与算子同目录)
    const SRC: &'static str;
    /// 参数表(**槽序唯一真源**)
    const PARAMS: &'static [Param];
    /// 形状族(P4 填充;缺省 = 未建)
    const SHAPE: ShapeFamily = ShapeFamily::NONE;
    /// 池访问(P4 填充;缺省 = 不触池)
    const POOL: &'static [PoolAccess] = &[];
    /// 数值律(装配期/运行期不变量门的机器可读名)
    const LAWS: &'static [Law] = &[];
    /// 捕获证据
    const CAPTURE: CaptureEvidence = CaptureEvidence::Unassessed;
    /// 确定性
    const DETERMINISM: Determinism = Determinism::Unassessed;
    /// 装配期自检(独立 oracle;缺省无)
    const SELFCHECK: Option<&'static dyn BootCheck> = None;
    /// **分面 tag**(2026-10-20;由 `#[kernel_tags(...)]` 属性宏注入)。
    ///
    /// tag 住算子声明处,随 [`KernelMeta`] 进编译期表;运行时只按 tag 过滤/
    /// 审计,不再手抄算子名清单。生产算子必须至少 1 个 tag(测试门锁)。
    const TAGS: &'static [KernelTag] = &[];

    /// 运行期检查(L3)。缺省 = 由 [`PARAMS`](Self::PARAMS) 自动生成;
    /// 只有**跨参耦合**的算子覆写(如 XQA 的 `q_seq_len ↔ mask` 耦合)。
    fn check(ctx: &CheckCtx<'_>) -> Result<(), OpError> {
        check_from_params(Self::ENTRY, Self::PARAMS, ctx)
    }
}

/// 编译期擦除:算子契约 → 纯数据表(**无 dyn 特征对象**)。
///
/// 实测可行:`const fn` + 关联常量 + trait 方法取 `fn` 指针,全在 const 上下文内。
pub const fn meta_of<T: KernelSpec>() -> KernelMeta {
    KernelMeta {
        family: T::FAMILY,
        entry: T::ENTRY,
        src: T::SRC,
        params: T::PARAMS,
        shape: T::SHAPE,
        pool: T::POOL,
        laws: T::LAWS,
        capture: T::CAPTURE,
        determinism: T::DETERMINISM,
        check: T::check,
        selfcheck: T::SELFCHECK,
        tags: T::TAGS,
    }
}

/// 擦除后的算子元数据(纯数据;可在 `static` 表里 const 构造)
pub struct KernelMeta {
    /// **执行家族**(`spec::NATIVE` = 原生 AOT 直发;否则 = 族 id)
    pub family: OpId,
    pub entry: &'static str,
    pub src: &'static str,
    pub params: &'static [Param],
    pub shape: ShapeFamily,
    pub pool: &'static [PoolAccess],
    pub laws: &'static [Law],
    pub capture: CaptureEvidence,
    pub determinism: Determinism,
    /// L3 检查函数(`<T as KernelSpec>::check` 的 fn 指针)
    pub check: fn(&CheckCtx<'_>) -> Result<(), OpError>,
    pub selfcheck: Option<&'static dyn BootCheck>,
    /// 分面 tag(声明处注入;空 = 未打 tag —— 生产算子测试门拒)
    pub tags: &'static [KernelTag],
}

impl std::fmt::Debug for KernelMeta {
    /// 手写 Debug:`selfcheck: Option<&dyn BootCheck>` 无 Debug(且打印它没意义),
    /// 诊断关心的是名字/槽序/律
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelMeta")
            .field("family", &self.family.0)
            .field("entry", &self.entry)
            .field("slots", &self.args_str())
            .field("laws", &self.laws)
            .field("capture", &self.capture)
            .field("determinism", &self.determinism)
            .field("selfcheck", &self.selfcheck.is_some())
            .field("tags", &self.tags.iter().map(|t| t.as_str()).collect::<Vec<_>>())
            .finish()
    }
}

impl KernelMeta {
    /// 是否家族算子(非原生)
    pub fn is_family(&self) -> bool {
        self.family.0 != crate::spec::NATIVE
    }

    /// 是否带某 tag(O(tags);tag 数 ≤64)
    pub fn has_tag(&self, tag: KernelTag) -> bool {
        self.tags.contains(&tag)
    }

    /// tag 位掩码(热路径/集合比较用)
    pub fn tag_mask(&self) -> u64 {
        tag_mask(self.tags)
    }

    /// tag id 列表(诊断/CLI 用)
    pub fn tag_ids(&self) -> Vec<&'static str> {
        self.tags.iter().map(|t| t.as_str()).collect()
    }

    /// 槽序 token 序列(由 [`PARAMS`](Self::params) 派生 —— 零第二份手抄)
    pub fn slot_tokens(&self) -> Vec<&'static str> {
        self.params.iter().map(token_of).collect()
    }

    /// **声明口径 dtype** = 输出槽的元素类型(F 轮:原登记表 `Entry.dtype`
    /// 的唯一信息,现由声明面派生 —— 每张 native 表都有输出槽,由
    /// `op_tables` 的「O 槽必须是指针位」门与 `ops::tests` 的
    /// `every_native_table_declares_out_dtype` 保证)。
    ///
    /// `None` = 无输出槽(部分 foreign 虚拟核)。调用方按需降级。
    pub fn out_dtype(&self) -> Option<Dtype> {
        self.params.iter().find(|p| p.role == ParamRole::Out).and_then(|p| match p.ty {
            ParamType::Tensor { dtype, .. } => Some(dtype),
            _ => None,
        })
    }

    /// 槽序串(`T,O,sz,...`;诊断/文档/`--print-op-table` 用)
    pub fn args_str(&self) -> String {
        self.slot_tokens().join(",")
    }

    /// 输出槽位置(`O` 所在下标;无 `O` = `None`)
    pub fn out_pos(&self) -> Option<usize> {
        self.params.iter().position(|p| p.role == ParamRole::Out)
    }

    /// 参数名(槽位诊断用;越界给 `?`)
    pub fn param_name(&self, i: usize) -> &'static str {
        self.params.get(i).map(|p| p.name).unwrap_or("?")
    }

    /// **槽序名序列**(唯一可读视图;回归锁/工具/文档面用)。
    ///
    /// 与 [`args_str`](Self::args_str)(token 串)互补:token 串看类型,名序列看语义。
    pub fn param_names(&self) -> Vec<&'static str> {
        self.params.iter().map(|p| p.name).collect()
    }

    /// **按名查槽下标**(声明面寻址)。
    ///
    /// 这是"`parse`/builder 按名取槽"(待办)的前置件 —— 现有手写解析器用
    /// 字面下标(`blocks[0]`/`scalars[1]`),本方法让它们可改成
    /// `slots.block("gate")?`,从而与 `PARAMS` 同源。
    /// 当前用途:`ops::tests::slot_order_lock` 把**解析器/装配器隐含的序**
    /// 与 `PARAMS` 钉死 —— 改 `PARAMS` 而不同步两侧 = 测试立刻红。
    pub fn slot_index(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    /// 发射期检查(统一入口;**server 唯一调用点**)
    pub fn check(&self, ctx: &CheckCtx<'_>) -> Result<(), OpError> {
        (self.check)(ctx)
    }
}

// ============================================================================
// §6 默认检查(由 PARAMS 自动生成;设计 §5.2)
// ============================================================================

/// 表驱动的默认检查:**服务端面**(线格式可判定的那半边)——
/// 数量对位 / 槽类型对位 / 角色面(Out 恒全块)/ 账长面(非零长·对齐·块在账本)/ 标量范围 / 几何通用律。
///
/// 能表达的自动机检;跨参耦合(如 `q_seq_len` 与 `mask` 长度)由算子覆写。
/// **dtype/rank 不在此面**(线格式不承载 ⇒ 服务端物理上看不见)—— 客户端面见
/// [`check_facts`]。
pub fn check_from_params(
    op: &'static str,
    params: &'static [Param],
    ctx: &CheckCtx<'_>,
) -> Result<(), OpError> {
    if ctx.args.len() != params.len() {
        return Err(OpError::Contract {
            op: op.to_string(),
            field: "args",
            expect: format!("槽数 = {} (参数表)", params.len()),
            got: format!("{}", ctx.args.len()),
        });
    }
    // ⓪ 几何通用律(2026-10-15 审查补:`ctx.geom` 此前**从不被读** ⇒ grid/out_elems
    //    任意值都过 L3)。grid == (0,0,0) 是**显式哨兵** = "发射配置由 runtime 自持"
    //    (foreign 家族全走这条:见 `client::gdn_chunked::to_launch`;native 路的哨兵
    //    在 `models::ops::lower_kernel` 已展开成 auto_grid ⇒ 这里看到全零 = 家族路)。
    if ctx.geom.out_elems == 0 {
        return Err(OpError::Contract {
            op: op.to_string(),
            field: "out_elems",
            expect: "≥ 1(零元素输出 = 上游形状 bug)".into(),
            got: "out_elems=0".into(),
        });
    }
    let (gx, gy, gz) = ctx.geom.grid;
    if gx == 0 && (gy, gz) != (0, 0) {
        return Err(OpError::Contract {
            op: op.to_string(),
            field: "grid",
            expect: "grid.x ≥ 1,或全零哨兵 (0,0,0) = runtime 自持".into(),
            got: format!("grid=({gx},{gy},{gz})"),
        });
    }
    for (i, (p, a)) in params.iter().zip(ctx.args.iter()).enumerate() {
        let field = p.name;
        // ① 槽类型对位(线格式词表 ↔ ParamType)
        let got_tok = arg_token(a);
        let want_tok = wire_token_of(p);
        // 位型走私:表声明 F32Bits/I32Bits,线槽是 sz —— 词表一致即通过
        if got_tok != want_tok {
            return Err(OpError::Contract {
                op: op.to_string(),
                field,
                expect: format!("{want_tok}({})", type_name(p)),
                got: format!("{got_tok}(槽 {i})"),
            });
        }
        // ② 张量槽:角色面(Out 恒全块)+ **事实面**(dtype/rank;线格式承载,
        //    客户端发过来 ⇒ 服务端权威校验,2026-10-15 用户裁决)+ 账长面
        if p.kind() == ParamKind::Tensor {
            check_out_role(op, p, i, a)?;
            check_tensor_facts(op, p, i, a)?;
            // 旧实现是 `if let Some(n) = ctx.arg_len(i)` —— 块不在账本时零长/对齐
            // **静默跳过**(审查 P2-2a)。现在改成硬拒:块不在账本 ⇒ 后续 resolve
            // 必失败,早报比晚炸准。
            let n = ctx.arg_len(i).ok_or_else(|| OpError::Contract {
                op: op.to_string(),
                field,
                expect: "块在账本(可查账长)".into(),
                got: format!("槽 {i} 的块不在本账本(未分配/已回收/非块参)"),
            })?;
            if n == 0 {
                return Err(OpError::Contract {
                    op: op.to_string(),
                    field,
                    expect: "numel ≥ 1(零长块/槽 = 上游形状 bug)".into(),
                    got: format!("槽 {i} numel=0"),
                });
            }
            if let Some(al) = p.align {
                if n % al != 0 {
                    return Err(OpError::Contract {
                        op: op.to_string(),
                        field,
                        expect: format!("numel 对齐 {al}"),
                        got: format!("槽 {i} numel={n}"),
                    });
                }
            }
        }
        // ③ 标量范围
        check_scalar_range(op, p, i, a)?;
    }
    Ok(())
}

/// 角色面:**输出槽恒为全块 `Block`**(线格式里 `BlockSlice` 仅入参)。
///
/// 客户端 `lower_kernel` 的装配保证也是这条(输出块固定追加在槽序里),这里是
/// 服务端侧的**线格式完整性**复核:切片输出会让回传句柄与写窗口对不上。
fn check_out_role(op: &str, p: &Param, i: usize, a: &Arg) -> Result<(), OpError> {
    if p.role == ParamRole::Out {
        if let Arg::BlockSlice { id, byte_offset, .. } = a {
            return Err(OpError::Contract {
                op: op.to_string(),
                field: p.name,
                expect: "输出槽 = 全块 Block(BlockSlice 仅入参)".into(),
                got: format!("槽 {i} BlockSlice(id={id}, off={byte_offset})"),
            });
        }
    }
    Ok(())
}

/// 标量范围(`RangeRule` 的唯一求值点;张量槽的 range **不在此求值** —— 见 `t_out`)
fn check_scalar_range(op: &str, p: &Param, i: usize, a: &Arg) -> Result<(), OpError> {
    let Some(v) = arg_scalar(a) else { return Ok(()) };
    let field = p.name;
    if let Some(set) = p.range.set {
        if !set.contains(&v) {
            return Err(OpError::Contract {
                op: op.to_string(),
                field,
                expect: format!("取值 ∈ {set:?}"),
                got: format!("槽 {i} = {v}"),
            });
        }
    }
    if let Some(min) = p.range.min {
        if (v as f64) < min {
            return Err(OpError::Contract {
                op: op.to_string(),
                field,
                expect: format!("≥ {min}"),
                got: format!("槽 {i} = {v}"),
            });
        }
    }
    if let Some(max) = p.range.max {
        if (v as f64) > max {
            return Err(OpError::Contract {
                op: op.to_string(),
                field,
                expect: format!("≤ {max}"),
                got: format!("槽 {i} = {v}"),
            });
        }
    }
    Ok(())
}

/// 槽的事实来源(声明驱动;`lower_kernel` 与事实面**单源**)。
///
/// 规则(与 `models::ops::lower_kernel` 的装配序逐字同):非输出 `T` 槽按序吃
/// 父张量;`O` 槽(无 `O` 时 = 末位 `T`)吃输出块;其余是标量槽。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotSrc {
    /// 第 n 个父张量(`T` 槽,入参)
    Parent(usize),
    /// 输出块(`O` 槽,或末位 `T`)
    Out,
    /// 标量槽
    Scalar,
}

/// 由声明表推导槽的事实来源(纯函数;客户端两条发射路 + 测试共用)。
pub fn slot_sources(params: &[Param]) -> Vec<SlotSrc> {
    let toks: Vec<&str> = params.iter().map(token_of).collect();
    slot_sources_of_tokens(&toks)
}

/// 由**槽序 token 串**推导事实来源(声明面与逃生舱签名同一条规则;
/// 与 `models::ops::lower_kernel` 的装配序**单源**)。
pub fn slot_sources_of_tokens(toks: &[&str]) -> Vec<SlotSrc> {
    let out_pos = toks.iter().position(|t| *t == "O");
    let n = toks.len();
    let mut pi = 0usize;
    toks.iter()
        .enumerate()
        .map(|(i, t)| match *t {
            "T" | "O" => {
                let is_out = match out_pos {
                    Some(pos) => i == pos,
                    None => i + 1 == n,
                };
                if is_out {
                    SlotSrc::Out
                } else {
                    let k = pi;
                    pi += 1;
                    SlotSrc::Parent(k)
                }
            }
            _ => SlotSrc::Scalar,
        })
        .collect()
}

// ============================================================================
// §6.1 事实面(dtype / rank;线格式承载 ⇒ 服务端权威校验)
// ============================================================================

/// **事实面校验**(由 [`check_from_params`] 调用;2026-10-15 用户裁决)。
///
/// dtype/rank 只存在于客户端(SSA 图的标注)—— **随 `Arg` 发过来**
/// (`Arg::Block { id, dtype, rank }` / `Arg::BlockSlice { … }`),服务端因此能
/// 权威校验:**单一咽喉 = server 的 `admission_check`**,覆盖原生/家族/未来所有
/// 发射路(客户端不再自查 —— 运动员不当裁判)。
///
/// 语义(用户裁决:**能多严就多严,不静默**):
/// - 张量槽拿不到事实(`facts() == None`,即块参写成了标量槽)⇒ 拒(上游 token 面
///   已先拦,这里兜底);
/// - `dtype` 精确相等;
/// - `rank`:声明 `0` = **显式阶数无关**(核按计数扁平索引);否则精确相等;
///   发送方 `rank = 0`(未定形)在张量槽 = **拒**(不许用"未定形"蒙混)。
fn check_tensor_facts(op: &str, p: &Param, i: usize, a: &Arg) -> Result<(), OpError> {
    let ParamType::Tensor { dtype: want_dtype, rank: want_rank } = p.ty else {
        return Ok(());
    };
    let Some((got_dtype, got_rank)) = a.facts() else {
        return Err(OpError::Contract {
            op: op.to_string(),
            field: p.name,
            expect: "块参(张量槽)".into(),
            got: format!("槽 {i} 无事实面"),
        });
    };
    if got_dtype != want_dtype {
        return Err(OpError::Contract {
            op: op.to_string(),
            field: p.name,
            expect: format!("dtype = {want_dtype:?}(声明)"),
            got: format!("槽 {i} 实得 {got_dtype:?}(线格式事实)"),
        });
    }
    if got_rank == 0 {
        return Err(OpError::Contract {
            op: op.to_string(),
            field: p.name,
            expect: "rank ≥ 1(发送方必须声明视图阶数)".into(),
            got: format!("槽 {i} rank=0(未定形)"),
        });
    }
    // 声明 rank = 0 ⇒ 显式阶数无关(核按计数扁平索引)
    if want_rank == 0 {
        // [临时审计 F5c] 声明 0 的槽,记录**实际喂进来的 rank** ——
        // 用途:找出哪些 "阶数无关" 其实在全部生产臂上都是固定阶数(可收紧)。
        // 去重键 = (op, 槽名, rank);`[probes] l3_audit` 才开(零成本关账)。
        if *L3_AUDIT_ON.get_or_init(|| {
            L3_AUDIT_REQUESTED.load(std::sync::atomic::Ordering::Relaxed)
        }) {
            l3_audit_note(op, p.name, got_rank);
        }
    }
    if want_rank != 0 && got_rank != want_rank {
        return Err(OpError::Contract {
            op: op.to_string(),
            field: p.name,
            expect: format!("rank = {want_rank}(声明)"),
            got: format!("槽 {i} 实得 {got_rank}(线格式事实)"),
        });
    }
    Ok(())
}

// ── [审计工具] 声明 rank=0 的槽实际喂进的阶数(F5c 立项面;`OWL_L3_AUDIT=1` 开)──
//
// 用途:rank=0 = 阶数无关(弱声明)。要判定某个槽**能不能收紧**,不能靠读
// 调用点猜 —— 跑一遂生产档(发几个请求)拿"实际阶数表",再逐槽判:
//   ① 同一 (op, 槽)出现 **多个不同 rank** ⇒ 必须维持 0(多臂共用/占位);
//   ② 只出现一个 rank 且跨档一致 ⇒ **候选收紧**(收紧前先确认其他 flavor
//      不走同一张表,否则 F5a 未验路径会当场被门拦);
//   ③ 槽本身是**占位/旗标**(如 `sinks` 槽恒不解引用)⇒ 诚实维持 0 。
// 关账成本 = 一次原子读(OnceLock)。开关唯一写点 = [`set_l3_audit`]
// (后端 boot 期由配置 `[probes] l3_audit` 下发;本 crate 零 env)。
static L3_AUDIT_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 写 L3 事实面审计开关(boot 期一次;重入以最后一次为准)
pub fn set_l3_audit(on: bool) {
    L3_AUDIT_REQUESTED.store(on, std::sync::atomic::Ordering::Relaxed);
}

static L3_AUDIT_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static L3_AUDIT_SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<(String, &'static str, u8)>>> =
    std::sync::OnceLock::new();

fn l3_audit_note(op: &str, slot: &'static str, rank: u8) {
    let seen = L3_AUDIT_SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::BTreeSet::new()));
    let fresh = match seen.lock() {
        Ok(mut s) => s.insert((op.to_string(), slot, rank)),
        Err(_) => false,
    };
    if fresh {
        eprintln!("[l3-audit] 声明0 实际{rank} {op}::{slot}");
    }
}

fn type_name(p: &Param) -> &'static str {    match p.ty {
        ParamType::Tensor { dtype, rank } => match (dtype, rank) {
            (Dtype::F32, _) => "tensor f32",
            (Dtype::F16, _) => "tensor f16",
            (Dtype::BF16, _) => "tensor bf16",
            (Dtype::U32, _) => "tensor u32",
            (Dtype::F8E4M3, _) => "tensor fp8-e4m3",
            (Dtype::F8E5M2, _) => "tensor fp8-e5m2",
            (Dtype::I8, _) => "tensor i8(int8)",
            (Dtype::U8, _) => "tensor u8(字节承载)",
        },
        ParamType::Scalar(ScalarTy::U64) => "u64",
        ParamType::Scalar(ScalarTy::I32) => "i32",
        ParamType::Scalar(ScalarTy::F32) => "f32",
        ParamType::Scalar(ScalarTy::F32Bits) => "f32bits",
        ParamType::Scalar(ScalarTy::I32Bits) => "i32bits",
    }
}

/// 线槽 → 词表 token
fn arg_token(a: &Arg) -> &'static str {
    match a {
        Arg::Block { .. } | Arg::BlockSlice { .. } => "T",
        Arg::U64(_) => "sz",
        Arg::I32(_) => "i32",
        Arg::F32(_) => "f32",
    }
}

/// 线槽 → 标量数值(范围检查用;块参返回 `None`)
fn arg_scalar(a: &Arg) -> Option<i64> {
    match a {
        Arg::U64(v) => Some(*v as i64),
        Arg::I32(v) => Some(*v as i64),
        Arg::F32(v) => Some(*v as i64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::bootcheck::{CheckCtx, Geom};
    use super::*;
    use crate::contract::Dtype;

    struct Demo;
    impl KernelSpec for Demo {
        const ENTRY: &'static str = "demo_op";
        const SRC: &'static str = "// demo";
        const PARAMS: &'static [Param] = &[
            Param::t_in("q", Dtype::F16, 2),
            Param::t_out("out", Dtype::F16, 2),
            Param::sz("n"),
            Param::f32("scale"),
        ];
        const LAWS: &'static [Law] = &[Law::OutputFinite];
    }

    /// 槽序是**派生**的:改 PARAMS 即改槽序,没有第二份手抄可漂移。
    #[test]
    fn args_derive_from_params() {
        let m = meta_of::<Demo>();
        assert_eq!(m.args_str(), "T,O,sz,f32");
        assert_eq!(m.out_pos(), Some(1));
        assert_eq!(m.entry, "demo_op");
        assert_eq!(m.laws, &[Law::OutputFinite]);
    }

    /// const 上下文可用(编译期建表):`args_len`/`args_bytes` 实测。
    #[test]
    fn args_derivable_in_const_context() {
        const N: usize = args_len(Demo::PARAMS);
        const B: [u8; N] = args_bytes::<{ N }>(Demo::PARAMS);
        const S: &str = unsafe { core::str::from_utf8_unchecked(&B) };
        assert_eq!(N, 10);
        assert_eq!(S, "T,O,sz,f32");
    }

    const GEOM: Geom = Geom { grid: (1, 1, 1), block: (256, 1, 1), smem: 0, out_elems: 4 };

    fn ctx<'a>(args: &'a [Arg], len: &'a dyn Fn(u64) -> Option<usize>) -> CheckCtx<'a> {
        CheckCtx { op: "demo_op", args, geom: &GEOM, len }
    }

    /// **事实面**:线格式携带的事实与声明一致 ⇒ 过;声明 `rank = 0` = 显式阶数无关。
    #[test]
    fn facts_plane_passes_and_rank_zero_is_wildcard() {
        let probe = |_id: u64| Some(8usize);
        // 声明事实 = 表自身(F16 rank2 / F16 rank2 / sz / f32)
        let args = [
            Arg::block(1, Dtype::F16, 2),
            Arg::block(2, Dtype::F16, 2),
            Arg::U64(8),
            Arg::F32(0.5),
        ];
        let c = ctx(&args, &probe);
        check_from_params("demo_op", Demo::PARAMS, &c).expect("事实一致应过");
        // 声明 rank = 0(阶数无关核)⇒ 任意阶数放行
        const ZERO_RANK: &[Param] =
            &[Param::t_in("a", Dtype::F16, 0), Param::t_out("out", Dtype::F16, 0)];
        let any = [Arg::block(1, Dtype::F16, 3), Arg::block(2, Dtype::F16, 5)];
        let c = ctx(&any, &probe);
        check_from_params("demo_op", ZERO_RANK, &c).expect("rank=0 应放行任意阶");
    }

    /// **负例**:dtype / rank 不符 / 发送方未定形(rank 0)⇒ 指名道姓的结构化拒。
    #[test]
    fn facts_plane_rejects_dtype_rank_and_unshaped() {
        let probe = |_id: u64| Some(8usize);
        let bad_dt = [
            Arg::block(1, Dtype::F32, 2), // 声明 F16
            Arg::block(2, Dtype::F16, 2),
            Arg::U64(8),
            Arg::F32(0.5),
        ];
        let e = check_from_params("demo_op", Demo::PARAMS, &ctx(&bad_dt, &probe)).unwrap_err();
        assert!(format!("{e}").contains("dtype"), "错应点名 dtype:{e}");
        let bad_rk = [
            Arg::block(1, Dtype::F16, 3), // 声明 rank2
            Arg::block(2, Dtype::F16, 2),
            Arg::U64(8),
            Arg::F32(0.5),
        ];
        let e = check_from_params("demo_op", Demo::PARAMS, &ctx(&bad_rk, &probe)).unwrap_err();
        assert!(format!("{e}").contains("rank"), "错应点名 rank:{e}");
        let unshaped = [
            Arg::block(1, Dtype::F16, 0), // 未定形(alloc 后未 view)
            Arg::block(2, Dtype::F16, 2),
            Arg::U64(8),
            Arg::F32(0.5),
        ];
        let e = check_from_params("demo_op", Demo::PARAMS, &ctx(&unshaped, &probe)).unwrap_err();
        assert!(format!("{e}").contains("未定形"), "未定形应拒:{e}");
    }

    /// 角色面:**输出槽必须是全块 `Block`**(BlockSlice 仅入参)。
    #[test]
    fn out_slot_rejects_blockslice() {
        let probe = |_id: u64| Some(8usize);
        let args = [
            Arg::block(1, Dtype::F16, 2),
            Arg::block_slice(2, 0, 4, Dtype::F16, 2),
            Arg::U64(8),
            Arg::F32(0.5),
        ];
        let e = check_from_params("demo_op", Demo::PARAMS, &ctx(&args, &probe)).unwrap_err();
        assert!(format!("{e}").contains("全块"), "线格式面应拒切片输出:{e}");
    }

    /// **块不在账本 ⇒ 拒**(旧实现是静默跳过零长/对齐检查)。
    #[test]
    fn default_check_rejects_unknown_block() {
        let unknown = |_id: u64| None;
        let args = [Arg::block(7, Dtype::F16, 2), Arg::block(2, Dtype::F16, 2), Arg::U64(8), Arg::F32(0.5)];
        let c = ctx(&args, &unknown);
        let e = check_from_params("demo_op", Demo::PARAMS, &c).unwrap_err();
        assert!(format!("{e}").contains("账本"), "未知块应早报:{e}");
    }

    /// 几何通用律:`out_elems ≥ 1`;grid 全零 = 显式"runtime 自持"哨兵。
    #[test]
    fn default_check_geometry_laws() {
        let probe = |_id: u64| Some(8usize);
        let args = [Arg::block(1, Dtype::F16, 2), Arg::block(2, Dtype::F16, 2), Arg::U64(8), Arg::F32(0.5)];
        let zero_out = Geom { grid: (1, 1, 1), block: (256, 1, 1), smem: 0, out_elems: 0 };
        let c = CheckCtx { op: "demo_op", args: &args, geom: &zero_out, len: &probe };
        let e = check_from_params("demo_op", Demo::PARAMS, &c).unwrap_err();
        assert!(format!("{e}").contains("out_elems"), "零元素输出应拒:{e}");
        let bad_grid = Geom { grid: (0, 2, 1), block: (256, 1, 1), smem: 0, out_elems: 4 };
        let c = CheckCtx { op: "demo_op", args: &args, geom: &bad_grid, len: &probe };
        let e = check_from_params("demo_op", Demo::PARAMS, &c).unwrap_err();
        assert!(format!("{e}").contains("grid"), "半零 grid 应拒:{e}");
        let sentinel = Geom { grid: (0, 0, 0), block: (0, 0, 0), smem: 0, out_elems: 4 };
        let c = CheckCtx { op: "demo_op", args: &args, geom: &sentinel, len: &probe };
        check_from_params("demo_op", Demo::PARAMS, &c).expect("全零哨兵 = runtime 自持,应放行");
    }

    /// 槽的事实来源:**输出槽 = `O` 位**(无 `O` ⇒ 末位 `T`)—— 与装配序单源。
    #[test]
    fn slot_sources_track_out_position() {
        let s = slot_sources(Demo::PARAMS); // T,O,sz,f32
        assert_eq!(s, vec![SlotSrc::Parent(0), SlotSrc::Out, SlotSrc::Scalar, SlotSrc::Scalar]);
        let legacy = [Param::t_in("a", Dtype::F16, 2), Param::t_in("out", Dtype::F16, 2)];
        assert_eq!(slot_sources(&legacy), vec![SlotSrc::Parent(0), SlotSrc::Out]);
    }

    /// 正例:槽数/类型/非零长全对 ⇒ 过
    #[test]
    fn default_check_passes_on_wellformed_args() {
        let probe = |_id: u64| Some(8usize);
        let args = [Arg::block(1, Dtype::F16, 2), Arg::block(2, Dtype::F16, 2), Arg::U64(8), Arg::F32(0.5)];
        let c = ctx(&args, &probe);
        assert!(meta_of::<Demo>().check(&c).is_ok());
    }

    /// 负例 ①:零长块 ⇒ 结构化报错(带算子名 + 参数名 + 槽位)
    #[test]
    fn default_check_rejects_zero_numel_with_names() {
        let probe = |id: u64| if id == 2 { Some(0usize) } else { Some(8) };
        let args = [Arg::block(1, Dtype::F16, 2), Arg::block(2, Dtype::F16, 2), Arg::U64(8), Arg::F32(0.5)];
        let c = ctx(&args, &probe);
        let e = meta_of::<Demo>().check(&c).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("demo_op"), "{s}");
        assert!(s.contains("out"), "未带参数名: {s}");
        assert!(s.contains("numel=0"), "{s}");
    }

    /// 负例 ②:槽类型错位(sz 槽喂 i32)⇒ 结构化报错带期望/实得
    #[test]
    fn default_check_rejects_slot_type_mismatch() {
        let probe = |_id: u64| Some(8usize);
        let args = [Arg::block(1, Dtype::F16, 2), Arg::block(2, Dtype::F16, 2), Arg::I32(8), Arg::F32(0.5)];
        let c = ctx(&args, &probe);
        let e = meta_of::<Demo>().check(&c).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("n") && s.contains("sz"), "{s}");
    }

    /// 几何门:声明驱动(只查已声明项;未声明 = 不约束)
    #[test]
    fn shape_family_accepts_declaration_driven() {
        let m = meta_of::<Demo>();
        // Demo 未声明 SHAPE ⇒ 任何几何都过
        assert!(m.shape.accepts(&ShapeQuery::default()).is_ok());

        let decl = ShapeFamily {
            kv_dtype: Some(Dtype::U8),
            head_dim: &[128, 256],
            page: &[32],
            grp: Some(6),
            tile: Some(16),
        };
        // 全对 ⇒ 过
        let ok = ShapeQuery {
            head_dim: 256,
            page: 32,
            grp: 6,
            tile: 8,
        };
        assert!(decl.accepts(&ok).is_ok());
        // head_dim 错 ⇒ 拒
        let bad_hd = ShapeQuery { head_dim: 192, ..ok };
        assert!(decl.accepts(&bad_hd).unwrap_err().contains("head_dim"));
        // page 错 ⇒ 拒
        let bad_page = ShapeQuery { page: 16, ..ok };
        assert!(decl.accepts(&bad_page).unwrap_err().contains("page"));
        // tile 超上界 ⇒ 拒
        let bad_tile = ShapeQuery { tile: 32, ..ok };
        assert!(decl.accepts(&bad_tile).unwrap_err().contains("行数上界"));
        // **未知事实不检查**(0/None ⇒ 放行)
        let unknown = ShapeQuery { head_dim: 0, page: 0, grp: 0, tile: 0 };
        assert!(decl.accepts(&unknown).is_ok());
    }

    /// 负例 ③:槽数不符 ⇒ 报期望/实得(组合期编程错,不是运行期猜)
    #[test]
    fn default_check_rejects_arity_mismatch() {
        let probe = |_id: u64| Some(8usize);
        let args = [Arg::block(1, Dtype::F16, 2), Arg::block(2, Dtype::F16, 2)];
        let c = ctx(&args, &probe);
        let e = meta_of::<Demo>().check(&c).unwrap_err();
        assert!(e.to_string().contains("槽数"), "{e}");
    }
}
