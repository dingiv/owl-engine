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
//! - 登记表([`crate::list`])对**已建表**的算子退化为"名字 → 源"的兜底,
//!   槽序/源码改从本模块的元数据取(未建表的旧入口继续走登记表字符串);
//! - [`crate::driver`] 的 `KernelPick` 携带 `&'static KernelMeta`(算子身份),
//!   不再携带裸名字字符串。

use crate::contract::{Arg, Dtype, Law, OpError};

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
    pub const fn t_out(name: &'static str, dtype: Dtype, rank: u8) -> Self {
        Self {
            name,
            role: ParamRole::Out,
            ty: ParamType::Tensor { dtype, rank },
            shape: ShapeRule::NONE,
            range: RangeRule::POSITIVE,
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

/// 自检缓冲区(设备指针 + 字节数;自检面零 feature 门 ⇒ 不复用 `ScratchBuf`)
#[derive(Debug, Clone, Copy)]
pub struct CheckBuf {
    pub ptr: u64,
    pub bytes: usize,
}

/// **自检运行时**(ungated 抽象;device 门内的适配器把 `DeviceRes × Exec`
/// 桥到本面 —— 这样 oracle 住零门模块,而 device 面照旧门控)。
///
/// 五个方法 = 装配期 oracle 的全部需求:分配 / 上传 / 回读 / 发射 / 同步。
pub trait CheckRunner {
    fn alloc(&mut self, bytes: usize, tag: &'static str) -> Result<CheckBuf, OpError>;
    /// **账本块分配**(foreign 算子自检必需:它们的参数是块句柄,不是 scratch)
    fn alloc_block(&mut self, bytes: usize, tag: &'static str) -> Result<crate::contract::Bytes, OpError>;
    /// 按块句柄上传(等价于 `resolve` + `upload`)
    fn upload_block(&mut self, b: &crate::contract::Bytes, src: &[u8]) -> Result<(), OpError>;
    /// 按块句柄回读(等价于 `resolve` + `download`)
    fn download_block(&mut self, b: &crate::contract::Bytes, dst: &mut [u8]) -> Result<(), OpError>;
    /// **经注册表发射**(与生产同一路径:`OpRegistry::execute`)—— foreign
    /// 算子(FFI/家族 runtime)只能这样跑;native 用 [`launch_nvrtc`](Self::launch_nvrtc)。
    fn dispatch(&mut self, msg: crate::contract::LaunchMsg) -> Result<crate::contract::Bytes, OpError>;
    fn upload(&mut self, dst: CheckBuf, src: &[u8]) -> Result<(), OpError>;
    /// 设备 → host 回读(**独立参考比对的唯一入口**)
    fn download(&mut self, src: CheckBuf, dst: &mut [u8]) -> Result<(), OpError>;
    /// NVRTC 源码发射(符号名 = `name`;值表见 [`crate::contract::LaunchVal`])
    fn launch_nvrtc(
        &mut self,
        name: &'static str,
        source: &'static str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        vals: &[crate::contract::LaunchVal],
    ) -> Result<(), OpError>;
    /// 流同步(回读前必须;自检是装配期一次性动作,不走热路径纪律)
    fn sync(&mut self) -> Result<(), OpError>;
}

/// 装配期自检(**独立 oracle**;设计 §5.5)。
///
/// 三条纪律:① 参考必须是**独立实现**(同源副本对不出双错,E8);
/// ② 装配期默认执行,失败 = **拒启**;③ 报告含逐元素 `max|err|` + 首个越界元素。
pub trait BootCheck: Send + Sync {
    /// 自检名(报告/boot 日志用)
    fn name(&self) -> &'static str;
    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError>;
}

/// 自检报告(逐元素误差 + 首个越界元素 + 毒值律结论)
#[derive(Debug, Clone)]
pub struct CheckReport {
    pub op: &'static str,
    pub worst_abs_err: f32,
    pub first_bad: Option<(usize, f32, f32)>,
    pub tolerance: f32,
    /// 毒值自检结论(`Some(未写字节数)` = 声明写集之外的字节仍为毒值 ⇒ 写集穷举成立)
    pub poison_untouched: Option<usize>,
}

impl CheckReport {
    /// 纯数值型报告(无数值越界 = `first_bad: None`)
    pub fn numeric(op: &'static str, worst_abs_err: f32, tolerance: f32) -> Self {
        Self { op, worst_abs_err, first_bad: None, tolerance, poison_untouched: None }
    }
}

// ============================================================================
// §4 发射期检查上下文(L3 哨兵)
// ============================================================================

/// 发射几何(从 [`crate::contract::LaunchMsg`] 提取)
#[derive(Debug, Clone, Copy)]
pub struct Geom {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
    pub out_elems: usize,
}

/// 块 id → **字节数**(server 账本查询;`None` = 该账本不查/未知)。
///
/// 口径 = 字节(账本以字节记账;`VramBlock.len` 在 cuda 面也是字节)。
/// 零长律(`> 0`)与对齐律(`align` 是字节对齐)在此口径下都成立。
pub type LenProbe<'a> = &'a dyn Fn(u64) -> Option<usize>;

/// 检查上下文:一次发射的全部可见事实
pub struct CheckCtx<'a> {
    pub op: &'static str,
    pub args: &'a [Arg],
    pub geom: &'a Geom,
    pub len: LenProbe<'a>,
}

impl CheckCtx<'_> {
    /// 槽位字节数(块参查账本;`BlockSlice` 用切片自带的元素数 —— 无 dtype
    /// 信息时按 1B/elem 上报,零长律不受影响)
    pub fn arg_len(&self, i: usize) -> Option<usize> {
        match self.args.get(i)? {
            Arg::Block { id } => (self.len)(*id),
            Arg::BlockSlice { elems, .. } => Some(*elems as usize),
            _ => None,
        }
    }
}

// ============================================================================
// §5 算子契约 trait(关联常量;编译期擦除)
// ============================================================================

/// 单算子契约。**只有关联常量 + 一个可默认的静态方法**(无 `&self`):
/// 带关联常量的 trait 不能做 trait 对象(rustc E0038 实证),这不是代价 ——
/// 它把算子侧彻底静态化,元数据在编译期擦成 [`KernelMeta`] 纯数据表。
pub trait KernelSpec: 'static {
    /// 线格式入口名(与某个 `.cu` 入口一对一)
    const ENTRY: &'static str;
    /// `.cu` 源码(`include_str!`;与算子同目录)
    const SOURCE: &'static str;
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
        entry: T::ENTRY,
        source: T::SOURCE,
        params: T::PARAMS,
        shape: T::SHAPE,
        pool: T::POOL,
        laws: T::LAWS,
        capture: T::CAPTURE,
        determinism: T::DETERMINISM,
        check: T::check,
        selfcheck: T::SELFCHECK,
    }
}

/// 擦除后的算子元数据(纯数据;可在 `static` 表里 const 构造)
pub struct KernelMeta {
    pub entry: &'static str,
    pub source: &'static str,
    pub params: &'static [Param],
    pub shape: ShapeFamily,
    pub pool: &'static [PoolAccess],
    pub laws: &'static [Law],
    pub capture: CaptureEvidence,
    pub determinism: Determinism,
    /// L3 检查函数(`<T as KernelSpec>::check` 的 fn 指针)
    pub check: fn(&CheckCtx<'_>) -> Result<(), OpError>,
    pub selfcheck: Option<&'static dyn BootCheck>,
}

impl std::fmt::Debug for KernelMeta {
    /// 手写 Debug:`selfcheck: Option<&dyn BootCheck>` 无 Debug(且打印它没意义),
    /// 诊断关心的是名字/槽序/律
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelMeta")
            .field("entry", &self.entry)
            .field("slots", &self.args_str())
            .field("laws", &self.laws)
            .field("capture", &self.capture)
            .field("determinism", &self.determinism)
            .field("selfcheck", &self.selfcheck.is_some())
            .finish()
    }
}

impl KernelMeta {
    /// 槽序 token 序列(由 [`PARAMS`](Self::params) 派生 —— 零第二份手抄)
    pub fn slot_tokens(&self) -> Vec<&'static str> {
        self.params.iter().map(token_of).collect()
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

    /// 发射期检查(统一入口;**server 唯一调用点**)
    pub fn check(&self, ctx: &CheckCtx<'_>) -> Result<(), OpError> {
        (self.check)(ctx)
    }
}

// ============================================================================
// §6 默认检查(由 PARAMS 自动生成;设计 §5.2)
// ============================================================================

/// 表驱动的默认检查:数量对位 / 槽类型对位 / 块非零长 / 标量范围。
///
/// 能表达的自动机检;跨参耦合(如 `q_seq_len` 与 `mask` 长度)由算子覆写。
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
        // ② 张量槽:非零长(全局硬律;xqa_sem 案)
        if p.kind() == ParamKind::Tensor {
            if let Some(n) = ctx.arg_len(i) {
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
        }
        // ③ 标量范围
        if let Some(v) = arg_scalar(a) {
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
        }
    }
    Ok(())
}

fn type_name(p: &Param) -> &'static str {
    match p.ty {
        ParamType::Tensor { dtype, rank } => match (dtype, rank) {
            (Dtype::F32, _) => "tensor f32",
            (Dtype::F16, _) => "tensor f16",
            (Dtype::BF16, _) => "tensor bf16",
            (Dtype::U32, _) => "tensor u32",
            (Dtype::U8, _) => "tensor u8(字节池)",
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
    use super::*;
    use crate::contract::Dtype;

    struct Demo;
    impl KernelSpec for Demo {
        const ENTRY: &'static str = "demo_op";
        const SOURCE: &'static str = "// demo";
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

    /// 正例:槽数/类型/非零长全对 ⇒ 过
    #[test]
    fn default_check_passes_on_wellformed_args() {
        let probe = |_id: u64| Some(8usize);
        let args = [Arg::Block { id: 1 }, Arg::Block { id: 2 }, Arg::U64(8), Arg::F32(0.5)];
        let c = ctx(&args, &probe);
        assert!(meta_of::<Demo>().check(&c).is_ok());
    }

    /// 负例 ①:零长块 ⇒ 结构化报错(带算子名 + 参数名 + 槽位)
    #[test]
    fn default_check_rejects_zero_numel_with_names() {
        let probe = |id: u64| if id == 2 { Some(0usize) } else { Some(8) };
        let args = [Arg::Block { id: 1 }, Arg::Block { id: 2 }, Arg::U64(8), Arg::F32(0.5)];
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
        let args = [Arg::Block { id: 1 }, Arg::Block { id: 2 }, Arg::I32(8), Arg::F32(0.5)];
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
        let args = [Arg::Block { id: 1 }, Arg::Block { id: 2 }];
        let c = ctx(&args, &probe);
        let e = meta_of::<Demo>().check(&c).unwrap_err();
        assert!(e.to_string().contains("槽数"), "{e}");
    }
}
