//! **装配期自检面**(G 轮 2026-10-13 自 `spec.rs` 拆出)。
//!
//! 与 `spec.rs` 的界:**声明面 = 能进 `const` 上下文的词汇与派生**(参数表/
//! 形状族/池/律,`KernelSpec`/`KernelMeta`);**自检面 = 要跑设备的执行面**
//! ([`CheckRunner`] 是 GPU 执行 trait,`BootCheck` 是它的实现合约)。
//!
//! ```text
//! KernelSpec::SELFCHECK: Option<&'static dyn BootCheck>   ← 声明指出"要自检"
//!         │  registry::assemble 逐算子跑
//!         ▼
//! BootCheck::run(&mut dyn CheckRunner) -> CheckReport     ← 本模块的执行合约
//! ```
//!
//! 消费者:`registry::assemble`(装配期门)· `server::selfcheck`(GPU 实现)·
//! `ops/oracles/*`(逐算子实现)。**零 feature 门**(`CheckBuf` 是裸设备指针
//! 包装,不复用 `device::ScratchBuf`)。

use crate::contract::{Arg, OpError};

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
    /// 算子(FFI/家族 runtime)只能这样跑;native 用 [`launch_aot`](Self::launch_aot)。
    fn dispatch(&mut self, msg: crate::contract::LaunchMsg) -> Result<crate::contract::Bytes, OpError>;
    fn upload(&mut self, dst: CheckBuf, src: &[u8]) -> Result<(), OpError>;
    /// 设备 → host 回读(**独立参考比对的唯一入口**)
    fn download(&mut self, src: CheckBuf, dst: &mut [u8]) -> Result<(), OpError>;
    /// **构建期资产发射**(符号名 = `name`;源哈希查资产,零运行期编译)。
    /// 值表见 [`crate::contract::LaunchVal`]。
    fn launch_aot(
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
            Arg::Block { id, .. } => (self.len)(*id),
            Arg::BlockSlice { elems, .. } => Some(*elems as usize),
            _ => None,
        }
    }
}
