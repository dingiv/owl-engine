//! Module:模型层统一接口(一切 layer/model 的计算声明入口)。
//!
//! 接口显化(2026-09-25 用户裁决):layer 与 model 的共性 = 「吃一个
//! 输入声明,吐一个输出声明」——
//!
//! ```rust,ignore
//! fn forward(&self, xs: &TensorOps) -> TensorOps;
//! ```
//!
//! 三律(与 TensorOps 同构):
//! - **同步**:零 await;
//! - **total**:零 `?` 零 panic —— 构造期违约与未装载槽均已毒值化,
//!   随链流动,`eval` 边界收割;
//! - **纯描述**:零执行(执行 = `interpreter::eval`)。
//!
//! 装载生命周期不在本接口 —— 见 `loader::Loadable`(layout + mount 已
//! 被执行器吸收,层侧唯一钩子是 layout)。计算与装载分离:
//! `Module` 管声明,`Loadable` 管数据需求,`interpreter` 管执行。
//!
//! 归位说明:Rope(位置上下文层,双输入 + pos)暂不进本接口 ——
//! 统一签名需 ForwardCtx(每步动态依赖 grab-bag,async-runtime.md §4.3),
//! 随 runner 立项。

use crate::TensorOps;

/// 每步计算上下文(forward 的 ctx;动态依赖 grab-bag,async-runtime §4.2:
/// 算子缺什么放什么,由 runner 每步构造注入 —— 层只透传,不自取)。
/// 按值传递(Copy);字段随域扩。
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelCtx {
    /// 本步 token 数(decode = 1;prefill = 块长)
    pub tokens: usize,
}

/// 模型层统一接口
pub trait Module {
    /// 计算声明:xs → y(同步 · total · 纯描述;ctx = 动态依赖)
    fn forward(&self, xs: &TensorOps, ctx: &KernelCtx) -> TensorOps;
}
