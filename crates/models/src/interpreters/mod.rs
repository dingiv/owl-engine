//! 解释器集合(装载/计算等执行域变体;声明式范式的执行侧)。
//!
//! 模型层是纯声明(TensorOps 链 / LoaderOps 清单);**解释器 = 声明的
//! 落地封装**:把声明翻译为后端原语(alloc / lower / launch / htod),
//! 后端经 face 泛型注入(`owl-iface::contract::DeviceClient`),解释器
//! 本身零后端代码。毒值在解释器边界落地(结构化报错 + depth 归因)。
//!
//! | 变体 | 文件 | 职责 | 入口 |
//! |---|---|---|---|
//! | 计算(推理) | [`eval`] | TensorOps DAG 归约(CSE + 毒值落地 + C1 断言) | [`eval`] / [`eval_ops`] / [`eval_ops_tap`] |
//! | 装载 | [`load`] | Loadable::layout → 取数/变换/物化(流式分块 + 并发流水) | [`eval_load`] |
//! | 观测面(tap) | [`observe`] | 节点级单步调试的事件词汇与协议(挂在 eval 归约点) | [`observe::Tap`] / [`observe::StatsTap`] |
//!
//! 新变体(图捕获回放、量化装载、其他平台)按域另起文件,公共词汇
//! 上提至本文件 —— 勿在变体间互相依赖。
//!
//! 后端契约 = `owl-iface::contract::DeviceClient`(线格式同源)。
//! 同步参考解释器(reduce/CpuInterpreter,对拍锚)见 [`crate::reference`]。
//! 毒值传播契约:构造期违约随链流动,`is_poisoned()` 可查,边界收割。
//!
//! **维度源头单一律(C1,2026-09-26 定案)**:一切维度推导只读声明
//! shape;`Bytes.len` 不参与语义(Block 叶子 len=0),仅边界断言。

pub mod eval;
pub mod load;
pub mod observe;

pub use eval::{eval, eval_ops, eval_ops_tap};
pub use load::eval_load;
pub use observe::{BlockRef, BlockStats, NodeEvent, StatRecord, StatsTap, Tap, Want};
