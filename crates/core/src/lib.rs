//! 引擎核心 —— 内存规划器(A1.1 落点)、调度与 runner(A3 同步隔离)。
//!
//! 依赖方向:core → {graph, comm, kernels};server crate(M2)只进 tokio 侧。

use owl_graph::{GraphAllowance, GraphGovernor};

/// 内存规划器 —— 池计算的第一句话就是扣图预算(A1.1)。
///
/// 与 xinfer 的本质差异:那里是"池建好后图来要内存",这里是
/// "图预算是规划输入,池天生是扣除后的余量"。
pub struct MemoryPlan {
    /// 规划输入:设备总显存(bytes/卡)
    pub total_bytes: u64,
    /// 权重驻留(量化后)
    pub weights_bytes: u64,
    /// 激活工作区峰值(由 batched_tokens 推)
    pub activation_bytes: u64,
    /// 图预算(来自 GraphGovernor,单一样式来源)
    pub allowance: GraphAllowance,
}

impl MemoryPlan {
    /// KV 池容量(字节/卡)。A1.1:图预算在一切池计算之前扣除。
    pub fn kv_pool_bytes(&self) -> u64 {
        self.total_bytes
            .saturating_sub(self.weights_bytes)
            .saturating_sub(self.activation_bytes)
            .saturating_sub(self.allowance.total())
    }
}

/// 调度决策的形状输入(REQ:并发是约束不是目标,≤8 档位静态定)。
pub struct ScheduleShape {
    pub batched_tokens: u32,
    pub graph_batches: Vec<u32>,
}

/// runner 契约(A3 同步隔离):
/// - runner 线程独占全部 CUDA 上下文与 device 内存句柄;
/// - tokio 侧通过有界 channel 递请求、收 logits,自己**永不**触碰 device;
/// - 全引擎唯一 D2H 点 = logits 回传,位于 runner 内。
///
/// M1 前不实现执行体;此 trait 先冻结边界,防止"顺手在 async 里碰 device"。
pub trait RunnerLoop: Send {
    fn run(self, plan: MemoryPlan, shape: ScheduleShape, governor: GraphGovernor);
}
