//! 引擎核心 —— 严格显存预算(A5)、内存规划器(A1.1)、调度与 runner(A3)。
//!
//! 依赖方向:core → {graph, comm, kernels, iface};server crate(M2)只进
//! tokio 侧。上层永远面向 owl-iface 编程,不感知后端(见 backends/iface)。

use owl_graph::GraphAllowance;

/// **A5 硬预算**:启动时声明,生命周期恒不超。预算是合同,不是愿望。
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// 预算上限(bytes/卡),如 23G
    pub bytes: u64,
    /// 运行时底价:CUDA context/驱动等不可避免开销(实测登记,非估算)
    pub runtime_floor_bytes: u64,
    /// 安全余量(未预见的碎片等;过大说明分解不完整,应回到登记)
    pub safety_margin_bytes: u64,
    /// A5.3 对账警线:free 低于此值告警
    pub reserve_floor: u64,
}

/// 内存规划器 —— 池计算的第一句话就是扣图预算(A1.1)。
pub struct MemoryPlan {
    /// 规划输入:设备总显存(bytes/卡)
    pub total_bytes: u64,
    /// 权重驻留(量化后)
    pub weights_bytes: u64,
    /// 激活工作区峰值(由 batched_tokens 推)
    pub activation_bytes: u64,
    /// 图预算(来自 GraphGovernor,单一来源)
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

    /// **A5.1 分解封闭性静态证明**:启动前校验。预算 B 的每一项都必须
    /// 有数字且总和 ≤ B;超出则拒绝启动——"先跑起来再看显存"不存在。
    pub fn validate_budget(
        &self,
        budget: &Budget,
    ) -> Result<BudgetProof, BudgetError> {
        let known = self
            .weights_bytes
            .saturating_add(self.activation_bytes)
            .saturating_add(self.allowance.total())
            .saturating_add(budget.runtime_floor_bytes)
            .saturating_add(budget.safety_margin_bytes);
        if known > budget.bytes {
            return Err(BudgetError::OverCommitted {
                budget: budget.bytes,
                known,
            });
        }
        Ok(BudgetProof {
            kv_pool_bytes: self.kv_pool_bytes(),
            // 未分配余量 = B - 已分解项;过大说明分解不完整(A5.1 审计点)
            unallocated: budget.bytes - known,
        })
    }
}

/// A5.1 通过后的预算证明:启动日志应打印它,审计有据可查
#[derive(Debug, Clone, Copy)]
pub struct BudgetProof {
    /// 静态证明通过后的 KV 池容量
    pub kv_pool_bytes: u64,
    /// 未分配余量(应接近安全余量;过大 = 分解项缺失)
    pub unallocated: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    /// A5.1:分解总和超预算,拒绝启动
    #[error("A5.1 预算超支:分解总和 {known}B > 预算 {budget}B,拒绝启动")]
    OverCommitted { budget: u64, known: u64 },
}

/// 调度决策的形状输入(REQ:并发是约束不是目标,≤8 档位静态定)。
pub struct ScheduleShape {
    pub batched_tokens: u32,
    pub graph_batches: Vec<u32>,
}

/// **A2.7 故障语义**:全有或全无。任一 runner 报错 → Unrecoverable。
/// 不做 rank 级热恢复;恢复 = 整进程重启。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthState {
    Starting,
    Running,
    /// 冻结态:调度停止,全部设备 LedgerSnapshot 已收集,错误链完整
    Unrecoverable,
}

/// 引擎健康看板:故障时从这里读"发生了什么 + 内存现状"
pub struct EngineHealth {
    pub state: HealthState,
    /// 违约/报错时的错误链(完整,不截断)
    pub error_chain: Vec<String>,
}

/// runner 契约(A3 同步隔离):
/// - runner 线程独占全部 CUDA 上下文与 device 内存句柄;
/// - tokio 侧通过有界 channel 递请求、收 logits,自己**永不**触碰 device;
/// - 全引擎唯一 D2H 点 = logits 回传,位于 runner 内。
///
/// M1 前不实现执行体;此 trait 先冻结边界,防止"顺手在 async 里碰 device"。
pub trait RunnerLoop: Send {
    /// 返回 Err = A2.7 触发:引擎进入 Unrecoverable,错误链上交,
    /// 本线程不再触碰任何 device(A3 纪律在死亡时也不破例)。
    fn run(
        self,
        plan: MemoryPlan,
        shape: ScheduleShape,
        budget: Budget,
    ) -> Result<(), Vec<String>>;
}
