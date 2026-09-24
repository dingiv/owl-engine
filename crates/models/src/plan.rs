//! 语义 Op 枚举:client 对 server 能力期望的清单。
//! server 照此实现 dispatch;新增能力 = 新变体 + server 一臂。
//! (原 Step/TensorMeta 已解体进 Tensor——2026-09-23 用户裁决。)

use crate::interpreter::Value;

/// 语义运算
#[derive(Clone, Debug)]
pub enum Op {
    // ---- 源节点 ----
    /// host 数据入块(server 侧 pinned 码头直填 + 池流 htod async+sync;契约 3)。
    /// 值语义:数据随树走(配置面;server 版换成 pinned 码头引用)。
    Htod { bytes: Vec<u8> },
    /// 清零分配
    Zeros,
    /// 引用已物化数据(rt::Tensor 的池块;id = 其全局唯一身份证)。
    /// eval 时零操作(数据已在),仅把"真数据"接进声明图的叶子。
    Block { id: u64 },

    // ---- 计算节点 ----
    /// [m,k] × [k,n]
    Matmul,
    Add,
    Silu,
    /// w_off = ×(1+w) 语义(use_norm_offset)
    Rmsnorm { eps: f32, w_off: bool },
    Rope { theta_base: f64 },
    Embedding,
    /// paged attention(带槽位;server 侧 kernel 从 attention-rs port)
    PagedAttn,

    // ---- 逃逸舱:client 侧自定义算子(fn 指针;CPU/客户端执行)----
    // ⚠️ 架构注记:fn 指针无法序列化成 kernel——这两个变体**不进捕获图、
    // 不上 GPU**,由 CPU 解释器就地执行。真 GPU 自定义算子的正路 =
    // load_kernel(装载源码)+ Launch(发射),能力面已覆盖。
    // f 指针保留 Clone+Debug+Send+Sync(fn 指针天生全有;boxed 闭包三样全丢)。
    /// f(Tensor) -> Tensor:一元 client 侧变换
    UnaryFn { f: fn(Value) -> Value },
    /// f(Tensor, Tensor) -> Tensor:二元 client 侧变换
    BinaryFn { f: fn(Value, Value) -> Value },

    // ---- 状态节点(唯一显式副作用;SSA 外形,物理原地由 server 解释)----
    /// KV 写槽:声明"本节目写 kv manager 的这些格"——
    /// 跨节目依赖机械可导:同一 kv 格的写/读节目之间插事件边。
    SlotWrite,

}

/// 计划节点能力(草稿,2026-09-23 用户手写骨架的形式化):
/// - `op()`:节点语义(server dispatch 的键);
/// - join/map:链式组合子(声明式运算的动词)。
/// Tensor 是首个实现;将来别的节点形态(如捕获烘焙产物)也可实现。
pub trait PlanNode {
    /// 节点语义
    fn op(&self) -> &Op;

    /// 二元组合:与 another 合并,产出新声明节点
    fn join(&self, another: &Self, op: Op) -> Self;

    /// 一元组合:本节点的变体(如激活/形状变换)
    fn map(&self, op: Op) -> Self;
}

impl PlanNode for crate::tensor::TensorOps {
    fn op(&self) -> &Op {
        &self.op
    }

    fn join(&self, another: &Self, op: Op) -> Self {
        crate::tensor::TensorOps::join(self, op, Some(another), self.dtype, self.shape.clone())
    }

    fn map(&self, op: Op) -> Self {
        crate::tensor::TensorOps::join(self, op, None, self.dtype, self.shape.clone())
    }
}