//! 语义 Op 枚举:client 对 server 能力期望的清单。
//! server 照此实现 dispatch;新增能力 = 新变体 + server 一臂。
//! (原 Step/TensorMeta 已解体进 Tensor——2026-09-23 用户裁决。)

use crate::kernel::Kernel;

/// Kernel 节点的参数槽(有序)
/// 类型化:标量按 kernel 形参宽度入槽(CUDA 参数空间自然对齐,
/// 宽槽顶窄形参会错位读參 —— 与 client::Arg 同一纪律)
#[derive(Clone, Debug)]
pub enum KernelArg {
    /// 张量依赖:归约序保证先算;发射时 server 解 id → 设备指针
    T { id: u64 },
    /// 8 字节标量(size_t/u64)
    Bits(u64),
    /// 4 字节有符号整数
    I32(i32),
    /// 4 字节浮点
    F32(f32),
}

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
    /// 同形逐元素乘(MLP 门控 / 注意力输出门)
    Mul,
    Silu,
    /// w_off = ×(1+w) 语义(use_norm_offset)
    Rmsnorm { eps: f32, w_off: bool },
    Rope { theta_base: f64 },
    Embedding,
    /// paged attention(带槽位;server 侧 kernel 从 attention-rs port)
    PagedAttn,

    // ---- Kernel 节点(2026-09-23 定稿:节点的本质形态,funio Pack 同源)----
    /// 携带核函数值(Kernel{name, source})+ 有序参数槽。
    /// 解释器:CPU = 结构化"需 GPU server";GPU = 懒编译(源哈希缓存)+ 发射。
    /// 参数槽有序:T(张量依赖)/ Bits(标量位型);归约时张量参数先入账。
    /// Kernel 节点(参数槽在 TensorOps.args;归约时随发射消息打包)
    Kernel { kernel: Kernel },

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

    /// 草稿语义:join = 相加(map = silu);正式组合子随声明式算子面扩展
    fn join(&self, another: &Self, op: Op) -> Self {
        match op {
            Op::Add => self.add(another),
            _ => self.add(another),
        }
    }

    fn map(&self, op: Op) -> Self {
        match op {
            Op::Silu => self.silu(),
            _ => self.clone(),
        }
    }
}