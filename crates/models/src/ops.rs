//! 声明算子面:语义 Op 枚举(声明什么)+ lower 动作表(怎么翻译)。
//!
//! 两半一个主题(2026-09-26 重组:原 plan.rs + actions.rs 合并):
//! - **Op**:client 对 server 能力期望的清单;server 照此实现 dispatch,
//!   新增能力 = 新变体 + server 一臂 + lower 一函数;
//! - **lower_***:具名算子 → LaunchMsg 的唯一翻译通道(声明语义落线格式)。
//!
//! kernel 源**不住这里** —— 经 [`crate::kernel`] 注册表按名取用(源码之家
//! = owl-kernels cu/;2026-09-25 垫子层裁决)。用户自定义 kernel 走
//! `TensorOps::of(Kernel)` 直带源码(不经注册表,组合面逃生舱)。

use crate::contract::{Arg, Bytes, KernelSpec, LaunchMsg};
use crate::kernel::Kernel;

// ============================================================================
// §1 Kernel 节点参数槽(有序;类型化)
// ============================================================================

/// Kernel 节点的**标量**参数槽(有序)。
///
/// 张量依赖不在本枚举 —— `TensorOps.parents` 即张量槽(每个节点自身
/// 声明输出类型 = f32 块,消费方查父输出即可,无需在 args 里重复标 T;
/// 2026-09-26 用户裁决)。发射时槽序由签名唯一权威(注册表 Entry.args /
/// 逃生舱 kernel.sig)对位:T ↔ 下一个父块,sz/i32/f32 ↔ 下一个标量。
/// 类型化纪律:CUDA 参数空间自然对齐,宽槽顶窄形参会错位读參。
#[derive(Clone, Debug)]
pub enum KernelArg {
    /// 8 字节标量(size_t/u64;与 kernel `size_t` 形参严格对位)
    Bits(u64),
    /// 4 字节有符号整数
    I32(i32),
    /// 4 字节浮点
    F32(f32),
}

// ============================================================================
// §2 语义 Op 枚举
// ============================================================================

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
    /// 逐元素 sigmoid(attn_output_gate 门;GDN beta 同族)
    Sigmoid,
    /// w_off = ×(1+w) 语义(use_norm_offset);归一化宽度由 alpha 定义
    /// (x 任意前导维折叠为行)。**C8 权威定案**:本变体 + owl_rmsnorm_f32
    /// (ops.cu)为该语义唯一权威,reference.rs / owl-cpu ops 为对拍副本;
    /// **C12 定案**:w_off 留 flag 不拆 op(qk-norm 是唯一 add_one 用户)。
    Rmsnorm { eps: f32, w_off: bool },
    Rope { theta_base: f64 },
    Embedding,
    /// paged attention(带槽位;server 侧 kernel 从 attention-rs port)
    PagedAttn,
    /// 形状重解释(纯元数据视图;元素数守恒;eval 透传父块,零拷贝)
    Reshape,

    // ---- Kernel 节点(2026-09-23 定稿:节点的本质形态,funio Pack 同源)----
    /// 携带核函数值(Kernel{name, source})+ 有序参数槽。
    /// 解释器:CPU = 结构化"需 GPU server";GPU = 懒编译(源哈希缓存)+ 发射。
    /// 参数槽有序:T(张量依赖)/ 标量;归约时张量参数先入账。
    Kernel { kernel: Kernel },

    // ---- 状态节点(唯一显式副作用;SSA 外形,物理原地由 server 解释)----
    /// KV 写槽:声明"本节目写 kv manager 的这些格"——
    /// 跨节目依赖机械可导:同一 kv 格的写/读节目之间插事件边。
    SlotWrite,
}

// ============================================================================
// §3 PlanNode(草稿,2026-09-23 用户手写骨架的形式化)
// ============================================================================

/// 计划节点能力(草稿):
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
            Op::Sigmoid => self.sigmoid(),
            _ => self.clone(),
        }
    }
}

// ============================================================================
// §4 lower 动作表:具名算子 → LaunchMsg(唯一翻译通道)
// ============================================================================

fn spec(name: &'static str) -> KernelSpec {
    KernelSpec { name: name.to_string(), source: crate::kernel::source(name).to_string() }
}

fn ceil_1d(n: usize) -> (u32, u32, u32) {
    ((n as u32 + 255) / 256, 1, 1)
}

/// lower:Add(同形二元;a/b 块等长由 server 账长校验兜底)
/// lower:Add(同形二元;a/b 块等长由 server 账长校验兜底)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)
pub fn lower_add(ins: &[Bytes], out: &Bytes, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: spec("owl_add_f32"),
        args: vec![
            Arg::Block { id: ins[0].id },
            Arg::Block { id: ins[1].id },
            Arg::Block { id: out.id },
            Arg::U64(n as u64),
        ],
        grid: ceil_1d(n),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// lower:Mul(同形逐元素乘)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)
pub fn lower_mul(ins: &[Bytes], out: &Bytes, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: spec("owl_mul_f32"),
        args: vec![Arg::Block { id: ins[0].id }, Arg::Block { id: ins[1].id }, Arg::Block { id: out.id }, Arg::U64(n as u64)],
        grid: ceil_1d(n),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// lower:Silu(一元)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)
pub fn lower_silu(ins: &[Bytes], out: &Bytes, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: spec("owl_silu_f32"),
        args: vec![Arg::Block { id: ins[0].id }, Arg::Block { id: out.id }, Arg::U64(n as u64)],
        grid: ceil_1d(n),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// lower:Sigmoid(一元;attn_output_gate 门 / GDN beta 同族)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)
pub fn lower_sigmoid(ins: &[Bytes], out: &Bytes, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: spec("owl_sigmoid_f32"),
        args: vec![Arg::Block { id: ins[0].id }, Arg::Block { id: out.id }, Arg::U64(n as u64)],
        grid: ceil_1d(n),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// lower:Matmul([m,k]×[k,n];m/k/n 全部来自声明 shape(C1))
pub fn lower_matmul(ins: &[Bytes], out: &Bytes, m: usize, k: usize, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: spec("owl_matmul_f32"),
        args: vec![
            Arg::Block { id: ins[0].id },
            Arg::Block { id: ins[1].id },
            Arg::Block { id: out.id },
            Arg::I32(m as i32),
            Arg::I32(k as i32),
            Arg::I32(n as i32),
        ],
        grid: ((m as u32 + 15) / 16, (n as u32 + 15) / 16, 1),
        block: (16, 16, 1),
        shared_mem: 0,
        out_elems: m * n,
    }
}

/// lower:Rmsnorm([rows, cols];per-channel alpha([cols] 广播);w_off = ×(1+w))
pub fn lower_rmsnorm(ins: &[Bytes], eps: f32, w_off: bool, out: &Bytes, rows: usize, cols: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: spec("owl_rmsnorm_f32"),
        args: vec![
            Arg::Block { id: ins[0].id },
            Arg::Block { id: ins[1].id },
            Arg::Block { id: out.id },
            Arg::I32(cols as i32),
            Arg::F32(eps),
            Arg::I32(w_off as i32),
        ],
        // 一 block 一行(owl_rmsnorm_f32:row = blockIdx.x)
        grid: (rows as u32, 1, 1),
        block: (256, 1, 1),
        shared_mem: 256 * 4,
        out_elems: rows * cols,
    }
}

// ============================================================================
// §5 Kernel 节点(动作表二期:用户自定义 kernel 的唯一 lower 通道)
// ============================================================================

/// lower:Kernel 节点。
///
/// 槽序契约:`.arg(t)` 同时追加参数槽与父依赖(同序),故 T 槽按序
/// 对齐归约结果 ins;**输出块固定追加在槽序末尾**(server 按"最后一个
/// Block"回传句柄;kernel 形参必须把输出声明在末位)。
/// grid = (0,0,0) 哨兵 → 按输出元素数自动 1D ceil/256。
/// out_elems = 声明元素数(C1)。
///
/// **C2 签名校验**:kernel 若在注册表(`kernel::lookup`),decl_args 的
/// 类型序列 + 末尾输出块必须与登记的形参槽序(`Entry.args`)严格一致,
/// 违约 panic(组合期编程错)—— u32/sz 错位、输出不在末参这两类雷在
/// 这里机器拦截。非注册 kernel(逃生舱直带源码)跳过校验。
pub fn lower_kernel(
    kernel: &crate::kernel::Kernel,
    scalars: &[KernelArg],
    ins: &[Bytes],
    out: &Bytes,
    out_elems: usize,
) -> LaunchMsg {
    // 槽序唯一权威:注册表签名优先,逃生舱 kernel 自带 sig
    let sig: &str = match crate::kernel::lookup(kernel.name) {
        Some(e) => e.args,
        None if !kernel.sig.is_empty() => kernel.sig,
        None => panic!(
            "lower_kernel({}): 非注册 kernel 必须带 with_sig(槽序契约无权威即拒绝发射)",
            kernel.name
        ),
    };
    let toks: Vec<&str> = sig.split(',').collect();
    // 输出块 = 末位 T;其余 T 槽数必须等于父依赖数
    let t_in = toks[..toks.len() - 1].iter().filter(|t| **t == "T").count();
    assert!(
        t_in == ins.len(),
        "lower_kernel({}): 签名 T 槽 {t_in} != 父依赖 {}(检查 .arg 链)",
        kernel.name,
        ins.len()
    );
    assert!(
        toks.len() - 1 - t_in == scalars.len(),
        "lower_kernel({}): 签名标量槽 {} != 标量参数 {}(sz/i32/f32 与 arg_usize/arg_i32/arg_f32 对位)",
        kernel.name,
        toks.len() - 1 - t_in,
        scalars.len()
    );

    // 按 sig 对位装配:T → 下一个父块(父输出类型 = 张量块,查父即得);
    // sz/i32/f32 → 下一个标量;末位 T → 输出块
    let mut args: Vec<Arg> = Vec::with_capacity(toks.len());
    let (mut pi, mut si) = (0usize, 0usize);
    for (i, tok) in toks.iter().enumerate() {
        let is_out = i == toks.len() - 1;
        match *tok {
            "T" if is_out => args.push(Arg::Block { id: out.id }),
            "T" => {
                args.push(Arg::Block { id: ins[pi].id });
                pi += 1;
            }
            "sz" => match &scalars[si] {
                KernelArg::Bits(v) => args.push(Arg::U64(*v)),
                other => panic!("lower_kernel({}): 槽 {i} 期望 sz,实得 {other:?}", kernel.name),
            },
            "i32" => match &scalars[si] {
                KernelArg::I32(v) => args.push(Arg::I32(*v)),
                other => panic!("lower_kernel({}): 槽 {i} 期望 i32,实得 {other:?}", kernel.name),
            },
            "f32" => match &scalars[si] {
                KernelArg::F32(v) => args.push(Arg::F32(*v)),
                other => panic!("lower_kernel({}): 槽 {i} 期望 f32,实得 {other:?}", kernel.name),
            },
            other => panic!("lower_kernel({}): 非法签名 token `{other}`", kernel.name),
        }
        if !is_out && *tok != "T" {
            si += 1;
        }
    }
    let (grid, block, shared_mem) = if kernel.launch.grid == (0, 0, 0) {
        (auto_grid(out_elems), kernel.launch.block, kernel.launch.shared_mem)
    } else {
        (kernel.launch.grid, kernel.launch.block, kernel.launch.shared_mem)
    };
    LaunchMsg {
        kernel: KernelSpec { name: kernel.name.to_string(), source: kernel.source.to_string() },
        args,
        grid,
        block,
        shared_mem,
        out_elems,
    }
}

/// 自动 1D grid(哨兵展开用)
pub fn auto_grid(out_elems: usize) -> (u32, u32, u32) {
    ceil_1d(out_elems)
}
