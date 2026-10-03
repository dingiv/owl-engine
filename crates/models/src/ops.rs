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

pub use owl_kernels::driver::OpId;

/// 语义算子词表(model 层语汇;**值 = 语义名空间,非 kernel 实现名**。
/// 实现住 owl-kernels::driver 分派表,耦合由 models 侧 resolve 测试把门:
/// 词表加票、driver 加臂,两侧不同票 = 单测红)
pub mod ids {
    use super::OpId;
    pub const GDN_GATING: OpId = OpId("gdn.gating_g");
    pub const GDN_L2NORM: OpId = OpId("gdn.l2norm");
    pub const GDN_CONV_UPD: OpId = OpId("gdn.conv_upd");
    pub const GDN_DELTA_DEC: OpId = OpId("gdn.delta_dec");
    /// D1:decode 整链融合(v-conv + l2norm×2 + gating + sigmoid + delta + norm_act)
    pub const GDN_DECODE_STEP: OpId = OpId("gdn.decode_step");
    pub const GDN_CONV_FWD: OpId = OpId("gdn.conv_fwd");
    pub const GDN_RECURRENCE: OpId = OpId("gdn.recurrence_varlen_gqa");
    pub const GDN_NORM_ACT: OpId = OpId("gdn.norm_act");
    pub const SIGMOID: OpId = OpId("ops.sigmoid");
    pub const ATTN_K0_WRITE: OpId = OpId("attn.k0_write");
    pub const ATTN_K0_DUAL: OpId = OpId("attn.k0_dual");
    pub const ATTN_K0_DUAL_FP8KV: OpId = OpId("attn.k0_dual_fp8kv");
    pub const CAST_F16_F32: OpId = OpId("elems.cast_f16_f32");
    pub const ATTN_PAGED_DECODE: OpId = OpId("attn.paged_decode");
    pub const ATTN_PAGED_PREFILL: OpId = OpId("attn.paged_prefill");
    pub const ATTN_PREFILL_SPLIT: OpId = OpId("attn.prefill_split");
    pub const ATTN_PREFILL_SPLIT_REDUCE: OpId = OpId("attn.prefill_split_reduce");
    pub const ATTN_NAIVE_DECODE: OpId = OpId("attn.naive_decode");
    pub const ATTN_GATE_MUL: OpId = OpId("attn.gate_mul");
    pub const OPS_NARROW: OpId = OpId("ops.narrow");
    pub const OPS_CONCAT: OpId = OpId("ops.concat");
    pub const OPS_ROPE: OpId = OpId("ops.rope");
    pub const OPS_EMBED: OpId = OpId("ops.embed");
    pub const LOAD_CT_REPACK: OpId = OpId("load.ct_repack");
    pub const ATTN_NORM_ROPE: OpId = OpId("attn.norm_rope");
    pub const MLP_SILU_AND_MUL: OpId = OpId("mlp.silu_and_mul");
    pub const LN_FUSED_ADD_RMSNORM: OpId = OpId("ln.fused_add_rmsnorm");
    pub const ATTN_QKV_NORM_ROPE_INSERT: OpId = OpId("attn.qkv_norm_rope_insert");
}

/// contract::Dtype → driver::DType(契约类型不过 kernels,转换住消费侧)
pub(crate) fn ddt(dt: crate::contract::Dtype) -> Result<owl_kernels::driver::DType, crate::contract::ModelError> {
    use owl_kernels::driver::DType;
    Ok(match dt {
        crate::contract::Dtype::F16 => DType::F16,
        crate::contract::Dtype::F32 => DType::F32,
        crate::contract::Dtype::U32 => DType::U32,
        other => {
            return Err(crate::contract::ModelError::Msg(format!(
                "driver::resolve: dtype {other:?} 无拾取臂(词表族仅 f16/f32)"
            )))
        }
    })
}

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
    /// [m,k] × (B 以 [n,k] 行主序直读)→ [m,n](nt = B 非转置存储;
    /// lm_head/tied 形态:权重保持 checkpoint 原布局,免 host 转置)
    MatmulNt,
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
    /// 刀1:块内连续切片视图(零内核零节点派发;eval 透传父块,
    /// 消费面发射 `Arg::BlockSlice{id, byte_offset: offset·esz}`)。
    /// 节点偏移在声明期已含父链合成(slice_of_slice);仅连续切片
    /// (narrow 的 outer==1 ∨ src_dim==out_dim)入此臂,非连续仍物化。
    SliceView { offset_elems: usize },

    // ---- 语义调用(Driver 立项,2026-10-01):model 层只描述「要什么」
    //      (OpId 语义词表)+ 张量 + 语义标量;名/变体/发射参数 = 解释器
    //      执行期经 owl-kernels::driver::resolve(OpEnv 必传,被动律)拾取。
    //      aux = 拾取推导常数(层语义几何:kd/vd/batch…;非核参数,不进
    //      签名)。**model 层由此不再感知 kernel 层的存在**(S2' 定稿)。
    Call { op: OpId, aux: Vec<usize> },

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

/// dtype 路由名(f16 基线 F2):`owl_<op>_<f32|f16>`;注册表缺条目 =
/// source panic(编程错误口径 —— 守门已在 eval 前置,此处对齐注册表)
fn dname(base: &str, dtype: crate::contract::Dtype) -> &'static str {
    match dtype {
        crate::contract::Dtype::F32 => to_static(concat_op(base, "f32")),
        crate::contract::Dtype::F16 => to_static(concat_op(base, "f16")),
        other => panic!("dname: 语义算子不支持 {other:?}(f16 基线:仅 F32/F16)"),
    }
}

fn concat_op(base: &str, suffix: &str) -> String {
    format!("{base}_{suffix}")
}

fn to_static(s: String) -> &'static str {
    // 路由名来自封闭集合(f32/f16 后缀),注册表键为 'static;
    // 用 Box::leak 承载(进程生命周期,量级 = 语义算子数 × 2,可忽略)
    Box::leak(s.into_boxed_str())
}

fn ceil_1d(n: usize) -> (u32, u32, u32) {
    ((n as u32 + 255) / 256, 1, 1)
}

/// lower:Add(同形二元;a/b 块等长由 server 账长校验兜底)
/// lower:Add(同形二元;a/b 块等长由 server 账长校验兜底)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)

/// f16 GEMM(foreign-kernel 通道,2026-09-26):核名 = cublas 虚拟核,
/// server 按名分派到 owl-kernels::cublas(非 nvrtc 注册表;source 空)。
/// 槽序契约:[T a, T b, T out, sz m, sz k, sz n, sz nt](cublas.rs 文档)。
pub fn lower_gemm(
    ins: &[Arg],
    out: &Bytes,
    m: usize,
    k: usize,
    n: usize,
    nt: bool,
) -> LaunchMsg {
    LaunchMsg {
        // 核名 = 线契约常量(权威定义 owl-kernels::cublas::GEMM_F16;
        // models 不开 cublas feature,此处字面量对齐,测试互证)
        kernel: KernelSpec {
            name: "cublas_gemm_f16".to_string(),
            source: String::new(),
        },
        args: vec![
            ins[0].clone(),
            ins[1].clone(),
            Arg::Block { id: out.id },
            Arg::U64(m as u64),
            Arg::U64(k as u64),
            Arg::U64(n as u64),
            Arg::U64(nt as u64),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: m * n,
    }
}

pub fn lower_add(ins: &[Arg], out: &Bytes, n: usize, dtype: crate::contract::Dtype) -> LaunchMsg {
    LaunchMsg {
        kernel: spec(dname("owl_add", dtype)),
        args: vec![
            ins[0].clone(),
            ins[1].clone(),
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
pub fn lower_mul(ins: &[Arg], out: &Bytes, n: usize, dtype: crate::contract::Dtype) -> LaunchMsg {
    LaunchMsg {
        kernel: spec(dname("owl_mul", dtype)),
        args: vec![ins[0].clone(), ins[1].clone(), Arg::Block { id: out.id }, Arg::U64(n as u64)],
        grid: ceil_1d(n),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// lower:Silu(一元)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)
pub fn lower_silu(ins: &[Arg], out: &Bytes, n: usize, dtype: crate::contract::Dtype) -> LaunchMsg {
    LaunchMsg {
        kernel: spec(dname("owl_silu", dtype)),
        args: vec![ins[0].clone(), Arg::Block { id: out.id }, Arg::U64(n as u64)],
        grid: ceil_1d(n),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// lower:Sigmoid(一元;attn_output_gate 门 / GDN beta 同族)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)
pub fn lower_sigmoid(ins: &[Arg], out: &Bytes, n: usize, dtype: crate::contract::Dtype) -> LaunchMsg {
    LaunchMsg {
        kernel: spec(dname("owl_sigmoid", dtype)),
        args: vec![ins[0].clone(), Arg::Block { id: out.id }, Arg::U64(n as u64)],
        grid: ceil_1d(n),
        block: (256, 1, 1),
        shared_mem: 0,
        out_elems: n,
    }
}

/// lower:Matmul([m,k]×[k,n];m/k/n 全部来自声明 shape(C1))
pub fn lower_matmul(ins: &[Arg], out: &Bytes, m: usize, k: usize, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: spec("owl_matmul_f32"),
        args: vec![
            ins[0].clone(),
            ins[1].clone(),
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

/// nt 变体:内核按 B [n,k] 直读(见 owl_matmul_nt_f32);网格同款
pub fn lower_matmul_nt(ins: &[Arg], out: &Bytes, m: usize, k: usize, n: usize) -> LaunchMsg {
    LaunchMsg {
        kernel: spec("owl_matmul_nt_f32"),
        args: vec![
            ins[0].clone(),
            ins[1].clone(),
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
pub fn lower_rmsnorm(ins: &[Arg], eps: f32, w_off: bool, out: &Bytes, rows: usize, cols: usize, dtype: crate::contract::Dtype) -> LaunchMsg {
    LaunchMsg {
        kernel: spec(dname("owl_rmsnorm", dtype)),
        args: vec![
            ins[0].clone(),
            ins[1].clone(),
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
    ins: &[Arg],
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
    // 输出槽位:E3 起 sig 可含 "O"(输出块所在槽,marlin foreign 等
    // out 非末参的核);无 O = 末位 T(向后兼容)
    let out_pos = toks.iter().position(|t| *t == "O");
    let is_out = |i: usize| match out_pos {
        Some(p) => i == p,
        None => i + 1 == toks.len(),
    };
    // 其余 T 槽数必须等于父依赖数
    let t_in = toks
        .iter()
        .enumerate()
        .filter(|(i, t)| **t == "T" && !is_out(*i))
        .count();
    assert!(
        t_in == ins.len(),
        "lower_kernel({}): 签名 T 槽 {t_in} != 父依赖 {}(检查 .arg 链)",
        kernel.name,
        ins.len()
    );
    assert!(
        toks.len() - t_in - 1 == scalars.len(), // -1 = 输出槽(末位 T 或 O)
        "lower_kernel({}): 签名标量槽 {} != 标量参数 {}(sz/i32/f32 与 arg_usize/arg_i32/arg_f32 对位)",
        kernel.name,
        toks.len() - t_in - 1,
        scalars.len()
    );

    // 按 sig 对位装配:T → 下一个父块(父输出类型 = 张量块,查父即得);
    // sz/i32/f32 → 下一个标量;末位 T → 输出块
    let mut args: Vec<Arg> = Vec::with_capacity(toks.len());
    let (mut pi, mut si) = (0usize, 0usize);
    for (i, tok) in toks.iter().enumerate() {
        let is_out = is_out(i);
        match *tok {
            "T" if is_out => args.push(Arg::Block { id: out.id }),
            "O" => args.push(Arg::Block { id: out.id }),
            "T" => {
                args.push(ins[pi].clone());
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

// ============================================================================
// 采样(E3;REQ-DEC-04 每步零大 D2H)
// ============================================================================

/// 设备侧贪心采样声明:logits 树的 argmax 索引,输出 [1] f32(索引数值
/// 过线,owl 契约 5;词表 id < 2²⁴ 精确)。平局取最小索引(与 host
/// argmax fold 语义一致)。核 = owl_argmax_f32idx_f16(单块两段规约;
/// cu/owl/argmax_f16.cu 头注)。`offset` = 末行起点(prefill 末块
/// [T,V] 传 (T-1)·V;decode T=1 传 0)。
pub fn argmax_f32idx(x: &crate::tensor::TensorOps, n: usize, offset: usize) -> crate::tensor::TensorOps {
    use crate::contract::Dtype;
    crate::tensor::TensorOps::of(crate::kernel::kernel_with(
        "owl_argmax_f32idx_f16",
        (1, 1, 1),
        (256, 1, 1),
        0,
    ))
    .arg(x)
    .arg_i32(n as i32)
    .arg_i32(offset as i32)
    .with_shape(Dtype::F32, vec![1])
}
