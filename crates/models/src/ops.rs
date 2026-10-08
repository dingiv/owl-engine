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

use crate::contract::{Arg, Bytes, Dtype, KernelSource, LaunchMsg};
use crate::kernel::Kernel;
use crate::tensor::TensorOps;

pub use owl_kernels::driver::OpId;

/// 语义算子词表(model 层语汇;**值 = 语义名空间,非 kernel 实现名**。
/// 实现住 owl-kernels::driver 分派表,耦合由 models 侧 resolve 测试把门:
/// 词表加票、driver 加臂,两侧不同票 = 单测红)
pub mod ids {
    use super::OpId;
    pub const GDN_GATING: OpId = OpId("gdn.gating_g");
    pub const GDN_L2NORM: OpId = OpId("gdn.l2norm");
    pub const GDN_CONV_UPD: OpId = OpId("gdn.conv_upd");
    /// 刀3b:q/k 双段 conv 槽更新单发(aux 无;标量 dq/dk/batch/silu 入参)
    pub const GDN_CONV_UPD_DUAL: OpId = OpId("gdn.conv_upd_dual");
    pub const GDN_DELTA_DEC: OpId = OpId("gdn.delta_dec");
    /// D1:decode 整链融合(v-conv + l2norm×2 + gating + sigmoid + delta + norm_act)
    pub const GDN_DECODE_STEP: OpId = OpId("gdn.decode_step");
    /// D1-v2:delta 相 float4 行组重写(同契约;sglang 刺探产物)
    pub const GDN_DECODE_STEP_V2: OpId = OpId("gdn.decode_step_v2");
    pub const GDN_CONV_FWD: OpId = OpId("gdn.conv_fwd");
    pub const GDN_RECURRENCE: OpId = OpId("gdn.recurrence_varlen_gqa");
    pub const GDN_NORM_ACT: OpId = OpId("gdn.norm_act");
    /// foreign 家族臂(2026-10-12 用户律:gdn 两臂入词表,与 native 同一
    /// Call 通道;名字/发射配置归 driver,槽序 sig 归家族线契约单源)
    pub const GDN_CHUNKED: OpId = OpId("gdn.chunked_delta");
    pub const GDN_SCALAR: OpId = OpId("gdn.scalar_delta");
    pub const SIGMOID: OpId = OpId("ops.sigmoid");
    pub const ATTN_K0_WRITE: OpId = OpId("attn.k0_write");
    pub const ATTN_K0_WRITE_FP8: OpId = OpId("attn.k0_write_fp8");
    pub const ATTN_K0_WRITE_FP8_BF16: OpId = OpId("attn.k0_write_fp8_bf16");
    pub const ATTN_PAGED_DECODE_V2_FP8: OpId = OpId("attn.paged_decode_v2_fp8");
    pub const ATTN_PAGED_PREFILL_FP8: OpId = OpId("attn.paged_prefill_fp8");
    pub const ATTN_K0_DUAL: OpId = OpId("attn.k0_dual");
    pub const ATTN_K0_DUAL_FP8KV: OpId = OpId("attn.k0_dual_fp8kv");
    pub const CAST_F16_F32: OpId = OpId("elems.cast_f16_f32");
    pub const ATTN_PAGED_DECODE: OpId = OpId("attn.paged_decode");
    /// v2 分页 decode 在线 softmax 主核(分块 PARTITION=512;aux =
    /// [hd, hq, hkv, nb, nparts];输出 = 未归一化 partials [1,hq·nparts·hd])
    pub const ATTN_PAGED_DECODE_V2: OpId = OpId("attn.paged_decode_v2");
    /// v2 LSE 归并(exp_sums/max_logits/tmp_out → out;aux = [hd, hq, nparts])
    pub const ATTN_PAGED_V2_REDUCE: OpId = OpId("attn.paged_v2_reduce");
    pub const ATTN_PAGED_PREFILL: OpId = OpId("attn.paged_prefill");
    pub const ATTN_PREFILL_SPLIT: OpId = OpId("attn.prefill_split");
    pub const ATTN_PREFILL_SPLIT_REDUCE: OpId = OpId("attn.prefill_split_reduce");
    pub const ATTN_NAIVE_DECODE: OpId = OpId("attn.naive_decode");
    pub const ATTN_GATE_MUL: OpId = OpId("attn.gate_mul");
    /// 刀3a':双权 GEMV 单发(b/a 投影;aux = [rows_b, rows_a, cols, tokens])
    pub const ELEMS_GEMV_DUAL: OpId = OpId("elems.gemv_dual");
    pub const OPS_NARROW: OpId = OpId("ops.narrow");
    pub const OPS_CONCAT: OpId = OpId("ops.concat");
    pub const OPS_ROPE: OpId = OpId("ops.rope");
    pub const OPS_EMBED: OpId = OpId("ops.embed");
    pub const LOAD_CT_REPACK: OpId = OpId("load.ct_repack");
    pub const ATTN_NORM_ROPE: OpId = OpId("attn.norm_rope");
    pub const MLP_SILU_AND_MUL: OpId = OpId("mlp.silu_and_mul");
    pub const LN_FUSED_ADD_RMSNORM: OpId = OpId("ln.fused_add_rmsnorm");
    pub const ATTN_QKV_NORM_ROPE_INSERT: OpId = OpId("attn.qkv_norm_rope_insert");
    /// B6.3:fp8 e4m3 主池变体(decode 融合插池路;池写 1B e4m3)
    pub const ATTN_QKV_NORM_ROPE_INSERT_FP8KV: OpId = OpId("attn.qkv_norm_rope_insert_fp8kv");
}

/// contract::Dtype → driver::DType(契约类型不过 kernels,转换住消费侧)
pub(crate) fn ddt(dt: crate::contract::Dtype) -> Result<owl_kernels::driver::DType, crate::contract::ModelError> {
    use owl_kernels::driver::DType;
    Ok(match dt {
        crate::contract::Dtype::F16 => DType::F16,
        crate::contract::Dtype::BF16 => DType::BF16,
        crate::contract::Dtype::F32 => DType::F32,
        crate::contract::Dtype::U32 => DType::U32,
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
// §1.5 KernelCall —— 声明期槽序状态机(E1/E2 缺口闭环,2026-10-12)
// ============================================================================
//
// sig 来自登记表(单源;**零 sig 字面量**),逐 token 消费:
// - .t()  消费一个 T 槽(挂张量父;out 槽自动跳过)
// - .sz()/.i32()/.f32() 消费对应宽度标量槽(E1 宽度错位:方法即宽度)
// - build():全消费断言(缺项/溢出 = 声明期 panic,非 eval 期)
//
// 配对要求(E1/E2 的杜绝)从「人眼对 sig」变成「状态机不放行」。

/// 进行中的 native kernel 声明(槽序状态机)
pub struct KernelCall {
    node: crate::tensor::TensorOps,
    toks: Vec<&'static str>,
    pos: usize,
    out_pos: usize,
    name: &'static str,
}

fn kernel_call_launch(name: &'static str, launch: crate::kernel::Kernel) -> KernelCall {
    let e = crate::kernel::lookup(name).unwrap_or_else(|| panic!("kernel_call: {name} 未登记"));
    let toks: Vec<&'static str> = e.args.split(',').collect();
    // out 保留位:显式 O,或(无 O 时)末位 T
    let out_pos = toks
        .iter()
        .position(|t| *t == "O")
        .unwrap_or(toks.len().saturating_sub(1));
    KernelCall {
        node: crate::tensor::TensorOps::of(launch),
        toks,
        pos: 0,
        out_pos,
        name: e.name,
    }
}

/// 登记表 kernel 声明(自动 1D 网格)
pub fn kernel_call(name: &'static str) -> KernelCall {
    kernel_call_launch(name, crate::kernel::kernel(name))
}

/// 登记表 kernel 声明(显式网格;行核 embed/rope/attn 等)
pub fn kernel_call_with(
    name: &'static str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
) -> KernelCall {
    kernel_call_launch(name, crate::kernel::kernel_with(name, grid, block, shared_mem))
}

impl KernelCall {
    fn consume(&mut self, kind: &str) {
        if self.pos == self.out_pos {
            self.pos += 1; // out 槽由 eval 追加,自动跳过
        }
        let tok = self
            .toks
            .get(self.pos)
            .unwrap_or_else(|| panic!("{}: 槽序溢出(已消费 {} 个,期望 {kind})", self.name, self.pos));
        assert!(
            *tok == kind,
            "{}: 第 {} 槽期望 {kind} 实得 {tok}(E1 宽度/类型错位,声明期拦截)",
            self.name,
            self.pos + 1
        );
        self.pos += 1;
    }

    /// T 槽(张量父;设备指针)
    pub fn t(mut self, t: &crate::tensor::TensorOps) -> Self {
        self.consume("T");
        self.node = self.node.arg(t);
        self
    }

    /// sz 标量(8B;size_t 形参)
    pub fn sz(mut self, v: usize) -> Self {
        self.consume("sz");
        self.node = self.node.arg_usize(v);
        self
    }

    /// i32 标量(4B)
    pub fn i32(mut self, v: i32) -> Self {
        self.consume("i32");
        self.node = self.node.arg_i32(v);
        self
    }

    /// f32 标量(4B)
    pub fn f32(mut self, v: f32) -> Self {
        self.consume("f32");
        self.node = self.node.arg_f32(v);
        self
    }

    /// 全消费断言(out 槽由 eval 追加,不计)+ 产出节点
    pub fn build(self) -> crate::tensor::TensorOps {
        let mut pos = self.pos;
        if pos == self.out_pos {
            pos += 1;
        }
        assert!(
            pos == self.toks.len(),
            "{}: 声明未消费完(pos {}/{};E2 断链类,声明期拦截)",
            self.name,
            pos,
            self.toks.len()
        );
        self.node
    }
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
    /// 胖算子(kernels 侧 struct;validate/wire 由算子自带,interpreter 强制)
    Spec { spec: std::sync::Arc<dyn owl_kernels::contract::OpSpec> },

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

fn spec(name: &'static str) -> KernelSource {
    KernelSource { name: name.to_string(), source: crate::kernel::source(name).to_string() }
}

/// dtype 路由名(f16 基线 F2):`owl_<op>_<f32|f16>`;注册表缺条目 =
/// source panic(编程错误口径 —— 守门已在 eval 前置,此处对齐注册表)
fn dname(base: &str, dtype: crate::contract::Dtype) -> &'static str {
    match dtype {
        crate::contract::Dtype::F32 => to_static(concat_op(base, "f32")),
        crate::contract::Dtype::F16 => to_static(concat_op(base, "f16")),
        // bf16 臂(E5-DF3 同日十四;owl_add_bf16/owl_mul_bf16/owl_rmsnorm_bf16)
        crate::contract::Dtype::BF16 => to_static(concat_op(base, "bf16")),
        other => panic!("dname: 语义算子不支持 {other:?}(f16 基线:仅 F32/F16/BF16)"),
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
    bf16: bool,
) -> LaunchMsg {
    LaunchMsg {
        // 核名 = 线契约常量(唯一字面量住址 owl_kernels::contract::names;
        // models 零 cublas feature 也经此引用 —— 字面量全库只写一次)
        kernel: KernelSource {
            name: if bf16 {
                owl_kernels::contract::names::GEMM_BF16.to_string()
            } else {
                owl_kernels::contract::names::GEMM_F16.to_string()
            },
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
        kernel: KernelSource { name: kernel.name.to_string(), source: kernel.source.to_string() },
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


// ============================================================================
// §6 层↔kernels 隔离面(2026-10-12 用户律:**layers 禁触 owl_kernels**;
// face builder/名字/槽序签名/OK 谓词知识全部收编本文件 —— 层只供语义
// 张量与纯标量,装配细节 = 解释层动作表职责)
// ============================================================================

use owl_kernels::client::gdn_chunked as fc_chunked;
use owl_kernels::client::gdn_scalar as fc_scalar;

/// foreign 名字(层侧唯一出口;字面量住址 = owl_kernels::contract::names)
pub use owl_kernels::driver::load::CT_REPACK;

/// foreign 家族臂线格式(名字 → 槽序 sig;解释器 Call 臂消费 —— native
/// 登记表无 foreign 条目,sig 住 client face 单源。gdn 两臂 Call 通道的
/// 桥:driver 拾取名 → 本表 → lower_kernel 对位装配)
pub(crate) fn foreign_sig(name: &str) -> Option<&'static str> {
    if name == owl_kernels::family::gdn_chunked::GDN_CHUNKED_FWD {
        Some(owl_kernels::client::gdn_chunked::SIG)
    } else if name == owl_kernels::family::gdn_scalar::GDN_SCALAR_FWD {
        Some(owl_kernels::client::gdn_scalar::SIG)
    } else {
        None
    }
}

/// marlin W4A16 臂装配(AWQ 7 块 / f16·bf16 6 块;签名/名字 face 单源)
#[allow(clippy::too_many_arguments)]
pub(crate) fn marlin_node(
    xs: &TensorOps,
    qw: &TensorOps,
    sc: &TensorOps,
    ws: &TensorOps,
    ctmp: &TensorOps,
    zs: Option<&TensorOps>,
    m: usize,
    in_dim: usize,
    n_pack: usize,
    group: usize,
    bf16_act: bool,
) -> TensorOps {
    use owl_kernels::family::marlin;
    let (name, sig): (&'static str, &'static str) = match zs {
        Some(_) => (marlin::GEMM_W4A16_AWQ, owl_kernels::client::marlin::SIG_AWQ),
        None if bf16_act => (marlin::GEMM_W4A16_BF16, owl_kernels::client::marlin::SIG),
        None => (marlin::GEMM_W4A16, owl_kernels::client::marlin::SIG),
    };
    let node = TensorOps::of(Kernel::new(name, "").with_sig(sig))
        .arg(xs)
        .arg(qw)
        .arg(sc);
    let node = match zs {
        Some(z) => node.arg(z),
        None => node,
    };
    node.arg(ws)
        .arg(ctmp)
        .arg_usize(m)
        .arg_usize(in_dim)
        .arg_usize(n_pack)
        .arg_usize(group)
        .with_shape(if bf16_act { Dtype::BF16 } else { Dtype::F16 }, vec![m, n_pack])
}

/// marlin workspace 长度(i32 个数;装载域 Want 计尺,层零 kernels 知识)
pub(crate) fn marlin_ws_elems(n: usize) -> usize {
    owl_kernels::family::marlin::v2_workspace_len(n)
}

/// FI paged prefill 臂装配(9 Block + 8 sz;签名/scale 编码 face 单源。
/// name 由调用方给 = env.fi_prefill_name() 收口)
#[allow(clippy::too_many_arguments)]
pub(crate) fn fi_prefill_node(
    name: &'static str,
    q: &TensorOps,
    kcs: &TensorOps,
    vcs: &TensorOps,
    q_cu: &TensorOps,
    indices: &TensorOps,
    indptr: &TensorOps,
    last_len: &TensorOps,
    wr: &TensorOps,
    total_rows: usize,
    ctx_total: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    page: usize,
) -> TensorOps {
    TensorOps::of(Kernel::new(name, "").with_sig(owl_kernels::client::fi_sig::SIG))
        .arg(q)
        .arg(kcs)
        .arg(vcs)
        .arg(q_cu)
        .arg(indices)
        .arg(indptr)
        .arg(last_len)
        .arg(wr) // 树序依赖边(K0 先于分块读池;FI 不解引用)
        .arg_usize(total_rows)
        .arg_usize(ctx_total)
        .arg_usize(total_rows)
        .arg_usize(hq)
        .arg_usize(hkv)
        .arg_usize(hd)
        .arg_usize(page)
        .arg_usize(owl_kernels::client::fi_sig::scale_bits(hd) as usize)
        .with_shape(q.dtype, vec![total_rows, hq * hd])
}

/// FI 虚核名选择(models 侧唯一出口;测试/层共用)
pub(crate) fn fi_name(fp8kv: bool) -> &'static str {
    if fp8kv {
        owl_kernels::contract::names::PREFILL_FI_FP8KV
    } else {
        owl_kernels::contract::names::PREFILL_FI
    }
}

/// 页配对谓词(decode;driver 单源的层侧出口,层零 driver 知识)
pub(crate) fn paged_decode_ok(hd: usize, page: usize) -> bool {
    owl_kernels::driver::attn::paged_decode_ok(hd, page)
}

/// 页配对谓词(prefill;bs32 契约)
pub(crate) fn paged_prefill_ok(hd: usize, page: usize) -> bool {
    owl_kernels::driver::attn::paged_prefill_ok(hd, page)
}

#[cfg(test)]
mod foreign_call_lock {
    //! gdn foreign 两臂 Call 通道锁(2026-10-12 用户律二番:具名算子入
    //! 词表):ids ↔ driver 臂 ↔ foreign sig ↔ server parse 四点同票 ——
    //! 层声明 → lower_kernel → client face parse 逐位对拍。
    use super::*;
    use owl_kernels::driver::{OpEnv, OpReq};

    fn env() -> OpEnv {
        OpEnv { hw: owl_kernels::driver::Hw { arch: owl_kernels::Arch::Sm86 }, page: 0 }
    }

    /// 层声明形态(chunked;与 layers/gdn.rs 逐字同构)
    fn declare_chunked(
        q: &TensorOps, k: &TensorOps, v: &TensorOps, beta: &TensorOps,
        gate: &TensorOps, state: &TensorOps, slot: usize,
        t: usize, nv: usize, nk: usize, kd: usize,
    ) -> TensorOps {
        TensorOps::call(ids::GDN_CHUNKED)
            .arg(q).arg(k).arg(v).arg(beta).arg(gate).arg(state)
            .arg_usize(t).arg_usize(slot).arg_usize(nv).arg_usize(nk).arg_usize(kd)
            .arg_bits((1.0f32 / (kd as f32).sqrt()).to_bits() as u64)
            .with_shape(Dtype::F16, vec![t, nv, kd])
    }

    #[test]
    fn ids_resolve_to_foreign_families() {
        // 词表 ↔ driver 分派表同票(加票不加臂 = 此处红)
        for (op, want) in [
            (ids::GDN_CHUNKED, owl_kernels::family::gdn_chunked::GDN_CHUNKED_FWD),
            (ids::GDN_SCALAR, owl_kernels::family::gdn_scalar::GDN_SCALAR_FWD),
        ] {
            let pick = owl_kernels::driver::resolve(OpReq {
                op, env: &env(), dt: owl_kernels::driver::DType::F16,
                shapes: &[], aux: &[], scalars: &[],
            });
            assert_eq!(pick.name, want);
            assert_eq!(
                (pick.shape.grid, pick.shape.block, pick.shape.smem),
                ((0, 0, 0), (0, 0, 0), 0),
                "foreign 臂发射配置应直通"
            );
            assert!(foreign_sig(pick.name).is_some(), "driver 拾取名必须有家族 sig");
        }
    }

    #[test]
    fn chunked_call_wire_survives_server_parse() {
        let q = TensorOps::of_block(1, Dtype::F16, vec![64, 16, 128]);
        let k = TensorOps::of_block(2, Dtype::F16, vec![64, 16, 128]);
        let v = TensorOps::of_block(3, Dtype::F16, vec![64, 48, 128]);
        let beta = TensorOps::of_block(4, Dtype::F16, vec![64, 48]);
        let gate = TensorOps::of_block(5, Dtype::F16, vec![64, 48]);
        let state = TensorOps::of_block(6, Dtype::F32, vec![8, 48, 128, 128]);
        let node = declare_chunked(&q, &k, &v, &beta, &gate, &state, 3, 64, 48, 16, 128);

        // 解释器 Call 臂同款:pick → foreign sig kernel → lower_kernel
        let pick = owl_kernels::driver::resolve(OpReq {
            op: ids::GDN_CHUNKED, env: &env(), dt: owl_kernels::driver::DType::F16,
            shapes: &[], aux: &[], scalars: &[],
        });
        let kernel = crate::kernel::Kernel::new(pick.name, "")
            .with_sig(foreign_sig(pick.name).expect("foreign sig"));
        let out = Bytes::new(99, 64 * 48 * 128);
        let ins: Vec<Arg> = [&q, &k, &v, &beta, &gate, &state]
            .iter().map(|t| Arg::Block { id: t.id }).collect();
        let msg = lower_kernel(&kernel, &node.args, &ins, &out, 64 * 48 * 128);

        // server 面 parse 逐位对拍(线格式四点锁的最后一环)
        let (call, out2) = fc_chunked::GdnChunkedCall::parse(&msg).expect("server parse");
        assert_eq!(out2.id, 99);
        // 块句柄 = 叶子节点 id(全局计数器;与套件其他测试共存,勿硬编码)
        assert_eq!(call.q().id, q.id);
        assert_eq!(call.k().id, k.id);
        assert_eq!(call.v().id, v.id);
        assert_eq!(call.beta().id, beta.id);
        assert_eq!(call.gate().0.id, gate.id);
        assert_eq!(call.state().pool.id, state.id);
        assert_eq!(call.state().slot, 3);
        assert_eq!(call.shape().t, 64);
        assert_eq!(call.shape().nv, 48);
        assert_eq!(call.shape().nk, 16);
        assert_eq!(call.shape().kd, 128);
        // 槽序数:7 Block + 6 sz(lower_kernel 对位 sig 全额)
        let blocks = msg.args.iter().filter(|a| matches!(a, Arg::Block { .. })).count();
        assert_eq!(blocks, 7);
        assert_eq!(msg.args.len(), 13);
    }

    #[test]
    fn scalar_call_wire_survives_server_parse() {
        let q = TensorOps::of_block(1, Dtype::F32, vec![32, 4, 64]);
        let k = TensorOps::of_block(2, Dtype::F32, vec![32, 4, 64]);
        let v = TensorOps::of_block(3, Dtype::F32, vec![32, 8, 64]);
        let g = TensorOps::of_block(4, Dtype::F32, vec![32, 8]);
        let beta = TensorOps::of_block(5, Dtype::F32, vec![32, 8]);
        let state = TensorOps::of_block(6, Dtype::F32, vec![2, 8, 64, 64]);
        let node = TensorOps::call(ids::GDN_SCALAR)
            .arg(&q).arg(&k).arg(&v).arg(&g).arg(&beta).arg(&state)
            .arg_usize(32).arg_usize(1).arg_usize(1)
            .arg_usize(8).arg_usize(4).arg_usize(64)
            .arg_bits((1.0f32 / (64f32).sqrt()).to_bits() as u64)
            .with_shape(Dtype::F16, vec![32, 8, 64]);

        let pick = owl_kernels::driver::resolve(OpReq {
            op: ids::GDN_SCALAR, env: &env(), dt: owl_kernels::driver::DType::F32,
            shapes: &[], aux: &[], scalars: &[],
        });
        let kernel = crate::kernel::Kernel::new(pick.name, "")
            .with_sig(foreign_sig(pick.name).expect("foreign sig"));
        let out = Bytes::new(99, 32 * 8 * 64);
        let ins: Vec<Arg> = [&q, &k, &v, &g, &beta, &state]
            .iter().map(|t| Arg::Block { id: t.id }).collect();
        let msg = lower_kernel(&kernel, &node.args, &ins, &out, 32 * 8 * 64);

        let (call, out2) = fc_scalar::GdnScalarCall::parse(&msg).expect("server parse");
        assert_eq!(out2.id, 99);
        assert_eq!(call.q().id, q.id);
        assert_eq!(call.state().id, state.id);
        assert_eq!(call.slot(), 1);
        assert_eq!(call.shape().t, 32);
        assert_eq!(call.shape().ns, 1);
        assert_eq!(call.shape().nv, 8);
        assert_eq!(call.shape().nk, 4);
        assert_eq!(call.shape().kd, 64);
    }
}

#[cfg(test)]
mod kernel_call_lock {
    //! 声明期状态机锁(E1 宽度错位 / E2 断链)+ 与手摆链同构互证。

    use super::*;
    use crate::tensor::TensorOps;

    #[test]
    fn wire_matches_legacy_chain() {
        // owl_narrow_strided_f16 sig = "T,sz,sz,sz,sz,T"
        let q = TensorOps::of_block(1, crate::contract::Dtype::F16, vec![64]);
        let kc = TensorOps::of_block(2, crate::contract::Dtype::F16, vec![64]);

        // out 槽(末位 T)由 eval 追加,调用链不含
        let new_node = kernel_call("owl_narrow_strided_f16")
            .t(&q)
            .sz(8)
            .sz(4)
            .sz(32)
            .sz(2)
            .build()
            .with_shape(crate::contract::Dtype::F16, vec![64]);

        let legacy = TensorOps::of(crate::kernel::Kernel::new("owl_narrow_strided_f16", ""))
            .arg(&q)
            .arg_usize(8)
            .arg_usize(4)
            .arg_usize(32)
            .arg_usize(2)
            .with_shape(crate::contract::Dtype::F16, vec![64]);

        assert_eq!(new_node.parents.len(), legacy.parents.len(), "T 槽数漂移");
        for (a, b) in new_node.parents.iter().zip(&legacy.parents) {
            assert_eq!(a.id, b.id, "T 槽序漂移");
        }
        assert_eq!(new_node.args.len(), legacy.args.len(), "标量槽数漂移");
        for (a, b) in new_node.args.iter().zip(&legacy.args) {
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "标量漂移");
        }
    }

    #[test]
    #[should_panic(expected = "第 2 槽期望 i32 实得 sz(E1")]
    fn width_mismatch_panics_at_declaration() {
        // E1:sz 槽用 i32 顶(4B 顶 8B,参数空间错位类)—— 声明期拦截
        let q = TensorOps::of_block(1, crate::contract::Dtype::F16, vec![64]);
        let _ = kernel_call("owl_narrow_strided_f16").t(&q).i32(7);
    }

    #[test]
    #[should_panic(expected = "声明未消费完")]
    fn missing_slot_panics_at_build() {
        // E2 断链类:少喂一个 sz 槽,build() 全消费断言拦
        let q = TensorOps::of_block(1, crate::contract::Dtype::F16, vec![64]);
        let _ = kernel_call("owl_narrow_strided_f16")
            .t(&q)
            .sz(8)
            .sz(4)
            .sz(32)
            .build();
    }
}
