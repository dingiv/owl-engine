//! 声明算子面:语义 Op 枚举(声明什么)+ lower 动作表(怎么翻译)。
//!
//! 两半一个主题(2026-09-26 重组:原 plan.rs + actions.rs 合并):
//! - **Op**:client 对 server 能力期望的清单;server 照此实现 dispatch,
//!   新增能力 = 新变体 + server 一臂 + lower 一函数;
//! - **lower_***:具名算子 → LaunchMsg 的唯一翻译通道(声明语义落线格式)。
//!
//! kernel 源**不住这里** —— 经 [`crate::kernel`] 注册表按名取用(源码之家
//! = owl-kernels cu/;2026-09-25 垫子层裁决)。用户自定义 kernel 走
//! 非注册内核 = 加源 + 登记表一条 + 语义变体 + driver 臂(逃生舱收编)。
//! (2026-10-12 用户律:Kernel/Spec 节点消灭,唯一算子节点 = Call。)

use crate::contract::{Arg, Bytes, Dtype, KernelSource, LaunchMsg};
use crate::kernel::Kernel;
use crate::tensor::TensorOps;

use owl_kernels::driver::OpId;

/// 语义算子动作词表(2026-10-12 用户律:层面向**抽象动作**,不面向实现
/// —— 层只描述「对张量做什么」,硬件归解释器/驱动关注。枚举变体 = 层侧
/// 唯一算子词汇;CUDA 域经 [`SemanticKernel::op_id`] 落 driver 分派表,CPU/
/// AMD 解释器直接 match 本枚举落自有实现,**零 NVIDIA 词汇入层**)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticKernel {
    // ---- gdn.*(aux 序 = driver 族函数文档)----
    /// g = -exp(A_log)·softplus(a + dt_bias) 门控
    GdnGatingG,
    /// 末维 L2 归一
    GdnL2Norm,
    /// conv 状态滑更新
    GdnConvUpd,
    /// 刀3b:q/k 双段 conv 槽更新单发(标量 dq/dk/batch/silu 入参)
    GdnConvUpdDual,
    /// decode 单步 delta 递推
    GdnDeltaDec,
    /// D1:decode 整链融合(v-conv + l2norm×2 + gating + sigmoid + delta + norm_act)
    GdnDecodeStep,
    /// D1-v2:delta 相 float4 行组重写(同契约;sglang 刺探产物)
    GdnDecodeStepV2,
    /// conv 前向(记录臂)
    GdnConvFwd,
    /// varlen GQA 递推批核(aux = [nv, kd, vd])
    GdnRecurrence,
    /// 门控归一化 + T 批量(aux = [rows, value_dim, group_size])
    GdnNormAct,
    /// foreign 家族臂:chunked 六核编排(FLA fork-bf16;slot 寻址状态池。
    /// 名字/发射配置归 driver,槽序 sig 归家族线契约单源)
    GdnChunkedDelta,
    /// foreign 家族臂:scalar 单核(lmdeploy pre_sm90 port)
    GdnScalarDelta,

    // ---- ops.* ----
    /// 逐元素 sigmoid(attn_output_gate 门 / GDN beta 同族)
    Sigmoid,
    /// 同形逐元素加(残差)
    Add,
    /// 同形逐元素乘(门控)
    Mul,
    /// 一元激活(独立 silu 核;MLP 走 SiluAndMul 融合)
    Silu,
    /// ×(1+w) rmsnorm(行归一,cols 由 alpha 定义;标量槽 = cols/eps/w_off)
    Rmsnorm,
    /// 窄切物化拷贝(outer/src_dim/start/out_dim)
    Narrow,
    /// 行拼接
    Concat,
    /// partial rope
    Rope,
    /// embedding 查表
    Embed,

    // ---- elems.* ----
    /// f16 → f32 铸(GDN scalar/chunked 臂层侧预铸)
    CastF16F32,
    /// 刀3a':双权 GEMV 单发(b/a 投影;aux = [rows_b, rows_a, cols, tokens])
    GemvDual,

    // ---- attn.*(aux 序见各变体)----
    /// KV 散写(classic 布局;aux = [tokens])
    K0Write,
    /// KV 散写 e4m3 池变体(aux = [tokens])
    K0WriteFp8,
    /// 同上 bf16 入(K0)
    K0WriteFp8Bf16,
    K0WriteKnhd,
    K0WriteFp8Knhd,
    /// K0 双写(classic + kNHD 影子;aux = [tokens])
    K0Dual,
    /// K0 双写 e4m3 影子
    K0DualFp8kv,
    /// 分页 decode v1(aux = [hd, hq, hkv, nb])
    PagedDecode,
    /// v2 分页 decode 在线 softmax 主核(分块 PARTITION=512;aux =
    /// [hd, hq, hkv, nb, nparts];输出 = 未归一化 partials [1,hq·nparts·hd])
    PagedDecodeV2,
    /// B6.1:v2 fp8 e4m3 KV 读变体(形状契约同 f16)
    PagedDecodeV2Fp8,
    /// §三十一:verify 8 伪序列变体(grid.y = seqs;aux 末位 = seqs)
    PagedDecodeV2Fp8Seq,
    PagedDecodeV2Seq,
    PagedDecodeV2Knhd,
    PagedDecodeV2Fp8Knhd,
    PagedDecodeV2SeqKnhd,
    PagedDecodeV2Fp8SeqKnhd,
    /// v2 LSE 归并(exp_sums/max_logits/tmp_out → out;aux = [hd, hq, nparts])
    PagedV2Reduce,
    /// chunked prefill 在线 softmax(aux = [hd, hkv, hq, tokens])
    PagedPrefill,
    /// B6.3:chunked prefill fp8 KV 读变体
    PagedPrefillFp8,
    PagedPrefillKnhd,
    PagedPrefillFp8Knhd,
    /// prefill split(flash-decoding;aux = [hd, hkv, hq, tokens, nparts];
    /// ctx_base 走核参数槽,层侧传入)
    PrefillSplit,
    /// B6.3 同族:split fp8 e4m3 KV 直读(2026-10-12;修雷 + 读量减半)
    PrefillSplitFp8kv,
    PrefillSplitKnhd,
    PrefillSplitFp8kvKnhd,
    /// prefill split 归并(aux = [tokens, hq])
    PrefillSplitReduce,
    /// naive decode(slot 直排)
    NaiveDecode,
    /// 注意力输出门(y *= sigmoid(g))
    GateMul,
    /// norm_rope 融合:qk-norm + rotate-half partial rope 三发合一
    /// (aux = [tokens, heads, hd])
    NormRope,
    /// qk-norm+rope+K/V 插池三合一(aux = [tokens, hq, hkv, hd, half])
    QkvNormRopeInsert,
    /// B6.3:fp8 e4m3 主池变体(几何同 f16;池写 1B e4m3)
    QkvNormRopeInsertFp8kv,
    QkvNormRopeInsertKnhd,
    QkvNormRopeInsertFp8kvKnhd,

    // ---- ln./mlp./load.* ----
    /// fused_add_rmsnorm:residual 原地 += mixed + rmsnorm·w(aux = [rows, n])
    FusedAddRmsnorm,
    /// SwiGLU 门控 silu(g)⊙u(aux = [n])
    SiluAndMul,
    /// ct packed → marlin B 设备重排(U32;核签名形参即 rows/cols)
    CtRepack,
    /// 刀D 层间时间戳探针(链式拷贝 + clock64;标量槽 = idx/n)
    TsStamp,
    /// f16 → bf16 铸边界(草稿装载)
    CastF16Bf16,
    /// bf16 → f16 铸边界(草稿采样前)
    CastBf16F16,
    /// 设备侧贪心 argmax(采样;输出 [1] f32 索引数值)
    Argmax,
    /// 逐行 top-16(值半区 + 索引半区双 f32;aux = [rows])
    Topk16,
    /// DFlash2 格打分 + 贪心 walk 融合(单块;输出 [rows·(K·K+1)])
    DflashSelect,
    /// DFlash2 分组动态深度 2-tap 卷积
    DflashConv,
    /// 非因果块 attention(草稿;自块直读 + 前缀池)
    NaiveAttnNc,
    /// 同上 fp8kv 前缀池变体
    NaiveAttnNcFp8kv,
    /// marlin W4A16 GEMM(三名同 runtime:AWQ 7 父 / bf16 dt / f16,driver 路由)
    MarlinW4A16,
    /// FlashInfer paged prefill(f16 KV)
    FiPrefill,
    /// FlashInfer paged prefill(e4m3 KV)
    FiPrefillFp8kv,
}

impl SemanticKernel {
    /// 语义动作 → 命名空间身份证(**CUDA 解释器现役 lowering 的分派键**;
    /// 值 = 语义名空间,非 kernel 实现名 —— 实现名由 driver 按 env 推导)。
    /// CPU/AMD 解释器不走此表:直接 match 本枚举落自有实现。
    pub fn op_id(self) -> OpId {
        OpId(match self {
            SemanticKernel::GdnGatingG => "gdn.gating_g",
            SemanticKernel::GdnL2Norm => "gdn.l2norm",
            SemanticKernel::GdnConvUpd => "gdn.conv_upd",
            SemanticKernel::GdnConvUpdDual => "gdn.conv_upd_dual",
            SemanticKernel::GdnDeltaDec => "gdn.delta_dec",
            SemanticKernel::GdnDecodeStep => "gdn.decode_step",
            SemanticKernel::GdnDecodeStepV2 => "gdn.decode_step_v2",
            SemanticKernel::GdnConvFwd => "gdn.conv_fwd",
            SemanticKernel::GdnRecurrence => "gdn.recurrence_varlen_gqa",
            SemanticKernel::GdnNormAct => "gdn.norm_act",
            SemanticKernel::GdnChunkedDelta => "gdn.chunked_delta",
            SemanticKernel::GdnScalarDelta => "gdn.scalar_delta",
            SemanticKernel::Sigmoid => "ops.sigmoid",
            SemanticKernel::Add => "ops.add",
            SemanticKernel::Mul => "ops.mul",
            SemanticKernel::Silu => "ops.silu",
            SemanticKernel::Rmsnorm => "ops.rmsnorm",
            SemanticKernel::Narrow => "ops.narrow",
            SemanticKernel::Concat => "ops.concat",
            SemanticKernel::Rope => "ops.rope",
            SemanticKernel::Embed => "ops.embed",
            SemanticKernel::CastF16F32 => "elems.cast_f16_f32",
            SemanticKernel::GemvDual => "elems.gemv_dual",
            SemanticKernel::K0Write => "attn.k0_write",
            SemanticKernel::K0WriteKnhd => "attn.k0_write_knhd",
            SemanticKernel::K0WriteFp8Knhd => "attn.k0_write_fp8_knhd",
            SemanticKernel::K0WriteFp8 => "attn.k0_write_fp8",
            SemanticKernel::K0WriteFp8Bf16 => "attn.k0_write_fp8_bf16",
            SemanticKernel::K0Dual => "attn.k0_dual",
            SemanticKernel::K0DualFp8kv => "attn.k0_dual_fp8kv",
            SemanticKernel::PagedDecode => "attn.paged_decode",
            SemanticKernel::PagedDecodeV2 => "attn.paged_decode_v2",
            SemanticKernel::PagedDecodeV2Seq => "attn.paged_decode_v2_seq",
            SemanticKernel::PagedDecodeV2Knhd => "attn.paged_decode_v2_knhd",
            SemanticKernel::PagedDecodeV2Fp8Knhd => "attn.paged_decode_v2_fp8_knhd",
            SemanticKernel::PagedDecodeV2SeqKnhd => "attn.paged_decode_v2_seq_knhd",
            SemanticKernel::PagedDecodeV2Fp8SeqKnhd => "attn.paged_decode_v2_fp8_seq_knhd",
            SemanticKernel::PagedDecodeV2Fp8 => "attn.paged_decode_v2_fp8",
            SemanticKernel::PagedDecodeV2Fp8Seq => "attn.paged_decode_v2_fp8_seq",
            SemanticKernel::PagedV2Reduce => "attn.paged_v2_reduce",
            SemanticKernel::PagedPrefill => "attn.paged_prefill",
            SemanticKernel::PagedPrefillFp8 => "attn.paged_prefill_fp8",
            SemanticKernel::PagedPrefillKnhd => "attn.chunked_prefill_knhd",
            SemanticKernel::PagedPrefillFp8Knhd => "attn.chunked_prefill_fp8_knhd",
            SemanticKernel::PrefillSplit => "attn.prefill_split",
            SemanticKernel::PrefillSplitFp8kv => "attn.prefill_split_fp8kv",
            SemanticKernel::PrefillSplitKnhd => "attn.prefill_split_knhd",
            SemanticKernel::PrefillSplitFp8kvKnhd => "attn.prefill_split_fp8_knhd",
            SemanticKernel::PrefillSplitReduce => "attn.prefill_split_reduce",
            SemanticKernel::NaiveDecode => "attn.naive_decode",
            SemanticKernel::GateMul => "attn.gate_mul",
            SemanticKernel::NormRope => "attn.norm_rope",
            SemanticKernel::QkvNormRopeInsert => "attn.qkv_norm_rope_insert",
            SemanticKernel::QkvNormRopeInsertFp8kv => "attn.qkv_norm_rope_insert_fp8kv",
            SemanticKernel::QkvNormRopeInsertKnhd => "attn.qkv_norm_rope_insert_knhd",
            SemanticKernel::QkvNormRopeInsertFp8kvKnhd => "attn.qkv_norm_rope_insert_fp8kv_knhd",
            SemanticKernel::FusedAddRmsnorm => "ln.fused_add_rmsnorm",
            SemanticKernel::SiluAndMul => "mlp.silu_and_mul",
            SemanticKernel::CtRepack => "load.ct_repack",
            SemanticKernel::TsStamp => "ops.ts_stamp",
            SemanticKernel::CastF16Bf16 => "elems.cast_f16_bf16",
            SemanticKernel::CastBf16F16 => "elems.cast_bf16_f16",
            SemanticKernel::Argmax => "ops.argmax",
            SemanticKernel::Topk16 => "ops.topk16",
            SemanticKernel::DflashSelect => "dflash.select",
            SemanticKernel::DflashConv => "dflash.conv",
            SemanticKernel::NaiveAttnNc => "attn.naive_nc",
            SemanticKernel::NaiveAttnNcFp8kv => "attn.naive_nc_fp8kv",
            SemanticKernel::MarlinW4A16 => "marlin.w4a16",
            SemanticKernel::FiPrefill => "attn.fi_prefill",
            SemanticKernel::FiPrefillFp8kv => "attn.fi_prefill_fp8kv",
        })
    }
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

    // ---- 计算节点(2026-10-12 瘦身:纯算子动作全部折叠进 Call ——
    //      Add/Mul/Silu/Sigmoid/Rmsnorm 已入 SemanticKernel 词表,双词汇
    //      时代结束;Matmul/MatmulNt 留守 = 多策略复合(dtypes 分派
    //      cublas/native + cm 映射),非单一 kernel 语义)----
    /// [m,k] × [k,n]
    Matmul,
    /// [m,k] × (B 以 [n,k] 行主序直读)→ [m,n](nt = B 非转置存储;
    /// lm_head/tied 形态:权重保持 checkpoint 原布局,免 host 转置)
    MatmulNt,
    /// 形状重解释(纯元数据视图;元素数守恒;eval 透传父块,零拷贝)
    Reshape,
    /// 刀1:块内连续切片视图(零内核零节点派发;eval 透传父块,
    /// 消费面发射 `Arg::BlockSlice{id, byte_offset: offset·esz}`)。
    /// 节点偏移在声明期已含父链合成(slice_of_slice);仅连续切片
    /// (narrow 的 outer==1 ∨ src_dim==out_dim)入此臂,非连续仍物化。
    SliceView { offset_elems: usize },

    // ---- 语义调用(Driver 立项,2026-10-01;2026-10-12 枚举化):model 层
    //      只描述「要什么」(SemanticKernel 动作词表)+ 张量 + 语义标量;名/
    //      变体/发射参数 = 解释器执行期拾取(CUDA 域 = op_id → driver
    //      resolve,OpEnv 必传被动律)。aux = 拾取推导常数(层语义几何:
    //      kd/vd/batch…;非核参数,不进签名)。**model 层由此不感知任何
    //      硬件算子的存在**(S2' 定稿;枚举 = 硬件无关动作契约)。
    Call { op: SemanticKernel, aux: Vec<usize> },

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
            Op::Call { op: SemanticKernel::Add, .. } => self.add(another),
            _ => self.add(another),
        }
    }

    fn map(&self, op: Op) -> Self {
        match op {
            Op::Call { op: SemanticKernel::Silu, .. } => self.silu(),
            Op::Call { op: SemanticKernel::Sigmoid, .. } => self.sigmoid(),
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

/// lower:Mul(同形逐元素乘)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)
/// lower:Silu(一元)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)
/// lower:Sigmoid(一元;attn_output_gate 门 / GDN beta 同族)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)
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
    crate::tensor::TensorOps::call(SemanticKernel::Argmax)
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

/// foreign 家族臂线格式(名字 → 槽序 sig;解释器 Call 臂消费 —— native
/// 登记表无 foreign 条目,sig 住 client face 单源。gdn 两臂 Call 通道的
/// 桥:driver 拾取名 → 本表 → lower_kernel 对位装配)
pub(crate) fn foreign_sig(name: &str) -> Option<&'static str> {
    use owl_kernels::contract::names as n;
    match name {
        x if x == owl_kernels::family::gdn_chunked::GDN_CHUNKED_FWD => Some(owl_kernels::client::gdn_chunked::SIG),
        x if x == owl_kernels::family::gdn_scalar::GDN_SCALAR_FWD => Some(owl_kernels::client::gdn_scalar::SIG),
        x if x == n::GEMM_W4A16 => Some(owl_kernels::client::marlin::SIG),
        x if x == n::GEMM_W4A16_AWQ => Some(owl_kernels::client::marlin::SIG_AWQ),
        x if x == n::GEMM_W4A16_BF16 => Some(owl_kernels::client::marlin::SIG),
        x if x == n::PREFILL_FI => Some(owl_kernels::client::fi_sig::SIG),
        x if x == n::PREFILL_FI_FP8KV => Some(owl_kernels::client::fi_sig::SIG),
        _ => None,
    }
}

/// marlin workspace 长度(i32 个数;装载域 Want 计尺,层零 kernels 知识)
pub(crate) fn marlin_ws_elems(n: usize) -> usize {
    owl_kernels::family::marlin::v2_workspace_len(n)
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
    use owl_kernels::client::gdn_chunked as fc_chunked;
    use owl_kernels::client::gdn_scalar as fc_scalar;
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
        TensorOps::call(SemanticKernel::GdnChunkedDelta)
            .arg(q).arg(k).arg(v).arg(beta).arg(gate).arg(state)
            .arg_usize(t).arg_usize(slot).arg_usize(nv).arg_usize(nk).arg_usize(kd)
            .arg_bits((1.0f32 / (kd as f32).sqrt()).to_bits() as u64)
            .with_shape(Dtype::F16, vec![t, nv, kd])
    }

    #[test]
    fn ids_resolve_to_foreign_families() {
        // 词表 ↔ driver 分派表同票(加票不加臂 = 此处红)
        for (op, want) in [
            (SemanticKernel::GdnChunkedDelta, owl_kernels::family::gdn_chunked::GDN_CHUNKED_FWD),
            (SemanticKernel::GdnScalarDelta, owl_kernels::family::gdn_scalar::GDN_SCALAR_FWD),
        ] {
            let pick = owl_kernels::driver::resolve(OpReq {
                op: op.op_id(), env: &env(), dt: owl_kernels::driver::DType::F16,
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
            op: SemanticKernel::GdnChunkedDelta.op_id(), env: &env(), dt: owl_kernels::driver::DType::F16,
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
        let node = TensorOps::call(SemanticKernel::GdnScalarDelta)
            .arg(&q).arg(&k).arg(&v).arg(&g).arg(&beta).arg(&state)
            .arg_usize(32).arg_usize(1).arg_usize(1)
            .arg_usize(8).arg_usize(4).arg_usize(64)
            .arg_bits((1.0f32 / (64f32).sqrt()).to_bits() as u64)
            .with_shape(Dtype::F16, vec![32, 8, 64]);

        let pick = owl_kernels::driver::resolve(OpReq {
            op: SemanticKernel::GdnScalarDelta.op_id(), env: &env(), dt: owl_kernels::driver::DType::F32,
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
