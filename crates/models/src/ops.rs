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

use crate::contract::{Arg, Bytes, KernelSource, LaunchMsg};

use owl_kernels::driver::OpId;
use owl_kernels::spec::SlotSrc;

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

    // ---- nvfp4.*(NVFP4 块缩放 GEMM;sm_12x 的 prefill 臂)----
    /// NVFP4 GEMM(`out = (ga·gb)·A×Bᵀ`;权重装载期打包一次,激活每次量化)
    NvFp4Gemm,

    // ---- comm.*(TP2 S4:allreduce 的语义面;家族臂)----
    /// AR stage:本端 payload → 本端 staging slot(发布 produced)
    CommArStage,
    /// AR reduce:原地累加对端 staging slot(发布 consumed)
    CommArReduce,

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
    /// gate/up 合并块的 silu_and_mul（单输入 [rows,2*inter]）
    SiluAndMulMerged,
    /// partial rope
    Rope,
    /// 激活 per-token int8 量化(W4A8 喂料手;S1c:行 absmax → scale → 饱和
    /// round-nearest-even;`mul` = 权重侧全局因子只进 scales 出口)
    /// —— 两个 op_id 各一档(输出恒 U8,`dt` 载不了输入档;与 cast 对同律)
    QuantInt8RowF16,
    /// 同上,bf16 入
    QuantInt8RowBf16,
    /// embedding 查表
    Embed,
    /// int8 表查表 + per-row 反量化(P8;aux = [tokens],输出 f16)
    EmbedI8,

    // ---- elems.* ----
    /// f16 → f32 铸(GDN scalar/chunked 臂层侧预铸)
    CastF16F32,
    /// 刀3a':双权 GEMV 单发(b/a 投影;aux = [rows_b, rows_a, cols, tokens])
    GemvDual,
    /// **int8 权重 GEMV × bf16 激活**(P1.5-a;lm_head 专用;aux = [rows, cols, tokens])
    GemvI8wBf16,
    /// 同上 × **f16 激活**(执行 dtype = f16 的档:W4A16Awq/F16 权重的 27B)——
    /// 两档必须各有一个 op_id:driver 的拾取键就是 op_id,核实例按激活 dtype 分家。
    GemvI8wF16,
    /// W8A8 head:per-token 激活 int8 量化(scale / q 两发;2026-10-22)
    QuantI8ActScaleF16,
    QuantI8ActScaleBf16,
    QuantI8ActQF16,
    QuantI8ActQBf16,
    /// W8A8 head:int8 权重 × int8 激活 dp4a(多 token;aux = [rows, cols, tokens])
    GemvI8wA8F16,
    GemvI8wA8Bf16,
    /// W8A8 head:int8 Tensor Core(多 token;M 补到 16,aux 同 dp4a)
    GemmI8wA8TcF16,
    GemmI8wA8TcBf16,
    /// W8A16 head:f16/bf16 Tensor Core + 在线 int8→half 反量化
    GemmI8wA16TcF16,
    GemmI8wA16TcBf16,

    // ---- attn.*(aux 序见各变体)----
    /// KV 散写(classic 布局;aux = [tokens])
    K0Write,
    /// KV 散写 e4m3 池变体(aux = [tokens])
    K0WriteFp8,
    /// 同上 bf16 入(K0)
    K0WriteFp8Bf16,
    K0WriteKnhd,
    K0WriteFp8Knhd,
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
    /// LayerNorm(affine;视觉塔专用):mean/var 两趟 + weight/bias。
    /// 父 = x [rows,n] / weight [n] / bias [n];标量槽 = n(i32), eps(f32)
    Layernorm,
    /// GELU(tanh 近似;视觉塔 MLP)。标量槽 = n(sz)
    GeluTanh,
    /// GELU(精确 erf;**视觉塔 merger 专用**;与 tanh 版不可互替)
    GeluErf,
    /// 行广播加(x [rows,cols] + bias [cols];视觉塔每条 Linear 的 bias)
    /// 标量槽 = cols(sz)
    BiasAdd,
    /// 位置嵌入 4-tap 重采样(out[n,d] = Σ_k wts[n,k]·table[idx[n,k],d];
    /// 视觉塔 48² 网格 → 图网格)。父 = table / idx(f32 索引) / wts;
    /// 标量槽 = n_taps(sz), dim(sz)
    PosEmbedBilinear,
    /// 非因果 varlen 全注意力(逐图;无 GQA;视觉塔 hd=72)。
    /// 父 = q / k / v / cu_seqlens(f32 段界);
    /// 标量槽 = rows(sz), heads(sz), head_dim(sz), n_seq(i32), scale(f32)
    AttnFull,
    /// VLM 图像 token 注入:out[t] = idx[t] ≥ 0 ? merged[idx[t]] : h[t](文本侧接线)
    /// 父 = h [L,D] / merged [N,D] / idx [L](f32;负 = 保留 h);标量槽 = dim(sz)
    ImageScatter,
    /// SwiGLU 门控 silu(g)⊙u(aux = [n])
    SiluAndMul,
    /// ct packed → marlin B 设备重排(U32;核签名形参即 rows/cols)
    CtRepack,
    /// W4A8 的 B 重排(S2):同输入契约、同输出字数,tile 几何不同
    CtRepackA8,
    /// 刀D 层间时间戳探针(链式拷贝 + clock64;标量槽 = idx/n)
    TsStamp,
    /// f16 → bf16 铸边界(草稿装载)
    CastF16Bf16,
    /// 逐元素乘常数(f16/bf16;GGUF 草稿 GEMM 的成对指数移位)
    Scale,
    /// bf16 → f16 铸边界(草稿采样前)
    CastBf16F16,
    /// 设备侧贪心 argmax(采样;输出 [1] f32 索引数值)
    Argmax,
    /// 设备侧 argmax 的 (max 值, 索引) 双输出(分布式 argmax;2026-10-26)
    ArgmaxVal,
    /// 逐行 top-16(值半区 + 索引半区双 f32;aux = [rows])
    Topk16,
    /// 逐行 top-16 split-K 段一(2026-10-14 D2;aux = [rows, nparts];
    /// 输出单缓冲 [2·rows·nparts·K]:值区 f32 + 索引区 int32 位型)
    Topk16Split,
    /// 逐行 top-16 split-K 段二(一行一 warp;aux = [rows];
    /// 输入 = 段一单缓冲,输出 = 与 Topk16 逐位同布局的 [2·rows·K])
    Topk16Merge,
    /// 逐行 top-32 split-K 段一(分布式采样;2026-10-26)
    Topk32Split,
    /// 逐行 top-32 split-K 段二(输出 [2·rows·32])
    Topk32Merge,
    /// 两半表候选合并(DFlash2 切档 propose 收割;2026-10-26 刀3)
    TopkMergeHalves,
    /// DFlash2 格打分 + 贪心 walk 融合(单块;输出 [rows·(K·K+1)])
    DflashSelect,
    /// DFlash2 分组动态深度 2-tap 卷积
    DflashConv,
    /// 非因果块 attention(草稿;自块直读 + 前缀池)
    NaiveAttnNc,
    /// 同上 fp8kv 前缀池变体
    NaiveAttnNcFp8kv,
    /// marlin W4A16 GEMM(四名同 runtime:AWQ 7 父 / bf16 dt / f16,driver 路由)
    MarlinW4A16,
    /// marlin W4A8 × AWQ GEMM(int8 激活 + per-token fp32 尺度;8 父块 ——
    /// W4A8 线,2026-10-16:层侧按档位/选层显式声明而非常见于两档)
    MarlinW4A8Awq,
    /// **GGUF 激活 → q8_1 块**(f16 入;MMVQ 喂料手;2026-10-19 gguf 轮)
    /// —— 与 `QuantInt8RowF16` 同律:输出恒 U8,`dt` 载不了输入档 ⇒ 按输入
    /// dtype 分 op_id(bf16 臂另立变体)
    GgufQuantActF16,
    /// 同上,bf16 入
    GgufQuantActBf16,
    /// **GGUF dequant-in-kernel GEMM**(MMVQ;权重 = GGUF 块字节,压缩驻留)。
    /// 块型住 `aux[0]`(ggml 型码)→ driver 经 `family::gguf::entry_for` 单一
    /// 映射拾取入口(H3:逐型一条准入声明)
    GgufGemm,
    /// **GGUF 权重复用 GEMM**(prefill;W-2):同一 warp 一行权重算 `TILE_MT`
    /// 个 token —— 与 [`Self::GgufGemm`] 同槽序,只是调度不同(门:
    /// 两臂**位级相同**)
    GgufGemmTile,
    /// **GGUF smem 复用 GEMM**(prefill 的算术杠杆;W-2b):权重窗解量化进 smem
    /// **每窗一次**,激活直吃 f16(不经 q8_1),寄存器 4×4 分块点积
    GgufGemmSmem,
    /// **GGUF 官方 MMQ**(tensor-core prefill;内核 = llama.cpp vendoring)。
    /// 与 smem/vec_dot 臂同槽序?**不**:MMQ 只吃 (weights, xq, out, k, n, m)。
    GgufMmq,
    /// **GGUF 内维列序 gather**(W-2e;f16):`out[.., 32·i] = x[.., 32·tbl[i]]`
    /// —— 把 `ssm_out` 的 tiled 列序一次性换成 grouped 序,使它也能吃
    /// cuBLAS/MMQ 臂(此前只能退回自研 tile 臂;
    /// 见设计 §19.5 定位 ①)
    GgufGatherCols,
    /// **GGUF int8-MMA MMQ**(W-2f;head 版 llama.cpp):权重块字节直读 +
    /// q8_1_mmq 激活 → f16 —— 走 `mma.sync.m16n8k16.s8`(sm80+ int8 张量核)
    GgufMmq2,
    /// 同上:**激活量化** f16 → `block_q8_1_mmq`(DS4 布局;Q4_K/Q5_K)
    GgufMmq2QuantDs4,
    /// 同上(D4 布局;Q8_0/Q6_K)
    GgufMmq2QuantD4,
    /// MMQ2 激活量化(BF16 入;DFlash2 草稿线 2026-10-21)
    GgufMmq2QuantDs4Bf16,
    /// MMQ2 激活量化(BF16 入;D4)
    GgufMmq2QuantD4Bf16,
    /// **GGUF 行窗解量化**(W-2d):量化块行窗 `[n0, n0+rows)` → f16 `[rows, k]`
    /// (写进调用方给的 scratch;配 `Op::CallDst`)
    GgufCvtRange,
    /// **写列窗 f16 GEMM**(W-2d;foreign cuBLAS):`C = y + off`,`ldc = y 行宽`
    /// —— 照 llama.cpp `ggml_cuda_op_mul_mat_cublas` 同代实现(f16 累加)
    GgufCublasDst,
    /// **GGUF 量化表的行解量化**(查表臂;`token_embd` 在 GGUF 里是量化块,
    /// 既不是 f16 表也不可解量化驻留 ⇒ 查表自带解量化;块型住 `aux[0]`,几何 = `rows_geom`)
    GgufDequantRows,
    /// **GGUF 激活侧列置换**(v-head 反重排:内维是 ggml tiled 序而 GDN 核
    /// 产出 grouped 序;置换跨量化块 ⇒ 只能在激活侧等价置换,
    /// **折进已要跑的激活量化核**:零额外发射零额外趟数。两档 dtype 各一入口。
    GgufQuantActTiledF16,
    /// 同上,bf16 入
    GgufQuantActTiledBf16,
    /// FlashInfer paged prefill(f16 KV)
    FiPrefill,
    /// FlashInfer paged prefill(e4m3 KV)
    FiPrefillFp8kv,
    /// XQA verify/decode 注意力(kNHD fp8 池;foreign-kernel)
    XqaVerify,
    /// XQA 裸 decode 注意力(q=1;第二份实例化 `M_TILESIZE=16`;foreign-kernel)
    XqaDecode,
    /// XQA prefill 注意力(chunked q=T;与 verify 同资产/同槽序,
    /// host 几何强制单块 —— 见 `kernels::server::xqa::XqaPrefillRuntime`)
    XqaPrefill,
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
            SemanticKernel::NvFp4Gemm => "nvfp4.gemm",
            SemanticKernel::CommArStage => "comm.ar_stage",
            SemanticKernel::CommArReduce => "comm.ar_reduce",
            SemanticKernel::Sigmoid => "ops.sigmoid",
            SemanticKernel::Add => "ops.add",
            SemanticKernel::Mul => "ops.mul",
            SemanticKernel::Silu => "ops.silu",
            SemanticKernel::Rmsnorm => "ops.rmsnorm",
            SemanticKernel::Narrow => "ops.narrow",
            SemanticKernel::Concat => "ops.concat",
            SemanticKernel::Rope => "ops.rope",
            SemanticKernel::QuantInt8RowF16 => "elems.quant_int8_row_f16",
            SemanticKernel::QuantInt8RowBf16 => "elems.quant_int8_row_bf16",
            SemanticKernel::Embed => "ops.embed",
            SemanticKernel::EmbedI8 => "elems.embed_i8",
            SemanticKernel::CastF16F32 => "elems.cast_f16_f32",
            SemanticKernel::GemvDual => "elems.gemv_dual",
            SemanticKernel::GemvI8wBf16 => "elems.gemv_i8w_bf16",
            SemanticKernel::GemvI8wF16 => "elems.gemv_i8w_f16",
            SemanticKernel::QuantI8ActScaleF16 => "elems.quant_i8_act_scale_f16",
            SemanticKernel::QuantI8ActScaleBf16 => "elems.quant_i8_act_scale_bf16",
            SemanticKernel::QuantI8ActQF16 => "elems.quant_i8_act_q_f16",
            SemanticKernel::QuantI8ActQBf16 => "elems.quant_i8_act_q_bf16",
            SemanticKernel::GemvI8wA8F16 => "elems.gemv_i8w_a8_f16",
            SemanticKernel::GemvI8wA8Bf16 => "elems.gemv_i8w_a8_bf16",
            SemanticKernel::GemmI8wA8TcF16 => "elems.gemm_i8w_a8_tc_f16",
            SemanticKernel::GemmI8wA8TcBf16 => "elems.gemm_i8w_a8_tc_bf16",
            SemanticKernel::GemmI8wA16TcF16 => "elems.gemm_i8w_a16_tc_f16",
            SemanticKernel::GemmI8wA16TcBf16 => "elems.gemm_i8w_a16_tc_bf16",
            SemanticKernel::K0Write => "attn.k0_write",
            SemanticKernel::K0WriteKnhd => "attn.k0_write_knhd",
            SemanticKernel::K0WriteFp8Knhd => "attn.k0_write_fp8_knhd",
            SemanticKernel::K0WriteFp8 => "attn.k0_write_fp8",
            SemanticKernel::K0WriteFp8Bf16 => "attn.k0_write_fp8_bf16",
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
            SemanticKernel::Layernorm => "vis.layernorm",
            SemanticKernel::GeluTanh => "vis.gelu_tanh",
            SemanticKernel::GeluErf => "vis.gelu_erf",
            SemanticKernel::BiasAdd => "vis.bias_add",
            SemanticKernel::PosEmbedBilinear => "vis.pos_embed",
            SemanticKernel::AttnFull => "vis.attn_full",
            SemanticKernel::ImageScatter => "vis.image_scatter",
            SemanticKernel::SiluAndMul => "mlp.silu_and_mul",
            SemanticKernel::SiluAndMulMerged => "mlp.silu_and_mul_merged",
            SemanticKernel::CtRepack => "load.ct_repack",
            SemanticKernel::CtRepackA8 => "load.ct_repack_a8",
            SemanticKernel::TsStamp => "ops.ts_stamp",
            SemanticKernel::CastF16Bf16 => "elems.cast_f16_bf16",
            SemanticKernel::Scale => "elems.scale",
            SemanticKernel::CastBf16F16 => "elems.cast_bf16_f16",
            SemanticKernel::Argmax => "ops.argmax",
            SemanticKernel::ArgmaxVal => "ops.argmax_val",
            SemanticKernel::Topk16 => "ops.topk16",
            SemanticKernel::Topk16Split => "ops.topk16_split",
            SemanticKernel::Topk16Merge => "ops.topk16_merge",
            SemanticKernel::Topk32Split => "ops.topk32_split",
            SemanticKernel::Topk32Merge => "ops.topk32_merge",
            SemanticKernel::TopkMergeHalves => "ops.topk_merge_halves",
            SemanticKernel::DflashSelect => "dflash.select",
            SemanticKernel::DflashConv => "dflash.conv",
            SemanticKernel::NaiveAttnNc => "attn.naive_nc",
            SemanticKernel::NaiveAttnNcFp8kv => "attn.naive_nc_fp8kv",
            SemanticKernel::GgufQuantActF16 => "gguf.quant_act_f16",
            SemanticKernel::GgufQuantActBf16 => "gguf.quant_act_bf16",
            SemanticKernel::GgufGemm => "gguf.gemm",
            SemanticKernel::GgufGemmTile => "gguf.gemm_tile",
            SemanticKernel::GgufGemmSmem => "gguf.gemm_smem",
            SemanticKernel::GgufMmq => "gguf.mmq",
            SemanticKernel::GgufMmq2 => "gguf.mmq2",
            SemanticKernel::GgufMmq2QuantDs4 => "gguf.mmq2_quant_ds4",
            SemanticKernel::GgufMmq2QuantD4 => "gguf.mmq2_quant_d4",
            SemanticKernel::GgufMmq2QuantDs4Bf16 => "gguf.mmq2_quant_ds4_bf16",
            SemanticKernel::GgufMmq2QuantD4Bf16 => "gguf.mmq2_quant_d4_bf16",
            SemanticKernel::GgufCvtRange => "gguf.dequant_range",
            SemanticKernel::GgufGatherCols => "gguf.gather_cols",
            SemanticKernel::GgufCublasDst => "cublas.gemm_f16_dst",
            SemanticKernel::GgufDequantRows => "gguf.dequant_rows",
            SemanticKernel::GgufQuantActTiledF16 => "gguf.quant_act_f16_tiled",
            SemanticKernel::GgufQuantActTiledBf16 => "gguf.quant_act_bf16_tiled",
            SemanticKernel::MarlinW4A16 => "marlin.w4a16",
            SemanticKernel::MarlinW4A8Awq => "marlin.w4a8_awq",
            SemanticKernel::FiPrefill => "attn.fi_prefill",
            SemanticKernel::FiPrefillFp8kv => "attn.fi_prefill_fp8kv",
            SemanticKernel::XqaVerify => "attn.xqa_verify",
            SemanticKernel::XqaDecode => "attn.xqa_decode",
            SemanticKernel::XqaPrefill => "attn.xqa_prefill",
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
        // **8 位三族各自点名**(2026-10-16):fp8-e4m3(fp8 池/激活)、
        // fp8-e5m2(并列格式)、**int8**(有符号;W4A8 激活)、以及真正不透明的
        // `U8` 字节块。driver 词汇里它们各自独立 —— 只有声明里写同一 dtype
        // 的算子才收得到,其余算子的 `match` 响亮拒。
        crate::contract::Dtype::F8E4M3 => DType::F8E4M3,
        crate::contract::Dtype::F8E5M2 => DType::F8E5M2,
        crate::contract::Dtype::I8 => DType::I8,
        crate::contract::Dtype::U8 => DType::U8,
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
    /// **无清零分配**(scratch 块;内容未定义)。
    ///
    /// 唯一合法用法 = **下一位写者先整体覆盖**(W-2d 的"反量化进 scratch"就是):
    /// 用 `Zeros` 会让每次 memset 吃掉省下的带宽(27B 一档 = 每 chunk 数十 GB)。
    /// 消费者若在写者之前读它 ⇒ 读到上一位使用者的旧值(契约:**调用方负责**,
    /// 与 `Block` 的"数据已在"是同一条纪律的两面)。
    Uninit,
    /// **写入既有块**(dest = `parents[0]`)。
    ///
    /// 语义:本节点把结果写进 `parents[0]` 的(子)窗口,节点值 = **同块别名**
    /// (eval 返回 `ins[0].clone()` 并 `track_alias`)。用途 = 一个输出由多个
    /// 分块核拼出来(W-2d 的 cuBLAS 列窗:每个 tile 写 `y[:, n0..]`,ldc = 行宽)。
    ///
    /// 序纪律:各分块节点之间**没有数据边**,靠"单计算流 + 尾序发射"保证先后
    /// (与 `SlotWrite` 的显式边不同 —— 那需要跨节目可导;这里是同一节点树内的
    /// 顺序发射,cuBLAS 与该流同序)。
    CallDst { op: SemanticKernel, aux: Vec<usize> },
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
///   Tensor 是首个实现;将来别的节点形态(如捕获烘焙产物)也可实现。
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
    ((n as u32).div_ceil(256), 1, 1)
}

/// lower:Add(同形二元;a/b 块等长由 server 账长校验兜底)
/// lower:Add(同形二元;a/b 块等长由 server 账长校验兜底)
/// n = 声明元素数(C1 维度源头单一律:只读声明 shape,不读块账长)
///
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
        // 核名 = 族身份常量(字面量住址 owl_kernels::family::cublas;
        // models 零 cublas feature 也经此引用 —— 字面量全库只写一次)
        kernel: KernelSource {
            name: if bf16 {
                owl_kernels::family::cublas::GEMM_BF16.to_string()
            } else {
                owl_kernels::family::cublas::GEMM_F16.to_string()
            },
            source: String::new(),
        },
        args: vec![
            ins[0].clone(),
            ins[1].clone(),
            Arg::block(out.id, out.dtype, out.rank),
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
            Arg::block(out.id, out.dtype, out.rank),
            Arg::I32(m as i32),
            Arg::I32(k as i32),
            Arg::I32(n as i32),
        ],
        grid: ((m as u32).div_ceil(16), (n as u32).div_ceil(16), 1),
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
            Arg::block(out.id, out.dtype, out.rank),
            Arg::I32(m as i32),
            Arg::I32(k as i32),
            Arg::I32(n as i32),
        ],
        grid: ((m as u32).div_ceil(16), (n as u32).div_ceil(16), 1),
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
    // 槽序唯一权威(优先序:算子元数据 → 登记表 → 逃生舱签名;
    // 2026-10-13 P3:已建表算子经 KernelSpec 的 PARAMS 派生,零第二份手抄)
    let toks: Vec<&str> = kernel.slot_tokens();
    // 槽的事实来源(父张量 / 输出 / 标量):**规则单源** =
    // `owl_kernels::spec::slot_sources_of_tokens`(2026-10-15:原 `is_out`
    // 闭包就地推导 —— 事实面检查要与装配序逐槽对齐,两处推导必漂)
    let srcs: Vec<SlotSrc> = kernel.slot_sources();
    // 父张量槽数必须等于父依赖数
    let t_in = srcs.iter().filter(|s| matches!(s, SlotSrc::Parent(_))).count();
    assert!(
        t_in == ins.len(),
        "lower_kernel({}): 签名 T 槽 {t_in} != 父依赖 {}(检查 .arg 链)",
        kernel.name,
        ins.len()
    );
    let n_scal = srcs.iter().filter(|s| matches!(s, SlotSrc::Scalar)).count();
    assert!(
        n_scal == scalars.len(),
        "lower_kernel({}): 签名标量槽 {n_scal} != 标量参数 {}(sz/i32/f32 与 arg_usize/arg_i32/arg_f32 对位)",
        kernel.name,
        scalars.len()
    );

    // 按槽源装配:Parent → 下一个父块;Out → 输出块;Scalar → 下一个标量
    let mut args: Vec<Arg> = Vec::with_capacity(toks.len());
    let mut si = 0usize;
    for (i, (tok, src)) in toks.iter().zip(srcs.iter()).enumerate() {
        match src {
            SlotSrc::Out => args.push(Arg::block(out.id, out.dtype, out.rank)),
            SlotSrc::Parent(pi) => args.push(ins[*pi].clone()),
            SlotSrc::Scalar => {
                match *tok {
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
                si += 1;
            }
        }
    }
    let (grid, block, shared_mem) = if kernel.launch.grid == (0, 0, 0) {
        (auto_grid(out_elems), kernel.launch.block, kernel.launch.shared_mem)
    } else {
        (kernel.launch.grid, kernel.launch.block, kernel.launch.shared_mem)
    };
    LaunchMsg {
        kernel: KernelSource { name: kernel.name.to_string(), source: kernel.src.to_string() },
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

/// **逐行 top-32 候选**(K=32;分布式采样用,2026-10-26):两段 split-K,
/// 输出单缓冲 `[2·rows·32]`(值区行主 + 索引区,索引 f32 过线)。
///
/// `stride` = 本 rank 的 logits 行宽(词表切档 = 半表)⇒ 索引是**半表域**,
/// 全局 id 由 host 合并面加偏移(与 DFlash2 候选同律)。
///
/// **设备端重复惩罚**:`hist` = 全局 id 升序去重 + 尾部 u32::MAX 的固定窗口;
/// `rep_penalty` = `[1]` f32 设备张量,核先按 `vocab_offset + local` 命中历史、
/// 再做 top-32。两者都走图输入槽(不随内容重捕获)。`rep_penalty <= 1.0`
/// 核内整段跳过,与旧 top-32 逐位一致。
pub fn topk32_of_logits(
    logits: &crate::tensor::TensorOps,
    hist: &crate::tensor::TensorOps,
    rep_penalty: &crate::tensor::TensorOps,
    rows: usize,
    nparts: usize,
    vocab_offset: usize,
) -> crate::tensor::TensorOps {
    use crate::contract::Dtype;
    const K: usize = 32;
    let stride = logits.shape()[1];
    let part = crate::tensor::TensorOps::call(SemanticKernel::Topk32Split)
        .aux(&[rows, nparts])
        .arg(logits)
        .arg(hist)
        .arg(rep_penalty)
        .arg_i32(stride as i32)
        .arg_i32(nparts as i32)
        .arg_i32(rows as i32)
        .arg_i32(hist.shape()[0] as i32)
        .arg_i32(vocab_offset as i32)
        .with_shape(Dtype::F32, vec![2 * rows * nparts * K]);
    crate::tensor::TensorOps::call(SemanticKernel::Topk32Merge)
        .aux(&[rows])
        .arg(&part)
        .arg_i32(nparts as i32)
        .arg_i32(rows as i32)
        .arg_i32(stride as i32)
        .with_shape(Dtype::F32, vec![2 * rows * K])
}

/// 设备侧 argmax 的 **(max 值, 索引)** 双输出([2] f32;值在前)。
///
/// 用于 TP 词表切的两半表合并:两 rank 各出本半的最大值+索引 ⇒ host 只比较
/// 2 个数(每行 8B),不再回读 4MB 半表 logits。平局语义与
/// [`argmax_f32idx`] 一致(最小索引)。
pub fn argmax_validx(x: &crate::tensor::TensorOps, n: usize, offset: usize) -> crate::tensor::TensorOps {
    use crate::contract::Dtype;
    crate::tensor::TensorOps::call(SemanticKernel::ArgmaxVal)
    .arg(x)
    .arg_i32(n as i32)
    .arg_i32(offset as i32)
    .with_shape(Dtype::F32, vec![2])
}

/// **激活 per-token int8 量化声明**(W4A8 喂料手;S1c)。
///
/// `x[rows, cols]` → `q[rows, cols]`(**u8 字节承载 int8**,引擎分配的瞬时块)
/// + `scales[rows]`(f32;受 `mul` 缩放)。
///
/// 槽序与 `ops::elems::QUANT_INT8_ROW_*` 表逐字对齐:
/// `x / rows(i32) / cols(i32) / mul(设备标量 f32[1]) / scales(入槽,写出) / q(出槽)`。
/// `rows` 同时进 `aux`(driver 的 grid = rows,与核的"一 token 一 block"对上)。
///
/// - **`mul` = 设备标量块**(权重侧全局因子;装载期由尺度数据算出 ⇒ 层声明 want
///   由装载器填,与 vLLM `input_global_scale` 同构);
/// - **`scales` 由调用方备块**(写出语义,同 xqa 的 `wr`/`scratch` 先例);
/// - `q` 是引擎按形状分配的**瞬时输出**(U8 字节承载)。
#[allow(clippy::too_many_arguments)]
pub fn quant_int8_row(
    x: &crate::tensor::TensorOps,
    mul: &crate::tensor::TensorOps,
    scales: &crate::tensor::TensorOps,
    rows: usize,
    cols: usize,
) -> crate::tensor::TensorOps {
    use crate::contract::Dtype;
    let op = match x.dtype {
        Dtype::F16 => SemanticKernel::QuantInt8RowF16,
        Dtype::BF16 => SemanticKernel::QuantInt8RowBf16,
        other => panic!("quant_int8_row: 激活只可能是 f16/bf16,实得 {other:?}"),
    };
    crate::tensor::TensorOps::call(op)
        .arg(x)
        .arg(mul)
        .arg(scales)
        .arg_i32(rows as i32)
        .arg_i32(cols as i32)
        .aux(&[rows])
        .with_shape(Dtype::I8, vec![rows, cols])
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
    match name {
        x if x == owl_kernels::family::gdn_chunked::GDN_CHUNKED_FWD => Some(owl_kernels::client::gdn_chunked::SIG),
        x if x == owl_kernels::family::gdn_scalar::GDN_SCALAR_FWD => Some(owl_kernels::client::gdn_scalar::SIG),
        x if x == owl_kernels::family::marlin::GEMM_W4A16 => Some(owl_kernels::client::marlin::SIG),
        x if x == owl_kernels::family::marlin::GEMM_W4A16_AWQ => Some(owl_kernels::client::marlin::SIG_AWQ),
        x if x == owl_kernels::family::marlin::GEMM_W4A8_AWQ => Some(owl_kernels::client::marlin::SIG_W4A8),
        x if x == owl_kernels::family::marlin::GEMM_W4A16_BF16 => Some(owl_kernels::client::marlin::SIG),
        x if x == owl_kernels::family::flashinfer::PREFILL_FI => Some(owl_kernels::client::fi_sig::SIG),
        x if x == owl_kernels::family::flashinfer::PREFILL_FI_FP8KV => Some(owl_kernels::client::fi_sig::SIG),
        // 通信族(TP2 S4;三 dtype 同 sig —— 槽序不随 dtype 变)
        x if x == owl_kernels::family::comm::asset::STAGE_F32 => Some(owl_kernels::client::comm::SIG),
        x if x == owl_kernels::family::comm::asset::STAGE_F16 => Some(owl_kernels::client::comm::SIG),
        x if x == owl_kernels::family::comm::asset::STAGE_BF16 => Some(owl_kernels::client::comm::SIG),
        x if x == owl_kernels::family::comm::asset::REDUCE_F32 => Some(owl_kernels::client::comm::SIG_REDUCE),
        x if x == owl_kernels::family::comm::asset::REDUCE_F16 => Some(owl_kernels::client::comm::SIG_REDUCE),
        x if x == owl_kernels::family::comm::asset::REDUCE_BF16 => Some(owl_kernels::client::comm::SIG_REDUCE),
        x if x == owl_kernels::family::xqa::VERIFY_XQA => Some(owl_kernels::family::xqa::VERIFY_XQA_SIG),
        x if x == owl_kernels::family::xqa::DECODE_XQA => Some(owl_kernels::family::xqa::DECODE_XQA_SIG),
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
    use crate::contract::Dtype;
    
    use crate::tensor::TensorOps;
    use owl_kernels::client::gdn_chunked as fc_chunked;
    use owl_kernels::client::gdn_scalar as fc_scalar;
    use owl_kernels::driver::{OpEnv, OpReq};

    fn env() -> OpEnv {
        OpEnv { hw: owl_kernels::driver::Hw { arch: owl_kernels::Arch::Sm86 }, page: 0, head_dim: 0, grp: 0 }
    }

    /// 层声明形态(chunked;与 layers/gdn.rs 逐字同构)
    #[allow(clippy::too_many_arguments)]
    /// GDN chunked 声明(P-6b:槽面 + `tables`(cu|coff|idx)与 `nt` 入参;
    /// 顺序与 `layers::gdn::chunked_delta` 逐字同源 —— 线格式单源)
    #[allow(clippy::too_many_arguments)]
    fn declare_chunked(
        q: &TensorOps, k: &TensorOps, v: &TensorOps, beta: &TensorOps,
        gate: &TensorOps, state: &TensorOps, tables: &TensorOps, nt: usize, slot: usize,
        t: usize, nv: usize, nk: usize, kd: usize,
    ) -> TensorOps {
        TensorOps::call(SemanticKernel::GdnChunkedDelta)
            .arg(q).arg(k).arg(v).arg(beta).arg(gate).arg(state).arg(tables)
            .arg_usize(t).arg_usize(slot).arg_usize(nv).arg_usize(nk).arg_usize(kd)
            .arg_bits((1.0f32 / (kd as f32).sqrt()).to_bits() as u64)
            .arg_usize(nt)
            .with_shape(Dtype::F16, vec![t, nv, kd])
    }

    #[test]
    fn ids_resolve_to_foreign_families() {
        // 词表 ↔ driver 分派表同票(加票不加臂 = 此处红)
        for (op, want) in [
            (SemanticKernel::GdnChunkedDelta, owl_kernels::family::gdn_chunked::GDN_CHUNKED_FWD),
            (SemanticKernel::GdnScalarDelta, owl_kernels::family::gdn_scalar::GDN_SCALAR_FWD),
            // 通信族(TP2 S4;本测试 dt = F16 ⇒ 期望 F16 实例名)
            (SemanticKernel::CommArStage, owl_kernels::family::comm::asset::STAGE_F16),
            (SemanticKernel::CommArReduce, owl_kernels::family::comm::asset::REDUCE_F16),
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

    /// **int8 头 GEMV 两档选择面**(P1.5-a;2026-10-20):激活 dtype → op_id →
    /// 核实例各一颗(f16/bf16)。缺第二条 = f16 执行档拿 bf16 核 ⇒ dtype 守门
    /// 当场红(实测:27B W4A16Awq 的 lm_head 装配即撞)。
    #[test]
    fn gemv_i8w_selection_face_is_wired() {
        for (sk, dt, want) in [
            (SemanticKernel::GemvI8wF16, owl_kernels::driver::DType::F16, "owl_gemv_i8w_f16"),
            (SemanticKernel::GemvI8wBf16, owl_kernels::driver::DType::BF16, "owl_gemv_i8w_bf16"),
        ] {
            let pick = owl_kernels::driver::resolve(OpReq {
                op: sk.op_id(),
                env: &env(),
                dt,
                shapes: &[],
                // aux = [rows, cols, tokens](cols 须 16 对齐)
                aux: &[124160, 5120, 1],
                scalars: &[],
            });
            assert_eq!(pick.name, want, "{sk:?}: op_id 应直指该档核名");
            let m = pick.meta().expect("必须有准入声明");
            assert_eq!(m.out_dtype().map(|d| format!("{d:?}")), Some(format!("{dt:?}")), "{sk:?}: 声明输出 dtype");
        }
    }

    /// **S1c 激活量化臂的选择面四点同票**(词表 → driver 拾取名 → 准入声明;
    /// 输出域 = U8 是本算子独有的,故另锁"node dtype = U8"这条事实面)
    #[test]
    fn quant_int8_row_selection_face_is_wired() {
        for (sk, want) in [
            (SemanticKernel::QuantInt8RowF16, "owl_quant_int8_row_f16"),
            (SemanticKernel::QuantInt8RowBf16, "owl_quant_int8_row_bf16"),
        ] {
            let pick = owl_kernels::driver::resolve(OpReq {
                op: sk.op_id(),
                env: &env(),
                // 输出域 = U8(字节承载;自 S1c 起进 driver 词汇,与 U32 同律)
                dt: owl_kernels::driver::DType::I8,
                shapes: &[],
                aux: &[8],
                scalars: &[],
            });
            assert_eq!(pick.name, want, "{sk:?}: op_id 应直指该档核名");
            assert_eq!(
                (pick.shape.grid, pick.shape.block),
                ((8, 1, 1), (256, 1, 1)),
                "{sk:?}: 几何 = 一 token 一 block(aux[0] = rows)"
            );
            let m = pick.meta().expect("必须有准入声明");
            assert_eq!(m.out_dtype(), Some(crate::contract::Dtype::I8), "{sk:?}: 出槽 I8(int8)");
            assert_eq!(m.param_names()[0], "x");
            assert_eq!(m.param_names().last().copied(), Some("q"), "{sk:?}: 出槽末位");
        }
        // ddt:I8 放行(否则本臂的节点 dtype 过不了 eval)
        assert_eq!(
            crate::ops::ddt(crate::contract::Dtype::I8).expect("I8 应放行"),
            owl_kernels::driver::DType::I8
        );
    }

    /// **W4A8×AWQ 臂的选择面四点同票**(2026-10-16 C④-3):词表
    /// (`SemanticKernel::MarlinW4A8Awq`)→ driver 拾取名 → 家族线格式 sig →
    /// 准入声明(建表)。层侧 S2 接线前先把这四点锁住 —— 任一环缺失
    /// (加票不加臂/加臂不加 sig/加名不上册)这里就红。
    #[test]
    fn w4a8_awq_selection_face_is_wired() {
        let pick = owl_kernels::driver::resolve(OpReq {
            op: SemanticKernel::MarlinW4A8Awq.op_id(),
            env: &env(),
            dt: owl_kernels::driver::DType::F16,
            shapes: &[],
            aux: &[],
            scalars: &[],
        });
        assert_eq!(pick.name, owl_kernels::family::marlin::GEMM_W4A8_AWQ);
        assert_eq!(
            (pick.shape.grid, pick.shape.block, pick.shape.smem),
            ((0, 0, 0), (0, 0, 0), 0),
            "foreign 臂发射配置应直通"
        );
        assert_eq!(
            foreign_sig(pick.name),
            Some(owl_kernels::client::marlin::SIG_W4A8),
            "拾取名必须能查到家族线格式 sig(8 Block + 4 sz)"
        );
        assert!(
            pick.meta().is_some(),
            "W4A8 臂必须有准入声明(`ops::marlin::W4A8_AWQ` 在建表里)"
        );
    }

    #[test]
    fn chunked_call_wire_survives_server_parse() {
        let q = TensorOps::of_block(1, Dtype::F16, vec![64, 16, 128]);
        let k = TensorOps::of_block(2, Dtype::F16, vec![64, 16, 128]);
        let v = TensorOps::of_block(3, Dtype::F16, vec![64, 48, 128]);
        let beta = TensorOps::of_block(4, Dtype::F16, vec![64, 48]);
        let gate = TensorOps::of_block(5, Dtype::F16, vec![64, 48]);
        let state = TensorOps::of_block(6, Dtype::F32, vec![8, 48, 128, 128]);
        // 序列表槽(cu|coff|idx;P-6b)= U32 位型 [IDX_OFF/4 + 4·nt] 词
        let nt = 1usize;
        let tbl_words = (owl_kernels::family::gdn_chunked::IDX_OFF / 4) + 4 * nt;
        let tables = TensorOps::of_block(7, Dtype::U32, vec![tbl_words]);
        let node = declare_chunked(&q, &k, &v, &beta, &gate, &state, &tables, nt, 3, 64, 48, 16, 128);

        // 解释器 Call 臂同款:pick → foreign sig kernel → lower_kernel
        let pick = owl_kernels::driver::resolve(OpReq {
            op: SemanticKernel::GdnChunkedDelta.op_id(), env: &env(), dt: owl_kernels::driver::DType::F16,
            shapes: &[], aux: &[], scalars: &[],
        });
        let kernel = crate::kernel::Kernel::new(pick.name, "")
            .with_sig(foreign_sig(pick.name).expect("foreign sig"));
        let out = Bytes::new(99, 64 * 48 * 128, Dtype::F16, 3); // [t, nv, kd]
        let ins: Vec<Arg> = [&q, &k, &v, &beta, &gate, &state, &tables]
            .iter().map(|t| Arg::block(t.id, t.dtype, t.shape.len() as u8)).collect();
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
        assert_eq!(call.tables().tables.id, tables.id);
        assert_eq!(call.tables().nt, nt);
        assert_eq!(call.shape().t, 64);
        assert_eq!(call.shape().nv, 48);
        assert_eq!(call.shape().nk, 16);
        assert_eq!(call.shape().kd, 128);
        // 槽序数:8 Block(含 tables)+ 7 sz(lower_kernel 对位 sig 全额)
        let blocks = msg.args.iter().filter(|a| matches!(a, Arg::Block { .. })).count();
        assert_eq!(blocks, 8);
        assert_eq!(msg.args.len(), 15);
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
        let out = Bytes::new(99, 32 * 8 * 64, Dtype::F16, 3);
        let ins: Vec<Arg> = [&q, &k, &v, &g, &beta, &state]
            .iter().map(|t| Arg::block(t.id, t.dtype, t.shape.len() as u8)).collect();
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
