//! DFlash2 草稿(E5-DF1;2026-10-07)—— z-lab/Qwen3.8-27B-DFlash2 检查点
//! (1.92B BF16 81 张量;sglang `models/dflash.py` DFlash2DraftModel 同构,
//! 逐式对拍锚,sglang-dflash-patch 分支)。
//!
//! ```text
//! memory = hidden_norm(fc(⊕5 taps))                # target hidden 投影
//! x = embed([anchor, MASK×7])                      # 8 行噪声块(无跨轮状态)
//! 5 × layer:  h = input_ln(x, residual)
//!             (ci, δ) = attention_conv.prepare(h)
//!             a = self_attn(ci)                    # GQA 32/8/128 + qk_norm + RoPE
//!             a = attention_conv.finish(a, δ)      # 分组动态 2-tap 卷积
//!             n2 = post_ln(a, residual)【fused add】
//!             (ci2, δ2) = mlp_conv.prepare(n2)
//!             h2 = mlp(ci2); h2 = mlp_conv.finish(h2, δ2)
//!             (x, residual) = (h2, residual)       # h2 未加残差(sglang 约定)
//! hidden = norm(x, residual)                       # final fused add + norm
//! drafts = selector(hidden[1:], anchor)            # 7 草稿(码本选择器)
//! ```
//!
//! selector(sglang CandidateSelector 同式):候选 = target lm_head 逐行
//! top-16(unary = 变换后 top-k 值;本家族 multiplier=1 无 softcap);
//! score[e,p,c] = unary[e,c] + ⟨A[pred]⊙proj(e), B[cand]⟩,pred(e=0)=锚、
//! pred(e>0)=cand[e-1];贪心 walk 单 GPU 核(全设备零 host 往返)。
//!
//! 与 MTP 的分野:草稿头非 248K 逐步 argmax(3×2.4GB lm_head 税)→
//! 一次前向 7 草稿 + 一次 lm_head top-16(7 行);草稿注意力 = 非因果
//! 双向块(ENCODER_ONLY,is_causal=false)+ **无逐 token decode 臂**
//! (唯一读路径 = propose 块;唯一写路径 = encode/自块槽写入)。
//!
//! 装载:键约定住 specs(qwen35.rs `impl Loadable` 家规);fc 平声明
//! [5120, 25600] 直读,消费面 reshape 重释 [25600, 5120](零拷贝转置)
//! 后 5 连续 slice_view 直吃 5 taps(免 concat 免虚拟键,Σ 5 GEMM)。

use crate::contract::Dtype;
use crate::layers::linear::Linear;
use crate::layers::mlp::Mlp;
use crate::layers::rmsnorm::RmsNorm;
use crate::layers::rope::Rope;
use crate::module::{ForwardCtx, KvBuffers, Module, QuantPlan, Weight};
use crate::TensorOps;

// ---- 检查点家族定形常量(dflash_config 实测;z-lab/Qwen3.8-27B-DFlash2)----
/// conv taps(conv_kernel_size)
pub const TAPS: usize = 2;
/// conv 组内通道(conv_group_size)
pub const GROUP: usize = 16;
/// 块宽 = 1 锚 + 7 草稿(block_size;conv 位置掩码模数)
pub const BLOCK: usize = 8;
/// 每轮草稿数
pub const DEPTH: usize = BLOCK - 1;
/// selector 每 slot 候选数(selector_top_k)
pub const TOP_K: usize = 16;
/// selector 码本秩(selector_rank)
pub const RANK: usize = 256;
/// 噪声块 MASK token id(mask_token_id)
pub const MASK_TOKEN_ID: u32 = 248070;

// ============================================================================
// GroupedConv:分组动态深度 2-tap 时间卷积(sglang DFlashGroupedConv 同构)
// ============================================================================

pub struct GroupedConv {
    /// base_kernel [2(side), TAPS, hidden](side-major;tap0 初始 1 恒等)
    base: Weight,
    /// kernel_projection [2·TAPS·G, hidden](动态系数投影)
    proj: Linear,
    hidden: usize,
    groups: usize,
    block_size: usize,
}

impl GroupedConv {
    pub fn new(hidden: usize, block_size: usize) -> Self {
        let groups = hidden / GROUP;
        Self {
            base: Weight::new_typed_f16("base_kernel", vec![2, TAPS, hidden]),
            proj: Linear::new(
                "kernel_projection",
                2 * TAPS * groups,
                hidden,
                QuantPlan::F16,
            ),
            hidden,
            groups,
            block_size,
        }
    }

    /// 一次卷积(side = 0 输入侧 / 1 输出侧;delta = prepare 产出的
    /// 系数投影 [T, 2·TAPS·G],side-major 布局同 sglang reshape 序)。
    fn conv(&self, x: &TensorOps, delta: &TensorOps, side: usize, ctx: &ForwardCtx) -> TensorOps {
        let _ = ctx;
        let t = x.shape()[0];
        let n = t * self.hidden;
        let (gx, _, _) = crate::ops::auto_grid(n);
        TensorOps::of(crate::kernel::kernel_with(
            "owl_dflash_conv_f16",
            (gx, 1, 1),
            (256, 1, 1),
            0,
        ))
        .arg(x)
        .arg(delta)
        .arg(&self.base.decl())
        .arg_i32(side as i32)
        .arg_i32(self.block_size as i32)
        .arg_i32(self.hidden as i32)
        .arg_i32(self.groups as i32)
        .arg_i32(t as i32)
        .with_shape(Dtype::F16, vec![t, self.hidden])
    }

    /// prepare(sglang 同名):coeffs = proj(x) → (conv_in(x), delta)。
    /// 返回 (卷积后输入, 输出侧系数 delta)。
    pub fn prepare(&self, x: &TensorOps, ctx: &ForwardCtx) -> (TensorOps, TensorOps) {
        let delta = self.proj.forward(x, ctx);
        let conv_in = self.conv(x, &delta, 0, ctx);
        (conv_in, delta)
    }

    /// finish(sglang 同名):输出侧系数卷积。
    pub fn finish(&self, y: &TensorOps, delta: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        self.conv(y, delta, 1, ctx)
    }

    pub(crate) fn base(&self) -> &Weight {
        &self.base
    }
    pub(crate) fn proj(&self) -> &Linear {
        &self.proj
    }
}

// ============================================================================
// DfAttn:草稿注意力(普通 qwen3 款 —— 无 q⊕gate 门控,q 独立投影;
// owl Attention 是 Qwen3Next 门控款,草稿不通用,另立此件)。
// 双臂:encode_kv(kv-only 物化,无 q 无注意力)/ propose(8 行双向块)。
// ============================================================================

pub struct DfAttn {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    hq: usize,
    hkv: usize,
    hd: usize,
    hidden: usize,
}

impl DfAttn {
    pub fn new(hq: usize, hkv: usize, hd: usize, hidden: usize, eps: f32, plan: QuantPlan) -> Self {
        Self {
            q_proj: Linear::new("q_proj", hq * hd, hidden, plan),
            k_proj: Linear::new("k_proj", hkv * hd, hidden, plan),
            v_proj: Linear::new("v_proj", hkv * hd, hidden, plan),
            o_proj: Linear::new("o_proj", hidden, hq * hd, plan),
            q_norm: RmsNorm::new_add_one("q_norm", hd, eps),
            k_norm: RmsNorm::new_add_one("k_norm", hd, eps),
            hq,
            hkv,
            hd,
            hidden,
        }
    }

    /// classic 池几何(页宽 / interleave 因子;叶子 shape [nb, hkv, hd/x, page, x])
    fn pool_geom(kv: &KvBuffers) -> (usize, usize) {
        let sh = kv.k_cache.shape();
        (sh[3], sh[4])
    }

    /// norm+rope(ATTN_NORM_ROPE plain 布局:row_stride = 行全长,
    /// head_stride = hd —— 融合核 k 链同款寻址)
    fn norm_rope(
        &self,
        x: &TensorOps,
        norm: &RmsNorm,
        tokens: usize,
        heads: usize,
        pos: &TensorOps,
        rope: &Rope,
        eps: f32,
    ) -> TensorOps {
        let (cos_d, sin_d) = rope.cos_sin_decl();
        let row = heads * self.hd;
        TensorOps::call(crate::ops::ids::ATTN_NORM_ROPE)
            .arg(x)
            .arg(&norm.alpha_decl())
            .arg(&cos_d)
            .arg(&sin_d)
            .arg(pos)
            .arg_f32(eps)
            .arg_usize(row)
            .arg_usize(self.hd)
            .arg_usize(rope.rotary_half())
            .arg_i32(1)
            .aux(&[tokens, heads, self.hd])
            .with_shape(Dtype::F16, vec![tokens, row])
    }

    /// encode 臂(sglang kv_proj_only + apply_k_norm + apply_k_rope 同式;
    /// sglang _append_target_hidden_sequential 逐层同构):memory [T, hidden]
    /// → k/k_norm/rope + v → classic 池槽写入。**无 q 无注意力无 o_proj**。
    /// 返回哑输出根(契约 4;调用方 multi-root eval)。
    pub fn encode_kv(
        &self,
        memory: &TensorOps,
        pos: &TensorOps,
        kv: &KvBuffers,
        rope: &Rope,
        eps: f32,
        ctx: &ForwardCtx,
    ) -> TensorOps {
        let t = memory.shape()[0];
        let k_raw = self.k_proj.forward(memory, ctx);
        let v = self.v_proj.forward(memory, ctx);
        let k = self.norm_rope(&k_raw, &self.k_norm, t, self.hkv, pos, rope, eps);
        let (page, x) = Self::pool_geom(kv);
        let row_kv = self.hkv * self.hd;
        TensorOps::call(crate::ops::ids::ATTN_K0_WRITE)
            .aux(&[t])
            .arg(&k)
            .arg(&v)
            .arg(&kv.k_cache)
            .arg(&kv.v_cache)
            .arg(&kv.slots)
            .arg_i32(row_kv as i32)
            .arg_i32(row_kv as i32)
            .arg_i32(self.hkv as i32)
            .arg_i32(self.hd as i32)
            .arg_i32(page as i32)
            .arg_i32(x as i32)
            .with_shape(Dtype::F16, vec![1])
    }

    /// propose 臂:xs [T, hidden](T = 8 噪声块)→ q/k/v → norm+rope →
    /// 自块写槽(K0;slots = kv.slots 尾段)→ 非因果块注意力(kv =
    /// 池 [0, kv_len) 全可见;ENCODER_ONLY)→ o_proj。
    /// `kv_len` = 前缀 + T(host 标量;图态化时换输入槽,DF-4)。
    #[allow(clippy::too_many_arguments)]
    pub fn propose_attn(
        &self,
        xs: &TensorOps,
        pos: &TensorOps,
        kv: &KvBuffers,
        rope: &Rope,
        eps: f32,
        kv_len: usize,
        ctx: &ForwardCtx,
    ) -> TensorOps {
        let t = xs.shape()[0];
        let q_raw = self.q_proj.forward(xs, ctx);
        let k_raw = self.k_proj.forward(xs, ctx);
        let v = self.v_proj.forward(xs, ctx);
        let q = self.norm_rope(&q_raw, &self.q_norm, t, self.hq, pos, rope, eps);
        let k = self.norm_rope(&k_raw, &self.k_norm, t, self.hkv, pos, rope, eps);
        // 自块写槽(写-后-打分;slots 表 = kv.slots [T])
        let (page, x) = Self::pool_geom(kv);
        let row_q = self.hq * self.hd;
        let row_kv = self.hkv * self.hd;
        // 非因果块注意力(naive NC 核:自块直读零 launch 内依赖;生产 FI
        // kNonCausal 变体挂 DF-4,KV 全走池 + wr 依赖边)
        let n = t * self.hq;
        let (gx, _, _) = crate::ops::auto_grid(n);
        let y = TensorOps::of(crate::kernel::kernel_with(
            "owl_naive_attn_nc_f16",
            (gx, 1, 1),
            (256, 1, 1),
            0,
        ))
        .arg(&q)
        .arg(&k)
        .arg(&v)
        .arg(&kv.k_cache)
        .arg(&kv.v_cache)
        .arg_i32(t as i32)
        .arg_i32((kv_len - t) as i32) // prefix_len = 窗口 - 自块
        .arg_i32(self.hq as i32)
        .arg_i32(self.hkv as i32)
        .arg_i32(self.hd as i32)
        .arg_i32(page as i32)
        .arg_i32(x as i32)
        .with_shape(Dtype::F16, vec![t, row_q]);
        self.o_proj.forward(&y, ctx)
    }

    pub(crate) fn q_proj(&self) -> &Linear {
        &self.q_proj
    }
    pub(crate) fn k_proj(&self) -> &Linear {
        &self.k_proj
    }
    pub(crate) fn v_proj(&self) -> &Linear {
        &self.v_proj
    }
    pub(crate) fn o_proj(&self) -> &Linear {
        &self.o_proj
    }
    pub(crate) fn q_norm(&self) -> &RmsNorm {
        &self.q_norm
    }
    pub(crate) fn k_norm(&self) -> &RmsNorm {
        &self.k_norm
    }
    /// 观测面(accessor 消费位:specs Loadable + 测试):(hq, hkv, hd, hidden)
    pub fn heads(&self) -> (usize, usize, usize, usize) {
        (self.hq, self.hkv, self.hd, self.hidden)
    }
}

// ============================================================================
// DfLayer:草稿解码层(sglang DFlashDecoderLayer 同构;双流残差 +
// conv 双包裹)
// ============================================================================

pub struct DfLayer {
    input_ln: RmsNorm,
    attn_conv: GroupedConv,
    attn: DfAttn,
    post_ln: RmsNorm,
    mlp_conv: GroupedConv,
    mlp: Mlp,
    hidden: usize,
}

impl DfLayer {
    pub fn new(
        hq: usize,
        hkv: usize,
        hd: usize,
        hidden: usize,
        inter: usize,
        eps: f32,
        block_size: usize,
        plan: QuantPlan,
    ) -> Self {
        Self {
            input_ln: RmsNorm::new_add_one("input_layernorm", hidden, eps),
            attn_conv: GroupedConv::new(hidden, block_size),
            attn: DfAttn::new(hq, hkv, hd, hidden, eps, plan),
            post_ln: RmsNorm::new_add_one("post_attention_layernorm", hidden, eps),
            mlp_conv: GroupedConv::new(hidden, block_size),
            mlp: Mlp::new(hidden, inter, plan),
            hidden,
        }
    }

    /// fused add + rmsnorm(LN_FUSED_ADD_RMSNORM;residual 原地 +=
    /// mixed,out = norm·w。decoder.rs 同款装配)
    fn fused_add_norm(
        norm: &RmsNorm,
        mixed: &TensorOps,
        residual: &TensorOps,
    ) -> TensorOps {
        let rows = mixed.shape()[0];
        let n = mixed.shape()[1];
        TensorOps::call(crate::ops::ids::LN_FUSED_ADD_RMSNORM)
            .arg(mixed)
            .arg(residual)
            .arg(&norm.alpha_decl())
            .arg_f32(norm.eps())
            .arg_usize(n)
            .arg_i32(norm.w_off() as i32)
            .aux(&[rows, n])
            .with_shape(mixed.dtype, mixed.shape().to_vec())
    }

    /// 层前向(sglang DFlashDecoderLayer.forward 逐式):双流残差。
    /// 返回 (h_local, residual) —— h_local **未加**残差(下游 fused
    /// add 消费;sglang 返回约定)。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        xs: &TensorOps,
        residual: Option<&TensorOps>,
        pos: &TensorOps,
        kv: &KvBuffers,
        rope: &Rope,
        eps: f32,
        kv_len: usize,
        ctx: &ForwardCtx,
    ) -> (TensorOps, TensorOps) {
        // ① input_ln(首层 residual = xs;后续 fused add)
        let (h, residual) = match residual {
            None => (self.input_ln.forward(xs, ctx), xs.clone()),
            Some(r) => (Self::fused_add_norm(&self.input_ln, xs, r), r.clone()),
        };
        // ② attention(conv 双包裹)
        let (conv_in, delta) = self.attn_conv.prepare(&h, ctx);
        let attn_out = self.attn.propose_attn(&conv_in, pos, kv, rope, eps, kv_len, ctx);
        let attn_out = self.attn_conv.finish(&attn_out, &delta, ctx);
        // ③ post_ln(fused add)
        let n2 = Self::fused_add_norm(&self.post_ln, &attn_out, &residual);
        // ④ mlp(conv 双包裹)
        let (conv_in2, delta2) = self.mlp_conv.prepare(&n2, ctx);
        let h2 = self.mlp.forward(&conv_in2, ctx);
        let h2 = self.mlp_conv.finish(&h2, &delta2, ctx);
        (h2, residual)
    }

    pub(crate) fn input_ln(&self) -> &RmsNorm {
        &self.input_ln
    }
    pub(crate) fn post_ln(&self) -> &RmsNorm {
        &self.post_ln
    }
    pub(crate) fn attn_conv(&self) -> &GroupedConv {
        &self.attn_conv
    }
    pub(crate) fn mlp_conv(&self) -> &GroupedConv {
        &self.mlp_conv
    }
    pub(crate) fn attn(&self) -> &DfAttn {
        &self.attn
    }
    pub(crate) fn mlp(&self) -> &Mlp {
        &self.mlp
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }
}

// ============================================================================
// CandidateSelector:码本选择器草稿头(sglang CandidateSelector 同构)
// ============================================================================

pub struct CandidateSelector {
    /// hidden_projection [rank, hidden]
    proj: Linear,
    /// predecessor_codebook [vocab, rank](A;前驱)
    a_code: Weight,
    /// successor_codebook [vocab, rank](B;后继)
    b_code: Weight,
    vocab: usize,
    rank: usize,
}

impl CandidateSelector {
    /// `rank` = selector_rank(检查点家族 = RANK 256;tiny 测试参数化)。
    pub fn new(hidden: usize, vocab: usize, rank: usize) -> Self {
        Self {
            proj: Linear::new("hidden_projection", rank, hidden, QuantPlan::F16),
            a_code: Weight::new_typed_f16("predecessor_codebook", vec![vocab, rank]),
            b_code: Weight::new_typed_f16("successor_codebook", vec![vocab, rank]),
            vocab,
            rank,
        }
    }

    /// 选 draft(hidden [1+DEPTH, hidden] post-norm;anchor = 噪声块
    /// 行 0 的 token id)。返回 (drafts [DEPTH] f32, scores [D,K,K] f32
    /// 对拍观测面)。两核各单发射:topk 双输出同缓冲 slice 拆分(SSA
    /// 单输出契约 —— 双声明 = 双发射,topk 双跑 2.4GB 扫描不可接受)。
    pub fn select(
        &self,
        hidden: &TensorOps,
        anchor: &TensorOps,
        embed: &crate::layers::embedding::Embedding,
        ctx: &ForwardCtx,
    ) -> (TensorOps, TensorOps) {
        let h = hidden.shape()[0];
        let rows = h - 1; // DEPTH
        let dim = hidden.shape()[1];
        // pred_hidden = hidden 行 1..(连续视图)
        let pred = hidden.slice_view(dim, vec![rows, dim]);
        // ① 候选:target lm_head top-16(sglang compute_candidates 同位;
        // unary 变换 = 恒等,本家族 multiplier=1 无 softcap)。单缓冲
        // [rows, 32]:值半区 [0..16) + 索引半区 [16..32)。
        let logits = embed.lm_head_matmul(&pred);
        let topk = TensorOps::of(crate::kernel::kernel_with(
            "owl_topk16_f16",
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
        ))
        .arg(&logits)
        .arg_i32(self.vocab as i32)
        .with_shape(Dtype::F32, vec![rows, 2 * TOP_K]);
        let cand = topk.slice_view(TOP_K, vec![rows, TOP_K]); // 索引半区
        let vals = topk.slice_view(0, vec![rows, TOP_K]); // 值半区(unary)
        // ② proj = hidden_projection(pred)[rows, RANK]
        let proj = self.proj.forward(&pred, ctx);
        // ③ 格打分 + 贪心 walk(单块融合核;单缓冲 toks + scores)
        let sel = TensorOps::of(crate::kernel::kernel_with(
            "owl_dflash_select_f16",
            (1, 1, 1),
            (256, 1, 1),
            0,
        ))
        .arg(&cand)
        .arg(&vals)
        .arg(&proj)
        .arg(anchor)
        .arg(&self.a_code.decl())
        .arg(&self.b_code.decl())
        .arg_i32(rows as i32)
        .arg_i32(TOP_K as i32)
        .arg_i32(self.rank as i32)
        .with_shape(Dtype::F32, vec![rows * (TOP_K * TOP_K + 1)]);
        let drafts = sel.slice_view(0, vec![rows]);
        let scores = sel.slice_view(rows, vec![rows, TOP_K, TOP_K]);
        (drafts, scores)
    }

    pub(crate) fn proj(&self) -> &Linear {
        &self.proj
    }
    pub(crate) fn a_code(&self) -> &Weight {
        &self.a_code
    }
    pub(crate) fn b_code(&self) -> &Weight {
        &self.b_code
    }
}

// ============================================================================
// DFlash2Draft:草稿本体(骨架 + memory 投影 + selector)
// ============================================================================

pub struct DFlash2Draft {
    /// fc(声明 = 转置装载:源 [hidden, n·hidden] → 存 [n·hidden, hidden]
    /// = W^T 行主序;W^T 行 i·h..(i+1)·h 连续 = W 列块 i → 逐 tap
    /// slice_view 直吃,Σ n GEMM 免 concat 免虚拟键)
    fc: Weight,
    hidden_norm: RmsNorm,
    layers: Vec<DfLayer>,
    norm: RmsNorm,
    selector: CandidateSelector,
    hidden: usize,
    eps: f32,
    vocab: usize,
}

impl DFlash2Draft {
    /// 几何 = 检查点 config(5 层 32H/8KV/128;hidden 5120;inter 17408;
    /// eps 1e-6;vocab 248320)。plan 恒 F16(检查点全 BF16,装载源
    /// bf16→f16 归一;W4A16 版 = 二期)。
    pub fn new(
        hidden: usize,
        inter: usize,
        hq: usize,
        hkv: usize,
        hd: usize,
        n_layers: usize,
        eps: f32,
        vocab: usize,
    ) -> Self {
        Self {
            fc: Weight::new_transposed("fc", hidden, n_layers * hidden),
            hidden_norm: RmsNorm::new_add_one("hidden_norm", hidden, eps),
            layers: (0..n_layers)
                .map(|_| DfLayer::new(hq, hkv, hd, hidden, inter, eps, BLOCK, QuantPlan::F16))
                .collect(),
            norm: RmsNorm::new_add_one("norm", hidden, eps),
            selector: CandidateSelector::new(hidden, vocab, RANK),
            hidden,
            eps,
            vocab,
        }
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// memory 投影(memory = hidden_norm(fc(⊕taps));encode/propose
    /// 共用面)。taps = n_layers 份 [T, hidden](target 层输出残差流,
    /// sglang capture residual 语义;顺序 = target_layer_ids 升序)。
    pub fn project_memory(&self, taps: &[TensorOps], ctx: &ForwardCtx) -> TensorOps {
        assert!(!taps.is_empty(), "project_memory: taps 空");
        let n = self.layers.len();
        assert_eq!(taps.len(), n, "taps 数 = 草稿层数");
        // wt = W^T [n·hidden, hidden](装载期已转置;行块 i = W 列块 i)
        let wt = self.fc.decl().tag("dflash.fc_t");
        let mut acc = taps[0].matmul(&wt.slice_view(0, vec![self.hidden, self.hidden]));
        for (i, tap) in taps.iter().enumerate().skip(1) {
            let wit = wt.slice_view(i * self.hidden * self.hidden, vec![self.hidden, self.hidden]);
            acc = acc.add(&tap.matmul(&wit));
        }
        self.hidden_norm.forward(&acc, ctx)
    }

    /// encode 臂(草稿 KV 物化):memory 逐层 kv-only 写(sglang
    /// _append_target_hidden 逐层同构)。返回 5 哑根(multi-root eval)。
    pub fn encode_kv(
        &self,
        memory: &TensorOps,
        pos: &TensorOps,
        kvs: &[KvBuffers],
        rope: &Rope,
        ctx: &ForwardCtx,
    ) -> Vec<TensorOps> {
        assert_eq!(kvs.len(), self.layers.len(), "草稿 KV 层数");
        self.layers
            .iter()
            .zip(kvs)
            .map(|(l, kv)| l.attn.encode_kv(memory, pos, kv, rope, self.eps, ctx))
            .collect()
    }

    /// propose 块(8 行噪声块 backbone + selector;全设备零 host 往返)。
    /// tokens = [anchor, MASK×7](f32 [BLOCK]);pos = 位置 [BLOCK];
    /// kvs[i].slots = 自块槽表 [BLOCK](前缀尾段 + 8 scratch 槽);
    /// `kv_len` = 前缀 + BLOCK(全可见窗口)。返回 (hidden [BLOCK, hidden]
    /// post-norm, drafts [DEPTH] f32, scores 对拍观测面)。
    #[allow(clippy::too_many_arguments)]
    pub fn propose_block(
        &self,
        tokens: &TensorOps,
        pos: &TensorOps,
        anchor: &TensorOps,
        kvs: &[KvBuffers],
        rope: &Rope,
        embed: &crate::layers::embedding::Embedding,
        kv_len: usize,
        ctx: &ForwardCtx,
    ) -> (TensorOps, TensorOps, TensorOps) {
        assert_eq!(kvs.len(), self.layers.len(), "草稿 KV 层数");
        let mut x = embed.embed(tokens, BLOCK);
        let mut residual: Option<TensorOps> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let (nx, res) = layer.forward(
                &x,
                residual.as_ref(),
                pos,
                &kvs[i],
                rope,
                self.eps,
                kv_len,
                ctx,
            );
            x = nx;
            residual = Some(res);
        }
        // final norm(fused add + norm;sglang `self.norm(hidden, residual)`)
        let hidden = match &residual {
            Some(r) => Self::fused_add_norm_layer(&self.norm, &x, r),
            None => self.norm.forward(&x, ctx),
        };
        let (drafts, scores) = self.selector.select(&hidden, anchor, embed, ctx);
        (hidden, drafts, scores)
    }

    /// fused add + norm(DfLayer 同款;独立于层以复用 norm 容器)
    fn fused_add_norm_layer(norm: &RmsNorm, mixed: &TensorOps, residual: &TensorOps) -> TensorOps {
        DfLayer::fused_add_norm(norm, mixed, residual)
    }

    pub(crate) fn fc(&self) -> &Weight {
        &self.fc
    }
    pub(crate) fn hidden_norm(&self) -> &RmsNorm {
        &self.hidden_norm
    }
    pub(crate) fn norm_head(&self) -> &RmsNorm {
        &self.norm
    }
    pub(crate) fn layers(&self) -> &[DfLayer] {
        &self.layers
    }
    pub(crate) fn selector(&self) -> &CandidateSelector {
        &self.selector
    }
}

// ---- 组件级装载(本地键;检查点键映射住 specs 家规:qwen35.rs impl Loadable)----
impl crate::module::Loadable for GroupedConv {
    fn layout(&self, ctx: &crate::module::LoaderCtx) -> crate::module::LoaderOps {
        self.base.layout(ctx).chain(self.proj.layout(ctx))
    }
}

impl crate::module::Loadable for CandidateSelector {
    fn layout(&self, ctx: &crate::module::LoaderCtx) -> crate::module::LoaderOps {
        self.proj.layout(ctx).chain(self.a_code.layout(ctx)).chain(self.b_code.layout(ctx))
    }
}

impl crate::module::Loadable for DfAttn {
    fn layout(&self, ctx: &crate::module::LoaderCtx) -> crate::module::LoaderOps {
        self.q_proj
            .layout(ctx)
            .chain(self.k_proj.layout(ctx))
            .chain(self.v_proj.layout(ctx))
            .chain(self.o_proj.layout(ctx))
            .chain(self.q_norm.layout(ctx))
            .chain(self.k_norm.layout(ctx))
    }
}

// ============================================================================
// 测试(E5-DF1 验收:conv / selector / NC-attn 往返 / fc 拆分 四金标;
// GPU 门控 OWL_TEST_DEVICE)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{LoaderCtx, Loadable};
    use crate::testkit::{assert_close, f32b, gpu_client, gpu_enabled, harvest, harvest_f16};
    use std::collections::HashMap;

    fn gen(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| half::f16::from_f32(((i as f32 + seed) * 0.23).sin() * 0.7).to_f32())
            .collect()
    }
    fn f16b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
    }

    /// 金标 1:conv kernel vs host f32 参考(sglang _grouped_conv 逐式;
    /// prepare/finish 双侧,T=8 位置掩码语义)。
    #[tokio::test]
    async fn gpu_dflash_conv_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (t, hidden, block) = (8usize, 64usize, 8usize);
        let groups = hidden / GROUP;
        let conv = GroupedConv::new(hidden, block);
        let mut src: HashMap<String, Vec<f32>> = HashMap::new();
        // base [2, 2, hidden](tap0 ≈ 1 + 小扰动;tap1 小值)
        let base: Vec<f32> = (0..2 * TAPS * hidden)
            .map(|i| {
                let (s, tap, d) = (i / (TAPS * hidden), (i / hidden) % TAPS, i % hidden);
                if tap == 0 { 1.0 + (d as f32 * 0.01).sin() * 0.1 } else { ((i % 7) as f32 - 3.0) * 0.05 }
            })
            .collect();
        src.insert("base_kernel".into(), base.clone());
        src.insert("kernel_projection".into(), gen(2 * TAPS * groups * hidden, 1.0));

        let mut gpu = gpu_client().await;
        crate::interpreters::eval_load(&conv, &mut gpu, &src, &LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false })
            .await
            .expect("conv 装载");

        // 输入 x [8, 64](f16)→ delta = proj(x) 亦 host 复算
        let x = gen(t * hidden, 2.0);
        let w_proj = gen(2 * TAPS * groups * hidden, 1.0);
        let _ = &w_proj;
        let x_t = TensorOps::from_host(Dtype::F16, vec![t, hidden], &f16b(&x));
        let ctx = ForwardCtx::minimal(t);
        let (conv_in, delta) = conv.prepare(&x_t, &ctx);
        // 分相①:delta = proj(x) 先对拍(定位投影 vs 核)
        let got_delta = harvest_f16(&mut gpu, &delta).await;
        let y_in = harvest_f16(&mut gpu, &conv_in).await;

        // host 参考:proj(x) = x @ W^T([out,in] 行主序,W^T 消费)
        let x32: Vec<f32> = x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        let delta_host = {
            // Linear 声明 [out, in] = [2·TAPS·G, hidden];matmul_nt 语义 y = x·W^T
            let mut dy = vec![0f32; t * 2 * TAPS * groups];
            for r in 0..t {
                for o in 0..2 * TAPS * groups {
                    let mut acc = 0f32;
                    for k in 0..hidden {
                        acc += x32[r * hidden + k] * half::f16::from_f32(w_proj[o * hidden + k]).to_f32();
                    }
                    dy[r * 2 * TAPS * groups + o] = acc;
                }
            }
            dy
        };
        // conv 参考式(sglang):coeff[t,side,tap,g,s] = base + delta
        let conv_ref = |side: usize, xin: &[f32], delta: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; t * hidden];
            for tt in 0..t {
                for d in 0..hidden {
                    let g = d / GROUP;
                    let c0 = base[(side * TAPS) * hidden + d]
                        + half::f16::from_f32(delta[(tt * 2 + side) * 2 * groups + g]).to_f32();
                    let c1 = base[(side * TAPS + 1) * hidden + d]
                        + half::f16::from_f32(delta[(tt * 2 + side) * 2 * groups + groups + g]).to_f32();
                    let x0 = xin[tt * hidden + d];
                    let x1 = if tt >= 1 { xin[(tt - 1) * hidden + d] } else { 0.0 };
                    let m1 = if tt % block >= 1 { 1.0 } else { 0.0 };
                    out[tt * hidden + d] = c0 * x0 + c1 * x1 * m1;
                }
            }
            out
        };
        assert_close(&got_delta, &delta_host, 2e-2, "delta = proj(x)");
        assert_close(&y_in, &conv_ref(0, &x32, &delta_host), 2e-2, "conv prepare(input 侧)");

        // finish(输出侧,吃 prepare 的 delta;输入 = conv_in)
        let y_out = harvest_f16(&mut gpu, &conv.finish(&conv_in, &delta, &ctx)).await;
        assert_close(&y_out, &conv_ref(1, &y_in, &delta_host), 3e-1, "conv finish(output 侧)");
        gpu.close().await.expect("关机");
    }

    /// 金标 2:selector(lattice 打分 + 贪心 walk)vs host 参考。
    /// tiny 形状:vocab 97,rank 16(核参数化;K=16 烘焙 ≤ vocab 合法)。
    #[tokio::test]
    async fn gpu_dflash_selector_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (vocab, rank, rows) = (97usize, 16usize, DEPTH);
        let sel = CandidateSelector::new(rank, vocab, rank); // hidden = rank(pred 维)
        let mut src: HashMap<String, Vec<f32>> = HashMap::new();
        let w_proj = gen(rank * rank, 3.0);
        let a_tab = gen(vocab * rank, 4.0);
        let b_tab = gen(vocab * rank, 5.0);
        src.insert("hidden_projection".into(), w_proj.clone());
        src.insert("predecessor_codebook".into(), a_tab.clone());
        src.insert("successor_codebook".into(), b_tab.clone());

        let mut gpu = gpu_client().await;
        crate::interpreters::eval_load(&sel, &mut gpu, &src, &LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false })
            .await
            .expect("selector 装载");

        // hidden [1+rows, rank] 随机;anchor token
        let hidden = gen((rows + 1) * rank, 6.0);
        let anchor = 3.0f32;
        let hidden_t = TensorOps::from_host(Dtype::F16, vec![rows + 1, rank], &f16b(&hidden));
        let anchor_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[anchor]));

        // 共享 embed(lm_head 直接以张量喂 topk —— 不走 Embedding 装载,
        // logits 直构:pred @ lm_head^T)
        let lm_head = gen(vocab * rank, 7.0);
        let pred: Vec<f32> = hidden[rank..].to_vec();
        let mut logits = vec![0f32; rows * vocab];
        for r in 0..rows {
            for v in 0..vocab {
                let mut acc = 0f32;
                for k in 0..rank {
                    acc += half::f16::from_f32(pred[r * rank + k]).to_f32()
                        * half::f16::from_f32(lm_head[v * rank + k]).to_f32();
                }
                // 与 device 同格:logits 先落 f16 网格再排序(f32→f16 张量
                // 送核;host 排序若用全精度,舍入创造的平局序会错位)
                logits[r * vocab + v] = half::f16::from_f32(acc).to_f32();
            }
        }
        // host top-16((值降,索引升)全序)
        let host_topk = |row: &[f32]| -> (Vec<f32>, Vec<usize>) {
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.sort_by(|&a, &b| {
                row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b))
            });
            idx.truncate(TOP_K);
            (idx.iter().map(|&i| row[i]).collect(), idx)
        };
        let mut cand_h = vec![0f32; rows * TOP_K];
        let mut unary_h = vec![0f32; rows * TOP_K];
        for r in 0..rows {
            let (vals, ids) = host_topk(&logits[r * vocab..(r + 1) * vocab]);
            for (k, (&v, &i)) in vals.iter().zip(ids.iter()).enumerate() {
                unary_h[r * TOP_K + k] = v;
                cand_h[r * TOP_K + k] = i as f32;
            }
        }
        // device 侧:同一 logits 从 host 送(嵌入路径由 wiring 测试另证)
        let logits_t = TensorOps::from_host(Dtype::F16, vec![rows, vocab], &f16b(&logits));
        let topk = TensorOps::of(crate::kernel::kernel_with(
            "owl_topk16_f16", (rows as u32, 1, 1), (256, 1, 1), 0,
        ))
        .arg(&logits_t)
        .arg_i32(vocab as i32)
        .with_shape(Dtype::F32, vec![rows, 2 * TOP_K]);
        let topk_full = harvest(&mut gpu, &topk).await; // slice dtoh = 整父块,host 切分
        // 布局 [rows, 32]:行内 [0..16) 值 + [16..32) 索引
        let mut got_cand = Vec::with_capacity(rows * TOP_K);
        let mut got_vals = Vec::with_capacity(rows * TOP_K);
        for r in 0..rows {
            got_vals.extend_from_slice(&topk_full[r * 2 * TOP_K..r * 2 * TOP_K + TOP_K]);
            got_cand.extend_from_slice(&topk_full[r * 2 * TOP_K + TOP_K..(r + 1) * 2 * TOP_K]);
        }
        assert_close(&got_cand, &cand_h, 1e-3, "topk16 索引");
        assert_close(&got_vals, &unary_h, 1e-2, "topk16 值");

        // lattice + walk(host 参考)
        let proj: Vec<f32> = {
            let mut pj = vec![0f32; rows * rank];
            for r in 0..rows {
                for o in 0..rank {
                    let mut acc = 0f32;
                    for k in 0..rank {
                        acc += half::f16::from_f32(pred[r * rank + k]).to_f32()
                            * half::f16::from_f32(w_proj[o * rank + k]).to_f32();
                    }
                    pj[r * rank + o] = acc;
                }
            }
            pj
        };
        let mut scores_h = vec![0f32; rows * TOP_K * TOP_K];
        for e in 0..rows {
            for p in 0..TOP_K {
                let pred_id = if e == 0 { anchor as usize } else { cand_h[(e - 1) * TOP_K + p] as usize };
                for c in 0..TOP_K {
                    let cid = cand_h[e * TOP_K + c] as usize;
                    let mut dot = 0f32;
                    for r in 0..rank {
                        dot += half::f16::from_f32(a_tab[pred_id * rank + r]).to_f32()
                            * proj[e * rank + r]
                            * half::f16::from_f32(b_tab[cid * rank + r]).to_f32();
                    }
                    scores_h[(e * TOP_K + p) * TOP_K + c] = unary_h[e * TOP_K + c] + dot;
                }
            }
        }
        // host walk
        let mut idx = 0usize;
        let mut toks_h = vec![0f32; rows];
        for e in 0..rows {
            let row = &scores_h[(e * TOP_K + idx) * TOP_K..(e * TOP_K + idx + 1) * TOP_K];
            let mut best = row[0]; let mut bi = 0;
            for (c, &v) in row.iter().enumerate().skip(1) {
                if v > best { best = v; bi = c; }
            }
            idx = bi;
            toks_h[e] = cand_h[e * TOP_K + idx];
        }

        // device select(直接喂 host 构的 cand/unary/proj)
        let cand_t = TensorOps::from_host(Dtype::F32, vec![rows, TOP_K], &f32b(&cand_h));
        let vals_t = TensorOps::from_host(Dtype::F32, vec![rows, TOP_K], &f32b(&unary_h));
        let proj_t = TensorOps::from_host(Dtype::F16, vec![rows, rank], &f16b(&proj));
        let sel_out = TensorOps::of(crate::kernel::kernel_with(
            "owl_dflash_select_f16", (1, 1, 1), (256, 1, 1), 0,
        ))
        .arg(&cand_t)
        .arg(&vals_t)
        .arg(&proj_t)
        .arg(&anchor_t)
        .arg(&sel.a_code().decl())
        .arg(&sel.b_code().decl())
        .arg_i32(rows as i32)
        .arg_i32(TOP_K as i32)
        .arg_i32(rank as i32)
        .with_shape(Dtype::F32, vec![rows * (TOP_K * TOP_K + 1)]);
        let sel_full = harvest(&mut gpu, &sel_out).await; // 同上,host 切分
        let got_toks = sel_full[..rows].to_vec();
        let got_scores = sel_full[rows..].to_vec();
        assert_close(&got_scores, &scores_h, 5e-2, "lattice scores");
        assert_close(&got_toks, &toks_h, 1e-3, "greedy walk toks");
        gpu.close().await.expect("关机");
    }

    /// 金标 3:fc 拆分消费面(reshape 重释 + slice_view)vs host 参考
    /// (Σ tap_i @ W_i^T;W 行主序 [hidden, fan] 列块)。
    #[tokio::test]
    async fn gpu_dflash_fc_split_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (hidden, n) = (32usize, 4usize);
        let fan = n * hidden;
        let draft = DFlash2Draft::new(hidden, hidden, 2, 1, 16, n, 1e-6, 23);
        let mut src: HashMap<String, Vec<f32>> = HashMap::new();
        let w_fc: Vec<f32> = (0..hidden * fan)
            .map(|i| half::f16::from_f32(((i as f32) * 0.11).sin() * 0.5).to_f32())
            .collect();
        src.insert("fc".into(), w_fc.clone());
        let hn = gen(hidden, 9.0);
        src.insert("hidden_norm".into(), hn.clone());

        let mut gpu = gpu_client().await;
        // 只装载 fc + hidden_norm(单 Weight 容器直 eval_load)
        struct FcOnly<'a>(&'a DFlash2Draft);
        impl Loadable for FcOnly<'_> {
            fn layout(&self, _ctx: &LoaderCtx) -> crate::module::LoaderOps {
                self.0.fc().layout(_ctx).chain(self.0.hidden_norm().layout(_ctx))
            }
        }
        crate::interpreters::eval_load(&FcOnly(&draft), &mut gpu, &src, &LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false })
            .await
            .expect("fc 装载");

        // taps 4 × [2, hidden]
        let taps: Vec<Vec<f32>> = (0..n).map(|i| gen(2 * hidden, 10.0 + i as f32)).collect();
        let taps_t: Vec<TensorOps> = taps
            .iter()
            .map(|tp| TensorOps::from_host(Dtype::F16, vec![2, hidden], &f16b(tp)))
            .collect();
        let ctx = ForwardCtx::minimal(2);
        let mem = draft.project_memory(&taps_t, &ctx);
        let got = harvest_f16(&mut gpu, &mem).await;

        // host:y = Σ tap_i @ W_i^T(W [hidden, fan] 行主序,列块 i);
        // 再 hidden_norm(×(1+w) rmsnorm)
        let t32 = |v: &[f32]| -> Vec<f32> {
            v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
        };
        let mut acc = vec![0f32; 2 * hidden];
        for (i, tp) in taps.iter().enumerate() {
            let tp32 = t32(tp);
            for r in 0..2 {
                for o in 0..hidden {
                    let mut s = 0f32;
                    for k in 0..hidden {
                        s += tp32[r * hidden + k]
                            * half::f16::from_f32(w_fc[o * fan + i * hidden + k]).to_f32();
                    }
                    acc[r * hidden + o] += s;
                }
            }
        }
        let hn32 = t32(&hn);
        for r in 0..2 {
            let row = &acc[r * hidden..(r + 1) * hidden];
            let ms = row.iter().map(|&v| v * v).sum::<f32>() / hidden as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for c in 0..hidden {
                acc[r * hidden + c] *= inv * (1.0 + hn32[c]);
            }
        }
        assert_close(&got, &acc, 2e-2, "fc 列块拆分 + hidden_norm");
        gpu.close().await.expect("关机");
    }

    /// 金标 4:草稿 KV 写-读往返(K0 write classic 布局 → NC attention
    /// 读)vs host 参考(读回 classic 池解包重算注意力)—— 布局一致性
    /// 与 norm_rope plain 寻址一锤定音。
    #[tokio::test]
    async fn gpu_dflash_nc_attn_roundtrip() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (hq, hkv, hd, hidden) = (4usize, 2usize, 16usize, 32usize);
        let eps = 1e-6f32;
        let attn = DfAttn::new(hq, hkv, hd, hidden, eps, QuantPlan::F16);
        let mut src: HashMap<String, Vec<f32>> = HashMap::new();
        src.insert("k_proj".into(), gen(hkv * hd * hidden, 21.0));
        src.insert("v_proj".into(), gen(hkv * hd * hidden, 22.0));
        src.insert("q_proj".into(), gen(hq * hd * hidden, 23.0));
        src.insert("o_proj".into(), gen(hidden * hq * hd, 24.0));
        src.insert("k_norm".into(), gen(hd, 25.0));
        src.insert("q_norm".into(), gen(hd, 26.0));

        let mut gpu = gpu_client().await;
        crate::interpreters::eval_load(&attn, &mut gpu, &src, &LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false })
            .await
            .expect("attn 装载");

        // 池:page 8 / x 8,nb 2(16 槽);前缀 3 行 + 自块 2 行,kv_len 5
        let (page, x, nb) = (8usize, 8usize, 2usize);
        let kcv = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd / x, page, x]).step(), &mut gpu)
            .await.expect("kc");
        let vcv = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd, page]).step(), &mut gpu)
            .await.expect("vc");
        let bt = TensorOps::from_host(Dtype::F32, vec![1, nb], &f32b(&(0..nb).map(|i| i as f32).collect::<Vec<_>>()));
        let slots_h = vec![0.0f32, 1.0, 2.0, 3.0, 4.0]; // 前缀 0..2 + 自块 3..4(连续;NC 核读 [0, kv_len) 直排槽)
        let kv = KvBuffers {
            k_cache: TensorOps::of_block(kcv.id, Dtype::F16, vec![nb, hkv, hd / x, page, x]),
            v_cache: TensorOps::of_block(vcv.id, Dtype::F16, vec![nb, hkv, hd, page]),
            slots: TensorOps::from_host(Dtype::F32, vec![slots_h.len()], &f32b(&slots_h)),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[5.0])),
            block_tables: bt,
        };
        let rope = Rope::new(64, hd, hd, 10_000.0).expect("rope");
        crate::interpreters::eval_load(&rope, &mut gpu, &rope.tables(), &LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false })
            .await
            .expect("rope 表");

        // 输入 5 行:前 3 行 = 前缀 memory(encode 臂写),后 2 行 = 自块
        let xs = gen(5 * hidden, 27.0);
        let xs_t = TensorOps::from_host(Dtype::F16, vec![5, hidden], &f16b(&xs));
        let pos_h = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let pos_t = TensorOps::from_host(Dtype::F32, vec![5], &f32b(&pos_h));
        let ctx = ForwardCtx::minimal(5);

        // encode 臂:前 3 行写前缀(xs 前 3 行窄视图)
        let mem3 = xs_t.slice_view(0, vec![3, hidden]);
        let pos3 = pos_t.slice_view(0, vec![3]);
        let kv3 = KvBuffers {
            k_cache: kv.k_cache.clone(),
            v_cache: kv.v_cache.clone(),
            slots: kv.slots.slice_view(0, vec![3]),
            kv_lens: kv.kv_lens.clone(),
            block_tables: kv.block_tables.clone(),
        };
        let dummy = attn.encode_kv(&mem3, &pos3, &kv3, &rope, eps, &ctx);
        let _ = harvest_f16(&mut gpu, &dummy).await; // 触发写

        // propose 臂:后 2 行(xs 后 2 行)+ 全可见窗口 5
        let self2 = xs_t.slice_view(3 * hidden, vec![2, hidden]);
        // pos 直构(非零偏移 slice 作标量表参数会丢偏移 —— 引擎侧恒
        // from_host 不受影响;此处为测试侧坑记录)
        let pos2 = TensorOps::from_host(Dtype::F32, vec![2], &f32b(&[3.0, 4.0]));
        let kv2 = KvBuffers {
            k_cache: kv.k_cache.clone(),
            v_cache: kv.v_cache.clone(),
            slots: kv.slots.slice_view(3, vec![2]),
            kv_lens: kv.kv_lens.clone(),
            block_tables: kv.block_tables.clone(),
        };
        // 手动复算 propose_attn 内部(拆黑盒;pre-o_proj 与 o_proj 双锚)
        let q_raw_self2 = attn.q_proj().forward(&self2, &ctx);
        let k_raw_self2 = attn.k_proj().forward(&self2, &ctx);
        let v_self2 = attn.v_proj().forward(&self2, &ctx);
        let q_self2 = {
            let (cos_d, sin_d) = rope.cos_sin_decl();
            TensorOps::call(crate::ops::ids::ATTN_NORM_ROPE)
                .arg(&q_raw_self2)
                .arg(&attn.q_norm().alpha_decl())
                .arg(&cos_d)
                .arg(&sin_d)
                .arg(&pos2)
                .arg_f32(eps)
                .arg_usize(hq * hd)
                .arg_usize(hd)
                .arg_usize(rope.rotary_half())
                .arg_i32(1)
                .aux(&[2, hq, hd])
                .with_shape(Dtype::F16, vec![2, hq * hd])
        };
        let k_self2 = {
            let (cos_d, sin_d) = rope.cos_sin_decl();
            TensorOps::call(crate::ops::ids::ATTN_NORM_ROPE)
                .arg(&k_raw_self2)
                .arg(&attn.k_norm().alpha_decl())
                .arg(&cos_d)
                .arg(&sin_d)
                .arg(&pos2)
                .arg_f32(eps)
                .arg_usize(hkv * hd)
                .arg_usize(hd)
                .arg_usize(rope.rotary_half())
                .arg_i32(1)
                .aux(&[2, hkv, hd])
                .with_shape(Dtype::F16, vec![2, hkv * hd])
        };
        let n2 = 2 * hq;
        let (gx2, _, _) = crate::ops::auto_grid(n2);
        let y_nc = TensorOps::of(crate::kernel::kernel_with(
            "owl_naive_attn_nc_f16", (gx2, 1, 1), (256, 1, 1), 0,
        ))
        .arg(&q_self2)
        .arg(&k_self2)
        .arg(&v_self2)
        .arg(&kv.k_cache)
        .arg(&kv.v_cache)
        .arg_i32(2)
        .arg_i32(3) // prefix_len
        .arg_i32(hq as i32)
        .arg_i32(hkv as i32)
        .arg_i32(hd as i32)
        .arg_i32(page as i32)
        .arg_i32(x as i32)
        .with_shape(Dtype::F16, vec![2, hq * hd]);
        let got_nc = harvest_f16(&mut gpu, &y_nc).await;
        let y = attn.o_proj().forward(&y_nc, &ctx);
        let got = harvest_f16(&mut gpu, &y).await;



        // host 参考:重放同一套数学(f32;k/v 从 host 公式直算,不读池
        // —— 写入正确性由 K0 语义另证,此处锚 NC 读 + norm_rope 布局)
        let f = |v: &[f32]| -> Vec<f32> {
            v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
        };
        let xs32 = f(&xs);
        let w = |k: &str, rows: usize, cols: usize| -> Vec<Vec<f32>> {
            (0..rows).map(|r| {
                (0..cols).map(|c| half::f16::from_f32(src[k][r * cols + c]).to_f32()).collect()
            }).collect()
        };
        let wk = w("k_proj", hkv * hd, hidden);
        let wv = w("v_proj", hkv * hd, hidden);
        let wq = w("q_proj", hq * hd, hidden);
        let wo = w("o_proj", hidden, hq * hd);
        let wkn = f(&src["k_norm"]);
        let wqn = f(&src["q_norm"]);
        // rope 表(theta 1e4,rotary 全维)
        let half = hd / 2;
        let cos = |p: usize, i: usize| ((p as f32) * 10_000f32.powf(-(2.0 * i as f32) / hd as f32)).cos();
        let sin = |p: usize, i: usize| ((p as f32) * 10_000f32.powf(-(2.0 * i as f32) / hd as f32)).sin();
        let norm_rope = |xin: &[f32], row: usize, heads: usize, w_n: &[f32], pos_t: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; row * heads * hd];
            for r in 0..row {
                for h in 0..heads {
                    let seg = &xin[r * heads * hd + h * hd..r * heads * hd + (h + 1) * hd];
                    let ms = seg.iter().map(|&v| v * v).sum::<f32>() / hd as f32;
                    let inv = 1.0 / (ms + eps).sqrt();
                    let mut nrm = vec![0f32; hd];
                    for d in 0..hd {
                        nrm[d] = seg[d] * inv * (1.0 + w_n[d]);
                    }
                    let p = pos_t[r] as usize;
                    for i in 0..half {
                        let (a, b) = (nrm[i], nrm[i + half]);
                        out[r * heads * hd + h * hd + i] = a * cos(p, i) - b * sin(p, i);
                        out[r * heads * hd + h * hd + i + half] = a * sin(p, i) + b * cos(p, i);
                    }
                }
            }
            out
        };
        let proj = |xin: &[f32], rows_n: usize, in_d: usize, wm: &Vec<Vec<f32>>, out_d: usize| -> Vec<f32> {
            let mut out = vec![0f32; rows_n * out_d];
            for r in 0..rows_n {
                for o in 0..out_d {
                    let mut s = 0f32;
                    for k in 0..in_d {
                        s += xin[r * in_d + k] * wm[o][k];
                    }
                    out[r * out_d + o] = s;
                }
            }
            out
        };
        // 次序 = 核同款:投影 → per-head norm → rope
        let k_all = norm_rope(&proj(&xs32, 5, hidden, &wk, hkv * hd), 5, hkv, &wkn, &pos_h);
        let v_all = proj(&xs32, 5, hidden, &wv, hkv * hd);
        let q_raw_self_h = proj(&xs32[3 * hidden..].to_vec(), 2, hidden, &wq, hq * hd);
        let k_raw_self_h = proj(&xs32, 5, hidden, &wk, hkv * hd)[3 * hkv * hd..].to_vec();
        let q_self = norm_rope(&q_raw_self_h, 2, hq, &wqn, &pos_h[3..]);
        // 分相:池内前缀 k/v 直读对拍(encode 写入 vs host classic 布局)
        // k_cache [nb, hkv, hd/x, page, x] f16 整块收割
        let kc_host = harvest_f16(&mut gpu, &kv.k_cache).await;
        let (w_kn2, w_qn2) = (&wkn, &wqn);
        let _ = (w_kn2, w_qn2);
        {
            // host 前缀 k(投影 + norm + rope)classic 布局解包比对
            let k_all_h = norm_rope(&proj(&xs32, 5, hidden, &wk, hkv * hd), 5, hkv, &wkn, &pos_h);
            let mut kbad = 0;
            for s in 0..3usize {
                let (b, off) = (s / page, s % page);
                for h in 0..hkv {
                    for d in 0..hd {
                        let flat = b * hkv * (hd / x) * page * x
                            + h * (hd / x) * page * x
                            + (d / x) * page * x
                            + off * x
                            + (d % x);
                        let want = k_all_h[s * hkv * hd + h * hd + d];
                        let got = kc_host[flat];
                        if (got - want).abs() > 3e-2 { kbad += 1; }
                    }
                }
            }
            eprintln!("prefix pool k bad = {kbad}/{}", 3 * hkv * hd);
            // V 池同法([nb, hkv, hd, page])
            let vc_host = harvest_f16(&mut gpu, &kv.v_cache).await;
            let v_all_h = proj(&xs32, 5, hidden, &wv, hkv * hd);
            let mut vbad = 0;
            for s in 0..3usize {
                let (b, off) = (s / page, s % page);
                for h in 0..hkv {
                    for d in 0..hd {
                        let flat = b * hkv * hd * page + h * hd * page + d * page + off;
                        let want = v_all_h[s * hkv * hd + h * hd + d];
                        let got = vc_host[flat];
                        if (got - want).abs() > 3e-2 { vbad += 1; }
                    }
                }
            }
            eprintln!("prefix pool v bad = {vbad}/{}", 3 * hkv * hd);
        }

        // NC attention:行 r(t = 3+r)可见 keys = 全部 5 行
        let scale = 1.0 / (hd as f32).sqrt();
        let mut y_ref = vec![0f32; 2 * hq * hd];
        for r in 0..2 {
            for h in 0..hq {
                let kvh = h / (hq / hkv);
                let mut scores = vec![0f32; 5];
                for s in 0..5 {
                    let mut dot = 0f32;
                    for d in 0..hd {
                        dot += q_self[r * hq * hd + h * hd + d] * k_all[s * hkv * hd + kvh * hd + d];
                    }
                    scores[s] = dot * scale;
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = scores.iter().map(|&s| (s - mx).exp()).collect();
                let den: f32 = ex.iter().sum();
                for d in 0..hd {
                    let mut acc = 0f32;
                    for s in 0..5 {
                        acc += (ex[s] / den) * v_all[s * hkv * hd + kvh * hd + d];
                    }
                    y_ref[r * hq * hd + h * hd + d] = acc;
                }
            }
        }
        // o_proj
        let y_proj = proj(&y_ref, 2, hq * hd, &wo, hidden);
        assert_close(&got_nc, &y_ref, 3e-2, "NC attention(pre-o_proj)");
        assert_close(&got, &y_proj, 3e-2, "NC attention 写-读往返 + o_proj");
        gpu.close().await.expect("关机");
    }
}
// scratch:隔离 Linear matmul_nt 行为
