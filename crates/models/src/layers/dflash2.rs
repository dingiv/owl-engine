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

// 铸边界助手(E5-DF3 同日十四):dtype 相同直通;否则单发 cast 核。
// 支持臂 = f16↔bf16(两向;f32 不在草稿路径)。
pub(crate) fn ensure_dt(x: &TensorOps, dt: Dtype) -> TensorOps {
    if x.dtype == dt {
        return x.clone();
    }
    let name = match (x.dtype, dt) {
        (Dtype::F16, Dtype::BF16) => "owl_cast_f16_bf16",
        (Dtype::BF16, Dtype::F16) => "owl_cast_bf16_f16",
        other => panic!("ensure_dt: 无 cast 臂 {other:?}(草稿路径仅 f16/bf16)"),
    };
    let shape = x.shape().to_vec();
    let n: usize = shape.iter().product();
    let (gx, _, _) = crate::ops::auto_grid(n);
    TensorOps::of(crate::kernel::kernel_with(name, (gx, 1, 1), (256, 1, 1), 0))
        .arg(x)
        .arg_i32(n as i32)
        .with_shape(dt, shape)
}

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
    /// 激活/输出 dtype(F16 存量测试 / BF16 生产 sglang 对齐;E5-DF3 同日十四)
    dt: Dtype,
}

impl GroupedConv {
    pub fn new(hidden: usize, block_size: usize, dt: Dtype) -> Self {
        let groups = hidden / GROUP;
        Self {
            base: Weight::new_typed("base_kernel", vec![2, TAPS, hidden], dt),
            proj: Linear::new(
                "kernel_projection",
                2 * TAPS * groups,
                hidden,
                QuantPlan::F16,
            ),
            hidden,
            groups,
            block_size,
            dt,
        }
    }

    /// 一次卷积(side = 0 输入侧 / 1 输出侧;delta = prepare 产出的
    /// 系数投影 [T, 2·TAPS·G],side-major 布局同 sglang reshape 序)。
    fn conv(&self, x: &TensorOps, delta: &TensorOps, side: usize, ctx: &ForwardCtx) -> TensorOps {
        let _ = ctx;
        let t = x.shape()[0];
        let n = t * self.hidden;
        let (gx, _, _) = crate::ops::auto_grid(n);
        let name = match self.dt {
            Dtype::F16 => "owl_dflash_conv_f16",
            Dtype::BF16 => "owl_dflash_conv_bf16",
            other => panic!("owl_dflash_conv 无 {other:?} 变体"),
        };
        TensorOps::of(crate::kernel::kernel_with(name, (gx, 1, 1), (256, 1, 1), 0))
            .arg(x)
            .arg(delta)
            .arg(&self.base.decl())
            .arg_i32(side as i32)
            .arg_i32(self.block_size as i32)
            .arg_i32(self.hidden as i32)
            .arg_i32(self.groups as i32)
            .arg_i32(t as i32)
            .with_shape(self.dt, vec![t, self.hidden])
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
    /// 激活/池 dtype(同日十四;池叶子由引擎侧同 dtype 分配)
    dt: Dtype,
}

impl DfAttn {
    pub fn new(hq: usize, hkv: usize, hd: usize, hidden: usize, eps: f32, plan: QuantPlan, dt: Dtype) -> Self {
        Self {
            q_proj: Linear::new("q_proj", hq * hd, hidden, plan),
            k_proj: Linear::new("k_proj", hkv * hd, hidden, plan),
            v_proj: Linear::new("v_proj", hkv * hd, hidden, plan),
            o_proj: Linear::new("o_proj", hidden, hq * hd, plan),
            // draft q/k-norm = **plain ×w**(sglang dflash.py RMSNorm 无
            // offset,"matching HF Qwen3";曾误用 add_one —— 全 norms 系统性
            // ×(1+w) 毒化注意力温度与 selector 打分,AL=0 根因,2026-10-08 定谳)
            q_norm: RmsNorm::new("q_norm", hd, eps),
            k_norm: RmsNorm::new("k_norm", hd, eps),
            hq,
            hkv,
            hd,
            hidden,
            dt,
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
            .arg_i32(norm.w_off() as i32)
            .aux(&[tokens, heads, self.hd])
            .with_shape(self.dt, vec![tokens, row])
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
            .with_shape(self.dt, vec![1])
    }

    /// propose 臂:xs [T, hidden](T = 8 噪声块)→ q/k/v → norm+rope →
    /// 自块写槽(K0;slots = kv.slots 尾段)→ 非因果块注意力(kv =
    /// 池 [0, kv_len) 全可见;ENCODER_ONLY)→ o_proj。
    /// `kv_len_t` = [1] f32 全窗张量(契约 5;E5-DF4 图化:prefix 曾为
    /// 宿主烘焙标量,每轮 fp 变 → 图化必须运行时读)。
    #[allow(clippy::too_many_arguments)]
    pub fn propose_attn(
        &self,
        xs: &TensorOps,
        pos: &TensorOps,
        kv: &KvBuffers,
        rope: &Rope,
        eps: f32,
        kv_len_t: &TensorOps,
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
        // v2:grid (T, Hq) × block (hd) —— block-per-head flash 式
        let nc_name = match self.dt {
            Dtype::F16 => "owl_naive_attn_nc_f16",
            Dtype::BF16 => "owl_naive_attn_nc_bf16",
            other => panic!("owl_naive_attn_nc 无 {other:?} 变体"),
        };
        let y = TensorOps::of(crate::kernel::kernel_with(
            nc_name,
            (t as u32, self.hq as u32, 1),
            (self.hd as u32, 1, 1),
            0,
        ))
        .arg(&q)
        .arg(&k)
        .arg(&v)
        .arg(&kv.k_cache)
        .arg(&kv.v_cache)
        .arg(kv_len_t)
        .arg_i32(self.hq as i32)
        .arg_i32(self.hkv as i32)
        .arg_i32(self.hd as i32)
        .arg_i32(page as i32)
        .arg_i32(x as i32)
        .with_shape(self.dt, vec![t, row_q]);
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
        dt: Dtype,
    ) -> Self {
        Self {
            input_ln: RmsNorm::new("input_layernorm", hidden, eps),
            attn_conv: GroupedConv::new(hidden, block_size, dt),
            attn: DfAttn::new(hq, hkv, hd, hidden, eps, plan, dt),
            post_ln: RmsNorm::new("post_attention_layernorm", hidden, eps),
            mlp_conv: GroupedConv::new(hidden, block_size, dt),
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
        kv_len_t: &TensorOps,
        ctx: &ForwardCtx,
        mut probe: Option<&mut Vec<TensorOps>>,
    ) -> (TensorOps, TensorOps) {
        // ① input_ln(首层 residual = xs;后续 fused add)
        let (h, residual) = match residual {
            None => (self.input_ln.forward(xs, ctx), xs.clone()),
            Some(r) => (Self::fused_add_norm(&self.input_ln, xs, r), r.clone()),
        };
        // ② attention(conv 双包裹)
        let (conv_in, delta) = self.attn_conv.prepare(&h, ctx);
        if let Some(p) = probe.as_deref_mut() {
            p.push(conv_in.clone().tag("probe.conv_in"));
        }
        let attn_out = self.attn.propose_attn(&conv_in, pos, kv, rope, eps, kv_len_t, ctx);
        if let Some(p) = probe.as_deref_mut() {
            p.push(attn_out.clone().tag("probe.attn_raw"));
        }
        let attn_out = self.attn_conv.finish(&attn_out, &delta, ctx);
        if let Some(p) = probe.as_deref_mut() {
            p.push(attn_out.clone().tag("probe.attn_fin"));
        }
        // ③ post_ln(fused add)
        let n2 = Self::fused_add_norm(&self.post_ln, &attn_out, &residual);
        // ④ mlp(conv 双包裹)
        let (conv_in2, delta2) = self.mlp_conv.prepare(&n2, ctx);
        if let Some(p) = probe.as_deref_mut() {
            p.push(conv_in2.clone().tag("probe.mlp_conv_in"));
        }
        let h2 = self.mlp.forward(&conv_in2, ctx);
        let h2 = self.mlp_conv.finish(&h2, &delta2, ctx);
        if let Some(p) = probe.as_deref_mut() {
            p.push(h2.clone().tag("probe.mlp_out"));
        }
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
    /// proj/码本 dtype(同日十四;BF16 原生检查点直载)
    dt: Dtype,
}

impl CandidateSelector {
    /// `rank` = selector_rank(检查点家族 = RANK 256;tiny 测试参数化)。
    pub fn new(hidden: usize, vocab: usize, rank: usize, dt: Dtype) -> Self {
        Self {
            proj: Linear::new("hidden_projection", rank, hidden, QuantPlan::F16),
            a_code: Weight::new_typed("predecessor_codebook", vec![vocab, rank], dt),
            b_code: Weight::new_typed("successor_codebook", vec![vocab, rank], dt),
            vocab,
            rank,
            dt,
        }
    }

    /// 选 draft(logits 外供版:引擎探针/调用方复用 lm_head 输出)。
    pub fn select_with_logits(
        &self,
        hidden: &TensorOps,
        anchor: &TensorOps,
        logits: &TensorOps,
        ctx: &ForwardCtx,
    ) -> (TensorOps, TensorOps) {
        let h = hidden.shape()[0];
        let rows = h - 1; // DEPTH
        let dim = hidden.shape()[1];
        let pred = hidden.slice_view(dim, vec![rows, dim]); // 行 1..(连续视图)
        let topk = TensorOps::of(crate::kernel::kernel_with(
            "owl_topk16_f16",
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
        ))
        .arg(&logits)
        .arg_i32(self.vocab as i32)
        .with_shape(Dtype::F32, vec![2 * rows * TOP_K]); // 值区 + 索引区(两段连续)
        let cand = topk.slice_view(rows * TOP_K, vec![rows, TOP_K]); // 索引区
        let vals = topk.slice_view(0, vec![rows, TOP_K]); // 值区(unary)
        // ② proj = hidden_projection(pred)[rows, RANK](dtype 随 hidden)
        let proj = self.proj.forward(&pred, ctx);
        // ③ 格打分 + 贪心 walk(单块融合核;单缓冲 toks + scores)
        // BF16 变体:proj/码本 bf16(同日十四);topk/格打分输出 f32 契约不变
        let sel_name = match self.dt {
            Dtype::F16 => "owl_dflash_select_f16",
            Dtype::BF16 => "owl_dflash_select_bf16",
            other => panic!("owl_dflash_select 无 {other:?} 变体"),
        };
        let sel = TensorOps::of(crate::kernel::kernel_with(
            sel_name,
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

/// fc 消费面形态(双源族):
/// - `F16T`:BF16 源,转置装载单权(W^T [n·h, h] 行块 = 列块,逐 tap slice)
/// - `Q`:W4A16 源,五拆 marlin Linear(packed 域列块连续;源侧
///   `fc_{i}.qweight/.scales` 列块派生,w4a16.rs fc_block_bytes)
pub enum DraftFc {
    F16T(Weight),
    Q(Vec<Linear>),
}

pub struct DFlash2Draft {
    fc: DraftFc,
    hidden_norm: RmsNorm,
    layers: Vec<DfLayer>,
    norm: RmsNorm,
    selector: CandidateSelector,
    hidden: usize,
    eps: f32,
    vocab: usize,
    /// 草稿激活 dtype(F16 存量测试面 / BF16 生产,sglang 对齐 ——
    /// 真实权重激活超 f16 范围,E5-DF3 同日十四定谳)
    dt: Dtype,
}

impl DFlash2Draft {
    /// 几何 = 检查点 config(5 层 32H/8KV/128;hidden 5120;inter 17408;
    /// eps 1e-6;vocab 248320)。plan 恒 F16(检查点全 BF16,装载源
    /// bf16→f16 归一;W4A16 版 = 二期)。dt = F16(存量测试面)。
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
        Self::new_with_plan_dt(
            hidden,
            inter,
            hq,
            hkv,
            hd,
            n_layers,
            eps,
            vocab,
            QuantPlan::F16,
            Dtype::F16,
        )
    }

    /// plan 参数化构造(W4A16:attn/mlp/fc 走量化计划,norms/conv 投影
    /// /codebooks 恒 F16 直读 —— 检查点 ignore 清单同构)。dt = F16。
    pub fn new_with_plan(
        hidden: usize,
        inter: usize,
        hq: usize,
        hkv: usize,
        hd: usize,
        n_layers: usize,
        eps: f32,
        vocab: usize,
        plan: QuantPlan,
    ) -> Self {
        Self::new_with_plan_dt(hidden, inter, hq, hkv, hd, n_layers, eps, vocab, plan, Dtype::F16)
    }

    /// 全参数构造(dt 显式;生产 = BF16,sglang 对齐,E5-DF3 同日十四:
    /// 草稿路径 BF16 全程 —— 权重原生 BF16 直载,激活 f16 溢出免疫;
    /// 池叶子由引擎侧同 dtype 分配)
    pub fn new_with_plan_dt(
        hidden: usize,
        inter: usize,
        hq: usize,
        hkv: usize,
        hd: usize,
        n_layers: usize,
        eps: f32,
        vocab: usize,
        plan: QuantPlan,
        dt: Dtype,
    ) -> Self {
        let fc = match plan {
            QuantPlan::W4A16 => DraftFc::Q((0..n_layers)
                .map(|i| {
                    Linear::new(
                        match i {
                            0 => "fc_0",
                            1 => "fc_1",
                            2 => "fc_2",
                            3 => "fc_3",
                            _ => "fc_4",
                        },
                        hidden,
                        hidden,
                        QuantPlan::W4A16,
                    )
                })
                .collect()),
            _ => DraftFc::F16T(Weight::new_transposed("fc", hidden, n_layers * hidden)),
        };
        Self {
            fc,
            hidden_norm: RmsNorm::new("hidden_norm", hidden, eps),
            layers: (0..n_layers)
                .map(|_| DfLayer::new(hq, hkv, hd, hidden, inter, eps, BLOCK, plan, dt))
                .collect(),
            norm: RmsNorm::new("norm", hidden, eps),
            selector: CandidateSelector::new(hidden, vocab, RANK, dt),
            hidden,
            eps,
            vocab,
            dt,
        }
    }

    /// fc 形态观测面(specs Loadable 键映射分派用)
    pub fn is_quant(&self) -> bool {
        matches!(self.fc, DraftFc::Q(_))
    }

    /// 草稿激活 dtype(引擎探针解析/池分配对账用)
    pub fn dtype(&self) -> Dtype {
        self.dt
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// memory 投影(memory = hidden_norm(fc(⊕taps));encode/propose
    /// 共用面)。taps = n_layers 份 [T, hidden](target 层输出残差流,
    /// sglang capture residual 语义;顺序 = target_layer_ids 升序)。
    ///
    /// 双 dtype 臂:
    /// - BF16(生产,sglang 对齐):无预缩放(bf16 指数域同 f32,无溢出);
    ///   taps f16(target 面)经 owl_cast_f16_bf16 铸入;
    /// - F16(存量测试面):保留 2⁻⁸ 预缩放护栏(见下)。
    pub fn project_memory(&self, taps: &[TensorOps], ctx: &ForwardCtx) -> TensorOps {
        assert!(!taps.is_empty(), "project_memory: taps 空");
        let n = self.layers.len();
        assert_eq!(taps.len(), n, "taps 数 = 草稿层数");
        let inputs: Vec<TensorOps> = if self.dt == Dtype::BF16 {
            // bf16 全程:fc 前免预缩放,语义同 sglang(taps 原样铸 bf16)
            taps.iter().map(|tp| ensure_dt(tp, Dtype::BF16)).collect()
        } else {
            // f16 溢出护栏(2026-10-07 引擎定谳):真实 target taps(元素 ~7-50,
            // 深层残差流)经 fc(权重量级 ~0.5)输出可达 ~5×10⁴,f16 加链
            // 47456+16168+… → INF → NaN 塌缩(golden 测试 taps ~0.7 从不触发,
            // 故单测全绿引擎崩)。rms 对行尺度不变 → fc 前预乘 2⁻⁸(f16 纯指
            // 数移位,零舍入损失),norm 后语义严格等价。sglang bf16 无此问题。
            let t = taps[0].shape()[0];
            let scale = TensorOps::from_host(
                taps[0].dtype,
                vec![t, self.hidden],
                &{
                    let mut v = Vec::with_capacity(t * self.hidden * 2);
                    for _ in 0..t * self.hidden {
                        v.extend_from_slice(&half::f16::from_f32(1.0 / 256.0).to_le_bytes());
                    }
                    v
                },
            );
            taps.iter().map(|tp| tp.mul(&scale)).collect()
        };
        let acc = match &self.fc {
            DraftFc::Q(lins) => {
                // 五拆 marlin:fc_i 整块 GEMM 直吃 tap_i(matmul_nt [T,h]×[h,h];
                // BF16 激活自动路由 marlin bf16 核,同日十二)
                let mut acc = lins[0].forward(&inputs[0], ctx);
                for (i, tap) in inputs.iter().enumerate().skip(1) {
                    acc = acc.add(&lins[i].forward(tap, ctx));
                }
                acc
            }
            DraftFc::F16T(w) => {
                // wt = W^T [n·hidden, hidden](装载期已转置;行块 i = W 列块 i)
                let wt = w.decl().tag("dflash.fc_t");
                let mut acc = inputs[0].matmul(&wt.slice_view(0, vec![self.hidden, self.hidden]));
                for (i, tap) in inputs.iter().enumerate().skip(1) {
                    let wit =
                        wt.slice_view(i * self.hidden * self.hidden, vec![self.hidden, self.hidden]);
                    acc = acc.add(&tap.matmul(&wit));
                }
                acc
            }
        };
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
    /// `kv_len_t` = [1] f32 全窗 = 前缀 + BLOCK(图化运行时读,契约 5)。返回 (hidden [BLOCK, hidden]
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
        kv_len_t: &TensorOps,
        ctx: &ForwardCtx,
        mut probe_layers: Option<&mut Vec<TensorOps>>,
    ) -> (TensorOps, TensorOps, TensorOps, TensorOps) {
        assert_eq!(kvs.len(), self.layers.len(), "草稿 KV 层数");
        // cast 边界①:embed 查表(f16 表,target 共享)→ 草稿主干入口(bf16)
        let mut x = ensure_dt(&embed.embed(tokens, BLOCK), self.dt);
        let probe = probe_layers.is_some();
        if probe {
            probe_layers.as_mut().unwrap().push(x.clone().tag("probe.embed"));
        }
        let mut residual: Option<TensorOps> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_probe = if probe {
                probe_layers.as_deref_mut()
            } else {
                None
            };
            let (nx, res) = layer.forward(
                &x,
                residual.as_ref(),
                pos,
                &kvs[i],
                rope,
                self.eps,
                kv_len_t,
                ctx,
                layer_probe,
            );
            x = nx;
            if probe {
                // 层出口和 = h2 + residual(可观测;add 纯节点)
                probe_layers
                    .as_mut()
                    .unwrap()
                    .push(x.add(&res).tag(format!("probe.L{i}")));
            }
            residual = Some(res);
        }
        // 探针:逐层 NaN 定位(OWL_DFLASH_PROBE;引擎收割挂账 —— 声明期
        // 无法收割,改由引擎侧逐层 eval:本计数仅供断点)
        // final norm(fused add + norm;sglang `self.norm(hidden, residual)`)
        let hidden = match &residual {
            Some(r) => Self::fused_add_norm_layer(&self.norm, &x, r),
            None => self.norm.forward(&x, ctx),
        };
        // cast 边界②:hidden(草稿 dtype)→ lm_head 入口 f16(cublas/topk
        // 链保持 f16 不动,sglang 同位 .to(f16) 式;索引量 f32 契约不变)
        let hidden_head = ensure_dt(
            &hidden.slice_view(self.hidden, vec![DEPTH, self.hidden]),
            Dtype::F16,
        );
        let logits = embed.lm_head_matmul(&hidden_head);
        let (drafts, scores) = self.selector.select_with_logits(&hidden, anchor, &logits, ctx);
        (hidden, drafts, scores, logits)
    }

    /// fused add + norm(DfLayer 同款;独立于层以复用 norm 容器)
    fn fused_add_norm_layer(norm: &RmsNorm, mixed: &TensorOps, residual: &TensorOps) -> TensorOps {
        DfLayer::fused_add_norm(norm, mixed, residual)
    }

    pub(crate) fn fc_t(&self) -> &Weight {
        match &self.fc {
            DraftFc::F16T(w) => w,
            _ => unreachable!("fc_t 仅 F16T 形态"),
        }
    }
    pub(crate) fn fc_q(&self) -> &[Linear] {
        match &self.fc {
            DraftFc::Q(lins) => lins,
            _ => unreachable!("fc_q 仅 Q 形态"),
        }
    }

    /// 诊断口:fc_0 单发(E5-DF3 引擎排查;视图 vs 物化输入对拍用)
    pub fn debug_fc0(&self, tap: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        self.fc_q()[0].forward(tap, ctx)
    }

    /// 诊断口:fc_i 单发(E5-DF3 五块对表用)
    pub fn debug_fci(&self, i: usize, tap: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        self.fc_q()[i].forward(tap, ctx)
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
    use crate::testkit::{assert_close, f32b, gpu_client, gpu_enabled, harvest, harvest_bf16, harvest_f16};
    use std::collections::HashMap;

    fn gen(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| half::f16::from_f32(((i as f32 + seed) * 0.23).sin() * 0.7).to_f32())
            .collect()
    }
    fn f16b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
    }
    fn bf16b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| half::bf16::from_f32(*f).to_le_bytes()).collect()
    }
    fn b32g(v: f32) -> f32 {
        half::bf16::from_f32(v).to_f32()
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
        let conv = GroupedConv::new(hidden, block, Dtype::F16);
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
        let sel = CandidateSelector::new(rank, vocab, rank, Dtype::F16); // hidden = rank(pred 维)
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
        // 布局两段连续:值区 [rows*K) + 索引区 [rows*K, 2*rows*K)
        let got_vals = topk_full[..rows * TOP_K].to_vec();
        let got_cand = topk_full[rows * TOP_K..].to_vec();
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
                let fc = match self.0.fc_shape() {
                    crate::specs::qwen35::FcShape::Q => {
                        let mut o = self.0.fc_q()[0].layout(_ctx);
                        for l in &self.0.fc_q()[1..] {
                            o = o.chain(l.layout(_ctx));
                        }
                        o
                    }
                    crate::specs::qwen35::FcShape::F16T => self.0.fc_t().layout(_ctx),
                };
                fc.chain(self.0.hidden_norm().layout(_ctx))
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
        // 再 hidden_norm(plain ×w rmsnorm,6.18 律)。
        // GPU 侧有 f16 溢出护栏:fc 前预乘 2⁻⁸(指数移位零舍入;rms
        // 尺度不变但 eps 不是 —— taps 缩后 ms~e-5 与 eps 同级,host 必须
        // 同步预缩才能逐式对齐,否则 2.6% 级偏差超 atol)。
        let pre = 1.0f32 / 256.0;
        let t32 = |v: &[f32]| -> Vec<f32> {
            v.iter()
                .map(|&x| half::f16::from_f32(x).to_f32() * pre)
                .collect()
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
        // gamma 原样入模(不预缩;t32 的 pre 因子只属于 taps)
        let hn32: Vec<f32> = hn.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect();
        for r in 0..2 {
            let row = &acc[r * hidden..(r + 1) * hidden];
            let ms = row.iter().map(|&v| v * v).sum::<f32>() / hidden as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for c in 0..hidden {
                // 6.18 律:sglang dflash 全系 plain ×w(×(1+w) 是 AL=0 真根因,
                // GPU 侧已改,host 参照同步 —— 本测试曾双方同错自洽)
                acc[r * hidden + c] *= inv * hn32[c];
            }
        }
        assert_close(&got, &acc, 2e-2, "fc 列块拆分 + hidden_norm");
        gpu.close().await.expect("关机");
    }

    /// 金标 5(E5-DF3 排查主案):真实 W4A16 权重 + golden 逐层对拍。
    /// 金标 = tools/dflash2_golden.py(torch f32,sglang 逐式)存
    /// testdata/dflash2_golden.safetensors。同输入(公式 E/LM/taps/ids)、
    /// 同权重(marlin 装载)—— 逐层定位首个发散算子。
    #[tokio::test]
    async fn gpu_dflash2_golden_matches() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let Ok(dir) = std::env::var("OWL_DFLASH2_DIR") else {
            eprintln!("skip: OWL_DFLASH2_DIR 未设");
            return;
        };
        use crate::module::{LoaderCtx, Loadable};
        use crate::testkit::{f32b, gpu_client};
        use std::collections::HashMap;

        let (hidden, inter, hq, hkv, hd, nl) = (5120usize, 17408usize, 32usize, 8usize, 128usize, 5usize);
        let (vocab, topk) = (248320usize, 16usize);
        let mut draft = DFlash2Draft::new_with_plan(hidden, inter, hq, hkv, hd, nl, 1e-6, vocab, QuantPlan::W4A16);

        // 公式 E/LM(f16 网格;与 python 同式)→ Embedding 装载
        let gen = |n: usize, seed: f64| -> Vec<f32> {
            (0..n)
                .map(|i| half::f16::from_f32(((((i as f64 + seed) * 0.23).sin()) as f32) * 0.7).to_f32())
                .collect()
        };
        let e_tab = gen(vocab * hidden, 20.0);
        let lm_tab = gen(vocab * hidden, 21.0);
        let mut esrc: HashMap<String, Vec<f32>> = HashMap::new();
        esrc.insert("weight".into(), e_tab);
        esrc.insert("lm_head.weight".into(), lm_tab.clone());
        let embed = crate::layers::embedding::Embedding::new_untied(vocab, hidden);
        struct EmbLoad<'a>(&'a crate::layers::embedding::Embedding);
        impl Loadable for EmbLoad<'_> {
            fn layout(&self, ctx: &LoaderCtx) -> crate::module::LoaderOps {
                self.0.layout(ctx).chain(self.0.layout_lm_head(ctx).unwrap())
            }
        }
        let lctx = LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false };
        let mut gpu = gpu_client().await;
        crate::interpreters::eval_load(&EmbLoad(&embed), &mut gpu, &esrc, &lctx)
            .await
            .expect("embed 装载");

        // 草稿装载(W4A16)
        let draft = {
            let d = DFlash2Draft::new_with_plan(hidden, inter, hq, hkv, hd, nl, 1e-6, vocab, QuantPlan::W4A16);
            let src = crate::formats::w4a16::W4A16Source::open_dir(std::path::Path::new(&dir)).unwrap();
            crate::interpreters::eval_load(&d, &mut gpu, &src, &lctx).await.expect("draft 装载");
            d
        };

        // 输入(ids/taps;E/LM 同式)
        let ids_v: Vec<f32> = [108820.0f32]
            .iter()
            .chain(std::iter::repeat(&(MASK_TOKEN_ID as f32)).take(7))
            .copied()
            .collect();
        let genf = |n: usize, seed: f64| -> Vec<f32> { gen(n, seed) };
        let taps_t: Vec<TensorOps> = (0..nl)
            .map(|i| {
                let t = genf(20 * hidden, 30.0 + i as f64 * 7.0);
                let tb: Vec<u8> = t.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect();
                TensorOps::from_host(Dtype::F16, vec![20, hidden], &tb)
            })
            .collect();
        let ids_t = TensorOps::from_host(Dtype::F32, vec![BLOCK], &f32b(&ids_v));
        let pos_t = TensorOps::from_host(Dtype::F32, vec![BLOCK], &f32b(&(0..BLOCK).map(|i| (19 + i) as f32).collect::<Vec<_>>()));
        let anchor_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[108820.0]));

        // 池 + encode(memory 前 19 行 @ 槽 0..18)+ rope
        let rope = Rope::new(262_144, hd, hd, 1.0e7).expect("rope");
        crate::interpreters::eval_load(&rope, &mut gpu, &rope.tables(), &lctx).await.expect("rope 表");
        // memory + encode + propose(全在 propose_block 前;用引擎同构装配)
        // memory = project_memory(taps)
        let ctx = ForwardCtx::minimal(20);
        let memory = draft.project_memory(&taps_t, &ctx);

        // 池(零块)与槽表
        let (page, xq) = (32usize, 8usize);
        let kcb = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![2, hkv, hd / xq, page, xq]).step(), &mut gpu)
            .await.expect("kc");
        let vcb = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![2, hkv, hd, page]).step(), &mut gpu)
            .await.expect("vc");
        let bt = TensorOps::from_host(Dtype::F32, vec![1, 2], &f32b(&[0.0, 1.0]));
        let f32b2 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let kvs: Vec<KvBuffers> = (0..nl)
            .map(|_| KvBuffers {
                k_cache: TensorOps::of_block(kcb.id, Dtype::F16, vec![2, hkv, hd / xq, page, xq]),
                v_cache: TensorOps::of_block(vcb.id, Dtype::F16, vec![2, hkv, hd, page]),
                slots: TensorOps::from_host(Dtype::F32, vec![19], &f32b2(&(0..19).map(|i| i as f32).collect::<Vec<_>>())),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![19], &f32b2(&(0..19).map(|i| (i + 1) as f32).collect::<Vec<_>>())),
                block_tables: bt.clone(),
            })
            .collect();
        let pos_enc = TensorOps::from_host(Dtype::F32, vec![19], &f32b2(&(0..19).map(|i| i as f32).collect::<Vec<_>>()));
        let mem19 = memory.slice_view(0, vec![19, hidden]);
        let enc_roots = draft.encode_kv(&mem19, &pos_enc, &kvs, &rope, &ctx);
        {
            // 取证:同一 multi 里挎带 memory 与 fc0 同埣收割(E5-DF3 同日十一:
            // golden encode 内 fc 到底是什么 —— 定谳“双重人格”矛盾)
            let memory_root = mem19.clone();
            let fc0_root = draft.debug_fc0(&taps_t[0], &ctx);
            let mut refs: Vec<&TensorOps> = enc_roots.iter().collect();
            refs.push(&memory_root);
            refs.push(&fc0_root);
            let face = &mut gpu;
            let enc_outs =
                crate::interpreters::eval_ops_multi(&refs, face).await.expect("encode");
            for (tag, o) in [("memory", &enc_outs[enc_roots.len()]), ("fc0", &enc_outs[enc_roots.len() + 1])] {
                let mut mb = vec![0u8; 20 * 5120 * 2];
                gpu.dtoh(o, &mut mb).await.expect("dtoh");
                let mx = mb[..5120 * 2]
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                    .fold(0f32, f32::max);
                let zeros = mb[..5120 * 2]
                    .chunks_exact(2)
                    .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() == 0.0)
                    .count();
                eprintln!("[enc-forensic] {tag} 行0: |max|={mx:.4} 零元={zeros}/5120");
            }
        }

        // propose(probe_roots)
        let mut probe_roots: Vec<TensorOps> = Vec::new();
        let kv_len_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b2(&[(19 + BLOCK) as f32]));
        let (hidden_t, drafts_t, _scores, logits_t) = draft.propose_block(
            &ids_t, &pos_t, &anchor_t, &kvs, &rope, &embed, &kv_len_t, &ctx,
            Some(&mut probe_roots),
        );
        let _ = &_scores;

        // 单遍 multi 求值(2026-10-07 定谳:havest_f16 逐根 eval_ops 每次
        // 新建 memo → 整棵子图重执行 → 原地 fused_add 把残差流反复累加
        // 进共享 embed 缓冲 —— 「L1 起全错 + embed 终态 NaN」均为取证伪
        // 影,核/模型无罪。共享 memo 单遍 = 每节点恰执行一次;收劁从返
        // 回块直读,零重执行)
        let mut roots: Vec<TensorOps> = probe_roots.clone();
        roots.push(hidden_t.clone());
        roots.push(logits_t.clone()); // 全量 logits(视图根会透传底层块)
        roots.push(drafts_t.clone());
        let refs: Vec<&TensorOps> = roots.iter().collect();
        let outs = crate::interpreters::eval_ops_multi(&refs, &mut gpu)
            .await
            .expect("单遍求值");
        let mut harvested: Vec<Vec<f32>> = Vec::with_capacity(outs.len());
        use owl_iface::DeviceClient as _;
        for (out, t) in outs.iter().zip(&roots) {
            // 视图根透传底层块:缓冲按块(Bytes.len × dtype 尺寸)开,解码后
            // 按视图逻辑元素数截断(drafts/logits_head 均为视图根)
            let f32blk = t.dtype == Dtype::F32;
            let mut buf = vec![0u8; out.len * if f32blk { 4 } else { 2 }];
            gpu.dtoh(out, &mut buf).await.expect("dtoh");
            let mut v: Vec<f32> = if f32blk {
                buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                buf.chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect()
            };
            let n: usize = t.shape().iter().product();
            v.truncate(n);
            harvested.push(v);
        }

        // 金标读取(safetensors 裸头)
        let golden = read_safetensors("testdata/dflash2_golden.safetensors");
        let compare = |key: &str, got: &[f32]| {
            let Some(want) = golden.get(key) else {
                eprintln!("[golden] {key}: 金标缺失");
                return;
            };
            let wf: Vec<f32> = want.iter().copied().collect();
            // f16 溢出容差(2026-10-07):真实权量下 L0 起激活 >65504,owl 与
            // python 同样写 inf —— 同号 inf 视为一致(根修 = 权重预缩放,另案)
            let ok = |g: &f32, w: &f32| {
                (*g - *w).abs() <= 6e-2
                    || (g.is_nan() && w.is_nan())
                    || (g.is_infinite() && w.is_infinite() && g.signum() == w.signum())
            };
            let bad = got.iter().zip(&wf).filter(|(g, w)| !ok(g, w)).count();
            eprintln!(
                "[golden] {key}: bad={bad}/{}{}",
                got.len(),
                if bad > 0 {
                    format!(" got[0]={} want[0]={}", got[0], wf[0])
                } else {
                    String::new()
                }
            );
        };
        // 显式配对:probe_roots = [embed] + 每层[conv_in, attn_raw, attn_fin,
        // mlp_conv_in, mlp_out, 出口和]×5 = 1 + 6×5 = 31 根;出口和纯观测跳比
        assert_eq!(probe_roots.len(), 1 + nl * 6, "probe 结构");
        let phase_keys = ["conv_in", "attn_raw", "attn_fin", "mlp_conv_in", "mlp_out"];
        for li in 0..nl {
            for (p, sfx) in phase_keys.iter().enumerate() {
                let idx = 1 + li * 6 + p;
                compare(&format!("L{li}_{sfx}"), &harvested[idx]);
            }
        }
        {
            // hidden
            let i = probe_roots.len();
            compare("hidden", &harvested[i]);
            // logits head(相对容差 4e-3 + 1.0 地板;取各行前 16 列 ——
            // 注意块是 [7, 248320] 行主序,前 112 元素只是行 0 的前 112 列,
            // 不能直接与 golden [7,16] 拍平对比 —— 2026-10-07 定谳:
            // 前版「logits bad=97/112」纯属此切片错位,gemm/BlockSlice 无罪)
            let vlog = *logits_t.shape().last().unwrap();
            let mut lw: Vec<f32> = Vec::with_capacity(7 * 16);
            for r in 0..7 {
                for c in 0..16 {
                    lw.push(harvested[i + 1][r * vlog + c]);
                }
            }
            // host 参考:owl 内积复算 hidden 行1 × LM 表前 16 列(f16 四舍五入
            // 同构)—— gemm 通道 vs 权重装载归因
            {
                let hid0 = &harvested[i][5120..5120 * 2]; // hidden 行1 = draft 首
                let mut ref0 = Vec::with_capacity(16);
                for j in 0..16 {
                    let col = &lm_tab[j * 5120..(j + 1) * 5120];
                    let mut acc = 0f32;
                    for t in 0..5120 {
                        acc +=
                            hid0[t] * half::f16::from_f32(col[t]).to_f32();
                    }
                    ref0.push(half::f16::from_f32(acc).to_f32());
                }
                eprintln!(
                    "[golden] logits host_ref[0..4]={:?} owl={:?} golden={:?}",
                    &ref0[..4],
                    &lw[..4],
                    golden.get("logits_head").map(|v| v[..4].to_vec())
                );
            }
            if let Some(want) = golden.get("logits_head") {
                let wf: Vec<f32> = want.iter().copied().collect();
                let lbad = lw
                    .iter()
                    .zip(&wf)
                    .filter(|(g, w)| !((**g - **w).abs() <= 4e-3 * w.abs() + 1.0))
                    .count();
                let samples: Vec<String> = lw
                    .iter()
                    .zip(&wf)
                    .enumerate()
                    .filter(|(_, (g, w))| !((**g - **w).abs() <= 4e-3 * w.abs() + 1.0))
                    .take(3)
                    .map(|(j, (g, w))| format!("[{j}]{g:.1}vs{w:.1}"))
                    .collect();
                eprintln!("[golden] logits_head: bad={lbad}/{} 样本 {}", lw.len(), samples.join(" "));
            }
            // drafts(打印 + 一致计数)
            let dt = &harvested[i + 2];
            if let Some(dw) = golden.get("drafts") {
                let dwf: Vec<f32> = dw.iter().copied().collect();
                let dmatch = dt
                    .iter()
                    .zip(&dwf)
                    .filter(|(g, w)| (**g - **w).abs() < 0.5)
                    .count();
                eprintln!(
                    "[golden] drafts got={:?} want={:?} match={dmatch}/{}",
                    &dt[..7.min(dt.len())],
                    &dwf[..7.min(dwf.len())],
                    dwf.len()
                );
            }
        }

        // gemm 多行诊断(2026-10-07:logits 行1+ 结构化垃圾;hidden 行1-7
        // 已证全同 → gemm 各行输出应全同。三变体归因:无切片 / 0偏移切片 /
        // 1行偏移切片)
        {
            let l_full = embed.lm_head_matmul(&hidden_t); // [8, vocab] 无切片
            let l_zs =
                embed.lm_head_matmul(&hidden_t.slice_view(5120 * 0, vec![7, 5120]));
            let variants: [(String, TensorOps); 3] = [
                ("full[8]".into(), l_full),
                ("slice0[7]".into(), l_zs),
                ("slice1[7]".into(), logits_t.clone()),
            ];
            let rs: Vec<&TensorOps> = variants.iter().map(|(_, t)| t).collect();
            let outs2 = crate::interpreters::eval_ops_multi(&rs, &mut gpu)
                .await
                .expect("gemm 诊断");
            for ((name, t), o) in variants.iter().zip(&outs2) {
                let rows = t.shape()[0];
                let vocab = *t.shape().last().unwrap();
                let mut buf = vec![0u8; o.len * 2];
                gpu.dtoh(o, &mut buf).await.expect("dtoh");
                let at = |r: usize, c: usize| {
                    half::f16::from_le_bytes([buf[(r * vocab + c) * 2], buf[(r * vocab + c) * 2 + 1]])
                        .to_f32()
                };
                let mut worst = (0f32, 0usize);
                for j in 1..rows {
                    let mut d = 0f32;
                    for c in 0..vocab {
                        d = d.max((at(j, c) - at(0, c)).abs());
                    }
                    if d > worst.0 {
                        worst = (d, j);
                    }
                }
                eprintln!(
                    "[gemm-diag] {name}: rows={rows} 行j-vs-行0 最大差={worst:?} 块id={} 行0[:3]={:?} 行1[:3]={:?}",
                    o.id,
                    &(0..3).map(|c| at(0, c)).collect::<Vec<_>>(),
                    &(0..3).map(|c| at(1.min(rows - 1), c)).collect::<Vec<_>>(),
                );
            }
            // eval1(主求值)与 eval2 的 hidden 头值对账
            eprintln!(
                "[gemm-diag] eval1 hidden[0,:3]={:?} [1,:3]={:?}; logits 行0[:3]={:?} 行1[:3]={:?}",
                &harvested[probe_roots.len()][0..3],
                &harvested[probe_roots.len()][5120..5123],
                &harvested[probe_roots.len() + 1][0..3],
                &harvested[probe_roots.len() + 1][248320..248323],
            );
            // eval3:hidden 再求值一次(第三次),看 hidden 块内容是否随求值漂移
            {
                let rs3: Vec<&TensorOps> = vec![&hidden_t, &logits_t];
                let outs3 = crate::interpreters::eval_ops_multi(&rs3, &mut gpu)
                    .await
                    .expect("eval3");
                let hb = &outs3[0];
                let mut hbuf = vec![0u8; hb.len * 2];
                gpu.dtoh(hb, &mut hbuf).await.expect("dtoh");
                let h_at = |i: usize| {
                    half::f16::from_le_bytes([hbuf[i * 2], hbuf[i * 2 + 1]]).to_f32()
                };
                let lb = &outs3[1];
                let mut lbuf = vec![0u8; lb.len * 2];
                gpu.dtoh(lb, &mut lbuf).await.expect("dtoh");
                let l_at = |r: usize, c: usize| {
                    half::f16::from_le_bytes([lbuf[(r * 248320 + c) * 2], lbuf[(r * 248320 + c) * 2 + 1]])
                        .to_f32()
                };
                eprintln!(
                    "[gemm-diag] eval3 hidden 块id={} [0,:3]={:?} [1,:3]={:?}; logits 块id={} 行0[:3]={:?} 行1[:3]={:?} 行差={}",
                    hb.id,
                    &(0..3).map(|c| h_at(c)).collect::<Vec<_>>(),
                    &(0..3).map(|c| h_at(5120 + c)).collect::<Vec<_>>(),
                    lb.id,
                    &(0..3).map(|c| l_at(0, c)).collect::<Vec<_>>(),
                    &(0..3).map(|c| l_at(1, c)).collect::<Vec<_>>(),
                    (0..248320).map(|c| (l_at(1, c) - l_at(0, c)).abs()).fold(0f32, f32::max),
                );
            }
        }
        gpu.close().await.expect("关机");
    }

    /// E5-DF3 m 扫掠:fc_0 marlin 的 m 无变性(m=20 前 m' 行 ≡ m=m')。
    /// 2026-10-07 引擎缺列案的混淆变量切分:引擎 fc m=18,golden 测试
    /// m=20/19/8 —— 若本进程 m=18 也缺列则为 marlin m 病(非进程态);
    /// 参考 = m=20 输出的前 m' 行(marlin 确定性,同权重同输入行)。
    #[tokio::test]
    async fn gpu_dflash2_fc_m_sweep() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let Ok(dir) = std::env::var("OWL_DFLASH2_DIR") else {
            eprintln!("skip: OWL_DFLASH2_DIR 未设");
            return;
        };
        use crate::module::{LoaderCtx, Loadable};
        use owl_iface::DeviceClient as _;
        let mut gpu = gpu_client().await;
        let lctx = LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false };
        let draft = DFlash2Draft::new_with_plan(
            5120, 17408, 32, 8, 128, 5, 1e-6, 248320, QuantPlan::W4A16,
        );
        let src = crate::formats::w4a16::W4A16Source::open_dir(std::path::Path::new(&dir))
            .expect("源打开");
        crate::interpreters::eval_load(&draft, &mut gpu, &src, &lctx)
            .await
            .expect("装载");

        // 公式输入(f16 字节;golden 同式 seed=30)
        let genb = |n: usize, seed: f64| -> Vec<u8> {
            (0..n)
                .map(|i| {
                    half::f16::from_f32(
                        ((((i as f64 + seed) * 0.23).sin()) as f32) * 0.7,
                    )
                    .to_le_bytes()
                })
                .flatten()
                .collect::<Vec<u8>>()
        };
        let ctx = ForwardCtx::minimal(20);
        let taps20v = TensorOps::from_host(Dtype::F16, vec![20, 5120], &genb(20 * 5120, 30.0));
        async fn run_fc(
            gpu: &mut owl_cuda::GpuClient,
            draft: &DFlash2Draft,
            ctx: &ForwardCtx<'_>,
            m: usize,
        ) -> Vec<u8> {
            let genb = |n: usize, seed: f64| -> Vec<u8> {
                (0..n)
                    .map(|i| {
                        half::f16::from_f32(
                            ((((i as f64 + seed) * 0.23).sin()) as f32) * 0.7,
                        )
                        .to_le_bytes()
                    })
                    .flatten()
                    .collect::<Vec<u8>>()
            };
            let taps =
                TensorOps::from_host(Dtype::F16, vec![m, 5120], &genb(m * 5120, 30.0));
            let out = draft.debug_fc0(&taps, ctx);
            let b = crate::interpreters::eval_ops(out.step(), gpu)
                .await
                .expect("fc eval");
            let mut buf = vec![0u8; m * 5120 * 2];
            gpu.dtoh(&b, &mut buf).await.expect("dtoh");
            buf
        }
        let ref_buf = run_fc(&mut gpu, &draft, &ctx, 20).await;
        let at = |buf: &[u8], r: usize, c: usize| {
            half::f16::from_le_bytes([buf[(r * 5120 + c) * 2], buf[(r * 5120 + c) * 2 + 1]])
                .to_f32()
        };
        for m in [16usize, 18, 19] {
            let buf = run_fc(&mut gpu, &draft, &ctx, m).await;
            // 行级:每行与参考(m=20 同行)的最大差 + 全零行计
            let mut bad_rows = 0usize;
            let mut first_bad = None;
            for r in 0..m {
                let dmax = (0..5120)
                    .map(|c| (at(&buf, r, c) - at(&ref_buf, r, c)).abs())
                    .fold(0f32, f32::max);
                let zeros = (0..5120).filter(|&c| at(&buf, r, c) == 0.0).count();
                if dmax > 1e-2 {
                    bad_rows += 1;
                    if first_bad.is_none() {
                        first_bad = Some((r, dmax, zeros));
                    }
                }
                if zeros > 256 {
                    eprintln!("[m-sweep] m={m} 行{r} 零元={zeros}/5120 (可疑缺写)");
                }
            }
            eprintln!(
                "[m-sweep] m={m}: bad_rows={bad_rows}/{m} first_bad={first_bad:?}"
            );
        }
        // m=18 行 0 列剖面(零段定位)
        let buf18 = run_fc(&mut gpu, &draft, &ctx, 18).await;
        let zeros64 = (0..64).filter(|&c| at(&buf18, 0, c) == 0.0).count();
        let zeros_all = (0..5120).filter(|&c| at(&buf18, 0, c) == 0.0).count();
        eprintln!(
            "[m-sweep] m=18 行0: cols0-63 零={zeros64}/64 全行零={zeros_all}/5120"
        );
        // 五块逐个对表 host 真值(python dequant:公式 taps 下
        // fc0 |max|=3.192 零0/102400, fc1 30.833/0, fc2 22.858/0,
        // fc3 148.713/0, fc4 0.684/0 —— 全稠密;另:m=20 raw 直喂)
        for i in 0..5 {
            let out = draft.debug_fci(i, &taps20v, &ctx);
            let b = crate::interpreters::eval_ops(out.step(), &mut gpu)
                .await
                .expect("fc eval");
            let mut buf = vec![0u8; 20 * 5120 * 2];
            gpu.dtoh(&b, &mut buf).await.expect("dtoh");
            let mx = buf
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                .fold(0f32, f32::max);
            let zeros = buf
                .chunks_exact(2)
                .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() == 0.0)
                .count();
            eprintln!(
                "[m-sweep] fc{i}(m=20 raw): |max|={mx:.3} 零元={zeros}/102400"
            );
        }
        // scaled 输入对照(除 256 后应 |max|≈真值/256:fc0 0.0125 … fc3 0.581)
        let scale = TensorOps::from_host(
            Dtype::F16,
            vec![20, 5120],
            &{
                let mut v = Vec::with_capacity(20 * 5120 * 2);
                for _ in 0..20 * 5120 {
                    v.extend_from_slice(&half::f16::from_f32(1.0 / 256.0).to_le_bytes());
                }
                v
            },
        );
        let scaled = taps20v.mul(&scale);
        for i in 0..5 {
            let out = draft.debug_fci(i, &scaled, &ctx);
            let b = crate::interpreters::eval_ops(out.step(), &mut gpu)
                .await
                .expect("fc eval");
            let mut buf = vec![0u8; 20 * 5120 * 2];
            gpu.dtoh(&b, &mut buf).await.expect("dtoh");
            let mx = buf
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                .fold(0f32, f32::max);
            let zeros = buf
                .chunks_exact(2)
                .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() == 0.0)
                .count();
            eprintln!(
                "[m-sweep] fc{i}(m=20 scaled): |max|={mx:.4} 零元={zeros}/102400"
            );
        }
        // mul 输出体检(1/256 缩放后的 taps;期望 max=0.7/256≈0.0027,零元 0)
        {
            let scale = TensorOps::from_host(
                Dtype::F16,
                vec![20, 5120],
                &{
                    let mut v = Vec::with_capacity(20 * 5120 * 2);
                    for _ in 0..20 * 5120 {
                        v.extend_from_slice(&half::f16::from_f32(1.0 / 256.0).to_le_bytes());
                    }
                    v
                },
            );
            let mulout = taps20v.mul(&scale);
            let b = crate::interpreters::eval_ops(mulout.step(), &mut gpu)
                .await
                .expect("mul eval");
            let mut mb = vec![0u8; 20 * 5120 * 2];
            gpu.dtoh(&b, &mut mb).await.expect("dtoh");
            let mx = mb
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                .fold(0f32, f32::max);
            let zeros = mb
                .chunks_exact(2)
                .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() == 0.0)
                .count();
            eprintln!(
                "[m-sweep] mul(taps,1/256): |max|={mx:.6} 零元={zeros}/102400(期望 max≈0.0027 零0)"
            );
        }
        // 合埣实验:同一 multi(CSE 共享执行),根 = [PM, fc0..fc4]
        {
            let taps_all: Vec<TensorOps> = (0..5)
                .map(|i| {
                    TensorOps::from_host(
                        Dtype::F16,
                        vec![20, 5120],
                        &genb(20 * 5120, 30.0 + i as f64 * 7.0),
                    )
                })
                .collect();
            let pm = draft.project_memory(&taps_all, &ctx);
            let mut roots: Vec<TensorOps> = vec![pm];
            for i in 0..5 {
                roots.push(draft.debug_fci(i, &taps_all[i], &ctx));
            }
            let refs: Vec<&TensorOps> = roots.iter().collect();
            let outs = crate::interpreters::eval_ops_multi(&refs, &mut gpu)
                .await
                .expect("合埣 eval");
            for (i, o) in outs.iter().enumerate() {
                let mut buf = vec![0u8; 20 * 5120 * 2];
                gpu.dtoh(o, &mut buf).await.expect("dtoh");
                let mx = buf
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32().abs())
                    .fold(0f32, f32::max);
                let zeros = buf
                    .chunks_exact(2)
                    .filter(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32() == 0.0)
                    .count();
                if i == 0 {
                    eprintln!(
                        "[同埣] PM: |max|={mx:.4} 零元={zeros}/102400(python 真值应稠密)"
                    );
                } else {
                    eprintln!(
                        "[同埣] fc{}: |max|={mx:.3} 零元={zeros}/102400",
                        i - 1
                    );
                }
            }
        }
        gpu.close().await.expect("关机");
    }

    fn read_safetensors(path: &str) -> HashMap<String, Vec<f32>> {
        let full = format!(
            "{}/{}",
            env!("CARGO_MANIFEST_DIR"),
            path.replace("testdata", "../../testdata")
        );
        let bytes = std::fs::read(&full).expect("golden 读");
        // 正规解析(safetensors crate;手写扫描器会被 __metadata__ 吞掉
        // 头部前几个键 —— 2026-10-07 对拍乱象根源之一)
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("golden 头解析");
        let mut out = HashMap::new();
        for (name, view) in st.iter() {
            let data = view.data();
            let vals: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::F16 => data
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                other => panic!("golden dtype 不支持: {other:?}"),
            };
            out.insert(name.to_string(), vals);
        }
        out
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
        let attn = DfAttn::new(hq, hkv, hd, hidden, eps, QuantPlan::F16, Dtype::F16);
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
                .arg_i32(attn.q_norm().w_off() as i32) // 6.18 律:plain ×w(曾误 ×(1+w))
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
                .arg_i32(attn.k_norm().w_off() as i32) // 同律
                .aux(&[2, hkv, hd])
                .with_shape(Dtype::F16, vec![2, hkv * hd])
        };
        // v2 语义:写-后-打分 —— 自块 k/v 先 K0 写池 slots [3,4],
        // NC 全窗 [0..5) 纯池读(生产流同式:encode/propose 前置写槽)
        let k0_write = TensorOps::call(crate::ops::ids::ATTN_K0_WRITE)
            .aux(&[2])
            .arg(&k_self2)
            .arg(&v_self2)
            .arg(&kv.k_cache)
            .arg(&kv.v_cache)
            .arg(&kv2.slots)
            .arg_i32((hkv * hd) as i32)
            .arg_i32((hkv * hd) as i32)
            .arg_i32(hkv as i32)
            .arg_i32(hd as i32)
            .arg_i32(page as i32)
            .arg_i32(x as i32)
            .with_shape(Dtype::F16, vec![1]);
        let _ = harvest_f16(&mut gpu, &k0_write).await; // 触发写(fire-and-forget)
        // v2 契约(6.19):prefix_len 去烘焙 → kv_len_ptr 张量读(契约 5);
        // grid (T, Hq) × block (hd) —— block-per-head flash 式;窗口 =
        // 全可见 5(前缀 3 + 自块 2,slots [0..5) 连续直排)
        let kv_len_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[5.0]));
        let y_nc = TensorOps::of(crate::kernel::kernel_with(
            "owl_naive_attn_nc_f16", (2, hq as u32, 1), (hd as u32, 1, 1), 0,
        ))
        .arg(&q_self2)
        .arg(&k_self2)
        .arg(&v_self2)
        .arg(&kv.k_cache)
        .arg(&kv.v_cache)
        .arg(&kv_len_t)
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
                        // 6.18 律:qk-norm plain ×w(曾误 ×(1+w);DfAttn::new
                        // 已改 RmsNorm::new 非 add_one,host 同步)
                        nrm[d] = seg[d] * inv * w_n[d];
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

    /// 金标 5(BF16 面;E5-DF3 同日十四):草稿路径 BF16 化三件套 ——
    /// ① conv bf16 vs host f32 参考(权重经 f32→bf16 装载);② cast
    /// f16→bf16→f16 往返(f16 值域内零损失);③ select bf16(proj/码本
    /// bf16)与 f16 版同 tok 序。用例容差按 bf16 尾数 7~8 位放行。
    #[tokio::test]
    async fn gpu_dflash_bf16_smoke() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let mut gpu = gpu_client().await;

        // ---- ① conv bf16(全 bf16:x/delta/base/out;proj 线性 bf16 want)----
        let (t, hidden, block) = (8usize, 64usize, 8usize);
        let groups = hidden / GROUP;
        let conv = GroupedConv::new(hidden, block, Dtype::BF16);
        let base: Vec<f32> = (0..2 * TAPS * hidden)
            .map(|i| {
                let (s, tap, d) = (i / (TAPS * hidden), (i / hidden) % TAPS, i % hidden);
                if tap == 0 { 1.0 + (d as f32 * 0.01).sin() * 0.1 } else { ((i % 7) as f32 - 3.0) * 0.05 }
            })
            .collect();
        let w_proj = gen(2 * TAPS * groups * hidden, 1.0);
        let mut src: HashMap<String, Vec<f32>> = HashMap::new();
        src.insert("base_kernel".into(), base.clone());
        src.insert("kernel_projection".into(), w_proj.clone());
        crate::interpreters::eval_load(&conv, &mut gpu, &src, &LoaderCtx { dtype: Dtype::BF16, shard: 1, device_repack: false })
            .await
            .expect("conv bf16 装载");

        let x = gen(t * hidden, 2.0);
        let x_t = TensorOps::from_host(Dtype::BF16, vec![t, hidden], &bf16b(&x));
        let ctx = ForwardCtx::minimal(t);
        let (conv_in, delta) = conv.prepare(&x_t, &ctx);
        let got_delta = harvest_bf16(&mut gpu, &delta).await;
        let y_in = harvest_bf16(&mut gpu, &conv_in).await;

        // host 参考(f32 中间;输入按 bf16 网格量化对齐)
        let xb32: Vec<f32> = x.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
        let delta_host = {
            let mut dy = vec![0f32; t * 2 * TAPS * groups];
            for r in 0..t {
                for o in 0..2 * TAPS * groups {
                    let mut acc = 0f32;
                    for k in 0..hidden {
                        acc += xb32[r * hidden + k]
                            * half::bf16::from_f32(w_proj[o * hidden + k]).to_f32();
                    }
                    dy[r * 2 * TAPS * groups + o] = acc;
                }
            }
            dy
        };
        let conv_ref = |side: usize, xin: &[f32], delta: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; t * hidden];
            for tt in 0..t {
                for d in 0..hidden {
                    let g = d / GROUP;
                    let c0 = base[(side * TAPS) * hidden + d]
                        + half::bf16::from_f32(delta[(tt * 2 + side) * 2 * groups + g]).to_f32();
                    let c1 = base[(side * TAPS + 1) * hidden + d]
                        + half::bf16::from_f32(delta[(tt * 2 + side) * 2 * groups + groups + g]).to_f32();
                    let x0 = xin[tt * hidden + d];
                    let x1 = if tt >= 1 { xin[(tt - 1) * hidden + d] } else { 0.0 };
                    let m1 = if tt % block >= 1 { 1.0 } else { 0.0 };
                    out[tt * hidden + d] = c0 * x0 + c1 * x1 * m1;
                }
            }
            out
        };
        // host 参考落 bf16 输出网格(核输出 = bf16;mag ~16 处 quantum
        // 0.0625,比 f16 版的 2e-2 宽 —— 累加序差一量子内放行)
        let delta_grid: Vec<f32> = delta_host.iter().map(|&v| b32g(v)).collect();
        assert_close(&got_delta, &delta_grid, 0.15, "bf16 delta = proj(x)");
        let y_ref: Vec<f32> = conv_ref(0, &xb32, &delta_host).iter().map(|&v| b32g(v)).collect();
        assert_close(&y_in, &y_ref, 0.15, "bf16 conv prepare");

        // ---- ② cast 往返(f16 → bf16 → f16;bf16 尾数 7 位,值域内损失 ≤ 0.4%)----
        let vals = gen(1024, 9.0);
        let v_f16 = TensorOps::from_host(Dtype::F16, vec![1024], &f16b(&vals));
        let (gx, _, _) = crate::ops::auto_grid(1024);
        let v_bf16 = TensorOps::of(crate::kernel::kernel_with(
            "owl_cast_f16_bf16", (gx, 1, 1), (256, 1, 1), 0,
        ))
        .arg(&v_f16)
        .arg_i32(1024)
        .with_shape(Dtype::BF16, vec![1024]);
        let round = TensorOps::of(crate::kernel::kernel_with(
            "owl_cast_bf16_f16", (gx, 1, 1), (256, 1, 1), 0,
        ))
        .arg(&v_bf16)
        .arg_i32(1024)
        .with_shape(Dtype::F16, vec![1024]);
        let got = harvest_f16(&mut gpu, &round).await;
        for (i, (&a, &b)) in vals.iter().zip(got.iter()).enumerate() {
            let want = half::bf16::from_f32(a).to_f32(); // f16→bf16 网格
            assert!((b - want).abs() < 1e-6, "cast 往返 [{i}] {b} vs {want}");
        }

        // ---- ③ select bf16(proj/码本 bf16;tok 序与 host 同律)----
        let (vocab, rank, rows) = (97usize, 16usize, DEPTH);
        let sel = CandidateSelector::new(rank, vocab, rank, Dtype::BF16);
        let w2 = gen(rank * rank, 3.0);
        let a_tab = gen(vocab * rank, 4.0);
        let b_tab = gen(vocab * rank, 5.0);
        let mut src2: HashMap<String, Vec<f32>> = HashMap::new();
        src2.insert("hidden_projection".into(), w2.clone());
        src2.insert("predecessor_codebook".into(), a_tab.clone());
        src2.insert("successor_codebook".into(), b_tab.clone());
        crate::interpreters::eval_load(&sel, &mut gpu, &src2, &LoaderCtx { dtype: Dtype::BF16, shard: 1, device_repack: false })
            .await
            .expect("selector bf16 装载");
        let hidden_h = gen((rows + 1) * rank, 6.0);
        let anchor = 3.0f32;
        let hidden_t = TensorOps::from_host(Dtype::BF16, vec![rows + 1, rank], &bf16b(&hidden_h));
        let anchor_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[anchor]));
        // logits 直构 f16(topk 面不变;值域按 bf16 网格对齐)
        let lm_head = gen(vocab * rank, 7.0);
        let pred: Vec<f32> = hidden_h[rank..].to_vec();
        let mut logits = vec![0f32; rows * vocab];
        for r in 0..rows {
            for v in 0..vocab {
                let mut acc = 0f32;
                for k in 0..rank {
                    acc += half::bf16::from_f32(pred[r * rank + k]).to_f32()
                        * half::bf16::from_f32(lm_head[v * rank + k]).to_f32();
                }
                logits[r * vocab + v] = half::f16::from_f32(acc).to_f32();
            }
        }
        let logits_t = TensorOps::from_host(Dtype::F16, vec![rows, vocab], &f16b(&logits));
        let (drafts, _scores) = sel.select_with_logits(&hidden_t, &anchor_t, &logits_t, &ctx);
        // drafts = select 单缓冲 SliceView —— dtoh 回读整父块(select 同坑,
        // f16 selector 测试同款:host 侧取前 rows)
        let drafts_bytes = crate::interpreters::eval_ops(drafts.step(), &mut gpu)
            .await
            .expect("eval drafts");
        let mut dbuf = vec![0u8; rows * (TOP_K * TOP_K + 1) * 4];
        owl_iface::contract::DeviceClient::dtoh(&mut gpu, &drafts_bytes, &mut dbuf).await.expect("dtoh drafts");
        let got_toks: Vec<f32> = dbuf[..rows * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // host 参考:topk((值降,索引升))+ bf16 网格格打分 + 贪心 walk
        let b32 = |v: f32| half::bf16::from_f32(v).to_f32();
        let host_topk = |row: &[f32]| -> (Vec<f32>, Vec<usize>) {
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
            idx.truncate(TOP_K);
            (idx.iter().map(|&i| row[i]).collect(), idx)
        };
        let mut cand_h = vec![0f32; rows * TOP_K];
        let mut unary_h = vec![0f32; rows * TOP_K];
        for r in 0..rows {
            let (vals2, ids) = host_topk(&logits[r * vocab..(r + 1) * vocab]);
            for (k, (&v, &i)) in vals2.iter().zip(ids.iter()).enumerate() {
                unary_h[r * TOP_K + k] = v;
                cand_h[r * TOP_K + k] = i as f32;
            }
        }
        let mut walk_idx = 0usize;
        for e in 0..rows {
            let mut best = (f32::MIN, 0usize);
            for c in 0..TOP_K {
                let pred_id = if e == 0 { anchor as usize } else { cand_h[(e - 1) * TOP_K + walk_idx] as usize };
                let cand_id = cand_h[e * TOP_K + c] as usize;
                let mut dot = 0f32;
                for r2 in 0..rank {
                    // proj 行 e = hidden_projection(hidden[e+1])(bf16 网格逐点)
                    let mut pv = 0f32;
                    for k2 in 0..rank {
                        pv += b32(hidden_h[(e + 1) * rank + k2]) * b32(w2[r2 * rank + k2]);
                    }
                    dot += b32(a_tab[pred_id * rank + r2])
                        * half::bf16::from_f32(pv).to_f32()
                        * b32(b_tab[cand_id * rank + r2]);
                }
                let score = unary_h[e * TOP_K + c] + dot;
                if score > best.0 {
                    best = (score, c);
                }
            }
            walk_idx = best.1;
            assert_eq!(
                got_toks[e] as u32,
                cand_h[e * TOP_K + walk_idx] as u32,
                "bf16 select tok 行 {e}(host walk c={walk_idx})"
            );
        }
        gpu.close().await.expect("关机");
    }

    /// E5-DF4 性能:propose 图各阶段 GPU 计时(真实 W4A16 草稿 BF16 面;
    /// 每阶段独立 eval + sync,预热 1 次计 5 次)。定位 16ms 构成。
    #[tokio::test]
    async fn gpu_dflash2_stage_bench() -> Result<(), crate::contract::ModelError> {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return Ok(());
        }
        let Ok(dir) = std::env::var("OWL_DFLASH2_DIR") else {
            eprintln!("skip: OWL_DFLASH2_DIR 未设");
            return Ok(());
        };
        use crate::module::{LoaderCtx, Loadable};
        use owl_iface::DeviceClient as _;
        let mut gpu = gpu_client().await;
        let lctx = LoaderCtx { dtype: Dtype::BF16, shard: 1, device_repack: false };
        let draft = DFlash2Draft::new_with_plan_dt(
            5120, 17408, 32, 8, 128, 5, 1e-6, 248320, QuantPlan::W4A16, Dtype::BF16,
        );
        let src = crate::formats::w4a16::W4A16Source::open_dir(std::path::Path::new(&dir))
            .expect("源打开");
        crate::interpreters::eval_load(&draft, &mut gpu, &src, &lctx)
            .await
            .expect("装载");

        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let f16b = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
        };
        let f32_to_bf16b = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|f| half::bf16::from_f32(*f).to_le_bytes()).collect()
        };
        let gen = |n: usize, seed: f64| -> Vec<f32> {
            (0..n).map(|i| (((i as f64 + seed) * 0.23).sin() as f32) * 20.0).collect()
        };
        let (hq, hkv, hd, hidden) = (32usize, 8usize, 128usize, 5120usize);
        let rope = Rope::new(262_144, hd, hd, 1.0e7).expect("rope");
        crate::interpreters::eval_load(&rope, &mut gpu, &rope.tables(), &LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false })
            .await
            .expect("rope 表");

        // 输入面
        let taps: Vec<TensorOps> = (0..5)
            .map(|i| TensorOps::from_host(Dtype::F16, vec![8, hidden], &f16b(&gen(8 * hidden, 30.0 + i as f64))))
            .collect();
        let pos_t = TensorOps::from_host(Dtype::F32, vec![8], &f32b(&(0..8).map(|i| i as f32).collect::<Vec<_>>()));
        let ids_t = TensorOps::from_host(Dtype::F32, vec![8], &f32b(&[109757.0, 248070.0, 248070.0, 248070.0, 248070.0, 248070.0, 248070.0, 248070.0]));
        let anchor_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[109757.0]));
        let kv_len_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[18.0]));

        // 池(单块 12 页几何同引擎)
        let (page, x, nb) = (32usize, 8usize, 12usize);
        let mut kvs = Vec::new();
        for _ in 0..5 {
            let kc = crate::interpreters::eval_ops(
                TensorOps::zeros(Dtype::BF16, vec![nb, hkv, hd / x, page, x]).step(), &mut gpu)
                .await.expect("kc");
            let vc = crate::interpreters::eval_ops(
                TensorOps::zeros(Dtype::BF16, vec![nb, hkv, hd, page]).step(), &mut gpu)
                .await.expect("vc");
            let slots = TensorOps::from_host(Dtype::F32, vec![8], &f32b(&(0..8).map(|i| i as f32).collect::<Vec<_>>()));
            kvs.push(KvBuffers {
                k_cache: TensorOps::of_block(kc.id, Dtype::BF16, vec![nb, hkv, hd / x, page, x]),
                v_cache: TensorOps::of_block(vc.id, Dtype::BF16, vec![nb, hkv, hd, page]),
                slots,
                kv_lens: TensorOps::from_host(Dtype::F32, vec![8], &f32b(&(0..8).map(|i| (i + 1) as f32).collect::<Vec<_>>())),
                block_tables: TensorOps::from_host(Dtype::F32, vec![1, nb], &f32b(&(0..nb).map(|i| i as f32).collect::<Vec<_>>())),
            });
        }

        let ctx = ForwardCtx::minimal(8);
        // 真实 embed/lm_head(target 表)——与引擎同位
        let mem = draft.project_memory(&taps, &ctx);
        let _ = crate::testkit::harvest_bf16(&mut gpu, &mem).await; // 预热

        // 阶段 A0:单发 fc marlin m=8(定位 marlin m=8 效率)
        {
            let tap1 = TensorOps::from_host(Dtype::BF16, vec![8, hidden], &f32_to_bf16b(&gen(8 * hidden, 50.0)));
            let mut t = Vec::new();
            for _ in 0..6 {
                let s = std::time::Instant::now();
                let fc0 = draft.debug_fc0(&tap1, &ctx);
                let _ = crate::testkit::harvest_bf16(&mut gpu, &fc0).await;
                t.push(s.elapsed());
            }
            eprintln!("[bench] A0 fc 单发 marlin m=8      = {:?}(末次)", t[5]);
        }
        // 阶段 A:memory
        {
            let mut t = Vec::new();
            for _ in 0..6 {
                let s = std::time::Instant::now();
                let root = draft.project_memory(&taps, &ctx);
                let _ = crate::testkit::harvest_bf16(&mut gpu, &root).await;
                t.push(s.elapsed());
            }
            eprintln!("[bench] A memory(5×fc+norm)      = {:?}(末次)", t[5]);
        }
        // 阶段 B:encode_kv 5 层
        {
            let mut t = Vec::new();
            for _ in 0..6 {
                let s = std::time::Instant::now();
                let roots = draft.encode_kv(&mem, &pos_t, &kvs, &rope, &ctx);
                let refs: Vec<&TensorOps> = roots.iter().collect();
                let _ = crate::interpreters::eval_ops_multi(&refs, &mut gpu).await?;
                gpu.sync().await.expect("sync");
                t.push(s.elapsed());
            }
            eprintln!("[bench] B encode_kv(5 层 kv 写)  = {:?}(末次)", t[5]);
        }
        // 阶段 C:噪声块 5 层(embed 经引擎同位 cast;层用 layers() 观测面)
        {
            let mut t = Vec::new();
            for _ in 0..6 {
                let s = std::time::Instant::now();
                let emb_out = TensorOps::from_host(Dtype::F16, vec![8, hidden], &f16b(&gen(8 * hidden, 40.0)));
                let xx = ensure_dt(&emb_out, Dtype::BF16);
                let mut residual: Option<TensorOps> = None;
                let mut cur = xx;
                for (i, layer) in draft.layers().iter().enumerate() {
                    let (nx, res) = layer.forward(
                        &cur, residual.as_ref(), &pos_t, &kvs[i], &rope, 1e-6,
                        &kv_len_t, &ctx, None,
                    );
                    cur = nx;
                    residual = Some(res);
                }
                let _ = crate::testkit::harvest_bf16(&mut gpu, &cur).await;
                t.push(s.elapsed());
            }
            eprintln!("[bench] C 噪声块 5 层            = {:?}(末次)", t[5]);
        }
        // 阶段 D:lm_head + topk + select
        {
            let lm = gen(248320 * hidden, 21.0);
            let lm_t = TensorOps::from_host(Dtype::F16, vec![248320, hidden], &f16b(&lm));
            let hid7 = TensorOps::from_host(Dtype::F16, vec![7, hidden], &f16b(&gen(7 * hidden, 31.0)));
            let mut t = Vec::new();
            for _ in 0..6 {
                let s = std::time::Instant::now();
                let logits = hid7.matmul_nt(&lm_t);
                let _ = crate::testkit::harvest_f16(&mut gpu, &logits).await;
                t.push(s.elapsed());
            }
            eprintln!("[bench] D lm_head cublas [7,5120]×[248320,5120] = {:?}(末次)", t[5]);
        }
        gpu.close().await.expect("关机");
        Ok(())
    }
}

// scratch:隔离 Linear matmul_nt 行为

