//! GDN:GatedDeltaNet 线性注意力(Qwen3.5 18/24 层;M-c 逐核小批移植)。
//!
//! 移植源:owl-nn `kernels/cu/gdn/gdn_kernels.cu`(旧世界已验证数学)+
//! owl-engine `models/layers/deltanet.rs`(垫片调用形态);
//! 金标准 = HF `Qwen3_5GatedDeltaNet` / `torch_recurrent_gated_delta_rule`
//! (gdn.py op 分发)。
//!
//! 批次账(每批收口全绿,勿一把梭):
//! - 批1 ✅ gating:g 臂新核 `owl_gdn_gating_g_f32`(softplus 无语义等价);
//!   **beta 臂改判复用** `owl_sigmoid_f32`(= sigmoid(b),M-b 同款律)
//! - 批2 ✅ l2norm:`owl_gdn_l2norm_f32`(块内归约一 block 一行;
//!   旧世界 warp/block 双变体合并,dim ≤ 256 走块内)
//! - 批3 ✅ conv_upd:`owl_gdn_conv_upd_f32`(decode 槽更新;state 原地 =
//!   已知副作用,C7 同族单流保序;三段 q/k/v 独立发射,零拼接算子;
//!   Qwen3.5 conv 无 bias;slot<0 = padding 跳写 state 与 out)
//! - 批4 ✅ delta_dec:`owl_gdn_delta_dec_f32`(单步递推;g log 空间核内
//!   exp;q 核内乘 q_scale = 1/√kd;kd ≤ 128 寄存器硬上界;
//!   HF `torch_recurrent_gated_delta_rule` 直跑对拍 out + state)
//! - 批5 ✅ norm_act:`owl_gdn_norm_act_f32`(×w **非零中心** ——
//!   RMSNormGated weight=ones 初始化,与 qk-norm 零中心相反,HF 实证;
//!   act=silu;一 block 一 (row, group))
//! - 批6 ⏳ GatedDeltaNet 层组装(SplitQkvZa 投影 + decode 全链)
//!
//! 形状账(0.8B 实测):HK=16 × hd_k 128 = key_dim 2048;HV=16 × hd_v 128;
//! conv_dim = 2·key_dim + value_dim = 6144;hidden 1024;conv k=4 无 bias。

use crate::contract::Dtype;
use crate::kernel;
use crate::layers::linear::Linear;
use crate::layers::{concat_rows, narrow_strided};
use crate::module::{ForwardCtx, Loadable, LoaderCtx, LoaderOps, Module, Weight};
use crate::TensorOps;

// ============================================================================
// GDN 常驻状态(runner 词汇;块归 runner,层零所有权 —— KvBuffers 同构)
// ============================================================================

/// GDN decode 常驻块组(0.8B 单层;每层一套,归 runner 池)。
/// 全部 `TensorOps::of_block` 常驻块引用(块只增不减;单流保序)。
/// slots [batch] f32 数值过线(负 = padding;契约 5)。
pub struct GdnBuffers {
    /// conv 状态三段(q/k/v 独立建块;零拼接算子)[max_slots, seg_dim, 3]
    pub conv_q: TensorOps,
    pub conv_k: TensorOps,
    pub conv_v: TensorOps,
    /// recurrent 状态 [max_slots, HV, hd_k, hd_v](恒 f32)
    pub rec: TensorOps,
    /// 本步槽位 [batch]
    pub slots: TensorOps,
}

// ============================================================================
// 批1:融合门控(g 臂;beta = b.sigmoid() 走语义算子)
// ============================================================================

/// g 臂声明:g[i] = -exp(A_log[h]) · softplus(a[i] + dt_bias[h]),h = i % heads。
/// a_log / dt_bias = Weight decl()([heads] 常驻块);a = in_proj_a 输出
/// [tokens, heads]。grid 哨兵自动 1D(逐元素形)。
pub fn gating_g(
    a_log: &TensorOps,
    a: &TensorOps,
    dt_bias: &TensorOps,
    tokens: usize,
    heads: usize,
) -> TensorOps {
    let dt = a.dtype;
    TensorOps::of(kernel::kernel_with(
        gname("owl_gdn_gating_g", dt),
        (0, 0, 0),
        (256, 1, 1),
        0,
    ))
    .arg(a_log)
    .arg(a)
    .arg(dt_bias)
    .arg_usize(tokens * heads)
    .arg_usize(heads)
    .with_shape(dt, vec![tokens, heads])
}

// ============================================================================
// 批2:末维 L2 归一
// ============================================================================

/// l2norm 声明:y[r][i] = x[r][i] · rsqrt(sum(x[r][·]²) + eps)。
/// 行核(非哨兵):grid (rows,1,1) × block (256,1,1)。q/k 头向量归一用
/// (decode:rows = tokens × heads,dim = head_k_dim 128;eps 1e-6)。
pub fn l2norm(x: &TensorOps, rows: usize, dim: usize, eps: f32) -> TensorOps {
    let dt = x.dtype;
    TensorOps::of(kernel::kernel_with(
        gname("owl_gdn_l2norm", dt),
        (rows as u32, 1, 1),
        (256, 1, 1),
        0,
    ))
    .arg(x)
    .arg_usize(rows)
    .arg_usize(dim)
    .arg_f32(eps)
    .with_shape(dt, vec![rows, dim])
}

// ============================================================================
// 批3:causal conv1d decode 槽更新
// ============================================================================

/// conv 槽更新声明(段级:一段一发射,q/k/v 三段各自调用,零拼接)。
/// x [batch, dim];w [conv_dim_total, 4] 单权重块(w_offset = 段基址行号:
/// q=0 / k=key_dim / v=2·key_dim);state [max_slots, dim, 3] 常驻块;
/// slots [batch] f32(负 = padding:跳写 state 与 out);silu 恒 true。
/// **已知副作用**:state 原地更新(C7 同族;块归 runner,单流保序)。
pub fn conv_upd(
    x: &TensorOps,
    w: &TensorOps,
    state: &TensorOps,
    slots: &TensorOps,
    batch: usize,
    dim: usize,
    w_offset: usize,
    silu: bool,
) -> TensorOps {
    let dt = x.dtype;
    TensorOps::of(kernel::kernel_with(
        gname("owl_gdn_conv_upd", dt),
        (0, 0, 0),
        (256, 1, 1),
        0,
    ))
    .arg(x)
    .arg(w)
    .arg(state)
    .arg(slots)
    .arg_usize(batch * dim)
    .arg_usize(dim)
    .arg_usize(w_offset)
    .arg_i32(silu as i32)
    .with_shape(dt, vec![batch, dim])
}

// ============================================================================
// 批4:gated delta rule decode(单步;slot 寻址;GQA)
// ============================================================================

/// 单步递推声明:state *= exp(g) → kv_mem = state·k → delta=(v−kv_mem)·beta
/// → state += k⊗delta → out = state·(q·q_scale)。g [B,HV] log 空间;
/// q_scale = 1/√kd(与 HF torch_recurrent 恒除 √K 同式)。
/// state [max_slots, HV, kd, vd] 常驻块原地更新(已知副作用,C7 同族);
/// slots [batch] f32(负 = padding:跳写 state 与 out)。
/// 发射 = (ceil(vd/64), B·HV) × (64,1,1),shared (128+128+2)·4。
pub fn delta_dec(
    q: &TensorOps,
    k: &TensorOps,
    v: &TensorOps,
    g: &TensorOps,
    beta: &TensorOps,
    state: &TensorOps,
    slots: &TensorOps,
    batch: usize,
    nv: usize,
    nk: usize,
    kd: usize,
    vd: usize,
    q_scale: f32,
) -> TensorOps {
    let dt = v.dtype;
    TensorOps::of(kernel::kernel_with(
        gname("owl_gdn_delta_dec", dt),
        (((vd + 63) / 64) as u32, (batch * nv) as u32, 1),
        (64, 1, 1),
        (2 * 128 + 2) * 4,
    ))
    .arg(q)
    .arg(k)
    .arg(v)
    .arg(g)
    .arg(beta)
    .arg(state)
    .arg(slots)
    .arg_usize(batch)
    .arg_usize(nv)
    .arg_usize(nk)
    .arg_usize(kd)
    .arg_usize(vd)
    .arg_f32(q_scale)
    .with_shape(dt, vec![batch, nv, vd])
}

// ============================================================================
// PF1b 批核 wrapper(2026-09-26 接线;xinfer attention_rs 同源,varlen 单发射)
// ============================================================================

/// causal conv1d prefill varlen(批核):x [T, d] 单发射吃全块,
/// state 槽寻址原地滑窗;slots [1] + cu_seqlens [0,T](单序列)。
/// f16 生产核(f32 锚链走展开降级路)。**无 w_offset 参数** —— 三段
/// 各传窄切后的权重行(段基址在层侧完成)。
pub(crate) fn conv_fwd(
    x: &TensorOps,
    w: &TensorOps,
    state: &TensorOps,
    slots: &TensorOps,
    cu_seqlens: &TensorOps,
    tokens: usize,
    d: usize,
    silu: bool,
) -> TensorOps {
    TensorOps::of(kernel::kernel_with(
        "owl_gdn_conv_fwd_f16",
        (1u32, ((d + 255) / 256) as u32, 1),
        (256, 1, 1),
        0,
    ))
    .arg(x)
    .arg(w)
    .arg(state)
    .arg(slots)
    .arg(cu_seqlens)
    .arg_i32(1) // batch = 1(单序列)
    .arg_i32(d as i32)
    .arg_i32(silu as i32)
    .with_shape(Dtype::F16, vec![tokens, d])
}

/// gated delta rule varlen 递推(批核):q/k/v/g/beta [T, ·] 单发射,
/// t 循环核内;state 槽寻址进出(读初态写终态同格,原地律同族);
/// cu_seqlens [0,T](vLLM 契约保真,M2 packed 直用)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn recurrence_varlen(
    q: &TensorOps,
    k: &TensorOps,
    v: &TensorOps,
    g: &TensorOps,
    beta: &TensorOps,
    state: &TensorOps,
    slots: &TensorOps,
    cu_seqlens: &TensorOps,
    tokens: usize,
    nv: usize,
    nk: usize,
    kd: usize,
    vd: usize,
    q_scale: f32,
) -> TensorOps {
    TensorOps::of(kernel::kernel_with(
        "owl_gdn_recurrence_varlen_gqa_f16",
        (((vd + 7) / 8) as u32, nv as u32, 1),
        (32, 8, 1),
        ((4 * kd + 4) * 4) as u32,
    ))
    .arg(q)
    .arg(k)
    .arg(v)
    .arg(g)
    .arg(beta)
    .arg(state)
    .arg(slots)
    .arg(cu_seqlens)
    .arg_usize(1) // batch = 1(单序列)
    .arg_usize(nv)
    .arg_usize(nk)
    .arg_usize(kd)
    .arg_usize(vd)
    .arg_f32(q_scale)
    .with_shape(Dtype::F16, vec![tokens, nv, vd])
}

// ============================================================================
// 批5:门控 RMSNorm × act(z)
// ============================================================================

/// 门控归一化声明:y = rmsnorm(x 组内)·gamma · act(z)。
/// x/z [rows, value_dim];gamma [group_size](×w 非零中心);
/// act_silu = true(Qwen3.5)/ false = sigmoid(Qwen4 预留)。
/// 发射 = (rows·value_dim/group_size,1,1) × (256,1,1)。
pub fn norm_act(
    x: &TensorOps,
    z: &TensorOps,
    gamma: &TensorOps,
    rows: usize,
    value_dim: usize,
    group_size: usize,
    eps: f32,
    act_silu: bool,
) -> TensorOps {
    let dt = x.dtype;
    TensorOps::of(kernel::kernel_with(
        gname("owl_gdn_norm_act", dt),
        ((rows * value_dim / group_size) as u32, 1, 1),
        (256, 1, 1),
        0,
    ))
    .arg(x)
    .arg(z)
    .arg(gamma)
    .arg_usize(rows)
    .arg_usize(value_dim)
    .arg_usize(group_size)
    .arg_f32(eps)
    // 核约定沿旧世界:act=0 → silu,act=1 → sigmoid
    .arg_i32(if act_silu { 0 } else { 1 })
    .with_shape(dt, vec![rows, value_dim])
}

// ============================================================================
// 批6:GatedDeltaNet 层(Qwen3.5 线性注意力层;decode 全链)
// ============================================================================

/// Qwen3.5 GatedDeltaNet(SplitQkvZa 投影形态 —— checkpoint 四独立投影;
/// 旧世界 GdnProjection::SplitQkvZaLegacy 同款)。
///
/// decode 数据流:
/// ```text
/// xs [T, hidden]
///   ├ in_proj_qkv → [T, 2K+V] ─ narrow 三段 → q/k/v
///   ├ in_proj_z → z [T, V];in_proj_b/a → b/a [T, HV]
///   ├ conv_upd(q/k/v 三段独立发射;state 滑窗 + silu)→ q'/k'/v'
///   ├ l2norm(q',k' per-head)→ g/beta 门控 → delta_dec(单步;rec 原地)
///   ├ norm_act(×w 非零中心 × silu(z) per-head)→ out_proj → [T, hidden]
/// ```
pub struct GatedDeltaNet {
    in_proj_qkv: Linear, // [2K+V, hidden]
    in_proj_z: Linear,   // [V, hidden]
    in_proj_b: Linear,   // [HV, hidden]
    in_proj_a: Linear,   // [HV, hidden]
    out_proj: Linear,    // [hidden, V]
    conv_w: Weight,      // [conv_dim, 4](checkpoint [conv_dim,1,4] 同布局直读)
    a_log: Weight,       // [HV]
    dt_bias: Weight,     // [HV]
    norm_w: Weight,      // [hd_v](×w 非零中心)
    nk: usize,           // key 头数(16)
    hk_dim: usize,       // key 头维(128)
    nv: usize,           // value 头数(16)
    hv_dim: usize,       // value 头维(128)
    hidden: usize,
    eps: f32,
}

impl GatedDeltaNet {
    /// 准备容器(纯元数据;0.8B = nk 16/hk_dim 128/nv 16/hv_dim 128/hidden 1024)
    pub fn new(nk: usize, hk_dim: usize, nv: usize, hv_dim: usize, hidden: usize, eps: f32) -> Self {
        let (key_dim, value_dim) = (nk * hk_dim, nv * hv_dim);
        let conv_dim = 2 * key_dim + value_dim;
        GatedDeltaNet {
            in_proj_qkv: Linear::new("in_proj_qkv", conv_dim, hidden),
            in_proj_z: Linear::new("in_proj_z", value_dim, hidden),
            in_proj_b: Linear::new("in_proj_b", nv, hidden),
            in_proj_a: Linear::new("in_proj_a", nv, hidden),
            out_proj: Linear::new("out_proj", hidden, value_dim),
            conv_w: Weight::new("conv1d", vec![conv_dim, 4]),
            a_log: Weight::new("A_log", vec![nv]),
            dt_bias: Weight::new("dt_bias", vec![nv]),
            norm_w: Weight::new("norm", vec![hv_dim]),
            nk, hk_dim, nv, hv_dim, hidden, eps,
        }
    }

    fn key_dim(&self) -> usize {
        self.nk * self.hk_dim
    }

    fn value_dim(&self) -> usize {
        self.nv * self.hv_dim
    }

    /// prefill 分派(PF1-0):f16 = PF1b 批核(conv_fwd + varlen 递推,
    /// 单发射吃全块);f32 锚链 = 展开降级路(锚链不再演进,展开保留为
    /// 对拍基准)。语义两者同:prefill(x, S) == 依次 T 次 decode 步
    /// (token t,槽恒 gdn_slot)后的状态与逐 token 输出(对拍即证)。
    fn forward_prefill_decl(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        if xs.dtype == Dtype::F16 {
            self.forward_prefill_batch(xs, ctx)
        } else {
            self.forward_prefill_expanded(xs, ctx)
        }
    }

    /// PF1b 批核路径(f16 生产路;2026-09-26 接线):节点账 ~25/层
    /// (展开 ~10T+62 坍缩),投递/norm 侧 T 批量单发不变。
    fn forward_prefill_batch(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        let tokens = ctx.tokens;
        let (gdn, gdn_slot) = match (ctx.gdn, ctx.gdn_slot) {
            (Some(g), Some(s)) => (g, s),
            _ => {
                return TensorOps::poisoned(
                    Dtype::F16,
                    vec![tokens, self.hidden],
                    "gdn prefill(batch): ctx 缺动态依赖(需 gdn 块组 + gdn_slot)",
                );
            }
        };
        let key_dim = self.key_dim();
        let value_dim = self.value_dim();
        let conv_dim = 2 * key_dim + value_dim;

        // T 批量投影
        let qkv = self.in_proj_qkv.forward(xs, ctx); // [T, 2K+V]
        let z = self.in_proj_z.forward(xs, ctx); // [T, V]
        let b = self.in_proj_b.forward(xs, ctx); // [T, HV]
        let a = self.in_proj_a.forward(xs, ctx); // [T, HV]
        let q = narrow_strided(&qkv, tokens, conv_dim, 0, key_dim, vec![tokens, key_dim]);
        let k = narrow_strided(&qkv, tokens, conv_dim, key_dim, key_dim, vec![tokens, key_dim]);
        let v = narrow_strided(&qkv, tokens, conv_dim, 2 * key_dim, value_dim, vec![tokens, value_dim]);

        // cu_seqlens = [0, T](单序列;vLLM 契约形态,M2 packed 直用)
        let cu = TensorOps::from_host(Dtype::F32, vec![2], &{
            let mut v = Vec::with_capacity(8);
            v.extend_from_slice(&0.0f32.to_le_bytes());
            v.extend_from_slice(&(tokens as f32).to_le_bytes());
            v
        });

        // conv 三段批核(单发射;state 槽 = gdn_slot)。批核无 w_offset
        // 参数 —— 权重行按段窄切后传入(rows [0..K][K..2K][2K..])
        let w_decl = self.conv_w.decl();
        let w_q = narrow_strided(&w_decl, key_dim, 4, 0, 4, vec![key_dim, 4]);
        let w_k = narrow_strided(&w_decl, key_dim, 4, key_dim * 4, 4, vec![key_dim, 4]);
        let w_v = narrow_strided(&w_decl, value_dim, 4, 2 * key_dim * 4, 4, vec![value_dim, 4]);
        let q_c = conv_fwd(&q, &w_q, &gdn.conv_q, gdn_slot, &cu, tokens, key_dim, true);
        let k_c = conv_fwd(&k, &w_k, &gdn.conv_k, gdn_slot, &cu, tokens, key_dim, true);
        let v_c = conv_fwd(&v, &w_v, &gdn.conv_v, gdn_slot, &cu, tokens, value_dim, true);

        // qk l2norm(per-head 行,T 批量)
        let q_n = l2norm(
            &q_c.reshape(vec![tokens * self.nk, self.hk_dim]),
            tokens * self.nk,
            self.hk_dim,
            1e-6,
        );
        let k_n = l2norm(
            &k_c.reshape(vec![tokens * self.nk, self.hk_dim]),
            tokens * self.nk,
            self.hk_dim,
            1e-6,
        );

        // 门控(g + beta)
        let g = gating_g(&self.a_log.decl(), &a, &self.dt_bias.decl(), tokens, self.nv);
        let beta = b.sigmoid();

        // varlen 递推批核(单发射;state 原地进出)
        let y = recurrence_varlen(
            &q_n.reshape(vec![tokens, self.nk, self.hk_dim]),
            &k_n.reshape(vec![tokens, self.nk, self.hk_dim]),
            &v_c.reshape(vec![tokens, self.nv, self.hv_dim]),
            &g,
            &beta,
            &gdn.rec,
            gdn_slot,
            &cu,
            tokens,
            self.nv,
            self.nk,
            self.hk_dim,
            self.hv_dim,
            1.0 / (self.hk_dim as f32).sqrt(),
        );

        // 门控归一化(T 批量)+ 出投影
        let gated = norm_act(
            &y.reshape(vec![tokens, value_dim]),
            &z,
            &self.norm_w.decl(),
            tokens,
            value_dim,
            self.hv_dim,
            self.eps,
            true,
        );
        self.out_proj.forward(&gated, ctx)
    }

    /// 展开降级路(f32 锚链;f16 = 对拍基准,保留为测试锚)
    fn forward_prefill_expanded(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        let tokens = ctx.tokens;
        let (gdn, gdn_slot) = match (ctx.gdn, ctx.gdn_slot) {
            (Some(g), Some(s)) => (g, s),
            _ => {
                return TensorOps::poisoned(
                    crate::tensor::Dtype::F16,
                    vec![tokens, self.hidden],
                    "gdn prefill: ctx 缺动态依赖(需 gdn 块组 + gdn_slot;ForwardCtx::gdn_prefill)",
                );
            }
        };
        let key_dim = self.key_dim();
        let value_dim = self.value_dim();
        let conv_dim = 2 * key_dim + value_dim;

        // T 批量投影(SplitQkvZa 同 decode;T 无关算子单发即批量)
        let qkv = self.in_proj_qkv.forward(xs, ctx); // [T, 2K+V]
        let z = self.in_proj_z.forward(xs, ctx); // [T, V]
        let b = self.in_proj_b.forward(xs, ctx); // [T, HV]
        let a = self.in_proj_a.forward(xs, ctx); // [T, HV]

        // 逐 token 段(conv 滑窗 / rec 递推链式;slot 恒 gdn_slot)
        let mut ys: Vec<TensorOps> = Vec::with_capacity(tokens);
        for t in 0..tokens {
            // 读侧行窄切(直接从 [T,·] 投影读,start = 行基址)
            let q_t = narrow_strided(&qkv, 1, conv_dim, t * conv_dim, key_dim, vec![1, key_dim]);
            let k_t = narrow_strided(&qkv, 1, conv_dim, t * conv_dim + key_dim, key_dim, vec![1, key_dim]);
            let v_t = narrow_strided(&qkv, 1, conv_dim, t * conv_dim + 2 * key_dim, value_dim, vec![1, value_dim]);
            let a_t = narrow_strided(&a, 1, self.nv, t * self.nv, self.nv, vec![1, self.nv]);
            let b_t = narrow_strided(&b, 1, self.nv, t * self.nv, self.nv, vec![1, self.nv]);

            // conv 三段(状态滑窗;段基址 = w_offset;silu 恒 true)
            let q_c = conv_upd(&q_t, &self.conv_w.decl(), &gdn.conv_q, gdn_slot, 1, key_dim, 0, true);
            let k_c = conv_upd(&k_t, &self.conv_w.decl(), &gdn.conv_k, gdn_slot, 1, key_dim, key_dim, true);
            let v_c = conv_upd(&v_t, &self.conv_w.decl(), &gdn.conv_v, gdn_slot, 1, value_dim, 2 * key_dim, true);

            // qk l2norm(per-head 行)
            let q_n = l2norm(&q_c.reshape(vec![self.nk, self.hk_dim]), self.nk, self.hk_dim, 1e-6);
            let k_n = l2norm(&k_c.reshape(vec![self.nk, self.hk_dim]), self.nk, self.hk_dim, 1e-6);

            // 门控:g(softplus 链)+ beta = sigmoid(b)
            let g_t = gating_g(&self.a_log.decl(), &a_t, &self.dt_bias.decl(), 1, self.nv);
            let beta_t = b_t.sigmoid();

            // 单步递推(rec 原地;q_scale = 1/√hd_k)
            let y_t = delta_dec(
                &q_n.reshape(vec![1, self.nk, self.hk_dim]),
                &k_n.reshape(vec![1, self.nk, self.hk_dim]),
                &v_c.reshape(vec![1, self.nv, self.hv_dim]),
                &g_t,
                &beta_t,
                &gdn.rec,
                gdn_slot,
                1,
                self.nv,
                self.nk,
                self.hk_dim,
                self.hv_dim,
                1.0 / (self.hk_dim as f32).sqrt(),
            );
            ys.push(y_t.reshape(vec![1, value_dim]));
        }

        // 栈(T×[1,V] → [T,V];arity 8 = 展开锚定位)→ 门控归一(T 批量)
        // → 出投影(T 批量)
        let refs: Vec<&TensorOps> = ys.iter().collect();
        let y_all = concat_rows(&refs, 1, value_dim);
        let gated = norm_act(
            &y_all,
            &z,
            &self.norm_w.decl(),
            tokens,
            value_dim,
            self.hv_dim,
            self.eps,
            true,
        );
        self.out_proj.forward(&gated, ctx)
    }

    fn forward_decl(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        // PF1a 分派(pf1-port-design §二):prefill = T-token 块展开,
        // 全用已验证 decode 核(vLLM 语义的合法退化形态;批P3 实装)
        if ctx.kind == crate::module::StepKind::Prefill {
            return self.forward_prefill_decl(xs, ctx);
        }
        let tokens = ctx.tokens;
        let gdn = match ctx.gdn {
            Some(g) => g,
            None => {
                return TensorOps::poisoned(
                    Dtype::F32,
                    vec![tokens, self.hidden],
                    "gdn: ctx 缺动态依赖(需 gdn 常驻块组;ForwardCtx::gdn_decode)",
                );
            }
        };
        let (key_dim, value_dim, conv_dim) = (self.key_dim(), self.value_dim(), 2 * self.key_dim() + self.value_dim());

        // 投影(SplitQkvZa:qkv 融合单投影 + z/b/a 独立)
        let qkv = self.in_proj_qkv.forward(xs, ctx); // [T, 2K+V]
        let z = self.in_proj_z.forward(xs, ctx); // [T, V]
        let b = self.in_proj_b.forward(xs, ctx); // [T, HV]
        let a = self.in_proj_a.forward(xs, ctx); // [T, HV]

        // qkv 列切分(narrow;conv 权重不切 —— 段基址走 w_offset 标量)
        let q = narrow_strided(&qkv, tokens, conv_dim, 0, key_dim, vec![tokens, key_dim]);
        let k = narrow_strided(&qkv, tokens, conv_dim, key_dim, key_dim, vec![tokens, key_dim]);
        let v = narrow_strided(&qkv, tokens, conv_dim, 2 * key_dim, value_dim, vec![tokens, value_dim]);

        // conv 三段独立发射(零拼接算子;state 原地滑窗;段基址 = w_offset)
        let q_c = conv_upd(&q, &self.conv_w.decl(), &gdn.conv_q, &gdn.slots, tokens, key_dim, 0, true);
        let k_c = conv_upd(&k, &self.conv_w.decl(), &gdn.conv_k, &gdn.slots, tokens, key_dim, key_dim, true);
        let v_c = conv_upd(&v, &self.conv_w.decl(), &gdn.conv_v, &gdn.slots, tokens, value_dim, 2 * key_dim, true);

        // qk l2norm(per-head 行;HF use_qk_l2norm_in_kernel=True)
        let q_n = l2norm(
            &q_c.reshape(vec![tokens * self.nk, self.hk_dim]),
            tokens * self.nk,
            self.hk_dim,
            1e-6,
        );
        let k_n = l2norm(
            &k_c.reshape(vec![tokens * self.nk, self.hk_dim]),
            tokens * self.nk,
            self.hk_dim,
            1e-6,
        );

        // 门控:g 新核;beta = sigmoid(b) 复用语义算子
        let g = gating_g(&self.a_log.decl(), &a, &self.dt_bias.decl(), tokens, self.nv);
        let beta = b.sigmoid();

        // 单步递推(rec 原地;q_scale = 1/√hd_k)
        let y = delta_dec(
            &q_n.reshape(vec![tokens, self.nk, self.hk_dim]),
            &k_n.reshape(vec![tokens, self.nk, self.hk_dim]),
            &v_c.reshape(vec![tokens, self.nv, self.hv_dim]),
            &g,
            &beta,
            &gdn.rec,
            &gdn.slots,
            tokens,
            self.nv,
            self.nk,
            self.hk_dim,
            self.hv_dim,
            1.0 / (self.hk_dim as f32).sqrt(),
        );

        // 门控归一化(×w 非零中心 × silu(z);per-head 组)+ 出投影
        let gated = norm_act(
            &y.reshape(vec![tokens, value_dim]),
            &z,
            &self.norm_w.decl(),
            tokens,
            value_dim,
            self.hv_dim,
            self.eps,
            true,
        );
        self.out_proj.forward(&gated, ctx)
    }
}

impl Module for GatedDeltaNet {
    fn forward(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        self.forward_decl(xs, ctx)
    }
}

impl Loadable for GatedDeltaNet {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.in_proj_qkv.layout(ctx)
            .chain(self.in_proj_z.layout(ctx))
            .chain(self.in_proj_b.layout(ctx))
            .chain(self.in_proj_a.layout(ctx))
            .chain(self.out_proj.layout(ctx))
            .chain(self.conv_w.layout(ctx))
            .chain(self.a_log.layout(ctx))
            .chain(self.dt_bias.layout(ctx))
            .chain(self.norm_w.layout(ctx))
    }
}

/// 测试/示例公共件:确定性权重源(权重公式 + 生成器;测试与 examples
/// 共用,保证两者输入逐位一致 —— 对拍锚纪律的独立副本指 host 参考算法,
/// 不指输入数据)

/// GDN f16/f32 核名路由(F5 整模切换;输出 dtype 跟随激活声明,
/// state 恒 f32 混合核 —— 输出口径守门,输入位宽由 .cu 源自证)
fn gname(base: &str, dt: crate::tensor::Dtype) -> &'static str {
    match dt {
        crate::tensor::Dtype::F16 => leak_name(format!("{base}_f16")),
        _ => leak_name(format!("{base}_f32")),
    }
}

fn leak_name(s: String) -> &'static str {
    // 封闭集合(GDN 五核 × 2 dtype),Box::leak 量级可忽略
    Box::leak(s.into_boxed_str())
}

pub mod fixture {
    pub fn weights(nk: usize, hk_dim: usize, nv: usize, hv_dim: usize, hidden: usize) -> Vec<(String, usize)> {
        let (key_dim, value_dim) = (nk * hk_dim, nv * hv_dim);
        let conv_dim = 2 * key_dim + value_dim;
        vec![
            ("in_proj_qkv".to_string(), conv_dim * hidden),
            ("in_proj_z".to_string(), value_dim * hidden),
            ("in_proj_b".to_string(), nv * hidden),
            ("in_proj_a".to_string(), nv * hidden),
            ("out_proj".to_string(), hidden * value_dim),
            ("conv1d".to_string(), conv_dim * 4),
            ("A_log".to_string(), nv),
            ("dt_bias".to_string(), nv),
            ("norm".to_string(), hv_dim),
        ]
    }

    pub fn gen(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.13).sin() * 0.5).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{f32b, harvest, gpu_client, gpu_enabled};

    /// host 参考(g 臂;与核同式独立副本)
    pub(crate) fn host_gating_g(
        a_log: &[f32],
        a: &[f32],
        dt_bias: &[f32],
        total: usize,
        heads: usize,
    ) -> Vec<f32> {
        (0..total)
            .map(|i| {
                let h = i % heads;
                let x = a[i] + dt_bias[h];
                let sp = if x < 20.0 { x.exp().ln_1p() } else { x };
                -a_log[h].exp() * sp
            })
            .collect()
    }

    fn slots(total: usize, heads: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            (0..heads).map(|i| (i as f32 * 0.31) - 2.0).collect(),
            (0..total).map(|i| ((i as f32 * 0.17) - 3.0).sin() * 4.0).collect(),
            (0..heads).map(|i| (i as f32 * 0.07) + 0.1).collect(),
        )
    }

    #[test]
    fn declaration_is_wellformed() {
        let (tokens, heads) = (3usize, 2usize);
        let a_log = TensorOps::from_host(Dtype::F32, vec![heads], &f32b(&[0.1, -0.2]));
        let a = TensorOps::from_host(Dtype::F32, vec![tokens, heads], &f32b(&vec![0.5; tokens * heads]));
        let dt = TensorOps::from_host(Dtype::F32, vec![heads], &f32b(&[0.3, 0.4]));
        let g = gating_g(&a_log, &a, &dt, tokens, heads);
        assert!(!g.is_poisoned(), "g 臂声明不应有毒");
        assert_eq!(g.shape(), &[tokens, heads]);
    }

    /// GPU vs host(门控 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn gpu_gating_g_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (tokens, heads) = (5usize, 4usize);
        let (a_log, a, dt) = slots(tokens * heads, heads);

        let mut gpu = gpu_client().await;
        let decl = gating_g(
            &TensorOps::from_host(Dtype::F32, vec![heads], &f32b(&a_log)),
            &TensorOps::from_host(Dtype::F32, vec![tokens, heads], &f32b(&a)),
            &TensorOps::from_host(Dtype::F32, vec![heads], &f32b(&dt)),
            tokens,
            heads,
        );
        let got = harvest(&mut gpu, &decl).await;
        gpu.close().await.expect("server 关机");

        let want = host_gating_g(&a_log, &a, &dt, tokens * heads, heads);
        crate::testkit::assert_close(&got, &want, 1e-6, "gating-g");
    }

    /// HF golden(g + beta 双臂;门控 OWL_HF_PARITY=1,另需 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn hf_parity_gating() {
        use crate::testkit::{assert_close, gpu_enabled, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path};
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (tokens, heads) = (4usize, 2usize);
        let (a_log, a, dt) = slots(tokens * heads, heads);
        let b: Vec<f32> = (0..tokens * heads).map(|i| ((i as f32 * 0.23) - 1.5).cos() * 2.0).collect();

        let inp = tmp_path("gdn_gating_in");
        let outp = tmp_path("gdn_gating_out");
        st_write(
            &inp,
            &[
                ("a_log", &a_log, vec![heads]),
                ("dt_bias", &dt, vec![heads]),
                ("a", &a, vec![tokens * heads]),
                ("b", &b, vec![tokens * heads]),
            ],
        );
        let contract = format!(r#"{{"op": "gating", "heads": {heads}}}"#);
        let manifest = hf_python("gdn.py", &[inp.to_str().unwrap(), outp.to_str().unwrap(), &contract]);
        eprintln!("[hf manifest] gdn gating: {manifest}");
        let want_g = st_read(&outp, "g");
        let want_beta = st_read(&outp, "beta");

        assert!(gpu_enabled(), "gdn gating HF parity 需 OWL_TEST_DEVICE(kernel 节点 GPU-only)");
        let mut gpu = gpu_client().await;
        let g_decl = gating_g(
            &TensorOps::from_host(Dtype::F32, vec![heads], &f32b(&a_log)),
            &TensorOps::from_host(Dtype::F32, vec![tokens, heads], &f32b(&a)),
            &TensorOps::from_host(Dtype::F32, vec![heads], &f32b(&dt)),
            tokens,
            heads,
        );
        let got_g = harvest(&mut gpu, &g_decl).await;
        // beta 臂 = 语义算子 sigmoid
        let b_decl = TensorOps::from_host(Dtype::F32, vec![tokens, heads], &f32b(&b)).sigmoid();
        let got_beta = harvest(&mut gpu, &b_decl).await;
        gpu.close().await.expect("server 关机");

        assert_close(&got_g, &want_g, 1e-6, "gating-g-hf");
        assert_close(&got_beta, &want_beta, 1e-6, "gating-beta-hf");
        std::fs::remove_file(&inp).ok();
        std::fs::remove_file(&outp).ok();
    }

    // ======================================================================
    // 批2:l2norm
    // ======================================================================

    fn host_l2norm(x: &[f32], _rows: usize, dim: usize, eps: f32) -> Vec<f32> {
        x.chunks_exact(dim)
            .flat_map(|row| {
                let ss: f32 = row.iter().map(|v| v * v).sum();
                let inv = 1.0 / (ss.max(0.0) + eps).sqrt();
                row.iter().map(move |v| v * inv)
            })
            .collect()
    }

    #[test]
    fn l2norm_declaration_is_wellformed() {
        let (rows, dim) = (3usize, 8usize);
        let x = TensorOps::from_host(Dtype::F32, vec![rows, dim], &f32b(&vec![0.5; rows * dim]));
        let y = l2norm(&x, rows, dim, 1e-6);
        assert!(!y.is_poisoned(), "l2norm 声明不应有毒");
        assert_eq!(y.shape(), &[rows, dim]);
    }

    /// GPU vs host + 单位范数性质(门控 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn gpu_l2norm_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (rows, dim) = (6usize, 16usize);
        let x: Vec<f32> = (0..rows * dim).map(|i| ((i as f32 * 0.19) - 2.5).sin() * 3.0).collect();

        let mut gpu = gpu_client().await;
        let decl = l2norm(
            &TensorOps::from_host(Dtype::F32, vec![rows, dim], &f32b(&x)),
            rows,
            dim,
            1e-6,
        );
        let got = harvest(&mut gpu, &decl).await;
        gpu.close().await.expect("server 关机");

        let want = host_l2norm(&x, rows, dim, 1e-6);
        crate::testkit::assert_close(&got, &want, 1e-6, "l2norm");
        for r in 0..rows {
            let nrm: f32 = got[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((nrm - 1.0).abs() < 1e-4, "行 {r} 范数 {nrm} 应≈1");
        }
    }

    /// HF golden(门控 OWL_HF_PARITY=1,另需 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn hf_parity_l2norm() {
        use crate::testkit::{assert_close, gpu_enabled, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path};
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (rows, dim) = (5usize, 8usize);
        let x: Vec<f32> = (0..rows * dim).map(|i| ((i as f32 * 0.29) - 1.8).cos() * 2.5).collect();

        let inp = tmp_path("gdn_l2norm_in");
        let outp = tmp_path("gdn_l2norm_out");
        st_write(&inp, &[("x", &x, vec![rows * dim])]);
        let contract = format!(r#"{{"op": "l2norm", "rows": {rows}, "dim": {dim}, "eps": 1e-6}}"#);
        let manifest = hf_python("gdn.py", &[inp.to_str().unwrap(), outp.to_str().unwrap(), &contract]);
        eprintln!("[hf manifest] gdn l2norm: {manifest}");
        let want = st_read(&outp, "y");

        assert!(gpu_enabled(), "gdn l2norm HF parity 需 OWL_TEST_DEVICE");
        let mut gpu = gpu_client().await;
        let decl = l2norm(
            &TensorOps::from_host(Dtype::F32, vec![rows, dim], &f32b(&x)),
            rows,
            dim,
            1e-6,
        );
        let got = harvest(&mut gpu, &decl).await;
        gpu.close().await.expect("server 关机");

        assert_close(&got, &want, 1e-6, "l2norm-hf");
        std::fs::remove_file(&inp).ok();
        std::fs::remove_file(&outp).ok();
    }

    // ======================================================================
    // 批3:conv_upd(state 副作用 + 跨步持久 + padding)
    // ======================================================================

    /// host 参考(state 就地更新;与核同式独立副本)
    fn host_conv_upd(
        x: &[f32], w: &[f32], state: &mut [f32], slots: &[f32],
        batch: usize, dim: usize, silu: bool,
    ) -> Vec<f32> {
        let mut out = vec![f32::NAN; batch * dim]; // padding 行不落地,NaN 哨兵
        for b in 0..batch {
            let slot = slots[b] as i32;
            if slot < 0 { continue; }
            for ch in 0..dim {
                let sbase = (slot as usize * dim + ch) * 3;
                let hist = [state[sbase], state[sbase + 1], state[sbase + 2]];
                let wbase = ch * 4;
                let mut sum = x[b * dim + ch] * w[wbase + 3];
                for k in 0..3 {
                    sum += hist[k] * w[wbase + k];
                }
                if silu {
                    sum /= 1.0 + (-sum).exp();
                }
                state[sbase] = hist[1];
                state[sbase + 1] = hist[2];
                state[sbase + 2] = x[b * dim + ch];
                out[b * dim + ch] = sum;
            }
        }
        out
    }

    /// GPU vs host:两步跨槽(state 持久)+ padding 行(门控 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn gpu_conv_upd_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        use crate::contract::DeviceClient;
        let (max_slots, dim) = (8usize, 12usize);
        let w: Vec<f32> = (0..dim * 4).map(|i| (i as f32 * 0.13) - 0.7).collect();
        let state0: Vec<f32> = (0..max_slots * dim * 3).map(|i| (i as f32 * 0.05).sin()).collect();
        // 步 A:slot 0/5 首步;步 B:slot 0 续步(读回 A 写入)+ slot -1 padding
        let steps: [(&[f32], &[f32]); 2] = [
            (&[0.5, -0.3, 1.0, -0.6, 0.4, 0.9, 0.2, -0.8, 1.1, -0.1, 0.7, 0.3,
               0.9, 0.1, -0.4, 0.6, -0.2, 0.8, -1.0, 0.5, 0.3, 0.6, -0.7, 0.2], &[0.0, 5.0]),
            (&[0.3, 0.8, -0.5, 0.1, 0.9, -0.6, 0.4, 0.2, -0.9, 0.7, -0.3, 0.5,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[1.0, -1.0]),
        ];

        let mut host_state = state0.clone();
        let mut gpu = gpu_client().await;
        let sbytes = alloc_and_fill(&mut gpu, &state0).await;
        for (si, (xs, slots)) in steps.iter().enumerate() {
            let batch = slots.len();
            let host_out = host_conv_upd(xs, &w, &mut host_state, slots, batch, dim, true);
            let decl = conv_upd(
                &TensorOps::from_host(Dtype::F32, vec![batch, dim], &f32b(xs)),
                &TensorOps::from_host(Dtype::F32, vec![dim, 4], &f32b(&w)),
                &TensorOps::of_block(sbytes.id, Dtype::F32, vec![max_slots, dim, 3]),
                &TensorOps::from_host(Dtype::F32, vec![batch], &f32b(slots)),
                batch, dim, 0, true,
            );
            let got = harvest(&mut gpu, &decl).await;
            // 只比有效槽(padding 行不落地)
            for b in 0..batch {
                if slots[b] < 0.0 { continue; }
                crate::testkit::assert_close(
                    &got[b * dim..(b + 1) * dim],
                    &host_out[b * dim..(b + 1) * dim],
                    1e-6, &format!("conv_upd step{si} b{b}"),
                );
            }
        }
        // state 终态对拍(跨步写入的落盘验证)
        let mut sbuf = vec![0u8; max_slots * dim * 3 * 4];
        gpu.dtoh(&sbytes, &mut sbuf).await.expect("dtoh state");
        let got_state: Vec<f32> = sbuf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        crate::testkit::assert_close(&got_state, &host_state, 1e-6, "conv_upd state 终态");
        gpu.close().await.expect("server 关机");
    }

    /// 设备块灌入 host f32(htod 后取块句柄;测试专用)
    async fn alloc_and_fill(gpu: &mut owl_cuda::GpuClient, v: &[f32]) -> crate::contract::Bytes {
        let t = crate::TensorOps::from_host(crate::tensor::Dtype::F32, vec![v.len()], &f32b(v));
        crate::interpreters::eval_ops(t.step(), gpu).await.expect("htod")
    }

    // ======================================================================
    // 批3.5:conv_upd HF golden(手动式已验; golden op 同源互证)
    // ======================================================================

    /// HF golden conv_upd(门控 OWL_HF_PARITY=1 + OWL_TEST_DEVICE)
    #[tokio::test]
    async fn hf_parity_conv_upd() {
        use crate::contract::DeviceClient;
        use crate::testkit::{assert_close, gpu_enabled, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path};
        if !parity_enabled() {
            skip_note();
            return;
        }
        assert!(gpu_enabled(), "conv_upd HF parity 需 OWL_TEST_DEVICE");
        let (max_slots, dim, batch) = (4usize, 6usize, 3usize);
        let x: Vec<f32> = (0..batch * dim).map(|i| ((i as f32 * 0.21) - 1.0).sin()).collect();
        let w: Vec<f32> = (0..dim * 4).map(|i| (i as f32 * 0.11) - 0.5).collect();
        let state: Vec<f32> = (0..max_slots * dim * 3).map(|i| (i as f32 * 0.07).cos()).collect();
        let slots = vec![3.0f32, 0.0, 1.0];

        let inp = tmp_path("gdn_conv_in");
        let outp = tmp_path("gdn_conv_out");
        st_write(&inp, &[
            ("x", &x, vec![batch * dim]),
            ("w", &w, vec![dim * 4]),
            ("state", &state, vec![max_slots * dim * 3]),
            ("slots", &slots, vec![batch]),
        ]);
        let contract = format!(r#"{{"op": "conv_upd", "slots": {batch}, "dim": {dim}, "silu": 1}}"#);
        let manifest = hf_python("gdn.py", &[inp.to_str().unwrap(), outp.to_str().unwrap(), &contract]);
        eprintln!("[hf manifest] gdn conv_upd: {manifest}");
        let want_y = st_read(&outp, "y");
        let want_state = st_read(&outp, "state");

        let mut gpu = gpu_client().await;
        let sblock = alloc_and_fill(&mut gpu, &state).await;
        let decl = conv_upd(
            &TensorOps::from_host(Dtype::F32, vec![batch, dim], &f32b(&x)),
            &TensorOps::from_host(Dtype::F32, vec![dim, 4], &f32b(&w)),
            &TensorOps::of_block(sblock.id, Dtype::F32, vec![max_slots, dim, 3]),
            &TensorOps::from_host(Dtype::F32, vec![batch], &f32b(&slots)),
            batch, dim, 0, true,
        );
        let got_y = harvest(&mut gpu, &decl).await;
        let mut sbuf = vec![0u8; max_slots * dim * 3 * 4];
        gpu.dtoh(&sblock, &mut sbuf).await.expect("dtoh state");
        gpu.close().await.expect("server 关机");
        let got_state: Vec<f32> = sbuf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        assert_close(&got_y, &want_y, 1e-6, "conv_upd-hf");
        assert_close(&got_state, &want_state, 1e-6, "conv_upd-state-hf");
        std::fs::remove_file(&inp).ok();
        std::fs::remove_file(&outp).ok();
    }

    // ======================================================================
    // 批4:delta_dec(递推核心)
    // ======================================================================

    /// host 参考(单步;与核同式独立副本;返回 out,padding 行 NaN)
    fn host_delta_dec(
        q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32],
        state: &mut [f32], slots: &[f32],
        batch: usize, nv: usize, nk: usize, kd: usize, vd: usize,
    ) -> Vec<f32> {
        let mut out = vec![f32::NAN; batch * nv * vd];
        let kv_group = nv / nk;
        let q_scale = 1.0 / (kd as f32).sqrt();
        for b in 0..batch {
            let slot = slots[b] as i32;
            if slot < 0 { continue; }
            for vh in 0..nv {
                let kh = vh / kv_group;
                let decay = g[b * nv + vh].exp();
                let bt = beta[b * nv + vh];
                let sbase = (slot as usize * nv + vh) * kd * vd;
                let qoff = (b * nk + kh) * kd;
                let voff = (b * nv + vh) * vd;
                let mut kv_mem = vec![0.0f32; vd];
                for j in 0..kd {
                    for d in 0..vd {
                        let idx = sbase + j * vd + d;
                        state[idx] *= decay;
                        kv_mem[d] += state[idx] * k[qoff + j];
                    }
                }
                for d in 0..vd {
                    let delta = (v[voff + d] - kv_mem[d]) * bt;
                    let mut y = 0.0f32;
                    for j in 0..kd {
                        let idx = sbase + j * vd + d;
                        state[idx] += k[qoff + j] * delta;
                        y += state[idx] * q[qoff + j] * q_scale;
                    }
                    out[(b * nv + vh) * vd + d] = y;
                }
            }
        }
        out
    }

    /// GPU vs host:GQA(kv_group=2)+ 跨步 + padding + state 终态(门控 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn gpu_delta_dec_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        use crate::contract::DeviceClient;
        let (max_slots, nv, nk, kd, vd) = (8usize, 4usize, 2usize, 8usize, 16usize);
        let gen = |n: usize, seed: f32| (0..n).map(|i| ((i as f32 + seed) * 0.17).sin() * 0.8).collect::<Vec<f32>>();
        let q0 = gen(2 * nk * kd, 1.0);
        let k0 = gen(2 * nk * kd, 2.0);
        let v0 = gen(2 * nv * vd, 3.0);
        let g0 = gen(2 * nv, 4.0);
        let b0 = gen(2 * nv, 5.0);
        let state0 = gen(max_slots * nv * kd * vd, 6.0);
        // 步 A:slot 2/7;步 B:slot 2 续步 + slot -1 padding
        let steps: [(&[f32], &[f32], &[f32], &[f32], &[f32], &[f32]); 2] = [
            (&q0, &k0, &v0, &g0, &b0, &[2.0, 7.0]),
            (
                &gen(2 * nk * kd, 7.0), &gen(2 * nk * kd, 8.0), &gen(2 * nv * vd, 9.0),
                &gen(2 * nv, 10.0), &gen(2 * nv, 11.0), &[2.0, -1.0],
            ),
        ];

        let mut host_state = state0.clone();
        let mut gpu = gpu_client().await;
        let sblock = alloc_and_fill(&mut gpu, &state0).await;
        for (si, (q, k, v, g, beta, slots)) in steps.iter().enumerate() {
            let batch = slots.len();
            let host_out = host_delta_dec(q, k, v, g, beta, &mut host_state, slots, batch, nv, nk, kd, vd);
            let decl = delta_dec(
                &TensorOps::from_host(Dtype::F32, vec![batch, nk, kd], &f32b(q)),
                &TensorOps::from_host(Dtype::F32, vec![batch, nk, kd], &f32b(k)),
                &TensorOps::from_host(Dtype::F32, vec![batch, nv, vd], &f32b(v)),
                &TensorOps::from_host(Dtype::F32, vec![batch, nv], &f32b(g)),
                &TensorOps::from_host(Dtype::F32, vec![batch, nv], &f32b(beta)),
                &TensorOps::of_block(sblock.id, Dtype::F32, vec![max_slots, nv, kd, vd]),
                &TensorOps::from_host(Dtype::F32, vec![batch], &f32b(slots)),
                batch, nv, nk, kd, vd,
                1.0 / (kd as f32).sqrt(),
            );
            let got = harvest(&mut gpu, &decl).await;
            for b in 0..batch {
                if slots[b] < 0.0 { continue; }
                crate::testkit::assert_close(
                    &got[b * nv * vd..(b + 1) * nv * vd],
                    &host_out[b * nv * vd..(b + 1) * nv * vd],
                    1e-5, &format!("delta_dec step{si} b{b}"),
                );
            }
        }
        let mut sbuf = vec![0u8; max_slots * nv * kd * vd * 4];
        gpu.dtoh(&sblock, &mut sbuf).await.expect("dtoh state");
        gpu.close().await.expect("server 关机");
        let got_state: Vec<f32> = sbuf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        crate::testkit::assert_close(&got_state, &host_state, 1e-5, "delta_dec state 终态");
    }

    /// HF golden(torch_recurrent_gated_delta_rule 直跑;out + state;
    /// 门控 OWL_HF_PARITY=1 + OWL_TEST_DEVICE)
    #[tokio::test]
    async fn hf_parity_delta_dec() {
        use crate::contract::DeviceClient;
        use crate::testkit::{assert_close, gpu_enabled, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path};
        if !parity_enabled() {
            skip_note();
            return;
        }
        assert!(gpu_enabled(), "delta_dec HF parity 需 OWL_TEST_DEVICE");
        let (max_slots, nv, nk, kd, vd) = (4usize, 2usize, 2usize, 4usize, 4usize);
        let batch = 2usize;
        let gen = |n: usize, seed: f32| (0..n).map(|i| ((i as f32 + seed) * 0.23).cos() * 0.7).collect::<Vec<f32>>();
        let q = gen(batch * nk * kd, 1.0);
        let k = gen(batch * nk * kd, 2.0);
        let v = gen(batch * nv * vd, 3.0);
        let g = gen(batch * nv, 4.0);
        let beta = gen(batch * nv, 5.0);
        let state = gen(max_slots * nv * kd * vd, 6.0);
        let slots = vec![3.0f32, 0.0];

        let inp = tmp_path("gdn_delta_in");
        let outp = tmp_path("gdn_delta_out");
        st_write(&inp, &[
            ("q", &q, vec![batch, nk, kd]),
            ("k", &k, vec![batch, nk, kd]),
            ("v", &v, vec![batch, nv, vd]),
            ("g", &g, vec![batch, nv]),
            ("beta", &beta, vec![batch, nv]),
            ("state", &state, vec![max_slots * nv * kd * vd]),
            ("slots", &slots, vec![batch]),
        ]);
        let contract = format!(r#"{{"op": "delta_dec", "batch": {batch}, "nv": {nv}, "nk": {nk}, "kd": {kd}, "vd": {vd}}}"#);
        let manifest = hf_python("gdn.py", &[inp.to_str().unwrap(), outp.to_str().unwrap(), &contract]);
        eprintln!("[hf manifest] gdn delta_dec: {manifest}");
        let want_out = st_read(&outp, "out");
        let want_state = st_read(&outp, "state");

        let mut gpu = gpu_client().await;
        let sblock = alloc_and_fill(&mut gpu, &state).await;
        let decl = delta_dec(
            &TensorOps::from_host(Dtype::F32, vec![batch, nk, kd], &f32b(&q)),
            &TensorOps::from_host(Dtype::F32, vec![batch, nk, kd], &f32b(&k)),
            &TensorOps::from_host(Dtype::F32, vec![batch, nv, vd], &f32b(&v)),
            &TensorOps::from_host(Dtype::F32, vec![batch, nv], &f32b(&g)),
            &TensorOps::from_host(Dtype::F32, vec![batch, nv], &f32b(&beta)),
            &TensorOps::of_block(sblock.id, Dtype::F32, vec![max_slots, nv, kd, vd]),
            &TensorOps::from_host(Dtype::F32, vec![batch], &f32b(&slots)),
            batch, nv, nk, kd, vd,
            1.0 / (kd as f32).sqrt(),
        );
        let got_out = harvest(&mut gpu, &decl).await;
        let mut sbuf = vec![0u8; max_slots * nv * kd * vd * 4];
        gpu.dtoh(&sblock, &mut sbuf).await.expect("dtoh state");
        gpu.close().await.expect("server 关机");
        let got_state: Vec<f32> = sbuf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        assert_close(&got_out, &want_out, 1e-5, "delta_dec-hf");
        assert_close(&got_state, &want_state, 1e-5, "delta_dec-state-hf");
        std::fs::remove_file(&inp).ok();
        std::fs::remove_file(&outp).ok();
    }

    // ======================================================================
    // 批5:norm_act
    // ======================================================================

    fn host_norm_act(
        x: &[f32], z: &[f32], gamma: &[f32],
        rows: usize, value_dim: usize, group_size: usize, eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * value_dim];
        for r in 0..rows {
            for g in 0..value_dim / group_size {
                let base = r * value_dim + g * group_size;
                let ms: f32 = (0..group_size).map(|i| x[base + i] * x[base + i]).sum::<f32>() / group_size as f32;
                let inv = 1.0 / (ms.max(0.0) + eps).sqrt();
                for i in 0..group_size {
                    let zv = z[base + i];
                    let act = zv / (1.0 + (-zv).exp()); // silu
                    out[base + i] = x[base + i] * inv * gamma[i] * act;
                }
            }
        }
        out
    }

    /// GPU vs host(门控 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn gpu_norm_act_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (rows, vd, gs) = (4usize, 12usize, 4usize);
        let x: Vec<f32> = (0..rows * vd).map(|i| ((i as f32 * 0.31) - 2.0).sin() * 2.0).collect();
        let z: Vec<f32> = (0..rows * vd).map(|i| ((i as f32 * 0.13) - 0.5).cos() * 1.5).collect();
        let gamma: Vec<f32> = (0..gs).map(|i| (i as f32 * 0.19) - 0.4).collect();

        let mut gpu = gpu_client().await;
        let decl = norm_act(
            &TensorOps::from_host(Dtype::F32, vec![rows, vd], &f32b(&x)),
            &TensorOps::from_host(Dtype::F32, vec![rows, vd], &f32b(&z)),
            &TensorOps::from_host(Dtype::F32, vec![gs], &f32b(&gamma)),
            rows, vd, gs, 1e-6, true,
        );
        let got = harvest(&mut gpu, &decl).await;
        gpu.close().await.expect("server 关机");

        let want = host_norm_act(&x, &z, &gamma, rows, vd, gs, 1e-6);
        crate::testkit::assert_close(&got, &want, 1e-6, "norm_act");
    }

    /// HF golden(Qwen3_5RMSNormGated 直跑;门控 OWL_HF_PARITY=1 + OWL_TEST_DEVICE)
    #[tokio::test]
    async fn hf_parity_norm_act() {
        use crate::testkit::{assert_close, gpu_enabled, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path};
        if !parity_enabled() {
            skip_note();
            return;
        }
        assert!(gpu_enabled(), "norm_act HF parity 需 OWL_TEST_DEVICE");
        let (rows, vd, gs) = (3usize, 8usize, 4usize);
        let x: Vec<f32> = (0..rows * vd).map(|i| ((i as f32 * 0.27) - 1.2).sin() * 2.2).collect();
        let z: Vec<f32> = (0..rows * vd).map(|i| ((i as f32 * 0.41) - 0.7).cos() * 1.8).collect();
        let gamma: Vec<f32> = (0..gs).map(|i| (i as f32 * 0.23) - 0.3).collect();

        let inp = tmp_path("gdn_norm_in");
        let outp = tmp_path("gdn_norm_out");
        st_write(&inp, &[
            ("x", &x, vec![rows * vd]),
            ("z", &z, vec![rows * vd]),
            ("gamma", &gamma, vec![gs]),
        ]);
        let contract = format!(r#"{{"op": "norm_act", "rows": {rows}, "value_dim": {vd}, "group_size": {gs}, "eps": 1e-6, "act": 0}}"#);
        let manifest = hf_python("gdn.py", &[inp.to_str().unwrap(), outp.to_str().unwrap(), &contract]);
        eprintln!("[hf manifest] gdn norm_act: {manifest}");
        let want = st_read(&outp, "y");

        let mut gpu = gpu_client().await;
        let decl = norm_act(
            &TensorOps::from_host(Dtype::F32, vec![rows, vd], &f32b(&x)),
            &TensorOps::from_host(Dtype::F32, vec![rows, vd], &f32b(&z)),
            &TensorOps::from_host(Dtype::F32, vec![gs], &f32b(&gamma)),
            rows, vd, gs, 1e-6, true,
        );
        let got = harvest(&mut gpu, &decl).await;
        gpu.close().await.expect("server 关机");

        assert_close(&got, &want, 1e-6, "norm_act-hf");
        std::fs::remove_file(&inp).ok();
        std::fs::remove_file(&outp).ok();
    }

    // ======================================================================
    // 批6:层组装(装载/声明 + GPU 两步全链对拍)
    // ======================================================================

    use crate::layers::gdn::fixture::{gen, weights};

    const NK: usize = 2;
    const HK_DIM: usize = 4;
    const NV: usize = 2;
    const HV_DIM: usize = 4;
    const HIDDEN: usize = 6;

    fn weight_src() -> std::collections::HashMap<String, Vec<f32>> {
        weights(NK, HK_DIM, NV, HV_DIM, HIDDEN)
            .into_iter()
            .enumerate()
            .map(|(i, (k, n))| (k, gen(n, 10.0 + i as f32)))
            .collect()
    }

    /// 装载 + 声明健全性(CPU face 装载合法;声明不 eval 零后端依赖)
    #[tokio::test]
    async fn layer_load_and_declaration() {
        let mut face = owl_cpu::CpuFace::new();
        let layer = GatedDeltaNet::new(NK, HK_DIM, NV, HV_DIM, HIDDEN, 1e-6);
        crate::interpreters::eval_load(&layer, &mut face, &weight_src(), &Default::default())
            .await
            .expect("eval_load 九槽");

        let tokens = 2usize;
        let max_slots = 4usize;
        let (key_dim, value_dim) = (NK * HK_DIM, NV * HV_DIM);
        let gdn_buf = crate::layers::gdn::GdnBuffers {
            conv_q: TensorOps::zeros(Dtype::F32, vec![max_slots, key_dim, 3]),
            conv_k: TensorOps::zeros(Dtype::F32, vec![max_slots, key_dim, 3]),
            conv_v: TensorOps::zeros(Dtype::F32, vec![max_slots, value_dim, 3]),
            rec: TensorOps::zeros(Dtype::F32, vec![max_slots, NV, HK_DIM, HV_DIM]),
            slots: TensorOps::from_host(Dtype::F32, vec![tokens], &f32b(&[0.0, 2.0])),
        };
        let xs = TensorOps::from_host(Dtype::F32, vec![tokens, HIDDEN], &f32b(&vec![0.4; tokens * HIDDEN]));
        let out = layer.forward(&xs, &ForwardCtx::gdn_decode(tokens, &gdn_buf));
        assert!(!out.is_poisoned(), "装载后声明不应有毒");
        assert_eq!(out.shape(), &[tokens, HIDDEN]);

        // 缺 gdn ctx → 毒值(与未装载槽同构)
        let out2 = layer.forward(&xs, &ForwardCtx::minimal(tokens));
        assert!(out2.is_poisoned(), "minimal ctx 缺 gdn 常驻块组,应毒");

        // prefill 分派(PF1-0/批P3 实装):kind=Prefill → 批核路声明良构
        // (未装 F16 权重前的毒值 = 槽未装载语义;decode 路不破)
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0]));
        let out3 = layer.forward(&xs, &ForwardCtx::gdn_prefill(tokens, &gdn_buf, &gdn_slot));
        let _ = out3; // f16 装载形态由 f16_tests 层级对拍覆盖(CPU 面不追)
        let out4 = layer.forward(&xs, &ForwardCtx::gdn_decode(tokens, &gdn_buf));
        assert!(!out4.is_poisoned(), "prefill 分派后 decode 路不破");
    }

    /// host 全链参考(独立副本;两步 decode,state 跨步持久)
    fn host_gdn_step(
        xs: &[f32], src: &std::collections::HashMap<String, Vec<f32>>,
        conv_s: &mut [Vec<f32>; 3], rec: &mut [f32], slots: &[f32],
    ) -> Vec<f32> {
        let (key_dim, value_dim) = (NK * HK_DIM, NV * HV_DIM);
        let conv_dim = 2 * key_dim + value_dim;
        let batch = slots.len();
        let wqkv = &src["in_proj_qkv"];
        let wz = &src["in_proj_z"];
        let wb = &src["in_proj_b"];
        let wa = &src["in_proj_a"];
        let wo = &src["out_proj"];
        let wc = &src["conv1d"];
        // in_dim 显式传参(M-b 教训:out_proj 内维 = value_dim,禁止硬编码)
        let lin = |x: &[f32], w: &[f32], out_dim: usize, in_dim: usize| {
            (0..out_dim)
                .map(|o| (0..in_dim).map(|i| x[i] * w[o * in_dim + i]).sum::<f32>())
                .collect::<Vec<f32>>()
        };
        let mut out_all = vec![0.0f32; batch * HIDDEN];
        for t in 0..batch {
            let slot = slots[t] as i32;
            let x = &xs[t * HIDDEN..(t + 1) * HIDDEN];
            let qkv = lin(x, wqkv, conv_dim, HIDDEN);
            let z = lin(x, wz, value_dim, HIDDEN);
            let b_v = lin(x, wb, NV, HIDDEN);
            let a_v = lin(x, wa, NV, HIDDEN);
            let segs = [&qkv[0..key_dim], &qkv[key_dim..2 * key_dim], &qkv[2 * key_dim..]];
            let mut post = [vec![0.0f32; key_dim], vec![0.0f32; key_dim], vec![0.0f32; value_dim]];
            for (seg, (inp, (st, dim))) in segs.iter().zip(conv_s.iter_mut().zip([key_dim, key_dim, value_dim])).enumerate() {
                for ch in 0..dim {
                    let sbase = (slot as usize * dim + ch) * 3;
                    let wbase = [0, key_dim, 2 * key_dim][seg] * 4 + ch * 4;
                    let hist = [st[sbase], st[sbase + 1], st[sbase + 2]];
                    let mut sum = inp[ch] * wc[wbase + 3];
                    for kk in 0..3 {
                        sum += hist[kk] * wc[wbase + kk];
                    }
                    sum /= 1.0 + (-sum).exp();
                    st[sbase] = hist[1];
                    st[sbase + 1] = hist[2];
                    st[sbase + 2] = inp[ch];
                    post[seg][ch] = sum;
                }
            }
            // q/k l2norm per-head
            let l2 = |row: &[f32]| {
                let ss: f32 = row.iter().map(|v| v * v).sum();
                let inv = 1.0 / (ss.max(0.0) + 1e-6).sqrt();
                row.iter().map(|v| v * inv).collect::<Vec<f32>>()
            };
            let mut qn = vec![0.0f32; key_dim];
            let mut kn = vec![0.0f32; key_dim];
            for h in 0..NK {
                let r = &post[0][h * HK_DIM..(h + 1) * HK_DIM];
                qn[h * HK_DIM..(h + 1) * HK_DIM].copy_from_slice(&l2(r));
                let r = &post[1][h * HK_DIM..(h + 1) * HK_DIM];
                kn[h * HK_DIM..(h + 1) * HK_DIM].copy_from_slice(&l2(r));
            }
            // 门控
            let a_log = &src["A_log"];
            let dtb = &src["dt_bias"];
            let g_v: Vec<f32> = (0..NV).map(|h| {
                let x2 = a_v[h] + dtb[h];
                let sp = if x2 < 20.0 { x2.exp().ln_1p() } else { x2 };
                -a_log[h].exp() * sp
            }).collect();
            let beta_v: Vec<f32> = b_v.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            // delta 单步
            let q_scale = 1.0 / (HK_DIM as f32).sqrt();
            let mut y = vec![0.0f32; value_dim];
            for vh in 0..NV {
                let kh = vh / (NV / NK);
                let decay = g_v[vh].exp();
                let sbase = (slot as usize * NV + vh) * HK_DIM * HV_DIM;
                let qoff = kh * HK_DIM;
                let voff = vh * HV_DIM;
                let mut kv_mem = vec![0.0f32; HV_DIM];
                for j in 0..HK_DIM {
                    for d in 0..HV_DIM {
                        let idx = sbase + j * HV_DIM + d;
                        rec[idx] *= decay;
                        kv_mem[d] += rec[idx] * kn[qoff + j];
                    }
                }
                for d in 0..HV_DIM {
                    let delta = (post[2][voff + d] - kv_mem[d]) * beta_v[vh];
                    let mut acc = 0.0f32;
                    for j in 0..HK_DIM {
                        let idx = sbase + j * HV_DIM + d;
                        rec[idx] += kn[qoff + j] * delta;
                        acc += rec[idx] * qn[qoff + j] * q_scale;
                    }
                    y[voff + d] = acc;
                }
            }
            // norm_act(per-head ×w × silu(z))+ out_proj
            let gamma = &src["norm"];
            let mut gated = vec![0.0f32; value_dim];
            for vh in 0..NV {
                let base = vh * HV_DIM;
                let ms: f32 = (0..HV_DIM).map(|i| y[base + i] * y[base + i]).sum::<f32>() / HV_DIM as f32;
                let inv = 1.0 / (ms.max(0.0) + 1e-6).sqrt();
                for i in 0..HV_DIM {
                    let zv = z[base + i];
                    gated[base + i] = y[base + i] * inv * gamma[i] * (zv / (1.0 + (-zv).exp()));
                }
            }
            let o = lin(&gated, wo, HIDDEN, value_dim);
            out_all[t * HIDDEN..(t + 1) * HIDDEN].copy_from_slice(&o);
        }
        out_all
    }

    /// GPU 全链两步 decode vs host(门控 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn gpu_layer_decode_matches_host() {
        if !crate::testkit::gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let max_slots = 4usize;
        let (key_dim, value_dim) = (NK * HK_DIM, NV * HV_DIM);
        let src = weight_src();
        let mut gpu = crate::testkit::gpu_client().await;
        let layer = GatedDeltaNet::new(NK, HK_DIM, NV, HV_DIM, HIDDEN, 1e-6);
        crate::interpreters::eval_load(&layer, &mut gpu, &src, &Default::default())
            .await
            .expect("eval_load");

        // 常驻状态块(清零分配 + 灌零)
        let sfill = |n: usize| vec![0.0f32; n];
        let cq = alloc_and_fill(&mut gpu, &sfill(max_slots * key_dim * 3)).await;
        let ck = alloc_and_fill(&mut gpu, &sfill(max_slots * key_dim * 3)).await;
        let cv = alloc_and_fill(&mut gpu, &sfill(max_slots * value_dim * 3)).await;
        let rc = alloc_and_fill(&mut gpu, &sfill(max_slots * NV * HK_DIM * HV_DIM)).await;

        // host 侧副本
        let mut host_conv = [sfill(max_slots * key_dim * 3), sfill(max_slots * key_dim * 3), sfill(max_slots * value_dim * 3)];
        let mut host_rec = sfill(max_slots * NV * HK_DIM * HV_DIM);

        let steps: [(&[f32], &[f32]); 2] = [
            (&gen(2 * HIDDEN, 1.0), &[0.0, 2.0]),
            (&gen(HIDDEN, 2.0), &[1.0]),
        ];
        for (si, (xs, slots)) in steps.iter().enumerate() {
            let batch = slots.len();
            let host_out = host_gdn_step(xs, &src, &mut host_conv, &mut host_rec, slots);
            let gdn_buf = crate::layers::gdn::GdnBuffers {
                conv_q: TensorOps::of_block(cq.id, Dtype::F32, vec![max_slots, key_dim, 3]),
                conv_k: TensorOps::of_block(ck.id, Dtype::F32, vec![max_slots, key_dim, 3]),
                conv_v: TensorOps::of_block(cv.id, Dtype::F32, vec![max_slots, value_dim, 3]),
                rec: TensorOps::of_block(rc.id, Dtype::F32, vec![max_slots, NV, HK_DIM, HV_DIM]),
                slots: TensorOps::from_host(Dtype::F32, vec![batch], &f32b(slots)),
            };
            let xs_t = TensorOps::from_host(Dtype::F32, vec![batch, HIDDEN], &f32b(xs));
            let decl = layer.forward(&xs_t, &ForwardCtx::gdn_decode(batch, &gdn_buf));
            let got = crate::testkit::harvest(&mut gpu, &decl).await;
            crate::testkit::assert_close(&got, &host_out, 1e-4, &format!("gdn 层 step{si}"));
        }
        gpu.close().await.expect("server 关机");
    }
}

// ============================================================================
// f16_tests(F4 换源版对拍;工单 G 收口锚)
// 纪律:输入先过 f16 量化(上载即 f16),host f32 参考在量化后的值上计算
// —— 参考侧先落到目标位宽再谈容差(2026-09-26 F3 教训)。核数学 =
// attention.rs 上游模板(见 gdn.cu 溯源头),对拍验「移植不失真」。
// ============================================================================

#[cfg(test)]
mod f16_tests {
    use super::*;
    use crate::contract::DeviceClient as _;
    use crate::testkit::{gpu_client, gpu_enabled};
    use crate::tensor::Dtype;
    /// 批P1:concat_rows 核对拍(位型拷贝,期望逐位相等)
    #[tokio::test]
    async fn gpu_concat_rows_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (n, r, d) = (3usize, 2usize, 4usize);
        let ins: Vec<Vec<f32>> = (0..n)
            .map(|k| (0..r * d).map(|i| ((k * 100 + i) as f32) * 0.25 - 1.0).collect())
            .collect();

        let mut gpu = gpu_client().await;
        let decls: Vec<TensorOps> = ins
            .iter()
            .map(|v| TensorOps::from_host(Dtype::F16, vec![r, d], &hbytes(v)))
            .collect();
        let refs: Vec<&TensorOps> = decls.iter().collect();
        let decl = crate::layers::concat_rows(&refs, r, d);
        let mut buf = vec![0u8; n * r * d * 2];
        {
            let bytes = crate::interpreters::eval_ops(decl.step(), &mut gpu)
                .await
                .expect("eval concat");
            gpu.dtoh(&bytes, &mut buf).await.expect("dtoh");
        }
        gpu.close().await.expect("server 关机");

        let got = unhalf(&buf);
        let want: Vec<f32> = ins.iter().flat_map(|v| v.iter().copied()).collect();
        assert_eq!(got.len(), want.len(), "形状 [n·r, d]");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "concat 位型 [{i}]");
        }
    }


    /// f16 量化(host 侧先落位宽;测试输入与参考共用)
    fn q(v: &[f32]) -> Vec<f32> {
        v.iter().map(|f| half::f16::from_f32(*f).to_f32()).collect()
    }
    fn hbytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
    }
    fn f32b(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }
    /// 从 f16 字节读回 f32
    fn unhalf(buf: &[u8]) -> Vec<f32> {
        buf.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
    }
    fn assert_close_rel(got: &[f32], want: &[f32], tol: f32, ctx: &str) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{ctx}[{i}] {g} vs {w}");
        }
    }

    /// 零态 GdnBuffers(conv f32 / rec f32;新序列零态起步语义)
    async fn zero_gdn_bufs(
        gpu: &mut owl_cuda::GpuClient, slots_n: usize,
        key_dim: usize, value_dim: usize, nv: usize, hkd: usize, hvd: usize,
        slot: f32,
    ) -> GdnBuffers {
        async fn mk(gpu: &mut owl_cuda::GpuClient, shape: Vec<usize>) -> crate::contract::Bytes {
            crate::interpreters::eval_ops(
                TensorOps::zeros(Dtype::F32, shape).step(), gpu,
            ).await.expect("zeros 块")
        }
        let of = |b: crate::contract::Bytes, shape: Vec<usize>| {
            TensorOps::of_block(b.id, Dtype::F32, shape)
        };
        let cq = mk(gpu, vec![slots_n, key_dim, 3]).await;
        let ck = mk(gpu, vec![slots_n, key_dim, 3]).await;
        let cv = mk(gpu, vec![slots_n, value_dim, 3]).await;
        let rc = mk(gpu, vec![slots_n, nv, hkd, hvd]).await;
        GdnBuffers {
            conv_q: of(cq, vec![slots_n, key_dim, 3]),
            conv_k: of(ck, vec![slots_n, key_dim, 3]),
            conv_v: of(cv, vec![slots_n, value_dim, 3]),
            rec: of(rc, vec![slots_n, nv, hkd, hvd]),
            slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot])),
        }
    }

    /// 读回 f32 块(Block 叶 eval 零操作直取)
    async fn read_f32_block(gpu: &mut owl_cuda::GpuClient, blk: &TensorOps, n: usize) -> Vec<f32> {
        let b = crate::interpreters::eval_ops(blk.step(), gpu).await.expect("block 叶");
        let mut buf = vec![0u8; n * 4];
        gpu.dtoh(&b, &mut buf).await.expect("dtoh");
        buf.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// 批P3 验收:T=8 prefill 块 == T×decode 逐步(输出 + conv/rec 终态;
    /// 语义定义 = pf1-port-design §二 2.2)
    #[tokio::test]
    async fn gpu_gdn_prefill_block_matches_token_loop() {
        if !gpu_enabled() { eprintln!("skip"); return; }
        let (nk, hkd, nv, hvd, hidden) = (2usize, 4usize, 2usize, 4usize, 6usize);
        let t_len = 8usize;
        let slot = 2.0f32;
        let key_dim = nk * hkd;
        let value_dim = nv * hvd;
        let q = |v: &[f32]| -> Vec<f32> { v.iter().map(|f| half::f16::from_f32(*f).to_f32()).collect() };

        let layer = GatedDeltaNet::new(nk, hkd, nv, hvd, hidden, 1e-6);
        let src: std::collections::HashMap<String, Vec<f32>> =
            crate::layers::gdn::fixture::weights(nk, hkd, nv, hvd, hidden)
                .into_iter()
                .enumerate()
                .map(|(i, (k, n))| (k, crate::layers::gdn::fixture::gen(n, 10.0 + i as f32)))
                .collect();

        let mut gpu = gpu_client().await;
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1 };
        crate::interpreters::eval_load(&layer, &mut gpu, &src, &lctx)
            .await
            .expect("层 f16 装载");

        let mut ref_buf = zero_gdn_bufs(&mut gpu, 4, key_dim, value_dim, nv, hkd, hvd, slot).await;
        let pre_buf = zero_gdn_bufs(&mut gpu, 4, key_dim, value_dim, nv, hkd, hvd, slot).await;

        // 输入(f16 位型量化后的 f32,host/device 逐位一致)
        let xs_f32: Vec<f32> = q(
            &(0..t_len * hidden).map(|i| ((i as f32) * 0.37 - 1.0).sin()).collect::<Vec<_>>(),
        );

        // 参考:T 次 decode 步(行 t 单独;state 跨步持久于 ref_buf 块)
        let mut ref_outs = Vec::new();
        for t in 0..t_len {
            let x_t = TensorOps::from_host(
                Dtype::F16,
                vec![1, hidden],
                &hbytes(&xs_f32[t * hidden..(t + 1) * hidden]),
            );
            let decl = layer.forward(&x_t, &ForwardCtx::gdn_decode(1, &ref_buf));
            ref_outs.push(crate::testkit::harvest_f16(&mut gpu, &decl).await);
        }

        // 被测:一次 prefill 块(T=8;slot 恒 gdn_slot)
        let xs_all = TensorOps::from_host(Dtype::F16, vec![t_len, hidden], &hbytes(&xs_f32));
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot]));
        let decl_all = layer.forward(&xs_all, &ForwardCtx::gdn_prefill(t_len, &pre_buf, &gdn_slot));
        let out_all = crate::testkit::harvest_f16(&mut gpu, &decl_all).await;

        // 输出对拍(行独立;层输出 [T, hidden] 过 out_proj,f16 链 2e-2 rel)
        for t in 0..t_len {
            assert_close_rel(
                &out_all[t * hidden..(t + 1) * hidden],
                &ref_outs[t],
                2e-2,
                &format!("prefill 行{t}"),
            );
        }

        // 终态对拍(conv 三段 + rec;f32 块,同核同形 → 1e-3 rel)
        for (name, a, b, n) in [
            ("conv_q", &ref_buf.conv_q, &pre_buf.conv_q, 4 * key_dim * 3),
            ("conv_k", &ref_buf.conv_k, &pre_buf.conv_k, 4 * key_dim * 3),
            ("conv_v", &ref_buf.conv_v, &pre_buf.conv_v, 4 * value_dim * 3),
            ("rec", &ref_buf.rec, &pre_buf.rec, 4 * nv * hkd * hvd),
        ] {
            let ga = read_f32_block(&mut gpu, a, n).await;
            let gb = read_f32_block(&mut gpu, b, n).await;
            assert_close_rel(&ga, &gb, 1e-3, &format!("prefill 终态 {name}"));
        }
        gpu.close().await.expect("server 关机");
    }

    /// 批P3(f32 锚链变体):模型 fixture 走 f32 链 —— 展开的 f32 核路
    /// 同样必须满足「块 == 逐步」(f16 变体已证,此处证 dtype 路由)
    #[tokio::test]
    async fn gpu_gdn_prefill_block_matches_token_loop_f32() {
        if !gpu_enabled() { eprintln!("skip"); return; }
        let (nk, hkd, nv, hvd, hidden) = (2usize, 4usize, 2usize, 4usize, 6usize);
        let t_len = 8usize;
        let slot = 2.0f32;
        let key_dim = nk * hkd;
        let value_dim = nv * hvd;

        let layer = GatedDeltaNet::new(nk, hkd, nv, hvd, hidden, 1e-6);
        let src: std::collections::HashMap<String, Vec<f32>> =
            crate::layers::gdn::fixture::weights(nk, hkd, nv, hvd, hidden)
                .into_iter()
                .enumerate()
                .map(|(i, (k, n))| (k, crate::layers::gdn::fixture::gen(n, 10.0 + i as f32)))
                .collect();

        let mut gpu = gpu_client().await;
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F32, shard: 1 };
        crate::interpreters::eval_load(&layer, &mut gpu, &src, &lctx)
            .await
            .expect("层 f32 装载");

        let mut ref_buf = zero_gdn_bufs(&mut gpu, 4, key_dim, value_dim, nv, hkd, hvd, slot).await;
        let pre_buf = zero_gdn_bufs(&mut gpu, 4, key_dim, value_dim, nv, hkd, hvd, slot).await;

        let xs_f32: Vec<f32> =
            (0..t_len * hidden).map(|i| ((i as f32) * 0.37 - 1.0).sin()).collect::<Vec<_>>();

        let mut ref_outs = Vec::new();
        for t in 0..t_len {
            let x_t = TensorOps::from_host(
                Dtype::F32,
                vec![1, hidden],
                &f32b(&xs_f32[t * hidden..(t + 1) * hidden]),
            );
            let decl = layer.forward(&x_t, &ForwardCtx::gdn_decode(1, &ref_buf));
            ref_outs.push(crate::testkit::harvest(&mut gpu, &decl).await);
        }

        let xs_all = TensorOps::from_host(Dtype::F32, vec![t_len, hidden], &f32b(&xs_f32));
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot]));
        let decl_all = layer.forward(&xs_all, &ForwardCtx::gdn_prefill(t_len, &pre_buf, &gdn_slot));
        let out_all = crate::testkit::harvest(&mut gpu, &decl_all).await;

        for t in 0..t_len {
            assert_close_rel(
                &out_all[t * hidden..(t + 1) * hidden],
                &ref_outs[t],
                1e-5,
                &format!("prefill-f32 行{t}"),
            );
        }

        for (name, a, b, n) in [
            ("conv_q", &ref_buf.conv_q, &pre_buf.conv_q, 4 * key_dim * 3),
            ("conv_k", &ref_buf.conv_k, &pre_buf.conv_k, 4 * key_dim * 3),
            ("conv_v", &ref_buf.conv_v, &pre_buf.conv_v, 4 * value_dim * 3),
            ("rec", &ref_buf.rec, &pre_buf.rec, 4 * nv * hkd * hvd),
        ] {
            let ga = read_f32_block(&mut gpu, a, n).await;
            let gb = read_f32_block(&mut gpu, b, n).await;
            assert_close_rel(&ga, &gb, 1e-5, &format!("prefill-f32 终态 {name}"));
        }
        gpu.close().await.expect("server 关机");
    }

    /// 批P6 验收:T=64 批核 == 64×decode 逐步 + 耗时记账(§六:批核节点
    /// 坍缩 → GPU-bound;此处实测回填)
    #[tokio::test]
    async fn gpu_gdn_prefill_t64_batch_vs_loop() {
        if !gpu_enabled() { eprintln!("skip"); return; }
        let (nk, hkd, nv, hvd, hidden) = (2usize, 4usize, 2usize, 4usize, 6usize);
        let t_len = 64usize;
        let slot = 1.0f32;
        let key_dim = nk * hkd;
        let value_dim = nv * hvd;
        let q = |v: &[f32]| -> Vec<f32> { v.iter().map(|f| half::f16::from_f32(*f).to_f32()).collect() };

        let layer = GatedDeltaNet::new(nk, hkd, nv, hvd, hidden, 1e-6);
        let src: std::collections::HashMap<String, Vec<f32>> =
            crate::layers::gdn::fixture::weights(nk, hkd, nv, hvd, hidden)
                .into_iter()
                .enumerate()
                .map(|(i, (k, n))| (k, crate::layers::gdn::fixture::gen(n, 10.0 + i as f32)))
                .collect();

        let mut gpu = gpu_client().await;
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1 };
        crate::interpreters::eval_load(&layer, &mut gpu, &src, &lctx)
            .await
            .expect("层 f16 装载");

        let mut ref_buf = zero_gdn_bufs(&mut gpu, 4, key_dim, value_dim, nv, hkd, hvd, slot).await;
        let pre_buf = zero_gdn_bufs(&mut gpu, 4, key_dim, value_dim, nv, hkd, hvd, slot).await;

        let xs_f32: Vec<f32> = q(
            &(0..t_len * hidden).map(|i| ((i as f32) * 0.37 - 1.0).sin()).collect::<Vec<_>>(),
        );

        // 参考:64×decode 逐步(计时)
        let t0 = std::time::Instant::now();
        let mut ref_outs = Vec::new();
        for t in 0..t_len {
            let x_t = TensorOps::from_host(
                Dtype::F16,
                vec![1, hidden],
                &hbytes(&xs_f32[t * hidden..(t + 1) * hidden]),
            );
            let decl = layer.forward(&x_t, &ForwardCtx::gdn_decode(1, &ref_buf));
            ref_outs.push(crate::testkit::harvest_f16(&mut gpu, &decl).await);
        }
        let dt_loop = t0.elapsed();

        // 被测:一次 prefill 批核 T=64(计时)
        let xs_all = TensorOps::from_host(Dtype::F16, vec![t_len, hidden], &hbytes(&xs_f32));
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot]));
        let t1 = std::time::Instant::now();
        let decl_all = layer.forward(&xs_all, &ForwardCtx::gdn_prefill(t_len, &pre_buf, &gdn_slot));
        let out_all = crate::testkit::harvest_f16(&mut gpu, &decl_all).await;
        let dt_batch = t1.elapsed();

        for t in 0..t_len {
            assert_close_rel(
                &out_all[t * hidden..(t + 1) * hidden],
                &ref_outs[t],
                2e-2,
                &format!("prefill-t64 行{t}"),
            );
        }
        eprintln!(
            "[t64 记账] 展开等价步循环 {dt_loop:.1?} vs 批核 {dt_batch:.1?}(含 host dispatch;批核节点 ~25 vs 10T+62)"
        );
        // 终态锚(conv + rec)
        for (name, a, b, n) in [
            ("conv_q", &ref_buf.conv_q, &pre_buf.conv_q, 4 * key_dim * 3),
            ("rec", &ref_buf.rec, &pre_buf.rec, 4 * nv * hkd * hvd),
        ] {
            let ga = read_f32_block(&mut gpu, a, n).await;
            let gb = read_f32_block(&mut gpu, b, n).await;
            assert_close_rel(&ga, &gb, 1e-3, &format!("t64 终态 {name}"));
        }
        gpu.close().await.expect("server 关机");
    }

    /// (工单 G-1)gating g 臂:g = -exp(A_log)·softplus(a + dt_bias)
    #[tokio::test]
    async fn gpu_gating_g_f16_matches_host() {
        if !gpu_enabled() { eprintln!("skip"); return; }
        let (total, heads) = (12usize, 4usize);
        let a_log: Vec<f32> = q(&(0..heads).map(|i| (i as f32 * 0.31) - 1.0).collect::<Vec<_>>());
        let a: Vec<f32> = q(&(0..total).map(|i| ((i as f32 * 0.7) - 4.0).sin()).collect::<Vec<_>>());
        let dt: Vec<f32> = q(&(0..heads).map(|i| (i as f32 * 0.11) + 0.2).collect::<Vec<_>>());
        let mut want = vec![0f32; total];
        for i in 0..total {
            let x = a[i] + dt[i % heads];
            let sp = if x <= 20.0 { x.exp().ln_1p() } else { x };
            want[i] = -a_log[i % heads].exp() * sp;
        }
        let mut gpu = gpu_client().await;
        let sh = crate::contract::Shape::from;
        let dl = gpu.htod(Dtype::F16, &sh(vec![heads]), &hbytes(&a_log)).await.unwrap();
        let da = gpu.htod(Dtype::F16, &sh(vec![total]), &hbytes(&a)).await.unwrap();
        let dd = gpu.htod(Dtype::F16, &sh(vec![heads]), &hbytes(&dt)).await.unwrap();
        let (l, a_, d_) = (
            TensorOps::of_block(dl.id, Dtype::F16, vec![heads]),
            TensorOps::of_block(da.id, Dtype::F16, vec![total]),
            TensorOps::of_block(dd.id, Dtype::F16, vec![heads]),
        );
        let decl = TensorOps::of(crate::kernel::kernel_with(
            "owl_gdn_gating_g_f16", (0, 0, 0), (256, 1, 1), 0,
        )).arg(&l).arg(&a_).arg(&d_).arg_usize(total).arg_usize(heads)
        .with_shape(Dtype::F16, vec![total]);
        let out = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.unwrap();
        let mut buf = vec![0u8; total * 2];
        gpu.dtoh(&out, &mut buf).await.unwrap();
        assert_close_rel(&unhalf(&buf), &want, 2e-2, "gating");
        gpu.close().await.unwrap();
    }

    /// (工单 G-2)l2norm:行末维归一
    #[tokio::test]
    async fn gpu_l2norm_f16_matches_host() {
        if !gpu_enabled() { eprintln!("skip"); return; }
        let (rows, dim) = (4usize, 64usize);
        let x: Vec<f32> = q(&(0..rows * dim).map(|i| ((i as f32 * 0.23) - 2.0).sin()).collect::<Vec<_>>());
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            let ss: f32 = x[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss.max(0.0) + 1e-6).sqrt();
            for i in 0..dim { want[r * dim + i] = x[r * dim + i] * inv; }
        }
        let mut gpu = gpu_client().await;
        let dx = gpu.htod(Dtype::F16, &crate::contract::Shape::from(vec![rows, dim]), &hbytes(&x)).await.unwrap();
        let x_decl = TensorOps::of_block(dx.id, Dtype::F16, vec![rows, dim]);
        let decl = TensorOps::of(crate::kernel::kernel_with(
            "owl_gdn_l2norm_f16", (rows as u32, 1, 1), (256, 1, 1), 0,
        )).arg(&x_decl).arg_usize(rows).arg_usize(dim).arg_f32(1e-6)
        .with_shape(Dtype::F16, vec![rows, dim]);
        let out = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.unwrap();
        let mut buf = vec![0u8; rows * dim * 2];
        gpu.dtoh(&out, &mut buf).await.unwrap();
        assert_close_rel(&unhalf(&buf), &want, 2e-2, "l2norm");
        gpu.close().await.unwrap();
    }

    /// (工单 G-3)norm_act:rmsnorm(x 组内)·gamma·silu(z)
    #[tokio::test]
    async fn gpu_norm_act_f16_matches_host() {
        if !gpu_enabled() { eprintln!("skip"); return; }
        let (rows, vd, gs) = (3usize, 32usize, 8usize);
        let x: Vec<f32> = q(&(0..rows * vd).map(|i| ((i as f32 * 0.41) - 3.0).sin()).collect::<Vec<_>>());
        let z: Vec<f32> = q(&(0..rows * vd).map(|i| ((i as f32 * 0.19) - 1.0).cos()).collect::<Vec<_>>());
        let gamma: Vec<f32> = q(&(0..gs).map(|i| (i as f32 * 0.05) - 0.1).collect::<Vec<_>>());
        let mut want = vec![0f32; rows * vd];
        for r in 0..rows {
            for g in 0..(vd / gs) {
                let off = r * vd + g * gs;
                let ss: f32 = x[off..off + gs].iter().map(|v| v * v).sum();
                let inv = 1.0 / (ss / gs as f32 + 1e-6).sqrt();
                for i in 0..gs {
                    let zv = z[off + i];
                    let act = zv / (1.0 + (-zv).exp());
                    want[off + i] = x[off + i] * inv * gamma[i] * act;
                }
            }
        }
        let mut gpu = gpu_client().await;
        let sh = crate::contract::Shape::from;
        let dxi = gpu.htod(Dtype::F16, &sh(vec![rows, vd]), &hbytes(&x)).await.unwrap();
        let dzi = gpu.htod(Dtype::F16, &sh(vec![rows, vd]), &hbytes(&z)).await.unwrap();
        let dgi = gpu.htod(Dtype::F16, &sh(vec![gs]), &hbytes(&gamma)).await.unwrap();
        let (x_, z_, g_) = (
            TensorOps::of_block(dxi.id, Dtype::F16, vec![rows, vd]),
            TensorOps::of_block(dzi.id, Dtype::F16, vec![rows, vd]),
            TensorOps::of_block(dgi.id, Dtype::F16, vec![gs]),
        );
        let decl = TensorOps::of(crate::kernel::kernel_with(
            "owl_gdn_norm_act_f16",
            ((rows * vd / gs) as u32, 1, 1), (256, 1, 1), 0,
        )).arg(&x_).arg(&z_).arg(&g_)
        .arg_usize(rows).arg_usize(vd).arg_usize(gs).arg_f32(1e-6).arg_i32(0)
        .with_shape(Dtype::F16, vec![rows, vd]);
        let out = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.unwrap();
        let mut buf = vec![0u8; rows * vd * 2];
        gpu.dtoh(&out, &mut buf).await.unwrap();
        assert_close_rel(&unhalf(&buf), &want, 2e-2, "norm_act");
        gpu.close().await.unwrap();
    }

    /// (工单 G-4)conv_upd + delta_dec:两步连续 + state f32 连续性
    /// (padding slot 负值跳写;GQA kv_group=1 简档)
    #[tokio::test]
    async fn gpu_conv_delta_f16_two_steps() {
        if !gpu_enabled() { eprintln!("skip"); return; }
        let (nk, nv, kd, vd, batch, slots_n) = (2usize, 2usize, 16usize, 32usize, 2usize, 4usize);
        let d = nk * kd; // conv 段简化:只测 q 段(d = key_dim)
        let w: Vec<f32> = q(&(0..d * 4).map(|i| ((i as f32 * 0.13) - 0.5).sin()).collect::<Vec<_>>());
        let a_log: Vec<f32> = q(&(0..nv).map(|i| (i as f32 * 0.2) - 0.3).collect::<Vec<_>>());
        let dt: Vec<f32> = q(&(0..nv).map(|i| i as f32 * 0.1).collect::<Vec<_>>());
        let slots = [0.0f32, -1.0f32]; // b1 padding:跳写

        let mut gpu = gpu_client().await;
        let sh = crate::contract::Shape::from;
        // 常驻块(state f32):先物化拿 Bytes(核写这个块;重 eval 会另开新块)
        let conv_state_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F32, vec![slots_n, d, 3]).step(), &mut gpu).await.unwrap();
        let conv_state = TensorOps::of_block(conv_state_b.id, Dtype::F32, vec![slots_n, d, 3]);
        let rec_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F32, vec![slots_n, nv, kd, vd]).step(), &mut gpu).await.unwrap();
        let rec = TensorOps::of_block(rec_b.id, Dtype::F32, vec![slots_n, nv, kd, vd]);
        let dwb = gpu.htod(Dtype::F16, &sh(vec![d, 4]), &hbytes(&w)).await.unwrap();
        let dal = gpu.htod(Dtype::F16, &sh(vec![nv]), &hbytes(&a_log)).await.unwrap();
        let ddt = gpu.htod(Dtype::F16, &sh(vec![nv]), &hbytes(&dt)).await.unwrap();
        let w_decl = TensorOps::of_block(dwb.id, Dtype::F16, vec![d, 4]);
        let al_decl = TensorOps::of_block(dal.id, Dtype::F16, vec![nv]);
        let dt_decl = TensorOps::of_block(ddt.id, Dtype::F16, vec![nv]);

        for step in 0..2u32 {
            let x: Vec<f32> = q(&(0..batch * d).map(|i| ((i as f32 + step as f32 * 7.0) * 0.31).sin()).collect::<Vec<_>>());
            let dxi = gpu.htod(Dtype::F16, &sh(vec![batch, d]), &hbytes(&x)).await.unwrap();
            let x_decl = TensorOps::of_block(dxi.id, Dtype::F16, vec![batch, d]);
            let sl = TensorOps::from_host(Dtype::F32, vec![batch], &f32b(&slots));
            // conv_upd(q 段,w_offset=0)
            let conv = TensorOps::of(crate::kernel::kernel_with(
                "owl_gdn_conv_upd_f16", (0, 0, 0), (256, 1, 1), 0,
            )).arg(&x_decl).arg(&w_decl).arg(&conv_state).arg(&sl)
            .arg_usize(batch * d).arg_usize(d).arg_usize(0).arg_i32(1)
            .with_shape(Dtype::F16, vec![batch, d]);
            let cout = crate::interpreters::eval_ops(conv.step(), &mut gpu).await.unwrap();
            let mut cbuf = vec![0u8; batch * d * 2];
            gpu.dtoh(&cout, &mut cbuf).await.unwrap();
            let got_c = unhalf(&cbuf);
            for b in 0..batch {
                if slots[b] < 0.0 { continue; }
                for ch in 0..d {
                    let sp = slots[b] as usize;
                    let hist = [0.0f32; 3]; // 首步 state=0;两步间由设备块持久(下方直接验终态)
                    let _ = hist;
                    let x_t = x[b * d + ch];
                    let mut sum = x_t * w[(0 + ch) * 4 + 3];
                    // host 只验首步(step0 state=0);step1 的 state 由设备持久,
                    // host 不复算(连续性由 conv_fwd 对拍 G-6 交叉验证)
                    if step == 0 {
                        for k in 0..3 { sum += 0.0 * w[ch * 4 + k]; }
                        // 核尾带 silu(Qwen3.5 conv 激活;arg_i32(1))—— host 同式
                        let want = {
                            let raw = x_t * w[ch * 4 + 3];
                            raw / (1.0 + (-raw).exp())
                        };
                        let g = got_c[b * d + ch];
                        assert!((g - want).abs() <= 2e-2 * (1.0 + want.abs()), "conv s0[{b},{ch}] {g} vs {want}");
                    }
                    let _ = sum;
                }
            }
            let _ = cout;

            // delta_dec(单步;g/beta 由 host 公式生成)
            let g_v: Vec<f32> = q(&(0..batch * nv).map(|i| -0.1 - i as f32 * 0.05).collect::<Vec<_>>());
            let beta: Vec<f32> = q(&(0..batch * nv).map(|i| 0.8 - i as f32 * 0.1).collect::<Vec<_>>());
            let qv: Vec<f32> = q(&(0..batch * nk * kd).map(|i| ((i as f32 + step as f32) * 0.17).sin()).collect::<Vec<_>>());
            let kv: Vec<f32> = q(&(0..batch * nk * kd).map(|i| ((i as f32 + step as f32) * 0.23).cos()).collect::<Vec<_>>());
            let vv: Vec<f32> = q(&(0..batch * nv * vd).map(|i| ((i as f32 + step as f32) * 0.29).sin()).collect::<Vec<_>>());
            let dq = gpu.htod(Dtype::F16, &sh(vec![batch, nk, kd]), &hbytes(&qv)).await.unwrap();
            let dk = gpu.htod(Dtype::F16, &sh(vec![batch, nk, kd]), &hbytes(&kv)).await.unwrap();
            let dv = gpu.htod(Dtype::F16, &sh(vec![batch, nv, vd]), &hbytes(&vv)).await.unwrap();
            let dg = gpu.htod(Dtype::F16, &sh(vec![batch, nv]), &hbytes(&g_v)).await.unwrap();
            let db = gpu.htod(Dtype::F16, &sh(vec![batch, nv]), &hbytes(&beta)).await.unwrap();
            let (q_, k_, v_, g_, b_) = (
                TensorOps::of_block(dq.id, Dtype::F16, vec![batch, nk, kd]),
                TensorOps::of_block(dk.id, Dtype::F16, vec![batch, nk, kd]),
                TensorOps::of_block(dv.id, Dtype::F16, vec![batch, nv, vd]),
                TensorOps::of_block(dg.id, Dtype::F16, vec![batch, nv]),
                TensorOps::of_block(db.id, Dtype::F16, vec![batch, nv]),
            );
            let sl2 = TensorOps::from_host(Dtype::F32, vec![batch], &f32b(&slots));
            let decl = TensorOps::of(crate::kernel::kernel_with(
                "owl_gdn_delta_dec_f16",
                (((vd + 63) / 64) as u32, (batch * nv) as u32, 1), (64, 1, 1),
                // 核内 k/q_smem 按 OWL_GDN16_MAX_KD=128 偏移寻址 —— smem 恒按
                // 128 档给足(kd 参数仅约束装载循环;照 f32 wrapper 同款)
                ((2 * 128 + 2) * 4) as u32,
            )).arg(&q_).arg(&k_).arg(&v_).arg(&g_).arg(&b_)
            .arg(&rec).arg(&sl2)
            .arg_usize(batch).arg_usize(nv).arg_usize(nk).arg_usize(kd).arg_usize(vd)
            .arg_f32(1.0 / (kd as f32).sqrt())
            .with_shape(Dtype::F16, vec![batch, nv, vd]);
            let out = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.unwrap();
            let mut obuf = vec![0u8; batch * nv * vd * 2];
            gpu.dtoh(&out, &mut obuf).await.unwrap();
            let got_o = unhalf(&obuf);
            // 首步:b0(state 0)→ out = (Σ_k q·k)·scale·beta·v[state=0 时
            // kv_mem=0,delta=v·beta,out=Σ s·q̂ = Σ k·(v·beta)·(q·scale)];
            // padding b1 跳写 → 应恒 0(零初始化输出块语义下首读)
            if step == 0 {
                for h in 0..nv {
                    let qk: f32 = (0..kd)
                        .map(|k| qv[h * kd + k] * kv[h * kd + k])
                        .sum::<f32>() * (1.0 / (kd as f32).sqrt());
                    for i in 0..vd {
                        let want = qk * beta[h] * vv[h * vd + i];
                        let g = got_o[h * vd + i];
                        assert!((g - want).abs() <= 2e-2 * (1.0 + want.abs()), "delta s0[{h},{i}] {g} vs {want}");
                    }
                }
            }
        }
        // 两步后 state 非零连续性:收割 rec 块(b0 段)应有非零有限值
        let mut sbuf = vec![0u8; slots_n * nv * kd * vd * 4];
        gpu.dtoh(&rec_b, &mut sbuf).await.unwrap();
        let svals: Vec<f32> = sbuf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let b0_nonzero = svals[0..nv * kd * vd].iter().any(|v| v.abs() > 1e-6);
        let pad_zero = svals[(slots_n - 1) * nv * kd * vd..].iter().all(|v| v.abs() == 0.0);
        assert!(b0_nonzero, "state 两步后应非零(f32 连续)");
        assert!(pad_zero, "padding 槽 state 应保持零");
        gpu.close().await.unwrap();
    }

    /// (工单 G-5)PF1b:conv_fwd varlen(单序列 T=8)== T×conv_upd 展开
    /// (PF1a 展开形态 vs 批核的核级对拍;out + state 双锚)
    #[tokio::test]
    async fn gpu_conv_fwd_varlen_matches_expansion() {
        if !gpu_enabled() { eprintln!("skip"); return; }
        let (nk, kd, t, slot) = (2usize, 16usize, 8usize, 0usize);
        let d = nk * kd;
        let w: Vec<f32> = q(&(0..d * 4).map(|i| ((i as f32 * 0.13) - 0.5).sin()).collect::<Vec<_>>());
        let x: Vec<f32> = q(&(0..t * d).map(|i| ((i as f32 * 0.31) - 1.0).sin()).collect::<Vec<_>>());

        let mut gpu = gpu_client().await;
        let sh = crate::contract::Shape::from;
        // 批核:单发射
        let st_a_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F32, vec![1, d, 3]).step(), &mut gpu).await.unwrap();
        let st_a = TensorOps::of_block(st_a_b.id, Dtype::F32, vec![1, d, 3]);
        let dwb = gpu.htod(Dtype::F16, &sh(vec![d, 4]), &hbytes(&w)).await.unwrap();
        let dxb = gpu.htod(Dtype::F16, &sh(vec![t, d]), &hbytes(&x)).await.unwrap();
        let w_decl = TensorOps::of_block(dwb.id, Dtype::F16, vec![d, 4]);
        let x_decl = TensorOps::of_block(dxb.id, Dtype::F16, vec![t, d]);
        let sl = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot as f32]));
        let cu = TensorOps::from_host(Dtype::F32, vec![2], &f32b(&[0.0, t as f32]));
        let fwd = TensorOps::of(crate::kernel::kernel_with(
            "owl_gdn_conv_fwd_f16", (1u32, ((d + 255) / 256) as u32, 1), (256, 1, 1), 0,
        )).arg(&x_decl).arg(&w_decl).arg(&st_a).arg(&sl).arg(&cu)
        .arg_i32(1).arg_i32(d as i32).arg_i32(1)
        .with_shape(Dtype::F16, vec![t, d]);
        let out_a = crate::interpreters::eval_ops(fwd.step(), &mut gpu).await.unwrap();

        // 展开:T×conv_upd(独立 state 块)
        let st_b_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F32, vec![1, d, 3]).step(), &mut gpu).await.unwrap();
        let st_b = TensorOps::of_block(st_b_b.id, Dtype::F32, vec![1, d, 3]);
        let mut outs = Vec::new();
        for tk in 0..t {
            let one = gpu.htod(Dtype::F16, &sh(vec![1, d]), &hbytes(&x[tk * d..(tk + 1) * d])).await.unwrap();
            let one_decl = TensorOps::of_block(one.id, Dtype::F16, vec![1, d]);
            let sl1 = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot as f32]));
            let c = TensorOps::of(crate::kernel::kernel_with(
                "owl_gdn_conv_upd_f16", (0, 0, 0), (256, 1, 1), 0,
            )).arg(&one_decl).arg(&w_decl).arg(&st_b).arg(&sl1)
            .arg_usize(d).arg_usize(d).arg_usize(0).arg_i32(1)
            .with_shape(Dtype::F16, vec![1, d]);
            let o = crate::interpreters::eval_ops(c.step(), &mut gpu).await.unwrap();
            let mut b1 = vec![0u8; d * 2];
            gpu.dtoh(&o, &mut b1).await.unwrap();
            outs.extend(unhalf(&b1));
        }
        let mut abuf = vec![0u8; t * d * 2];
        gpu.dtoh(&out_a, &mut abuf).await.unwrap();
        let got_a = unhalf(&abuf);
        assert_close_rel(&got_a, &outs, 1e-4, "conv_fwd vs expansion");
        // state 双锚
        let mut sa = vec![0u8; d * 3 * 4];
        let mut sb = vec![0u8; d * 3 * 4];
        gpu.dtoh(&st_a_b, &mut sa).await.unwrap();
        gpu.dtoh(&st_b_b, &mut sb).await.unwrap();
        for (i, (a, b)) in sa.chunks_exact(4).zip(sb.chunks_exact(4)).enumerate() {
            let (va, vb) = (f32::from_le_bytes(a.try_into().unwrap()), f32::from_le_bytes(b.try_into().unwrap()));
            assert!((va - vb).abs() < 1e-6, "state[{i}] {va} vs {vb}");
        }
        gpu.close().await.unwrap();
    }

    /// (工单 G-6)PF1b:recurrence varlen gqa(单序列 T=8)== T×delta_dec 展开
    /// (out + 终态 f32 双锚;两步连续的批核侧验证)
    #[tokio::test]
    async fn gpu_recurrence_varlen_matches_expansion() {
        if !gpu_enabled() { eprintln!("skip"); return; }
        let (nk, nv, kd, vd, t, slot) = (2usize, 2usize, 16usize, 32usize, 8usize, 0usize);
        let qv: Vec<f32> = q(&(0..t * nk * kd).map(|i| ((i as f32 * 0.17) - 1.0).sin()).collect::<Vec<_>>());
        let kv: Vec<f32> = q(&(0..t * nk * kd).map(|i| ((i as f32 * 0.23) - 0.5).cos()).collect::<Vec<_>>());
        let vv: Vec<f32> = q(&(0..t * nv * vd).map(|i| ((i as f32 * 0.29) + 0.3).sin()).collect::<Vec<_>>());
        let gv: Vec<f32> = q(&(0..t * nv).map(|i| -0.05 - (i as f32 % 4.0) * 0.02).collect::<Vec<_>>());
        let bv: Vec<f32> = q(&(0..t * nv).map(|i| 0.9 - (i as f32 % 4.0) * 0.05).collect::<Vec<_>>());
        let q_scale = 1.0 / (kd as f32).sqrt();

        let mut gpu = gpu_client().await;
        let sh = crate::contract::Shape::from;
        let st_a_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F32, vec![1, nv, kd, vd]).step(), &mut gpu).await.unwrap();
        let st_a = TensorOps::of_block(st_a_b.id, Dtype::F32, vec![1, nv, kd, vd]);
        let dq = gpu.htod(Dtype::F16, &sh(vec![t, nk, kd]), &hbytes(&qv)).await.unwrap();
        let dk = gpu.htod(Dtype::F16, &sh(vec![t, nk, kd]), &hbytes(&kv)).await.unwrap();
        let dv = gpu.htod(Dtype::F16, &sh(vec![t, nv, vd]), &hbytes(&vv)).await.unwrap();
        let dg = gpu.htod(Dtype::F16, &sh(vec![t, nv]), &hbytes(&gv)).await.unwrap();
        let db = gpu.htod(Dtype::F16, &sh(vec![t, nv]), &hbytes(&bv)).await.unwrap();
        let (q_, k_, v_, g_, b_) = (
            TensorOps::of_block(dq.id, Dtype::F16, vec![t, nk, kd]),
            TensorOps::of_block(dk.id, Dtype::F16, vec![t, nk, kd]),
            TensorOps::of_block(dv.id, Dtype::F16, vec![t, nv, vd]),
            TensorOps::of_block(dg.id, Dtype::F16, vec![t, nv]),
            TensorOps::of_block(db.id, Dtype::F16, vec![t, nv]),
        );
        let sl = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot as f32]));
        let cu = TensorOps::from_host(Dtype::F32, vec![2], &f32b(&[0.0, t as f32]));
        let fwd = TensorOps::of(crate::kernel::kernel_with(
            "owl_gdn_recurrence_varlen_gqa_f16",
            (((vd + 7) / 8) as u32, (nv) as u32, 1), (32, 8, 1),
            ((4 * kd + 4) * 4) as u64 as u32,
        )).arg(&q_).arg(&k_).arg(&v_).arg(&g_).arg(&b_)
        .arg(&st_a).arg(&sl).arg(&cu)
        .arg_usize(1).arg_usize(nv).arg_usize(nk).arg_usize(kd).arg_usize(vd)
        .arg_f32(q_scale)
        .with_shape(Dtype::F16, vec![t, nv, vd]);
        let out_a = crate::interpreters::eval_ops(fwd.step(), &mut gpu).await.unwrap();

        // 展开:T×delta_dec(独立 state)
        let st_b_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F32, vec![1, nv, kd, vd]).step(), &mut gpu).await.unwrap();
        let st_b = TensorOps::of_block(st_b_b.id, Dtype::F32, vec![1, nv, kd, vd]);
        let mut outs = Vec::new();
        for tk in 0..t {
            let (rq, rk, rv, rg, rb) = (
                gpu.htod(Dtype::F16, &sh(vec![1, nk, kd]), &hbytes(&qv[tk * nk * kd..(tk + 1) * nk * kd])).await.unwrap(),
                gpu.htod(Dtype::F16, &sh(vec![1, nk, kd]), &hbytes(&kv[tk * nk * kd..(tk + 1) * nk * kd])).await.unwrap(),
                gpu.htod(Dtype::F16, &sh(vec![1, nv, vd]), &hbytes(&vv[tk * nv * vd..(tk + 1) * nv * vd])).await.unwrap(),
                gpu.htod(Dtype::F16, &sh(vec![1, nv]), &hbytes(&gv[tk * nv..(tk + 1) * nv])).await.unwrap(),
                gpu.htod(Dtype::F16, &sh(vec![1, nv]), &hbytes(&bv[tk * nv..(tk + 1) * nv])).await.unwrap(),
            );
            let (q_, k_, v_, g_, b_) = (
                TensorOps::of_block(rq.id, Dtype::F16, vec![1, nk, kd]),
                TensorOps::of_block(rk.id, Dtype::F16, vec![1, nk, kd]),
                TensorOps::of_block(rv.id, Dtype::F16, vec![1, nv, vd]),
                TensorOps::of_block(rg.id, Dtype::F16, vec![1, nv]),
                TensorOps::of_block(rb.id, Dtype::F16, vec![1, nv]),
            );
            let sl1 = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot as f32]));
            let d = TensorOps::of(crate::kernel::kernel_with(
                "owl_gdn_delta_dec_f16",
                (((vd + 63) / 64) as u32, (nv) as u32, 1), (64, 1, 1),
                ((2 * 128 + 2) * 4) as u32,
            )).arg(&q_).arg(&k_).arg(&v_).arg(&g_).arg(&b_)
            .arg(&st_b).arg(&sl1)
            .arg_usize(1).arg_usize(nv).arg_usize(nk).arg_usize(kd).arg_usize(vd)
            .arg_f32(q_scale)
            .with_shape(Dtype::F16, vec![1, nv, vd]);
            let o = crate::interpreters::eval_ops(d.step(), &mut gpu).await.unwrap();
            let mut b1 = vec![0u8; nv * vd * 2];
            gpu.dtoh(&o, &mut b1).await.unwrap();
            outs.extend(unhalf(&b1));
        }
        let mut abuf = vec![0u8; t * nv * vd * 2];
        gpu.dtoh(&out_a, &mut abuf).await.unwrap();
        assert_close_rel(&unhalf(&abuf), &outs, 1e-4, "recurrence vs expansion");
        // 终态双锚
        let mut sa = vec![0u8; nv * kd * vd * 4];
        let mut sb = vec![0u8; nv * kd * vd * 4];
        gpu.dtoh(&st_a_b, &mut sa).await.unwrap();
        gpu.dtoh(&st_b_b, &mut sb).await.unwrap();
        for (i, (a, b)) in sa.chunks_exact(4).zip(sb.chunks_exact(4)).enumerate() {
            let (va, vb) = (f32::from_le_bytes(a.try_into().unwrap()), f32::from_le_bytes(b.try_into().unwrap()));
            // varlen = warp 树归约 / 展开 = 顺序累加:求和序不同,容差取相对
            assert!((va - vb).abs() <= 1e-5 * (1.0 + vb.abs()), "state[{i}] {va} vs {vb}");
        }
        gpu.close().await.unwrap();
    }
}
