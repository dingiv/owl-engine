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
use crate::layers::narrow_strided;
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
    TensorOps::of(kernel::kernel_with(
        "owl_gdn_gating_g_f32",
        (0, 0, 0),
        (256, 1, 1),
        0,
    ))
    .arg(a_log)
    .arg(a)
    .arg(dt_bias)
    .arg_usize(tokens * heads)
    .arg_usize(heads)
    .with_shape(Dtype::F32, vec![tokens, heads])
}

// ============================================================================
// 批2:末维 L2 归一
// ============================================================================

/// l2norm 声明:y[r][i] = x[r][i] · rsqrt(sum(x[r][·]²) + eps)。
/// 行核(非哨兵):grid (rows,1,1) × block (256,1,1)。q/k 头向量归一用
/// (decode:rows = tokens × heads,dim = head_k_dim 128;eps 1e-6)。
pub fn l2norm(x: &TensorOps, rows: usize, dim: usize, eps: f32) -> TensorOps {
    TensorOps::of(kernel::kernel_with(
        "owl_gdn_l2norm_f32",
        (rows as u32, 1, 1),
        (256, 1, 1),
        0,
    ))
    .arg(x)
    .arg_usize(rows)
    .arg_usize(dim)
    .arg_f32(eps)
    .with_shape(Dtype::F32, vec![rows, dim])
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
    TensorOps::of(kernel::kernel_with(
        "owl_gdn_conv_upd_f32",
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
    .with_shape(Dtype::F32, vec![batch, dim])
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
    TensorOps::of(kernel::kernel_with(
        "owl_gdn_delta_dec_f32",
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
    .with_shape(Dtype::F32, vec![batch, nv, vd])
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
    TensorOps::of(kernel::kernel_with(
        "owl_gdn_norm_act_f32",
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
    .with_shape(Dtype::F32, vec![rows, value_dim])
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

    fn forward_decl(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
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
