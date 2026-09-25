//! DecoderLayer:transformer 层编排(Full-attention / GDN 双 mixer 枚举;
//! Qwen3.5 layer_types:3:1 交错 —— 3 层 linear_attention + 1 层
//! full_attention,共 24 层)。
//!
//! 结构(HF Qwen3_5DecoderLayer 同构,标准双残差 pre-norm):
//! ```text
//! xs ─┬────────────────────────────┐
//!     └→ input_ln → mixer ────(+)─┬┘  mixer = Full(Attention) | Gdn(GatedDeltaNet)
//!         ┌───────────────────────┘
//!         └→ post_ln → mlp ───(+)──→ out
//! ```
//! norms 全系 add_one 零中心(Qwen3.5 实证律,§四 10)。
//! mixer 的动态依赖由分支各自检查(Full 需 rope+pos+kv;Gdn 需 gdn
//! 常驻块组)—— 缺依赖沿分支透传毒值,与层内检查同构。

use crate::contract::Dtype;
use crate::layers::attention::Attention;
use crate::layers::gdn::GatedDeltaNet;
use crate::layers::mlp::Mlp;
use crate::layers::rmsnorm::RmsNorm;
use crate::module::{ForwardCtx, Loadable, LoaderCtx, LoaderOps, Module};
use crate::TensorOps;

/// token mixer 枚举(layer_types 分派;同层二选一)
pub enum TokenMixer {
    /// full_attention 层(4:1 间隔)
    Full(Attention),
    /// linear_attention 层(GatedDeltaNet;24 层中占 18)
    Gdn(GatedDeltaNet),
}

pub struct DecoderLayer {
    input_ln: RmsNorm,
    post_ln: RmsNorm,
    mixer: TokenMixer,
    mlp: Mlp,
    hidden: usize,
}

impl DecoderLayer {
    /// full_attention 层容器
    pub fn new_full(hq: usize, hkv: usize, hd: usize, hidden: usize, inter: usize, eps: f32) -> Self {
        DecoderLayer {
            input_ln: RmsNorm::new_add_one("input_layernorm", hidden, eps),
            post_ln: RmsNorm::new_add_one("post_attention_layernorm", hidden, eps),
            mixer: TokenMixer::Full(Attention::new(hq, hkv, hd, hidden, eps)),
            mlp: Mlp::new(hidden, inter),
            hidden,
        }
    }

    /// linear_attention(GDN)层容器
    pub fn new_gdn(nk: usize, hk_dim: usize, nv: usize, hv_dim: usize, hidden: usize, inter: usize, eps: f32) -> Self {
        DecoderLayer {
            input_ln: RmsNorm::new_add_one("input_layernorm", hidden, eps),
            post_ln: RmsNorm::new_add_one("post_attention_layernorm", hidden, eps),
            mixer: TokenMixer::Gdn(GatedDeltaNet::new(nk, hk_dim, nv, hv_dim, hidden, eps)),
            mlp: Mlp::new(hidden, inter),
            hidden,
        }
    }
}

impl Module for DecoderLayer {
    /// 双残差声明(h = x + mixer(ln1(x));out = h + mlp(ln2(h)))
    fn forward(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        let n1 = self.input_ln.forward(xs, ctx);
        let mixed = match &self.mixer {
            TokenMixer::Full(a) => a.forward(&n1, ctx),
            TokenMixer::Gdn(g) => g.forward(&n1, ctx),
        };
        let h = xs.add(&mixed);
        let n2 = self.post_ln.forward(&h, ctx);
        let mlp_out = self.mlp.forward(&n2, ctx);
        h.add(&mlp_out)
    }
}

impl Loadable for DecoderLayer {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        let mixer = match &self.mixer {
            TokenMixer::Full(a) => a.layout(ctx),
            TokenMixer::Gdn(g) => g.layout(ctx),
        };
        self.input_ln.layout(ctx)
            .chain(self.post_ln.layout(ctx))
            .chain(mixer)
            .chain(self.mlp.layout(ctx))
    }
}

// Layer 的形状语义 = hidden(结构参数;声明期常量,毒值路径也带它)
impl DecoderLayer {
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// 测试专用:分段收割(层树内 mixed / n2 / mlp_out 的声明,非公开 API)。
    #[cfg(test)]
    pub(crate) fn forward_stages<'a>(
        &self,
        xs: &TensorOps,
        ctx: &ForwardCtx<'a>,
    ) -> (TensorOps, TensorOps, TensorOps, TensorOps, TensorOps) {
        let n1 = self.input_ln.forward(xs, ctx);
        let mixed = match &self.mixer {
            TokenMixer::Full(a) => a.forward(&n1, ctx),
            TokenMixer::Gdn(g) => g.forward(&n1, ctx),
        };
        let h = xs.add(&mixed);
        let n2 = self.post_ln.forward(&h, ctx);
        let mlp_out = self.mlp.forward(&n2, ctx);
        (mixed, h.clone(), n2, mlp_out.clone(), h.add(&mlp_out))
    }
}

// 保持 Dtype 引用(层文件统一风格)
const _: Dtype = Dtype::F32;

#[cfg(test)]
mod tests {
    use super::*;

    fn src_map(entries: &[(&str, usize, f32)]) -> std::collections::HashMap<String, Vec<f32>> {
        entries
            .iter()
            .map(|(k, n, seed)| {
                (
                    k.to_string(),
                    (0..*n).map(|i| ((i as f32 + seed) * 0.13).sin() * 0.5).collect::<Vec<f32>>(),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn load_and_declaration_both_branches() {
        let mut face = owl_cpu::CpuFace::new();

        // Full 分支(HQ=2 HKV=1 HD=4 hidden=3 inter=5)
        let full = DecoderLayer::new_full(2, 1, 4, 3, 5, 1e-6);
        let full_src = src_map(&[
            ("input_layernorm", 3, 1.0),
            ("post_attention_layernorm", 3, 2.0),
            ("q_proj", 2 * 4 * 2 * 3, 3.0),
            ("k_proj", 1 * 4 * 3, 4.0),
            ("v_proj", 1 * 4 * 3, 5.0),
            ("o_proj", 3 * 2 * 4, 6.0),
            ("q_norm", 4, 7.0),
            ("k_norm", 4, 8.0),
            ("gate_proj", 5 * 3, 9.0),
            ("up_proj", 5 * 3, 10.0),
            ("down_proj", 3 * 5, 11.0),
        ]);
        crate::interpreter::eval_load(&full, &mut face, &full_src, &Default::default())
            .await
            .expect("full 层 eval_load");

        // Gdn 分支(NK=2 HK=4 NV=2 HV=4 hidden=6 inter=5)
        let gdn = DecoderLayer::new_gdn(2, 4, 2, 4, 6, 5, 1e-6);
        let gdn_src = src_map(&[
            ("input_layernorm", 6, 1.0),
            ("post_attention_layernorm", 6, 2.0),
            ("in_proj_qkv", (2 * 8 + 8) * 6, 3.0),
            ("in_proj_z", 8 * 6, 4.0),
            ("in_proj_b", 2 * 6, 5.0),
            ("in_proj_a", 2 * 6, 6.0),
            ("out_proj", 6 * 8, 7.0),
            ("conv1d", 24 * 4, 8.0),
            ("A_log", 2, 9.0),
            ("dt_bias", 2, 10.0),
            ("norm", 4, 11.0),
            ("gate_proj", 5 * 6, 12.0),
            ("up_proj", 5 * 6, 13.0),
            ("down_proj", 6 * 5, 14.0),
        ]);
        crate::interpreter::eval_load(&gdn, &mut face, &gdn_src, &Default::default())
            .await
            .expect("gdn 层 eval_load");

        // Gdn 声明(GdnBuffers 零块)
        let tokens = 2usize;
        let gdn_buf = crate::layers::gdn::GdnBuffers {
            conv_q: TensorOps::zeros(Dtype::F32, vec![4, 8, 3]),
            conv_k: TensorOps::zeros(Dtype::F32, vec![4, 8, 3]),
            conv_v: TensorOps::zeros(Dtype::F32, vec![4, 8, 3]),
            rec: TensorOps::zeros(Dtype::F32, vec![4, 2, 4, 4]),
            slots: TensorOps::from_host(Dtype::F32, vec![tokens], &crate::testkit::f32b(&[0.0, 2.0])),
        };
        let xs6 = TensorOps::from_host(Dtype::F32, vec![tokens, 6], &crate::testkit::f32b(&vec![0.4; tokens * 6]));
        let out = gdn.forward(&xs6, &ForwardCtx::gdn_decode(tokens, &gdn_buf));
        assert!(!out.is_poisoned(), "gdn 层声明不应有毒");
        assert_eq!(out.shape(), &[tokens, 6]);
        assert_eq!(gdn.hidden(), 6);

        // Full 声明缺 rope/pos/kv → 毒值(mixer 分支检查透传)
        let xs3 = TensorOps::from_host(Dtype::F32, vec![tokens, 3], &crate::testkit::f32b(&vec![0.4; tokens * 3]));
        let out2 = full.forward(&xs3, &ForwardCtx::minimal(tokens));
        assert!(out2.is_poisoned(), "full 层 minimal ctx 应毒");
    }

    // ======================================================================
    // 批7 GPU 数值对拍(两分支;host 全链独立副本)
    // ======================================================================

    const EPS: f32 = 1e-6;

    /// rmsnorm ×(1+w)(零中心)
    fn host_ln(x: &[f32], w: &[f32], cols: usize, _t: usize) -> Vec<f32> {
        let row = &x[0..cols];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + EPS).sqrt();
        row.iter().zip(w).map(|(v, g)| v * inv * (g + 1.0)).collect()
    }

    /// mlp:silu(gate·x) ⊙ (up·x) → down
    fn host_mlp(s: &std::collections::HashMap<String, Vec<f32>>, x: &[f32], inter: usize, hidden: usize) -> Vec<f32> {
        let lin = |w: &[f32]| {
            (0..w.len() / hidden)
                .map(|o| (0..hidden).map(|i| x[i] * w[o * hidden + i]).sum::<f32>())
                .collect::<Vec<f32>>()
        };
        let gate = lin(&s["gate_proj"]);
        let up = lin(&s["up_proj"]);
        let mut h = vec![0.0f32; inter];
        for o in 0..inter {
            let silu = gate[o] / (1.0 + (-gate[o]).exp());
            h[o] = silu * up[o];
        }
        (0..hidden)
            .map(|o| (0..inter).map(|i| h[i] * s["down_proj"][o * inter + i]).sum::<f32>())
            .collect()
    }

    fn gen(seed: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.13).sin() * 0.5).collect()
    }

    /// 零块(htod 清零;勿用裸 alloc —— 垃圾内存会进 conv/rec 状态)
    async fn zero_block(client: &mut owl_cuda::GpuClient, n: usize, shape: Vec<usize>) -> (crate::contract::Bytes, TensorOps) {
        let b = crate::interpreter::eval_ops(
            TensorOps::from_host(Dtype::F32, vec![n], &crate::testkit::f32b(&vec![0.0; n])).step(),
            client,
        )
        .await
        .expect("zero block");
        let t = TensorOps::of_block(b.id, Dtype::F32, shape);
        (b, t)
    }

    /// Full 分支单步 decode vs host(门控 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn gpu_full_layer_matches_host() {
        use crate::layers::rope::Rope;
        use crate::module::KvBuffers;
        if !crate::testkit::gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (hq, hkv, hd, hidden, inter) = (2usize, 1usize, 4usize, 3usize, 5usize);
        let layer = DecoderLayer::new_full(hq, hkv, hd, hidden, inter, EPS);
        let mut gpu = crate::testkit::gpu_client().await;
        let src = src_map(&[
            ("input_layernorm", hidden, 1.0),
            ("post_attention_layernorm", hidden, 2.0),
            ("q_proj", hq * hd * 2 * hidden, 3.0),
            ("k_proj", hkv * hd * hidden, 4.0),
            ("v_proj", hkv * hd * hidden, 5.0),
            ("o_proj", hidden * hq * hd, 6.0),
            ("q_norm", hd, 7.0),
            ("k_norm", hd, 8.0),
            ("gate_proj", inter * hidden, 9.0),
            ("up_proj", inter * hidden, 10.0),
            ("down_proj", hidden * inter, 11.0),
        ]);
        crate::interpreter::eval_load(&layer, &mut gpu, &src, &Default::default()).await.expect("load");
        let rp = Rope::new(64, hd, hd, 10_000.0).expect("rope"); // 全旋转(rotary = head_dim)
        crate::interpreter::eval_load(&rp, &mut gpu, &rp.tables(), &Default::default()).await.expect("rope 表");
        let (_, kc) = zero_block(&mut gpu, 8 * hkv * hd, vec![8, hkv, hd]).await;
        let (_, vc) = zero_block(&mut gpu, 8 * hkv * hd, vec![8, hkv, hd]).await;

        let tokens = 1usize;
        let xs = gen(1.0, hidden);
        let pos_v = [7.0f32];
        let kv = KvBuffers {
            k_cache: kc,
            v_cache: vc,
            slots: TensorOps::from_host(Dtype::F32, vec![tokens], &crate::testkit::f32b(&[0.0])),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![tokens], &crate::testkit::f32b(&[1.0])),
        };
        let pos = TensorOps::from_host(Dtype::F32, vec![tokens], &crate::testkit::f32b(&pos_v));
        let xs_t = TensorOps::from_host(Dtype::F32, vec![tokens, hidden], &crate::testkit::f32b(&xs));
        let ctx = ForwardCtx::decode(tokens, &pos, &kv, &rp);
        let decl = layer.forward(&xs_t, &ctx);
        let got = crate::testkit::harvest(&mut gpu, &decl).await;
        gpu.close().await.expect("关机");

        // ---- host 全链(独立副本)----
        let lin = |x: &[f32], w: &[f32], out_dim: usize, in_dim: usize| {
            (0..out_dim)
                .map(|o| (0..in_dim).map(|i| x[i] * w[o * in_dim + i]).sum::<f32>())
                .collect::<Vec<f32>>()
        };
        let n1 = host_ln(&xs, &src["input_layernorm"], hidden, 0);
        // attention(qk-norm → rope → 单步 attn → gate)—— 数学同 M-b HostRef
        let attn = |n: &[f32]| -> Vec<f32> {
            let q_raw = lin(n, &src["q_proj"], hq * hd * 2, hidden);
            let k_raw = lin(n, &src["k_proj"], hkv * hd, hidden);
            let v_raw = lin(n, &src["v_proj"], hkv * hd, hidden);
            let mut q = vec![0.0; hq * hd];
            let mut gate = vec![0.0; hq * hd];
            for h in 0..hq {
                for d in 0..hd {
                    q[h * hd + d] = q_raw[h * hd * 2 + d];
                    gate[h * hd + d] = q_raw[h * hd * 2 + hd + d];
                }
            }
            let rms = |x: &[f32], g: &[f32]| {
                let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / hd as f32;
                let inv = 1.0 / (ms + EPS).sqrt();
                x.iter().zip(g).map(|(v, gg)| v * inv * (gg + 1.0)).collect::<Vec<f32>>()
            };
            let rope = |x: &[f32], heads: usize, p: f32| -> Vec<f32> {
                let half = hd / 2;
                let mut out = x.to_vec();
                for h in 0..heads {
                    for i in 0..half {
                        let ang = p * 10_000f32.powf(-(2.0 * i as f32) / hd as f32);
                        let (c, sn) = (ang.cos(), ang.sin());
                        let (a, b) = (x[h * hd + i], x[h * hd + i + half]);
                        out[h * hd + i] = a * c - b * sn;
                        out[h * hd + i + half] = b * c + a * sn;
                    }
                }
                out
            };
            let mut qn = vec![0.0f32; hq * hd];
            let mut kn = vec![0.0f32; hkv * hd];
            for h in 0..hq {
                qn[h * hd..(h + 1) * hd].copy_from_slice(&rms(&q[h * hd..(h + 1) * hd], &src["q_norm"]));
            }
            for h in 0..hkv {
                kn[h * hd..(h + 1) * hd].copy_from_slice(&rms(&k_raw[h * hd..(h + 1) * hd], &src["k_norm"]));
            }
            let q = rope(&qn, hq, pos_v[0]);
            let k = rope(&kn, hkv, pos_v[0]);
            // 单步:kv_len=1,窗 [0,1);cache 零行 = 本步写入
            let mut y = vec![0.0f32; hq * hd];
            for h in 0..hq {
                let kvh = h / (hq / hkv);
                let qs = &q[h * hd..(h + 1) * hd];
                let scale = 1.0 / (hd as f32).sqrt();
                let acc: f32 = qs.iter().zip(&k[kvh * hd..(kvh + 1) * hd]).map(|(a2, b2)| a2 * b2).sum::<f32>() * scale;
                let wgt = 1.0; // 单窗 softmax 恒 1
                let _ = acc;
                for d in 0..hd {
                    y[h * hd + d] = wgt * v_raw[kvh * hd + d];
                }
            }
            for (yi, gg) in y.iter_mut().zip(&gate) {
                *yi *= 1.0 / (1.0 + (-gg).exp());
            }
            lin(&y, &src["o_proj"], hidden, hq * hd)
        };
        let mixed = attn(&n1);
        let mut h = vec![0.0f32; hidden];
        for i in 0..hidden {
            h[i] = xs[i] + mixed[i];
        }
        let n2 = host_ln(&h, &src["post_attention_layernorm"], hidden, 0);
        let mlp_out = host_mlp(&src, &n2, inter, hidden);
        let mut want = vec![0.0f32; hidden];
        for i in 0..hidden {
            want[i] = h[i] + mlp_out[i];
        }
        crate::testkit::assert_close(&got, &want, 1e-4, "decoder-full");
    }

    /// Gdn 分支单步 decode vs host(手搭 mixer 链逐段收割;门控 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn gpu_gdn_layer_matches_host() {
        use crate::layers::gdn::GdnBuffers;
        // if !crate::testkit::gpu_enabled() {
        //     eprintln!("skip: OWL_TEST_DEVICE 未设");
        //     return;
        // }
        let (nk, hk_dim, nv, hv_dim, hidden, inter) = (2usize, 4usize, 2usize, 4usize, 6usize, 5usize);
        let key_dim = nk * hk_dim;
        let value_dim = nv * hv_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let layer = DecoderLayer::new_gdn(nk, hk_dim, nv, hv_dim, hidden, inter, EPS);
        let mut gpu = crate::testkit::gpu_client().await;
        let src = src_map(&[
            ("input_layernorm", hidden, 1.0),
            ("post_attention_layernorm", hidden, 2.0),
            ("in_proj_qkv", conv_dim * hidden, 3.0),
            ("in_proj_z", value_dim * hidden, 4.0),
            ("in_proj_b", nv * hidden, 5.0),
            ("in_proj_a", nv * hidden, 6.0),
            ("out_proj", hidden * value_dim, 7.0),
            ("conv1d", conv_dim * 4, 8.0),
            ("A_log", nv, 9.0),
            ("dt_bias", nv, 10.0),
            ("norm", hv_dim, 11.0),
            ("gate_proj", inter * hidden, 12.0),
            ("up_proj", inter * hidden, 13.0),
            ("down_proj", hidden * inter, 14.0),
        ]);
        crate::interpreter::eval_load(&layer, &mut gpu, &src, &Default::default()).await.expect("load");
        let tokens = 1usize;
        let xs = gen(2.0, hidden);
        let slots_v = [1.0f32];
        let xs_t = TensorOps::from_host(Dtype::F32, vec![tokens, hidden], &crate::testkit::f32b(&xs));
        // 五段各配全新零状态,**每轮只 harvest 目标段**(第 i 轮的第一次
        // eval 是唯一干净态:状态副作用下,同树二次 eval = 滑过态,不可比)
        let mut harvested = Vec::new();
        for i in 0..5 {
            let buf = GdnBuffers {
                conv_q: zero_block(&mut gpu, 4 * key_dim * 3, vec![4, key_dim, 3]).await.1,
                conv_k: zero_block(&mut gpu, 4 * key_dim * 3, vec![4, key_dim, 3]).await.1,
                conv_v: zero_block(&mut gpu, 4 * value_dim * 3, vec![4, value_dim, 3]).await.1,
                rec: zero_block(&mut gpu, 4 * nv * hk_dim * hv_dim, vec![4, nv, hk_dim, hv_dim]).await.1,
                slots: TensorOps::from_host(Dtype::F32, vec![tokens], &crate::testkit::f32b(&slots_v)),
            };
            let (m, hd, n2d, md, fd) = layer.forward_stages(&xs_t, &ForwardCtx::gdn_decode(tokens, &buf));
            // 每轮只 harvest 目标段:本段 eval = 该轮第一次 = 唯一干净态
            // (状态副作用下,同树二次 eval 已滑状态,不可作对拍锚)
            let t = [m, hd, n2d, md, fd][i].clone();
            let v = crate::testkit::harvest(&mut gpu, &t).await;
            harvested.push(v);
        }
        let got = harvested[4].clone();

        // ---- host 全链(独立副本)----
        let lin = |x: &[f32], w: &[f32], out_dim: usize, in_dim: usize| {
            (0..out_dim)
                .map(|o| (0..in_dim).map(|i| x[i] * w[o * in_dim + i]).sum::<f32>())
                .collect::<Vec<f32>>()
        };
        let n1 = host_ln(&xs, &src["input_layernorm"], hidden, 0);
        let qkv = lin(&n1, &src["in_proj_qkv"], conv_dim, hidden);
        let z = lin(&n1, &src["in_proj_z"], value_dim, hidden);
        let b_v = lin(&n1, &src["in_proj_b"], nv, hidden);
        let _a_v = lin(&n1, &src["in_proj_a"], nv, hidden);
        let segs = [&qkv[0..key_dim], &qkv[key_dim..2 * key_dim], &qkv[2 * key_dim..]];
        let seg_dim = [key_dim, key_dim, value_dim];
        let seg_off = [0usize, key_dim, 2 * key_dim];
        let mut post = [vec![0.0f32; key_dim], vec![0.0f32; key_dim], vec![0.0f32; value_dim]];
        for (seg, inp) in segs.iter().enumerate() {
            for ch in 0..seg_dim[seg] {
                let wb = (seg_off[seg] + ch) * 4;
                let w = &src["conv1d"];
                let mut sum = inp[ch] * w[wb + 3]; // state 零起步:hist 项 = 0
                sum /= 1.0 + (-sum).exp();
                post[seg][ch] = sum;
            }
        }
        let l2 = |row: &[f32]| {
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss.max(0.0) + 1e-6).sqrt();
            row.iter().map(|v| v * inv).collect::<Vec<f32>>()
        };
        let mut qn = vec![0.0f32; key_dim];
        let mut kn = vec![0.0f32; key_dim];
        for h in 0..nk {
            qn[h * hk_dim..(h + 1) * hk_dim].copy_from_slice(&l2(&post[0][h * hk_dim..(h + 1) * hk_dim]));
            kn[h * hk_dim..(h + 1) * hk_dim].copy_from_slice(&l2(&post[1][h * hk_dim..(h + 1) * hk_dim]));
        }
        let beta_v: Vec<f32> = b_v.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        let q_scale = 1.0 / (hk_dim as f32).sqrt();
        let mut y = vec![0.0f32; value_dim];
        for vh in 0..nv {
            let kh = vh / (nv / nk);
            let qoff = kh * hk_dim;
            let voff = vh * hv_dim;
            for d in 0..hv_dim {
                let delta = post[2][voff + d] * beta_v[vh]; // kv_mem = 0(rec 零起步)
                let mut acc = 0.0f32;
                for j in 0..hk_dim {
                    acc += kn[qoff + j] * delta * qn[qoff + j] * q_scale;
                }
                y[voff + d] = acc;
            }
        }
        let mut gated = vec![0.0f32; value_dim];
        for vh in 0..nv {
            let base = vh * hv_dim;
            let ms: f32 = (0..hv_dim).map(|i| y[base + i] * y[base + i]).sum::<f32>() / hv_dim as f32;
            let inv = 1.0 / (ms.max(0.0) + EPS).sqrt();
            for i in 0..hv_dim {
                let zv = z[base + i];
                gated[base + i] = y[base + i] * inv * src["norm"][i] * (zv / (1.0 + (-zv).exp()));
            }
        }
        let mixed = lin(&gated, &src["out_proj"], hidden, value_dim);
        let mut h = vec![0.0f32; hidden];
        for i in 0..hidden {
            h[i] = xs[i] + mixed[i];
        }
        let n2 = host_ln(&h, &src["post_attention_layernorm"], hidden, 0);
        let mlp_out = host_mlp(&src, &n2, inter, hidden);
        let mut want = vec![0.0f32; hidden];
        for i in 0..hidden {
            want[i] = h[i] + mlp_out[i];
        }

        // 终极裁定:同 fresh buf 连续两次 harvest full
        {
            let (cqb, cq_t) = zero_block(&mut gpu, 4 * key_dim * 3, vec![4, key_dim, 3]).await;
            let (ckb, ck_t) = zero_block(&mut gpu, 4 * key_dim * 3, vec![4, key_dim, 3]).await;
            let (cvb, cv_t) = zero_block(&mut gpu, 4 * value_dim * 3, vec![4, value_dim, 3]).await;
            let (rcb, rc_t) = zero_block(&mut gpu, 4 * nv * hk_dim * hv_dim, vec![4, nv, hk_dim, hv_dim]).await;
            let _ = (&cqb, &ckb, &cvb, &rcb);
            let buf5 = GdnBuffers {
                conv_q: cq_t,
                conv_k: ck_t,
                conv_v: cv_t,
                rec: rc_t,
                slots: TensorOps::from_host(Dtype::F32, vec![tokens], &crate::testkit::f32b(&slots_v)),
            };
            let (_, _, _, _, f1) = layer.forward_stages(&xs_t, &ForwardCtx::gdn_decode(tokens, &buf5));
            let g1 = crate::testkit::harvest(&mut gpu, &f1).await;
            let (_, _, _, _, f2) = layer.forward_stages(&xs_t, &ForwardCtx::gdn_decode(tokens, &buf5));
            let g2 = crate::testkit::harvest(&mut gpu, &f2).await;
            println!("  [final] fresh-buf full #1 = {g1:?}");
            println!("  [final] fresh-buf full #2 = {g2:?}");
        }
        crate::testkit::assert_close(&got, &want, 1e-4, "decoder-gdn");
        gpu.close().await.expect("关机");
    }
}
