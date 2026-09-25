//! Attention:full-attention 层(Qwen3.5 自注意力;decode slot 直排)。
//!
//! 数据流(0.8B 实测配置:Hq=8 / Hkv=2(GQA 4:1)/ head_dim=256;
//! rope theta 1e7 partial 0.25 interleaved —— 表与发射归 [`crate::layers::rope`]):
//!
//! ```text
//! xs [T, hidden]
//!   ├ q_proj → q_raw [T, 2*Hq*HD](attn_output_gate:value|gate per-head 拼接)
//!   │   ├ narrow(start=0)  → q    [T, Hq*HD]
//!   │   └ narrow(start=HD) → gate [T, Hq*HD]
//!   ├ k_proj → k [T, Hkv*HD] ── k_norm ── rope ─┐
//!   ├ v_proj → v [T, Hkv*HD] ───────────────────┤
//!   │            q: q_norm → rope ──────────────┤
//!   │        naive decode attn(slot 直排 KV)←─┘ → y [T, Hq*HD]
//!   │        y × sigmoid(gate) → o_proj → out [T, hidden]
//! ```
//!
//! 语义算子面(matmul/rmsnorm/sigmoid/mul)双 face 可跑;三个 Kernel 节点
//! (`owl_narrow_strided_f32` ×2 / `owl_naive_decode_attn_f32`)GPU server
//! 懒编译执行(qk-norm 复用 `owl_rmsnorm_f32` w_off=1 —— per-head 行 =
//! [T×H, HD] rows,无需专用核;对计划文档"新写 owl_qknorm_addone"的改判)。
//!
//! 动态依赖(rope 表 / pos / KV 槽)经参数显式传入(Rope 先例:不进
//! `Module` trait;统一 ForwardCtx 随 runner 立项)。

use crate::contract::Dtype;
use crate::kernel;
use crate::layers::linear::Linear;
use crate::layers::narrow_strided;
use crate::layers::rmsnorm::RmsNorm;
use crate::module::{ForwardCtx, Loadable, LoaderCtx, LoaderOps, Module};
use crate::TensorOps;

pub struct Attention {
    /// q_proj [2*Hq*HD, hidden](value|gate per-head 拼接;装载期转置)
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    /// o_proj [hidden, Hq*HD]
    o_proj: Linear,
    /// qk-norm(per-head ×(1+w);rows = T×H,eps = rms_norm_eps 1e-6)
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    hq: usize,
    hkv: usize,
    hd: usize,
    hidden: usize,
}

impl Attention {
    /// 准备容器(0.8B:hq=8, hkv=2, hd=256 → q_proj out 4096 / kv out 512)
    pub fn new(hq: usize, hkv: usize, hd: usize, hidden: usize, eps: f32) -> Attention {
        Attention {
            q_proj: Linear::new("q_proj", hq * hd * 2, hidden),
            k_proj: Linear::new("k_proj", hkv * hd, hidden),
            v_proj: Linear::new("v_proj", hkv * hd, hidden),
            o_proj: Linear::new("o_proj", hidden, hq * hd),
            q_norm: RmsNorm::new_add_one("q_norm", hd, eps),
            k_norm: RmsNorm::new_add_one("k_norm", hd, eps),
            hq,
            hkv,
            hd,
            hidden,
        }
    }

    /// 计算声明(decode;xs [T, hidden],T = ctx.tokens;C4 后回归 Module)。
    /// rope 表与 pos / KV 引用由 ctx 注入(rope 全局一份,表已是设备块);
    /// 缺任一动态依赖 → 毒值声明(eval 边界收割,与未装载槽同构)。
    fn forward_decl(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        let tokens = ctx.tokens;
        let (rope, pos, kv) = match (ctx.rope, ctx.pos, ctx.kv) {
            (Some(r), Some(p), Some(k)) => (r, p, k),
            _ => {
                return TensorOps::poisoned(
                    Dtype::F32,
                    vec![tokens, self.hidden],
                    "attention: ctx 缺动态依赖(需 pos + kv + rope;ForwardCtx::decode)",
                );
            }
        };
        let q_raw = self.q_proj.forward(xs, ctx); // [T, 2*Hq*HD]
        let k = self.k_proj.forward(xs, ctx); // [T, Hkv*HD]
        let v = self.v_proj.forward(xs, ctx); // [T, Hkv*HD]

        // per-head [value|gate] 切分(两段同形 [T, Hq*HD])
        let flat_shape = vec![tokens, self.hq * self.hd];
        let q = narrow_strided(&q_raw, tokens * self.hq, self.hd * 2, 0, self.hd, flat_shape.clone());
        let gate = narrow_strided(&q_raw, tokens * self.hq, self.hd * 2, self.hd, self.hd, flat_shape);

        // qk-norm(per-head 行 = [T×H, HD];×(1+w))→ rope
        let q = self.q_norm.forward(&q, ctx);
        let k = self.k_norm.forward(&k, ctx);
        let q = rope.forward_q(&q, pos, tokens, self.hq);
        let k = rope.forward_k(&k, pos, tokens, self.hkv);

        // naive decode attention(slot 直排;一线程一 (t, q_head))
        let y = TensorOps::of(kernel::kernel_with(
            "owl_naive_decode_attn_f32",
            (0, 0, 0), // 哨兵;核内有 bs 上界 guard
            (256, 1, 1),
            0,
        ))
        .arg(&q)
        .arg(&k)
        .arg(&v)
        .arg(&kv.k_cache)
        .arg(&kv.v_cache)
        .arg(&kv.slots)
        .arg(&kv.kv_lens)
        .arg_usize(tokens)
        .arg_usize(self.hq)
        .arg_usize(self.hkv)
        .arg_usize(self.hd)
        .with_shape(Dtype::F32, vec![tokens, self.hq * self.hd]);

        // 输出门:attn 输出 × sigmoid(gate)(per-head;o_proj 之前)
        let y = y.mul(&gate.sigmoid());
        self.o_proj.forward(&y, ctx)
    }
}

impl Module for Attention {
    fn forward(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        self.forward_decl(xs, ctx)
    }
}

impl Loadable for Attention {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.q_proj.layout(ctx)
            .chain(self.k_proj.layout(ctx))
            .chain(self.v_proj.layout(ctx))
            .chain(self.o_proj.layout(ctx))
            .chain(self.q_norm.layout(ctx))
            .chain(self.k_norm.layout(ctx))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::rope::Rope as RopeT;
    use crate::module::KvBuffers;
    use crate::testkit::{f32b, harvest, Src};
    use crate::tensor::Dtype;

    fn six_slots(hq: usize, hkv: usize, hd: usize, hidden: usize) -> Src {
        Src::from([
            ("q_proj".to_string(), (0..hq * hd * 2 * hidden).map(|i| (i as f32 * 0.011) - 0.3).collect()),
            ("k_proj".to_string(), (0..hkv * hd * hidden).map(|i| (i as f32 * 0.013) - 0.2).collect()),
            ("v_proj".to_string(), (0..hkv * hd * hidden).map(|i| (i as f32 * 0.017) - 0.1).collect()),
            ("o_proj".to_string(), (0..hidden * hq * hd).map(|i| (i as f32 * 0.019) - 0.4).collect()),
            ("q_norm".to_string(), (0..hd).map(|i| (i as f32 * 0.02) - 0.1).collect()),
            ("k_norm".to_string(), (0..hd).map(|i| (i as f32 * 0.03) - 0.15).collect()),
        ])
    }

    #[tokio::test]
    async fn load_and_declaration_wellformed() {
        let (hq, hkv, hd, hidden) = (2usize, 1usize, 4usize, 3usize);
        let mut face = owl_cpu::CpuFace::new();
        let attn = Attention::new(hq, hkv, hd, hidden, 1e-6);
        crate::interpreter::eval_load(&attn, &mut face, &six_slots(hq, hkv, hd, hidden), &Default::default())
            .await
            .expect("eval_load 六槽");

        let tokens = 1usize;
        let xs = TensorOps::from_host(Dtype::F32, vec![tokens, hidden], &f32b(&[0.5, -0.25, 1.0]));
        let pos = TensorOps::from_host(Dtype::F32, vec![tokens], &f32b(&[7.0]));
        let rp = RopeT::new(64, hd, 2, 10_000.0).expect("rope new");
        crate::interpreter::eval_load(&rp, &mut face, &rp.tables(), &Default::default())
            .await
            .expect("rope 表物化");
        let kv = KvBuffers {
            k_cache: TensorOps::zeros(Dtype::F32, vec![4, hkv, hd]),
            v_cache: TensorOps::zeros(Dtype::F32, vec![4, hkv, hd]),
            slots: TensorOps::from_host(Dtype::F32, vec![tokens], &f32b(&[0.0])),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![tokens], &f32b(&[1.0])),
        };
        let ctx = crate::module::ForwardCtx::decode(tokens, &pos, &kv, &rp);

        let out = attn.forward(&xs, &ctx);
        assert!(!out.is_poisoned(), "装载后声明不应有毒");
        assert_eq!(out.shape(), &[tokens, hidden]);

        // 毒值契约:未装载容器 → forward 声明立即带毒
        let attn2 = Attention::new(hq, hkv, hd, hidden, 1e-6);
        assert!(attn2.forward(&xs, &ctx).is_poisoned(), "未装载槽的声明应立即带毒");

        // 缺动态依赖的 ctx → 毒值(与未装载槽同构)
        let bare = crate::module::ForwardCtx::minimal(tokens);
        assert!(attn.forward(&xs, &bare).is_poisoned(), "minimal ctx 缺 pos/kv/rope,应毒");
    }

    /// narrow_strided:GPU-only kernel vs host 参考(OWL_TEST_DEVICE 门控)
    #[tokio::test]
    async fn gpu_narrow_matches_host() {
        if !crate::testkit::gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let q_raw: Vec<f32> = (0..2 * 32).map(|i| (i as f32 * 0.17) - 1.4).collect();
        let (outer, src_dim, start, out_dim) = (4usize, 16usize, 8usize, 8usize); // start = HD(gate 半段)

        let mut want = vec![0.0f32; outer * out_dim];
        for r in 0..outer {
            for d in 0..out_dim {
                want[r * out_dim + d] = q_raw[r * src_dim + start + d];
            }
        }

        let mut gpu = crate::testkit::gpu_client().await;
        let src = TensorOps::from_host(Dtype::F32, vec![2, 32], &f32b(&q_raw));
        let decl = TensorOps::of(kernel::kernel_with(
            "owl_narrow_strided_f32",
            (0, 0, 0),
            (256, 1, 1),
            0,
        ))
        .arg(&src)
        .arg_usize(outer)
        .arg_usize(src_dim)
        .arg_usize(start)
        .arg_usize(out_dim)
        .with_shape(Dtype::F32, vec![outer * out_dim]);
        let got = harvest(&mut gpu, &decl).await;
        gpu.close().await.expect("server 关机");

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-6, "[{i}] {g} vs {w}");
        }
    }

    /// Module 统一入口 GPU 冒烟:装载六槽 + 缺依赖毒值收割(OWL_TEST_DEVICE 门控)
    #[tokio::test]
    async fn gpu_module_smoke() {
        if !crate::testkit::gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        use crate::module::Module as _;
        let mut gpu = crate::testkit::gpu_client().await;
        let attn = Attention::new(2, 1, 8, 3, 1e-6);
        crate::interpreter::eval_load(&attn, &mut gpu, &six_slots(2, 1, 8, 3), &Default::default())
            .await
            .expect("eval_load");
        let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&[0.5, -0.25, 1.0]));
        let out = attn.forward(&xs, &crate::module::ForwardCtx::minimal(1));
        assert!(out.is_poisoned(), "minimal ctx 缺 pos/kv/rope → 毒");
        let err = crate::interpreter::eval_ops(out.step(), &mut gpu)
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("缺动态依赖"), "毒值应带 ctx 归因:{err:?}");
        gpu.close().await.expect("server 关机");
    }
}
