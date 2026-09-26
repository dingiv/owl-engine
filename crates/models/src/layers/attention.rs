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

        // naive decode attention(slot 直排;一线程一 (t, q_head));
        // 核名/输出 dtype 跟随 q 声明(F5 整模切换;KV cache f16)
        let dt = q.dtype;
        let attn_name = if dt == Dtype::F16 {
            "owl_naive_decode_attn_f16"
        } else {
            "owl_naive_decode_attn_f32"
        };
        let y = TensorOps::of(kernel::kernel_with(
            attn_name,
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
        .with_shape(dt, vec![tokens, self.hq * self.hd]);

        // 输出门:attn 输出 × sigmoid(gate)(per-head;o_proj 之前)
        // f16 路径:gate_mul 融合单发(工单 N 产 owl_sigmoid_gate_mul_f16,
        // out = x · sigmoid(gate);n 偶数由 head_dim 全偶保证)
        // f32 路径:语义算子 sigmoid + mul(CPU 单元锚)
        let y = if dt == Dtype::F16 {
            let n = tokens * self.hq * self.hd;
            TensorOps::of(kernel::kernel_with(
                "owl_sigmoid_gate_mul_f16",
                (0, 0, 0),
                (256, 1, 1),
                0,
            ))
            .arg(&gate)
            .arg(&y)
            .arg_usize(n)
            .with_shape(Dtype::F16, vec![tokens, self.hq * self.hd])
        } else {
            y.mul(&gate.sigmoid())
        };
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
        crate::interpreters::eval_load(&attn, &mut face, &six_slots(hq, hkv, hd, hidden), &Default::default())
            .await
            .expect("eval_load 六槽");

        let tokens = 1usize;
        let xs = TensorOps::from_host(Dtype::F32, vec![tokens, hidden], &f32b(&[0.5, -0.25, 1.0]));
        let pos = TensorOps::from_host(Dtype::F32, vec![tokens], &f32b(&[7.0]));
        let rp = RopeT::new(64, hd, 2, 10_000.0).expect("rope new");
        crate::interpreters::eval_load(&rp, &mut face, &rp.tables(), &Default::default())
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
        crate::interpreters::eval_load(&attn, &mut gpu, &six_slots(2, 1, 8, 3), &Default::default())
            .await
            .expect("eval_load");
        let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&[0.5, -0.25, 1.0]));
        let out = attn.forward(&xs, &crate::module::ForwardCtx::minimal(1));
        assert!(out.is_poisoned(), "minimal ctx 缺 pos/kv/rope → 毒");
        let err = crate::interpreters::eval_ops(out.step(), &mut gpu)
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("缺动态依赖"), "毒值应带 ctx 归因:{err:?}");
        gpu.close().await.expect("server 关机");
    }

    /// 连续槽窗语义钉(PF1-0 契约锚;OWL_TEST_DEVICE 门控):
    /// 打分窗 = [slot-kv_len+1, slot]。缓存四行已知 k/v,q 对齐第 1 行,
    /// 逐案验证窗口落点 —— 这也是现存“slots 恒 0”接线疑云的定谳探针。
    #[tokio::test]
    async fn gpu_attn_window_semantics() {
        if !crate::testkit::gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        // hq=hkv=1, hd=2;rows: 0=[1,0] 1=[0,3] 2=[.5,.5] 3=[9,9](哨兵行不进窗)
        let k_cache = [1.0, 0.0, 0.0, 3.0, 0.5, 0.5, 9.0, 9.0];
        let v_cache = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 9.0, 9.0];
        let q = [0.0, 3.0]; // = k1 → 正确窗内应压倒性取 v1
        // 当前 token 的 k/v = 它自己槽位的值(核先写 cache 后打分,dummy 会脏化窗口)
        let cur = |slot: usize| (vec![k_cache[slot * 2], k_cache[slot * 2 + 1]], vec![v_cache[slot * 2], v_cache[slot * 2 + 1]]);
        let probe = |slot: f32, kv_len: f32| {
            let s = slot as usize;
            let (ck, cv) = cur(s);
            let kc = TensorOps::from_host(Dtype::F32, vec![4, 1, 2], &f32b(&k_cache));
            let vc = TensorOps::from_host(Dtype::F32, vec![4, 1, 2], &f32b(&v_cache));
            TensorOps::of(kernel::kernel_with(
                "owl_naive_decode_attn_f32",
                (0, 0, 0),
                (256, 1, 1),
                0,
            ))
            .arg(&TensorOps::from_host(Dtype::F32, vec![1, 1, 2], &f32b(&q)))
            .arg(&TensorOps::from_host(Dtype::F32, vec![1, 1, 2], &f32b(&ck)))
            .arg(&TensorOps::from_host(Dtype::F32, vec![1, 1, 2], &f32b(&cv)))
            .arg(&kc)
            .arg(&vc)
            .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot])))
            .arg(&TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[kv_len])))
            .arg_usize(1)
            .arg_usize(1)
            .arg_usize(1)
            .arg_usize(2)
            .with_shape(Dtype::F32, vec![1, 2])
        };

        let mut gpu = crate::testkit::gpu_client().await;
        // A1:slot=2, kv_len=3 → 窗 [0,2]:q·k=[0,9,1.5] → softmax ≈ v1
        let out = harvest(&mut gpu, &probe(2.0, 3.0)).await;
        eprintln!("[probe] A1 slot=2 kv=3 → {:?}(期望 ≈ v1=[0,1])", out);
        assert!(out[0] < 0.01 && (out[1] - 1.0).abs() < 0.01, "A1 窗 [0,2] 应≈v1,得 {out:?}");
        // A2:slot=1, kv_len=1 → 窗 [1,1] 单行 → out == v1 逐位
        let out = harvest(&mut gpu, &probe(1.0, 1.0)).await;
        eprintln!("[probe] A2 slot=1 kv=1 → {:?}(期望 == v1=[0,1])", out);
        assert!(out[0].abs() < 1e-5 && (out[1] - 1.0).abs() < 1e-5, "A2 单行窗应==v1,得 {out:?}");
        // A3:slot=0, kv_len=2 → 窗 [-1,0]:现存“slots 恒 0”接线的真实落点。
        // 行 −1 在块外(池内邻块垃圾);仅打印定谳,不做断言。
        let out = harvest(&mut gpu, &probe(0.0, 2.0)).await;
        eprintln!("[probe] A3 slot=0 kv=2 → {:?}(现存接线:窗 [-1,0],行-1=块外垃圾)", out);
        // A4:对照 —— 同写窗但合法:slot=1, kv_len=2 → 窗 [0,1] → 仍≈v1
        let out = harvest(&mut gpu, &probe(1.0, 2.0)).await;
        eprintln!("[probe] A4 slot=1 kv=2 → {:?}(窗 [0,1] 期望≈v1)", out);
        assert!(out[0] < 0.01 && (out[1] - 1.0).abs() < 0.01, "A4 窗 [0,1] 应≈v1,得 {out:?}");
        gpu.close().await.expect("server 关机");
    }
}

#[cfg(test)]
mod f16_tests {
    use super::*;
    use crate::contract::DeviceClient as _;
    use crate::testkit::{gpu_client, gpu_enabled};
    use crate::tensor::Dtype;

    /// GPU:f16 narrow gather(位型直搬,逐位一致;OWL_TEST_DEVICE 门控)
    #[tokio::test]
    async fn gpu_narrow_f16_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        // [2, 16] half → start=8 out_dim=8(gate 半段)
        // 源值过 f16 量化(上载即 f16;纯拷贝 = 对量化后参考逐位)
        let src: Vec<f32> = (0..32)
            .map(|i| half::f16::from_f32((i as f32 * 0.17) - 1.4).to_f32())
            .collect();
        let (outer, src_dim, start, out_dim) = (2usize, 16usize, 8usize, 8usize);
        let mut gpu = gpu_client().await;
        let ds = gpu.htod(Dtype::F16, &crate::contract::Shape::from(vec![2, 16]),
            &src.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>()).await.expect("htod");
        let s_decl = TensorOps::of_block(ds.id, Dtype::F16, vec![2, 16]);
        let decl = TensorOps::of(crate::kernel::kernel_with(
            "owl_narrow_strided_f16", (0, 0, 0), (256, 1, 1), 0,
        ))
        .arg(&s_decl)
        .arg_usize(outer)
        .arg_usize(src_dim)
        .arg_usize(start)
        .arg_usize(out_dim)
        .with_shape(Dtype::F16, vec![outer * out_dim]);
        let out = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.expect("eval");
        let mut buf = vec![0u8; outer * out_dim * 2];
        gpu.dtoh(&out, &mut buf).await.expect("dtoh");
        for r in 0..outer {
            for d in 0..out_dim {
                let got = half::f16::from_le_bytes([buf[(r * out_dim + d) * 2], buf[(r * out_dim + d) * 2 + 1]]).to_f32();
                assert_eq!(got, src[r * src_dim + start + d], "[{r},{d}] 纯拷贝须逐位");
            }
        }
        gpu.close().await.expect("关机");
    }
}

#[cfg(test)]
mod owl_port_tests {
    use super::*;
    use crate::contract::DeviceClient as _;
    use crate::testkit::{gpu_client, gpu_enabled};
    use crate::tensor::Dtype;

    /// GPU:owl_sigmoid_gate_mul_f16(NInfer 移植)vs host f32
    /// 验融合等价:silu 链上的 sigmoid+mul 双发射 → 单核(工单 N 收口锚)
    #[tokio::test]
    async fn gpu_sigmoid_gate_mul_f16_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let n = 1024usize; // 偶数(hidden/head_dim 全偶,契约)
        let gate: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.23) - 3.0).sin() * 2.5).collect();
        let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.11) - 1.0).cos() * 1.8).collect();
        let q = |f: f32| half::f16::from_f32(f).to_f32(); // f16 量化参考
        let mut gpu = gpu_client().await;
        let sh = crate::contract::Shape::from(vec![n]);
        let dg = gpu.htod(Dtype::F16, &sh, &gate.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>()).await.expect("gate");
        let dxx = gpu.htod(Dtype::F16, &sh, &x.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>()).await.expect("x");
        let dout = gpu.alloc(Dtype::F16, n).await.expect("alloc");
        let g_decl = TensorOps::of_block(dg.id, Dtype::F16, vec![n]);
        let x_decl = TensorOps::of_block(dxx.id, Dtype::F16, vec![n]);
        let decl = TensorOps::of(crate::kernel::kernel_with(
            "owl_sigmoid_gate_mul_f16", (0, 0, 0), (256, 1, 1), 0,
        ))
        .arg(&g_decl).arg(&x_decl).arg_usize(n)
        .with_shape(Dtype::F16, vec![n]);
        let out = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.expect("eval");
        let mut buf = vec![0u8; n * 2];
        gpu.dtoh(&out, &mut buf).await.expect("dtoh");
        let mut max_diff = 0f32;
        for i in 0..n {
            let want = q(x[i]) * (1.0 / (1.0 + (-q(gate[i])).exp()));
            let got = half::f16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]).to_f32();
            max_diff = max_diff.max((got - want).abs());
        }
        eprintln!("[gate_mul f16] max_diff = {max_diff:.5}");
        assert!(max_diff < 2e-2, "融合门乘容差 2e-2,得 {max_diff}");
        gpu.close().await.expect("关机");
    }
}

#[cfg(test)]
mod naive_attn_f16_tests {
    use super::*;
    use crate::contract::DeviceClient as _;
    use crate::testkit::{gpu_client, gpu_enabled};
    use crate::tensor::Dtype;

    fn hle(f: f32) -> [u8; 2] {
        half::f16::from_f32(f).to_le_bytes()
    }

    /// GPU:naive_attn f16 窗语义(两 token 递进窗;host f32 参考)
    /// owl 窗契约专属核(上游无对应物;数学沿 f32 版,已探针钉死)
    #[tokio::test]
    async fn gpu_naive_attn_f16_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (hq, hkv, hd, slots_n) = (2usize, 1usize, 8usize, 8usize);
        let q: Vec<f32> = (0..2 * hq * hd).map(|i| ((i as f32 * 0.31) - 2.0).sin()).collect();
        let k: Vec<f32> = (0..2 * hkv * hd).map(|i| ((i as f32 * 0.17) - 1.0).cos()).collect();
        let v: Vec<f32> = (0..2 * hkv * hd).map(|i| ((i as f32 * 0.13) + 0.5).sin()).collect();
        let mut gpu = gpu_client().await;
        let sh = crate::contract::Shape::from;
        let hb = |v: &[f32]| v.iter().flat_map(|f| hle(*f)).collect::<Vec<u8>>();
        let dq = gpu.htod(Dtype::F16, &sh(vec![2, hq, hd]), &hb(&q)).await.unwrap();
        let dk = gpu.htod(Dtype::F16, &sh(vec![2, hkv, hd]), &hb(&k)).await.unwrap();
        let dv = gpu.htod(Dtype::F16, &sh(vec![2, hkv, hd]), &hb(&v)).await.unwrap();
        let kc = gpu.alloc(Dtype::F16, slots_n * hkv * hd).await.unwrap();
        let vc = gpu.alloc(Dtype::F16, slots_n * hkv * hd).await.unwrap();

        // 两步:token0(slot0,win[0,0]) → token1(slot1,win[0,1])
        let mut outs = Vec::new();
        for t in 0..2usize {
            // 逐 token 展开(bs=1):切片上载,槽/窗随 t 递进(真·展开形态)
            let sl = t as f32;
            let kl = (t + 1) as f32;
            let qt = &q[t * hq * hd..(t + 1) * hq * hd];
            let kt = &k[t * hkv * hd..(t + 1) * hkv * hd];
            let vt = &v[t * hkv * hd..(t + 1) * hkv * hd];
            let dq = gpu.htod(Dtype::F16, &sh(vec![1, hq, hd]), &hb(qt)).await.unwrap();
            let dk = gpu.htod(Dtype::F16, &sh(vec![1, hkv, hd]), &hb(kt)).await.unwrap();
            let dv = gpu.htod(Dtype::F16, &sh(vec![1, hkv, hd]), &hb(vt)).await.unwrap();
            let (q_t, k_t, v_t) = (
                TensorOps::of_block(dq.id, Dtype::F16, vec![1, hq, hd]),
                TensorOps::of_block(dk.id, Dtype::F16, vec![1, hkv, hd]),
                TensorOps::of_block(dv.id, Dtype::F16, vec![1, hkv, hd]),
            );
            let kc_d = TensorOps::of_block(kc.id, Dtype::F16, vec![slots_n, hkv, hd]);
            let vc_d = TensorOps::of_block(vc.id, Dtype::F16, vec![slots_n, hkv, hd]);
            let decl = TensorOps::of(crate::kernel::kernel_with(
                "owl_naive_decode_attn_f16", (0, 0, 0), (256, 1, 1), 0,
            ))
            .arg(&q_t).arg(&k_t).arg(&v_t).arg(&kc_d).arg(&vc_d)
            .arg(&TensorOps::from_host(Dtype::F32, vec![1], &sl.to_le_bytes().to_vec()))
            .arg(&TensorOps::from_host(Dtype::F32, vec![1], &kl.to_le_bytes().to_vec()))
            .arg_usize(1).arg_usize(hq).arg_usize(hkv).arg_usize(hd)
            .with_shape(Dtype::F16, vec![1, hq * hd]);
            let o = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.unwrap();
            let mut b = vec![0u8; 1 * hq * hd * 2];
            gpu.dtoh(&o, &mut b).await.unwrap();
            outs.push(b);
        }

        // host f32 参考(token1:窗 [0,1],token0 权重可解析计算)
        let scale = 1.0 / (hd as f32).sqrt();
        let mut max_diff = 0f32;
        for t in 0..2usize {
            let kv_len = t + 1;
            for h in 0..hq {
                let kvh = h * hkv / hq;
                // 每头先算窗内权重,再对每个输出维加权求和
                let mut w = [0f32; 8];
                let mut wsum = 0f32;
                for s in 0..kv_len {
                    let mut score = 0f32;
                    for dd in 0..hd {
                        score += q[(t * hq + h) * hd + dd] * k[(s * hkv + kvh) * hd + dd];
                    }
                    w[s] = (score * scale).exp();
                    wsum += w[s];
                }
                for d in 0..hd {
                    let mut want = 0f32;
                    for s in 0..kv_len {
                        want += (w[s] / wsum) * v[(s * hkv + kvh) * hd + d];
                    }
                    let got = half::f16::from_le_bytes([
                        outs[t][(h * hd + d) * 2],
                        outs[t][(h * hd + d) * 2 + 1],
                    ]).to_f32();
                    max_diff = max_diff.max((got - want).abs());
                }
            }
        }
        eprintln!("[naive_attn f16] max_diff = {max_diff:.5}");
        assert!(max_diff < 5e-2, "窗语义容差 5e-2,得 {max_diff}");
        gpu.close().await.expect("关机");
    }
}
