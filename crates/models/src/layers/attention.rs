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

    /// 非连续窄切物化(行展平 r = t*H + h,行内 stride = src_dim)。
    /// q_raw [T, H×2HD] → q(start=0)/ gate(start=HD),各 [T, H×HD]。
    fn narrow(src: &TensorOps, outer: usize, src_dim: usize, start: usize, out_dim: usize, shape: crate::contract::Shape) -> TensorOps {
        TensorOps::of(kernel::kernel_with(
            "owl_narrow_strided_f32",
            (0, 0, 0), // 哨兵:逐元素核,自动 1D ceil/256
            (256, 1, 1),
            0,
        ))
        .arg(src)
        .arg_usize(outer)
        .arg_usize(src_dim)
        .arg_usize(start)
        .arg_usize(out_dim)
        .with_shape(Dtype::F32, shape)
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
        let q = Self::narrow(&q_raw, tokens * self.hq, self.hd * 2, 0, self.hd, flat_shape.clone());
        let gate = Self::narrow(&q_raw, tokens * self.hq, self.hd * 2, self.hd, self.hd, flat_shape);

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
