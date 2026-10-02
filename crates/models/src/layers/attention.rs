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
use owl_kernels::driver;
use crate::ops::ids;
use crate::layers::linear::Linear;
use crate::layers::{concat_rows_hier, narrow_strided};
use crate::layers::rmsnorm::RmsNorm;
use crate::module::{ForwardCtx, Loadable, LoaderCtx, LoaderOps, Module, QuantPlan, KvBuffers};
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
    /// 准备容器(0.8B:hq=8, hkv=2, hd=256 → q_proj out 4096 / kv out 512;
    /// `plan` = 量化计划构造期注入,尺寸门控在 Linear 内定形)
    pub fn new(hq: usize, hkv: usize, hd: usize, hidden: usize, eps: f32, plan: QuantPlan) -> Attention {
        Attention {
            q_proj: Linear::new("q_proj", hq * hd * 2, hidden, plan),
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

    /// 计算声明(decode;xs [T, hidden],T = ctx.tokens;C4 后回归 Module)。
    /// rope 表与 pos / KV 引用由 ctx 注入(rope 全局一份,表已是设备块);
    /// 缺任一动态依赖 → 毒值声明(eval 边界收割,与未装载槽同构)。
    fn forward_decl(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        // PF1a 分派(批P4 实装):槽/kv_len 表 [T],逐 token 窄视图 +
        // naive_attn 单发(写-打分同 launch 的跨 token 竞态 ⇒ 不批)
        if ctx.kind == crate::module::StepKind::Prefill {
            return self.forward_prefill_decl(xs, ctx);
        }
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

        // per-head [value|gate] 切分(C1):value 半段由 norm_rope strided
        // 直取(q 链 narrow+norm+rope 三发合一);gate 段仍 narrow(W2)
        let dt_raw = q_raw.dtype;
        // C1-W2 融合分派(默认关,挂案:27B 真权重域 k 值偏差立案中):
        // OWL_QKV_FUSE=1 → qkv_norm_rope_insert 单发(norm+rope+KV 插入)
        let pol = crate::module::kv_paged_policy(dt_raw);
        let page_ok = pol.as_ref()
            .map(|p| driver::attn::paged_decode_ok(self.hd, p.page))
            .unwrap_or(false);
        let force_naive = ctx.env.attn.force_naive_prefill;
        let use_fused_insert =
            dt_raw == Dtype::F16 && !force_naive && ctx.env.attn.qkv_fuse && page_ok;
        if use_fused_insert {
            let page = pol.as_ref().unwrap().page;
            let (cos_d, sin_d) = rope.cos_sin_decl();
            let gate = narrow_strided(&q_raw, tokens * self.hq, self.hd * 2, self.hd, self.hd,
                vec![tokens, self.hq * self.hd]);
            let q = TensorOps::call(ids::ATTN_QKV_NORM_ROPE_INSERT)
                .arg(&q_raw)
                .arg(&k)
                .arg(&v)
                .arg(&kv.k_cache)
                .arg(&kv.v_cache)
                .arg(&kv.slots)
                .arg(&self.q_norm.alpha_decl())
                .arg(&self.k_norm.alpha_decl())
                .arg(&cos_d)
                .arg(&sin_d)
                .arg(pos)
                .arg_f32(self.q_norm.eps())
                .arg_i32(self.hkv as i32)
                .arg_i32(rope.rotary_half() as i32)
                .arg_i32(page as i32)
                .arg_i32(1)                                 // w_off = ×(1+w)
                .aux(&[tokens, self.hq, self.hkv, self.hd, rope.rotary_half()])
                .with_shape(Dtype::F16, vec![tokens, self.hq * self.hd]);
            // 融合分派:v1 免 K0(cache 已由融合核写;树序:q 为 v1 父)
            return self.paged_decode_v1(&q, &gate, kv, tokens, ctx, pol.as_ref().unwrap());
        }
        let (q, gate, k) = if dt_raw == Dtype::F16 {
            // 融合:qk-norm(×(1+w))+ rotate-half partial rope 单发;
            // strided 读 q_raw(stride = q_raw 行长,head 步 = 2HD)
            let (cos_d, sin_d) = rope.cos_sin_decl();
            let gate = narrow_strided(&q_raw, tokens * self.hq, self.hd * 2, self.hd, self.hd,
                vec![tokens, self.hq * self.hd]);
            let q = TensorOps::call(ids::ATTN_NORM_ROPE)
                .arg(&q_raw)
                .arg(&self.q_norm.alpha_decl())
                .arg(&cos_d)
                .arg(&sin_d)
                .arg(pos)
                .arg_f32(self.q_norm.eps())
                .arg_usize(self.hq * self.hd * 2) // row_stride(q_raw 行长)
                .arg_usize(self.hd * 2)                     // head_stride
                .arg_usize(rope.rotary_half())
                .arg_i32(1)                                 // w_off = ×(1+w)
                .aux(&[tokens, self.hq, self.hd])
                .with_shape(Dtype::F16, vec![tokens, self.hq * self.hd]);
            let k = TensorOps::call(ids::ATTN_NORM_ROPE)
                .arg(&k)
                .arg(&self.k_norm.alpha_decl())
                .arg(&cos_d)
                .arg(&sin_d)
                .arg(pos)
                .arg_f32(self.k_norm.eps())
                .arg_usize(self.hkv * self.hd)     // row_stride(k 连续)
                .arg_usize(self.hd)
                .arg_usize(rope.rotary_half())
                .arg_i32(1)
                .aux(&[tokens, self.hkv, self.hd])
                .with_shape(Dtype::F16, vec![tokens, self.hkv * self.hd]);
            (q, gate, k)
        } else {
            // f32 语义锚链(CPU face / f32 全模;逐算子组合保持)
            let flat_shape = vec![tokens, self.hq * self.hd];
            let q = narrow_strided(&q_raw, tokens * self.hq, self.hd * 2, 0, self.hd, flat_shape.clone());
            let gate = narrow_strided(&q_raw, tokens * self.hq, self.hd * 2, self.hd, self.hd, flat_shape);
            let q = self.q_norm.forward(&q, ctx);
            let k = self.k_norm.forward(&k, ctx);
            let q = rope.forward_q(&q, pos, tokens, self.hq);
            let k = rope.forward_k(&k, pos, tokens, self.hkv);
            (q, gate, k)
        };

        // naive decode attention(slot 直排;一线程一 (t, q_head));
        // 核名/输出 dtype 跟随 q 声明(F5 整模切换;KV cache f16)
        let dt = q.dtype;
        // 诊断开关(OWL_FORCE_NAIVE=1):强制 naive 回退路径 —— paged 核
        // 数值质量的 A/B 对照(2026-09-27 文本退化排查)
        let force_naive = ctx.env.attn.force_naive_prefill;
        if let Some(pol) = ctx.env.kv.policy() {
            // paged 分派(K1):K0 写核 + v1 分页打分(vLLM classic 布局;
            // 页/x 来自 kv_paged_policy,块表语义见 block_tables 头注。
            // 页配对律住 driver(attn::paged_decode_ok / resolve 内 wrapper
            // 选择)—— 谓词门控声明,执行期 env.page 终审)
            if !force_naive && driver::attn::paged_decode_ok(self.hd, pol.page) {
                // 可达性:use_fused_insert 已覆盖同谓词;防御臂(理论不可达)
                // —— K0 形态保留以防 future 分派变化
                let wr = TensorOps::call(ids::ATTN_K0_WRITE).aux(&[tokens])
                .arg(&k)
                .arg(&v)
                .arg(&kv.k_cache)
                .arg(&kv.v_cache)
                .arg(&kv.slots)
                .arg_i32(self.hkv as i32 * self.hd as i32)
                .arg_i32(self.hkv as i32 * self.hd as i32)
                .arg_i32(self.hkv as i32)
                .arg_i32(self.hd as i32)
                .arg_i32(pol.page as i32)
                .arg_i32(pol.x as i32)
                .with_shape(dt, vec![1]);
                return self.paged_decode_v1_k0(&q, &gate, &wr, kv, tokens, ctx, &pol);
            }
        }
        let y = TensorOps::call(ids::ATTN_NAIVE_DECODE) // 哨兵;核内有 bs 上界 guard
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
            TensorOps::call(ids::ATTN_GATE_MUL)
            .arg(&gate)
            .arg(&y)
            .arg_usize(n)
            .with_shape(Dtype::F16, vec![tokens, self.hq * self.hd])
        } else {
            y.mul(&gate.sigmoid())
        };
        self.o_proj.forward(&y, ctx)
    }

    /// v1 分页打分(融合核变体;KV 写入已由融合核完成)。
    fn paged_decode_v1(
        &self,
        q: &TensorOps,
        gate: &TensorOps,
        kv: &KvBuffers,
        tokens: usize,
        ctx: &ForwardCtx,
        pol: &crate::module::KvPagedPolicy,
    ) -> TensorOps {
        let wr = q.clone();
        self.paged_decode_v1_k0(q, gate, &wr, kv, tokens, ctx, pol)
    }

    /// v1 分页打分(K0 形态;alibi 槽 = K0 哑输出,树序依赖边)
    fn paged_decode_v1_k0(
        &self,
        q: &TensorOps,
        gate: &TensorOps,
        wr: &TensorOps,
        kv: &KvBuffers,
        tokens: usize,
        ctx: &ForwardCtx,
        pol: &crate::module::KvPagedPolicy,
    ) -> TensorOps {
        let dt = q.dtype;
        let page = pol.page as i32;
        // v1 分页打分(不写 cache;context_lens = kv_lens)
        let scale = 1.0 / (self.hd as f32).sqrt();
        let nb = kv.block_tables.shape().last().cloned().unwrap_or(1);
        // wrapper 页配对 + smem 契约公式 = driver 单源(env.page 终审;
        // 原手抄 smem 越界事故与 bs16/32 错配案的结构性封点)
        let y = TensorOps::call(ids::ATTN_PAGED_DECODE)
            .aux(&[self.hd, self.hq, self.hkv, nb as usize])
        .arg(q)
        .arg(&kv.k_cache)
        .arg(&kv.v_cache)
        .arg(&kv.block_tables)
        .arg(&kv.kv_lens) // context_lens
        .arg(wr) // alibi 槽 = 写核输出块(树序依赖边;旗标 0 不解引用)
        .arg_i32(self.hkv as i32)
        .arg_f32(scale)
        .arg_i32(nb as i32)
        .arg_i32(self.hq as i32 * self.hd as i32) // q_stride
        .arg_i32(self.hkv as i32 * self.hd as i32 * page) // kv_block_stride
        .arg_i32(self.hd as i32 * page) // kv_head_stride
        .arg_f32(1.0) // softscapping 直通
        .arg_i32(-1) // sliding_window 关
        .arg_i32(0) // use_alibi 关
        .with_shape(dt, vec![tokens, self.hq * self.hd]);
        // ③ 输出门(f16 融合单发)+ 出投影(与 legacy 尾巴同)
        let n = tokens * self.hq * self.hd;
        let y = TensorOps::call(ids::ATTN_GATE_MUL)
        .arg(gate)
        .arg(&y)
        .arg_usize(n)
        .with_shape(dt, vec![tokens, self.hq * self.hd]);
        self.o_proj.forward(&y, ctx)
    }



    /// paged prefill(PF1 终;F16 hd∈{128,256}):K0 批量写池(T 行散写)
    /// + chunked prefill 批核(bs16 特化;因果语义 = 查询 token t 看
    /// [0, seq_start+t])。naive 逐 token 路径保留为回退。
    fn paged_prefill_output(
        &self,
        q: &TensorOps,
        k: &TensorOps,
        v: &TensorOps,
        gate: &TensorOps,
        kv: &KvBuffers,
        tokens: usize,
        ctx: &ForwardCtx,
        pol: &crate::module::KvPagedPolicy,
        kv_slots: &TensorOps,
        kv_lens: &TensorOps,
    ) -> TensorOps {
        let dt = q.dtype;
        let page = pol.page as i32;
        let x = pol.x;
        // ① K0 批量写池(slots [T] = 物理槽表;恒等分页下 = pos)。
        // 槽/长度表单源 = ctx(与下方 naive 循环同源):KvBuffers.slots/
        // kv_lens 是 decode 步字段(尺寸 [B]),prefill 误读曾致 K0 grid=T
        // 越界读 + seq_lens 垃圾 → 输出全零/ILLEGAL_ADDRESS(2026-10-01)
        let wr = TensorOps::call(ids::ATTN_K0_WRITE).aux(&[tokens])
        .arg(k)
        .arg(v)
        .arg(&kv.k_cache)
        .arg(&kv.v_cache)
        .arg(kv_slots)
        .arg_i32(self.hkv as i32 * self.hd as i32)
        .arg_i32(self.hkv as i32 * self.hd as i32)
        .arg_i32(self.hkv as i32)
        .arg_i32(self.hd as i32)
        .arg_i32(page)
        .arg_i32(x as i32)
        .with_shape(dt, vec![1]); // 哑输出(契约 4)
        // ② chunked prefill 批核(bs16;seq_lens [1] = kv_lens 末元;narrow 树序亦成立)
        let seq_lens = narrow_strided(kv_lens, 1, 1, tokens - 1, 1, vec![1]);
        let qsl: Vec<u8> = [0.0f32, tokens as f32].iter().flat_map(|f| f.to_le_bytes()).collect();
        let scale = 1.0 / (self.hd as f32).sqrt();
        let nb = kv.block_tables.shape().last().cloned().unwrap_or(1);
        // 名/网格/smem(bs32 契约)= driver 单源(env.page 终审)
        let y = TensorOps::call(ids::ATTN_PAGED_PREFILL).aux(&[
            self.hd, self.hkv, self.hq, tokens,
        ])
        .arg(q)
        .arg(&kv.k_cache)
        .arg(&kv.v_cache)
        .arg(&kv.block_tables)
        .arg(&seq_lens)
        .arg(&TensorOps::from_host(Dtype::F32, vec![2], &qsl)) // query_start_len(急切路径,from_host 合法)
        .arg(&wr)       // alibi 槽 = 写核输出(树序依赖;旗标 0 不解引用)
        .arg(&kv.slots) // sinks 槽(旗标 0 不解引用)
        .arg_i32(self.hkv as i32)
        .arg_f32(scale)
        .arg_i32(1) // block_table_stride(单序列)
        .arg_i32(1) // num_seqs
        .arg_i32(self.hq as i32)
        .arg_i32(tokens as i32)
        .arg_f32(1.0) // softscapping 直通
        .arg_i32(self.hq as i32 * self.hd as i32) // o_stride_tokens
        .arg_i32(-1) // sliding_window 关
        .arg_i32(nb as i32) // total_num_blocks
        .arg_i32(self.hkv as i32 * self.hd as i32 * page) // kv_block_stride
        .arg_i32(self.hd as i32 * page) // kv_head_stride
        .arg_i32(0) // use_alibi 关
        .arg_i32(0) // use_sinks 关
        .with_shape(dt, vec![tokens, self.hq * self.hd]);
        // 输出门(f16 融合单发)+ 出投影(与 legacy 同)
        let n = tokens * self.hq * self.hd;
        let y = TensorOps::call(ids::ATTN_GATE_MUL)
        .arg(gate)
        .arg(&y)
        .arg_usize(n)
        .with_shape(dt, vec![tokens, self.hq * self.hd]);
        self.o_proj.forward(&y, ctx)
    }

    /// 长 ctx split attention(flash-decoding;2026-10-02;prefill 主案):
    /// nparts = ceil(max_ctx/512) ≥ 4(max_ctx > 2048)即走;K0 写池 →
    /// K1 分块在线 softmax(partial + (m,l) 入 scratch)→ K2 归一化合并。
    /// 块数 ×nparts(occupancy 主升),scratch 由活性回收随 chunk 归池。
    fn paged_prefill_split_output(
        &self,
        q: &TensorOps,
        k: &TensorOps,
        v: &TensorOps,
        gate: &TensorOps,
        kv: &KvBuffers,
        tokens: usize,
        ctx: &ForwardCtx,
        pol: &crate::module::KvPagedPolicy,
        kv_slots: &TensorOps,
        nparts: usize,
    ) -> TensorOps {
        let dt = q.dtype;
        let page = pol.page as i32;
        // ① K0 批量写池(本 chunk k/v 先入池;slots [T] = 物理槽表,engine 单源)
        let wr = TensorOps::call(ids::ATTN_K0_WRITE).aux(&[tokens])
        .arg(k)
        .arg(v)
        .arg(&kv.k_cache)
        .arg(&kv.v_cache)
        .arg(kv_slots)
        .arg_i32(self.hkv as i32 * self.hd as i32)
        .arg_i32(self.hkv as i32 * self.hd as i32)
        .arg_i32(self.hkv as i32)
        .arg_i32(self.hd as i32)
        .arg_i32(page)
        .arg_i32(pol.x as i32)
        .with_shape(dt, vec![1]); // 哑输出(契约 4)
        // ② K1 分块在线 softmax(partial + (m,l) 入 scratch;wr = 树序依赖边)
        let scr_out = TensorOps::zeros(dt, vec![tokens * self.hq * nparts * self.hd]);
        let scr_stat = TensorOps::zeros(Dtype::F32, vec![tokens * self.hq * nparts * 2]);
        let scale = 1.0 / (self.hd as f32).sqrt();
        let sp = TensorOps::call(ids::ATTN_PREFILL_SPLIT)
        .arg(q)
        .arg(&kv.k_cache)
        .arg(&kv.v_cache)
        .arg(&kv.block_tables)
        .arg(&scr_out)
        .arg(&scr_stat)
        .arg(&wr) // 树序依赖边(K0 先于分块读池;核不解引用)
        .arg_f32(scale)
        .arg_i32(self.hkv as i32)
        .arg_i32(tokens as i32)
        .arg_i32(ctx.ctx_base as i32)
        .arg_i32(nparts as i32)
        .arg_i32(self.hkv as i32 * self.hd as i32 * page) // kv_block_stride
        .arg_i32(self.hd as i32 * page) // kv_head_stride
        .arg_i32(page)
        .arg_i32(self.hq as i32)
        .aux(&[self.hd, self.hkv, self.hq, tokens, nparts])
        .with_shape(dt, vec![1]); // 哑输出(真输出 = reduce)
        // ③ K2 归一化合并(split 哑输出 = 树序依赖边;out [T, Hq*hd])
        let y = TensorOps::call(ids::ATTN_PREFILL_SPLIT_REDUCE)
        .arg(&sp)
        .arg(&scr_out)
        .arg(&scr_stat)
        .arg_i32(nparts as i32)
        .arg_i32(self.hq as i32)
        .arg_i32(self.hd as i32)
        .arg_i32(tokens as i32)
        .aux(&[tokens, self.hq])
        .with_shape(dt, vec![tokens, self.hq * self.hd]);
        // ④ 输出门(f16 融合单发)+ 出投影(与旧路径同)
        let n = tokens * self.hq * self.hd;
        let y = TensorOps::call(ids::ATTN_GATE_MUL)
        .arg(gate)
        .arg(&y)
        .arg_usize(n)
        .with_shape(dt, vec![tokens, self.hq * self.hd]);
        self.o_proj.forward(&y, ctx)
    }

    /// FlashInfer paged prefill(E1.5;OWL_FLASHINFER=1,engine 注入
    /// ForwardCtx.fi):K0-dual 写池(classic + kHND 影子)→ FI 虚拟核
    /// (server plan 缓存 + run;FA2 级 tensor-core,causal chunked 语义
    /// = q 对齐 kv 尾部,与 K0 先行的池读序配套)。
    /// 槽序契约:7 Block + O + 8 sz(owl_kernels::flashinfer::PREFILL_FI_SLOTS;
    /// 字面量对齐 —— models 不开 kernels feature,marlin 先例)。
    #[allow(clippy::too_many_arguments)]
    fn paged_prefill_fi_output(
        &self,
        q: &TensorOps,
        k: &TensorOps,
        v: &TensorOps,
        gate: &TensorOps,
        kv: &KvBuffers,
        tokens: usize,
        ctx: &ForwardCtx,
        pol: &crate::module::KvPagedPolicy,
        kv_slots: &TensorOps,
        fi: &crate::module::FiPrefillCtx<'_>,
    ) -> TensorOps {
        let dt = q.dtype;
        let page = pol.page as i32;
        let fp8kv = ctx.env.kv.quant == crate::env::KvQuant::Fp8E4M3;
        // ① K0-dual:classic K/V + kNHD K/V 影子(f16 或 e4m3;slots = 物理槽表)
        let wr = TensorOps::call(if fp8kv { ids::ATTN_K0_DUAL_FP8KV } else { ids::ATTN_K0_DUAL })
            .aux(&[tokens])
            .arg(k)
            .arg(v)
            .arg(&kv.k_cache)
            .arg(&kv.v_cache)
            .arg(&fi.kcs[ctx.fi_kvi])
            .arg(&fi.vcs[ctx.fi_kvi])
            .arg(kv_slots)
            .arg_i32(self.hkv as i32 * self.hd as i32)
            .arg_i32(self.hkv as i32 * self.hd as i32)
            .arg_i32(self.hkv as i32)
            .arg_i32(self.hd as i32)
            .arg_i32(page)
            .arg_i32(pol.x as i32)
            .with_shape(dt, vec![1]); // 哑输出(契约 4)
        // ② FI prefill(q 已是 [T, Hq*hd] 投影+norm+rope 后;out [T, Hq*hd])
        let ctx_total = ctx.ctx_base + tokens;
        let y = TensorOps::of(
            crate::kernel::Kernel::new(
                if fp8kv { "flashinfer_prefill_paged_fp8kv" } else { "flashinfer_prefill_paged_f16" },
                "",
            )
            .with_sig("T,T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz,sz"),
        )
        .arg(q)
        .arg(&fi.kcs[ctx.fi_kvi])
        .arg(&fi.vcs[ctx.fi_kvi])
        .arg(fi.q_cu)
        .arg(fi.indices)
        .arg(fi.indptr)
        .arg(fi.last_len)
        .arg(&wr) // 树序依赖边(K0 先于分块读池;FI 不解引用)
        .arg_usize(tokens)          // total_rows(本 chunk q 行)
        .arg_usize(ctx_total)       // kv_indptr host 端点
        .arg_usize(tokens)          // T
        .arg_usize(self.hq)
        .arg_usize(self.hkv)
        .arg_usize(self.hd)
        .arg_usize(pol.page)
        .arg_usize((1.0f32 / (self.hd as f32).sqrt()).to_bits() as usize)
        .with_shape(dt, vec![tokens, self.hq * self.hd]);
        // ③ 输出门(f16 融合单发)+ 出投影(与旧路径同)
        let n = tokens * self.hq * self.hd;
        let y = TensorOps::call(ids::ATTN_GATE_MUL)
            .arg(gate)
            .arg(&y)
            .arg_usize(n)
            .with_shape(dt, vec![tokens, self.hq * self.hd]);
        self.o_proj.forward(&y, ctx)
    }

    /// prefill 展开声明(PF1a;批P4):投影/qk-norm/rope/gate T 批量单发


    /// (rope grid=(tokens,1,1) 逐行读 pos[t],T 就绪);唯逐 token 段 =
    /// naive_attn(写-打分同 launch 的跨 token 竞态 ⇒ 不批;槽/kv_len 取
    /// [T] 表行 t 窄视图)。语义 = T 次 decode 步(验收锚,批P4 对拍)。
    fn forward_prefill_decl(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        let tokens = ctx.tokens;
        let (rope, pos, kv, kv_slots, kv_lens) = match (
            ctx.rope, ctx.pos, ctx.kv, ctx.kv_slots, ctx.kv_lens,
        ) {
            (Some(r), Some(p), Some(k), Some(s), Some(l)) => (r, p, k, s, l),
            _ => {
                return TensorOps::poisoned(
                    Dtype::F16,
                    vec![tokens, self.hidden],
                    "attention prefill: ctx 缺动态依赖(需 pos+kv+rope+槽表/kv_len表;ForwardCtx::attn_prefill)",
                );
            }
        };
        let row_q = self.hq * self.hd;
        let row_kv = self.hkv * self.hd;

        // T 批量投影 + per-head [value|gate] 切分(同 decode)
        let q_raw = self.q_proj.forward(xs, ctx); // [T, 2*Hq*HD]
        let k = self.k_proj.forward(xs, ctx); // [T, Hkv*HD]
        let v = self.v_proj.forward(xs, ctx); // [T, Hkv*HD]
        // C1 融合:同 decode 臂(norm_rope strided 直取 value 半段)
        let dt_raw = q_raw.dtype;
        let (q, gate, k) = if dt_raw == Dtype::F16 {
            let (cos_d, sin_d) = rope.cos_sin_decl();
            let gate = narrow_strided(&q_raw, tokens * self.hq, self.hd * 2, self.hd, self.hd,
                vec![tokens, row_q]);
            let q = TensorOps::call(ids::ATTN_NORM_ROPE)
                .arg(&q_raw)
                .arg(&self.q_norm.alpha_decl())
                .arg(&cos_d)
                .arg(&sin_d)
                .arg(pos)
                .arg_f32(self.q_norm.eps())
                .arg_usize(self.hq * self.hd * 2)
                .arg_usize(self.hd * 2)
                .arg_usize(rope.rotary_half())
                .arg_i32(1)
                .aux(&[tokens, self.hq, self.hd])
                .with_shape(Dtype::F16, vec![tokens, row_q]);
            let k = TensorOps::call(ids::ATTN_NORM_ROPE)
                .arg(&k)
                .arg(&self.k_norm.alpha_decl())
                .arg(&cos_d)
                .arg(&sin_d)
                .arg(pos)
                .arg_f32(self.k_norm.eps())
                .arg_usize(self.hkv * self.hd)
                .arg_usize(self.hd)
                .arg_usize(rope.rotary_half())
                .arg_i32(1)
                .aux(&[tokens, self.hkv, self.hd])
                .with_shape(Dtype::F16, vec![tokens, self.hkv * self.hd]);
            (q, gate, k)
        } else {
            let flat_shape = vec![tokens, row_q];
            let q = narrow_strided(&q_raw, tokens * self.hq, self.hd * 2, 0, self.hd, flat_shape.clone());
            let gate = narrow_strided(&q_raw, tokens * self.hq, self.hd * 2, self.hd, self.hd, flat_shape);
            let q = self.q_norm.forward(&q, ctx);
            let k = self.k_norm.forward(&k, ctx);
            let q = rope.forward_q(&q, pos, tokens, self.hq);
            let k = rope.forward_k(&k, pos, tokens, self.hkv);
            (q, gate, k)
        };

        let dt = q.dtype;
        if dt == Dtype::F16 {
            if let Some(pol) = crate::module::kv_paged_policy(dt) {
                // 页守卫(2026-10-01 配对律):prefill 核 bs32 特化(vendor
                // 契约 BLOCK∈{32,64}),仅页 32 池可进;页 16 池回退 naive
                //(bs16 prefill 实例化 = 越契约,挂账)。smem 公式的 page 项
                // 与核 BLOCK 同源,见 paged_prefill_output
                if driver::attn::paged_prefill_ok(self.hd, pol.page) {
                    // 诊断二分开关保留(OWL_FORCE_NAIVE_PREFILL=1 走逐 token
                    // 对照;2026-10-01 撤销遗留的 if false 硬禁用 —— 它让
                    // 全部 prefill 注意力落 naive 逐 token 路径)
                    if !ctx.env.attn.force_naive_prefill {
                        // FlashInfer prefill(E1.5;OWL_FLASHINFER=1 → engine
                        // 注入 ForwardCtx.fi):FA2 级 tensor-core,主臂
                        if let Some(fi) = &ctx.fi {
                            return self.paged_prefill_fi_output(
                                &q, &k, &v, &gate, kv, tokens, ctx, &pol, kv_slots, fi,
                            );
                        }
                        // 长 ctx split(flash-decoding;2026-10-02):nparts ≥ 4
                        // (max_ctx > 2048)走分块路径;仅 hd256 核(hd128 变体
                        // 挂账)—— W3 结案:两雷已清,默认仍 opt-in(引擎实测
                        // 后裁决翻默认)
                        let nparts = (ctx.ctx_base + tokens).div_ceil(512);
                        if nparts >= 4 && self.hd == 256 && ctx.env.attn.prefill_split {
                            return self.paged_prefill_split_output(
                                &q, &k, &v, &gate, kv, tokens, ctx, &pol, kv_slots, nparts,
                            );
                        }
                        return self.paged_prefill_output(
                            &q, &k, &v, &gate, kv, tokens, ctx, &pol, kv_slots, kv_lens,
                        );
                    }
                }
            }
        }

        // 逐 token 段:naive_attn 单发(bs=1;槽/kv_len 取表行 t 窄视图)
        let mut ys: Vec<TensorOps> = Vec::with_capacity(tokens);
        for t in 0..tokens {
            let q_t = narrow_strided(&q, 1, row_q, t * row_q, row_q, vec![1, row_q]);
            let k_t = narrow_strided(&k, 1, row_kv, t * row_kv, row_kv, vec![1, row_kv]);
            let v_t = narrow_strided(&v, 1, row_kv, t * row_kv, row_kv, vec![1, row_kv]);
            let slot_t = narrow_strided(kv_slots, 1, 1, t, 1, vec![1]);
            let len_t = narrow_strided(kv_lens, 1, 1, t, 1, vec![1]);
            let y_t = TensorOps::call(ids::ATTN_NAIVE_DECODE) // 哨兵;核内 bs 上界 guard
            .arg(&q_t)
            .arg(&k_t)
            .arg(&v_t)
            .arg(&kv.k_cache)
            .arg(&kv.v_cache)
            .arg(&slot_t)
            .arg(&len_t)
            .arg_usize(1)
            .arg_usize(self.hq)
            .arg_usize(self.hkv)
            .arg_usize(self.hd)
            .with_shape(dt, vec![1, row_q]);
            ys.push(y_t);
        }

        // 栈(T×[1,Hq·HD] → [T,Hq·HD];层级栈,T≤64)→ 门(T 批量)→ 出投影
        let refs: Vec<&TensorOps> = ys.iter().collect();
        let y_all = concat_rows_hier(&refs, row_q);
        let y = if dt == Dtype::F16 {
            let n = tokens * row_q;
            TensorOps::call(ids::ATTN_GATE_MUL)
            .arg(&gate)
            .arg(&y_all)
            .arg_usize(n)
            .with_shape(Dtype::F16, vec![tokens, row_q])
        } else {
            y_all.mul(&gate.sigmoid())
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
        let attn = Attention::new(hq, hkv, hd, hidden, 1e-6, QuantPlan::F16);
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
            block_tables: TensorOps::zeros(Dtype::F32, vec![1]), // 哑表(legacy 路径不解引用)
        };
        let ctx = crate::module::ForwardCtx::decode(tokens, &pos, &kv, &rp);

        let out = attn.forward(&xs, &ctx);
        assert!(!out.is_poisoned(), "装载后声明不应有毒");
        assert_eq!(out.shape(), &[tokens, hidden]);

        // prefill 分派(PF1-0):kind=Prefill → forward_prefill_decl
        // (批P4 前为结构化毒值;decode 路不破)
        let t8 = 2usize;
        let pos8 = TensorOps::from_host(Dtype::F32, vec![t8], &f32b(&[7.0, 8.0]));
        let slots8 = TensorOps::from_host(Dtype::F32, vec![t8], &f32b(&[0.0, 1.0]));
        let lens8 = TensorOps::from_host(Dtype::F32, vec![t8], &f32b(&[1.0, 2.0]));
        let xs8 = TensorOps::from_host(Dtype::F32, vec![t8, hidden], &f32b(&vec![0.5; t8 * hidden]));
        let pctx = crate::module::ForwardCtx::attn_prefill(t8, &pos8, &kv, &rp, &slots8, &lens8);
        let out_p = attn.forward(&xs8, &pctx);
        let _ = out_p; // f16 装载形态由 f16_tests 层级对拍覆盖(CPU 面不追)
        let out_d = attn.forward(&xs, &ctx);
        assert!(!out_d.is_poisoned(), "prefill 分派后 decode 路不破");

        // 毒值契约:未装载容器 → forward 声明立即带毒
        let attn2 = Attention::new(hq, hkv, hd, hidden, 1e-6, QuantPlan::F16);
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
        let attn = Attention::new(2, 1, 8, 3, 1e-6, QuantPlan::F16);
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
    use crate::module::KvBuffers;
    use crate::testkit::{f32b, gpu_client, gpu_enabled};
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

    /// v1 paged decode 隔离对拍(2026-10-01 立案;历史盲区:
    /// 既有 decode 对照测试 hd=4 不进 paged 分派,v1 真实形态零覆盖)——
    /// paged 池 + 真块表;参考臂 = paged prefill(bs32,vendor 契约档)。
    /// **配对律结案(2026-10-01,三轮反转终审)**:曾判“v1 多块数值缺陷”
    /// 的行 16+ 偏差 = 测试/层配置自伤两连 ——
    /// ① wrapper×池页错配:v1_name 曾指向 bs16 wrapper 而池页 32,
    ///   逻辑块(16 token)地址换算对不上物理块(32 token),行 16+ 全错;
    /// ② prefill 参考臂 KvBuffers 传 1 元素 decode 步表:K0 grid=T 逐线程
    ///   读 slots[t] 越界、seq_lens narrow 越界 → 垃圾 seq_len →
    ///   valid_block 全灭 → 输出全零;T=4096 档直接 ILLEGAL_ADDRESS
    ///   (即“paged_prefill 本体有病/4k 崩”立案的真凶)。
    /// 修复后:页 32 生产档(policy 单源)+ v1_name(hd,page) 配对裁决,
    /// 全矩阵收紧为硬门(无 allow_red)。
    #[tokio::test]
    async fn gpu_attn_v1_decode_matches_prefill_paged() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        // 变因矩阵:hd × GQA(hkv)× 跨块;页 32 = policy 生产档,
        // wrapper bs32 由 v1_name 页配对律裁决。
        // (2026-10-02 W2 融合核 k 支转置案结案:allow_red 立案拐杖撤除,
        // 全档硬门;融合核写池金标另见 gpu_w2_fused_kprobe)
        v1_isolation_case_inner(128, 2, 1, 8, 2, false).await;
        v1_isolation_case_inner(256, 2, 1, 8, 2, false).await;
        v1_isolation_case_inner(256, 8, 4, 8, 2, false).await; // GQA 单块
        v1_isolation_case_inner(256, 2, 1, 64, 2, false).await; // 跨 2 块(行 32 = 页边界首槽)
        v1_isolation_case_inner(256, 8, 4, 64, 2, false).await; // engine 全参档(GQA + 跨页)
        // 规模档:真模型规模 T=4096/nb=128 单发射(4k e2e 同规模;
        // 曾以越界 slots 表复现 ILLEGAL_ADDRESS,修复后应为绿)
        v1_isolation_case_inner(256, 2, 1, 4096, 128, false).await;
        gpu_client_close().await;
    }

    async fn v1_isolation_case(hd: usize, hq: usize, hkv: usize, t_len: usize, nb: usize) {
        v1_isolation_case_inner(hd, hq, hkv, t_len, nb, false).await;
    }

    async fn v1_isolation_case_inner(hd: usize, hq: usize, hkv: usize, t_len: usize, nb: usize, allow_red: bool) {
        let hidden = 6usize;
        let (page, x) = (32usize, 8usize); // 二分:暂回页 32
        let row_q = hq * hd;
        let halfb = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
        };

        let mut src = std::collections::HashMap::new();
        src.insert("q_proj".to_string(), (0..hq * hd * 2 * hidden).map(|i| ((i as f32 + 3.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("k_proj".to_string(), (0..hkv * hd * hidden).map(|i| ((i as f32 + 4.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("v_proj".to_string(), (0..hkv * hd * hidden).map(|i| ((i as f32 + 5.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("o_proj".to_string(), (0..hidden * hq * hd).map(|i| ((i as f32 + 6.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("q_norm".to_string(), (0..hd).map(|i| ((i as f32 + 7.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("k_norm".to_string(), (0..hd).map(|i| ((i as f32 + 8.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        let attn = Attention::new(hq, hkv, hd, hidden, 1e-6, QuantPlan::F16);

        let mut gpu = gpu_client().await;
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false };
        crate::interpreters::eval_load(&attn, &mut gpu, &src, &lctx)
            .await
            .expect("attention f16 装载");
        let rp = crate::layers::rope::Rope::new(t_len * 2, hd, 32, 10_000.0).expect("rope new");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &lctx)
            .await
            .expect("rope 表物化");

        // paged 池(K0 classic 布局):kc [nb, hkv, hd/x, page, x] / vc [nb, hkv, hd, page]
        let kc_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd / x, page, x]).step(), &mut gpu)
            .await.expect("kc");
        let vc_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd, page]).step(), &mut gpu)
            .await.expect("vc");
        // ⚠️ 块表必须 = 物理块号恒等表 [0,1,..,nb-1] —— 曾写成全零并
        // 误注"零即真":全零 = 所有逻辑块映射物理块 0,v1 读回块 0 旧数据
        // (2026-10-01 复读案真凶:测试 bug,非核 bug)
        let bt = TensorOps::from_host(
            Dtype::F32,
            vec![1, nb],
            &f32b(&(0..nb).map(|i| i as f32).collect::<Vec<_>>()),
        );
        let paged_kv = |slots: Vec<f32>, lens: Vec<f32>| KvBuffers {
            k_cache: TensorOps::of_block(kc_b.id, Dtype::F16, vec![nb, hkv, hd / x, page, x]),
            v_cache: TensorOps::of_block(vc_b.id, Dtype::F16, vec![nb, hkv, hd, page]),
            slots: TensorOps::from_host(Dtype::F32, vec![slots.len()], &f32b(&slots)),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![lens.len()], &f32b(&lens)),
            block_tables: bt.clone(),
        };

        let xs_f32: Vec<f32> = (0..t_len * hidden)
            .map(|i| half::f16::from_f32(((i as f32) * 0.37 - 1.0).sin()).to_f32())
            .collect();

        // 参考臂:一次 prefill 块(paged;slots [0..T] / lens [1..T])
        let xs_all = TensorOps::from_host(Dtype::F16, vec![t_len, hidden], &halfb(&xs_f32));
        let pos_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&(0..t_len).map(|t| t as f32).collect::<Vec<_>>()));
        let slots_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&(0..t_len).map(|t| t as f32).collect::<Vec<_>>()));
        let lens_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&(0..t_len).map(|t| t as f32 + 1.0).collect::<Vec<_>>()));
        // ⚠️ 参考臂 KvBuffers 的 slots/kv_lens 传 [T] 全量表(与 ctx 表
        // 同内容,双保险):KvBuffers.slots/kv_lens 是 decode 步字段(尺寸
        // [B]),曾致 paged_prefill 越界(现已层侧单源修复:prefill 读 ctx
        // 表,见 paged_prefill_output 头注);测试侧仍传全量表防回归
        let kv_p = paged_kv(
            (0..t_len).map(|t| t as f32).collect::<Vec<_>>(),
            (0..t_len).map(|t| t as f32 + 1.0).collect::<Vec<_>>(),
        );
        let decl_all = attn.forward(
            &xs_all,
            &ForwardCtx::attn_prefill(t_len, &pos_all, &kv_p, &rp, &slots_all, &lens_all),
        );
        let out_prefill = crate::testkit::harvest_f16(&mut gpu, &decl_all).await;
        let mut bad = false;

        // 被测臂:T × v1 decode 步(同一持久池;slots=[t] / lens=[t+1];
        // K0 重写同槽同值 = 幂等,池内容与 prefill 后一致)
        for t in 0..t_len {
            let x_t = TensorOps::from_host(Dtype::F16, vec![1, hidden], &halfb(&xs_f32[t * hidden..(t + 1) * hidden]));
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32]));
            let kv_t = paged_kv(vec![t as f32], vec![t as f32 + 1.0]);
            let decl = attn.forward(&x_t, &ForwardCtx::decode(1, &pos_t, &kv_t, &rp));
            let got = crate::testkit::harvest_f16(&mut gpu, &decl).await;
            let want = &out_prefill[t * hidden..(t + 1) * hidden];
            let maxd = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if std::env::var_os("OWL_RESOLVE_TRACE").is_some() {
                eprintln!(
                    "[v1-isolation][r{t}] hd{hd} hq{hq} hkv{hkv} t{t_len}: got={:?} want={:?}",
                    &got[..6], &want[..6]
                );
            }
            if maxd > 2e-2 {
                eprintln!(  // 诊断期:RED 全量打印(定位 hd128 档)
                
                    "[v1-isolation][RED] hd{hd} hq{hq} hkv{hkv} t{t_len} 行{t} 偏差 {maxd:.4}\n  got = {:?}\n  want= {:?}\n  |got|={:?}",
                    &got[..6], &want[..6],
                    got.iter().fold(0f32, |a, &b| f32::max(a, b))
                );
                bad = true;
            }
        }
        if bad && !allow_red {
            panic!("[v1-isolation] hd{hd} hq{hq} hkv{hkv} t{t_len} nb{nb} RED");
        }
        // ── 池内容取证:host 直算 V[t] = W_v × x_t(V 路径纯线性,无 norm/rope);
        // vc 布局 [nb, hkv, hd, page]:b=s/32, 头 0, 元素 d*page + s%32 ──
        {
            let mut vpool = vec![0u8; nb * hkv * hd * page * 2];
            let face = &mut gpu;
            face.dtoh(&vc_b, &mut vpool).await.expect("vc dtoh");
            let vf = |i: usize| -> f32 {
                half::f16::from_le_bytes([vpool[i * 2], vpool[i * 2 + 1]]).to_f32()
            };
            let wv = &src["v_proj"]; // flat [hkv*hd, hidden]
            let mut worst = (0f32, 0usize);
            for t in [0usize, 31, 32, 33, 63].into_iter().filter(|&t| t < t_len) {
                let (b, off) = (t / page, t % page);
                for d in 0..hd.min(8) {
                    let mut acc = 0f32;
                    for c in 0..hidden {
                        let w = wv[(d) * hidden + c];
                        let xv = half::f16::from_f32(xs_f32[t * hidden + c]).to_f32();
                        acc += w * xv;
                    }
                    let got = vf(((b * hkv + 0) * hd + d) * page + off);
                    let dd = (got - acc).abs();
                    if dd > worst.0 {
                        worst = (dd, t);
                    }
                    if t == 32 && d < 3 {
                        eprintln!("[vc-probe] t32 d{d}: pool={got:.4} host={acc:.4}");
                    }
                }
                if worst.0 > 0.0 && t == 63 {
                    eprintln!("[vc-probe] 最坏偏差 {worst:.2?}(容差域 = f16 装载量化)");
                }
            }
        }
        eprintln!("[v1-isolation] hd{hd} hq{hq} hkv{hkv} t{t_len} nb{nb} 全绿");
    }

    async fn gpu_client_close() {
        let mut gpu = gpu_client().await;
        gpu.close().await.expect("关机");
    }

    // ═══════════════════════════════════════════════════════════════════
    // W2 融合核 KV 写池金标门(k-probe;2026-10-02 立案→结案转正):
    // 三臂 × host f32 金标 ——
    //   A 臂 = 默认 decode 分派(norm_rope ×2 + K0;pos=t)
    //   B 臂 = 融合分派(owl_qknorm_rope_kv_insert;pos=t)
    //   C 臂 = 融合分派 + pos≡0(θ=0 → 池内容 = 纯 norm,norm/rope 解耦)
    //   G/N = host 金标(f16 输入/权重 → f32 线性 → rmsnorm(1+w) → rope /
    //         纯 norm;cos/sin 过 f16 量化 = 设备表同口径)
    // 硬门:|A−G|/|B−G|/|C−N|/|A−B| ≤ 3e-2(f16 噪声 + cublas 累序容差)。
    // 立案战果:融合核 k 支 [half,2half) 转置(n[d−h]·cos + n[d]·sin),
    // pos=0 时整段写 n[d−h] —— 引擎 kv-hex 3-4% 偏差唯一现行犯
    // (2026-10-02 一行修复,详见 fused.cu 同日头注)。
    // ⚠️ 教训双条:金标自身曾有两处病(p×cos(freq) 运算优先级 /
    // 转置式与被告同构)—— 判决前必须先证金标清白(A 臂双锚 + 直通维)。
    // ═══════════════════════════════════════════════════════════════════
    #[tokio::test]
    async fn gpu_w2_fused_kprobe() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        kprobe_case(128, 2, 1, 32).await;
        kprobe_case(256, 2, 1, 32).await;
        kprobe_case(256, 24, 4, 64).await; // 27B 全参档
        gpu_client_close().await;
    }

    async fn kprobe_case(hd: usize, hq: usize, hkv: usize, rotary: usize) {
        const TOL: f32 = 3e-2;
        let hidden = 6usize;
        let (page, x) = (32usize, 8usize);
        let t_len = 8usize;
        let nb = 1usize; // t_len=8 < page=32 → 单块
        let half = rotary / 2;
        let theta = 10_000.0f32;
        let eps = 1e-6f32;
        let halfb = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
        };
        let f16r = |v: f32| -> f32 { half::f16::from_f32(v).to_f32() };

        // 权重(同隔离档确定式;f16 装载)
        let mut src = std::collections::HashMap::new();
        src.insert("q_proj".to_string(), (0..hq * hd * 2 * hidden).map(|i| ((i as f32 + 3.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("k_proj".to_string(), (0..hkv * hd * hidden).map(|i| ((i as f32 + 4.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("v_proj".to_string(), (0..hkv * hd * hidden).map(|i| ((i as f32 + 5.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("o_proj".to_string(), (0..hidden * hq * hd).map(|i| ((i as f32 + 6.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("q_norm".to_string(), (0..hd).map(|i| ((i as f32 + 7.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("k_norm".to_string(), (0..hd).map(|i| ((i as f32 + 8.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        let attn = Attention::new(hq, hkv, hd, hidden, eps, QuantPlan::F16);

        let mut gpu = gpu_client().await;
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false };
        crate::interpreters::eval_load(&attn, &mut gpu, &src, &lctx)
            .await
            .expect("attention f16 装载");
        let rp = crate::layers::rope::Rope::new(t_len * 2, hd, rotary, 10_000.0).expect("rope new");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &lctx)
            .await
            .expect("rope 表物化");

        // 双臂独立池(classic 布局)
        let kc_a = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd / x, page, x]).step(), &mut gpu)
            .await.expect("kc_a");
        let vc_a = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd, page]).step(), &mut gpu)
            .await.expect("vc_a");
        let kc_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd / x, page, x]).step(), &mut gpu)
            .await.expect("kc_b");
        let vc_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd, page]).step(), &mut gpu)
            .await.expect("vc_b");

        // 输入(f16 位型量化)
        let xs_f32: Vec<f32> = (0..t_len * hidden)
            .map(|i| half::f16::from_f32(((i as f32) * 0.37 - 1.0).sin()).to_f32())
            .collect();
        let xs_bytes = halfb(&xs_f32);

        // ── A 臂:默认分派(norm_rope + K0;pos=t)──
        let kc_shape = vec![nb, hkv, hd / x, page, x];
        let pool_a = kprobe_run_arm(&attn, &rp, &xs_bytes, t_len, hidden, hkv, hd, page,
            false, false, &kc_a, &vc_a, &kc_shape, &mut gpu).await;
        // ── B 臂:融合分派(pos=t)──
        let pool_b = kprobe_run_arm(&attn, &rp, &xs_bytes, t_len, hidden, hkv, hd, page,
            false, true, &kc_b, &vc_b, &kc_shape, &mut gpu).await;
        // ── C 臂:融合分派 + pos≡0(θ=0 → 池内容 = 纯 norm 输出,与 rope 解耦)──
        let kc_c = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd / x, page, x]).step(), &mut gpu)
            .await.expect("kc_c");
        let vc_c = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd, page]).step(), &mut gpu)
            .await.expect("vc_c");
        let pool_c = kprobe_run_arm(&attn, &rp, &xs_bytes, t_len, hidden, hkv, hd, page,
            true, true, &kc_c, &vc_c, &kc_shape, &mut gpu).await;

        // ── host 金标 G ──
        let kw: Vec<f32> = src["k_proj"].iter().map(|&v| f16r(v)).collect();
        let alpha: Vec<f32> = src["k_norm"].iter().map(|&v| f16r(v)).collect();
        let xsr: Vec<f32> = xs_f32.iter().map(|&v| f16r(v)).collect();
        let mut gold = vec![0f32; t_len * hkv * hd];
        let mut gold_n = vec![0f32; t_len * hkv * hd]; // 纯 norm 期望(C 臂,θ=0)
        for t in 0..t_len {
            for kh in 0..hkv {
                let mut k = vec![0f32; hd];
                for d in 0..hd {
                    let mut acc = 0f32;
                    for c in 0..hidden {
                        acc += kw[(kh * hd + d) * hidden + c] * xsr[t * hidden + c];
                    }
                    k[d] = acc;
                }
                let sum: f32 = k.iter().map(|v| v * v).sum();
                let inv = 1.0 / (sum / hd as f32 + eps).sqrt();
                let n: Vec<f32> = (0..hd).map(|d| k[d] * inv * (alpha[d] + 1.0)).collect();
                let p = t as f32;
                for d in 0..hd {
                    let g = if d < half {
                        let ang = p * theta.powf(-(2.0 * d as f32) / rotary as f32);
                        let (cf, sf) = (f16r(ang.cos()), f16r(ang.sin()));
                        n[d] * cf - n[d + half] * sf
                    } else if d < 2 * half {
                        let dd = d - half;
                        let ang = p * theta.powf(-(2.0 * dd as f32) / rotary as f32);
                        let (cf, sf) = (f16r(ang.cos()), f16r(ang.sin()));
                        // HF rotate-half:out[i+h] = x[i+h]·cos + x[i]·sin(基值 = 自己)
                        n[d] * cf + n[dd] * sf
                    } else {
                        n[d]
                    };
                    gold[(t * hkv + kh) * hd + d] = f16r(g);
                    gold_n[(t * hkv + kh) * hd + d] = f16r(n[d]);
                }
            }
        }

        // ── 池提取(classic 寻址)+ 三方对拍(硬门)──
        let vf = |pool: &[u8], t: usize, kh: usize, d: usize| -> f32 {
            let b = t / page;
            let off = t % page;
            let i = (((b * hkv + kh) * (hd / x) + d / x) * page * x + off * x + d % x) * 2;
            half::f16::from_le_bytes([pool[i], pool[i + 1]]).to_f32()
        };
        let g_at = |t: usize, kh: usize, d: usize| gold[(t * hkv + kh) * hd + d];
        let (mut w_a, mut w_b, mut w_ab, mut w_c) = (0f32, 0f32, 0f32, 0f32);
        let (mut rot_b, mut pass_b) = (0f32, 0f32); // 结构判据(旋转/直通分账,失败时定位用)
        let mut worst = (0f32, 0usize, 0usize, 0usize); // (dev, t, kh, d)
        for t in 0..t_len {
            for kh in 0..hkv {
                for d in 0..hd {
                    let (a, b, g) = (vf(&pool_a, t, kh, d), vf(&pool_b, t, kh, d), g_at(t, kh, d));
                    let c = vf(&pool_c, t, kh, d);
                    w_c = w_c.max((c - gold_n[(t * hkv + kh) * hd + d]).abs());
                    w_a = w_a.max((a - g).abs());
                    w_b = w_b.max((b - g).abs());
                    w_ab = w_ab.max((a - b).abs());
                    if d < 2 * half {
                        rot_b = rot_b.max((b - g).abs());
                    } else {
                        pass_b = pass_b.max((b - g).abs());
                    }
                    if (b - g).abs() > worst.0 {
                        worst = ((b - g).abs(), t, kh, d);
                    }
                }
            }
        }
        let r0 = 2 * half;
        eprintln!(
            "[k-probe hd{hd} r{rotary}] |A−G|={w_a:.4} |B−G|={w_b:.4} |C−N|={w_c:.4} |A−B|={w_ab:.4} \
             (B 旋转维={rot_b:.4} 直通维={pass_b:.4} 最坏点 t={} kh={} d={})",
            worst.1, worst.2, worst.3,
        );
        assert!(w_a <= TOL && w_b <= TOL && w_c <= TOL,
            "[k-probe hd{hd} r{rotary}] 金标超差:A={w_a:.4} B={w_b:.4} C={w_c:.4}(门 {TOL})");
        assert!(w_ab <= TOL,
            "[k-probe hd{hd} r{rotary}] 融合/默认分派漂移:A−B={w_ab:.4}(门 {TOL})");
    }

    /// k-probe 单臂:逐 token decode 写池 → dtoh 终态 k 池(字节)
    #[allow(clippy::too_many_arguments)]
    async fn kprobe_run_arm<D: crate::contract::DeviceClient>(
        attn: &Attention,
        rp: &crate::layers::rope::Rope,
        xs_bytes: &[u8],
        t_len: usize,
        hidden: usize,
        hkv: usize,
        hd: usize,
        page: usize,
        pos_zero: bool,
        qkv_fuse: bool,
        kc: &crate::contract::Bytes,
        vc: &crate::contract::Bytes,
        kc_shape: &[usize],
        gpu: &mut D,
    ) -> Vec<u8> {
        for t in 0..t_len {
            let x_t = TensorOps::from_host(Dtype::F16, vec![1, hidden],
                &xs_bytes[t * hidden * 2..(t + 1) * hidden * 2]);
            let pv = if pos_zero { 0.0f32 } else { t as f32 };
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[pv]));
            let kv = KvBuffers {
                k_cache: TensorOps::of_block(kc.id, Dtype::F16, kc_shape.to_vec()),
                v_cache: TensorOps::of_block(vc.id, Dtype::F16,
                    vec![kc_shape[0], hkv, hd, page]),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32 + 1.0])),
                block_tables: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[0.0])),
            };
            let mut ctx = ForwardCtx::decode(1, &pos_t, &kv, rp);
            ctx.env.attn.qkv_fuse = qkv_fuse;
            let decl = attn.forward(&x_t, &ctx);
            let _ = crate::testkit::harvest_f16(gpu, &decl).await;
        }
        let mut pool = vec![0u8; kc_shape.iter().product::<usize>() * 2];
        gpu.dtoh(kc, &mut pool).await.expect("kc dtoh");
        pool
    }


    /// 批P4 验收:T=8 prefill 块 == T×decode 逐步(输出行 + KV 终态位型;
    /// 表/槽/kv_len 均走 [T] 表的正确姿势 —— F16 ctx 装表)
    #[tokio::test]
    async fn gpu_attn_prefill_block_matches_token_loop() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (hq, hkv, hd, hidden) = (2usize, 1usize, 4usize, 6usize);
        let t_len = 8usize;
        let row_q = hq * hd;
        let halfb = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
        };

        // 权重(确定式;f16 装载)
        let mut src = std::collections::HashMap::new();
        src.insert("q_proj".to_string(), (0..hq * hd * 2 * hidden).map(|i| ((i as f32 + 3.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("k_proj".to_string(), (0..hkv * hd * hidden).map(|i| ((i as f32 + 4.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("v_proj".to_string(), (0..hkv * hd * hidden).map(|i| ((i as f32 + 5.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("o_proj".to_string(), (0..hidden * hq * hd).map(|i| ((i as f32 + 6.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("q_norm".to_string(), (0..hd).map(|i| ((i as f32 + 7.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("k_norm".to_string(), (0..hd).map(|i| ((i as f32 + 8.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        let attn = Attention::new(hq, hkv, hd, hidden, 1e-6, QuantPlan::F16);

        let mut gpu = gpu_client().await;
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false };
        crate::interpreters::eval_load(&attn, &mut gpu, &src, &lctx)
            .await
            .expect("attention f16 装载");

        // rope(表 F16 ctx 装载 —— 正确姿势;smoke 的 Default 传法见挂账)
        let rp = crate::layers::rope::Rope::new(64, hd, 2, 10_000.0).expect("rope new");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &lctx)
            .await
            .expect("rope 表物化(f16)");

        // KV cache(f16;8 槽)
        let kc_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![8, hkv, hd]).step(), &mut gpu)
            .await.expect("kc");
        let vc_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![8, hkv, hd]).step(), &mut gpu)
            .await.expect("vc");
        let kv_zero = |slots: Vec<f32>, lens: Vec<f32>| KvBuffers {
            k_cache: TensorOps::of_block(kc_b.id, Dtype::F16, vec![8, hkv, hd]),
            v_cache: TensorOps::of_block(vc_b.id, Dtype::F16, vec![8, hkv, hd]),
            slots: TensorOps::from_host(Dtype::F32, vec![slots.len()], &f32b(&slots)),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![lens.len()], &f32b(&lens)),
            block_tables: TensorOps::zeros(Dtype::F32, vec![1]), // 哑表(legacy 路径不解引用)
        };

        // 输入(f16 位型量化)
        let xs_f32: Vec<f32> = (0..t_len * hidden)
            .map(|i| half::f16::from_f32(((i as f32) * 0.37 - 1.0).sin()).to_f32())
            .collect();
        let pos_f32: Vec<f32> = (0..t_len).map(|t| t as f32).collect();

        // 参考:T 次 decode 步(slot=t,kv_len=t+1,pos=t;KV 块跨步持久)
        let mut ref_outs = Vec::new();
        for t in 0..t_len {
            let x_t = TensorOps::from_host(Dtype::F16, vec![1, hidden], &halfb(&xs_f32[t * hidden..(t + 1) * hidden]));
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32]));
            let kv_t = kv_zero(vec![t as f32], vec![t as f32 + 1.0]);
            let decl = attn.forward(&x_t, &ForwardCtx::decode(1, &pos_t, &kv_t, &rp));
            ref_outs.push(crate::testkit::harvest_f16(&mut gpu, &decl).await);
        }

        // 被测:一次 prefill 块(slots/kv_lens/pos [T] 表)
        let xs_all = TensorOps::from_host(Dtype::F16, vec![t_len, hidden], &halfb(&xs_f32));
        let pos_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&pos_f32));
        let slots_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&(0..t_len).map(|t| t as f32).collect::<Vec<_>>()));
        let lens_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&(0..t_len).map(|t| t as f32 + 1.0).collect::<Vec<_>>()));
        let kv_p = kv_zero(vec![0.0], vec![1.0]); // 表形式,单引 slots 不被读
        let decl_all = attn.forward(
            &xs_all,
            &ForwardCtx::attn_prefill(t_len, &pos_all, &kv_p, &rp, &slots_all, &lens_all),
        );
        let out_all = crate::testkit::harvest_f16(&mut gpu, &decl_all).await;

        // 输出行对拍([T, hidden];f16 链 2e-2 rel)
        for t in 0..t_len {
            crate::testkit::assert_close(
                &out_all[t * hidden..(t + 1) * hidden],
                &ref_outs[t],
                2e-2,
                &format!("attn prefill 行{t}"),
            );
        }

        // KV 终态位型对拍(写路径同核 → 逐位一致;只核前 t_len 槽)
        let read_kc = {
            let b = crate::interpreters::eval_ops(
                TensorOps::of_block(kc_b.id, Dtype::F16, vec![8, hkv, hd]).step(), &mut gpu)
                .await.expect("kc 叶");
            let mut buf = vec![0u8; 8 * hkv * hd * 2];
            gpu.dtoh(&b, &mut buf).await.expect("dtoh");
            buf
        };
        let mut wrote = 0usize;
        for (i, ch) in read_kc.chunks_exact(2).enumerate() {
            let v = half::f16::from_le_bytes([ch[0], ch[1]]).to_f32();
            if v != 0.0 { wrote += 1; }
        }
        assert!(wrote >= t_len * hkv * hd, "KV 槽应已被 T 步写满: wrote={wrote}");
        gpu.close().await.expect("关机");
    }

    /// 批P4(f32 锚链变体):模型 fixture = f32 链,展开的 f32 核路同验
    #[tokio::test]
    async fn gpu_attn_prefill_block_matches_token_loop_f32() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (hq, hkv, hd, hidden) = (2usize, 1usize, 4usize, 6usize);
        let t_len = 8usize;
        let mut src = std::collections::HashMap::new();
        src.insert("q_proj".to_string(), (0..hq * hd * 2 * hidden).map(|i| ((i as f32 + 3.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("k_proj".to_string(), (0..hkv * hd * hidden).map(|i| ((i as f32 + 4.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("v_proj".to_string(), (0..hkv * hd * hidden).map(|i| ((i as f32 + 5.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("o_proj".to_string(), (0..hidden * hq * hd).map(|i| ((i as f32 + 6.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("q_norm".to_string(), (0..hd).map(|i| ((i as f32 + 7.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        src.insert("k_norm".to_string(), (0..hd).map(|i| ((i as f32 + 8.0) * 0.13).sin() * 0.5).collect::<Vec<f32>>());
        let attn = Attention::new(hq, hkv, hd, hidden, 1e-6, QuantPlan::F16);

        let mut gpu = gpu_client().await;
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F32, shard: 1, device_repack: false };
        crate::interpreters::eval_load(&attn, &mut gpu, &src, &lctx)
            .await
            .expect("attention f32 装载");
        let rp = crate::layers::rope::Rope::new(64, hd, 2, 10_000.0).expect("rope new");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &lctx)
            .await
            .expect("rope 表物化(f32)");

        let kc_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F32, vec![8, hkv, hd]).step(), &mut gpu)
            .await.expect("kc");
        let vc_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F32, vec![8, hkv, hd]).step(), &mut gpu)
            .await.expect("vc");
        let kv_zero = |slots: Vec<f32>, lens: Vec<f32>| KvBuffers {
            k_cache: TensorOps::of_block(kc_b.id, Dtype::F32, vec![8, hkv, hd]),
            v_cache: TensorOps::of_block(vc_b.id, Dtype::F32, vec![8, hkv, hd]),
            slots: TensorOps::from_host(Dtype::F32, vec![slots.len()], &f32b(&slots)),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![lens.len()], &f32b(&lens)),
            block_tables: TensorOps::zeros(Dtype::F32, vec![1]), // 哑表(legacy 路径不解引用)
        };

        let xs_f32: Vec<f32> = (0..t_len * hidden)
            .map(|i| ((i as f32) * 0.37 - 1.0).sin())
            .collect();

        let mut ref_outs = Vec::new();
        for t in 0..t_len {
            let x_t = TensorOps::from_host(Dtype::F32, vec![1, hidden], &f32b(&xs_f32[t * hidden..(t + 1) * hidden]));
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32]));
            let kv_t = kv_zero(vec![t as f32], vec![t as f32 + 1.0]);
            let decl = attn.forward(&x_t, &ForwardCtx::decode(1, &pos_t, &kv_t, &rp));
            ref_outs.push(crate::testkit::harvest(&mut gpu, &decl).await);
        }

        let xs_all = TensorOps::from_host(Dtype::F32, vec![t_len, hidden], &f32b(&xs_f32));
        let pos_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&(0..t_len).map(|t| t as f32).collect::<Vec<_>>()));
        let slots_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&(0..t_len).map(|t| t as f32).collect::<Vec<_>>()));
        let lens_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&(0..t_len).map(|t| t as f32 + 1.0).collect::<Vec<_>>()));
        let kv_p = kv_zero(vec![0.0], vec![1.0]);
        let decl_all = attn.forward(
            &xs_all,
            &ForwardCtx::attn_prefill(t_len, &pos_all, &kv_p, &rp, &slots_all, &lens_all),
        );
        let out_all = crate::testkit::harvest(&mut gpu, &decl_all).await;

        for t in 0..t_len {
            crate::testkit::assert_close(
                &out_all[t * hidden..(t + 1) * hidden],
                &ref_outs[t],
                1e-5,
                &format!("attn prefill-f32 行{t}"),
            );
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




#[cfg(test)]
mod split_probe_tests {
    use super::*;

    // W3 结案硬门(2026-10-02):split attention 小规模真权重对拍 + host 金标。
    // 曾两雷:① store 隐式转换截断 ② 在线 softmax 漏 acc 重缩放 —— 本门
    // 双双拦截(单 token 档护 ①,多 token 档护 ②)。
    #[tokio::test]
    async fn gpu_split_probe() {
    use crate::contract::DeviceClient as _;
    if !crate::testkit::gpu_enabled() { return; }
    let (hq, hkv, hd, t, ctx_base, nparts) = (2usize, 1usize, 256usize, 8usize, 0usize, 1usize);
    let page = 32usize; let x = 8usize; let nb = 1usize;
    let mut gpu = crate::testkit::gpu_client().await;

    // 输入
    let qr: Vec<f32> = (0..t * hq * hd).map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.13).collect();
    let kr: Vec<f32> = (0..t * hkv * hd).map(|i| ((i * 11 % 19) as f32 - 9.0) * 0.17).collect();
    let vr: Vec<f32> = (0..t * hkv * hd).map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.19).collect();
    let halfb = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect() };
    let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };

    let q = TensorOps::from_host(Dtype::F16, vec![t, hq * hd], &halfb(&qr));
    let k = TensorOps::from_host(Dtype::F16, vec![t, hkv * hd], &halfb(&kr));
    let v = TensorOps::from_host(Dtype::F16, vec![t, hkv * hd], &halfb(&vr));
    let kc = crate::interpreters::eval_ops(
        TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd / x, page, x]).step(), &mut gpu).await.unwrap();
    let vc = crate::interpreters::eval_ops(
        TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd, page]).step(), &mut gpu).await.unwrap();
    let bt = TensorOps::from_host(Dtype::F32, vec![1, nb], &f32b(&(0..nb).map(|i| i as f32).collect::<Vec<_>>()));
    let slots = TensorOps::from_host(Dtype::F32, vec![t], &f32b(&(0..t).map(|i| i as f32).collect::<Vec<_>>()));
    let scr_out_b = crate::interpreters::eval_ops(
        TensorOps::zeros(Dtype::F16, vec![t * hq * nparts * hd]).step(), &mut gpu).await.unwrap();
    let scr_stat_b = crate::interpreters::eval_ops(
        TensorOps::zeros(Dtype::F32, vec![t * hq * nparts * 2]).step(), &mut gpu).await.unwrap();
    let scr_out = TensorOps::of_block(scr_out_b.id, Dtype::F16, vec![t * hq * nparts * hd]);
    let scr_stat = TensorOps::of_block(scr_stat_b.id, Dtype::F32, vec![t * hq * nparts * 2]);

    // K0 写池
    let wr = TensorOps::call(ids::ATTN_K0_WRITE).aux(&[t])
        .arg(&k).arg(&v)
        .arg(&TensorOps::of_block(kc.id, Dtype::F16, vec![nb, hkv, hd / x, page, x]))
        .arg(&TensorOps::of_block(vc.id, Dtype::F16, vec![nb, hkv, hd, page]))
        .arg(&slots)
        .arg_i32((hkv * hd) as i32).arg_i32((hkv * hd) as i32)
        .arg_i32(hkv as i32).arg_i32(hd as i32).arg_i32(page as i32).arg_i32(x as i32)
        .with_shape(Dtype::F16, vec![1]);
    let _ = crate::interpreters::eval_ops(wr.step(), &mut gpu).await.unwrap();

    // split + reduce
    let sp = TensorOps::call(ids::ATTN_PREFILL_SPLIT)
        .arg(&q)
        .arg(&TensorOps::of_block(kc.id, Dtype::F16, vec![nb, hkv, hd / x, page, x]))
        .arg(&TensorOps::of_block(vc.id, Dtype::F16, vec![nb, hkv, hd, page]))
        .arg(&bt)
        .arg(&scr_out).arg(&scr_stat)
        .arg(&wr)
        .arg_f32(1.0 / (hd as f32).sqrt())
        .arg_i32(hkv as i32).arg_i32(t as i32).arg_i32(ctx_base as i32)
        .arg_i32(nparts as i32)
        .arg_i32((hkv * hd * page) as i32).arg_i32((hd * page) as i32).arg_i32(page as i32)
        .arg_i32(hq as i32)
        .aux(&[hd, hkv, hq, t, nparts])
        .with_shape(Dtype::F16, vec![1]);
    let y = TensorOps::call(ids::ATTN_PREFILL_SPLIT_REDUCE)
        .arg(&sp).arg(&scr_out).arg(&scr_stat)
        .arg_i32(nparts as i32).arg_i32(hq as i32).arg_i32(hd as i32).arg_i32(t as i32)
        .aux(&[t, hq])
        .with_shape(Dtype::F16, vec![t, hq * hd]);
    let got = crate::testkit::harvest_f16(&mut gpu, &y).await;
    {
        use crate::contract::DeviceClient as _;
        let mut sb = vec![0u8; 128];
        gpu.dtoh(&scr_stat_b, &mut sb).await.expect("stat dtoh");
        let sv: Vec<f32> = sb.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        eprintln!("[split-probe] scr_stat(m,l): {sv:?}");
        let mut ob = vec![0u8; 8192];
        gpu.dtoh(&scr_out_b, &mut ob).await.expect("scr_out dtoh");
        let nz: usize = ob.chunks_exact(2).filter(|c| *c != [0, 0]).count();
        eprintln!("[split-probe] scr_out 非零 halves = {nz}/4096");
        let hf16 = |b: &[u8], i: usize| half::f16::from_le_bytes([b[i * 2], b[i * 2 + 1]]).to_f32();
        // 诊断:scr_out 按 (tok,head) 槽 × g 段(64 dims)非零分布
        for tok in 0..t {
            for h in 0..hq {
                let base = (tok * hq + h) * nparts * hd;
                let seg: Vec<String> = (0..4).map(|g| {
                    let nz = (0..64).filter(|&d| ob[(base + g * 64 + d) * 2..] != [0, 0]).count();
                    format!("g{g}:{nz}")
                }).collect();
                eprintln!("[split-probe] scr_out t{tok}h{h} [{}]", seg.join(" "));
            }
        }
        let vr0: Vec<f32> = (0..256usize).map(|d| ((d * 13 % 17) as f32 - 8.0) * 0.19).collect();
        for slot in 0..2usize {
            let lo = slot * 256;
            let nz_s = ob[lo * 2..(lo + 256) * 2].chunks_exact(2).filter(|c| *c != [0, 0]).count();
                let d0: Vec<String> = (0..8).map(|d| format!("{d}:{:.5}", hf16(&ob, lo + d))).collect();
            eprintln!("[split-probe] slot{slot}[0..8]={:?} (金标 vr[0..8][0]=-1.52)", &d0);
        }
        eprintln!("[split-probe] v0 金标[0..12]: {:?}", &vr0[..12]);
        // vc 池取证:V 写入 = d×32 + kt
        let mut vcb = vec![0u8; 16384];
        gpu.dtoh(&vc, &mut vcb).await.expect("vc dtoh");
        let hf16v = |b: &[u8], i: usize| half::f16::from_le_bytes([b[i * 2], b[i * 2 + 1]]).to_f32();
        let nz_v = vcb.chunks_exact(2).filter(|c| *c != [0, 0]).count();
        eprintln!("[split-probe] vc 非零 halves = {nz_v}/8192");
        for db in 0..4usize {
            let mut nzr = 0usize; let mut totr = 0usize;
            for d in db * 64..(db + 1) * 64 {
                for kt in 0..8usize {
                    totr += 1;
                    let i = (d * 32 + kt) * 2;
                    if vcb[i..i + 2] != [0, 0] { nzr += 1; }
                }
            }
            eprintln!("[split-probe] vc d[{},{}) 非零 {nzr}/{totr}", db * 64, (db + 1) * 64);
        }
        for kt in [0usize, 3] {
            let row: Vec<String> = (0..6).map(|d| format!("d{d}={:.3}", hf16v(&vcb, d * 32 + kt))).collect();
            eprintln!("[split-probe] vc kt{kt}: {:?}(金标 v[kt×256+d])", &row);
        }
    }

    gpu.close().await.expect("close");

    // host 金标
    let hf = |b: &[u8], i: usize| half::f16::from_le_bytes([b[i * 2], b[i * 2 + 1]]).to_f32();
    let kb = halfb(&kr); let vb = halfb(&vr); let qb = halfb(&qr);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut worst = 0f32;
    for tok in 0..t {
        for h in 0..hq {
            let mut m = f32::MIN; let mut scores = vec![0f32; tok + 1];
            for kt in 0..=tok {
                let mut dot = 0f32;
                for d in 0..hd {
                    dot += hf(&qb, (tok * hq + h) * hd + d) * hf(&kb, kt * hd + d);
                }
                let s = dot * scale; scores[kt] = s; m = m.max(s);
            }
            let mut l = 0f32;
            for kt in 0..=tok { scores[kt] = (scores[kt] - m).exp(); l += scores[kt]; }
            for d in 0..hd {
                let mut acc = 0f32;
                for kt in 0..=tok { acc += scores[kt] * hf(&vb, kt * hd + d); }
                let want = half::f16::from_f32(acc / l).to_f32();
                let got_v = got[(tok * hq + h) * hd + d];
                worst = worst.max((want - got_v).abs());
            }
        }
    }
    eprintln!("[split-probe] worst |dev| = {worst:.4}");
    for (tok, h) in [(0usize, 0usize), (0, 1), (3, 0), (7, 1)] {
        for d in [0usize, 64, 128, 192] {
            let g = got[(tok * hq + h) * hd + d];
            let mut m = f32::MIN; let mut sc = vec![0f32; tok + 1];
            for kt in 0..=tok {
                let mut dot = 0f32;
                for dd in 0..hd { dot += hf(&qb, (tok*hq+h)*hd + dd) * hf(&kb, kt*hd + dd); }
                let sv = dot * scale; sc[kt] = sv; m = m.max(sv);
            }
            let mut l = 0f32;
            for kt in 0..=tok { sc[kt] = (sc[kt] - m).exp(); l += sc[kt]; }
            let mut acc = 0f32;
            for kt in 0..=tok { acc += sc[kt] * hf(&vb, kt*hd + d); }
            eprintln!("[split-probe] t{tok} h{h} d{d}: got={g:.5} want={:.5}", acc / l);
        }
    }
    assert!(worst < 5e-2, "split attention 偏差 {worst}");
}

    // FlashInfer paged prefill 对拍(E1.5 硬门;真数据 + host 金标):
    // 全 ctx 一次 K0-dual 灌池(classic + kHND 影子),q = 末 T token,
    // FI causal chunked 语义 = q[i] attend [0, ctx_total-T+i]。
    #[tokio::test]
    async fn gpu_fi_prefill_probe() {
        use crate::contract::DeviceClient as _;
        if !crate::testkit::gpu_enabled() { return; }
        // 引擎全参档(27B:24/4/256;chunk=page=32;双 chunk = 跨 chunk K0)
        let fp8kv = std::env::var_os("FI_FP8").is_some();
        let (hq, hkv, hd, page, x) = (24usize, 4usize, 256usize, 32usize, 8usize);
        let nb = 2usize;
        let ctx_total = nb * page;     // 64 = 两个 chunk
        let t = 32usize;               // 末 32 token 为 q(chunk 2)
        let ctx_base = ctx_total - t;
        let mut gpu = crate::testkit::gpu_client().await;

        // 输入(确定性伪随机;q 取末 t 行)
        let qr: Vec<f32> = (0..t * hq * hd).map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.13).collect();
        let kr: Vec<f32> = (0..ctx_total * hkv * hd).map(|i| ((i * 11 % 19) as f32 - 9.0) * 0.17).collect();
        let vr: Vec<f32> = (0..ctx_total * hkv * hd).map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.19).collect();
        let halfb = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect() };
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let i32b = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };

        let q = TensorOps::from_host(Dtype::F16, vec![t, hq * hd], &halfb(&qr));
        let k_all = TensorOps::from_host(Dtype::F16, vec![ctx_total, hkv * hd], &halfb(&kr));
        let v_all = TensorOps::from_host(Dtype::F16, vec![ctx_total, hkv * hd], &halfb(&vr));
        let kc = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd / x, page, x]).step(), &mut gpu).await.unwrap();
        let vc = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd, page]).step(), &mut gpu).await.unwrap();
        let kc_t = TensorOps::of_block(kc.id, Dtype::F16, vec![nb, hkv, hd / x, page, x]);
        let vc_t = TensorOps::of_block(vc.id, Dtype::F16, vec![nb, hkv, hd, page]);
        let (kfi, vfi, kfi_t, vfi_t) = if fp8kv {
            // e4m3 字节池(U32 承载:elems = bytes/4)
            let n_u32 = nb * page * hkv * hd / 4;
            let kf = crate::interpreters::eval_ops(
                TensorOps::zeros(Dtype::U32, vec![n_u32]).step(), &mut gpu).await.unwrap();
            let vf = crate::interpreters::eval_ops(
                TensorOps::zeros(Dtype::U32, vec![n_u32]).step(), &mut gpu).await.unwrap();
            (
                kf.clone(),
                vf.clone(),
                TensorOps::of_block(kf.id, Dtype::U32, vec![n_u32]),
                TensorOps::of_block(vf.id, Dtype::U32, vec![n_u32]),
            )
        } else {
            let kf = crate::interpreters::eval_ops(
                TensorOps::zeros(Dtype::F16, vec![nb, page, hkv, hd]).step(), &mut gpu).await.unwrap();
            let vf = crate::interpreters::eval_ops(
                TensorOps::zeros(Dtype::F16, vec![nb, page, hkv, hd]).step(), &mut gpu).await.unwrap();
            (
                kf.clone(),
                vf.clone(),
                TensorOps::of_block(kf.id, Dtype::F16, vec![nb, page, hkv, hd]),
                TensorOps::of_block(vf.id, Dtype::F16, vec![nb, page, hkv, hd]),
            )
        };

        // K0-dual:全 ctx 一次灌池(slots = 0..ctx_total;恒等页表)
        let slots = TensorOps::from_host(Dtype::F32, vec![ctx_total], &f32b(&(0..ctx_total).map(|i| i as f32).collect::<Vec<_>>()));
        let k0 = TensorOps::call(ids::ATTN_K0_DUAL).aux(&[ctx_total])
            .arg(&k_all).arg(&v_all)
            .arg(&kc_t).arg(&vc_t).arg(&kfi_t).arg(&vfi_t)
            .arg(&slots)
            .arg_i32((hkv * hd) as i32).arg_i32((hkv * hd) as i32)
            .arg_i32(hkv as i32).arg_i32(hd as i32).arg_i32(page as i32).arg_i32(x as i32)
            .with_shape(Dtype::F16, vec![1]);
        let _ = crate::interpreters::eval_ops(k0.step(), &mut gpu).await.unwrap();
        if !fp8kv {
            // 影子池取证(观测窗内:k0 eval 后立即读;kNHD [nb, page, hkv, hd];
            // K0-dual 写入与 FI 消费双把关;fp8 = e4m3 字节池,金标换算另案)
            use crate::contract::DeviceClient as _;
            let n1 = ctx_total * hkv * hd;
            let mut kb = vec![0u8; n1 * 2];
            gpu.dtoh(&kfi, &mut kb).await.expect("kfi dtoh");
            let mut vb = vec![0u8; n1 * 2];
            gpu.dtoh(&vfi, &mut vb).await.expect("vfi dtoh");
            let hf = |b: &[u8], i: usize| half::f16::from_le_bytes([b[i * 2], b[i * 2 + 1]]).to_f32();
            let mut wk = 0f32; let mut wv = 0f32; let mut nz_v = 0usize;
            for tok in 0..ctx_total {
                for hh in 0..hkv {
                    for d in 0..hd {
                        let off = tok * hkv * hd + hh * hd + d;
                        wk = wk.max((hf(&kb, off) - kr[off]).abs());
                        wv = wv.max((hf(&vb, off) - vr[off]).abs());
                        if vb[off * 2..] != [0, 0] { nz_v += 1; }
                    }
                }
            }
            eprintln!("[fi-probe] 影子池对照:kfi worst {wk:.5} / vfi worst {wv:.5} / vfi 非零 {nz_v}/{}", n1);
        }

        // FI 表四件套(i32;预物化块 + of_block 引用,杀 from_host 时序变量)
        let q_cu_b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::U32, vec![2], &i32b(&[0, t as i32])).step(), &mut gpu).await.unwrap();
        let idx_b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::U32, vec![nb], &i32b(&(0..nb as i32).collect::<Vec<_>>())).step(), &mut gpu).await.unwrap();
        let ind_b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::U32, vec![2], &i32b(&[0, nb as i32])).step(), &mut gpu).await.unwrap();
        let ll_b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::U32, vec![1],
                &i32b(&[(ctx_total - (ctx_total.div_ceil(page) - 1) * page) as i32])).step(), &mut gpu).await.unwrap();
        let q_cu = TensorOps::of_block(q_cu_b.id, Dtype::U32, vec![2]);
        let indices = TensorOps::of_block(idx_b.id, Dtype::U32, vec![nb]);
        let indptr = TensorOps::of_block(ind_b.id, Dtype::U32, vec![2]);
        let last_len = TensorOps::of_block(ll_b.id, Dtype::U32, vec![1]);

        // FI 虚拟核(生产同款槽序;wr 依赖边 = 已完成的 k0)
        let wr = TensorOps::call(if fp8kv { ids::ATTN_K0_DUAL_FP8KV } else { ids::ATTN_K0_DUAL })
            .aux(&[ctx_total])
            .arg(&k_all).arg(&v_all).arg(&kc_t).arg(&vc_t).arg(&kfi_t).arg(&vfi_t).arg(&slots)
            .arg_i32((hkv * hd) as i32).arg_i32((hkv * hd) as i32)
            .arg_i32(hkv as i32).arg_i32(hd as i32).arg_i32(page as i32).arg_i32(x as i32)
            .with_shape(Dtype::F16, vec![1]);
        let y = TensorOps::of(
            crate::kernel::Kernel::new(
                if fp8kv { "flashinfer_prefill_paged_fp8kv" } else { "flashinfer_prefill_paged_f16" },
                "",
            )
            .with_sig("T,T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz,sz"),
        )
        .arg(&q).arg(&kfi_t).arg(&vfi_t)
        .arg(&q_cu).arg(&indices).arg(&indptr).arg(&last_len)
        .arg(&wr)
        .arg_usize(t).arg_usize(ctx_total).arg_usize(t)
        .arg_usize(hq).arg_usize(hkv).arg_usize(hd).arg_usize(page)
        .arg_usize((1.0f32 / (hd as f32).sqrt()).to_bits() as usize)
        .with_shape(Dtype::F16, vec![t, hq * hd]);
        let got = crate::testkit::harvest_f16(&mut gpu, &y).await;
                gpu.close().await.expect("close");

        // host 金标:q[i](绝对 ctx_base+i)attend kv[0..=ctx_base+i]
        let hf = |b: &[u8], i: usize| half::f16::from_le_bytes([b[i * 2], b[i * 2 + 1]]).to_f32();
        let kb = halfb(&kr); let vb = halfb(&vr); let qb = halfb(&qr);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut worst = 0f32;
        for tok in 0..t {
            let abs = ctx_base + tok;
            for h in 0..hq {
                let kvh = h / (hq / hkv); // GQA:q 头 → kv 头
                let mut m = f32::MIN; let mut scores = vec![0f32; abs + 1];
                for kt in 0..=abs {
                    let mut dot = 0f32;
                    for d in 0..hd {
                        dot += hf(&qb, (tok * hq + h) * hd + d)
                            * hf(&kb, kt * hkv * hd + kvh * hd + d);
                    }
                    let s = dot * scale; scores[kt] = s; m = m.max(s);
                }
                let mut l = 0f32;
                for kt in 0..=abs { scores[kt] = (scores[kt] - m).exp(); l += scores[kt]; }
                for d in 0..hd {
                    let mut acc = 0f32;
                    for kt in 0..=abs { acc += scores[kt] * hf(&vb, kt * hkv * hd + kvh * hd + d); }
                    let want = half::f16::from_f32(acc / l).to_f32();
                    let got_v = got[(tok * hq + h) * hd + d];
                    worst = worst.max((want - got_v).abs());
                }
            }
        }
        // 分 token/head 偏差分布(定位用)
        let mut per_tok = vec![0f32; t];
        for tok in 0..t {
            let abs = ctx_base + tok;
            for h in 0..hq {
                let mut m = f32::MIN; let mut sc = vec![0f32; abs + 1];
                for kt in 0..=abs {
                    let mut dot = 0f32;
                    for d in 0..hd { dot += hf(&qb, (tok*hq+h)*hd + d) * hf(&kb, kt*hd + d); }
                    let sv = dot * scale; sc[kt] = sv; m = m.max(sv);
                }
                let mut l = 0f32;
                for kt in 0..=abs { sc[kt] = (sc[kt] - m).exp(); l += sc[kt]; }
                for d in 0..hd {
                    let mut acc = 0f32;
                    for kt in 0..=abs { acc += sc[kt] * hf(&vb, kt*hd + d); }
                    let want = half::f16::from_f32(acc / l).to_f32();
                    let got_v = got[(tok*hq+h)*hd + d];
                    per_tok[tok] = per_tok[tok].max((want - got_v).abs());
                }
            }
        }
        let pairs: Vec<String> = (0..6).map(|d| {
            let abs = ctx_base; let mut m = f32::MIN; let mut sc = vec![0f32; abs+1];
            for kt in 0..=abs { let mut dot = 0f32; for dd in 0..hd { dot += hf(&qb, dd) * hf(&kb, kt*hd + dd); } let sv = dot*scale; sc[kt]=sv; m=m.max(sv); }
            let mut l = 0f32; for kt in 0..=abs { sc[kt]=(sc[kt]-m).exp(); l+=sc[kt]; }
            let mut acc = 0f32; for kt in 0..=abs { acc += sc[kt]*hf(&vb, kt*hd+d); }
            format!("{}: got={:.5} want={:.5}", d, got[d], acc/l)
        }).collect();
        eprintln!("[fi-probe] t0h0[0..6] {:?}", pairs);
        // 对照:got 的 t0 行是否更接近「无 scale」或「2× scale」?
        for cand in [0.03125f32, 0.0625, 0.125] {
            let mut w = 0f32;
            for d in 0..hd {
                let abs = ctx_base; let mut m = f32::MIN; let mut sc = vec![0f32; abs+1];
                for kt in 0..=abs { let mut dot = 0f32; for dd in 0..hd { dot += hf(&qb, dd) * hf(&kb, kt*hd+dd); } let sv = dot*cand; sc[kt]=sv; m=m.max(sv); }
                let mut l = 0f32; for kt in 0..=abs { sc[kt]=(sc[kt]-m).exp(); l+=sc[kt]; }
                let mut acc = 0f32; for kt in 0..=abs { acc += sc[kt]*hf(&vb, kt*hd+d); }
                w = w.max((got[d] - acc/l).abs());
            }
            eprintln!("[fi-probe] scale={cand} → t0 worst {w:.4}");
        }
        eprintln!("[fi-probe] worst |dev| = {worst:.4}");
        let tol = if fp8kv { 9e-2 } else { 5e-2 }; // e4m3 KV 量化噪声(实测 ~0.05-0.08)
        assert!(worst < tol, "FlashInfer prefill 偏差 {worst}(fp8kv={fp8kv})");
        // 双断言:K0-dual 影子写入与 FI 消费各自把关(见影子池对照打印)
    }

}
