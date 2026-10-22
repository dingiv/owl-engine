//! Embedding:词表查表(Kernel 注册表:`owl_embed_f32`)。
//! tied 权重:同一份 `[vocab, D]` checkpoint 原布局数据兼 embedding 与
//! lm_head —— lm_head 走 **nt matmul**(`owl_matmul_nt_f32`,B 按
//! [n,k]=[vocab,D] 直读,抄 candle/mistral "W 保持 [out,in]" 惯例):
//! 零 host 转置、零第二份显存(2026-09-26 w_t 双槽形态作废)。
//! 容器 + LoaderOps 装载形态。

#[cfg(test)]
use crate::ops::SemanticKernel;
use crate::tensor::Dtype;
use crate::module::{Loadable, LoaderCtx, LoaderOps, QuantPlan, Weight};
use crate::module::{ForwardCtx, Module};
use crate::layers::linear::{gguf_quant_gemm, Linear};
use crate::TensorOps;
use owl_kernels::family::gguf::GgmlTy;

pub struct Embedding {
    /// [vocab, D](checkpoint 原布局;查表 + nt matmul 共用一份)
    w: Weight,
    /// untied lm_head 独立槽(tied = None,nt matmul 复用 w)。
    /// C1(2026-10-11):槽体 = Linear —— f16 直读或量化臂
    /// (W4A16Awq:装载源 F16 兕底 = 现场RTN int4;HyperQwen 实证
    /// lm_head int4 +0.6%PPL/GSM8K 不变,推翻旧“输出头不量化”裁决)
    lm_head: Option<Linear>,
    /// **int8 头**(P1.5-a;`QuantPlan::I8W` ⇒ 本槽在场、`lm_head` 为 None):
    /// `q i8 [rows, k]` 行主 + `s f32 [rows]`(装载期现算,见 `rtn_i8_row_sym`)
    head_i8: Option<HeadI8>,
    d_dim: usize,
    vocab: usize,
    /// **GGUF 量化表臂**(tied + 量化表;设计 §11.5 ②):`Some` ⇒ 槽是
    /// `U8` 块字节 `[vocab, row_bytes]`,查表走 `GgufDequantRows`、头走
    /// `GgufGemm`(同一张表两用 —— 4B 实测 `token_embd` = Q6_K 且
    /// `tie_word_embeddings=True`)。
    gguf_ty: Option<GgmlTy>,
    /// **int8 表**(P8,2026-10-22):q [vocab, d] i8 + s [vocab] f32。
    /// `Some` ⇒ 不装载 f16 表(`w` 仅占位),查表走 `owl_embed_i8_f16`。
    /// 仅 untied(头有独立 lm_head;tied 的头无法直读 int8 表)。
    w_i8q: Option<Weight>,
    w_i8s: Option<Weight>,
}

/// int8 lm_head 的两块(行数随词表切 ⇒ 与 `head_vocab` 同源)
pub struct HeadI8 {
    q: Weight,
    s: Weight,
    rows: usize,
    rule: crate::plan::rules::WeightRule,
}

impl Embedding {
    /// 准备容器(局部键 "weight":C10 前缀源下 →
    /// `{base}.embed_tokens.weight`;单槽,tied 经 nt matmul 复用)
    pub fn new(vocab: usize, d_dim: usize) -> Embedding {
        Embedding { w: Weight::new("weight", vec![vocab, d_dim]), lm_head: None, head_i8: None, d_dim, vocab, gguf_ty: None, w_i8q: None, w_i8s: None }
    }

    /// **tied + 计划定形**(2026-10-19 刀 7g):`QuantPlan::Gguf(ty)` ⇒ 槽 = 块字节
    /// (`U8`、`[vocab, 行字节]`);其余计划 ⇒ f16 表(行为逐位不变)。
    ///
    /// **响亮拒**:该型无行解量化入口(`ROWS_PORTED` 外,如 Q8_0 —— 快照无
    /// `dequantize_block_q8_0`)⇒ 构造期就报错,不做“装了才发现读不了”。
    pub fn new_by_plan(vocab: usize, d_dim: usize, plan: QuantPlan) -> Result<Embedding, crate::contract::ModelError> {
        let mut e = Embedding::new(vocab, d_dim);
        e.set_table_plan(plan)?;
        Ok(e)
    }

    /// **表槽按计划定形**(tied/untied 两臂共用;**唯一产地**):
    /// `QuantPlan::Gguf(ty)` ⇒ 槽 = 块字节(`U8`、`[vocab, 行字节]`)+ 查表走
    /// `GgufDequantRows`;其余计划 ⇒ f16 表(既有行为逐位不变)。
    ///
    /// ⚠️ **untied 也必须走这里**(2026-10-19 27B 实测):`token_embd` 在 27B 里
    /// 是 **Q4_K**,而 untied 臂原来只给 `lm_head` 传计划、表自己按 f16 建槽
    /// ⇒ 装载期源侧响亮拒(「量化键只能以 U8 交付」)。tied/untied 是**头部**
    /// 的差别,与"表怎么存"无关 —— 两件事不许混。
    fn set_table_plan(&mut self, plan: QuantPlan) -> Result<(), crate::contract::ModelError> {
        if matches!(plan, QuantPlan::I8W) {
            // int8 表只服务 untied(头有自己的 lm_head;tied 头会拿这张表
            // 做 nt matmul,而 int8 表没有 matmul 臂 —— 构造期响亮拒)。
            if self.lm_head.is_none() {
                return Err(crate::contract::ModelError::Msg(
                    "Embedding: int8 表要求 untied 模型(独立 lm_head);\
                     tied 头无法直读 int8 表".into(),
                ));
            }
            // `w` = F16 占位(不声明/不装载):`embed` 的输出 dtype 由它携带,
            // 不能留默认 F32(否则 dtype 守门拒 `owl_embed_i8_f16`)。
            self.w = Weight::new_typed("weight", vec![self.vocab, self.d_dim], crate::tensor::Dtype::F16);
            self.w_i8q = Some(Weight::new_typed("i8q", vec![self.vocab, self.d_dim], crate::tensor::Dtype::I8));
            self.w_i8s = Some(Weight::new_typed("i8s", vec![self.vocab], crate::tensor::Dtype::F32));
            return Ok(());
        }
        let QuantPlan::Gguf(ty) = plan else { return Ok(()) };
        if owl_kernels::family::gguf::rows_entry_for(ty).is_none() {
            return Err(crate::contract::ModelError::Msg(format!(
                "Embedding: 量化表型 {} 无行解量化入口(行臂集 = {:?})—— \
                 查表自带解量化是本臂的前提,响亮拒(设计 §11.5 ②)",
                ty.name(),
                owl_kernels::family::gguf::ROWS_PORTED
            )));
        }
        let (be, bb) = ty.block();
        // 判据(结构上不可失败):`GgufPlan::plan_at` 已要求索引内维 ==
        // 声明的 `d_dim`;而索引的 `nbytes` 定义是 `elems/be*bb` ⇒ 内维必被
        // 块元素数整除(否则该 GGUF 的字节账本身就不自洽,reader 已拒)
        assert!(
            be > 0 && self.d_dim % be == 0,
            "Embedding: GGUF {} 内维 {} 不是块大小 {be} 的整数倍(块布局无定义)",
            ty.name(),
            self.d_dim
        );
        self.w = Weight::new_typed("weight", vec![self.vocab, self.d_dim / be * bb], Dtype::U8);
        self.gguf_ty = Some(ty);
        Ok(())
    }

    /// untied(lm_head 独立权重;f16 直读;测试 fixture/mtp/dflash2 同构)
    pub fn new_untied(vocab: usize, d_dim: usize) -> Embedding {
        Embedding::new_untied_quant(vocab, d_dim, QuantPlan::F16)
    }

    /// untied + 头部量化计划(C1 生产入口;plan 量化且 eligible 时
    /// lm_head 走 marlin 臂,否则 f16 直读)
    pub fn new_untied_quant(vocab: usize, d_dim: usize, plan: QuantPlan) -> Embedding {
        Embedding::new_untied_head(vocab, vocab, d_dim, plan) // 全量头(= 词表切关)
    }

    /// **untied + 词表切**(P1.1):embed 表**全量**,lm_head 只建 `head_vocab` 行。
    pub fn new_untied_head(vocab: usize, head_vocab: usize, d_dim: usize, plan: QuantPlan) -> Embedding {
        // int8 头(§72/§73):不进 `Linear`(见 QuantPlan::I8W 的裁决注记)
        if matches!(plan, QuantPlan::I8W) {
            let rule = if head_vocab < vocab {
                crate::plan::rules::WeightRule::UniformCol
            } else {
                crate::plan::rules::WeightRule::Replicate
            };
            let head_i8 = Some(HeadI8 {
                q: Weight::new_typed("i8q", vec![head_vocab, d_dim], crate::tensor::Dtype::I8),
                s: Weight::new_typed("i8s", vec![head_vocab], crate::tensor::Dtype::F32),
                rows: head_vocab,
                rule,
            });
            return Embedding {
                w: Weight::new("weight", vec![vocab, d_dim]),
                lm_head: None,
                head_i8,
                d_dim,
                vocab,
                gguf_ty: None,
                w_i8q: None,
                w_i8s: None,
            };
        }
        let head = Linear::new("lm_head", head_vocab, d_dim, plan);
        let head = if head_vocab < vocab {
            head.with_tp_rule(crate::plan::rules::WeightRule::UniformCol)
        } else {
            head
        };
        Embedding { w: Weight::new("weight", vec![vocab, d_dim]), lm_head: Some(head), head_i8: None, d_dim, vocab, gguf_ty: None, w_i8q: None, w_i8s: None }
    }

    /// **untied + 词表切 + 表/头两个独立计划**(并集:GGUF 27B 的真实形态 ——
    /// `token_embd` = Q4_K、`output.weight` = Q6_K,且词表切档 `head_vocab < vocab`)
    pub fn new_untied_head_two_plans(
        vocab: usize,
        head_vocab: usize,
        d_dim: usize,
        table_plan: QuantPlan,
        head_plan: QuantPlan,
    ) -> Result<Embedding, crate::contract::ModelError> {
        let mut e = Embedding::new_untied_head(vocab, head_vocab, d_dim, head_plan);
        e.set_table_plan(table_plan)?;
        Ok(e)
    }

    /// 头部 TP 规则(**只读**;判据用;tied = None ⇒ 取 embed 表规则 = Replicate)
    pub fn head_rule(&self) -> crate::plan::rules::WeightRule {
        if let Some(h) = &self.head_i8 {
            return h.rule;
        }
        match &self.lm_head {
            Some(lin) => lin.tp_rule(),
            None => crate::plan::rules::WeightRule::Replicate,
        }
    }

    /// 头部输出行数(= 本 rank 的 logits 宽度;tied = 词表全量)
    pub fn head_vocab(&self) -> usize {
        if let Some(h) = &self.head_i8 {
            return h.rows;
        }
        match &self.lm_head {
            Some(lin) => lin.out_dim(),
            None => self.w.decl().shape[0],
        }
    }


    /// untied lm_head 槽装载声明(键 = **局部键** `lm_head[*]`;全键由调用方
    /// 经 [`crate::module::KeyConvention::head_key`] 改写 —— **第 4 类键**:
    /// HF 叫 `lm_head.weight`,GGUF 叫 `output.weight`)。
    pub fn layout_lm_head(&self, ctx: &LoaderCtx) -> Option<LoaderOps> {
        if let Some(h) = &self.head_i8 {
            // int8 头:只声明两个**派生键**(源 = 检查点 bf16 `lm_head.weight`,
            // 由 `formats::awq_build` 的 F16 臂现算)⇒ 不驻留 bf16 块。
            // 键 = **局部键**(main 的新约定:全键由调用方 `KeyConvention::head_key`
            // 改写;`lm_head.i8q/.i8s` 不是 `lm_head` 本身 ⇒ 原样直通,
            // 正是 AWQ 源认的派生键)。
            //
            // **层声明规则覆盖**(与 `Linear::layout` 同律):派生键的键表默认
            // 是 `Replicate`(lm_head 不切),词表切档必须显式覆盖成 `UniformCol`
            // —— 否则两 rank 读同一段(高半 = 低半的复制,实测撞过)。
            let ops =
                h.q.layout_as("lm_head.i8q", ctx).chain(h.s.layout_as("lm_head.i8s", ctx));
            return Some(if h.rule != crate::plan::rules::rule_of("lm_head") {
                ops.with_rule(h.rule)
            } else {
                ops
            });
        }
        self.lm_head.as_ref().map(|lin| lin.layout(ctx))
    }

    /// 装载完备性
    pub fn is_loaded(&self) -> bool {
        match (&self.w_i8q, &self.w_i8s) {
            (Some(q), Some(s)) => q.is_loaded() && s.is_loaded(),
            _ => self.w.is_loaded(),
        }
    }

    /// 查表(Module 统一入口的实体;tokens 由 ctx 提供 —— 每步动态依赖)。
    /// grid = tokens(核内 blockIdx.x = token 行;发射配置声明期显式)。
    /// 槽序契约:kernel 签名 (w, ids, d_dim, out) → T 槽序 w、ids,输出块末尾
    /// (w = 物化块引用,eval 时 Block 叶子零操作)。
    pub fn embed(&self, ids: &TensorOps, tokens: usize) -> TensorOps {
        // **GGUF 量化表臂**:查表自带解量化(`token_embd` 在 GGUF 里是量化块)
        if let Some(ty) = self.gguf_ty {
            return TensorOps::call(crate::ops::SemanticKernel::GgufDequantRows)
                .arg(&self.w.decl()) // [vocab, 行字节] U8
                .arg(ids) // 词表 id(f32 承载设备侧索引)
                .arg_i32(self.d_dim as i32)
                .arg_i32(tokens as i32)
                .arg_i32(self.vocab as i32)
                .aux(&[ty.code() as usize, tokens, self.d_dim])
                .with_shape(Dtype::F16, vec![tokens, self.d_dim]);
        }
        // **int8 表**(P8):查表 + per-row 反量化(输出恒 f16;与 f16 表同形)
        if let (Some(q), Some(s)) = (&self.w_i8q, &self.w_i8s) {
            let dt = self.w.dtype();
            return TensorOps::call(crate::ops::SemanticKernel::EmbedI8)
                .aux(&[tokens])
                .arg(&q.decl())
                .arg(&s.decl())
                .arg(ids)
                .arg_usize(self.d_dim)
                .with_shape(dt, vec![tokens, self.d_dim]);
        }
        // 输出 dtype 跟随表 Weight(F5;LoaderCtx 自动跟随)
        let dt = self.w.dtype();
        TensorOps::call(crate::ops::SemanticKernel::Embed).aux(&[tokens])
        .arg(&self.w.decl())
        .arg(ids)
        .arg_usize(self.d_dim)
        .with_shape(dt, vec![tokens, self.d_dim])
    }

    /// lm_head:hidden [.., D] → logits [.., vocab]
    /// nt 直读原始 [vocab, D] 布局(matmul_nt;零转置;untied 用独立槽)。
    /// C1:untied 量化臂经 Linear::forward_inner(marlin GEMV/GEMM;
    /// f16 臂 = 原语义 matmul_nt,零行为变更)
    pub fn lm_head_matmul(&self, hidden: &TensorOps, arms: &crate::env::GgufEnv) -> TensorOps {
        // ⚠️ **分支键 = `lm_head.is_none()`(tied 事实),不是 `gguf_ty`**
        // (2026-10-19 27B 实测定谳):`gguf_ty` 说的是"**表**是不是量化块",
        // 而 tied/untied 说的是"**头**用不用这张表" —— 两件事正交。
        // 早先拿 `gguf_ty` 当判据 ⇒ **untied 的 GGUF 模型把头算成了
        // `hidden @ token_embdᵀ`**(静默错值:首 token 直接垃圾,而权重面全对,
        // 极难归因;4B tied 档照不到这条雷)。
        // int8 头:GEMV(aux = [rows, cols, tokens];输出 [tokens, rows],dtype 随激活)
        if let Some(h) = &self.head_i8 {
            let sh = hidden.shape();
            let tokens: usize = sh[..sh.len() - 1].iter().product();
            // **激活 dtype 定实例**(2026-10-20:单一 op_id 会撞 dtype 守门 ——
            // 执行 dtype = f16 的档(W4A16Awq/F16 27B)拿到 bf16 核 = 输出域不符)
            let kern = match hidden.dtype {
                crate::tensor::Dtype::BF16 => crate::ops::SemanticKernel::GemvI8wBf16,
                crate::tensor::Dtype::F16 => crate::ops::SemanticKernel::GemvI8wF16,
                other => panic!("int8 头:GEMV 输出 dtype 只可能是 f16/bf16,实得 {other:?}"),
            };
            return TensorOps::call(kern)
                .arg(&h.q.decl())
                .arg(&h.s.decl())
                .arg(hidden)
                .arg_i32(h.rows as i32)
                .arg_i32(self.d_dim as i32)
                .aux(&[h.rows, self.d_dim, tokens])
                .with_shape(hidden.dtype, vec![tokens, h.rows]);
        }
        match &self.lm_head {
            // untied:独立头(marlin / GGUF 量化 / f16 三臂都在 `Linear` 里)
            Some(lin) => lin.forward_inner(hidden, arms),
            // tied:**表兼头** —— 表是量化块(GGUF 档)时走同一套两发路径
            // (设计 §11.5 ②;与 `Linear` 的 GGUF 臂**同一实现**)
            None => match self.gguf_ty {
                Some(ty) => gguf_quant_gemm(&self.w, ty, hidden, self.vocab, self.d_dim, None, arms),
                None => hidden.matmul_nt(&self.w.decl()),
            },
        }
    }
}

impl Loadable for Embedding {
    /// tied 单槽:一个 Want 兼查表与 lm_head(nt matmul)
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        // int8 表:局部键 `i8q`/`i8s` 经 `KeyConvention::embed_key` 变成
        // `...embed_tokens.i8q/.i8s`;AWQ 派生键解析器按 base=...embed_tokens
        // 读 bf16/f16 源现算(见 awq_build 的 i8 分支)。
        match (&self.w_i8q, &self.w_i8s) {
            (Some(q), Some(s)) => q.layout(ctx).chain(s.layout(ctx)),
            _ => self.w.layout(ctx),
        }
    }

}

impl Module for Embedding {
    /// 查表声明(tokens 由 ctx 提供 —— 每步动态依赖)
    fn forward(&self, ids: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        self.embed(ids, ctx.tokens)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, f32b, harvest, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path, Src};
    
    #[tokio::test]
    async fn declaration_is_wellformed() {
        let mut face = owl_cpu::CpuFace::new();
        let emb = Embedding::new(16, 4);
        let src = Src::from([
            ("weight".to_string(), (0..64).map(|i| i as f32 * 0.1).collect()),
        ]);
        crate::interpreters::eval_load(&emb, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");
        assert!(emb.is_loaded(), "tied 槽应有块");

        let ids = TensorOps::from_host(Dtype::F32, vec![2], &f32b(&[3.0, 7.0]));
        let out = emb.embed(&ids, 2);
        assert!(!out.is_poisoned(), "embedding 声明不应有毒");
        assert_eq!(out.shape(), &[2, 4]);

        let hidden = TensorOps::from_host(Dtype::F32, vec![1, 4], &f32b(&[0.1; 4]));
        let logits = harvest(&mut face, &emb.lm_head_matmul(&hidden, &crate::env::GgufEnv::default())).await;
        assert_eq!(logits.len(), 16);
    }

    /// HF parity(查表 + tied lm_head;门控 OWL_HF_PARITY=1)
    #[tokio::test]
    async fn parity_matches_hf() {
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (tokens, vocab, d) = (3usize, 16usize, 4usize);
        let ids: Vec<f32> = vec![3.0, 15.0, 0.0];
        let w: Vec<f32> = (0..vocab * d).map(|i| (i as f32 * 0.1) - 0.8).collect();
        let x: Vec<f32> = vec![0.2, -0.5, 1.1, 0.0];

        let inp = tmp_path("embed_in");
        let outp = tmp_path("embed_out");
        st_write(&inp, &[
            ("ids", &ids, vec![tokens]),
            ("w", &w, vec![vocab, d]),
            ("x", &x, vec![1, d]),
        ]);
        let manifest = hf_python("embedding.py", &[
            inp.to_str().unwrap(), outp.to_str().unwrap(),
            &format!(r#"{{"vocab": {vocab}, "d": {d}}}"#),
        ]);
        eprintln!("[hf manifest] embedding: {manifest}");
        let want_y = st_read(&outp, "y");
        let want_logits = st_read(&outp, "logits");

        for (face_tag, on_gpu) in [("cpu", false), ("gpu", true)] {
            if on_gpu && !crate::testkit::gpu_enabled() {
                continue;
            }
            let (_, y, logits) = if !on_gpu {
                let mut face = owl_cpu::CpuFace::new();
                let emb = Embedding::new(vocab, d);
                let src = Src::from([
                    ("weight".to_string(), w.clone()),
                ]);
                crate::interpreters::eval_load(&emb, &mut face, &src, &Default::default())
                    .await.expect("eval_load");
                // embed = Kernel 节点(CPU face 不执行);lm_head = 语义 matmul 可跑
                let x_t = TensorOps::from_host(Dtype::F32, vec![1, d], &f32b(&x));
                let y = want_y.clone(); // CPU 臂无 embed 执行,占位由 GPU 臂覆盖
                let logits = harvest(&mut face, &emb.lm_head_matmul(&x_t, &crate::env::GgufEnv::default())).await;
                ("cpu", y, logits)
            } else {
                let mut gpu = crate::testkit::gpu_client().await;
                let emb = Embedding::new(vocab, d);
                let src = Src::from([
                    ("weight".to_string(), w.clone()),
                ]);
                crate::interpreters::eval_load(&emb, &mut gpu, &src, &Default::default())
                    .await.expect("eval_load");
                let ids_t = TensorOps::from_host(Dtype::F32, vec![tokens], &f32b(&ids));
                let x_t = TensorOps::from_host(Dtype::F32, vec![1, d], &f32b(&x));
                let y = harvest(&mut gpu, &emb.embed(&ids_t, tokens)).await;
                let logits = harvest(&mut gpu, &emb.lm_head_matmul(&x_t, &crate::env::GgufEnv::default())).await;
                gpu.close().await.expect("server 关机");
                ("gpu", y, logits)
            };
            assert_close(&y, &want_y, 1e-5, &format!("embed-{face_tag}"));
            assert_close(&logits, &want_logits, 1e-5, &format!("lmhead-{face_tag}"));
        }
        owl_shared::file_loader::remove_file(&inp).ok();
        owl_shared::file_loader::remove_file(&outp).ok();
    }
}

#[cfg(test)]
mod f16_tests {
    use super::*;
    use crate::testkit::{gpu_client, gpu_enabled};
    use crate::contract::DeviceClient as _;
    
    fn half_le(f: f32) -> [u8; 2] {
        half::f16::from_f32(f).to_le_bytes()
    }

    /// GPU:f16 embed 查表 vs host f32(F3 收口;OWL_TEST_DEVICE 门控)
    #[tokio::test]
    async fn gpu_embed_f16_matches_host() {
        if !gpu_enabled() {
            owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
            return;
        }
        let (vocab, d) = (8usize, 16usize);
        // 源值过 f16 量化(上载即 f16;查表正确性 = 对量化后参考逐位)
        let w: Vec<f32> = (0..vocab * d)
            .map(|i| half::f16::from_f32(((i as f32 * 0.37) - 3.0).sin()).to_f32())
            .collect();
        let ids = [2.0f32, 5.0f32, 7.0f32];
        let tokens = 3usize;

        let mut gpu = gpu_client().await;
        let dw = owl_iface::htod!(gpu, Dtype::F16, &crate::contract::Shape::from(vec![vocab, d]),
            &w.iter().flat_map(|f| half_le(*f)).collect::<Vec<u8>>()).await.expect("htod w");
        let di = owl_iface::htod!(gpu, Dtype::F32, &crate::contract::Shape::from(vec![tokens]),
            &crate::testkit::f32b(&ids)).await.expect("htod ids");

        // htod 块 → Block 叶子声明(f16 链不走 CpuFace 的 from_host)
        let w_decl = TensorOps::of_block(dw.id, Dtype::F16, vec![vocab, d]);
        let i_decl = TensorOps::of_block(di.id, Dtype::F32, vec![tokens]);
        let decl = TensorOps::call(SemanticKernel::Embed).aux(&[tokens])
        .arg(&w_decl)
        .arg(&i_decl)
        .arg_usize(d)
        .with_shape(Dtype::F16, vec![tokens, d]);
        let out = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.expect("eval");
        let mut buf = vec![0u8; tokens * d * 2];
        gpu.dtoh(&out, &mut buf).await.expect("dtoh");

        for t in 0..tokens {
            let id = ids[t] as usize;
            for dd in 0..d {
                let got = half::f16::from_le_bytes([buf[(t * d + dd) * 2], buf[(t * d + dd) * 2 + 1]]).to_f32();
                assert_eq!(got, w[id * d + dd], "查表须逐位一致 [{t},{dd}]");
            }
        }
        gpu.close().await.expect("关机");
    }
}
