//! 通用解码器主干(批8,2026-09-26 拆分):embed + N×DecoderLayer
//! + final norm + tied lm_head —— **所有模型共有**的机制,凡本主干
//!   词汇表内(Full|Gdn 双 mixer)的 hybrid / dense 解码器共用。
//!   旧世界 `Qwen3_5ForCausalLM`(engine/src/models/qwen3_5.rs)的
//!   声明式重铸 —— 砍掉的是命令式缓存管理/MTP/DFlash 支线,留下的是
//!   主干数据流。
//!
//! **拆分律(2026-09-26 用户裁决)**:
//! - 本文件只住**共有机制**(主干结构/单树求值/序列 ctx 派生/C10
//!   分组件装载/fixture 测试),零模型特定参数;
//! - **模型特有**住 `specs/<model>.rs`(维度预设/层型表约定/检查点
//!   特例,纯参数事实零机制)—— 现存 [`crate::formats::qwen35`];
//! - 新模型 = 新 spec 文件;mixer 词汇超出 Full|Gdn(MoE/MLA/…)时
//!   另立 DecoderLayer 扩展口(需求基线:arch 分发表不写死),不动主干。
//!
//! ```text
//! ids ─→ embed ─→ layers[0] ─→ … ─→ layers[N-1] ─→ norm ─→ lm_head(tied) ─→ logits
//! ```
//!
//! ```text
//! ids ─→ embed ─→ layers[0] ─→ … ─→ layers[N-1] ─→ norm ─→ lm_head(tied) ─→ logits
//! ```
//!
//! 三条既有律在此收口:
//! - **C5 编排**:整模单树 —— 一次 `forward` = 一棵树 = 一次 eval;
//!   捕获/回放同一棵树(M-f 前提);
//! - **C10 装载**:Want 键恒层内短键,checkpoint 全局键由
//!   [`module::KeyConvention`] 改写(默认 [`module::LlamaFamily`];
//!   模型特有约定住 specs/<model>.rs);tied embedding 单源键双 Want;
//! - **CSE/DAG**:层内残差共享子树只求值一次;跨步状态(KV/GDN)
//!   靠常驻块 + 每步重建声明推进。

use crate::contract::{Dtype, ModelError};
use crate::layers::decoder::DecoderLayer;
use crate::layers::embedding::Embedding;
use crate::layers::rmsnorm::RmsNorm;
use crate::module::{ForwardCtx, KeyConvention, Loadable, LoaderCtx, LoaderOps, Module, PlanSource};
use crate::TensorOps;

// ============================================================================
// §1 ModelSpec:结构参数(声明期常量;纯数据零逻辑 —— 预设构造器
// 归 specs/<model>.rs,此处不认识任何具体模型)
// ============================================================================

/// 模型结构参数(旧世界 Config 的主干切片;全部声明期常量)
///
/// **TP 切分的形状单源**(S4a):[`ModelSpec::sharded`] 是本结构唯一的
/// 切分入口 —— 层构造照常从本结构取维度 ⇒ **层代码零感知切分**。
#[derive(Clone)]
pub struct ModelSpec {
    pub vocab: usize,
    /// **lm_head 的输出行数**(P1.1 词表切;缺省 = `vocab`)。
    /// 由 [`ModelSpec::sharded`] 从 `head_split` 派生 —— 唯一的"本 rank
    /// logits 宽度"真源(引擎侧 logits 槽形状/采样拼接都读它)。
    pub head_vocab: usize,
    /// **词表切开关**(P1.1;意图位,来自引擎配置 `runtime.tp_vocab_split`)。
    /// `sharded()` 只读它一次:`true` ⇒ `head_vocab = vocab / world`(必须整除,
    /// 否则响亮拒)。缺省 `false` ⇒ 逐字节等价于旧行为。
    pub head_split: bool,
    pub hidden: usize,
    pub inter: usize,
    /// 模型计算 dtype(f16 基线 F5;装载/核名/KV 池的单一事实源)
    pub dtype: crate::contract::Dtype,
    /// full attention 头 (hq, hkv, hd)
    pub full_heads: (usize, usize, usize),
    /// GDN 头 (nk, hk_dim, nv, hv_dim)
    pub gdn_heads: (usize, usize, usize, usize),
    pub eps: f32,
    /// 层混型表(true = full_attention;dense 全 true;hybrid 按检查点
    /// layer_types 填)
    pub layer_types: Vec<bool>,
    /// 分词器声明(家族特有事实:eos 族 + chat 文本格式;机制在
    /// [`crate::tokenizer`] —— 模型用什么分词器是声明,不是函数)
    pub tokenizer: crate::tokenizer::TokenizerSpec,
    /// tie_word_embeddings(true = lm_head 复用 embed;tied 仓特例;
    /// Qwen3.8-27B 等独立 lm_head 仓 = false)
    pub tied: bool,
}

impl ModelSpec {
    /// **形状级切分**(S4a):全量 spec → 本 rank 的分片 spec(纯函数)。
    ///
    /// 逐字段语义(设计 §15.2):
    /// - `hidden` **不切**(层间通道保持全宽;切它要在每层入口 allgather,不划算);
    /// - `inter` 切(MLP gate/up 是 ColParallel(out),down 是 RowParallel(in));
    /// - `full_heads` 切 `(hq, hkv)`、`hd` 不切(GQA 比例保持 ⇒ 两者都要整除);
    /// - `gdn_heads` 切 `(nk, nv)`、`(hk, hv)` 不切;
    /// - `vocab` 不切(v1 复制 lm_head;AG 词表切分是将来项);
    /// - `dtype`/`eps`/`layer_types`/`tokenizer`/`tied` 原样。
    ///
    /// **整除硬门**:任一切分维不被 world 整除 ⇒ 结构化拒(不许 pad)。
    pub fn sharded(&self, plan: crate::plan::tp::TpPlan) -> Result<ModelSpec, String> {
        // **tied 表切不了**（P1.1）：lm_head 复用 embed 表（查任意 token 必须全量）
        // ⇒ “每 rank 只持半表头”无据可依。响亮拒（静默忽略 = 配置说谎）。
        if self.head_split && self.tied {
            return Err(
                "词表切(TP vocab split)需要 untied 检查点：本模型 tie_word_embeddings = true，\
                 lm_head 复用 embed 全量表（tied 档请关 runtime.tp_vocab_split）"
                    .into(),
            );
        }
        if plan.is_single() {
            let mut s = self.clone();
            s.head_vocab = self.vocab; // 单卡退化:头也全量(与 head_split 无关)
            return Ok(s);
        }
        let (hq, hkv, hd) = self.full_heads;
        let (nk, hk, nv, hv) = self.gdn_heads;
        Ok(ModelSpec {
            vocab: self.vocab,
            // **词表切(P1.1)**:embed 表保持全量(查表要任意 token);
            // lm_head 行数按 world 等分(`head_split = false` ⇒ 全量)。
            head_vocab: if self.head_split { plan.shard_len(self.vocab)? } else { self.vocab },
            head_split: self.head_split,
            hidden: self.hidden,
            inter: plan.shard_len(self.inter)?,
            dtype: self.dtype,
            full_heads: (plan.shard_len(hq)?, plan.shard_len(hkv)?, hd),
            gdn_heads: (plan.shard_len(nk)?, hk, plan.shard_len(nv)?, hv),
            eps: self.eps,
            layer_types: self.layer_types.clone(),
            tokenizer: self.tokenizer.clone(),
            tied: self.tied,
        })
    }
}

// ============================================================================
// §2 Model:主干容器(纯声明)
// ============================================================================

pub struct Model {
    /// tied:[vocab, hidden] 直读 + [hidden, vocab] 转置槽(单源键)
    pub embed: Embedding,
    pub layers: Vec<DecoderLayer>,
    /// final norm(checkpoint 键 `{base}.norm.weight`)
    pub norm: RmsNorm,
    /// 检查点键名约定(layout 期改写局部键;模型特有约定住 specs)
    keys: Box<dyn KeyConvention + Send + Sync>,
    hidden: usize,
    vocab: usize,
}

impl Model {
    /// 准备容器(纯元数据;零数据零副作用)。keys = 检查点键名约定
    /// (默认 [`LlamaFamily`];Qwen3.5 等特有约定由 specs 传入)。
    /// `plan` = 量化计划构造期注入(禁 enable_* 可变后置范式);
    /// `head_plan` = untied lm_head 量化计划(C1,2026-10-11:HyperQwen
    /// 实证 int4 lm_head +0.6%PPL/GSM8K 不变,推翻旧“输出头不量化”
    /// 裁决;embed 表恒 f16 —— 查表增益微小,int8 化另立项)。tied
    /// 模型 head_plan 无效(lm_head 复用 w)。
    ///
    /// **2026-10-19 延展(GGUF 轮)**:两个 plan 参数型改为
    /// `impl Into<PlanSource>` —— `QuantPlan` 字面量经 `From` 直落
    /// `PlanKind::Uniform` 臂(既有调用点零改动),逐张量混型源(GGUF)
    /// 改传 [`PlanSource::keyed`](逐键查表 + 源事实)。
    ///
    /// **返回值 = `Result`**(新增):Keyed 臂下模型可能问出表外的键
    /// (缺键 / K-quant 形态不合法)⇒ 此处**响亮拒**——禁静默落 F16
    /// (那会让显存账静默失真;设计 I-2)。
    pub fn new<'a>(
        spec: &ModelSpec,
        keys: impl KeyConvention + Send + Sync + 'static,
        plan: impl Into<PlanSource<'a>>,
        head_plan: impl Into<PlanSource<'a>>,
    ) -> Result<Model, ModelError> {
        let plan: PlanSource<'a> = plan.into();
        let head: PlanSource<'a> = head_plan.into();
        let mut layers = Vec::with_capacity(spec.layer_types.len());
        for (i, &full) in spec.layer_types.iter().enumerate() {
            let src = plan.at_layer(i);
            layers.push(if full {
                let (hq, hkv, hd) = spec.full_heads;
                DecoderLayer::new_full(hq, hkv, hd, spec.hidden, spec.inter, spec.eps, src)
            } else {
                let (nk, hk_dim, nv, hv_dim) = spec.gdn_heads;
                DecoderLayer::new_gdn(nk, hk_dim, nv, hv_dim, spec.hidden, spec.inter, spec.eps, src)
            });
        }
        // 计划表覆盖门(响亮;Keyed 臂)。为什么在**构造期**判:计划在此定形,
        // 槽形状/显存账随之确定 ⇒ 缺键必须当场拒,而不能等到装载期
        // (那时已按 f16 建槽,账已失真)。
        let mut unresolved = plan.take_unresolved();
        unresolved.extend(head.take_unresolved());
        unresolved.sort();
        unresolved.dedup();
        if !unresolved.is_empty() {
            return Err(ModelError::Msg(format!(
                "Model::new: 计划表未覆盖 {} 个键(缺键或 K-quant 形态不合法 ⇒ 拒建):{}",
                unresolved.len(),
                unresolved.join(", ")
            )));
        }
        // **tied 臂**:表真的要按计划定形 —— GGUF 的 `token_embd` 是量化块
        // (4B = Q6_K)⇒ 槽= 块字节 + 查表自带解量化;HF 一切档位= f16 表。
        // 键 = `KeyConvention::embed_key`(“词表键”的唯一出口)。
        let embed = if spec.tied {
            Embedding::new_by_plan(
                spec.vocab,
                spec.hidden,
                plan.at(&keys.embed_key("weight"), spec.vocab, spec.hidden),
            )?
        } else {
            // P1.1:head_vocab < vocab ⇒ 词表切;GGUF 档表/头**两个独立计划**
            Embedding::new_untied_head_two_plans(
                spec.vocab,
                spec.head_vocab,
                spec.hidden,
                plan.at(&keys.embed_key("weight"), spec.vocab, spec.hidden),
                head.head(spec.head_vocab, spec.hidden),
            )?
        };
        Ok(Model {
            embed,
            layers,
            // 最终 norm 的语义随**源事实**:GGUF 已烘焙 +1 ⇒ ×w(设计 §11.3)
            norm: RmsNorm::new_by_facts("norm", spec.hidden, spec.eps, plan.facts()),
            keys: Box::new(keys),
            hidden: spec.hidden,
            vocab: spec.vocab,
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    /// **本 rank 的 logits 宽度**(P1.1):词表切档 = `vocab / world`;
    /// 未切档 = `vocab`(与 `vocab_size()` 同值)。
    /// ⚠️ 引擎侧的 logits 槽形状、采样拼接、门判据都以**本函数**为真源;
    /// `vocab_size()` 仍是"词表全量"(embed 表/tokenizer/draft 面用它)。
    pub fn logits_vocab(&self) -> usize {
        self.embed.head_vocab()
    }

    /// VLM 图像 token 注入(embed 之后 / 层循环之前 —— 与 HF
    /// `masked_scatter` 同位)。`ctx.vlm_image = None` ⇒ 原地返回
    /// (零节点零开销;纯文本路径不受影响)。
    fn vlm_inject(&self, xs: TensorOps, ctx: &ForwardCtx) -> TensorOps {
        match ctx.vlm_image {
            Some(inj) => xs.image_scatter(inj.merged, inj.idx).tag("vlm_image"),
            None => xs,
        }
    }


    /// per-layer 子 ctx(整模 ctx → 单层 ctx;kvs/gdns 按混型游标取,
    /// Attention/Gdn 层读单引用零感知)
    pub(crate) fn layer_ctx<'a>(
        &self,
        ctx: &'a ForwardCtx<'a>,
        kvi: usize,
        gi: usize,
    ) -> ForwardCtx<'a> {
        ForwardCtx {

            tokens: ctx.tokens,
            kind: ctx.kind,
            pos: ctx.pos,
            kv: ctx.kvs.and_then(|s| s.get(kvi)),
            rope: ctx.rope,
            gdn: ctx.gdns.and_then(|s| s.get(gi)),
            kvs: None,
            gdns: None,
            kv_slots: ctx.kv_slots,
            kv_lens: ctx.kv_lens,
            gdn_slot: ctx.gdn_slot,
            ctx_base: ctx.ctx_base,
            fi_plan_ctx: ctx.fi_plan_ctx,
            fi: ctx.fi,
            fi_kvi: kvi,
            seq_cu: ctx.seq_cu,
            gdn_tap: ctx.gdn_tap.clone(),
            env: ctx.env,
            trace_gate: ctx.trace_gate,
            gdn_slot_host: ctx.gdn_slot_host,
            gdn_tables: ctx.gdn_tables,
            ts_buf: ctx.ts_buf.clone(),
            attn_v2: ctx.attn_v2.clone(),
            v2_bt8: ctx.v2_bt8,
            xqa: ctx.xqa.clone(),
            vlm_image: ctx.vlm_image,
            vlm_rope: ctx.vlm_rope,
            q_sink: ctx.q_sink,
        }
    }

    /// E5-DF4 崩坏排查:逐层 bisect + 指定层五站 stage tap(非 dflash
    /// 诊断路;与 forward_stages 同式非原地展开,自洽即可)。调用方
    /// multi eval 后自行收割 checksum;gdn 层的 ctx.gdn_tap 若设,
    /// stage 层的 8 件(gdn 前向内部节点)一并带出。
    /// 返回 (fin, [(名, 节点)]):名 = embed / l{i} / s{L}.{n1|mixed|h|n2|mlp|out|g{j}.f} / fin。
    pub fn prefill_bisect(
        &self,
        ids: &TensorOps,
        ctx: &ForwardCtx,
        stage_layer: Option<usize>,
    ) -> (TensorOps, Vec<(String, TensorOps)>) {
        let mut nodes: Vec<(String, TensorOps)> = Vec::new();
        let mut xs = self.embed.forward(ids, ctx).tag("embed");
        xs = self.vlm_inject(xs, ctx);
        nodes.push(("embed".into(), xs.clone()));
        let (mut kvi, mut gi) = (0usize, 0usize);
        for (li, layer) in self.layers.iter().enumerate() {
            let sub = self.layer_ctx(ctx, kvi, gi);
            if Some(li) == stage_layer {
                // 六站展开(与 forward_stages 逐式同源;gdn_tap 若设,
                // mixer 内自推 8 件 → 层后收割)
                let st = layer.forward_bisect_stages(&xs, &sub);
                let names = ["n1", "mixed", "h", "n2", "mlp", "out"];
                for (j, n) in st.iter().enumerate() {
                    nodes.push((format!("s{li}.{}", names[j]), n.clone()));
                }
                if let Some(tap) = &sub.gdn_tap {
                    let gdn_fields = ["q", "k", "v", "qn", "kn", "vc", "g", "beta"];
                    for (j, n) in tap.borrow_mut().drain(..).enumerate() {
                        nodes.push((format!("s{li}.g{}", gdn_fields[j % 8]), n));
                    }
                }
                xs = st[5].clone();
            } else {
                xs = layer.forward(&xs, &sub);
            }
            if layer.is_full() {
                kvi += 1;
            } else {
                gi += 1;
            }
            nodes.push((format!("l{li}"), xs.clone()));
        }
        let fin = self.norm.forward(&xs, ctx);
        nodes.push(("fin".into(), fin.clone()));
        (fin, nodes)
    }

    /// 主干声明至 last hidden(pre-lm_head;MTP/深栈支线的挂点)
    pub fn last_hidden(&self, ids: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        // 层根自动打标(interpreter-tap.md §3.2):Model 是唯一知道层坐标
        // 的声明者;tag 是纯标注(同 id 不变,CSE memo 不受影响),是观测
        // 面 tap 曲线/golden 指纹的定位键。
        let mut xs = self.embed.forward(ids, ctx).tag("embed");
        xs = self.vlm_inject(xs, ctx);
        // 刀D 取证:层间 clock64 时间戳(OWL_TS_PROBE;embed 后/每层后/
        // final_norm 后各一拍,回放后 dtoh ts_buf 读原生层间时间线)
        let ts = ctx.ts_buf.clone().filter(|_| ctx.env.diag.ts_probe);
        let mut ti = 0usize;
        if let Some(buf) = &ts {
            xs = ts_stamp(&xs, buf, ti);
            ti += 1;
        }
        let (mut kvi, mut gi) = (0usize, 0usize);
        for (li, layer) in self.layers.iter().enumerate() {
            let sub = self.layer_ctx(ctx, kvi, gi);
            xs = layer
                .forward(&xs, &sub)
                .tag(format!("L{li}.{}", if layer.is_full() { "attn" } else { "gdn" }));
            if layer.is_full() {
                kvi += 1;
            } else {
                gi += 1;
            }
            if let Some(buf) = &ts {
                xs = ts_stamp(&xs, buf, ti);
                ti += 1;
            }
        }
        let xs = self.norm.forward(&xs, ctx).tag("final_norm");
        match &ts {
            Some(buf) => ts_stamp(&xs, buf, ti),
            None => xs,
        }
    }

    /// 层输出 taps 采集(E5-DF2;DFlash2 memory 面配套):层循环同构,
    /// tap_ids 层输出残差流克隆收集(sglang capture residual 同语义;
    /// tap = 树内别名节点,eval 共享前向零额外计算)。返回 (末层 hidden
    /// pre-final-norm, taps 按 tap_ids 序)。
    pub fn tapped_hidden(
        &self,
        ids: &TensorOps,
        ctx: &ForwardCtx,
        tap_ids: &[usize],
    ) -> (TensorOps, Vec<TensorOps>) {
        let mut taps: Vec<Option<TensorOps>> = vec![None; tap_ids.len()];
        let mut xs = self.embed.forward(ids, ctx).tag("embed");
        xs = self.vlm_inject(xs, ctx);
        let (mut kvi, mut gi) = (0usize, 0usize);
        for (li, layer) in self.layers.iter().enumerate() {
            let sub = self.layer_ctx(ctx, kvi, gi);
            xs = layer.forward(&xs, &sub);
            if layer.is_full() {
                kvi += 1;
            } else {
                gi += 1;
            }
            if let Some(pos) = tap_ids.iter().position(|&t| t == li) {
                taps[pos] = Some(xs.clone().tag(format!("tap{li}")));
            }
        }
        // final norm(与 last_hidden 对齐;调用方 tok/hid 消费同一口径 ——
        // 2026-10-07 恒等门分歧根因:pre-norm hidden 进 lm_head = tok 全错)
        let fin = self.norm.forward(&xs, ctx).tag("final_norm");
        (
            fin,
            taps.into_iter().map(|t| t.expect("tap_ids 层号非法")).collect(),
        )
    }
}

impl Module for Model {
    /// 整模单树(C5):embed → 层链 → final norm → tied lm_head
    fn forward(&self, ids: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        self.embed
            .lm_head_matmul(&self.last_hidden(ids, ctx), &ctx.env.gguf)
            .tag("logits")
    }
}

/// 刀D 取证:层间时间戳内核(OWL_TS_PROBE)。拷贝 in→out(链式接续)+
/// thread(0,0) 写 clock64 到 ts_buf[idx]。发射 = 哨兵自动网格
/// (2026-10-12 源收编 kernels cu/owl/ts_probe.cu + 登记表;三形态归一 Call)。
fn ts_stamp(xs: &TensorOps, buf: &TensorOps, idx: usize) -> TensorOps{
    let shape = xs.shape.clone();
    TensorOps::call(crate::ops::SemanticKernel::TsStamp)
        .arg(xs)
        .arg(buf)
        .arg_usize(idx)
        .arg_usize(shape.iter().product::<usize>())
        .with_shape(xs.dtype, shape)
}

impl Model {
    /// Prefill 末行头(2026-10-02;末块专用):last_hidden 窄到末行再
    /// lm_head —— logits [1,V] 而非 [T,V](chunk 1024 @248320 词表:
    /// 508MB → 0.5MB,GEMM 2.6 TFLOP → 2.6 GFLOP;prefill 采样只吃
    /// 末行)。中间 chunk 本就免算(last_hidden 直根)。
    pub fn forward_last(&self, ids: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        let hidden = self.last_hidden(ids, ctx);
        let t = ctx.tokens;
        let d = hidden.shape[1];
        let last = crate::layers::narrow_strided(
            &hidden, 1, d, (t - 1) * d, d, vec![1, d],
        );
        self.embed.lm_head_matmul(&last, &ctx.env.gguf).tag("logits_last")
    }
}

impl Loadable for Model {
    /// 整模装载需求(C10):embed / layers.{i} / norm 三段子清单 chain
    /// 聚合,局部键经 KeyConvention 改写为 checkpoint 全键 —— 改写
    /// 发生在声明期,Want 清单里已是最终键,解释器零约定感知。
    /// tied embedding 的双 Want(直读 + 转置)同键改写,天然保持同键。
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        let mut ops = self.embed.layout(ctx).map_keys(|k| self.keys.embed_key(k));
        // untied lm_head:独立槽(**局部键**;全键经 head_key 改写 ——
        // 第 4 类键:HF `lm_head.weight` / GGUF `output.weight`)
        if let Some(lm) = self.embed.layout_lm_head(ctx) {
            ops = ops.chain(lm.map_keys(|k| self.keys.head_key(k)));
        }
        for (i, layer) in self.layers.iter().enumerate() {
            ops = ops.chain(layer.layout(ctx).map_keys(|k| self.keys.layer_key(i, k)));
        }
        ops.chain(self.norm.layout(ctx).map_keys(|k| self.keys.norm_key(k)))
    }
}

// 保持 Dtype 引用(层文件统一风格)
const _: Dtype = Dtype::F32;

// ============================================================================
// 测试:fixture 装载 + 声明良构(CPU)+ 两步 decode vs host 全链(GPU)
// ============================================================================

#[cfg(test)]
mod spec_shard_tests {
    //! S4a:`ModelSpec::sharded` 的形状级切分(纯逻辑;27B 真实维度)。
    use super::ModelSpec;
    use crate::plan::tp::TpPlan;

    fn plan(r: usize, w: usize) -> TpPlan {
        TpPlan::new(r, w).expect("合法计划")
    }

    /// 27B 真实维度 × world=2:逐字段切分正确(设计 §15.2)
    #[test]
    fn sharded_27b_world2_matches_design() {
        let full = crate::formats::qwen35::qwen3_8_27b();
        let s = full.sharded(plan(0, 2)).expect("27B 可切");
        assert_eq!(s.hidden, 5120, "hidden 不切");
        assert_eq!(s.vocab, 248320, "vocab 不切(v1 复制)");
        assert_eq!(s.inter, 8704, "inter 切半");
        assert_eq!(s.full_heads, (12, 2, 256), "(hq,hkv) 切半,hd 不切");
        assert_eq!(s.gdn_heads, (8, 128, 24, 128), "(nk,nv) 切半,(hk,hv) 不切");
        assert_eq!(s.layer_types, full.layer_types, "层型表原样");
        assert_eq!(s.tied, full.tied);
        // 两 rank 的分片尺寸相同(等分)
        let s1 = full.sharded(plan(1, 2)).expect("rank1");
        assert_eq!((s1.inter, s1.full_heads, s1.gdn_heads), (s.inter, s.full_heads, s.gdn_heads));
    }

    /// **P1.1 词表切**:`head_split = true` ⇒ `head_vocab = vocab / world`,
    /// embed 表仍全量;`false` ⇒ 逐字段不变(缺省零行为变化)
    #[test]
    fn sharded_head_split_splits_only_the_head() {
        let mut full = crate::formats::qwen35::qwen3_8_27b();
        assert_eq!(full.head_vocab, full.vocab, "缺省:头 = 全量");
        let s0 = full.sharded(plan(0, 2)).expect("未开切:可切");
        assert_eq!(s0.head_vocab, 248320, "未开切 ⇒ 头全量(旧行为)");
        full.head_split = true;
        let s = full.sharded(plan(0, 2)).expect("词表切:可切");
        assert_eq!(s.vocab, 248320, "**embed 表仍全量**(每 rank 查任意 token)");
        assert_eq!(s.head_vocab, 124160, "lm_head 切半");
        assert!(s.head_split, "意图位透传");
        // 两 rank 行区间恰好覆盖 [0, vocab) 且互不重叠
        let r0 = TpPlan::new(0, 2).unwrap().shard_range(248320).unwrap();
        let r1 = TpPlan::new(1, 2).unwrap().shard_range(248320).unwrap();
        assert_eq!((r0.start, r0.end, r1.start, r1.end), (0, 124160, 124160, 248320));
        // world=1 退化:即使开了切,头也是全量
        let s1 = full.sharded(TpPlan::single()).expect("单卡");
        assert_eq!(s1.head_vocab, 248320);
    }

    /// **整除硬门**:词表不被 world 整除 ⇒ 响亮拒(不许 pad)
    #[test]
    fn sharded_head_split_refuses_uneven_vocab() {
        let full = ModelSpec {
            vocab: 5, // 奇数:world=2 切不动
            head_vocab: 5,
            head_split: true,
            hidden: 8,
            inter: 4,
            dtype: crate::contract::Dtype::F16,
            full_heads: (2, 2, 4), // hkv 也要被 2 整除(否则拒因不在词表那一维)
            gdn_heads: (2, 4, 2, 4),
            eps: 1e-6,
            layer_types: vec![true],
            tokenizer: crate::formats::qwen35::qwen3_5_0_8b().tokenizer,
            tied: false,
        };
        let e = match full.sharded(plan(0, 2)) {
            Err(e) => e,
            Ok(_) => panic!("vocab 5 不被 2 整除应拒"),
        };
        assert!(e.contains("整除") || e.contains("shard"), "拒因应指向整除:{e}");
        // 同 spec 关掉切 ⇒ 可切(证明拒因只在词表切那一维)
        let mut ok = full.clone();
        ok.head_split = false;
        assert!(ok.sharded(plan(0, 2)).is_ok());
    }

    /// **Model::new 的头部**:切档 ⇒ `logits_vocab = vocab/world` 且头规则 = UniformCol;
    /// 未切档 ⇒ 全量 + 默认规则(Replicate)
    #[test]
    fn model_head_vocab_and_rule_follow_the_split() {
        let full = crate::formats::qwen35::qwen3_8_27b();
        let keys = || crate::formats::qwen35::Qwen35Convention::new("model");
        let m_full = super::Model::new(&full, keys(), crate::module::QuantPlan::F16, crate::module::QuantPlan::F16).expect("模型构造");
        assert_eq!(m_full.logits_vocab(), 248320);
        assert_eq!(m_full.vocab_size(), 248320);
        assert_eq!(m_full.embed.head_rule(), crate::plan::rules::WeightRule::Replicate, "未切:头不切");

        let mut sp = full.clone();
        sp.head_split = true;
        let sharded = sp.sharded(plan(0, 2)).expect("切");
        let m = super::Model::new(&sharded, keys(), crate::module::QuantPlan::F16, crate::module::QuantPlan::F16).expect("模型构造");
        assert_eq!(m.logits_vocab(), 124160, "本 rank 的 logits 宽度");
        assert_eq!(m.vocab_size(), 248320, "词表全量(embed/tokenizer 面)");
        assert_eq!(m.embed.head_rule(), crate::plan::rules::WeightRule::UniformCol, "切档:头按 out 维切");
    }

    /// world=1 = 逐字段恒等(退化路径零特判)
    #[test]
    fn sharded_single_is_identity() {
        let full = crate::formats::qwen35::qwen3_8_27b();
        let s = full.sharded(TpPlan::single()).expect("单卡退化");
        assert_eq!(s.hidden, full.hidden);
        assert_eq!(s.inter, full.inter);
        assert_eq!(s.full_heads, full.full_heads);
        assert_eq!(s.gdn_heads, full.gdn_heads);
        assert_eq!(s.vocab, full.vocab);
    }

    /// 整除硬门:任一切分维不被 world 整除 ⇒ 响亮拒(不许 pad)
    #[test]
    fn sharded_rejects_non_divisible() {
        let full = crate::formats::qwen35::qwen3_8_27b();
        // 17408 = 2^10×17 ⇒ world=3 / 5 都不整除
        assert!(full.sharded(plan(0, 3)).is_err(), "inter 不整除应拒");
        assert!(full.sharded(plan(0, 5)).is_err(), "inter 不整除应拒");
        // hkv = 4:world=3 时 inter 也拒(先撞哪个都算拒)
        assert!(full.sharded(plan(0, 3)).is_err());
        // 0.8B:inter 3584 = 2^9×7 ⇒ world=3 拒
        let small = crate::formats::qwen35::qwen3_5_0_8b();
        assert!(small.sharded(plan(0, 3)).is_err());
        // 0.8B world=2:hq 8→4, hkv 2→1, gdn (16,16)→(8,8);inter 1792
        let s = small.sharded(plan(0, 2)).expect("0.8B 可切");
        assert_eq!((s.inter, s.full_heads, s.gdn_heads), (1792, (4, 1, 256), (8, 128, 8, 128)));
    }

    /// **几何闭合**:所有 rank 的分片**拼接回全量**(被切维),未切维原样
    /// —— 这是层构造"内外关系闭合"的前提(q_proj 输出 = hq·hd = o_proj 输入,
    /// 由同一 spec 派生 ⇒ 自动一致)
    #[test]
    fn sharded_geometry_tiles_back_to_full() {
        let full = crate::formats::qwen35::qwen3_8_27b();
        for world in [1usize, 2] {
            let (mut inter, mut hq, mut hkv, mut nk, mut nv) = (0usize, 0, 0, 0, 0);
            for r in 0..world {
                let s = full.sharded(plan(r, world)).unwrap();
                inter += s.inter;
                hq += s.full_heads.0;
                hkv += s.full_heads.1;
                nk += s.gdn_heads.0;
                nv += s.gdn_heads.2;
                // 未切维:每个 rank 都保持全量
                assert_eq!(s.hidden, full.hidden, "hidden 每 rank 全量");
                assert_eq!(s.full_heads.2, full.full_heads.2, "hd 每 rank 全量");
                assert_eq!(s.gdn_heads.1, full.gdn_heads.1, "hk 每 rank 全量");
                assert_eq!(s.gdn_heads.3, full.gdn_heads.3, "hv 每 rank 全量");
            }
            assert_eq!(
                (inter, hq, hkv, nk, nv),
                (
                    full.inter,
                    full.full_heads.0,
                    full.full_heads.1,
                    full.gdn_heads.0,
                    full.gdn_heads.2
                ),
                "world={world}:分片拼接 ≠ 全量(切分丢/多)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::QuantPlan;
    use crate::layers::gdn::GdnBuffers;
    use crate::module::LlamaFamily;
    use crate::layers::rope::Rope;
    use crate::module::KvBuffers;
    use crate::testkit::{assert_close, f32b, gpu_client, gpu_enabled, skip_note};
    use crate::tensor::Dtype;
    use std::collections::HashMap;

    // fixture 维度(缩水档;两 mixer + 3:1 周期一轮)
    const VOCAB: usize = 16;
    const HIDDEN: usize = 6;
    const INTER: usize = 5;
    const HQ: usize = 2;
    const HKV: usize = 1;
    const HD: usize = 4;
    const NK: usize = 2;
    const HK_DIM: usize = 4;
    const NV: usize = 2;
    const HV_DIM: usize = 4;
    const EPS: f32 = 1e-6;
    const MAX_SLOTS: usize = 4;
    const KV_ROWS: usize = 8;

    /// 层型单一来源(spec / checkpoint_src / HostModel 三处共用;
    /// 二分批改这里)
    fn layer_pattern() -> Vec<bool> {
        vec![false, false, false, true] // G,G,G,F
    }

    fn spec() -> ModelSpec {
        ModelSpec {
            head_vocab: 248320,
            head_split: false,
            vocab: VOCAB,
            hidden: HIDDEN,
            inter: INTER,
            dtype: Dtype::F32, // 单元锚保持 f32(CPU 面;fixture 测试)
            full_heads: (HQ, HKV, HD),
            gdn_heads: (NK, HK_DIM, NV, HV_DIM),
            eps: EPS,
            layer_types: layer_pattern(),
            tokenizer: crate::tokenizer::TokenizerSpec {
                eos_tokens: vec!["<|endoftext|>"],
                chat: crate::tokenizer::ChatFormat {
                    prefix: String::new(),
                    suffix: String::new(),
                    im_open: String::new(),
                    im_close: String::new(),
                    assistant_open: String::new(),
                    think_prefill: String::new(),
                },
            },
            tied: true,
        }
    }

    fn gen(seed: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.13).sin() * 0.5).collect()
    }

    /// checkpoint 全键源(经 LlamaFamily 约定走 C10 真路径)
    fn checkpoint_src() -> HashMap<String, Vec<f32>> {
        let base = "model.language_model";
        let mut m: HashMap<String, Vec<f32>> = HashMap::new();
        let mut put = |k: String, n: usize, seed: f32| {
            m.insert(k, gen(seed, n));
        };
        put(format!("{base}.embed_tokens.weight"), VOCAB * HIDDEN, 100.0);
        let pat = layer_pattern();
        #[allow(clippy::needless_range_loop)]
        for i in 0..pat.len() {
            let mut p = |k: &str, n: usize, seed: f32| {
                put(format!("{base}.layers.{i}.{k}.weight"), n, seed);
            };
            p("input_layernorm", HIDDEN, 1.0 + i as f32);
            p("post_attention_layernorm", HIDDEN, 2.0 + i as f32);
            if !pat[i] {
                // GDN 层
                let key_dim = NK * HK_DIM;
                let value_dim = NV * HV_DIM;
                p("in_proj_qkvz", (2 * key_dim + 2 * value_dim) * HIDDEN, 3.0);
                p("in_proj_b", NV * HIDDEN, 5.0);
                p("in_proj_a", NV * HIDDEN, 6.0);
                p("out_proj", HIDDEN * value_dim, 7.0);
                p("conv1d", (2 * key_dim + value_dim) * 4, 8.0);
                p("A_log", NV, 9.0);
                p("dt_bias", NV, 10.0);
                p("norm", HV_DIM, 11.0);
            } else {
                // full 层
                p("q_proj", HQ * HD * 2 * HIDDEN, 3.0);
                p("k_proj", HKV * HD * HIDDEN, 4.0);
                p("v_proj", HKV * HD * HIDDEN, 5.0);
                p("o_proj", HIDDEN * HQ * HD, 6.0);
                p("q_norm", HD, 7.0);
                p("k_norm", HD, 8.0);
            }
            p("gate_proj", INTER * HIDDEN, 12.0);
            p("up_proj", INTER * HIDDEN, 13.0);
            p("down_proj", HIDDEN * INTER, 14.0);
        }
        put(format!("{base}.norm.weight"), HIDDEN, 15.0);
        m
    }

    /// 零块(htod 清零;勿用裸 alloc —— 垃圾内存会进 conv/rec/KV 状态)
    async fn zero_block(client: &mut owl_cuda::GpuClient, n: usize, shape: Vec<usize>) -> TensorOps {
        let b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F32, vec![n], &f32b(&vec![0.0; n])).step(),
            client,
        )
        .await
        .expect("zero block");
        TensorOps::of_block(b.id, Dtype::F32, shape)
    }

    fn gdn_buf(gi: usize) -> GdnBuffers {
        // GPU 臂用零块构造;此构造器供 CPU 声明测试(零块声明即可)
        let key_dim = NK * HK_DIM;
        let value_dim = NV * HV_DIM;
        GdnBuffers {
            conv_q: TensorOps::zeros(Dtype::F32, vec![MAX_SLOTS, key_dim, 3]),
            conv_k: TensorOps::zeros(Dtype::F32, vec![MAX_SLOTS, key_dim, 3]),
            conv_v: TensorOps::zeros(Dtype::F32, vec![MAX_SLOTS, value_dim, 3]),
            rec: TensorOps::zeros(Dtype::F32, vec![MAX_SLOTS, NV, HK_DIM, HV_DIM]),
            slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[gi as f32])),
        }
    }

    /// CPU 臂:checkpoint 键形装载 + 声明良构(kvs/gdns 派生 + 双分支)
    #[tokio::test]
    async fn loads_checkpoint_keys_and_declares() {
        let mut face = owl_cpu::CpuFace::new();
        let model = Model::new(
            &spec(),
            LlamaFamily::new("model.language_model"),
            QuantPlan::F16,
            QuantPlan::F16,
        ).expect("Model::new");
        crate::interpreters::eval_load(&model, &mut face, &checkpoint_src(), &Default::default())
            .await
            .expect("model.eval_load(C10 声明路径)");

        // 装载完备性:embed 双槽 + final norm(all 层槽位由缺键 Err 兼容性覆盖)
        assert!(model.embed.is_loaded(), "embed 双槽");
        assert!(model.norm.is_loaded(), "final norm");

        // 声明良构:logits [1, vocab] 无毒(CPU face 不执行 Kernel,只验声明)
        let ids = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[3.0]));
        let pos = TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[0.0]));  // 契约 pos r2
        let kvs: Vec<KvBuffers> = (0..1)
            .map(|_| KvBuffers {
                k_cache: TensorOps::zeros(Dtype::F32, vec![KV_ROWS, HKV, HD]),
                v_cache: TensorOps::zeros(Dtype::F32, vec![KV_ROWS, HKV, HD]),
                slots: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[0.0])),  // 契约 slots/kv_lens r2(B6 收敛)
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[1.0])),  // 契约 slots/kv_lens r2(B6 收敛)
                block_tables: TensorOps::zeros(Dtype::F32, vec![1]), // 哑表(legacy 路径不解引用)
            })
            .collect();
        let gdns: Vec<GdnBuffers> = (0..3).map(gdn_buf).collect();
        let rp = Rope::new(64, HD, HD, 10_000.0).expect("rope");
        let ctx = ForwardCtx::model_decode(1, &pos, &kvs, &rp, &gdns);
        let logits = model.forward(&ids, &ctx);
        assert!(!logits.is_poisoned(), "整模声明不应有毒(缺依赖会沿分支毒化)");
        assert_eq!(logits.shape(), &[1, VOCAB]);
        assert_eq!(model.hidden(), HIDDEN);
        assert_eq!(model.vocab_size(), VOCAB);
    }

    // ======================================================================
    // GPU 两步 decode vs host 全链(门控 OWL_TEST_DEVICE)
    // ======================================================================

    /// host 全链参考(独立副本;内维显式传参 —— §四 16 律)。
    /// 两步跨步持状态:GDN conv/rec per-layer、full 层 KV cache 增行。
    struct HostModel {
        src: HashMap<String, Vec<f32>>,
        gdn_conv: Vec<[Vec<f32>; 3]>,       // per gdn 层 q/k/v 段 [slot, dim, 3]
        gdn_rec: Vec<Vec<f32>>,             // per gdn 层 [slot, NV, HK_DIM, HV_DIM]
        kv: Vec<(Vec<f32>, Vec<f32>)>,      // per full 层 (k, v) [row, hkv*hd]
    }

    impl HostModel {
        fn new(src: HashMap<String, Vec<f32>>) -> Self {
            let key_dim = NK * HK_DIM;
            let value_dim = NV * HV_DIM;
            let pat = layer_pattern();
            let ngdn = pat.iter().filter(|f| !**f).count();
            let nfull = pat.len() - ngdn;
            HostModel {
                gdn_conv: (0..ngdn)
                    .map(|_| {
                        [
                            vec![0.0; MAX_SLOTS * key_dim * 3],
                            vec![0.0; MAX_SLOTS * key_dim * 3],
                            vec![0.0; MAX_SLOTS * value_dim * 3],
                        ]
                    })
                    .collect(),
                gdn_rec: (0..ngdn)
                    .map(|_| vec![0.0; MAX_SLOTS * NV * HK_DIM * HV_DIM])
                    .collect(),
                kv: (0..nfull)
                    .map(|_| (vec![0.0; KV_ROWS * HKV * HD], vec![0.0; KV_ROWS * HKV * HD]))
                    .collect(),
                src,
            }
        }

        fn s(&self, i: usize, k: &str) -> &Vec<f32> {
            &self.src[&format!("model.language_model.layers.{i}.{k}.weight")]
        }

        fn lin(&self, x: &[f32], w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
            (0..out_dim)
                .map(|o| (0..in_dim).map(|j| x[j] * w[o * in_dim + j]).sum::<f32>())
                .collect()
        }

        fn ln(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
            let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = 1.0 / (ms + EPS).sqrt();
            x.iter().zip(w).map(|(v, g)| v * inv * (g + 1.0)).collect()
        }

        fn rope_one(x: &[f32], heads: usize, p: f32) -> Vec<f32> {
            let half = HD / 2;
            let mut out = x.to_vec();
            for h in 0..heads {
                for i in 0..half {
                    let ang = p * 10_000f32.powf(-(2.0 * i as f32) / HD as f32);
                    let (c, sn) = (ang.cos(), ang.sin());
                    let (a, b) = (x[h * HD + i], x[h * HD + i + half]);
                    out[h * HD + i] = a * c - b * sn;
                    out[h * HD + i + half] = b * c + a * sn;
                }
            }
            out
        }

        fn rms_hd(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
            let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / HD as f32;
            let inv = 1.0 / (ms + EPS).sqrt();
            x.iter().zip(w).map(|(v, g)| v * inv * (g + 1.0)).collect()
        }

        /// full mixer(单 token decode;cache 增行 + 窗 softmax)
        fn attn(&mut self, fi: usize, i: usize, n1: &[f32], pos: f32, slot: usize, kv_len: usize) -> Vec<f32> {
            let q_raw = self.lin(n1, self.s(i, "q_proj"), HQ * HD * 2, HIDDEN);
            let k_raw = self.lin(n1, self.s(i, "k_proj"), HKV * HD, HIDDEN);
            let v_raw = self.lin(n1, self.s(i, "v_proj"), HKV * HD, HIDDEN);
            let mut q = [0.0; HQ * HD];
            let mut gate = vec![0.0; HQ * HD];
            for h in 0..HQ {
                for d in 0..HD {
                    q[h * HD + d] = q_raw[h * HD * 2 + d];
                    gate[h * HD + d] = q_raw[h * HD * 2 + HD + d];
                }
            }
            let mut qn = vec![0.0f32; HQ * HD];
            let mut kn = vec![0.0f32; HKV * HD];
            for h in 0..HQ {
                qn[h * HD..(h + 1) * HD]
                    .copy_from_slice(&self.rms_hd(&q[h * HD..(h + 1) * HD], self.s(i, "q_norm")));
            }
            for h in 0..HKV {
                kn[h * HD..(h + 1) * HD]
                    .copy_from_slice(&self.rms_hd(&k_raw[h * HD..(h + 1) * HD], self.s(i, "k_norm")));
            }
            let q = Self::rope_one(&qn, HQ, pos);
            let k = Self::rope_one(&kn, HKV, pos);
            // 写 cache 第 slot 行
            let (kc, vc) = &mut self.kv[fi];
            kc[slot * HKV * HD..(slot + 1) * HKV * HD].copy_from_slice(&k);
            vc[slot * HKV * HD..(slot + 1) * HKV * HD].copy_from_slice(&v_raw);
            // 窗 [slot-kv_len+1, slot] 打分
            let kc = &self.kv[fi].0;
            let vc = &self.kv[fi].1;
            let scale = 1.0 / (HD as f32).sqrt();
            let mut y = vec![0.0f32; HQ * HD];
            for h in 0..HQ {
                let kvh = h / (HQ / HKV);
                let qs = &q[h * HD..(h + 1) * HD];
                let mut scores = Vec::new();
                for r in (slot + 1 - kv_len)..=slot {
                    let ks = &kc[r * HKV * HD + kvh * HD..r * HKV * HD + (kvh + 1) * HD];
                    scores.push(qs.iter().zip(ks).map(|(a, b)| a * b).sum::<f32>() * scale);
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let es: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f32 = es.iter().sum();
                for d in 0..HD {
                    let mut acc = 0.0f32;
                    for (j, r) in ((slot + 1 - kv_len)..=slot).enumerate() {
                        acc += es[j] / sum * vc[r * HKV * HD + kvh * HD + d];
                    }
                    y[h * HD + d] = acc;
                }
            }
            for (yi, g) in y.iter_mut().zip(&gate) {
                *yi *= 1.0 / (1.0 + (-g).exp());
            }
            self.lin(&y, self.s(i, "o_proj"), HIDDEN, HQ * HD)
        }

        /// GDN mixer(单 token;per-layer conv/rec 状态跨步)
        fn gdn(&mut self, gi: usize, i: usize, n1: &[f32], slot: usize) -> Vec<f32> {
            let key_dim = NK * HK_DIM;
            let value_dim = NV * HV_DIM;
            let conv_dim = 2 * key_dim + value_dim;
            let qkvz = self.lin(n1, self.s(i, "in_proj_qkvz"), conv_dim + value_dim, HIDDEN);
            let qkv = &qkvz[..conv_dim];
            let z = &qkvz[conv_dim..];
            let b_v = self.lin(n1, self.s(i, "in_proj_b"), NV, HIDDEN);
            let a_v = self.lin(n1, self.s(i, "in_proj_a"), NV, HIDDEN);
            let segs = [&qkv[0..key_dim], &qkv[key_dim..2 * key_dim], &qkv[2 * key_dim..]];
            let seg_dim = [key_dim, key_dim, value_dim];
            let seg_off = [0usize, key_dim, 2 * key_dim];
            let mut post = [vec![0.0f32; key_dim], vec![0.0f32; key_dim], vec![0.0f32; value_dim]];
            for (seg, inp) in segs.iter().enumerate() {
                let w = self.s(i, "conv1d").clone();
                let st = &mut self.gdn_conv[gi][seg];
                for ch in 0..seg_dim[seg] {
                    let sb = (slot * seg_dim[seg] + ch) * 3;
                    let wb = (seg_off[seg] + ch) * 4;
                    let hist = [st[sb], st[sb + 1], st[sb + 2]];
                    let mut sum = inp[ch] * w[wb + 3];
                    for kk in 0..3 {
                        sum += hist[kk] * w[wb + kk];
                    }
                    sum /= 1.0 + (-sum).exp();
                    st[sb] = hist[1];
                    st[sb + 1] = hist[2];
                    st[sb + 2] = inp[ch];
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
            for h in 0..NK {
                qn[h * HK_DIM..(h + 1) * HK_DIM]
                    .copy_from_slice(&l2(&post[0][h * HK_DIM..(h + 1) * HK_DIM]));
                kn[h * HK_DIM..(h + 1) * HK_DIM]
                    .copy_from_slice(&l2(&post[1][h * HK_DIM..(h + 1) * HK_DIM]));
            }
            let a_log = self.s(i, "A_log");
            let dtb = self.s(i, "dt_bias");
            let g_v: Vec<f32> = (0..NV)
                .map(|h| {
                    let x2 = a_v[h] + dtb[h];
                    let sp = if x2 < 20.0 { x2.exp().ln_1p() } else { x2 };
                    -a_log[h].exp() * sp
                })
                .collect();
            let beta_v: Vec<f32> = b_v.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            let q_scale = 1.0 / (HK_DIM as f32).sqrt();
            let mut y = vec![0.0f32; value_dim];
            for vh in 0..NV {
                let kh = vh / (NV / NK);
                let decay = g_v[vh].exp();
                let sb = (slot * NV + vh) * HK_DIM * HV_DIM;
                let qoff = kh * HK_DIM;
                let voff = vh * HV_DIM;
                let mut kv_mem = [0.0f32; HV_DIM];
                for j in 0..HK_DIM {
                    #[allow(clippy::needless_range_loop)]
                    for d in 0..HV_DIM {
                        let idx = sb + j * HV_DIM + d;
                        self.gdn_rec[gi][idx] *= decay;
                        kv_mem[d] += self.gdn_rec[gi][idx] * kn[qoff + j];
                    }
                }
                for d in 0..HV_DIM {
                    let delta = (post[2][voff + d] - kv_mem[d]) * beta_v[vh];
                    let mut acc = 0.0f32;
                    for j in 0..HK_DIM {
                        let idx = sb + j * HV_DIM + d;
                        self.gdn_rec[gi][idx] += kn[qoff + j] * delta;
                        acc += self.gdn_rec[gi][idx] * qn[qoff + j] * q_scale;
                    }
                    y[voff + d] = acc;
                }
            }
            let gamma = self.s(i, "norm");
            let mut gated = vec![0.0f32; value_dim];
            for vh in 0..NV {
                let base = vh * HV_DIM;
                let ms: f32 = (0..HV_DIM).map(|k| y[base + k] * y[base + k]).sum::<f32>() / HV_DIM as f32;
                let inv = 1.0 / (ms.max(0.0) + EPS).sqrt();
                for k in 0..HV_DIM {
                    let zv = z[base + k];
                    gated[base + k] = y[base + k] * inv * gamma[k] * (zv / (1.0 + (-zv).exp()));
                }
            }
            self.lin(&gated, self.s(i, "out_proj"), HIDDEN, value_dim)
        }

        /// 一步 decode(bs=1):ids/pos/slot/kv_len → logits [VOCAB]
        fn step(&mut self, id: f32, pos: f32, slot: usize, kv_len: usize) -> Vec<f32> {
            self.step2(id, pos, slot, kv_len, slot)
        }

        /// kv_slot / gdn_slot 分离(单序列 prefill 对拍:KV 行随 token 走,
        /// GDN 状态格恒定;旧双参 step 保留 = 两槽同值的两 token 语义)
        fn step2(&mut self, id: f32, pos: f32, kv_slot: usize, kv_len: usize, gdn_slot: usize) -> Vec<f32> {
            let w = self.src["model.language_model.embed_tokens.weight"].clone();
            let mut x = w[(id as usize) * HIDDEN..(id as usize + 1) * HIDDEN].to_vec();
            let (mut kvi, mut gi) = (0usize, 0usize);
            for (i, full) in layer_pattern().iter().enumerate() {
                let ln1 = self.ln(&x, self.s(i, "input_layernorm"));
                let mixed = if *full {
                    let m = self.attn(kvi, i, &ln1, pos, kv_slot, kv_len);
                    kvi += 1;
                    m
                } else {
                    let m = self.gdn(gi, i, &ln1, gdn_slot);
                    gi += 1;
                    m
                };
                for (xv, mv) in x.iter_mut().zip(&mixed) {
                    *xv += mv;
                }
                let ln2 = self.ln(&x, self.s(i, "post_attention_layernorm"));
                let gate = self.lin(&ln2, self.s(i, "gate_proj"), INTER, HIDDEN);
                let up = self.lin(&ln2, self.s(i, "up_proj"), INTER, HIDDEN);
                let h: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                    .collect();
                let down = self.lin(&h, self.s(i, "down_proj"), HIDDEN, INTER);
                for (xv, dv) in x.iter_mut().zip(&down) {
                    *xv += dv;
                }
            }
            let f = self.ln(&x, &self.src["model.language_model.norm.weight"]);
            (0..VOCAB)
                .map(|v| f.iter().enumerate().map(|(d, fv)| fv * w[v * HIDDEN + d]).sum::<f32>())
                .collect()
        }
    }

    /// GPU 两步 decode vs host 全链(状态跨步:GDN conv/rec + KV 增行;
    /// 门控 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn gpu_model_two_steps_match_host() {
        if !gpu_enabled() {
            skip_note();
            return;
        }
        let model = Model::new(
            &spec(),
            LlamaFamily::new("model.language_model"),
            QuantPlan::F16,
            QuantPlan::F16,
        ).expect("Model::new");
        let src = checkpoint_src();
        let mut gpu = gpu_client().await;
        crate::interpreters::eval_load(&model, &mut gpu, &src, &Default::default())
            .await
            .expect("model.eval_load");
        let rp = Rope::new(64, HD, HD, 10_000.0).expect("rope");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &Default::default())
            .await
            .expect("rope 表");

        // 常驻缓冲:1 full 层 KV + 3 gdn 层状态(全零起步)
        let kv_zero = [0.0f32; KV_ROWS * HKV * HD];
        let kvs: Vec<KvBuffers> = {
            let mut v = Vec::new();
            for _ in 0..1 {
                v.push(KvBuffers {
                    k_cache: zero_block(&mut gpu, kv_zero.len(), vec![KV_ROWS, HKV, HD]).await,
                    v_cache: zero_block(&mut gpu, kv_zero.len(), vec![KV_ROWS, HKV, HD]).await,
                    slots: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[0.0])),  // 契约 slots/kv_lens r2(B6 收敛)
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[1.0])),  // 契约 slots/kv_lens r2(B6 收敛)
                    block_tables: TensorOps::zeros(Dtype::F32, vec![1]), // 哑表(legacy 路径不解引用)
                });
            }
            v
        };
        let key_dim = NK * HK_DIM;
        let value_dim = NV * HV_DIM;
        let gdns: Vec<GdnBuffers> = {
            let mut v = Vec::new();
            for _ in 0..3 {
                v.push(GdnBuffers {
                    conv_q: zero_block(&mut gpu, MAX_SLOTS * key_dim * 3, vec![MAX_SLOTS, key_dim, 3]).await,
                    conv_k: zero_block(&mut gpu, MAX_SLOTS * key_dim * 3, vec![MAX_SLOTS, key_dim, 3]).await,
                    conv_v: zero_block(&mut gpu, MAX_SLOTS * value_dim * 3, vec![MAX_SLOTS, value_dim, 3]).await,
                    rec: zero_block(&mut gpu, MAX_SLOTS * NV * HK_DIM * HV_DIM, vec![MAX_SLOTS, NV, HK_DIM, HV_DIM]).await,
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                });
            }
            v
        };

        // host 参考(独立副本)
        let mut host = HostModel::new(src);

        // 两步:step1 (id=5, pos=0, slot=0, kv_len=1) → step2 (id=9, pos=1, slot=1, kv_len=2)
        let steps: [(f32, f32, f32, f32); 2] = [(5.0, 0.0, 0.0, 1.0), (9.0, 1.0, 1.0, 2.0)];
        for (si, &(id, pos, slot, kv_len)) in steps.iter().enumerate() {
            let ids = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[id]));
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[pos]));  // 契约 pos r2
            // 每步重写动态标量块(slots/kv_lens 由 runner 步间改写 —— C7)
            let kvs_step: Vec<KvBuffers> = kvs
                .iter()
                .map(|kv| KvBuffers {
                    k_cache: kv.k_cache.clone(),
                    v_cache: kv.v_cache.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[slot])),  // 契约 slots/kv_lens r2(B6 收敛)
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[kv_len])),  // 契约 slots/kv_lens r2(B6 收敛)
                    block_tables: TensorOps::zeros(Dtype::F32, vec![1]), // 哑表(legacy 路径不解引用)
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdns
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot])),
                })
                .collect();
            let ctx = ForwardCtx::model_decode(1, &pos_t, &kvs_step, &rp, &gdns_step);
            let logits = model.forward(&ids, &ctx);
            let got = crate::testkit::harvest(&mut gpu, &logits).await;
            let want = host.step(id, pos, slot as usize, kv_len as usize);
            assert_close(&got, &want, 1e-4, &format!("model-step{si}"));
        }
        gpu.close().await.expect("关机");
    }

    /// **批语义 M0 门**(B5.1,2026-10-20):两会话一次 T=2 decode ≡ 两次 T=1。
    ///
    /// fixture 的小几何(hd=4)不在分页配对里 ⇒ 落 `NaiveDecode` 槽直排
    /// (与 T=1 同一条回退律);验的是**批语义本身**:
    /// 每行自己的 KV 槽窗(`slots[]`/`kv_lens[]`,两条互不相邻的链)与
    /// GDN 行(`gdn.slots[]`)。串行臂/合批臂各用**一套独立缓冲**(同零初态 +
    /// 相同 prefill),判据 = logits 逐位/近位相同 + 第二步(状态续跑)仍相同。
    /// 分页 v2 seq 臂(不同 bt 行)由 verify T=8 已证同核,引擎接入另测。
    async fn model_step(
        gpu: &mut owl_cuda::GpuClient,
        model: &Model,
        rp: &Rope,
        kvs: &[KvBuffers],
        gdns: &[GdnBuffers],
        tokens: &[f32],
        pos: &[f32],
        slots: &[f32],
        lens: &[f32],
        gslots: &[f32],
    ) -> Vec<f32> {
        let t = tokens.len();
        let ids = TensorOps::from_host(Dtype::F32, vec![t], &f32b(tokens));
        let pos_t = TensorOps::from_host(Dtype::F32, vec![1, t], &f32b(pos));
        let kvs_step: Vec<KvBuffers> = kvs
            .iter()
            .map(|kv| KvBuffers {
                k_cache: kv.k_cache.clone(),
                v_cache: kv.v_cache.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![1, t], &f32b(slots)),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1, t], &f32b(lens)),
                block_tables: kv.block_tables.clone(),
            })
            .collect();
        let gdns_step: Vec<GdnBuffers> = gdns
            .iter()
            .map(|g| GdnBuffers {
                conv_q: g.conv_q.clone(),
                conv_k: g.conv_k.clone(),
                conv_v: g.conv_v.clone(),
                rec: g.rec.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![t], &f32b(gslots)),
            })
            .collect();
        let ctx = ForwardCtx::model_decode(t, &pos_t, &kvs_step, &rp, &gdns_step);
        let logits = model.forward(&ids, &ctx);
        crate::testkit::harvest(gpu, &logits).await
    }

    #[tokio::test]
    async fn gpu_model_batch_decode_matches_serial() {
        if !gpu_enabled() {
            skip_note();
            return;
        }
        let model = Model::new(
            &spec(),
            LlamaFamily::new("model.language_model"),
            QuantPlan::F16,
            QuantPlan::F16,
        ).expect("模型构造");
        let src = checkpoint_src();
        let mut gpu = gpu_client().await;
        crate::interpreters::eval_load(&model, &mut gpu, &src, &Default::default())
            .await
            .expect("model.eval_load");
        let rp = Rope::new(64, HD, HD, 10_000.0).expect("rope");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &Default::default())
            .await
            .expect("rope 表");

        // 两套独立缓冲:会话 A(slot 0..2 / GDN 0)、会话 B(slot 4..6 / GDN 1)
        // —— KV_ROWS=8 / MAX_SLOTS=4 内。
        let (kvs_s, gdns_s) = model_bufs(&mut gpu, 1, 3).await;
        let (kvs_b, gdns_b) = model_bufs(&mut gpu, 1, 3).await;

        // prefill:每会话两步(T=1),两臂逐字相同。
        let steps: [(&[f32], &[f32], &[f32], &[f32], &[f32]); 4] = [
            (&[5.0], &[0.0], &[0.0], &[1.0], &[0.0]),
            (&[9.0], &[1.0], &[1.0], &[2.0], &[0.0]),
            (&[3.0], &[0.0], &[4.0], &[1.0], &[1.0]),
            (&[7.0], &[1.0], &[5.0], &[2.0], &[1.0]),
        ];
        for (tok, pos, slot, len, gs) in steps {
            let _ = model_step(&mut gpu, &model, &rp, &kvs_s, &gdns_s, tok, pos, slot, len, gs).await;
            let _ = model_step(&mut gpu, &model, &rp, &kvs_b, &gdns_b, tok, pos, slot, len, gs).await;
        }

        // 串行臂:两步单行(A 再 B)。
        let a1 = model_step(&mut gpu, &model, &rp, &kvs_s, &gdns_s, &[11.0], &[2.0], &[2.0], &[3.0], &[0.0]).await;
        let b1 = model_step(&mut gpu, &model, &rp, &kvs_s, &gdns_s, &[13.0], &[2.0], &[6.0], &[3.0], &[1.0]).await;
        // 合批臂:一次 T=2(各走各的链/行)。
        let batch1 = model_step(&mut gpu, &model, &rp, &kvs_b, &gdns_b, &[11.0, 13.0], &[2.0, 2.0], &[2.0, 6.0], &[3.0, 3.0], &[0.0, 1.0]).await;
        assert_eq!(batch1.len(), 2 * VOCAB, "合批 logits 行数");
        assert_close(&batch1[..VOCAB], &a1, 1e-5, "batch-step1-rowA");
        assert_close(&batch1[VOCAB..], &b1, 1e-5, "batch-step1-rowB");

        // 第二步:验**状态续跑**(KV 写入 + GDN conv/rec 推进)两臂仍同。
        let a2 = model_step(&mut gpu, &model, &rp, &kvs_s, &gdns_s, &[17.0], &[3.0], &[3.0], &[4.0], &[0.0]).await;
        let b2 = model_step(&mut gpu, &model, &rp, &kvs_s, &gdns_s, &[19.0], &[3.0], &[7.0], &[4.0], &[1.0]).await;
        let batch2 = model_step(&mut gpu, &model, &rp, &kvs_b, &gdns_b, &[17.0, 19.0], &[3.0, 3.0], &[3.0, 7.0], &[4.0, 4.0], &[0.0, 1.0]).await;
        assert_close(&batch2[..VOCAB], &a2, 1e-5, "batch-step2-rowA");
        assert_close(&batch2[VOCAB..], &b2, 1e-5, "batch-step2-rowB");

        // 反向判别:两行必须**不同**(否则判据无分辨率)
        let d: f32 = batch1[..VOCAB].iter().zip(&batch1[VOCAB..]).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(d > 1e-3, "两行 logits 相同(maxdiff={d})→ 判据无分辨率");
        println!("✅ batch decode: T=2 与串行 T=1×2 两行逐位同级(maxdiff row={d:.3e})");
        gpu.close().await.expect("关机");
    }

    /// **M2 前置门**(B5.1):多序列 prefill(T = Σ 各会话行,`seq_cu`
    /// 分段)与逐会话串行 prefill **logits 同级**。这是 spec 回合合批
    /// (verify 8 行 × B 会话)的模型语义底:GDN 走 varlen 的 `slots[seq_idx]`,
    /// 注意力走逐行 `kv_slots/kv_lens`。
    async fn model_prefill_step(
        gpu: &mut owl_cuda::GpuClient,
        model: &Model,
        rp: &Rope,
        kvs: &[KvBuffers],
        gdns: &[GdnBuffers],
        tokens: &[f32],
        pos: &[f32],
        slots: &[f32],
        lens: &[f32],
        gslots: &[f32],
        seq_cu: &[f32],
    ) -> Vec<f32> {
        let t = tokens.len();
        let ids = TensorOps::from_host(Dtype::F32, vec![t], &f32b(tokens));
        let pos_t = TensorOps::from_host(Dtype::F32, vec![1, t], &f32b(pos));
        let slots_t = TensorOps::from_host(Dtype::F32, vec![t], &f32b(slots));
        let lens_t = TensorOps::from_host(Dtype::F32, vec![t], &f32b(lens));
        let gslot_t = TensorOps::from_host(Dtype::F32, vec![gslots.len()], &f32b(gslots));
        let cu = TensorOps::from_host(Dtype::F32, vec![seq_cu.len()], &f32b(seq_cu));
        let kvs_step: Vec<KvBuffers> = kvs
            .iter()
            .map(|kv| KvBuffers {
                k_cache: kv.k_cache.clone(),
                v_cache: kv.v_cache.clone(),
                slots: slots_t.clone(),
                kv_lens: lens_t.clone(),
                block_tables: kv.block_tables.clone(),
            })
            .collect();
        let gdns_step: Vec<GdnBuffers> = gdns
            .iter()
            .map(|g| GdnBuffers {
                conv_q: g.conv_q.clone(),
                conv_k: g.conv_k.clone(),
                conv_v: g.conv_v.clone(),
                rec: g.rec.clone(),
                slots: gslot_t.clone(),
            })
            .collect();
        let mut ctx = ForwardCtx::model_prefill(
            t, &pos_t, &kvs_step, &rp, &gdns_step, &slots_t, &lens_t, &gslot_t, None, 0, None,
        );
        ctx.seq_cu = Some(&cu);
        let logits = model.forward(&ids, &ctx);
        crate::testkit::harvest_f16(gpu, &logits).await
    }

    /// F16 KV 缓冲套(KV 池 = F16,GDN 状态恒 F32)
    async fn model_bufs_f16(
        gpu: &mut owl_cuda::GpuClient,
        n_full: usize,
        n_gdn: usize,
    ) -> (Vec<KvBuffers>, Vec<GdnBuffers>) {
        let kv_zero = vec![0u8; KV_ROWS * HKV * HD * 2];
        let mut kvs = Vec::new();
        for _ in 0..n_full {
            let k = crate::interpreters::eval_ops(
                TensorOps::from_host(Dtype::F16, vec![KV_ROWS * HKV * HD], &kv_zero).step(),
                gpu,
            )
            .await
            .expect("kv zero");
            let v = crate::interpreters::eval_ops(
                TensorOps::from_host(Dtype::F16, vec![KV_ROWS * HKV * HD], &kv_zero).step(),
                gpu,
            )
            .await
            .expect("kv zero");
            kvs.push(KvBuffers {
                k_cache: TensorOps::of_block(k.id, Dtype::F16, vec![KV_ROWS, HKV, HD]),
                v_cache: TensorOps::of_block(v.id, Dtype::F16, vec![KV_ROWS, HKV, HD]),
                slots: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[0.0])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[1.0])),
                block_tables: TensorOps::zeros(Dtype::F32, vec![1]),
            });
        }
        let key_dim = NK * HK_DIM;
        let value_dim = NV * HV_DIM;
        let mut gdns = Vec::new();
        for _ in 0..n_gdn {
            gdns.push(GdnBuffers {
                conv_q: zero_block(gpu, MAX_SLOTS * key_dim * 3, vec![MAX_SLOTS, key_dim, 3]).await,
                conv_k: zero_block(gpu, MAX_SLOTS * key_dim * 3, vec![MAX_SLOTS, key_dim, 3]).await,
                conv_v: zero_block(gpu, MAX_SLOTS * value_dim * 3, vec![MAX_SLOTS, value_dim, 3]).await,
                rec: zero_block(gpu, MAX_SLOTS * NV * HK_DIM * HV_DIM, vec![MAX_SLOTS, NV, HK_DIM, HV_DIM]).await,
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
            });
        }
        (kvs, gdns)
    }

    #[tokio::test]
    async fn gpu_model_batch_prefill_matches_serial() {
        if !gpu_enabled() {
            skip_note();
            return;
        }
        // 必须走 **F16 批核 prefill**(`forward_prefill_batch`,varlen `slots[seq_idx]`)：
        // F32 语义锚链的 expanded 路径按 token 行索引取 slots（单序列语义），
        // 多序列会恒读 slots[0] —— 那是测试面限制，不是生产路径。
        // 装载 dtype 由 `LoaderCtx.dtype` 定（`Model::new` 不读 spec.dtype）。
        let model = Model::new(
            &spec(),
            LlamaFamily::new("model.language_model"),
            QuantPlan::F16,
            QuantPlan::F16,
        ).expect("模型构造");
        let src = checkpoint_src();
        let mut gpu = gpu_client().await;
        let lctx = crate::module::LoaderCtx { dtype: Dtype::F16, ..Default::default() };
        crate::interpreters::eval_load(&model, &mut gpu, &src, &lctx)
            .await
            .expect("model.eval_load(F16)");
        let rp = Rope::new(64, HD, HD, 10_000.0).expect("rope");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &lctx)
            .await
            .expect("rope 表(F16)");
        let (kvs_s, gdns_s) = model_bufs_f16(&mut gpu, 1, 3).await;
        let (kvs_b, gdns_b) = model_bufs_f16(&mut gpu, 1, 3).await;

        // ── prefill:会话 A 2 行(slot 0,1 / GDN 0)、B 1 行(slot 4 / GDN 1)──
        // 串行臂:两次单序列 prefill
        let _ = model_prefill_step(&mut gpu, &model, &rp, &kvs_s, &gdns_s,
            &[5.0, 9.0], &[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0], &[0.0], &[0.0, 2.0]).await;
        let _ = model_prefill_step(&mut gpu, &model, &rp, &kvs_s, &gdns_s,
            &[3.0], &[0.0], &[4.0], &[1.0], &[1.0], &[0.0, 1.0]).await;
        // 合批臂:一次 T=3,seq_cu = [0,2,3]
        let batch_pre = model_prefill_step(&mut gpu, &model, &rp, &kvs_b, &gdns_b,
            &[5.0, 9.0, 3.0], &[0.0, 1.0, 0.0], &[0.0, 1.0, 4.0], &[1.0, 2.0, 1.0],
            &[0.0, 1.0], &[0.0, 2.0, 3.0]).await;
        // 串行臂的逐会话 logits(重跑以取输出;状态已推进 ⇒ 用第二套缓冲)
        let (kvs_s2, gdns_s2) = model_bufs_f16(&mut gpu, 1, 3).await;
        let s_a = model_prefill_step(&mut gpu, &model, &rp, &kvs_s2, &gdns_s2,
            &[5.0, 9.0], &[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0], &[0.0], &[0.0, 2.0]).await;
        let s_b = model_prefill_step(&mut gpu, &model, &rp, &kvs_s2, &gdns_s2,
            &[3.0], &[0.0], &[4.0], &[1.0], &[1.0], &[0.0, 1.0]).await;
        assert_close(&batch_pre[..2 * VOCAB], &s_a[..], 1e-5, "batch-prefill-A");
        assert_close(&batch_pre[2 * VOCAB..], &s_b[..], 1e-5, "batch-prefill-B");

        // ── verify 形(槽区不重叠):A 2 行(slot 2,3;pos 2,3)、
        //    B 3 行(slot 5..7;pos 1..3)──
        let a1 = model_prefill_step(&mut gpu, &model, &rp, &kvs_s2, &gdns_s2,
            &[11.0, 12.0], &[2.0, 3.0], &[2.0, 3.0], &[3.0, 4.0],
            &[0.0], &[0.0, 2.0]).await;
        let b1 = model_prefill_step(&mut gpu, &model, &rp, &kvs_s2, &gdns_s2,
            &[21.0, 22.0, 23.0], &[1.0, 2.0, 3.0], &[5.0, 6.0, 7.0], &[2.0, 3.0, 4.0],
            &[1.0], &[0.0, 3.0]).await;
        // 合批臂:一次 T=5,seq_cu = [0,2,5]
        let batch_v = model_prefill_step(&mut gpu, &model, &rp, &kvs_b, &gdns_b,
            &[11.0, 12.0, 21.0, 22.0, 23.0],
            &[2.0, 3.0, 1.0, 2.0, 3.0],
            &[2.0, 3.0, 5.0, 6.0, 7.0],
            &[3.0, 4.0, 2.0, 3.0, 4.0],
            &[0.0, 1.0], &[0.0, 2.0, 5.0]).await;
        assert_close(&batch_v[..2 * VOCAB], &a1[..], 1e-5, "batch-verify-A");
        assert_close(&batch_v[2 * VOCAB..], &b1[..], 1e-5, "batch-verify-B");
        println!("✅ 多序列 prefill/verify(2 会话,seq_cu 分段)与串行逐位同级");
        gpu.close().await.expect("关机");
    }

    /// 常驻缓冲一套(full 1 KV + gdn 3 状态;全零起步)
    async fn model_bufs(gpu: &mut owl_cuda::GpuClient, n_full: usize, n_gdn: usize) -> (Vec<KvBuffers>, Vec<GdnBuffers>) {
        let kv_zero = [0.0f32; KV_ROWS * HKV * HD];
        let mut kvs = Vec::new();
        for _ in 0..n_full {
            kvs.push(KvBuffers {
                k_cache: zero_block(gpu, kv_zero.len(), vec![KV_ROWS, HKV, HD]).await,
                v_cache: zero_block(gpu, kv_zero.len(), vec![KV_ROWS, HKV, HD]).await,
                slots: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[0.0])),  // 契约 slots/kv_lens r2(B6 收敛)
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[1.0])),  // 契约 slots/kv_lens r2(B6 收敛)
                block_tables: TensorOps::zeros(Dtype::F32, vec![1]), // 哑表(legacy 路径不解引用)
            });
        }
        let key_dim = NK * HK_DIM;
        let value_dim = NV * HV_DIM;
        let mut gdns = Vec::new();
        for _ in 0..n_gdn {
            gdns.push(GdnBuffers {
                conv_q: zero_block(gpu, MAX_SLOTS * key_dim * 3, vec![MAX_SLOTS, key_dim, 3]).await,
                conv_k: zero_block(gpu, MAX_SLOTS * key_dim * 3, vec![MAX_SLOTS, key_dim, 3]).await,
                conv_v: zero_block(gpu, MAX_SLOTS * value_dim * 3, vec![MAX_SLOTS, value_dim, 3]).await,
                rec: zero_block(gpu, MAX_SLOTS * NV * HK_DIM * HV_DIM, vec![MAX_SLOTS, NV, HK_DIM, HV_DIM]).await,
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
            });
        }
        (kvs, gdns)
    }

    /// 批P5:fixture 全模 prefill T=4 块 vs host 全链 4 步(路径等价;
    /// T ≤ MAX_SLOTS = GDN 槽位约束;真模型 T=8 见 qwen35 spec 测试)。
    /// ⚠️ 重放律(testkit harvest 注):带状态树一次 eval 只收一个根 ——
    /// last_hidden 与 logits 各走独立缓冲套,禁止同树双根。
    #[tokio::test]
    async fn gpu_model_prefill_block_matches_host_loop() {
        if !gpu_enabled() {
            skip_note();
            return;
        }
        let t_len = 4usize;
        let model = Model::new(
            &spec(),
            LlamaFamily::new("model.language_model"),
            QuantPlan::F16,
            QuantPlan::F16,
        ).expect("Model::new");
        let src = checkpoint_src();
        let mut gpu = gpu_client().await;
        crate::interpreters::eval_load(&model, &mut gpu, &src, &Default::default())
            .await
            .expect("model.eval_load");
        let rp = Rope::new(64, HD, HD, 10_000.0).expect("rope");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &Default::default())
            .await
            .expect("rope 表");

        let ids: [f32; 4] = [5.0, 9.0, 3.0, 7.0];
        let pos_of = |t: usize| TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[t as f32]));  // 契约 pos r2

        // ── 参考:T 次 decode 步,单根 = last_hidden(缓冲套 A)──
        let (kvs_a, gdns_a) = model_bufs(&mut gpu, 1, 3).await;
        let mut ref_hidden: Vec<Vec<f32>> = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for t in 0..t_len {
            let ids_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[ids[t]]));
            let pos_t = pos_of(t);
            let kvs_step: Vec<KvBuffers> = kvs_a
                .iter()
                .map(|kv| KvBuffers {
                    k_cache: kv.k_cache.clone(),
                    v_cache: kv.v_cache.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[t as f32])),  // 契约 slots/kv_lens r2(B6 收敛)
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1, 1], &f32b(&[t as f32 + 1.0])),  // 契约 slots/kv_lens r2(B6 收敛)
                    block_tables: TensorOps::zeros(Dtype::F32, vec![1]), // 哑表(legacy 路径不解引用)
                })
                .collect();
            // 单序列语义:GDN 状态格恒 gdn_slot(=0,与 prefill 一致);
            // 槽随步换 = 每步新序列,状态不携带 —— 与 prefill 不可比
            let gdns_step: Vec<GdnBuffers> = gdns_a
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                })
                .collect();
            let ctx = ForwardCtx::model_decode(1, &pos_t, &kvs_step, &rp, &gdns_step);
            ref_hidden.push(crate::testkit::harvest(&mut gpu, &model.last_hidden(&ids_t, &ctx)).await);
        }

        // ── 被测:一次 prefill 块(缓冲套 B;单根 = last_hidden)──
        let (kvs_b, gdns_b) = model_bufs(&mut gpu, 1, 3).await;
        let ids_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&ids));
        let pos_all = TensorOps::from_host(Dtype::F32, vec![1, t_len], &f32b(&(0..t_len).map(|t| t as f32).collect::<Vec<_>>()));
        let slots_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&(0..t_len).map(|t| t as f32).collect::<Vec<_>>()));
        let lens_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&(0..t_len).map(|t| t as f32 + 1.0).collect::<Vec<_>>()));
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0]));
        let ctx = ForwardCtx::model_prefill(t_len, &pos_all, &kvs_b, &rp, &gdns_b, &slots_all, &lens_all, &gdn_slot, None, 0, None);
        let pre_hidden = crate::testkit::harvest(&mut gpu, &model.last_hidden(&ids_all, &ctx)).await;
        for t in 0..t_len {
            let d: f32 = pre_hidden[t * HIDDEN..(t + 1) * HIDDEN]
                .iter()
                .zip(&ref_hidden[t])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(d < 1e-4, "prefill last_hidden 行{t} maxdiff={d}");
        }

        // ── logits 根(缓冲套 C)+ host 锚 ──
        let (kvs_c, gdns_c) = model_bufs(&mut gpu, 1, 3).await;
        let ctx_c = ForwardCtx::model_prefill(t_len, &pos_all, &kvs_c, &rp, &gdns_c, &slots_all, &lens_all, &gdn_slot, None, 0, None);
        let got_all = crate::testkit::harvest(&mut gpu, &model.forward(&ids_all, &ctx_c)).await;
        let mut host2 = HostModel::new(checkpoint_src());
        for t in 0..t_len {
            let want = host2.step2(ids[t], t as f32, t, t + 1, 0);
            eprintln!("[dbg] logits 行{t}: prefill {:?} host {:?}", &got_all[t * VOCAB..(t + 1) * VOCAB], want);
            assert_close(
                &got_all[t * VOCAB..(t + 1) * VOCAB],
                &want,
                1e-3,
                &format!("model-prefill 行{t}"),
            );
        }
        gpu.close().await.expect("关机");
    }

}
