//! 引擎面:构造(资源绑定,零执行)→ ModelLoader(模型装载)→
//! run(装配 GraphPlan + warmup,进入执行态)→ submit/pump(turn 流)。
//!
//! 生命周期(用户裁决 2026-09-26):
//! 1. [`Engine::new`] / [`Engine::on`] —— 构造:设备绑定 + 容量参数,
//!    **不执行**;
//! 2. [`Engine::loader`] —— 面向上层的 [`ModelLoader`]:模型(权重 +
//!    tokenizer + rope)按 spec 声明装载;
//! 3. [`Engine::run`] —— 模型交引擎:状态块/槽/GraphPlan 组建 + warmup,
//!    返回执行态 [`RunningEngine`](仍不跑任何 turn);
//! 4. [`RunningEngine::submit`] —— 提交 turn(非阻塞入队);
//!    [`RunningEngine::pump`] —— 推进到下一个事件点(`Idle` = 队列排空)。
//!
//! 依赖分工:models 主依赖零后端(纯声明层);**engine 主依赖 owl-cuda**
//! —— 绑定设备本就是引擎职责,`Engine::on` 保留任意 DeviceClient 注入
//! (CPU 测试/自定义后端)。
//!
//! 并发:M0.5 单槽串行(一次一个 turn 占 0 号槽;GDN 状态 per-turn 零化,
//! KV 由「kv_len 窗口 + 先写后打分」语义天然隔离——本 turn 打分的槽位
//! 全部由本 turn 写过);真并发随 M2 batching 换入,submit/pump 形状不变。

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use owl_cuda::{DeviceSelector, GpuClient};
use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_models::interpreters::eval_ops;
use owl_models::layers::gdn::GdnBuffers;
use owl_models::layers::rope::Rope;
use owl_models::model::{Model, ModelSpec};
use owl_models::module::{ForwardCtx, KvBuffers, Module};
use owl_models::specs::{load_0_8b, load_tokenizer, qwen3_5_0_8b};
use owl_models::tokenizer::Tokenizer;
use owl_models::{TensorOps};

use crate::graph_plan::{GraphPlan, GraphPlanDesc, InputSlot, OutputSlot, PlanCtx};
use crate::turn::{TurnEvent, TurnSpec};

type Result<T> = std::result::Result<T, ModelError>;

// ============================================================================
// §1 构造参数 / 已装载模型 / ModelLoader
// ============================================================================

/// 引擎构造参数(纯资源面,零模型语义)
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// 设备序(GpuClient Ordinal;UUID 钉卡 A2.6 挂账)
    pub device_ordinal: usize,
    /// 单 turn token 预算 = KV 直排槽位(prompt + 生成 ≤ 此值)
    pub max_seq_tokens: usize,
    /// prefill 块长(token/块;W1 块式喂入)。约束:块内 kv_len 峰值
    /// (块基址 + 块长)≤ OWL_MAX_KV=256(attention 窗寄存器上限);
    /// 默认 128。长 ctx 批核另案(§五.3)
    pub prefill_chunk: usize,
}

/// 已装载模型(权重在设备 + tokenizer 解析 + rope 表;[`Engine::run`] 的输入)
pub struct LoadedModel {
    pub(crate) model: Arc<Model>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) rope: Rope,
    pub(crate) spec: ModelSpec,
}

/// 面向上层的模型加载器(经引擎的 face;spec 声明驱动)
pub struct ModelLoader<'a, D: DeviceClient> {
    face: &'a mut D,
}

impl<D: DeviceClient + 'static> ModelLoader<'_, D> {
    /// Qwen3.5-0.8B(维度/键约定/分词器事实全在 specs/qwen35.rs 声明;
    /// 未来 `load(dir)` 按 config.json 分发,挂账)
    pub async fn load_qwen35_0_8b(&mut self, dir: &Path) -> Result<LoadedModel> {
        let spec = qwen3_5_0_8b();
        let model = Arc::new(load_0_8b(dir, self.face).await?);
        let tokenizer = load_tokenizer(dir)?;
        let rope = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        let ctx = owl_models::module::LoaderCtx { dtype: spec.dtype, shard: 1 };
        owl_models::interpreters::eval_load(&rope, self.face, &rope.tables(), &ctx).await?;
        Ok(LoadedModel { model, tokenizer, rope, spec })
    }
}

// ============================================================================
// §2 Engine:构造(资源绑定,零执行)
// ============================================================================

pub struct Engine<D: DeviceClient> {
    face: D,
    cfg: EngineConfig,
}

impl Engine<GpuClient> {
    /// 常规构造:按 cfg 绑定 GPU(ordinal)
    pub fn new(cfg: EngineConfig) -> Result<Self> {
        let face = GpuClient::spawn(DeviceSelector::Ordinal(cfg.device_ordinal))
            .map_err(|e| ModelError::Msg(format!("engine: gpu 绑定失败 {e:?}")))?;
        Self::on(cfg, face)
    }
}

impl<D: DeviceClient> Engine<D> {
    /// 注入 face(CPU 测试/自定义后端;唯一构造机制,new 是其 GPU 特化)
    pub fn on(cfg: EngineConfig, face: D) -> Result<Self> {
        Ok(Self { face, cfg })
    }

    /// 模型加载器(借用引擎 face;装载即校验,错误在 load 边界落地)
    pub fn loader(&mut self) -> ModelLoader<'_, D> {
        ModelLoader { face: &mut self.face }
    }

    /// 模型交引擎:状态块/槽/Session 组建 + warmup —— 仍不跑任何 turn,
    /// 返回执行态引擎(请求入口)。
    pub async fn run(mut self, loaded: LoadedModel) -> Result<RunningEngine<D>> {
        let s = self.cfg.max_seq_tokens;
        let (_, hkv, hd) = loaded.spec.full_heads;
        let (nk, hk, nv, hv) = loaded.spec.gdn_heads;
        let n_full = loaded.spec.layer_types.iter().filter(|&&f| f).count();
        let n_gdn = loaded.spec.layer_types.len() - n_full;

        // 状态块(零初始化;GDN 段每 turn 开始重置)。
        // dtype 定盘(F5 整模切换,战役 §二):KV cache f16;GDN state 恒 f32
        // (三方先例 + 旧世界律);slots/kv_lens f32 契约 5
        let mut kvs_b: Vec<KvBlocks> = Vec::new();
        for _ in 0..n_full {
            let k_cache = zero_block_dt(&mut self.face, s * hkv * hd, loaded.spec.dtype).await?;
            let v_cache = zero_block_dt(&mut self.face, s * hkv * hd, loaded.spec.dtype).await?;
            kvs_b.push(KvBlocks { k_cache, v_cache });
        }
        let mut gdns_b: Vec<GdnBlocks> = Vec::new();
        for _ in 0..n_gdn {
            let conv_q = zero_block(&mut self.face, s * nk * hk * 3).await?;
            let conv_k = zero_block(&mut self.face, s * nv * hv * 3).await?;
            let conv_v = zero_block(&mut self.face, s * nv * hv * 3).await?;
            let rec = zero_block(&mut self.face, s * nv * hk * hv).await?;
            gdns_b.push(GdnBlocks { conv_q, conv_k, conv_v, rec });
        }
        let kv_dt = loaded.spec.dtype;
        let dims = ModelDims {
            hkv, hd, nk, hk, nv, hv, n_full, n_gdn,
            hidden: loaded.spec.hidden, dtype: loaded.spec.dtype,
        };
        let kv_leaf = |b: &KvBlocks| KvBuffers {
            k_cache: block_leaf_dt(&(b.k_cache.0), vec![s, hkv, hd], kv_dt),
            v_cache: block_leaf_dt(&(b.v_cache.0), vec![s, hkv, hd], kv_dt),
            slots: TensorOps::zeros(Dtype::F32, vec![1]),
            kv_lens: TensorOps::zeros(Dtype::F32, vec![1]),
        };
        let gdn_leaf = |b: &GdnBlocks| GdnBuffers {
            conv_q: block_leaf(&(b.conv_q.0), vec![s, nk * hk, 3]),
            conv_k: block_leaf(&(b.conv_k.0), vec![s, nv * hv, 3]),
            conv_v: block_leaf(&(b.conv_v.0), vec![s, nv * hv, 3]),
            rec: block_leaf(&(b.rec.0), vec![s, nv, hk, hv]),
            slots: TensorOps::zeros(Dtype::F32, vec![1]),
        };
        let kvs: Vec<KvBuffers> = kvs_b.iter().map(kv_leaf).collect();
        let gdns: Vec<GdnBuffers> = gdns_b.iter().map(gdn_leaf).collect();

        // Session 闭包:槽 → 整模单树(状态句柄捕获;模型 Arc 共享)。
        // 捕获期禁 Htod:常量标量走持久槽;KV/GDN 双槽分立 ——
        // KV 槽 = 本 token 自己的格子(窗 = [slot-kv_len+1, slot],随步进;
        // 2026-09-26 窗口语义探针定谳:slots 恒 0 读块外垃圾行);
        // GDN 槽 = 序列状态格,turn 内恒 0(递推状态按序列累积)。
        let model = Arc::clone(&loaded.model);
        let rp = loaded.rope.clone();
        let vocab = loaded.model.vocab_size();
        let forward = move |sc: &PlanCtx| -> Result<()> {
            let ids = sc.input("frontier")?;
            let pos = sc.input("pos")?;
            let kv_len = sc.input("kv_len")?;
            let kv_slot = sc.input("kv_slot")?;
            let gdn_slot = sc.input("gdn_slot")?;
            let kvs_step: Vec<KvBuffers> = kvs
                .iter()
                .map(|kv| KvBuffers {
                    k_cache: kv.k_cache.clone(),
                    v_cache: kv.v_cache.clone(),
                    slots: kv_slot.clone(),
                    kv_lens: kv_len.clone(),
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdns
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: gdn_slot.clone(),
                })
                .collect();
            let ctx = ForwardCtx::model_decode(1, &pos, &kvs_step, &rp, &gdns_step);
            let tree = model.forward(&ids, &ctx);
            sc.output("logits", &tree)
        };

        let (session, capture_outcome) = GraphPlan::plan(
            self.face,
            GraphPlanDesc {
                inputs: vec![
                    InputSlot::f32("frontier", 1),
                    InputSlot::f32("pos", 1),
                    InputSlot::f32("kv_len", 1),
                    InputSlot::f32("kv_slot", 1),
                    InputSlot::f32("gdn_slot", 1).init(vec![0.0]),
                ],
                outputs: vec![OutputSlot { name: "logits", shape: vec![1, vocab], dtype: loaded.spec.dtype }],
                capture: true,
            },
            forward,
        )
        .await?;
        if let crate::graph_plan::PlanOutcome::EagerFallback { reason } = &capture_outcome {
            eprintln!("[engine] 捕获降级为 eager:{reason}");
        }

        Ok(RunningEngine {
            session,
            tok: loaded.tokenizer,
            cfg: self.cfg,
            capture_outcome,
            kvs_b,
            gdns_b,
            queue: VecDeque::new(),
            active: None,
            next_id: 1,
            model: loaded.model,
            rope: loaded.rope,
            dims,
        })
    }
}

// ============================================================================
// §3 RunningEngine:执行态(请求入口)
// ============================================================================

struct Turn {
    id: u64,
    spec: TurnSpec,
    prompt_ids: Vec<u32>,
}

struct ActiveTurn {
    id: u64,
    prompt_ids: Vec<u32>,
    max_new: usize,
    out: Vec<u32>,
    fed: usize,
    decoded: String,
}

type BlockN = (Bytes, usize); // (块句柄, 元素数;重置分块写用)

struct KvBlocks {
    k_cache: BlockN,
    v_cache: BlockN,
}

struct GdnBlocks {
    conv_q: BlockN,
    conv_k: BlockN,
    conv_v: BlockN,
    rec: BlockN,
}

pub struct RunningEngine<D: DeviceClient> {
    session: GraphPlan<D>,
    tok: Tokenizer,
    cfg: EngineConfig,
    /// 捕获三态结果(Captured = 回放态;EagerFallback = 同闭包直发)
    pub capture_outcome: crate::graph_plan::PlanOutcome,
    #[allow(dead_code)]
    kvs_b: Vec<KvBlocks>,
    gdns_b: Vec<GdnBlocks>,
    queue: VecDeque<Turn>,
    active: Option<ActiveTurn>,
    next_id: u64,
    /// prefill 块式喂入(W1)的模型面:树根构造 + 维度账
    model: Arc<Model>,
    rope: Rope,
    dims: ModelDims,
}

/// prefill 块构造所需维度账(run 期从 spec 提取;Copy 免 Clone 传播)
#[derive(Clone, Copy)]
struct ModelDims {
    hkv: usize,
    hd: usize,
    nk: usize,
    hk: usize,
    nv: usize,
    hv: usize,
    n_full: usize,
    n_gdn: usize,
    hidden: usize,
    dtype: Dtype,
}

impl<D: DeviceClient> RunningEngine<D> {
    /// 提交 turn(非阻塞;入队即返回 id)。预算越界 fail-fast:
    /// prompt + 生成 ≤ max_seq_tokens(= KV 槽位)。
    pub fn submit(&mut self, prompt: impl Into<String>, max_new: usize) -> Result<u64> {
        let prompt = prompt.into();
        let prompt_ids = {
            let wrapped = self.tok.chat_wrap(&prompt);
            self.tok.encode(&wrapped)
        };
        if prompt_ids.is_empty() {
            return Err(ModelError::Msg("submit: 空 prompt".into()));
        }
        if prompt_ids.len() + max_new > self.cfg.max_seq_tokens {
            return Err(ModelError::Msg(format!(
                "submit: prompt {} + 生成 {} 超预算 {}",
                prompt_ids.len(),
                max_new,
                self.cfg.max_seq_tokens
            )));
        }
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(Turn { id, spec: TurnSpec { prompt, max_new }, prompt_ids });
        Ok(id)
    }

    /// 推进到下一个事件点(每调用 = 活跃 turn 的一个采样步,或 turn 的
    /// 启动切换;`Idle` = 队列排空且无活跃,调用方可安全挂起等新提交)。
    pub async fn pump(&mut self) -> Result<TurnEvent> {
        // 无活跃 → 取队首开 turn:重置 GDN 状态 + prefill 全部 prompt +
        // 首采样,一气推进到首个 Token 事件(prompt 中段回 Prefill 事件)
        if self.active.is_none() {
            let Some(turn) = self.queue.pop_front() else {
                return Ok(TurnEvent::Idle);
            };
            let max_new = turn.spec.max_new;
            self.reset_gdn().await?;
            self.active = Some(ActiveTurn {
                id: turn.id,
                prompt_ids: turn.prompt_ids,
                max_new,
                out: Vec::new(),
                fed: 0,
                decoded: String::new(),
            });
        }

        // ── prompt 相位:一次 pump = 一个 chunk(块式 prefill;W1/PF1)──
        {
            // 先取标量(prompt 切片经 &self 借出后即刻拷走,避免与
            // prefill_chunk 的 &mut self 相交)
            let plan = {
                let act = self.active.as_ref().expect("active 已保证");
                if act.fed >= act.prompt_ids.len() {
                    None
                } else {
                    let base = act.fed;
                    let remaining = act.prompt_ids.len() - base;
                    // 块长:配置块长 ∩ 剩余 ∩ 窗约束(块内 kv_len 峰值
                    // base+chunk ≤ OWL_MAX_KV=256,attention 窗寄存器上限)
                    let chunk = remaining
                        .min(self.cfg.prefill_chunk)
                        .min(256usize.saturating_sub(base))
                        .max(1);
                    Some((
                        base,
                        act.id,
                        chunk,
                        base + chunk >= act.prompt_ids.len(),
                        act.prompt_ids[base..base + chunk].to_vec(),
                    ))
                }
            };
            if let Some((base, turn_id, chunk, is_last_prompt, chunk_ids)) = plan {
                let last_row = self
                    .prefill_chunk(&chunk_ids, base, is_last_prompt)
                    .await?;
                    let act = self.active.as_mut().expect("active 已保证");
                act.fed += chunk;
                if !is_last_prompt {
                    // 块落定,无文本产出 —— 不谎报 Idle
                    return Ok(TurnEvent::Prefill {
                        turn: turn_id,
                        fed: act.fed,
                        total: act.prompt_ids.len(),
                    });
                }
                // 末块:末行采样(eos / 预算 / Token)
                return self.sample_and_emit(last_row.expect("末块必有 logits")).await;
            }
        }

        // ── 生成相位:decode 单步(捕获回放)+ 采样 ──
        let (tid, pos) = {
            let act = self.active.as_mut().expect("active 已保证");
            let tid = *act.out.last().expect("生成中");
            (tid, act.fed)
        };
        self.session
            .step(&[
                ("frontier", &[tid as f32]),
                ("pos", &[pos as f32]),
                ("kv_len", &[(pos + 1) as f32]),
                ("kv_slot", &[pos as f32]),
                ("gdn_slot", &[0.0]),
            ])
            .await?;

        let act = self.active.as_mut().expect("active 已保证");
        act.fed += 1;
        let logits = self.session.read_output_f32("logits").await?;
        self.sample_and_emit(logits).await
    }

    /// 采样 + 事件产出(Greedy;host argmax —— 设备采样挂账)。
    /// eos 命中或预算尽 → Completed;否则 Token(delta = 解码文本增量)。
    async fn sample_and_emit(&mut self, logits: Vec<f32>) -> Result<TurnEvent> {
        let turn_id = self.active.as_ref().expect("active 已保证").id;
        let nt = argmax(&logits) as u32;
        if self.tok.is_eos(nt) {
            return self.complete().await;
        }
        let (delta, budget_done) = {
            let act = self.active.as_mut().expect("active 已保证");
            act.out.push(nt);
            (decode_delta(&self.tok, &act.out, &mut act.decoded), act.out.len() >= act.max_new)
        };
        if budget_done {
            return self.complete().await;
        }
        Ok(TurnEvent::Token { turn: turn_id, delta })
    }

    /// 块式 prefill(W1;批P5 契约):ids [T] 从 KV 行 base 起步,
    /// 单序列语义(gdn_slot 恒 0;KV 行随 token 走)。末块走 logits 根
    /// 返回末行;中间块走 last_hidden 根(免 lm_head [T,V] 计算与大
    /// dtoh)返回 None。
    async fn prefill_chunk(
        &mut self,
        ids: &[u32],
        base: usize,
        is_last: bool,
    ) -> Result<Option<Vec<f32>>> {
        let d = self.dims;
        let s = self.cfg.max_seq_tokens;
        let t = ids.len();
        let f32seq = |start: usize, n: usize| -> Vec<u8> {
            (0..n)
                .flat_map(|i| ((start + i) as f32).to_le_bytes())
                .collect()
        };
        let kvs_step: Vec<KvBuffers> = self
            .kvs_b
            .iter()
            .map(|b| KvBuffers {
                k_cache: block_leaf_dt(&b.k_cache.0, vec![s, d.hkv, d.hd], d.dtype),
                v_cache: block_leaf_dt(&b.v_cache.0, vec![s, d.hkv, d.hd], d.dtype),
                slots: TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base, t)),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base + 1, t)),
            })
            .collect();
        let gdns_step: Vec<GdnBuffers> = self
            .gdns_b
            .iter()
            .map(|g| GdnBuffers {
                conv_q: block_leaf(&g.conv_q.0, vec![s, d.nk * d.hk, 3]),
                conv_k: block_leaf(&g.conv_k.0, vec![s, d.nv * d.hv, 3]),
                conv_v: block_leaf(&g.conv_v.0, vec![s, d.nv * d.hv, 3]),
                rec: block_leaf(&g.rec.0, vec![s, d.nv, d.hk, d.hv]),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32seq(0, 1)),
            })
            .collect();
        let ids_t = TensorOps::from_host(
            Dtype::F32,
            vec![t],
            &ids.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect::<Vec<u8>>(),
        );
        let pos_t = TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base, t));
        let slots_t = TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base, t));
        let lens_t = TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base + 1, t));
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32seq(0, 1));
        let ctx = ForwardCtx::model_prefill(
            t, &pos_t, &kvs_step, &self.rope, &gdns_step, &slots_t, &lens_t, &gdn_slot,
        );
        let face = self.session.face_mut();
        let vocab = self.model.vocab_size();
        if is_last {
            let tree = self.model.forward(&ids_t, &ctx);
            let b = eval_ops(tree.step(), face).await?;
            let esz = if d.dtype == Dtype::F16 { 2 } else { 4 };
            let mut buf = vec![0u8; t * vocab * esz];
            face.dtoh(&b, &mut buf).await?;
            let off = (t - 1) * vocab * esz;
            let row = buf[off..off + vocab * esz]
                .chunks_exact(esz)
                .map(|c| {
                    if esz == 2 {
                        half::f16::from_le_bytes([c[0], c[1]]).to_f32()
                    } else {
                        f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    }
                })
                .collect();
            Ok(Some(row))
        } else {
            // 中间块:last_hidden 根(状态推进完整;lm_head 免算)
            let tree = self.model.last_hidden(&ids_t, &ctx);
            let b = eval_ops(tree.step(), face).await?;
            let esz = if d.dtype == Dtype::F16 { 2 } else { 4 };
            let mut buf = vec![0u8; t * d.hidden * esz];
            face.dtoh(&b, &mut buf).await?;
            Ok(None)
        }
    }

    /// 便捷入口:提交一个 turn 并泵到其完成,返回全文
    /// (多 turn 并发场景用 submit + pump 手动驱动)
    pub async fn generate(&mut self, prompt: impl Into<String>, max_new: usize) -> Result<String> {
        let id = self.submit(prompt, max_new)?;
        loop {
            match self.pump().await? {
                TurnEvent::Completed { turn, text } if turn == id => return Ok(text),
                TurnEvent::Idle => {
                    return Err(ModelError::Msg(format!("generate: turn {id} 队列丢失")))
                }
                _ => {}
            }
        }
    }

    async fn complete(&mut self) -> Result<TurnEvent> {
        let act = self.active.take().expect("active");
        let text = self.tok.decode(&act.out);
        Ok(TurnEvent::Completed { turn: act.id, text })
    }

    /// GDN 状态 per-turn 零化(conv 三段 + recurrent;分块写免 host 大向
    /// 量 —— rec 单层 16M 元素)。KV 不重置:kv_len 窗口 + 先写后打分
    /// 语义下,本 turn 打分的槽位全部由本 turn 写过。
    /// GDN 状态重置(turn-open;新序列零态语义)。设备侧 memset
    /// (2026-09-26 定谳:host 往返清零 1.2GB = 14s/turn,黑洞实测)
    async fn reset_gdn(&mut self) -> Result<()> {
        for g in &self.gdns_b {
            let blocks = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (bn, bytes) in blocks {
                self.session
                    .face_mut()
                    .memset_zero(bn, bytes * 4)
                    .await?;
            }
        }
        self.session.face_mut().sync().await?;
        Ok(())
    }
}

// ============================================================================
// §4 小件
// ============================================================================

/// 按 dtype 清零分配(KV 池;f16 = 2B/元素字节口径)
async fn zero_block_dt<D: DeviceClient>(face: &mut D, n: usize, dt: Dtype) -> Result<BlockN> {
    match dt {
        Dtype::F16 => {
            let bytes = vec![0u8; n * 2];
            let b = eval_ops(TensorOps::from_host(Dtype::F16, vec![n], &bytes).step(), face).await?;
            Ok((Bytes::new(b.id, 0), n))
        }
        _ => zero_block(face, n).await,
    }
}

fn block_leaf_dt(b: &Bytes, shape: Vec<usize>, dt: Dtype) -> TensorOps {
    match dt {
        Dtype::F16 => TensorOps::of_block(b.id, Dtype::F16, shape),
        _ => TensorOps::of_block(b.id, Dtype::F32, shape),
    }
}

async fn zero_block<D: DeviceClient>(face: &mut D, n: usize) -> Result<BlockN> {
    let bytes: Vec<u8> = vec![0.0f32; n].iter().flat_map(|f| f.to_le_bytes()).collect();
    let b = eval_ops(TensorOps::from_host(Dtype::F32, vec![n], &bytes).step(), face).await?;
    Ok((Bytes::new(b.id, 0), n))
}

fn block_leaf(b: &Bytes, shape: Vec<usize>) -> TensorOps {
    TensorOps::of_block(b.id, Dtype::F32, shape)
}

/// 块零化(分块 write_block;1M 元素 = 4MB/笔)

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a })
        .0
}

/// 增量解码:全量重解取后缀差分(byte-level BPE 多字节字符跨 token 兜底)
fn decode_delta(tok: &Tokenizer, out: &[u32], decoded: &mut String) -> String {
    let full = tok.decode(out);
    let delta = full[decoded.len()..].to_string();
    *decoded = full;
    delta
}

// ============================================================================
// §5 测试(CPU:构造/装载门禁;GPU 门控:双 turn 生命周期)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use owl_cpu::CpuFace;

    fn asset_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../models/assets/Qwen3.5-0.8B")
    }

    fn gpu_ordinal() -> Option<usize> {
        std::env::var("OWL_TEST_DEVICE").ok().and_then(|v| v.parse().ok())
    }

    /// CPU:构造(零执行)+ ModelLoader 装载门禁(权重/tokenizer 解析)
    #[tokio::test]
    async fn cpu_construct_and_load() {
        let mut engine = Engine::on(
            EngineConfig { device_ordinal: 0, max_seq_tokens: 64, prefill_chunk: 16 },
            CpuFace::new(),
        )
        .expect("engine 构造");
        let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
        assert_eq!(loaded.model.layers.len(), 24);
        assert!(!loaded.tokenizer.encode("你好").is_empty(), "tokenizer 活性");
    }

    /// GPU 门控:双 turn 生命周期 —— submit 入队 / pump 事件流 / per-turn
    /// GDN 重置(第二个 turn 在脏状态后仍须产出连贯文本)。
    #[tokio::test]
    async fn gpu_two_turns_lifecycle() {
        let Some(ordinal) = gpu_ordinal() else {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        };
        let dir = asset_dir();
        let mut engine =
            Engine::new(EngineConfig { device_ordinal: ordinal, max_seq_tokens: 64, prefill_chunk: 16 })
                .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        assert!(
            matches!(running.capture_outcome, crate::graph_plan::PlanOutcome::Captured),
            "真模型捕获应成功(回放态),得 {:?}",
            running.capture_outcome
        );

        let t1 = running.submit("Hello, who are you?", 16).expect("t1");
        let t2 = running.submit("用一句话介绍长城。", 16).expect("t2");
        let (mut ttft, mut t0) = (None, std::time::Instant::now());
        let mut done = Vec::new();
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Idle => break,
                TurnEvent::Prefill { turn, fed, total } => {
                    eprintln!("[ev] prefill t{turn} {fed}/{total} (+{:?})", t0.elapsed());
                }
                TurnEvent::Token { turn, .. } if ttft.is_none() => {
                    ttft = Some(t0.elapsed());
                    eprintln!("[ttft] 首 Token(turn {turn}): {:?}", ttft.unwrap());
                }
                TurnEvent::Completed { turn, text } => {
                    eprintln!("[ttft] turn {turn} 完成(累计 {:?})", t0.elapsed());
                    done.push((turn, text))
                }
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                _ => {}
            }
        }
        assert_eq!(done.len(), 2, "两 turn 都完成");
        assert!(done.iter().any(|(t, _)| *t == t1));
        assert!(done.iter().any(|(t, _)| *t == t2));
        for (t, text) in &done {
            eprintln!("[test] t{t}: {text}");
            assert!(!text.trim().is_empty(), "t{t} 产出非空");
        }
    }
}
