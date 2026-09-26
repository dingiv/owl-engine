//! 引擎面:构造(资源绑定,零执行)→ ModelLoader(模型装载)→
//! run(装配 Session + warmup,进入执行态)→ submit/pump(turn 流)。
//!
//! 生命周期(用户裁决 2026-09-26):
//! 1. [`Engine::new`] / [`Engine::on`] —— 构造:设备绑定 + 容量参数,
//!    **不执行**;
//! 2. [`Engine::loader`] —— 面向上层的 [`ModelLoader`]:模型(权重 +
//!    tokenizer + rope)按 spec 声明装载;
//! 3. [`Engine::run`] —— 模型交引擎:状态块/槽/Session 组建 + warmup,
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

use crate::session::{InputSlot, OutputSlot, Session, SessionDesc, StepCtx};
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

impl<D: DeviceClient> ModelLoader<'_, D> {
    /// Qwen3.5-0.8B(维度/键约定/分词器事实全在 specs/qwen35.rs 声明;
    /// 未来 `load(dir)` 按 config.json 分发,挂账)
    pub async fn load_qwen35_0_8b(&mut self, dir: &Path) -> Result<LoadedModel> {
        let spec = qwen3_5_0_8b();
        let model = Arc::new(load_0_8b(dir, self.face).await?);
        let tokenizer = load_tokenizer(dir)?;
        let rope = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        owl_models::interpreters::eval_load(&rope, self.face, &rope.tables(), &Default::default())
            .await?;
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

        // 状态块(零初始化;GDN 段每 turn 开始重置)
        let mut kvs_b: Vec<KvBlocks> = Vec::new();
        for _ in 0..n_full {
            let k_cache = zero_block(&mut self.face, s * hkv * hd).await?;
            let v_cache = zero_block(&mut self.face, s * hkv * hd).await?;
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
        let kv_leaf = |b: &KvBlocks| KvBuffers {
            k_cache: block_leaf(&(b.k_cache.0), vec![s, hkv, hd]),
            v_cache: block_leaf(&(b.v_cache.0), vec![s, hkv, hd]),
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
        // 捕获期禁 Htod:所有常量标量(slots 0 号槽)走持久槽 slot0。
        let model = Arc::clone(&loaded.model);
        let rp = loaded.rope;
        let vocab = loaded.model.vocab_size();
        let forward = move |sc: &StepCtx| -> Result<()> {
            let ids = sc.input("frontier")?;
            let pos = sc.input("pos")?;
            let kv_len = sc.input("kv_len")?;
            let slot0 = sc.input("slot0")?;
            let kvs_step: Vec<KvBuffers> = kvs
                .iter()
                .map(|kv| KvBuffers {
                    k_cache: kv.k_cache.clone(),
                    v_cache: kv.v_cache.clone(),
                    slots: slot0.clone(),
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
                    slots: slot0.clone(),
                })
                .collect();
            let ctx = ForwardCtx::model_decode(1, &pos, &kvs_step, &rp, &gdns_step);
            let tree = model.forward(&ids, &ctx);
            sc.output("logits", &tree)
        };

        let (session, capture_outcome) = Session::plan(
            self.face,
            SessionDesc {
                inputs: vec![
                    InputSlot::f32("frontier", 1),
                    InputSlot::f32("pos", 1),
                    InputSlot::f32("kv_len", 1),
                    InputSlot::f32("slot0", 1).init(vec![0.0]),
                ],
                outputs: vec![OutputSlot::f32("logits", &[1, vocab])],
                capture: true,
            },
            forward,
        )
        .await?;
        if let crate::session::PlanOutcome::EagerFallback { reason } = &capture_outcome {
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
    session: Session<D>,
    tok: Tokenizer,
    cfg: EngineConfig,
    /// 捕获三态结果(Captured = 回放态;EagerFallback = 同闭包直发)
    pub capture_outcome: crate::session::PlanOutcome,
    #[allow(dead_code)]
    kvs_b: Vec<KvBlocks>,
    gdns_b: Vec<GdnBlocks>,
    queue: VecDeque<Turn>,
    active: Option<ActiveTurn>,
    next_id: u64,
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

        // 单步:喂一个 token(prompt 相位 teacher-forcing;生成相位喂上次采样)
        let (tid, pos, is_last_prompt, turn_id) = {
            let act = self.active.as_mut().expect("active 已保证");
            let tid = if act.fed < act.prompt_ids.len() {
                act.prompt_ids[act.fed]
            } else {
                *act.out.last().expect("生成中")
            };
            (tid, act.fed, act.fed + 1 >= act.prompt_ids.len(), act.id)
        };
        self.session
            .step(&[
                ("frontier", &[tid as f32]),
                ("pos", &[pos as f32]),
                ("kv_len", &[(pos + 1) as f32]),
            ])
            .await?;

        let act = self.active.as_mut().expect("active 已保证");
        act.fed += 1;
        if !is_last_prompt {
            // prompt 中段:状态推进一个 step,无文本产出 —— 不谎报 Idle
            return Ok(TurnEvent::Prefill {
                turn: turn_id,
                fed: act.fed,
                total: act.prompt_ids.len(),
            });
        }

        // 采样(Greedy;host argmax —— 设备采样挂账)
        let logits = self.session.read_output_f32("logits").await?;
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
    async fn reset_gdn(&mut self) -> Result<()> {
        for g in &self.gdns_b {
            let blocks = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (bn, elems) in blocks {
                zero_fill(self.session.face_mut(), bn, elems).await?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// §4 小件
// ============================================================================

async fn zero_block<D: DeviceClient>(face: &mut D, n: usize) -> Result<BlockN> {
    let bytes: Vec<u8> = vec![0.0f32; n].iter().flat_map(|f| f.to_le_bytes()).collect();
    let b = eval_ops(TensorOps::from_host(Dtype::F32, vec![n], &bytes).step(), face).await?;
    Ok((Bytes::new(b.id, 0), n))
}

fn block_leaf(b: &Bytes, shape: Vec<usize>) -> TensorOps {
    TensorOps::of_block(b.id, Dtype::F32, shape)
}

/// 块零化(分块 write_block;1M 元素 = 4MB/笔)
async fn zero_fill<D: DeviceClient>(face: &mut D, b: &Bytes, elems: usize) -> Result<()> {
    const CHUNK: usize = 1 << 20;
    let zeros = vec![0.0f32; CHUNK.min(elems)];
    let mut off = 0usize;
    while off < elems {
        let n = CHUNK.min(elems - off);
        face.write_block_f32(b, off, &zeros[..n]).await?;
        off += n;
    }
    Ok(())
}

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
            EngineConfig { device_ordinal: 0, max_seq_tokens: 64 },
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
            Engine::new(EngineConfig { device_ordinal: ordinal, max_seq_tokens: 64 })
                .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        assert!(
            matches!(running.capture_outcome, crate::session::PlanOutcome::Captured),
            "真模型捕获应成功(回放态),得 {:?}",
            running.capture_outcome
        );

        let t1 = running.submit("Hello, who are you?", 16).expect("t1");
        let t2 = running.submit("用一句话介绍长城。", 16).expect("t2");
        let mut done = Vec::new();
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Idle => break,
                TurnEvent::Completed { turn, text } => done.push((turn, text)),
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
