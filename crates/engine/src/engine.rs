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
//!
//! **会话接线(S0/S1,2026-09-26)**:SessionTable 入引擎 ——
//! `submit_session(Some(id), …)` = 连续会话(同会话跳过 GDN 重置,
//! 只 prefill `cached_len..` 增量段;token 账前缀守卫失配回退全量);
//! `submit(None)` = 临时会话终了即焚,行为同 M0.5(每 turn 全量重算)。

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use owl_cuda::{DeviceSelector, GpuClient};
use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use crate::blocks::BlockManager;
use crate::graph_plan::f32b;
use owl_models::interpreters::{eval_ops, eval_ops_scoped};
use owl_models::layers::gdn::GdnBuffers;
use owl_models::layers::rope::Rope;
use owl_models::model::{Model, ModelSpec};
use owl_models::module::{ForwardCtx, KvBuffers, Module};
use owl_models::specs::{load_0_8b, load_tokenizer, qwen3_5_0_8b};
use owl_models::tokenizer::Tokenizer;
use owl_models::{TensorOps};

use crate::graph_plan::{GraphPlan, GraphPlanDesc, InputSlot, OutputSlot, PlanCtx};
use crate::session::SessionTable;
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
        let t_run = std::time::Instant::now();
        let s = self.cfg.max_seq_tokens;
        let (_, hkv, hd) = loaded.spec.full_heads;
        let (nk, hk, nv, hv) = loaded.spec.gdn_heads;
        let n_full = loaded.spec.layer_types.iter().filter(|&&f| f).count();
        let n_gdn = loaded.spec.layer_types.len() - n_full;

        // 状态块(零初始化;GDN 段每 turn 开始重置)。
        // dtype 定盘(F5 整模切换,战役 §二):KV cache f16;GDN state 恒 f32
        // (三方先例 + 旧世界律);slots/kv_lens f32 契约 5。
        // KV 布局由 kv_paged_policy(dtype) 分发(REQ-HW-01 表驱动):
        //   Some = paged classic(kc [nb,Hkv,hd/x,page,x] / vc [nb,Hkv,hd,page];
        //   恒等块表:物理块 = 逻辑块,物理槽 = pos);
        //   None = legacy token-major 池 + naive 回退(层同表驱动,两边一致)
        let pol = owl_models::module::kv_paged_policy(loaded.spec.dtype);
        let (page, x, nb, paged) = match &pol {
            Some(p) => (p.page, p.x, (s + p.page - 1) / p.page, true),
            None => (s, 1usize, 1usize, false),
        };
        // E2b 块池容量:默认 = 2 × 单会话容量(两会话满载共存);
        // OWL_POOL_TOKENS 可覆写(token 口径,向上取整到页)
        let pool_tokens = std::env::var("OWL_POOL_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2 * s);
        let nb = paged.then(|| pool_tokens.div_ceil(page)).unwrap_or(nb).max(nb);
        let blocks_m = BlockManager::new(nb, page);
        let mut kvs_b: Vec<KvBlocks> = Vec::new();
        for _ in 0..n_full {
            let k_cache = zero_block_dt(&mut self.face, nb * hkv * hd * page, loaded.spec.dtype).await?;
            let v_cache = zero_block_dt(&mut self.face, nb * hkv * hd * page, loaded.spec.dtype).await?;
            kvs_b.push(KvBlocks { k_cache, v_cache });
        }
        // 块表持久块(E2b):内容 = 活跃会话块链,turn 切换/增长时重写
        // (write_bt);legacy = 哑表
        let bt_b = if paged {
            eval_ops(
                TensorOps::zeros(Dtype::F32, vec![1, nb]).step(),
                &mut self.face,
            )
            .await?
        } else {
            eval_ops(TensorOps::zeros(Dtype::F32, vec![1]).step(), &mut self.face).await?
        };
        let bt_shape = if paged { vec![1, nb] } else { vec![1] };
        let bt_leaf = TensorOps::of_block(bt_b.id, Dtype::F32, bt_shape.clone());
        let mut gdns_b: Vec<GdnBlocks> = Vec::new();
        for _ in 0..n_gdn {
            let conv_q = zero_block(&mut self.face, GDN_SLOTS * nk * hk * 3).await?;
            let conv_k = zero_block(&mut self.face, GDN_SLOTS * nv * hv * 3).await?;
            let conv_v = zero_block(&mut self.face, GDN_SLOTS * nv * hv * 3).await?;
            let rec = zero_block(&mut self.face, GDN_SLOTS * nv * hk * hv).await?;
            gdns_b.push(GdnBlocks { conv_q, conv_k, conv_v, rec });
        }
        let kv_dt = loaded.spec.dtype;
        eprintln!(
            "[boot] 状态块分配 {:.2}s(kv f16 ×{} / gdn ×{},槽位 {})",
            t_run.elapsed().as_secs_f32(),
            kvs_b.len(),
            gdns_b.len(),
            s
        );
        let dims = ModelDims {
            hkv, hd, nk, hk, nv, hv,
            hidden: loaded.spec.hidden, dtype: loaded.spec.dtype,
        };
        let k_shape = if paged { vec![nb, hkv, hd / x, page, x] } else { vec![s, hkv, hd] };
        let v_shape = if paged { vec![nb, hkv, hd, page] } else { vec![s, hkv, hd] };
        let kv_leaf = |b: &KvBlocks, bt: &TensorOps| KvBuffers {
            k_cache: block_leaf_dt(&(b.k_cache.0), k_shape.clone(), kv_dt),
            v_cache: block_leaf_dt(&(b.v_cache.0), v_shape.clone(), kv_dt),
            slots: TensorOps::zeros(Dtype::F32, vec![1]),
            kv_lens: TensorOps::zeros(Dtype::F32, vec![1]),
            block_tables: bt.clone(),
        };
        let gdn_leaf = |b: &GdnBlocks| GdnBuffers {
            conv_q: block_leaf(&(b.conv_q.0), vec![GDN_SLOTS, nk * hk, 3]),
            conv_k: block_leaf(&(b.conv_k.0), vec![GDN_SLOTS, nv * hv, 3]),
            conv_v: block_leaf(&(b.conv_v.0), vec![GDN_SLOTS, nv * hv, 3]),
            rec: block_leaf(&(b.rec.0), vec![GDN_SLOTS, nv, hk, hv]),
            slots: TensorOps::zeros(Dtype::F32, vec![1]),
        };
        let kvs: Vec<KvBuffers> = kvs_b.iter().map(|b| kv_leaf(b, &bt_leaf)).collect();
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
        eprintln!("[boot] GraphPlan plan(warmup+捕获) {:.2}s", t_run.elapsed().as_secs_f32());

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
            sessions: SessionTable::new(),
            model: loaded.model,
            rope: loaded.rope,
            dims,
            page,
            nb,
            blocks_m,
            bt: bt_b,
            paged,
            x,
        })
    }
}

// ============================================================================
// §3 RunningEngine:执行态(请求入口)
// ============================================================================

struct Turn {
    id: u64,
    /// 归属会话(已解析 id;None 提交在 submit 时已建临时账)
    session_id: u64,
    /// 临时会话(终了即焚;行为同 M0.5 每 turn 全量重算)
    ephemeral: bool,
    spec: TurnSpec,
    prompt_ids: Vec<u32>,
}

struct ActiveTurn {
    id: u64,
    session_id: u64,
    ephemeral: bool,
    /// GDN 状态格号(会话绑定;多会话各清各格、各续各态)
    gdn_slot: usize,
    prompt_ids: Vec<u32>,
    max_new: usize,
    out: Vec<u32>,
    /// KV/prompt 推进指针(会话 turn 从 `cached_len` 起步 = 增量 prefill)
    fed: usize,
    decoded: String,
}

type BlockN = (Bytes, usize); // (块句柄, 元素数;重置分块写用)

struct KvBlocks {
    k_cache: BlockN,
    v_cache: BlockN,
}

/// GDN 状态格容量(**与会话数绑定,与 max_seq_tokens 解耦**)。
/// 格语义 = 每会话一格(SessionTable 分配/释放);按位分配是历史包袱:
/// s=4096 时 rec(1 MiB/格/层 × 18 层)将达 72 GB,而真实需求 = 格数。
/// 8 = 覆盖需求并发上限(§二 并发 ≤8);会话 close 即释放格号。
const GDN_SLOTS: usize = 8;

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
    /// 会话账(S0):同会话跨 turn 复用 KV/GDN,增量 prefill
    sessions: SessionTable,
    /// prefill 块式喂入(W1)的模型面:树根构造 + 维度账
    model: Arc<Model>,
    rope: Rope,
    dims: ModelDims,
    /// paged 池几何(页;块数)+ 恒等块表(E1 接线;legacy 模式 page=nb=1)
    page: usize,
    nb: usize,
    /// KV 物理块账房(E2b;块链按会话分派,池内 ref 计数)
    blocks_m: BlockManager,
    bt: Bytes,
    paged: bool,
    x: usize,
}

impl<D: DeviceClient> RunningEngine<D> {
    /// KV 池形状(模式相关;与层分派同源策略)
    fn k_shape(&self) -> Vec<usize> {
        let d = &self.dims;
        if self.paged {
            vec![self.nb, d.hkv, d.hd / self.x, self.page, self.x]
        } else {
            vec![self.nb, d.hkv, d.hd] // legacy:nb=1,page=容量
        }
    }
    fn v_shape(&self) -> Vec<usize> {
        let d = &self.dims;
        if self.paged {
            vec![self.nb, d.hkv, d.hd, self.page]
        } else {
            vec![self.nb, d.hkv, d.hd]
        }
    }
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
    hidden: usize,
    dtype: Dtype,
}

impl<D: DeviceClient> RunningEngine<D> {
    /// 提交 turn(临时会话:每 turn 独立,终了即焚;行为同 M0.5)。
    /// 预算越界 fail-fast:prompt + 生成 ≤ max_seq_tokens(= KV 槽位)。
    pub fn submit(&mut self, prompt: impl Into<String>, max_new: usize) -> Result<u64> {
        self.submit_session(None, prompt, max_new).map(|(_, t)| t)
    }

    /// 会话化提交(S0):`session = None` 自动开临时会话;`Some(id)` =
    /// 连续会话 —— prompt 语义为**全量重发**(历史 + 新消息,客户端保证
    /// 前缀一致):引擎按 token 账前缀守卫,命中则跳过 GDN 重置、只
    /// prefill `cached_len..` 增量段;失配回退全量重算(S1,正确性优先)。
    /// 返回 (session_id, turn_id)。
    pub fn submit_session(
        &mut self,
        session: impl Into<Option<u64>>,
        prompt: impl Into<String>,
        max_new: usize,
    ) -> Result<(u64, u64)> {
        let session = session.into();
        let prompt = prompt.into();
        let prompt_ids = {
            let wrapped = self.tok.chat_wrap(&prompt);
            self.tok.encode(&wrapped)
        };
        if prompt_ids.is_empty() {
            return Err(ModelError::Msg("submit: 空 prompt".into()));
        }
        // 预算(会话感知):命中账本的 turn 按 max(prompt, cached) 记账
        // (增量段写 `[cached, prompt)` 行,生成续写其后;失配回退全量时
        // prompt 为准,同一上界)
        let cached_hit = session
            .and_then(|id| self.sessions.get(id).map(|s| (s.cached_len, s.guard(&prompt_ids))))
            .filter(|&(_, guard)| guard)
            .map(|(cached, _)| cached)
            .unwrap_or(0);
        let need = prompt_ids.len().max(cached_hit) + max_new;
        if need > self.cfg.max_seq_tokens {
            return Err(ModelError::Msg(format!(
                "submit: prompt {} + 生成 {} 超预算 {}",
                prompt_ids.len(),
                max_new,
                self.cfg.max_seq_tokens
            )));
        }
        let (session_id, ephemeral) = match session {
            Some(id) => (self.sessions.get_or_create(Some(id), GDN_SLOTS)?.0, false),
            None => (self.sessions.get_or_create(None, GDN_SLOTS)?.0, true),
        };
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(Turn {
            id,
            session_id,
            ephemeral,
            spec: TurnSpec { prompt, max_new },
            prompt_ids,
        });
        Ok((session_id, id))
    }

    /// 显式关会话(客户端声明不再续;账本清票,块链归还账房)
    pub fn close_session(&mut self, id: u64) -> Result<()> {
        if let Some(s) = self.sessions.get_mut(id) {
            let mut table = std::mem::take(&mut s.block_table);
            self.blocks_m.release_table(&mut table);
        }
        self.sessions.close(id)
    }

    /// 会话账观测(已入 KV 的 token 数;None = 会话不存在)
    pub fn session_len(&self, id: u64) -> Option<usize> {
        self.sessions.get(id).map(|s| s.cached_len)
    }

    /// 推进到下一个事件点(每调用 = 活跃 turn 的一个采样步,或 turn 的
    /// 启动切换;`Idle` = 队列排空且无活跃,调用方可安全挂起等新提交)。
    pub async fn pump(&mut self) -> Result<TurnEvent> {
        // 无活跃 → 取队首开 turn:会话守卫(命中 = 跳过 GDN 重置 +
        // 增量起步;失配/新会话 = 重置 + 全量),再推进到首个 Token 事件
        if self.active.is_none() {
            let Some(turn) = self.queue.pop_front() else {
                return Ok(TurnEvent::Idle);
            };
            let (cached_len, gdn_slot) = {
                let s = self
                    .sessions
                    .get_mut(turn.session_id)
                    .expect("submit 已建会话账");
                if s.guard(&turn.prompt_ids) {
                    (s.cached_len, s.gdn_slot)
                } else {
                    // S1 回退:改写历史/模板漂移 → 块链全还 + 清账,全量重算
                    let mut table = std::mem::take(&mut s.block_table);
                    self.blocks_m.release_table(&mut table);
                    s.reset();
                    (0, s.gdn_slot)
                }
            };
            if cached_len == 0 {
                self.reset_gdn(gdn_slot).await?;
            }
            // E2b:块表确保覆盖 prompt(增量 turn 块链已存,只长新增段)
            {
                let s = self
                    .sessions
                    .get_mut(turn.session_id)
                    .expect("submit 已建会话账");
                self.blocks_m.ensure_for_len(&mut s.block_table, turn.prompt_ids.len())?;
            }
            self.write_bt(turn.session_id).await?;
            self.active = Some(ActiveTurn {
                id: turn.id,
                session_id: turn.session_id,
                ephemeral: turn.ephemeral,
                gdn_slot,
                prompt_ids: turn.prompt_ids,
                max_new: turn.spec.max_new,
                out: Vec::new(),
                fed: cached_len,
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
                    // 块长:配置块长 ∩ 剩余(E1 paged 核后无 256 窗钳制;
                    // 全局注意力语义,长 ctx 安全)
                    let chunk = remaining.min(self.cfg.prefill_chunk).max(1);
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
        let (sid, tid, pos) = {
            let act = self.active.as_ref().expect("active 已保证");
            let tid = *act.out.last().expect("生成中");
            (act.session_id, tid, act.fed)
        };
        // E2b:物理槽 + 跨页增长(块链增长时重写持久块表)
        let (kv_slot, gdn_slot_v, grew) = {
            let s = self.sessions.get_mut(sid).expect("账在");
            let before = s.block_table.len();
            self.blocks_m.ensure_for_len(&mut s.block_table, pos + 1)?;
            let grew = s.block_table.len() != before;
            let page = self.page;
            let b = s.block_table[pos / page];
            (b * page as u32 + (pos % page) as u32, s.gdn_slot, grew)
        };
        if grew {
            self.write_bt(sid).await?;
        }
        self.session
            .step(&[
                ("frontier", &[tid as f32]),
                ("pos", &[pos as f32]),
                ("kv_len", &[(pos + 1) as f32]),
                ("kv_slot", &[kv_slot as f32]),
                ("gdn_slot", &[gdn_slot_v as f32]),
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

    /// 活跃会话块表 → 持久 bt 块(设备)。图烘焙的是 bt 指针,内容随
    /// 会话切换/块链增长重写(指针稳定 = 图不重捕);f32 过线(契约 5)
    async fn write_bt(&mut self, sid: u64) -> Result<()> {
        let table = self.sessions.get(sid).expect("账在").block_table.clone();
        let mut v = vec![0f32; self.nb];
        for (i, b) in table.iter().enumerate() {
            v[i] = *b as f32;
        }
        let face = self.session.face_mut();
        face.write_block_f32(&self.bt, 0, &v).await
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
        let t = ids.len();
        // E2b:活跃会话块链 + GDN 格(逻辑位 → 物理槽由块表换算)
        let (bt_chain, gdn_slot_v) = {
            let act = self.active.as_ref().expect("active 已保证");
            let s = self.sessions.get(act.session_id).expect("账在");
            (s.block_table.clone(), s.gdn_slot)
        };
        let f32seq = |start: usize, n: usize| -> Vec<u8> {
            (0..n)
                .flat_map(|i| ((start + i) as f32).to_le_bytes())
                .collect()
        };
        let kvs_step: Vec<KvBuffers> = self
            .kvs_b
            .iter()
            .map(|b| KvBuffers {
                k_cache: block_leaf_dt(&b.k_cache.0, self.k_shape(), d.dtype),
                v_cache: block_leaf_dt(&b.v_cache.0, self.v_shape(), d.dtype),
                slots: TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base, t)),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base + 1, t)),
                block_tables: TensorOps::of_block(self.bt.id, Dtype::F32, vec![1, self.nb]),
            })
            .collect();
        let gdns_step: Vec<GdnBuffers> = self
            .gdns_b
            .iter()
            .map(|g| GdnBuffers {
                conv_q: block_leaf(&g.conv_q.0, vec![GDN_SLOTS, d.nk * d.hk, 3]),
                conv_k: block_leaf(&g.conv_k.0, vec![GDN_SLOTS, d.nv * d.hv, 3]),
                conv_v: block_leaf(&g.conv_v.0, vec![GDN_SLOTS, d.nv * d.hv, 3]),
                rec: block_leaf(&g.rec.0, vec![GDN_SLOTS, d.nv, d.hk, d.hv]),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32seq(0, 1)),
            })
            .collect();
        let ids_t = TensorOps::from_host(
            Dtype::F32,
            vec![t],
            &ids.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect::<Vec<u8>>(),
        );
        let pos_t = TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base, t));
        // E2b:slots = 物理槽(块链换算);legacy 恒等直排不变
        let slots_t = if self.paged {
            let page = self.page;
            let phys: Vec<f32> = (base..base + t)
                .map(|pos| {
                    let b = bt_chain[pos / page];
                    (b * page as u32 + (pos % page) as u32) as f32
                })
                .collect();
            TensorOps::from_host(Dtype::F32, vec![t], &f32b(&phys))
        } else {
            TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base, t))
        };
        let lens_t = TensorOps::from_host(Dtype::F32, vec![t], &f32seq(base + 1, t));
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[gdn_slot_v as f32]));
        let ctx = ForwardCtx::model_prefill(
            t, &pos_t, &kvs_step, &self.rope, &gdns_step, &slots_t, &lens_t, &gdn_slot,
        );
        let face = self.session.face_mut();
        let vocab = self.model.vocab_size();
        if is_last {
            let tree = self.model.forward(&ids_t, &ctx);
            let (b, arena) = eval_ops_scoped(tree.step(), face).await?;
            let esz = if d.dtype == Dtype::F16 { 2 } else { 4 };
            let mut buf = vec![0u8; t * vocab * esz];
            face.dtoh(&b, &mut buf).await?;
            face.free(&arena).await?; // E2a:根已收割,中间块归池
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
            let (b, arena) = eval_ops_scoped(tree.step(), face).await?;
            let esz = if d.dtype == Dtype::F16 { 2 } else { 4 };
            let mut buf = vec![0u8; t * d.hidden * esz];
            face.dtoh(&b, &mut buf).await?;
            face.free(&arena).await?; // E2a:中间块归池(每 chunk 零净增)
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
        // 会话账落地(S1 收口):KV 此刻 = prompt 全量 + 生成段;GDN 状态
        // 跨 turn 连续。临时会话终了即焚(表不随 turn 数无界增长)
        if let Some(s) = self.sessions.get_mut(act.session_id) {
            s.commit_turn(&act.prompt_ids, &act.out);
        }
        if act.ephemeral {
            // 终了即焚:块链归还账房 + 会话表移除(GDN 格随之释放)
            if let Some(s) = self.sessions.get_mut(act.session_id) {
                let mut table = std::mem::take(&mut s.block_table);
                self.blocks_m.release_table(&mut table);
            }
            self.sessions.close(act.session_id).ok();
        }
        let text = self.tok.decode(&act.out);
        Ok(TurnEvent::Completed { turn: act.id, text })
    }

    /// GDN 状态 per-turn 零化(conv 三段 + recurrent;分块写免 host 大向
    /// 量 —— rec 单层 16M 元素)。KV 不重置:kv_len 窗口 + 先写后打分
    /// 语义下,本 turn 打分的槽位全部由本 turn 写过。
    /// GDN 状态重置(turn-open;新序列零态语义)。设备侧 memset
    /// (2026-09-26 定谳:host 往返清零 1.2GB = 14s/turn,黑洞实测)
    async fn reset_gdn(&mut self, gdn_slot: usize) -> Result<()> {
        // E2b:按格重置(offset = 格号 × 行字节)—— 多会话各清各格,
        // 其余格的其他会话状态不受扰
        for g in &self.gdns_b {
            let blocks = [
                (&g.conv_q.0, g.conv_q.1),
                (&g.conv_k.0, g.conv_k.1),
                (&g.conv_v.0, g.conv_v.1),
                (&g.rec.0, g.rec.1),
            ];
            for (bn, elems) in blocks {
                let row = elems / GDN_SLOTS;
                self.session
                    .face_mut()
                    .memset_zero_at(bn, gdn_slot * row * 4, row * 4)
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

/// 按 dtype 清零分配(KV/GDN 状态块)。**设备侧 alloc + memset**,不走
/// host 零向量 —— 曾用 `vec![0f32; n]` + to_le_bytes 逐元素造 64MB(rec
/// 单层 16M 元素),debug 循环烧掉 ~90s(F5-4 同款 debug 转换税);池
/// alloc 无零保证,必须显式 memset(F5-4 已去 fill(0),不能省)
async fn zero_block_dt<D: DeviceClient>(face: &mut D, n: usize, dt: Dtype) -> Result<BlockN> {
    // 2B 激活域显式臂(BF16 曾落 else 按 4B 误解释 —— dtype 表驱动律)
    let esz = if matches!(dt, Dtype::F16 | Dtype::BF16) { 2 } else { 4 };
    let b = face.alloc(dt, n).await?;
    face.memset_zero(&b, n * esz).await?;
    Ok((b, n))
}

fn block_leaf_dt(b: &Bytes, shape: Vec<usize>, dt: Dtype) -> TensorOps {
    match dt {
        Dtype::F16 | Dtype::BF16 => TensorOps::of_block(b.id, dt, shape),
        _ => TensorOps::of_block(b.id, Dtype::F32, shape),
    }
}

async fn zero_block<D: DeviceClient>(face: &mut D, n: usize) -> Result<BlockN> {
    zero_block_dt(face, n, Dtype::F32).await
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

/// 增量解码:全量重解取后缀差分。多字节字符跨 token 时,半截字节经
/// tokenizers 解出 U+FFFD 占位(字节数与真前缀不等,**不能按字节
/// index 切**)—— 差分按字节公共前缀比对,双侧回退到字符边界;尾部
/// 占位符扣住不发,等拼全后随下一笔 delta 出(终文由 complete()
/// 全文重解兜底,流式末字符可能延后一笔)
fn decode_delta(tok: &Tokenizer, out: &[u32], decoded: &mut String) -> String {
    const REPL: &str = "\u{FFFD}";
    let full = tok.decode(out);
    let common = decoded
        .as_bytes()
        .iter()
        .zip(full.as_bytes())
        .take_while(|(a, b)| a == b)
        .count();
    let mut cut = common;
    while cut > 0 && (!full.is_char_boundary(cut) || !decoded.is_char_boundary(cut)) {
        cut -= 1;
    }
    let mut emit = full[cut..].to_string();
    if emit.ends_with(REPL) {
        emit.truncate(emit.len() - REPL.len());
    }
    *decoded = full[..cut + emit.len()].to_string();
    emit
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

    /// CPU:增量解码多字节字符跨 token(🌟 4 字节;逐 token 喂入,
    /// 差分不得 panic 且终态与前缀累计一致)
    #[tokio::test]
    async fn cpu_incremental_decode_multibyte() {
        let mut engine = Engine::on(
            EngineConfig { device_ordinal: 0, max_seq_tokens: 64, prefill_chunk: 16 },
            CpuFace::new(),
        )
        .expect("engine 构造");
        let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
        let ids = loaded.tokenizer.encode("你好，一个🌟加一句中文。");
        assert!(ids.len() >= 4, "测试需要多 token");
        let mut acc = String::new();
        for k in 1..=ids.len() {
            let _delta = decode_delta(&loaded.tokenizer, &ids[..k], &mut acc);
        }
        let final_text = loaded.tokenizer.decode(&ids);
        assert!(
            final_text.starts_with(acc.trim_end()),
            "累计差分应是终文前缀:acc={acc:?} final={final_text:?}"
        );
        assert!(final_text.contains('🌟'), "终文应含拆跨字符");
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

    /// GPU 门控:同会话连续 turn(S0 验收)—— turn2 应命中账本
    /// (跳过 GDN 重置,只 prefill 增量段);临时会话路径由生命周期
    /// 测试覆盖,此处验 Some(id) 连续性 + 收口账推进。
    #[tokio::test]
    async fn gpu_session_continuity() {
        let Some(ordinal) = gpu_ordinal() else {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        };
        let dir = asset_dir();
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");

        // turn1:建立会话账(显式 id;None = 临时会话终了即焚,不归此测)
        let (sid, t1) = running
            .submit_session(Some(42), "请记住:我最喜欢的数字是七。", 16)
            .expect("t1");
        assert_eq!(sid, 42);
        let text1 = drive_turn(&mut running, t1).await;
        eprintln!("[test] t{t1}(session {sid}): {text1}");
        assert!(!text1.trim().is_empty());
        assert!(running.session_len(sid).unwrap_or(0) > 0, "turn1 收口账非零");

        // turn2:同会话全量重发(历史 + 新问题)→ 守卫应命中,增量 prefill
        let (sid2, t2) = running
            .submit_session(
                Some(sid),
                "请记住:我最喜欢的数字是七。我刚才说我最喜欢的数字是什么?",
                16,
            )
            .expect("t2");
        assert_eq!(sid2, sid, "同会话归队");
        let text2 = drive_turn(&mut running, t2).await;
        eprintln!("[test] t{t2}(session {sid2}): {text2}");
        assert!(!text2.trim().is_empty());

        // 收口验证:账本应推到 turn2 全量 + 生成段;记忆质量(答“七”)
        // 属 S3 金标验收,此处不断言文本内容
        let cached = running.session_len(sid).expect("账在");
        assert!(cached > 0, "同会话收口后账本非零");
    }

    /// GPU 门控:4k 长 ctx 前缀记忆 QA(E1 收尾 P4)—— 暗号埋在 prompt
    /// 开头(≈token 10),问题压在 ≈3900 token 处;真全局注意力下应召回,
    /// 旧 OWL_MAX_KV=256 滑窗截断下物理不可过(暗号在窗外)。兼验收:
    /// 4k 预算守卫、分页 prefill 百级 chunk、v1 decode 长程、同会话
    /// turn2(前缀失配回退全量路径 @4k)。
    #[tokio::test]
    async fn gpu_longctx_4k_prefix_qa() {
        let Some(ordinal) = gpu_ordinal() else {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        };
        let dir = asset_dir();
        let seq: usize = std::env::var("OWL_E2E_SEQ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: seq,
            prefill_chunk: 32,
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");

        // 组装:暗号开头 + 中性填充 + 补全式探针收尾(裸 LM 无 chat 模板,
        // 问答式会退化成文本续写 —— 探针改为让模型续写暗号本体)
        let marker = "本会话暗号:「蓝鲸二十一号」。\n\n";
        let question = "\n\n清单结束。按约定,暗号重复一遍:暗号是「";
        let filler = |i: usize| {
            format!(
                "第{i}条:观测点{i}记录到信号{i},强度{i}级,来源坐标({i},{i}),持续{i}天。\n"
            )
        };
        let mut body = String::from(marker);
        let mut i = 0usize;
        let nofill = std::env::var_os("OWL_E2E_NOFILL").is_some();
        loop {
            if nofill {
                break; // 对照实验:短 ctx 直问(验模型/模板,不验长程)
            }
            let cand = format!("{body}{}", filler(i));
            let full = format!("{cand}{question}");
            if loaded.tokenizer.encode(&full).len() + 32 > seq {
                break; // 留 32 token 生成余量(turn2 预算同享)
            }
            body = cand;
            i += 1;
        }
        let prompt_t1 = format!("{body}{question}");
        let n_tok = loaded.tokenizer.encode(&prompt_t1).len();
        eprintln!("[test] 长 ctx prompt = {n_tok} tok(填充 {i} 条)");
        assert!(
            nofill || n_tok > seq * 4 / 5,
            "长文应填满窗口 ≥80%,得 {n_tok}/{seq}"
        );

        let mut running = engine.run(loaded).await.expect("装配");
        let t0 = std::time::Instant::now();
        let (sid, t1) = running
            .submit_session(Some(7), prompt_t1.as_str(), 16)
            .expect("t1 应在 4k 预算内");
        assert_eq!(sid, 7);
        let mut ttft = None;
        let mut prefill_last = None;
        let text1;
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Idle => panic!("turn {t1} 队列丢失"),
                TurnEvent::Prefill { turn, fed, total } if turn == t1 => {
                    if fed % (32 * 16) == 0 || fed == total {
                        eprintln!("[ev] t{t1} prefill {fed}/{total} @ {:?}", t0.elapsed());
                    }
                    prefill_last = Some((fed, total, t0.elapsed()));
                }
                TurnEvent::Token { turn, .. } if turn == t1 && ttft.is_none() => {
                    ttft = Some(t0.elapsed());
                }
                TurnEvent::Completed { turn, text } if turn == t1 => {
                    text1 = text;
                    break;
                }
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                _ => {}
            }
        }
        let wall1 = t0.elapsed();
        if let Some((fed, total, el)) = prefill_last {
            eprintln!(
                "[bench] t{t1} prefill {fed}/{total} tok @ {el:?}({:.0} tok/s,debug) | TTFT {:?} | 总 {:?}",
                fed as f64 / el.as_secs_f64(),
                ttft.unwrap(),
                wall1
            );
        }
        eprintln!("[test] t{t1} 答: {text1}");
        // 记忆质量金标挂 S3(0.8B + greedy + 裸模板的答句质量不可靠;
        // 对照实验:短/长 ctx 行为一致 ⇒ 机械等价)。此处验收 =
        // 4k 全链不崩 + 帐本收口 + 计时(v1 smem 越界已修,见上)
        assert!(!text1.trim().is_empty(), "t{t1} 产出非空");

        // turn2 同会话再问(前缀失配 → 回退全量路径)。E2a 后唯一验收:
        // 修复前此处在 4k 下 OOM(server 中间块零回收,双 turn 累计
        // ~24GB;现 eval 竞技场每 chunk 归池,显存恒平)
        {
            let (_, t2) = running
                .submit_session(Some(sid), format!("{body}再重复一遍:暗号是「"), 16)
                .expect("t2 应在预算内");
            let t0b = std::time::Instant::now();
            let text2 = drive_turn(&mut running, t2).await;
            eprintln!("[bench] t{t2} 全量回退重灌 + 生成 @ {:?}", t0b.elapsed());
            eprintln!("[test] t{t2} 答: {text2}");
            assert!(!text2.trim().is_empty(), "t{t2} 产出非空");
            assert!(running.session_len(sid).unwrap_or(0) >= n_tok, "账本收口");
        }
    }

    /// GPU 门控:多会话隔离(E2b 验收)—— A/B 两会话交替 turn:
    /// ① B 的 turn 前后,A 的块链/账本逐字节不受扰;② A/B 物理块互异
    /// (无前缀缓存时零共享);③ 双会话产出非空。隔离 = 块管理器的
    /// 直接行为证据(块链不同 ⇒ KV 物理槽不同)。
    #[tokio::test]
    async fn gpu_multi_session_isolation() {
        let Some(ordinal) = gpu_ordinal() else {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        };
        let dir = asset_dir();
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");

        let (_, ta1) = running
            .submit_session(Some(101), "我的代号是阿尔法。", 12)
            .expect("A1");
        let ta1_text = drive_turn(&mut running, ta1).await;
        eprintln!("[test] A1: {ta1_text}");
        assert!(!ta1_text.trim().is_empty());
        let a_table_1 = running.sessions.get(101).expect("A 账在").block_table.clone();
        let a_len_1 = running.session_len(101).expect("A 账在");
        assert!(!a_table_1.is_empty(), "A 块链非空");

        let (_, tb1) = running
            .submit_session(Some(202), "我的代号是贝塔。", 12)
            .expect("B1");
        let tb1_text = drive_turn(&mut running, tb1).await;
        eprintln!("[test] B1: {tb1_text}");
        assert!(!tb1_text.trim().is_empty());

        // 隔离断言 ①:B 的 turn 后,A 的块链/账本逐字节不变
        let a_table_2 = running.sessions.get(101).expect("A 账在").block_table.clone();
        assert_eq!(a_table_1, a_table_2, "B 的 turn 不得扰动 A 块链");
        assert_eq!(running.session_len(101), Some(a_len_1), "B 的 turn 不得扰动 A 账本");
        // 隔离断言 ②:物理块零共享(无前缀缓存 = 全新分配)
        let b_table = running.sessions.get(202).expect("B 账在").block_table.clone();
        assert!(b_table.iter().all(|b| !a_table_2.contains(b)), "A/B 物理块互异");
        eprintln!("[test] 块链 A = {a_table_2:?} / B = {b_table:?}");

        // A 二轮(全量重发,前缀守卫失配回退全量也须稳)
        let (_, ta2) = running
            .submit_session(Some(101), "我的代号是阿尔法。我的代号是什么?", 12)
            .expect("A2");
        let ta2_text = drive_turn(&mut running, ta2).await;
        eprintln!("[test] A2: {ta2_text}");
        assert!(!ta2_text.trim().is_empty());
        assert!(running.session_len(101).unwrap_or(0) >= a_len_1, "A 账本推进");
    }

    /// 泵到目标 turn 完成,回吐全文(其间事件仅观测)
    async fn drive_turn<D: DeviceClient>(running: &mut RunningEngine<D>, id: u64) -> String {
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Idle => panic!("turn {id} 队列丢失"),
                TurnEvent::Prefill { turn, fed, total } if turn == id => {
                    eprintln!("[ev] t{turn} prefill {fed}/{total}")
                }
                TurnEvent::Completed { turn, text } if turn == id => break text,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                _ => {}
            }
        }
    }
}
