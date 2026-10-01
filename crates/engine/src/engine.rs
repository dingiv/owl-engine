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
//! **会话接线(S0/S1,2026-09-26)**:SessionTable 入引擎 ——
//! `submit_session(Some(id), …)` = 连续会话(同会话跳过 GDN 重置,
//! 只 prefill `cached_len..` 增量段;token 账前缀守卫失配回退全量);
//! `submit(None)` = 临时会话终了即焚,行为同 M0.5(每 turn 全量重算)。
//!
//! # 模块地图(2026-10-01 拆分;原单文件 1782 行 → 六户)
//!
//! - [`crate::engine`](self):Engine 构造面 + `run()` 装配(状态块经
//!   StatePool、图闭包、GraphPlan warmup)
//! - [`crate::loader`]:LoadedModel + ModelLoader(spec 声明驱动装载)
//! - [`crate::running`]:RunningEngine(turn 账 + submit/pump/generate)
//! - [`crate::scheduler`]:调度决策词汇 + `schedule()` 纯主机侧决策
//! - [`crate::exec`]:执行器(设备侧照办)+ 采样收口 + decode_delta
//! - [`crate::state`]:StatePool 设备状态块治理(KV 池/GDN 格/快照/块表)

use std::sync::Arc;

use owl_cuda::{DeviceSelector, GpuClient};
use owl_iface::contract::{DeviceClient, Dtype, ModelError};
use owl_models::layers::gdn::GdnBuffers;
use owl_models::module::{ForwardCtx, KvBuffers, Module};

use crate::blocks::BlockManager;
use crate::graph_plan::{GraphPlan, GraphPlanDesc, InputSlot, OutputSlot, PlanCtx};
use crate::loader::{LoadedModel, ModelLoader};
use crate::running::RunningEngine;
use crate::session::SessionTable;
use crate::state::{ModelDims, StatePool};

type Result<T> = std::result::Result<T, ModelError>;

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

impl<D: DeviceClient + 'static> Engine<D> {
    /// 注入 face(CPU 测试/自定义后端;唯一构造机制,new 是其 GPU 特化)
    pub fn on(cfg: EngineConfig, face: D) -> Result<Self> {
        Ok(Self { face, cfg })
    }

    /// 模型加载器(借用引擎 face;装载即校验,错误在 load 边界落地)
    pub fn loader(&mut self) -> ModelLoader<'_, D> {
        ModelLoader::new(&mut self.face)
    }

    /// 模型交引擎:状态块/槽/Session 组建 + warmup —— 仍不跑任何 turn,
    /// 返回执行态引擎(请求入口)。
    pub async fn run(mut self, loaded: LoadedModel) -> Result<RunningEngine<D>> {
        let t_run = std::time::Instant::now();
        let s = self.cfg.max_seq_tokens;
        let (_, hkv, hd) = loaded.spec.full_heads;
        let (nk, hk, nv, hv) = loaded.spec.gdn_heads;
        let dims = ModelDims {
            hkv, hd, nk, hk, nv, hv,
            hidden: loaded.spec.hidden, dtype: loaded.spec.dtype,
        };

        // 状态块(StatePool::alloc,零初始化;GDN 段每 turn 开始按格重置)。
        // KV 布局表驱动(kv_paged_policy;REQ-HW-01);块池容量 E2b:
        // 默认 = 2 × 单会话容量(两会话满载共存),OWL_POOL_TOKENS 覆写
        let pool_tokens = std::env::var("OWL_POOL_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2 * s);
        let pool = StatePool::alloc(&mut self.face, dims, &loaded.spec.layer_types, s, pool_tokens).await?;

        // 块账房(E2b)+ E2c 前缀缓存启用(容量 = 池半;OWL_PREFIX_CACHE=0 关闭)
        let mut blocks_m = BlockManager::new(pool.nb, pool.page);
        if pool.paged
            && std::env::var("OWL_PREFIX_CACHE").map(|v| v != "0").unwrap_or(true)
        {
            blocks_m.enable_prefix_cache((pool.nb / 2).max(1));
        }

        // Session 闭包:槽 → 整模单树(状态句柄捕获;模型 Arc 共享)。
        // 捕获期禁 Htod:常量标量走持久槽;KV/GDN 双槽分立 ——
        // KV 槽 = 本 token 自己的格子(窗 = [slot-kv_len+1, slot],随步进;
        // 2026-09-26 窗口语义探针定谳:slots 恒 0 读块外垃圾行);
        // GDN 槽 = 序列状态格,turn 内恒 0(递推状态按序列累积)。
        let model = Arc::clone(&loaded.model);
        let rp = loaded.rope.clone();
        let vocab = loaded.model.vocab_size();
        let kv_caches = pool.kv_caches();
        let gdn_caches = pool.gdn_caches();
        let bt_leaf = pool.bt_leaf();
        let forward = move |sc: &PlanCtx| -> Result<()> {
            let ids = sc.input("frontier")?;
            let pos = sc.input("pos")?;
            let kv_len = sc.input("kv_len")?;
            let kv_slot = sc.input("kv_slot")?;
            let gdn_slot = sc.input("gdn_slot")?;
            let kvs_step: Vec<KvBuffers> = kv_caches
                .iter()
                .map(|(k, v)| KvBuffers {
                    k_cache: k.clone(),
                    v_cache: v.clone(),
                    slots: kv_slot.clone(),
                    kv_lens: kv_len.clone(),
                    block_tables: bt_leaf.clone(),
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdn_caches
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
            // E3 设备采样:token = argmax(logits)(f32 数值过线,契约 5)
            let tok = owl_models::ops::argmax_f32idx(&tree, vocab, 0);
            sc.output("token", &tok);
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
                outputs: vec![
                    OutputSlot { name: "logits", shape: vec![1, vocab], dtype: loaded.spec.dtype },
                    OutputSlot { name: "token", shape: vec![1], dtype: Dtype::F32 },
                ],
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
            pool,
            queue: std::collections::VecDeque::new(),
            active: None,
            next_id: 1,
            sessions: SessionTable::new(),
            model: loaded.model,
            rope: loaded.rope,
            blocks_m,
        })
    }
}
