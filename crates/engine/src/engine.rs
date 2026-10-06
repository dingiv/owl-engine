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

/// E5-M5:fold 图的状态叶子(flat 单槽行;slot 恒 0 = 基指针;
/// 自由函数 —— 图闭包 move 语义不可借 self)
fn leaf_flat(b: &crate::state::BlockN) -> owl_models::tensor::TensorOps {
    let n = b.1;
    owl_models::tensor::TensorOps::of_block(b.0.id, owl_iface::contract::Dtype::F32, vec![n])
}

pub struct Engine<D: DeviceClient> {
    face: D,
    cfg: EngineConfig,
}

/// boot 序号(metrics 打点命名空间;进程内自增,b1/b2…)
static BOOT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
        let (hq, hkv, hd) = loaded.spec.full_heads;
        let (nk, hk, nv, hv) = loaded.spec.gdn_heads;
        let dims = ModelDims {
            hq, hkv, hd, nk, hk, nv, hv,
            hidden: loaded.spec.hidden, dtype: loaded.spec.dtype,
        };

        // 状态块(StatePool::alloc,零初始化;GDN 段每 turn 开始按格重置)。
        // KV 布局表驱动(kv_paged_policy;REQ-HW-01);块池容量 E2b:
        // 默认 = 2 × 单会话容量(两会话满载共存),OWL_POOL_TOKENS 覆写
        let pool_tokens = std::env::var("OWL_POOL_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2 * s);
        // FlashInfer prefill 面(OWL_FLASHINFER=1;E1.5):K 影子池 + 每
        // chunk 表四件套(ForwardCtx.fi)。f16 + paged 才有意义。
        // 解释器执行环境(EnvProvider;E1.5 抽象):from_env 兼容面 +
        // 硬件档自 iface op_env 合入(被动律:引擎 = 环境事实来源)
        let mut env = owl_models::env::EnvProvider::from_env();
        if let Some(oe) = self.face.op_env() {
            env.hw = owl_models::env::HwEnv::Cuda(oe.hw.arch);
        }
        env.quant = loaded.quant_plan;
        let fi_quant = if env.attn.fi
            && loaded.spec.dtype == owl_models::tensor::Dtype::F16
        {
            Some(env.kv.quant)
        } else {
            None
        };
        // E5-M2b/C7:spec 模式三态裁决 —— depth>0 且检查点有 mtp.* →
        // Mtp(真草稿);OWL_SPEC_DUMB=1 → Dumb(哑草稿诊断面);否则
        // Off(spec_depth 归零,调度回落 DecodeBatch —— 无草稿器不白发
        // verify 税)。depth 上限 3(kv_slots [u32;4] 契约)。
        let spec_depth_raw = std::env::var("OWL_SPEC_DEPTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&d| d > 0)
            .unwrap_or(0);
        let dflash2_dir = std::env::var("OWL_DFLASH2_DIR").ok();
        let spec_mode = if spec_depth_raw == 0 {
            crate::running::SpecMode::Off
        } else if dflash2_dir.is_some() {
            crate::running::SpecMode::DFlash2
        } else if loaded.mtp_dir.is_some() {
            crate::running::SpecMode::Mtp
        } else if std::env::var_os("OWL_SPEC_DUMB").is_some() {
            crate::running::SpecMode::Dumb
        } else {
            eprintln!("[boot] OWL_SPEC_DEPTH>0 但无草稿器(非 mtp/dflash2 检查点)→ C7 回落 DecodeBatch");
            crate::running::SpecMode::Off
        };
        let spec_depth = if spec_mode == crate::running::SpecMode::Off {
            0
        } else if spec_mode == crate::running::SpecMode::DFlash2 {
            // DFlash2:block_size 8 = 1 锚 + 7 草稿(检查点家族定形)
            spec_depth_raw.min(7)
        } else {
            spec_depth_raw.min(3)
        };
        if spec_mode != crate::running::SpecMode::Off {
            eprintln!(
                "[boot] spec 模式 = {:?} depth={spec_depth}(注:spec 轮恒 greedy,采样 parity 挂 D4 device accept)",
                spec_mode
            );
        }
        // 草稿器装载(C7;embed/lm_head 均共享 target)
        let drafter = match spec_mode {
            crate::running::SpecMode::Mtp => {
                let dir = loaded.mtp_dir.as_ref().expect("mtp_dir");
                let (mtp, _) = owl_models::specs::qwen35::load_27b_mtp(dir, &mut self.face).await?;
                eprintln!("[boot] MTP 草稿头装载(spec depth={spec_depth})");
                Some(crate::running::Drafter::Mtp(std::sync::Arc::new(mtp)))
            }
            crate::running::SpecMode::DFlash2 => {
                let dir = dflash2_dir.as_ref().expect("dflash2_dir");
                let (d2, _) = owl_models::specs::qwen35::load_27b_dflash2(std::path::Path::new(dir), &mut self.face).await?;
                eprintln!("[boot] DFlash2 草稿装载(depth={spec_depth};1.92B BF16 激活,sglang 对齐)");
                Some(crate::running::Drafter::DFlash2(std::sync::Arc::new(d2)))
            }
            _ => None,
        };
        // DFlash2 草稿 rope(θ 1e7,rotary 全维 128;z-lab 检查点家族)
        let draft_rope = match &drafter {
            Some(crate::running::Drafter::DFlash2(_)) => {
                let dr = owl_models::layers::rope::Rope::new(262_144, 128, 128, 1.0e7)?;
                {
                    let ctx = owl_models::module::LoaderCtx {
                        dtype: owl_models::contract::Dtype::F16,
                        shard: 1,
                        device_repack: false,
                    };
                    owl_models::interpreters::eval_load(&dr, &mut self.face, &dr.tables(), &ctx).await?;
                }
                Some(std::sync::Arc::new(dr))
            }
            _ => None,
        };
        let pool = StatePool::alloc(&mut self.face, dims, &loaded.spec.layer_types, s, pool_tokens, fi_quant, spec_depth > 0, spec_mode == crate::running::SpecMode::Mtp, spec_mode == crate::running::SpecMode::DFlash2).await?;
        if fi_quant.is_some() {
            eprintln!(
                "[boot] FlashInfer prefill 面启用(影子池 ×{},quant={:?})",
                pool.k_fis.len(),
                fi_quant.unwrap()
            );
        }

        // 块账房(E2b)+ E2c 前缀缓存启用(容量 = 池半;OWL_PREFIX_CACHE=0 关闭)
        let mut blocks_m = BlockManager::new(pool.nb, pool.page);
        if pool.paged
            && std::env::var("OWL_PREFIX_CACHE").map(|v| v != "0").unwrap_or(true)
        {
            blocks_m.enable_prefix_cache((pool.nb / 2).max(1));
        }
        // MTP 链页账房(E5-M2b):独立页池同几何;无前缀缓存。
        // 非 mtp 模式 0 块 —— 误用 = 结构化报错(池耗尽语义)。
        let blocks_mtp = BlockManager::new(
            if spec_mode == crate::running::SpecMode::Mtp { pool.nb } else { 0 },
            pool.page,
        );

        let vface = self.face.face_clone();
        // E5-M5:fold/propose 桶形图族的 face 句柄(depth × 2)
        let mut extra_faces: Vec<D> = (0..2 * spec_depth.max(1) + 2).filter_map(|_| self.face.face_clone()).collect();
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
        let attn_v2 = pool.attn_v2_scratch();
        let forward = move |sc: &PlanCtx| -> Result<()> {
            let ids = sc.input("frontier")?;
            let pos = sc.input("pos")?;
            let kv_len = sc.input("kv_len")?;
            let kv_slot = sc.input("kv_slot")?;
            let gdn_slot = sc.input("gdn_slot")?;
            let ts_buf = sc.input_opt("ts_buf")?;
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
            let mut ctx = ForwardCtx::model_decode(1, &pos, &kvs_step, &rp, &gdns_step);
            ctx.env = env; // 捕获期烘焙(图闭包 env 定格;回放沿用)
            ctx.ts_buf = ts_buf; // 刀D 时间戳探针(None = 关)
            ctx.attn_v2 = attn_v2.clone(); // v2 分页 decode scratch(None = v1)
            let tree = model.forward(&ids, &ctx);
            // E3 设备采样:token = argmax(logits)(f32 数值过线,契约 5)
            let tok = owl_models::ops::argmax_f32idx(&tree, vocab, 0);
            sc.output("token", &tok);
            sc.output("logits", &tree)
        };

        let (session, capture_outcome) = GraphPlan::plan(
            self.face,
            GraphPlanDesc {
                inputs: {
                    let mut v = vec![
                        InputSlot::f32("frontier", 1),
                        InputSlot::f32("pos", 1),
                        InputSlot::f32("kv_len", 1),
                        InputSlot::f32("kv_slot", 1),
                        InputSlot::f32("gdn_slot", 1).init(vec![0.0]),
                    ];
                    if std::env::var_os("OWL_TS_PROBE").is_some() {
                        v.push(InputSlot::f32("ts_buf", 4096));
                    }
                    v
                },
                outputs: vec![
                    OutputSlot { name: "logits".into(), shape: vec![1, vocab], dtype: loaded.spec.dtype },
                    OutputSlot { name: "token".into(), shape: vec![1], dtype: Dtype::F32 },
                ],
                // 逃生开关(C1 禁 graph 裁决配套):OWL_NO_GRAPH=1 → eager
                // 直发(decode 逐步 eval,无捕获回放)—— 图内/图外行为
                // A/B 的对照臂(2026-10-01 paged decode 质量案)
                capture: std::env::var_os("OWL_NO_GRAPH").is_none(),
            },
            forward,
        )
        .await?;
        if let crate::graph_plan::PlanOutcome::EagerFallback { reason } = &capture_outcome {
            eprintln!("[engine] 捕获降级为 eager:{reason}");
        }
        eprintln!("[boot] GraphPlan plan(warmup+捕获) {:.2}s", t_run.elapsed().as_secs_f32());

        // E5-M4:verify 桶形图(T = depth+1 定形;spec 模式专用)—— 消
        // 投机轮的整模 eager 发射税(~830 发 × ~25µs)。FI 不入图(T=4
        // attention 成本可忽略;回避 plan-in-graph);输出 = tok/hid +
        // **GDN fold 记录**(每层 8 件,tap 收集 → 多根共享 memo 单次归约,
        // 刀 1.6 语义)。捕获失败 → None → 执行器落旧 eager verify+重放路。
        let verify_graph = if spec_depth > 0
            && vface.is_some()
            && matches!(capture_outcome, crate::graph_plan::PlanOutcome::Captured)
        {
            let vface = vface.unwrap();
            let depth1 = spec_depth + 1;
            let model_v = Arc::clone(&loaded.model);
            let rp_v = loaded.rope.clone();
            let kv_caches = pool.kv_caches();
            let gdn_caches = pool.gdn_caches();
            let bt_leaf = pool.bt_leaf_flat();
            let env_v = env;
            let key_dim = dims.nk * dims.hk;
            let value_dim = dims.nv * dims.hv;
            let mut vouts = vec![
                OutputSlot { name: "tok".into(), shape: vec![depth1, 1], dtype: Dtype::F32 },
                OutputSlot { name: "hid".into(), shape: vec![depth1, dims.hidden], dtype: dims.dtype },
            ];
            // DFlash2 target taps(E5-DF2;层输出残差流,sglang capture 同语义)
            // OWL_DFLASH_NOTAPS=1 → 退回 last_hidden(排查开关:隔离 taps 图)
            let dflash_taps = matches!(spec_mode, crate::running::SpecMode::DFlash2)
                && std::env::var_os("OWL_DFLASH_NOTAPS").is_none();
            // TAPDECL:声明期 tapped(收集)但不挂输出槽 —— 二分「tap 节点
            // 本身」vs「输出槽机制」(2026-10-07 恒等门排查)
            let dflash_tap_decl = dflash_taps
                || (matches!(spec_mode, crate::running::SpecMode::DFlash2)
                    && std::env::var_os("OWL_DFLASH_TAPDECL").is_some());
            if dflash_taps {
                for i in 0..5 {
                    vouts.push(OutputSlot { name: format!("tap{i}"), shape: vec![depth1, dims.hidden], dtype: dims.dtype });
                }
            }
            for i in 0..gdn_caches.len() * 8 {
                let (gi, f) = (i / 8, i % 8);
                let shape = match f {
                    0 | 1 => vec![depth1, key_dim],
                    2 => vec![depth1, value_dim],
                    3 | 4 => vec![depth1, dims.nk, dims.hk],
                    5 => vec![depth1, dims.nv, dims.hv],
                    _ => vec![depth1, dims.nv],
                };
                vouts.push(OutputSlot { name: format!("r{i}") , shape, dtype: dims.dtype });
            }
            let (mut vg, voutcome) = GraphPlan::plan(
                vface,
                GraphPlanDesc {
                    inputs: vec![
                        InputSlot::f32("ids", depth1),
                        InputSlot::f32("pos", depth1),
                        InputSlot::f32("kv_slots", depth1),
                        InputSlot::f32("kv_lens", depth1),
                        InputSlot::f32("gdn_slot", 1),
                        InputSlot::f32("gdn_cu", 2).init(vec![0.0, depth1 as f32]),
                    ],
                    outputs: vouts,
                    capture: std::env::var_os("OWL_NO_GRAPH").is_none(),
                },
                move |sc: &PlanCtx| -> Result<()> {
                    let ids = sc.input("ids")?;
                    let pos = sc.input("pos")?;
                    let slots_in = sc.input("kv_slots")?;
                    let lens_in = sc.input("kv_lens")?;
                    let gdn_slot = sc.input("gdn_slot")?;
                    let gdn_cu = sc.input("gdn_cu")?;
                    let kvs_step: Vec<KvBuffers> = kv_caches
                        .iter()
                        .map(|(k, v)| KvBuffers {
                            k_cache: k.clone(),
                            v_cache: v.clone(),
                            slots: slots_in.clone(),
                            kv_lens: lens_in.clone(),
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
                    let mut ctx = ForwardCtx::model_prefill(
                        depth1, &pos, &kvs_step, &rp_v, &gdns_step, &slots_in, &lens_in,
                        &gdn_slot, 0, None,
                    );
                    ctx.seq_cu = Some(&gdn_cu);
                    ctx.env = env_v;
                    let tap: std::rc::Rc<std::cell::RefCell<Vec<owl_models::tensor::TensorOps>>> =
                        std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
                    ctx.gdn_tap = Some(tap.clone());
                    let vocab = model_v.vocab_size();
                    let (hidden, taps) = if dflash_tap_decl {
                        let (h, t) = model_v.tapped_hidden(&ids, &ctx, &[5, 19, 33, 47, 61]);
                        if dflash_taps {
                            for (i, t) in t.iter().enumerate() {
                                sc.output(format!("tap{i}"), t)?;
                            }
                        }
                        (h, t)
                    } else {
                        let h = model_v.last_hidden(&ids, &ctx);
                        (h.clone(), vec![h])
                    };
                    let _ = &taps;
                    let logits = model_v.embed.lm_head_matmul(&hidden);
                    let rows: Vec<owl_models::tensor::TensorOps> = (0..depth1)
                        .map(|r| owl_models::ops::argmax_f32idx(&logits, vocab, r * vocab))
                        .collect();
                    let mut root = owl_models::tensor::TensorOps::call(owl_models::ops::ids::OPS_CONCAT);
                    for i in 0..8 {
                        root = root.arg(rows.get(i).unwrap_or(&rows[0]));
                    }
                    let root = root
                        .arg_usize(depth1)
                        .arg_usize(1)
                        .arg_usize(1)
                        .with_shape(Dtype::F32, vec![depth1, 1]);
                    sc.output("tok", &root)?;
                    sc.output("hid", &hidden)?;
                    for (i, t) in tap.borrow().iter().enumerate() {
                        sc.output(format!("r{i}"), t)?;
                    }
                    Ok(())
                },
            )
            .await?;
            match voutcome {
                crate::graph_plan::PlanOutcome::Captured => {
                    eprintln!("[boot] verify 桶形图捕获(T={depth1})");
                    Some(vg)
                }
                other => {
                    eprintln!("[boot] verify 图降级({other:?})→ eager verify+重放路");
                    None
                }
            }
        } else {
            None
        };

        // E5-M5:fold 桶形图族(index = m;状态 = spec 快照 buf 就地,
        // slot 恒 0 —— 快照即工作缓冲,fold 后 restore 回写会话格;
        // 记录 = verify 图输出块前缀视图,指针捕获期烘焙)。96 发射的
        // eager fold → 单 graph_launch。
        let mut fold_graphs = Vec::new();
        let mut propose_graphs = Vec::new();
        let mut dflash_graph = None;
        if let Some(vg) = verify_graph.as_ref() {
            let n_rec = pool.gdn_count() * 8;
            let hid_blk = vg.output_block("hid").expect("hid 输出槽");
            let rec_blks: Vec<owl_iface::contract::Bytes> = (0..n_rec)
                .map(|i| vg.output_block(&format!("r{i}")).expect("fold 记录槽"))
                .collect();
            let snap_bufs = pool.spec_snap_bufs();
            let n_roots = pool.gdn_count() * 4;
            for m in 0..spec_depth {
                let m1 = m + 1;
                let model_f = Arc::clone(&loaded.model);
                let gdn_caches = pool.gdn_caches();
                let snap = snap_bufs.clone();
                let rec_blks = rec_blks.clone();
                let scalar = env.gdn.scalar;
                let d = dims;
                let mut fouts = Vec::with_capacity(n_roots);
                for i in 0..n_roots {
                    let shape = match i % 4 {
                        0 => vec![m1, d.nk * d.hk],
                        1 => vec![m1, d.nk * d.hk],
                        2 => vec![m1, d.nv * d.hv],
                        _ => vec![m1, d.nv, d.hv],
                    };
                    fouts.push(OutputSlot { name: format!("f{i}"), shape, dtype: Dtype::F16 });
                }
                let Some(fface) = extra_faces.pop() else {
                    eprintln!("[boot] fold 图 face 耗尽 → eager fold");
                    break;
                };
                let (fg, fout) = GraphPlan::plan(
                    fface,
                    GraphPlanDesc {
                        inputs: vec![
                            InputSlot::f32("slots", 1).init(vec![0.0]),
                            InputSlot::f32("cu", 2).init(vec![0.0, m1 as f32]),
                        ],
                        outputs: fouts,
                        capture: std::env::var_os("OWL_NO_GRAPH").is_none(),
                    },
                    move |sc: &PlanCtx| -> Result<()> {
                        let slots = sc.input("slots")?;
                        let cu = sc.input("cu")?;
                        let rec_views: Vec<owl_models::tensor::TensorOps> = rec_blks
                            .iter()
                            .enumerate()
                            .map(|(i, b)| {
                                let shape = match i % 8 {
                                    0 | 1 => vec![m1, d.nk * d.hk],
                                    2 => vec![m1, d.nv * d.hv],
                                    3 | 4 => vec![m1, d.nk, d.hk],
                                    5 => vec![m1, d.nv, d.hv],
                                    _ => vec![m1, d.nv],
                                };
                                owl_models::tensor::TensorOps::of_block(b.id, d.dtype, shape)
                            })
                            .collect();
                        let gdns: Vec<owl_models::layers::gdn::GdnBuffers> = (0..gdn_caches.len())
                            .map(|li| owl_models::layers::gdn::GdnBuffers {
                                conv_q: leaf_flat(&snap[li * 4]),
                                conv_k: leaf_flat(&snap[li * 4 + 1]),
                                conv_v: leaf_flat(&snap[li * 4 + 2]),
                                rec: leaf_flat(&snap[li * 4 + 3]),
                                slots: slots.clone(),
                            })
                            .collect();
                        let roots = owl_models::spec::fold_tree(
                            &model_f, &rec_views, &gdns, &slots, &cu, 0, m, scalar,
                        )?;
                        for (i, r) in roots.iter().enumerate() {
                            sc.output(format!("f{i}"), r)?;
                        }
                        Ok(())
                    },
                )
                .await?;
                match fout {
                    crate::graph_plan::PlanOutcome::Captured => {
                        eprintln!("[boot] fold 桶形图捕获(m={m})");
                        fold_graphs.push(fg);
                    }
                    other => eprintln!("[boot] fold 图降级(m={m},{other:?})→ eager fold"),
                }
            }

            // E5-DF4:DFlash2 propose **单图**(encode 全 8 行 + 噪声块 +
            // selector 固定几何;多编行下轮被自块写/重编覆盖,幂等安全)。
            // 输入槽 = tokens/anchor/enc_pos/enc_slots/prop_pos/prop_slots/
            // kv_len;烘焙叶 = verify taps 槽 + 草稿池叶。prefix_len 曾为
            // 宿主烘焙标量,已改核内读 kv_len 张量(契约 5)。
            if let Some(crate::running::Drafter::DFlash2(d2)) = drafter.as_ref() {
                let tap_blks: Vec<owl_iface::contract::Bytes> = (0..5)
                    .map(|i| vg.output_block(&format!("tap{i}")).expect("tap 输出槽"))
                    .collect();
                let kv_leaves = pool.dflash_kv_leaves().expect("dflash 池");
                let bt_leaf = pool.bt_leaf_flat();
                let model_df = Arc::clone(&loaded.model);
                let rp_df = draft_rope.as_ref().expect("draft rope").clone();
                let d2g = Arc::clone(d2);
                let hidd = d2.hidden();
                match extra_faces.pop() {
                    Some(gface) => {
                        let (dg, dout) = GraphPlan::plan(
                            gface,
                            GraphPlanDesc {
                                inputs: vec![
                                    InputSlot::f32("tokens", 8),
                                    InputSlot::f32("anchor", 1),
                                    InputSlot::f32("enc_pos", 8),
                                    InputSlot::f32("enc_slots", 8),
                                    InputSlot::f32("prop_pos", 8),
                                    InputSlot::f32("prop_slots", 8),
                                    InputSlot::f32("kv_len", 1),
                                ],
                                outputs: vec![
                                    OutputSlot::f32("drafts", &[7 * (16 * 16 + 1)]),
                                    OutputSlot { name: "e0".into(), shape: vec![1], dtype: Dtype::BF16 },
                                    OutputSlot { name: "e1".into(), shape: vec![1], dtype: Dtype::BF16 },
                                    OutputSlot { name: "e2".into(), shape: vec![1], dtype: Dtype::BF16 },
                                    OutputSlot { name: "e3".into(), shape: vec![1], dtype: Dtype::BF16 },
                                    OutputSlot { name: "e4".into(), shape: vec![1], dtype: Dtype::BF16 },
                                ],
                                capture: std::env::var_os("OWL_NO_GRAPH").is_none(),
                            },
                            move |sc: &PlanCtx| -> Result<()> {
                                let taps: Vec<owl_models::tensor::TensorOps> = tap_blks
                                    .iter()
                                    .map(|b| owl_models::tensor::TensorOps::of_block(b.id, Dtype::F16, vec![8, hidd]))
                                    .collect();
                                let tokens = sc.input("tokens")?;
                                let anchor = sc.input("anchor")?;
                                let enc_pos = sc.input("enc_pos")?;
                                let enc_slots = sc.input("enc_slots")?;
                                let prop_pos = sc.input("prop_pos")?;
                                let prop_slots = sc.input("prop_slots")?;
                                let kv_len = sc.input("kv_len")?;
                                let bt = bt_leaf.clone();
                                let enc_kvs: Vec<owl_models::module::KvBuffers> = kv_leaves
                                    .iter()
                                    .map(|(k, v)| owl_models::module::KvBuffers {
                                        k_cache: k.clone(),
                                        v_cache: v.clone(),
                                        slots: enc_slots.clone(),
                                        kv_lens: enc_pos.clone(),
                                        block_tables: bt.clone(),
                                    })
                                    .collect();
                                let prop_kvs: Vec<owl_models::module::KvBuffers> = kv_leaves
                                    .iter()
                                    .map(|(k, v)| owl_models::module::KvBuffers {
                                        k_cache: k.clone(),
                                        v_cache: v.clone(),
                                        slots: prop_slots.clone(),
                                        kv_lens: kv_len.clone(),
                                        block_tables: bt.clone(),
                                    })
                                    .collect();
                                let ctx = owl_models::module::ForwardCtx::minimal(8);
                                let mem = d2g.project_memory(&taps, &ctx);
                                let enc_roots = d2g.encode_kv(&mem, &enc_pos, &enc_kvs, &rp_df, &ctx);
                                for (i, r) in enc_roots.iter().enumerate() {
                                    sc.output(format!("e{i}"), r)?;
                                }
                                let (_h, drafts, _s, _l) = d2g.propose_block(
                                    &tokens, &prop_pos, &anchor, &prop_kvs, &rp_df,
                                    &model_df.embed, &kv_len, &ctx, None,
                                );
                                // drafts = select 单缓冲前 7 元(offset 0;输出槽
                                // 形状 = 整缓冲 —— dtoh 整父块契约)
                                sc.output("drafts", &drafts)
                            },
                        )
                        .await?;
                        match dout {
                            crate::graph_plan::PlanOutcome::Captured => {
                                eprintln!("[boot] dflash propose 图捕获(encode+propose 单图,固定 8 行)");
                                dflash_graph = Some(dg);
                            }
                            other => eprintln!("[boot] dflash propose 图降级({other:?})→ eager propose"),
                        }
                    }
                    None => eprintln!("[boot] dflash propose 图 face 耗尽 → eager propose"),
                }
            }

            // E5-M5:propose 桶形图族(index = m ∈ 0..=depth —— 全接受
            // m=depth 也要出下轮草稿;extend m+1 行 + k-1 链步;
            // hidden = verify hid 槽前缀视图;lm_head/链全部入图)。
            if let Some(crate::running::Drafter::Mtp(mtp)) = drafter.as_ref() {
                let dd = dims;
                let key_dim = dims.nk * dims.hk;
                let value_dim = dims.nv * dims.hv;
                for m in 0..=spec_depth {
                    let m1 = m + 1;
                    let model_p = Arc::clone(&loaded.model);
                    let mtp = std::sync::Arc::clone(mtp);
                    let rp_p = loaded.rope.clone();
                    let kv_leaf = pool.mtp_kv_leaf().expect("mtp 池");
                    let bt_leaf = pool.bt_mtp_leaf().expect("bt_mtp 槽");
                    let hid = hid_blk.clone();
                    let env_p = env;
                    let vocab = loaded.model.vocab_size();
                    let nb = pool.nb;
                    let mut pins = vec![InputSlot::f32("tok_ext", m1), InputSlot::f32("pos_ext", m1), InputSlot::f32("slots_ext", m1), InputSlot::f32("lens_ext", m1)];
                    for j in 0..spec_depth - 1 {
                        pins.push(InputSlot::f32(format!("c{j}_pos"), 1));
                        pins.push(InputSlot::f32(format!("c{j}_slots"), 1));
                        pins.push(InputSlot::f32(format!("c{j}_lens"), 1));
                    }
                    pins.push(InputSlot::f32("seq_cu", 2).init(vec![0.0, m1 as f32]));
                    let pface = extra_faces.pop().expect("extra face");
                    let (pg, pout) = GraphPlan::plan(
                        pface,
                        GraphPlanDesc {
                            inputs: pins,
                            outputs: vec![OutputSlot { name: "drafts".into(), shape: vec![spec_depth, 1], dtype: Dtype::F32 }],
                            capture: std::env::var_os("OWL_NO_GRAPH").is_none(),
                        },
                        move |sc: &PlanCtx| -> Result<()> {
                            // PlanCtx 输入声明形 [1, len] → flatten(行主前缀)
                            let tok_ext = sc.input("tok_ext")?.reshape(vec![m1]);
                            let pos_ext = sc.input("pos_ext")?.reshape(vec![m1]);
                            let slots_ext = sc.input("slots_ext")?.reshape(vec![m1]);
                            let lens_ext = sc.input("lens_ext")?.reshape(vec![m1]);
                            let seq_cu = sc.input("seq_cu")?;
                            let bt = bt_leaf.clone();
                            let kv_ext = owl_models::module::KvBuffers {
                                k_cache: kv_leaf.0.clone(),
                                v_cache: kv_leaf.1.clone(),
                                slots: slots_ext.clone(),
                                kv_lens: lens_ext.clone(),
                                block_tables: bt.clone(),
                            };
                            let hid_view =
                                owl_models::tensor::TensorOps::of_block(hid.id, dd.dtype, vec![m1, dd.hidden]);
                            let mut chain: Vec<(owl_models::tensor::TensorOps, owl_models::module::KvBuffers)> = Vec::new();
                            for j in 0..spec_depth - 1 {
                                let pos_c = sc.input(&format!("c{j}_pos"))?;
                                let slots_c = sc.input(&format!("c{j}_slots"))?;
                                let lens_c = sc.input(&format!("c{j}_lens"))?;
                                chain.push((
                                    pos_c,
                                    owl_models::module::KvBuffers {
                                        k_cache: kv_leaf.0.clone(),
                                        v_cache: kv_leaf.1.clone(),
                                        slots: slots_c.clone(),
                                        kv_lens: lens_c.clone(),
                                        block_tables: bt.clone(),
                                    },
                                ));
                            }
                            let chain_refs: Vec<(&owl_models::tensor::TensorOps, &owl_models::module::KvBuffers)> =
                                chain.iter().map(|(p, k)| (p, k)).collect();
                            let droot = mtp.propose_ext(
                                &tok_ext,
                                &hid_view,
                                &pos_ext,
                                &kv_ext,
                                &slots_ext,
                                &lens_ext,
                                &seq_cu,
                                &chain_refs,
                                &model_p.embed,
                                &rp_p,
                                vocab,
                                env_p,
                            );
                            sc.output("drafts", &droot)
                        },
                    )
                    .await?;
                    match pout {
                        crate::graph_plan::PlanOutcome::Captured => {
                            eprintln!("[boot] propose 桶形图捕获(m={m})");
                            propose_graphs.push(pg);
                        }
                        other => eprintln!("[boot] propose 图降级(m={m},{other:?})→ eager propose"),
                    }
                }
            }
        };

        Ok(RunningEngine {
            probes: crate::running::StepProbes::from_env(),
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
            env,
            spec_depth,
            spec_mode,
            drafter,
            blocks_mtp,
            verify_graph,
            fold_graphs,
            propose_graphs,
            dflash_graph,
            spec_drafts_host: None,
            draft_rope,
            dflash_tap_count: if matches!(spec_mode, crate::running::SpecMode::DFlash2)
                && std::env::var_os("OWL_DFLASH_NOTAPS").is_none()
            {
                5
            } else {
                0
            },
            dflash_mem_dumped: false,
            dflash_mem_dumped2: false,
            dflash_hid_dumped: false,
            boot_seq: BOOT_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1,
            spec_seed_hidden: None,
            spec_stats: crate::running::SpecStats::default(),
            spec_snap_valid: false,
            pending_events: std::collections::VecDeque::new(),
        })
    }
}
