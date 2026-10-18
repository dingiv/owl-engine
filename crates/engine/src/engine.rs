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

use owl_cuda::{DeviceSelector, GpuClient};
use owl_iface::contract::{DeviceClient, ModelError};

use crate::graph_fleet::{FleetBoot, GraphFleet};
use crate::loader::{LoadedModel, ModelLoader};
use crate::running::RunningEngine;
use crate::spec_runtime::{SpecBootInput, SpecMode};
use crate::state::{ModelDims, StatePool};

type Result<T> = std::result::Result<T, ModelError>;

/// boot 旋钮(显式依赖律,2026-10-10:环境变量是隐式依赖,引擎内部
/// **零 env 读取** —— env 派生数据由入口(server config)/ 测试用例经
/// 本结构显式传入;[`EngineKnobs::from_env`] 是唯一的入口侧构造器,
/// [`Default`] = 生产基线)。
#[derive(Clone, Debug)]
pub struct EngineKnobs {
    /// 步级探针(诊断面;Default = 全关)
    pub probes: crate::running::StepProbes,
    /// 块池容量覆写(None = 2 × 单会话容量;原 OWL_POOL_TOKENS)
    pub pool_tokens: Option<usize>,
    /// 草稿域(深度/哑草稿/家族目录/池量化/verify 诊断)
    pub draft: DraftCfg,
    /// 前缀缓存(缺省开;原 OWL_PREFIX_CACHE≠"0")
    pub prefix_cache: bool,
    /// **宿主层预算 MiB**(状态存储三层的中层;缺省 **0 = 关层**,REQ C4)。
    /// 环境变量 `OWL_HOST_CACHE_MB`;S3-3b(回灌)接线前开了也没用
    pub host_cache_mb: usize,
    /// eager 直发,跳过图捕获(原 OWL_NO_GRAPH;**总闸**)
    pub no_graph: bool,
    /// **prefill 图档位**(P1;`[graph] prefill_bucket`;缺省 `off` = 今天行为)
    pub prefill_bucket: owl_shared::config::PrefillBucket,
    /// **FI plan 图档瓦**(`[dispatch] fi_cuda_graph`;2026-10-18 图档单图立案):
    /// 开 ⇒ 形状对 ctx 全域恒定(可烘图的先决条件)。装配期经
    /// `owl_kernels::family::flashinfer::configure_plan_env` 投递给核。
    pub fi_cuda_graph: bool,
    /// **decode 单图烘 plan 上界**(`[dispatch] fi_plan_bake`;None = 现状 piecewise)
    pub fi_plan_bake: Option<usize>,
    /// TS 探针槽(原 OWL_TS_PROBE)
    pub ts_probe: bool,
    /// GDN 槽数覆写(缺省 8;原 OWL_GDN_SLOTS)
    pub gdn_slots: usize,
    /// spec 快照轮回滚缓冲槽数(缺省 4;原 OWL_SNAP_MAX)
    pub snap_max: usize,
    /// VRAM 预算目标占比(缺省 0.97;原 OWL_VRAM_TARGET)
    pub vram_target: f64,
    /// VRAM 预留 MiB(缺省 1024;原 OWL_VRAM_RESERVE_MB)
    pub vram_reserve_mb: u64,
    /// 裸续写输出,绕过 chat 模板(原 OWL_RAW_COMPLETION)
    pub raw_completion: bool,
    /// 采样开关(缺省开;OWL_SAMPLER=greedy → false)
    pub sampler_enabled: bool,
    /// 采样参数(temp/topk/topp/rep;缺省 = 反循环服务档)
    pub sampler: crate::sampler::SamplerCfg,
    /// GDN 画像 dump 全槽(原 OWL_GDN_DUMP_ALL)
    pub gdn_dump_all: bool,
    /// prefill 逐层 checksum 二分(原 OWL_PF_BISECT)
    pub pf_bisect: bool,
    /// prefill 指定层五站 stage tap(原 OWL_PF_STAGES)
    pub pf_stages: Option<usize>,
    /// prefill 尾块校验(原 OWL_PF_FIN_CHECK)
    pub pf_fin_check: bool,
    /// 设备诊断/调优选项(srv_timing/cap_prof/launch_sync/free_legacy/
    /// graph_flags/capture_slab_mb;原散点 OWL_* 直读 →
    /// owl_cuda::DiagOpts 一体携带,入口经 DiagOpts::from_env 填充)
    pub cuda: owl_cuda::DiagOpts,
    /// 解释器执行环境(硬件档/KV 策略/量化/分派旋钮;Default = 生产基线)
    pub env: owl_models::env::EnvProvider,
}

impl Default for EngineKnobs {
    fn default() -> Self {
        Self {
            probes: crate::running::StepProbes {
                // 命中面/调试打印:env 覆盖(实测排障用 `OWL_PROBES_DEBUG=1`)
                debug: owl_shared::env_reader::flag("OWL_DEBUG"),
                ..Default::default()
            },
            pool_tokens: None,
            draft: DraftCfg::default(),
            prefix_cache: true,
            host_cache_mb: 0,
            no_graph: false,
            prefill_bucket: owl_shared::config::PrefillBucket::Off,
            fi_cuda_graph: false,
            fi_plan_bake: None,
            ts_probe: false,
            gdn_slots: 8,
            snap_max: 4,
            vram_target: 0.97,
            vram_reserve_mb: 1024,
            raw_completion: false,
            sampler_enabled: true,
            sampler: crate::sampler::SamplerCfg::default(),
            gdn_dump_all: false,
            pf_bisect: false,
            pf_stages: None,
            pf_fin_check: false,
            cuda: owl_cuda::DiagOpts::default(),
            env: owl_models::env::EnvProvider::default(),
        }
    }
}

impl EngineKnobs {
    /// **入口侧组装器**(server config / 测试用例专用):自全量配置对象
    /// [`owl_shared::config::OwlConfig`] 显式组装,引擎内部零 env 读取。
    /// env → OwlConfig 的映射统一在 config loader(总账唯一登记点);
    /// 本函数只做 module → 旋钮的字段搬运。
    pub fn from_config(cfg: &owl_shared::config::OwlConfig) -> Self {
        Self {
            probes: crate::running::StepProbes::from_config(cfg),
            // 池容量:env 覆盖(实测调压必需)> preset > 默认 2×单会话(§22 实测)
            pool_tokens: owl_shared::env_reader::parse("OWL_POOL_TOKENS").or(cfg.pool.pool_tokens),
            draft: DraftCfg {
                depth: cfg.spec.depth,
                dumb: cfg.spec.dumb,
                dflash2_dir: cfg.model.dflash2_dir.clone(),
                kv_quant: if cfg.spec.draft_kv_fp8 {
                    owl_models::env::KvQuant::Fp8E4M3
                } else {
                    owl_models::env::KvQuant::None
                },
                notaps: cfg.spec.notaps,
                tapdecl: cfg.spec.tapdecl,
            },
            prefix_cache: cfg.pool.prefix_cache,
            // 宿主层预算(S3-3):环境的 opt-in 旋钮(不进 OwlConfig —— 未接线前
            // 开它只会让命中/驱逐响亮拒;接线后按档进 config 再谈)
            host_cache_mb: owl_shared::env_reader::parse("OWL_HOST_CACHE_MB").unwrap_or(0),
            no_graph: cfg.graph.no_graph,
            prefill_bucket: cfg.graph.prefill_bucket,
            fi_cuda_graph: cfg.dispatch.fi_cuda_graph,
            // 哨兵:0 = 取 max_seq(FI 调度表的图档上界 = 单序列 ctx 的天然上界;
            // 这样档位改 max_seq 时上界跟着走,不必两处手改)
            fi_plan_bake: cfg.dispatch.fi_plan_bake.map(|b| if b == 0 { cfg.runtime.max_seq } else { b }),
            ts_probe: cfg.dispatch.ts_probe,
            gdn_slots: cfg.pool.gdn_slots,
            // 快照池容量:env 覆盖(实测调压;命中面要求边界快照在场)
            snap_max: owl_shared::env_reader::parse("OWL_SNAP_MAX").unwrap_or(cfg.pool.snap_max),
            vram_target: cfg.pool.vram_target,
            vram_reserve_mb: cfg.pool.vram_reserve_mb,
            raw_completion: cfg.probes.raw_completion,
            sampler_enabled: cfg.sampling.mode == owl_shared::config::SamplerMode::Sampling,
            sampler: crate::sampler::SamplerCfg::from_sampling(&cfg.sampling),
            gdn_dump_all: cfg.probes.gdn_dump_all,
            pf_bisect: cfg.probes.pf_bisect,
            pf_stages: cfg.probes.pf_stages,
            pf_fin_check: cfg.probes.pf_fin_check,
            cuda: owl_cuda::DiagOpts::from_config(cfg),
            env: owl_models::env::EnvProvider::from_dispatch(&cfg.dispatch),
        }
    }
}

/// 草稿域配置(spec 草稿器;深度/家族目录/池量化/verify 诊断 ——
/// 原 knobs 顶层六散字段收编;量化用 KvQuant 枚举,bool 和稀泥废除)
#[derive(Clone, Debug)]
pub struct DraftCfg {
    /// 投机深度(0 = off;上限 3 / DFlash2 7)
    pub depth: usize,
    /// 哑草稿诊断面(无草稿器时)
    pub dumb: bool,
    /// DFlash2 草稿检查点目录(⚠️ W4A16 量化家族 syvai 系)
    pub dflash2_dir: Option<String>,
    /// 草稿池 KV 存储(None = f16;Fp8E4M3 = 偷显存档,原 draft_kv_fp8 布尔)
    pub kv_quant: owl_models::env::KvQuant,
    /// verify taps 退回 last_hidden(排查开关)
    pub notaps: bool,
    /// TAPDECL 二分臂(声明期 tapped 不挂输出槽)
    pub tapdecl: bool,
}

impl Default for DraftCfg {
    fn default() -> Self {
        Self {
            depth: 0,
            dumb: false,
            dflash2_dir: None,
            kv_quant: owl_models::env::KvQuant::None,
            notaps: false,
            tapdecl: false,
        }
    }
}

/// 引擎构造参数(纯资源面 + boot 旋钮,零模型语义)
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
    /// boot 旋钮(显式依赖;入口/测试填充,Default = 生产基线)
    pub knobs: EngineKnobs,
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
        // 诊断下沉口(boot 一次组装,显式下传):日志 + 发射/回收计时
        let face = GpuClient::spawn_with(
            DeviceSelector::Ordinal(cfg.device_ordinal),
            cfg.knobs.cuda.clone(),
            cfg.knobs.probes.debug,
            owl_shared::log::sink(),
        )
            .map_err(|e| ModelError::Msg(format!("engine: gpu 绑定失败 {e:?}")))?;
        Self::on(cfg, face)
    }
}

impl<D: DeviceClient + 'static> Engine<D> {
    /// 注入 face(CPU 测试/自定义后端;唯一构造机制,new 是其 GPU 特化)
    pub fn on(cfg: EngineConfig, face: D) -> Result<Self> {
        // 注:`[graph] prefill_bucket` 的**真接线在 [`Self::run`]**(桶捕获需要
        // 模型与状态池;静止点律 ⇒ 只在 boot 做)。此处不再拒启(2026-10-15
        // M2b 落地;此前的"配了没实现就拒"临时门已被真接线取代)。
        Ok(Self { face, cfg })
    }

    /// 模型加载器(借用引擎 face;装载即校验,错误在 load 边界落地)
    pub fn loader(&mut self) -> ModelLoader<'_, D> {
        ModelLoader::new(&mut self.face)
    }

    /// 模型交引擎:状态块/槽/Session 组建 + warmup —— 仍不跑任何 turn,
    /// 返回执行态引擎(请求入口)。
    // A3 同步隔离:引擎单线程模型(每卡一线程);Arc 仅作共享句柄,
    // 不跨线程发送 —— clippy 的 Send+Sync 建议不适用。
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn run(mut self, loaded: LoadedModel) -> Result<RunningEngine<D>> {
        let t_run = std::time::Instant::now();
        // **投机深度定形门**(2026-10-16):verify 块 = `depth + 1` 行,而首轮
        // 拼接核的输入槽数是**声明面事实**(`CONCAT_ROWS_SLOTS` = 8)⇒ depth 超界
        // 会在**首个 verify 步**才爆(且那时树已经 pad 到错槽数)。
        // 放在**任何设备动作之前**(连草稿器都不加载)⇒ 坏配置毫秒级拒。
        if self.cfg.knobs.draft.depth >= owl_models::kernel::CONCAT_ROWS_SLOTS {
            return Err(owl_iface::contract::ModelError::Msg(format!(
                "[spec] depth = {} 超界:verify 行数 = depth+1 必须 ≤ {}(首轮拼接槽数);\
                 请调小 depth(或关 spec 走 dumb 草稿)",
                self.cfg.knobs.draft.depth,
                owl_models::kernel::CONCAT_ROWS_SLOTS
            )));
        }
        let s = self.cfg.max_seq_tokens;
        let (hq, hkv, hd) = loaded.spec.full_heads;
        let (nk, hk, nv, hv) = loaded.spec.gdn_heads;
        // ⚠️ **已知缺口(§18.5 挂账①)**:CPU/mock 档若要跑引擎真前向,
        // `dtype` 不能简单归一为 F32 —— `kv_paged_policy(dtype)` 与
        // attn v2 scratch 注入都以**模型 dtype** 为前提(F32 ⇒ 无分页策略
        // ⇒ 走 legacy v1 臂 ⇒ attention.rs 的退役 panic)。所需修法:
        // 让分页策略/scratch 与执行域 dtype 解耦(单独一刀),不在 mock 刀里扫。
        let dims = ModelDims {
            hq, hkv, hd, nk, hk, nv, hv,
            hidden: loaded.spec.hidden,
            dtype: loaded.spec.dtype,
        };

        // 解释器执行环境(EnvProvider;E1.5 抽象):旋钮携带(入口经
        // from_config 填充)+ 硬件档自 iface op_env 合入(被动律:引擎 = 环境事实来源)
        let mut env = self.cfg.knobs.env;
        if let Some(oe) = self.face.op_env() {
            env.hw = owl_models::env::HwEnv::Cuda(oe.hw.arch);
        }
        // 模型几何事实合入(2026-10-13 P4:算子准入的几何门对账面)——
        // 与 hw 同律:引擎 = 环境事实来源,算子层零探测
        env.kv.head_dim = hd;
        env.kv.grp = hq.checked_div(hkv).unwrap_or(0);
        env.quant = loaded.quant_plan;
        // FI 面启用(量化恒 = 主池;影子池已删,本值只作"启用 + 池量化"携带)
        // FI 面启用条件 = **能力条件**(f16 池 + paged 页表),不是开关:
        // 2026-10-13 起配置面无 flashinfer 选项,FI 恒启用(env.attn.fi 仅门/诊断用)
        let fi_quant = if env.attn.fi
            && env.kv.paged
            && loaded.spec.dtype == owl_models::tensor::Dtype::F16
        {
            Some(env.kv.quant)
        } else {
            None
        };

        // spec 域装载(spec_runtime.rs:模式三态裁决 + 草稿器 + 草稿 rope;
        // E5-M2b/C7 —— depth>0 且检查点有 mtp.* → Mtp;dumb 诊断位 → Dumb;
        // 否则 Off 回落 DecodeBatch,无草稿器不白发 verify 税)
        let spec = crate::spec_runtime::load_spec_runtime(
            &mut self.face,
            SpecBootInput {
                depth_raw: self.cfg.knobs.draft.depth,
                dflash2_dir: self.cfg.knobs.draft.dflash2_dir.as_deref(),
                mtp_dir: loaded.mtp_dir.as_deref(),
                dumb: self.cfg.knobs.draft.dumb,
                draft_kv_fp8: self.cfg.knobs.draft.kv_quant == owl_models::env::KvQuant::Fp8E4M3,
                notaps: self.cfg.knobs.draft.notaps,
            },
        )
        .await?;

        // B6.2:主 KV 池 fp8 承载(draft_kv_fp8 / kv.fp8 → env.kv.quant)
        // KV 池清单门(收口律机器门;E1 2026-10-17 改为**声明面投影**):
        // 期望集从每个算子的 `POOL` 声明(`池名/布局/dtype`)投影出来,逐个
        // 核 `.cu` 源在场;缺声明/缺源 = 装配失败(池 dtype 改造漏网即启动
        // 即拦,不等 E2E 红灯)。原手抄核名清单已删(新池核忘写清单 ⇒ 门
        // 静默变窄的那种形状)。
        owl_models::kernel::kv_manifest_gate(
            owl_models::kernel::kv_pool_dtype(env.kv.quant, dims.dtype),
            env.attn.fi,
        )?;
        if env.kv.quant == owl_models::env::KvQuant::Fp8E4M3 {
            owl_shared::log_info!("boot", "KV 池 fp8 e4m3(容量减半/ctx 翻倍;读核 *_fp8 变体)");
        }

        // 状态块(StatePool::alloc,零初始化;GDN 段每 turn 开始按格重置)。
        // KV 布局表驱动(kv_paged_policy;REQ-HW-01);块池容量 E2b:
        // 默认 = 2 × 单会话容量(两会话满载共存),knobs.pool_tokens 覆写
        let pool_tokens = self.cfg.knobs.pool_tokens.unwrap_or(2 * s);
        let plan = crate::state::PoolPlan {
            seq_tokens: s,
            pool_tokens,
            kv: env.kv.quant,
            fi_quant,
            draft_kv: if self.cfg.knobs.draft.kv_quant == owl_models::env::KvQuant::Fp8E4M3 {
                owl_models::env::KvQuant::Fp8E4M3
            } else {
                owl_models::env::KvQuant::None
            },
            spec_mode: spec.mode,
            spec_depth: spec.depth,
            xqa_verify: self.cfg.knobs.env.attn.verify_xqa,
            xqa_decode: self.cfg.knobs.env.attn.decode_xqa,
            xqa_prefill: self.cfg.knobs.env.attn.xqa_prefill,
            gdn_slots: self.cfg.knobs.gdn_slots,
            snap_max: self.cfg.knobs.snap_max,
            vram_target: self.cfg.knobs.vram_target,
            vram_reserve_mb: self.cfg.knobs.vram_reserve_mb,
            dump_all: self.cfg.knobs.gdn_dump_all,
            // prefill 桶 padding 的 dump 页(M2b):档位非 off 才多分配那 1 页
            prefill_dump_page: self.cfg.knobs.prefill_bucket
                != owl_shared::config::PrefillBucket::Off,
            gdn_dbg: self.cfg.knobs.probes.gdn_dbg,
        };
        let pool = StatePool::alloc(&mut self.face, dims, &loaded.spec.layer_types, &plan).await?;
        if let Some(q) = fi_quant {
            owl_shared::log_info!("boot", "FlashInfer prefill 面启用(FI 直读主池 kNHD 单池;quant={:?})",
                q
            );
        }

        // 会话账 + 双块账房(E2b/E2c;session.rs:target 链前缀缓存挂载 +
        // MTP 链独立页池 —— 复合构造一次落位)
        let bank = crate::session::SessionBank::new(
            crate::session::BankPlan {
                nb: pool.nb,
                page: pool.page,
                block_bytes: pool.block_bytes(),
                mtp_block_bytes: pool.mtp_block_bytes(),
                paged: pool.paged,
                mtp: spec.mode == SpecMode::Mtp,
                prefix_cache: self.cfg.knobs.prefix_cache,
                // 宿主层预算(REQ C4:默认关;0 = 关层)—— 开它之前先做 S3-3b(回灌),
                // 否则命中/驱逐遇宿主格会响亮拒(设计卷 §10.4 律⑤)
                host_bytes: self.cfg.knobs.host_cache_mb << 20,
            },
            pool.pool_layout(0),
            pool.mtp_pool_layout(0),
        )?;

        // 图族总装配(graph_fleet.rs:① 主会话图(warmup+捕获)→
        // ② verify 桶形图 → ③ fold/DFlash2/MTP 族;face 句柄派生
        // vface + extra_faces(pop 消费桶))
        let vface = self.face.face_clone();
        // E5-M5:fold/propose 桶形图族的 face 句柄(depth × 2)
        let extra_faces: Vec<D> = (0..2 * spec.depth.max(1) + 2).filter_map(|_| self.face.face_clone()).collect();
        let graphs = GraphFleet::assemble(
            self.face,
            vface,
            extra_faces,
            FleetBoot {
                env,
                dims,
                pool: &pool,
                model: std::sync::Arc::clone(&loaded.model),
                rope: loaded.rope.clone(),
                spec: &spec,
                knobs: &self.cfg.knobs,
            },
            t_run,
        )
        .await?;

        // ── DFlash2 spec × 无 verify 图 = **boot 拒启**(2026-10-14 修)──
        // DFlash2 的 verify 路径**硬依赖 verify 图**(taps 取自图的输出槽,
        // 见 `exec/spec.rs`)。捕获失败 / `[graph] no_graph = true`(含
        // eager 诊断档)下 verify 图 = None ⇒ 原行为是**首个 spec 轮
        // panic**(`dflash 模式要求 verify 图(taps 输出)`)→ 引擎线程退役
        // (实测:第二个请求收到「引擎退役」)。即"捕获失败 → eager 逃生"
        // 这条路径对 DFlash2 既不可用、也**死得太晚**(跑到一半才炸)。
        // ⇒ 收口为**装配期拒启**:要跑该档就显式 `[spec] depth = 0`
        // 或 `[spec] notaps = true`(退 last_hidden,不需要图)。
        // 注:`dflash_tap_count > 0` ⟺ (DFlash2 模式 && !notaps)。
        if spec.mode == SpecMode::DFlash2
            && spec.dflash_tap_count > 0
            && graphs.verify.is_none()
        {
            return Err(owl_iface::contract::ModelError::Msg(
                "DFlash2 spec 需要 verify 图(taps 输出),但本档未建图\
                 ([graph] no_graph = true 或捕获失败)。请二选一:去掉 no_graph,\
                 或显式关闭该档([spec] depth = 0;或 [spec] notaps = true 退 last_hidden)"
                    .to_string(),
            ));
        }
        Ok(RunningEngine {
            probes: self.cfg.knobs.probes,
            tok: loaded.tokenizer,
            cfg: self.cfg,
            pool,
            model: loaded.model,
            rope: loaded.rope,
            env,
            sched_acc_ns: 0,
            sched_cnt: 0,
            boot_seq: BOOT_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1,
            turns: crate::turn::TurnLedger::new(),
            bank,
            graphs,
            spec,
        })
    }
}
