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
    /// **选择层索引采集(B 轴 B-0;缺省关)**。`OWL_INDEX_PROBE=1` / `[kvmem].index`。
    ///
    /// 开则:块边界(与 GDN 快照同处)回读该页 K → host 逆 RoPE 求 Mean-K →
    /// 登记进 store 的索引账(`AssetKind::Index`,host 层)。
    /// ⚠️ 一期是**回读**形态(每页 ~3MB D2H,一次性建索引);核内累加是后续优化。
    /// `select_decode` 蕴含它(打分的前置件)。
    pub index_probe: bool,
    /// **选择层覆盖率探针 —— 诊断面**(缺省关)。`OWL_COVERAGE_PROBE=1`。
    ///
    /// 开则额外做**真重诊断**:K/V 全量回读 + host 逐元素解码 +
    /// `true_block_mass`(逐 query 行 × (层,头) 扫**全 ctx** 的因果 softmax)
    /// + 论文口径保真对照 + 方案 A/B 对照 + 覆盖率曲线/上界报告。
    /// 单边界的成本**含 `ctx` 因子** ⇒ 全军一遍是 **O(ctx²)**
    /// (27B/8k 实测单次 6.3s = 墙钟 32%)⇒ **只在末个边界跑一次**
    /// (`OWL_COV_ANALYZE_EVERY` 可改)。诊断档另有一个副作用:
    /// prefill 桶图**让位 eager**(与 `pf_bisect` 同律,响亮告知一次)。
    ///
    /// 🔴 **测性能不要把生产面跟它一起开** —— 它会把整跑拖成诊断账
    /// (生产面只需 `select_decode`)。
    pub coverage_probe: bool,
    /// **选择层接进 decode —— 生产总开关**(缺省关)。`OWL_SELECT_DECODE=1` / `[kvmem].select_decode`。
    ///
    /// 开则:召回点之后的本 turn decode 步,注意力只读**视图页表**
    /// (选中块 ++ 新生成尾部;方案 A = 原位置,**不做 re-RoPE**),
    /// `sel_pt`/`sel_len` 两个图输入槽每步写;召回点之前/无视图 ⇒ 写全量链
    /// (与不开时逐字等价)。
    /// **前置**:图档(视图走输入槽)+ `index_probe` + 宿主层(超卖档)。
    /// 不依赖 `coverage_probe` —— 那是诊断面。
    /// XQA 裸 decode 臂与视图**不共戴**(页表语义两套)⇒ 当前需 `decode_xqa=0`。
    pub select_decode: bool,
    /// **B-3e-2:部分宿主驻留**(缺省关)。`OWL_SELECT_PARTIAL=1` / `[kvmem].select_partial`。
    ///
    /// 开则:每个召回点把**被踢出工作集**的链块 `decref` + 逐块下沉到宿主
    /// (会话表写占位),并把选中的宿主块回灌回来;prefill 的"过去"改走
    /// **子集页链**(否则会读到占位符)。前提:`select_decode`
    /// + 宿主层(`OWL_HOST_CACHE_MB` / `OWL_SNAP_HOST_MB`)。
    /// 这正是**低显存档位**(池 < ctx)成立的那一件 —— 不开它,
    /// 链无法超出池容量。
    pub select_partial: bool,
    /// **驱逐策略**(B-4;缺省 `lru`)。`OWL_EVICT_POLICY=lrfu[:λ]`。
    ///
    /// - `lru` = 现役基线(`last_use` 升序);
    /// - `lrfu` = **attention-aware LRFU**:召回点选中的块 keep-score 抬升
    ///   (`KS ← λ·KS + (1−λ)·hit`)⇒ 出序时"又冷又没被注意力用过"的先下沉。
    ///   我们只**下沉**不丢内容 ⇒ 该策略决定的是"谁的设备副本先让位"。
    pub evict_policy: owl_store::EvictPolicyKind,
    /// **分析边界目标 ctx**(诊断;0 = 最后一个边界)。`[kvmem].analyze_at_ctx`。
    pub coverage_analyze_at: usize,
    /// **召回预算**(块数;缺省 64,`OWL_SELECT_BLOCKS`)。视图 = 选中的这些块
    /// ⇒ 这是"选择层丢多少"的唯一旋钮。
    pub select_blocks: usize,
    /// 保底 sink / recent(块;缺省 2 / 4)
    pub select_sink: usize,
    pub select_recent: usize,
    /// **召回节奏**(块边界):每 k 个边界做一次**完整召回**(打分+按分选择),
    /// 其余边界**只采索引、不干预驻留**。缺省 **1 = 每个边界都完整召回**
    /// (旧行为,逐位不变)。`OWL_RECALL_EVERY` / `[kvmem].recall_every`。
    ///
    /// 为什么要有这一档:单次打分是 O(块数) **线性**,但"每块边界都打"⇒ 一次
    /// prefill 累计 **O(块数²)** —— 262k 下 ≈57s(release)/边界数 2048。
    /// 参考实现把这条曲线压回线性,靠的是**每请求只召回一次**。
    ///
    /// 廉价档的语义 = **两次完整召回之间不改变选择**(驻留交给 A 轴,
    /// 下一次完整召回时一次性对齐)⇒ 搬运量由"选择变了多少"决定,
    /// 而不是由"过了多少个边界"决定。
    /// ⚠️ 第一版曾让廉价档走"近期选择 + 调和",**实测净亏**:
    /// 两份名单互相覆盖 ⇒ 调和 +92% / 下沉回灌翻倍 / 末边界覆盖率读数失真。
    /// **末个边界永远是完整召回**(decode 消费的视图取自它)。
    pub recall_every: usize,
    /// **虚拟化是否开着**(`[kvmem] select_decode && select_partial`;boot 搬运,
    /// 唯一产地 = `KvmemCfg::virtualization_enabled`)。
    /// 用途:**所有解锁类护栏的判据** —— 非 Strict 地板、池小于上下文,
    /// 都只在这条为真时放行。
    pub kvmem_virtualization: bool,
    /// **准入地板是否是显式配置的**(`[kvmem].dense_floor_mib` 给了数)。
    /// 不是 ⇒ boot 在池几何已知后自动填**工作集字节**(见 `run()` 的虚拟化校验)。
    pub dense_floor_explicit: bool,
    /// **状态腿宿主预算 MiB**(S2 §25.2;缺省 **0 = 关层**)。`OWL_SNAP_HOST_MB`。
    ///
    /// 与 [`Self::host_cache_mb`] 同属 store 中层总账的两个**分片**
    /// (KV 块腿 / GDN 快照腿)—— 总账 = 两者之和(`Rel::open` 交叉校验分片
    /// 不超总账)。为什么分开:两条腿的搬运形态与热区完全不同,共用一个数
    /// 只会让一边饿死(实测:全给 KV 腿 ⇒ 快照下沉无位 ⇒ 冷快照被丢)。
    pub snap_host_mb: usize,
    /// **准入满员姿态**(S3b;缺省响亮拒)。`OWL_ADMIT_ON_FULL=reject|queue[:ms]`。
    ///
    /// `Reject` = 准入拒(客户端该降并发/重试,不是压缩上下文);
    /// `Queue { max_wait_ms }` = 留队首等下一次事件重判(超时则拒)。
    pub admission_on_full: owl_store::QueuePolicy,
    /// **per-turn 直接显存地板的缺省值**(S3b;设计卷 §2.4)。
    ///
    /// `Strict` = 全量 dense 驻留(选择层未落地时的唯一诚实值);
    /// 请求可用 `TurnOpts::dense_floor` 逐 turn 覆写 —— 非 `Strict` 当前**响亮拒**
    /// (B 轴未落地,地板语义无从兑现,§23.5)。
    pub dense_floor: owl_store::TurnFloor,
    /// eager 直发,跳过图捕获(原 OWL_NO_GRAPH;**总闸**)
    pub no_graph: bool,
    /// **prefill 图档位**(P1;`[graph] prefill_bucket`;缺省 `off` = 今天行为)
    pub prefill_bucket: owl_shared::config::PrefillBucket,
    /// TS 探针槽(原 OWL_TS_PROBE)
    pub ts_probe: bool,
    /// GDN 槽数覆写(缺省 8;原 OWL_MAX_ACTIVE_TURNS)
    pub max_active_turns: usize,
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
            snap_host_mb: 0,
            index_probe: false,
            coverage_probe: false,
            coverage_analyze_at: 0,
            select_decode: false,
            select_partial: false,
            evict_policy: owl_store::EvictPolicyKind::Lru,
            select_blocks: 64,
            select_sink: 2,
            select_recent: 4,
            recall_every: 1,
            kvmem_virtualization: false,
            dense_floor_explicit: false,
            admission_on_full: owl_store::QueuePolicy::Reject,
            dense_floor: owl_store::TurnFloor::Strict,
            no_graph: false,
            prefill_bucket: owl_shared::config::PrefillBucket::Off,
            ts_probe: false,
            max_active_turns: 8,
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

/// 准入满员姿态解析(`OWL_ADMIT_ON_FULL`):`reject`(缺省)/ `queue` / `queue:30000`。
///
/// 坏值 = **响亮拒启动**(解析失败不许静默回落到缺省 —— 那就变成"配了没生效")。

fn parse_admission_on_full() -> std::result::Result<owl_store::QueuePolicy, String> {
    use owl_store::QueuePolicy;
    let Some(s) = owl_shared::env_reader::parse::<String>("OWL_ADMIT_ON_FULL") else {
        return Ok(QueuePolicy::Reject);
    };
    let s = s.trim().to_ascii_lowercase();
    match s.split_once(':') {
        None if s == "reject" => Ok(QueuePolicy::Reject),
        None if s == "queue" => Ok(QueuePolicy::Queue { max_wait_ms: 30_000 }),
        Some(("queue", ms)) => match ms.parse::<u64>() {
            Ok(max_wait_ms) => Ok(QueuePolicy::Queue { max_wait_ms }),
            Err(_) => Err(format!("OWL_ADMIT_ON_FULL 坏值:{s}(期望 queue:<毫秒>)")),
        },
        _ => Err(format!(
            "OWL_ADMIT_ON_FULL 坏值:{s}(期望 reject | queue | queue:<毫秒>)"
        )),
    }
}

impl EngineKnobs {
    /// **驱逐策略映射**(单源):`[kvmem].evict_policy` → store 词汇。
    /// (分层:shared 不依赖 store ⇒ 映射在本层;与地板映射同律。)
    fn evict_policy_of(cfg: &owl_shared::config::KvmemCfg) -> owl_store::EvictPolicyKind {
        match cfg.evict_policy {
            owl_shared::config::EvictPolicyCfg::Lru => owl_store::EvictPolicyKind::Lru,
            owl_shared::config::EvictPolicyCfg::Lrfu => owl_store::EvictPolicyKind::Lrfu(0.9),
        }
    }

    /// **准入地板映射**(单源):配置三态 → `TurnFloor`。
    /// 配置层只存事实(`virtualization_enabled` + `dense_floor_mib`),
    /// 到 store 词汇的映射在本层(分层:shared 不得依赖 store)。
    fn dense_floor_of(cfg: &owl_shared::config::KvmemCfg) -> owl_store::TurnFloor {
        Self::dense_floor_for(cfg.virtualization_enabled(), cfg.dense_floor_mib)
    }

    /// 地板映射的**唯一实现**(按两份事实入参;探针/测试档直接构 knobs 时
    /// 也走这一条 —— 不得在别处再写一次“虚拟化开 ⇒ Bytes(0)”:那样两处
    /// 迟早对不上)。
    pub fn dense_floor_for(
        virtualization: bool,
        floor_mib: Option<u64>,
    ) -> owl_store::TurnFloor {
        if !virtualization {
            // 虚拟化不在场 ⇒ 唯一诚实值(非 Strict 会被 boot 拦)
            return owl_store::TurnFloor::Strict;
        }
        match floor_mib {
            Some(mib) => owl_store::TurnFloor::Bytes((mib as usize) << 20),
            None => owl_store::TurnFloor::Bytes(0),
        }
    }


    /// 准入满员姿态的等待上限(毫秒;`Reject` ⇒ 0 = 不等)。
    /// **一次运行的条件摘要**(单行;boot 与探针都打)。
    ///
    /// 目的 = **让每个数字自带条件** —— 之前"左一个数字右一个数字对不上",
    /// 就是因为条件散在多个入口/env 里。凡影响行为的旋钮都在这行里;
    /// 不在这行里的就不影响行为。
    pub fn digest(&self) -> String {
        let q = match self.env.kv.quant {
            owl_models::env::KvQuant::None => "f16",
            owl_models::env::KvQuant::Fp8E4M3 => "fp8",
        };
        let ev = match self.evict_policy {
            owl_store::EvictPolicyKind::Lru => "lru".to_string(),
            owl_store::EvictPolicyKind::Lrfu(l) => format!("lrfu:{l}"),
        };
        format!(
            "pool={:?} turns={} snap_max={} host={}M snap_host={}M | kvmem={}[idx={} cov={} view={} part={} budget={}+{}+{} every={} evict={} floor={:?}] | fi={} split={} gdn_chunked={} kv={} graph_off={}",
            self.pool_tokens,
            self.max_active_turns,
            self.snap_max,
            self.host_cache_mb,
            self.snap_host_mb,
            if self.kvmem_virtualization { "ON" } else { "off" },
            self.index_probe as u8,
            self.coverage_probe as u8,
            self.select_decode as u8,
            self.select_partial as u8,
            self.select_blocks,
            self.select_sink,
            self.select_recent,
            self.recall_every,
            ev,
            self.dense_floor,
            self.env.attn.fi as u8,
            self.env.attn.prefill_split as u8,
            self.env.gdn.chunked as u8,
            q,
            self.no_graph as u8,
        )
    }

    pub fn admission_max_wait_ms(&self) -> u64 {
        match self.admission_on_full {
            owl_store::QueuePolicy::Reject => 0,
            owl_store::QueuePolicy::Queue { max_wait_ms } => max_wait_ms,
        }
    }

    /// **入口侧组装器**(server config / 测试用例专用):自全量配置对象
    /// [`owl_shared::config::OwlConfig`] 显式组装,引擎内部零 env 读取。
    /// env → OwlConfig 的映射统一在 config loader(总账唯一登记点);
    /// 本函数只做 module → 旋钮的字段搬运。
    pub fn from_config(cfg: &owl_shared::config::OwlConfig) -> Self {
        Self {
            probes: crate::running::StepProbes::from_config(cfg),
            // ── **配置是唯一来源**(2026-10-19 统一)──
            // 本函数只做 `OwlConfig` → 旋钮的字段搬运,**不读任何 env**
            // (探针档/生产档走同一条路 ⇒ 同一份配置 = 同一套条件)。
            // 测试面只保留"运行器级"env(选卡/资产路径/场景过滤),见
            // `docs/arch/配置面-单一来源.md`。
            pool_tokens: cfg.pool.pool_tokens,
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
            // 宿主层预算(S3-3;`[pool].host_cache_mb`)
            host_cache_mb: cfg.pool.host_cache_mb,
            // 状态腿宿主预算(S2;`[pool].snap_host_mb`;两条腿各有热区,别互相饿死)
            snap_host_mb: cfg.pool.snap_host_mb,
            // ── KVMem 选择层(B 轴):全部来自 `[kvmem]`,零 env ──
            // 生产面(`select_decode`/`select_partial`/索引/预算/节奏)与
            // 诊断面(`coverage_probe`)是**两扇门**:低显存档只开前者。
            // ⚠️ `index` **蕴含于 `select_decode`**(打分的前置件;少了它
            // 选择层无从算分)—— 这条只是在旋钮上取并,不是第二个配置项。
            index_probe: cfg.kvmem.index || cfg.kvmem.select_decode,
            coverage_probe: cfg.kvmem.coverage_probe,
            coverage_analyze_at: cfg.kvmem.analyze_at_ctx,
            select_decode: cfg.kvmem.select_decode,
            select_partial: cfg.kvmem.select_partial,
            evict_policy: Self::evict_policy_of(&cfg.kvmem),
            select_blocks: cfg.kvmem.select_blocks,
            select_sink: cfg.kvmem.select_sink,
            select_recent: cfg.kvmem.select_recent,
            recall_every: cfg.kvmem.recall_every,
            kvmem_virtualization: cfg.kvmem.virtualization_enabled(),
            dense_floor_explicit: cfg.kvmem.dense_floor_mib.is_some(),
            // 准入姿态(S3b):env 覆盖(`reject` / `queue[:ms]`);缺省响亮拒。
            // 坏值 = **响亮拒**(§24.5④:不再 panic;启动期结构化失败,
            // 归 §18.5 的"配置错误静默 exit 1"红线同一族)
            admission_on_full: parse_admission_on_full().unwrap_or_else(|e| {
                owl_shared::log_error!("boot", "{e} —— 拒绝启动");
                std::process::exit(2);
            }),
            // per-turn 地板缺省。三态(2026-10-19 配置化):
            // - 虚拟化未开 ⇒ **Strict**(全量 dense;唯一诚实值)
            // - 虚拟化开 + `[kvmem].dense_floor_mib = N` ⇒ `Bytes(N MiB)`
            // - 虚拟化开 + 未给数 ⇒ `Bytes(0)` = "只要工作集装得下就准入"
            //   (搬运与腾位交给 A/B 两轴;这是低显存档的默认姿态)
            dense_floor: Self::dense_floor_of(&cfg.kvmem),
            no_graph: cfg.graph.no_graph,
            prefill_bucket: cfg.graph.prefill_bucket,
            ts_probe: cfg.dispatch.ts_probe,
            // S2b 正名:env 新名优先,旧名 `OWL_GDN_SLOTS` 兼容(旧档脚本不炸)
            max_active_turns: owl_shared::env_reader::parse("OWL_MAX_ACTIVE_TURNS")
                .or_else(|| owl_shared::env_reader::parse("OWL_GDN_SLOTS"))
                .unwrap_or(cfg.pool.max_active_turns),
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
    /// boot 显存快照(准入预算装配的真源;S3b §23.3)
    boot: BootVram,
}

/// boot 显存快照(设计卷 §23.3 的实测口径;只读两份数字)
#[derive(Clone, Copy, Debug, Default)]
struct BootVram {
    /// 开卡后 `cuMemGetInfo` free(字节;CUDA ctx 已在 free 里扣掉 —— 不进底价)。
    /// 无设备账档(CPU mock)= 0。
    v_card_bytes: u64,
    /// `Engine::new` 时刻的引擎 live 账(权重 = 装载后 − 此值)。
    live_at_new: u64,
}

/// **准入预算装配**(实测真源;设计卷 §23.3 表)。
///
/// 纪律:能测的必须测;无设备账档(CPU mock)由**固定池档**放宽规则接管
/// (`kv_face_bytes > 0` ⇒ 准入走池墙,卡级三项可为 0 —— 但由
/// `VramBudget::card_measured()` 标出,不装成"测过了")。
fn assemble_budget(
    boot: &BootVram,
    cfg: &EngineConfig,
    pool: &StatePool,
    live_loaded: u64,
    live_now: u64,
) -> Result<owl_store::BudgetInputs> {
    // 准入地板语义(S3b §23.5):**2026-10-19 拆护栏**。
    //
    // 旧护栏:非 Strict 一律拒 —— 因为当时选择层(B 轴)未落地,`Bytes` 的
    // 地板语义无从兑现。**B 轴已落地并拆出生产门**(`[kvmem].select_decode`),
    // 所以改成**条件放行**:只有配置里**显式开了虚拟化**
    // (`[kvmem].select_decode` + `select_partial` 两条同时为真)才允许非 Strict;
    // 否则依旧响亮拒(不许静默当 Strict 用 —— 那等于把"选择层会不会兜住"
    // 交给调用方的默认值)。
    if !cfg.knobs.dense_floor.is_strict() && !cfg.knobs.kvmem_virtualization {
        return Err(owl_iface::contract::ModelError::Msg(
            "boot: 非 Strict 的 dense_floor 需要**显式开启虚拟化** \
             ([kvmem].select_decode = true **且** select_partial = true)—— \
             地板语义只有选择层在场才能兜住;两者任一为假就响亮拒,不静默降级"
                .into(),
        ));
    }
    let block_bytes = pool.block_bytes();
    let page = pool.page.max(1);
    let kv_pool = (pool.nb as u64).saturating_mul(block_bytes as u64);
    let gdn_pool = (pool.max_active_turns as u64).saturating_mul(pool.gdn_row_bytes() as u64);
    let snap_pool = pool.snapshot_pool_bytes() as u64;
    let draft_pool = pool.draft_pool_bytes() as u64;
    let weights = live_loaded.saturating_sub(boot.live_at_new);
    // 运行底价 = run 末 live − 装载后 live − 已归类的池面(KV/GDN/快照/草稿)
    let runtime_floor = live_now
        .saturating_sub(live_loaded)
        .saturating_sub(kv_pool + gdn_pool + snap_pool + draft_pool);
    let inputs = owl_store::BudgetInputs {
        v_card_bytes: boot.v_card_bytes as usize,
        weights_bytes: weights as usize,
        runtime_floor_bytes: runtime_floor as usize,
        draft_pool_bytes: draft_pool as usize,
        gdn_row_bytes: pool.gdn_row_bytes(),
        kv_bytes_per_token: (block_bytes / page).max(1),
        // 中层总账 = KV 块腿分片 + 状态腿(快照)分片(S2 §25.2)
        host_bytes: (cfg.knobs.host_cache_mb + cfg.knobs.snap_host_mb) << 20,
        disk_bytes: 0,
        kv_face_bytes: kv_pool as usize,
    };
    let mb = |b: u64| b as f64 / 1048576.0;
    if owl_shared::vram::total_bytes() > 0 && weights == 0 {
        // 有设备账却算出权重 0 = 测量病(不是"没权重");不许静默
        owl_shared::log_warn!(
            "boot",
            "准入预算:权重字节实测为 0(引擎 live 账未动?装载路径绕过账本?)—— 请核对"
        );
    }
    owl_shared::log_info!(
        "boot",
        "准入预算:卡可用 {:.1}MiB · 权重 {:.1}MiB · 运行底价 {:.1}MiB · KV 池面 {:.1}MiB({} 槽 × {}B) · GDN 行 {:.2}MiB × {} · 快照池 {:.1}MiB · 草稿池 {:.1}MiB · host {:.1}MiB · 卡级实测={}",
        mb(boot.v_card_bytes),
        mb(weights),
        mb(runtime_floor),
        mb(kv_pool),
        pool.nb,
        block_bytes,
        pool.gdn_row_bytes() as f64 / 1048576.0,
        pool.max_active_turns,
        mb(snap_pool),
        mb(draft_pool),
        mb(cfg.knobs.host_cache_mb as u64 * 1048576),
        weights > 0 && runtime_floor > 0,
    );
    Ok(inputs)
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
        // boot 显存快照(准入预算的真源;S3b §23.3)—— 必须在任何池/权重之前取
        let boot = BootVram {
            v_card_bytes: owl_shared::vram::free_bytes(),
            live_at_new: owl_shared::vram::live_bytes(),
        };
        Self::on_with(cfg, face, boot)
    }

    fn on_with(cfg: EngineConfig, face: D, boot: BootVram) -> Result<Self> {
        // 注:`[graph] prefill_bucket` 的**真接线在 [`Self::run`]**(桶捕获需要
        // 模型与状态池;静止点律 ⇒ 只在 boot 做)。此处不再拒启(2026-10-15
        // M2b 落地;此前的"配了没实现就拒"临时门已被真接线取代)。
        Ok(Self { face, cfg, boot })
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
        // 装载后的引擎 live 账(权重与运行底价的基线;S3b §23.3)
        let live_loaded = owl_shared::vram::live_bytes();
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
        // KV 写者/读者清单门(收口律机器门):按活跃 quant 逐族校验内核
        // 在登记表与 .cu 源在场;缺族/缺名 = 装配失败(池 dtype 改造漏网
        // 即启动即拦,不等 E2E 红灯)
        owl_models::kernel::kv_manifest_gate(env.kv.quant, env.attn.fi)?;
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
            // **锁①门控**:只有虚拟化真的开着(配置显式 `select_decode` +
            // `select_partial`)才允许池小于上下文 —— 否则链超出池容量必
            // `Exhausted`(无洞可停,即无选择层可挤)。
            pool_overcommit: self.cfg.knobs.kvmem_virtualization,
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
            // B-3e-2:部分驻留档**不带 XQA**(页表语义不共戴;见上方醒目告知)
            xqa_decode: self.cfg.knobs.env.attn.decode_xqa && !self.cfg.knobs.select_partial,
            xqa_prefill: self.cfg.knobs.env.attn.xqa_prefill,
            max_active_turns: self.cfg.knobs.max_active_turns,
            snap_max: self.cfg.knobs.snap_max,
            vram_target: self.cfg.knobs.vram_target,
            vram_reserve_mb: self.cfg.knobs.vram_reserve_mb,
            dump_all: self.cfg.knobs.gdn_dump_all,
            // prefill 桶 padding 的 dump 页(M2b):档位非 off 才多分配那 1 页
            prefill_dump_page: self.cfg.knobs.prefill_bucket
                != owl_shared::config::PrefillBucket::Off,
            gdn_dbg: self.cfg.knobs.probes.gdn_dbg,
            // B-2:选择层视图面(**生产档也要** —— 视图是 decode 的消费方;
            // 与诊断面 `coverage_probe` 解耦)
            select_view: self.cfg.knobs.coverage_probe || self.cfg.knobs.select_decode,
        };
        // **B-3e-2 前置校验**(响亮拒,不许静默):
        // ① 部分驻留会让会话链**出洞**(占位符)⇒ 必须有"懂洞"的消费方
        //    (视图页表)⇒ `select_decode` 必开;
        // ② XQA 裸 decode 臂的页表直接来自会话链(`xqa_pt_of`)⇒ 洞 = u32::MAX 当页号 =
        //    ILLEGAL_ADDRESS(27B Stage2 实测)。本档**自动关 XQA 并响亮告知**。
        // **下沉的前提 = 缓存树在场**(2026-10-19 实测定谳):
        // 下沉机制 = "缓存树 + 会话双持(refs==2) → decref → 缓存独占 → 下沉";
        // 而 `session.rs` 的 `cache_cap = if paged && prefix_cache { .. } else { 0 }`
        // ⇒ **prefix_cache 关着 ⇒ 缓存树禁用 ⇒ `insert_inc` 返空 ⇒ 块永远 refs==1
        // ⇒ 结构性不可沉**(实测:每 chunk 送 4 片、入树 0、下沉 0 块)。
        // 不许静默:开了 partial 却没开 prefix_cache = 配置错,拒启。
        if self.cfg.knobs.select_partial && !self.cfg.knobs.prefix_cache {
            return Err(owl_iface::contract::ModelError::Msg(
                "boot: `[kvmem].select_partial = true` 要求 `[pool].prefix_cache = true` \
                 —— 下沉靠「缓存树 + 会话双持」(refs==2)才能 decref 放给宿主;\
                 关着 prefix_cache ⇒ 缓存树 cap=0 ⇒ 块永远 refs==1 ⇒ **一块都沉不下去** \
                 (响亮拒:跑起来会白跑一趟还报 '不可沉 N 块')"
                    .into(),
            ));
        }
        if self.cfg.knobs.select_partial && !self.cfg.knobs.select_decode {
            owl_shared::log_error!(
                "boot",
                "OWL_SELECT_PARTIAL=1 需同时开 OWL_SELECT_DECODE=1(链上出洞后必须有视图消费方)—— 拒绝启动"
            );
            std::process::exit(2);
        }
        if self.cfg.knobs.select_partial && self.cfg.knobs.env.attn.decode_xqa {
            owl_shared::log_warn!(
                "boot",
                "部分驻留档:本档**关掉 XQA 裸 decode 臂**(其页表语义读整条会话链,遇占位符 = 非法地址;\n                    \
                 视图×XQA 的对接是挂账,不在此刀)—— 本档走 v2 分页 decode"
            );
        }
        let pool = StatePool::alloc(&mut self.face, dims, &loaded.spec.layer_types, &plan).await?;

        // **锁①:虚拟化档的池容量硬校验**(2026-10-19;响亮拒,不静默落空槽)。
        //
        // 几何一律从**池实况**取(`pool.page`/`pool.nb`),不另算一份。
        //
        // # 下限的算式(为什么乘 N_active)
        //
        // ```text
        // 池页数 ≥ N_active × (select_blocks + sink + recent + chunk/page)
        //          └─ 每个活跃 turn 都要**自己一套**工作集与 chunk 工作区
        // ```
        //
        // `N_active` = `max_active_turns`(并发上限)。**这一项不能省** ——
        // 并发档下每个 turn 各有自己的工作集,池要能同时装下它们。
        // 27B f16 算例(页 = 32 tok = 2 MiB):
        //   select_blocks 32 + sink 2 + recent 4 + chunk 256 ⇒ 每 turn 46 页 = 92 MiB
        //   8 并发 ⇒ 368 页 = 736 MiB ‖ 对照 dense 8 并发 = 8×256 页 = 4 GiB ⇒ 省 5.6×
        if self.cfg.knobs.kvmem_virtualization {
            let chunk_pages = self.cfg.prefill_chunk.div_ceil(pool.page.max(1)).max(1);
            let per_turn = self.cfg.knobs.select_blocks.max(1)
                + self.cfg.knobs.select_sink
                + self.cfg.knobs.select_recent
                + chunk_pages;
            let n_active = self.cfg.knobs.max_active_turns.max(1);
            let need = per_turn * n_active;
            let page_bytes = pool.block_bytes().max(1);
            if pool.nb < need {
                return Err(owl_iface::contract::ModelError::Msg(format!(
                    "boot: 设备池 {} 页装不下 {n_active} 并发 × {per_turn} 页 = {need} 页\
                     (每 turn = select_blocks {} + sink {} + recent {} + chunk {chunk_pages});\
                     pool_tokens={} page={} —— 调大池 / 减预算 / 降 max_active_turns\
                     (响亮拒:跑起来必 Exhausted)",
                    pool.nb,
                    self.cfg.knobs.select_blocks,
                    self.cfg.knobs.select_sink,
                    self.cfg.knobs.select_recent,
                    plan.pool_tokens,
                    pool.page
                )));
            }
            // **准入地板缺省 = 本 turn 工作集字节**(不是 0!)
            //
            // 地板语义 = "本 turn 至少多少字节 KV 直接驻留设备"。并发档下若给 0,
            // N 个 turn 会各自被无条件准入、然后互抢池 ⇒ 抖动;给工作集字节
            // = "每个进来的 turn 都先留出它自己那套",与上面的池下限算式**同源**
            // (同一套页数,乘的是字节而不是并发数)。显式给了 `[kvmem].dense_floor_mib`
            // 则尊重显式值。
            if !self.cfg.knobs.dense_floor_explicit {
                let ws_pages = (self.cfg.knobs.select_blocks.max(1)
                    + self.cfg.knobs.select_sink
                    + self.cfg.knobs.select_recent)
                    .max(1);
                self.cfg.knobs.dense_floor =
                    owl_store::TurnFloor::Bytes(ws_pages * page_bytes);
            }
            owl_shared::log_warn!(
                "boot",
                "KVMem 虚拟化档:**池 = {} tok / {} 页** vs 上下文 {} tok / {} 页 ⇒ 超卖 {:.2}×;\
                 并发上限 {} × 每 turn {} 页(工作集 {}+{}+{} + chunk {}) = {need} 页;\
                 每页 {} KiB;准入地板 = {:?} / turn;宿主 {} MiB",
                pool.nb * pool.page,
                pool.nb,
                s,
                s.div_ceil(pool.page.max(1)),
                s as f64 / (pool.nb * pool.page).max(1) as f64,
                n_active,
                per_turn,
                self.cfg.knobs.select_blocks,
                self.cfg.knobs.select_sink,
                self.cfg.knobs.select_recent,
                chunk_pages,
                page_bytes / 1024,
                self.cfg.knobs.dense_floor,
                self.cfg.knobs.host_cache_mb + self.cfg.knobs.snap_host_mb,
            );
        }
        if let Some(q) = fi_quant {
            owl_shared::log_info!("boot", "FlashInfer prefill 面启用(FI 直读主池 kNHD 单池;quant={:?})",
                q
            );
        }

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

        // ── 准入预算组装 + 会话/块账房(S3b;§23.3)──
        // 此刻固定面全部落位(权重 + 池 + 图 slab + 表件)⇒ 实测数字齐;
        // 装配失败 = 拒绝启动(A5.1:没有数字的项不许进预算式)。
        let budget_inputs = assemble_budget(&self.boot, &self.cfg, &pool, live_loaded, owl_shared::vram::live_bytes())?;
        let admission_policy = owl_store::AdmissionPolicy::new(
            self.cfg.knobs.max_active_turns.max(1),
            self.cfg.knobs.admission_on_full,
        );
        let host_bytes = self.cfg.knobs.host_cache_mb << 20;
        // **S2:状态腿三层化**(§25.2)—— 注册快照池布局(快照 = 「槽 × 片」),
        // 并把 store 的状态腿宿主层挂上(KV 腿宿主分片之外的部分)。
        // 放在 `SessionBank::new` 之前:布局进 store 后 `Rel` 才知道快照的
        // 单条字节(宿主层的块粒度)。
        let bank = {
            let mut b = crate::session::SessionBank::new(
                crate::session::BankPlan {
                    nb: pool.nb,
                    page: pool.page,
                    block_bytes: pool.block_bytes(),
                    mtp_block_bytes: pool.mtp_block_bytes(),
                    paged: pool.paged,
                    mtp: spec.mode == SpecMode::Mtp,
                    prefix_cache: self.cfg.knobs.prefix_cache,
                    // 宿主层预算(REQ C4:默认关;0 = 关层)—— 与预算入参**同源**
                    // (腿不得超总账;`Rel::open` 交叉校验)
                    host_bytes,
                },
                budget_inputs,
                admission_policy,
                pool.pool_layout(0),
                pool.mtp_pool_layout(0),
            )?;
            // **B-4 驱逐策略装配**(boot 一次;缺省 LRU ⇒ 与历史逐字一致)
            b.rel.set_evict_policy(self.cfg.knobs.evict_policy.build());
            owl_shared::log_info!("store", "驱逐策略 = {}", b.rel.evict_policy_name());
            if self.cfg.knobs.index_probe {
                // B 轴索引几何:层 = 全注意力层数,头 = KV 头数,dim = head_dim
                b.rel.attach_index(dims.hd, pool.kvs.len(), dims.hkv)?;
                owl_shared::log_info!(
                    "boot",
                    "选择层索引:采集开(dim={} 层={} KV 头={};每块 {}B)",
                    dims.hd,
                    pool.kvs.len(),
                    dims.hkv,
                    b.rel.index.as_ref().map(|i| i.block_bytes()).unwrap_or(0)
                );
            }
            if pool.paged && pool.snap_capacity() > 0 {
                let epoch = b.rel.attach_snap_layout(pool.snap_pool_layout(0))?;
                owl_shared::log_info!(
                    "boot",
                    "状态腿三层化:快照 {} 槽 × {}B/条(布局 epoch {epoch})· 宿主预算 {:.1}MiB",
                    pool.snap_capacity(),
                    pool.snap_bytes(),
                    b.rel.snap_host_budget() as f64 / 1048576.0
                );
            }
            b
        };

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
            cov_pending: None,
            cov_last: None,
            sel_view: None,
            select_view_steps: 0,
            sel_view_fallback: 0,
            stage: crate::running::StageStats::default(),
            partial: crate::running::PartialStats::default(),
            prof: crate::running::BAxisProfile::default(),
            sel_keep: Vec::new(),
        })
    }
}
