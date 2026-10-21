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
use owl_models::layers::rope::Rope;
use owl_models::model::{Model, ModelSpec};
use owl_iface::contract::{DeviceClient, ModelError};
use owl_iface::handle::{AssetClass, DeviceHandle, Owner};

use crate::graph_fleet::{FleetBoot, GraphFleet};
use crate::loader::{LoadedModel, ModelLoader, RankModel};
use crate::running::RunningEngine;
use crate::spec_runtime::{SpecBootInput, SpecMode, SpecRuntime};
use crate::state::{ModelDims, StatePool};

type Result<T> = std::result::Result<T, ModelError>;

/// boot 旋钮(显式依赖律,2026-10-10:环境变量是隐式依赖,引擎内部
/// **零 env 读取** —— env 派生数据由入口(server config)/ 测试用例经
/// 本结构显式传入;[`EngineKnobs::from_config`] 是唯一的入口侧构造器,
/// [`Default`] = 生产基线)。
#[derive(Clone, Debug)]
pub struct EngineKnobs {
    /// **装载旋钮**(`[load]` 段;从配置文件来,不读 env —— 用户裁决 2026-10-19)
    pub load: owl_shared::config::LoadCfg,
    /// **rope 外推/缩放**(`[model.rope]`;GGUF 档生效;缺省关)
    pub rope: owl_shared::config::RopeCfg,
    /// **视觉塔**(`[model].vision`):装载 `model.visual.*` + 允许 VLM 请求。
    /// 缺省 false = 纯文本路径逐字节不变(27B 视觉档显式开)。
    pub vision: bool,
    /// **GGUF 视觉塔单文件**(`[model].vision_path`;None = 同目录唯一候选)。
    /// 仅 GGUF 档 + `vision = true` 消费。
    pub mmproj: Option<std::path::PathBuf>,
    /// lm_head 量化方案(`[model].head_quant`;boot 投影到 tag filter)
    pub head_quant: owl_shared::config::HeadQuant,
    /// tag 过滤通配符(`[dispatch] tag_wildcard`;诊断 `*`)
    pub tag_wildcard: bool,
    /// 步级探针(诊断面;Default = 全关)
    pub probes: crate::running::StepProbes,
    /// 块池容量覆写(None = 2 × 单会话容量;原 OWL_POOL_TOKENS)
    pub pool_tokens: Option<usize>,
    /// 草稿域(深度/哑草稿/家族目录/池量化/verify 诊断)
    pub draft: DraftCfg,
    /// 前缀缓存(缺省开;原 OWL_PREFIX_CACHE≠"0")
    pub prefix_cache: bool,
    /// **前缀缓存份额**(0..1;`[pool].prefix_cache_share`;缺省 0.5 = 旧 `nb/2`)。
    /// 容量算式在 store(`CachePolicy::Share`),这里只是**政策输入**。
    pub prefix_cache_share: f64,
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
    /// **方案对照档**(`OWL_SCHEME_PROBE`;诊断):多读一份 V + host 算 A/B/dense。
    /// 入口面一次性解析(引擎内部零 env 读取)。
    pub coverage_scheme_probe: bool,
    /// 方案对照的最小 ctx(`OWL_SCHEME_MIN_CTX`;0 = 每个边界都算)
    pub coverage_scheme_min_ctx: usize,
    /// **重诊断步进**(`OWL_COV_ANALYZE_EVERY`;None = 每边界都诊断)
    pub coverage_analyze_every: Option<usize>,
    /// **每 turn 工作集的对外事实(配置面)** —— `[kvmem].workspace_per_req`(token;
    /// 含 sink/recent 保底)。装配期由 [`derive_turn_ws`] 派生下面三件:
    /// `select_blocks`(召回预算块数)/ `dense_floor`(准入地板字节)/ 池下限里的每 turn 页数。
    /// **零 env 覆盖**(配置是唯一来源,见 §8.27)。
    pub workspace_per_req: usize,
    /// **召回预算**(块数)= `ceil(workspace_per_req/page) − sink − recent`。
    /// **派生量**(唯一产地 = [`derive_turn_ws`],装配期写;测试档可直接构)。
    /// 视图 = 保底 + 按分 top-k ⇒ 这是"选择层丢多少"的内部旋钮。
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
    /// **准入地板的事实(配置面)** = `[kvmem].dense_floor_mib`(MiB;`None` = 未给)。
    /// 装配期按**池页宽**换成页数(`ceil(MiB/page_bytes)`,页数 0 保留 0)。
    /// `None` = 跟随虚拟化姿态:开 ⇒ 页数 = 工作集页数;关 ⇒ `Strict`。
    /// **配置只存 MiB 事实,页数在装配期落**(A7:地板与池下限同一把尺)。
    pub dense_floor_mib: Option<u64>,
    /// **状态腿宿主预算 MiB**(S2 §25.2;缺省 **0 = 关层**)。`OWL_SNAP_HOST_MB`。
    ///
    /// **复用池容量上限**(MiB;缺口 4 收尾 P1-1;2026-10-19)。
    ///
    /// 缺省 256:27B 单轮释放后池天然囤到 669 MiB、复用率仅 ~46% ⇒ 无界池 = 新形态囤积;
    /// 256 覆盖实测热集大头,超出部分逐块顶出回驱动。0 = 不限(仅供诊断/对照)。
    pub pool_cap_mb: usize,
    /// 与 [`Self::host_cache_mb`] 同属 store 中层总账的两个**分片**
    /// (KV 块腿 / GDN 快照腿)—— 总账 = 两者之和(`Rel::open` 交叉校验分片
    /// 不超总账)。为什么分开:两条腿的搬运形态与热区完全不同,共用一个数
    /// 只会让一边饿死(实测:全给 KV 腿 ⇒ 快照下沉无位 ⇒ 冷快照被丢)。
    pub snap_host_mb: usize,
    /// **磁盘 KV 腿预算 MiB**(S6;0 = 关层)。宿主两级都满时的最后一站(**内容不丢**)。
    pub disk_cache_mb: usize,
    /// **磁盘状态腿预算 MiB**(S6;0 = 关层;与 [`Self::disk_cache_mb`] 同属盘层总账)。
    pub snap_disk_mb: usize,
    /// **磁盘内容目录**(空 = 关层)。
    pub disk_dir: String,
    /// **准入满员姿态**(S3b;缺省响亮拒)。`OWL_ADMIT_ON_FULL=reject|queue[:ms]`。
    ///
    /// `Reject` = 准入拒(客户端该降并发/重试,不是压缩上下文);
    /// `Queue { max_wait_ms }` = 留队首等下一次事件重判(超时则拒)。
    pub admission_on_full: owl_store::QueuePolicy,
    /// **per-turn 直接显存地板的缺省值**(S3b;设计卷 §2.4;**页口径**)。
    ///
    /// 装配期由 [`Self::derive_turn_ws`] 落值(`dense_floor_mib` 事实 × 池页宽)。
    /// `Strict` = 全量 dense 驻留(非虚拟化档的唯一诚实值);
    /// 请求可用 `TurnOpts::dense_floor` 逐 turn 覆写 —— 覆写面当前只认 `Strict`
    /// (**响亮拒**,见 `running::submit_session_opts`:接受非 Strict 而无执行面
    /// 就是静默降级)。
    pub dense_floor: owl_store::TurnFloor,
    /// eager 直发,跳过图捕获(原 OWL_NO_GRAPH;**总闸**)
    pub no_graph: bool,
    /// **prefill 尾块合并**(tokens;L4,2026-10-18;`[runtime] prefill_tail_merge`):
    /// 剩余 ≤ `prefill_chunk + 本值` ⇒ 本步吃完(消灭与尺寸无关的固定宿主税
    /// ~250ms/chunk 的尾块;缺省 0 = 关 = 旧行为)。见 `scheduler::prefill_step_chunk`。
    pub prefill_tail_merge: usize,
    /// 相位时间片(0 = 关;`[runtime] slice_every/slice_tokens`)
    pub slice_every: usize,
    pub slice_tokens: usize,
    /// 准入三旋钮(`[runtime]`;见 `sched_policy::Limits`)
    pub newcomer_min_tokens: usize,
    pub min_gen_tokens: usize,
    /// 池保留页(0 = 自动 `min(nb/64, 8)`)
    pub reserve_pages: usize,
    /// 等待队列上限(0 = 无界;超出 ⇒ 429 `no_slot`)
    pub max_queue: usize,
    /// **末段页对齐**(前缀复用闭环,2026-10-19;`[pool] prefill_align_tail`,
    /// 缺省 **true**)。块边界快照只在**页对齐**处拍 ⇒ 单块档(整个 prompt 落进
    /// 一个 chunk)L4 会把尾端不对齐地一口吃完 ⇒ 一个快照都没有 ⇒ 下一轮全量
    /// 重算。开 ⇒ 单块档切一刀(尾巴 <1 页,多付一份 ~250ms chunk 宿主税),
    /// 换"prompt 最靠后的整页"有快照;多 chunk 档不受影响(前段尾本来就对齐)。
    pub prefill_align_tail: bool,
    /// **生成相位页边界快照页步长**(§9.1(6)a;`[pool] snap_decode_stride`,缺省 1)。
    /// spec 轮提交跨过页边界 ⇒ 用 fold 把 GDN 态回卷到边界拍一张再卷回;
    /// 本键节流(`边界页号 % stride == 0` 才拍)。0 = 关(旧行为:生成相位不拍)。
    pub snap_decode_stride: usize,
    /// **FI plan 图档瓦**(`[dispatch] fi_cuda_graph`;2026-10-18 图档单图立案):
    /// 开 ⇒ 形状对 ctx 全域恒定(可烘图的先决条件)。装配期经
    /// `owl_kernels::family::flashinfer::configure_plan_env` 投递给核。
    pub fi_cuda_graph: bool,
    /// **decode 单图烘 plan 上界**(`[dispatch] fi_plan_bake`;None = 现状 piecewise)
    pub fi_plan_bake: Option<usize>,
    /// **图自携 FI plan restore**(`[dispatch] fi_plan_restore`;缺省 true):
    /// true = 捕获窗内把本 plan 表烘成图内 memcpy(桶图旧行为);
    /// false = 图不烘表 ⇒ **host 是唯一权威**,每个图回放前窗外 `fi_plan_sync`
    /// (decode/verify 既有;prefill 桶已退役,2026-10-21)。
    pub fi_plan_restore: bool,
    /// TS 探针槽(原 OWL_TS_PROBE)
    pub ts_probe: bool,
    /// GDN 槽数覆写(缺省 8;原 OWL_MAX_ACTIVE_TURNS)
    pub gdn_slots: usize,
    /// **跨会话合批上限**(B5.1;缺省 1 = 串行,逐字节现状)。
    /// >1 = 调度面拼 T=N 裸 decode(M1;spec off / 单卡 / 无选择层)。
    pub max_batch: usize,
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
    /// owl_cuda::DiagOpts 一体携带,入口经 DiagOpts::from_config 填充)
    pub cuda: owl_cuda::DiagOpts,
    /// 解释器执行环境(硬件档/KV 策略/量化/分派旋钮;Default = 生产基线)
    pub env: owl_models::env::EnvProvider,
}

impl Default for EngineKnobs {
    fn default() -> Self {
        Self {
            load: owl_shared::config::LoadCfg::default(),
            rope: owl_shared::config::RopeCfg::default(),
            vision: false,
            mmproj: None,
            head_quant: owl_shared::config::HeadQuant::Off,
            tag_wildcard: false,
            probes: crate::running::StepProbes {
                // 命中面/调试打印:env 覆盖(实测排障用 `OWL_PROBES_DEBUG=1`)
                debug: owl_shared::env_reader::flag("OWL_DEBUG"),
                ..Default::default()
            },
            pool_tokens: None,
            draft: DraftCfg::default(),
            prefix_cache: true,
            prefix_cache_share: 0.5,
            host_cache_mb: 0,
            snap_host_mb: 0,
            disk_cache_mb: 0,
            snap_disk_mb: 0,
            disk_dir: String::new(),
            pool_cap_mb: 256,
            index_probe: false,
            coverage_probe: false,
            coverage_analyze_at: 0,
            coverage_scheme_probe: false,
            coverage_scheme_min_ctx: 0,
            coverage_analyze_every: None,
            select_decode: false,
            select_partial: false,
            evict_policy: owl_store::EvictPolicyKind::Lru,
            workspace_per_req: 2240,
            select_blocks: 64,
            select_sink: 2,
            select_recent: 4,
            recall_every: 1,
            kvmem_virtualization: false,
            dense_floor_mib: None,
            admission_on_full: owl_store::QueuePolicy::Reject,
            dense_floor: owl_store::TurnFloor::Strict,
            no_graph: false,
            prefill_tail_merge: 0,
            slice_every: 0,
            slice_tokens: 0,
            newcomer_min_tokens: 4096,
            min_gen_tokens: 256,
            reserve_pages: 0,
            max_queue: 0,
            prefill_align_tail: true,
            snap_decode_stride: 1,
            fi_cuda_graph: false,
            fi_plan_bake: None,
            fi_plan_restore: true,
            ts_probe: false,
            gdn_slots: 8,
            max_batch: 1,
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
            owl_shared::config::EvictPolicyCfg::Lrfu { lambda } => {
                owl_store::EvictPolicyKind::Lrfu(lambda as f64)
            }
        }
    }

    /// **准入地板映射**(单源):配置三态 → `TurnFloor`(装配期定谳)。
    /// 配置层只存事实(`virtualization_enabled` + `dense_floor_mib`),
    /// 到 store 词汇的映射在本层(分层:shared 不得依赖 store)。
    ///
    /// ⚠️ 这里只能给**页几何未知时的占位**:
    /// - 非虚拟化 ⇒ `Strict`(唯一诚实值,不会变);
    /// - 虚拟化 ⇒ `Pages(0)`,真值由 [`Self::derive_turn_ws`] 在池页宽已知后落
    ///   (显式 MiB → 页 / 未给 ⇒ 工作集页数)。
    fn dense_floor_of(cfg: &owl_shared::config::KvmemCfg) -> owl_store::TurnFloor {
        Self::dense_floor_for(cfg.virtualization_enabled(), cfg.dense_floor_mib)
    }

    /// 地板映射的**唯一实现**(按两份事实入参;探针/测试档直接构 knobs 时
    /// 也走这一条 —— 不得在别处再写一次“虚拟化开 ⇒ Pages(0)”:那样两处
    /// 迟早对不上)。
    ///
    /// `floor_mib = Some(n)` 时仍需页宽才能定页数 ⇒ 本函数给的是
    /// `Pages(0)` 占位,**真值在装配期**(见 [`Self::dense_floor_with_mib`])。
    pub fn dense_floor_for(
        virtualization: bool,
        floor_mib: Option<u64>,
    ) -> owl_store::TurnFloor {
        if !virtualization {
            // 虚拟化不在场 ⇒ 唯一诚实值(非 Strict 会被 boot 拦)
            return owl_store::TurnFloor::Strict;
        }
        // 页宽未知:先落 0 页占位(装配期 `derive_turn_ws` 用池页宽覆盖)。
        let _ = floor_mib;
        owl_store::TurnFloor::Pages(0)
    }

    /// **MiB 事实 → 页数地板**(唯一换算;页宽已知时用)。
    ///
    /// 向上取整到整页(少一页 = “承诺了又给不够”);`0 MiB` ⇒ `0 页`
    /// (= 允许全量下沉)。
    pub fn dense_floor_with_mib(mib: u64, page_bytes: usize) -> owl_store::TurnFloor {
        let page_bytes = page_bytes.max(1);
        let bytes = (mib as usize).saturating_mul(1 << 20);
        owl_store::TurnFloor::Pages(bytes.div_ceil(page_bytes))
    }


    /// **每 turn 工作集的派生(唯一产地)** —— 对外一个 token 数,内部三件:
    /// 工作集页数 / 召回预算(块)/ 准入地板(**页数**,A7 起)。
    ///
    /// 为什么必须单点(2026-10-19):这三件在旧实现里**各算一次**
    /// (池下限算式算一遍页数、准入地板又算一遍、召回预算再写死一个块数)
    /// —— 迟早对不上;更糟的是**对外口径**只能靠人脑在"块/字节/token"之间换算。
    /// 现在配置只给 `workspace_per_req`(token),三件都从它派生,
    /// 且 boot 打 `[turn-ws]` 自证行(含池下限算式)。
    ///
    /// 语义:工作集 = **常驻块数**(含保底)= `sink + recent + 按分 top-k`。
    /// `top_k = 0`("只留保底")合法;`工作集 < 保底` = 配置不自洽 ⇒ **响亮拒**。
    /// ⚠️ `select_blocks` 是**总选中数**(`owl_store::select` 的预算就是总数,保底从
    /// 里面优先占),所以它 = `ws_pages`,**不是** `ws_pages − keep`(2026-10-19 踩过)。
    pub(crate) fn derive_turn_ws(
        workspace_tokens: usize,
        page_tokens: usize,
        page_bytes: usize,
        knobs: &mut EngineKnobs,
    ) -> Result<(usize, usize)> {
        let page = page_tokens.max(1);
        let keep = knobs.select_sink + knobs.select_recent;
        let ws_pages = workspace_tokens.div_ceil(page).max(1);
        if ws_pages < keep {
            return Err(owl_iface::contract::ModelError::Msg(format!(
                "boot: 每 turn 工作集不自洽 —— workspace_per_req={} tok = {} 页 < 保底                  (sink {} + recent {} = {} 页);要么调大工作集,要么调小保底",
                workspace_tokens, ws_pages, knobs.select_sink, knobs.select_recent, keep
            )));
        }
        // `select_blocks` = **总选中数(含保底)** —— 工作集页数就是它;保底是
        // 其中被优先占用的那部分(top-k = ws_pages − keep)。
        knobs.select_blocks = ws_pages;
        let ws_bytes = ws_pages * page_bytes.max(1);
        // 地板**只在虚拟化档**落:非虚拟化档的唯一诚实地板是 `Strict`
        // (boot 会拦非 Strict),别在这里越权改。
        //
        // A7(2026-10-19):地板是**页口径** —— 显式 MiB 按池页宽向上取整成页数;
        // 未给数则“地板 = 工作集页数”(与池下限/召回预算**同一套页数**,
        // 不再各处各算一遍字节)。
        if knobs.kvmem_virtualization {
            knobs.dense_floor = match knobs.dense_floor_mib {
                Some(mib) => Self::dense_floor_with_mib(mib, page_bytes),
                None => owl_store::TurnFloor::Pages(ws_pages),
            };
        }
        Ok((ws_pages, ws_bytes))
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
            // 主干新增变体(暂无核臂,选了会响亮拒);digest 只做标注
            owl_models::env::KvQuant::Fp8E5M2 => "fp8e5m2",
        };
        let ev = match self.evict_policy {
            owl_store::EvictPolicyKind::Lru => "lru".to_string(),
            owl_store::EvictPolicyKind::Lrfu(l) => format!("lrfu:{l}"),
        };
        // 地板一列打**事实**(配置面口径),不打占位页数(A7):
        // 页数要在池页宽已知的装配期才定 ⇒ 真值在 `[turn-ws]` 自证行。
        let fl = if !self.kvmem_virtualization {
            "Strict".to_string()
        } else {
            match self.dense_floor_mib {
                Some(mib) => format!("{mib}MiB→页"),
                None => format!("ws({}tok)→页", self.workspace_per_req),
            }
        };
        format!(
            "pool={:?} turns={} snap_max={} host={}M snap_host={}M disk={}M snap_disk={}M dir={:?} prefix_cache={}:{:.2} align_tail={} bnd_stride={} | kvmem={}[idx={} cov={} view={} part={} ws={}tok(sink{}+recent{}+topk{}@32tok/page) every={} evict={} floor={}] | fi={} split={} gdn_chunked={} kv={} graph_off={}",
            self.pool_tokens,
            self.gdn_slots,
            self.snap_max,
            self.host_cache_mb,
            self.snap_host_mb,
            self.disk_cache_mb,
            self.snap_disk_mb,
            self.disk_dir,
            self.prefix_cache as u8,
            self.prefix_cache_share,
            self.prefill_align_tail as u8,
            self.snap_decode_stride,
            if self.kvmem_virtualization { "ON" } else { "off" },
            self.index_probe as u8,
            self.coverage_probe as u8,
            self.select_decode as u8,
            self.select_partial as u8,
            self.workspace_per_req,
            self.select_sink,
            self.select_recent,
            self.select_blocks,
            self.recall_every,
            ev,
            fl,
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
            load: cfg.load.clone(),
            rope: cfg.model.rope.clone(),
            vision: cfg.model.vision,
            mmproj: cfg
                .model
                .vision_path
                .clone()
                .filter(|p| !p.as_os_str().is_empty()),
            head_quant: cfg.model.head_quant,
            tag_wildcard: cfg.kernel.tag_wildcard,
            probes: crate::running::StepProbes::from_config(cfg),
            // ── **配置是唯一来源**(2026-10-19 统一)──
            // 本函数只做 `OwlConfig` → 旋钮的字段搬运,**不读任何 env**
            // (探针档/生产档走同一条路 ⇒ 同一份配置 = 同一套条件)。
            // 测试面只保留"运行器级"env(选卡/资产路径/场景过滤),见
            // `docs/arch/配置面-单一来源.md`。
            pool_tokens: cfg.pool.pinned_pool_tokens(),
            pool_cap_mb: cfg.pool.pool_cap_mb,
            draft: DraftCfg {
                depth: cfg.spec.depth,
                family: cfg.spec.draft,
                dumb: cfg.spec.dumb,
                // 空串 = 未设(TOML 里写 `draft_path = ""` 的语义)
                dflash2_dir: cfg
                    .spec
                    .draft_path
                    .as_ref()
                    .filter(|p| !p.as_os_str().is_empty())
                    .map(|p| p.display().to_string()),
                mtp_dir: cfg.model.mtp_dir.as_ref().map(std::path::PathBuf::from),
                kv_quant: if cfg.spec.draft_kv_fp8 {
                    owl_models::env::KvQuant::Fp8E4M3
                } else {
                    owl_models::env::KvQuant::None
                },
                mtp_window: cfg.spec.mtp_window,
                mtp_sink: cfg.spec.mtp_sink,
                mtp_window_min_ctx: cfg.spec.mtp_window_min_ctx,
                adaptive_depth: cfg.spec.adaptive_depth,
                depth_min: cfg.spec.depth_min,
                depth_switch_ctx: cfg.spec.depth_switch_ctx,
                adaptive_min_tpr: cfg.spec.adaptive_min_tpr,
                notaps: cfg.spec.notaps,
                tapdecl: cfg.spec.tapdecl,
                tp_draft: cfg.spec.tp_draft,
            },
            prefix_cache: cfg.pool.prefix_cache,
            prefix_cache_share: cfg.pool.prefix_cache_share,
            // 宿主层预算(S3-3;`[pool].host_cache_mb`)
            host_cache_mb: cfg.pool.host_cache_mb,
            disk_cache_mb: cfg.pool.disk_cache_mb,
            snap_disk_mb: cfg.pool.snap_disk_mb,
            disk_dir: cfg.pool.disk_dir.clone(),
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
            // 诊断面的两个 env 旋钮在**入口**一次性收编(引擎内部零 env 读取)
            coverage_scheme_probe: owl_shared::env_reader::flag("OWL_SCHEME_PROBE"),
            coverage_scheme_min_ctx: owl_shared::env_reader::parse::<usize>("OWL_SCHEME_MIN_CTX")
                .unwrap_or(0),
            coverage_analyze_every: owl_shared::env_reader::parse::<usize>("OWL_COV_ANALYZE_EVERY"),
            select_decode: cfg.kvmem.select_decode,
            select_partial: cfg.kvmem.select_partial,
            evict_policy: Self::evict_policy_of(&cfg.kvmem),
            workspace_per_req: cfg.kvmem.workspace_per_req,
            // `select_blocks` = 派生量(装配期 `derive_turn_ws` 按池页宽写);
            // 此处保留构造器缺省,测试档可直接覆盖。
            select_blocks: 64,
            select_sink: cfg.kvmem.select_sink,
            select_recent: cfg.kvmem.select_recent,
            recall_every: cfg.kvmem.recall_every,
            kvmem_virtualization: cfg.kvmem.virtualization_enabled(),
            // 地板 = **事实**(MiB 配置项);页数在装配期按池页宽落(`derive_turn_ws`)。
            dense_floor_mib: cfg.kvmem.dense_floor_mib,
            // 准入姿态(S3b):env 覆盖(`reject` / `queue[:ms]`);缺省响亮拒。
            // 坏值 = **响亮拒**(§24.5④:不再 panic;启动期结构化失败,
            // 归 §18.5 的"配置错误静默 exit 1"红线同一族)
            admission_on_full: parse_admission_on_full().unwrap_or_else(|e| {
                owl_shared::log_error!("boot", "{e} —— 拒绝启动");
                std::process::exit(2);
            }),
            // per-turn 地板缺省。三态(2026-10-19 配置化;A7 改页口径):
            // - 虚拟化未开 ⇒ **Strict**(全量 dense;唯一诚实值)
            // - 虚拟化开 + `[kvmem].dense_floor_mib = N` ⇒ `Pages(ceil(N·MiB / 页字节))`
            // - 虚拟化开 + 未给数 ⇒ `Pages(工作集页数)` = "只要自己那套装得下就准入"
            //   (搬运与腾位交给 A/B 两轴;这是低显存档的默认姿态)
            // 两个虚拟化分支的真值都由 `derive_turn_ws`(页宽已知)落 —— 这里
            // 只放 `dense_floor_of` 的三态占位。
            dense_floor: Self::dense_floor_of(&cfg.kvmem),
            no_graph: !cfg.cuda.graph,
            fi_cuda_graph: cfg.kernel.fi_cuda_graph,
            // 哨兵:0 = 取 max_seq(FI 调度表的图档上界 = 单序列 ctx 的天然上界;
            // 这样档位改 max_seq 时上界跟着走,不必两处手改)
            fi_plan_bake: cfg.kernel.fi_plan_bake.then_some(cfg.runtime.max_seq),
            fi_plan_restore: cfg.kernel.fi_plan_restore,
            ts_probe: cfg.kernel.ts_probe,
            // S2b 正名:env 新名优先,旧名 `OWL_GDN_SLOTS` 兼容(旧档脚本不炸)
            gdn_slots: owl_shared::env_reader::parse("OWL_MAX_ACTIVE_TURNS")
                .or_else(|| owl_shared::env_reader::parse("OWL_GDN_SLOTS"))
                .unwrap_or(cfg.pool.gdn_slots),
            // 合批上限:配置文件单源(`[runtime] max_batch`;env 不参与 ——
            // 用户裁决 2026-10-19:引擎行为参数只许配置文件显式声明)
            max_batch: cfg.runtime.max_batch,
            prefill_tail_merge: cfg.runtime.prefill_tail_merge,
            slice_every: cfg.runtime.slice_every,
            slice_tokens: cfg.runtime.slice_tokens,
            newcomer_min_tokens: cfg.runtime.newcomer_min_tokens,
            min_gen_tokens: cfg.runtime.min_gen_tokens,
            reserve_pages: cfg.runtime.reserve_pages,
            max_queue: cfg.runtime.max_queue,
            prefill_align_tail: cfg.pool.prefill_align_tail,
            snap_decode_stride: cfg.pool.snap_decode_stride,
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
            env: owl_models::env::EnvProvider::from_config(cfg),
        }
    }
}

/// 草稿域配置(spec 草稿器;深度/家族目录/池量化/verify 诊断 ——
/// 原 knobs 顶层六散字段收编;量化用 KvQuant 枚举,bool 和稀泥废除)
#[derive(Clone, Debug)]
pub struct DraftCfg {
    /// 投机深度(0 = off;结构上限 7)
    pub depth: usize,
    /// 草稿族(auto/mtp/dflash2;显式选择见 `SpecDraft`)
    pub family: owl_shared::config::SpecDraft,
    /// 哑草稿诊断面(无草稿器时)
    pub dumb: bool,
    /// DFlash2 草稿检查点目录(⚠️ W4A16 量化家族 syvai 系)
    pub dflash2_dir: Option<String>,
    /// MTP 头覆盖路径(目录或 `.gguf`;None = 检查点自带)
    pub mtp_dir: Option<std::path::PathBuf>,
    /// 草稿池 KV 存储(None = f16;Fp8E4M3 = 偷显存档,原 draft_kv_fp8 布尔)
    pub kv_quant: owl_models::env::KvQuant,
    /// MTP 草稿注意力滑动窗口(tok;0 = 关;仅 MTP)
    pub mtp_window: usize,
    /// MTP 草稿注意力 sink(tok;最老前缀;仅 window>0 生效)
    pub mtp_sink: usize,
    /// 窗口生效的最小绝对 ctx(tok;0 = 不设额外门槛)
    pub mtp_window_min_ctx: usize,
    /// 自适应 depth（仅 MTP；两档图族 boot 期预捕获）
    pub adaptive_depth: bool,
    /// 自适应档下限 depth
    pub depth_min: usize,
    /// 切档 ctx 阈值（pos >= 阈值用下限）
    pub depth_switch_ctx: usize,
    /// T1b 滚动降档门（tokens/round）
    pub adaptive_min_tpr: f32,
    /// verify taps 退回 last_hidden(排查开关)
    pub notaps: bool,
    /// TAPDECL 二分臂(声明期 tapped 不挂输出槽)
    pub tapdecl: bool,
    /// **TP 档放行投机**(`[spec] tp_draft`;缺省 false ⇒ `tp.rs` 的 v1 护栏仍拒)
    pub tp_draft: bool,
}

impl Default for DraftCfg {
    fn default() -> Self {
        Self {
            tp_draft: false,
            depth: 0,
            family: owl_shared::config::SpecDraft::Auto,
            dumb: false,
            dflash2_dir: None,
            mtp_dir: None,
            kv_quant: owl_models::env::KvQuant::None,
            mtp_window: 0,
            mtp_sink: 64,
            mtp_window_min_ctx: 0,
            adaptive_depth: false,
            depth_min: 3,
            depth_switch_ctx: 49152,
            adaptive_min_tpr: 3.0,
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
    /// 设备面(**唯一咽喉**:分配/归还/换层都在这里发事件;见 `iface::handle`)
    face: DeviceHandle<D>,
    /// **共享账本单元**(与执行态 store 的 `Rel` 同一本;boot 造、随句柄传)
    ledger: owl_store::SharedLedger,
    /// 本引擎句柄使用的**稳定** `LedgerCell`(TP rank1 预接管/幂等 adopt 的键;
    /// `ledger_entry` 每次调用会包新 Arc,不能拿它做同一性判断)
    ledger_cell: owl_iface::handle::LedgerCell,
    cfg: EngineConfig,
    /// boot 显存快照(准入预算装配的真源;S3b §23.3)
    boot: BootVram,
}

/// boot 显存快照(设计卷 §23.3 的实测口径;只读两份数字)
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BootVram {
    /// 开卡后 `cuMemGetInfo` free(字节;CUDA ctx 已在 free 里扣掉 —— 不进底价)。
    /// 无设备账档(CPU mock)= 0。
    v_card_bytes: u64,
    /// `Engine::new` 时刻的引擎 live 账(权重 = 装载后 − 此值)。
    live_at_new: u64,
}

impl BootVram {
    /// 装载前 live 快照(TP 装配的 per-card 权重基线用)
    pub(crate) fn live_at_new(&self) -> u64 {
        self.live_at_new
    }
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
    weights_bytes: u64,
    world: usize,
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
    let page = pool.geom.page.max(1);
    let kv_pool = (pool.geom.nb as u64).saturating_mul(block_bytes as u64);
    let gdn_pool = (pool.geom.gdn_slots as u64).saturating_mul(pool.gdn_row_bytes() as u64);
    let snap_pool = pool.snapshot_pool_bytes() as u64;
    let draft_pool = pool.draft_pool_bytes() as u64;
    let weights = weights_bytes;
    // 运行底价 = run 末 live − 装载后 live − **两 rank** 已归类池面
    // (live 是进程级合计;单卡 world=1 退化为原式)
    let pools_total = (kv_pool + gdn_pool + snap_pool + draft_pool)
        .saturating_mul(world.max(1) as u64);
    let runtime_floor = live_now
        .saturating_sub(live_loaded)
        .saturating_sub(pools_total);
    // 中层预算(host/盘):**唯一算式在 store**(`tier_budget_of`)—— boot 造账与
    // 本处预算入参必须同源(两处各算一遍 = 迟早漂;`Rel::open` 还会交叉校验兜底)。
    let tier_budget = owl_store::tier_budget_of(
        cfg.knobs.host_cache_mb,
        cfg.knobs.snap_host_mb,
        cfg.knobs.disk_cache_mb,
        cfg.knobs.snap_disk_mb,
    );
    let inputs = owl_store::BudgetInputs {
        v_card_bytes: boot.v_card_bytes as usize,
        weights_bytes: weights as usize,
        runtime_floor_bytes: runtime_floor as usize,
        draft_pool_bytes: draft_pool as usize,
        gdn_row_bytes: pool.gdn_row_bytes(),
        kv_bytes_per_token: (block_bytes / page).max(1),
        page_tokens: page,
        // 中层总账 = KV 块腿分片 + 状态腿(快照)分片(S2 §25.2)
        host_bytes: tier_budget.host_bytes,
        disk_bytes: tier_budget.disk_bytes,
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
        pool.geom.nb,
        block_bytes,
        pool.gdn_row_bytes() as f64 / 1048576.0,
        pool.geom.gdn_slots,
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
        let face = spawn_gpu_face(&cfg)?;
        Self::on(cfg, face)
    }

    /// **TP 共享账本构造**(2026-10-20):把"**本 rank 写哪本书、按哪个 rank 记账**"
    /// 在**任何设备动作之前**落定。
    ///
    /// 为什么必须是构造期(不是装配期):事件账的接管判据是 **boot 期(零事件)**
    /// (`DeviceHandle::adopt_ledger`)—— 而**装载即发事件**(0.8B 就有 312 条)
    /// ⇒ 想让两 rank 写同一本,合流只能发生在装载之前。
    ///
    /// `dev_rank` 同样不能晚设:事件按 rank 分组,池复用键带 rank(P0-1)⇒
    /// 漏设会把两 rank 的临时块混池(反例 = 跨卡发句柄)。
    ///
    /// 用法(TP2):rank0 用 [`Self::new`](自造账本) → `rank0.ledger_cell()`
    /// 交给 rank1 的本构造;两侧**然后**再各自 `loader().slice(..)` 装载。
    pub fn new_with_ledger(
        cfg: EngineConfig,
        ledger: owl_store::SharedLedger,
        dev_rank: u8,
    ) -> Result<Self> {
        let face = spawn_gpu_face(&cfg)?;
        let engine = Self::on_with_ledger(cfg, face, ledger)?;
        engine.device().set_dev_rank(dev_rank);
        Ok(engine)
    }
}

/// GPU face 绑定(两个构造入口共用;诊断下沉口在 boot 只组装一次)。
fn spawn_gpu_face(cfg: &EngineConfig) -> Result<GpuClient> {
    GpuClient::spawn_with(
        DeviceSelector::Ordinal(cfg.device_ordinal),
        cfg.knobs.cuda.clone(),
        cfg.knobs.probes.debug,
        owl_shared::log::sink(),
    )
    .map_err(|e| ModelError::Msg(format!("engine: gpu 绑定失败 {e:?}")))
}

impl<D: DeviceClient + 'static> Engine<D> {
    /// 注入 face(CPU 测试/自定义后端;唯一构造机制,`new` 是其 GPU 特化)。
    ///
    /// **账本在这里造**(boot 唯一入口):装载/捕获期的分配也要进账 ⇒
    /// 必须在**任何设备动作之前**拥有账本;窗口侧 store 的 `Rel::open` 拿的是
    /// 同一本(`into_running` 传入)。
    pub fn on(cfg: EngineConfig, face: D) -> Result<Self> {
        let ledger = owl_store::shared_ledger(owl_store::tier_budget_of(
            cfg.knobs.host_cache_mb,
            cfg.knobs.snap_host_mb,
            cfg.knobs.disk_cache_mb,
            cfg.knobs.snap_disk_mb,
        ));
        Self::on_with_ledger(cfg, face, ledger)
    }

    /// 注入 face + **指定账本**(TP:两 rank 必须写同一本;测试:预置账本读)。
    pub fn on_with_ledger(
        cfg: EngineConfig,
        face: D,
        ledger: owl_store::SharedLedger,
    ) -> Result<Self> {
        // boot 显存快照(准入预算的真源;S3b §23.3)—— 必须在任何池/权重之前取
        let boot = BootVram {
            v_card_bytes: owl_shared::vram::free_bytes(),
            live_at_new: owl_shared::vram::live_bytes(),
        };
        let ledger_cell = owl_store::ledger_entry(&ledger);
        // 输出口注入(2026-10-21):`[device-handle]` 的告警/自愈面走统一日志门
        // (level + tag 白黑名单);iface 不依赖 owl-shared ⇒ 在这里显式给。
        let face = DeviceHandle::new(face, Some(ledger_cell.clone()))
            .with_sink(owl_shared::log::sink());
        Ok(Self { face, ledger, ledger_cell, cfg, boot })
    }

    /// **接管已包好的句柄**(TP:两 rank 共用一本账;句柄由 `into_parts` 交出)。
    pub(crate) fn from_handle(
        cfg: EngineConfig,
        face: DeviceHandle<D>,
        ledger: owl_store::SharedLedger,
        boot: BootVram,
    ) -> Result<Self> {
        let ledger_cell = face
            .ledger_cell()
            .unwrap_or_else(|| owl_store::ledger_entry(&ledger));
        Ok(Self { face, ledger, ledger_cell, cfg, boot })
    }

    /// 本引擎的**账本单元**(TP 接管/诊断面)。
    pub fn ledger_cell(&self) -> owl_store::SharedLedger {
        self.ledger.clone()
    }

    /// 设备面(boot 期上下文设置/诊断面)。
    pub fn device(&self) -> &DeviceHandle<D> {
        &self.face
    }

    /// 模型加载器(借用引擎 face;装载即校验,错误在 load 边界落地)。
    /// 面类型 = 句柄(`DeviceHandle<D>`)—— 装载期的每一块都进事件账(Weight 类);
    /// `load` 旋钮来自配置(`[load]` 段;用户裁决 2026-10-19:不读 env)。
    pub fn loader(&mut self) -> ModelLoader<'_, DeviceHandle<D>> {
        ModelLoader::new(&mut self.face, self.cfg.knobs.load.clone())
            .mtp_override(self.cfg.knobs.draft.mtp_dir.clone())
            .rope(self.cfg.knobs.rope.clone(), self.cfg.max_seq_tokens)
            .vision(self.cfg.knobs.vision)
            .mmproj(self.cfg.knobs.mmproj.clone())
    }

    /// 模型交引擎:状态块/槽/Session 组建 + warmup —— 仍不跑任何 turn,
    /// 返回执行态引擎(请求入口)。
    // A3 同步隔离:引擎单线程模型(每卡一线程);Arc 仅作共享句柄,
    // 不跨线程发送 —— clippy 的 Send+Sync 建议不适用。
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn run(self, loaded: LoadedModel) -> Result<RunningEngine<DeviceHandle<D>>> {
        let t_run = std::time::Instant::now();
        let Engine { face, ledger, ledger_cell: _, mut cfg, boot } = self;
        let (rm, tokenizer) = loaded.into_parts();
        // 池分配前的 live 快照 = 权重/杂项(不含池)。单卡直接当 per-card 基线;
        // weights = 装载增量。TP2 由 `TpEngine` 取两 rank 合计后摊分。
        let live_before_pool = owl_shared::vram::live_bytes();
        let weights_bytes = live_before_pool.saturating_sub(boot.live_at_new);
        // 设备侧装配(单卡:tp = None;双 rank 走 `TpEngine` 的同一 `boot_rank`)
        let dev = boot_rank(face, &mut cfg, rm, None, live_before_pool, t_run).await?;
        Ok(into_running(cfg, tokenizer, boot, dev, None, ledger, weights_bytes)?)
    }

    /// 本引擎账本单元(TP2 rank1 **装载前**接管用;与 `TpEngine` 的合流同源)
    pub fn ledger_entry(&self) -> owl_iface::handle::LedgerCell {
        self.ledger_cell.clone()
    }

    /// **装载前接管账本**(TP2 rank1;2026-10-20):
    /// rank1 的模型/视觉塔装载事件必须直接进 rank0 的账本 —— 否则
    /// `TpEngine::assemble` 的 boot-only `adopt_ledger` 会拒(已发事件进孤儿账)。
    /// 幂等:同一本账重复接管 = Ok(`TpEngine` 后续再调一次)。
    pub fn adopt_ledger(
        &mut self,
        cell: owl_iface::handle::LedgerCell,
    ) -> Result<()> {
        self.face.adopt_ledger(cell)
    }

    /// **拆件**(TP 双 rank 装配面:face/cfg/账本同源单持有;S4 ⑤-K2)
    pub(crate) fn into_parts(
        self,
    ) -> (DeviceHandle<D>, EngineConfig, owl_store::SharedLedger, BootVram) {
        (self.face, self.cfg, self.ledger, self.boot)
    }

    /// 绑定设备序号(TP 探测面:UUID/BAR1/P2P 按 ordinal 查)
    pub(crate) fn device_ordinal(&self) -> usize {
        self.cfg.device_ordinal
    }

    /// **boot 显存快照**(TP 装配面:两 rank 共用 rank0 的实测底价)
    pub(crate) fn boot_vram(&self) -> BootVram {
        self.boot
    }
}

/// **进执行态**(RunningEngine 组装):host 侧账本单一,设备束 = rank0(+可选
/// peer rank1);单卡 `Engine::run` 与双 rank `TpEngine::run` 共用。
pub(crate) fn into_running<D: DeviceClient>(
    cfg: EngineConfig,
    tokenizer: owl_models::tokenizer::Tokenizer,
    boot: BootVram,
    dev: RankDev<D>,
    peer: Option<RankDev<D>>,
    ledger: owl_store::SharedLedger,
    weights_bytes: u64,
) -> Result<RunningEngine<D>> {
    // **store 装配(host 半;S3b §23.3)** —— 预算/策略/层布局/快照腿:
    // 单卡与 TP 都**只做一份**(host 账单一)。
    let live_loaded = owl_shared::vram::live_bytes();
    let world = if peer.is_some() { 2 } else { 1 };
    let budget_inputs = assemble_budget(
        &boot,
        &cfg,
        &dev.pool,
        weights_bytes,
        world,
        live_loaded,
        owl_shared::vram::live_bytes(),
    )?;
        let admission_policy = owl_store::AdmissionPolicy::new(
            cfg.knobs.gdn_slots.max(1),
            cfg.knobs.admission_on_full,
        );
        let host_bytes = cfg.knobs.host_cache_mb << 20;
        // **S2:状态腿三层化**(§25.2)—— 注册快照池布局(快照 = 「槽 × 片」),
        // 并把 store 的状态腿宿主层挂上(KV 腿宿主分片之外的部分)。
        // 放在 `SessionBank::new` 之前:布局进 store 后 `Rel` 才知道快照的
        // 单条字节(宿主层的块粒度)。
        let bank = {
            let mut b = crate::session::SessionBank::new(
                crate::session::BankPlan {
                    nb: dev.pool.geom.nb,
                    page: dev.pool.geom.page,
                    block_bytes: dev.pool.block_bytes(),
                    mtp_block_bytes: dev.pool.mtp_block_bytes(),
                    paged: dev.pool.geom.paged,
                    mtp: dev.spec_rt.mode == SpecMode::Mtp,
                    prefix_cache: cfg.knobs.prefix_cache,
                    cache_share_permille: (cfg.knobs.prefix_cache_share * 1000.0)
                        .round()
                        .clamp(0.0, 1000.0) as u16,
                    // 宿主层预算(REQ C4:默认关;0 = 关层)—— 与预算入参**同源**
                    // (腿不得超总账;`Rel::open` 交叉校验)
                    host_bytes,
                    // **TP 宿主腿平面数 = world**(单卡 1;双 rank 2 —— 两个 rank
                    // 的半页 KV 各占一平面,预算仍是两 rank 合计)
                    world: if peer.is_some() { 2 } else { 1 },
                },
                budget_inputs,
                admission_policy,
                dev.pool.pool_layout(0),
                dev.pool.mtp_pool_layout(0),
                ledger,
            )?;
            // **TP:peer 平面布局**(2026-10-20 双卡 KVMem)。两 rank 的池句柄 id
            // 是两套 ⇒ 每个平面必须用**自己那套** `PieceSpec` 组片(靠偏移平移不行)。
            if let Some(pr) = peer.as_ref() {
                b.rel.leg.attach_pool_peer(pr.pool.pool_layout(0))?;
            }
            // **复用池上限**(缺口 4 收尾 P1-1):配置面显式(=0 ⇒ 不限,仅诊断用)
            let cap = cfg.knobs.pool_cap_mb << 20;
            b.rel.set_pool_cap(cap.max(0) as usize);
            owl_shared::log_info!(
                "store",
                "复用池上限 = {}(缺省 256MiB;超限逐块顶出回驱动)",
                if cap == 0 { "不限".to_string() } else { format!("{}MiB", cfg.knobs.pool_cap_mb) }
            );
            // **B-4 驱逐策略装配**(boot 一次;缺省 LRU ⇒ 与历史逐字一致)
            b.rel.set_evict_policy(cfg.knobs.evict_policy.build());
            owl_shared::log_info!("store", "驱逐策略 = {}", b.rel.evict_policy_name());
            if cfg.knobs.index_probe {
                // B 轴索引几何:层 = 全注意力层数,头 = KV 头数,dim = head_dim
                b.rel.attach_index(dev.pool.geom.dims.hd, dev.pool.kvs().len(), dev.pool.geom.dims.hkv)?;
                owl_shared::log_info!(
                    "boot",
                    "选择层索引:采集开(dim={} 层={} KV 头={};每块 {}B)",
                    dev.pool.geom.dims.hd,
                    dev.pool.kvs().len(),
                    dev.pool.geom.dims.hkv,
                    b.rel.index.as_ref().map(|i| i.block_bytes()).unwrap_or(0)
                );
            }
            if dev.pool.geom.paged && dev.pool.snap_capacity() > 0 {
                let epoch = b.rel.attach_snap_layout(dev.pool.snap_pool_layout(0))?;
                if let Some(pr) = peer.as_ref() {
                    b.rel.attach_snap_layout_peer(pr.pool.snap_pool_layout(0))?;
                }
                owl_shared::log_info!(
                    "boot",
                    "状态腿三层化:快照 {} 槽 × {}B/条(布局 epoch {epoch})· 宿主预算 {:.1}MiB",
                    dev.pool.snap_capacity(),
                    dev.pool.snap_bytes(),
                    b.rel.snap_host_budget() as f64 / 1048576.0
                );
            }
            // **磁盘层接线 + 启动回加载**(S6;用户 2026-10-19):配置给配额与目录 ⇒
            // 接盘(两腿分片)+ **扫目录回填账**(内容惰性读:27B 一条快照 159 MiB,
            // 全量预读会把启动拖垮)。大小不符/超预算/重复 = 隔离并逐条 WARN。
            if (cfg.knobs.disk_cache_mb + cfg.knobs.snap_disk_mb) > 0 && !cfg.knobs.disk_dir.is_empty()
            {
                let snap_disk_bytes = if dev.pool.geom.paged && dev.pool.snap_capacity() > 0 {
                    cfg.knobs.snap_disk_mb << 20
                } else {
                    0 // 没快照池 ⇒ 状态腿盘分片无意义(不许静默吃掉配额)
                };
                let attach = b.rel.attach_disk(
                    cfg.knobs.disk_dir.clone(),
                    cfg.knobs.disk_cache_mb << 20,
                    snap_disk_bytes,
                )?;
                // **S6.1:按盘上链索引重建树**(跨重启复用的真身)。
                //
                // 顺序:①内容入账(`adopt_disk`)②索引解码(每条链 ≈32 KB,同步读一次)
                // ③`adopt_disk_chains` 逐链重算块身份 + 按"内容文件在场"截断到连续前缀
                // ④重建条目为**非设备层**(`slot = None`)⑤盘上没被任何重建链引用的
                // 内容 = 孤儿(占配额且永远用不上)⇒ 清掉(索引文件保留:只增不删)。
                let snap_entries =
                    crate::disk_io::scan_dir_sync::<owl_store::SnapKey>(&attach.snap_root);
                let kv_entries =
                    crate::disk_io::scan_dir_sync::<owl_store::ChainHash>(&attach.kv_root);
                let (n_kv, n_snap) = (kv_entries.len(), snap_entries.len());
                let (ra, rb) = b.rel.adopt_disk(kv_entries, snap_entries);
                owl_shared::log_info!(
                    "store",
                    "磁盘层:配额 {}MiB(KV {} / 快照 {})· 目录 {} · 回填内容 KV {}/{} 快照 {}/{} 条 · 隔离 {}+{} 条(内容惰性读回)",
                    cfg.knobs.disk_cache_mb + cfg.knobs.snap_disk_mb,
                    cfg.knobs.disk_cache_mb,
                    snap_disk_bytes >> 20,
                    cfg.knobs.disk_dir,
                    ra.adopted,
                    n_kv,
                    rb.adopted,
                    n_snap,
                    ra.skipped.len(),
                    rb.skipped.len(),
                );
                for (p, why) in ra.skipped.iter().chain(rb.skipped.iter()) {
                    owl_shared::log_warn!("store", "磁盘回填隔离 {}:{why}", p.display());
                }

                let idx_files =
                    crate::disk_io::scan_index_sync::<owl_store::ChainHash>(&attach.kv_root);
                let mut idxs = Vec::with_capacity(idx_files.len());
                for (_, p) in &idx_files {
                    let decoded = std::fs::read(p)
                        .map_err(|e| format!("{e}"))
                        .and_then(|bytes| {
                            owl_store::ChainIndex::decode(&bytes).map_err(|e| format!("{e}"))
                        });
                    match decoded {
                        Ok(i) => idxs.push(i),
                        Err(e) => {
                            owl_shared::log_warn!("store", "链索引不可用 {}:{e}", p.display());
                        }
                    }
                }
                let rep = b.rel.adopt_disk_chains(&idxs)?;
                {
                    let kept: std::collections::HashSet<_> = rep.kept.iter().copied().collect();
                    let orphans: Vec<owl_store::ChainHash> = b
                        .rel
                        .disk
                        .as_ref()
                        .map(|d| {
                            d.resident_keys()
                                .into_iter()
                                .filter(|k| !kept.contains(k))
                                .collect()
                        })
                        .unwrap_or_default();
                    if !orphans.is_empty() {
                        let paths: Vec<std::path::PathBuf> = b
                            .rel
                            .disk
                            .as_ref()
                            .map(|d| {
                                orphans
                                    .iter()
                                    .filter_map(|k| d.path_of(*k).map(|p| p.to_path_buf()))
                                    .collect()
                            })
                            .unwrap_or_default();
                        for p in &paths {
                            let _ = std::fs::remove_file(p);
                        }
                        let freed = b
                            .rel
                            .disk
                            .as_mut()
                            .map(|d| d.commit_evict(&orphans))
                            .unwrap_or(0);
                        owl_shared::log_warn!(
                            "store",
                            "盘上孤儿内容:清掉 {} 块 / {freed}B(无索引或索引截断;索引文件保留)",
                            orphans.len()
                        );
                    }
                }
                owl_shared::log_info!(
                    "store",
                    "跨重启复用:索引 {} 条 ⇒ 重建 {} 链 / {} 块 · 跳过 {} · 截断 {}",
                    idxs.len(),
                    rep.chains,
                    rep.blocks,
                    rep.skipped.len(),
                    rep.truncated.len()
                );
                for (what, why) in &rep.skipped {
                    owl_shared::log_warn!("store", "链重建跳过({what}):{why}");
                }
                for (what, n) in &rep.truncated {
                    owl_shared::log_info!(
                        "store",
                        "链截断({what}):盘上只覆盖前缀 {n} 块(其余内容不在盘上)"
                    );
                }
            }
            b
        };

    let boot = crate::running::BootCtx {
        probes: cfg.knobs.probes,
        tok: tokenizer,
        cfg,
        seq: BOOT_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1,
    };
    Ok(RunningEngine {
        boot,
        turns: crate::turn::TurnLedger::new(),
        bank,
        sel: crate::select::SelectRuntime::default(),
        hk: crate::housekeeping::Housekeeping::default(),
        dev,
        peer,
    })
}

/// **图族 face 数**(spec depth 列表 → 句柄数)。
///
/// 每个 active depth 需要 fold(depth 张)+ propose(depth+1 张) + verify(1 张)；
/// 自适应档两档 = 各自的 fold/propose + 两张 verify。另加 DFlash2 单图 1 张。
/// 第一张 verify 用 `vface`，其余从本预算里领 ⇒ 预算只算**额外** verify。
fn fleet_faces(depths: &[usize]) -> usize {
    let mut n = 1; // DFlash2 propose 单图（非 DFlash2 档留空）
    for (i, &d) in depths.iter().enumerate() {
        n += d.max(1) + (d + 1);
        if i > 0 {
            n += 1; // 额外 depth 的 verify face
        }
    }
    n
}

/// **rank 束**(S4 ⑤-K2/K3):一个 rank 的**全部**状态 —— 模型件 + 执行环境 +
/// 运行态 + 状态池 + 图族。`Engine::run`(单卡)与 `TpEngine`(双 rank)
/// 共用同一装配(`boot_rank`)与同一执行码(设备侧体 = 本类型的方法)。
///
/// # 结构(2026-10-21 S4:装载期事实 / 运行态两分)
///
/// 一个 rank 的状态其实有**两种寿命**:
/// - [`RankBoot`]:装载期定型,之后**只读**(权重/rope/几何/环境/tp/视觉塔);
/// - 运行态:池([`StatePool`])/ 图族([`GraphFleet`])/ spec 域([`SpecRuntime`]),
///   它们会分配、捕获、回放、换页 —— 只有这三件需要 `&mut`。
///
/// 分开之后,"谁能改什么" 在签名上就看得见:设备侧方法拿 `&self` 只能读权重/环境,
/// 要动设备资源必须显式 `&mut` 那三件。
pub(crate) struct RankDev<D: DeviceClient> {
    /// 装载期事实(只读)
    pub(crate) boot: RankBoot,
    /// **当前求值相位**(`env.phase` 的可变部分)。
    ///
    /// 为什么单独立出来:整份 `EnvProvider` 只在 `phase` 上可变 —— 其余
    /// (量化/硬件档/几何/分页)是 boot 期事实。把这一位提成显式字段后,
    /// `env()` = 只读基座 + 本相位(值语义交给解释器),
    /// 相位游标由 [`Self::set_phase`] 单点推进。
    pub(crate) phase: owl_models::env::EvalPhase,
    /// 状态块池(KV/GDN/快照;按 dims 分配)
    pub(crate) pool: StatePool,
    /// 图族(捕获完成 / eager 降级;每 rank 自成一族)
    pub(crate) graphs: GraphFleet<D>,
    /// 引擎级 spec 运行态(模式/深度/草稿器)
    pub(crate) spec_rt: SpecRuntime,
}

/// **装载期定型、运行期只读**的 rank 事实(权重是分片的,每 rank 一份)。
///
/// 纪律:本结构的字段**任何运行期代码都不许写**(只能在 `boot_rank` 构造);
/// 因此字段可直接读,不需要 setter —— 只读性由类型 + 构造点保证。
pub(crate) struct RankBoot {
    /// 模型声明树
    pub(crate) model: Arc<Model>,
    /// rope 表(Replicate;每 rank 一份设备表)
    pub(crate) rope: Rope,
    /// **分片 spec**(形状单源;池与层同源)
    pub(crate) spec: ModelSpec,
    /// 解释器执行环境**基座**(相位除外;相位住 `RankDev::phase`)
    pub(crate) env: owl_models::env::EnvProvider,
    /// 张量并行上下文(本 rank 的;`None` = 单卡)—— 执行面所有求值调用
    /// 把它传给解释器(行并行 allreduce 注入靠它;装配期填入,烘进每张图)
    pub(crate) tp: Option<owl_models::module::TpCtx>,
    /// **视觉塔装配**(`[model].vision`;Some = VLM)。塔是 prefill-only 的
    /// 额外权重;文本 `model` 与它是同一份 Arc(见 `Vlm::from_model`)。
    pub(crate) vlm: Option<std::sync::Arc<owl_models::vlm::Vlm>>,
}

impl<D: DeviceClient> RankDev<D> {
    /// 当前求值环境 = boot 基座 + 本相位(**值语义**;直接交给解释器)。
    ///
    /// 为什么不返回 `&EnvProvider`:解释器吃的是按值传的 env(相位要在调用点
    /// 定住,不随后续 `set_phase` 漂移)。
    pub(crate) fn env(&self) -> owl_models::env::EnvProvider {
        self.boot.env.with_phase(self.phase)
    }

    /// 当前相位(相位栈保存/恢复用)。
    pub(crate) fn phase(&self) -> owl_models::env::EvalPhase {
        self.phase
    }

    /// 推进相位游标(不碰图族 —— 图族相位由引擎 [`crate::RunningEngine`]
    /// 同步推进;两处语义不同,不能隐式联动)。
    pub(crate) fn set_phase(&mut self, p: owl_models::env::EvalPhase) {
        self.phase = p;
    }
}

/// **rank 设备侧装配(唯实现)**:几何 → 环境 → spec 域 → KV 写者门 →
/// 状态池 → 图族(warmup + 捕获)。
///
/// `tp` = 张量并行上下文(**必须在图捕获之前**传入 —— 捕获期拒三个 `Peer*`
/// 命令;staging 的 VMM 握手由调用方在更早完成,见 `tp.rs`)。
#[allow(clippy::arc_with_non_send_sync)]
/// **boot 配置定形门**(S6a;纯判据,任何设备动作之前)。
///
/// 这些组合一旦放进运行期才爆,现场都很难查(0.8B 桶门 IMA / 首个 verify
/// 步越界 / 图档瓦与表内容不同构)。集中在这里 = 坏配置毫秒级拒。
fn boot_knobs_check<D: DeviceClient>(
    face: &mut DeviceHandle<D>,
    cfg: &EngineConfig,
    tp: Option<&owl_models::module::TpCtx>,
) -> Result<()> {
    // **FI plan 自携开关:引擎 = 配置持有者 ⇒ 装配期下发**(2026-10-19 桶门立案)。
    //
    // 双源曾分叉:引擎旋钮 `knobs.fi_plan_restore`(桶图用它决定"图自携 vs
    // host 步前刷表")与后端 `DiagOpts.fi_plan_restore`(家族用它决定"窗内
    // 烘不烘 restore 节点")各读各的。测试档只设前者 ⇒ 家族按 false 跑
    // ⇒ **图里没有 restore 节点、引擎又不刷表** ⇒ 回放读到上一个 plan 的
    // 调度表 ⇒ 0.8B 桶门 IMA(`q_indptr[5]` 越界)/错值。
    // 单源 = 引擎旋钮(生产两处同源于 `[dispatch]`,行为不变;测试/直连档
    // 以后者为准)。写点必须在**图族装配之前**(boot_rank 内)。
    face.configure_fi_plan(cfg.knobs.fi_plan_restore, cfg.knobs.fi_cuda_graph)?;
    // **合批档自洽门**(B5.1;2026-10-20):每会话一格 GDN + 单卡 AR 语义未审
    if cfg.knobs.max_batch > 1 {
        if tp.is_some() {
            return Err(owl_iface::contract::ModelError::Msg(
                "[runtime] max_batch > 1 与 TP 不共戴天(AR 的批语义未审;先单卡)".into(),
            ));
        }
        if cfg.knobs.max_batch > 8 {
            return Err(owl_iface::contract::ModelError::Msg(format!(
                "max_batch({}) > 8:逐行 token 输出走 Concat 8 槽",
                cfg.knobs.max_batch
            )));
        }
        if cfg.knobs.gdn_slots < cfg.knobs.max_batch {
            return Err(owl_iface::contract::ModelError::Msg(format!(
                "gdn_slots({}) < max_batch({})—— 合批每会话要一格 GDN 状态",
                cfg.knobs.gdn_slots, cfg.knobs.max_batch
            )));
        }
        if cfg.knobs.select_decode || cfg.knobs.select_partial {
            return Err(owl_iface::contract::ModelError::Msg(
                "max_batch > 1 与选择层视图不共戴天(M1 页表按行铺)".into(),
            ));
        }
    }
    // **投机深度定形门**(2026-10-16):verify 块 = `depth + 1` 行,而首轮
    // 拼接核的输入槽数是**声明面事实**(`CONCAT_ROWS_SLOTS` = 8)⇒ depth 超界
    // 会在**首个 verify 步**才爆(且那时树已经 pad 到错槽数)。
    // 放在**任何设备动作之前**(连草稿器都不加载)⇒ 坏配置毫秒级拒。
    if cfg.knobs.draft.depth >= owl_models::kernel::CONCAT_ROWS_SLOTS {
        return Err(owl_iface::contract::ModelError::Msg(format!(
            "[spec] depth = {} 超界:verify 行数 = depth+1 必须 ≤ {}(首轮拼接槽数);\
             请调小 depth(或关 spec 走 dumb 草稿)",
            cfg.knobs.draft.depth,
            owl_models::kernel::CONCAT_ROWS_SLOTS
        )));
    }
    Ok(())
}

/// **每 turn 工作集派生 + 池下限校验 + 自证行**(S6a)。
///
/// 语义:配置只写 `workspace_per_req`(token),这里派生成三件(页/字节/召回 top-k),
/// 再按"并发 × (工作集 + chunk) ≤ 池页数"校验(仅虚拟化档)。
fn boot_pool_floor(
    cfg: &mut EngineConfig,
    pool: &StatePool,
    s: usize,
    plan_pool_tokens: usize,
) -> Result<()> {
    let cfg: &mut EngineConfig = cfg;
    let page_bytes = pool.block_bytes().max(1);
    let (ws_pages, ws_bytes) = EngineKnobs::derive_turn_ws(
        cfg.knobs.workspace_per_req,
        pool.geom.page,
        page_bytes,
        &mut cfg.knobs,
    )?;
    // chunk 增量页 / 每 turn 页 / 并发下限(与下面的池校验**同一批数**)
    let keep_pages = cfg.knobs.select_sink + cfg.knobs.select_recent;
    let chunk_pages = cfg.prefill_chunk.div_ceil(pool.geom.page.max(1)).max(1);
    let per_turn = ws_pages.max(1) + chunk_pages;
    let n_active = cfg.knobs.gdn_slots.max(1);
    let need = per_turn * n_active;
    let vm_on = cfg.knobs.kvmem_virtualization;
    // **boot 自证行**(装配期无条件打):工作集口径与三件派生一次打全 ——
    // 配置里只写 token,这一行告诉你它变成了几页/几字节/多少找回预算。
    // 打印用 `eprintln!`(取证行惯例):`log_*` 宏在无 `log` feature 的
    // 测试/探针构建里被**编译剔除**(`if false`)—— 自证行必须在探针里可见。
    //
    // 地板一列打**装配后的真值**(A7):非虚拟化档 = `Strict`,虚拟化档 =
    // 工作集页数或显式 MiB 换来的页数 —— 不再拿 `ws_bytes` 冒充地板。
    let floor_desc = match cfg.knobs.dense_floor {
        owl_store::TurnFloor::Strict => "Strict(全 ctx 页)".to_string(),
        owl_store::TurnFloor::Pages(n) => {
            format!("{n} 页 = {} KiB/turn", (n * page_bytes) >> 10)
        }
    };
    owl_shared::log_info!(
        "turn-ws",
        "每 turn 工作集 = {} tok = {} 页 = {} KiB(页 {} tok / {} KiB)⇒ \
         常驻 {} 块 = 保底 sink {} + recent {} + 召回 top-k {};准入地板 = {};\
         池下限 = {} 并发 × (工作集 {} + chunk {} = {} 页) = {} 页(池 {} 页){}\
         (改工作集只动 `[kvmem].workspace_per_req`;`select_blocks` 是派生量)",
        cfg.knobs.workspace_per_req,
        ws_pages,
        ws_bytes >> 10,
        pool.geom.page,
        page_bytes >> 10,
        ws_pages,
        cfg.knobs.select_sink,
        cfg.knobs.select_recent,
        ws_pages - keep_pages,
        floor_desc,
        n_active,
        ws_pages,
        chunk_pages,
        per_turn,
        need,
        pool.geom.nb,
        if !vm_on {
            "(非虚拟化档:不做池下限校验)"
        } else if pool.geom.nb >= need {
            " ✓"
        } else {
            " ✗"
        }
    );
    if cfg.knobs.kvmem_virtualization {
        if pool.geom.nb < need {
            return Err(owl_iface::contract::ModelError::Msg(format!(
                "boot: 设备池 {} 页装不下 {n_active} 并发 × {per_turn} 页 = {need} 页\
                 (每 turn = workspace_per_req {} tok = {} 页 + chunk {chunk_pages} 页;\
                  其中工作集由 sink {} + recent {} + top-k {} 组成);\
                 pool_tokens={} page={} —— 调大池 / 减工作集 / 降 gdn_slots\
                 (响亮拒:跑起来必 Exhausted)",
                pool.geom.nb,
                cfg.knobs.workspace_per_req,
                ws_pages,
                cfg.knobs.select_sink,
                cfg.knobs.select_recent,
                cfg.knobs.select_blocks,
                plan_pool_tokens,
                pool.geom.page
            )));
        }
        // **准入地板缺省 = 本 turn 工作集页数**(不是 0 页!)
        //
        // 地板语义 = "本 turn 至少多少 KV **页**直接驻留设备"。并发档下若给 0,
        // N 个 turn 会各自被无条件准入、然后互抢池 ⇒ 抖动;给工作集页数
        // = "每个进来的 turn 都先留出它自己那套",与上面的池下限算式**同源**
        // (同一套页数)。显式给了 `[kvmem].dense_floor_mib` 则尊重显式值。
        // (两件都**已由 `derive_turn_ws` 按池页宽填好** —— 这里不再二次算式。)
        let _ = ws_bytes;
        owl_shared::log_warn!(
            "boot",
            "KVMem 虚拟化档:**池 = {} tok / {} 页** vs 上下文 {} tok / {} 页 ⇒ 超卖 {:.2}×;\
             并发上限 {} × 每 turn {} 页(工作集 {} tok=sink{}+recent{}+top-k{} + chunk {}) = {need} 页;\
             每页 {} KiB;准入地板 = {} / turn;宿主 {} MiB",
            pool.geom.nb * pool.geom.page,
            pool.geom.nb,
            s,
            s.div_ceil(pool.geom.page.max(1)),
            s as f64 / (pool.geom.nb * pool.geom.page).max(1) as f64,
            n_active,
            per_turn,
            cfg.knobs.workspace_per_req,
            cfg.knobs.select_sink,
            cfg.knobs.select_recent,
            cfg.knobs.select_blocks,
            chunk_pages,
            page_bytes / 1024,
            floor_desc,
            cfg.knobs.host_cache_mb + cfg.knobs.snap_host_mb,
        );
    }
    Ok(())
}

/// **装配后门**(S6a):依赖"图族已建"的组合检查。
fn boot_post_check<D: DeviceClient>(
    spec: &crate::spec_runtime::SpecRuntime,
    graphs: &GraphFleet<D>,
    model: &Model,
) -> Result<()> {
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
        && graphs.verify_of(spec.depth).is_none()
    {
        return Err(owl_iface::contract::ModelError::Msg(
            "DFlash2 spec 需要 verify 图(taps 输出),但本档未建图\
             ([graph] no_graph = true 或捕获失败)。请二选一:去掉 no_graph,\
             或显式关闭该档([spec] depth = 0;或 [spec] notaps = true 退 last_hidden)"
                .to_string(),
        ));
    }

    // ── 词表切 × 草稿档（DFlash2/MTP）= **boot 拒启**（P1.1 × P2.1，2026-10-19）──
    // 两族的草稿头都**共享 target 的 lm_head**（`dflash2.rs` 的 propose /
    // `mtp.rs` 的 `embed.lm_head_matmul`），且选择面按**全量 vocab** 索引
    // （DFlash2 码本 [vocab, rank] / MTP argmax stride = vocab）⇒ 词表切档下
    // 草稿 logits 只剩半表：候选塌到半词表。
    // **实测**（3080 对，depth=7）：AL 2.4 → 1.4 ⇒ B4 降级把 spec 档拖到裸
    // decode（**104 → 40 t/s**）。该组合的解 = 草稿选择面半表化（两 rank 各出
    // 候选 + host 合并 + 码本行偏移）——**未施工** ⇒ 此档**响亮拒**
    // （静默放行 = 用户跑出“投机失效”的黑盒）。
    // dumb 草稿无头 ⇒ 不受影响；`depth = 0` ⇒ 无草稿档。
    // ── 词表切 × 草稿档：**DFlash2 图档已支持 / 其余仍拒**（P1.1 × P2.1）──
    // 两族的草稿头都**共享 target 的 lm_head**（`dflash2.rs` / `mtp.rs`），
    // 选择面按**全量 vocab** 索引 ⇒ 切档后草稿 logits 只剩半表（实测
    // AL 2.4→1.4 ⇒ 降级裸 decode 104→40 t/s）。
    // **已解**：DFlash2 **图档** —— propose 图出半表 top-16 候选 + proj，
    // host 合并两 rank 候选（rank1 id + half），rank0 跑 select（全表域）。
    // **仍拒**：MTP（未做）/ DFlash2 无图（eager 降级档，未做）。
    if matches!(spec.mode, SpecMode::DFlash2 | SpecMode::Mtp)
        && model.logits_vocab() != model.vocab_size()
    {
        let dflash_graph_ok =
            spec.mode == SpecMode::DFlash2 && graphs.dflash.is_some();
        if !dflash_graph_ok {
            return Err(owl_iface::contract::ModelError::Msg(
                "词表切(runtime.tp_vocab_split)与**本草稿档**不兼容：".to_string()
                    + "草稿头共享 target lm_head ⇒ 切档后草稿 logits 只剩半表。\
                       已解 = DFlash2 **图档**（需 `[graph] no_graph = false` 且捕获成功）；\
                       未解 = MTP / DFlash2 eager 降级档。\
                       请二选一：关 runtime.tp_vocab_split，或换到已解档。",
            ));
        }
    }

    Ok(())
}

/// **执行环境 + spec 域 + 选核快照 + KV 清单门**(S6a)。
///
/// 纪律:引擎 = 环境事实来源(硬件档/几何/量化都在这里合入),算子层零探测。
async fn boot_env_and_spec<D: DeviceClient + 'static>(
    face: &mut DeviceHandle<D>,
    cfg: &EngineConfig,
    rm: &RankModel,
    tp: Option<&owl_models::module::TpCtx>,
) -> Result<(
    ModelDims,
    owl_models::env::EnvProvider,
    Option<owl_models::env::KvQuant>,
    crate::spec_runtime::SpecRuntime,
)> {
    let (hq, hkv, hd) = rm.spec.full_heads;
    let (nk, hk, nv, hv) = rm.spec.gdn_heads;
    let dims = ModelDims {
        hq, hkv, hd, nk, hk, nv, hv,
        hidden: rm.spec.hidden,
        dtype: rm.spec.dtype,
    };

    // 解释器执行环境(EnvProvider;E1.5 抽象):旋钮携带(入口经
    // from_config 填充)+ 硬件档自 iface op_env 合入(被动律:引擎 = 环境事实来源)
    let mut env = cfg.knobs.env;
    if let Some(oe) = face.op_env() {
        env.hw = owl_models::env::HwEnv::Cuda(oe.hw.arch);
    }
    // 模型几何事实合入(2026-10-13 P4:算子准入的几何门对账面)——
    // 与 hw 同律:引擎 = 环境事实来源,算子层零探测
    env.kv.head_dim = hd;
    env.kv.grp = hq.checked_div(hkv).unwrap_or(0);
    // 逐张量计划(GGUF)不注入单值 —— 计划的**真源**已在构造期烧进层容器
    // (`PlanSource::Keyed`);此字段只服务"单一方案"档位的一致性/准入面。
    if let Some(q) = rm.quant_plan {
        env.quant = q;
    }
    // FI 面启用(量化恒 = 主池;影子池已删,本值只作"启用 + 池量化"携带)
    // FI 面启用条件 = **能力条件**(f16 池 + paged 页表),不是开关:
    // 2026-10-13 起配置面无 flashinfer 选项,FI 恒启用(env.attn.fi 仅门/诊断用)
    let fi_quant = if env.attn.fi
        && env.kv.paged
        && rm.spec.dtype == owl_models::tensor::Dtype::F16
    {
        Some(env.kv.quant)
    } else {
        None
    };

    // spec 域装载(spec_runtime.rs:模式三态裁决 + 草稿器 + 草稿 rope;
    // E5-M2b/C7 —— depth>0 且检查点有 mtp.* → Mtp;dumb 诊断位 → Dumb;
    // 否则 Off 回落 DecodeBatch,无草稿器不白发 verify 税)
    let mut spec = crate::spec_runtime::load_spec_runtime(
        face,
        SpecBootInput {
            depth_raw: cfg.knobs.draft.depth,
            family: cfg.knobs.draft.family,
            dflash2_dir: cfg.knobs.draft.dflash2_dir.as_deref(),
            mtp_dir: rm.mtp_dir.as_deref(),
            spec: &rm.spec,
            dumb: cfg.knobs.draft.dumb,
            draft_kv_fp8: cfg.knobs.draft.kv_quant == owl_models::env::KvQuant::Fp8E4M3,
            mtp_window: cfg.knobs.draft.mtp_window,
            mtp_sink: cfg.knobs.draft.mtp_sink,
            mtp_window_min_ctx: cfg.knobs.draft.mtp_window_min_ctx,
            adaptive_depth: cfg.knobs.draft.adaptive_depth,
            depth_min: cfg.knobs.draft.depth_min,
            depth_switch_ctx: cfg.knobs.draft.depth_switch_ctx,
            adaptive_min_tpr: cfg.knobs.draft.adaptive_min_tpr,
            // 草稿共享的 target 事实(GGUF 草稿声明校验 + 码本行数)
            target: owl_models::formats::gguf::TargetFacts {
                vocab: rm.spec.vocab,
                layers: rm.spec.layer_types.len(),
                hidden: rm.spec.hidden,
            },
            notaps: cfg.knobs.draft.notaps,
            // P2.1-a:本 rank 的 TP 计划(单卡 = None ⇒ 旧行为)
            tp: tp.as_ref().map(|c| c.plan),
            load: cfg.knobs.load.clone(),
            // 草稿环境基底(已合入目标几何/硬件档;草稿只覆盖 hd/grp)
            base_env: env,
        },
    )
    .await?;

    // **KernelSelectionEnv 快照**(2026-10-20):boot 事实投影给 picker。
    // 必须在此处冻结(硬件/几何/量化/spec 已定,池/图尚未分配);之后
    // `EnvProvider::op_env()` 只读这份快照,不再从散字段重算。
    let features = owl_models::kernel_env::FeatureCaps {
        vision: cfg.knobs.vision,
        spec_depth: spec.depth,
        tp_ranks: tp.as_ref().map(|c| c.plan.world()).unwrap_or(1),
        head_int8: cfg.knobs.head_quant == owl_shared::config::HeadQuant::RtnInt8Row,
        head_int4: cfg.knobs.head_quant == owl_shared::config::HeadQuant::RtnInt4G32,
        main_kv: Some(owl_models::kernel_env::KvCaps::from_env(&env)),
        draft_kv: (spec.depth > 0)
            .then(|| owl_models::kernel_env::KvCaps::from_env(&spec.draft_env)),
        tag_wildcard: cfg.knobs.tag_wildcard,
    };
    if let Some(k) = owl_models::kernel_env::KernelSelectionEnv::from_env(&env, features) {
        owl_shared::log_info!("kernel-env", "target: {}", k.describe());
        env.kernel_env = Some(k);
    }
    // 草稿有自己的几何(hd=128 / grp):快照必须从 draft_env 单独投影。
    // MTP 复用标准 attention 核，其 F16 链池声明在 `kv.main.*`
    // 命名空间（KnHD）；因此草稿快照的 main_kv 事实 = 草稿自己的 F16 池，
    // 而不是 target 的 fp8 池。DFlash2 仍按草稿池 tag 走。
    let mut draft_features = features;
    if spec.mode == SpecMode::Mtp {
        draft_features.main_kv =
            Some(owl_models::kernel_env::KvCaps::from_env(&spec.draft_env));
    }
    if let Some(k) =
        owl_models::kernel_env::KernelSelectionEnv::from_env(&spec.draft_env, draft_features)
    {
        owl_shared::log_info!("kernel-env", "draft:  {}", k.describe());
        spec.draft_env.kernel_env = Some(k);
    }

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

    Ok((dims, env, fi_quant, spec))
}

/// **PoolPlan 构造**(S6a):把配置/spec/几何折成池的分配需求。
fn boot_pool_plan(
    cfg: &EngineConfig,
    s: usize,
    env: &owl_models::env::EnvProvider,
    spec: &crate::spec_runtime::SpecRuntime,
    fi_quant: Option<owl_models::env::KvQuant>,
    tp: Option<&owl_models::module::TpCtx>,
    live_baseline_per_card: u64,
) -> crate::state::PoolPlan {
    // **自动档**(2026-10-21):`[pool].pool_tokens` 未写 ⇒ 池 = 本档 ctx,
    // 由 `StatePool::alloc` 按显存预算校验并回报 `ctx_max`(不再 2× 单会话)。
    // `pool_tokens` 缺省**或 0** ⇒ 自动(= 本档 max_seq)。0 是必要的逃生口:
    // 档位走 `extends` 继承时,子档"不写"会继承父档的值,只有显式 `0` 能表达
    // "本档要自动"(而 `PoolCfg::pinned_pool_tokens` 也把 0 视为未钉)。
    let pool_auto = cfg.knobs.pool_tokens.map_or(true, |n| n == 0);
    let pool_tokens = cfg.knobs.pool_tokens.filter(|n| *n > 0).unwrap_or(s);
    let plan = crate::state::PoolPlan {
        seq_tokens: s,
        pool_tokens,
        pool_auto,
        ws_tokens: cfg.knobs.workspace_per_req,
        // **锁①门控**:只有虚拟化真的开着(配置显式 `select_decode` +
        // `select_partial`)才允许池小于上下文 —— 否则链超出池容量必
        // `Exhausted`(无洞可停,即无选择层可挤)。
        pool_overcommit: cfg.knobs.kvmem_virtualization,
        kv: env.kv.quant,
        fi_quant,
        draft_kv: if cfg.knobs.draft.kv_quant == owl_models::env::KvQuant::Fp8E4M3 {
            owl_models::env::KvQuant::Fp8E4M3
        } else {
            owl_models::env::KvQuant::None
        },
        // P2.1-b:草稿池几何(**分片后**;来自草稿声明 + 本 rank 计划)
        draft_geom: spec.draft_geom,
        // 治理器按卡摊 `live`(TP2 下 LIVE 含两卡之和;2026-10-19 定谳)
        tp_ranks: tp.as_ref().map(|c| c.plan.world()).unwrap_or(1),
        // 两 rank 共用的池前 live 基线(消除顺序记账)
        live_baseline_per_card: Some(live_baseline_per_card),
        // 草稿池 dtype 同源(量化族 BF16 / 未量化族 F16;见 spec_runtime::draft_dt)
        draft_dt: spec.draft_dt,
        spec_mode: spec.mode,
        spec_depth: spec.depth,
        xqa_verify: matches!(cfg.knobs.env.attn.verify_arm, owl_shared::config::VerifyArm::Xqa),
        // B-3e-2:部分驻留档**不带 XQA**(页表语义不共戴;见上方醒目告知)
        xqa_decode: matches!(cfg.knobs.env.attn.decode_arm, owl_shared::config::DecodeArm::Xqa)
            && !cfg.knobs.select_partial,
        xqa_prefill: cfg.knobs.env.attn.xqa_prefill,
        gdn_slots: cfg.knobs.gdn_slots,
        max_batch: cfg.knobs.max_batch,
        snap_max: cfg.knobs.snap_max,
        vram_target: cfg.knobs.vram_target,
        vram_reserve_mb: cfg.knobs.vram_reserve_mb,
        dump_all: cfg.knobs.gdn_dump_all,
        gdn_dbg: cfg.knobs.probes.gdn_dbg,
        // B-2:选择层视图面(**生产档也要** —— 视图是 decode 的消费方;
        // 与诊断面 `coverage_probe` 解耦)
        select_view: cfg.knobs.coverage_probe || cfg.knobs.select_decode,
    };
    plan
}

/// **KVMem 前置校验**(S6a):虚拟化三件套的自洽门(响亮拒,不静默降级)。
fn boot_kvmem_check(cfg: &EngineConfig) -> Result<()> {
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
    if cfg.knobs.select_partial && !cfg.knobs.prefix_cache {
        return Err(owl_iface::contract::ModelError::Msg(
            "boot: `[kvmem].select_partial = true` 要求 `[pool].prefix_cache = true` \
             —— 下沉靠「缓存树 + 会话双持」(refs==2)才能 decref 放给宿主;\
             关着 prefix_cache ⇒ 缓存树 cap=0 ⇒ 块永远 refs==1 ⇒ **一块都沉不下去** \
             (响亮拒:跑起来会白跑一趟还报 '不可沉 N 块')"
                .into(),
        ));
    }
    if cfg.knobs.select_partial && !cfg.knobs.select_decode {
        owl_shared::log_error!(
            "boot",
            "OWL_SELECT_PARTIAL=1 需同时开 OWL_SELECT_DECODE=1(链上出洞后必须有视图消费方)—— 拒绝启动"
        );
        std::process::exit(2);
    }
    if cfg.knobs.select_partial
        && matches!(cfg.knobs.env.attn.decode_arm, owl_shared::config::DecodeArm::Xqa)
    {
        owl_shared::log_warn!(
            "boot",
            "部分驻留档:本档**关掉 XQA 裸 decode 臂**(其页表语义读整条会话链,遇占位符 = 非法地址;\n                    \
             视图×XQA 的对接是挂账,不在此刀)—— 本档走 v2 分页 decode"
        );
    }
    Ok(())
}

pub(crate) async fn boot_rank<D: DeviceClient + 'static>(
    mut face: DeviceHandle<D>,
    cfg: &mut EngineConfig,
    rm: RankModel,
    tp: Option<owl_models::module::TpCtx>,
    live_baseline_per_card: u64,
    t_run: std::time::Instant,
) -> Result<RankDev<DeviceHandle<D>>> {
        boot_knobs_check(&mut face, cfg, tp.as_ref())?;
        let s = cfg.max_seq_tokens;
        // ⚠️ **已知缺口(§18.5 挂账①)**:CPU/mock 档若要跑引擎真前向,
        // `dtype` 不能简单归一为 F32 —— `kv_paged_policy(dtype)` 与
        // attn v2 scratch 注入都以**模型 dtype** 为前提(F32 ⇒ 无分页策略
        // ⇒ 走 legacy v1 臂 ⇒ attention.rs 的退役 panic)。所需修法:
        // 让分页策略/scratch 与执行域 dtype 解耦(单独一刀),不在 mock 刀里扫。
        let (dims, env, fi_quant, spec) =
            boot_env_and_spec(&mut face, cfg, &rm, tp.as_ref()).await?;
        // 状态块(StatePool::alloc,零初始化;GDN 段每 turn 开始按格重置)。
        // KV 布局表驱动(kv_paged_policy;REQ-HW-01);块池容量 E2b:
        // 默认 = 2 × 单会话容量(两会话满载共存),knobs.pool_tokens 覆写
        let plan = boot_pool_plan(cfg, s, &env, &spec, fi_quant, tp.as_ref(), live_baseline_per_card);
        boot_kvmem_check(cfg)?;
        // **池骨架 = `Pool` 类**(boot 相位;不加这个上下文就会落“未分类”桶 ——
        // 可见但无用)。作用域 RAII:块分配途中任何早退都会还原上下文。
        let pool = {
            let _scope = face.scoped_ctx(Owner::System, AssetClass::Pool);
            StatePool::alloc(&mut face, dims, &rm.spec.layer_types, &plan).await?
        };

        // **MTP window 的 v2 依赖门**(2026-10-21):window 只改 draft 读，
        // 但必须走 v2 伪序列（decode T=1 / extend 多行 / chain）；缺
        // `attn_v2_scratch` 或 `verify_attn_v2_scratch` 就会落 full lens
        // 的 chunked 语义，window 表会静默错位 ⇒ boot 拒，不自动回退。
        if spec.mtp_window.active() {
            if pool.attn_v2_scratch().is_none()
                || pool.verify_attn_v2_scratch().is_none()
            {
                return Err(owl_iface::contract::ModelError::Msg(
                    "[spec] mtp_window > 0 需要 v2 注意力 scratch（attn_v2 + verify_attn_v2）；\
                     当前池缺其一，window 读会落 chunked/full lens 语义 ⇒ 拒启"
                        .into(),
                ));
            }
            let page = pool.geom.page.max(1);
            let sink_eff = spec.mtp_window.sink.div_ceil(page) * page;
            let window_eff = spec.mtp_window.window.div_ceil(page) * page;
            owl_shared::log_info!(
                "boot",
                "MTP window: window={} sink={} min_ctx={} page={} => sink_pages={} window_pages={} (effective sink={} window={})",
                spec.mtp_window.window,
                spec.mtp_window.sink,
                spec.mtp_window.min_ctx,
                page,
                sink_eff / page,
                window_eff / page,
                sink_eff,
                window_eff
            );
        }

        // **锁①:虚拟化档的池容量硬校验**(2026-10-19;响亮拒,不静默落空槽)。
        //
        // 几何一律从**池实况**取(`pool.geom.page`/`pool.geom.nb`),不另算一份。
        //
        // # 下限的算式(为什么乘 N_active)
        //
        // ```text
        // 池页数 ≥ N_active × (select_blocks + sink + recent + chunk/page)
        //          └─ 每个活跃 turn 都要**自己一套**工作集与 chunk 工作区
        // ```
        //
        // `N_active` = `gdn_slots`(并发上限)。**这一项不能省** ——
        // 并发档下每个 turn 各有自己的工作集,池要能同时装下它们。
        // 27B f16 算例(页 = 32 tok = 2 MiB):
        //   select_blocks 32 + sink 2 + recent 4 + chunk 256 ⇒ 每 turn 46 页 = 92 MiB
        //   8 并发 ⇒ 368 页 = 736 MiB ‖ 对照 dense 8 并发 = 8×256 页 = 4 GiB ⇒ 省 5.6×
        // **每 turn 工作集派生**(唯一产地;对外一个 token 数 ⇒ 内部三件)。
        // 位置 = 装配期(页几何已知)、**与虚拟化档无关** —— 诊断档(只跑选择器
        // 读数)也要同一口径,否则配置写了 token 却被静默忽略。
        boot_pool_floor(cfg, &pool, s, plan.pool_tokens)?;
    {
        // **池的落账行**(2026-10-21):算式上限(`[ctx-max] cap`)与真实可服务
        // 上限之间还差一块**未入账固定开销**(CUDA ctx + FI ws + 图 slab + 碎片,
        // 本机实测 ≈1.6 GiB)+ 每 turn 瞬态(34~300 MiB)。本行只报池本身,
        // 经验值留在 profile 档头/bench 卷(216k ✓ / 220k 差 17 MiB / 224k ✗)。
        let pool_mib = (pool.geom.nb as u64 * pool.block_bytes() as u64) >> 20;
        eprintln!(
            "[ctx-max] 池落账 {} MiB({} tok × {} 页宽;工作集 {} tok 常驻,Strict 准入)",
            pool_mib,
            pool.geom.nb * pool.geom.page,
            pool.geom.page,
            cfg.knobs.workspace_per_req
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
        let vface = face.face_clone();
        // E5-M5:fold/propose 桶形图族的 face 句柄(depth × 2)
        let depth_candidates = spec.depth_candidates();
        let extra_faces: Vec<DeviceHandle<D>> =
            (0..fleet_faces(&depth_candidates)).filter_map(|_| face.face_clone()).collect();
        let graphs = GraphFleet::assemble(
            face,
            vface,
            extra_faces,
            FleetBoot {
                env,
                tp: tp.clone(),
                dims,
                pool: &pool,
                model: std::sync::Arc::clone(&rm.model),
                rope: rm.rope.clone(),
                spec: &spec,
                knobs: &cfg.knobs,
            },
            t_run,
        )
        .await?;

        // ── 准入预算组装 + 会话/块账房(S3b;§23.3)──
        // 此刻固定面全部落位(权重 + 池 + 图 slab + 表件)⇒ 实测数字齐;
        // 装配失败 = 拒绝启动(A5.1:没有数字的项不许进预算式)。

        boot_post_check(&spec, &graphs, &rm.model)?;
    let boot_env = env;
    Ok(RankDev {
        boot: RankBoot {
            vlm: rm.vlm,
            model: rm.model,
            rope: rm.rope,
            spec: rm.spec,
            env: boot_env,
            tp,
        },
        phase: boot_env.phase,
        spec_rt: spec,
        pool,
        graphs,
    })}
