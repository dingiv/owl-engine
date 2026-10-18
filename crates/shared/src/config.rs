//! 全量配置总账 + 统一 loader(2026-10-10 显式配置模块;显式依赖律终点)。
//!
//! **定位**:全 workspace 唯一的配置**定义面** —— 每个配置键在此登记
//! (键名/类型/缺省/归属 module),[`OwlConfig::load`] 是唯一 loader。
//! 消费方(engine/cuda/models/server)一律 `from_config(&OwlConfig)`
//! 组装各自的 knobs/options,**零 env 直读、零散点 parse**。
//!
//! **叠加式加载**(2026-10-16:配置层不认识 "flavor" —— 预设档只是预先写好的
//! 一份配置文件,由入口(`flavors::preset_path`)解析成路径当"基准文件"传入):
//!
//! ```text
//! Default(代码缺省) → Flavor(内置命名预设) → 用户文件(toml) → CLI 启动参数
//! ```
//!
//! **继承 `extends`**(2026-10-18):任一被加载的文件可在顶层写
//! `extends = "other.toml"`(相对路径 = 相对本文件目录)—— 被继承者先合并、
//! 本文件覆盖之,链可嵌套;环/超深/类型非法 = 响亮拒。子档只写「与 base 不同的
//! 项」(站点档 base.toml → 设备档 → 实验档)。字符串直载(`from_toml_str`)
//! 无文件上下文 ⇒ 含 extends 即拒。
//!
//! 预设档只锁"**必须成套**"的旋钮(避免手拼组合踩约束,如 spec depth
//! 无草稿目录空转、verify 图撞缺省 slab);实验旋钮在用户文件临时改即可,
//! 无需绕预设。**env 彻底退出正式配置链**(缩编裁决:env 只剩 OWL_CONFIG
//! 路径兜底等隐藏键 + 测试域账外自读)。
//!
//! **类型纪律**:枚举严格(ModelKind/SamplerMode —— 反序列化非法值直接
//! 报错;GraphInstantiateFlags 结构化布尔位面),不许 String 和稀泥;
//! **`deny_unknown_fields` 全 module** —— 键名打错字拒启,不静默忽略。
//!
//! **全量配置总账**(键 → module;新键必须在此登记,禁止账外键):
//!
//! | module | 键 | 类型/缺省 |
//! |---|---|---|
//! | (顶层) | extends | String / 无 —— 继承另一配置档(相对 = 相对本文件目录;
//! |   |   | 链可嵌套;环/超深/类型非法 = 拒启。直载 `from_toml_str` 不解析)|
//! | (顶层) | flavor | String / 无 —— 预设档名(等效 `extends` 到 flavors/<名>.toml;
//! |   |   | 测试环境档/站点档用它自带基准) |
//! | runtime | bind | String / "127.0.0.1:8135" |
//! | runtime | device | usize / 0 |
//! | runtime | max_seq | usize / 4096 |
//! | runtime | prefill_chunk | usize / 512 |
//! | runtime | test_device | Option<usize> / None(测试域,不入 toml) |
//! | runtime | server_url | Option<String> / None(cli,不入 toml) |
//! | model | kind | ModelKind 枚举("0.8b"\\|"awq27b")/ 0.8b |
//! | model | dir / name | Option / None(workspace 资产缺省) |
//! | model | awq27b_dir | Option<String> / None(27B AWQ 检查点门控) |
//! | model | dflash2_dir | Option<String>(⚠️ W4A16 量化家族 syvai 系;
//! |   |   | BF16 源(z-lab)喂引擎 = 草稿全噪声 AL=0,装载期嗅探拒启) |
//! | model | quant | QuantScheme 枚举("w4a16"\|"w4a8")/ w4a16(主干量化方案;|
//! |   |   | W4A8 = 权重 a8 布局 + per-token int8 激活 + kS8×kU4 GEMM) |
//! | model | head_quant | HeadQuant 枚举("off"\|"rtn-int4-g32")/ off(C1;|
//! |   |   | 竖式/gate 质量门后转正) |
//! | pool | pool_tokens | Option<usize> / None(=2×单会话) |
//! | pool | gdn_slots / snap_max | usize / 8、4 |
//! | pool | vram_target / vram_reserve_mb | f64、u64 / 0.97、1024 |
//! | pool | prefix_cache | bool / true |
//! | spec | depth / dumb / draft_kv_fp8 | usize 0(off)、flag、flag |
//! | spec | notaps / tapdecl | flag / false(verify 图诊断) |
//! | spec | degrade_after / probe_every | usize / 6、8(降级首探窗口;旧 64 = 长尾税) |
//! | sampling | mode | SamplerMode 枚举("greedy"\\|"sampling")/ sampling |
//! | sampling | temp / topk / topp / rep_penalty | 1.0 / 20 / 0.95 / 1.0 |
//! | load | verify / debug_tap | flag / false |
//! | dispatch | qkv_fuse / gdn_fused_decode / gdn_fused_decode_v2 | bool / 全真(**正语义**) |
//! | dispatch | prefill_split / force_naive | flag / false |
//! | dispatch | ~~flashinfer~~ | **已移除(2026-10-13):FI 恒启用,无开关** |
//! | dispatch | gdn_chunked / gdn_scalar / kv_fp8 | flag / false |
//! | dispatch | resolve_trace / host_argmax / ts_probe | flag / false |
//! | probes | step_profile / perf_level / gdn_dump / debug / prefill_cksum / fact_probe | flag / false |
//! | probes | step_gate / step_gate_ms / step_gate_us / step_gate_every | StepMode("off"\|"sticky"\|"single") / 50 / 0(覆盖 ms) / 1 |
//! | log | level / tags_allow / tags_deny / file / max_mb / keep / timestamp | 见模块 |
//! | probes | trace_gate / dflash_probe / propose_eager | flag / false |
//! | probes | dflash_eager / dflash_noencode / dflash_dumb | flag / false |
//! | probes | gdn_dump_all / pf_bisect / pf_fin_check / raw_completion / prompt_dump / gdn_dbg / pf_fence | flag / false |
//! | probes | pf_stages | Option<usize> / None |
//! | cuda | srv_timing / cap_prof / launch_sync / launch_time | flag / false |
//! | cuda | trace_ring | usize / 0(关;>0 = 发射账容量) |
//! | cuda | d2h_prof / gpu_prof / free_legacy | flag / false |
//! | cuda | graph_flags | { upload, device_launch } / 全 false |
//! | cuda | capture_slab_mb | Option<usize> / None(hint 定量) |
//! | cuda | *(nvrtc_include/native_aot 2026-10-13 退役 —— NVRTC 已删)* | — |
//! | dispatch | fi_min_tokens | usize / 1(FI 接管的 T 下限;128 = 复现旧行为) |
//! | dispatch | xqa_prefill | flag / false(prefill 尾块走 XQA;主路径禁止开) |
//! | graph | no_graph | flag / false(总闸:eager 直发,跳过**全部**图捕获) |
//! | graph | prefill_bucket | off\|t512\|geobuckets / **off**(prefill 图**独立**
//! |   |   | 开关:off = 今天行为;t512 = 只建 512 桶;geobuckets = 全桶族) |
//! | agent | template_path | Option<String> / None(相对路径 = 相对 model.dir) |
//! | agent | implicit_session | flag / false(AGT-09:不带 session 键 ⇒ 会话 0) |
//! | agent | default_max_new / max_new_cap | usize / 4096、8192 |
//! | agent | parse_tool_calls | flag / true(工具三通道解析) |
//! | agent | tool_call_format | ToolCallFormat 枚举("xml"\|"json")/ xml |
//! | agent | auto_disable_thinking_with_tools | flag / false(有工具即关思考) |
//! | agent | reasoning_effort | Option<String> / None(OpenAI 同名协议值,
//! |   |   | 模板值域 none/low/medium/high…;None = 不传让模板取默认) |
//!
//! 账外键(不入公共面):测试私有旋钮(OWL_GATE_*/OWL_E2E_*/OWL_SWEEP_*/
//! OWL_WS_MUL/SPLIT_*/MARLIN_*/FI_DEBUG/OWL_27B_STRICT/OWL_HF_PARITY ——
//! 测试用例自由直读)、构建期键(build.rs 例外)、OWL_CONFIG(loader 路径
//! 兜底,server 入口读)、OWL_TEST_CONF(`#[owl_test]` 的**唯一**测试环境档
//! 入口 —— 每用例 overlay 改走 `testdata/cases/**` 约定路径,见 testkit::case
//! 与 docs/arch/owl-test-macro.md)。

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::file_loader;

// ============================================================================
// §1 枚举与位面(严格类型;非法值反序列化即报错,不许 String 和稀泥)
// ============================================================================

/// 模型档位(model.kind)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
pub enum ModelKind {
    #[serde(rename = "0.8b")]
    #[default]
    Qwen35_08b,
    #[serde(rename = "awq27b")]
    Awq27b,
}

/// 采样模式(sampling.mode;原 `!= "greedy"` 字符串和稀泥废除)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SamplerMode {
    /// 贪心(恒等门/测试口径)
    Greedy,
    /// 采样(缺省 = 成功推理默认姿势)
    #[default]
    Sampling,
}

/// lm_head 量化方案(model.head_quant;2026-10-11 由 bool 升格:
/// 量化方案是一等配置面,新增方案 = 加枚举值 + loader 映射,
/// 禁回生字符串/布尔和稀泥)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HeadQuant {
    /// 关闭(untied lm_head 保持 f16 直读;对照系)
    #[default]
    Off,
    /// 装载期现场 RTN int4 g32 对称(zp≡8,marlin kU4;检查点不动)。
    /// 质量†:HyperQwen GPTQ 口径 +0.6%PPL/GSM8K 不变;本地 RTN g32
    /// 口径 A/B 恒等门已过(prose/数学逐字同,竖式同错=模型固有)
    RtnInt4G32,
    // 预留:GptqInt4G128(需自采 hidden states 产量化,另案)、
    // RtnInt8G128(中间档)——加值时同步 loader 映射与质量门台账
}

/// **主干量化方案**(model.quant;2026-10-17 由 `w4a8: bool` 升格 ——
/// 量化方案是一等配置面,同 [`HeadQuant`] 先例)。
///
/// 命名 = 行业记法 `W{权重精度}A{激活精度}`:**W4A16 = weight-only
/// quantization(WOQ)**,**W4A8 = weight-and-activation quantization**
/// (激活侧也量化)。完整方案 = WxAy × 粒度 × 算法 × 动静态,本键只承载
/// **主干层**这一维(head_quant / 草稿 / mtp 各自独立 plan)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
pub enum QuantScheme {
    /// W4A16 × AWQ g32-asym(权重 4bit + zp,激活 f16)= **今天行为**
    #[serde(rename = "w4a16")]
    #[default]
    W4a16,
    /// W4A8 × AWQ 同源权重(权重 = a8 布局三件套;激活 per-token int8 +
    /// `marlin_gemm_w4a8_awq`(kS8×kU4,int8 张量核))。
    /// 实测(单卡 3090 Ti @32k):prefill +48~50% / TTFT −33% / decode 不变;
    /// 代价 = 装载 +14.5s(a8 重物化,设备重排核待补)+ **质量门 S3 待过**
    /// (非逐字同 —— int8 激活有损)。施工/门/回滚见
    /// `docs/arch/W4A8-算子家族-设计与施工.md`。
    #[serde(rename = "w4a8")]
    W4a8,
    // 预留:W8A8 / W8A16 / W4A4 —— 加值时同步 loader 映射
    // (engine::quant_plan_of)与质量门台账。
}

/// 图实例化旗标(cuda.graph_flags;原裸 u64 魔数 2/4 收口为结构化位面)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraphInstantiateFlags {
    /// UPLOAD=0b10(实例化后预热上传)
    pub upload: bool,
    /// DEVICE_LAUNCH=0b100(设备侧派发;派发税排查面)
    pub device_launch: bool,
}

impl GraphInstantiateFlags {
    /// cuda GraphInstantiateWithFlags 位面(唯一出口;与位定义同尺)
    pub fn to_bits(self) -> u64 {
        ((self.upload as u64) << 1) | ((self.device_launch as u64) << 2)
    }
}

// ============================================================================
// §2 config modules(每 module 管一个功能域;deny_unknown = 键名打错拒启)
// ============================================================================

/// 进程装配面
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeCfg {
    pub bind: String,
    /// **声明式选卡**:`device = { bdf = "42:00.0" }` / `{ find_name = "3090" }` /
    /// `{ uuid = "GPU-…" }`(多填 = 交集,多命中全收);**整行不写 = 自动挑空闲卡**。
    /// 与 `device_ordinal` 互斥(同时给 = 拒启)。
    pub device: crate::device::DeviceSpec,
    /// 显式 ordinal(测试/特殊场景;不走设备表查找)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_ordinal: Option<usize>,
    /// **自动选卡的空闲占比阈值**(0..1;默认 0.9):候选卡的 `memory.free/total` 低于它
    /// ⇒ auto-pick 跳过(避让非 owl 进程/未持锁实例);显式 spec 命中且全被跳 ⇒ 响亮拒启。
    pub gpu_min_free_ratio: f64,
    pub max_seq: usize,
    pub prefill_chunk: usize,
    /// **尾块合并**(tokens;L4,2026-10-18):`剩余 ≤ prefill_chunk + 本值`
    /// ⇒ 本步一口气吃完剩余(块长上限 = chunk + tail_merge)。
    ///
    /// 动机(实测,`perf-runs/prefill-prof-2026-10-18/REPORT.md` §4):每 chunk
    /// 有一份**与尺寸无关**的固定宿主税(~250ms:6395 条命令 alloc/launch/free
    /// 往返;4-token 的尾块也要 249ms)⇒ `<chunk` 的尾块是纯亏。
    /// 默认 **0 = 关 = 逐字节旧行为**(块长 = min(chunk, 剩余))。
    pub prefill_tail_merge: usize,
    /// **测试设备门控值**(不入 toml;测试入口自填):`OWL_TEST_DEVICE` 的 **BDF**
    /// (如 `41:00.0`)。`Some` = GPU 门已过。**CUDA 序号禁用**(序会漂)。
    #[serde(skip)]
    pub test_device: Option<String>,
    /// cli 上行端点(不入 toml;cli 入口自填)
    #[serde(skip)]
    pub server_url: Option<String>,
}

/// 模型资产面
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[derive(Default)]
pub struct ModelCfg {
    pub kind: ModelKind,
    pub dir: Option<PathBuf>,
    pub name: Option<String>,
    /// 27B AWQ 检查点目录(测试/E2E 门控)
    pub awq27b_dir: Option<String>,
    /// DFlash2 草稿检查点目录(⚠️ 须 W4A16 量化家族 syvai 系;BF16 源
    /// (z-lab)喂引擎 = 草稿全噪声 AL=0;装载期家族嗅探拒启)
    pub dflash2_dir: Option<String>,
    /// C1(2026-10-11):untied lm_head 头部量化方案(枚举严格;映射点
    /// 在 engine loader:HeadQuant → QuantPlan)。质量门台账见枚举值注。
    pub head_quant: HeadQuant,
    /// **主干量化方案**(量化方案 = 配置枚举,loader 零硬编码;映射点 =
    /// `engine::quant_plan_of`)。**只作用于主干层** —— lm_head / 草稿 /
    /// mtp 各自独立 plan(选层排除在本仓是结构性的,不需要名字正则)。
    pub quant: QuantScheme,
}

/// 池几何与显存预算(engine StatePool)
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PoolCfg {
    /// 设备 KV 池(token;`None` = 2 × 单会话容量)。
    /// **虚拟化档可小于上下文**(`[kvmem].pool_overcommit`;boot 硬校验兜底)。
    pub pool_tokens: Option<usize>,
    /// 并发上限(每活跃 turn 一套 GDN 行;也是池下限算式的 N_active)
    pub gdn_slots: usize,
    /// GDN 快照池槽数
    pub snap_max: usize,
    /// **宿主 KV 腿**(MiB;0 = 关层)。下沉的落点之一。
    pub host_cache_mb: usize,
    /// **宿主状态腿**(MiB;0 = 关层)。GDN 快照下沉的落点 ——
    /// ⚠️ 超卖档会把压力推给它(实测 27B 单快照 607 MiB),别只给 KV 腿留量。
    pub snap_host_mb: usize,
    pub vram_target: f64,
    pub vram_reserve_mb: u64,
    pub prefix_cache: bool,
}

/// **驱逐策略**(`[kvmem].evict_policy`;TOML 字符串)
///
/// | 值 | 语义 |
/// |---|---|
/// | `lru` | 纯近期(现役基线;`last_use` 升序出序) |
/// | `lrfu` | attention-aware LRFU(缺省 λ = 0.9) |
/// | `lrfu:0.8` | 同上,显式 λ(**0 < λ < 1**) |
///
/// ⚠️ 这条是**配置面单源**:λ 只在这里解析(非法值 = boot 响亮拒),
/// `engine::EngineKnobs::evict_policy_of` 只做词汇映射。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvictPolicyCfg {
    /// 纯近期(按 `last_use` 升序出序)。
    Lru,
    /// attention-aware LRFU:被召回点命中的块 keep-score 抬升。
    Lrfu {
        /// 衰减系数(0,1);`lrfu` 缺省 = 0.9。
        lambda: f32,
    },
}

impl Default for EvictPolicyCfg {
    fn default() -> Self {
        Self::Lru
    }
}

impl EvictPolicyCfg {
    /// **LRFU 缺省 λ**(与 `owl_store::LrfuPolicy::new` 的文档同值)。
    pub const LRFU_DEFAULT_LAMBDA: f32 = 0.9;
}

impl<'de> Deserialize<'de> for EvictPolicyCfg {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        let s = String::deserialize(d)?;
        match s.as_str() {
            "lru" => Ok(Self::Lru),
            "lrfu" => Ok(Self::Lrfu { lambda: Self::LRFU_DEFAULT_LAMBDA }),
            other => {
                let Some(raw) = other.strip_prefix("lrfu:") else {
                    return Err(D::Error::custom(format!(
                        "[kvmem].evict_policy = \"{other}\" 非法:期望 `lru` | `lrfu` | `lrfu:<λ>`(0<λ<1)"
                    )));
                };
                let lambda: f32 = raw.parse().map_err(|_| {
                    D::Error::custom(format!(
                        "[kvmem].evict_policy = \"{other}\" 非法:λ 不是数(期望 0<λ<1 的浮点)"
                    ))
                })?;
                if !(lambda > 0.0 && lambda < 1.0) {
                    return Err(D::Error::custom(format!(
                        "[kvmem].evict_policy = \"{other}\" 非法:λ={lambda} 须 0<λ<1(0=只记最近一次,1=永不衰减)"
                    )));
                }
                Ok(Self::Lrfu { lambda })
            }
        }
    }
}

/// **KVMem 选择层(B 轴)** —— 低显存档位的突破项。
///
/// 适用区间:**设备 KV 池 < 上下文长**(超卖档)。它让"设备上只留工作集、
/// 全量 KV 在宿主"成立 —— 同池尺寸下"全量常驻"引擎必然 OOM。
/// **高显存档不需要它**(池 ≥ ctx 时请关掉:每边界一次召回都是纯开销)。
///
/// 语义(与设计卷 `docs/arch/kvmem-选择层-设计与施工.md` 同源):
/// - 索引 = 逐 (块,层,KV 头) 的 **Mean-K**(按块采,不随召回变);
/// - 召回点 = **prefill 块边界**:末 W=8 行的 query 面 → 打分 → sink/recent 保底 + 按分 top-k;
/// - 位置方案 **A**(原位置稀疏):K 一字不改,只换页表 ⇒ 零 re-RoPE;
/// - 视图只接 **decode**(prefill 用全链);
/// - `recall_every` 是**唯一改渐近线**的旋钮:单次打分是 O(块数) 线性,
///   但"每边界都打" ⇒ 一次 prefill 累计 **O(块数²)**。
///
/// ⚠️ 与 `probes.coverage_probe` 的分工:那个是**诊断面**(真实质量全量扫描,
/// 单次成本含 `ctx` 因子 ⇒ 全军 O(ctx²))。**生产档只需 `select_decode`**,
/// 不要为了开功能而把探针一起打开。
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KvmemCfg {
    /// **生产总开关**:选择层接进 decode(建视图页表 + 每步展开)。
    pub select_decode: bool,
    /// **部分宿主驻留**:被逐块下沉到宿主、链上留占位符。
    /// 需 `select_decode` 同开(链上出洞后必须有懂洞的消费方 = 视图页表);
    /// 开了它会**自动关 XQA 裸 decode 臂**(页表语义不共戴)。
    pub select_partial: bool,
    /// **索引采集**(打分的前置件;`select_decode` 蕴含它,单独开无意义)。
    pub index: bool,
    /// **每 turn 工作集(对外口径;`token` 数)** —— 一个**活跃 turn** 在设备 KV
    /// 池里最多**常驻**多少 KV(含 `select_sink` / `select_recent` 保底)。
    ///
    /// 为什么要这个口径而不是"召回预算块数":它同时是
    /// ①**准入地板**(本 turn 至少要有这么多 KV 直接驻留设备)
    /// ②**召回预算**(视图 = 保底 + 按分 top-k)
    /// ③**池下限算式**里每 turn 的那一项
    /// —— 三处必须是同一个数,所以配置只给**一个** token 数,
    /// 内部三件(块数/字节/每 turn 页数)在**装配期**由页几何派生
    /// (唯一产地 = `engine::derive_turn_ws`;boot 打 `[turn-ws]` 自证行)。
    ///
    /// 缺省 = 2240 tok(= 70 页 × 32 tok = 保底 6 + 召回 64 块,与旧 `select_blocks=64` 等值)。
    /// 页宽契约 = 32 token(设计卷 §1 铁律②)。
    pub workspace_per_req: usize,
    /// 保底 sink(时间序最前的若干块;论文 always-retained)。
    pub select_sink: usize,
    /// 保底 recent(时间序最后的若干块)。
    pub select_recent: usize,
    /// **召回节奏**:每 k 个块边界做一次**完整召回**(打分+按分选择),
    /// 其余边界**只采索引、不干预驻留**(驻留交给 A 轴,下一次完整召回时对齐)。
    /// **末个边界永远是完整召回**(decode 消费的视图取自它)。
    pub recall_every: usize,
    /// **覆盖率探针(诊断面;缺省关)**。开则在末个边界额外做**真重诊断**:
    /// K/V 全量回读 + host 逐元素解码 + `true_block_mass`(逐 query 行 ×(层,头)
    /// 扫**全 ctx**)+ 论文口径对照 + 方案 A/B 对照 + 覆盖率/上界报告。
    /// 单边界成本**含 ctx 因子** ⇒ 全军 O(ctx²)(27B/8k 实测单次 6.3s = 墙钟 32%)。
    /// 🔴 **测性能不要开它** —— 生产面只需 `select_decode`。
    pub coverage_probe: bool,
    /// **分析边界目标 ctx**(诊断面;0/未设 = **最后一个边界**)。
    ///
    /// 为什么要有它:边界只在"chunk 末且页对齐"处发生 ⇒ 同一个 prompt 下,
    /// "最后一个边界"的位置**随 chunk 漂移** ⇒ 覆盖率读数跨 chunk **不可比**
    /// (实测 chunk 1024/512/256 ⇒ 分析@3072/3584/3840 ⇒ top8 87.5%/12.4%/6.8%)。
    /// 钉住 ctx 后:query 面 `[ctx−8, ctx)` 与块数 `ctx/page` **都与 chunk 无关**
    /// ⇒ 读数逐位可比。要求该 ctx 是 chunk 的整数倍(否则取不到该边界)。
    pub analyze_at_ctx: usize,
    /// **驱逐策略**(缺省 `lru`):`lru` = 纯近期;`lrfu[:λ]` = attention-aware
    /// LRFU(召回点选中的块抬 keep-score)⇒ 出序时"又冷又没被注意力用过"的先下沉。
    /// λ 参数在此解析(A4 收口,2026-10-19)。
    pub evict_policy: EvictPolicyCfg,
    /// **准入地板**(MiB;`None` = 跟随虚拟化姿态)。
    ///
    /// 语义 = "本 turn 至少要有多少 **KV 页**直接驻留在设备"
    /// (`owl_store::TurnFloor::Pages`)。MiB 在装配期按**池页宽**换成页数
    /// (向上取整;与池下限算式同一把尺)。它决定**准入判据**:
    /// - `None` + 虚拟化开 ⇒ **0** = "只要工作集装得下就准入"(低显存档默认);
    /// - `None` + 虚拟化关 ⇒ **Strict** = 要全量 dense 放得下(旧行为);
    /// - `Some(n)` ⇒ `n` MiB(仅当虚拟化开才被接受,否则 boot 响亮拒)。
    #[serde(default)]
    pub dense_floor_mib: Option<u64>,
    /// **允许设备池小于上下文**(锁①;缺省关)。
    ///
    /// 关着时池有硬下限 `nb ≥ ceil(seq_tokens/page)`(`state.rs`)—— 即
    /// **选择层只省访问面、不省显存**。开着才允许 `pool_tokens` 真的压小。
    /// 只有虚拟化开才被接受。
    #[serde(default)]
    pub pool_overcommit: bool,
}

impl KvmemCfg {
    /// **虚拟化是否真的开着**:两件缺一不可。
    ///
    /// - `select_decode` = 有懂洞的消费方(视图页表接进 decode);
    /// - `select_partial` = 链上会真的出洞(被逐块下沉)。
    ///
    /// 两件必须同真才能"让链超出池容量"——否则链要么出洞无人懂(静默错页),
    /// 要么永远不出洞(池装不下就 `Exhausted`)。这是**所有解锁类护栏的唯一判据**。
    pub fn virtualization_enabled(&self) -> bool {
        self.select_decode && self.select_partial
    }
}

/// 投机解码(spec 三态/草稿池量化/B4 降级参数/verify 图诊断)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpecCfg {
    pub depth: usize,
    pub dumb: bool,
    pub draft_kv_fp8: bool,
    /// verify 图 taps 退回 last_hidden(排查开关)
    pub notaps: bool,
    /// TAPDECL 二分臂(声明期 tapped 不挂输出槽)
    pub tapdecl: bool,
    /// **m=0 连击阈值**(达此连击 → 降级裸 decode;0 = 禁用降级)
    pub degrade_after: usize,
    /// **降级后的首次探测窗口(tokens)** —— 探测 = 跑一个投机轮试水,命中(m≥1)即恢复 spec。
    ///
    /// ⚠️ **2026-10-18 定谳:必须小**(缺省 8,旧缺省 64 是**长尾税**):
    /// 降级后窗口内每 token 走裸 decode(**实测 27.5 ms/token**)而 spec 档
    /// 6.4 token/轮 ≈ **5.7 ms/token** ⇒ 卡在窗口里一个 token 就是 22 ms 的损失。
    /// 窗口 64 ⇒ 一次**偶发** m=0 连击最长罚 **1.4 s**;窗口 8 ⇒ 罚 0.18 s。
    /// 反向成本(探测失败的代价)= 一轮投机轮(36.7ms)换掉一个裸步(27.5ms)
    /// ≈ **+9 ms/次**,且探测失败后**退避 ×2**(8→16→…→512 封顶)⇒ 持久失败
    /// (真复读上下文)总浪费 ~0.1 s 量级。**两头都便宜,故取小**。
    pub probe_every: usize,
}

/// 采样参数
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingCfg {    pub mode: SamplerMode,
    pub temp: f32,
    pub topk: usize,
    pub topp: f32,
    pub rep_penalty: f32,
}

/// 装载域(校验/观测)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoadCfg {
    pub verify: bool,
    pub debug_tap: bool,
}

/// 解释器分派面(models EnvProvider 组装源;**正语义** —— true = 开,
/// env 面的 NO_FUSE 反开关是兼容遗留,不入 toml)
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DispatchCfg {
    /// W2 qkv 融合(缺省真;2026-10-04 k-probe 硬门定谳)
    pub qkv_fuse: bool,
    pub prefill_split: bool,
    pub force_naive_prefill: bool,
    /// verify(T=depth+1)走的注意力臂:true = v2 伪序列臂(现状缺省);
    /// false = 回落 prefill 族(--split / --chunked_opt)。
    /// 2026-10-13 立(长 ctx 衰减 D3 立案):v2 伪序列臂在 ctx 8192 实测
    /// 4.63ms/层(fp8),而 prefill 族是 qo_len=8 单遍形态 —— 本开关是
    /// 两族 A/B 的唯一入口,定谳后固化其一,不留双路径。
    pub verify_v2: bool,
    /// **decode 走"单图烘 plan"**(2026-10-18 图档单图 W3):
    /// `Some(ctx)` = ① decode **不进 piecewise**(FI 核留捕获图内);
    /// ② 喂 FI 的 `ctx_total` 用该上界烘死 ⇒ 图自携该上界的调度表。
    /// **上界必须 ≥ 任何可能出现的 ctx**(校验拒启兜底)。
    /// **`0` = 哨兵「取 [`RuntimeCfg::max_seq`]」** —— 单序列 ctx 的天然上界,
    /// 且它随档位走(不同 max_seq 的档不必手改上界)。
    /// `None`(缺省)= 现状(piecewise 断点 + 每步重算精确表)。
    #[serde(default)]
    pub fi_plan_bake: Option<usize>,
    /// **图自携 plan restore**(缺省 `true` = 捕获窗内把本 plan 的表烘进图的
    /// memcpy 节点;prefill 桶图靠它)。
    /// `false` = 图不烘表 ⇒ **host 为表的唯一权威**:引擎每次图 launch 前
    /// 用 `fi_plan_sync` 刷当步**精确**表(单图档 `fi_plan_bake` 的标配)。
    /// ⚠️ 与 `fi_plan_bake` 必须配套(false),冲突配置由 `validate` 拒启。
    #[serde(default = "default_true")]
    pub fi_plan_restore: bool,
    /// **FI plan 走图档瓦**(上游 `enable_cuda_graph=true`):`cta_tile_q` 取最坏情形、
    /// `padded_batch_size = max(2·SM/hkv, tiles)` ⇒ **plan 形状对 ctx 全域恒定**
    /// (2026-10-18 实测:ctx 32→262144 一条直线 `padded=42/cta=16|64/split=1/int_bytes=618|650`)
    /// ⇒ 这是"FI 进捕获图"的结构前提(形状恒定 ⇒ 一张图 + 换表内容即可)。
    /// 关 = eager 瓦(历史行为;形状随 ctx 跳变 16 段)。
    /// 投递:C++ 侧读 `getenv("FI_CUDA_GRAPH")`,故由引擎 boot 期经
    /// `owl_kernels::family::flashinfer::configure_plan_env` **唯一**写入。
    #[serde(default)]
    pub fi_cuda_graph: bool,
    /// **verify 走 FI 断点臂**(V1b;需 `verify_piecewise`)——FI+split 换掉 XQA
    pub verify_fi: bool,
    /// **decode(T=1)走 FI 断点臂**(V1c;session 常驻图改分段,断点挂 decode 核)
    pub decode_fi: bool,
    /// verify 臂改走 **XQA**(kNHD fp8 池 + spec-dec q=1+K;feature `xqa`)。
    /// 2026-10-13 D3:XQA 单算子实测 ctx8192 0.495ms vs v2 2.673ms(5.40×),
    /// 数值 max|XQA−host| 4.5e-4(v2 4.6e-4)。缺省 false(定谳后再翻)。
    pub verify_xqa: bool,
    /// 裸 decode(T=1)改走 **XQA**(第二份实例化 `M_TILESIZE=16`;feature `xqa`)。
    /// 2026-10-13 ② 立案:kernels crate 单算子实测(kNHD fp8/q=1/3090 Ti)
    /// ctx 512/2048/8192 = 1.90× / 2.25× / **1.46×** vs v2 主核+reduce
    /// (`crates/kernels/tests/xqa_decode_probe.rs`)。
    /// 与 `verify_xqa` **独立**(走不同图:decode 常驻捕获图 vs verify 桶形图)。
    pub decode_xqa: bool,
    /// prefill(chunked q=T)注意力改走 **XQA**(与 verify **同一份
    /// `M_TILESIZE=32` 实例化**;feature `xqa`;自 `feat/prefill-graph` 合入)。
    ///
    /// ⚠️ **只作 FI 缺口交接臂,不许放到 FI 之前**(CUPTI 逐核同请求 A/B:
    /// XQA `kernel_mha` 258.5ms vs FI `BatchPrefill` 63.0ms = **慢 4.1×**;
    /// XQA 是 decode 形状核,q=512 只有 ~9 TFLOPS)。价值面 = `t < fi_min_tokens`
    /// 的尾块(现役回落核 24 CTA)⇒ 探针 T=128 同形状 **3.9×**
    /// (`crates/kernels/tests/xqa_prefill_probe.rs`),长 prompt 净省 ~75ms
    /// (期望值 ≈ 0.25×75ms)。缺省 **false** = 生产零行为变化。
    pub xqa_prefill: bool,
    /// **FI 接管的 T 下限**(行)。**缺省 1 = FI 通吃**(2026-10-14 定谳,自
    /// `feat/prefill-graph` 合入)。
    ///
    /// 历史缺省 128(2026-10-12 立),依据是"FI 对小 qo 低效(18→36ms/tok)"。
    /// ⚠️ 2026-10-14 单测推翻该依据(`crates/kernels/tests/fi_prefill_probe.rs`,
    /// 生产核的第一个单测):FI 在 T=1/8/54/128 × ctx=3072 实测
    /// **0.287/0.435/0.437/0.438 ms/层**(数值 max|FI−host| 3.2e-4,超门行数 0),
    /// 而旧门槛以下的回落核(`chunked knhd`,24 CTA 低并行)同形状 **6.7 ms/层**
    /// (in-engine CUPTI)⇒ **FI 快 15×**,且差距随 ctx 线性放大(T=128:ctx512
    /// 1.38 → ctx8192 20.05 ms/层)。
    ///
    /// 引擎 A/B(3 轮交替 boot,中位,同 prompt 只差本键;尾块 47/36 行):
    /// **−73.8ms @3.1k ctx(−3.0%)· −272.2ms @8.2k ctx(−4.1%)**;
    /// 短 prompt(58 tok)与满 chunk(567 tok)各 −2.8/−1.9ms(无回归)。
    /// 本键留作诊断旋钮(128 = 复现旧行为)。
    pub fi_min_tokens: usize,
    /// FlashInfer prefill 面
    pub gdn_chunked: bool,
    pub gdn_scalar: bool,
    /// GDN decode 融合 D1(缺省真)
    pub gdn_fused_decode: bool,
    /// D1-v2(缺省真;2026-10-04 三重验证后翻)
    pub gdn_fused_decode_v2: bool,
    /// KV 池 fp8(→ KvQuant::Fp8E4M3)
    pub kv_fp8: bool,
    pub resolve_trace: bool,
    pub host_argmax: bool,
    pub ts_probe: bool,
    /// **生产算子白名单**(设计 §八 P4「白名单硬门」;`None`/空 = 门关)。
    ///
    /// 非空 ⇒ 只允许这些算子发射:未列出的算子**发射即结构化拒**
    /// (deny-by-default)。语义 = "本档位需要的算子集合" —— 生产档显式列全,
    /// dev/诊断档留空(门关)。
    ///
    /// 为什么是硬门:未建表(仅引入)的算子不该在不知道它契约的情况下进生产;
    /// 而"哪些算子在跑"此前**无人知道**(实测靠 `launch_time` 日志数)。
    /// 装配期还会用本表跑**池写集穷举审计**(`ops::audit_active`)。
    pub allowed_ops: Option<Vec<String>>,
}

/// 性能框架分级(映射到 `owl_shared::metrics::level`;见
/// docs/arch/perf-framework.md §一)。`off` = 零成本基线对照臂。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PerfLevel {
    Off,
    Counters,
    #[default]
    Host,
    Gpu,
}

impl PerfLevel {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Off => 0,
            Self::Counters => 1,
            Self::Host => 2,
            Self::Gpu => 3,
        }
    }
}

/// 日志级别(映射到 `owl_shared::log::Level`;`off` = 静音)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Off,
    Error,
    #[default]
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Off => crate::log::LVL_OFF,
            Self::Error => crate::log::LVL_ERROR,
            Self::Warn => crate::log::LVL_WARN,
            Self::Info => crate::log::LVL_INFO,
            Self::Debug => crate::log::LVL_DEBUG,
            Self::Trace => crate::log::LVL_TRACE,
        }
    }
}

/// 日志面(logger 框架;模块头见 `crates/shared/src/log.rs`)
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LogCfg {
    /// 级别阈值(off/error/warn/info/debug/trace;**缺省 warn**)
    pub level: LogLevel,
    /// tag 前缀白名单(空 = 全放行)
    pub tags_allow: Vec<String>,
    /// tag 前缀黑名单(优先于白名单)
    pub tags_deny: Vec<String>,
    /// tee 落盘路径(None = 仅 stderr)
    pub file: Option<std::path::PathBuf>,
    /// 单文件上限 MiB(0 = 不 rotation)
    pub max_mb: u64,
    /// rotation 保留份数
    pub keep: usize,
    /// 前置 `2026-10-14T00:12:33.123Z ` 时间戳
    pub timestamp: bool,
}

/// serde 缺省 = true(布尔配置项的显式缺省;避免 `#[serde(default)]` 落到 false)
fn default_true() -> bool {
    true
}

/// 诊断探针族(全域布尔;boot 一次解析,热路径零 env)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProbesCfg {
    pub step_profile: bool,
    /// 性能分级(off/counters/host/gpu;缺省 host = 历史行为)
    pub perf_level: PerfLevel,
    pub gdn_dump: bool,
    pub debug: bool,
    pub prefill_cksum: bool,
    pub fact_probe: bool,
    pub trace_gate: bool,
    pub dflash_probe: bool,
    pub propose_eager: bool,
    pub dflash_eager: bool,
    pub dflash_noencode: bool,
    pub dflash_dumb: bool,
    pub gdn_dump_all: bool,
    pub pf_bisect: bool,
    pub pf_stages: Option<usize>,
    pub pf_fin_check: bool,
    pub raw_completion: bool,
    /// 渲染后 prompt 落盘(/tmp/owl-prompt-*.txt;诊断面 —— 原
    /// OWL_PROMPT_DUMP env,2026-10-15 随 env 退出配置链收编)
    #[serde(default)]
    pub prompt_dump: bool,
    /// reset_gdn 后 GDN rec 格画像(§三十二 会话污染定位诊断面;
    /// 原 OWL_GDN_DBG)
    #[serde(default)]
    pub gdn_dbg: bool,
    /// prefill H2D 叶子后插流栅栏(A13 竞态排障;
    /// 原 OWL_PF_FENCE)
    #[serde(default)]
    pub pf_fence: bool,
    /// **单步/粘滞调试门**(2026-10-14;见 `crate::stepgate` 模块头):
    /// 解释器检查点的运行期档位。⚠️ 需以 `--features stepgate` 构建才生效;
    /// 未编入时入口会**可见地告警**(不静默失效)。
    pub step_gate: crate::stepgate::StepMode,
    /// **KV 页字节差分落点**(`<pos>` = 在该 decode 步前后各 d2h 一页,一次;
    /// 写者足迹取证面)—— 原 `OWL_KV_DUMP` env(同上收编)
    #[serde(default)]
    pub kv_dump_at: Option<usize>,
    /// 粘滞档每检查点睡眠毫秒(缺省 50)
    #[serde(default = "default_step_gate_ms")]
    pub step_gate_ms: u64,
    /// **亚毫秒档**(µs;>0 时覆盖 `step_gate_ms`,粘滞等待走自旋)
    #[serde(default)]
    pub step_gate_us: u64,
    /// 每 N 个检查点停一次(缺省 1 = 每点都停)
    #[serde(default = "default_step_gate_every")]
    pub step_gate_every: u64,
}

fn default_step_gate_ms() -> u64 {
    50
}

fn default_step_gate_every() -> u64 {
    1
}

/// CUDA 设备面(诊断旗标 + 图实例化位面 + slab 定档 + nvrtc 工具链)
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaCfg {
    pub srv_timing: bool,
    pub cap_prof: bool,
    pub launch_sync: bool,
    /// **同步点分解掩码**(诊断 A/B,2026-10-15):`launch_sync` 是「全部同步点」
    /// 的合称,不是单一变量 —— 要做归因必须逐点开关。位定义见 [`sync_bit`];
    /// 0(缺省)= 生产语义。`launch_sync = true` 时本字段被覆盖为 [`sync_bit::ALL`]。
    #[serde(default)]
    pub sync_mask: u32,
    pub launch_time: bool,
    pub d2h_prof: bool,
    pub gpu_prof: bool,
    pub free_legacy: bool,
    pub graph_flags: GraphInstantiateFlags,
    /// 捕获 slab 固定档 MiB(None = warmup 计量定量)
    pub capture_slab_mb: Option<usize>,
    /// **发射账容量**(0 = 关;>0 ⇒ server 记最近 N 条命令,含流/核名/块地址)。
    /// 查询:`GET /debug/launches[/<n>[/<kind>]]`;见 cuda::trace 模块头。
    #[serde(default)]
    pub trace_ring: usize,
    // nvrtc_include(2026-10-13 退役):NVRTC 编译路径已删 —— 设备码一律
    // 构建期 cubin 资产(源哈希查表),include 路径是构建期 `nvcc` 的事。
    // 键已从配置面移除(不许留死键,`deny_unknown_fields` 会让旧档拒启 →
    // 显式暴露"该档还在用已退役旋钮")。
}

/// 同步点分解掩码的位定义(`cuda.sync_mask`;诊断 A/B 用)。
///
/// 6 处同步点分两类:**跨流 IO 序**(HTOD/DTOH/FENCE —— 拷贝流与 COMPUTE 之间的
/// 内存可见性)与 **host 领先**(LAUNCH/GRAPH/ALLOC —— host 是否等设备追上)。
/// `launch_sync = true` 曾把两类一起钉死 ⇒ "能治好" 但归因不明;分开才能指认
/// 是哪一对跨流/哪一处领先在竞速。
pub mod sync_bit {
    /// `try_htod`:H2D DMA 后 `sync` H2D 流(host 等上传真落定)
    pub const HTOD: u32 = 1 << 0;
    /// `dtoh`:退回整流 `COMPUTE.synchronize()`(旧语义;否则 = 事件定向等待)
    pub const DTOH: u32 = 1 << 1;
    /// 块回收栅栏:任一 Owned 块 ⇒ H2D + D2H 都等(旧语义;否则 = 按块定向)
    pub const FENCE: u32 = 1 << 2;
    /// 逐 eager launch 后 `COMPUTE.synchronize()`
    pub const LAUNCH: u32 = 1 << 3;
    /// 图回放后 `COMPUTE.synchronize()`
    pub const GRAPH: u32 = 1 << 4;
    /// alloc/memset 后 `COMPUTE.synchronize()`
    pub const ALLOC: u32 = 1 << 5;
    /// **家族 runtime scratch 上传·实验位**(A13,2026-10-15):拷贝前先排空
    /// COMPUTE(让流序分配落定)。定谳时用过;现生产缺省已修为同流 async。
    pub const SCRATCH_SYNC_UP: u32 = 1 << 6;
    /// **家族 runtime scratch 上传·回退位**:**复现 A13** —— `DeviceRes::upload`
    /// 非捕获分支退回 "NULL 流同步拷贝"(2026-10-09 试刀行为)。
    /// ⚠️ 置位 = 故意把写序抹掉(写进尚未落定的异步分配 ⇒ 静默丢写),
    /// 仅供 A/B 与仪器验证;生产不得置位。
    pub const SCRATCH_UP_LEGACY: u32 = 1 << 7;
    /// `launch_sync` 的全部六点(仍不含 scratch 上传位 —— 它们在语义上不是"多一道同步")
    pub const ALL: u32 = HTOD | DTOH | FENCE | LAUNCH | GRAPH | ALLOC;
}

/// 图装配面(engine GraphPlanDesc)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraphCfg {
    /// eager 直发,跳过图捕获(**总闸**:连 decode/verify/fold/propose 族一起关)
    pub no_graph: bool,
    /// **prefill 图的独立档位**(P1;2026-10-15 立 —— 与总闸分离,可单关
    /// prefill 而不动现役 decode 图族)。
    ///
    /// 分层而非 bool 的理由:回滚时要能**逐级降**,不必在"全开/全关"之间跳
    /// - `off`(缺省):完全不建 prefill 图 = 今天行为,零影响;
    /// - `t512`:只建 512 桶(最小面:显存最小、命中面 = 满块);
    /// - `geobuckets`:全桶族 {32,64,128,256,512}(尾块也吃图,显存 ≈ 374 MiB)。
    ///
    /// 运行时另有**熔断**:某桶捕获失败/回放出错 ⇒ 该桶永久降级 eager
    /// (只 warn 一次,不重试),与静态档位叠加。
    pub prefill_bucket: PrefillBucket,
}

/// prefill 图档(见 [`GraphCfg::prefill_bucket`];serde 小写:off/t512/geobuckets)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrefillBucket {
    /// 不建 prefill 图(= 今天行为)
    #[default]
    Off,
    /// 只建 t=512 桶(最小面)
    T512,
    /// 全桶族 {32,64,128,256,512}
    Geobuckets,
}

/// 工具调用输出格式(agent.tool_call_format;**jinja 模板同名变量**)。
/// xml = `<function=…><parameter=…>` 块;json = `<tool_call>{"name":…}` 单行。
/// 解析器两格式都认(`toolcall.rs`),**切换时必须与模板同批**。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallFormat {
    /// 模板缺省(厂商推荐)
    #[default]
    Xml,
    /// 单行 JSON(部分实现/工具链更易解析)
    Json,
}

/// agent 服务面(2026-10-14 T5;agent-服务面-设计与施工.md §三/§九)。
/// 这些旋钮此前散落为 `openai.rs` 硬编码常量与 adapter 侧约定 ——
/// 迁入配置总账后由 `ServerConfig` 显式搬运,账外键拒启。
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentCfg {
    /// 模板路径覆盖(B2):None = 模型快照 `<model.dir>/chat_template.jinja`;
    /// 相对路径 = 相对 model.dir。**不动快照**换模板的运维抓手
    /// (模板漂移调查 / 模板 A/B 的载体)
    pub template_path: Option<String>,
    /// 隐式默认会话(AGT-09):请求不带 `session` 键时落到会话 0。
    /// pi 每轮全量重发历史 ⇒ 前缀增量命中;false = 每请求临时会话
    /// (旧行为,每次全量重算 + GDN 格轮转)
    pub implicit_session: bool,
    /// 未带 `max_tokens` 时的生成步数
    pub default_max_new: usize,
    /// `max_tokens` 夹紧上限(须 ≤ 池容量 − prompt 量级;随池定档放开)
    pub max_new_cap: usize,
    /// 工具三通道解析(关 = 工具文本原样进 `content` —— 诊断臂)
    pub parse_tool_calls: bool,
    /// 工具格式(**必须与模板同值**:模板按它选分支,解析器按它选状态机)
    pub tool_call_format: ToolCallFormat,
    /// 有工具即关思考(模板同名变量;产品选择待真机 A/B)
    pub auto_disable_thinking_with_tools: bool,
    /// 思考档位缺省(OpenAI 同名协议值:模板值域 none/low/medium/high/xhigh…;
    /// 请求带 `reasoning_effort` 时以请求为准)。None = 不传 ⇒ 模板取自己的缺省
    pub reasoning_effort: Option<String>,
}

impl Default for AgentCfg {
    fn default() -> Self {
        Self {
            template_path: None,
            implicit_session: false,
            default_max_new: 4096,
            max_new_cap: 8192,
            parse_tool_calls: true,
            tool_call_format: ToolCallFormat::Xml,
            auto_disable_thinking_with_tools: false,
            reasoning_effort: None,
        }
    }
}

/// 本地服务面(并发/排队;owl 定位 = 本地单机,不做对外高并发)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerCfg {
    /// 最大并发请求数(在跑 + 排队;超出者排队等待)。
    /// 默认 8;须 ≤ pool 的 GDN 行数(knobs.gdn_slots,每并发会话
    /// 占一格)—— 启动校验不足即拒。本地多副本 agent 各占一并发。
    pub max_concurrency: usize,
}

impl Default for ServerCfg {
    fn default() -> Self {
        Self { max_concurrency: 8 }
    }
}

// ============================================================================
// §3 总对象
// ============================================================================

/// 全量配置对象(loader 唯一产物;进程内冻结,只读下传)
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OwlConfig {
    pub runtime: RuntimeCfg,
    pub model: ModelCfg,
    pub pool: PoolCfg,
    pub kvmem: KvmemCfg,
    pub spec: SpecCfg,
    pub sampling: SamplingCfg,
    pub load: LoadCfg,
    pub dispatch: DispatchCfg,
    pub probes: ProbesCfg,
    pub cuda: CudaCfg,
    pub graph: GraphCfg,
    pub server: ServerCfg,
    pub agent: AgentCfg,
    pub log: LogCfg,
}

impl Default for RuntimeCfg {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8135".into(),
            device: crate::device::DeviceSpec::default(),
            device_ordinal: None,
            gpu_min_free_ratio: 0.9,
            max_seq: 4096,
            prefill_chunk: 512,
            prefill_tail_merge: 0,
            test_device: None,
            server_url: None,
        }
    }
}


impl Default for PoolCfg {
    fn default() -> Self {
        Self {
            pool_tokens: None,
            gdn_slots: 8,
            snap_max: 4,
            host_cache_mb: 0,
            snap_host_mb: 0,
            vram_target: 0.97,
            vram_reserve_mb: 1024,
            prefix_cache: true,
        }
    }
}

impl Default for KvmemCfg {
    /// 缺省全关 = "高显存档直接关掉它"(功能只在池 < ctx 的档位有意义)。
    /// `recall_every = 1` = 每个边界都完整召回(旧行为;逐位可回退)。
    fn default() -> Self {
        Self {
            select_decode: false,
            select_partial: false,
            index: false,
            // 2240 tok = 70 页 × 32 tok = 保底 6 + 召回 64 块(与旧 `select_blocks=64` 等值)
            workspace_per_req: 2240,
            select_sink: 2,
            select_recent: 4,
            recall_every: 1,
            coverage_probe: false,
            analyze_at_ctx: 0,
            evict_policy: EvictPolicyCfg::Lru,
            dense_floor_mib: None,
            pool_overcommit: false,
        }
    }
}

impl Default for SpecCfg {
    fn default() -> Self {
        Self {
            depth: 0,
            dumb: false,
            draft_kv_fp8: false,
            notaps: false,
            tapdecl: false,
            degrade_after: 6,
            probe_every: 8,
        }
    }
}

impl Default for DispatchCfg {
    /// 分派缺省:三大融合生产默认开(2026-10-04 k-probe/金标/步时定谳)
    fn default() -> Self {
        Self {
            qkv_fuse: true,
            prefill_split: false,
            force_naive_prefill: false,
            verify_v2: true,
            verify_xqa: false,
            // piecewise 段驱动件:**缺省关**(2026-10-15 立项;A/B 先证机制
            // 零回归,定谱后再翻默认 —— 同 verify_xqa 的翻法)
            fi_plan_bake: None,
            fi_plan_restore: true,
            // FI plan 图档瓦(2026-10-18 图档单图立案):**缺省关** —— 图档单图的
            // 其余件(W2 关自携 / W3 单图)未落地前开它只是换瓦片口径,不产生图。
            fi_cuda_graph: false,
            verify_fi: false,
            decode_fi: false,
            decode_xqa: false,
            xqa_prefill: false,
            fi_min_tokens: 1,
            gdn_chunked: false,
            gdn_scalar: false,
            gdn_fused_decode: true,
            gdn_fused_decode_v2: true,
            kv_fp8: false,
            resolve_trace: false,
            host_argmax: false,
            ts_probe: false,
            allowed_ops: None,
        }
    }
}

impl Default for SamplingCfg {
    fn default() -> Self {
        Self {
            mode: SamplerMode::default(),
            temp: 1.0,
            topk: 20,
            topp: 0.95,
            rep_penalty: 1.0,
        }
    }
}


// ============================================================================
// §5 统一 loader(Default → 基准文件 → 覆盖文件;两文件均可 `extends` 继承链;
// deny_unknown 在最终反序列化生效)
// ============================================================================

/// TOML Value 深合并(覆盖层写回基座;表递归,标量/数组整体覆盖)
fn merge(base: &mut toml::Value, over: toml::Value) {
    match (base, over) {
        (toml::Value::Table(b), toml::Value::Table(o)) => {
            for (k, v) in o {
                match b.get_mut(&k) {
                    Some(bv) if bv.is_table() && v.is_table() => merge(bv, v),
                    _ => {
                        b.insert(k, v);
                    }
                }
            }
        }
        (b, o) => *b = o,
    }
}

/// **退役键迁移门**(2026-10-17):键**改名**不是“静默忽略”,而是响亮拒绝 +
/// 当面指明新键。`deny_unknown_fields` 本就会拒(unknown field),这里追加的
/// 是**可操作**的那一半(点名新键 + 值域)—— 使用者不必去翻旧分支。
///
/// 新增退役键 = 在表里加一行(`旧写法 → 新写法`)。
fn reject_retired_keys(v: &toml::Value) -> Result<(), String> {
    /// (module, 退役键, 迁移指引)
    const RETIRED: &[(&str, &str, &str)] = &[
        (
            "model",
            "w4a8",
            "W4A8 档位已从 bool 升格为量化方案枚举 —— 改用 `[model] quant = \"w4a8\"`(值域:\"w4a16\"|\"w4a8\";缺省 \"w4a16\")",
        ),
        (
            "dispatch",
            "verify_piecewise",
            "piecewise(分段捕获 + 断点)已整体退役(2026-10-18 W5):FI 核恒留捕获图内、表由 `fi_plan_sync` 窗外刷 —— 删掉该键",
        ),
        (
            "dispatch",
            "break_at_out",
            "断点面随 piecewise 退役(W5):该诊断键无数据源 —— 删掉该键",
        ),
        (
            "dispatch",
            "fi_plan_bake_verify",
            "verify 恒随 `fi_plan_bake` 走单图(W4 转正 + W5) —— 删掉该键",
        ),
        (
            "probes",
            "fi_bucket",
            "FI plan 桶化随宿主标量槽退役(W5;表恒按当轮精确 ctx 刷) —— 删掉该键",
        ),
        (
            "probes",
            "pw_dump_tok",
            "piecewise 断点面已退役(W5):该 dump 无数据源 —— 删掉该键",
        ),
    ];
    for (module, key, hint) in RETIRED {
        if v.get(*module).and_then(|m| m.get(*key)).is_some() {
            return Err(format!("[{module}] {key} 退役键:{hint}"));
        }
    }
    Ok(())
}

/// **配置继承**(`extends`;顶层键;2026-10-18)。
///
/// 语义:`extends = "other.toml"` ⇒ 先合并 other.toml(相对路径 = 相对本文件
/// 目录),再合并本文件(本文件覆盖被继承者) ⇒ 子档只写「与 base 不同的项」。
/// CLI 的 `--flavor` 仍是整条链最底层的基准文件。链可嵌套;环/超深 = 响亮拒。
const EXTENDS_KEY: &str = "extends";
/// **预设档引用键**(`flavor = "spec-dflash"`;顶层;2026-10-18)。
///
/// 语义与 `extends` 同型,但父档由**名字**解析(`crate::flavors::preset_path`,
/// 即 `--flavor` 的同一解析面):父链顺序 = `flavor` 预设 → `extends` 父档 →
/// 自身。用途:测试环境档/站点档自带基准(“以 `spec-dflash` 为底,只改几项”),
/// 不必让调用方同时记得传 `--flavor`。
const FLAVOR_KEY: &str = "flavor";
/// extends 链深上限(含起点;防病态链/手滑)
const EXTENDS_MAX_DEPTH: usize = 8;

/// TOML 值的类型名(报错用;显式 match —— 不依赖 toml crate 的辅助 API 漂移)
fn toml_type_name(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::String(_) => "字符串",
        toml::Value::Integer(_) => "整数",
        toml::Value::Float(_) => "浮点",
        toml::Value::Boolean(_) => "布尔",
        toml::Value::Datetime(_) => "日期时间",
        toml::Value::Array(_) => "数组",
        toml::Value::Table(_) => "表",
    }
}

/// 读一个配置档并**递归展开 `extends`**,把「被继承者 → 自身」依序合并进 `out`。
///
/// `depth` = 当前链深;`chain` = 已展开的**规范化**路径(环检测 + 报错直接打印链)。
/// 只做文件面,不碰配置语义(反序列化/validate 在 `load` 末尾统一做)。
fn load_config_into(
    path: &Path,
    out: &mut toml::Value,
    depth: usize,
    chain: &mut Vec<PathBuf>,
) -> Result<(), String> {
    if depth >= EXTENDS_MAX_DEPTH {
        let mut shown: Vec<String> = chain.iter().map(|p| p.display().to_string()).collect();
        shown.push(path.display().to_string());
        return Err(format!(
            "extends 链深超上限 {EXTENDS_MAX_DEPTH}:{}",
            shown.join(" → ")
        ));
    }
    let canon = path
        .canonicalize()
        .map_err(|e| format!("配置 {} 读取失败: {e}", path.display()))?;
    if let Some(pos) = chain.iter().position(|p| p == &canon) {
        let mut shown: Vec<String> = chain[pos..]
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        shown.push(canon.display().to_string());
        return Err(format!("extends 成环:{}", shown.join(" → ")));
    }
    let raw = file_loader::read_to_string(path)
        .map_err(|e| format!("配置 {} 读取失败: {e}", path.display()))?;
    let mut v: toml::Value = toml::from_str(&raw)
        .map_err(|e| format!("配置 {} 解析失败: {e}", path.display()))?;
    // 取走顶层 flavor(预设档名 → 路径;与 extends 同为父层,flavor 在前)
    let flavor_base = match v.as_table_mut().and_then(|t| t.remove(FLAVOR_KEY)) {
        None => None,
        Some(toml::Value::String(s)) if !s.trim().is_empty() => {
            Some(crate::flavors::preset_path(s.trim()).map_err(|e| {
                format!("配置 {}:{FLAVOR_KEY} 解析失败: {e}", path.display())
            })?)
        }
        Some(toml::Value::String(_)) => {
            return Err(format!("配置 {}:{FLAVOR_KEY} 不能是空字符串", path.display()))
        }
        Some(other) => {
            return Err(format!(
                "配置 {}:{FLAVOR_KEY} 必须是预设档名(字符串),得到 {}",
                path.display(),
                toml_type_name(&other)
            ))
        }
    };
    // 取走顶层 extends(必须在最终反序列化前摘掉;类型非法 = 响亮拒,不静默忽略)
    let parent = match v.as_table_mut().and_then(|t| t.remove(EXTENDS_KEY)) {
        None => None,
        Some(toml::Value::String(s)) if !s.trim().is_empty() => Some(PathBuf::from(s)),
        Some(toml::Value::String(_)) => {
            return Err(format!("配置 {}:{EXTENDS_KEY} 不能是空字符串", path.display()))
        }
        Some(other) => {
            return Err(format!(
                "配置 {}:{EXTENDS_KEY} 必须是单个文件路径(字符串),得到 {};\n              多条继承请用链式(父档自己再 extends)",
                path.display(),
                toml_type_name(&other)
            ))
        }
    };
    chain.push(canon);
    if let Some(p) = flavor_base {
        load_config_into(&p, out, depth + 1, chain)?;
    }
    if let Some(p) = parent {
        let p = if p.is_absolute() {
            p
        } else {
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .join(p)
        };
        load_config_into(&p, out, depth + 1, chain)?;
    }
    chain.pop();
    merge(out, v);
    Ok(())
}

impl OwlConfig {
    /// **统一 loader**(Default → 基准文件 → 覆盖文件;CLI 由入口决定前两者的
    /// 取值)。每个文件都可声明 `extends` 继承另一档 —— 被继承者先合并、本文件
    /// 覆盖之(见 [`load_config_into`])。`deny_unknown_fields` 在最终反序列化
    /// 生效 —— 任一文件里任何账外键、枚举非法值 = Err 拒启。
    pub fn load(base: Option<&Path>, file: Option<&Path>) -> Result<Self, String> {
        match file {
            Some(f) => Self::load_chain(base, &[f]),
            None => Self::load_chain(base, &[]),
        }
    }

    /// **多层链加载**(2026-10-18):`base`(可选)→ `layers` 按序叠加。
    /// 每层都支援 `flavor`(预设档名 → 该层的父层)与 `extends`;后者覆盖前者。
    /// 用途:`OWL_TEST_CONF → 用例 overlay` 这条测试链(见 `testkit::case`)。
    pub fn load_chain(base: Option<&Path>, layers: &[&Path]) -> Result<Self, String> {
        let mut v = toml::Value::Table(toml::map::Map::new());
        if let Some(b) = base {
            load_config_into(b, &mut v, 0, &mut Vec::new())?;
        }
        for p in layers {
            load_config_into(p, &mut v, 0, &mut Vec::new())?;
        }
        reject_retired_keys(&v)?;
        v.try_into()
            .map_err(|e| format!("配置反序列化失败(账外键或非法枚举值): {e}"))
            .and_then(|c: Self| {
                c.validate()?;
                Ok(c)
            })
    }

    /// TOML 字符串直载(测试/ops 便捷;与 load 同一反序列化面 + 同一语义门)。
    /// **不解析 `extends`**(无文件上下文 —— 继承请走文件加载)。
    pub fn from_toml_str(s: &str) -> Result<Self, String> {
        let v: toml::Value =
            toml::from_str(s).map_err(|e| format!("配置解析失败(账外键或非法枚举值): {e}"))?;
        if v.get(EXTENDS_KEY).is_some() {
            return Err(format!(
                "配置含 `{EXTENDS_KEY}`:继承需要文件上下文 —— 用 OwlConfig::load / 服务端配置档(字符串直载不解析 extends)"
            ));
        }
        if v.get(FLAVOR_KEY).is_some() {
            return Err(format!(
                "配置含 `{FLAVOR_KEY}`:预设档引用需要文件上下文 —— 用 OwlConfig::load / 服务端配置档(字符串直载不解析 flavor)"
            ));
        }
        reject_retired_keys(&v)?;
        let c: Self = v
            .try_into()
            .map_err(|e| format!("配置反序列化失败(账外键或非法枚举值): {e}"))?;
        c.validate()?;
        Ok(c)
    }

    /// **跨字段语义门**(D1,2026-10-17):反序列化只能保证**形状**(账外键 /
    /// 枚举合法),字段之间的**语义**在这里拒 —— 三处入口(server / cli /
    /// 配置档)同守,不再各自写一份。
    ///
    /// 规则①`pool.gdn_slots ≥ server.max_concurrency`:GDN 状态格 = 每并发会话
    /// 一格;并发超格数 ⇒ 第 N+1 会话把**活跃**会话的状态格 LRU 逐出 =
    /// 跨会话状态污染(不是“排队等待”那种良性降级)。
    /// 原写法在 `apps/server/src/main.rs`(server 要读引擎内部容量、直接
    /// `exit(1)`),且**只有 server 一条入口守** —— 直接构造 engine 的路径
    /// (tests / cli)裸奔。
    pub fn validate(&self) -> Result<(), String> {
        if self.pool.gdn_slots < self.server.max_concurrency {
            return Err(format!(
                "[pool] gdn_slots({}) < [server] max_concurrency({})—— GDN 状态格 = 每并发会话一格,\n              并发超格数会把活跃会话的状态格 LRU 逐出(跨会话状态污染);请调 gdn_slots ≥ max_concurrency",
                self.pool.gdn_slots, self.server.max_concurrency
            ));
        }
        // **FI 单图烘 plan 的上界律**(2026-10-18 W3):喂进去的 `ctx_total` 被烘死
        // ⇒ 图自携的表按该上界枚举工作项;若真实 ctx 超过它,运行期算出的 chunk 数
        // 会超过表里的工作项数 ⇒ partial 行 stride 错(设计卷 §2.2 的 V1 形态,
        // 实测 T=8 max|Δ|=1.39)。**只许上界 ≥ 任何可能 ctx**,这里 fail-fast。
        // bake 档与 prefill 桶图**互斥**:桶图的 FI 表靠"图自携 restore",
        // 而 bake 档全进程关掉了 restore(host 每步 `fi_plan_sync` 为唯一权威);
        // 桶图没有那条刷新面 ⇒ 会读到别人的表。要同时用 = 先给桶路也接 sync。
        // 单图烘 plan 档必须"图不烘表"(否则回放那次 memcpy 把同步进去的表盖掉
        // ⇒ 又回 inert 表)。**拒启**而不是自动纠正:静默改语义比拒启危险。
        if self.dispatch.fi_plan_bake.is_some() && self.dispatch.fi_plan_restore {
            return Err(
                "[dispatch] fi_plan_bake 需要 fi_plan_restore = false \
                 (单图档的 plan 表由 host 每步 `fi_plan_sync` 刷新,图内不得再烘自携 restore)"
                    .to_string(),
            );
        }
        if self.dispatch.fi_plan_bake.is_some()
            && self.graph.prefill_bucket != crate::config::PrefillBucket::Off
        {
            return Err(
                "[graph] prefill_bucket 与 [dispatch] fi_plan_bake **互斥**(待桶路接 plan sync 后解禁)"
                    .to_string(),
            );
        }
        // **W5 意图配套门**（2026-10-18）：FI 臂只有"图内臂"一种承载（核留图内 +
        // `ctx_total` 烘上界 + 表由 `fi_plan_sync` 窗外刷）。缺上界 ⇒ 捕获期会拿
        // **占位 ctx** 烘表（文本连贯但错，历史踩过）⇒ 响亮拒绝，不给静默错值。
        if (self.dispatch.decode_fi || self.dispatch.verify_fi)
            && self.dispatch.fi_plan_bake.is_none()
        {
            return Err(
                "[dispatch] decode_fi/verify_fi 需要 fi_plan_bake 在场（断点面已退役：FI 核只能走                 图内臂 + 窗外刷表）—— 要么设 `fi_plan_bake`（0 = 取 max_seq），要么关掉 FI 臂"
                    .to_string(),
            );
        }
        if self.dispatch.fi_plan_bake.is_some() && !self.dispatch.fi_cuda_graph {
            return Err(
                "[dispatch] fi_plan_bake 需要 fi_cuda_graph = true（图档瓦让 plan 形状对 ctx 恒定，                 单图才可能服务任意 ctx）"
                    .to_string(),
            );
        }
        if let Some(bake) = self.dispatch.fi_plan_bake {
            if bake != 0 && bake < self.runtime.max_seq {
                return Err(format!(
                    "[dispatch] fi_plan_bake({bake}) < [runtime] max_seq({})—— \n                                   单图烘 plan 的上界必须 ≥ 引擎能服务的最大 ctx(否则表的工作项数\n                                  少于运行期 chunk 数 ⇒ partial 行 stride 错);请设 fi_plan_bake ≥ max_seq",
                    self.runtime.max_seq
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_全缺省() {
        let c = OwlConfig::default();
        assert_eq!(c.runtime.bind, "127.0.0.1:8135");
        assert_eq!(c.runtime.max_seq, 4096);
        assert_eq!(c.model.kind, ModelKind::Qwen35_08b);
        assert!(c.pool.prefix_cache);
        assert_eq!(c.sampling.mode, SamplerMode::Sampling);
        assert!(c.dispatch.qkv_fuse && c.dispatch.gdn_fused_decode);
        assert!(!c.cuda.graph_flags.upload);
        assert!(!c.graph.no_graph);
    }

    #[test]
    fn agent_面缺省与账外键() {
        let c = OwlConfig::default();
        assert!(c.agent.template_path.is_none());
        assert!(!c.agent.implicit_session, "隐式会话默认关(旧行为不变)");
        assert_eq!(c.agent.default_max_new, 4096);
        assert_eq!(c.agent.max_new_cap, 8192);
        assert!(c.agent.parse_tool_calls);
        assert_eq!(c.agent.tool_call_format, ToolCallFormat::Xml);
        assert!(!c.agent.auto_disable_thinking_with_tools);
        assert!(c.agent.reasoning_effort.is_none());
        // 账外键 / 非法枚举 = 拒启(deny_unknown_fields + 严格枚举)
        assert!(OwlConfig::from_toml_str("[agent]\ntemplate_pathz = \"x\"").is_err());
        assert!(OwlConfig::from_toml_str("[agent]\ntool_call_format = \"yaml\"").is_err());
        let c = OwlConfig::from_toml_str(
            "[agent]\nimplicit_session = true\ntool_call_format = \"json\"\nmax_new_cap = 32768\nreasoning_effort = \"high\"\n",
        )
        .expect("合法 agent 键应通过");
        assert!(c.agent.implicit_session);
        assert_eq!(c.agent.tool_call_format, ToolCallFormat::Json);
        assert_eq!(c.agent.max_new_cap, 32768);
        assert_eq!(c.agent.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn 枚举严格_非法值拒() {
        assert!(OwlConfig::from_toml_str("[model]\nkind = \"awq27bb\"").is_err());
        assert!(OwlConfig::from_toml_str("[sampling]\nmode = \"greedy1\"").is_err());
        assert!(OwlConfig::from_toml_str("[pool]\nprefix_cache = \"nope\"").is_err());
    }

    #[test]
    fn 账外键拒_deny_unknown() {
        assert!(OwlConfig::from_toml_str("[model]\nkindz = \"awq27b\"").is_err());
        assert!(OwlConfig::from_toml_str("[no such module]\nx = 1").is_err());
    }

    /// **主干量化方案**(2026-10-17 由 `w4a8: bool` 升格):缺省 = w4a16
    /// (今天行为);合法值两档;非法值**结构化拒**;**退役键 `w4a8` 响亮拒**
    /// —— 拒语要点名新键与值域(可行动,不是一律 unknown field)。
    #[test]
    fn quant_scheme_枚举与退役键门() {
        assert_eq!(OwlConfig::default().model.quant, QuantScheme::W4a16);
        let c = OwlConfig::from_toml_str("[model]\nquant = \"w4a8\"").expect("a8 档");
        assert_eq!(c.model.quant, QuantScheme::W4a8);
        let c = OwlConfig::from_toml_str("[model]\nquant = \"w4a16\"").expect("a16 档");
        assert_eq!(c.model.quant, QuantScheme::W4a16);
        let e = OwlConfig::from_toml_str("[model]\nquant = \"w4a9\"").expect_err("非法档拒");
        assert!(e.contains("w4a9"), "拒语应带现场: {e}");
        for old in ["[model]\nw4a8 = true", "[model]\nw4a8 = false"] {
            let e = OwlConfig::from_toml_str(old).expect_err("退役键必须拒(开/关同罪)");
            assert!(
                e.contains("退役键") && e.contains("quant = \"w4a8\""),
                "拒语应点名新键: {e}"
            );
        }
    }

    /// **跨字段语义门**(D1,2026-10-17):`gdn_slots < max_concurrency` 必拒,
    /// 拒语要点出病因(状态格逐出 = 跨会话污染)与调法;合规组合(生产档 1/1、
    /// 缺省 8/8)必须过。
    #[test]
    fn validate_rejects_concurrency_over_gdn_slots() {
        let e = OwlConfig::from_toml_str("[pool]\ngdn_slots = 1\n[server]\nmax_concurrency = 8\n")
            .expect_err("1 格 vs 8 并发必须拒");
        assert!(
            e.contains("gdn_slots") && e.contains("跨会话状态污染"),
            "拒语应可行动: {e}"
        );
        OwlConfig::from_toml_str("[pool]\ngdn_slots = 1\n[server]\nmax_concurrency = 1\n")
            .expect("相等 = 合规(生产 spec-dflash 档即此)");
        OwlConfig::from_toml_str("[pool]\ngdn_slots = 8\n[server]\nmax_concurrency = 2\n")
            .expect("格多于并发 = 合规");
        OwlConfig::default().validate().expect("缺省档 8/8");
    }

    /// **A4 收口(2026-10-19)**:`lrfu:λ` 解析回配置面。
    ///
    /// 病根 = 配置枚举只落 `lru|lrfu`(单元枚举),λ 固定 0.9 ⇒ 文档写的
    /// `lrfu:0.8` 在 `deny_unknown_fields` 下**直接拒启**(配置面/实现两边不一致)。
    /// 本门钉三条:①三档字面量 ②λ 真传下去 ③非法值(boot 边界)响亮拒。
    #[test]
    fn evict_policy_parses_lambda() {
        let get = |toml: &str| -> EvictPolicyCfg {
            toml::from_str::<KvmemCfg>(toml).expect("解析").evict_policy
        };
        assert_eq!(get(""), EvictPolicyCfg::Lru, "缺省 = lru");
        assert_eq!(get("evict_policy = \"lru\""), EvictPolicyCfg::Lru);
        assert_eq!(get("evict_policy = \"lrfu\""), EvictPolicyCfg::Lrfu { lambda: 0.9 });
        assert_eq!(
            get("evict_policy = \"lrfu:0.8\""),
            EvictPolicyCfg::Lrfu { lambda: 0.8 },
            "λ 必须真进枚举(旧版这里直接拒)"
        );
        assert_eq!(get("evict_policy = \"lrfu:0.25\""), EvictPolicyCfg::Lrfu { lambda: 0.25 });
        // λ 边界与拼写错 = 结构化拒(不许静默落 lru / 0.9)
        for bad in [
            "[kvmem]\nevict_policy = \"lru:0.5\"",
            "[kvmem]\nevict_policy = \"lrfu:0\"",
            "[kvmem]\nevict_policy = \"lrfu:1\"",
            "[kvmem]\nevict_policy = \"lrfu:1.5\"",
            "[kvmem]\nevict_policy = \"lrfu:-0.1\"",
            "[kvmem]\nevict_policy = \"lrfu:x\"",
            "[kvmem]\nevict_policy = \"LRFU\"",
            "[kvmem]\nevict_policy = \"\"",
        ] {
            let e = OwlConfig::from_toml_str(bad).expect_err("必须拒");
            assert!(e.contains("evict_policy"), "拒语应点名键: {e}");
        }
    }

    #[test]
    fn 正语义分派与合法文件() {
        // 注:`max_concurrency = 1` 是 **D1 语义门**(2026-10-17)要求的 ——
        // 本 fixture 原为 `gdn_slots = 1` + 缺省 `max_concurrency = 8`,
        // 那是“能反序列化但语义非法”的组合(server 侧原先会在 boot 拒启)。
        let c = OwlConfig::from_toml_str(
            "[dispatch]\nqkv_fuse = false\ngdn_fused_decode_v2 = false\n[pool]\ngdn_slots = 1\n[server]\nmax_concurrency = 1\n",
        )
        .expect("合法文件应通过");
        assert!(!c.dispatch.qkv_fuse);
        assert!(!c.dispatch.gdn_fused_decode_v2);
        assert_eq!(c.pool.gdn_slots, 1);
    }

}

#[cfg(test)]
mod prefill_tail_merge_tests {
    use super::*;

    /// L4(2026-10-18):`[runtime] prefill_tail_merge` 缺省 **0 = 关**(旧行为),
    /// 显式给值可开;非法(负数)由类型拒。
    #[test]
    fn prefill_tail_merge_defaults_off_and_parses() {
        let c = OwlConfig::from_toml_str("[runtime]\nmax_seq = 128\n").expect("解析");
        assert_eq!(c.runtime.prefill_tail_merge, 0, "缺省必须关(旧行为)");
        let c2 = OwlConfig::from_toml_str("[runtime]\nprefill_tail_merge = 1024\n").expect("解析");
        assert_eq!(c2.runtime.prefill_tail_merge, 1024);
        assert!(OwlConfig::from_toml_str("[runtime]\nprefill_tail_merge = -1\n").is_err());
    }
}

#[cfg(test)]
mod extends_tests {
    use super::*;

    /// 建临时目录/写档(每个用例一个目录;退出前自清)
    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("owl_cfg_ext_{}_{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        file_loader::create_dir_all(&d).expect("mkdir");
        d
    }
    fn write(p: &Path, body: &str) {
        file_loader::write(p, body).expect("写档");
    }

    /// 真实链路:flavor(基准) + 站点档(extends 父档) + 子档覆盖。
    /// 父档提供部署面、子档只写差异 —— 预设档未覆盖处仍保留。
    #[test]
    fn extends_父档先合并子档覆盖() {
        let d = tmp_dir("merge");
        let site = d.join("site.toml");
        write(&site, "[runtime]\nmax_seq = 12345\n[pool]\nsnap_max = 2\n");
        let child = d.join("child.toml");
        write(&child, "extends = \"site.toml\"\n\n[spec]\ndepth = 5\n");
        let flavor = crate::flavors::preset_path("spec-dflash").expect("预设档");
        let c = OwlConfig::load(Some(&flavor), Some(&child)).expect("load");
        assert_eq!(c.spec.depth, 5, "子档覆盖 flavor/父档");
        assert_eq!(c.runtime.max_seq, 12345, "父档提供、flavor 未写者生效");
        assert_eq!(c.pool.snap_max, 2, "父档覆盖 flavor");
        assert_eq!(c.sampling.mode, SamplerMode::Greedy, "flavor 未被子/父覆盖处保留");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// 链式继承(孙 extends 子 extends 父);相对路径 = 相对各自文件目录。
    #[test]
    fn extends_链式与相对路径() {
        let d = tmp_dir("chain");
        let sub = d.join("sub");
        file_loader::create_dir_all(&sub).expect("mkdir sub");
        write(&d.join("grand.toml"), "[runtime]\nmax_seq = 777\n");
        write(
            &sub.join("mid.toml"),
            "extends = \"../grand.toml\"\n[spec]\ndepth = 4\n",
        );
        let leaf = sub.join("leaf.toml");
        write(&leaf, "extends = \"mid.toml\"\n[pool]\nsnap_max = 3\n");
        let c = OwlConfig::load(None, Some(&leaf)).expect("load");
        assert_eq!(c.runtime.max_seq, 777, "祖父档向上传递");
        assert_eq!(c.spec.depth, 4, "子档在链中保留");
        assert_eq!(c.pool.snap_max, 3, "孙档自己的键在最上层");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// 成环 = 响亮拒(不静默截断);报错点名链上文件。
    #[test]
    fn extends_成环响亮拒() {
        let d = tmp_dir("cycle");
        write(&d.join("a.toml"), "extends = \"b.toml\"\n");
        write(&d.join("b.toml"), "extends = \"a.toml\"\n");
        let e = OwlConfig::load(None, Some(&d.join("a.toml"))).expect_err("必须拒");
        assert!(e.contains("成环"), "报错要点名环:{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// 非字符串 extends / 空串 / 缺失父档 = 各自响亮拒(带路径与类型名)。
    #[test]
    fn extends_非法与缺失都响亮拒() {
        let d = tmp_dir("bad");
        let bad_ty = d.join("bad_ty.toml");
        write(&bad_ty, "extends = 5\n");
        let e = OwlConfig::load(None, Some(&bad_ty)).expect_err("类型非法必须拒");
        assert!(e.contains("字符串") && e.contains("整数"), "要点名得到什么:{e}");

        let empty = d.join("empty.toml");
        write(&empty, "extends = \"  \"\n");
        let e = OwlConfig::load(None, Some(&empty)).expect_err("空串必须拒");
        assert!(e.contains("空字符串"), "{e}");

        let missing = d.join("missing.toml");
        write(&missing, "extends = \"nope.toml\"\n");
        let e = OwlConfig::load(None, Some(&missing)).expect_err("缺失父档必须拒");
        assert!(e.contains("nope.toml"), "要点名找不到的父档:{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// 字符串直载没有文件上下文 ⇒ 含 extends 的文本响亮拒(不静默忽略)。
    #[test]
    fn extends_字符串直载拒() {
        let e = OwlConfig::from_toml_str("extends = \"x.toml\"\n").expect_err("必须拒");
        assert!(e.contains("文件上下文"), "{e}");
    }

    /// `flavor = "<预设名>"` = 该文件自带的基准层(复用 flavors/ 而不靠 CLI)。
    #[test]
    fn flavor_键解析为预设基准() {
        let d = tmp_dir("flavor_key");
        let conf = d.join("test-conf.toml");
        write(&conf, "flavor = \"spec-dflash\"\n\n[spec]\ndepth = 5\n\n[runtime]\nmax_seq = 2048\n");
        let c = OwlConfig::load(None, Some(&conf)).expect("load");
        assert_eq!(c.spec.depth, 5, "文件覆盖预设");
        assert_eq!(c.runtime.max_seq, 2048, "文件键到手");
        assert_eq!(c.sampling.mode, SamplerMode::Greedy, "预设未覆盖处保留(spec-dflash)");
        assert_eq!(c.dispatch.kv_fp8, true, "预设行为面保留");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// 未知预设名 / 直载含 flavor = 响亮拒。
    #[test]
    fn flavor_非法响亮拒() {
        let d = tmp_dir("flavor_bad");
        let bad = d.join("bad.toml");
        write(&bad, "flavor = \"no-such-flavor\"\n");
        let e = OwlConfig::load(None, Some(&bad)).expect_err("未知预设必须拒");
        assert!(e.contains("no-such-flavor"), "{e}");
        let e2 = OwlConfig::from_toml_str("flavor = \"spec-dflash\"\n").expect_err("直载必须拒");
        assert!(e2.contains("文件上下文"), "{e2}");
        let _ = std::fs::remove_dir_all(&d);
    }
}

#[cfg(test)]
mod prefill_bucket_tests {
    use super::*;

    /// 三档字面量 + 缺省 off + 非法值**结构化拒**(deny_unknown_fields 同族纪律:
    /// 配置面不许有"写了但没人认"的值)
    #[test]
    fn prefill_bucket_parses_three_levels() {
        let get = |toml: &str| -> PrefillBucket {
            toml::from_str::<GraphCfg>(toml).expect("解析").prefill_bucket
        };
        assert_eq!(get(""), PrefillBucket::Off, "缺省 = off(= 今天行为)");
        assert_eq!(get("prefill_bucket = \"off\""), PrefillBucket::Off);
        assert_eq!(get("prefill_bucket = \"t512\""), PrefillBucket::T512);
        assert_eq!(get("prefill_bucket = \"geobuckets\""), PrefillBucket::Geobuckets);
        assert!(
            toml::from_str::<GraphCfg>("prefill_bucket = \"t1024\"").is_err(),
            "非法档位必须拒(不许静默落 off)"
        );
    }

    /// 基准文件 + 覆盖文件叠加(flavor 不再是配置层概念:入口把预设档解析成路径当基准)
    #[test]
    fn load_基准文件与覆盖文件叠加() {
        let dir = std::env::temp_dir().join(format!("owl_cfg_two_{}", std::process::id()));
        file_loader::create_dir_all(&dir).expect("mkdir");
        let over = dir.join("over.toml");
        file_loader::write(&over, "[spec]\ndepth = 5\n").expect("w");
        let base = crate::flavors::preset_path("spec-dflash").expect("预设档");
        let c = OwlConfig::load(Some(&base), Some(&over)).expect("load");
        assert_eq!(c.spec.depth, 5, "覆盖文件优先");
        assert_eq!(c.sampling.mode, SamplerMode::Greedy, "基准未覆盖处保留");
        let c2 = OwlConfig::load(Some(&base), None).expect("load base only");
        assert_eq!(c2.spec.depth, 7, "spec-dflash 预设 = d7");
        assert_eq!(c2.model.head_quant, HeadQuant::RtnInt4G32, "C1 int4 头");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
