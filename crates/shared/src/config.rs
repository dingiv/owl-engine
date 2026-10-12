//! 全量配置总账 + 统一 loader(2026-10-10 显式配置模块;显式依赖律终点)。
//!
//! **定位**:全 workspace 唯一的配置**定义面** —— 每个配置键在此登记
//! (键名/类型/缺省/归属 module),[`OwlConfig::load`] 是唯一 loader。
//! 消费方(engine/cuda/models/server)一律 `from_config(&OwlConfig)`
//! 组装各自的 knobs/options,**零 env 直读、零散点 parse**。
//!
//! **四层叠加**(flavor 机制,2026-10-10;参考 vllm flavors.py 三层律):
//!
//! ```text
//! Default(代码缺省) → Flavor(内置命名预设) → 用户文件(toml) → CLI 启动参数
//! ```
//!
//! flavor 只锁"**必须成套**"的旋钮(避免手拼组合踩约束,如 spec depth
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
//! | model | head_quant | HeadQuant 枚举("off"\|"rtn-int4-g32")/ off(C1;|
//! |   |   | 竖式/gate 质量门后转正) |
//! | pool | pool_tokens | Option<usize> / None(=2×单会话) |
//! | pool | gdn_slots / snap_max | usize / 8、4 |
//! | pool | vram_target / vram_reserve_mb | f64、u64 / 0.97、1024 |
//! | pool | prefix_cache | bool / true |
//! | spec | depth / dumb / draft_kv_fp8 | usize 0(off)、flag、flag |
//! | spec | notaps / tapdecl | flag / false(verify 图诊断) |
//! | spec | degrade_after / probe_every | usize / 6、64 |
//! | sampling | mode | SamplerMode 枚举("greedy"\\|"sampling")/ sampling |
//! | sampling | temp / topk / topp / rep_penalty | 1.0 / 20 / 0.95 / 1.15 |
//! | load | verify / debug_tap | flag / false |
//! | dispatch | qkv_fuse / gdn_fused_decode / gdn_fused_decode_v2 | bool / 全真(**正语义**) |
//! | dispatch | prefill_split / force_naive | flag / false |
//! | dispatch | ~~flashinfer~~ | **已移除(2026-10-13):FI 恒启用,无开关** |
//! | dispatch | gdn_chunked / gdn_scalar / kv_fp8 | flag / false |
//! | dispatch | resolve_trace / host_argmax / ts_probe | flag / false |
//! | probes | step_profile / gdn_dump / debug / prefill_cksum / fact_probe | flag / false |
//! | probes | trace_gate / dflash_probe / propose_eager | flag / false |
//! | probes | dflash_eager / dflash_noencode / dflash_dumb | flag / false |
//! | probes | gdn_dump_all / pf_bisect / pf_fin_check / raw_completion | flag / false |
//! | probes | pf_stages | Option<usize> / None |
//! | cuda | srv_timing / cap_prof / launch_sync / launch_time | flag / false |
//! | cuda | d2h_prof / gpu_prof / free_legacy | flag / false |
//! | cuda | graph_flags | { upload, device_launch } / 全 false |
//! | cuda | capture_slab_mb | Option<usize> / None(hint 定量) |
//! | cuda | *(nvrtc_include/native_aot 2026-10-13 退役 —— NVRTC 已删)* | — |
//! | graph | no_graph | flag / false |
//!
//! 账外键(不入公共面):测试私有旋钮(OWL_GATE_*/OWL_E2E_*/OWL_SWEEP_*/
//! OWL_WS_MUL/SPLIT_*/MARLIN_*/FI_DEBUG/OWL_27B_STRICT/OWL_HF_PARITY ——
//! 测试用例自由直读)、构建期键(build.rs 例外)、OWL_CONFIG(loader 路径
//! 兜底,server 入口读)。

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
    pub device: usize,
    pub max_seq: usize,
    pub prefill_chunk: usize,
    /// 测试域设备序(不入 toml;测试入口自填)
    #[serde(skip)]
    pub test_device: Option<usize>,
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
}

/// 池几何与显存预算(engine StatePool)
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PoolCfg {
    pub pool_tokens: Option<usize>,
    pub gdn_slots: usize,
    pub snap_max: usize,
    pub vram_target: f64,
    pub vram_reserve_mb: u64,
    pub prefix_cache: bool,
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
    pub degrade_after: usize,
    pub probe_every: usize,
}

/// 采样参数
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingCfg {
    pub mode: SamplerMode,
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

/// 诊断探针族(全域布尔;boot 一次解析,热路径零 env)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProbesCfg {
    pub step_profile: bool,
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
}

/// CUDA 设备面(诊断旗标 + 图实例化位面 + slab 定档 + nvrtc 工具链)
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaCfg {
    pub srv_timing: bool,
    pub cap_prof: bool,
    pub launch_sync: bool,
    pub launch_time: bool,
    pub d2h_prof: bool,
    pub gpu_prof: bool,
    pub free_legacy: bool,
    pub graph_flags: GraphInstantiateFlags,
    /// 捕获 slab 固定档 MiB(None = warmup 计量定量)
    pub capture_slab_mb: Option<usize>,
    // nvrtc_include(2026-10-13 退役):NVRTC 编译路径已删 —— 设备码一律
    // 构建期 cubin 资产(源哈希查表),include 路径是构建期 `nvcc` 的事。
    // 键已从配置面移除(不许留死键,`deny_unknown_fields` 会让旧档拒启 →
    // 显式暴露"该档还在用已退役旋钮")。
}

/// 图装配面(engine GraphPlanDesc)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraphCfg {
    /// eager 直发,跳过图捕获
    pub no_graph: bool,
}

/// 本地服务面(并发/排队;owl 定位 = 本地单机,不做对外高并发)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerCfg {
    /// 最大并发请求数(在跑 + 排队;超出者排队等待)。
    /// 默认 8;须 ≤ pool 的 GDN 状态格数(knobs.gdn_slots,每并发会话
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
    pub spec: SpecCfg,
    pub sampling: SamplingCfg,
    pub load: LoadCfg,
    pub dispatch: DispatchCfg,
    pub probes: ProbesCfg,
    pub cuda: CudaCfg,
    pub graph: GraphCfg,
    pub server: ServerCfg,
}

impl Default for RuntimeCfg {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8135".into(),
            device: 0,
            max_seq: 4096,
            prefill_chunk: 512,
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
            vram_target: 0.97,
            vram_reserve_mb: 1024,
            prefix_cache: true,
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
            probe_every: 64,
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
            decode_xqa: false,
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
            rep_penalty: 1.15,
        }
    }
}

// ============================================================================
// §4 flavor 预设(只锁"必须成套"的旋钮;每档注释 = 为什么)
// ============================================================================

/// 内置 flavor 表(name → TOML patch;与用户文件同一 serde 解析路径,
/// 零额外机制)。`--flavor <name>` 激活;文件在其上覆盖。
pub const FLAVORS: &[(&str, &str)] = &[
    (
        // 0.8B 开发基线(dev 面;= 代码缺省 + bind 明示。CI/冒烟快捷档)
        "dev-08b",
        r#"
[runtime]
bind = "127.0.0.1:8135"
"#,
    ),
    (
        // 27B AWQ 生产档(24G 贴顶三旋钮:单会话门不需要多格/多快照/
        // 前缀缓存 —— 实测各省 ~300MB;greedy = 恒等门口径)
        // 需配套:model.dir(检查点目录)
        "awq27b",
        r#"
[model]
kind = "awq27b"

[pool]
gdn_slots = 1
snap_max = 1
prefix_cache = false

[sampling]
mode = "greedy"

[dispatch]
# 裸 27B 档白名单(实测 27 个;与 kernels `ops::PROD_AWQ27B_2026_10_13` 一致)
allowed_ops = [
        "cublas_gemm_f16", "flashinfer_prefill_paged_fp8kv",
        "marlin_gemm_w4a16_awq", "owl_add_f16",
        "owl_argmax_f32idx_f16", "owl_ct_repack_u32",
        "owl_embed_f16", "owl_fused_add_rmsnorm_f16",
        "owl_gdn_conv_fwd_f16", "owl_gdn_conv_upd_dual_f16",
        "owl_gdn_decode_step_v2_f16", "owl_gdn_gating_g_f16",
        "owl_gdn_l2norm_f16", "owl_gdn_norm_act_f16",
        "owl_gdn_recurrence_varlen_gqa_f16", "owl_gemv_dual_f16",
        "owl_narrow_strided_f16", "owl_norm_rope_f16",
        "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd", "owl_reshape_and_cache_fp8kv_knhd",
        "owl_rmsnorm_f16", "owl_sigmoid_f16",
        "owl_sigmoid_gate_mul_f16", "owl_silu_and_mul_f16",
        "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256", "vllm_paged_attention_v2_fp8_knhd_hd256bs32",
        "vllm_paged_attention_v2_reduce_f16_hd256",
]
"#,
    ),
    (
        // 27B + DFlash2 真草稿生产档(2026-10-10 实测配方;24G 单卡贴顶):
        // decode 34.56@512(对照 perf-roadmap 首测 fp8 主池+d7 = 35.16,同口径
        // 吻合)、Captured 零降级、装配 16.6s。
        // 锁"必须成套":
        // - d7 + fp8 主池(perf-roadmap 首测口径);
        // - slots=1(单会话门;slots=8 的 GDN 格 = fixed_after 1.7G,
        //   budget 直接负 → 池 0 → OOM,实测立案);
        // - capture_slab 用 hint 定量(**勿设固定档**:160MB × ~10 图族
        //   = 1.6G,贴顶必死,实测立案);
        // - vram_target 0.99 + 钉池 4352(治理只做护栏,池由显式钉值定)。
        // 需配套:model.dir + model.dflash2_dir(检查点目录);
        // 建议 model.name 显式给(缺省应答名是 0.8b 的,health 会误导)。
        "spec-dflash",
        r#"
[model]
kind = "awq27b"
# C1(2026-10-11):lm_head int4 装载期现场 RTN(marlin kU4);质量门
# 已过(prose/数学 A/B 逐字同,竖式同错=模型固有),数学域 159→177
head_quant = "rtn-int4-g32"

[runtime]
# 2026-10-13 重定档(影子池时代旧值 = 2560/池 4352):单池 + FI 直读后
# 内存不再是约束 —— 实测 vram 治理 `fixed_after=0.68G budget=4.66G
# per_tok=53248B → 池上限 94024 tok`(fp8 目标池 32KB/tok + DFlash2
# bf16 草稿池 20KB/tok)。故**池值现为产品选择而非内存必需**:
# 取 16k ctx(相对旧档 6.4×),实测代价 ~2.5% prefill、~0.5% decode
# (池/nparts 9472/19 → 16896/33 的 A/B)。
max_seq = 16384

[spec]
depth = 7

[server]
# 单会话语义:GDN 格锁 1 → 并发闸门必须 ≤1(启动 fail-fast 校验
# gdn_slots ≥ max_concurrency)。2026-10-13 实测:本档原**缺此项** →
# `--flavor spec-dflash` 独立起不来(此前 e2e/bench 各自在配置文件里
# 覆盖了 gdn_slots=8 才蒙混过去)。多会话档 = 这里改 8 + gdn_slots=8。
max_concurrency = 1

[pool]
pool_tokens = 16896   # = ceil(16384/page)+slack(页 32;528 页;≈0.9G)
vram_target = 0.99
vram_reserve_mb = 512
gdn_slots = 1
snap_max = 1
prefix_cache = false

[dispatch]
kv_fp8 = true
# **生产算子白名单**(P4 硬门;清单 = 实测台账,与 kernels
# `ops::PROD_SPEC_DFLASH_2026_10_13` 逐项一致 —— 由 apps/server 的门测试锁死)。
# 非空 ⇒ 未列出的算子发射即结构化拒(deny-by-default);装配期还会用它跑
# 池写集穷举审计(读必有人写 + 声明自洽 + 活跃写者不歧义)。
allowed_ops = [
        "cublas_gemm_bf16", "cublas_gemm_f16",
        "flashinfer_prefill_paged_fp8kv", "gdn_chunked_delta_rule_fwd",
        "marlin_gemm_w4a16_awq", "marlin_gemm_w4a16_bf16",
        "owl_add_bf16", "owl_add_f16",
        "owl_argmax_f32idx_f16", "owl_cast_bf16_f16",
        "owl_cast_f16_bf16", "owl_concat_rows_f16",
        "owl_concat_rows_f32", "owl_ct_repack_u32",
        "owl_dflash_conv_bf16", "owl_dflash_select_bf16",
        "owl_embed_f16", "owl_fused_add_rmsnorm_bf16",
        "owl_fused_add_rmsnorm_f16", "owl_gdn_conv_fwd_f16",
        "owl_gdn_conv_upd_dual_f16", "owl_gdn_decode_step_v2_f16",
        "owl_gdn_gating_g_f16", "owl_gdn_l2norm_f16",
        "owl_gdn_norm_act_f16", "owl_gdn_recurrence_varlen_gqa_f16",
        "owl_gemv_dual_f16", "owl_naive_attn_nc_bf16_v2",
        "owl_narrow_strided_f16", "owl_norm_rope_bf16",
        "owl_norm_rope_f16", "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",
        "owl_reshape_and_cache_fp8kv_knhd", "owl_rmsnorm_bf16",
        "owl_rmsnorm_f16", "owl_sigmoid_f16",
        "owl_sigmoid_gate_mul_f16", "owl_silu_and_mul_bf16",
        "owl_silu_and_mul_f16", "owl_topk16_f16",
        "owl_xqa_decode", "owl_xqa_verify",
        "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256", "vllm_reshape_and_cache_bf16",
]
# XQA verify 臂(2026-10-13 D3 **转正**):恒等门 3/3 稳定绿(spec ≡ 无 spec
# 逐 token;fp8 池 + 27B + DFlash2),verify 纯时 65.2→31.2ms、8192 spec
# decode 27.7→47.0 t/s(+70%,且首次反超同池裸档 36.4)。形状不匹配时层侧
# 自动回落 v2 臂(graceful)。根因与修法见工单 §2.7(semaphores 走图输入槽,
# 不用跨账本的 memset 命令)。
verify_xqa = true
# 裸 decode 走 XQA(2026-10-13 ②):第二份实例化 `M_TILESIZE=16`。
# 注:短 ctx 收益略低(512 档 1.90× vs 2048 档 2.25×),但**不缺省加 ctx
# 阈值门** —— 最小值也在 1.4× 以上,且阈值本身是额外的一支分支/形状面。
decode_xqa = true
# GDN prefill 臂(2026-10-11 终版):fork-bf16 全家桶转正 —— g 语义律
# 修复(kkt/h/o 必须吃 cumsum 而非 raw g;raw g 在深谷 -24 时
# exp(±24)=e²⁴ 爆炸)。恒等门:四域 vs recurrence 逐位一致;std 矩阵
# prefill 1299(超 scalar 1294)。fork 六核 bf16 配方全在 handler 肠内。
gdn_chunked = true

[sampling]
mode = "greedy"
"#,
    ),
    (
        // fp8 KV 池档(2026-10-13:FlashInfer **恒启用**,故本档只剩 fp8 KV
        // 这一维;原"FI 开关 + 27B FI 面挂账"随开关移除一并作废)
        "fi-kv8",
        r#"
[dispatch]
kv_fp8 = true
"#,
    ),
    (
        // eager 诊断档(图内/图外行为 A/B 的对照臂;C1 禁 graph 配套)
        "eager-debug",
        r#"
[graph]
no_graph = true

[probes]
debug = true
"#,
    ),
];

/// 可用 flavor 名(──list-flavors 消费面)
pub fn flavor_names() -> impl Iterator<Item = &'static str> {
    FLAVORS.iter().map(|(n, _)| *n)
}

fn flavor_toml(name: &str) -> Option<&'static str> {
    FLAVORS.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

// ============================================================================
// §5 统一 loader(四层叠加;deny_unknown 在最终反序列化生效)
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

impl OwlConfig {
    /// **统一 loader**(四层叠加:Default → flavor → 文件;CLI 由入口
    /// 决定前两者的取值)。`deny_unknown_fields` 在最终反序列化生效 ——
    /// 文件/flavor 里任何账外键、枚举非法值 = Err 拒启。
    pub fn load(flavor: Option<&str>, file: Option<&Path>) -> Result<Self, String> {
        let mut v = toml::Value::Table(toml::map::Map::new());
        if let Some(name) = flavor {
            let patch = flavor_toml(name)
                .ok_or_else(|| format!("未知 flavor {name:?}(可用:{:?})", flavor_names().collect::<Vec<_>>()))?;
            let pv: toml::Value =
                toml::from_str(patch).map_err(|e| format!("内置 flavor {name:?} 解析失败: {e}"))?;
            merge(&mut v, pv);
        }
        if let Some(p) = file {
            let raw = file_loader::read_to_string(p)
                .map_err(|e| format!("配置文件 {} 读取失败: {e}", p.display()))?;
            let fv: toml::Value = toml::from_str(&raw)
                .map_err(|e| format!("配置文件 {} 解析失败: {e}", p.display()))?;
            merge(&mut v, fv);
        }
        v.try_into()
            .map_err(|e| format!("配置反序列化失败(账外键或非法枚举值): {e}"))
    }

    /// TOML 字符串直载(测试/ops 便捷;与 load 同一反序列化面)
    pub fn from_toml_str(s: &str) -> Result<Self, String> {
        toml::from_str(s).map_err(|e| format!("配置解析失败(账外键或非法枚举值): {e}"))
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

    #[test]
    fn 正语义分派与合法文件() {
        let c = OwlConfig::from_toml_str(
            "[dispatch]\nqkv_fuse = false\ngdn_fused_decode_v2 = false\n[pool]\ngdn_slots = 1\n",
        )
        .expect("合法文件应通过");
        assert!(!c.dispatch.qkv_fuse);
        assert!(!c.dispatch.gdn_fused_decode_v2);
        assert_eq!(c.pool.gdn_slots, 1);
    }

    #[test]
    fn flavor_全表可解析且叠加生效() {
        for (name, patch) in FLAVORS {
            let c = OwlConfig::from_toml_str(patch)
                .unwrap_or_else(|e| panic!("flavor {name} 应可解析: {e}"));
            let _ = c;
        }
        // spec-dflash(2026-10-10 实测配方):d7 + fp8 主池 + 贴顶池预算 +
        // hint slab(勿设固定档,160MB × 图族贴顶必死)
        let c = OwlConfig::from_toml_str(flavor_toml("spec-dflash").expect("存在")).expect("解析");
        assert_eq!(c.spec.depth, 7);
        assert_eq!(c.sampling.mode, SamplerMode::Greedy);
        assert_eq!(c.model.kind, ModelKind::Awq27b);
        assert_eq!(c.model.head_quant, HeadQuant::RtnInt4G32, "C1 int4 头(2026-10-11 固化)");
        assert!(c.dispatch.kv_fp8);
        assert!(c.cuda.capture_slab_mb.is_none(), "slab 走 hint 定量");
    }

    #[test]
    fn load_文件覆盖_flavor_文件优先() {
        let dir = std::env::temp_dir().join(format!("owl_cfg_test_{}", std::process::id()));
        file_loader::create_dir_all(&dir).expect("mkdir");
        let p = dir.join("owl.toml");
        file_loader::write(&p, "[spec]\ndepth = 5\n").expect("write");
        let c = OwlConfig::load(Some("spec-dflash"), Some(&p)).expect("load");
        assert_eq!(c.spec.depth, 5, "文件覆盖 flavor");
        assert_eq!(c.sampling.mode, SamplerMode::Greedy, "flavor 未覆盖处保留");
        let _ = std::fs::remove_dir_all(&dir);

        assert!(OwlConfig::load(Some("nope"), None).is_err(), "未知 flavor 拒");
        assert!(OwlConfig::load(None, Some(Path::new("/nonexistent/owl.toml"))).is_err());
    }
}
