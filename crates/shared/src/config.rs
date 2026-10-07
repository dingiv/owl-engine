//! 全量配置总账 + 统一 loader(2026-10-10 显式配置模块;显式依赖律终点)。
//!
//! **定位**:全 workspace 唯一的配置**定义面** —— 每个配置键在此登记
//! (键名/类型/缺省/归属 module),[`OwlConfig::from_env`] 是唯一 loader
//! (一次读齐 → 强类型对象返回)。消费方(engine/cuda/models/server)
//! 一律 `from_config(&OwlConfig)` 组装各自的 knobs/options,**零 env 直读、
//! 零散点 parse**;入口(server config/测试用例)构造 OwlConfig 后显式下传。
//!
//! **类型纪律**:枚举语义严格枚举(ModelKind/SamplerMode/
//! GraphInstantiateFlags)—— 非法值 = `Err` fail-fast(消息带合法值),
//! 不许 String 和稀泥;数值解析失败 = warn + 缺省(env_reader 降级可见)。
//!
//! **全量配置总账**(键 → module;新键必须在此登记,禁止账外键):
//!
//! | module | 键 | 类型/缺省 |
//! |---|---|---|
//! | runtime | OWL_DEVICE | usize / 0 |
//! | runtime | OWL_MAX_SEQ | usize / 4096 |
//! | runtime | OWL_PREFILL_CHUNK | usize / 512 |
//! | runtime | OWL_TEST_DEVICE | Option<usize> / None(测试域) |
//! | runtime | OWL_SERVER_URL | Option<String> / None(cli) |
//! | runtime | OWL_BIND | String / 127.0.0.1:8135(server) |
//! | model | OWL_MODEL_KIND | ModelKind 枚举 / Qwen35_08b |
//! | model | OWL_MODEL_DIR | Option<PathBuf> / None |
//! | model | OWL_MODEL_NAME | Option<String> / None |
//! | model | OWL_AWQ27B_DIR | Option<String> / None(27B 资产门控) |
//! | model | OWL_DFLASH2_DIR | Option<String> / None(草稿检查点) |
//! | pool | OWL_POOL_TOKENS | Option<usize> / None(=2×单会话) |
//! | pool | OWL_GDN_SLOTS | usize / 8 |
//! | pool | OWL_SNAP_MAX | usize / 4 |
//! | pool | OWL_VRAM_TARGET | f64 / 0.97 |
//! | pool | OWL_VRAM_RESERVE_MB | u64 / 1024 |
//! | pool | OWL_PREFIX_CACHE | bool(严格 0/1/off/on)/ true |
//! | spec | OWL_SPEC_DEPTH | usize / 0(off) |
//! | spec | OWL_SPEC_DUMB | flag / false |
//! | spec | OWL_DFLASH_KV_FP8 | flag / false |
//! | spec | OWL_DFLASH_NOTAPS | flag / false |
//! | spec | OWL_DFLASH_TAPDECL | flag / false |
//! | spec | OWL_SPEC_DEGRADE_AFTER | usize / 6 |
//! | spec | OWL_SPEC_PROBE_EVERY | usize / 64 |
//! | sampling | OWL_SAMPLER | SamplerMode 枚举 / Sampling |
//! | sampling | OWL_TEMP | f32 / 1.0 |
//! | sampling | OWL_TOPK | usize / 20 |
//! | sampling | OWL_TOPP | f32 / 0.95 |
//! | sampling | OWL_REP_PENALTY | f32 / 1.15 |
//! | load | OWL_LOAD_VERIFY | flag / false |
//! | load | OWL_LOAD_DEBUG | flag / false |
//! | dispatch | OWL_QKV_NO_FUSE | 反 flag(qkv_fuse 缺省真) |
//! | dispatch | OWL_PREFILL_SPLIT | flag / false |
//! | dispatch | OWL_FORCE_NAIVE | flag / false |
//! | dispatch | OWL_FLASHINFER | flag / false |
//! | dispatch | OWL_GDN_CHUNKED | flag / false |
//! | dispatch | OWL_GDN_SCALAR | flag / false |
//! | dispatch | OWL_GDN_NO_FUSE_DECODE | 反 flag(缺省真) |
//! | dispatch | OWL_GDN_NO_FUSE_DECODE_V2 | 反 flag(缺省真) |
//! | dispatch | OWL_KV_FP8 | flag / false(→ KvQuant::Fp8E4M3) |
//! | dispatch | OWL_RESOLVE_TRACE | flag / false |
//! | dispatch | OWL_HOST_ARGMAX | flag / false |
//! | dispatch | OWL_TS_PROBE | flag / false |
//! | probes | OWL_STEP_PROFILE | flag / false |
//! | probes | OWL_GDN_DUMP | flag / false |
//! | probes | OWL_DEBUG | flag / false |
//! | probes | OWL_PREFILL_CKSUM | flag / false |
//! | probes | OWL_FACT_PROBE | flag / false |
//! | probes | OWL_TRACE_GATE | flag / false |
//! | probes | OWL_DFLASH_PROBE | flag / false |
//! | probes | OWL_PROPOSE_EAGER | flag / false |
//! | probes | OWL_DFLASH_EAGER | flag / false |
//! | probes | OWL_DFLASH_NOENCODE | flag / false |
//! | probes | OWL_DFLASH_DUMB | flag / false |
//! | probes | OWL_GDN_DUMP_ALL | flag / false |
//! | probes | OWL_PF_BISECT | flag / false |
//! | probes | OWL_PF_STAGES | Option<usize> / None |
//! | probes | OWL_PF_FIN_CHECK | flag / false |
//! | probes | OWL_RAW_COMPLETION | flag / false |
//! | cuda | OWL_SRV_TIMING | flag / false |
//! | cuda | OWL_CAP_PROF | flag / false |
//! | cuda | OWL_LAUNCH_SYNC | flag / false |
//! | cuda | OWL_LAUNCH_TIME | flag / false |
//! | cuda | OWL_D2H_PROF | flag / false |
//! | cuda | OWL_GPU_PROF | flag / false |
//! | cuda | OWL_FREE_LEGACY | flag / false |
//! | cuda | OWL_GRAPH_FLAGS | GraphInstantiateFlags 枚举 / 空 |
//! | cuda | OWL_CAPTURE_SLAB_MB | Option<usize> / None(hint 定量) |
//! | cuda | CUDA_HOME / CUDA_PATH | Option<PathBuf>(nvrtc include) |
//! | graph | OWL_NO_GRAPH | flag / false(capture = !no_graph) |
//!
//! 账外键(不入公共面):测试私有旋钮(OWL_GATE_*/OWL_E2E_*/OWL_SWEEP_*/
//! OWL_WS_MUL/SPLIT_*/MARLIN_*/FI_DEBUG/OWL_27B_STRICT/OWL_HF_PARITY ——
//! 测试用例自由直读)与构建期键(OWL_CUDA_ARCH/OWL_NVCC/OUT_DIR/AR ——
//! build.rs 例外)。

use std::path::PathBuf;

use crate::env_reader;

// ============================================================================
// §1 枚举类型(严格枚举;非法值 fail-fast,不许 String 和稀泥)
// ============================================================================

/// 模型档位(OWL_MODEL_KIND)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelKind {
    /// Qwen3.5-0.8B f16(开发基线)
    #[default]
    Qwen35_08b,
    /// Qwen3.8-27B AWQ-INT4(生产档)
    Awq27b,
}

impl ModelKind {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "0.8b" => Ok(Self::Qwen35_08b),
            "awq27b" => Ok(Self::Awq27b),
            other => Err(format!(
                "OWL_MODEL_KIND={other:?} 非法(合法:0.8b | awq27b)"
            )),
        }
    }
}

/// 采样模式(OWL_SAMPLER;原 `!= "greedy"` 字符串和稀泥废除 —— 未知值
/// 静默当采样的隐患收口)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SamplerMode {
    /// 贪心(恒等门/测试口径)
    Greedy,
    /// 采样(缺省 = 成功推理默认姿势)
    #[default]
    Sampling,
}

impl SamplerMode {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "greedy" => Ok(Self::Greedy),
            "sampling" | "1" => Ok(Self::Sampling),
            other => Err(format!(
                "OWL_SAMPLER={other:?} 非法(合法:greedy | sampling)"
            )),
        }
    }
}

/// 图实例化旗标(OWL_GRAPH_FLAGS;原裸 u64 魔数 2/4 收口)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphInstantiateFlags {
    /// UPLOAD=2(实例化后预热上传)
    pub upload: bool,
    /// DEVICE_LAUNCH=4(设备侧派发;10.8µs/节点派发税排查面)
    pub device_launch: bool,
}

impl GraphInstantiateFlags {
    fn parse(raw: &str) -> Result<Self, String> {
        let bits: u64 = raw
            .parse()
            .map_err(|_| format!("OWL_GRAPH_FLAGS={raw:?} 非法(合法:0 | 2 | 4 | 6)"))?;
        if bits & !0b110 != 0 {
            return Err(format!("OWL_GRAPH_FLAGS={raw:?} 含未定位(合法:0 | 2 | 4 | 6)"));
        }
        Ok(Self { upload: bits & 0b10 != 0, device_launch: bits & 0b100 != 0 })
    }

    /// cuda GraphInstantiateWithFlags 位面(唯一出口;UPLOAD=0b10 /
    /// DEVICE_LAUNCH=0b100,与解析同尺)
    pub fn to_bits(self) -> u64 {
        ((self.upload as u64) << 1) | ((self.device_launch as u64) << 2)
    }
}

// ============================================================================
// §2 config modules(每 module 管一个功能域;字段 = 强类型,见总账)
// ============================================================================

/// 进程装配面(设备/窗口/块长/测试设备/cli 端点)
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeCfg {
    /// server 监听地址
    pub bind: String,
    pub device: usize,
    pub max_seq: usize,
    pub prefill_chunk: usize,
    /// 测试域设备序(None = 测试跳过)
    pub test_device: Option<usize>,
    /// cli 上行端点
    pub server_url: Option<String>,
}

/// 模型资产面(档位/目录/名/检查点门控)
#[derive(Clone, Debug, PartialEq)]
pub struct ModelCfg {
    pub kind: ModelKind,
    pub model_dir: Option<PathBuf>,
    pub model_name: Option<String>,
    /// 27B AWQ 检查点目录(测试/E2E 门控)
    pub awq27b_dir: Option<String>,
    /// DFlash2 草稿检查点目录
    pub dflash2_dir: Option<String>,
}

/// 池几何与显存预算(engine StatePool)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoolCfg {
    pub pool_tokens: Option<usize>,
    pub gdn_slots: usize,
    pub snap_max: usize,
    pub vram_target: f64,
    pub vram_reserve_mb: u64,
    pub prefix_cache: bool,
}

/// 投机解码(spec 三态/草稿池量化/B4 降级参数/verify 图诊断)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// 采样参数(mode 枚举 + generation_config 同款三维 + 反循环惩罚)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingCfg {
    pub mode: SamplerMode,
    pub temp: f32,
    pub topk: usize,
    pub topp: f32,
    pub rep_penalty: f32,
}

/// 装载域(校验/观测)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadCfg {
    /// 装载校验(整块回读 vs staged 校验和)
    pub verify: bool,
    /// 装载观测 tap(stderr 逐键)
    pub debug_tap: bool,
}

/// 解释器分派面(models EnvProvider 组装源;被动律:引擎 = 环境事实来源)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchCfg {
    /// W2 qkv 融合(缺省真;反开关退出)
    pub qkv_fuse_off: bool,
    pub prefill_split: bool,
    pub force_naive_prefill: bool,
    /// FlashInfer prefill 面
    pub flashinfer: bool,
    pub gdn_chunked: bool,
    pub gdn_scalar: bool,
    /// GDN decode 融合(缺省真;反开关退出)
    pub gdn_no_fuse_decode: bool,
    /// D1-v2(缺省真;反开关退出)
    pub gdn_no_fuse_decode_v2: bool,
    /// KV 池 fp8(→ KvQuant::Fp8E4M3)
    pub kv_fp8: bool,
    pub resolve_trace: bool,
    pub host_argmax: bool,
    pub ts_probe: bool,
}

/// 诊断探针族(全域布尔;boot 一次解析,热路径零 env)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

/// CUDA 设备面(诊断旗标 + 图实例化枚举 + slab 定档 + nvrtc 工具链)
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
    /// nvrtc include(入口自 CUDA_HOME/CUDA_PATH 解析)
    pub nvrtc_include: Option<PathBuf>,
}

/// 图装配面(engine GraphPlanDesc)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphCfg {
    /// eager 直发,跳过图捕获
    pub no_graph: bool,
}

// ============================================================================
// §3 总对象 + 统一 loader
// ============================================================================

/// 全量配置对象(loader 唯一产物;进程内冻结,只读下传)
#[derive(Clone, Debug, Default, PartialEq)]
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
}

/// 严格布尔(值语义开关;flag 族用 env_reader::flag,不入此)
fn bool_strict(key: &str, default: bool) -> Result<bool, String> {
    match env_reader::str(key).as_deref() {
        None => Ok(default),
        Some("0" | "false" | "off") => Ok(false),
        Some("1" | "true" | "on") => Ok(true),
        Some(other) => Err(format!(
            "{key}={other:?} 非法布尔(合法:0 | false | off | 1 | true | on)"
        )),
    }
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

impl Default for ModelCfg {
    fn default() -> Self {
        Self { kind: ModelKind::default(), model_dir: None, model_name: None, awq27b_dir: None, dflash2_dir: None }
    }
}

impl Default for PoolCfg {
    fn default() -> Self {
        Self { pool_tokens: None, gdn_slots: 8, snap_max: 4, vram_target: 0.97, vram_reserve_mb: 1024, prefix_cache: true }
    }
}

impl Default for SpecCfg {
    fn default() -> Self {
        Self { depth: 0, dumb: false, draft_kv_fp8: false, notaps: false, tapdecl: false, degrade_after: 6, probe_every: 64 }
    }
}

impl Default for SamplingCfg {
    fn default() -> Self {
        Self { mode: SamplerMode::default(), temp: 1.0, topk: 20, topp: 0.95, rep_penalty: 1.15 }
    }
}

impl OwlConfig {
    /// **统一 loader**(全 workspace 唯一多键读取点;入口调用一次,
    /// 强类型对象返回后进程内冻结)。枚举非法 = Err(fail-fast);
    /// 数值解析失败 = warn + 缺省(env_reader 降级可见)。
    pub fn from_env() -> Result<Self, String> {
        let flag = |k: &str| env_reader::flag(k);
        Ok(Self {
            runtime: RuntimeCfg {
                bind: env_reader::str_or("OWL_BIND", "127.0.0.1:8135"),
                device: env_reader::parse_or("OWL_DEVICE", 0),
                max_seq: env_reader::parse_or("OWL_MAX_SEQ", 4096),
                prefill_chunk: env_reader::parse_or("OWL_PREFILL_CHUNK", 512),
                test_device: env_reader::parse("OWL_TEST_DEVICE"),
                server_url: env_reader::str("OWL_SERVER_URL"),
            },
            model: ModelCfg {
                kind: match env_reader::str("OWL_MODEL_KIND") {
                    None => ModelKind::default(),
                    Some(raw) => ModelKind::parse(&raw)?,
                },
                model_dir: env_reader::str("OWL_MODEL_DIR").map(PathBuf::from),
                model_name: env_reader::str("OWL_MODEL_NAME"),
                awq27b_dir: env_reader::str("OWL_AWQ27B_DIR"),
                dflash2_dir: env_reader::str("OWL_DFLASH2_DIR"),
            },
            pool: PoolCfg {
                pool_tokens: env_reader::parse("OWL_POOL_TOKENS"),
                gdn_slots: env_reader::parse_or("OWL_GDN_SLOTS", 8),
                snap_max: env_reader::parse_or("OWL_SNAP_MAX", 4),
                vram_target: env_reader::parse_or("OWL_VRAM_TARGET", 0.97),
                vram_reserve_mb: env_reader::parse_or("OWL_VRAM_RESERVE_MB", 1024u64),
                prefix_cache: bool_strict("OWL_PREFIX_CACHE", true)?,
            },
            spec: SpecCfg {
                depth: env_reader::parse_or("OWL_SPEC_DEPTH", 0),
                dumb: flag("OWL_SPEC_DUMB"),
                draft_kv_fp8: flag("OWL_DFLASH_KV_FP8"),
                notaps: flag("OWL_DFLASH_NOTAPS"),
                tapdecl: flag("OWL_DFLASH_TAPDECL"),
                degrade_after: env_reader::parse_or("OWL_SPEC_DEGRADE_AFTER", 6),
                probe_every: env_reader::parse_or("OWL_SPEC_PROBE_EVERY", 64).max(1),
            },
            sampling: SamplingCfg {
                mode: match env_reader::str("OWL_SAMPLER") {
                    None => SamplerMode::default(),
                    Some(raw) => SamplerMode::parse(&raw)?,
                },
                temp: env_reader::parse_or("OWL_TEMP", 1.0),
                topk: env_reader::parse_or("OWL_TOPK", 20usize),
                topp: env_reader::parse_or("OWL_TOPP", 0.95),
                rep_penalty: env_reader::parse_or("OWL_REP_PENALTY", 1.15),
            },
            load: LoadCfg {
                verify: flag("OWL_LOAD_VERIFY"),
                debug_tap: flag("OWL_LOAD_DEBUG"),
            },
            dispatch: DispatchCfg {
                qkv_fuse_off: flag("OWL_QKV_NO_FUSE"),
                prefill_split: flag("OWL_PREFILL_SPLIT"),
                force_naive_prefill: flag("OWL_FORCE_NAIVE"),
                flashinfer: flag("OWL_FLASHINFER"),
                gdn_chunked: flag("OWL_GDN_CHUNKED"),
                gdn_scalar: flag("OWL_GDN_SCALAR"),
                gdn_no_fuse_decode: flag("OWL_GDN_NO_FUSE_DECODE"),
                gdn_no_fuse_decode_v2: flag("OWL_GDN_NO_FUSE_DECODE_V2"),
                kv_fp8: flag("OWL_KV_FP8"),
                resolve_trace: flag("OWL_RESOLVE_TRACE"),
                host_argmax: flag("OWL_HOST_ARGMAX"),
                ts_probe: flag("OWL_TS_PROBE"),
            },
            probes: ProbesCfg {
                step_profile: flag("OWL_STEP_PROFILE"),
                gdn_dump: flag("OWL_GDN_DUMP"),
                debug: flag("OWL_DEBUG"),
                prefill_cksum: flag("OWL_PREFILL_CKSUM"),
                fact_probe: flag("OWL_FACT_PROBE"),
                trace_gate: flag("OWL_TRACE_GATE"),
                dflash_probe: flag("OWL_DFLASH_PROBE"),
                propose_eager: flag("OWL_PROPOSE_EAGER"),
                dflash_eager: flag("OWL_DFLASH_EAGER"),
                dflash_noencode: flag("OWL_DFLASH_NOENCODE"),
                dflash_dumb: flag("OWL_DFLASH_DUMB"),
                gdn_dump_all: flag("OWL_GDN_DUMP_ALL"),
                pf_bisect: flag("OWL_PF_BISECT"),
                pf_stages: env_reader::parse("OWL_PF_STAGES"),
                pf_fin_check: flag("OWL_PF_FIN_CHECK"),
                raw_completion: flag("OWL_RAW_COMPLETION"),
            },
            cuda: CudaCfg {
                srv_timing: flag("OWL_SRV_TIMING"),
                cap_prof: flag("OWL_CAP_PROF"),
                launch_sync: flag("OWL_LAUNCH_SYNC"),
                launch_time: flag("OWL_LAUNCH_TIME"),
                d2h_prof: flag("OWL_D2H_PROF"),
                gpu_prof: flag("OWL_GPU_PROF"),
                free_legacy: flag("OWL_FREE_LEGACY"),
                graph_flags: match env_reader::str("OWL_GRAPH_FLAGS") {
                    None => GraphInstantiateFlags::default(),
                    Some(raw) => GraphInstantiateFlags::parse(&raw)?,
                },
                capture_slab_mb: env_reader::parse("OWL_CAPTURE_SLAB_MB"),
                nvrtc_include: env_reader::str("CUDA_HOME")
                    .or_else(|| env_reader::str("CUDA_PATH"))
                    .map(|h| PathBuf::from(h).join("include")),
            },
            graph: GraphCfg { no_graph: flag("OWL_NO_GRAPH") },
        })
    }

    /// 缺省装配(测试便捷;= 全缺省值,零 env 读取)
    pub fn test_defaults() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env_reader::EnvGuard;
    use std::sync::{Mutex, MutexGuard};

    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn default_全缺省零env() {
        let c = OwlConfig::test_defaults();
        assert_eq!(c.runtime.device, 0);
        assert_eq!(c.model.kind, ModelKind::Qwen35_08b);
        assert!(c.pool.prefix_cache);
        assert_eq!(c.sampling.mode, SamplerMode::Sampling);
        assert_eq!(c.sampling.rep_penalty, 1.15);
        assert!(!c.cuda.graph_flags.upload);
        assert!(!c.graph.no_graph);
    }

    #[test]
    fn 枚举严格解析_非法fail_fast() {
        let _g = env_lock();
        let _k = EnvGuard::set("OWL_MODEL_KIND", "awq27bb");
        assert!(OwlConfig::from_env().is_err(), "档位打错字必须拒启");
        let _k2 = EnvGuard::set("OWL_SAMPLER", "greedy1");
        let _k3 = EnvGuard::set("OWL_MODEL_KIND", "0.8b");
        assert!(OwlConfig::from_env().is_err(), "采样模式和稀泥必须拒启");
    }

    #[test]
    fn 枚举合法值与旗标位面() {
        let _g = env_lock();
        let _k = EnvGuard::set("OWL_MODEL_KIND", "awq27b");
        let _k2 = EnvGuard::set("OWL_SAMPLER", "greedy");
        let _k3 = EnvGuard::set("OWL_GRAPH_FLAGS", "6");
        let _k4 = EnvGuard::set("OWL_PREFIX_CACHE", "0");
        let c = OwlConfig::from_env().expect("合法值应通过");
        assert_eq!(c.model.kind, ModelKind::Awq27b);
        assert_eq!(c.sampling.mode, SamplerMode::Greedy);
        assert_eq!(c.cuda.graph_flags.to_bits(), 0b110);
        assert!(!c.pool.prefix_cache);
    }

    #[test]
    fn prefix_cache_布尔和稀泥值拒收() {
        let _g = env_lock();
        let _k = EnvGuard::set("OWL_PREFIX_CACHE", "off");
        assert!(OwlConfig::from_env().is_ok());
        let _k2 = EnvGuard::set("OWL_PREFIX_CACHE", "nope");
        assert!(OwlConfig::from_env().is_err());
    }
}
