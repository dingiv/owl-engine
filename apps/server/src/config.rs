//! 服务配置(boot 边界一次解析;进程内冻结)。
//!
//! **显式依赖律终点 + 配置文件面(2026-10-10)**:全量配置统一定义在
//! `owl_shared::config`(总账 + 强类型 modules + 四层叠加 loader:
//! Default → Flavor → 用户文件 → CLI);本模块只做 server app 面的
//! 组装 —— env 彻底退出正式配置链(缩编裁决),档位分派与引擎旋钮
//! 全部从配置对象搬运。枚举非法/账外键 fail-fast 于 boot。

use std::path::{Path, PathBuf};

use owl_shared::config::OwlConfig;
pub use owl_shared::config::ModelKind;

// ── app 面缺省(总账外的 server 组装缺省;R2 命名常量)──
const DEFAULT_MODEL_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/models/assets/Qwen3.5-0.8B"
);
const DEFAULT_MODEL_NAME: &str = "qwen3.5-0.8b";

/// CLI 启动参数(唯一参数面;手写解析零 clap —— 启动参数爆炸的反面)
#[derive(Clone, Debug, Default)]
pub struct CliArgs {
    /// 位置参数:配置文件路径(OWL_CONFIG env 兜底;都不给 = 纯 flavor/缺省)
    pub config: Option<PathBuf>,
    /// --flavor <NAME>:内置命名预设
    pub flavor: Option<String>,
    /// --list-flavors:列出可用预设后退出
    pub list_flavors: bool,
    /// --perf-dump <PATH>:每 turn 终了覆写性能账(CSV + 同名 .json;
    /// 见 docs/arch/perf-framework.md §三)
    pub perf_dump: Option<PathBuf>,
    /// --perf-level <off|counters|host|gpu>:覆盖 [probes] perf_level
    pub perf_level: Option<String>,
    /// --log-level <off|error|warn|info|debug|trace>:覆盖 [log] level
    pub log_level: Option<String>,
    /// --log-file <PATH>:覆盖 [log] file(tee 落盘)
    pub log_file: Option<PathBuf>,
    /// --step-gate <off|sticky|single>:单步/粘滞调试门档位(需 `--features
    /// stepgate` 构建;见 docs/arch/perf-framework.md §五)
    pub step_gate: Option<String>,
    /// --step-gate-ms <N>:粘滞档每检查点睡眠毫秒
    pub step_gate_ms: Option<u64>,
    /// --step-gate-us <N>:粘滞间隔(µs;覆盖 --step-gate-ms)
    pub step_gate_us: Option<u64>,
    /// --step-gate-every <N>:每 N 个检查点停一次
    pub step_gate_every: Option<u64>,
}

impl CliArgs {
    /// 手写解析(未知参数 = Err 拒启,不静默忽略 —— 与配置同纪律)
    pub fn parse<I: Iterator<Item = String>>(mut it: I) -> Result<Self, String> {
        let mut out = Self::default();
        // config 路径兜底:OWL_CONFIG(隐藏键;env 退出正式链后的唯一遗留)
        if let Ok(path) = std::env::var("OWL_CONFIG") {
            out.config = Some(PathBuf::from(path));
        }
        while let Some(a) = it.next() {
            match a.as_str() {
                "--flavor" => {
                    out.flavor = Some(
                        it.next().ok_or("--flavor 需要值(--list-flavors 查可用档)")?,
                    );
                }
                "--list-flavors" => out.list_flavors = true,
                "--perf-dump" => {
                    out.perf_dump = Some(PathBuf::from(
                        it.next().ok_or("--perf-dump 需要路径(CSV 基名;同名 .json 一并写)")?,
                    ));
                }
                "--perf-level" => {
                    let v = it.next().ok_or("--perf-level 需要值(off|counters|host|gpu)")?;
                    if owl_shared::metrics::level::parse(&v).is_none() {
                        return Err(format!("--perf-level 非法值 {v:?}(off|counters|host|gpu)"));
                    }
                    out.perf_level = Some(v);
                }
                "--log-level" => {
                    let v = it.next().ok_or("--log-level 需要值(off|error|warn|info|debug|trace)")?;
                    if owl_shared::log::Level::parse_threshold(&v).is_none() {
                        return Err(format!(
                            "--log-level 非法值 {v:?}(off|error|warn|info|debug|trace)"
                        ));
                    }
                    out.log_level = Some(v);
                }
                "--log-file" => {
                    out.log_file = Some(PathBuf::from(
                        it.next().ok_or("--log-file 需要路径(tee:stderr + 文件)")?,
                    ));
                }
                "--step-gate" => {
                    let v = it.next().ok_or("--step-gate 需要值(off|sticky|single)")?;
                    if owl_shared::stepgate::StepMode::parse(&v).is_none() {
                        return Err(format!("--step-gate 非法值 {v:?}(off|sticky|single)"));
                    }
                    out.step_gate = Some(v);
                }
                "--step-gate-ms" => {
                    let v = it.next().ok_or("--step-gate-ms 需要毫秒数")?;
                    out.step_gate_ms = Some(
                        v.parse().map_err(|_| format!("--step-gate-ms 非法值 {v:?}"))?,
                    );
                }
                "--step-gate-us" => {
                    let v = it.next().ok_or("--step-gate-us 需要微秒数")?;
                    out.step_gate_us = Some(
                        v.parse().map_err(|_| format!("--step-gate-us 非法值 {v:?}"))?,
                    );
                }
                "--step-gate-every" => {
                    let v = it.next().ok_or("--step-gate-every 需要正整数")?;
                    out.step_gate_every = Some(
                        v.parse().map_err(|_| format!("--step-gate-every 非法值 {v:?}"))?,
                    );
                }
                _ if a.starts_with('-') => {
                    return Err(format!(
                        "未知参数 {a:?}(支持:[CONFIG] --flavor <N> --perf-dump <PATH> --perf-level <L> --log-level <L> --log-file <P> --list-flavors)"
                    ));
                }
                _ => {
                    if out.config.is_some() {
                        return Err(format!("配置文件路径重复:{a:?}"));
                    }
                    out.config = Some(PathBuf::from(a));
                }
            }
        }
        Ok(out)
    }
}

/// 服务配置(唯一构造口 [`ServerConfig::load`];boot 后只读)
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// 监听地址(OWl_BIND 遗留键已废;runtime.bind)
    pub bind: String,
    /// 模型目录(model.dir;构造期已回退 workspace 资产缺省)
    pub model_dir: PathBuf,
    /// 模型档位(model.kind;非法值 loader 拒启)
    pub model_kind: ModelKind,
    /// 对外模型名(model.name;构造期已回退服务端缺省)
    pub model_name: String,
    /// 引擎旋钮(全量配置 engine/cuda/sampling module 的显式搬运)
    pub knobs: owl_engine::EngineKnobs,
    /// 已选设备 ordinal(声明式选卡/自动挑空闲后的结果;交引擎 cudaSetDevice)
    pub device: usize,
    /// 已选设备 UUID(卡锁 + 日志身份)
    pub device_uuid: String,
    /// attention 窗上限(runtime.max_seq)
    pub max_seq: usize,
    /// 预填 chunk(runtime.prefill_chunk)
    pub prefill_chunk: usize,
    /// lm_head 头部量化方案(model.head_quant;C1 2026-10-11;枚举严格)
    pub head_quant: owl_shared::config::HeadQuant,
    /// **W4A8 档位**(model.w4a8;S2 2026-10-17;默认 off = 今天行为)
    pub w4a8: bool,
    /// 最大并发请求(server.max_concurrency;默认 8;须 ≤ knobs.gdn_slots)
    pub max_concurrency: usize,
    /// 性能账落盘基名(--perf-dump;None = 不落盘)
    pub perf_dump: Option<PathBuf>,
    /// 性能分级(probes.perf_level,可被 --perf-level 覆盖)
    pub perf_level: u8,
    /// 日志面([log] module;CLI 可覆盖 level/file)
    pub log: owl_shared::config::LogCfg,
    /// agent 服务面([agent] module;模板覆盖/隐式会话/工具解析/档位)
    pub agent: owl_shared::config::AgentCfg,
    /// 单步/粘滞调试门(probes.step_gate*;CLI 可覆盖;见 perf-framework.md §五)
    pub step_gate: owl_shared::stepgate::StepMode,
    pub step_gate_ms: u64,
    /// 亚毫秒档(µs;>0 覆盖 ms)
    pub step_gate_us: u64,
    pub step_gate_every: u64,
}

impl ServerConfig {
    /// boot 边界唯一入口:四层叠加加载 → app 面组装。
    /// `cli` 携带 --perf-dump / --perf-level 两个 app 面覆盖。
    pub fn load(flavor: Option<&str>, file: Option<&Path>, cli: &CliArgs) -> Result<Self, String> {
        // 预设档(flavor)= 预先写好的配置文件:入口把名字解析成路径,配置层只认文件
        let base = match flavor {
            Some(name) => Some(owl_shared::assets::preset_path(name)?),
            None => None,
        };
        let cfg = OwlConfig::load(base.as_deref(), file)?;
        // 声明式选卡(2026-10-16):规格 → 设备表过滤 → 剔 gpu_lock 已占 → 取首张
        // (空规格 = 全部候选 ⇒ 自动挑空闲卡);命中全列日志,无候选/全被占 = 响亮拒启
        let (device, device_uuid) = resolve_device(&cfg.runtime)?;
        let mut out = Self::from_config(cfg);
        out.device = device;
        out.device_uuid = device_uuid;
        if let Some(l) = cli.perf_level.as_deref() {
            out.perf_level =
                owl_shared::metrics::level::parse(l).ok_or("perf-level 非法")?;
        }
        out.perf_dump = cli.perf_dump.clone();
        if let Some(l) = cli.log_level.as_deref() {
            out.log.level = match owl_shared::log::Level::parse_threshold(l) {
                Some(v) if v == owl_shared::log::LVL_OFF => owl_shared::config::LogLevel::Off,
                Some(v) if v == owl_shared::log::LVL_ERROR => owl_shared::config::LogLevel::Error,
                Some(v) if v == owl_shared::log::LVL_WARN => owl_shared::config::LogLevel::Warn,
                Some(v) if v == owl_shared::log::LVL_DEBUG => owl_shared::config::LogLevel::Debug,
                Some(v) if v == owl_shared::log::LVL_TRACE => owl_shared::config::LogLevel::Trace,
                Some(_) => owl_shared::config::LogLevel::Info,
                None => return Err("--log-level 非法".into()),
            };
        }
        if let Some(f) = cli.log_file.clone() {
            out.log.file = Some(f);
        }
        // 调试门:CLI 覆盖配置面(档位/间隔/步长;非法档位在 CLI 解析期已拒)
        if let Some(m) = cli.step_gate.as_deref() {
            out.step_gate = owl_shared::stepgate::StepMode::parse(m)
                .ok_or_else(|| format!("--step-gate 非法值 {m:?}"))?;
        }
        if let Some(v) = cli.step_gate_ms {
            out.step_gate_ms = v;
        }
        if let Some(v) = cli.step_gate_us {
            out.step_gate_us = v;
        }
        if let Some(v) = cli.step_gate_every {
            out.step_gate_every = v;
        }
        Ok(out)
    }

    /// app 面组装(已加载配置 → server 形态;测试可直接喂 default/定制)
    pub fn from_config(cfg: OwlConfig) -> Self {
        Self {
            bind: cfg.runtime.bind.clone(),
            model_dir: cfg
                .model
                .dir
                .clone()
                .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR)),
            model_kind: cfg.model.kind,
            model_name: cfg
                .model
                .name
                .clone()
                .unwrap_or_else(|| DEFAULT_MODEL_NAME.into()),
            knobs: owl_engine::EngineKnobs::from_config(&cfg),
            // 占位:真值由 `load()` 的声明式选卡回填(设备表 + 卡锁都在那边)
            device: 0,
            device_uuid: String::new(),
            max_seq: cfg.runtime.max_seq,
            prefill_chunk: cfg.runtime.prefill_chunk,
            head_quant: cfg.model.head_quant,
            w4a8: cfg.model.w4a8,
            max_concurrency: cfg.server.max_concurrency,
            perf_dump: None,
            perf_level: cfg.probes.perf_level.to_u8(),
            log: cfg.log,
            agent: cfg.agent,
            step_gate: cfg.probes.step_gate,
            step_gate_ms: cfg.probes.step_gate_ms,
            step_gate_us: cfg.probes.step_gate_us,
            step_gate_every: cfg.probes.step_gate_every,
        }
    }

    /// agent 面缺省模板变量(`[agent]` → jinja 变量;请求 kwargs 覆盖之)。
    /// **只装非缺省项** ⇒ 缺省档与 HF oracle(无 kwargs)逐字节同源。
    pub fn agent_template_defaults(&self) -> serde_json::Map<String, serde_json::Value> {
        use owl_shared::config::ToolCallFormat;
        let mut m = serde_json::Map::new();
        if self.agent.tool_call_format != ToolCallFormat::Xml {
            m.insert("tool_call_format".into(), serde_json::json!("json"));
        }
        if self.agent.auto_disable_thinking_with_tools {
            m.insert("auto_disable_thinking_with_tools".into(), serde_json::json!(true));
        }
        if let Some(e) = &self.agent.reasoning_effort {
            m.insert("reasoning_effort".into(), serde_json::json!(e));
        }
        m
    }
}

/// nvidia-smi 实占(按 UUID → 空闲占比)。None = 取不到(命令失败/不可用)
fn vram_free_ratio_map() -> Option<std::collections::HashMap<String, f64>> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=uuid,memory.free,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let map = owl_shared::device::parse_nvidia_smi_mem(&String::from_utf8_lossy(&out.stdout));
    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

/// **声明式选卡**(2026-10-16):见 `owl_shared::device`。
/// - `device_ordinal` 显式给 ⇒ 直通(测试/特殊场景;仍查 UUID 供锁用);
/// - 否则按 `device` 规格从设备表里选(**空规格 = 全部候选**),
///   再依次用 ①`gpu_lock`(owl 实例)②nvidia-smi 实占内存(阈值 `runtime.gpu_min_free_ratio`)
///   剔掉不可用的卡,取**首张**;
/// - 规格未命中任何设备 / 候选全被占 ⇒ 响亮 Err(带设备表或持有者),不静默降级。
fn resolve_device(rt: &owl_shared::config::RuntimeCfg) -> Result<(usize, String), String> {
    if let Some(o) = rt.device_ordinal {
        let uuid = owl_cuda::ffi::device_uuid(o)
            .map(|b| owl_shared::device::uuid_display(&b))
            .map_err(|e| format!("device_ordinal={o} 的 UUID 查不到: {e:?}"))?;
        // 此刻 logger 尚未初始化(log 被丢弃)⇒ 裸 eprintln 保可见(仓规:boot 关键行恒输出)
        eprintln!("[boot] 选卡(显式 ordinal):{o} → {uuid}");
        return Ok((o, uuid));
    }
    let rows = owl_cuda::ffi::device_rows()?;
    let hit = rt.device.select(&rows);
    if hit.is_empty() {
        let table: Vec<String> = rows
            .iter()
            .map(|r| format!("{}[{}|{}]", r.ordinal, r.bdf, r.name))
            .collect();
        return Err(format!("device 规格未命中任何设备(spec={:?};设备表={table:?})", rt.device));
    }
    let live = owl_shared::gpu_lock::live_locks();
    let free: Vec<&owl_shared::device::DeviceRow> = hit
        .iter()
        .copied()
        .filter(|r| !live.iter().any(|l| l.uuid == r.uuid))
        .collect();
    // ② 实占内存过滤(锁只认 owl 实例;别的进程/未持锁的旧实例也要避让)
    let (free, mem_ok) = match vram_free_ratio_map() {
        Some(m) => {
            let mut keep = Vec::new();
            let mut skipped: Vec<String> = Vec::new();
            for r in free {
                match m.get(&r.uuid) {
                    Some(ratio) if *ratio < rt.gpu_min_free_ratio => {
                        skipped.push(format!("{} {} free={:.0}%", r.uuid, r.bdf, ratio * 100.0));
                    }
                    _ => keep.push(r),
                }
            }
            if !skipped.is_empty() {
                eprintln!(
                    "[boot] 选卡:按实占内存跳过 {} 台(阈值 {:.0}%):{skipped:?}",
                    skipped.len(),
                    rt.gpu_min_free_ratio * 100.0
                );
            }
            (keep, true)
        }
        // nvidia-smi 不可用 ⇒ 不阻断启动(锁过滤已生效),但响亮提示
        None => {
            eprintln!("[boot] 选卡:nvidia-smi 内存查询不可用 ⇒ 跳过实占内存过滤(只按卡锁避让)");
            (free, false)
        }
    };
    let _ = mem_ok;
    let picked = free.first().copied().ok_or_else(|| {
        let holders: Vec<String> = live
            .iter()
            .map(|l| format!("{} pid={} port={}", l.uuid, l.pid, l.port))
            .collect();
        format!("候选设备全部被占(spec={:?};持有者={holders:?})", rt.device)
    })?;
    eprintln!(
        "[boot] 选卡:命中 {} 台(spec={:?})→ 取 ordinal {} [{}|{}] {};空闲候选 {} 台(logger 前 ⇒ 裸输出)",
        hit.len(),
        rt.device,
        picked.ordinal,
        picked.bdf,
        picked.name,
        picked.uuid,
        free.len()
    );
    Ok((picked.ordinal, picked.uuid.clone()))
}
