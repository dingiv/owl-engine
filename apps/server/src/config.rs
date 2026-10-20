//! 服务配置(boot 边界一次解析;进程内冻结)。
//!
//! **显式依赖律终点 + 配置文件面(2026-10-10)**:全量配置统一定义在
//! `owl_shared::config`(总账 + 强类型 modules + 四层叠加 loader:
//! Default → Flavor → 用户文件 → CLI);本模块只做 server app 面的
//! 组装 —— env 彻底退出正式配置链(缩编裁决),档位分派与引擎旋钮
//! 全部从配置对象搬运。枚举非法/账外键 fail-fast 于 boot。

use std::path::{Path, PathBuf};

use owl_shared::config::{OwlConfig, ResolvedConfig};
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
    /// 位置参数:可选覆盖文件(OWL_CONFIG env 兜底;都不给 = 纯缺省)
    pub config: Option<PathBuf>,
    /// -p/--profile <PATH>:配置档**文件路径**(源码 `profiles/` 复制修改)
    pub profile: Option<PathBuf>,
    /// --site <PATH>:站点层(路径/卡位/端口;`[site]` 或普通配置表)
    pub site: Option<PathBuf>,
    /// --help:打印用法后退出
    pub help: bool,
    /// --check:预检(不加载权重;打印意图派生/内存账/文件校验)
    pub check: bool,
    /// --print-config:打印生效配置(TOML,或 `--format json` 含来源/估算)
    pub print_config: bool,
    /// --format <toml|json>(`--print-config` 用;缺省 toml)
    pub format: Option<String>,
    /// --explain <KEY>:解释单键(点路径;可重复)
    pub explain: Vec<String>,
    /// --set <KEY=VALUE>:临时覆盖(点路径;可重复;替代启动脚本里的 extends 拼档)
    pub sets: Vec<(String, toml::Value)>,
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
                    return Err(
                        "--flavor 已移除:配置档统一为 `-p/--profile <profile_path>`\
                         (源码 profiles/ 下可直接复制修改)"
                            .into(),
                    );
                }
                "--profile" | "-p" => {
                    out.profile = Some(PathBuf::from(
                        it.next().ok_or("-p/--profile 需要配置文件路径")?,
                    ));
                }
                "--site" => {
                    out.site = Some(PathBuf::from(
                        it.next().ok_or("--site 需要路径(站点档:[site] 或普通配置表)")?,
                    ));
                }
                "--list-flavors" | "--list-profiles" => {
                    return Err(
                        "--list-flavors/--list-profiles 已移除:配置层不再做目录名查找;\
                         直接从源码 profiles/ 复制文件,启动用 -p <profile_path>"
                            .into(),
                    );
                }
                "--check" => out.check = true,
                "--print-config" => out.print_config = true,
                "--format" => {
                    let v = it.next().ok_or("--format 需要值(toml|json)")?;
                    if !matches!(v.as_str(), "toml" | "json") {
                        return Err(format!("--format 非法值 {v:?}(toml|json)"));
                    }
                    out.format = Some(v);
                }
                "--explain" => {
                    out.explain.push(
                        it.next().ok_or("--explain 需要键(点路径,如 pool.pool_tokens)")?,
                    );
                }
                "--set" => {
                    let raw = it.next().ok_or("--set 需要 key=value(点路径)")?;
                    out.sets.push(owl_shared::config::parse_cli_set(&raw)?);
                }
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
                "--help" | "-h" => out.help = true,
                _ if a.starts_with('-') => {
                    return Err(format!(
                        "未知参数 {a:?}\n\n{HELP}"
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

    /// **CLI 覆盖层**(与 `--set` 同一条合并链;来源 = `cli`)
    ///
    /// 日志/性能/调试门旗标也走这里 ⇒ `--print-config` 看到的就是 boot 用的
    /// 同一份配置(不再有 "print 之后 ServerConfig 再改一遍" 的双源)。
    pub fn overrides(&self) -> Vec<(String, toml::Value)> {
        let mut v = self.sets.clone();
        let mut push = |k: &str, val: toml::Value| v.push((k.to_string(), val));
        if let Some(x) = &self.log_level {
            push("log.level", toml::Value::String(x.clone()));
        }
        if let Some(x) = &self.log_file {
            push(
                "log.file",
                toml::Value::String(x.display().to_string()),
            );
        }
        if let Some(x) = &self.perf_level {
            push("probes.perf_level", toml::Value::String(x.clone()));
        }
        if let Some(x) = &self.step_gate {
            push("probes.step_gate", toml::Value::String(x.clone()));
        }
        if let Some(x) = self.step_gate_ms {
            push("probes.step_gate_ms", toml::Value::Integer(x as i64));
        }
        if let Some(x) = self.step_gate_us {
            push("probes.step_gate_us", toml::Value::Integer(x as i64));
        }
        if let Some(x) = self.step_gate_every {
            push("probes.step_gate_every", toml::Value::Integer(x as i64));
        }
        v
    }
}

/// 使用说明(打印到 stderr;`--help` / 未知参数)
pub(crate) const HELP: &str = "用法: server -p <profile_path> [CONFIG] [选项]\n\
  -p, --profile <PATH>   配置档文件路径(源码 profiles/ 下可复制修改)\n\
  --site <PATH>          可选站点覆盖层(路径/卡位/端口)\n\
  --check                预检(不加载权重;打印派生/内存账/文件校验)\n\
  --print-config         打印生效配置(缺省 TOML;--format json 含来源/估算)\n\
  --format <toml|json>   --print-config 的输出格式\n\
  --explain <KEY>        解释单键(点路径;可重复)\n\
  --set <KEY=VALUE>      临时覆盖(点路径;可重复)\n\
  --log-level/-file      日志覆盖;--perf-level/--perf-dump 性能账\n\
  --step-gate[-ms|-us|-every]  单步/粘滞调试门\n";

/// **本进程持有的设备束**(boot 边界定形;rank 序 = 列表序)。
///
/// 单卡档 world = 1(声明式选卡/自动挑空闲的结果);TP 档 world = 2
/// (`runtime.tp_ordinals`;S6 2026-10-17)。**唯一真源** —— 引擎 ordinal、卡锁、
/// 日志身份全从这里取,单卡/TP 两条路径不各存一份。
#[derive(Clone, Debug)]
pub struct Devices {
    /// 引擎 ordinal(rank 序 = 列表序;交引擎 cudaSetDevice)
    ordinals: Vec<usize>,
    /// 设备 UUID(卡锁 + 日志身份;与 `ordinals` 同长同序)
    uuids: Vec<String>,
}

impl Devices {
    /// 构造即校验(空/长度不齐 = 拒;字段私有 ⇒ 这是唯一入口)
    fn new(ordinals: Vec<usize>, uuids: Vec<String>) -> Result<Self, String> {
        if ordinals.is_empty() {
            return Err("设备束为空(至少要一张卡)".into());
        }
        if ordinals.len() != uuids.len() {
            return Err(format!(
                "设备束长度不齐:ordinals={} uuids={}(内部不变量)",
                ordinals.len(),
                uuids.len()
            ));
        }
        Ok(Self { ordinals, uuids })
    }

    /// `from_config` 用的占位(world = 1、ordinal 0、UUID 空)—— 真值由
    /// [`ServerConfig::load`] 的选卡回填;喂 `from_config` 的测试不碰设备面。
    fn placeholder() -> Self {
        Self { ordinals: vec![0], uuids: vec![String::new()] }
    }

    /// rank 数(单卡 = 1;TP v1 恒 2)
    pub fn world(&self) -> usize {
        self.ordinals.len()
    }
    /// 全部 rank 的 ordinal(rank 序 = 列表序)
    pub fn ordinals(&self) -> &[usize] {
        &self.ordinals
    }
    /// 全部 rank 的 UUID(与 [`Self::ordinals`] 同长同序)
    pub fn uuids(&self) -> &[String] {
        &self.uuids
    }
    /// rank0(单卡档即唯一那张)的引擎 ordinal
    pub fn device(&self) -> usize {
        self.ordinals[0]
    }
}

/// 服务配置(构造口 = [`ServerConfig::load_resolved`] + [`ServerConfig::from_resolved`];boot 后只读)
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
    /// **本进程持有的设备束**(单卡 world = 1 / TP world = 2;rank 序 = 列表序)。
    /// 单卡档 = 声明式选卡/自动挑空闲的结果;TP 档 = `runtime.tp_ordinals`。
    pub devices: Devices,
    /// attention 窗上限(runtime.max_seq)
    pub max_seq: usize,
    /// 预填 chunk(runtime.prefill_chunk)
    pub prefill_chunk: usize,
    /// lm_head 头部量化方案(model.head_quant;C1 2026-10-11;枚举严格)
    pub head_quant: owl_shared::config::HeadQuant,
    /// **主干量化方案**(model.quant;S2 2026-10-17;枚举严格 —— 原
    /// `w4a8: bool` 已退役,旧键响亮拒)
    pub quant: owl_shared::config::QuantScheme,
    /// 最大并发请求(server.max_concurrency;默认 8;须 ≤ knobs.gdn_slots)
    pub max_concurrency: usize,
    /// **合批上限**([runtime] max_batch;1 = 串行现状)。服务面用它做两件事:
    /// 会话闸的配套告警(隐式会话 + 合批 ⇒ 串行)与 `/debug/vram` 的如实报数。
    pub max_batch: usize,
    /// **TP 通信数据腿:push 档**(runtime.tp_push;2026-10-18 §36.6)⇒ 交 `TpOpts`
    pub tp_push: bool,
    /// **TP AR 载荷 fp8 档**（runtime.tp_ar_f8；§47 第二步）⇒ 交 `TpOpts.ar_f8`
    pub tp_ar_f8: bool,
    /// **lm_head 词表切**（runtime.tp_vocab_split；P1.1）⇒ 装载面
    /// `ModelSpec::head_split`（每 rank 只持半表头；logits 由引擎 host 拼接）
    pub tp_vocab_split: bool,
    /// **v1 两核融合档**（runtime.tp_ar_fused；§11.2）⇒ 交 `TpOpts.ar_fused`
    pub tp_ar_fused: bool,
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
    /// **只解析配置 + 意图派生 + 预检元数据**(不选卡、不碰 GPU;`--check` 用)
    ///
    /// 层序:`-p/--profile` 路径 → 位置参数 CONFIG → `--site` → CLI
    /// (`--set` / 日志/性能/调试门)。`[intent]` 在同一函数里展开成低层键。
    pub fn load_resolved(args: &CliArgs) -> Result<ResolvedConfig, String> {
        let base = args.profile.as_deref();
        let mut layers: Vec<&Path> = Vec::new();
        if let Some(p) = args.config.as_deref() {
            layers.push(p);
        }
        if let Some(p) = args.site.as_deref() {
            layers.push(p);
        }
        let mut resolved =
            OwlConfig::load_resolved(base.as_deref(), &layers, &args.overrides())?;
        if let Some(m) = resolved.meta.as_mut() {
            m.intent.profile = args
                .profile
                .as_ref()
                .map(|p| p.display().to_string());
        }
        Ok(resolved)
    }

    /// 完整解析产物 → server 形态(**选卡在这里**;`--check` 不走)。
    pub fn from_resolved(
        resolved: ResolvedConfig,
        args: &CliArgs,
    ) -> Result<Self, String> {
        // 选卡(S6 2026-10-17 起两条规范:单卡声明式 / TP 显式双卡;互斥):
        //   单卡 = 声明式选卡(2026-10-16):规格 → 设备表过滤 → 剔 gpu_lock 已占
        //          → 取首张(空规格 = 全部候选 ⇒ 自动挑空闲卡);
        //   TP   = `runtime.tp_ordinals` 显式双卡(零自动挑卡,见 resolve_tp_devices);
        //   命中全列日志,无候选/全被占/口径混填 = 响亮拒启。
        let devices = resolve_devices(&resolved.config.runtime)?;
        let mut out = Self::from_config(resolved.config);
        out.devices = devices;
        out.perf_dump = args.perf_dump.clone();
        Ok(out)
    }

    /// **对外模型名的缺省**(唯一产地;`model.name` 显式给则用给的)。
    ///
    /// 为什么 GGUF 档要回落到**文件名**:GGUF 是**单文件档**,一个目录里可能并排放
    /// 着同一模型的多个量化档(`…-Q4_K_M.gguf` / `…-Q8_0.gguf`)—— 用固定缺省名
    /// (`qwen3.5-0.8b`)会让 `/v1/models`、`/health`、日志三处都显示别人的名字
    /// (2026-10-19 实测:GGUF 档起机报 `model=qwen3.5-0.8b`)。词干就是实例身份
    /// (`Qwen_Qwen3.5-4B-Q4_K_M`),与"声明即事实"同律。
    ///
    /// 其余档位保持历史缺省(改它会动既有档位的对外面,不在本轮射程)。
    fn default_model_name(kind: ModelKind, dir: Option<&std::path::Path>) -> String {
        if kind == ModelKind::Gguf {
            if let Some(stem) = dir.and_then(|d| d.file_stem()).and_then(|s| s.to_str()) {
                if !stem.is_empty() {
                    return stem.to_string();
                }
            }
        }
        DEFAULT_MODEL_NAME.to_string()
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
                .unwrap_or_else(|| Self::default_model_name(cfg.model.kind, cfg.model.dir.as_deref())),
            knobs: owl_engine::EngineKnobs::from_config(&cfg),
            // 占位:真值由 `load()` 的声明式选卡回填(设备表 + 卡锁都在那边)
            devices: Devices::placeholder(),
            // 上下文/块长:**单一来源 = 配置**(2026-10-20 删除 OWL_MAX_SEQ /
            // OWL_PREFILL_CHUNK —— 它们会在意图派生后再改一次 ⇒ max_seq 与
            // pool_tokens 失配/池夹死的双源;要临时调请 `--set`）。
            max_seq: cfg.runtime.max_seq,
            prefill_chunk: cfg.runtime.prefill_chunk,
            head_quant: cfg.model.head_quant,
            quant: cfg.model.quant,
            tp_push: cfg.runtime.tp_push,
            tp_ar_f8: cfg.runtime.tp_ar_f8,
            tp_vocab_split: cfg.runtime.tp_vocab_split,
            tp_ar_fused: cfg.runtime.tp_ar_fused,
            max_concurrency: cfg.server.max_concurrency,
            max_batch: cfg.runtime.max_batch,
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

/// **选卡总入口**:两条规范互斥 —— `runtime.tp_ordinals` 给值 ⇒ TP 双卡档
/// (零自动挑卡);不给 ⇒ 单卡档(声明式选卡/自动挑空闲)。
fn resolve_devices(rt: &owl_shared::config::RuntimeCfg) -> Result<Devices, String> {
    match &rt.tp_ordinals {
        None => {
            let (o, u) = resolve_device(rt)?;
            Devices::new(vec![o], vec![u])
        }
        Some(ord) => resolve_tp_devices(rt, ord),
    }
}

/// **TP 档选卡**(S6 2026-10-17;`runtime.tp_ordinals`)。
///
/// 与单卡档的差别 = **不做自动挑卡**:分片协议要的是"确定的双卡 + 确定的
/// rank 序",而自动挑卡结果随当下占用漂移(同 uuid 会撞 `TpEngine::assemble`
/// 的同卡护栏;挑到混插对会被 `PeerPlan::plan` 拒)。故此处:
/// - 必须正好 2 项(v1 world = 2 边界),且互不相同;
/// - 与 `device` / `device_ordinal` 互斥(同时给 = 拒启,不静默取一方 ——
///   否则卡身份有两个真源);
/// - 每项查 UUID 作身份(卡锁 + 日志);查不到 ⇒ 响亮 Err。
fn resolve_tp_devices(
    rt: &owl_shared::config::RuntimeCfg,
    ord: &[usize],
) -> Result<Devices, String> {
    if !rt.device.is_empty() {
        return Err(format!(
            "选卡口径混填:`runtime.tp_ordinals = {ord:?}` 与 `runtime.device = {:?}` 互斥(TP 档不自动挑卡)",
            rt.device
        ));
    }
    if let Some(o) = rt.device_ordinal {
        return Err(format!(
            "选卡口径混填:`runtime.tp_ordinals = {ord:?}` 与 `runtime.device_ordinal = {o}` 互斥"
        ));
    }
    if ord.len() != 2 {
        return Err(format!("TP v1 只支持 world = 2(tp_ordinals 得 {} 项 {ord:?})", ord.len()));
    }
    if ord[0] == ord[1] {
        return Err(format!(
            "TP 两个 rank 指向同一张卡(ordinal {};双 rank 协议要求两张不同的卡)",
            ord[0]
        ));
    }
    let mut uuids = Vec::with_capacity(ord.len());
    for &o in ord {
        let uuid = owl_cuda::ffi::device_uuid(o)
            .map(|b| owl_shared::device::uuid_display(&b))
            .map_err(|e| format!("TP rank 的 device_ordinal={o} 的 UUID 查不到: {e:?}"))?;
        uuids.push(uuid);
    }
    // 此刻 logger 尚未初始化(log 被丢弃)⇒ 裸 eprintln 保可见(仓规:boot 关键行恒输出)
    eprintln!(
        "[boot] 选卡(TP 档):rank0 = ordinal {} → {};rank1 = ordinal {} → {}(rank 序 = 列表序;logger 前 ⇒ 裸输出)",
        ord[0], uuids[0], ord[1], uuids[1]
    );
    Devices::new(ord.to_vec(), uuids)
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


// ============================================================================
// 预检 / 解释面(`--check` / `--print-config` / `--explain`;不加载权重)
// ============================================================================

fn gib(bytes: u64) -> String {
    format!("{:.2}G", bytes as f64 / (1u64 << 30) as f64)
}

/// boot 一行摘要(意图档;设计卷 §9 的最小版)
///
/// 位置:配置解析后、选卡/装权重前(裸 `eprintln`,logger 尚未初始化)。
/// 详细依据用 `--check` / `--print-config`;这里只给"看一眼就知道档对不对"。
pub fn summary_line(r: &ResolvedConfig) -> Option<String> {
    let m = r.intent()?;
    let c = &r.config;
    let e = r.estimate()?;
    let arms = format!(
        "prefill={} decode={} gdn={} bucket={:?}",
        if c.graph.prefill_bucket == owl_shared::config::PrefillBucket::Off {
            "eager"
        } else {
            "FI"
        },
        if c.dispatch.decode_fi {
            "FI(single)"
        } else if c.dispatch.decode_xqa {
            "XQA"
        } else {
            "eager"
        },
        if c.dispatch.gdn_chunked { "chunked" } else { "scalar" },
        c.graph.prefill_bucket,
    );
    Some(format!(
        "[owl] profile={} card={} ctx={} pool={}({}) {} mem={}/{} (余 {}) reuse={} spec={} c={}",
        m.profile.as_deref().unwrap_or("(inline)"),
        m.intent.hardware.as_deref().unwrap_or("auto"),
        c.runtime.max_seq,
        c.pool.pool_tokens.unwrap_or(0),
        gib(e.pool_bytes),
        arms,
        gib(e.used_bytes()),
        gib(e.total_bytes),
        gib(e.total_bytes.saturating_sub(e.used_bytes())),
        if c.pool.prefix_cache { format!("snap{}", c.pool.snap_max) } else { "off".into() },
        if c.spec.depth > 0 { format!("dflash{}", c.spec.depth) } else { "off".into() },
        c.server.max_concurrency,
    ))
}

/// 预检报告(`--check`):文本 + 是否通过
pub struct CheckReport {
    pub text: String,
    pub ok: bool,
}

/// `--check`:意图派生 + 内存账 + 模型文件几何校验(不加载权重、不选卡)
///
/// 文件几何错误**收集进报告**(不中途返回)⇒ 一次看到全部问题;
/// 解析/注册表级错误(意图非法等)仍直接 `Err`(没有报告可言)。
pub fn check_report(r: &ResolvedConfig) -> Result<CheckReport, String> {
    let mut out = String::new();
    let mut ok = true;
    let c = &r.config;
    out.push_str(&format!(
        "[check] layers: {}\n",
        if r.layers.is_empty() { "(纯缺省)".to_string() } else { r.layers.join(" → ") }
    ));
    let intent = r.intent();
    if let Some(m) = intent {
        out.push_str(&format!(
            "[check] intent: model={} hardware={} goal={} context={} concurrency={} \
             spec={} kv={} kvmem={} graphs={}\n",
            m.intent.model,
            m.intent.hardware.as_deref().unwrap_or("(推断)"),
            m.intent.goal.name(),
            m.intent.context.name(),
            m.intent.concurrency,
            m.intent.spec.name(),
            m.intent.kv.name(),
            m.intent.kvmem.name(),
            m.intent.graphs.name(),
        ));
        out.push_str(&format!(
            "[check] resolved: max_seq={} pool_tokens={} chunk={} tail={} \
             gdn_slots={} max_batch={} max_concurrency={} kv_fp8={} bucket={:?}\n",
            c.runtime.max_seq,
            c.pool.pool_tokens.unwrap_or(0),
            c.runtime.prefill_chunk,
            c.runtime.prefill_tail_merge,
            c.pool.gdn_slots,
            c.runtime.max_batch,
            c.server.max_concurrency,
            c.dispatch.kv_fp8,
            c.graph.prefill_bucket,
        ));
        if let Some(e) = r.estimate() {
            out.push_str(&format!(
                "[check] mem: weights {} + fixed {} + pool {} + act {} + margin {} = {} / {} (余 {})\n",
                gib(e.weights_bytes),
                gib(e.fixed_bytes),
                gib(e.pool_bytes),
                gib(e.activation_bytes),
                gib(e.stability_margin_bytes),
                gib(e.used_bytes()),
                gib(e.total_bytes),
                gib(e.total_bytes.saturating_sub(e.used_bytes())),
            ));
            out.push_str(&format!(
                "[check] governor: budget {} cap {} tok; pool {} {} cap\n",
                gib(e.budget_bytes),
                e.cap_tokens,
                e.pool_tokens,
                if e.pool_tokens <= e.cap_tokens { "≤" } else { ">" },
            ));
            out.push_str(&format!(
                "[check] basis: per_tok {}B(主 {} + 草稿 {})·激活 {}·固定 = gdn {} + snap {} + slab {} + warmup {}\n",
                e.per_tok,
                e.per_tok_main,
                e.per_tok_draft,
                e.activation_note,
                gib(e.fixed.gdn_grid),
                gib(e.fixed.gdn_snap),
                gib(e.fixed.graph_slab),
                gib(e.fixed.warmup_misc),
            ));
            if e.pool_tokens != c.pool.pool_tokens.unwrap_or(0) {
                out.push_str(
                    "[check] note: 估算池 ≠ 显式 pool_tokens(非超卖档引擎会把池下限抬到上下文)\n",
                );
            }
        }
    } else {
        out.push_str(
            "[check] 无 [intent]:仅做配置解析/语义门 + 模型几何校验(低层档/覆盖路径)。\n\
             [check] 要内存账/意图派生,请用 -p <profile_path> 或写 [intent]。\n",
        );
        out.push_str(&format!(
            "[check] resolved: max_seq={} pool_tokens={:?} chunk={} tail={} \
             gdn_slots={} max_batch={} max_concurrency={} kv_fp8={} bucket={:?}\n",
            c.runtime.max_seq,
            c.pool.pool_tokens,
            c.runtime.prefill_chunk,
            c.runtime.prefill_tail_merge,
            c.pool.gdn_slots,
            c.runtime.max_batch,
            c.server.max_concurrency,
            c.dispatch.kv_fp8,
            c.graph.prefill_bucket,
        ));
    }

    // ── 模型/草稿目录几何校验(目录不存在 ⇒ 告警跳过,CI 假路径可跑) ──
    let profile = match intent {
        Some(m) => Some(owl_shared::config::intent::model_profile(&m.intent.model)?),
        None => owl_shared::config::intent::model_by_kind(c.model.kind),
    };
    // 27B 没有"缺省资产":不给 dir ⇒ 真启动必失败,`--check` 直接拒;
    // 0.8B 有 workspace 缺省资产,走它做几何校验。
    let default_08b = || PathBuf::from(DEFAULT_MODEL_DIR);
    let dir_owned = match &c.model.dir {
        Some(d) => Some(d.clone()),
        None if c.model.kind == ModelKind::Qwen35_08b => Some(default_08b()),
        None => None,
    };
    match (dir_owned.as_deref(), profile) {
        (Some(dir), Some(p)) => match check_model_dir(dir, p, c.runtime.max_seq, c.model.vision) {
            Ok(Some(note)) => out.push_str(&format!("[check] model dir: {note}\n")),
            Ok(None) => {}
            Err(e) => {
                ok = false;
                out.push_str(&format!("[check] FAIL: {e}\n"));
            }
        },
        (None, _) => {
            ok = false;
            out.push_str(
                "[check] FAIL: 未配置 [model].dir / [site].model_dir —— 27B 档无法定位检查点\
                 (真启动必失败);请给站点档\n",
            );
        }
        (Some(_), None) => {
            out.push_str("[check] model dir: 注册表无该 kind ⇒ 几何校验跳过\n")
        }
    }
    if c.spec.depth > 0 {
        match c.model.dflash2_dir.as_deref() {
            Some(dir) => match check_draft_dir(Path::new(dir)) {
                Ok(note) => out.push_str(&format!("[check] draft dir: {note}\n")),
                Err(e) => {
                    ok = false;
                    out.push_str(&format!("[check] FAIL: {e}\n"));
                }
            },
            None => {
                ok = false;
                out.push_str("[check] FAIL: spec.depth > 0 但 model.dflash2_dir 为空\n");
            }
        }
    }

    // ── 组合告警(不拒;`--check` 的 UX 价值就在把隐患提前说出来) ──
    if c.graph.prefill_bucket != owl_shared::config::PrefillBucket::Off
        && c.runtime.prefill_chunk + c.runtime.prefill_tail_merge > 1024
    {
        out.push_str(&format!(
            "[check] WARN: chunk+tail = {}+{} > 1024:尾块没有桶图 ⇒ eager(激活约 1.7G);\
             贴顶档建议拆成 512+512 或 1024+0\n",
            c.runtime.prefill_chunk, c.runtime.prefill_tail_merge
        ));
    }
    if !c.kvmem.pool_overcommit {
        if let Some(p) = c.pool.pool_tokens {
            if p < c.runtime.max_seq {
                out.push_str(&format!(
                    "[check] WARN: pool_tokens={p} < max_seq={}:非超卖档引擎会把池下限抬到单会话上下文;\
                     并发满上下文仍可能 Exhausted\n",
                    c.runtime.max_seq
                ));
            }
        }
    }

    // ── 臂摘要 ──
    let prefill = match c.graph.prefill_bucket {
        owl_shared::config::PrefillBucket::Off => "eager",
        owl_shared::config::PrefillBucket::T512 => "FI(t512)",
        owl_shared::config::PrefillBucket::Geobuckets => "FI(全桶族)",
    };
    let decode = if c.dispatch.decode_fi {
        "FI(单图)"
    } else if c.dispatch.decode_xqa {
        "XQA"
    } else {
        "eager/piecewise"
    };
    out.push_str(&format!(
        "[check] arms: prefill={prefill} decode={decode} gdn={} spec=depth{} reuse={} batch={} vision={}\n",
        if c.dispatch.gdn_chunked { "chunked" } else { "scalar" },
        c.spec.depth,
        if c.pool.prefix_cache { format!("snap{}", c.pool.snap_max) } else { "off".into() },
        c.runtime.max_batch,
        if c.model.vision { "on" } else { "off" },
    ));
    if c.model.vision {
        if c.spec.depth > 0 {
            out.push_str(
                "[check] note: vision=on + spec>0:图像 × 投机已接(V5)—— verify 的 rope \
                 pos = 真实位置 + m-RoPE delta,KV slots/lens 仍用真实位置;\
                 带图为单图/非合批;文本+draft 同档可用\n",
            );
        } else {
            out.push_str(
                "[check] note: vision=on + spec=off:图像路径可用(单图;data URL;\
                 TP2 两 rank 各备塔输出)\n",
            );
        }
    }

    if let Some(m) = intent {
        for w in &m.warnings {
            out.push_str(&format!("[check] WARN: {w}\n"));
        }
    }
    if ok {
        out.push_str("[check] OK(estimated:权重未加载;真机以 [vram-gov] 实测为准)\n");
    } else {
        out.push_str("[check] 未通过:修正上方 FAIL 后重试(其余项已全部检查)\n");
    }
    Ok(CheckReport { text: out, ok })
}

/// 模型 `config.json` 几何校验(读 `text_config`;与注册表逐项比对)
fn check_model_dir(
    dir: &Path,
    profile: &owl_shared::config::intent::ModelProfile,
    ctx: usize,
    vision: bool,
) -> Result<Option<String>, String> {
    let p = dir.join("config.json");
    if !p.is_file() {
        return Ok(Some(format!(
            "{} 不可读(config.json 缺失)—— 几何校验跳过(真启动会失败)",
            dir.display()
        )));
    }
    let raw = owl_shared::file_loader::read_to_string(&p)
        .map_err(|e| format!("读取 {} 失败: {e}", p.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("{} 不是合法 JSON: {e}", p.display()))?;
    if vision {
        if v.get("vision_config").is_none() {
            return Err(format!(
                "[model].vision = true,但 {} 的 config.json 无 vision_config(不是 VLM 检查点)",
                dir.display()
            ));
        }
        if v.get("image_token_id").and_then(|x| x.as_u64()).is_none() {
            return Err(format!(
                "[model].vision = true,但 {} 的 config.json 缺 image_token_id",
                dir.display()
            ));
        }
    }
    let g = v.get("text_config").unwrap_or(&v);
    let getu = |k: &str| g.get(k).and_then(|x| x.as_u64()).map(|n| n as usize);
    let geom = profile.geom;
    for (k, want) in [
        ("hidden_size", geom.hidden),
        ("num_hidden_layers", geom.n_layers),
        ("num_key_value_heads", geom.n_kv_heads),
        ("head_dim", geom.head_dim),
        ("vocab_size", geom.vocab),
    ] {
        if let Some(got) = getu(k) {
            if got != want {
                return Err(format!(
                    "模型目录与 intent.model={:?} 不匹配:{k}={got} 期望 {want}\n\
                     —— 检查 [model].dir / [site].model_dir 是否指向了别的检查点",
                    profile.key
                ));
            }
        }
    }
    if let Some(pos) = getu("max_position_embeddings") {
        if ctx > pos {
            return Err(format!(
                "max_seq={ctx} 超过检查点 max_position_embeddings={pos}(模型装不下)"
            ));
        }
        if pos != geom.max_pos {
            return Err(format!(
                "检查点 max_position_embeddings={pos} 与注册表 {:?} 的 {} 不一致 —— \
                 注册表需按该检查点重建(或换回注册表对应模型)",
                profile.key, geom.max_pos
            ));
        }
    }
    if let Some(interval) = getu("full_attention_interval") {
        if interval > 0 && geom.n_layers / interval != geom.n_full {
            return Err(format!(
                "检查点 full_attention_interval={interval} ⇒ n_full={},注册表 n_full={} —— 几何漂移",
                geom.n_layers / interval,
                geom.n_full
            ));
        }
    }
    Ok(Some(format!(
        "{} ✓ hidden={} layers={}(full {}) kv_heads={} hd={} max_pos={}",
        dir.display(),
        geom.hidden,
        geom.n_layers,
        geom.n_full,
        geom.n_kv_heads,
        geom.head_dim,
        geom.max_pos
    )))
}

/// 草稿 `config.json` 校验(DFlash2 + W4A16 pack-quantized;BF16 源 = AL 0)
fn check_draft_dir(dir: &Path) -> Result<String, String> {
    let p = dir.join("config.json");
    if !p.is_file() {
        return Ok(format!("{} 不可读(config.json 缺失)—— 校验跳过(真启动会失败)", dir.display()));
    }
    let raw = owl_shared::file_loader::read_to_string(&p)
        .map_err(|e| format!("读取 {} 失败: {e}", p.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("{} 不是合法 JSON: {e}", p.display()))?;
    let arch: Vec<String> = v
        .get("architectures")
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    if !arch.iter().any(|a| a.contains("DFlash2")) {
        return Err(format!(
            "草稿目录 {} 的 architectures={arch:?} 不是 DFlash2 —— 检查 [site].draft_dir",
            dir.display()
        ));
    }
    let fmt = v
        .get("quantization_config")
        .and_then(|q| q.get("format"))
        .and_then(|x| x.as_str());
    if fmt != Some("pack-quantized") {
        return Err(format!(
            "草稿目录 {} 不是 W4A16 pack-quantized(quantization_config.format={fmt:?})\n\
             —— BF16 源(z-lab 系)喂引擎 = 草稿全噪声 AL=0;生产用 syvai W4A16 系",
            dir.display()
        ));
    }
    let layers = v.get("num_hidden_layers").and_then(|x| x.as_u64());
    Ok(format!(
        "{} ✓ DFlash2 / pack-quantized{}",
        dir.display(),
        layers.map(|n| format!(" / {n} 层")).unwrap_or_default()
    ))
}

/// `--print-config` / `--explain` 输出(TOML 或 JSON;JSON 含来源/估算)
pub fn inspect(args: &CliArgs, r: &ResolvedConfig) -> Result<(), String> {
    let json_mode = args.print_config && args.format.as_deref() == Some("json");
    let mut check_ok = true;
    let check = if args.check {
        let rep = check_report(r)?;
        check_ok = rep.ok;
        Some(rep.text)
    } else {
        None
    };
    if let Some(t) = check.as_deref() {
        if json_mode {
            // stdout 必须是纯 JSON(--print-config --format json 可管道给 jq)
            eprint!("{t}");
        } else {
            print!("{t}");
        }
    }
    if args.print_config {
        match args.format.as_deref().unwrap_or("toml") {
            "json" => {
                let v = serde_json::json!({
                    "layers": r.layers,
                    "intent": r.intent(),
                    "config": &r.config,
                    "provenance": &r.origins,
                    "estimate": r.estimate(),
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&v)
                        .map_err(|e| format!("--print-config JSON 序列化失败: {e}"))?
                );
            }
            _ => {
                println!("# owl 生效配置(层:{})", r.layers.join(" → "));
                print!("{}", r.to_toml_pretty()?);
            }
        }
    }
    for k in &args.explain {
        print!("{}", r.explain(k)?);
    }
    if !check_ok {
        return Err("预检未通过(见上方 [check] FAIL)".into());
    }
    Ok(())
}
