//! 服务配置(boot 边界一次解析;进程内冻结)。
//!
//! **显式依赖律终点 + 配置文件面(2026-10-10)**:全量配置统一定义在
//! `owl_shared::config`(总账 + 强类型 modules + 四层叠加 loader:
//! Default → Flavor → 用户文件 → CLI);本模块只做 server app 面的
//! 组装 —— env 彻底退出正式配置链(缩编裁决),档位分派与引擎旋钮
//! 全部从配置对象搬运。枚举非法/账外键 fail-fast 于 boot。

use std::path::{Path, PathBuf};

use owl_shared::config::{OwlConfig, ResolvedConfig, SpecDraft};
pub use owl_shared::config::ModelFormat;

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
                        it.next().ok_or("--explain 需要键(点路径,如 runtime.max_seq)")?,
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

/// **服务面处理器视图**(`[runtime]` + `[server]` 的组装体;不是配置段)。
///
/// 为什么保留这个名字:模板/工具解析/会话缺省在 `openai.rs` / `load.rs` /
/// `main.rs` 是**整包传递**的(处理一条请求要同时看"格式 + 缺省 + 上限");
/// 拆成两个结构反而每次都要搬两份参数。配置面已按职责拆散
/// (`[runtime] implicit_session/...` + `[server] parse_tool_calls/...`),
/// 本结构只是它们在 app 层的投影。
#[derive(Clone, Debug, Default)]
pub struct AgentCfg {
    /// 模板路径覆盖(None = 模型快照 `<model.path>/chat_template.jinja`)
    pub template_path: Option<String>,
    /// 隐式默认会话(请求不带 session 键时落到会话 0)
    pub implicit_session: bool,
    /// 未带 `max_tokens` 时的生成步数
    pub default_max_new: usize,
    /// `max_tokens` 夹紧上限
    pub max_new_cap: usize,
    /// 工具三通道解析
    pub parse_tool_calls: bool,
    /// 工具格式(必须与模板同值)
    pub tool_call_format: owl_shared::config::ToolCallFormat,
    /// 有工具即关思考
    pub auto_disable_thinking_with_tools: bool,
    /// 思考档位缺省(None = 模板自决)
    pub reasoning_effort: Option<String>,
    /// 排队超时毫秒(0 = 关)
    pub queue_wait_ms: usize,
    /// SSE 静默保活间隔毫秒(0 = 关)—— 长 prefill 期间探断流
    pub keepalive_ms: usize,
}

impl AgentCfg {
    /// **唯一组装点**:`[runtime]` 的会话面 + `[server]` 的模板/工具面。
    pub fn from_config(cfg: &OwlConfig) -> Self {
        Self {
            // 空串 = 未设(TOML 里写 `template_path = ""` 的语义)
            template_path: cfg
                .server
                .template_path
                .clone()
                .filter(|s| !s.trim().is_empty()),
            implicit_session: cfg.runtime.implicit_session,
            default_max_new: cfg.runtime.default_max_new,
            max_new_cap: cfg.runtime.max_new_cap,
            parse_tool_calls: cfg.server.parse_tool_calls,
            tool_call_format: cfg.server.tool_call_format,
            auto_disable_thinking_with_tools: cfg.server.auto_disable_thinking_with_tools,
            reasoning_effort: cfg
                .server
                .reasoning_effort
                .clone()
                .filter(|s| !s.trim().is_empty()),
            queue_wait_ms: cfg.runtime.queue_wait_ms,
            keepalive_ms: cfg.runtime.keepalive_ms,
        }
    }
}

/// 服务配置(构造口 = [`ServerConfig::load_resolved`] + [`ServerConfig::from_resolved`];boot 后只读)
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// 监听地址(`[server] bind`)
    pub bind: String,
    /// 原始请求 JSON 落盘目录（空 = 关；前端联调复现用）
    pub request_dump_dir: Option<PathBuf>,
    /// 单图像素上限（视觉预处理封顶；0 = 内置 4096²=不缩）
    pub vision_max_pixels: usize,
    /// 模型路径(`[model] path`;目录或 .gguf 单文件;构造期回退 workspace 缺省)
    pub model_dir: PathBuf,
    /// 模型格式(`[model] format`;装载器分档的唯一判据)
    pub model_format: ModelFormat,
    /// 对外模型名(`[model] name`;构造期已回退服务端缺省)
    pub model_name: String,
    /// 引擎旋钮(全量配置 engine/cuda/sampling module 的显式搬运)
    pub knobs: owl_engine::EngineKnobs,
    /// 设备所有权面（`[parallel]` + 空闲阈值；卡锁端口由 main 回填）。
    /// **选卡在引擎 boot 内**（2026-11-15 业务下沉 P1），server 只搬配置字段。
    pub device_opts: owl_engine::DeviceOpts,
    /// attention 窗上限(runtime.max_seq)
    pub max_seq: usize,
    /// 预填 chunk(runtime.prefill_chunk)
    pub prefill_chunk: usize,
    /// lm_head 头部量化方案(model.head_quant;C1 2026-10-11;枚举严格)
    pub head_quant: owl_shared::config::HeadQuant,
    /// embedding 表量化(model.embed_quant;P8 2026-10-22;默认 f16)
    pub embed_quant: owl_shared::config::EmbedQuant,
    /// **主干量化方案**(model.quant;S2 2026-10-17;枚举严格 —— 原
    /// `w4a8: bool` 已退役,旧键响亮拒)
    pub quant: owl_shared::config::QuantScheme,
    /// 最大并发请求(runtime.max_concurrency;默认 8;须 ≤ knobs.gdn_slots)
    pub max_concurrency: usize,
    /// **合批上限**([runtime] max_batch;1 = 串行现状)。服务面用它做两件事:
    /// 会话串行的配套告警(隐式会话 + 合批 ⇒ 串行)与 `/debug/vram` 的如实报数。
    pub max_batch: usize,
    /// **TP AR 载荷 fp8 档**（parallel.tp_ar_f8；§47 第二步）⇒ 交 `TpOpts.ar_f8`
    pub tp_ar_f8: bool,
    /// **lm_head 词表切**（runtime.tp_vocab_split；P1.1）⇒ 装载面
    /// `ModelSpec::head_split`（每 rank 只持半表头；logits 由引擎 host 拼接）
    pub tp_vocab_split: bool,
    /// **AR 拓扑档**（parallel.tp_ar;v1|fused|ar2,2026-10-27 单键化）⇒ 交 `TpOpts`
    pub tp_ar: owl_shared::config::TpArMode,
    /// **TP 卡间通信通道**（parallel.tp_comm;p2p|host,2026-11-04）⇒ 交 `TpOpts`
    pub tp_comm: owl_shared::config::TpCommMode,
    /// 性能账落盘基名(--perf-dump;None = 不落盘)
    pub perf_dump: Option<PathBuf>,
    /// 性能分级(probes.perf_level,可被 --perf-level 覆盖)
    pub perf_level: u8,
    /// 日志面([log] module;CLI 可覆盖 level/file)
    pub log: owl_shared::config::LogCfg,
    /// 服务面处理器要的模板/会话旋钮(**由 `[runtime]`+`[server]` 组装的 app 面
    /// 视图**,不是配置段 —— 配置段已按职责拆散,这里只为处理器少搬参数)
    pub agent: AgentCfg,
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
        // `--print-config` 是迁移工具:它要把半份旧档补成全量档 ⇒ 放行缺键;
        // 其余路径(`--check` / boot / 测试)一律缺键拒启。
        let completeness = if args.print_config {
            owl_shared::config::Completeness::Tolerate
        } else {
            owl_shared::config::Completeness::Enforce
        };
        let mut resolved =
            OwlConfig::load_resolved(base, &layers, &args.overrides(), completeness)?;
        if let Some(m) = resolved.meta.as_mut() {
            m.intent.profile = base.map(|p| p.display().to_string());
        }
        Ok(resolved)
    }

    /// 完整解析产物 → server 形态（设备选卡已下沉到 `EngineService::boot`）。
    pub fn from_resolved(
        resolved: ResolvedConfig,
        args: &CliArgs,
    ) -> Result<Self, String> {
        let mut out = Self::from_config(resolved.config)?;
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
    fn default_model_name(format: ModelFormat, dir: Option<&std::path::Path>) -> String {
        if format == ModelFormat::Gguf {
            if let Some(stem) = dir.and_then(|d| d.file_stem()).and_then(|s| s.to_str()) {
                if !stem.is_empty() {
                    return stem.to_string();
                }
            }
        }
        DEFAULT_MODEL_NAME.to_string()
    }

    /// app 面组装(已加载配置 → server 形态;测试可直接喂 default/定制)
    pub fn from_config(cfg: OwlConfig) -> Result<Self, String> {
        // 先组装 app 面视图(要整份 cfg);后面按值搬移各 module
        let agent = AgentCfg::from_config(&cfg);
        Ok(Self {
            bind: cfg.server.bind.clone(),
            request_dump_dir: {
                let p = cfg.server.request_dump_dir.trim();
                (!p.is_empty()).then(|| PathBuf::from(p))
            },
            vision_max_pixels: cfg.vision.max_pixels,
            model_dir: cfg
                .model
                .path
                .clone()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR)),
            model_format: cfg.model.format,
            model_name: cfg
                .model
                .name
                .clone()
                .unwrap_or_else(|| Self::default_model_name(cfg.model.format, cfg.model.path.as_deref())),
            knobs: owl_engine::EngineKnobs::from_config(&cfg)
                .map_err(|e| format!("engine knobs: {e}"))?,
            device_opts: owl_engine::DeviceOpts {
                parallel: cfg.parallel.clone(),
                gpu_min_free_ratio: cfg.runtime.gpu_min_free_ratio,
                lock_port: 0,
                lock_role: None,
            },
            // 上下文/块长:**单一来源 = 配置**(2026-10-20 删除 OWL_MAX_SEQ /
            // OWL_PREFILL_CHUNK —— 它们会在意图派生后再改一次 ⇒ max_seq 与
            // max_seq 失配的双源;要临时调请 `--set`）。
            max_seq: cfg.runtime.max_seq,
            prefill_chunk: cfg.runtime.prefill_chunk,
            head_quant: cfg.model.head_quant,
            embed_quant: cfg.model.embed_quant,
            quant: cfg.model.quant,
            tp_ar_f8: cfg.parallel.tp_ar_f8,
            tp_comm: cfg.parallel.tp_comm,
            tp_vocab_split: cfg.parallel.tp_vocab_split,
            tp_ar: cfg.parallel.tp_ar,
            max_concurrency: cfg.runtime.max_concurrency,
            max_batch: cfg.runtime.max_batch,
            perf_dump: None,
            perf_level: cfg.probes.perf_level.to_u8(),
            log: cfg.log,
            agent,
            step_gate: cfg.probes.step_gate,
            step_gate_ms: cfg.probes.step_gate_ms,
            step_gate_us: cfg.probes.step_gate_us,
            step_gate_every: cfg.probes.step_gate_every,
        })
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
        "prefill={} decode={} gdn={}",
        "eager",
        c.kernel.decode_arm.as_str(),
        if c.kernel.gdn_chunked { "chunked" } else { "scalar" },
    );
    Some(format!(
        "[owl] profile={} card={} ctx={} pool=auto({}) {} mem={}/{} (余 {}) reuse={} spec={} c={}",
        m.profile.as_deref().unwrap_or("(inline)"),
        m.intent.hardware.as_deref().unwrap_or("auto"),
        c.runtime.max_seq,
        gib(e.pool_bytes),
        arms,
        gib(e.used_bytes()),
        gib(e.total_bytes),
        gib(e.total_bytes.saturating_sub(e.used_bytes())),
        if c.pool.prefix_cache { format!("snap{}", c.pool.snap_max) } else { "off".into() },
        if c.spec.depth > 0 { format!("dflash{}", c.spec.depth) } else { "off".into() },
        c.runtime.max_concurrency,
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
            "[check] resolved: max_seq={} pool=auto(剩余显存;boot 报 cap) chunk={} tail={} \
             gdn_slots={} max_batch={} max_concurrency={} kv_dtype={}\n",
            c.runtime.max_seq,
            c.runtime.prefill_chunk,
            c.runtime.prefill_tail_merge,
            c.pool.gdn_slots,
            c.runtime.max_batch,
            c.runtime.max_concurrency,
            c.pool.kv_dtype,
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
                "[check] governor: budget {} cap {} tok; logical_pool {} {} cap\n",
                gib(e.budget_bytes),
                e.cap_tokens,
                e.pool_tokens,
                if e.pool_tokens <= e.cap_tokens { "≤" } else { ">" },
            ));
            out.push_str(&format!(
                "[check] basis: per_tok {}B(主 {} + 草稿 {})·激活 {}·固定 = gdn {} + snap {}·预留 {}\n",
                e.per_tok,
                e.per_tok_main,
                e.per_tok_draft,
                e.activation_note,
                gib(e.fixed.gdn_grid),
                gib(e.fixed.gdn_snap),
                gib(e.reserve_bytes),
            ));
            out.push_str(
                "[check] note: 估算池 = 逻辑下限(上下文×并发);物理池 = 无缓存→min(cap, ctx×并发+1页),缓存开→cap\n",
            );
        }
    } else {
        out.push_str(
            "[check] 无 [intent]:仅做配置解析/语义门 + 模型几何校验(低层档/覆盖路径)。\n\
             [check] 要内存账/意图派生,请用 -p <profile_path> 或写 [intent]。\n",
        );
        out.push_str(&format!(
            "[check] resolved: max_seq={} pool=auto(剩余显存;boot 报 cap) chunk={} tail={} \
             gdn_slots={} max_batch={} max_concurrency={} kv_dtype={}\n",
            c.runtime.max_seq,
            c.runtime.prefill_chunk,
            c.runtime.prefill_tail_merge,
            c.pool.gdn_slots,
            c.runtime.max_batch,
            c.runtime.max_concurrency,
            c.pool.kv_dtype,
        ));
    }

    // ── 模型/草稿目录几何校验(目录不存在 ⇒ 告警跳过,CI 假路径可跑) ──
    //
    // **检查点事实归 engine**（2026-11-16 业务下沉 P2）：臂分派
    // （`load_arm_of`）→ `LoadIntent` → `open_decl`/几何对账/视觉声明、
    // 草稿族判定、MTP 键全部在 `owl_engine::preflight`（与 boot 同一份解析）。
    // server 只算"该查哪个目录"（配置语义）并按既有前缀打印 —— 不再认识
    // `LoadArm` / `pack-quantized` / `mtp.*`（历史错位：`format=awq` +
    // `.gguf` 路径"检查通过、启动失败"）。
    let profile = match intent {
        Some(m) => Some(owl_shared::config::intent::model_profile(&m.intent.model)?),
        None => owl_shared::config::intent::model_by_format(c.model.format),
    };
    // 27B 没有"缺省资产":不给 dir ⇒ 真启动必失败,`--check` 直接拒;
    // 0.8B 有 workspace 缺省资产,走它做几何校验。
    let default_08b = || PathBuf::from(DEFAULT_MODEL_DIR);
    let dir_owned = match &c.model.path {
        Some(d) => Some(d.clone()),
        None if c.model.format == ModelFormat::Safetensors => Some(default_08b()),
        None => None,
    };
    match dir_owned.as_deref() {
        Some(dir) => {
            // **一条路径**:臂 → 源族 → 声明事实;几何对账 / 视觉声明 / 摘要
            // 全部走 `CheckpointDecl` 接口(与 boot 同一份解析)。
            let vision_path = c.vision.path.as_deref().filter(|p| !p.as_os_str().is_empty());
            let spec = owl_engine::preflight::ModelCheckSpec {
                path: dir,
                format: c.model.format,
                quant: c.model.quant,
                head_quant: c.model.head_quant,
                embed_quant: c.model.embed_quant,
                max_seq: c.runtime.max_seq,
                vision: c.vision.enabled,
                vision_path,
                profile,
            };
            match owl_engine::preflight::check_model_dir(&spec) {
                Ok(lines) => push_check(&mut out, &mut ok, "model dir", lines),
                Err(e) => {
                    ok = false;
                    out.push_str(&format!("[check] FAIL: {e}\n"));
                }
            }
        }
        None => {
            ok = false;
            out.push_str(
                "[check] FAIL: 未配置 [model] path / [site].model_dir —— 无法定位检查点\
                 (真启动必失败);请给站点档\n",
            );
        }
    }
    if c.spec.depth > 0 {
        let dflash2 = match c.spec.draft {
            SpecDraft::Dflash2 => true,
            SpecDraft::Mtp => false,
            SpecDraft::Auto => c.spec.draft_path.is_some(),
        };
        if dflash2 {
            match c.spec.draft_path.as_deref() {
                Some(dir) => push_check(
                    &mut out,
                    &mut ok,
                    "draft dir",
                    owl_engine::preflight::check_draft_dir(dir),
                ),
                None => {
                    ok = false;
                    out.push_str("[check] FAIL: spec.draft=dflash2 但 spec.draft_path 为空\n");
                }
            }
        } else if matches!(c.spec.draft, SpecDraft::Auto) && c.spec.dumb {
            out.push_str("[check] draft: dumb(诊断哑草稿;不需要 MTP/DFlash2 资产)\n");
        } else {
            if c.spec.draft_path.is_some() {
                out.push_str(
                    "[check] note: spec.draft=mtp 忽略 spec.draft_path(显式选择优先)\n",
                );
            }
            match c.model.path.as_deref() {
                Some(dir) => {
                    let mtp = c
                        .model
                        .mtp_dir
                        .as_deref()
                        .map(Path::new)
                        .unwrap_or_else(|| Path::new(dir));
                    push_check(
                        &mut out,
                        &mut ok,
                        "draft",
                        owl_engine::preflight::check_mtp_head(mtp),
                    );
                }
                None => out.push_str(
                    "[check] draft: MTP 头随检查点装载;缺 model.path 时无法预检 mtp.* 键\n",
                ),
            }
        }
    }

    // ── 组合告警(不拒;`--check` 的 UX 价值就在把隐患提前说出来) ──

    // ── 臂摘要 ──
    let prefill = "eager";
    let decode = c.kernel.decode_arm.as_str();
    out.push_str(&format!(
        "[check] arms: prefill={prefill} decode={decode} gdn={} spec=depth{}({:?}) reuse={} batch={} vision={}\n",
        if c.kernel.gdn_chunked { "chunked" } else { "scalar" },
        c.spec.depth,
        c.spec.draft,
        if c.pool.prefix_cache { format!("snap{}", c.pool.snap_max) } else { "off".into() },
        c.runtime.max_batch,
        if c.vision.enabled { "on" } else { "off" },
    ));
    if c.vision.enabled {
        if c.spec.depth > 0 {
            out.push_str(
                "[check] note: vision=on + spec>0:图像 × 投机已接(V5)—— verify 的 rope \
                 pos = 真实位置 + m-RoPE delta,KV slots/lens 仍用真实位置;\
                 带图为单请求多图/非合批;文本+draft 同档可用\n",
            );
        } else {
            out.push_str(
                "[check] note: vision=on + spec=off:图像路径可用(单请求多图;data URL;\
                 TP2 两 rank 各备塔输出;`[vision] image_evict` = 前缀无感删除)\n",
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

/// 预检行 → 报告文本(顺序即报告顺序;`Fail` 统一打 `[check] FAIL:`)。
///
/// 检查点事实由 `owl_engine::preflight` 产出(与 boot 同源),这里只排版。
fn push_check(
    out: &mut String,
    ok: &mut bool,
    label: &str,
    lines: owl_engine::preflight::CheckLines,
) {
    for item in lines.items {
        match item {
            owl_engine::preflight::CheckItem::Pass(t) => {
                out.push_str(&format!("[check] {label}: {t}\n"))
            }
            owl_engine::preflight::CheckItem::Fail(e) => {
                *ok = false;
                out.push_str(&format!("[check] FAIL: {e}\n"));
            }
        }
    }
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
