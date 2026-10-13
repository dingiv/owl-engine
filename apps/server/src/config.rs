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
    /// 设备 ordinal(runtime.device)
    pub device: usize,
    /// attention 窗上限(runtime.max_seq)
    pub max_seq: usize,
    /// 预填 chunk(runtime.prefill_chunk)
    pub prefill_chunk: usize,
    /// lm_head 头部量化方案(model.head_quant;C1 2026-10-11;枚举严格)
    pub head_quant: owl_shared::config::HeadQuant,
    /// 最大并发请求(server.max_concurrency;默认 8;须 ≤ knobs.gdn_slots)
    pub max_concurrency: usize,
    /// 性能账落盘基名(--perf-dump;None = 不落盘)
    pub perf_dump: Option<PathBuf>,
    /// 性能分级(probes.perf_level,可被 --perf-level 覆盖)
    pub perf_level: u8,
    /// 日志面([log] module;CLI 可覆盖 level/file)
    pub log: owl_shared::config::LogCfg,
}

impl ServerConfig {
    /// boot 边界唯一入口:四层叠加加载 → app 面组装。
    /// `cli` 携带 --perf-dump / --perf-level 两个 app 面覆盖。
    pub fn load(flavor: Option<&str>, file: Option<&Path>, cli: &CliArgs) -> Result<Self, String> {
        let cfg = OwlConfig::load(flavor, file)?;
        let mut out = Self::from_config(cfg);
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
            device: cfg.runtime.device,
            max_seq: cfg.runtime.max_seq,
            prefill_chunk: cfg.runtime.prefill_chunk,
            head_quant: cfg.model.head_quant,
            max_concurrency: cfg.server.max_concurrency,
            perf_dump: None,
            perf_level: cfg.probes.perf_level.to_u8(),
            log: cfg.log,
        }
    }
}
