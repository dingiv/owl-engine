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
                _ if a.starts_with('-') => {
                    return Err(format!("未知参数 {a:?}(支持:[CONFIG] --flavor <N> --list-flavors)"));
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
}

impl ServerConfig {
    /// boot 边界唯一入口:四层叠加加载 → app 面组装。
    pub fn load(flavor: Option<&str>, file: Option<&Path>) -> Result<Self, String> {
        let cfg = OwlConfig::load(flavor, file)?;
        Ok(Self::from_config(cfg))
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
        }
    }
}
