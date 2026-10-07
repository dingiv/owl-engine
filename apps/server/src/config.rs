//! 服务配置(boot 边界一次解析;进程内冻结)。
//!
//! **显式依赖律终点(2026-10-10)**:全量配置统一定义在
//! `owl_shared::config`(总账 + 强类型 modules + 唯一 loader);
//! 本模块只做 server app 面的组装 —— [`OwlConfig`] 加载一次,
//! 档位分派([`ModelKind`])与引擎旋钮([`EngineKnobs::from_config`])
//! 全部从配置对象搬运,零散点 parse。枚举非法 fail-fast 于 boot
//! (`OWL_MODEL_KIND` 打错字拒启,不再静默落 0.8b 基线)。

use std::path::PathBuf;

use owl_shared::config::OwlConfig;
pub use owl_shared::config::ModelKind;

// ── app 面缺省(总账外的 server 组装缺省;R2 命名常量)──
const DEFAULT_BIND: &str = "127.0.0.1:8135";
const DEFAULT_MODEL_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../crates/models/assets/Qwen3.5-0.8B"
);
const DEFAULT_MODEL_NAME: &str = "qwen3.5-0.8b";

/// 服务配置(唯一构造口 [`ServerConfig::from_env`];boot 后只读)
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// 监听地址(OWL_BIND)
    pub bind: String,
    /// 模型目录(OWL_MODEL_DIR;构造期已回退 workspace 资产缺省)
    pub model_dir: PathBuf,
    /// 模型档位(OWL_MODEL_KIND;非法值 boot 拒启)
    pub model_kind: ModelKind,
    /// 对外模型名(OWL_MODEL_NAME;构造期已回退服务端缺省;应答/列表回显)
    pub model_name: String,
    /// 引擎旋钮(全量配置 engine/cuda/sampling module 的显式搬运;
    /// 引擎内部零 env 读取)
    pub knobs: owl_engine::EngineKnobs,
    /// 设备 ordinal(OWL_DEVICE)
    pub device: usize,
    /// attention 窗上限(OWL_MAX_SEQ;E1 后 paged 布局解除旧 256 顶)
    pub max_seq: usize,
    /// 预填 chunk(OWL_PREFILL_CHUNK;2026-10-10 提档 32→512,1024+ 收益递减待测)
    pub prefill_chunk: usize,
}

impl ServerConfig {
    /// boot 边界唯一入口:统一 loader 一次加载 → app 面组装。
    /// (枚举非法 = Err 透传,main fail-fast。)
    pub fn from_env() -> Result<Self, String> {
        let cfg = OwlConfig::from_env()?;
        Ok(Self {
            bind: cfg.runtime.bind.clone(),
            model_dir: cfg
                .model
                .model_dir
                .clone()
                .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR)),
            model_kind: cfg.model.kind,
            model_name: cfg
                .model
                .model_name
                .clone()
                .unwrap_or_else(|| DEFAULT_MODEL_NAME.into()),
            knobs: owl_engine::EngineKnobs::from_config(&cfg),
            device: cfg.runtime.device,
            max_seq: cfg.runtime.max_seq,
            prefill_chunk: cfg.runtime.prefill_chunk,
        })
    }



    /// bind 引用(编译期缺省兜底;config loader 已填)
    pub fn bind_or_default(&self) -> &str {
        if self.bind.is_empty() {
            DEFAULT_BIND
        } else {
            &self.bind
        }
    }
}
