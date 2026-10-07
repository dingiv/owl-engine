//! 服务配置(boot 边界一次解析;进程内冻结)。
//!
//! 环境变量全部经 [`owl_shared::env_reader`] 读取(解析失败降级可见);
//! 档位分派([`ModelKind`])在此落地 —— engine 线程零 env 读取、零字符串
//! 魔数,非法档位 fail-fast 于 boot(`OWL_MODEL_KIND` 打错字不再静默落
//! 0.8b 基线)。

use std::path::PathBuf;

use owl_shared::env_reader;

// ── 缺省值(全 env 缺席时的基线;R2 命名常量)──
const DEFAULT_BIND: &str = "127.0.0.1:8135";
const DEFAULT_DEVICE: usize = 0;
const DEFAULT_MAX_SEQ: usize = 4096;
const DEFAULT_MODEL_KIND: &str = "0.8b";
const DEFAULT_MODEL_NAME: &str = "qwen3.5-0.8b";
const DEFAULT_PREFILL_CHUNK: usize = 512;

/// 模型档位(OWL_MODEL_KIND;2026-10-01 立两档)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    /// Qwen3.5-0.8B f16(开发基线;权重/rope 随 workspace 资产)
    Qwen35_08b,
    /// Qwen3.8-27B AWQ-INT4(生产档;model_dir = 检查点目录,tokenizer 同目录)
    Awq27b,
}

impl ModelKind {
    fn from_env() -> Result<Self, String> {
        match env_reader::str_or("OWL_MODEL_KIND", DEFAULT_MODEL_KIND).as_str() {
            "0.8b" => Ok(Self::Qwen35_08b),
            "awq27b" => Ok(Self::Awq27b),
            other => Err(format!(
                "OWL_MODEL_KIND={other:?} 未知档位(合法:0.8b | awq27b)"
            )),
        }
    }
}

/// 服务配置(唯一构造口 [`ServerConfig::from_env`];boot 后只读)
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// 监听地址(OWL_BIND)
    pub bind: String,
    /// 模型目录(OWL_MODEL_DIR;缺省 = workspace 资产 Qwen3.5-0.8B)
    pub model_dir: PathBuf,
    /// 模型档位(OWL_MODEL_KIND;非法值 boot 拒启)
    pub model_kind: ModelKind,
    /// 对外模型名(OWL_MODEL_NAME;应答/列表回显)
    pub model_name: String,
    /// 引擎 boot 旋钮(显式依赖律:全引擎 env→旋钮映射在此闭合,
    /// 引擎内部零 env 读取)
    pub knobs: owl_engine::EngineKnobs,
    /// 设备 ordinal(OWL_DEVICE;解析失败降级可见,缺省 0)
    pub device: usize,
    /// attention 窗上限(OWL_MAX_SEQ;E1 后 paged 布局解除旧 256 顶)
    pub max_seq: usize,
    /// 预填 chunk(OWL_PREFILL_CHUNK;2026-10-10 提档 32→512,1024+ 收益递减待测)
    pub prefill_chunk: usize,
}

impl ServerConfig {
    /// boot 边界唯一 env 读取点(解析失败/非法档位 = Err,main fail-fast)
    pub fn from_env() -> Result<Self, String> {
        Ok(Self {
            bind: env_reader::str_or("OWL_BIND", DEFAULT_BIND),
            model_dir: env_reader::str("OWL_MODEL_DIR").map(PathBuf::from).unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../crates/models/assets/Qwen3.5-0.8B")
            }),
            model_kind: ModelKind::from_env()?,
            knobs: owl_engine::EngineKnobs::from_env(),
            model_name: env_reader::str_or("OWL_MODEL_NAME", DEFAULT_MODEL_NAME),
            device: env_reader::parse_or("OWL_DEVICE", DEFAULT_DEVICE),
            max_seq: env_reader::parse_or("OWL_MAX_SEQ", DEFAULT_MAX_SEQ),
            prefill_chunk: env_reader::parse_or("OWL_PREFILL_CHUNK", DEFAULT_PREFILL_CHUNK),
        })
    }
}
