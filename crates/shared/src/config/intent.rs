//! **意图层(Intent)**:`profiles/*.toml` 只写 6–10 个意图键,低层键由
//! [`super::resolve`] 派生。
//!
//! 设计卷:`docs/arch/配置面-体验优化-设计与施工.md` §5。
//!
//! 纪律:
//! - 意图是**输入**,不是第二份配置真源 —— 派生结果仍是一份 [`super::OwlConfig`];
//! - 意图表不认识的键 = `deny_unknown_fields` 响亮拒(附可用键清单);
//! - 所有注册表(模型/硬件)**只在这里登记**;内存模型/解析器都从这里取数,
//!   不许第二处写死几何/显存。
//!
//! 模型/硬件数字为 **2026-10-20 实测校准**(见记忆:专用 `[vram-gov]` 行);
//! `calibrated = false` 的条目只作保守估计,`--check` 会标注。

use serde::{Deserialize, Serialize};

use super::{HeadQuant, ModelKind, QuantScheme};

// ============================================================================
// §1 意图 schema
// ============================================================================

/// 目标档(`[intent].goal`)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Goal {
    /// 最大上下文(单会话/贴顶;复用关)
    MaxContext,
    /// 最大吞吐(图全开/尾合并;复用全局)
    MaxSpeed,
    /// 折中(复用会话级)
    #[default]
    Balanced,
    /// 低显存(池小于上下文;虚拟化)
    LowMemory,
}

impl Goal {
    pub const NAMES: &'static [&'static str] =
        &["max-context", "max-speed", "balanced", "low-memory"];
    pub fn name(self) -> &'static str {
        match self {
            Self::MaxContext => "max-context",
            Self::MaxSpeed => "max-speed",
            Self::Balanced => "balanced",
            Self::LowMemory => "low-memory",
        }
    }
}

/// 上下文意图(`"max"` 或数字 token)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextSpec {
    /// 显式 token 数
    Tokens(usize),
    /// 内存模型求解(启动前给出依据)
    Max,
}

impl<'de> Deserialize<'de> for ContextSpec {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        let v = toml::Value::deserialize(d)?;
        match v {
            toml::Value::Integer(n) if n > 0 => Ok(Self::Tokens(n as usize)),
            toml::Value::Integer(n) => {
                Err(D::Error::custom(format!("[intent].context = {n} 非法:须 > 0")))
            }
            toml::Value::String(s) if s.trim().eq_ignore_ascii_case("max") => Ok(Self::Max),
            toml::Value::String(s) => Err(D::Error::custom(format!(
                "[intent].context = {s:?} 非法:写数字(token)或 \"max\""
            ))),
            other => Err(D::Error::custom(format!(
                "[intent].context 类型非法:{} —— 写数字(token)或 \"max\"",
                super::toml_type_name(&other)
            ))),
        }
    }
}

impl ContextSpec {
    pub fn name(self) -> String {
        match self {
            Self::Tokens(n) => n.to_string(),
            Self::Max => "max".into(),
        }
    }
}

impl Serialize for ContextSpec {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Tokens(n) => s.serialize_u64(*n as u64),
            Self::Max => s.serialize_str("max"),
        }
    }
}

/// 投机意图
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpecIntent {
    #[default]
    Off,
    /// DFlash2 草稿(depth 7;需 `[site].draft_dir`)
    Dflash,
}

impl SpecIntent {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Dflash => "dflash",
        }
    }
}

/// KV 池 dtype 意图
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvIntent {
    /// fp8_e4m3(容量 ×2;FI/XQA 读核前提)
    #[default]
    Fp8,
    /// f16
    F16,
}

impl KvIntent {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::F16 => "f16",
        }
    }
}

/// KVMem 意图(一个键替四键)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvmemIntent {
    /// 不超卖:池 ≥ 上下文(高显存档的纯收益姿势)
    #[default]
    Off,
    /// 虚拟化:池 < 上下文(需宿主/磁盘腿)
    Overcommit,
}

impl KvmemIntent {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Overcommit => "overcommit",
        }
    }
}

/// 图意图
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphIntent {
    /// 能开则开;开不了自动降级并打印原因
    #[default]
    Auto,
    /// 全关(eager 直发;A/B 与诊断)
    Off,
}

impl GraphIntent {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
        }
    }
}

/// 前缀复用意图
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReuseIntent {
    /// 不建前缀缓存/快照(单会话最大上下文档)
    Off,
    /// 同会话续写(1 条快照)
    Session,
    /// 跨会话前缀(4 条快照 + 宿主腿)
    Global,
}

impl ReuseIntent {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Session => "session",
            Self::Global => "global",
        }
    }
}

/// `[intent]` 表(**唯一意图输入面**)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Intent {
    /// 模型注册表键(如 `qwen3.8-27b-awq`)
    pub model: String,
    /// 硬件注册表键(如 `3090ti-24g`);不写 = 从 `[runtime].device` 推断
    #[serde(default)]
    pub hardware: Option<String>,
    #[serde(default)]
    pub goal: Goal,
    /// 必填:显式 token 或 `"max"`(启动前求解)
    pub context: ContextSpec,
    /// 并发会话数(派生 `gdn_slots / max_concurrency / max_batch`)
    #[serde(default = "one")]
    pub concurrency: usize,
    #[serde(default)]
    pub spec: SpecIntent,
    #[serde(default)]
    pub kv: KvIntent,
    #[serde(default)]
    pub kvmem: KvmemIntent,
    #[serde(default)]
    pub graphs: GraphIntent,
    /// 不写 = 由 `goal` 决定
    #[serde(default)]
    pub reuse: Option<ReuseIntent>,
}

fn one() -> usize {
    1
}

impl Intent {
    pub const KEYS: &'static [&'static str] = &[
        "model",
        "hardware",
        "goal",
        "context",
        "concurrency",
        "spec",
        "kv",
        "kvmem",
        "graphs",
        "reuse",
    ];
}

// ============================================================================
// §2 模型注册表(几何 + 校准显存;唯一登记处)
// ============================================================================

/// 文本主干几何(与 checkpoint `config.json` 的 `text_config` 对齐)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ModelGeom {
    pub hidden: usize,
    pub n_layers: usize,
    /// 全注意力层数(其余 = GDN 线性层)
    pub n_full: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    /// checkpoint 的 `max_position_embeddings`
    pub max_pos: usize,
    /// GDN 单行状态元素数(`nk*hk*3 + nv*hv*3 + nv*hk*hv`;f32 元素)
    pub gdn_slot_elems: usize,
}

/// 草稿(DFlash2)几何
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DraftGeom {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// 每 token 草稿 KV 元素(f16;per_tok 再 ×2)
    pub kv_elems_per_token: usize,
}

/// 模型档(注册表项)
#[derive(Clone, Copy, Debug)]
pub struct ModelProfile {
    pub key: &'static str,
    pub aliases: &'static [&'static str],
    pub kind: ModelKind,
    pub quant: QuantScheme,
    pub head_quant: HeadQuant,
    pub geom: ModelGeom,
    /// **装载后、池分配前的单卡 live**(字节;实测校准)
    pub weights_base_bytes: u64,
    /// 草稿模型额外权重(字节;`spec = dflash` 时加)
    pub draft_weights_bytes: u64,
    pub draft: Option<DraftGeom>,
    /// 实测校准(true = 有 `[vram-gov]` 现场;false = 保守估计)
    pub calibrated: bool,
}

impl ModelProfile {
    /// 主池每 token 字节(K+V × n_full 层 × 头 × 头维 × elem)
    pub fn kv_bytes_per_token(&self, fp8: bool) -> u64 {
        let elem = if fp8 { 1 } else { 2 };
        (self.geom.n_full * 2 * self.geom.n_kv_heads * self.geom.head_dim * elem) as u64
    }

    /// 草稿池每 token 字节(f16;草稿池固定 bf16 档)
    pub fn draft_bytes_per_token(&self) -> u64 {
        match self.draft {
            Some(d) => (d.n_layers * 2 * d.n_kv_heads * d.head_dim * 2) as u64,
            None => 0,
        }
    }

    /// GDN 单行状态字节(f32;与 `engine::FixedAfter` 同式:
    /// `n_gdn 层 × 每层元素 × 4B`)
    pub fn gdn_row_bytes(&self) -> u64 {
        let n_gdn = self.geom.n_layers.saturating_sub(self.geom.n_full);
        (n_gdn * self.geom.gdn_slot_elems * 4) as u64
    }

    pub fn weights_bytes(&self, spec_dflash: bool) -> u64 {
        self.weights_base_bytes + if spec_dflash { self.draft_weights_bytes } else { 0 }
    }
}

/// 27B AWQ-INT4(Qwen3.8;生产主模型)
///
/// 校准来源(2026-10-20,单 3090 Ti):
/// `[vram-gov] total=23.60G ... live=16.26G ... fixed_after=0.90G`(spec off)
/// 与 `live=17.52G`(spec dflash,含草稿权重)。几何来自
/// `models/cyankiwi/Qwen3.8-27B-AWQ-INT4/config.json` 的 `text_config`。
pub const QWEN38_27B_AWQ: ModelProfile = ModelProfile {
    key: "qwen3.8-27b-awq",
    aliases: &["27b-awq", "awq27b", "qwen3.8-27b"],
    kind: ModelKind::Awq27b,
    quant: QuantScheme::W4a16,
    head_quant: HeadQuant::RtnInt4G32,
    geom: ModelGeom {
        hidden: 5120,
        n_layers: 64,
        n_full: 16,
        n_kv_heads: 4,
        head_dim: 256,
        vocab: 248320,
        max_pos: 262144,
        // 16*128*3 + 48*128*3 + 48*128*128 = 811,008(f32 元素)
        gdn_slot_elems: 811_008,
    },
    // 16.26 GiB = 16,650 MiB(实测 live,含 int4 头节省后的权重/运行底价)
    weights_base_bytes: 16_650 << 20,
    // 17.52 − 16.26 = 1.26 GiB(DFlash2 W4A16 草稿权重 + 运行底价)
    draft_weights_bytes: 1_290 << 20,
    draft: Some(DraftGeom { n_layers: 5, n_kv_heads: 8, head_dim: 128, kv_elems_per_token: 10240 }),
    calibrated: true,
};

/// 0.8B 开发档(Qwen3.5)
///
/// ⚠️ `calibrated = false`:权重字节为 f16 参数量的保守上界(1.6 GiB),
/// 只在 `context="max"` 时提供保守解;真机数字以 `[vram-gov] live=` 为准。
pub const QWEN35_08B: ModelProfile = ModelProfile {
    key: "qwen3.5-0.8b",
    aliases: &["0.8b", "dev-08b"],
    kind: ModelKind::Qwen35_08b,
    quant: QuantScheme::W4a16,
    head_quant: HeadQuant::Off,
    geom: ModelGeom {
        hidden: 1024,
        n_layers: 24,
        n_full: 6,
        n_kv_heads: 2,
        head_dim: 256,
        vocab: 248320,
        max_pos: 262144,
        // 16*128*3 + 16*128*3 + 16*128*128 = 274,432
        gdn_slot_elems: 274_432,
    },
    weights_base_bytes: 1_650 << 20,
    draft_weights_bytes: 0,
    draft: None,
    calibrated: false,
};

pub const MODELS: &[ModelProfile] = &[QWEN38_27B_AWQ, QWEN35_08B];

/// 模型键 → 档(主键或别名;未命中 = Err 带清单)
pub fn model_profile(key: &str) -> Result<&'static ModelProfile, String> {
    let k = key.trim();
    MODELS
        .iter()
        .find(|p| p.key == k || p.aliases.iter().any(|a| *a == k))
        .ok_or_else(|| {
            format!(
                "[intent].model = {k:?} 未登记(可用:{:?})",
                MODELS.iter().map(|p| p.key).collect::<Vec<_>>()
            )
        })
}

// ============================================================================
// §3 硬件注册表(卡总显存;治理器口径)
// ============================================================================

/// 硬件档(卡总量 = `[vram-gov] total=` 的实测口径,不是 nvidia-smi 标称)
#[derive(Clone, Copy, Debug)]
pub struct HardwareProfile {
    pub key: &'static str,
    pub aliases: &'static [&'static str],
    /// 卡总显存字节(治理器 `cuMemGetInfo` 口径)
    pub total_bytes: u64,
    /// nvidia-smi 标称 MiB(仅展示/核对)
    pub nominal_mib: u64,
    pub calibrated: bool,
}

pub const RTX_3090TI_24G: HardwareProfile = HardwareProfile {
    key: "3090ti-24g",
    aliases: &["3090ti", "3090-ti", "3090", "24g"],
    // 实测 `[vram-gov] total=23.60G`(GiB ⇒ 24,166 MiB)
    total_bytes: 24_166 << 20,
    nominal_mib: 24564,
    calibrated: true,
};

pub const RTX_3080_20G: HardwareProfile = HardwareProfile {
    key: "3080-20g",
    aliases: &["3080", "20g-3080"],
    // 实测 `[vram-gov] total=19.63G`(GiB ⇒ 20,101 MiB)
    total_bytes: 20_101 << 20,
    nominal_mib: 20480,
    calibrated: true,
};

pub const HARDWARE: &[HardwareProfile] = &[RTX_3090TI_24G, RTX_3080_20G];

/// 按 `[model].kind` 找注册表档(几何校验用;头部量化不同不影响几何)
pub fn model_by_kind(kind: ModelKind) -> Option<&'static ModelProfile> {
    MODELS.iter().find(|p| p.kind == kind)
}

/// 硬件键 → 档(未命中 = Err 带清单)
pub fn hardware_profile(key: &str) -> Result<&'static HardwareProfile, String> {
    let k = key.trim();
    HARDWARE
        .iter()
        .find(|p| p.key == k || p.aliases.iter().any(|a| *a == k))
        .ok_or_else(|| {
            format!(
                "[intent].hardware = {k:?} 未登记(可用:{:?})",
                HARDWARE.iter().map(|p| p.key).collect::<Vec<_>>()
            )
        })
}

/// 从选卡规格推断硬件(只认名字;BDF/UUID 不含型号 ⇒ 推断失败)
pub fn infer_hardware(device: &crate::device::DeviceSpec) -> Option<&'static HardwareProfile> {
    let name = device.find_name.as_deref()?.to_ascii_lowercase();
    if name.contains("3090") {
        Some(&RTX_3090TI_24G)
    } else if name.contains("3080") {
        Some(&RTX_3080_20G)
    } else {
        None
    }
}

/// 硬件键清单(`--check` 报错用)
pub fn hardware_keys() -> Vec<&'static str> {
    HARDWARE.iter().map(|p| p.key).collect()
}

/// 模型键清单
pub fn model_keys() -> Vec<&'static str> {
    MODELS.iter().map(|p| p.key).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 27B 几何派生值与 `engine::state` 的实测校准逐项相等(防止注册表漂移)
    #[test]
    fn m27b_geometry_matches_engine_calibration() {
        let p = model_profile("qwen3.8-27b-awq").expect("注册表");
        assert_eq!(p.gdn_row_bytes(), 155_713_536, "GDN 单行 = 148.5 MiB(engine 实测)");
        assert_eq!(p.kv_bytes_per_token(true), 32768, "fp8 主池 = 32 KiB/token");
        assert_eq!(p.kv_bytes_per_token(false), 65536, "f16 主池 = 64 KiB/token");
        assert_eq!(p.draft_bytes_per_token(), 20480, "草稿池 bf16 = 20 KiB/token");
        assert_eq!(p.weights_bytes(false), 16_650 << 20);
        assert_eq!(p.weights_bytes(true), (16_650 + 1_290) << 20);
        assert!(p.calibrated);
    }

    /// 0.8B 几何(与 checkpoint `text_config` 对齐;per_tok 由该几何派生)
    #[test]
    fn m08b_geometry() {
        let p = model_profile("0.8b").expect("别名");
        assert_eq!(p.geom.n_full, 6);
        assert_eq!(p.kv_bytes_per_token(true), 6144);
        assert_eq!(p.kv_bytes_per_token(false), 12288);
        assert_eq!(p.gdn_row_bytes(), 19_759_104, "= 18 层 × 274,432 × 4");
        assert!(!p.calibrated);
    }

    #[test]
    fn context_spec_parses_number_and_max() {
        let i: Intent = toml::from_str(
            "model = \"qwen3.8-27b-awq\"\ncontext = 8192\n",
        )
        .expect("数字");
        assert_eq!(i.context, ContextSpec::Tokens(8192));
        let i: Intent = toml::from_str(
            "model = \"qwen3.8-27b-awq\"\ncontext = \"max\"\n",
        )
        .expect("max");
        assert_eq!(i.context, ContextSpec::Max);
        assert!(toml::from_str::<Intent>("model = \"qwen3.8-27b-awq\"\ncontext = 0\n").is_err());
        assert!(toml::from_str::<Intent>("model = \"qwen3.8-27b-awq\"\ncontext = \"MAX!\"\n").is_err());
    }

    #[test]
    fn intent_unknown_key_rejected_with_key_list() {
        let e = toml::from_str::<Intent>(
            "model = \"qwen3.8-27b-awq\"\ncontext = 1024\nmax_seq = 4096\n",
        )
        .expect_err("账外键必须拒");
        let msg = e.to_string();
        assert!(msg.contains("max_seq"), "拒语应点名键: {msg}");
        // Intent::KEYS 是 CLI 报错面;保证非空且包含核心键
        assert!(Intent::KEYS.contains(&"context") && Intent::KEYS.contains(&"model"));
    }
}
