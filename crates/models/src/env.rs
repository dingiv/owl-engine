//! 解释器执行环境(EnvProvider;2026-10-03 抽象落地)。
//!
//! **定位**:解释器执行环境的**完备描述** —— 硬件档、量化方案、KV 策略、
//! 分派旋钮、诊断面,全部一等公民化。默认值齐备([`EnvProvider::default`]
//! = 生产基线:全 opt-in 关、f16 paged KV、无硬件环境),engine / 测试
//! 构造后按需定制,经 [`crate::module::ForwardCtx`] 供给解释器与层声明,
//! 解释器按它执行。
//!
//! **被动律**(charter A):环境必传参,kernels/解释器零探测 —— 本对象是
//! 「引擎 = 事实来源」的载体;std::env 读取**只**发生在 [`EnvProvider::from_env`]
//! (ops 工作流兼容面),组件内部禁止直读环境变量。
//!
//! **单一来源**:
//! - 硬件档 ← iface `op_env()`(引擎 boot 合入;GpuClient = Sm86)
//! - KV 几何 ← [`KvEnv::policy`](原 `kv_paged_policy` 的 dtype 键表收编,
//!   新 KV dtype = 加 KvEnv 构造行,调用点零改动)
//! - 量化方案 ← [`QuantPlan`](构造期注入的同一枚举;装载面显式入口照旧,
//!   env 携带为环境真值,W4A8 一 flag 切换的 REQ-CTX-01 宿主)

use crate::contract::Dtype;
use crate::module::{KvPagedPolicy, QuantPlan};
use owl_kernels::Arch;

/// 硬件环境(被动律:不供给 = [`HwEnv::Cpu`],Call op 结构化拒绝)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HwEnv {
    Cpu,
    Cuda(Arch),
}

/// KV 量化(REQ-CTX-03;None = f16 与 Q/Out 同精度)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvQuant {
    /// f16(默认;与对照口径一致)
    #[default]
    None,
    /// e4m3(FI 软件路径;池/影子 ×0.5;scale 隐式 1.0)
    Fp8E4M3,
}

/// KV 缓存策略(量化 × 布局几何;原 `kv_paged_policy` 键表收编)。
/// 现役 F16 paged 页 32(配对律定谳);新 dtype = 新构造行。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvEnv {
    /// KV 存储量化(None = f16;Fp8E4M3 = e4m3 字节池/影子)
    pub quant: KvQuant,
    /// Q/Out dtype(F16 现役;KV 量化不改 Q/Out 精度)
    pub dtype: Dtype,
    /// 页宽(tokens/block;0 = legacy 直排)
    pub page: usize,
    /// K 向量宽(16B / sizeof(dtype))
    pub x: usize,
    /// paged(false = legacy 直排)
    pub paged: bool,
}

impl KvEnv {
    /// 生产档:f16 paged 页 32(配对律定谳;x = 16B/2B = 8)
    pub fn f16_paged() -> Self {
        Self { quant: KvQuant::None, dtype: Dtype::F16, page: 32, x: 8, paged: true }
    }

    /// classic 布局策略(层/引擎几何消费;None = legacy 回退)
    pub fn policy(&self) -> Option<KvPagedPolicy> {
        if !self.paged {
            return None;
        }
        Some(KvPagedPolicy { page: self.page, x: self.x })
    }
}

/// 注意力分派旋钮(层侧谓词消费;全部默认关 = 生产基线)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttnEnv {
    /// C1-W2 qknorm_rope_kv_insert 单发(已结案;opt-in)
    pub qkv_fuse: bool,
    /// 自研 flash-decoding split(对照臂;FI 在场时优先 FI)
    pub prefill_split: bool,
    /// 诊断二分:强制 naive 回退(paged 数值质量 A/B)
    pub force_naive_prefill: bool,
    /// FlashInfer prefill 面(engine 据此注入 ForwardCtx.fi 影子池/表)
    pub fi: bool,
}

/// 诊断面
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiagEnv {
    /// driver resolve 追踪(OWL_RESOLVE_TRACE)
    pub resolve_trace: bool,
    /// host argmax 对照(OWL_HOST_ARGMAX)
    pub host_argmax: bool,
}

/// GDN 分派旋钮
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdnEnv {
    /// chunked delta rule(FLA AOT 五核;prefill;默认关)
    pub chunked: bool,
    /// 标量门 chunked(lmdeploy pre_sm90 port 单核;prefill;默认关;
    /// 与 chunked 同开时 scalar 优先 —— 层臂三分序 scalar > chunked > recurrence)
    pub scalar: bool,
}

/// 解释器执行环境完备描述(默认 = 生产基线 + 无硬件环境)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EnvProvider {
    pub gdn: GdnEnv,
    pub hw: HwEnv,
    /// 量化方案(F16 / W4A16 / W4A16Awq;W4A8 = REQ-CTX-01 一 flag 宿主)
    pub quant: QuantPlan,
    pub kv: KvEnv,
    pub attn: AttnEnv,
    pub diag: DiagEnv,
}

impl Default for HwEnv {
    fn default() -> Self {
        HwEnv::Cpu
    }
}

impl Default for KvEnv {
    fn default() -> Self {
        KvEnv::f16_paged()
    }
}

impl EnvProvider {
    /// GPU 环境快捷构造(生产基线 + 指定 arch)
    pub fn gpu(arch: Arch) -> Self {
        Self { hw: HwEnv::Cuda(arch), ..Self::default() }
    }

    /// env-var 兼容构造(ops 工作流;组件内部禁止直读环境变量)。
    /// 硬件档不在 env(引擎 boot 从 iface `op_env()` 合入)。
    pub fn from_env() -> Self {
        let has = |k: &str| std::env::var_os(k).is_some();
        Self {
            attn: AttnEnv {
                qkv_fuse: has("OWL_QKV_FUSE"),
                prefill_split: has("OWL_PREFILL_SPLIT"),
                force_naive_prefill: has("OWL_FORCE_NAIVE"),
                fi: has("OWL_FLASHINFER"),
            },
            gdn: GdnEnv { chunked: has("OWL_GDN_CHUNKED"), scalar: has("OWL_GDN_SCALAR") },
            kv: KvEnv { quant: if has("OWL_KV_FP8") { KvQuant::Fp8E4M3 } else { KvQuant::None }, ..KvEnv::default() },
            diag: DiagEnv {
                resolve_trace: has("OWL_RESOLVE_TRACE"),
                host_argmax: has("OWL_HOST_ARGMAX"),
            },
            ..Self::default()
        }
    }

    /// driver OpEnv 投影(kernels 零依赖律:契约类型不过 kernels,
    /// 本处是 env → OpEnv 的唯一投影点)
    pub fn op_env(&self) -> Option<owl_kernels::driver::OpEnv> {
        match self.hw {
            HwEnv::Cpu => None,
            HwEnv::Cuda(arch) => Some(owl_kernels::driver::OpEnv {
                hw: owl_kernels::driver::Hw { arch },
                page: if self.kv.paged { self.kv.page } else { 0 },
            }),
        }
    }
}
