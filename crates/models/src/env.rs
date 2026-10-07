//! 解释器执行环境(EnvProvider;2026-10-03 抽象落地)。
//!
//! **定位**:解释器执行环境的**完备描述** —— 硬件档、量化方案、KV 策略、
//! 分派旋钮、诊断面,全部一等公民化。默认值齐备([`EnvProvider::default`]
//! = 生产基线:全 opt-in 关、f16 paged KV、无硬件环境),engine / 测试
//! 构造后按需定制,经 [`crate::module::ForwardCtx`] 供给解释器与层声明,
//! 解释器按它执行。
//!
//! **被动律**(charter A):环境必传参,kernels/解释器零探测 —— 本对象是
//! 「引擎 = 事实来源」的载体;env 读取统一在 owl_shared::config loader,
//! 本对象经 [`EnvProvider::from_dispatch`] 显式组装
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

    // ── KV 内核选择唯一出口(收口律,2026-10-10)─────────────────
    // 主池/影子的每个写者/读者族一个具名方法;分派点零分支调用。
    // 病史:融合插池 f16 臂 / K0-DUAL classic 臂 / NC 悬空名三次漏网,
    // 根因 = 分派点散布 `if quant == ...`;新增写者 = 新增方法 + 登记表
    // 行 + 清单门(kv_manifest_gate)三件套,禁止在分派点手写分支。

    /// K0 批量写(classic 池;naive decode / chunked prefill / naive prefill 共用)
    pub fn k0_write_op(&self) -> crate::ops::OpId {
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::ids::ATTN_K0_WRITE_FP8,
            KvQuant::None => crate::ops::ids::ATTN_K0_WRITE,
        }
    }

    /// K0 双写(classic + FI kNHD 影子;两臂随 quant 同 dtype)
    pub fn k0_dual_op(&self) -> crate::ops::OpId {
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::ids::ATTN_K0_DUAL_FP8KV,
            KvQuant::None => crate::ops::ids::ATTN_K0_DUAL,
        }
    }

    /// chunked paged prefill 批读
    pub fn prefill_paged_attn_op(&self) -> crate::ops::OpId {
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::ids::ATTN_PAGED_PREFILL_FP8,
            KvQuant::None => crate::ops::ids::ATTN_PAGED_PREFILL,
        }
    }

    /// v2 分页 decode 打分
    pub fn decode_v2_op(&self) -> crate::ops::OpId {
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::ids::ATTN_PAGED_DECODE_V2_FP8,
            KvQuant::None => crate::ops::ids::ATTN_PAGED_DECODE_V2,
        }
    }

    /// decode 融合插池(qk-norm+rope+K/V 插池三合一)
    pub fn fused_insert_op(&self) -> crate::ops::OpId {
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::ids::ATTN_QKV_NORM_ROPE_INSERT_FP8KV,
            KvQuant::None => crate::ops::ids::ATTN_QKV_NORM_ROPE_INSERT,
        }
    }

    /// FI paged prefill 虚核名(Kernel::new 直名族)
    pub fn fi_prefill_name(&self) -> &'static str {
        match self.quant {
            KvQuant::Fp8E4M3 => "flashinfer_prefill_paged_fp8kv",
            KvQuant::None => "flashinfer_prefill_paged_f16",
        }
    }

    /// 主池是否 e4m3 字节承载(池账/几何消费;非内核选择)
    pub fn is_fp8(&self) -> bool {
        self.quant == KvQuant::Fp8E4M3
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
    /// 刀D 时间戳探针(OWL_TS_PROBE;层间 clock64 原地写 ts_buf)
    pub ts_probe: bool,
}

/// GDN 分派旋钮
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GdnEnv {
    /// chunked delta rule(FLA AOT 五核;prefill;默认关)
    pub chunked: bool,
    /// 标量门 chunked(lmdeploy pre_sm90 port 单核;prefill;默认关;
    /// 与 chunked 同开时 scalar 优先 —— 层臂三分序 scalar > chunked > recurrence)
    pub scalar: bool,
    /// D1:decode 整链融合核(v-conv + l2norm×2 + gating + sigmoid(beta) +
    /// delta + norm_act 六算子一发;**生产默认开**。2026-10-03 结案:
    /// 初版 norm_act 误写 sigmoid(旧链生产语义 = silu),单测 host 参考
    /// 同错相消全绿,引擎域 y 逐通道错 1~10 倍致乱码;修 silu 后 27B
    /// e2e greedy/S3 采样双态文本与基线逐字一致,图内 nsys -1.7ms/步;
    /// 取证方法论(状态格画像层扫 + logits dump)OWL_GDN_DUMP/OWL_DEBUG);
    /// OWL_GDN_NO_FUSE_DECODE 反开关退出
    pub fused_decode: bool,
    /// D1-v2:delta 相 float4 行组重写(2026-10-04 sglang 刺探;同契约同
    /// 数学,状态装载 16B/lane 合并;实测靶 ≤10µs vs v1 28.4µs。opt-in
    /// 验证后翻默认)
    pub fused_decode_v2: bool,
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

    /// **入口侧组装器**:自全量配置 dispatch module 显式组装
    /// (owl_shared::config::OwlConfig;组件内部禁止直读环境变量)。
    /// 硬件档不在 config(引擎 boot 从 iface `op_env()` 合入)。
    /// 分派缺省:D1/D1-v2/qkv 融合生产默认开(2026-10-04 定谳)。
    pub fn from_dispatch(d: &owl_shared::config::DispatchCfg) -> Self {
        Self {
            attn: AttnEnv {
                qkv_fuse: d.qkv_fuse,
                prefill_split: d.prefill_split,
                force_naive_prefill: d.force_naive_prefill,
                fi: d.flashinfer,
            },
            gdn: GdnEnv {
                chunked: d.gdn_chunked,
                scalar: d.gdn_scalar,
                fused_decode: d.gdn_fused_decode,
                fused_decode_v2: d.gdn_fused_decode_v2,
            },
            kv: KvEnv {
                quant: if d.kv_fp8 { KvQuant::Fp8E4M3 } else { KvQuant::None },
                ..KvEnv::default()
            },
            diag: DiagEnv {
                resolve_trace: d.resolve_trace,
                host_argmax: d.host_argmax,
                ts_probe: d.ts_probe,
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
