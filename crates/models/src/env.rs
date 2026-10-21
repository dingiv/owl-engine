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
//! 本对象经 [`EnvProvider::from_config`] 显式组装
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
#[derive(Default)]
pub enum HwEnv {
    #[default]
    Cpu,
    Cuda(Arch),
}

/// KV 量化(REQ-CTX-03;None = f16 与 Q/Out 同精度)
///
/// **FP8 的两种格式并列**(2026-10-16):e4m3 与 e5m2 是**两个格式**,
/// 不许糊成一个"FP8";`dtype()` 是"配置 → 类型"的**单一映射**(禁 bool 传格式)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvQuant {
    /// f16(默认;与对照口径一致)
    #[default]
    None,
    /// **e4m3**(FI 软件路径;池/影子 ×0.5;scale 隐式 1.0)
    Fp8E4M3,
    /// **e5m2**(范围更大、精度更低;与 e4m3 并列的另一种 fp8 ——
    /// 暂无核臂(选了会在臂选择处**响亮拒**),先让类型体系能表达)
    Fp8E5M2,
}

#[cfg(test)]
mod kvquant_tests {
    use super::*;

    /// **配置 → 类型 的单一映射门**(2026-10-16):FP8 的两种格式**并列**,
    /// 且字节数由枚举自己给 —— 禁"用 bool 传格式"(那正是 F16/BF16 案的翻版)。
    #[test]
    fn fp8_formats_are_two_and_single_sourced() {
        use crate::contract::Dtype;
        assert_eq!(KvQuant::None.dtype(), Dtype::F16);
        assert_eq!(KvQuant::None.size_bytes(), 2);
        // 两种 fp8 **并列**,不是"FP8"一个东西
        assert_eq!(KvQuant::Fp8E4M3.dtype(), Dtype::F8E4M3);
        assert_eq!(KvQuant::Fp8E5M2.dtype(), Dtype::F8E5M2);
        assert_ne!(
            KvQuant::Fp8E4M3.dtype(),
            KvQuant::Fp8E5M2.dtype(),
            "e4m3 与 e5m2 必须是两个 dtype"
        );
        for q in [KvQuant::Fp8E4M3, KvQuant::Fp8E5M2] {
            assert_eq!(q.size_bytes(), 1, "fp8 一律 1B/elem");
        }
        // 与 int8 不同族(不同 dtype)
        assert_ne!(Dtype::F8E4M3, Dtype::I8);
        assert_eq!(Dtype::I8.size_bytes(), 1);
    }
}

impl KvQuant {
    /// 配置 → **存储/声明 dtype**(单一映射;字节算术、池声明、臂选择都取自这里)
    pub fn dtype(self) -> crate::contract::Dtype {
        use crate::contract::Dtype;
        match self {
            KvQuant::None => Dtype::F16,
            KvQuant::Fp8E4M3 => Dtype::F8E4M3,
            // 类型映射**必须**能表达两种 fp8(拒的是"核臂"那一层:见
            // state.rs 的启动门与 KvEnv 的读者选择子)
            KvQuant::Fp8E5M2 => Dtype::F8E5M2,
        }
    }

    /// 每元素字节(1B = fp8 两族;2B = f16)
    pub fn size_bytes(self) -> usize {
        self.dtype().size_bytes()
    }
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
    /// K 向量宽(16B / sizeof(dtype))= **classic 布局专参**(P6-C 随草稿池
    /// 翻 kNHD 一并退役;当前仅 `dflash2::pool_geom` 读它)
    pub x: usize,
    /// paged(false = legacy 直排)
    pub paged: bool,
    /// 注意力头维(**模型几何事实**;0 = 未知 ⇒ 拾取期几何门跳过该项)。
    /// 由引擎 boot 从 `ModelDims` 合入 —— 池几何 [nb,page,hkv,hd] 的一半,
    /// 也是算子 `ShapeFamily.head_dim` 声明的对账面。
    pub head_dim: usize,
    /// GQA 分组比 hq/hkv(0 = 未知/不适用)
    pub grp: usize,
}

/// kV 池内元素偏移(kNHD 单布局,唯一真源):[nb, page, hkv, hd] 页内连续
///
/// 2026-10-13(P6-A):classic 臂与其 x-block 交错数学已删 —— 统一契约终态
/// 是**单布局零枚举**(roadmap.local/kv布局统一契约 §2.2 D1 大爆炸切换)。
/// 探针/诊断面回读偏移仍走本函数(布局单源,核层零字面量索引)。
pub fn knhd_offset(slot: usize, h: usize, d: usize, hkv: usize, hd: usize) -> usize {
    (slot * hkv + h) * hd + d
}

impl KvEnv {
    /// 生产档:f16 paged 页 32(配对律定谳)
    ///
    /// **2026-10-13 kNHD 单布局**(统一契约 §2.2 D1 大爆炸切换;P6-A 已删
    /// classic 臂与布局枚举 —— 终态单布局零枚举)。boot 静态定形。
    pub fn f16_paged() -> Self {
        Self {
            quant: KvQuant::None,
            dtype: Dtype::F16,
            page: 32,
            x: 8,
            paged: true,
            // 模型几何:boot 由 ModelDims 合入(缺省 0 = 未知,门跳过)
            head_dim: 0,
            grp: 0,
        }
    }

    // ── KV 内核选择唯一出口(收口律,2026-10-10)─────────────────
    // 主池/影子的每个写者/读者族一个具名方法;分派点零分支调用。
    // 病史:融合插池 f16 臂 / K0-DUAL classic 臂 / NC 悬空名三次漏网,
    // 根因 = 分派点散布 `if quant == ...`;新增写者 = 新增方法 + 登记表
    // 行 + 清单门(kv_manifest_gate)三件套,禁止在分派点手写分支。

    /// K0 批量写(classic 池;naive decode / chunked prefill / naive prefill 共用)
    pub fn k0_write_op(&self) -> crate::ops::SemanticKernel {
        // 单布局(P6-A):knhd 名唯一(臂/名成对律见 kernels::list 不变量门)
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::SemanticKernel::K0WriteFp8Knhd,
            // e5m2 与 e4m3 并列但**暂无核臂** ⇒ 响亮拒(不静默当 e4m3 跑)
            KvQuant::Fp8E5M2 => unimplemented!("fp8-e5m2 暂无核臂(本仓只有 e4m3)"),
            KvQuant::None => crate::ops::SemanticKernel::K0WriteKnhd,
        }
    }

    /// chunked paged prefill 批读
    pub fn prefill_paged_attn_op(&self) -> crate::ops::SemanticKernel {
        // 单布局(P6-A):knhd 名唯一(臂/名成对律见 kernels::list 不变量门)
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::SemanticKernel::PagedPrefillFp8Knhd,
            // e5m2 与 e4m3 并列但**暂无核臂** ⇒ 响亮拒(不静默当 e4m3 跑)
            KvQuant::Fp8E5M2 => unimplemented!("fp8-e5m2 暂无核臂(本仓只有 e4m3)"),
            KvQuant::None => crate::ops::SemanticKernel::PagedPrefillKnhd,
        }
    }

    /// prefill split 读(flash-decoding;fp8 池必须走 fp8kv 直读 —— f16 核
    /// 读 e4m3 字节池 = 字节错位,2026-10-12 埋雷实录)
    pub fn prefill_split_op(&self) -> crate::ops::SemanticKernel {
        // 单布局(P6-A):knhd 名唯一(臂/名成对律见 kernels::list 不变量门)
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::SemanticKernel::PrefillSplitFp8kvKnhd,
            // e5m2 与 e4m3 并列但**暂无核臂** ⇒ 响亮拒(不静默当 e4m3 跑)
            KvQuant::Fp8E5M2 => unimplemented!("fp8-e5m2 暂无核臂(本仓只有 e4m3)"),
            KvQuant::None => crate::ops::SemanticKernel::PrefillSplitKnhd,
        }
    }

    /// v2 分页 decode 打分
    pub fn decode_v2_op(&self) -> crate::ops::SemanticKernel {
        // 单布局(P6-A):knhd 名唯一(臂/名成对律见 kernels::list 不变量门)
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::SemanticKernel::PagedDecodeV2Fp8Knhd,
            // e5m2 与 e4m3 并列但**暂无核臂** ⇒ 响亮拒(不静默当 e4m3 跑)
            KvQuant::Fp8E5M2 => unimplemented!("fp8-e5m2 暂无核臂(本仓只有 e4m3)"),
            KvQuant::None => crate::ops::SemanticKernel::PagedDecodeV2Knhd,
        }
    }

    /// verify 伪序列变体(grid.y = seqs;§三十一)
    pub fn decode_v2_seq_op(&self) -> crate::ops::SemanticKernel {
        // 单布局(P6-A):knhd 名唯一(臂/名成对律见 kernels::list 不变量门)
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::SemanticKernel::PagedDecodeV2Fp8SeqKnhd,
            // e5m2 与 e4m3 并列但**暂无核臂** ⇒ 响亮拒(不静默当 e4m3 跑)
            KvQuant::Fp8E5M2 => unimplemented!("fp8-e5m2 暂无核臂(本仓只有 e4m3)"),
            KvQuant::None => crate::ops::SemanticKernel::PagedDecodeV2SeqKnhd,
        }
    }

    /// decode 融合插池(qk-norm+rope+K/V 插池三合一)
    pub fn fused_insert_op(&self) -> crate::ops::SemanticKernel {
        // 单布局(P6-A):knhd 名唯一(臂/名成对律见 kernels::list 不变量门)
        match self.quant {
            KvQuant::Fp8E4M3 => crate::ops::SemanticKernel::QkvNormRopeInsertFp8kvKnhd,
            // e5m2 与 e4m3 并列但**暂无核臂** ⇒ 响亮拒(不静默当 e4m3 跑)
            KvQuant::Fp8E5M2 => unimplemented!("fp8-e5m2 暂无核臂(本仓只有 e4m3)"),
            KvQuant::None => crate::ops::SemanticKernel::QkvNormRopeInsertKnhd,
        }
    }

    /// FI paged prefill 虚核名(Kernel::new 直名族;字面量住址 =
    /// `owl_kernels::family::flashinfer`,零本地字面量)
    pub fn fi_prefill_name(&self) -> &'static str {
        match self.quant {
            KvQuant::Fp8E4M3 => owl_kernels::family::flashinfer::PREFILL_FI_FP8KV,
            // e5m2 与 e4m3 并列但**暂无核臂** ⇒ 响亮拒(不静默当 e4m3 跑)
            KvQuant::Fp8E5M2 => unimplemented!("fp8-e5m2 暂无核臂(本仓只有 e4m3)"),
            KvQuant::None => owl_kernels::family::flashinfer::PREFILL_FI,
        }
    }

    /// KV 池 head stride(单布局 = hd;页内 [page,hkv,hd] 连续。
    /// classic 的 hd·page 随 P6-A 退役)
    pub fn kv_head_stride(&self, hd: usize, _page: i32) -> i32 {
        hd as i32
    }
    pub fn kv_block_stride(&self, hkv: usize, hd: usize, page: i32) -> i32 {
        hkv as i32 * hd as i32 * page
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
    /// **verify 臂(单值;2026-10-21 取代 verify_v2/verify_xqa/verify_fi 三布尔)**
    ///
    /// 图层按本字段**线性**选臂(每级能力门不满足 → 下一级),不再是
    /// 「三个布尔谁先命中」;`plain` 缺省 = `V2`(生产基线零变化)。
    pub verify_arm: owl_shared::config::VerifyArm,
    /// **decode 臂(单值;取代 decode_fi/decode_xqa)**
    pub decode_arm: owl_shared::config::DecodeArm,
    /// FI 接管的 T 下限(`dispatch.fi_min_tokens`;t ≥ 此值走 FI 主臂)
    pub fi_min_tokens: usize,
    /// prefill(chunked q=T)走 XQA(同一份 `M_TILESIZE=32` 实例化;缺省 false =
    /// 只作 FI 缺口交接臂 —— 主路径换它净亏 4.1×,见 `DispatchCfg::xqa_prefill`)
    pub xqa_prefill: bool,
    /// FlashInfer prefill 面 —— **生产恒真**(2026-10-13:配置面开关已移除,
    /// 没有不启用的情况)。本字段保留仅供**门/诊断**做读核 A/B
    /// (engine::tests::gpu_longctx_fi_direct_gate 用它切 FI/自研读核)
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
    /// 解释器活性收尾归因打印(`[probes] live_diag`;原 OWL_LIVE_DIAG)
    pub live_diag: bool,
}

/// 解释器内存治理(`[pool]`;原 `OWL_BLOCK_POOL` / `OWL_NO_RECLAIM` env)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterpEnv {
    /// 块池预算字节(0 = 关 = 逐节点 alloc/free)
    pub block_pool_bytes: usize,
    /// 关活性中途回收(诊断)
    pub no_reclaim: bool,
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

/// 解释器求值阶段(性能面定相;docs/arch/perf-framework.md §四)。
///
/// 用途:性能钩子的 tag 前缀段 —— `interp.host.node.<phase>.<class>`。
/// 定相不靠线程栈(span 跨 await 有错位风险),而由引擎逐相位赋值。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EvalPhase {
    /// 未定相(eager 诊断、测试、装载面)
    #[default]
    Unk,
    /// 预填充 chunk
    Prefill,
    /// 裸 decode 步
    Decode,
    /// 投机轮(verify + propose/encode)
    Spec,
    /// 装载域(权重转换/repack)
    Load,
}

impl EvalPhase {
    /// tag 前缀段(有界词表)
    pub fn tag(self) -> &'static str {
        match self {
            Self::Unk => "unk",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Spec => "spec",
            Self::Load => "load",
        }
    }
}

/// **GGUF 块型臂的旋钮面**(models 侧消费;配置真源 = `[kernel]`/`[dispatch]`)
///
/// 纯数据、值语义;`Default` = 生产基线(五臂全开、行窗走预算档)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GgufEnv {
    /// 小 n 走 MMVQ 的张量字节上限(MiB;`0` = 关)
    /// —— 语义见 `owl_shared::config::KernelCfg::gguf_mmvq_l2_mib`
    pub mmvq_l2_mib: usize,
    pub tile: bool,
    pub cublas: bool,
    pub mmq2: bool,
    pub mmq: bool,
    pub smem: bool,
    /// cuBLAS 臂行窗行数(None = 96 MiB 预算档)
    pub cublas_tile: Option<usize>,
}

impl Default for GgufEnv {
    fn default() -> Self {
        Self {
            mmvq_l2_mib: 4,
            tile: true,
            cublas: true,
            mmq2: true,
            mmq: true,
            smem: true,
            cublas_tile: None,
        }
    }
}

/// 解释器执行环境完备描述(默认 = 生产基线 + 无硬件环境)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EnvProvider {
    pub gdn: GdnEnv,
    /// GGUF 块型臂旋钮(小 n 判据等)
    pub gguf: GgufEnv,
    pub hw: HwEnv,
    /// 量化方案(F16 / W4A16 / W4A16Awq;W4A8 = REQ-CTX-01 一 flag 宿主)
    pub quant: QuantPlan,
    pub kv: KvEnv,
    pub attn: AttnEnv,
    pub diag: DiagEnv,
    /// 解释器内存治理(块池预算 / 回收开关)
    pub interp: InterpEnv,
    /// 求值阶段(性能面定相;不影响计算语义)
    pub phase: EvalPhase,
    /// **boot 后冻结的选核环境快照**(引擎装配期填;None = 旧路/CPU)
    pub kernel_env: Option<crate::kernel_env::KernelSelectionEnv>,
}

impl EnvProvider {
    /// 阶段切换(值语义;引擎逐相位赋值 —— 显式依赖律)
    pub fn with_phase(self, p: EvalPhase) -> Self {
        Self { phase: p, ..self }
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

    /// **入口侧组装器**:自全量配置对象显式组装(唯一构造口)
    /// (owl_shared::config::OwlConfig;组件内部禁止直读环境变量)。
    /// 硬件档不在 config(引擎 boot 从 iface `op_env()` 合入)。
    /// 分派缺省:D1/D1-v2/qkv 融合生产默认开(2026-10-04 定谳)。
    pub fn from_config(cfg: &owl_shared::config::OwlConfig) -> Self {
        let d = &cfg.kernel;
        Self {
            attn: AttnEnv {
                qkv_fuse: d.qkv_fuse,
                prefill_split: d.prefill_split,
                force_naive_prefill: d.force_naive_prefill,
                verify_arm: d.verify_arm,
                decode_arm: d.decode_arm,
                fi_min_tokens: d.fi_min_tokens,
                xqa_prefill: d.xqa_prefill,
                // FI **恒启用**(2026-10-13 用户令:移除开关,没有不启用的情况;
                // 原 dispatch.flashinfer 已删)。
                // ⚠️ 诊断 A/B(2026-10-14 A13 立案):`OWL_DIAG_NO_FI=1` 只服务
                // 「FI 读核 vs 自研读核」同 boot 同分块的单变量对照(缺省 = 恒启用,
                // 生产语义零变化;确认病灶后可撤)
                fi: !owl_shared::env_reader::flag("OWL_DIAG_NO_FI"),
            },
            gguf: GgufEnv {
                mmvq_l2_mib: d.gguf_mmvq_l2_mib,
                tile: d.gguf_tile,
                cublas: d.gguf_cublas,
                mmq2: d.gguf_mmq2,
                mmq: d.gguf_mmq,
                smem: d.gguf_smem,
                cublas_tile: d.gguf_cublas_tile,
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
                live_diag: cfg.probes.live_diag,
            },
            interp: InterpEnv {
                block_pool_bytes: cfg.pool.block_pool_bytes,
                no_reclaim: cfg.pool.no_reclaim,
            },
            ..Self::default()
        }
    }

    /// driver OpEnv 投影(kernels 零依赖律:契约类型不过 kernels,
    /// 本处是 env → OpEnv 的唯一投影点)
    pub fn op_env(&self) -> Option<owl_kernels::driver::OpEnv> {
        if let Some(k) = self.kernel_env {
            return Some(k.op_env());
        }
        match self.hw {
            HwEnv::Cpu => None,
            HwEnv::Cuda(arch) => Some(owl_kernels::driver::OpEnv {
                hw: owl_kernels::driver::Hw { arch },
                page: if self.kv.paged { self.kv.page } else { 0 },
                head_dim: self.kv.head_dim,
                grp: self.kv.grp,
            }),
        }
    }
}

#[cfg(test)]
mod kv_geometry_tests {
    //! kV 几何契约门(P6-A 单布局版:探针/诊断回读偏移的唯一真源)

    use super::*;

    /// 偏移双射:同逻辑元素集 (slot,h,d) 在池内唯一且不碰撞
    #[test]
    fn 偏移映射_双射性() {
        let (hkv, hd, slots) = (4usize, 256usize, 96usize);
        let n = slots * hkv * hd;
        let mut seen = vec![false; n];
        for slot in 0..slots {
            for h in 0..hkv {
                for d in 0..hd {
                    let o = knhd_offset(slot, h, d, hkv, hd);
                    assert!(o < n && !seen[o], "偏移碰撞 off={o}");
                    seen[o] = true;
                }
            }
        }
        assert!(seen.iter().all(|&b| b), "偏移未覆盖池");
    }

    /// 页内序 = [page, hkv, hd](kNHD 契约);块内偏移 = 槽内偏移 + 页序×页宽
    #[test]
    fn 页内序与块内算术一致() {
        let (hkv, hd, page) = (4usize, 256usize, 32usize);
        for slot in [0usize, 1, 31, 32, 33, 95] {
            let (b, p) = (slot / page, slot % page);
            for &(h, d) in &[(0usize, 0usize), (1, 7), (3, hd - 1)] {
                // 池内偏移 = 块内偏移(块号×块 stride)+ 页序×页内 stride + 头/维偏移
                let expect = b * (hkv * hd * page) + p * (hkv * hd) + h * hd + d;
                assert_eq!(knhd_offset(slot, h, d, hkv, hd), expect, "slot{slot} h{h} d{d}");
            }
        }
    }

}
