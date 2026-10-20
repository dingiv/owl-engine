//! **KernelSelectionEnv** —— 引擎 boot 后投影给 picker 的**只读环境子集**。
//!
//! 纪律(被动律):本结构只由引擎在装配期构造一次;`driver::resolve` 只读它,
//! 不读配置/不探测硬件/不碰全局。`EnvProvider` 持有快照,`op_env()` 投影成
//! `owl_kernels::driver::OpEnv`;白名单/锁文件也从同一份快照派生。
//!
//! 这里刻意只放**影响选核**的事实:
//! - `HwCaps`   : 架构(后续可扩 sm_count/smem/tensor-core 能力)
//! - `QuantCaps`: 权重计划 + 激活 dtype
//! - `KvCaps`   : KV 量化/布局/页/几何
//! - `FeatureCaps`: 视觉/投机/TP 等"走哪条语义链"的开关
//!
//! 端口/并发/池大小/日志/采样等**不进**本结构。

use crate::contract::Dtype;
use crate::env::{EnvProvider, HwEnv, KvQuant};
use crate::module::QuantPlan;
use owl_kernels::driver::{Hw, KernelPick, OpEnv, OpId, OpReq, resolve};
use owl_kernels::spec::KernelTag;
use owl_kernels::tag_filter::TagFilter;
use owl_kernels::Arch;

/// 硬件能力(当前只有 arch;新字段 = 新能力,不改调用方)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HwCaps {
    pub arch: Arch,
}

/// 量化能力:权重计划 + 激活/输出 dtype
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantCaps {
    pub plan: QuantPlan,
    pub act: Dtype,
}

/// KV/attention 几何与存储
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCaps {
    pub quant: KvQuant,
    pub dtype: Dtype,
    pub page: usize,
    pub head_dim: usize,
    pub grp: usize,
    pub paged: bool,
}

impl KvCaps {
    /// 从已合入事实的 EnvProvider 投影（主/草稿池共用）
    pub fn from_env(env: &EnvProvider) -> Self {
        Self {
            quant: env.kv.quant,
            dtype: env.kv.dtype,
            page: env.kv.page,
            head_dim: env.kv.head_dim,
            grp: env.kv.grp,
            paged: env.kv.paged,
        }
    }
}

/// 一条待解析的语义 op(白名单派生的输入单元)
#[derive(Clone, Copy, Debug)]
pub struct OpSpec<'a> {
    pub op: &'static str,
    pub dt: Dtype,
    pub shapes: &'a [Vec<usize>],
    pub aux: &'a [usize],
    pub scalars: &'a [i64],
}

/// 影响语义算子选择的 feature 开关
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FeatureCaps {
    pub vision: bool,
    pub spec_depth: usize,
    pub tp_ranks: usize,
    /// lm_head 是否 int8 per-row(`owl_gemv_i8w_*` / `quant.int8` tag)
    pub head_int8: bool,
    /// 主 KV 池事实（None = 用当前 env；引擎 boot 显式给）
    pub main_kv: Option<KvCaps>,
    /// 草稿 KV 池事实（spec.depth>0 时给；None = 无草稿）
    pub draft_kv: Option<KvCaps>,
    /// 诊断通配符：`*` = 跳过 tag 过滤（profile 显式开）
    pub tag_wildcard: bool,
}

/// boot 后冻结的选核环境(不可变;Copy 便于随 `ForwardCtx` 借传)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelSelectionEnv {
    pub hw: HwCaps,
    pub quant: QuantCaps,
    pub kv: KvCaps,
    pub features: FeatureCaps,
    /// **配置派生的选核 tag 过滤**(W4A16/GGUF/KV/vision/spec/tp/head)
    pub tag_filter: TagFilter,
}

impl KernelSelectionEnv {
    /// 从引擎已合入事实的 `EnvProvider` 投影;CPU 档 = `None`。
    pub fn from_env(env: &EnvProvider, features: FeatureCaps) -> Option<Self> {
        let HwEnv::Cuda(arch) = env.hw else {
            return None;
        };
        if features.tag_wildcard {
            return Some(Self {
                hw: HwCaps { arch },
                quant: QuantCaps { plan: env.quant, act: env.kv.dtype },
                kv: KvCaps::from_env(env),
                features,
                tag_filter: TagFilter::wildcard(),
            });
        }
        let mut tags: Vec<KernelTag> = Vec::new();
        match env.quant {
            QuantPlan::W4A16 => tags.push(KernelTag::QUANT_W4A16),
            QuantPlan::W4A16Awq => {
                tags.push(KernelTag::QUANT_W4A16);
                tags.push(KernelTag::QUANT_AWQ);
            }
            QuantPlan::W4A8Awq => {
                tags.push(KernelTag::QUANT_W4A8);
                tags.push(KernelTag::QUANT_AWQ);
            }
            QuantPlan::Gguf(_) => tags.push(KernelTag::QUANT_GGUF),
            QuantPlan::F16 | QuantPlan::I8W => {}
        }
        let kv_tag = |kv: KvCaps, main: bool| -> KernelTag {
            match kv.quant {
                KvQuant::Fp8E4M3 | KvQuant::Fp8E5M2 => {
                    if main { KernelTag::KV_MAIN_FP8 } else { KernelTag::KV_DRAFT_FP8 }
                }
                KvQuant::None => match kv.dtype {
                    Dtype::BF16 => {
                        if main { KernelTag::KV_MAIN_BF16 } else { KernelTag::KV_DRAFT_BF16 }
                    }
                    _ => {
                        if main { KernelTag::KV_MAIN_F16 } else { KernelTag::KV_DRAFT_F16 }
                    }
                },
            }
        };
        let main_kv = features.main_kv.unwrap_or_else(|| KvCaps::from_env(env));
        tags.push(kv_tag(main_kv, true));
        if features.spec_depth > 0 {
            if let Some(d) = features.draft_kv {
                tags.push(kv_tag(d, false));
            }
        }
        if features.vision {
            tags.push(KernelTag::FEAT_VISION);
        }
        if features.spec_depth > 0 {
            tags.push(KernelTag::FEAT_DFLASH);
            tags.push(KernelTag::FEAT_SPEC);
        }
        if features.tp_ranks > 1 {
            tags.push(KernelTag::FEAT_TP);
        }
        if features.head_int8 {
            tags.push(KernelTag::QUANT_INT8);
            tags.push(KernelTag::FEAT_HEAD);
        }
        Some(Self {
            hw: HwCaps { arch },
            quant: QuantCaps {
                plan: env.quant,
                act: env.kv.dtype,
            },
            kv: KvCaps {
                quant: env.kv.quant,
                dtype: env.kv.dtype,
                page: env.kv.page,
                head_dim: env.kv.head_dim,
                grp: env.kv.grp,
                paged: env.kv.paged,
            },
            features,
            tag_filter: TagFilter::from_tags(&tags),
        })
    }

    /// picker 直接消费的投影(唯一投影点;`OpEnv` 保持最小)
    pub fn op_env(&self) -> OpEnv {
        OpEnv {
            hw: Hw { arch: self.hw.arch },
            page: if self.kv.paged { self.kv.page } else { 0 },
            head_dim: self.kv.head_dim,
            grp: self.kv.grp,
        }
    }

    /// **解析一条语义 op**(白名单派生的最小单元;picker 仍是纯函数)
    pub fn resolve(&self, spec: &OpSpec<'_>) -> Result<KernelPick, crate::contract::ModelError> {
        let env = self.op_env();
        let dt = crate::ops::ddt(spec.dt)?;
        let pick = resolve(OpReq {
            op: OpId(spec.op),
            env: &env,
            dt,
            shapes: spec.shapes,
            aux: spec.aux,
            scalars: spec.scalars,
        });
        let meta = pick.meta().ok_or_else(|| crate::contract::ModelError::Msg(format!(
            "算子 `{}` 未建表 —— 无法做 tag 过滤(生产档拒)", pick.name
        )))?;
        if let Some(facet) = self.tag_filter.denied_facet(meta) {
            return Err(crate::contract::ModelError::Msg(format!(
                "算子 `{}` 被 tag filter 拒绝(facet={facet}; allowed=[{}]; tags={:?})",
                meta.entry,
                self.tag_filter.describe(),
                meta.tag_ids()
            )));
        }
        Ok(pick)
    }

    /// 批量解析并收集实现名(白名单/锁文件的输入;去重排序)
    pub fn collect_ops(
        &self,
        specs: &[OpSpec<'_>],
    ) -> Result<std::collections::BTreeSet<&'static str>, crate::contract::ModelError> {
        specs
            .iter()
            .map(|s| self.resolve(s).map(|p| p.name))
            .collect()
    }

    /// boot 日志一行(可审计:这个进程到底把什么投射给了 picker)
    pub fn describe(&self) -> String {
        format!(
            "arch={:?} quant={:?} act={:?} kv={:?}/{:?} page={} hd={} grp={} vision={} spec_depth={} tp={} tags=[{}]",
            self.hw.arch,
            self.quant.plan,
            self.quant.act,
            self.kv.quant,
            self.kv.dtype,
            self.kv.page,
            self.kv.head_dim,
            self.kv.grp,
            self.features.vision,
            self.features.spec_depth,
            self.features.tp_ranks,
            self.tag_filter.describe(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{EnvProvider, HwEnv, KvEnv, KvQuant};

    #[test]
    fn cpu_has_no_kernel_env() {
        let env = EnvProvider::default();
        assert!(KernelSelectionEnv::from_env(&env, FeatureCaps::default()).is_none());
    }

    #[test]
    fn cuda_projection_is_minimal_and_complete() {
        let env = EnvProvider {
            hw: HwEnv::Cuda(Arch::Sm86),
            kv: KvEnv {
                quant: KvQuant::Fp8E4M3,
                page: 32,
                head_dim: 256,
                grp: 6,
                ..KvEnv::default()
            },
            ..EnvProvider::default()
        };
        let features = FeatureCaps { vision: true, spec_depth: 7, tp_ranks: 2, head_int8: true, ..FeatureCaps::default() };
        let k = KernelSelectionEnv::from_env(&env, features).expect("CUDA");
        assert_eq!(k.hw.arch, Arch::Sm86);
        assert_eq!(k.quant.plan, env.quant);
        assert_eq!(k.kv.quant, KvQuant::Fp8E4M3);
        assert_eq!(k.kv.page, 32);
        assert_eq!(k.kv.head_dim, 256);
        assert_eq!(k.kv.grp, 6);
        assert_eq!(k.features, features);
        let oe = k.op_env();
        assert_eq!(oe.page, 32);
        assert_eq!(oe.head_dim, 256);
        assert_eq!(oe.grp, 6);
    }

    #[test]
    fn collect_ops_from_snapshot_uses_picker() {
        let env = EnvProvider {
            hw: HwEnv::Cuda(Arch::Sm86),
            ..EnvProvider::default()
        };
        let k = KernelSelectionEnv::from_env(
            &env,
            FeatureCaps { vision: true, ..FeatureCaps::default() },
        )
        .expect("CUDA");
        let s = [vec![128usize, 1152usize]];
        let specs = [
            OpSpec { op: "vis.layernorm", dt: Dtype::F16, shapes: &s, aux: &[], scalars: &[] },
            OpSpec { op: "vis.bias_add", dt: Dtype::F16, shapes: &s, aux: &[], scalars: &[] },
            OpSpec { op: "vis.gelu_tanh", dt: Dtype::F16, shapes: &s, aux: &[], scalars: &[] },
            OpSpec { op: "vis.gelu_erf", dt: Dtype::F16, shapes: &s, aux: &[], scalars: &[] },
        ];
        let got = k.collect_ops(&specs).expect("collect");
        for want in [
            "owl_layernorm_f16",
            "owl_bias_add_f16",
            "owl_gelu_tanh_f16",
            "owl_gelu_erf_f16",
        ] {
            assert!(got.contains(want), "缺 {want}: {got:?}");
        }
    }

    #[test]
    fn config_derived_tag_filter_matches_picker_family() {
        use crate::module::QuantPlan;
        let env = EnvProvider {
            hw: HwEnv::Cuda(Arch::Sm86),
            quant: QuantPlan::W4A16Awq,
            kv: KvEnv {
                quant: KvQuant::Fp8E4M3,
                page: 32,
                head_dim: 256,
                grp: 6,
                ..KvEnv::default()
            },
            ..EnvProvider::default()
        };
        let k = KernelSelectionEnv::from_env(
            &env,
            FeatureCaps { vision: true, spec_depth: 7, tp_ranks: 2, head_int8: true, ..FeatureCaps::default() },
        )
        .expect("CUDA");
        let allow = |entry: &str| {
            let m = owl_kernels::ops::meta(entry).unwrap_or_else(|| panic!("缺 {entry}"));
            k.tag_filter.allows(m)
        };
        assert!(allow("marlin_gemm_w4a16_awq"), "W4A16 核应放行");
        assert!(!allow("marlin_gemm_w4a8_awq"), "W4A8 核应拒");
        assert!(!allow("owl_gguf_gemm_q4k"), "GGUF 核应拒");
        assert!(allow("vllm_paged_attention_v2_fp8_knhd_hd256bs32"));
        assert!(!allow("vllm_paged_attention_v2_f16_knhd_hd256bs32"));
        assert!(allow("owl_rmsnorm_f16"), "通用核无 quant/kv tag ⇒ 放行");
        assert!(allow("owl_gemv_i8w_f16"), "head int8 facet 命中");
    }

    #[test]
    fn legacy_non_paged_page_is_zero() {
        let env = EnvProvider {
            hw: HwEnv::Cuda(Arch::Sm86),
            kv: KvEnv { paged: false, page: 64, ..KvEnv::default() },
            ..EnvProvider::default()
        };
        let k = KernelSelectionEnv::from_env(&env, FeatureCaps::default()).expect("CUDA");
        assert_eq!(k.op_env().page, 0, "legacy 直排 page 传 0");
    }
}
