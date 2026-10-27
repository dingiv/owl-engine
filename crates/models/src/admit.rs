//! **代际准入门(引擎侧投影)** —— 把「当前配置」投影成 [`OpFamily`] 并逐项裁决。
//!
//! 分工:
//! - 判据表住 `owl_kernels::admit`(零依赖单源);
//! - **映射**住这里(`QuantPlan` / `EnvProvider` 都属 models ⇒ 不把引擎概念塞进 kernels);
//! - 调用点 = 引擎 boot(拒启)+ 预检(可选)。
//!
//! 纪律(与 `admit` 表一致):`Deny` 一律带"为什么 + 怎么办",直接进拒启信息。

use owl_kernels::{Admit, AdmitCtx, OpFamily};

use crate::env::{EnvProvider, HwEnv};
use crate::module::QuantPlan;

/// 权重格式 → 算子族(单一映射;新增 plan 必须在此表态)
pub fn family_of_plan(p: QuantPlan) -> OpFamily {
    match p {
        QuantPlan::W4A16 | QuantPlan::W4A16Awq => OpFamily::MarlinW4A16,
        QuantPlan::W4A8Awq => OpFamily::MarlinW4A8,
        QuantPlan::Gguf(_) => OpFamily::GgufMmq,
        QuantPlan::F16 | QuantPlan::I8W | QuantPlan::I8W8 => OpFamily::CublasF16,
    }
}

/// 环境 → 当前会用到哪些算子族(顺序即日志顺序)
pub fn families_of_env(env: &EnvProvider, plan: Option<QuantPlan>) -> Vec<(&'static str, OpFamily)> {
    let mut v: Vec<(&'static str, OpFamily)> = Vec::new();
    if let Some(p) = plan {
        let f = family_of_plan(p);
        v.push((f.name(), f));
    }
    let gdn = if env.gdn.scalar {
        OpFamily::GdnScalar
    } else if env.gdn.chunked {
        OpFamily::GdnChunked
    } else {
        OpFamily::GdnScalar
    };
    v.push((gdn.name(), gdn));
    if env.attn.fi {
        v.push((OpFamily::FiPrefill.name(), OpFamily::FiPrefill));
    }
    v
}

/// 逐项裁决。`Ok(summary)` = 一行摘要(进启动日志);`Err(why)` = **拒启**(带原因与替代)。
///
/// `spec_depth` 参与裁决:GDN chunked 只在 spec verify(T>1)形状上在 sm_120 出错,
/// 所以同一族在 `spec_depth = 0` 与 `> 0` 下结论不同 —— 这正是 [`AdmitCtx`] 存在的理由。
pub fn check_env(
    env: &EnvProvider,
    plan: Option<QuantPlan>,
    spec_depth: usize,
) -> Result<String, String> {
    let arch = match env.hw {
        HwEnv::Cuda(a) => a,
        // 纯 host 面(CPU/mock/无卡测试):没有硬件档 ⇒ 门关
        HwEnv::Cpu => return Ok("host(无硬件档,门关)".to_string()),
    };
    let ctx = AdmitCtx::of(arch).with_spec(spec_depth);
    let mut brief: Vec<String> = vec![match plan {
        Some(_) => String::new(),
        // GGUF 的计划是**逐张量**的(`PlanSource::Keyed`),单值 plan 为 None;
        // 该族的准入由 loader 的逐型拒启兜底,这里只做标注。
        None => "plan=逐张量(loader 逐型校验)".to_string(),
    }]
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect();
    for (name, fam) in families_of_env(env, plan) {
        let a = fam.admit(ctx);
        if let Admit::Deny(why) = a {
            return Err(format!(
                "代际准入门:{name} 不允许在 {arch} 上跑 —— {why}"
            ));
        }
        brief.push(format!("{name}={}", a.describe()));
    }
    // XQA 臂(kNHD fp8 池的 decode/verify);当前 GGUF 档不走,但臂可被配置选中
    if matches!(
        format!("{:?}{:?}", env.attn.verify_arm, env.attn.decode_arm).as_str(),
        s if s.contains("Xqa")
    ) {
        match OpFamily::Xqa.admit(ctx) {
            Admit::Deny(why) => {
                return Err(format!("代际准入门:xqa 不允许在 {arch} 上跑 —— {why}"))
            }
            a => brief.push(format!("xqa={}", a.describe())),
        }
    }
    Ok(format!("{arch} · spec={spec_depth} ⇒ {}", brief.join(" · ")))
}
