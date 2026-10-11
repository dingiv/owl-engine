//! 生产算子白名单硬门的一致性锁(2026-10-13 P4)。
//!
//! **两处声明必须同票**:
//! - 配置面 = flavor 的 `dispatch.allowed_ops`(TOML 文本);
//! - 权威台账 = `owl_kernels::ops::PROD_{SPEC_DFLASH,AWQ27B}_2026_10_13`
//!   (实测:真跑生产档 + `[cuda] launch_time` 逐发射记账)。
//!
//! 锁死三件事:
//! ① 两份清单逐项一致(改一处必须改另一处,否则红);
//! ② 清单里每个算子都已建表(白名单不许列"仅引入"的算子);
//! ③ 清单通过**池写集穷举审计**(读必有人写 + 声明自洽 + 活跃写者不歧义)。

use owl_shared::config::OwlConfig;

fn whitelist(flavor: &str) -> Vec<&'static str> {
    // flavor 文本是 'static 常量 ⇒ 借出 'static 视图
    let cfg = OwlConfig::load(Some(flavor), None).expect("flavor 解析");
    let list = cfg
        .dispatch
        .allowed_ops
        .unwrap_or_else(|| panic!("{flavor} 必须显式列 dispatch.allowed_ops(生产档硬门)"));
    assert!(!list.is_empty(), "{flavor} 白名单为空 = 门关");
    list.into_iter().collect::<Vec<String>>().leak().iter().map(|s| s.as_str()).collect()
}

#[test]
fn flavor_whitelists_match_audited_tables() {
    for (flavor, want) in [
        ("spec-dflash", owl_kernels::ops::PROD_SPEC_DFLASH_2026_10_13),
        ("awq27b", owl_kernels::ops::PROD_AWQ27B_2026_10_13),
    ] {
        let mut got = whitelist(flavor);
        got.sort_unstable();
        let mut want = want.to_vec();
        want.sort_unstable();
        assert_eq!(got, want, "{flavor}: 白名单与实测台账不一致(改一处须改两处)");
    }
}

#[test]
fn whitelists_are_built_and_pass_pool_audit() {
    for flavor in ["spec-dflash", "awq27b"] {
        let names = whitelist(flavor);
        for n in &names {
            assert!(
                owl_kernels::ops::meta(n).is_some(),
                "{flavor}: 白名单算子 `{n}` 未建表(不许列仅引入的算子)"
            );
        }
        owl_kernels::ops::audit_active(Some(&names))
            .unwrap_or_else(|e| panic!("{flavor}: 池写集审计失败 —— {e}"));
        eprintln!("[whitelist] {flavor}: {} 项,已建表 ✓ 池审计 ✓", names.len());
    }
}
