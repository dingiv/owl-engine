//! 配置面 P0 回归:意图档预检 / 生效配置 / 单键解释(纯 CPU,不加载权重)。
//!
//! 判据(设计卷 §11 P0):
//! - `--check` < 2s(实测 10–20ms),不碰 GPU/权重,输出 128k + 内存依据;
//! - `--print-config --format json` 带 provenance / estimate,且可被 JSON 解析;
//! - 旧 flavor 仍可走同一 CLI(无 intent 分支)。
//!
//! 走黑盒(拉起 `CARGO_BIN_EXE_server`),因为配置面只从 `main` 暴露;
//! cwd = `apps/server`,仓库路径由 `CARGO_MANIFEST_DIR` 拼。

use std::process::Command;
use std::time::{Duration, Instant};

fn repo(path: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(path)
}

fn profile(name: &str) -> std::path::PathBuf {
    repo(&format!("profiles/{name}.toml"))
}

fn legacy(name: &str) -> std::path::PathBuf {
    repo(&format!("profiles/legacy/{name}.toml"))
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_server"))
        .args(args)
        .output()
        .expect("拉起 server 二进制")
}

#[test]
fn profile_check_reports_128k_and_memory_basis() {
    let site = repo("testdata/args/base.toml");
    let t0 = Instant::now();
    let out = run(&[
        "-p",
        profile("awq27b-3090-max").to_str().expect("路径"),
        "--site",
        site.to_str().expect("路径"),
        "--check",
    ]);
    let dt = t0.elapsed();
    assert!(
        out.status.success(),
        "check 应成功:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(dt < Duration::from_secs(2), "--check 必须 <2s(实测 {dt:?})");
    let s = String::from_utf8_lossy(&out.stdout);
    for needle in [
        "max_seq=131072",
        "pool_tokens=131072",
        "budget 5.95G",
        "OK(estimated",
        "model dir:",
        "arms: prefill=FI(全桶族)",
    ] {
        assert!(s.contains(needle), "check 输出缺 {needle:?}:\n{s}");
    }
    assert!(!s.contains("FAIL"), "不应有失败行:\n{s}");
}

#[test]
fn print_config_json_carries_provenance_and_estimate() {
    let site = repo("testdata/args/base.toml");
    let out = run(&[
        "-p",
        profile("awq27b-3090-max").to_str().expect("路径"),
        "--site",
        site.to_str().expect("路径"),
        "--print-config",
        "--format",
        "json",
    ]);
    assert!(
        out.status.success(),
        "print-config 应成功:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout 必须是合法 JSON");
    assert_eq!(v["config"]["runtime"]["max_seq"], 131072);
    assert_eq!(v["config"]["pool"]["pool_tokens"], 131072);
    assert_eq!(v["estimate"]["pool_tokens"], 131072);
    // provenance:上下文/池都由 intent 派生(不是文件里的手写值)
    assert_eq!(v["provenance"]["pool.pool_tokens"]["kind"], "derived");
    assert_eq!(v["provenance"]["pool.pool_tokens"]["detail"], "非超卖:池 = 上下文");
    assert_eq!(
        v["intent"]["profile"],
        profile("awq27b-3090-max").to_str().expect("路径")
    );
}

#[test]
fn explain_prints_origin_and_rule() {
    let site = repo("testdata/args/base.toml");
    let out = run(&[
        "-p",
        profile("awq27b-3090-max").to_str().expect("路径"),
        "--site",
        site.to_str().expect("路径"),
        "--explain",
        "runtime.max_seq",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("runtime.max_seq = 131072"), "{s}");
    assert!(s.contains("derived:context=\"max\""), "应给派生规则:\n{s}");
    assert!(s.contains("稳定裕度"), "应给内存依据:\n{s}");
}

#[test]
fn set_overrides_derived_value() {
    let site = repo("testdata/args/base.toml");
    let out = run(&[
        "-p",
        profile("awq27b-3090-max").to_str().expect("路径"),
        "--site",
        site.to_str().expect("路径"),
        "--set",
        "runtime.max_seq=65536",
        "--check",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("max_seq=65536"), "--set 应覆盖派生值:\n{s}");
    assert!(s.contains("pool_tokens=65536"), "池应跟随显式上下文:\n{s}");
}

#[test]
fn legacy_profile_still_works() {
    let site = repo("testdata/args/base.toml");
    let out = run(&[
        "-p",
        legacy("awq27b").to_str().expect("路径"),
        "--site",
        site.to_str().expect("路径"),
        "--check",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("无 [intent]"), "旧档应走兼容分支:\n{s}");
    assert!(s.contains("OK"), "{s}");
}

#[test]
fn check_without_site_fails_loudly_with_report() {
    // 常用档现在是自包含的(带 model.dir);用历史档验证"缺站点资产"的响亮拒。
    let out = run(&[
        "-p",
        legacy("spec-dflash").to_str().expect("路径"),
        "--check",
    ]);
    assert!(!out.status.success(), "缺站点档应失败(27B 无缺省资产)");
    let s = String::from_utf8_lossy(&out.stdout);
    // 报告仍打印(内存账在前),FAIL 行点名病因,末尾不是 OK
    assert!(s.contains("max_seq="), "报告应保留派生结果:\n{s}");
    assert!(s.contains("[check] FAIL: 未配置 [model].dir"), "{s}");
    assert!(s.contains("未通过"), "{s}");
    assert!(!s.contains("[check] OK"), "不应给假绿:\n{s}");
}
