//! 配置面 P0 回归:意图档预检 / 生效配置 / 单键解释(纯 CPU,不加载权重)。
//!
//! 判据(设计卷 §11 P0):
//! - `--check` < 2s(实测 10–20ms),不碰 GPU/权重,输出 max_seq + 内存依据;
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

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_server"))
        .args(args)
        .output()
        .expect("拉起 server 二进制")
}

#[test]
fn profile_check_reports_max_seq_and_memory_basis() {
    let t0 = Instant::now();
    let out = run(&[
        "-p",
        profile("qwen38_27b-3090").to_str().expect("路径"),
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
        "max_seq=79872",
        "pool=auto",
        "OK(estimated",
        "model dir:",
        "arms: prefill=eager",
    ] {
        assert!(s.contains(needle), "check 输出缺 {needle:?}:\n{s}");
    }
    assert!(!s.contains("FAIL"), "不应有失败行:\n{s}");
}

#[test]
fn print_config_json_carries_provenance_and_estimate() {
    let out = run(&[
        "-p",
        profile("qwen38_27b-3090").to_str().expect("路径"),
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
    assert_eq!(v["config"]["runtime"]["max_seq"], 79872);
    // 2026-10-22:池容量配置项删除;池由 [vram-gov] 按剩余显存自动定。
    assert!(v["config"]["pool"].get("pool_tokens").is_none());
    // 完整档没有 [intent] ⇒ meta.intent 为空是预期(派生面已退役)
    assert!(v["intent"].is_null() || v["intent"]["profile"].is_string());
}

#[test]
fn explain_prints_origin_and_rule() {
    let out = run(&[
        "-p",
        profile("qwen38_27b-3090").to_str().expect("路径"),
        "--explain",
        "runtime.max_seq",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("runtime.max_seq = 79872"), "{s}");
    // 完整档:来源必须是文件(file:…),不再是 derived
    assert!(s.contains("来源: file:"), "应给文件来源:\n{s}");
}

#[test]
fn set_overrides_derived_value() {
    let out = run(&[
        "-p",
        profile("qwen38_27b-3090").to_str().expect("路径"),
        "--set",
        "runtime.max_seq=32768",
        "--check",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("max_seq=32768"), "--set 应覆盖文件值:\n{s}");
    // 池容量无配置项;改 max_seq 只改单请求上限,物理池仍自动。
    assert!(s.contains("pool=auto"), "池应为自动:\n{s}");
}

#[test]
fn legacy_profile_still_works() {
    // 旧档名 `awq27b` 已随 profile 重命名退场;此门守"无 [intent] 的档
    // 仍走同一 CLI 分支"(兼容路径),档名取现役单 3090 满血档。
    let out = run(&[
        "-p",
        profile("qwen38_27b-3090").to_str().expect("路径"),
        "--check",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("无 [intent]"), "旧档应走兼容分支:\n{s}");
    assert!(s.contains("OK"), "{s}");
}

#[test]
fn incomplete_profile_is_rejected_with_key_list() {
    // **全字段显式**(2026-10-21):从**全量模板**里删一个键 ⇒ 拒启并列出缺项。
    // (派生档现在是 `extends` 半份档,完整性由合并结果判 —— 不能拿它当残缺源)
    let full = std::fs::read_to_string(profile("full")).expect("读全量档");
    let broken: String = full
        .lines()
        .filter(|l| !l.trim_start().starts_with("gguf_tile ="))
        .collect::<Vec<_>>()
        .join("\n");
    let path = std::env::temp_dir().join("owl-incomplete-profile.toml");
    std::fs::write(&path, broken).expect("写残缺档");
    let out = run(&["-p", path.to_str().expect("路径"), "--check"]);
    assert!(!out.status.success(), "缺字段应拒启");
    let s = format!("{}{}", String::from_utf8_lossy(&out.stderr), String::from_utf8_lossy(&out.stdout));
    assert!(s.contains("未显式声明"), "应点名缺字段:\n{s}");
    assert!(s.contains("kernel.gguf_tile"), "缺项应列出键路径:\n{s}");
}
