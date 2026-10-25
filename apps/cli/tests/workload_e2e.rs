//! Live agent-workload gate. Requires a running owl server configured for the
//! suite (8135 image-evict test profile) and `OWL_E2E=1`.
//!
//!   OWL_E2E=1 OWL_SERVER_URL=http://127.0.0.1:8135 \
//!   OWL_SERVER_LOG=/tmp/owl-8135-image-evict.log \
//!   cargo test -p cli --test workload_e2e -- --nocapture
//!
//! `OWL_WORKLOAD_SUITE` selects the suite (default `core`; `gui` is the
//! GUI-agent screenshot-window scenario and needs `snap_max>=2`). For the
//! rebase arm start the server with `--set pool.image_evict_delta_limit=2` and
//! set `OWL_WORKLOAD_SUITE=image-evict-rebase`.

use std::process::Command;

#[test]
fn agent_workload_suite() {
    if std::env::var("OWL_E2E").as_deref() != Ok("1") {
        eprintln!("skip: OWL_E2E=1 未开(需要已启动的 owl server)");
        return;
    }
    let url = std::env::var("OWL_SERVER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8135".to_string());
    let log = std::env::var("OWL_SERVER_LOG").ok();
    let suite = std::env::var("OWL_WORKLOAD_SUITE").unwrap_or_else(|_| "core".to_string());
    let bin = env!("CARGO_BIN_EXE_cli");
    let mut cmd = Command::new(bin);
    cmd.args(["--workload", &suite, "--url", &url]);
    if let Some(log) = log.as_deref() {
        cmd.args(["--server-log", log]);
    }
    let out = cmd.output().expect("启动 cli workload");
    eprintln!("--- stderr ---\n{}", String::from_utf8_lossy(&out.stderr));
    eprintln!("--- stdout ---\n{}", String::from_utf8_lossy(&out.stdout));
    assert!(
        out.status.success(),
        "workload {suite} 失败(server {url})"
    );
}
