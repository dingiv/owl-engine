//! engine facade 端到端(2026-09-26):构造(不执行)→ ModelLoader →
//! run → submit/pump(turn 流;两 turn 入队验证 per-turn 状态重置)。
//!
//! 用法:
//! ```text
//! OWL_TEST_DEVICE=3 cargo run -q -p owl-engine --example generate -- "你好" 24
//! ```

use std::path::PathBuf;

use owl_engine::{Engine, EngineConfig, TurnEvent};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "请用一句话介绍你自己。".into());
    let max_new: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    let ordinal: usize = std::env::var("OWL_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/assets/Qwen3.5-0.8B");

    // ── 生命周期:构造(不执行)→ 装载 → run(进入执行态)→ 提交/泵 ──
    let t0 = std::time::Instant::now();
    let mut engine = Engine::new(EngineConfig { device_ordinal: ordinal, max_seq_tokens: 64 })
        .expect("engine 构造");
    let model = engine
        .loader()
        .load_qwen35_0_8b(&dir)
        .await
        .expect("模型装载");
    let mut running = engine.run(model).await.expect("装配(含 warmup)");

    // 并发形态:两个 turn 入队(M0.5 单槽串行;M2 batching 真并发)
    let t1 = running.submit(&prompt, max_new).expect("submit t1");
    let t2 = running.submit("用一句话说明什么是 P2P。", max_new).expect("submit t2");
    println!("[gen] turn t{t1} t{t2} 已入队");

    let t_gen = std::time::Instant::now();
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => break,
            TurnEvent::Prefill { turn, fed, total } => {
                if fed % 4 == 0 || fed == total {
                    eprintln!("[gen] t{turn} prefill {fed}/{total}");
                }
            }
            TurnEvent::Token { turn, delta } => {
                print!("[t{turn}]{delta}");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            TurnEvent::Completed { turn, text } => {
                println!("\n[gen] t{turn} 完成: {text}");
            }
            TurnEvent::Failed { turn, err } => {
                println!("\n[gen] t{turn} 失败: {err}");
            }
        }
    }
    println!(
        "[gen] 泵 2 turn x {} tok / {:.2} s = {:.3} s/tok(回放态;eager 基线 0.11)",
        max_new,
        t_gen.elapsed().as_secs_f32(),
        t_gen.elapsed().as_secs_f32() / (max_new * 2) as f32
    );
    println!("[gen] 队列排空,引擎 Idle(总耗时 {:.1}s)", t0.elapsed().as_secs_f32());
}
