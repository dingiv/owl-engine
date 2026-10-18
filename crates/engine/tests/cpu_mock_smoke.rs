//! **CPU mock 档冒烟(§18)**:引擎跑在 `CpuFace` 上 —— **无卡机**可跑全链
//! (boot → 装载 → warmup → 图捕获 → turn)。
//!
//! 界线(必须说清,否则会被误用):
//! - **等价面** = 账与搬运:块分配/释放、容量墙、`dtoh/h tod`、`swap`、
//!   图捕获/重放、命令序(发射点什么、什么块被写);
//! - **不等价面** = 数值:空转档(`null_kernels`)不执行算子,输出块清零 ⇒
//!   生成内容是垃圾(但流程不崩、token 合法)。
//! - 性能数**不可用**(CPU 解释器 + 空转)。
//!
//! 跑法(默认 skip;装载 0.8B 需数十秒,不进默认套件):
//! ```bash
//! OWL_CPU_MOCK=1 cargo test -p owl-engine --test cpu_mock_smoke -- --nocapture
//! ```

use owl_cpu::{CpuFace, MockLimits};
use owl_engine::{Engine, EngineConfig, EngineKnobs};

/// 资产目录(与 e2e 同源:0.8B 最小档)
fn asset_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/assets/Qwen3.5-0.8B")
}

#[tokio::test(flavor = "current_thread")]
async fn cpu_mock_engine_boots_loads_and_runs_a_turn() {
    if std::env::var_os("OWL_CPU_MOCK").is_none() {
        eprintln!("skip: OWL_CPU_MOCK 未设(CPU mock 冒烟按需开)");
        return;
    }
    let cfg = EngineConfig {
        device_ordinal: 0,
        max_seq_tokens: 256,
        prefill_chunk: 32,
        knobs: EngineKnobs { sampler_enabled: false, ..Default::default() },
    };
    // 空转档:账与搬运真、数值不算
    let face = CpuFace::mock(MockLimits { null_kernels: true, graph_capture: true, ..Default::default() });
    let mut engine = Engine::on(cfg, face).expect("构造(CPU face)");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配(warmup + 图捕获)");
    let text = running.generate("你好", 4).await.expect("生成");
    eprintln!("[cpu-mock] 空转档生成(数值无意义,只有流程意义): {text:?}");
    eprintln!("[cpu-mock] 流程门通过:boot/装载/warmup/图捕获/turn 全程无卡");
}

/// **容量墙档(§18)**:把"显存"拧到装不下池 ⇒ boot 必须**响亮拒**
/// (真卡上要熬到显存满才能测的错误路径,这里一行配置)。
#[tokio::test(flavor = "current_thread")]
async fn cpu_mock_vram_wall_rejects_boot_loudly() {
    if std::env::var_os("OWL_CPU_MOCK").is_none() {
        eprintln!("skip: OWL_CPU_MOCK 未设(CPU mock 冒烟按需开)");
        return;
    }
    let cfg = EngineConfig {
        device_ordinal: 0,
        max_seq_tokens: 256,
        prefill_chunk: 32,
        knobs: EngineKnobs { sampler_enabled: false, ..Default::default() },
    };
    // 1MiB "显存":连权重都装不下 ⇒ 拒启(而非静默降级/静默 OOM)
    let face = CpuFace::mock(MockLimits { vram_bytes: 1 << 20, ..Default::default() });
    let mut engine = Engine::on(cfg, face).expect("构造");
    let err = match engine.loader().load_qwen35_0_8b(&asset_dir()).await {
        Ok(_) => panic!("容量墙内装载必须失败"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    eprintln!("[cpu-mock] 容量墙拒原文: {msg}");
    assert!(
        msg.contains("PoolExhausted") || msg.contains("cpu-mock"),
        "拒绝须自带容量归因: {msg}"
    );
}

/// **装载耗时分解**(§18 诊断):各契约方法的调用次数与累计秒数。
#[tokio::test(flavor = "current_thread")]
async fn cpu_mock_load_cost_breakdown() {
    if std::env::var_os("OWL_CPU_MOCK").is_none() {
        eprintln!("skip: OWL_CPU_MOCK 未设");
        return;
    }
    let cfg = EngineConfig {
        device_ordinal: 0,
        max_seq_tokens: 256,
        prefill_chunk: 32,
        knobs: EngineKnobs { sampler_enabled: false, ..Default::default() },
    };
    let face = CpuFace::mock(MockLimits { null_kernels: true, graph_capture: true, ..Default::default() });
    let mut engine = Engine::on(cfg, face).expect("构造");
    owl_cpu::mock_stats_reset();
    let t = std::time::Instant::now();
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    let wall = t.elapsed().as_secs_f64();
    let names = ["alloc", "htod", "upload_pinned", "upload_and_launch", "copy_block_at", "launch"];
    let st = owl_cpu::mock_stats();
    eprintln!("[cpu-mock] 装载墙钟 {wall:.2}s;契约面累计:");
    let mut acc = 0.0f64;
    for (i, (calls, ns)) in st.iter().enumerate() {
        eprintln!("[cpu-mock]   {:<18} calls={:<8} cum={:>8.2}s", names[i], calls, *ns as f64 / 1e9);
        acc += *ns as f64 / 1e9;
    }
    eprintln!("[cpu-mock]   契约面合计 {acc:.2}s(其余 = 装载器 host 侧:repack/读盘/校验)");
    drop(loaded);
}


