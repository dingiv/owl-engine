//! **显存账健康门**(防呆,2026-10-20;§79 的强制回归项)。
//!
//! 它把 §79 那条血泪教训钉成**机器判据**——"每个 malloc 必须找到它自己的 free,
//! 且只能在自己那个 server 上" 的三条可观测推论:
//!
//! | 判据 | 读法 | 为什么是它 |
//! |---|---|---|
//! | **H1 无打空/双 free/跨面** | `srv.free_miss` 增量 = 0 且 `hook_fail` 增量 = 0 | 服务器 `free_blocks` 找不到块 / 存储层拒收**跨 rank 或二次**归还 —— §79 的泄漏两种形态都在这里冒头 |
//! | **H2 turn 收尾无欠账** | 每轮 `N4` 结清之后,该 turn 的借出登记 = 0 | 有欠账 = 引擎自己的收尾没还完(漏一个调用点就长一截) |
//! | **H3 设备块 = 账本块** | 每轮结束 `device_blocks − ledger_ids` **不增**(基线 = 启动后那点"固定的死东西") | 这条等价于 `/debug/vram.recon.equal`;增量 > 0 = 有块在设备上但账本看不见 |
//!
//! 任何 future 改动只要把这三条打破,本门就红 —— **不需要下一次靠人肉探针去发现**。
//!
//! # 运行(两张 3080;0.8B 档,秒级)
//!
//! ```bash
//! OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 \
//!   OWL_0_8B_DIR=/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B \
//!   cargo test -p owl-engine --test tp_vram_health_gate -- --nocapture
//! ```

use std::path::PathBuf;

use owl_engine::{Engine, EngineConfig, EngineKnobs, TpEngine};

fn dir_of() -> PathBuf {
    std::env::var("OWL_0_8B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../models/Qwen/Qwen3.5-0.8B"))
}

fn cfg_of(ordinal: usize) -> EngineConfig {
    let mut knobs = EngineKnobs::default();
    knobs.sampler_enabled = false;
    knobs.prefix_cache = false;
    EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 2048,
        prefill_chunk: 128,
        knobs,
    }
}

#[test]
fn tp_vram_health_gate() {
    let (Some(oa), Some(ob)) = (
        std::env::var("OWL_TP_TEST_DEV_A").ok().and_then(|s| s.parse().ok()),
        std::env::var("OWL_TP_TEST_DEV_B").ok().and_then(|s| s.parse().ok()),
    ) else {
        eprintln!("skip: 需 OWL_TP_TEST_DEV_A / _B(TP2 双 rank;CUDA ordinal)");
        return;
    };
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        eprintln!("skip: 0.8B 检查点不在 {dir:?}(设 OWL_0_8B_DIR)");
        return;
    }
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(body(oa, ob, dir))
        })
        .expect("spawn")
        .join()
        .expect("join");
}

async fn body(oa: usize, ob: usize, dir: PathBuf) {
    // H1 不再靠全局计数器:free 打空/双 free/跨 rank **在 DeviceHandle 与
    // GpuServer 两侧都是硬错误** ⇒ 只要 turn 能跑完(下面的 pump 不失败),
    // 就说明这一类没有发生。
    let mut e0 = Engine::new(cfg_of(oa)).expect("engine(r0)");
    let m0 = e0
        .loader()
        .slice(0, 2)
        .load_qwen35_0_8b(&dir)
        .await
        .expect("rank0 装载");
    let mut e1 = Engine::new_with_ledger(cfg_of(ob), e0.ledger_cell(), 1).expect("engine(r1)");
    let m1 = e1
        .loader()
        .slice(1, 2)
        .load_qwen35_0_8b(&dir)
        .await
        .expect("rank1 装载");
    let tp = TpEngine::assemble(vec![e0, e1], vec![m0, m1]).await.expect("TP2 装配");
    let mut r = tp.run().expect("进执行态");

    // H3 基线:装配之后(权重/图/池都已就位)的设备块数 vs 账本块数
    let (dev, led) = dev_and_ledger(&mut r).await;
    let base_gap = dev.saturating_sub(led) as i64;
    eprintln!("[vram-gate] 基线:device={dev} ledger={led} gap={base_gap}");

    let prompts = ["用五十字介绍长城。", "一个水池有甲乙两个进水管。", "用 Rust 写一个求中位数的函数。"];
    for (i, p) in prompts.iter().enumerate() {
        let sid = 11 + i as u64;
        let (_sid, turn) = r.submit_session_ex(Some(sid), *p, 24, true).expect("submit");
        loop {
            match r.pump().await.expect("pump") {
                owl_engine::TurnEvent::Idle => break,
                owl_engine::TurnEvent::Failed { turn: t, err } => panic!("t{t} 失败:{err}"),
                _ => {}
            }
        }
        // H1:账本拒收必须为 0(hook_fail = 跨 rank 归还 / 双 free / 幽灵)
        let hf = r.store_report().events.hook_fail;
        assert_eq!(hf, 0, "H1 破:账本 hook_fail = {hf}(跨 rank / 双 free / 幽灵都会记在这)");
        // H2:收尾之后**任何** turn 归属都不应再有在借(兜底释放已逐 rank 结清)
        let left = r
            .store_report()
            .events
            .open_lifecycle
            .iter()
            .filter(|l| matches!(l.owner, owl_iface::handle::Owner::Turn(_)))
            .count();
        assert_eq!(left, 0, "H2 破:第 {i} 轮收尾后仍有 {left} 块挂在 turn 归属下");
        let _ = turn;
        // H3:设备块 − 账本块 的缺口不得增长
        let (dev, led) = dev_and_ledger(&mut r).await;
        let gap = dev as i64 - led as i64;
        eprintln!("[vram-gate] 轮 {i}:device={dev} ledger={led} gap={gap}(基线 {base_gap})");
        assert!(
            gap <= base_gap,
            "H3 破:设备块比账本多出 {gap} 块(基线 {base_gap})—— 账本看不见的显存又长出来了"
        );
    }
    eprintln!("[vram-gate] 全绿:free 打空 0 · hook_fail 0 · 收尾无欠账 · 设备/账本缺口不增");
    let _ = r.shutdown_tp().await;
}

/// `(设备块数, 账本在借块数)` —— **两 rank 合计**(引擎已带 `block_ledger_probe`)。
///
/// 为什么用合计:账本两 rank 共用一本,设备块表各 rank 一份;
/// 只比合计会让"rank0 少、rank1 多"的偏差互相抵消 ⇒ 但也正好能抓
/// §79 那类"整块跑到账外"的泄漏(总量对不上就是不对)。
async fn dev_and_ledger<D: owl_iface::contract::DeviceClient>(
    r: &mut owl_engine::RunningEngine<D>,
) -> (usize, usize) {
    let probe = r.block_ledger_probe().await;
    let mut dev = 0usize;
    for (rank, got) in &probe {
        match got {
            Ok(rows) => dev += rows.len(),
            Err(e) => panic!("rank{rank} 设备块表不可用: {e}"),
        }
    }
    let led: usize = r.store_report().events.ids.iter().map(|(_, v)| v.len()).sum();
    (dev, led)
}
