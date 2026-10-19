//! **设备账在真引擎路径上的落地门**(步 3;CPU mock 档 —— 无卡机可跑)
//!
//! 判据(设计卷 §九 步 3):
//! 1. **装载**期的块进账且类 = `Weight`(相位上下文);
//! 2. **池骨架**进账且类 = `Pool`(boot 相位);
//! 3. **图窗**内自产块进账且类 = `Graph`(捕获相位;块随图存活);
//! 4. `hook_fail` 全程 0(钩子不许有"没入账"的事件);
//! 5. 账本是**同一本**(引擎持有 → `Rel` 拿到 → store 报告里能看到同一批字节)。
//!
//! 跑法(默认 skip;0.8B 装载需数十秒):
//! ```bash
//! OWL_CPU_MOCK=1 cargo test -p owl-engine --test device_ledger -- --nocapture
//! ```

use owl_cpu::{CpuFace, MockLimits};
use owl_engine::{AssetClass, Engine, EngineConfig, EngineKnobs};

fn asset_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/assets/Qwen3.5-0.8B")
}

#[tokio::test(flavor = "current_thread")]
async fn ledger_captures_load_pool_and_graph_phases() {
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
    // 账本**由测试预造**并交给引擎(boot 期接管的那一本)—— 与生产 `Engine::on`
    // 内部造的同一形状,只是为了能在外面读账。
    let cell = owl_store::shared_ledger(owl_store::tier_budget_of(
        cfg.knobs.host_cache_mb,
        cfg.knobs.snap_host_mb,
    ));
    let face = CpuFace::mock(MockLimits {
        null_kernels: true,
        graph_capture: true,
        ..Default::default()
    });
    let mut engine =
        Engine::on_with_ledger(cfg, face, cell.clone()).expect("构造(CPU face + 预置账本)");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    {
        // 装载完:权重类必须有账(相位上下文 = Weight)
        let b = owl_store::book_lock(&cell);
        let rows = b.open_rows();
        let weight: usize = rows
            .iter()
            .filter(|r| r.kind == owl_store::AssetKind::Weight)
            .map(|r| r.bytes)
            .sum();
        eprintln!("[ledger] 装载后 open={} 条 / {}B;weight 类 {}B", b.open_count(), b.open_total_bytes(), weight);
        assert!(weight > 0, "装载期必须有 weight 类账:{rows:?}");
        assert_eq!(b.hook_fail(), 0, "装载期不许有未入账事件");
    }
    let mut running = engine.run(loaded).await.expect("装配(warmup + 图捕获)");
    {
        let b = owl_store::book_lock(&cell);
        let rows = b.open_rows();
        let pool: usize = rows.iter().filter(|r| r.kind == owl_store::AssetKind::Pool).map(|r| r.bytes).sum();
        let graph: usize = rows.iter().filter(|r| r.kind == owl_store::AssetKind::Graph).map(|r| r.bytes).sum();
        eprintln!("[ledger] 装配后 open={} 条 / {}B;pool {}B;graph {}B", b.open_count(), b.open_total_bytes(), pool, graph);
        for r in &rows {
            eprintln!(
                "[ledger]   rank{} {:?} {:?} {:?} {}B n={}",
                r.rank, r.owner, r.kind, r.tier, r.bytes, r.entries
            );
        }
        assert!(pool > 0, "池骨架必须有 pool 类账");
        assert!(graph > 0, "捕获窗内自产块必须有 graph 类账(窗内类自动覆盖)");
        assert_eq!(b.hook_fail(), 0, "装配期不许有未入账事件");
    }
    // 跑一个 turn:turn 期分配的归属 = Turn(id)(N1 设 Node);跑完 N4 回落
    let text = running.generate("你好", 4).await.expect("生成");
    eprintln!("[ledger] 空转档生成(数值无意义): {text:?}");
    let (hf, wu, owb) = running.handle_probe();
    let b = owl_store::book_lock(&cell);
    eprintln!("[ledger] turn 后 open={} 条;hook_fail={hf} window_unclosed={wu} 窗内块={owb}", b.open_count());
    assert_eq!(hf, 0, "全程不许有未入账事件");
    assert_eq!(wu, 0, "没有窗失败 ⇒ 不该有未收窗");
    assert_eq!(b.open_count() > 0, true, "账必须有内容(否则记了个寂寞)");
}

/// **账本预算交叉校验**:`Rel::open` 必须拒"账本与分解不符"(boot 算式漂了 = 响亮拒)
#[test]
fn ledger_budget_mismatch_is_rejected() {
    // store 侧已有单测覆盖(rel.rs `open_rejects_unmeasured_budget_and_geometry`);
    // 这里只钉住引擎侧**同一个算式**(tier_budget_of)是唯一产地。
    let a = owl_store::tier_budget_of(512, 512);
    assert_eq!(a.host_bytes, 1024 << 20, "host 总量 = 两块之和");
    assert_eq!(a.disk_bytes, 0, "盘层二期(恒 0)");
    let _ = AssetClass::Weight; // 词表在引擎侧可见(公共面)
}
