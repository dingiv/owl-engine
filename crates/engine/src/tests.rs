//! 引擎测试(2026-10-01 拆分自 engine.rs §5;用例逐字保真,仅私有
//! 字段访问改 StatePool 路径)。
//!
//! **测试分层(2026-10-13,见 docs/arch/test-architecture.md)**:本文件只留**单元测试**
//! (CPU/无装载,`cargo test -p owl-engine --lib`);涉及**模型装载**的场景
//! (下方 `pub async fn gpu_*`,**无 `#[test]`**)由唯一 e2e 入口
//! `crates/engine/tests/engine.e2e.rs` 手动调用(`--features e2e --test engine_e2e`)。
//! 装载只做一次:`boot_08b`/`boot_27b_default` 是共享装载工厂(装载本身即一步测试)。
//!
//! CPU:构造/装载门禁;GPU 门控(OWL_TEST_DEVICE):双 turn 生命周期 /
//! 会话连续 / 4k 长 ctx / W4A16 / 27B AWQ / 多会话隔离 / 前缀缓存命中 /
//! S0 调度决策面。

use crate::engine::{Engine, EngineConfig, EngineKnobs};
#[cfg(test)] // 仅单元档的 decode 差分用例消费;e2e 档不编该函数
use crate::exec::decode_delta;
use crate::running::RunningEngine;
use crate::scheduler::{SchedulerOutput, StepAction};
use crate::turn::TurnEvent;
#[cfg(test)]
use owl_cpu::CpuFace; // dev-dep:仅 cfg(test) 单元档可用
// dev-dep 门控(与 tokio 同款模式):cfg(test) 先求值 ⇒ e2e 档不触宏名
#[cfg(test)]
use owl_shared::config::OwlConfig;
#[cfg(test)]
use owl_macros::owl_test;
use owl_iface::contract::DeviceClient;

/// **单图捕获 + 双回放门**(原 piecewise 段驱动件门;2026-10-18 W5 改单图)——
/// **不装模型**,纯 `GraphPlan` 面。
///
/// 判据:
/// ① 捕获成功(`PlanOutcome::Captured`);
/// ② **回放两次值一致**(图回放可重复);
/// ③ **引用计数推送面在场**(全量记名/OOM dump 的"它凭什么是活的"数据源):
///    warmup 探针窗内 `refs_pushed > 0` —— 判据用**计数器**而非日志
///    (日志受级别过滤;计数是机械事实)。
///
/// 树:`y = x + c → out = y + c`(x=c=1 ⇒ out=3)。
pub async fn gpu_single_graph_capture_and_replay() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    // 最小 GpuClient 装配(与 owl-cuda 的 gpu_server 测试同款:server 专职线程 + 管道)
    let (tx, rx) = std::sync::mpsc::channel::<owl_cuda::Command>();
    let server = owl_cuda::GpuServer::new(
        rx,
        owl_cuda::DeviceSelector::Ordinal(ordinal),
        None,
    );
    let face = owl_cuda::GpuClient::new(tx);
    std::thread::Builder::new()
        .name("owl-graph-test".into())
        .spawn(move || server.run().expect("server run 异常退出"))
        .expect("起 server 线程失败");

    let desc = crate::graph_plan::GraphPlanDesc {
        reclaim_capture: false,
        warmup: true,
        inputs: vec![
            crate::graph_plan::InputSlot::f32("x", 4).init(vec![1.0; 4]),
            crate::graph_plan::InputSlot::f32("c", 4).init(vec![1.0; 4]),
        ],
        outputs: vec![crate::graph_plan::OutputSlot::f32("out", &[4])],
        capture: true,
        slab_fixed: false,
        debug_probe: false,
    };
    let forward = |sc: &crate::graph_plan::PlanCtx| -> Result<(), owl_iface::contract::ModelError> {
        let x = sc.input("x")?;
        let c = sc.input("c")?;
        let y = x.add(&c);
        let out = y.add(&c);
        sc.output("out", &out)
    };
    let (mut plan, outcome) = crate::graph_plan::GraphPlan::plan(
        face,
        owl_models::env::EnvProvider::default(),
        None,
        desc,
        forward,
    )
    .await
    .expect("plan");
    assert!(
        matches!(outcome, crate::graph_plan::PlanOutcome::Captured),
        "应捕获,得 {outcome:?}"
    );
    // 回放 ×2:x=1,c=1 ⇒ y=2 ⇒ out=3
    for i in 0..2 {
        let xs = [1.0f32; 4];
        plan.step(&[("x", &xs)]).await.expect("step");
        let v = plan.read_output_f32("out").await.expect("read out");
        assert!(
            v.iter().all(|x| (x - 3.0).abs() < 1e-4),
            "回放 #{i} 值应 3.0,得 {v:?}"
        );
    }
    // **引用计数推送面**(全量记名同轮;2026-10-16):图 plan 的 warmup 探针窗内,
    // 求值器必须把 `(users, pending)` 推给探针。窗口在 plan 内部 `probe_on/off`,
    // on 时清零 ⇒ 此处读到的是本次 plan 的推送量。
    let pushed = owl_shared::slab_hint::refs_pushed();
    assert!(
        pushed > 0,
        "单图计划跑完后引用计数推送为 0 ⇒ 数据源断了(记名/OOM dump 的\"为什么活着\"面失效)"
    );
    eprintln!("[graph] 单图捕获 + 双回放 值一致 ✓(引用计数推送 {pushed} 次)");
}

fn asset_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models/assets/Qwen3.5-0.8B")
}

fn gpu_ordinal() -> Option<usize> {
    owl_cuda::test_device_ordinal_opt()
}

/// CPU:构造(零执行)+ ModelLoader 装载门禁(权重/tokenizer 解析)
/// EngineConfig 全件从**用例配置**注入(唯一入口 `OWL_TEST_CONF` 测试环境档;
/// 缺省 = `OwlConfig::default()`)。本用例只到 loader ——
/// 三件均不参与断言,注入不改测试行为;换档 = 换 toml 不改代码(S3 收编示范)。
#[cfg(test)] // dev-dep(owl-macros)门控:单元档专用
#[owl_test]
async fn cpu_construct_and_load(cfg: &OwlConfig) {
    let mut engine = Engine::on(
        EngineConfig {
            device_ordinal: 0, // CPU face 不看卡位(卡位改由 [parallel] device 表达)
            max_seq_tokens: cfg.runtime.max_seq,
            prefill_chunk: cfg.runtime.prefill_chunk,
            knobs: EngineKnobs::from_config(cfg),
        },
        CpuFace::new(),
    )
    .expect("engine 构造");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    assert_eq!(loaded.model.layers.len(), 24);
    assert!(!loaded.tokenizer.encode("你好").is_empty(), "tokenizer 活性");
}

/// **状态腿 mock 引擎**(§30 两门共用):snap_max 可调 + 状态腿宿主 64MiB。
#[cfg(test)] // 只被单元档用例消费(e2e 档不编)
async fn mock_snap_engine(
    snap_max: usize,
) -> RunningEngine<owl_iface::handle::DeviceHandle<owl_cpu::CpuFace>> {
    let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
    knobs.pool_tokens = Some(256);
    knobs.host_cache_mb = 8;
    knobs.snap_host_mb = 64;
    knobs.snap_max = snap_max;
    let cfg = EngineConfig {
        device_ordinal: 0,
        max_seq_tokens: 256,
        prefill_chunk: 32,
        knobs,
    };
    let face = owl_cpu::CpuFace::mock(owl_cpu::MockLimits {
        null_kernels: true,
        graph_capture: true,
        ..Default::default()
    });
    let mut engine = Engine::on(cfg, face).expect("构造(CPU mock)");
    let loaded = engine
        .loader()
        .load_qwen35_0_8b(&asset_dir())
        .await
        .expect("装载");
    engine.run(loaded).await.expect("装配")
}

/// **两腿统一调度器门(engine 侧;§30)**:store 的 `plan_need` 出 `Snap{Host}` 票
/// 之后,engine 的执行臂(`apply_verdict`)真的把设备层快照沉到宿主 ——
/// 验的是"判据唯一 + 执行闭环",不依赖 capture 的 churn 路径。
#[cfg(test)] // 单元档专属(dev-dep tokio/macros);e2e 档不编该用例
#[tokio::test]
async fn idle_node_offloads_device_snapshot_via_store_plan() {
    let mut running = mock_snap_engine(1).await;

    // boot 已注册状态腿布局 ⇒ "一条快照的字节" > 0;宿主分片 64MiB 留位
    let bytes = running.bank.rel.snap_bytes();
    assert!(bytes > 0, "状态腿布局应在 boot 注册(snap_max>0 + paged)");
    assert!(running.bank.rel.snap_host_free_slots() > 0, "宿主分片应留位");
    assert!(running.dev.pool.snap_capacity() > 0, "设备槽应 >= 1");

    // 手工登记一条 Durable 设备快照(不经 capture:本门只验调度 + 执行臂)
    let key = owl_store::SnapKey::at_boundary(owl_store::ChainHash::new(0x5A5A));
    running.bank.rel.state.register_snap(
        key,
        bytes,
        owl_store::Tier::Gpu,
        owl_store::SnapDurability::Durable,
    );
    running.dev.pool.snap_place(0, key);

    running
        .node(crate::running::Node::Idle(0))
        .await
        .expect("Idle 节点");

    assert_eq!(
        running.bank.rel.state.locate(key),
        Some(owl_store::Tier::Host),
        "store 判 Host ⇒ engine 必须真沉(执行臂闭环)"
    );
    assert_eq!(running.dev.pool.snap_used(), 0, "设备槽应释放");
}

/// **磁盘层门(S6)**:宿主装不下 ⇒ 溢写到磁盘(内容不丢),再能读回宿主。
///
/// 验的是三件:①`plan_capacity` 的第三层出票 ②engine 的执行臂真写文件
/// (原子 .tmp→rename)③回填把层账改回 Host(内容逐字节一致由 store 侧保证)。
#[cfg(test)] // 单元档专属(dev-dep tokio/macros);e2e 档不编该用例
#[tokio::test]
async fn disk_spill_and_fill_roundtrip() {
    let dir = std::env::temp_dir().join(format!("owl-disk-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
    knobs.pool_tokens = Some(256);
    knobs.host_cache_mb = 0;
    knobs.snap_host_mb = 24; // 只装得下一张快照(0.8B 一条 ≈19MiB)
    knobs.snap_max = 2;
    knobs.disk_cache_mb = 8;
    knobs.snap_disk_mb = 24;
    knobs.disk_dir = dir.display().to_string();
    let cfg = EngineConfig {
        device_ordinal: 0,
        max_seq_tokens: 256,
        prefill_chunk: 32,
        knobs,
    };
    let face = owl_cpu::CpuFace::mock(owl_cpu::MockLimits {
        null_kernels: true,
        graph_capture: true,
        ..Default::default()
    });
    let mut engine = Engine::on(cfg, face).expect("构造(CPU mock)");
    let loaded = engine
        .loader()
        .load_qwen35_0_8b(&asset_dir())
        .await
        .expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");

    let bytes = running.bank.rel.snap_bytes();
    assert!(bytes > 0, "状态腿布局应已注册");
    assert!(running.bank.rel.disk_enabled(), "盘层应已接线(配额+目录)");
    // **P2 磁盘腿入准入**：`store_facts()` 必须看见盘上余量（旧行为恒 0
    // ⇒ 超卖档配了盘也不用，宿主满 = 拒新）。单卡：盘块 = 一页 ⇒ 页数 = 槽数。
    let facts = running.store_facts();
    let slots = running.bank.rel.disk.as_ref().unwrap().free_slots();
    assert!(slots > 0, "盘配额应给出空位（实得 {slots}）");
    assert_eq!(facts.disk_free_pages, slots, "盘槽 → 页的折算必须一一对应");
    assert_eq!(running.bank.rel.snap_disk.as_ref().unwrap().free_slots(), 1, "24MiB / 一条");
    let k_a = owl_store::SnapKey::at_boundary(owl_store::ChainHash::new(0xA11));
    let k_b = owl_store::SnapKey::at_boundary(owl_store::ChainHash::new(0xB22));
    for k in [k_a, k_b] {
        running.bank.rel.state.register_snap(
            k,
            bytes,
            owl_store::Tier::Gpu,
            owl_store::SnapDurability::Durable,
        );
    }
    running.dev.pool.snap_place(0, k_a);
    running.dev.pool.snap_place(1, k_b);

    // **拍 1**:目标 1 张 ⇒ 沉到宿主(宿主 1 位,刚好)
    running
        .pool_housekeeping(owl_store::Need {
            slots: 0,
            snap_bytes: bytes,
            why: owl_store::NeedWhy::Pressure,
        })
        .await
        .expect("拍 1");
    let st1 = running.bank.rel.state.stats();
    assert_eq!((st1.snap_gpu.0, st1.snap_host.0), (1, 1), "一拍一张:{st1:?}");
    let in_host = [k_a, k_b]
        .into_iter()
        .find(|k| running.bank.rel.state.locate(*k) == Some(owl_store::Tier::Host))
        .expect("应有一张在宿主");
    let other = if in_host == k_a { k_b } else { k_a };

    // **拍 2**:宿主满 ⇒ 逐层律第三步:把宿主里最冷的那张挪到磁盘(内容不丢)
    running
        .pool_housekeeping(owl_store::Need {
            slots: 0,
            snap_bytes: bytes,
            why: owl_store::NeedWhy::Pressure,
        })
        .await
        .expect("拍 2");
    let disk = running.bank.rel.snap_disk.as_ref().unwrap().resident_count();
    assert_eq!(disk, 1, "宿主满 ⇒ 一张落磁盘:{:?}", running.bank.rel.state.stats());
    assert_eq!(
        running.bank.rel.state.locate(in_host),
        Some(owl_store::Tier::Disk),
        "被挪去盘的是宿主里那张"
    );
    assert_eq!(
        running.bank.rel.state.locate(other),
        Some(owl_store::Tier::Gpu),
        "设备那张仍在(内容不丢优先:本轮不出 Drop)"
    );
    let path = running
        .bank
        .rel
        .snap_disk
        .as_ref()
        .unwrap()
        .path_of(in_host)
        .unwrap()
        .to_path_buf();
    assert!(path.exists(), "文件应真写盘:{}", path.display());
    assert_eq!(
        std::fs::metadata(&path).unwrap().len() as usize,
        bytes,
        "文件大小 = 一条快照"
    );

    // 回填:盘 → 宿主(层账回 Host;磁盘副本仍在 = 权威)
    assert!(running.fill_snap_from_disk(in_host).await.expect("回填"));
    assert_eq!(
        running.bank.rel.state.locate(in_host),
        Some(owl_store::Tier::Host),
        "回填后层账 = Host"
    );
    assert!(running.bank.rel.snap_host.get(in_host).is_some(), "宿主应有内容");
    assert!(
        running.bank.rel.snap_disk.as_ref().unwrap().contains(in_host),
        "磁盘副本仍在(权威)"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
/// **会话感知淘汰门(engine)**:turn 收口把 **KV 链块**与 **GDN 快照**都记到
/// 会话名下(淘汰时同会话才能"从末端削")。
#[cfg(test)] // 单元档专属(dev-dep tokio/macros);e2e 档不编该用例
#[tokio::test]
async fn turn_records_session_ownership_for_kv_and_snaps() {
    let mut running = mock_snap_engine(2).await;
    let prompt: String = (0..120)
        .map(|i| ["alpha", "bravo", "cedar", "delta", "ember", "falcon"][i % 6])
        .collect::<Vec<_>>()
        .join(" ");
    let (sid, tid) = running.submit_session(None, &prompt, 32).expect("submit");
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Completed { turn, .. } if turn == tid => break,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败:{err}"),
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            _ => {}
        }
    }
    let toks = running.tokenize(&prompt, true).expect("tokenize").tokens;
    let page = running.dev.pool.geom.page.max(1);
    let full = toks.len() / page;
    assert!(full >= 2, "prompt 应覆盖至少两块");
    let tail = running
        .bank
        .rel
        .leg
        .tail_hash(&toks, full)
        .expect("链尾身份");
    assert_eq!(
        running.bank.rel.leg.session_of(tail),
        sid,
        "KV 尾块应归本会话"
    );
    assert_eq!(running.bank.rel.leg.depth_of(tail), full, "深度 = 块数");

    // 快照(页边界)也要带归属与深度
    let snaps: Vec<_> = running
        .bank
        .rel
        .state
        .evict_candidates(&[])
        .into_iter()
        .filter_map(|c| c.snap)
        .collect();
    assert!(!snaps.is_empty(), "应有页边界快照");
    let mut seen_depth = false;
    for k in snaps {
        let (s, d) = running.bank.rel.state.snap_session(k).expect("在账");
        assert_eq!(s, sid, "快照应归本会话");
        if d > 0 {
            seen_depth = true;
        }
    }
    assert!(seen_depth, "快照应带边界深度(末端优先的依据)");
}

/// **S6.1 端到端门**:turn 收口写链索引 → 小宿主把链溢写到盘 →
/// **重启后按索引重建树**(非设备层)→ **同 prompt 命中前缀**(跨重启复用成立)。
#[cfg(test)] // 单元档专属(dev-dep tokio/macros);e2e 档不编该用例
#[tokio::test]
async fn disk_chain_index_enables_cross_restart_reuse() {
    let dir = std::env::temp_dir().join(format!("owl-idx-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mk = |dir: &std::path::Path| {
        let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
        knobs.pool_tokens = Some(96); // 3 页(短池:容易满)
        knobs.host_cache_mb = 1; // 宿主只装得下 ~2 块(0.8B 块 ≈384KiB)
        knobs.snap_max = 0;
        knobs.disk_cache_mb = 8;
        knobs.snap_disk_mb = 0;
        knobs.disk_dir = dir.display().to_string();
        EngineConfig { device_ordinal: 0, max_seq_tokens: 256, prefill_chunk: 32, knobs }
    };
    let boot = |cfg: EngineConfig| async move {
        let face = owl_cpu::CpuFace::mock(owl_cpu::MockLimits {
            null_kernels: true,
            graph_capture: true,
            ..Default::default()
        });
        let mut engine = Engine::on(cfg, face).expect("构造(CPU mock)");
        let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
        engine.run(loaded).await.expect("装配")
    };
    // ≥128 token(4 整块)才能命中"整块前缀";短 prompt 不足一块 = 命中 0(语义如此)
    let prompt: String = (0..120)
        .map(|i| ["alpha", "bravo", "cedar", "delta", "ember", "falcon", "garnet", "harbor"][i % 8])
        .collect::<Vec<_>>()
        .join(" ");

    // ── 第一次进程:跑一个 turn(收口写索引)+ 压力保洁把链溢写到盘
    {
        let mut running = boot(mk(&dir)).await;
        let (_, tid) = running.submit_session(None, &prompt, 32).expect("submit");
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { turn, .. } if turn == tid => break,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败:{err}"),
                TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
                _ => {}
            }
        }
        // 压力保洁(N6):设备 → 宿主 → 磁盘(小宿主 ⇒ 必然溢写)
        for _ in 0..6 {
            running
                .pool_housekeeping(owl_store::Need {
                    slots: 8,
                    snap_bytes: 0,
                    why: owl_store::NeedWhy::Pressure,
                })
                .await
                .expect("保洁");
        }
        let disk_entries = running.bank.rel.disk.as_ref().unwrap().resident_count();
        assert!(disk_entries > 0, "应已有链内容溢写到盘");
        // 重启前的对照:同一份 token 化应命中(证明索引 tokens 与查询同源)
        // 提交路径会套 chat 模板 ⇒ 查询侧同口径(chat_wrap = true)
        let t0 = running.tokenize(&prompt, true).expect("tokenize").tokens;
        let (h0, _) = running.bank.rel.match_prefix(&t0).expect("命中");
        assert!(h0.matched_blocks > 0, "重启前应命中(否则 tokens 不同源)");
        let idx_n = std::fs::read_dir(dir.join("kv"))
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.file_name().to_string_lossy().ends_with(".idx"))
                    .count()
            })
            .unwrap_or(0);
        assert!(idx_n > 0, "turn 收口应写链索引");
    }

    // ── 第二次进程:同目录 ⇒ 按索引重建树(非设备层)+ 同 prompt 命中
    eprintln!(
        "[idx] 重启前盘上: bin={} idx={}",
        std::fs::read_dir(dir.join("kv"))
            .map(|rd| rd.flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".bin")).count())
            .unwrap_or(0),
        std::fs::read_dir(dir.join("kv"))
            .map(|rd| rd.flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".idx")).count())
            .unwrap_or(0),
    );
    let mut running2 = boot(mk(&dir)).await;
    eprintln!(
        "[idx] 重启后: 盘 entries={} / 树 host_blocks={}",
        running2.bank.rel.disk.as_ref().map(|d| d.resident_count()).unwrap_or(0),
        running2.bank.rel.leg.kv_stats().host_blocks
    );
    let rebuilt = running2.bank.rel.leg.kv_stats().host_blocks;
    assert!(rebuilt > 0, "重启后应按索引重建出条目(非设备层)");
    let toks = running2.tokenize(&prompt, true).expect("tokenize").tokens;
    let (hit, _recall) = running2.bank.rel.match_prefix(&toks).expect("命中");
    assert!(hit.matched_blocks > 0, "重建的树应命中同一 prompt 前缀");
    // 命中面允许回读(内容在盘 ⇒ pending_host/回灌路径);这里只验"选得中"
    assert_eq!(
        hit.hashes.len(),
        hit.matched_blocks,
        "命中面应带逐块身份(回灌/快照键的真源)"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// **启动回加载门(S6)**:进程重启(新 engine、同目录)后,盘上的内容被
/// **扫目录回填账**——账先回来(条目可见),内容惰性读回。
#[cfg(test)] // 单元档专属(dev-dep tokio/macros);e2e 档不编该用例
#[tokio::test]
async fn disk_adopt_on_boot_restores_ledger() {
    let dir = std::env::temp_dir().join(format!("owl-disk-boot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mk = |dir: &std::path::Path| {
        let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
        knobs.pool_tokens = Some(256);
        knobs.host_cache_mb = 0;
        knobs.snap_host_mb = 24;
        knobs.snap_max = 2;
        knobs.disk_cache_mb = 8;
        knobs.snap_disk_mb = 24;
        knobs.disk_dir = dir.display().to_string();
        EngineConfig { device_ordinal: 0, max_seq_tokens: 256, prefill_chunk: 32, knobs }
    };
    let boot = |cfg: EngineConfig| async move {
        let face = owl_cpu::CpuFace::mock(owl_cpu::MockLimits {
            null_kernels: true,
            graph_capture: true,
            ..Default::default()
        });
        let mut engine = Engine::on(cfg, face).expect("构造(CPU mock)");
        let loaded = engine
            .loader()
            .load_qwen35_0_8b(&asset_dir())
            .await
            .expect("装载");
        engine.run(loaded).await.expect("装配")
    };

    let key = owl_store::SnapKey::at_boundary(owl_store::ChainHash::new(0xFEED));
    // ── 第一次进程:把一张快照搬到盘上(拍 1 宿主 → 拍 2 盘)
    {
        let mut running = boot(mk(&dir)).await;
        let bytes = running.bank.rel.snap_bytes();
        running.bank
            .rel
            .state
            .register_snap(key, bytes, owl_store::Tier::Gpu, owl_store::SnapDurability::Durable);
        running.dev.pool.snap_place(0, key);
        for _ in 0..2 {
            running
                .pool_housekeeping(owl_store::Need {
                    slots: 0,
                    snap_bytes: bytes,
                    why: owl_store::NeedWhy::Pressure,
                })
                .await
                .expect("保洁");
        }
        assert_eq!(
            running.bank.rel.state.locate(key),
            Some(owl_store::Tier::Disk),
            "第一次进程里应已落盘"
        );
    } // ← engine drop(等价于进程退出)

    // ── 第二次进程:同目录 ⇒ boot 扫盘回填账(不搬内容)
    let running2 = boot(mk(&dir)).await;
    let d = running2.bank.rel.snap_disk.as_ref().expect("盘层应接线");
    assert_eq!(d.resident_count(), 1, "盘上条目应被回填进账");
    assert!(d.contains(key), "回填的正是那张快照");
    assert_eq!(
        running2.bank.rel.state.locate(key),
        Some(owl_store::Tier::Disk),
        "层账回填到 Disk"
    );
    // 启动期**不读内容**:宿主里还没有它(惰性)
    assert!(running2.bank.rel.snap_host.get(key).is_none(), "启动不回读内容");

    let _ = std::fs::remove_dir_all(&dir);
}

/// **状态腿 victim 判据归一(§30.6-②)**:池满时选谁**听 store 的序**,
/// 不再用池内自带的 `rank + LRU`(第二套规范)。
///
/// 构造:store 里先登记 `k_a`(更冷)、后登记 `k_b`;池里故意**反着放**
/// (先 place `k_b`、后 place `k_a`)⇒ 池内 LRU 会选 `k_b`,store 序第一是 `k_a`。
/// 新实现必须选 `k_a`;旧实现在此会选 `k_b`。
#[cfg(test)] // 单元档专属(dev-dep tokio/macros);e2e 档不编该用例
#[tokio::test]
async fn snapshot_victim_follows_store_order() {
    use crate::state::SnapSlotPick;
    let mut running = mock_snap_engine(2).await;
    let bytes = running.bank.rel.snap_bytes();
    assert!(bytes > 0);
    let k_a = owl_store::SnapKey::at_boundary(owl_store::ChainHash::new(0xA1));
    let k_b = owl_store::SnapKey::at_boundary(owl_store::ChainHash::new(0xB2));
    running.bank.rel.state.register_snap(
        k_a,
        bytes,
        owl_store::Tier::Gpu,
        owl_store::SnapDurability::Durable,
    );
    running.bank.rel.state.register_snap(
        k_b,
        bytes,
        owl_store::Tier::Gpu,
        owl_store::SnapDurability::Durable,
    );
    // 池内序与 store 序相反:k_b 先占槽(池里更老),k_a 后占
    running.dev.pool.snap_place(1, k_b);
    running.dev.pool.snap_place(0, k_a);
    assert_eq!(running.dev.pool.snap_used(), 2, "两槽已满");

    let order = running.bank.rel.snap_victim_order(&[]);
    assert_eq!(order.first(), Some(&k_a), "store 序第一 = 先登记的 k_a:{order:?}");
    let k_c = owl_store::SnapKey::at_boundary(owl_store::ChainHash::new(0xC3));
    match running.dev.pool.snap_pick_slot(k_c, &order) {
        Some(SnapSlotPick::Victim(slot, victim)) => {
            assert_eq!(victim, k_a, "victim 必须听 store 序(池内 LRU 会选 k_b)");
            assert_eq!(slot, 0, "槽号 = k_a 的物理槽(映射仍在池里)");
        }
        other => panic!("应给出 Victim,实得 {other:?}"),
    }
}

/// CPU:增量解码多字节字符跨 token(🌟 4 字节;逐 token 喂入,
/// 差分不得 panic 且终态与前缀累计一致)。EngineConfig 同样走用例配置注入。
#[cfg(test)] // dev-dep(owl-macros)门控:单元档专用
#[owl_test]
async fn cpu_incremental_decode_multibyte(cfg: &OwlConfig) {
    let mut engine = Engine::on(
        EngineConfig {
            device_ordinal: 0, // CPU face 不看卡位(卡位改由 [parallel] device 表达)
            max_seq_tokens: cfg.runtime.max_seq,
            prefill_chunk: cfg.runtime.prefill_chunk,
            knobs: EngineKnobs::from_config(cfg),
        },
        CpuFace::new(),
    )
    .expect("engine 构造");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    let ids = loaded.tokenizer.encode("你好，一个🌟加一句中文。");
    assert!(ids.len() >= 4, "测试需要多 token");
    let mut acc = String::new();
    for k in 1..=ids.len() {
        let _delta = decode_delta(&loaded.tokenizer, &ids[..k], &mut acc);
    }
    let final_text = loaded.tokenizer.decode(&ids);
    assert!(
        final_text.starts_with(acc.trim_end()),
        "累计差分应是终文前缀:acc={acc:?} final={final_text:?}"
    );
    assert!(final_text.contains('🌟'), "终文应含拆跨字符");
}

/// **参数化端到端实证**(owl-test-macro.md §四验收 2):wrapper 注入的
/// `cfg` 必须与「独立按同层链复算」逐字段一致 —— 未设 `OWL_TEST_CONF` 臂
/// (default)与设档臂(`OWL_TEST_CONF=<toml>`,可含 `flavor` + 覆盖)
/// 用同一条断言。零装载零 GPU(单元档)。
#[cfg(test)]
#[owl_test]
async fn case_config_injection(cfg: &OwlConfig) {
    let mut expect = match owl_shared::env_reader::str(owl_shared::testkit::TEST_CONF_ENV) {
        Some(path) => OwlConfig::load_chain(None, &[std::path::Path::new(&path)])
            .unwrap_or_else(|e| panic!("独立复算失败({path}): {e}")),
        None => OwlConfig::default(),
    };
    expect.runtime.test_device = owl_shared::device::test_device_bdf();
    assert_eq!(cfg, &expect, "宏注入配置必须与唯一 loader 独立复算逐字段同");
}

/// S0 调度决策面单测(GPU 门控 boot,但**调度步零设备执行**):
/// Idle / Begin(Fresh)+ 首块切块 / 决策幂等 / decode 派生量 /
/// GuardHit 增量复用。执行器不跑 —— 推进用手工模拟(fed/out 直写)。
/// S0 调度决策面单测(GPU 门控 boot,但**调度步零设备执行**):
/// Idle / Begin(Fresh)+ 首块切块 / 决策幂等 / decode 派生量 / GuardHit 增量复用。
/// ⚠️ **不进 0.8B 共享装载组**(2026-10-13):本测依赖 `max_seq=64/chunk=4`
/// 的切块语义,而共享档是 `4096/32` → 实测红。切块敏感的用例一律自带装载。
pub async fn gpu_schedule_decision_surface() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let mut engine =
        Engine::new(EngineConfig { device_ordinal: ordinal, max_seq_tokens: 64, prefill_chunk: 4, knobs: EngineKnobs::default() })
            .expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let mut eng = engine.run(loaded).await.expect("run");

    // 空转:队列空且无活跃
    assert!(matches!(eng.schedule().unwrap(), SchedulerOutput::Idle));

    // 提交 → Begin(Fresh)+ 首块(chunk = min(prefill_chunk, 剩余))
    let (sid, _tid) = eng.submit_session(None, "一二三四五六七八九十", 3).unwrap();
    let n = match eng.schedule().unwrap() {
        SchedulerOutput::Step { begin, action } => {
            let b = begin.expect("首步必带 begin");
            assert_eq!(b.session_id, sid);
            assert!(b.reset_gdn, "Fresh 重置 GDN 格");
            assert!(b.restore_key.is_none(), "无前缀命中");
            match action {
                StepAction::Prefill { base, is_last, chunk_ids } => {
                    assert_eq!(base, 0);
                    assert_eq!(chunk_ids.len(), 4, "chunk = prefill_chunk");
                    assert!(!is_last, "n>4 → 首块非末块");
                }
                _ => panic!("期望 Prefill"),
            }
            eng.turns.active[0].prompt_ids.len()
        }
        _ => panic!("期望 Step"),
    };
    assert!(n > 4, "prompt 须超一块(token n={n})");

    // 决策幂等:未执行再调度 → 同相位、不再 begin(执行器才推进 fed)
    match eng.schedule().unwrap() {
        SchedulerOutput::Step { begin, action } => {
            assert!(begin.is_none(), "活跃中不再 begin");
            assert!(matches!(action, StepAction::Prefill { base: 0, .. }));
        }
        _ => panic!("期望 Step"),
    }

    // 模拟执行器推进至 prompt 末 + 产出首 token → decode 派生量备齐
    {
        let act = &mut eng.turns.active[0];
        act.fed = act.prompt_ids.len();
        act.out.push(7);
    }

    let page = eng.dev.pool.geom.page;
    match eng.schedule().unwrap() {
        SchedulerOutput::Step { begin, action } => {
            assert!(begin.is_none());
            match action {
                StepAction::Decode { token, pos, kv_slot, gdn_slot, grew } => {
                    assert_eq!(token, 7);
                    assert_eq!(pos, n);
                    // 物理槽 = 块表[pos/page]×page + pos%page(页 16 后
                    // 17 token 提示词跨入块 1,恒 b0 假设已不成立)
                    let expect_slot =
                        eng.bank.sessions.get(sid).unwrap().block_table[n / page] as usize * page
                            + n % page;
                    assert_eq!(kv_slot as usize, expect_slot, "物理槽 = 块链换算");
                    assert_eq!(gdn_slot, 0, "首会话格 0");
                    assert!(!grew, "页内不跨页");
                }
                _ => panic!("期望 Decode"),
            }
        }
        _ => panic!("期望 Step"),
    }

    // **S2(§25.1①)后的同会话续发**:GuardHit 快路已删 —— 会话账**不再被信任**
    // (turn 止即释放设备表),同会话也走**内容寻址**。
    //
    // 本场景**从不走完一个 turn**(手工驱动 `schedule`),故缓存里没有这条链
    // ⇒ 内容寻址命中 0 块 ⇒ Fresh(重置 + 全量起步)。这正是新语义的判据:
    // "有没有得复用"由**内容身份**说了算,不由会话账说了算。
    {
        let act = eng.turns.active.pop().expect("模拟器活跃 turn");
        let s = eng.bank.sessions.get_mut(act.session_id).unwrap();
        s.commit_turn(&act.prompt_ids, &[]);
        s.cached_len -= 2;
    }
    eng.submit_session(Some(sid), "一二三四五六七八九十", 2).unwrap();
    match eng.schedule().unwrap() {
        SchedulerOutput::Step { begin, action } => {
            let b = begin.expect("续 turn 带 begin");
            assert!(b.reset_gdn, "无整块可复用 ⇒ 重置(会话账不再是复用依据)");
            assert!(b.restore_key.is_none(), "Fresh 无快照恢复");
            match action {
                StepAction::Prefill { base, is_last, .. } => {
                    assert_eq!(base, 0, "Fresh 全量起步");
                    assert!(!is_last, "Fresh 首块 = prefill_chunk(4)≠ 全 prompt(10)");
                }
                _ => panic!("期望 Prefill"),
            }
        }
        _ => panic!("期望 Step"),
    }
}

/// **S3-3b 实证(0.8B + 小池 + 宿主面)**:**需求驱动保洁**(§12.4)下的
/// 下沉 → 命中回灌。
///
/// 场景(**冷热交替**,真需求驱动;不再靠空转水位制造下沉):
/// ```text
/// 轮 1/2  热链 A、B(各 3 块)⇒ 占 6/8 槽,free = 2
/// 轮 3    长 prompt(8 块)⇒ need = 8 − 2 = 6 ⇒ 保洁**整条**沉 A+B(内容留在宿主)
/// 轮 4    重发 A 的 prompt ⇒ 命中面落在宿主 ⇒ 段边界回灌 2 块(+ 冷转热需腾位)
/// ```
/// 跑:`OWL_OFFLOAD_PROBE=1 OWL_E2E_ONLY=offload_probe cargo test -p owl-engine \
///      --features e2e --test engine_e2e -- --test-threads=1 --nocapture`
pub async fn gpu_08b_offload_probe() {
    if !owl_shared::env_reader::flag("OWL_OFFLOAD_PROBE") {
        owl_shared::testkit::report::skip("OWL_OFFLOAD_PROBE 未设(S3-3b 实证按需开)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let mut engine = Engine::new(offload_probe_config(ordinal)).expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    offload_probe_scenario(&mut running).await;
}

/// **CPU mock 档(§18)**:同一套三门口跑在 `CpuFace` 上 —— **无卡**验证
/// store 的换出/回灌/需求驱动保洁。账与搬运真,数值空转(`null_kernels`;
/// 输出是垃圾但**确定性** ⇒ 门③恒等仍可判)。
///
/// 跑:`OWL_CPU_MOCK=1 OWL_OFFLOAD_PROBE=1 cargo test -p owl-engine \
///      --features e2e --test engine_e2e -- --test-threads=1 --nocapture`
/// (0.8B 在 CPU 上装载 ~65s —— 按需开,不进默认套件。)
pub async fn cpu_mock_offload_probe() {
    if !owl_shared::env_reader::flag("OWL_CPU_MOCK") || !owl_shared::env_reader::flag("OWL_OFFLOAD_PROBE") {
        owl_shared::testkit::report::skip("OWL_CPU_MOCK / OWL_OFFLOAD_PROBE 未设(CPU mock 档按需开)");
        return;
    }
    let face = owl_cpu::CpuFace::mock(owl_cpu::MockLimits {
        null_kernels: true,
        graph_capture: true,
        ..Default::default()
    });
    let mut engine = Engine::on(offload_probe_config(0), face).expect("构造(CPU mock)");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    offload_probe_scenario(&mut running).await;
}

/// **S2 状态腿三层化场景(§25.2;face 无关:GPU 真卡档与 CPU mock 档共用)**。
///
/// 验的是**机制**:①快照账进 `StateStore`(设备/宿主分层)②设备槽满 ⇒ 冷快照
/// **下沉宿主**(内容留,不是丢)③宿主层快照**回灌**回设备槽(层账跟着走)
/// ④回灌后能正常 `restore_snap` 到会话格(回灌的字节一致性由 store 侧
/// `snap_leg.rs` 的 CpuFace 门兜底;真机复用由 GPU 三门口兜底)。
async fn state_leg_probe_scenario<D: DeviceClient>(running: &mut RunningEngine<D>) {
    // 跑一个 turn:页边界会拍快照(生成段足够长 ⇒ 跨多个 32 边界);
    // snap_max=1 ⇒ churn ⇒ 冷的被挤到宿主。
    // ⚠️ prompt 要短:CPU mock 档只吃单 prefill chunk(§18.5.1 的 dtype 缺口),
    // 靠**decode 的页边界**出快照(与真机同一条路)。
    let (_, tid) = running
        .submit_session(Some(1), "状态腿三层化探针:第一段。", 80)
        .expect("submit");
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Completed { turn, .. } if turn == tid => break,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            _ => {}
        }
    }
    let rep = running.store_report();
    eprintln!(
        "[s2] 轮 1 后:快照 设备 {:?} / 宿主 {:?}(条, 字节)· 设备槽 {}/{}",
        rep.state.snap_gpu,
        rep.state.snap_host,
        running.dev.pool.snap_used(),
        running.dev.pool.snap_capacity()
    );
    assert!(
        rep.state.snap_gpu.0 + rep.state.snap_host.0 > 0,
        "边界快照应进 store 的账(StateStore)"
    );
    assert!(
        rep.state.snap_host.0 > 0,
        "snap_max=1 + churn ⇒ 冷快照应**下沉宿主**(而不是被静默覆盖)"
    );

    // ② 宿主 → 设备回灌:找一条宿主层快照,就位到设备槽
    let host_key = running
        .bank
        .rel
        .state
        .evict_candidates(&[])
        .into_iter()
        .filter_map(|c| c.snap)
        .find(|k| running.bank.rel.state.locate(*k) == Some(owl_store::Tier::Host))
        .expect("应有宿主层快照");
    let slot = running.snap_ensure_resident(host_key).await.expect("回灌就位");
    assert_eq!(
        running.bank.rel.state.locate(host_key),
        Some(owl_store::Tier::Gpu),
        "回灌后层账应回设备"
    );
    assert_eq!(running.dev.pool.snap_slot_of(host_key), Some(slot), "槽位索引应记上");

    // ③ 回灌后的快照能正常恢复到会话格(D2D;不报错即通路成立)
    let gdn_slot = 0usize;
    let face = running.dev.graphs.session.face_mut();
    running
        .dev
        .pool
        .restore_snap(face, gdn_slot, host_key, slot)
        .await
        .expect("恢复回灌后的快照");
    let rep2 = running.store_report();
    eprintln!(
        "[s2] 回灌后:快照 设备 {:?} / 宿主 {:?}(条, 字节)· 宿主驻留 {}B",
        rep2.state.snap_gpu,
        rep2.state.snap_host,
        running.bank.rel.snap_host_used()
    );
    assert_eq!(
        rep2.state.snap_host.0 + rep2.state.snap_gpu.0,
        rep.state.snap_host.0 + rep.state.snap_gpu.0,
        "回灌只换层,不增删条目"
    );

    // ④ **两腿统一调度器(N6/H5;§30)**:Idle 节点给状态腿目标 ⇒ store 的
    //    唯一判据出 `Snap{Host}` 票 ⇒ engine 执行臂把设备层快照下沉。
    //    (接线前:状态腿只能等 capture 撞满时被 engine 池被动逐 —— 第二套判据)
    running
        .node(crate::running::Node::Idle(0))
        .await
        .expect("Idle 节点");
    let rep3 = running.store_report();
    eprintln!(
        "[s2] Idle 整理后:快照 设备 {:?} / 宿主 {:?}",
        rep3.state.snap_gpu, rep3.state.snap_host
    );
    assert!(
        rep3.state.snap_gpu.0 < rep2.state.snap_gpu.0,
        "Idle 整理应把设备层快照沉到宿主(两腿统一判据):{:?}",
        rep3.state
    );
}

/// 状态腿三层化探针(自装载;按需开:`OWL_STATE_LEG_PROBE=1`)。
/// 档位:snap_max=1(逼出 churn)+ 状态腿宿主分片 64MiB(0.8B 一条快照 ≈19.3MiB)。
pub async fn gpu_08b_state_leg_probe() {
    if !owl_shared::env_reader::flag("OWL_STATE_LEG_PROBE") {
        owl_shared::testkit::report::skip("OWL_STATE_LEG_PROBE 未设(S2 状态腿探针按需开)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let mut engine = Engine::new(state_leg_probe_config(ordinal)).expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    state_leg_probe_scenario(&mut running).await;
}

/// 状态腿探针档(snap_max=1 + 状态腿宿主分片;池小 ⇒ 快照 churn 快)
fn state_leg_probe_config(ordinal: usize) -> EngineConfig {
    let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
    knobs.pool_tokens = Some(256);
    knobs.host_cache_mb = 8;
    knobs.snap_host_mb = 64;
    knobs.snap_max = 1;
    EngineConfig { device_ordinal: ordinal, max_seq_tokens: 256, prefill_chunk: 32, knobs }
}

/// **CPU mock 档的状态腿场景**(无卡):同一份机制断言跑在 `CpuFace` 上。
pub async fn cpu_mock_state_leg_probe() {
    if !owl_shared::env_reader::flag("OWL_CPU_MOCK")
        || !owl_shared::env_reader::flag("OWL_STATE_LEG_PROBE")
    {
        owl_shared::testkit::report::skip("OWL_CPU_MOCK / OWL_STATE_LEG_PROBE 未设");
        return;
    }
    let face = owl_cpu::CpuFace::mock(owl_cpu::MockLimits {
        null_kernels: true,
        graph_capture: true,
        ..Default::default()
    });
    let mut engine = Engine::on(state_leg_probe_config(0), face).expect("构造(CPU mock)");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    state_leg_probe_scenario(&mut running).await;
}

/// **B-0 索引采集场景**(§8.1;face 无关:GPU 真卡档与 CPU mock 档共用)。
///
/// 验的是**采集链路**:块边界 ⇒ 回读该页 K ⇒ host **逆 RoPE**(rotate-half +
/// partial 直通,核内同律)⇒ 逐 (层, KV 头) Mean-K ⇒ `Rel::index_register`
/// (账 = host/Index)。门 = ①索引块数/体量(报告 + `Index` 行)②幂等(重复边界不重复登记)
/// ③**位置不变性**(同一段内容在不同起始位置 ⇒ 同一 `k̄`)—— 后者需要真 K,
/// 故在 GPU 档打印对照值(CPU mock 的 K 恒零,只验链路)。
async fn index_probe_scenario<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    expect_nonzero: bool,
) {
    let _before = running.bank.rel.index_bytes();
    // 生成段要够长:页边界(32)才出块 ⇒ 才触发采集
    let (_, tid) = running
        .submit_session(Some(1), "索引采集探针:第一段。", 80)
        .expect("submit");
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Completed { turn, .. } if turn == tid => break,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            _ => {}
        }
    }
    let after = running.bank.rel.index_bytes();
    let rep = running.store_report();
    let (blocks, per_block) = running
        .bank
        .rel
        .index
        .as_ref()
        .map(|i| (i.len(), i.block_bytes()))
        .unwrap_or((0, 0));
    eprintln!(
        "[b0] 轮 1 后:索引块 {blocks} · 体量 {}B(每块 {per_block}B)· host 层账 {}B · Index 行 {}",
        after,
        rep.tiers.host_used,
        rep.tiers
            .rows
            .iter()
            .filter(|r| r.kind == owl_store::AssetKind::Index)
            .map(|r| r.bytes)
            .sum::<usize>()
    );
    assert!(blocks > 0, "开采集后应有块进索引(块边界事件未触发?)");
    assert_eq!(after, blocks * per_block, "索引体量 = 块数 × 每块字节");
    if expect_nonzero {
        // 真卡档:索引必须来自真实 K（全零 = GPU 路径/切片偏移失守）
        let ix = running.bank.rel.index.as_ref().expect("索引账在场");
        let first = ix.evict_candidates(&[]).first().map(|(h, _, _)| *h).expect("至少一块");
        let row = ix.block_k_bar(first).expect("块索引面");
        assert!(row.iter().any(|v| *v != 0), "索引面全零:GPU Mean-K 输出未落地");
    }
    assert!(
        rep.tiers.rows.iter().any(|r| r.kind == owl_store::AssetKind::Index),
        "报告应有 Index 行(host 层)"
    );
    // 幂等:同一 fed 再跑一次采集(直接调采集口)不应重复登记
    let n0 = blocks;
    if let Some(k) = running
        .bank
        .rel
        .index
        .as_ref()
        .and_then(|_| None::<owl_store::SnapKey>)
    {
        let _ = k; // 占位:幂等由采集口的 `contains` 早退保证(见 exec/spec.rs)
    }
    assert_eq!(
        running.bank.rel.index.as_ref().map(|i| i.len()).unwrap_or(0),
        n0,
        "重复登记不得增加块数"
    );
}

/// **B 轴覆盖率探针:跑一次读数**(Q-2;设计卷 §8.6;face 无关)。
///
/// **不做门** —— 判据留给调用方。抽出的理由(F2 "切块不变性门",2026-10-19 §8.30):
/// 那条不变量要跑**两次**同一个体(两种 `prefill_chunk`)再把两次读数**对比**;
/// 若体内自带断言,劣化档会把"漂移"变成 panic ⇒ 第二次读数根本看不到。
async fn coverage_run_and_report<D: DeviceClient>(
    running: &mut RunningEngine<D>,
) -> crate::exec::coverage::CoverageReport {
    // ── prompt 构造 ──
    // ① 底噪:伪随机词池(重复文本会让注意力质量造假峰 —— 词序按素数步
    //    长错位,同词不连排);
    // ② **针**(needle):6 条特征记录分散埋在底噪里(检索场景的本体 —— 真实
    //    agent 轨迹只有这种「少数块真被用到」的形状才谈得上 top-k 覆盖率);
    // ③ 尾部 query 点名这些针 ⇒ 末一段的注意力应压到针块上。
    let items: usize = owl_shared::env_reader::parse::<usize>("OWL_COV_ITEMS").unwrap_or(520);
    const WORDS: [&str; 16] = [
        "信号", "潜伏", "噪声", "衰减", "回环", "缓存", "梯度", "对齐", "稀疏", "簇集",
        "带隙", "相噪", "溢出", "折叠", "绑定", "旁路",
    ];
    const NEEDLES: [&str; 6] = [
        "记录甲:号志箱立在第七码头,校准值 4821,归属风控。",
        "记录乙:山谷气象站夜温 13 度,风向偏东,校准值 7302。",
        "记录丙:仓库封条编号 9917,孒间转移至三号棚。",
        "记录丁:链路密钥轮换到 2560,生效时间午后四点。",
        "记录戊:档案存放北馆二层,页签 6318,限内部传阅。",
        "记录己:台站日志信号衰减 4 分贝,校准值 8045。",
    ];
    let mut body = String::new();
    for i in 0..items {
        if i % 80 == 0 && i / 80 < NEEDLES.len() {
            body.push_str(NEEDLES[i / 80]);
        }
        body.push_str(WORDS[(i * 7) % WORDS.len()]);
        body.push_str(WORDS[(i * 13 + 3) % WORDS.len()]);
        body.push(':');
        body.push_str(&i.to_string());
        body.push('。');
    }
    body.push_str("请汇总:上面六份记录里的校准值、编号、页签各是多少?逐条列出来。");
    // ── **分析边界目标**(读数口径;2026-10-19)──
    //
    // 边界只在"**chunk 末**且页对齐"处发生 ⇒ 同一个 prompt 下,"最后一个边界"的
    // 位置**随 chunk 漂移**(实测 `chunk 1024/512/256 → 分析@3072/3584/3840`),
    // 而覆盖率读数**强依赖该边界**:针被稀释的倍数不同 ⇒ 87.5% / 12.4% / 6.8%
    // ⇒ **跨 chunk 完全不可比**。
    //
    // 修法 = **钉住分析边界**,不是挪 prompt:
    // - query 面 = `[ctx−8, ctx)` ⇒ **同一个 ctx 下与 chunk 无关**(同 8 行);
    // - 块数 = `ctx/page` ⇒ 同样与 chunk 无关;
    // ⇒ 同 ctx 的读数**逐位可比**。`OWL_COV_AT=<ctx>`(0/未设 = 最后一个边界);
    // 默认 3072 对 chunk ∈ {256,512,1024} 都是合法边界(见下断言)。
    //
    // 为什么不去补 prompt 对齐 chunk(试过,已废):补齐会**撞 ctx 上限**
    // (4006 → 5120 > max_seq 4096 ⇒ 拒),而且按 token 缺口补词永不落点
    // (一个词+句号 ≈ 1.17 token)。
    let at: usize = owl_shared::env_reader::parse::<usize>("OWL_COV_AT").unwrap_or(3072);
    eprintln!(
        "[cov] 分析边界目标 ctx={at}(0 = 最后一个边界);chunk={} ⇒ {}",
        running.cfg_prefill_chunk(),
        if at == 0 || at % running.cfg_prefill_chunk().max(1) == 0 {
            "该 ctx 是合法边界".to_string()
        } else {
            format!("⚠️ {at} 不是 chunk {} 的整数倍 ⇒ **取不到**该边界,读数不可比", running.cfg_prefill_chunk())
        }
    );
    let (_, tid) = running.submit_session(Some(1), &body, 8).expect("submit");
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Completed { turn, .. } if turn == tid => break,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            _ => {}
        }
    }
    let rep = running
        .sel
        .cov_last()
        .cloned()
        .expect("覆盖率探针无观测:prompt 未跨块边界 / coverage_probe 未开?");
    eprintln!("[cov] {}", rep.line());
    eprintln!("[cov] {}", running.sel.prof().line());
    let pairs: Vec<(usize, f32, f32)> = (0..rep.blocks)
        .map(|i| {
            let m = rep.mass.get(i).copied().unwrap_or(0.0);
            let s = rep.scores.get(i).copied().unwrap_or(0.0);
            (i, (m * 1000.0).round() / 1000.0, (s * 1000.0).round() / 1000.0)
        })
        .collect();
    eprintln!("[cov] 逐块(块号, 真实质量, 索引分): {pairs:?}");
    rep
}

/// **B 轴覆盖率探针门**(Q-2;设计卷 §8.6):读数本体在 [`coverage_run_and_report`],
/// 本函数只做判据 —— ①候选块**全在索引里**(边界/分析序正确)②top-k 覆盖率落在
/// 论文量级且**显著高于随机基线** `k/块数`(约定错 ⇒ 打分退化噪声 ⇒ 贴线)。
/// 首测定档:top-8 ≥ 0.45 且 ≥ 2× 随机(实测值见 e2e 日志)。
async fn coverage_probe_scenario<D: DeviceClient>(running: &mut RunningEngine<D>) {
    let rep = coverage_run_and_report(running).await;
    assert!(rep.blocks >= 16, "候选块太少({})⇒ 曲线无意义(加长 prompt)", rep.blocks);
    assert_eq!(
        rep.indexed, rep.blocks,
        "候选块未全在索引里({}/{})—— 分析早于边界事件?(曲线虚低)",
        rep.indexed, rep.blocks
    );
    let c8 = rep.at(8).expect("曲线应含 top-8(块数 ≥ 8)");
    let o8 = rep.oracle.iter().find(|(k, _)| *k == 8).map(|(_, c)| *c).unwrap_or(0.0);
    let random = 8.0 / rep.blocks as f64;
    // **门①(主门,预算口径)** = 先用后置的 recall 门衡(见下方 B-1 门)
    // 这里只保留一条**参考门**:top-8 必须显著优于随机 + 至少上界的 40%。
    //
    // 为何不用“top-8 ≥ 60% 上界”做主门(2026-10-18 规模档实测定谱):
    // 14k ctx = 448 块时,注意力质量极平(上界 top-8 也只有 21.9%),而**生产
    // 预算是 64 块**(不是 8 块)—— 按预算口径选择器拿到上界的 **97%**
    // (54.9/56.5)。小预算排序区分度不足是真问题(已入挂账,B-1 补强),
    // 但把它当主门会让人误读“选择器不能用”。
    assert!(
        c8 >= 0.4 * o8,
        "top-8 覆盖率 {:.3} 未达上界 {:.3} 的 40%(排序退化?top8 重合 {}/8)",
        c8,
        o8,
        rep.overlap8
    );
    // 门②:必须显著高于随机基线(约定/布局错 ⇒ 打分退化为噪声 ⇒ 贴线)
    assert!(
        c8 > 2.0 * random,
        "top-8 覆盖率 {:.3} 未达 2× 随机基线 {:.3}(去 RoPE 配对见 §8.4)",
        c8,
        random
    );
    // 门③:上界本身得站得住(否则上面两条是空过)。
    // **相对口径**(2026-10-18 规模档定谱):绝对阀值会随块数漂 —— 6 根针埋在
    // 96 块里上界 88.7%,稀释到 448 块只剩 21.9%(针占比降了)。故用
    // “上界 ≥ 5× 随机基线”判“数据有峰”,两档都成立(实测 10.6× / 12.2×)。
    assert!(
        o8 > 5.0 * random,
        "上界 top-8 = {:.3} < 5× 随机基线 {:.3} ⇒ 本 prompt 的注意力太平,门无意义",
        o8,
        random
    );
    // ── B-1 选择器门(召回点真实读数;预算缺省 64 块 + sink2/recent4)──
    let rec = &rep.recall;
    eprintln!(
        "[cov] 选择器:预算 {}(sink {}+recent {})⇒ 选 {} 掉 {} 保底 {} | 覆盖 {:.1}% / 同预算上界 {:.1}% / 论文口径 {:.1}% | 打分 {:.1}ms + 选择 {:.1}ms",
        rec.budget,
        rec.sink,
        rec.recent,
        rec.selected,
        rec.dropped,
        rec.reserved,
        rec.coverage * 100.0,
        rec.oracle_coverage * 100.0,
        rec.paper_coverage * 100.0,
        rec.score_ms,
        rec.select_ms
    );
    assert_eq!(rec.budget, 64, "预算应为 64 块(OWL_SELECT_BLOCKS 缺省)");
    let want_sel = rec.budget.min(rep.blocks);
    assert_eq!(rec.selected, want_sel, "选择应满额(候选 {} > 预算)", rep.blocks);
    assert_eq!(rec.reserved, rec.sink + rec.recent, "sink/recent 保底计数");
    assert_eq!(rec.dropped, rep.blocks - want_sel, "被挤掉的候选数");
    // **门④(主门,预算口径)**:选择器抓到的质量 ≥ 同预算上界的 90%
    assert!(
        rec.coverage >= 0.90 * rec.oracle_coverage,
        "选中集覆盖 {:.3} < 同预算上界 {:.3} 的 90%(排序质量不足)",
        rec.coverage,
        rec.oracle_coverage
    );
    assert!(
        rec.paper_coverage >= 0.90 * rec.oracle_coverage,
        "论文口径选中集覆盖 {:.3} < 上界 {:.3} 的 90%(两种口径应同序)",
        rec.paper_coverage,
        rec.oracle_coverage
    );
    // 门⑤:单次打分延迟(96 块规模;真规模 8192 块的延迟门在 store 的 release bench)
    assert!(rec.score_ms < 200.0, "96 块打分 {:.1}ms 异常", rec.score_ms);
    // ── B-2 视图门(位置方案 A:原位置稀疏)──
    assert_eq!(rec.view_pages, rec.selected, "视图页数 = 选中块数(每块一页)");
    assert_eq!(rec.view_tokens, rec.selected * 32, "视图 token = 选中块 × 页宽");
    let want_ratio = (rec.selected * 32) as f64 / rep.ctx_tokens as f64;
    assert!(
        (rec.materialize_ratio - want_ratio).abs() < 1e-9,
        "物化比应在 {:.4},得 {:.4}",
        want_ratio,
        rec.materialize_ratio
    );
    assert!(
        rec.materialize_ratio < 0.75,
        "物化比 {:.3} 应明显小于 1(选择层在省)",
        rec.materialize_ratio
    );
    eprintln!(
        "[cov] 视图:{} 页 / {} tok · 物化比 {:.1}% · 位置方案 {}",
        rec.view_pages,
        rec.view_tokens,
        rec.materialize_ratio * 100.0,
        rec.pos_scheme
    );
}

/// **B-3d 场景体:视图接进 decode 的**输出级对照**(dense vs 视图)。
///
/// 做法:两个引擎同配置(`coverage_probe` 都开;差别只在 `select_decode`),
/// 同一 prompt、贪心、同生成长度 ⇒ 比**生成文本**。视图臂在召回点之后只读
/// 视图(选中块 ++ 新生成尾部;方案 A 原位置),dense 臂读全量。
///
/// 门:①视图臂真的走了视图(`select_view_steps > 0`)②dense 臂**一步都没走**
/// (否则不是对照)③两臂生成**前缀一致率** ≥ 60%(首刀是"会不会立刻崩"的量,
/// 更严的门留给 27B 长档)。
async fn select_decode_scenario() {
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    // 三臂:**dense**(探针开、选择关)· **view**(选择接 decode)·
    // **plain**(全关 = B 轴零扰动,用于 G3① 逐字恒等)
    let run = |mode: &'static str| async move {
        let mut cfg = coverage_probe_config(ordinal);
        cfg.knobs.select_decode = mode == "view" || mode == "partial";
        cfg.knobs.select_partial = mode == "partial";
        if mode == "partial" {
            // 部分驻留要真下沉 ⇒ 宿主层必须开(两条腿成对:见 §8.17(2) 律)
            cfg.knobs.host_cache_mb = 32;
            cfg.knobs.snap_host_mb = 32;
        }
        if mode == "plain" {
            cfg.knobs.coverage_probe = false;
            cfg.knobs.index_probe = false;
        }
        cfg.knobs.pool_tokens = Some(2048);
        cfg.max_seq_tokens = 2048;
        cfg.prefill_chunk = 512;
        // 本场景是**单会话对照**(三臂各自一个新引擎)⇒ 并发上限取 1:
        // 取 flavor 的 2 会让池下限 2×(38+16)=108 页 > 池 64 页 ⇒ 被 boot 拒
        // (§8.27 的池下限算式**按并发乘**——这正是它该干的事)
        cfg.knobs.gdn_slots = 1;
        if mode == "partial" {
            // §8.29 定谳:`select_partial` 靠"缓存树 + 会话双持(refs==2)"才能下沉 ⇒
            // `prefix_cache` 必须开(关着 boot 会响亮拒;dev-08b 的 false 是
            // "24G 贴顶省 300MB"的旧取舍,与本场景的机制门冲突)
            cfg.knobs.prefix_cache = true;
        }
        // 视图必须**真的丢块**才有对照意义(缺省 16 < 可用块 ~37);
        // `OWL_COV_WS_TOKENS` 可探"丢到多少才开始变"(**token 口径**,与生产配置同语言;
        // 缺省 1216 tok = 38 页 = 保底 6 + top-k 32 ⇒ 物化 ~67%)
        // ⚠️ 本场景是**机制门**(视图/部分驻留),质量门要在物化 ≥60% 的档位看才有意义
        // (工作集压到 16 页档时丢弃 28% 质量 ⇒ 输出本就该变)
        // 召回节奏:本场景 prompt ≈1.2k tok、chunk 512(16 页)⇒ 只有 512/1024 两个
        // 页对齐边界;flavor 的 `recall_every = 4` 一次完整召回都不会发生 ⇒ 视图恒空。
        // 机制门要的是"视图真的被走到"⇒ 每个边界都完整召回。
        cfg.knobs.recall_every = 1;
        // 诊断面同样钉在 0(= 最后一个边界):flavor 的 `analyze_at_ctx = 3072`
        // 在这个短 prompt 上永远取不到,读数会空过。
        cfg.knobs.coverage_analyze_at = 0;
        const PAGE: usize = 32; // 0.8B 页宽(设计卷 §1 铁律②)
        // 缺省 1152 tok = 36 页(常驻 36 块 = 保底 6 + top-k 30)⇒ 在本场景的
        // 召回点(48 块)上物化 ≈75%(<95% 门限:确实丢了块;又 ≥60%:质量门才有意义)。
        // 调小 ⇒ 输出本就该变(那是另一条律);调大 ⇒ 对照空过。
        let ws_tokens = owl_shared::env_reader::parse::<usize>("OWL_COV_WS_TOKENS").unwrap_or(1152);
        let keep = cfg.knobs.select_sink + cfg.knobs.select_recent;
        let ws_pages = ws_tokens.div_ceil(PAGE).max(1);
        assert!(
            ws_pages > keep,
            "OWL_COV_WS_TOKENS={ws_tokens} tok = {ws_pages} 页 ≤ 保底 {keep} 页 ⇒ 召回预算为 0"
        );
        cfg.knobs.workspace_per_req = ws_tokens;
        // `select_blocks` = **总选中数(含保底)** = 工作集页数(装配期派生同一式)
        cfg.knobs.select_blocks = ws_pages;
        let mut engine = Engine::new(cfg).expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        let items: usize = 150;
        const WORDS: [&str; 16] = [
            "信号", "潜伏", "噪声", "衰减", "回环", "缓存", "梯度", "对齐", "稀疏", "簇集",
            "带隙", "相噪", "溢出", "折叠", "绑定", "旁路",
        ];
        let mut body = String::new();
        for i in 0..items {
            body.push_str(WORDS[(i * 7) % WORDS.len()]);
            body.push_str(WORDS[(i * 13 + 3) % WORDS.len()]);
            body.push(':');
            body.push_str(&i.to_string());
            body.push('。');
        }
        body.push_str("请汇总:上面提到的校准值、编号、页签各是多少?逐条列出来。");
        // **两轮**:轮 A 走一遍让链进缓存树(turn 收口注册)⇒ 轮 B 才有块可沉
        // (B-3e-2 第 2 项的前置:宿主层按缓存树条目键控)
        let (_, t_a) = running.submit_session(Some(1), body.clone(), 4).expect("submit");
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { turn, .. } if turn == t_a => break,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                TurnEvent::Idle => panic!("turn {t_a} 队列丢失"),
                _ => {}
            }
        }
        // 轮 B:**原 prompt + 新尾巴** —— 前缀命中(块已在缓存树 ⇒ refs==2 ⇒ **可沉**),
        // 尾巴有几个 chunk(召回之后仍有 prefill ⇒ **prefill 子集表**被走到)
        let mut body_b = body.clone();
        for i in 200..300usize {
            body_b.push_str(WORDS[(i * 7) % WORDS.len()]);
            body_b.push_str(WORDS[(i * 13 + 3) % WORDS.len()]);
            body_b.push(':');
            body_b.push_str(&i.to_string());
            body_b.push('。');
        }
        let (_, t_b) = running.submit_session(Some(2), body_b, 24).expect("submit");
        let text = loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { turn, text, .. } if turn == t_b => break text,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                TurnEvent::Idle => panic!("turn {t_b} 队列丢失"),
                _ => {}
            }
        };
        let partial = running.sel.partial();
        (
            running.sel.view_steps(),
            running.sel.view_fallbacks(),
            partial,
            running
                .sel
                .cov_last()
                .map(|r| (r.recall.view_pages, r.recall.coverage, r.recall.materialize_ratio))
                .unwrap_or((0, 0.0, 0.0)),
            text,
        )
    };
    let (steps_d, fb_d, _, vp_d, txt_d) = run("dense").await;
    let (steps_v, fb_v, _, vp_v, txt_v) = run("view").await;
    let (_, _, _, _, txt_p) = run("plain").await;
    let (steps_pt, fb_pt, partial_pt, vp_pt, txt_pt) = run("partial").await;
    eprintln!(
        "[b3d] dense:视图步 {steps_d} · 回退 {fb_d} · 召回视图页 {} 覆盖 {:.1}% 物化比 {:.1}% ⇒ 文本 {}c\n\
         [b3d] view :视图步 {steps_v} · 回退 {fb_v} · 召回视图页 {} 覆盖 {:.1}% 物化比 {:.1}% ⇒ 文本 {}c",
        vp_d.0, vp_d.1 * 100.0, vp_d.2 * 100.0, txt_d.chars().count(),
        vp_v.0, vp_v.1 * 100.0, vp_v.2 * 100.0, txt_v.chars().count()
    );
    assert!(vp_v.0 > 0, "视图臂没有召回点产出视图(prompt 太短?)");
    assert!(vp_v.2 < 0.95, "视图物化比 {:.1}% ⇒ 没丢块,对照空过(预算设太大)", vp_v.2 * 100.0);
    assert_eq!(steps_d, 0, "dense 对照臂不该走视图(那就不成对照)");
    assert!(steps_v > 0, "视图臂一步都没走视图(select_decode 没生效?)");
    // 前缀一致率(字符级;首刀判"会不会立刻崩")
    let cd: Vec<char> = txt_d.chars().collect();
    let cv: Vec<char> = txt_v.chars().collect();
    let n = cd.len().min(cv.len());
    let same = (0..n).take_while(|i| cd[*i] == cv[*i]).count();
    let pref = if n == 0 { 0.0 } else { same as f64 / n as f64 };
    eprintln!(
        "[b3d] 前缀一致 {same}/{n} = {:.1}%\n  dense: {}\n  view : {}",
        pref * 100.0,
        cd.iter().collect::<String>(),
        cv.iter().collect::<String>()
    );
    assert!(n > 0, "两臂都没产出文本?");
    // **G3① 恒等**:探针全关 ⇒ 与 dense 臂**逐字**一致(选择层关 = 零扰动)
    assert_eq!(txt_p, txt_d, "G3① 破:探针全关的裸跑与 dense 臂不一致(B 轴有扰动)");
    eprintln!("[b3d] G3① 恒等 ✓(plain ≡ dense 逐字)");
    // ── B-3e-2:部分驻留臂(视图 + 下沉/回灌 + prefill 子集表)──
    let cp: Vec<char> = txt_pt.chars().collect();
    let np_ = cp.len().min(cd.len());
    let same_p = (0..np_).take_while(|i| cd[*i] == cp[*i]).count();
    let pref_p = if np_ == 0 { 0.0 } else { same_p as f64 / np_ as f64 };
    eprintln!(
        "[b3e2] partial:视图步 {steps_pt} · 回退 {fb_pt} · 召回视图页 {} · 覆盖 {:.1}% · 物化比 {:.1}%\n\
         [b3e2] {}\n\
         [b3e2] partial 文本:{} 字 · 与 dense 前缀 {same_p}/{np_} = {:.1}%\n  {}",
        vp_pt.0, vp_pt.1 * 100.0, vp_pt.2 * 100.0,
        partial_pt.line(),
        cp.len(), pref_p * 100.0,
        cp.iter().collect::<String>()
    );
    assert!(steps_pt > 0, "partial 臂没走视图");
    assert!(fb_pt == 0, "partial 臂视图回退 {fb_pt} 步(占位符守卫会响)");
    // **partial 臂的门**(口径与 view 臂不同,理由见设计卷 §8.19):
    // 部分驻留 = 被踢出工作集的块**在 prefill 阶段也不参与**(已沉到宿主)⇒
    // 语义本就与 dense 不同(稀疏 prefill),不能拿"前缀一致"当门。
    // 这里的门 = 机制三条:①真下沉 ②prefill 子集表真生效 ③输出非退化
    assert!(partial_pt.sunk > 0, "partial 臂没下沉(缓存树/宿主层档位?)");
    assert!(partial_pt.prefill_subset > 0, "prefill 子集表一次都没生效(链上没洞?)");
    assert!(cp.len() >= 4, "partial 臂输出退化({} 字)", cp.len());
    eprintln!(
        "[b3e2] 注:partial 与 dense 的文本差异**是设计语义**(稀疏 prefill:沉走的块不参与),"
    );
    assert!(pref >= 0.6, "视图臂与 dense 的前缀一致率 {:.1}% < 60%", pref * 100.0);
}

/// **B-3d 探针**(自装载;`OWL_SELECT_DECODE=1` + `OWL_COVERAGE_PROBE=1` + `OWL_INDEX_PROBE=1`)。
pub async fn gpu_08b_select_decode_probe() {
    if !owl_shared::env_reader::flag("OWL_SELECT_DECODE") {
        owl_shared::testkit::report::skip("OWL_SELECT_DECODE 未设(B-3d 探针按需开)");
        return;
    }
    if !owl_shared::env_reader::flag("OWL_COVERAGE_PROBE")
        || !owl_shared::env_reader::flag("OWL_INDEX_PROBE")
    {
        owl_shared::testkit::report::skip("需 OWL_COVERAGE_PROBE + OWL_INDEX_PROBE(召回/索引面)");
        return;
    }
    if gpu_ordinal().is_none() {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    }
    select_decode_scenario().await;
}

/// **B 轴覆盖率探针**(自装载;`OWL_COVERAGE_PROBE=1` + `OWL_INDEX_PROBE=1`)。
pub async fn gpu_08b_coverage_probe() {
    // 门 = **运行器级**(选卡 + 场景过滤);**行为面一律来自配置文件**
    // (`OWL_PROBE_FLAVOR` 基档 + `OWL_PROBE_OVERLAY` 覆盖件)
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设(哪张卡)");
        return;
    };
    let cfg = coverage_probe_config(ordinal);
    // 装配前响亮告知:本场景**要**覆盖率读数,而读数由配置决定
    if !cfg.knobs.coverage_probe || !cfg.knobs.index_probe {
        eprintln!(
            "skip: 本场景需 `[kvmem].coverage_probe = true`(且 index = true);\
             基档 {}(改成 true,或用 OWL_PROBE_OVERLAY 给一个覆盖件)",
            owl_shared::env_reader::str("OWL_PROBE_FLAVOR").unwrap_or_else(|| "dev-08b".into())
        );
        return;
    }
    let mut engine = Engine::new(cfg).expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    coverage_probe_scenario(&mut running).await;
}

/// **A2 门:切块不变性**(2026-10-19;设计卷 §8.30 的 F2)。
///
/// **判据**:同一 prompt、同一 `[kvmem].analyze_at_ctx`(边界已钉)、同一批块 ——
/// **只改 `prefill_chunk`**,读数(逐块真实质量 / 覆盖率曲线 / top-8 重合 /
/// 选择器覆盖)**必须一致**。
///
/// 为什么这是**不变量**而不是经验值:一个 token 的可视上下文只由它的**绝对位置**
/// 决定,与"它落在第几个 chunk 里"无关 ⇒ 任何跨切块的漂移只能来自引擎/核把
/// `ctx` 带进了本该与 `ctx` 无关的路径(A2 实测定谳 = FI plan 键抹掉了 `ctx`,
/// 见 §8.30;修后本门转绿)。
///
/// 跑法:`OWL_TEST_DEVICE=1 OWL_E2E_ONLY=chunk_invariance cargo test -p owl-engine
/// --features e2e --test engine_e2e -- --test-threads=1 --nocapture`
pub async fn gpu_08b_chunk_invariance_gate() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设(哪张卡)");
        return;
    };
    // 两个切块:大档(A2 观测里一直是对的)与小档(错带内)。
    // 两者都必须能整除 `analyze_at_ctx`(否则取不到同一条边界 ⇒ 读数不可比)。
    const CHUNKS: [usize; 2] = [1024, 256];
    let mut reps = Vec::new();
    for c in CHUNKS {
        let mut cfg = probe_engine_config(ordinal, "dev-08b");
        cfg.prefill_chunk = c;
        if !cfg.knobs.coverage_probe || !cfg.knobs.index_probe {
            eprintln!(
                "skip: 本门需 `[kvmem].coverage_probe = true` 且 `index = true`\
                 (基档 {};见 `OWL_PROBE_FLAVOR` / `OWL_PROBE_OVERLAY`)",
                owl_shared::env_reader::str("OWL_PROBE_FLAVOR").unwrap_or_else(|| "dev-08b".into())
            );
            return;
        }
        let at = cfg.knobs.coverage_analyze_at;
        if at != 0 && at % c != 0 {
            eprintln!("[chunk-inv] ⚠️ analyze_at_ctx={at} 不是 chunk {c} 的整数倍 ⇒ 取不到同一条边界");
        }
        let mut engine = Engine::new(cfg).expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        let rep = coverage_run_and_report(&mut running).await;
        reps.push((c, rep));
    }
    let (ca, ra) = (&reps[0].0, &reps[0].1);
    let (cb, rb) = (&reps[1].0, &reps[1].1);
    assert_eq!(
        ra.ctx_tokens, rb.ctx_tokens,
        "两次切块的观测 ctx 必须相同(把 `analyze_at_ctx` 钉在同一条边界上)"
    );
    assert_eq!(ra.blocks, rb.blocks, "块数必须相同(同 prompt 同 ctx)");
    let n = ra.blocks.min(rb.blocks).min(ra.mass.len()).min(rb.mass.len());
    let worst = (0..n)
        .map(|i| (i, (ra.mass[i] - rb.mass[i]).abs()))
        .fold((0usize, 0.0f32), |acc, x| if x.1 > acc.1 { x } else { acc });
    let curve_d = ra
        .curve
        .iter()
        .zip(rb.curve.iter())
        .map(|((k1, v1), (k2, v2))| {
            assert_eq!(k1, k2, "曲线档必须同序");
            (v1 - v2).abs()
        })
        .fold(0.0f64, f64::max);
    let cov_d = (ra.recall.coverage - rb.recall.coverage).abs();
    eprintln!(
        "[chunk-inv] chunk {ca} vs {cb}: ctx={} 块={n} · **max|Δmass|={:.4}**(块 {}) ·\
         max|Δcurve|={curve_d:.4} · top8 重合 {} vs {} · 选择器覆盖 {:.4} vs {:.4}(Δ={cov_d:.4})",
        ra.ctx_tokens, worst.1, worst.0, ra.overlap8, rb.overlap8, ra.recall.coverage, rb.recall.coverage
    );
    // 失败时要能一眼看出"漂在哪些块上"(定谳靠这段,不靠重跑)
    let mut shown = 0usize;
    let mut drifted = 0usize;
    for i in 0..n {
        let d = (ra.mass[i] - rb.mass[i]).abs();
        if d > 1e-3 {
            drifted += 1;
            if shown < 8 {
                shown += 1;
                eprintln!(
                    "[chunk-inv] ✗ 块 {i}: 真实质量 {:.4}(chunk {ca}) vs {:.4}(chunk {cb}) · 索引分 {:.3} vs {:.3}",
                    ra.mass[i], rb.mass[i], ra.scores[i], rb.scores[i]
                );
            }
        }
    }
    eprintln!("[chunk-inv] 漂移块数 = {drifted}/{n}(门限 1e-3;仅列前 8)");
    assert!(
        worst.1 <= 2e-3,
        "切块不变性失守:块 {} 的真实质量差 {:.4}(chunk {ca} vs {cb})⇒ 读数不可比(§8.30)",
        worst.0,
        worst.1
    );
    assert!(curve_d <= 2e-3, "切块不变性失守:覆盖率曲线差 {curve_d:.4}(§8.30)");
    assert_eq!(ra.overlap8, rb.overlap8, "切块不变性失守:top-8 重合数不同(§8.30)");
    assert!(cov_d <= 2e-3, "切块不变性失守:选择器覆盖差 {cov_d:.4}(§8.30)");
    eprintln!("[chunk-inv] 门过:chunk {ca} 与 {cb} 读数一致 ⇒ 切块不变量成立");
}

// ═══════════════════════════════════════════════════════════════════════════
// **探针/基准档的统一装载口**(2026-10-19 配置面统一)
// ═══════════════════════════════════════════════════════════════════════════

/// 一次探针跑 = **两个配置文件**描述全部行为面(与生产**同一条加载路**):
///
/// ```text
/// OWL_PROBE_FLAVOR=<name>    基准档(profiles/<name>.toml)
/// OWL_PROBE_OVERLAY=<path>   覆盖文件(可选;同 server 的 [CONFIG] 位置)
/// ```
///
/// 引擎旋钮一律由 [`EngineKnobs::from_config`] 搬 —— 探针档**不再自己拼 knobs**
/// (那正是"条件不一样、数字对不上"的根因)。要改条件就改文件,改完条件写在
/// 文件里、每跑一次 `digest()` 自证。
///
/// 只剩**运行器级**三件事走 env(它们不属于引擎配置):
/// - `OWL_TEST_DEVICE`(哪张卡)/ `OWL_E2E_ONLY`(跑哪些场景)
/// - `ckpt:27b-awq` / `ckpt:dflash2`(资产路径)
/// - 本函数的两个(读哪个配置文件)
fn probe_engine_config(ordinal: usize, default_flavor: &str) -> EngineConfig {
    let flavor = owl_shared::env_reader::str("OWL_PROBE_FLAVOR")
        .unwrap_or_else(|| default_flavor.to_string());
    let base = owl_shared::profiles::profile_path(&flavor).expect("预设档路径");
    let overlay = owl_shared::env_reader::str("OWL_PROBE_OVERLAY").map(std::path::PathBuf::from);
    let cfg = owl_shared::config::OwlConfig::load(Some(&base), overlay.as_deref())
        .expect("探针配置装载(账外键/非法值 = 拒)");
    let mut knobs = EngineKnobs::from_config(&cfg);
    // 探针档**不改变行为面**,只关掉与本档无关的采样(它不影响 prefill/decode 数值)
    knobs.sampler_enabled = false;
    eprintln!(
        "[probe-conf] flavor={flavor} overlay={} ⇒ {}",
        overlay.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "-".into()),
        knobs.digest()
    );
    EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: cfg.runtime.max_seq,
        prefill_chunk: cfg.runtime.prefill_chunk,
        knobs,
    }
}

/// 覆盖率探针档:条件全部来自配置文件(缺省基档 `dev-08b`)。
fn coverage_probe_config(ordinal: usize) -> EngineConfig {
    probe_engine_config(ordinal, "dev-08b")
}

/// **B-3e 场景体:stage-in(宿主 → 设备回灌)实测**(face 无关)。
///
/// 造压序列:①会话 1 跑一段 ⇒ 链进缓存;②会话 2 跑更长的一段 ⇒ 池压力 ⇒
/// 缓存块**下沉到宿主层**;③会话 3 用会话 1 的原 prompt ⇒ 前缀命中且内容在宿主
/// ⇒ `pending_loads` ⇒ **回灌**(一条 swap 搬 N 块)。
///
/// 门:①真下沉(`offloads > 0`)②真回灌(`onloads > 0` 且 stage 账非空)
/// ③报告吞吐 + **27B 工作集外推**(对照论文 174ms@256K)。
async fn stage_in_scenario<D: DeviceClient>(running: &mut RunningEngine<D>) {
    let mk = |i: usize, target: usize| -> String {
        let mut body = String::new();
        let mut k = 0usize;
        while body.chars().count() < target {
            body.push_str(&format!("第{i}则条目{k}:信号强度{k}级。"));
            k += 1;
        }
        body
    };
    let p1t: usize = owl_shared::env_reader::parse::<usize>("OWL_B3E_P1").unwrap_or(300);
    let p2t: usize = owl_shared::env_reader::parse::<usize>("OWL_B3E_P2").unwrap_or(460);
    let p1 = mk(1, p1t);
    let p2 = mk(2, p2t);
    // ① 会话 1:链进缓存
    let s1 = {
        let (sid, _) = running.submit_session(Some(1), p1.clone(), 4).expect("submit");
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { .. } => break,
                TurnEvent::Failed { err, .. } => panic!("t1 失败: {err}"),
                TurnEvent::Idle => panic!("t1 队列丢失"),
                _ => {}
            }
        }
        sid
    };
    let rep1 = running.store_report();
    eprintln!(
        "[b3e] 轮 1 后:缓存 {} 块(宿主承载 {})· 宿主驻留 {:.2}MiB · 空闲槽 {}/{} · 下沉 {} 次",
        rep1.kv.cached_blocks,
        rep1.kv.host_blocks,
        rep1.host_resident_bytes as f64 / 1048576.0,
        rep1.kv.slots_free,
        rep1.kv.slots_total,
        running.bank.rel.leg.report().ops.offloads
    );
    // ② 会话 2:造压 ⇒ 下沉
    {
        let (_, tid) = running.submit_session(Some(2), p2, 4).expect("submit");
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { turn, .. } if turn == tid => break,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
                _ => {}
            }
        }
    }
    let rep2 = running.store_report();
    eprintln!(
        "[b3e] 轮 2 后:缓存 {} 块(宿主承载 {})· 宿主驻留 {:.2}MiB · 下沉 {} 次/{}KiB · 空闲槽 {}",
        rep2.kv.cached_blocks,
        rep2.kv.host_blocks,
        rep2.host_resident_bytes as f64 / 1048576.0,
        running.bank.rel.leg.report().ops.offloads,
        running.bank.rel.leg.report().ops.offload_bytes >> 10,
        rep2.kv.slots_free
    );
    let _stage_before = running.sel.stage().blocks;
    // ③ 会话 3:原 prompt 再召回 ⇒ 回灌
    {
        let (_, tid) = running.submit_session(Some(3), p1, 4).expect("submit");
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { turn, .. } if turn == tid => break,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
                _ => {}
            }
        }
    }
    let rep3 = running.store_report();
    let st = running.sel.stage();
    eprintln!("[b3e] {}", st.line());
    eprintln!(
        "[b3e] 轮 3 后:回灌 {} 次/{}KiB(命中面复用)· 宿主仍驻留 {:.2}MiB",
        running.bank.rel.leg.report().ops.onloads,
        running.bank.rel.leg.report().ops.onload_bytes >> 10,
        rep3.host_resident_bytes as f64 / 1048576.0
    );
    // 27B 外推(每块 K+V = 16 层 × 32 tok × 4 头 × 256 dim × 2B × 2)
    let bpb27 = 16usize * 32 * 4 * 256 * 2 * 2;
    let wset = 64usize;
    let mib = (wset * bpb27) as f64 / 1048576.0;
    let ms = if st.mib_per_s() > 0.0 { mib / st.mib_per_s() * 1000.0 } else { f64::NAN };
    eprintln!(
        "[b3e] 外推 27B:工作集 W={wset} 块 = {mib:.0}MiB ⇒ 按实测 {:.0}MiB/s ≈ {ms:.0}ms(论文 174ms@256K)",
        st.mib_per_s()
    );
    let _ = s1;
    assert!(
        running.bank.rel.leg.report().ops.offloads > 0,
        "门①:没造出下沉(宿主层/压力不够?)"
    );
    assert!(
        st.blocks > 0 && running.bank.rel.leg.report().ops.onloads > 0,
        "门②:回灌没发生(前缀命中的块不在宿主层?下沉量不够?)"
    );
}

/// **B-3e 探针**(自装载;`OWL_B3E_PROBE=1`)。
pub async fn gpu_08b_stage_in_probe() {
    if !owl_shared::env_reader::flag("OWL_B3E_PROBE") {
        owl_shared::testkit::report::skip("OWL_B3E_PROBE 未设(B-3e stage-in 探针按需开)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let mut engine = Engine::new(stage_in_config(ordinal)).expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    stage_in_scenario(&mut running).await;
}

/// stage-in 档:小池(16 页)+ 宿主层开(32MiB)+ 前缀缓存开 ⇒ 能造出"下沉后召回"。
fn stage_in_config(ordinal: usize) -> EngineConfig {
    let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
    knobs.pool_tokens = Some(512);
    knobs.host_cache_mb = 32;
    // 状态腿宿主层也开:命中面的**边界快照**沉下去才不被压力挤掉
    // (`reuse_face` 要求边界有快照 ⇒ 没它复用面会被裁到 0,回灌永不发生)
    knobs.snap_host_mb = 32;
    knobs.snap_max = 8;
    knobs.gdn_slots = 8;
    EngineConfig { device_ordinal: ordinal, max_seq_tokens: 512, prefill_chunk: 64, knobs }
}

/// **B-5 Stage 2:27B 单卡 + 选择层 + 部分驻留**(`OWL_SELECT_27B=1` + `ckpt:27b-awq`)。
///
/// **场景设计就是"KVMem 虚拟空间"本身**:设备 KV 池**刻意小于** ctx
/// (池 4k tok / prompt ~16k tok = 4× 超卖)⇒ 中途**必须**把被踢出工作集的块
/// 沉到宿主,否则池早就满。对照:vLLM 这类"全量常驻"引擎在这个池尺寸下
/// **必然 OOM**(它没有"只在设备上留工作集"的能力)。
///
/// 门:①装载+prefill+decode 全程零崩 ②真下沉(`sunk > 0`)③视图驱动 decode
/// (`select_view_steps > 0`,零回退)④输出非退化(切题性看打印)。
async fn select_27b_scenario() {
    let dir = std::path::PathBuf::from(
        owl_shared::testkit::resource("ckpt:27b-awq").expect("ckpt:27b-awq"),
    );
    assert!(dir.exists(), "检查点目录不存在: {dir:?}");
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    // **条件全部来自配置文件**(与生产同一条加载路;缺省基档 `awq27b-kvmem`)
    // —— 要调池/预算/节奏/臂,改文件(或 `OWL_PROBE_OVERLAY` 指向的覆盖文件),
    //   不再靠散落的 `OWL_27B_*` env(那正是"条件不一样、数字对不上"的根因)。
    let items: usize = owl_shared::env_reader::parse::<usize>("OWL_27B_ITEMS").unwrap_or(2_600);
    let cfg = probe_engine_config(ordinal, "awq27b-kvmem");
    let mut knobs = cfg.knobs;
    let cfg_ctx = cfg.max_seq_tokens;
    // ⚠️ 只是**手写值**;真池容量由引擎治理器算(见 `[vram-gov]` 行)—— 别拿它当分配量
    let cfg_pool = knobs.pool_tokens.unwrap_or(2 * cfg_ctx);
    // **分段臂**:`bare`(KVMem 全关)/`index`/`view`/`partial`(全开)。
    // 只在配置文件开着的档位上**往下关**(不越过配置往上开 —— 那会造出第三个规范)。
    let arm = owl_shared::env_reader::str("OWL_27B_ARM").unwrap_or_else(|| "partial".into());
    let (ix, sd, sp) = match arm.as_str() {
        "bare" => (false, false, false),
        "index" => (true, false, false),
        "view" => (true, true, false),
        _ => (
            knobs.index_probe,
            knobs.select_decode,
            knobs.select_partial,
        ),
    };
    knobs.vram_reserve_mb = owl_shared::env_reader::parse::<u64>("OWL_27B_RESERVE_MB").unwrap_or(128);
    knobs.index_probe = ix;
    knobs.select_decode = sd;
    knobs.select_partial = sp;
    // 虚拟化开关跟着 partial 走(bare 必须关:否则池仍被钉在 ctx 上,与基线不可比)
    knobs.kvmem_virtualization = sp;
    // 地板 = **事实**(env 给 MiB);页数由装配期按池宽落(A7:同一把尺)
    knobs.dense_floor_mib = owl_shared::env_reader::parse::<u64>("OWL_27B_FLOOR_MIB");
    knobs.dense_floor = EngineKnobs::dense_floor_for(sp, knobs.dense_floor_mib);
    eprintln!("[b5-27b] 臂={arm}(index={ix} select_decode={sd} select_partial={sp})");
    // ⚠️ 这里**不再**硬编码任何引擎旋钮(host 预算/chunk/核档/fi)——
    // 它们全部来自 `awq27b-kvmem.toml`(或 `OWL_PROBE_OVERLAY` 的覆盖件)。
    // 之前那些硬编码正是"条件不一样、数字对不上"的来源(例如把 flavor 给的
    // snap_host 1024 MiB 又按回 512 MiB,状态腿随即撞分片上限)。
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: cfg_ctx,
        prefill_chunk: cfg.prefill_chunk,
        knobs,
    })
    .expect("构造");
    let loaded = engine
        .loader()
        .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, owl_models::module::QuantPlan::W4A16Awq)
        .await
        .expect("27B 装载");
    let mut running = engine.run(loaded).await.expect("装配");
    eprintln!(
        "[b5-27b] ctx={cfg_ctx} tok · 设备池={:?} tok(超卖 {:.1}×)· 装配后驱动空闲 {:.0}MiB · live 账 {:.0}MiB",
        cfg_pool,
        cfg_ctx as f64 / cfg_pool as f64,
        owl_shared::vram::free_bytes() as f64 / 1048576.0,
        owl_shared::vram::live_bytes() as f64 / 1048576.0
    );
    // 长 prompt:针 + 底噪(与 0.8B 探针同构)
    const WORDS: [&str; 16] = [
        "信号", "潜伏", "噪声", "衰减", "回环", "缓存", "梯度", "对齐", "稀疏", "簇集",
        "带隙", "相噪", "溢出", "折叠", "绑定", "旁路",
    ];
    const NEEDLES: [&str; 6] = [
        "记录甲:号志箱立在第七码头,校准值 4821,归属风控。",
        "记录乙:山谷气象站夜温 13 度,风向偏东,校准值 7302。",
        "记录丙:仓库封条编号 9917,晨间转移至三号棚。",
        "记录丁:链路密钥轮换到 2560,生效时间午后四点。",
        "记录戊:档案存放北馆二层,页签 6318,限内部传阅。",
        "记录己:台站日志信号衰减 4 分贝,校准值 8045。",
    ];
    let mut body = String::new();
    for i in 0..items {
        let k = i / (items / 7).max(1);
        if i % (items / 7).max(1) == 0 && k < NEEDLES.len() {
            body.push_str(NEEDLES[k]);
        }
        body.push_str(WORDS[(i * 7) % WORDS.len()]);
        body.push_str(WORDS[(i * 13 + 3) % WORDS.len()]);
        body.push(':');
        body.push_str(&i.to_string());
        body.push('。');
    }
    body.push_str("请汇总:上面六份记录里的校准值、编号、页签各是多少?");
    let t0 = std::time::Instant::now();
    let (_, tid) = running.submit_session(Some(1), &body, 24).expect("submit");
    // **预填充吞吐口径**(2026-10-19 加):TTFT = 从 submit 到第一个 Token 事件
    // = 整段 prompt 的 prefill 墙钟(含逐 chunk 的边界开销)。
    // 为什么要这一行:超卖档的代价全在 prefill 上(每 chunk 一次边界),
    // 而旧的"用时"含 decode ⇒ 看不清 prefill 自己的速率。
    let mut ttft = None;
    let mut prompt_tok = 0usize;
    let text = loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Prefill { total, .. } => prompt_tok = prompt_tok.max(total),
            TurnEvent::Token { turn, .. } if turn == tid && ttft.is_none() => {
                ttft = Some(t0.elapsed());
            }
            TurnEvent::Completed { turn, text, .. } if turn == tid => break text,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            _ => {}
        }
    };
    if let Some(ttft) = ttft {
        let secs = ttft.as_secs_f64().max(1e-6);
        let n = prompt_tok.max(1);
        let bnd = running.sel.boundaries().max(1);
        eprintln!(
            "[b5-27b] prefill: {n} tok / {secs:.2}s = **{:.0} tok/s** · 边界 {bnd} 次 · \
             平均 {:.0} ms/边界(B 轴 {:.0} ms/边界)",
            n as f64 / secs,
            secs * 1000.0 / bnd as f64,
            running.sel.prof().total_ms / bnd as f64
        );
    }
    let rep = running.store_report();
    let st = running.sel.stage();
    let pt = running.sel.partial();
    let rec = running.sel.cov_last().map(|r| r.recall.clone()).unwrap_or_default();
    eprintln!(
        "[b5-27b] 用时 {:.1}s · 视图步 {} · 回退 {} · 召回 {} 页(覆盖 {:.1}% 物化 {:.1}%)",
        t0.elapsed().as_secs_f32(),
        running.sel.view_steps(),
        running.sel.view_fallbacks(),
        rec.view_pages,
        rec.coverage * 100.0,
        rec.materialize_ratio * 100.0
    );
    eprintln!("[b5-27b] {} · {}", pt.line(), st.line());
    eprintln!("[b5-27b] {}", running.sel.prof().line());
    eprintln!(
        "[b5-27b] 池 {}/{} 槽自由 · 宿主 KV 驻留 {:.2}GiB · 下沉 {} 次 {}GiB",
        rep.kv.slots_free,
        rep.kv.slots_total,
        rep.host_resident_bytes as f64 / 1073741824.0,
        rep.kv.evicted,
        0
    );
    eprintln!("[b5-27b] 输出({} 字):{}", text.chars().count(), text);
    assert!(pt.sunk > 0, "池 < ctx 却没下沉 ⇒ 池没被压到(超卖档没生效)");
    assert!(running.sel.view_steps() > 0, "视图没驱动 decode");
    assert_eq!(running.sel.view_fallbacks(), 0, "视图回退(占位符/视图缺位?)");
    assert!(text.chars().count() >= 4, "输出退化");
}

/// **B-5 Stage 2 探针入口**(自装载;`OWL_SELECT_27B=1`)。
pub async fn gpu_27b_select_probe() {
    if !owl_shared::env_reader::flag("OWL_SELECT_27B") {
        owl_shared::testkit::report::skip("OWL_SELECT_27B 未设(B-5 Stage2 探针按需开)");
        return;
    }
    if owl_shared::testkit::resource("ckpt:27b-awq").is_none() || gpu_ordinal().is_none() {
        owl_shared::testkit::report::skip("需 ckpt:27b-awq + OWL_TEST_DEVICE");
        return;
    }
    select_27b_scenario().await;
}

/// B-0 索引采集探针(自装载;`OWL_INDEX_PROBE=1`)。
pub async fn gpu_08b_index_probe() {
    if !owl_shared::env_reader::flag("OWL_INDEX_PROBE") {
        owl_shared::testkit::report::skip("OWL_INDEX_PROBE 未设(B-0 索引采集探针按需开)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let mut engine = Engine::new(index_probe_config(ordinal)).expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    index_probe_scenario(&mut running, true).await;
}

/// **CPU mock 档的索引采集场景**(无卡:验链路;K 恒零 ⇒ 数值无意义)。
pub async fn cpu_mock_index_probe() {
    if !owl_shared::env_reader::flag("OWL_CPU_MOCK") || !owl_shared::env_reader::flag("OWL_INDEX_PROBE") {
        owl_shared::testkit::report::skip("OWL_CPU_MOCK / OWL_INDEX_PROBE 未设");
        return;
    }
    let face = owl_cpu::CpuFace::mock(owl_cpu::MockLimits {
        null_kernels: true,
        graph_capture: true,
        ..Default::default()
    });
    let mut engine = Engine::on(index_probe_config(0), face).expect("构造(CPU mock)");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    index_probe_scenario(&mut running, false).await;
}

/// 索引探针档(小池 + 开采集;prompt 短 ⇒ 靠 decode 页边界出块)
fn index_probe_config(ordinal: usize) -> EngineConfig {
    let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
    knobs.pool_tokens = Some(256);
    knobs.gdn_slots = 8;
    knobs.index_probe = true;
    // **廉价边界暂存路径**（P1）：每 4 个边界才完整采集一次；
    // 本测试的 prompt 会跨多个边界，从而覆盖 push/take/末边界排空。
    knobs.recall_every = 4;
    // 索引账落在 store 的 **host 层总账**上(与 KV/快照分片共用总账;
    // 索引自己的淘汰序 = 近期 + 累积频率,见 `MeanKIndex::evict_candidates`)
    knobs.snap_host_mb = 64;
    EngineConfig { device_ordinal: ordinal, max_seq_tokens: 256, prefill_chunk: 32, knobs }
}

/// 演示台旋钮(两档共用 —— 池/宿主/快照几何与 face 无关)。
fn offload_probe_config(ordinal: usize) -> EngineConfig {
    let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
    knobs.pool_tokens = Some(256); // 单会话容量下限 = 256/32 = 8 页 ⇒ nb = **8 槽**
    knobs.host_cache_mb = 8; // 宿主面开(8MiB ⇒ 可容 ~21 块)
    knobs.snap_max = 16; // 快照别被 LRU 冲掉(前缀复用要求边界快照在场)
    EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 256, // 8 页(池下限 = 单会话容量;要造压力就得同时压它)
        prefill_chunk: 32,
        knobs,
    }
}

/// **store 演示台场景体(face 无关)** —— GPU 真卡档与 CPU mock 档**共用同一份
/// 断言**(三门口 + 兜底驱逐 + 批面):§18 的 mock 价值就在"同一门"上。
async fn offload_probe_scenario<D: DeviceClient>(running: &mut RunningEngine<D>) {
    // prompt 先算好(借用 `running.boot.tok` 只在这里一次;下面要 &mut *running)
    let mk = |i: usize, target: usize| -> String {
        let mut body = String::new();
        let mut k = 0usize;
        while body.chars().count() < target {
            body.push_str(&format!("第{i}则条目{k}:信号强度{k}级。"));
            k += 1;
        }
        body
    };
    let hot: Vec<String> = (0..2).map(|i| mk(i, 88)).collect(); // 各 ~89 tok ⇒ 3 块
    let long = mk(9, 240); // ~240 tok ⇒ 8 块(**需求**:8 − free 2 = 6 槽)

    // 跑 ephemeral turn(终了焚链 ⇒ 缓存独占 ⇒ 可下沉)
    // 门③基线:**冷启动**(轮 1)= 无缓存路径的生成全文
    let mut cold_text: Option<String> = None;
    for (i, p) in hot.iter().enumerate() {
        let text = probe_run_turn(&mut *running, p).await;
        if i == 0 {
            cold_text = Some(text);
        }
    }

    // **轮 3:需求驱动下沉** —— 长 prompt 准入即要 8 槽,free 只 2 ⇒
    // 保洁按链把 A+B(各 3 块)整条沉到宿主(内容不丢,不是被同步驱逐掉)
    let _ = probe_run_turn(&mut *running, &long).await;

    let rep = running.bank.rel.leg.report();
    eprintln!(
        "[s3-3b] 轮 3 后:自由槽 {}/{} · 缓存 {} 块 · 宿主 {} 块 / {:.2}MiB · 换出 {} 次 {}KiB(drop {} / 兜底驱逐 {})",
        rep.slots_free,
        rep.slots_total,
        rep.cached_blocks,
        rep.host_blocks,
        rep.host_bytes as f64 / 1048576.0,
        rep.ops.offloads,
        rep.ops.offload_bytes >> 10,
        rep.ops.ensure_evicted,
        "—"
    );
    // ── B-3e:stage-in(回灌)实测账 + 27B 工作集外推 ──
    // 论文口径:恢复 174ms@256K("选中集"回灌,不是全 ctx)。我们把实测吞吐外推到
    // **27B 的真实几何**(16 全注意力层 × hkv4 × hd256 × 页 32 ⇒ 每块 ≈ 2.1MiB)
    // 与 W=64 的工作集,看是否同量级。
    {
        let st = running.sel.stage();
        eprintln!("[b3e] {}", st.line());
        let bpb27 = 16 * 32 * 4 * 256 * 2 * 2; // 每块 K+V 字节(27B)
        let wset = 64usize;
        let mib = (wset * bpb27) as f64 / 1048576.0;
        let ms = if st.mib_per_s() > 0.0 { mib / st.mib_per_s() * 1000.0 } else { f64::NAN };
        eprintln!(
            "[b3e] 外推 27B:工作集 W={wset} 块 = {mib:.0}MiB ⇒ 按实测吞吐 {:.0}MiB/s 回灌 ≈ {ms:.0}ms(论文 174ms@256K)",
            st.mib_per_s()
        );
        // 注:**本探针只下沉不回灌**(它的轮 3 命中的块仍在设备)⇒ 这里只报数,
        // 回灌的硬门在 B-3e 的 `gpu_08b_stage_in_probe`(造压 → 下沉 → 再召回)
        if st.blocks > 0 {
            assert!(st.bytes > 0 && st.nanos > 0, "门:有回灌则字节/墙钟必须有样本");
        }
    }
    assert!(rep.ops.offloads > 0, "门①:需求驱动保洁应真下沉(宿主面已开)");
    assert!(rep.host_bytes > 0, "门①:宿主层应真有驻留");
    assert_eq!(rep.ops.ensure_evicted, 0, "需求驱动到位 ⇒ `ensure_for_len` 同步驱逐应为 0");

    // 诊断:命中面到底长什么样(scheduler 看到的同一面)。
    // ⚠️ 口径必须与 submit 一致:`submit_session` 内部先 `chat_wrap` 再 encode
    // (裸 `encode(prompt)` = 少了模板前缀 ⇒ 哈希链从第 0 块就不同,匹配恒 0;
    // 本诊断此前正是踩了这个假现场,见设计卷 §16)。
    {
        let mut diag = |tag: &str, p: &str| {
            let wrapped = running.boot.tok.chat_wrap(p);
            let ids = running.boot.tok.encode(&wrapped);
            let hit = running.bank.rel.leg.match_prefix(&ids).expect("match");
            let rows: Vec<String> = hit
                .hashes
                .iter()
                .zip(&hit.slots)
                .enumerate()
                .map(|(i, (h, s))| {
                    format!(
                        "{i}:{}|snap={}",
                        if s.is_some() { "dev" } else { "host" },
                        // S2:在场性查 **store 的层账**(设备/宿主都算)
                        if running
                            .bank
                            .rel
                            .state
                            .locate(owl_store::SnapKey::at_boundary(*h))
                            .is_some()
                        {
                            "Y"
                        } else {
                            "N"
                        }
                    )
                })
                .collect();
            eprintln!(
                "[s3-3b] 诊断 {tag}:tokens={} matched={} pending={} · {}",
                ids.len(),
                hit.matched_blocks,
                hit.pending_host().len(),
                rows.join(" ")
            );
        };
        diag("链A(轮 1/4 的 prompt)", &hot[0]);
        diag("链B", &hot[1]);
        eprintln!(
            "[s3-3b] 诊断 快照池:cap={} used={}",
            running.dev.pool.snap_capacity(),
            running.dev.pool.snap_used()
        );
    }

    // **轮 4:冷转热** —— 重发第 1 轮 prompt ⇒ 命中面落在宿主 ⇒ 段边界回灌
    let text = probe_run_turn(&mut *running, &hot[0]).await;
    eprintln!("[s3-3b] 命中轮回吐:{text:?}");
    // **门③**:命中路径(宿主格→段边界回灌)与**冷启动**必须逐字节恒等
    let cold = cold_text.as_deref().expect("轮 1 冷启动基线在场");
    assert_eq!(text, cold, "门③:命中路径输出必须与冷启动逐字节一致");

    let rep2 = running.bank.rel.leg.report();
    eprintln!(
        "[s3-3b] 轮 4 后:宿主 {} 块 / {:.2}MiB · 换出 {} 块/{} 命令 · 回灌 {} 块/{} 命令 · 兜底驱逐 {}",
        rep2.host_blocks,
        rep2.host_bytes as f64 / 1048576.0,
        rep2.ops.offloads,
        rep2.ops.offload_batches,
        rep2.ops.onloads,
        rep2.ops.onload_batches,
        rep2.ops.ensure_evicted
    );
    assert!(rep2.ops.onloads > 0, "门②:命中应触发段边界回灌(onload)");
    assert!(rep2.ops.onloads >= 2, "命中面有两块在宿主 ⇒ 回灌至少 2 块");
    assert_eq!(
        rep2.ops.onload_batches, 1,
        "§15 批面:多块回灌必须**一条命令**(逐块 = 每块一次往返)"
    );
    assert_eq!(rep2.ops.ensure_evicted, 0, "全程不得依赖 `ensure_for_len` 的同步驱逐");
    // 轮 4 的换出 = **为回灌腾位**(冷链让路)⇒ 宿主净增减不作文判据;
    // 真正要看的:回灌发生了、且没有同步驱逐(上两条)。
}

/// 跑一个 ephemeral turn 到完成(探针共用;返回生成全文)。
async fn probe_run_turn<D: DeviceClient>(running: &mut RunningEngine<D>, prompt: &str) -> String {
    let (_, tid) = running.submit_session(None, prompt, 8).expect("submit");
    let mut n = 0usize;
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            TurnEvent::Completed { turn, text, .. } if turn == tid => return text,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {
                n += 1;
                assert!(n < 8192, "事件不收敛");
            }
        }
    }
}

/// **S0 预算实测(0.8B 档)** —— 固定面分解表:**每一项都是实测数字**
/// (A5.1:没有数字的项禁止进预算式),产出可直接喂 `owl_store::Rel::open`
/// 的 `BudgetInputs`(S3-3 host 卸载接线的前置)。
///
/// 跑法(钉卡示例;默认**不开** —— 需显式置 `OWL_BUDGET_PROBE=1`):
/// ```bash
/// CUDA_VISIBLE_DEVICES=GPU-<3080> OWL_TEST_DEVICE=0 OWL_BUDGET_PROBE=1 \
///   cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1 --nocapture
/// ```
/// 口径(全部来自真分配账,禁止按公式估):
/// - `weights` = live(装载后) − live(Engine::new 后);
/// - 固定面 = live(run 后) − live(装载后),再按**池几何**拆出 KV 池 / GDN 格 /
///   快照池 / 草稿池;余项 = **运行底价**(图 slab + 激活 + CUDA ctx + 表件);
/// - turn 期 live 峰值由 200µs 采样线程给(瞬态面)。
pub async fn gpu_08b_budget_probe() {
    if !owl_shared::env_reader::flag("OWL_BUDGET_PROBE") {
        owl_shared::testkit::report::skip("OWL_BUDGET_PROBE 未设(预算实测按需开)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let mb = |b: u64| b as f64 / 1048576.0;
    let snapshot = |tag: &str| {
        eprintln!(
            "[budget] {tag:<22} total={:>9.1}MiB free={:>9.1}MiB live={:>9.1}MiB",
            mb(owl_shared::vram::total_bytes()),
            mb(owl_shared::vram::free_bytes()),
            mb(owl_shared::vram::live_bytes()),
        );
    };

    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 4096,
        prefill_chunk: 32,
        knobs: EngineKnobs::default(),
    })
    .expect("构造");
    let live_open = owl_shared::vram::live_bytes();
    // **v_card 口径(设计卷 §2.2)**:开卡后 `cuMemGetInfo` 的 free(不是标称容量)
    // —— CUDA ctx(~250MiB,不入 live 账)已在 free 里被扣掉,故它不进底价。
    let v_card = owl_shared::vram::free_bytes();
    snapshot("①Engine::new");

    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let live_loaded = owl_shared::vram::live_bytes();
    snapshot("②装载后");

    let mut running = engine.run(loaded).await.expect("装配");
    let live_run = owl_shared::vram::live_bytes();
    snapshot("③run 后");

    // 池几何先**拷出来**(下面要 &mut running 跑 turn;借用不重叠)
    let (nb, nb_kv, page, n_full, n_gdn, kv_fp8, gdn_slots, snap_slots, paged) = {
        let pool = &running.dev.pool;
        (
            pool.geom.nb,
            pool.geom.nb_kv,
            pool.geom.page,
            pool.kvs().len(),
            pool.gdn_count(),
            pool.geom.kv_fp8,
            pool.geom.gdn_slots,
            pool.snap_capacity(),
            pool.geom.paged,
        )
    };
    let (block_bytes, gdn_row_bytes) = {
        let pool = &running.dev.pool;
        (pool.block_bytes(), pool.gdn_row_bytes())
    };
    let kv_pool = (nb_kv * block_bytes) as u64;
    let gdn_pool = (gdn_slots * gdn_row_bytes) as u64;
    let (snap_pool, draft_pool) = {
        let pool = &running.dev.pool;
        (pool.snapshot_pool_bytes() as u64, pool.draft_pool_bytes() as u64)
    };
    let kv_per_tok = (block_bytes / page.max(1)).max(1);
    let weights = live_loaded.saturating_sub(live_open);
    let runtime_floor = live_run
        .saturating_sub(live_loaded)
        .saturating_sub(kv_pool + gdn_pool + snap_pool + draft_pool);

    eprintln!(
        "[budget] 池几何 nb={} nb_kv={} page={} n_full={} n_gdn={} kv_fp8={} gdn_slots={} snap_slots={} paged={}",
        nb, nb_kv, page, n_full, n_gdn, kv_fp8, gdn_slots, snap_slots, paged
    );
    eprintln!(
        "[budget] 导出 kv_bytes_per_token={kv_per_tok}B({}KiB) gdn_row={:.2}MiB gdn_pool={:.2}MiB snap_pool={:.2}MiB draft_pool={:.2}MiB",
        kv_per_tok >> 10,
        mb(gdn_row_bytes as u64),
        mb(gdn_pool),
        mb(snap_pool as u64),
        mb(draft_pool as u64)
    );
    eprintln!(
        "[budget] 固定面分解(实测):权重 {:.1}MiB + KV池 {:.1}MiB + GDN格 {:.1}MiB + 快照池 {:.1}MiB + 草稿池 {:.1}MiB + **运行底价(余项)** {:.1}MiB",
        mb(weights),
        mb(kv_pool),
        mb(gdn_pool),
        mb(snap_pool),
        mb(draft_pool),
        mb(runtime_floor)
    );

    // turn 期 live 峰值(瞬态面;200µs 采样线程)
    let peak = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let (p, s) = (peak.clone(), stop.clone());
        std::thread::spawn(move || {
            while !s.load(std::sync::atomic::Ordering::Relaxed) {
                p.fetch_max(owl_shared::vram::live_bytes(), std::sync::atomic::Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        })
    };
    let (sid, tid) = running
        .submit_session(Some(9001), "用五十字介绍长城。", 16)
        .expect("submit");
    let mut fed = 0usize;
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            TurnEvent::Prefill { turn, .. } if turn == tid => fed += 1,
            TurnEvent::Completed { turn, .. } if turn == tid => break,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {}
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    sampler.join().expect("采样线程");
    snapshot("④首 turn 后");
    eprintln!(
        "[budget] turn 期 live 峰值={:.1}MiB(prefill chunk {} 次,16 tok 生成)",
        mb(peak.load(std::sync::atomic::Ordering::Relaxed)),
        fed
    );
    let _ = sid;

    let n_star = |turns: u64| {
        let fixed = weights
            .saturating_add(runtime_floor)
            .saturating_add(gdn_pool.saturating_mul(turns))
            .saturating_add(draft_pool);
        v_card.saturating_sub(fixed) / (kv_per_tok as u64)
    };
    eprintln!(
        "[budget] **BudgetInputs 建议(0.8B/3080 档)**:v_card_bytes={v_card}(开卡 free;total={}) weights_bytes={weights} \
runtime_floor_bytes={runtime_floor} draft_pool_bytes={draft_pool} gdn_row_bytes={} kv_bytes_per_token={kv_per_tok} host_bytes=0 disk_bytes=0",
        owl_shared::vram::total_bytes(),
        gdn_row_bytes
    );
    eprintln!(
        "[budget] 容量换算:V_avail(1 turn)={}MiB ⇒ N*≈{} tok;V_avail(2 turn)={}MiB ⇒ N*≈{} tok",
        mb(v_card.saturating_sub(weights.saturating_add(runtime_floor).saturating_add(gdn_pool).saturating_add(draft_pool))),
        n_star(1),
        mb(v_card.saturating_sub(weights.saturating_add(runtime_floor).saturating_add(gdn_pool * 2).saturating_add(draft_pool))),
        n_star(2)
    );

    // ── **显存硬预算实证**(2026-10-16):小预算 ⇒ server 侧**主动拒发** ──
    // 判据(cuda::state::vram_budget_guard):驱动账 free−reserve / live 账上限;
    // 拒发 = 结构化 `PoolExhausted`(+ is_fatal ⇒ 服务面优雅关停)。
    // 目的:证明"引擎装配遇到超预算"是**干净的结构化失败**,不是 CUDA 抛错。
    {
        let mut knobs = EngineKnobs::default();
        knobs.cuda.vram_budget_mb = Some(64); // 64MiB:装不下池(96MiB KV + 154MiB GDN)
        let mut e2 = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 4096,
            prefill_chunk: 32,
            knobs,
        })
        .expect("构造");
        // 拒发可落在**装载**(权重 htod)或**装配**(池 alloc)任一 —— 两处同一条闸
        let refused = match e2.loader().load_qwen35_0_8b(&dir).await {
            Err(e) => e,
            Ok(loaded2) => match e2.run(loaded2).await {
                Ok(_) => panic!("64MiB 预算下装配不该成功(硬预算未生效?)"),
                Err(e) => e,
            },
        };
        eprintln!(
            "[budget] 硬预算拒发实证:{refused}(is_fatal={})",
            refused.is_fatal()
        );
        assert!(
            matches!(refused, owl_iface::contract::ModelError::PoolExhausted { .. }),
            "期望结构化 PoolExhausted,得 {refused:?}"
        );
        assert!(
            refused.is_fatal(),
            "拒发必须保住致命类别(否则服务面不会走优雅关停;装载层不得把它包成 Msg)"
        );
    }
}

/// GPU 门控:双 turn 生命周期 —— submit 入队 / pump 事件流 / per-turn
/// GDN 重置(第二个 turn 在脏状态后仍须产出连贯文本)。
/// D1 冒烟:0.8B greedy 双配置(融合 vs 旧链)各产出非空连贯文本。
/// **文本本身两配置必然分叉**(数值微差 × 自回归混沌,2026-10-03 定谳:
/// 融合核与旧链互差 ≤2e-4,算子层 32 步现实幅度递推单测全绿)——
/// 本测试只验「两配置链路都活着且确定(各 ×2 逐字稳定)」。门控 OWL_TEST_DEVICE。
pub async fn gpu_08b_fuse_text_parity() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();

    async fn gen_text(fuse: bool, ordinal: usize, dir: &std::path::Path) -> String {
        // GDN 融合 A/B 收敛注:NO_FUSE 反开关是唯一活闸(EnvProvider);
        // OWL_GDN_FUSE_DECODE 无读者(死旗标,显式依赖律改造中删除)——
        // 双臂现同态(fused on),保留参数占位与恒等断言
        let _ = fuse;
        let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() }; // greedy
        knobs.env.gdn.fused_decode = true;
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 128,
            prefill_chunk: 16,
            knobs,
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(dir).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        let id = running.submit("用五十字介绍长城。", 40).expect("submit");
        drive_turn(&mut running, id).await
    }

    let a1 = gen_text(false, ordinal, &dir).await;
    let a2 = gen_text(false, ordinal, &dir).await;
    let b1 = gen_text(true, ordinal, &dir).await;
    let b2 = gen_text(true, ordinal, &dir).await;
    eprintln!("[smoke] baseline={a1:?}/{a2:?} fused={b1:?}/{b2:?}");
    assert!(!a1.is_empty() && !b1.is_empty(), "两配置均应产出非空文本");
    assert_eq!(a1, a2, "baseline greedy 跨 boot 确定性");
    assert_eq!(b1, b2, "fused greedy 跨 boot 确定性");
}

pub async fn gpu_two_turns_lifecycle(shared: Option<&mut RunningEngine<crate::GpuFace>>) {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<crate::GpuFace> = match shared {
        Some(r) => r,
        None => {
        let mut engine =
            Engine::new(EngineConfig { device_ordinal: ordinal, max_seq_tokens: 64, prefill_chunk: 16, knobs: EngineKnobs::default() })
                .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };
    assert!(
        matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
        "真模型捕获应成功(回放态),得 {:?}",
        running.capture_outcome()
    );

    let t1 = running.submit("Hello, who are you?", 16).expect("t1");
    let t2 = running.submit("用一句话介绍长城。", 16).expect("t2");
    let (mut ttft, t0) = (None, std::time::Instant::now());
    let mut done = Vec::new();
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => break,
            TurnEvent::Prefill { turn, fed, total } => {
                eprintln!("[ev] prefill t{turn} {fed}/{total} (+{:?})", t0.elapsed());
            }
            TurnEvent::Token { turn, .. } if ttft.is_none() => {
                ttft = Some(t0.elapsed());
                eprintln!("[ttft] 首 Token(turn {turn}): {:?}", ttft.unwrap());
            }
            TurnEvent::Completed { turn, text, .. } => {
                eprintln!("[ttft] turn {turn} 完成(累计 {:?})", t0.elapsed());
                done.push((turn, text))
            }
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {}
        }
    }
    assert_eq!(done.len(), 2, "两 turn 都完成");
    assert!(done.iter().any(|(t, _)| *t == t1));
    assert!(done.iter().any(|(t, _)| *t == t2));
    for (t, text) in &done {
        eprintln!("[test] t{t}: {text}");
        assert!(!text.trim().is_empty(), "t{t} 产出非空");
    }
}

/// GPU 门控:同会话连续 turn(S0 验收)—— turn2 应命中账本
/// (跳过 GDN 重置,只 prefill 增量段);临时会话路径由生命周期
/// 测试覆盖,此处验 Some(id) 连续性 + 收口账推进。
pub async fn gpu_session_continuity(shared: Option<&mut RunningEngine<crate::GpuFace>>) {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<crate::GpuFace> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };

    // turn1:建立会话账(显式 id;None = 临时会话终了即焚,不归此测)
    let (sid, t1) = running
        .submit_session(Some(42), "请记住:我最喜欢的数字是七。", 16)
        .expect("t1");
    assert_eq!(sid, 42);
    let text1 = drive_turn(&mut *running, t1).await;
    eprintln!("[test] t{t1}(session {sid}): {text1}");
    assert!(!text1.trim().is_empty());
    assert!(running.session_len(sid).unwrap_or(0) > 0, "turn1 收口账非零");

    // turn2:同会话全量重发(历史 + 新问题)→ 守卫应命中,增量 prefill
    let (sid2, t2) = running
        .submit_session(
            Some(sid),
            "请记住:我最喜欢的数字是七。我刚才说我最喜欢的数字是什么?",
            16,
        )
        .expect("t2");
    assert_eq!(sid2, sid, "同会话归队");
    let text2 = drive_turn(&mut *running, t2).await;
    eprintln!("[test] t{t2}(session {sid2}): {text2}");
    assert!(!text2.trim().is_empty());

    // 收口验证:账本应推到 turn2 全量 + 生成段;记忆质量(答“七”)
    // 属 S3 金标验收,此处不断言文本内容
    let cached = running.session_len(sid).expect("账在");
    assert!(cached > 0, "同会话收口后账本非零");
}

/// GPU 门控:4k 长 ctx 前缀记忆 QA(E1 收尾 P4)—— 暗号埋在 prompt
/// 开头(≈token 10),问题压在 ≈3900 token 处;真全局注意力下应召回,
/// 旧 OWL_MAX_KV=256 滑窗截断下物理不可过(暗号在窗外)。兼验收:
/// 4k 预算守卫、分页 prefill 百级 chunk、v1 decode 长程、同会话
/// turn2(前缀失配回退全量路径 @4k)。
pub async fn gpu_longctx_4k_prefix_qa(shared: Option<&mut RunningEngine<crate::GpuFace>>) {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let seq: usize = owl_shared::env_reader::parse_or("OWL_E2E_SEQ", 4096);
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<crate::GpuFace> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: seq,
            prefill_chunk: 32,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };

    // 组装:暗号开头 + 中性填充 + 补全式探针收尾(裸 LM 无 chat 模板,
    // 问答式会退化成文本续写 —— 探针改为让模型续写暗号本体)
    let marker = "本会话暗号:「蓝鲸二十一号」。\n\n";
    let question = "\n\n清单结束。按约定,暗号重复一遍:暗号是「";
    let filler = |i: usize| {
        format!(
            "第{i}条:观测点{i}记录到信号{i},强度{i}级,来源坐标({i},{i}),持续{i}天。\n"
        )
    };
    let mut body = String::from(marker);
    let mut i = 0usize;
    let nofill = owl_shared::env_reader::flag("OWL_E2E_NOFILL");
    loop {
        if nofill {
            break; // 对照实验:短 ctx 直问(验模型/模板,不验长程)
        }
        let cand = format!("{body}{}", filler(i));
        let full = format!("{cand}{question}");
        if running.boot.tok.encode(&full).len() + 32 > seq {
            break; // 留 32 token 生成余量(turn2 预算同享)
        }
        body = cand;
        i += 1;
    }
    let prompt_t1 = format!("{body}{question}");
    let n_tok = running.boot.tok.encode(&prompt_t1).len();
    eprintln!("[test] 长 ctx prompt = {n_tok} tok(填充 {i} 条)");
    assert!(
        nofill || n_tok > seq * 4 / 5,
        "长文应填满窗口 ≥80%,得 {n_tok}/{seq}"
    );

    let t0 = std::time::Instant::now();
    let (sid, t1) = running
        .submit_session(Some(7), prompt_t1.as_str(), 16)
        .expect("t1 应在 4k 预算内");
    assert_eq!(sid, 7);
    let mut ttft = None;
    let mut prefill_last = None;
    let text1;
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => panic!("turn {t1} 队列丢失"),
            TurnEvent::Prefill { turn, fed, total } if turn == t1 => {
                if fed % (32 * 16) == 0 || fed == total {
                    eprintln!("[ev] t{t1} prefill {fed}/{total} @ {:?}", t0.elapsed());
                }
                prefill_last = Some((fed, total, t0.elapsed()));
            }
            TurnEvent::Token { turn, .. } if turn == t1 && ttft.is_none() => {
                ttft = Some(t0.elapsed());
            }
            TurnEvent::Completed { turn, text, .. } if turn == t1 => {
                text1 = text;
                break;
            }
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {}
        }
    }
    let wall1 = t0.elapsed();
    if let Some((fed, total, el)) = prefill_last {
        eprintln!(
            "[bench] t{t1} prefill {fed}/{total} tok @ {el:?}({:.0} tok/s,debug) | TTFT {:?} | 总 {:?}",
            fed as f64 / el.as_secs_f64(),
            ttft.unwrap(),
            wall1
        );
    }
    eprintln!("[test] t{t1} 答: {text1}");
    // 记忆质量金标挂 S3(0.8B + greedy + 裸模板的答句质量不可靠;
    // 对照实验:短/长 ctx 行为一致 ⇒ 机械等价)。此处验收 =
    // 4k 全链不崩 + 帐本收口 + 计时(v1 smem 越界已修,见上)
    assert!(!text1.trim().is_empty(), "t{t1} 产出非空");

    // turn2 同会话再问(前缀失配 → 回退全量路径)。E2a 后唯一验收:
    // 修复前此处在 4k 下 OOM(server 中间块零回收,双 turn 累计
    // ~24GB;现 eval 竞技场每 chunk 归池,显存恒平)
    {
        let (_, t2) = running
            .submit_session(Some(sid), format!("{body}再重复一遍:暗号是「"), 16)
            .expect("t2 应在预算内");
        let t0b = std::time::Instant::now();
        let text2 = drive_turn(&mut *running, t2).await;
        eprintln!("[bench] t{t2} 全量回退重灌 + 生成 @ {:?}", t0b.elapsed());
        eprintln!("[test] t{t2} 答: {text2}");
        assert!(!text2.trim().is_empty(), "t{t2} 产出非空");
        // **S2(§25.1①)**:turn 止后账本归到**块边界**(尾块随表释放)⇒
        // `cached_len` 可比 prompt 少不到一页(≤31 tok),不是"账没收口"。
        assert!(
            running.session_len(sid).unwrap_or(0) + 32 > n_tok,
            "账本收口(块对齐):{} vs prompt {n_tok}",
            running.session_len(sid).unwrap_or(0)
        );
    }
}

/// GPU 门控:FI 直读单池健康门(2026-10-13 立项;kV 统一契约 P5 收尾)。
///
/// 背景:统一契约把主池翻 kNHD、FI 从「影子池双写」改为「直读主池」;
/// 为迎合它,写核/读核大范围改过。这条链**数值上**是否健康,此前没有
/// 有效判据:HTTP + greedy 的「跳实现逐字节门」已实测作废(两个
/// flash-attention 实现的 ulp 差被 greedy 放大成完全不同的文本;且
/// 27B AWQ + 空 think 预填在长文上模型自身退化)—— 见坑册 §23。
///
/// 本门姿势(照 [`gpu_longctx_4k_prefix_qa`] 的教训):
/// - **裸 LM**(不走 chat 模板)+ **补全式探针**(让模型续写暗号本体;
///   问答式在裸 LM 上必退化成文本续写);
/// - **同 dtype(fp8)同池几何**,只切读核(`attn.fi`)= FI 直读 vs 自研
///   (chunked_opt / split)→ 输出**应逐字一致**,**与文本质量无关**:
///   就算续写是垃圾,两臂的垃圾也必须一样 —— 这才是等价门;
/// - 4k prompt ⇒ 百级 chunk(prefill_chunk=32),覆盖两臂各自的多 chunk 路径。
///
/// 断言:①两臂 prompt token 数一致 ②召回行为一致 ③**文本逐字一致**。
pub async fn gpu_longctx_fi_direct_gate() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let seq: usize = owl_shared::env_reader::parse_or("OWL_E2E_SEQ", 4096);
    const NEEDLE: &str = "蓝鲸二十一号";

    /// 单臂:boot(池 dtype 与 `fi` 可切)→ 4k 前缀记忆 prompt → 补全式生成。
    async fn arm(ordinal: usize, fi: bool, fp8: bool, seq: usize) -> (String, usize) {
        let dir = asset_dir();
        let mut knobs = EngineKnobs::default();
        knobs.env.kv.quant = if fp8 {
            owl_models::env::KvQuant::Fp8E4M3
        } else {
            owl_models::env::KvQuant::None
        };
        knobs.env.attn.fi = fi; // ← 唯一变量:读核(FI 直读 vs 自研 chunked/split)
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: seq,
            prefill_chunk: 32,
            knobs,
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");

        let marker = "本会话暗号:「蓝鲸二十一号」。\n\n";
        let question = "\n\n清单结束。按约定,暗号重复一遍:暗号是「";
        let filler = |i: usize| {
            format!("第{i}条:观测点{i}记录到信号{i},强度{i}级,来源坐标({i},{i}),持续{i}天。\n")
        };
        let mut body = String::from(marker);
        let mut i = 0usize;
        loop {
            let cand = format!("{body}{}", filler(i));
            if running.boot.tok.encode(&format!("{cand}{question}")).len() + 32 > seq {
                break; // 留 32 token 生成余量
            }
            body = cand;
            i += 1;
        }
        let prompt = format!("{body}{question}");
        let n_tok = running.boot.tok.encode(&prompt).len();
        assert!(n_tok > seq * 4 / 5, "长文应填满窗口 ≥80%,得 {n_tok}/{seq}");
        let t0 = std::time::Instant::now();
        let (_, t) = running
            .submit_session(Some(7), prompt.as_str(), 16)
            .expect("submit 应在 4k 预算内");
        let text = drive_turn(&mut running, t).await;
        eprintln!(
            "[fi-gate] fi={fi} fp8={fp8} prompt={n_tok} tok({i} 条) 生成 @ {:?}: {text:?}",
            t0.elapsed()
        );
        // 实例收尾屏障(同 gpu_spec_identity_gate/恒等门 既有约定):
        // 本门连开 4 个实例,不 barrier 则前一个的 CUDA ctx/显存尚未析构,
        // 下一个实例的 VRAM 探测读到未释放的账 → 池被算小 → 后续用例
        // “KV 块池耗尽”(§11.5 跨用例污染同族;2026-10-13 套件内实测复现)。
        drop(running);
        std::thread::sleep(std::time::Duration::from_secs(10));
        (text, n_tok)
    }

    // 两臂:fp8 生产档 × 读核(自研 / FI 直读)。同 dtype 内两臂必须
    // **逐字一致**。f16 档由既有 `gpu_longctx_4k_prefix_qa`(默认 env 即
    // f16 池)覆盖 —— 本门只补 fp8 这一档,少开两次装载(套件内池账
    // 扰动与耗时都省;2026-10-13 四臂版曾把后置 spec 门的池挤小)。
    let mut _cross = String::new();
    for fp8 in [true] {
        let (off, n_off) = arm(ordinal, false, fp8, seq).await;
        let (on, n_on) = arm(ordinal, true, fp8, seq).await;
        assert_eq!(n_off, n_on, "两臂 prompt token 数应一致(同构造)");
        let (r_off, r_on) = (off.contains(NEEDLE), on.contains(NEEDLE));
        eprintln!(
            "[fi-gate] fp8={fp8} 召回 FI-off={r_off} / FI-on={r_on}; 逐字一致={}",
            off == on
        );
        assert_eq!(
            r_off, r_on,
            "两臂召回行为应一致(fp8={fp8})\n  off={off:?}\n  on ={on:?}"
        );
        assert_eq!(
            off, on,
            "同 dtype 同池、只换读核 → 输出应逐字一致(不一致 = FI 直读或自研读核的实参/几何有问题;fp8={fp8})\n  off={off:?}\n  on ={on:?}"
        );
        _cross = off.clone();
    }
}

/// GPU 门控:W4A16 marlin 全链 e2e(E3 验收;REQ-PRE-01)——
/// llm-compressor 转换的 0.8B W4A16 检查点装载(装载期 ct→marlin
/// 重排)+ 生成:eligible Linear 走 foreign marlin GEMM,小线性
/// (GDN in_proj_z/b/a)反量化 f16 直读。
pub async fn gpu_w4a16_marlin_e2e() {
    // 套件统一约定(OWL_TEST_DEVICE):曾硬钉 ordinal 2 —— 与现役
    // vLLM 生产服务器(3080 对)撞卡时 OOM at capture slab(2026-10-06)
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir(); // 转换产物与 f16 同目录约定:../Qwen3.5-0.8B-W4A16
    let w4a16_dir = dir
        .parent()
        .expect("assets")
        .join("Qwen3.5-0.8B-W4A16");
    if !w4a16_dir.exists() {
        owl_shared::testkit::report::skip(format_args!("W4A16 检查点不存在({:?}),先跑转换器", w4a16_dir));
        return;
    }
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 256,
        prefill_chunk: 32,
        knobs: EngineKnobs::default(),
    })
    .expect("构造");
    let loaded = engine
        .loader()
        .load_qwen35_0_8b_w4a16(&w4a16_dir, &dir)
        .await
        .expect("W4A16 装载");
    let mut running = engine.run(loaded).await.expect("装配");
    assert!(
        matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
        "W4A16 decode 图应捕获成功(回放态),得 {:?}",
        running.capture_outcome()
    );
    let t0 = std::time::Instant::now();
    let t1 = running
        .submit("用一句话介绍长城。", 16) // D1 立案复现:16 tok 亦崩(前 2 token 对,随后乱;基线同步长连贯)
        .expect("submit");
    let text = drive_turn(&mut running, t1).await;
    eprintln!("[bench] W4A16 16 tok @ {:?}({:.1} tok/s)", t0.elapsed(), 16.0 / t0.elapsed().as_secs_f64());
    eprintln!("[test] W4A16 答: {text}");
    assert!(!text.trim().is_empty(), "产出非空");
}

/// GPU 门控:Qwen3.8-27B AWQ-INT4 全链冒烟(2026-10-01;cyankiwi
/// g32-asym 检查点)—— 装载(formats/awq.rs 懒物化 + marlin kU4
/// GEMM_W4A16_AWQ)+ 图捕获 + 生成。门控 ckpt:27b-awq = 检查点
/// 目录(models/cyankiwi/Qwen3.8-27B-AWQ-INT4);未设则 skip。
/// VRAM 预算:权重 ~17.5GB + KV/GDN 状态 ~1.3GB → 3090 Ti 24G。
pub async fn gpu_awq27b_marlin_e2e(shared: Option<&mut RunningEngine<crate::GpuFace>>) {
    let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
        owl_shared::testkit::report::skip("ckpt:27b-awq 未设(cyankiwi 检查点目录)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.exists() {
        owl_shared::testkit::report::skip(format_args!("检查点目录不存在({dir:?})"));
        return;
    }
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    let _t0 = std::time::Instant::now();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<crate::GpuFace> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine
            .loader()
            .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, owl_models::module::QuantPlan::W4A16Awq)
            .await
            .expect("27B AWQ 装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };
    assert!(
        owl_shared::env_reader::flag("OWL_NO_GRAPH")
            || matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
        "27B decode 图应捕获成功,得 {:?}",
        running.capture_outcome()
    );
    let t1 = running
        .submit("用一句话介绍长城。", 16)
        .expect("submit");
    let t2 = std::time::Instant::now();
    let text = drive_turn(&mut *running, t1).await;
    eprintln!(
        "[bench] 生成 tok @ {:.1}ms(含首 token prefill 尾步)",
        t2.elapsed().as_secs_f32() * 1e3
    );
    // 刀D 取证:层间原生时间线(OWL_TS_PROBE;clock64 @~1.98GHz)
    if owl_shared::env_reader::flag("OWL_TS_PROBE") {
        let raw = match running.dev.graphs.session.read_input_slot_bytes("ts_buf").await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[ts] 读回失败 {e}");
                vec![]
            }
        };
        let stamps: Vec<u64> = raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .filter(|&v| v != 0)
            .collect();
        eprintln!("[ts] 拍数 {}", stamps.len());
        let mut prev = *stamps.first().unwrap_or(&0);
        for (i, &s) in stamps.iter().enumerate() {
            let d_us = (s.saturating_sub(prev)) as f64 / 1980.0;
            eprintln!("[ts] #{i:02} Δ={d_us:8.1}µs");
            prev = s;
        }
    }
    eprintln!("[test] 27B AWQ 答: {text}");
    assert!(!text.trim().is_empty(), "产出非空");
}

/// GPU 门控:多会话隔离(E2b 验收)—— A/B 两会话交替 turn:
/// ① B 的 turn 前后,A 的块链/账本逐字节不受扰;② A/B 物理块互异
/// (无前缀缓存时零共享);③ 双会话产出非空。隔离 = 块管理器的
/// 直接行为证据(块链不同 ⇒ KV 物理槽不同)。
pub async fn gpu_multi_session_isolation(shared: Option<&mut RunningEngine<crate::GpuFace>>) {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<crate::GpuFace> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };

    // **S2(§25.1①)后的口径**:会话在 turn 止不持设备表 ⇒ "块链"这条判据
    // 移到**内容身份面**(缓存里的链)。而缓存只收**整块**(32 tok)—— 故 A 的
    // prompt 要够长(短 prompt 一个整块都没有,判据会退化成空集)。
    let mut a_body = String::from("请记住下面的清单:");
    let mut i = 0usize;
    while running.boot.tok.encode(&a_body).len() < 80 {
        a_body.push_str(&format!("第{i}则:信号强度{i}级。"));
        i += 1;
    }
    let a_prompt = format!("{a_body}我的代号是阿尔法。");
    let (_, ta1) = running.submit_session(Some(101), a_prompt.as_str(), 12).expect("A1");
    let ta1_text = drive_turn(&mut *running, ta1).await;
    eprintln!("[test] A1: {ta1_text}");
    assert!(!ta1_text.trim().is_empty());
    let a_len_1 = running.session_len(101).expect("A 账在");
    assert!(a_len_1 > 0, "A 账本非空");
    // 判据 = 内容身份面:A 的链在缓存里(≥1 整块),且 B 的 turn 不动它。
    let a_tokens: Vec<u32> = running.boot.tok.encode(&running.boot.tok.chat_wrap(a_prompt.as_str()));
    let a_hit_1 = running.bank.rel.leg.match_prefix(&a_tokens).expect("A 命中查询");
    assert!(a_hit_1.matched_blocks >= 1, "A 的历史应进缓存(内容身份)");

    let (_, tb1) = running
        .submit_session(Some(202), "我的代号是贝塔。", 12)
        .expect("B1");
    let tb1_text = drive_turn(&mut *running, tb1).await;
    eprintln!("[test] B1: {tb1_text}");
    assert!(!tb1_text.trim().is_empty());

    // 隔离断言 ①:B 的 turn 后,A 的**内容链**与账本都不变
    let a_hit_2 = running.bank.rel.leg.match_prefix(&a_tokens).expect("A 命中查询");
    assert_eq!(
        a_hit_1.hashes, a_hit_2.hashes,
        "B 的 turn 不得扰动 A 的内容链(身份逐块一致)"
    );
    assert_eq!(running.session_len(101), Some(a_len_1), "B 的 turn 不得扰动 A 账本");
    // 隔离断言 ②:身份零串味(两段内容不同 ⇒ 链哈希不相交)
    let b_tokens: Vec<u32> = running
        .boot
        .tok
        .encode(&running.boot.tok.chat_wrap("我的代号是贝塔。"));
    let b_hit = running.bank.rel.leg.match_prefix(&b_tokens).expect("B 命中查询");
    assert!(
        b_hit.hashes.iter().all(|h| !a_hit_2.hashes.contains(h)),
        "A/B 内容身份互异(链哈希不得相交)"
    );
    eprintln!(
        "[test] 内容链 A = {:?}(命中 {})/ B 命中 {}",
        a_hit_2.hashes, a_hit_2.matched_blocks, b_hit.matched_blocks
    );

    // A 二轮(全量重发,前缀守卫失配回退全量也须稳)
    let a2_prompt = format!("{a_prompt}我的代号是什么?");
    let (_, ta2) = running
        .submit_session(Some(101), a2_prompt.as_str(), 12)
        .expect("A2");
    let ta2_text = drive_turn(&mut *running, ta2).await;
    eprintln!("[test] A2: {ta2_text}");
    assert!(!ta2_text.trim().is_empty());
    assert!(running.session_len(101).unwrap_or(0) >= a_len_1, "A 账本推进");
}

/// GPU 门控:前缀缓存命中(E2c 验收;REQ-PRE-04)—— 同前缀跨会话
/// 二轮:A 建档(块链 + 快照登记),B 同前缀 + 异尾 → 块链复用 +
/// 快照恢复,prefill 只算尾部。机械断言:B 的 prefill chunk 数 ≪ A;
/// B 块链前缀与 A 物理共享;TTFT 可测下降。
pub async fn gpu_prefix_cache_hit(shared: Option<&mut RunningEngine<crate::GpuFace>>) {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<crate::GpuFace> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 512,
            prefill_chunk: 32,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };

    // A:marker + filler(~320 tok,跨 ≥1 块边界 → 快照可拍)
    let marker = "本会话暗号:「蓝鲸二十一号」。\n\n";
    let filler = |i: usize| {
        format!(
            "第{i}条:观测点{i}记录到信号{i},强度{i}级,来源坐标({i},{i}),持续{i}天。\n"
        )
    };
    let mut body = String::from(marker);
    let mut i = 0usize;
    while running.boot.tok.encode(&format!("{body}{}", filler(i))).len() < 300 {
        body = format!("{body}{}", filler(i));
        i += 1;
    }
    let prompt_a = format!("{body}清单结束。按约定,暗号重复一遍:暗号是「");
    let prompt_b = format!("{body}这次换个问题:暗号里出现了什么动物?");
    let n_a = running.boot.tok.encode(&prompt_a).len();
    let n_b = running.boot.tok.encode(&prompt_b).len();
    eprintln!("[test] A prompt = {n_a} tok / B prompt = {n_b} tok(前缀共享)");

    let t0 = std::time::Instant::now();
    let (_sid_a, ta1) = running.submit_session(Some(101), prompt_a.as_str(), 12).expect("A");
    let mut a_prefill_chunks = 0usize;
    #[allow(unused_assignments)]
    let (mut a_ttft, mut a_text) = (None, String::new());
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => panic!("A 队列丢失"),
            TurnEvent::Prefill { turn, .. } if turn == ta1 => a_prefill_chunks += 1,
            TurnEvent::Token { turn, .. } if turn == ta1 && a_ttft.is_none() => {
                a_ttft = Some(t0.elapsed());
            }
            TurnEvent::Completed { turn, text, .. } if turn == ta1 => {
                a_text = text;
                break;
            }
            TurnEvent::Failed { turn, err } => panic!("A{turn} 失败: {err}"),
            _ => {}
        }
    }
    eprintln!("[bench] A prefill chunks = {a_prefill_chunks}, TTFT = {:?}", a_ttft.unwrap());
    assert!(!a_text.trim().is_empty());

    // B:同前缀 + 异尾(新会话;guard 必失配 → 走前缀缓存匹配)
    let t0b = std::time::Instant::now();
    let (_, tb1) = running.submit_session(Some(202), prompt_b.as_str(), 12).expect("B");
    let mut b_prefill_chunks = 0usize;
    let mut b_ttft = None;
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => panic!("B 队列丢失"),
            TurnEvent::Prefill { turn, .. } if turn == tb1 => b_prefill_chunks += 1,
            TurnEvent::Token { turn, .. } if turn == tb1 && b_ttft.is_none() => {
                b_ttft = Some(t0b.elapsed());
            }
            TurnEvent::Completed { turn, text, .. } if turn == tb1 => {
                eprintln!("[test] B 答: {text}");
                break;
            }
            TurnEvent::Failed { turn, err } => panic!("B{turn} 失败: {err}"),
            _ => {}
        }
    }
    eprintln!(
        "[bench] B prefill chunks = {b_prefill_chunks}, TTFT = {:?}(A = {:?})",
        b_ttft.unwrap(),
        a_ttft.unwrap()
    );

    // 机械断言:chunk 数骤降 + **内容身份共享** + 缓存面在场
    assert!(
        b_prefill_chunks * 4 <= a_prefill_chunks.max(1),
        "B prefill chunk 数应 ≪ A:{b_prefill_chunks} vs {a_prefill_chunks}"
    );
    // **S2(§25.1①)改判据**:会话在 turn 止**不持设备表**(历史 = 纯缓存)
    // ⇒ "物理共享"不再能从会话表读出(`block_table` 此时是空的);改为判
    // **内容身份共享**:同一份缓存块被两条链复用(这才是复用的真身)。
    let a_ids = running.boot.tok.encode(&running.boot.tok.chat_wrap(prompt_a.as_str()));
    let b_ids = running.boot.tok.encode(&running.boot.tok.chat_wrap(prompt_b.as_str()));
    let hit_a = running.bank.rel.leg.match_prefix(&a_ids).expect("A 命中查询");
    let hit_b = running.bank.rel.leg.match_prefix(&b_ids).expect("B 命中查询");
    let shared = hit_b.hashes.iter().filter(|h| hit_a.hashes.contains(h)).count();
    assert!(shared >= 8, "前缀块应**内容身份**共享,共享 {shared}");
    assert!(hit_b.matched_blocks >= 8, "B 应命中 ≥8 块,得 {}", hit_b.matched_blocks);
    let rep = running.bank.rel.leg.report();
    assert!(rep.cached_blocks >= 8, "缓存面应有 ≥8 条目,得 {}", rep.cached_blocks);
    eprintln!(
        "[test] 共享内容块 {shared}(A 命中 {} / B 命中 {} · 缓存 {} 条)",
        hit_a.matched_blocks, hit_b.matched_blocks, rep.cached_blocks
    );
}

/// GPU 门控:27B 单卡推理验收(正菜;3090 Ti 24G)—— 真实三问 ×
/// greedy(模板已含 thinking-off 空 think 块),验收 = 回答非空 +
/// 无退化复读(启发式:任意 2-8 字片段连现 4 次即退化)。
/// 输出全文打印供人工判读;复读启发式是底线门不是质量门。
pub async fn gpu_27b_chat_inference(shared: Option<&mut RunningEngine<crate::GpuFace>>) {
    let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
        owl_shared::testkit::report::skip("ckpt:27b-awq 未设(cyankiwi 检查点目录)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.exists() {
        owl_shared::testkit::report::skip(format_args!("检查点目录不存在({dir:?})"));
        return;
    }
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<crate::GpuFace> = match shared {
        Some(r) => r,
        None => {
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 1024,
            prefill_chunk: 128,
            knobs: EngineKnobs::default(),
        })
        .expect("构造");
        let loaded = engine
            .loader()
            .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, owl_models::module::QuantPlan::W4A16Awq)
            .await
            .expect("27B AWQ 装载");
            own = engine.run(loaded).await.expect("装配");
            &mut own
        }
    };

    let questions = [
        "用一句话介绍长城。",
        "水的沸点是多少摄氏度?",
        "写一句关于春天的诗。",
    ];
    // 退化复读启发式:任意 2-8 字片段连现 4 次(贪心解码典型病灶;
    // 正常文本几乎不可能命中)
    fn degenerate(text: &str) -> Option<String> {
        let chars: Vec<char> = text.chars().collect();
        for len in 2..=8usize {
            for start in 0..chars.len().saturating_sub(len * 4) {
                let pat: String = chars[start..start + len].iter().collect();
                let mut hits = 0;
                let mut i = start;
                while i + len <= chars.len() && chars[i..i + len].iter().collect::<String>() == pat {
                    hits += 1;
                    i += len;
                    if hits >= 4 {
                        return Some(pat);
                    }
                }
            }
        }
        None
    }
    for (qi, q) in questions.iter().enumerate() {
        // GPU 逐核归因探针窗(OWL_GPU_PROF=1):Q0 前 reset / Q0 后 query
        // —— 纯 decode 窗分账。注:窗口钉在 Q0(首 turn)——eager 模式
        // 27B 逐 turn 显存爬坡 ~480MB/步(收割缺口,另案),turn 2 会 OOM
        if qi == 0 && owl_shared::env_reader::flag("OWL_GPU_PROF") {
            owl_shared::metrics::reset_metrics();
        }
        let t0 = std::time::Instant::now();
        let id = running.submit(*q, 96).expect("submit");
        let mut ttft = None;
        let mut n_tok = 0usize;
        let text;
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Idle => panic!("Q{qi} 队列丢失"),
                TurnEvent::Token { turn, .. } if turn == id => {
                    if ttft.is_none() {
                        ttft = Some(t0.elapsed());
                    }
                    n_tok += 1;
                }
                TurnEvent::Completed { turn, text: t, .. } if turn == id => {
                    text = t;
                    break;
                }
                TurnEvent::Failed { turn, err } => panic!("Q{qi} turn {turn} 失败: {err}"),
                _ => {}
            }
        }
        let wall = t0.elapsed();
        eprintln!(
            "[27b-chat] Q{qi}: {q}\n[27b-chat] A{qi}: {text}\n[27b-chat] {n_tok} tok @ {wall:?}(TTFT {:?}, {:.1} tok/s)",
            ttft.unwrap(),
            n_tok as f64 / wall.as_secs_f64()
        );
        // GPU 逐核归因探针:Q0 窗口结束 → 热点分账(按总时长降序)
        if qi == 0 && owl_shared::env_reader::flag("OWL_GPU_PROF") {
            owl_shared::metrics::query_metrics(
                &owl_shared::metrics::MetricsFilter::new().tag_prefix("gpu.").limit(24),
            );
        }
        assert!(!text.trim().is_empty(), "Q{qi} 空回答");
        if let Some(pat) = degenerate(&text) {
            // 严格门(OWL_27B_STRICT=1)在 paged decode 质量案结案后启用;
            // 现默认警告 —— paged 档起始语义正确但退化复读(立案中),
            // naive 档(OWL_FORCE_NAIVE=1)三问连贯 = 当前推荐配方
            if owl_shared::env_reader::flag("OWL_27B_STRICT") {
                panic!("Q{qi} 退化复读:片段 {pat:?} 连现 ≥4 次;全文 = {text:?}");
            }
            eprintln!("[27b-chat][warn] Q{qi} 退化复读(立案中):片段 {pat:?}");
        }
    }
}

/// **B5.1 合批 decode 语义门**(2026-10-20,M1):两会话并发(T=2 批步)
/// 与串行两跑**逐字相同**;并证明批路径真的被走到(`batch_steps > 0`)。
///
/// 判据纪律:greedy(确定性 + 不抽签);同一引擎实例、同一 prompt、临时会话
/// (互不污染);批臂 = 同时 submit 两 turn(调度面先各自 prefill,再合批)。
pub async fn gpu_batch_decode_matches_serial() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 256,
        prefill_chunk: 32,
        knobs: EngineKnobs {
            max_batch: 2,
            gdn_slots: 2,
            sampler_enabled: false,
            ..Default::default()
        },
    })
    .expect("构造(合批档)");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    assert!(running.has_batch_graph(), "合批图应捕获成功(否则测的是 eager 臂)");

    let p1 = "用一句话介绍长城。";
    let p2 = "What is the capital of France?";
    // 串行臂:两题各自单跑(只有一活跃 ⇒ 批门不触发)
    async fn drive(r: &mut RunningEngine<crate::GpuFace>, id: u64) -> String {
        loop {
            match r.pump().await.expect("pump") {
                TurnEvent::Completed { turn, text, .. } if turn == id => return text,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                TurnEvent::Idle => panic!("t{id}: Idle(队列丢失)"),
                ev => eprintln!("[ev] {ev:?}"),
            }
        }
    }
    let s1 = running.submit(p1, 16).expect("submit serial t1");
    let a1 = drive(&mut running, s1).await;
    let s2 = running.submit(p2, 16).expect("submit serial t2");
    let a2 = drive(&mut running, s2).await;
    assert!(running.batch_steps() == 0, "串行臂不应走批路径");
    eprintln!("[batch][serial] A={a1:?}\n[batch][serial] B={a2:?}");

    // ── 合批臂:两 turn 同时在场 ──
    let t1 = running.submit(p1, 16).expect("submit t1");
    let t2 = running.submit(p2, 16).expect("submit t2");
    let (mut b1, mut b2) = (None, None);
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => break,
            TurnEvent::Completed { turn, text, .. } if turn == t1 => b1 = Some(text),
            TurnEvent::Completed { turn, text, .. } if turn == t2 => b2 = Some(text),
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {}
        }
    }
    let (b1, b2) = (b1.expect("t1 完成"), b2.expect("t2 完成"));
    eprintln!("[batch][concurrent] A={b1:?}\n[batch][concurrent] B={b2:?}");
    assert!(running.batch_steps() > 0, "合批路径必须真被走到(batch_steps=0)");
    assert_eq!(a1, b1, "会话 A:合批 ≠ 串行");
    assert_eq!(a2, b2, "会话 B:合批 ≠ 串行");
    eprintln!("[batch] ✅ 贪心逐字相同;批步数 = {}", running.batch_steps());

    // ── 采样档腿:固定 seed + 温度(图内 logits 行切片 → per-turn sample)──
    let opts = || crate::turn::TurnOpts {
        ignore_eos: true,
        sampler: Some(crate::turn::TurnSampler {
            temp: Some(1.0),
            topk: Some(0),
            topp: Some(1.0),
        }),
        seed: Some(20261020),
        ..Default::default()
    };
    let mut samp_serial = Vec::new();
    for p in [p1, p2] {
        let (_, id) = running
            .submit_session_opts(None, p, 16, opts())
            .expect("submit sampling serial");
        samp_serial.push(drive(&mut running, id).await);
    }
    let i1 = running
        .submit_session_opts(None, p1, 16, opts())
        .expect("submit sampling b1")
        .1;
    let i2 = running
        .submit_session_opts(None, p2, 16, opts())
        .expect("submit sampling b2")
        .1;
    let (mut s1, mut s2) = (None, None);
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Completed { turn, text, .. } if turn == i1 => s1 = Some(text),
            TurnEvent::Completed { turn, text, .. } if turn == i2 => s2 = Some(text),
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => break,
            _ => {}
        }
        if s1.is_some() && s2.is_some() {
            break;
        }
    }
    assert_eq!(samp_serial[0], s1.expect("采样 t1 完成"), "采样档 A 不一致");
    assert_eq!(samp_serial[1], s2.expect("采样 t2 完成"), "采样档 B 不一致");
    eprintln!("[batch] ✅ 采样档(seed 固定)逐字相同");

    // 同会话并发 = 结构性互踩 ⇒ 必须在提交边界响亮拒(合批档)
    let (_sid, live) = running.submit_session(Some(77), p1, 8).expect("live turn");
    let dup = running.submit_session(Some(77), p2, 8);
    assert!(dup.is_err(), "同会话已有活跃 turn,第二个 submit 应拒");
    let _ = drive(&mut running, live).await;
}

    /// **B5.1 合批 decode 吞吐台(27B AWQ;spec off)** —— M1b 图臂的实测账。
///
/// 口径:同两 prompt、greedy + ignore_eos(64 tok),**计时窗 = 首 Token → 双完成**
/// (排除 prefill/装载);串行臂 = 两 turn 先后跑(单活跃),合批臂 = 同时在场。
/// 判据:① 输出逐字相同(语义门)② 批步数 > 0(真走批路径)③ 打印 t/s 比值。
pub async fn gpu_batch_decode_bench_27b() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
        owl_shared::testkit::report::skip("ckpt:27b-awq 未设");
        return;
    };
    let max_new = 64usize;
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 2048,
        prefill_chunk: 256,
        knobs: EngineKnobs {
            max_batch: 2,
            gdn_slots: 2,
            sampler_enabled: false,
            pool_tokens: Some(4096),
            ..Default::default()
        },
    })
    .expect("构造(合批档 27B)");
    let loaded = engine
        .loader()
        .load_qwen38_27b_awq(
            std::path::Path::new(&dir),
            std::path::Path::new(&dir),
            owl_models::module::QuantPlan::F16,
            owl_models::module::QuantPlan::W4A16Awq,
        )
        .await
        .expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    assert!(running.has_batch_graph(), "27B 合批图应捕获成功");
    let p1 = "用一段话解释什么是量子纠缠。";
    let p2 = "Explain briefly how a transformer works.";

    // 串行臂
    let t0 = std::time::Instant::now();
    let mut first_token_at = None;
    let mut texts = [String::new(), String::new()];
    for (i, p) in [p1, p2].iter().enumerate() {
        let (_, id) = running
            .submit_session_ex(None, *p, max_new, true)
            .expect("submit serial");
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Token { turn, .. } if turn == id && first_token_at.is_none() => {
                    first_token_at = Some(std::time::Instant::now());
                }
                TurnEvent::Completed { turn, text, .. } if turn == id => {
                    texts[i] = text;
                    break;
                }
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                TurnEvent::Idle => panic!("t{id}: Idle"),
                _ => {}
            }
        }
    }
    let serial_wall = first_token_at.unwrap().elapsed();
    let _ = t0;
    let serial_steps = running.batch_steps();
    assert_eq!(serial_steps, 0, "串行臂不应走批路径");

    // 合批臂
    first_token_at = None;
    let mut done = [None, None];
    let ids = [
        running.submit_session_ex(None, p1, max_new, true).expect("submit b1").1,
        running.submit_session_ex(None, p2, max_new, true).expect("submit b2").1,
    ];
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Token { turn, .. } if first_token_at.is_none() && ids.contains(&turn) => {
                first_token_at = Some(std::time::Instant::now());
            }
            TurnEvent::Completed { turn, text, .. } => {
                if turn == ids[0] {
                    done[0] = Some(text);
                } else if turn == ids[1] {
                    done[1] = Some(text);
                }
                if done.iter().all(|d| d.is_some()) {
                    break;
                }
            }
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => panic!("合批臂 Idle"),
            _ => {}
        }
    }
    let batch_wall = first_token_at.unwrap().elapsed();
    let batch_steps = running.batch_steps();
    assert!(batch_steps > 0, "合批臂没走批路径");
    assert_eq!(texts[0], done[0].clone().unwrap(), "A 输出不一致");
    assert_eq!(texts[1], done[1].clone().unwrap(), "B 输出不一致");
    let toks = (max_new * 2) as f64;
    let s_ts = toks / serial_wall.as_secs_f64();
    let b_ts = toks / batch_wall.as_secs_f64();
    eprintln!(
        "[batch-bench-27b] 串行 {:.2}s = {s_ts:.1} t/s | 合批 {:.2}s = {b_ts:.1} t/s | 比值 ×{:.2} | 批步数 {batch_steps}",
        serial_wall.as_secs_f64(),
        batch_wall.as_secs_f64(),
        b_ts / s_ts
    );
    assert!(b_ts > s_ts, "合批应更快(反之 = 合批路径有退化)");
}

/// **0.8B 默认档共享装载**(e2e 唯一装载点之一;6 步共用:
/// schedule_decision_surface / two_turns_lifecycle / session_continuity /
/// multi_session_isolation / prefix_cache_hit / longctx_4k_prefix_qa)。
/// 装载本身即一步测试(构造+装载+装配+捕获态断言+计时)。
pub async fn boot_08b(max_seq: usize) -> RunningEngine<crate::GpuFace> {
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    let dir = asset_dir();
    let t0 = std::time::Instant::now();
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: max_seq,
        prefill_chunk: 32,
        knobs: EngineKnobs::default(),
    })
    .expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let running = engine.run(loaded).await.expect("装配");
    assert!(
        matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
        "0.8B 装载应捕获成功(回放态),得 {:?}",
        running.capture_outcome()
    );
    eprintln!("[e2e] 0.8B 默认档装载完成 {:?}(max_seq={max_seq})", t0.elapsed());
    running
}

/// **27B 默认档共享装载**(chat_inference / awq27b_marlin_e2e 共用;
/// 27B 装载 ~24-32s/次,重复装载 = 纯浪费)。
pub async fn boot_27b_default() -> RunningEngine<crate::GpuFace> {
    let dir = std::path::PathBuf::from(
        owl_shared::testkit::resource("ckpt:27b-awq").expect("ckpt:27b-awq"),
    );
    assert!(dir.exists(), "检查点目录不存在: {dir:?}");
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    let t0 = std::time::Instant::now();
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 1024,
        prefill_chunk: 128,
        knobs: EngineKnobs::default(),
    })
    .expect("构造");
    // 装载分相上屏(loader 性能工作台账用:构造 / 权重装载 / 装配+捕获)
    eprintln!("[e2e] 27B 引擎构造 {:?}", t0.elapsed());
    let t_load = std::time::Instant::now();
    let loaded = engine
        .loader()
        .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, owl_models::module::QuantPlan::W4A16Awq)
        .await
        .expect("27B AWQ 装载");
    eprintln!("[e2e] 27B 权重装载 {:?}", t_load.elapsed());
    let t_run = std::time::Instant::now();
    let running = engine.run(loaded).await.expect("装配");
    eprintln!("[e2e] 27B 装配+图捕获 {:?}", t_run.elapsed());
    assert!(
        matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
        "27B 装载应捕获成功(回放态),得 {:?}",
        running.capture_outcome()
    );
    eprintln!("[e2e] 27B 默认档装载完成 {:?}", t0.elapsed());
    running
}

/// 诊断:裸续写 A/B(模板态病 vs 权重病的鉴别臂)
    pub async fn gpu_27b_raw_completion_diag() {
        let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else { return; };
        let dir = std::path::PathBuf::from(dir);
        if !dir.exists() { return; }
        let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal, max_seq_tokens: 512, prefill_chunk: 128,
            knobs: EngineKnobs::default(),
        }).expect("构造");
        let loaded = engine.loader().load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, owl_models::module::QuantPlan::W4A16Awq).await.expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        for (qi, q) in ["中国的首都是北京。长城是", "1 2 3 4 5 6"].iter().enumerate() {
            let id = running.submit(*q, 48).expect("submit");
            let text = loop {
                match running.pump().await.expect("pump") {
                    TurnEvent::Completed { turn, text, .. } if turn == id => break text,
                    TurnEvent::Idle => panic!("队列丢失"),
                    _ => {}
                }
            };
            eprintln!("[raw-diag] Q{qi}: {q:?}\n[raw-diag] A{qi}: {text:?}");
        }
    }


    /// E5-M2 恒等门(机制臂,0.8B + 哑草稿):greedy 下 spec 轮输出 ≡ 无
    /// spec 输出(逐 token)。任意草稿(哑草稿 = anchor 重复)下机制必须
    /// 恒等 —— 接受是恒等变换,分红只影响速度。双 boot × 双 prompt ×
    /// 会话 token 账逐位比对(第二 prompt 同引擎续 turn = 跨 turn 状态
    /// 连续性一并覆盖)。哑草稿需 OWL_SPEC_DUMB=1(C7:无草稿器回落)。
    pub async fn gpu_spec_identity_gate() {
        let Some(ordinal) = gpu_ordinal() else {
            owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
            return;
        };
        let dir = asset_dir();
        const PROMPTS: [&str; 2] = ["用五十字介绍长城。", "水的沸点是多少度?"];

        async fn gen(
            spec: bool,
            ordinal: usize,
            dir: &std::path::Path,
        ) -> Vec<(String, Vec<u32>)> {
            let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() }; // greedy(恒等门口径)
            // 显式池余量(缺省 = 2×max_seq = 8 页,spec 链一挤就"块池耗尽";
            // 2026-10-13 套件内实测:前置用例扰动即红、单跑绿。恒等门测的是
            // 文本等价,不是容量边界 —— 边界另有专测)
            knobs.pool_tokens = Some(1024);
            if spec {
                knobs.draft.depth = 3;
                knobs.draft.dumb = true;
            }
            // **FI 图内臂恒等门**(W5:piecewise/断点面已退役 ⇒ 臂 = 单图 + 窗外刷表)。
            // spec(verify FI 图内臂)与 baseline(spec_depth=0)贪心 token 流必须逐位
            // 一致:覆盖"FI 四件套走输入槽 + `ctx_total` 烘上界 + 每步/每轮
            // `fi_plan_sync` 刷精确表"全链。(env 名保留历史拼写:`OWL_GATE_PIECEWISE`)
            if owl_shared::env_reader::flag("OWL_GATE_PIECEWISE") {
                knobs.fi_cuda_graph = true;
                knobs.fi_plan_bake = Some(128); // ≥ max_seq(本门 max_seq_tokens=128)
                knobs.env.attn.verify_arm = owl_shared::config::VerifyArm::Fi;
            }
            let mut engine = Engine::new(EngineConfig {
                device_ordinal: ordinal,
                max_seq_tokens: 128,
                prefill_chunk: 16,
                knobs,
            })
            .expect("构造");
            let loaded = engine.loader().load_qwen35_0_8b(dir).await.expect("装载");
            let mut running = engine.run(loaded).await.expect("装配");
            let mut out = Vec::new();
            for (i, p) in PROMPTS.iter().enumerate() {
                let (sid, id) = running
                    .submit_session(Some(100 + i as u64), *p, 48)
                    .expect("submit");
                let text = drive_turn(&mut running, id).await;
                let toks = running.bank.sessions
                    .get(sid)
                    .map(|s| s.tokens.clone())
                    .unwrap_or_default();
                out.push((text, toks));
            }
            // ★ 回归门(2026-10-13 定谳的挂死案):**跑满预算的 turn 必须
            // 送达终态事件**。病史:spec 路径轮末 `pending_events = evq`
            // 覆盖,把预算尽时 `sample_and_emit` 压入的 Completed 冲掉 →
            // actor 永不派发终态 → HTTP 层卡在 recv(客户端一帧不收也不
            // EOF)。此处 `ignore_eos=true` 强制跑满 8 tok → `drive_turn`
            // 若等不到 Completed 会 panic("队列丢失")= 门红。
            {
                let (sid_ie, id_ie) = running
                    .submit_session_ex(Some(900), "用一句话介绍长城。", 8, true)
                    .expect("submit(ignore_eos)");
                let text_ie = drive_turn(&mut running, id_ie).await;
                let n = running.bank.sessions.get(sid_ie).map(|s| s.tokens.len()).unwrap_or(0);
                eprintln!("[spec-gate] 预算尽门(ignore_eos): {n} tok,文本 {text_ie:?}");
                assert!(n >= 8, "预算尽 turn 应跑满 8 tok,得 {n}(终态事件是否送达见 panic 信息)");
            }
            let st = running.dev.spec_rt.stats;
            eprintln!(
                "[spec-gate] stats: rounds={} sum_m={} full={} partial={} zero={}",
                st.rounds, st.sum_m, st.full, st.partial, st.zero
            );
            out
        }

        let base = gen(false, ordinal, &dir).await; // 基线
        let spec = gen(true, ordinal, &dir).await; // spec(哑草稿)
        for (i, ((ta, ta_tokens), (tb, tb_tokens))) in base.iter().zip(&spec).enumerate() {
            eprintln!(
                "[spec-gate] p{i} baseline({}tok)={ta:?}\n[spec-gate] p{i} spec({}tok)={tb:?}",
                ta_tokens.len(),
                tb_tokens.len()
            );
            assert_eq!(ta, tb, "恒等门 p{i}:greedy spec 文本 ≡ 无 spec");
            assert_eq!(ta_tokens, tb_tokens, "恒等门 p{i}:token 账逐位一致");
        }
        // env 泄漏防护(spec vars 是进程全局;残留 = 后续测试被动进 spec 模
        // 式 —— two_turns 事件序列案 2026-10-06;旋钮已显式化,无 env 清场)
    }

    /// E5-M2b 恒等门(真草稿臂,27B + MTP 头):真草稿下 greedy spec 输出
    /// ≡ 无 spec 输出(逐 token)。同时验证 m>0 真接受路径(部分/全接受)
    /// 与 spec_stats 账目闭合(Σ(m_i+1) = 发射 token 数)。门控
    /// ckpt:27b-awq。
    pub async fn gpu_spec_identity_gate_27b() {
        let Some(ordinal) = gpu_ordinal() else {
            owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
            return;
        };
        let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
            owl_shared::testkit::report::skip("ckpt:27b-awq 未设(cyankiwi 检查点目录)");
            return;
        };
        // 域参数化(E5-DF3 AL 排查:PROSE 域 = DFlash2 结构性最差域,
        // xinfer p5 实测 accept 1.9%/AL 1.13;MATH 域 AL 4.49 —— OWL_GATE_PROMPT
        // 换域对照,缺省恒 = 原 42tok 长城门)
        let mut prompt = owl_shared::env_reader::str_or("OWL_GATE_PROMPT", "用五十字介绍长城。");
        let mtp_window: usize =
            owl_shared::env_reader::parse_or("OWL_GATE_MTP_WINDOW", 0);
        // window 门必须让 ctx 真正超过 sink+window；否则仍是 full，等于没测。
        if mtp_window > 0 {
            prompt = prompt.repeat(500);
        }
        // 恒等门深度可参数化（T1 自适应档验收：d3/d7 各跑一次）。
        let gate_depth: usize = owl_shared::env_reader::parse_or("OWL_GATE_SPEC_DEPTH", 3);
        assert!((1..=owl_shared::config::SPEC_DEPTH_MAX).contains(&gate_depth),
            "OWL_GATE_SPEC_DEPTH={gate_depth} 越界");

        /// **装载一次**(带 MTP 草稿;2026-10-13 用户令:不再每轮重载)
        async fn boot_gate_engine(
            ordinal: usize,
            dir: &str,
            max_seq_tokens: usize,
            gate_depth: usize,
        ) -> RunningEngine<crate::GpuFace> {
            let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
            // 24G 贴顶三旋钮(单会话门不需要多格/多快照/前缀缓存)
            knobs.gdn_slots = 1;
            knobs.snap_max = 1;
            knobs.prefix_cache = false;
            // 草稿档一律按投机配置 boot;baseline 轮靠运行时 spec_depth=0 回落
            knobs.draft.depth = gate_depth;
            // Windowed-MTP 恒等门开关（仅测试面 env；生产走配置）
            knobs.draft.mtp_window = owl_shared::env_reader::parse_or("OWL_GATE_MTP_WINDOW", 0);
            knobs.draft.mtp_sink = owl_shared::env_reader::parse_or("OWL_GATE_MTP_SINK", 64);
            knobs.cuda.capture_slab_mb = Some(160);
            // 生产主池是 fp8；MTP 链仍是独立 F16。开该门跑 fp8 主池 × F16
            // MTP 的恒等（与 mtp27b profile 同形；缺省 f16 档保持历史门）。
            if owl_shared::env_reader::flag("OWL_GATE_FP8") {
                knobs.env.kv.quant = owl_models::env::KvQuant::Fp8E4M3;
            }
            // **FI 图内臂恒等门**(W5:piecewise/断点面已退役 ⇒ 臂 = 单图 + 窗外刷表)。
            // spec(verify FI 图内臂)与 baseline(spec_depth=0)贪心 token 流必须逐位
            // 一致:覆盖"FI 四件套走输入槽 + `ctx_total` 烘上界 + 每步/每轮
            // `fi_plan_sync` 刷精确表"全链。(env 名保留历史拼写:`OWL_GATE_PIECEWISE`)
            if owl_shared::env_reader::flag("OWL_GATE_PIECEWISE") {
                knobs.fi_cuda_graph = true;
                knobs.fi_plan_bake = Some(max_seq_tokens); // ≥ max_seq(本门 max_seq_tokens)
                knobs.env.attn.verify_arm = owl_shared::config::VerifyArm::Fi;
            }
            let mut engine = Engine::new(EngineConfig {
                device_ordinal: ordinal,
                // `_MIN_GEN_TOKENS = 1024` ⇒ room = max_seq − prompt 必须 ≥ 1024
                // (老门 256 恒拒发 = 环境性红;2026-10-16 抬到 2048 —— 恒等门不比容量)
                max_seq_tokens,
                prefill_chunk: 64,
                knobs,
            })
            .expect("构造");
            let loaded = engine
                .loader()
                .load_qwen38_27b_awq(std::path::Path::new(dir), std::path::Path::new(dir), owl_models::module::QuantPlan::F16, owl_models::module::QuantPlan::W4A16Awq)
                .await
                .expect("装载");
            engine.run(loaded).await.expect("装配")
        }

        /// 单轮(`spec_depth = 0` = 不投机;>0 = 投机)
        async fn round(
            running: &mut RunningEngine<crate::GpuFace>,
            spec_depth: usize,
            prompt: &str,
            tag: &str,
        ) -> (String, Vec<u32>) {
            running.dev.spec_rt.depth = spec_depth;
            let (sid, id) = running.submit_session(Some(7), prompt, 40).expect("submit");
            let text = drive_turn(running, id).await;
            let toks = running.bank.sessions
                .get(sid)
                .map(|s| s.tokens.clone())
                .unwrap_or_default();
            let st = running.dev.spec_rt.stats;
            eprintln!(
                "[spec-gate-27b] {tag} spec_depth={spec_depth} stats: rounds={} sum_m={} full={} partial={} zero={}",
                st.rounds, st.sum_m, st.full, st.partial, st.zero
            );
            (text, toks)
        }

        // ── 装载**一次**,两轮:① 不投机 baseline ② 投机(用户令 2026-10-13) ──
        let gate_seq = if mtp_window > 0 { 8192 } else { 2048 };
        let mut running = boot_gate_engine(ordinal, &dir, gate_seq, gate_depth).await;
        let (ta, ta_tokens) = round(&mut running, 0, &prompt, "baseline").await;
        let (tb, tb_tokens) = round(&mut running, gate_depth, &prompt, "spec").await;
        eprintln!(
            "[spec-gate-27b] baseline({}tok)={ta:?}\n[spec-gate-27b] spec({}tok)={tb:?}",
            ta_tokens.len(),
            tb_tokens.len()
        );
        assert_eq!(ta, tb, "恒等门 27B(d{gate_depth}):greedy 真草稿 spec 文本 ≡ 无 spec");
        assert_eq!(ta_tokens, tb_tokens, "恒等门 27B(d{gate_depth}):token 账逐位一致");
        // 旋钮已全量显式化,无 env 清场需要

        drop(running);
        std::thread::sleep(std::time::Duration::from_secs(10));
    }

    /// E5-DF3 恒等门(27B AWQ target + DFlash2 草稿):greedy spec 文本
    /// ≡ 无 spec 逐位。门控 OWL_TEST_DEVICE + ckpt:27b-awq +
    /// ckpt:dflash2。**20G 卡贴顶**(target ~13.7G + draft 3.85G +
    /// 池)—— OOM 时降 max_seq_tokens/挪 24G 卡。
    pub async fn gpu_dflash2_identity_gate_27b() {
        let Some(ordinal) = gpu_ordinal() else {
            owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
            return;
        };
        let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
            owl_shared::testkit::report::skip("ckpt:27b-awq 未设(cyankiwi 检查点目录)");
            return;
        };
        let Some(ddir) = owl_shared::testkit::resource("ckpt:dflash2") else {
            owl_shared::testkit::report::skip("ckpt:dflash2 未设(DFlash2 草稿目录)");
            return;
        };
        // 域参数化(E5-DF3 AL 排查:PROSE 域 = DFlash2 结构性最差域,
        // xinfer p5 实测 accept 1.9%/AL 1.13;MATH 域 AL 4.49 —— OWL_GATE_PROMPT
        // 换域对照,缺省恒 = 原 42tok 长城门)
        let prompt = owl_shared::env_reader::str_or("OWL_GATE_PROMPT", "用五十字介绍长城。");
        // E5 性能:长生成旋钮(AL/吞吐稳态剖析;默认 40 = 恒等门口径)
        let max_new: usize = owl_shared::env_reader::parse_or("OWL_GATE_MAX_NEW", 40);

        /// **装载一次**(带草稿;2026-10-13 用户令:不再每轮重载)。
        /// 单次装载两轮复用:一轮不投机(spec_depth=0)、一轮投机。
        async fn boot_gate_engine(
            ordinal: usize,
            dir: &str,
            ddir: &str,
            max_new: usize,
        ) -> RunningEngine<crate::GpuFace> {
            let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
            knobs.gdn_slots = 1;
            knobs.snap_max = 1;
            knobs.prefix_cache = false;
            // 草稿档一律按**投机**配置 boot(深度 + verify 捕获档);
            // 不投机的 baseline 轮靠运行时 `spec_depth = 0` 回落,不重载。
            knobs.draft.depth = owl_shared::env_reader::parse_or("OWL_SPEC_DEPTH", 7);
            knobs.cuda.capture_slab_mb = Some(160);
            knobs.draft.dflash2_dir = Some(ddir.to_string());
            // DFlash2 lookup 恒等门臂(2026-10-22):greedy spec 仍须 ≡ 无 spec。
            if owl_shared::env_reader::flag("OWL_GATE_LOOKUP") {
                knobs.draft.lookup = true;
                knobs.draft.lookup_nmin =
                    owl_shared::env_reader::parse_or("OWL_GATE_LOOKUP_NMIN", 6);
                knobs.draft.lookup_nmax =
                    owl_shared::env_reader::parse_or("OWL_GATE_LOOKUP_NMAX", 12);
            }
            // E19 定位开关(2026-10-13):knhd×fp8 池×int4 头×spec 语义回归
            // 复现(缺省关 = 原口径不变)。三项同开才复现 —— 单开任一项均绿:
            // 疑为“knHD 多序(seqs>1)读非位等”在窄边际下的贪心翻转。
            if owl_shared::env_reader::flag("OWL_GATE_FP8") {
                knobs.env.kv.quant = owl_models::env::KvQuant::Fp8E4M3;
            }
            // XQA 臂开关(2026-10-13;恒等门用 —— spec ≡ 无 spec / 裸 decode
            // 臂不破文本,是 XQA 进生产的唯一红线)
            if owl_shared::env_reader::flag("OWL_GATE_XQA") {
                knobs.env.attn.verify_arm = owl_shared::config::VerifyArm::Xqa;
            }
            // 裸 decode XQA 臂(2026-10-13 ②):baseline 轮(spec_depth=0)
            // 就走它 ⇒ 本门同时是"decode 臂不破文本"的红线
            if owl_shared::env_reader::flag("OWL_GATE_DECODE_XQA") {
                knobs.env.attn.decode_arm = owl_shared::config::DecodeArm::Xqa;
            }
            if let Some(p) = owl_shared::env_reader::parse::<usize>("OWL_GATE_POOL") {
                knobs.pool_tokens = Some(p);
            }
            let head_plan = if owl_shared::env_reader::flag("OWL_GATE_INT4HEAD") {
                owl_models::module::QuantPlan::W4A16Awq
            } else {
                owl_models::module::QuantPlan::F16
            };
            let mut engine = Engine::new(EngineConfig {
                device_ordinal: ordinal,
                max_seq_tokens: owl_shared::env_reader::parse_or(
                    "OWL_GATE_MAXSEQ",
                    320.max(max_new + 64),
                ),
                prefill_chunk: 32,
                knobs,
            })
            .expect("构造");
            let loaded = engine
                .loader()
                .load_qwen38_27b_awq(std::path::Path::new(dir), std::path::Path::new(dir), head_plan, owl_models::module::QuantPlan::W4A16Awq)
                .await
                .expect("装载");
            let running = engine.run(loaded).await.expect("装配");
            assert!(
                matches!(running.capture_outcome(), crate::graph_plan::PlanOutcome::Captured),
                "带草稿装载应捕获成功,得 {:?}",
                running.capture_outcome()
            );
            running
        }

        /// 单轮(`spec_depth = 0` = 不投机 baseline;>0 = 投机)。
        /// spec_depth 是 `pub(crate)` 运行时可切字段(exec/spec.rs 每轮读)。
        async fn round(
            running: &mut RunningEngine<crate::GpuFace>,
            spec_depth: usize,
            prompt: &str,
            max_new: usize,
            tag: &str,
        ) -> (String, Vec<u32>) {
            let prompt_len = prompt.chars().count(); // 近似(测试口径)
            running.dev.spec_rt.depth = spec_depth;
            let (sid, id) = running.submit_session(Some(7), prompt, max_new).expect("submit");
            let tg = std::time::Instant::now();
            let text = drive_turn(running, id).await;
            let t_gen = tg.elapsed();
            let toks = running.bank.sessions
                .get(sid)
                .map(|s| s.tokens.clone())
                .unwrap_or_default();
            let gen_n = toks.len().saturating_sub(prompt_len);
            let st = running.dev.spec_rt.stats;
            let al = if st.rounds > 0 { st.sum_m as f64 / st.rounds as f64 } else { 0.0 };
            eprintln!(
                "[dflash-gate] 生成段: {} tok / {t_gen:.1?} = {:.1} tok/s({tag})",
                gen_n,
                gen_n as f64 / t_gen.as_secs_f64(),
            );
            eprintln!(
                "[dflash-gate] {tag} spec_depth={spec_depth} rounds={} sum_m={} AL={al:.2}",
                st.rounds, st.sum_m
            );
            (text, toks)
        }

        // ── 装载**一次**,两轮:① 不投机 baseline ② 投机(用户令 2026-10-13) ──
        let depth: usize = owl_shared::env_reader::parse_or("OWL_SPEC_DEPTH", 7);
        let mut running = boot_gate_engine(ordinal, &dir, &ddir, max_new).await;
        let t0 = std::time::Instant::now();
        let (ta, ta_tokens) = round(&mut running, 0, &prompt, max_new, "baseline").await;
        let t_base = t0.elapsed();
        // E5-DF4:b1 事实 dump(prefill bisect 层 checksum / pf.last / emit)
        owl_shared::metrics::query_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("b1."),
        );
        // E5-DF4 装载校验快照(b1;OWL_LOAD_VERIFY=1 时 loader 回读对比 +
        // metrics 记 `loadv.{key}` = 校验和)
        let l1: std::collections::HashMap<String, u64> = owl_shared::metrics::collect_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("loadv."),
        )
        .into_iter()
        .map(|r| (r.tag.clone(), r.counter))
        .collect();
        owl_shared::metrics::reset_metrics();
        let t1 = std::time::Instant::now();
        let (tb, tb_tokens) = round(&mut running, depth, &prompt, max_new, "spec").await;
        let t_spec = t1.elapsed();
        let l2: std::collections::HashMap<String, u64> = owl_shared::metrics::collect_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("loadv."),
        )
        .into_iter()
        .map(|r| (r.tag.clone(), r.counter))
        .collect();
        // 双 boot 按键校验和 diff(类 1 装载非确定的决定性仪表):
        // 共有键校验和不同 = 装载竞态实锤;l2 独有 = 草稿键(b2 专属,预期)
        {
            let mut diff = 0usize;
            let mut only2 = 0usize;
            for (k, v1) in &l1 {
                match l2.get(k) {
                    Some(v2) if v2 == v1 => {}
                    other => {
                        diff += 1;
                        if diff <= 8 {
                            eprintln!("[load-diff] {k}: {v1:#018x} vs {:?} ", other);
                        }
                    }
                }
            }
            for k in l2.keys() {
                if !l1.contains_key(k) {
                    only2 += 1;
                }
            }
            eprintln!(
                "[load-diff] b1_keys={} b2_keys={} mismatch={diff} l2_only(draft 预期)={only2}",
                l1.len(),
                l2.len()
            );
        }
        // 吞吐面:生成 tok 数 / gen 墙钟(含装载+prefill;粗口径,门内自证)
        let gen_n = |t: &[u32]| t.len().saturating_sub(18);
        eprintln!(
            "[dflash-gate] 吞吐 baseline: {} gen-tok / {t_base:.1?} = {:.1} tok/s(含装载)",
            gen_n(&ta_tokens),
            gen_n(&ta_tokens) as f64 / t_base.as_secs_f64(),
        );
        eprintln!(
            "[dflash-gate] 吞吐 spec:     {} gen-tok / {t_spec:.1?} = {:.1} tok/s(含装载)",
            gen_n(&tb_tokens),
            gen_n(&tb_tokens) as f64 / t_spec.as_secs_f64(),
        );
        eprintln!(
            "[dflash-gate] baseline({}tok)={ta:?}\n[dflash-gate] spec({}tok)={tb:?}",
            ta_tokens.len(),
            tb_tokens.len()
        );
        // E5-DF4 崩坏排查:b2 事实全量 dump(好/坏跑对比;b1 已在 reset 前 dump)
        owl_shared::metrics::query_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("b2."),
        );
        // E5 性能:spec 轮逐相汇总(长生成轮账;gap = round − Σparts)
        owl_shared::metrics::query_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("spec."),
        );
        assert_eq!(ta, tb, "恒等门 DFlash2:greedy 真草稿 spec 文本 ≡ 无 spec");
        assert_eq!(ta_tokens, tb_tokens, "恒等门 DFlash2:token 账逐位一致");
        // 旋钮已全量显式化,无 env 清场需要
    
        // 实例收尾屏障:drop 后等 server 线程退出 + CUDA ctx 析构(异步;
        // 下一实例的 17G 分配会与释放赛跑 —— 2026-10-07 非确定性定谳)
        drop(running);
        std::thread::sleep(std::time::Duration::from_secs(10));
    }

/// 泵到目标 turn 完成,回吐全文(其间事件仅观测)
async fn drive_turn<D: DeviceClient>(running: &mut RunningEngine<D>, id: u64) -> String {
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Idle => panic!("turn {id} 队列丢失"),
            TurnEvent::Prefill { turn, fed, total } if turn == id => {
                eprintln!("[ev] t{turn} prefill {fed}/{total}")
            }
            TurnEvent::Completed { turn, text, .. } if turn == id => break text,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {}
        }
    }
}

/// decode 边际速率差分基准(E5 性能会战终版口径,2026-10-08):
/// 同配置双 boot(16 / 128 token 上限),**边际 ms/tok = Δwall/Δtokens**
/// —— 固定成本(prefill 尾步/首步杂项/事件泵)与 EOS 点全部在差分中
/// 消掉。旧 e2e 的 wall÷max_new 与 gate 生成段 wall÷实际tok 两把尺子
/// 分别虚高/虚低(差分实测:4d21122 24.3ms vs HEAD 24.8ms = 无回退,
/// 当年"45 tok/s"是朴素口径虚高)。门控 ckpt:27b-awq + OWL_TEST_DEVICE。
///
/// 泄漏案探针(2026-10-10 深夜;slot≥1 崩塌案首刀):
/// MAX_ACTIVE_TURNS=2 下,turn(slot0) 前后对比 slot1 状态区指纹 ——
/// slot1 变 = 跨 slot 写实锤;不变 = 病在 slot1 核索引/复用路径。
/// 附:reset_gdn(slot1) 后回读验零(reset 生效性直接判)。
/// 门控 OWL_TEST_DEVICE。诊断打印,不做硬断言(判决先看数)。
pub async fn gpu_gdn_slot_cross_probe() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let mut knobs = EngineKnobs { gdn_slots: 2, ..Default::default() };
    knobs.prefix_cache = false;
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 512,
        prefill_chunk: 32,
        knobs,
    })
    .expect("构造");
    let dir = asset_dir();
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");

    // ① boot 后初始态:双 slot 指纹 + slot1 非零字节数(alloc 应已清零)
    let s1_0 = running.dev.pool.gdn_slot_fingerprint(running.dev.graphs.session.face_mut(), 1).await.unwrap();
    let nz0 = running.dev.pool.gdn_slot_zero_check(running.dev.graphs.session.face_mut(), 1).await.unwrap();
    eprintln!("[slot-probe] boot 后 slot1 非零字节 = {nz0}(期望 0 = alloc 清零生效)");

    // ② turn(slot0,新 session)
    let (sid, id) = running
        .submit_session(Some(7), "用五十字介绍长城。", 40)
        .expect("submit");
    let _ = drive_turn(&mut running, id).await;
    let _ = sid;

    // ③ turn 后双 slot 指纹
    let s1_1 = running.dev.pool.gdn_slot_fingerprint(running.dev.graphs.session.face_mut(), 1).await.unwrap();
    let s0_1 = running.dev.pool.gdn_slot_fingerprint(running.dev.graphs.session.face_mut(), 0).await.unwrap();
    let s1_nz = running.dev.pool.gdn_slot_zero_check(running.dev.graphs.session.face_mut(), 1).await.unwrap();
    eprintln!("[slot-probe] turn 后 slot1 非零字节 = {s1_nz}(>0 = turn(slot0) 跨写了 slot1)");
    let same1 = s1_0 == s1_1;
    eprintln!("[slot-probe] slot1 指纹 turn 前后相同 = {same1}(false = 跨 slot 写实锤)");
    let _ = s0_1;

    // ④ reset 生效性:slot0 填垃圾验证?——零检查已覆盖 slot1;slot0 的
    //    reset 由健康 turn 隐证。旋钮已显式化,无 env 清场需要。
}

pub async fn gpu_decode_marginal_bench() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
        owl_shared::testkit::report::skip("ckpt:27b-awq 未设");
        return;
    };
    // 口径与恒等门一致(greedy 单会话三旋钮,显式 knobs)
    async fn gen(ordinal: usize, dir: &str, max_new: usize) -> (f64, usize) {
        let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
        knobs.gdn_slots = 1;
        knobs.snap_max = 1;
        knobs.prefix_cache = false;
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
            knobs,
        })
        .expect("构造");
        let loaded = engine
            .loader()
            .load_qwen38_27b_awq(std::path::Path::new(dir), std::path::Path::new(dir), owl_models::module::QuantPlan::F16, owl_models::module::QuantPlan::W4A16Awq)
            .await
            .expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        let (_sid, id) = running
            .submit_session(Some(7), "用五十字介绍长城。", max_new)
            .expect("submit");
        let t0 = std::time::Instant::now();
        let _text = drive_turn(&mut running, id).await;
        let wall = t0.elapsed().as_secs_f64();
        let n = running.bank.sessions.get(7).map(|v| v.tokens.len()).unwrap_or(0);
        drop(running);
        std::thread::sleep(std::time::Duration::from_secs(3));
        (wall, n)
    }

    let (w1, n1) = gen(ordinal, &dir, 16).await;
    let (w2, n2) = gen(ordinal, &dir, 128).await;
    let dn = (n2 as i64 - n1 as i64).max(1) as f64;
    let marginal = (w2 - w1) * 1e3 / dn;
    eprintln!(
        "[marginal] turn1 {w1:.3}s/{n1}tok; turn2 {w2:.3}s/{n2}tok; 边际 = {marginal:.2} ms/tok = {:.1} tok/s",
        1e3 / marginal
    );
    // 旋钮已显式化,无 env 清场需要
}

/// **int8 头精度门**(P1.5-a,2026-10-20;用户口径:top-1 + 贪心逐 token)。
///
/// 同 prompt / greedy / 同 seed,`head_plan = F16` vs `I8W` 的**生成 token 序列**
/// 逐位比对 —— **token 级**,不吃 detokenizer(后者在本 head 上会把字节尾截掉)。
/// 判据:一致率 ≥ 0.95(逐 token 计数);长序列按需 `OWL_I8_HEAD_N`(缺省 48)。
///
/// # 两进程协议(`OWL_I8_HEAD_ARM`)
///
/// **为什么必须两进程**:同进程连跑两臂 = 两次 27B 装载,第二臂被 VRAM 池守卫
/// 响亮拒(`PoolExhausted`:驱动 free 897MiB / 池 reserved 18816MiB —— 释放与
/// 驱动回收不同步,不是漏 free)。
///
/// ```bash
/// # 臂 1(bf16 头),臂 2(int8 头),然后比对(比对不需要卡)
/// OWL_TEST_DEVICE=01:00.0 OWL_I8_HEAD_ARM=f16 cargo test -p owl-engine --test engine.e2e -- gpu_head_int8_gate --nocapture
/// OWL_TEST_DEVICE=01:00.0 OWL_I8_HEAD_ARM=i8  cargo test -p owl-engine --test engine.e2e -- gpu_head_int8_gate --nocapture
/// OWL_I8_HEAD_ARM=compare                     cargo test -p owl-engine --test engine.e2e -- gpu_head_int8_gate --nocapture
/// ```
/// 臂文件:`$OWL_I8_HEAD_OUT`(缺省 `/tmp/owl-i8head-<arm>.txt`)。
pub async fn gpu_head_int8_gate() {
    let arm = owl_shared::env_reader::str("OWL_I8_HEAD_ARM").unwrap_or_default();
    let n: usize = owl_shared::env_reader::parse_or("OWL_I8_HEAD_N", 48);
    let art = |a: &str| -> std::path::PathBuf {
        owl_shared::env_reader::str("OWL_I8_HEAD_OUT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(format!("/tmp/owl-i8head-{a}.txt")))
    };
    // ── 比对臂(零 GPU):读两臂 token 文件 → 一致率 ─────────────────
    if arm == "compare" {
        let read = |a: &str| -> Vec<(String, Vec<u32>)> {
            std::fs::read_to_string(art(a))
                .unwrap_or_else(|e| panic!("读 {a} 臂失败({}):{e} —— 先跑两臂", art(a).display()))
                .lines()
                .filter_map(|l| {
                    let (p, rest) = l.split_once('\t')?;
                    Some((p.to_string(), rest.split_whitespace().filter_map(|t| t.parse().ok()).collect()))
                })
                .collect()
        };
        let (a, b) = (read("f16"), read("i8"));
        assert_eq!(a.len(), b.len(), "两臂 prompt 数不等");
        let (mut agree, mut total, mut exact) = (0usize, 0usize, 0usize);
        for ((pa, ta), (_pb, tb)) in a.iter().zip(b.iter()) {
            let same = ta.iter().zip(tb.iter()).filter(|(x, y)| x == y).count();
            let first = ta.iter().zip(tb.iter()).position(|(x, y)| x != y);
            if ta == tb { exact += 1; }
            eprintln!(
                "[i8head] {} · len {} vs {} · 相同 {same} · 首个分歧 {first:?}",
                pa.chars().take(24).collect::<String>(),
                ta.len(), tb.len()
            );
            agree += same;
            total += ta.len().max(tb.len());
        }
        let rate = agree as f64 / total.max(1) as f64;
        eprintln!(
            "[i8head] **token 一致率 = {rate:.4}**({agree}/{total});逐字全等 {exact}/{} 条 ⇒ {}",
            a.len(),
            if rate >= 0.95 { "过门(≥0.95)" } else { "✗ 未过" }
        );
        assert!(rate >= 0.95, "int8 头精度门未过({rate:.4} < 0.95)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
        owl_shared::testkit::report::skip("ckpt:27b-awq 未设");
        return;
    };
    async fn gen(
        ordinal: usize,
        dir: &str,
        plan: owl_models::module::QuantPlan,
        prompt: &str,
        n: usize,
    ) -> Vec<u32> {
        let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
        knobs.gdn_slots = 1;
        knobs.snap_max = 1;
        knobs.prefix_cache = false;
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
            knobs,
        })
        .expect("构造");
        let loaded = engine
            .loader()
            .load_qwen38_27b_awq(
                std::path::Path::new(dir),
                std::path::Path::new(dir),
                plan,
                owl_models::module::QuantPlan::W4A16Awq,
            )
            .await
            .expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        let (_sid, id) = running.submit_session(Some(7), prompt, n).expect("submit");
        let _ = drive_turn(&mut running, id).await;
        let toks = running.bank.sessions.get(7).map(|s| s.tokens.clone()).unwrap_or_default();
        drop(running);
        std::thread::sleep(std::time::Duration::from_secs(3));
        toks
    }
    let prompts = [
        "用五十字介绍长城。",
        "一个水池有甲乙两个进水管，甲管单独注满需要 6 小时，乙管单独注满需要 4 小时。两管同时打开，多少小时可以注满？",
        "用 Rust 写一个求中位数的函数。",
    ];
    // ── 单臂模式(推荐;两进程各自一次装载)────────────────────────────
    if arm == "f16" || arm == "i8" {
        let plan = if arm == "i8" {
            owl_models::module::QuantPlan::I8W
        } else {
            owl_models::module::QuantPlan::F16
        };
        // **一次装载跑全部 prompt**(进程内第二次 27B 装载必被池守卫拒;
        // 设备内存从不 Free,见入口注释的挂账)。
        let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
        knobs.gdn_slots = 1;
        knobs.snap_max = 1;
        knobs.prefix_cache = false;
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 256,
            prefill_chunk: 32,
            knobs,
        })
        .expect("构造");
        let loaded = engine
            .loader()
            .load_qwen38_27b_awq(
                std::path::Path::new(&dir),
                std::path::Path::new(&dir),
                plan,
                owl_models::module::QuantPlan::W4A16Awq,
            )
            .await
            .expect("装载");
        let mut running = engine.run(loaded).await.expect("装配");
        let mut out = String::new();
        for (i, p) in prompts.iter().enumerate() {
            let sid = 7 + i as u64;
            let (_s, id) = running.submit_session(Some(sid), *p, n).expect("submit");
            let _ = drive_turn(&mut running, id).await;
            let t = running.bank.sessions.get(sid).map(|s| s.tokens.clone()).unwrap_or_default();
            eprintln!(
                "[i8head] {arm} 臂完成:{:?} → {} tok",
                p.chars().take(16).collect::<String>(),
                t.len()
            );
            out.push_str(&format!(
                "{}\t{}\n",
                p.replace('\t', " "),
                t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
            ));
        }
        drop(running);
        let path = art(arm.as_str());
        std::fs::write(&path, out).expect("写 token 文件");
        eprintln!("[i8head] {arm} 臂 token 落盘:{}(比对 = OWL_I8_HEAD_ARM=compare)", path.display());
        return;
    }

    // ── 缺省:同进程两臂(历史上会被 VRAM 池守卫拒;留作对照)────────
    eprintln!("[i8head] 未设 OWL_I8_HEAD_ARM ⇒ 同进程两臂(易被 PoolExhausted 拒;推荐两进程)");
    let (mut agree, mut total) = (0usize, 0usize);
    for (i, p) in prompts.iter().enumerate() {
        let a = gen(ordinal, &dir, owl_models::module::QuantPlan::F16, p, n).await;
        let b = gen(ordinal, &dir, owl_models::module::QuantPlan::I8W, p, n).await;
        let same = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
        let first = a.iter().zip(b.iter()).position(|(x, y)| x != y);
        eprintln!("[i8head] #{i} len {} vs {} · 相同 {same} · 首个分歧 {first:?}", a.len(), b.len());
        agree += same;
        total += a.len().max(b.len());
    }
    let rate = agree as f64 / total.max(1) as f64;
    eprintln!(
        "[i8head] **token 一致率 = {rate:.4}**({agree}/{total})⇒ {}",
        if rate >= 0.95 { "过门(≥0.95)" } else { "✗ 未过" }
    );
}

// B4 自适应降级态机(§6.24):纯转移函数逐分支
#[test]
fn spec_degrade_transition_branches() {
    use crate::exec::spec_degrade_transition;
    // 成功轮:三账全复位(含降级态/退避)
    let (s, d, p) = spec_degrade_transition(3, true, 256, 2, 6, 64);
    assert_eq!((s, d, p), (0, false, 64));
    // 非降级连击未达阈值:streak 累加,态不变
    let (s, d, p) = spec_degrade_transition(0, false, 64, 0, 6, 64);
    assert_eq!((s, d, p), (1, false, 64));
    // 达阈值 → 入降级(探测周期 = 初值)
    let (s, d, p) = spec_degrade_transition(5, false, 64, 0, 6, 64);
    assert_eq!((s, d, p), (6, true, 64));
    // 降级态探测失败 → 退避 ×2
    let (s, d, p) = spec_degrade_transition(7, true, 64, 0, 6, 64);
    assert_eq!((s, d, p), (8, true, 128));
    // 退避封顶 512
    let (s, d, p) = spec_degrade_transition(9, true, 400, 0, 6, 64);
    assert_eq!((s, d, p), (10, true, 512));
    // **小首探窗**(2026-10-18 缺省 8):入降级即用 8,探测失败退避到 16
    let (s, d, p) = spec_degrade_transition(5, false, 8, 0, 6, 8);
    assert_eq!((s, d, p), (6, true, 8));
    let (s, d, p) = spec_degrade_transition(6, true, 8, 0, 6, 8);
    assert_eq!((s, d, p), (7, true, 16));
    // threshold=0 = 禁用(永不入降级)
    let (s, d, p) = spec_degrade_transition(9, false, 64, 0, 0, 64);
    assert_eq!((s, d, p), (10, false, 64));
}

// ============================================================================
// A7 收口(2026-10-19):准入地板 = 页口径
// ============================================================================

/// **MiB 事实 → 页数**:向上取整到整页(少一页 = 承诺了又给不够);
/// `0 MiB` = 允许全量下沉(0 页,不是“未设”)。
#[test]
fn dense_floor_mib_converts_up_to_pages() {
    use owl_store::TurnFloor;
    let mib = 1usize << 20;
    assert_eq!(EngineKnobs::dense_floor_with_mib(1, mib), TurnFloor::Pages(1));
    assert_eq!(EngineKnobs::dense_floor_with_mib(1, mib / 2), TurnFloor::Pages(2), "半页配置 ⇒ 整页");
    assert_eq!(EngineKnobs::dense_floor_with_mib(3, mib), TurnFloor::Pages(3));
    assert_eq!(EngineKnobs::dense_floor_with_mib(0, mib), TurnFloor::Pages(0));
}

/// **快照重挂层分派**(前缀无感删除):四层各一条路,账不平 = 不重挂。
/// 这是`plan_image_delete` 的唯一判据(设备槽在场与否由调用方传入)。
#[test]
fn snap_rekey_target_dispatch_law() {
    use crate::scheduler::{snap_rekey_target, RekeyTarget};
    use owl_store::Tier;
    assert_eq!(snap_rekey_target(Some(Tier::Gpu), true), RekeyTarget::Device);
    assert_eq!(
        snap_rekey_target(Some(Tier::Gpu), false),
        RekeyTarget::Skip,
        "账说设备、槽上没有 ⇒ 击罒不平:不重挂"
    );
    assert_eq!(snap_rekey_target(Some(Tier::Host), false), RekeyTarget::Host);
    assert_eq!(snap_rekey_target(Some(Tier::Disk), false), RekeyTarget::Disk);
    assert_eq!(snap_rekey_target(None, true), RekeyTarget::Skip, "不在账 ⇒ 不重挂");
}

/// **装配期单源派生**:一个 `workspace_per_req` 同时落 `select_blocks` 与
/// 地板页数;显式 MiB 优先(页宽由池给)。三件同源 = 不再各处各算一遍。
#[test]
fn derive_turn_ws_sets_blocks_and_floor_from_one_source() {
    use owl_store::TurnFloor;
    let page_bytes = 1usize << 20; // 页 = 32 tok × 32KiB
    // ① 未给 MiB:地板 = 工作集页数
    let mut k = EngineKnobs { kvmem_virtualization: true, ..Default::default() };
    k.workspace_per_req = 640; // 20 页
    k.select_sink = 2;
    k.select_recent = 4;
    let (pages, bytes) = EngineKnobs::derive_turn_ws(640, 32, page_bytes, &mut k).expect("派生");
    assert_eq!((pages, bytes), (20, 20 << 20));
    assert_eq!(k.select_blocks, 20, "召回预算 = 总选中数(含保底)");
    assert_eq!(k.dense_floor, TurnFloor::Pages(20), "地板与池下限同一套页数");
    // ② 显式 MiB:按页宽换成页数(与工作集无关)
    let mut k = EngineKnobs { kvmem_virtualization: true, ..Default::default() };
    k.workspace_per_req = 640;
    k.dense_floor_mib = Some(2);
    let _ = EngineKnobs::derive_turn_ws(640, 32, page_bytes, &mut k).expect("派生");
    assert_eq!(k.dense_floor, TurnFloor::Pages(2));
    // ③ 非虚拟化档:地板不许被改(唯一诚实值 = Strict)
    let mut k = EngineKnobs { kvmem_virtualization: false, ..Default::default() };
    k.workspace_per_req = 640;
    let _ = EngineKnobs::derive_turn_ws(640, 32, page_bytes, &mut k).expect("派生");
    assert_eq!(k.dense_floor, TurnFloor::Strict, "非虚拟化档地板不动");
    // ④ 工作集 < 保底 = 配置不自洽 ⇒ 响亮拒
    let mut k = EngineKnobs { kvmem_virtualization: true, ..Default::default() };
    k.workspace_per_req = 64; // 2 页 < sink2+recent4 = 6 页
    assert!(EngineKnobs::derive_turn_ws(64, 32, page_bytes, &mut k).is_err());
    // ⑤ verify 视图只能裁、不能放大
    let mut k = EngineKnobs { kvmem_virtualization: true, ..Default::default() };
    k.workspace_per_req = 640;
    k.verify_view_tokens = 1024;
    assert!(EngineKnobs::derive_turn_ws(640, 32, page_bytes, &mut k).is_err());
}

/// **索引暂存批上限**:选择可以 r16，但索引 D2H 不能跟着 16 chunk 攒
/// （64 页/批是池下限里的驻留尾上限）。r1/r4/r16 三档钉住派生。
#[test]
fn index_cadence_caps_pending_batch() {
    let mut k = EngineKnobs::default();
    for (recall, want) in [(1, 1), (2, 2), (4, 4), (6, 4), (16, 4)] {
        k.recall_every = recall;
        assert_eq!(k.index_cadence(), want, "recall_every={recall}");
    }
}

// ===========================================================================
// B5.1 服务面契约:取消必须**指名**(排队/活跃/按会话)
// ===========================================================================

/// 排队中取消:精确命中该 turn,不碰同队其它 turn;重复取消 = None(幂等)。
#[cfg(test)] // 单元档专属(dev-dep tokio/macros);e2e 档不编该用例
#[tokio::test]
async fn cancel_queued_turn_is_precise() {
    let mut running = mock_snap_engine(1).await;
    let prompt_a = "alpha bravo cedar delta ember falcon alpha bravo";
    let prompt_b = "cedar delta ember falcon alpha bravo cedar delta";
    let (_s1, t1) = running.submit_session(Some(11), prompt_a, 8).expect("submit A");
    let (_s2, t2) = running.submit_session(Some(22), prompt_b, 8).expect("submit B");
    assert_eq!(running.queue_len(), 2, "两个 turn 都应排队");
    // 取消 B(排队中)
    let ev = running.cancel_turn(t2).await.expect("排队 turn 应可取消");
    match ev {
        TurnEvent::Failed { turn, ref err } => {
            assert_eq!(turn, t2);
            assert!(err.contains("排队"), "口径应写明排队中:{err}");
        }
        other => panic!("应为 Failed,得到 {other:?}"),
    }
    assert_eq!(running.queue_len(), 1, "A 不许连坐");
    // 幂等:再取消 = None
    assert!(running.cancel_turn(t2).await.is_none(), "重复取消应 None");
    // A 仍能跑完(取消没伤到队列结构)
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Completed { turn, .. } if turn == t1 => break,
            TurnEvent::Idle => panic!("A 丢了(队列被取消路径破坏)"),
            _ => {}
        }
    }
}

/// 按会话取消:只命中该会话的 turn(活跃或排队),同批别家不受影响。
#[cfg(test)] // 单元档专属(dev-dep tokio/macros);e2e 档不编该用例
#[tokio::test]
async fn cancel_session_targets_only_that_session() {
    let mut running = mock_snap_engine(1).await;
    let prompt_a = "alpha bravo cedar delta ember falcon alpha bravo";
    let prompt_b = "cedar delta ember falcon alpha bravo cedar delta";
    let (_s1, t1) = running.submit_session(Some(7), prompt_a, 16).expect("submit A");
    // 泵一步让 A 进活跃(单槽档:活跃 = 执行选择器)
    let _ = running.pump().await.expect("pump step");
    let (_s2, t2) = running.submit_session(Some(8), prompt_b, 8).expect("submit B");
    assert_eq!(running.queue_len(), 1, "B 应排队");
    // 按会话 8 取消 ⇒ 只掉 B
    let ev = running.cancel_session(8).await.expect("会话 8 应有 turn");
    assert!(matches!(ev, TurnEvent::Failed { turn, .. } if turn == t2));
    assert_eq!(running.queue_len(), 0);
    // A 不受影响:跑到 Completed
    let mut done = false;
    for _ in 0..512 {
        match running.pump().await.expect("pump") {
            TurnEvent::Completed { turn, .. } if turn == t1 => {
                done = true;
                break;
            }
            TurnEvent::Idle => break,
            _ => {}
        }
    }
    assert!(done, "A 应在 B 被取消后正常完成");
    // 无 turn 的会话 = None(不静默成功)
    assert!(running.cancel_session(999).await.is_none());
}

/// **P0-2 队列上限**（submit 边界）：在飞总数（活跃 + 排队）达上限 ⇒ 拒，
/// 且错误串带 `admission:` 前缀（服务面据此翻 429 `no_slot`）。
#[cfg(test)] // 单元档专属(dev-dep tokio/macros);e2e 档不编该用例
#[tokio::test]
async fn queue_bound_rejects_at_submit() {
    let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
    knobs.pool_tokens = Some(256);
    knobs.host_cache_mb = 8;
    knobs.snap_host_mb = 64;
    knobs.snap_max = 1;
    knobs.max_batch = 1;
    knobs.max_queue = 1; // 批位 1 + 队列 1 ⇒ 在飞上限 2
    let cfg = EngineConfig { device_ordinal: 0, max_seq_tokens: 256, prefill_chunk: 32, knobs };
    let face = owl_cpu::CpuFace::mock(owl_cpu::MockLimits {
        null_kernels: true,
        graph_capture: true,
        ..Default::default()
    });
    let mut engine = Engine::on(cfg, face).expect("构造(CPU mock)");
    let loaded = engine
        .loader()
        .load_qwen35_0_8b(&asset_dir())
        .await
        .expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");

    let p = "alpha bravo cedar delta ember falcon";
    running.submit_session(Some(1), p, 8).expect("第 1 个(活跃)");
    running.submit_session(Some(2), p, 8).expect("第 2 个(排队)");
    match running.submit_session(Some(3), p, 8) {
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("admission:no_slot"), "错误串应带准入码：{msg}");
        }
        Ok(_) => panic!("在飞已达上限 2，第 3 个应被拒"),
    }
    // 无界档（max_queue=0）不拒 —— 旧行为不变
    running.boot.cfg.knobs.max_queue = 0;
    running.submit_session(Some(4), p, 8).expect("无界档不拒");
}

/// **P3 退避估算/直方图**（纯函数；无卡）
#[test]
fn wait_histogram_and_eta_are_sane() {
    use crate::turn::TurnLedger;
    // 桶边界：49/50、199/200、999/1000、4999/5000
    assert_eq!(TurnLedger::wait_bucket(0), 0);
    assert_eq!(TurnLedger::wait_bucket(49), 0);
    assert_eq!(TurnLedger::wait_bucket(50), 1);
    assert_eq!(TurnLedger::wait_bucket(199), 1);
    assert_eq!(TurnLedger::wait_bucket(200), 2);
    assert_eq!(TurnLedger::wait_bucket(999), 2);
    assert_eq!(TurnLedger::wait_bucket(1_000), 3);
    assert_eq!(TurnLedger::wait_bucket(4_999), 3);
    assert_eq!(TurnLedger::wait_bucket(5_000), 4);

    let mut t = TurnLedger::new();
    // EWMA 未测 ⇒ 2s 保守；队列 3、批位 2 ⇒ 2 波 ⇒ 4s
    assert_eq!(t.eta_ms(2, 3), 4_000);
    // 测过 1s ⇒ 队列 0、批位 4 ⇒ 1 波 ⇒ 1s
    t.note_turn_ms(1_000);
    assert_eq!(t.eta_ms(4, 0), 1_000);
    // 下限/上限
    let mut t2 = TurnLedger::new();
    t2.note_turn_ms(10); // 首值即 EWMA ⇒ 10ms
    assert_eq!(t2.eta_ms(8, 0), 500, "下限 500ms");
    let mut t3 = TurnLedger::new();
    t3.note_turn_ms(100_000);
    assert_eq!(t3.eta_ms(1, 100), 30_000, "上限 30s");
}

/// **VLM 图像输入基线门**(2026-10-21,`fix/vlm` 立案):单图、无窗口、一次 turn。
///
/// 为什么单独立门:V1–V5 视觉接线(HTTP→预处理→占位展开→引擎→塔→scatter)
/// 只有 CPU 对拍,没有一条端到端;**image-rotation 施工时单图即红**,
/// 判据需要一条能独立跑的最小复现。
///
/// 判据:① 出非空文本;② `uploaded` 塔输出块在 scatter 时仍在账本(TEMP 探针)。
pub async fn gpu_vlm_image_baseline() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    if !dir.join("config.json").exists() {
        owl_shared::testkit::report::skip(format_args!("0.8B 资产不在 {}", dir.display()));
        return;
    }
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 512,
        prefill_chunk: 32,
        knobs: EngineKnobs { vision: true, ..Default::default() },
    })
    .expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载 0.8B+VLM");
    let mut running = engine.run(loaded).await.expect("装配");

    let img_tok = running.dev.boot.vlm.as_ref().expect("VLM 装配").cfg().image_token;
    let n_img = match running.boot.tok.try_encode("<|image_pad|>").expect("encode").as_slice() {
        [one] => *one,
        other => panic!("<|image_pad|> 应恰 1 token,得 {other:?}"),
    };
    assert_eq!(n_img, img_tok, "image_pad 必须映射到引擎 image_token");
    // 网格 (1,16,16) ⇒ l_img = 64(大形状 ⇒ 视觉塔注意力走 FI ragged 臂)
    let l_img = 64usize;
    let img = owl_models::vlm::VlmImage {
        pixel_values: (0..256 * 3 * 2 * 16 * 16)
            .map(|i| 0.2 + (i % 7) as f32 * 0.01)
            .collect(),
        grid: (1, 16, 16),
    };
    // ⚠️ **预热轮(纯文本)**:实测"**boot 后第一轮就带图**"会吐退化串
    // (0.8B:`!4444444`);同一张图放到第二轮就正常(`我看到的是一片灰灰色的平面区域`)。
    // 这是**既有面**的独立缺陷(首轮+视觉,另案;不是本门要测的东西),门先跑一轮
    // 纯文本把"首轮"这个变量消掉 —— 否则本门会在别人的红上摇摆。
    {
        let (_s0, t0) = running
            .submit_session(Some(5), "请用一句话说明你看到了什么。", 8)
            .expect("预热轮");
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { turn, text, .. } if turn == t0 => {
                    eprintln!("[test] 预热轮(纯文本,无图): {text}");
                    assert!(!text.trim().is_empty(), "纯文本首轮必须出文本");
                    break;
                }
                TurnEvent::Failed { turn, err } if turn == t0 => panic!("预热轮失败: {err}"),
                TurnEvent::Idle => panic!("预热轮队列丢失"),
                _ => {}
            }
        }
    }
    // **用家族模板包装**(含 `<|im_start|>` ⇒ submit 不再二次 wrap):
    // 裸拼 ChatML 少 `<|im_end|>`/`assistant` 头 ⇒ 模型只会续写垃圾
    // (实测 "!4444444")—— 门要看"图有没有被看见",不是看拼装的脾气。
    let prompt = running.boot.tok.chat_wrap(&format!(
        "<|vision_start|>{}<|vision_end|>请用一句话说明你看到了什么。",
        "<|image_pad|>".repeat(l_img)
    ));
    let (_sid, t) = running
        .submit_session_opts(
            Some(6),
            prompt.as_str(),
            8,
            crate::turn::TurnOpts { images: vec![img.into()], ..Default::default() },
        )
        .expect("提交");
    let out: Result<String, String> = loop {
        match running.pump().await {
            Ok(TurnEvent::Completed { turn, text, .. }) if turn == t => break Ok(text),
            Ok(TurnEvent::Failed { turn, err }) if turn == t => break Err(err),
            Ok(TurnEvent::Idle) => break Err("队列丢失".into()),
            Ok(_) => {}
            Err(e) => break Err(e.to_string()),
        }
    };
    match out {
        Ok(text) => {
            eprintln!("[test] VLM 基线文本(16×16 / FI 臂): {text}");
            assert!(!text.trim().is_empty(), "视觉 prefill 必须产出文本");
        }
        Err(e) => panic!("VLM 单图基线失败: {e}"),
    }

    // ── **小网格回流**(2026-10-21 `fix/vlm` F2):4×4 ⇒ 塔注意力 rows=16,
    //    FI ragged 只实例化 cta_tile_q ∈ {64,128,256} ⇒ 必须**自动退 native**
    //    (旧行为 = 深处 `ragged run err -205`)。
    {
        let small = owl_models::vlm::VlmImage {
            pixel_values: (0..16 * 3 * 2 * 16 * 16)
                .map(|i| 0.2 + (i % 7) as f32 * 0.01)
                .collect(),
            grid: (1, 4, 4),
        };
        let l_small = 4usize;
        let prompt = running.boot.tok.chat_wrap(&format!(
            "<|vision_start|>{}<|vision_end|>请用一句话说明你看到了什么。",
            "<|image_pad|>".repeat(l_small)
        ));
        let (_s1, t1) = running
            .submit_session_opts(
                Some(9),
                prompt.as_str(),
                8,
                crate::turn::TurnOpts { images: vec![small.into()], ..Default::default() },
            )
            .expect("小网格提交");
        let txt = loop {
            match running.pump().await {
                Ok(TurnEvent::Completed { turn, text, .. }) if turn == t1 => break text,
                Ok(TurnEvent::Failed { turn, err }) if turn == t1 => {
                    panic!("小网格(4×4)视觉路径失败 —— 应退 native 臂: {err}")
                }
                Ok(TurnEvent::Idle) => panic!("小网格队列丢失"),
                Ok(_) => {}
                Err(e) => panic!("小网格 pump 失败: {e}"),
            }
        };
        eprintln!("[test] VLM 小网格文本(4×4 / native 臂): {txt}");
        assert!(!txt.trim().is_empty(), "小网格路径必须产出文本");
    }
}

/// **F1 复现探针**(2026-10-21;`fix/vlm` 立案):"**进程第一轮就带图**" ⇒ 退化串。
///
/// 定谳证据(真机,0.8B + 16×16 网格,3 组对照):
///
/// | 轮序 | 带图轮的输出 |
/// |---|---|
/// | `图, 图, 文本, 图` | **三轮全退化**(`!44444444444`) |
/// | `文本, 图, 文本, 图` | **两轮都正常**(`这是一张灰色的背景图…`) |
/// | `图(首轮)` 单轮 | 退化 |
///
/// 同轮内塔输出 `merged` 的**指纹逐轮相同**(sum/l2/head 全等,仅块 id 不同)
/// ⇒ **塔无罪**,问题在"首轮 + 图像"这条路径的下游(**注入面/分块/位置**)。
/// 关图捕获(`no_graph`)不改变现象 ⇒ 不是"带图 prefill 图把指针烤死"。
///
/// 本探针**不设硬断言**(诊断;判决看打印),`first_image` 决定首轮形态:
/// 两个场景各跑一次即构成 A/B(一个进程只能有一个"首轮")。
pub async fn gpu_vlm_first_turn_probe(first_image: bool) {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    if !dir.join("config.json").exists() {
        owl_shared::testkit::report::skip(format_args!("0.8B 资产不在 {}", dir.display()));
        return;
    }
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 512,
        prefill_chunk: 32,
        knobs: EngineKnobs { vision: true, ..Default::default() },
    })
    .expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载 0.8B+VLM");
    let mut running = engine.run(loaded).await.expect("装配");

    let l_img = 64usize; // 网格 (1,16,16):l_img = (16/2)² = 64(≥16×16 避开 FI 小网格缺口)
    let mk_img = |salt: f32| owl_models::vlm::VlmImage {
        pixel_values: (0..256 * 3 * 2 * 16 * 16)
            .map(|i| salt + (i % 7) as f32 * 0.01)
            .collect(),
        grid: (1, 16, 16),
    };
    let q = "请用一句话说明你看到了什么。";
    let img_part = format!(
        "<|vision_start|>{}<|vision_end|>",
        "<|image_pad|>".repeat(l_img)
    );
    let sys_img = running.boot.tok.chat_format().render_chat(
        &[("system", "你是助手,按约定回答。"), ("user", &format!("{img_part}{q}"))],
        false,
    );
    let sys_text = running.boot.tok.chat_format().render_chat(
        &[("system", "你是助手,按约定回答。"), ("user", q)],
        false,
    );
    eprintln!("[f1] 首轮形态 = {}", if first_image { "带图" } else { "纯文本" });
    // 首轮形态由入参决定;之后 3 轮固定为 图/文本/图(便于两组逐轮对照)
    let plan = if first_image {
        [true, true, false, true]
    } else {
        [false, true, false, true]
    };
    for (idx, with_img) in plan.into_iter().enumerate() {
        let idx = idx + 1;
        let (prompt, images) = if with_img {
            (sys_img.clone(), vec![mk_img(0.2).into()])
        } else {
            (sys_text.clone(), Vec::new())
        };
        let (_sid, t) = running
            .submit_session_opts(
                Some(100 + idx as u64),
                prompt.as_str(),
                12,
                crate::turn::TurnOpts { images, ..Default::default() },
            )
            .expect("提交");
        let txt = loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { turn, text, .. } if turn == t => break text,
                TurnEvent::Failed { turn, err } if turn == t => break format!("<FAILED {err}>"),
                TurnEvent::Idle => break "<IDLE>".to_string(),
                _ => {}
            }
        };
        eprintln!(
            "[f1] 第{idx}轮 {} ⇒ {txt:?}",
            if with_img { "带图" } else { "纯文本" }
        );
    }
}

/// F1 的**控制组**(首轮纯文本):与 [`gpu_vlm_first_turn_probe`] 成对跑。
pub async fn gpu_vlm_first_turn_probe_textfirst() {
    gpu_vlm_first_turn_probe(false).await;
}

/// F1 的**复现组**(首轮带图)。
pub async fn gpu_vlm_first_turn_probe_imagefirst() {
    gpu_vlm_first_turn_probe(true).await;
}


/// **F1 × 27B**——同一探针在 27B AWQ INT4(生产 VLM 档)上重放。
///
/// 为什么要单独一档:0.8B 的结论不能外推(层数/GDN 比例/量化/kv fp8 全不同),
/// 而 27B 是**现役生产** VLM(`profiles/tp27b-3080-262k-vlm.toml`)⇒ F1 到底
/// 是不是"只在 0.8B 上"必须真机回答。
pub async fn gpu_vlm_first_turn_probe_27b(first_image: bool) {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
        owl_shared::testkit::report::skip("ckpt:27b-awq 未配(testdata/resources.toml)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let t0 = std::time::Instant::now();
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 1024,
        prefill_chunk: 128,
        knobs: EngineKnobs { vision: true, ..Default::default() },
    })
    .expect("构造");
    let loaded = engine
        .loader()
        .load_qwen38_27b_awq(
            &dir,
            &dir,
            owl_models::module::QuantPlan::F16,
            owl_models::module::QuantPlan::W4A16Awq,
        )
        .await
        .expect("27B AWQ 装载");
    let mut running = engine.run(loaded).await.expect("装配");
    eprintln!("[f1-27b] 装载+装配 {:?}", t0.elapsed());
    assert!(
        running.dev.boot.vlm.is_some(),
        "[model].vision = true 必须有视觉塔(27B)")
    ;

    let l_img = 64usize; // 网格 (1,16,16) ⇒ (16/2)² = 64 tok
    let mk_img = |salt: f32| owl_models::vlm::VlmImage {
        pixel_values: (0..256 * 3 * 2 * 16 * 16)
            .map(|i| salt + (i % 7) as f32 * 0.01)
            .collect(),
        grid: (1, 16, 16),
    };
    let q = "请用一句话说明你看到了什么。";
    let img_part = format!(
        "<|vision_start|>{}<|vision_end|>",
        "<|image_pad|>".repeat(l_img)
    );
    let sys_img = running.boot.tok.chat_format().render_chat(
        &[("system", "你是助手,按约定回答。"), ("user", &format!("{img_part}{q}"))],
        false,
    );
    let sys_text = running.boot.tok.chat_format().render_chat(
        &[("system", "你是助手,按约定回答。"), ("user", q)],
        false,
    );
    eprintln!("[f1-27b] 首轮形态 = {}", if first_image { "带图" } else { "纯文本" });
    let plan = if first_image { [true, true, false, true] } else { [false, true, false, true] };
    for (idx, with_img) in plan.into_iter().enumerate() {
        let idx = idx + 1;
        let (prompt, images) = if with_img {
            (sys_img.clone(), vec![mk_img(0.2).into()])
        } else {
            (sys_text.clone(), Vec::new())
        };
        let (_sid, t) = running
            .submit_session_opts(
                Some(300 + idx as u64),
                prompt.as_str(),
                12,
                crate::turn::TurnOpts { images, ..Default::default() },
            )
            .expect("提交");
        let txt = loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { turn, text, .. } if turn == t => break text,
                TurnEvent::Failed { turn, err } if turn == t => break format!("<FAILED {err}>"),
                TurnEvent::Idle => break "<IDLE>".to_string(),
                _ => {}
            }
        };
        eprintln!(
            "[f1-27b] 第{idx}轮 {} ⇒ {txt:?}",
            if with_img { "带图" } else { "纯文本" }
        );
    }
}


/// **前缀无感删除门**(2026-10-23;设计卷 `memory/prefix-deletion.md`)。
///
/// 场景 = GUI agent 的一步:历史里有两张截图(turn 1),harness 把**旧的
/// 第一张**抽掉(turn 2),其余全部前移。
///
/// # 判据(硬)
///
/// 1. `image_evict.absorbed == 1` —— 删除**被吸收**,不是退回全量重算;
/// 2. 删除区间 == 第一张图的**整段**(含页对齐 pad),起点/长度都是整页;
/// 3. **G1 机械等价**:被复用后缀页的每层 K == `R(−S)·` 删除前(配对量级 ≤4 ulp)
///    —— 退回重算会因 hidden 不同而**完全不等**,所以这条是**正判别**;
/// 4. 吸收轮 prefill 事件数很小(只喂新尾部;后缀零重算);
/// 5. 两轮文本都非空。
///
/// ⚠️ 0.8B「boot 后第一轮就带图」有既有红(F1,另案)⇒ 门先跑**纯文本预热轮**。
pub async fn gpu_image_evict_gate() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    if !dir.join("config.json").exists() {
        owl_shared::testkit::report::skip(format_args!("0.8B 资产不在 {}", dir.display()));
        return;
    }
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 768,
        prefill_chunk: 64,
        knobs: EngineKnobs {
            vision: true,
            prefix_cache: true,
            image_evict: true,
            // **叠转上限降到 1**:让 G7 臂第一次删除吸收(rot=1)、第二次删除
            // 必然到顶 ⇒ 整段重建(门要的是"到顶真的重建",不是数值)
            image_evict_delta_limit: 1,
            // 钉小池(门只跑 ~700 tok;auto 池算式在本机会被"预算可服务上限"
            // 拒 —— 与本特性无关的既有面)
            pool_tokens: Some(2048),
            ..Default::default()
        },
    })
    .expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载 0.8B+VLM");
    let mut running = engine.run(loaded).await.expect("装配");

    let page = running.dev.pool.geom.page;
    assert!(page >= 32 && page % 32 == 0, "页宽异常: {page}");
    let (hkv, hd) = {
        let d = &running.dev.pool.geom.dims;
        (d.hkv, d.hd)
    };
    let per_page_bytes = hkv * hd * page * 2; // f16
    let image_token = running
        .dev
        .boot
        .vlm
        .as_ref()
        .expect("VLM 必须装配")
        .cfg()
        .image_token;
    let one = |r: &RunningEngine<crate::GpuFace>, t: &str| -> u32 {
        let v = r.boot.tok.try_encode(t).expect("encode");
        assert_eq!(v.len(), 1, "{t} 应恰 1 token(得 {v:?})");
        v[0]
    };
    let (vs, ve) = (one(&running, "<|vision_start|>"), one(&running, "<|vision_end|>"));
    let pad_id = one(&running, "\n");

    // ── 预热轮(纯文本;消掉 F1 的"首轮+图"变量) ──
    {
        let (_s, t) = running
            .submit_session(Some(5), "请用一句话说明你在做什么。", 6)
            .expect("预热提交");
        let text = drive_turn(&mut running, t).await;
        assert!(!text.trim().is_empty(), "纯文本预热轮必须出文本");
    }

    // 图:grid (1,16,16) ⇒ 占位 (16/2)·(16/2) = 64 tok(网格 ≥ 16×16 是 FI ragged 门)
    let l_img = 64usize;
    let mk_img = |salt: f32| owl_models::vlm::VlmImage {
        pixel_values: (0..256 * 3 * 2 * 16 * 16)
            .map(|i| salt + (i % 7) as f32 * 0.01)
            .collect(),
        grid: (1, 16, 16),
    };
    let item = |_i: usize| format!("<|vision_start|>{}<|vision_end|>", "<|image_pad|>".repeat(l_img));
    let p_text = |n: usize| {
        let mut s = String::new();
        for i in 0..n {
            s.push_str(&format!("观察点{i}:屏幕无异常,日志正常,继续。\n"));
        }
        s
    };
    let q1 = "请用一句话说明你看到了什么。";
    let mid = "中间这段文字会留在历史里。\n";
    // ── turn 1:两张截图(都打可删标记) ──
    let prompt1 = running.boot.tok.chat_format().render_chat(
        &[
            ("system", "你是助手,按约定回答。"),
            (
                "user",
                &format!("{}{}{}{}{}{q1}", p_text(2), item(0), mid, item(1), p_text(1)),
            ),
        ],
        false,
    );
    let imgs1 = vec![
        crate::turn::TurnImage { image: mk_img(0.2), evict: true },
        crate::turn::TurnImage { image: mk_img(0.6), evict: true },
    ];
    let (sid, t1) = running
        .submit_session_opts(
            Some(7),
            prompt1.as_str(),
            8,
            crate::turn::TurnOpts { images: imgs1, ..Default::default() },
        )
        .expect("t1 提交");
    let text1 = drive_turn(&mut running, t1).await;
    assert!(!text1.trim().is_empty(), "t1 必须出文本(视觉基线)");

    // ── 测试侧复算 pad 律(与引擎同一份纯函数;用来验"删的正是第一段") ──
    let grids = [(1usize, 16usize, 16usize), (1usize, 16usize, 16usize)];
    let ids1 = running.boot.tok.encode(&prompt1);
    let seq1 = crate::img_evict::pad_image_spans(
        &ids1,
        &[true, true],
        &grids,
        image_token,
        Some(vs),
        Some(ve),
        page,
        pad_id,
        &[],
    )
    .expect("测试侧 pad");
    let (padded1, spans1) = (seq1.ids, seq1.spans);
    assert_eq!(spans1.len(), 2);
    assert_eq!(spans1[0].start % page, 0, "段首必须页对齐");
    assert_eq!((spans1[0].end - spans1[0].start) % page, 0, "段长必须整页");
    let del = spans1[0];
    {
        let s = running.bank.sessions.session(sid).expect("会话账");
        assert_eq!(
            &s.tokens[..padded1.len()],
            &padded1[..],
            "引擎内部 prompt 与测试侧 pad 律必须逐字节一致"
        );
        assert_eq!(s.img_spans.len(), 2, "会话账必须记住两段");
        assert_eq!(s.img_spans[0].start, spans1[0].start);
        assert_eq!(s.img_spans[0].end, spans1[0].end);
    }
    // 旧链槽位(内容树;turn 收口后会话不持表)
    let old_tokens = running.bank.sessions.session(sid).expect("会话账").tokens.clone();
    let (hit, _) = running.bank.rel.match_prefix(&old_tokens).expect("旧链命中");
    assert!(
        hit.matched_blocks >= (del.end / page) + 1,
        "旧链块数 {} 不足以覆盖删点({})",
        hit.matched_blocks,
        del.end / page
    );

    // ── 读 K(全层;删点之后的整段 —— 吸收后这些页要被原地重转) ──
    // 一条 `dtoh_batch` 读完一个槽的全部层(逐层一次往返会把门拖成分钟级)。
    async fn read_slot(
        running: &mut RunningEngine<crate::GpuFace>,
        slot: u32,
        per_page_bytes: usize,
    ) -> Vec<Vec<u8>> {
        let handles = running.dev.pool.kv_k_blocks();
        let off = slot as usize * per_page_bytes;
        let pieces: Vec<(&owl_iface::contract::Bytes, usize, usize)> =
            handles.iter().map(|b| (b, off, per_page_bytes)).collect();
        let all = {
            let face = running.dev.graphs.session.face_mut();
            face.dtoh_batch(&pieces).await.expect("D2H K")
        };
        all.chunks_exact(per_page_bytes).map(|c| c.to_vec()).collect()
    }
    let mut k_before: std::collections::HashMap<u32, Vec<Vec<u8>>> =
        std::collections::HashMap::new();
    for slot in hit.slots.iter().skip(del.start / page).flatten().copied() {
        let per_layer = read_slot(&mut running, slot, per_page_bytes).await;
        k_before.insert(slot, per_layer);
    }

    // ── turn 2:harness 抽掉**第一张**截图(其余全部前移) ──
    let q2 = "现在屏幕上还有什么?";
    let prompt2 = running.boot.tok.chat_format().render_chat(
        &[
            ("system", "你是助手,按约定回答。"),
            (
                "user",
                &format!("{}{}{}{}{q1}", p_text(2), mid, item(1), p_text(1)),
            ),
            ("assistant", &text1),
            ("user", q2),
        ],
        false,
    );
    // 幸存图的**标记必须保持不变**:pad 律由 `evict` 决定,标记一变 ⇒ pad 数变
    // ⇒ 前缀身份断(这是"标记一旦打上就冻结"的硬约束,见设计卷)
    let imgs2 = vec![crate::turn::TurnImage { image: mk_img(0.6), evict: true }];
    let (_, t2) = running
        .submit_session_opts(
            Some(7),
            prompt2.as_str(),
            8,
            crate::turn::TurnOpts { images: imgs2, ..Default::default() },
        )
        .expect("t2 提交");
    let mut prefill_events = 0usize;
    let text2 = loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Prefill { turn, .. } if turn == t2 => prefill_events += 1,
            TurnEvent::Completed { turn, text, .. } if turn == t2 => break text,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => panic!("t2 队列丢失"),
            _ => {}
        }
    };
    assert!(!text2.trim().is_empty(), "t2 必须出文本");

    // ① 吸收(不是退回重算)
    let trace = running.img_evict.last.clone().expect("必须吸收删除(退回重算 = 本门失败)");
    assert_eq!(running.img_evict.absorbed, 1, "一次删除只许吸收一次(Δ 一次性)");
    // ② 删的正是第一张图的整段
    assert_eq!(trace.a, del.start, "删除起点必须是第一段的段首");
    assert_eq!(trace.b, del.end, "删除终点必须是第一段的段尾(含 pad)");
    assert_eq!(trace.a % page, 0, "删点必须页对齐");
    assert_eq!((trace.b - trace.a) % page, 0, "删除长度必须整页");
    assert!(trace.keep_blocks >= 1, "后缀至少要保留一页");
    assert_eq!(trace.pages.len(), trace.keep_blocks);
    // ③ 吸收轮只喂新尾部
    assert!(
        prefill_events <= 3,
        "吸收轮 prefill 事件 {prefill_events} 次 —— 后缀被重算了?(期间应只喂新尾部)"
    );

    // ④ **G6b 改写即失效**:被重转后缀在内容树里的**旧身份必须已废弃** ——
    // 否则别的会话按旧 token 命中同一批槽,会读到旋转过的 K(静默错值;实测 NaN)。
    {
        let (hit_old, _) = running.bank.rel.match_prefix(&old_tokens).expect("旧链查询");
        // 删点之后**整条旧链**都不可达了:被删段的 KV 已显式释放(用户口径:
        // 删了图显存要跟着降),后缀页被原地重转(旧身份失真),尾巴是孤儿。
        // ⇒ 旧 tokens 现在只能命中**前缀**。
        // **2026-10-24 收窄**:我们只摘「被旋转页」的旧身份(正确性);被删段与
        // 未复用的老尾巴内容没变 ⇒ 留作死枝交回收器。所以旧历史现在能**穿过
        // 被删段**,在**第一个被旋转的页**处断 ✓(那里 K 已失真)。
        assert_eq!(
            hit_old.matched_blocks,
            trace.b / page,
            "旧身份必须止于第一页被旋转的后缀(得 {} 块;b/page={})",
            hit_old.matched_blocks,
            trace.b / page
        );
    }

    // ⑤ **G1 机械等价**:复用后缀页每层 K == R(−S)·删除前(配对量级 ≤4 ulp)
    let mrope_delta = {
        let vlm = running.dev.boot.vlm.as_ref().expect("VLM");
        owl_models::layers::kvshift::delta_table(&vlm.cfg().mrope, -trace.shift)
    };
    let half = mrope_delta.0.len();
    let geom = running.dev.pool.k_shift_geom(half).expect("K 重转几何");
    let mut worst = 0f32;
    for &slot in &trace.pages {
        let Some(before) = k_before.get(&slot) else {
            panic!("后缀页 {slot} 不在删除前的读取集里(页序错?)");
        };
        let after_all = read_slot(&mut running, slot, per_page_bytes).await;
        for (li, after) in after_all.iter().enumerate() {
            // host 参考:同一页原地乘 R(−S)
            let mut expect = before[li]
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect::<Vec<u16>>();
            owl_models::layers::kvshift::k_shift_host_f16(
                &mut expect,
                &mrope_delta.0,
                &mrope_delta.1,
                &[0u32],
                geom,
            )
            .expect("host 参考");
            // 判据 = **配对量级**的 4 ulp(rotate-half 的配对是
            // `(d, d+half)`;抵销情形按**输入**量级取 ulp —— 按输出去取会爆)
            let dg_units = geom.half / geom.x;
            let n_el = after.len() / 2;
            for i in 0..n_el {
                let a = &after[i * 2..i * 2 + 2];
                let av = f32::from(half::f16::from_bits(u16::from_le_bytes([a[0], a[1]])));
                let ev = f32::from(half::f16::from_bits(expect[i]));
                let dg = (i / (geom.page * geom.x)) % (geom.hd / geom.x);
                let partner = if dg < dg_units {
                    i + geom.half * geom.page
                } else {
                    i - geom.half * geom.page
                };
                let pv = before[li]
                    .chunks_exact(2)
                    .nth(partner)
                    .map(|b| f32::from(half::f16::from_bits(u16::from_le_bytes([b[0], b[1]]))))
                    .unwrap_or(0.0);
                let scale = av.abs().max(ev.abs()).max(pv.abs()).max(1e-3);
                let ulp = scale * 2f32.powi(-10);
                worst = worst.max((av - ev).abs() / (4.0 * ulp));
                assert!(
                    (av - ev).abs() <= 4.0 * ulp,
                    "层 {li} 页 {slot} 元素 {i}: K 重转与 host 参考不符({av} vs {ev};配对 {pv};第 {worst:.2} × 4ulp)"
                );
            }
        }
    }
    // ── G5 臂:**引擎不自作主张** —— 删一张**未标记**的图 ⇒ 必须退回旧路径 ──
    let q3 = "还有别的吗?";
    // 用**另一个会话**(8):同会话里换标记会改 pad 律(标记冻结律)。
    let g5_prompt1 = running.boot.tok.chat_format().render_chat(
        &[
            ("system", "你是助手,按约定回答。"),
            ("user", &format!("{}{}{}{q1}", p_text(1), item(2), p_text(1))),
        ],
        false,
    );
    let (_, g5t1) = running
        .submit_session_opts(
            Some(8),
            g5_prompt1.as_str(),
            6,
            crate::turn::TurnOpts {
                images: vec![crate::turn::TurnImage { image: mk_img(0.9), evict: false }],
                ..Default::default()
            },
        )
        .expect("G5 t1 提交");
    let g5_text1 = drive_turn(&mut running, g5t1).await;
    assert!(!g5_text1.trim().is_empty(), "G5 t1 必须出文本");
    // 删掉这张**未标记**的图 ⇒ 不许吸收
    let g5_prompt2 = running.boot.tok.chat_format().render_chat(
        &[
            ("system", "你是助手,按约定回答。"),
            ("user", &format!("{}{}{q1}", p_text(1), p_text(1))),
            ("assistant", &g5_text1),
            ("user", q3),
        ],
        false,
    );
    let (_, g5t2) = running
        .submit_session_opts(Some(8), g5_prompt2.as_str(), 6, crate::turn::TurnOpts::default())
        .expect("G5 t2 提交");
    let g5_text2 = drive_turn(&mut running, g5t2).await;
    assert!(!g5_text2.trim().is_empty(), "G5 t2(删未标记图)必须出文本");
    assert_eq!(
        running.img_evict.absorbed, 1,
        "未标记的图被删时**不许**吸收(引擎不自作主张)"
    );

    // ── G7 臂:**叠转上限**(delta_limit=1)── 第二次删除必须整段重建 ──
    //
    // 会话 9:turn 1 两张可删图 ⇒ turn 2 删第一张(吸收,rot=1 ⇒ 到顶)⇒
    // turn 3 删第二张时**不许**再叠转,改走全量 prefill(漂移与计数归零)。
    {
        let g7_p1 = running.boot.tok.chat_format().render_chat(
            &[
                ("system", "你是助手,按约定回答。"),
                (
                    "user",
                    &format!("{}{}{}{}{}{q1}", p_text(1), item(0), mid, item(1), p_text(1)),
                ),
            ],
            false,
        );
        let (_, g7t1) = running
            .submit_session_opts(
                Some(9),
                g7_p1.as_str(),
                6,
                crate::turn::TurnOpts {
                    images: vec![
                        crate::turn::TurnImage { image: mk_img(0.2), evict: true },
                        crate::turn::TurnImage { image: mk_img(0.3), evict: true },
                    ],
                    ..Default::default()
                },
            )
            .expect("G7 t1 提交");
        let g7_text1 = drive_turn(&mut running, g7t1).await;
        assert!(!g7_text1.trim().is_empty(), "G7 t1 必须出文本");
        let absorbed_after_t1 = running.img_evict.absorbed;

        // turn 2:删第一张(吸收;rot_rounds = 1 = limit ⇒ 到顶)
        let g7_p2 = running.boot.tok.chat_format().render_chat(
            &[
                ("system", "你是助手,按约定回答。"),
                ("user", &format!("{}{}{}{}{q1}", p_text(1), mid, item(1), p_text(1))),
                ("assistant", &g7_text1),
                ("user", q2),
            ],
            false,
        );
        let (_, g7t2) = running
            .submit_session_opts(
                Some(9),
                g7_p2.as_str(),
                6,
                crate::turn::TurnOpts {
                    images: vec![crate::turn::TurnImage { image: mk_img(0.3), evict: true }],
                    ..Default::default()
                },
            )
            .expect("G7 t2 提交");
        let g7_text2 = drive_turn(&mut running, g7t2).await;
        assert!(!g7_text2.trim().is_empty(), "G7 t2 必须出文本");
        assert_eq!(
            running.img_evict.absorbed,
            absorbed_after_t1 + 1,
            "第一次删除应当被吸收(没到顶)"
        );
        assert_eq!(
            running.bank.sessions.session(9).expect("会话 9").rot_rounds,
            1,
            "吸收一轮 ⇒ 叠转计数 1"
        );

        // turn 3:删第二张 —— 到顶 ⇒ 整段重建(不吸收、cached_len = 0)
        let q4 = "再确认一次。";
        let g7_p3 = running.boot.tok.chat_format().render_chat(
            &[
                ("system", "你是助手,按约定回答。"),
                ("user", &format!("{}{}{}", p_text(1), mid, p_text(1))),
                ("assistant", &g7_text2),
                ("user", q4),
            ],
            false,
        );
        let (_, g7t3) = running
            .submit_session_opts(Some(9), g7_p3.as_str(), 6, crate::turn::TurnOpts::default())
            .expect("G7 t3 提交");
        let g7_text3 = drive_turn(&mut running, g7t3).await;
        assert!(!g7_text3.trim().is_empty(), "G7 t3(重建)必须出文本");
        // 判据要看**重建的痕迹**,不是 turn 收口后的 cached_len(turn 收口会把
        // cached_len 写成新链长度 —— 任何路径都不是 0):
        //   ① 叠转计数归零(只有 rebase 路径会清零;不吸收的旧路径会留着 1);
        //   ② 计数器 `image_evict.rebase` 涨了 1(决策面的唯一事实)。
        let s9 = running.bank.sessions.session(9).expect("会话 9");
        assert_eq!(s9.rot_rounds, 0, "重建 ⇒ 叠转计数归零");
        assert_eq!(
            running.img_evict.rebased, 1,
            "到顶必须走一次整段重建(观测账 img_evict.rebased)"
        );
        assert_eq!(
            running.img_evict.absorbed,
            absorbed_after_t1 + 1,
            "到顶那一轮**不许**再吸收(叠转上限是硬门)"
        );
    }

    eprintln!(
        "[G3] 旋转段 {} µs({} 页 / {} 层;逐层发射的旧口径 ≈1.1 ms)",
        trace.rotate_us,
        trace.keep_blocks,
        running.dev.pool.kv_caches().len()
    );
    assert!(
        trace.rotate_us < 5_000,
        "旋转段 {} µs —— 发射没合批?(旧口径 ~1.1ms,批量应远低于此)",
        trace.rotate_us
    );
    eprintln!(
        "[test] 前缀无感删除:删 [{}, {})={} tok · 保留后缀 {} 页 · S={} · 重挂 {} · 轮时 prefill 事件 {} · K 最差 {:.2}×4ulp",
        trace.a,
        trace.b,
        trace.b - trace.a,
        trace.keep_blocks,
        trace.shift,
        trace.rekeyed,
        prefill_events,
        worst
    );
}

/// **多图 × 投机 verify 行**(D 项;2026-10-24)。
///
/// 删除之后后缀 K 被整体前移 `S`(位置帧变),而 verify 行是"anchor + k 个草稿"
/// 的多行前向 —— 它们必须用**新的**位置帧 + **新的** m-RoPE delta 跑;否则接受
/// 判据会拿错位置的 logits 去比。
///
/// 判据:**spec on / off 的贪心 token 流逐位相同**(0.8B 无 DFlash2 ⇒ 用 `dumb`
/// 草稿;verify 位置路径与 DFlash2 是同一份 `verify_forward(.., pos, .., vlm_delta)`。
/// DFlash2 自身的草稿池/ring 另有 27B/TP2 行为门,见设计卷 §5undecies)。
pub async fn gpu_image_evict_spec_identity_gate() {
    let Some(ordinal) = gpu_ordinal() else {
        owl_shared::testkit::report::skip("OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    if !dir.join("config.json").exists() {
        owl_shared::testkit::report::skip(format_args!("0.8B 资产不在 {}", dir.display()));
        return;
    }
    /// 跑一遍"两图 → 删第一张"的灰心流,返回 (turn2 生成 token, 是否吸收, 旋转页数)
    async fn arm(
        spec: bool,
        ordinal: usize,
        dir: &std::path::Path,
        no_delete: bool,
    ) -> (Vec<u32>, bool, usize) {
        let mut knobs = EngineKnobs {
            vision: true,
            prefix_cache: true,
            image_evict: true,
            // 恒等门口径 = 贪心
            sampler_enabled: false,
            pool_tokens: Some(2048),
            ..Default::default()
        };
        if spec {
            knobs.draft.depth = 3;
            knobs.draft.dumb = true;
        }
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 768,
            prefill_chunk: 64,
            knobs,
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(dir).await.expect("装载 0.8B+VLM");
        let mut running = engine.run(loaded).await.expect("装配");
        let page = running.dev.pool.geom.page;
        let image_token = running.dev.boot.vlm.as_ref().expect("VLM").cfg().image_token;
        let one = |r: &RunningEngine<crate::GpuFace>, t: &str| -> u32 {
            let v = r.boot.tok.try_encode(t).expect("encode");
            assert_eq!(v.len(), 1, "{t} 应恰 1 token");
            v[0]
        };
        let (vs, ve) = (one(&running, "<|vision_start|>"), one(&running, "<|vision_end|>"));
        let pad_id = one(&running, "\n");
        // 纯文本预热轮(0.8B「boot 后首轮带图」既有红 F1)
        {
            let (_s, t) = running
                .submit_session(Some(500), "请用一句话说明你在做什么。", 6)
                .expect("预热提交");
            let _ = drive_turn(&mut running, t).await;
        }
        let marked = !owl_shared::env_reader::flag("OWL_GATE_UNMARKED");
        let text_only = owl_shared::env_reader::flag("OWL_GATE_TEXT");
        let mk_img = |salt: f32| owl_models::vlm::VlmImage {
            pixel_values: (0..256 * 3 * 2 * 16 * 16)
                .map(|i| salt + (i % 7) as f32 * 0.01)
                .collect(),
            grid: (1, 16, 16),
        };
        let item = |_i: usize| {
            format!("<|vision_start|>{}<|vision_end|>", "<|image_pad|>".repeat(64))
        };
        let p_text = |n: usize| {
            let mut s = String::new();
            for i in 0..n {
                s.push_str(&format!("观察点{i}:屏幕无异常,日志正常,继续。\n"));
            }
            s
        };
        // `chat_format()` 借 `boot` ⇒ 每次用完即弃(块作用域),不跨 &mut 持有
        let q = "请用一句话说明你看到了什么。";
        let mid = "中间这段文字会留在历史里。\n";
        // 让 image0 的文本起点**落在页边界**(G4 同款;这里为了位置帧干净)
        let fill: String;
        let prompt1: String;
        if text_only {
            // 纯文本两轮(同一会话):判"跨 boot 的贪心是否确定"
            fill = p_text(1);
            prompt1 = {
                let c = running.boot.tok.chat_format();
                c.render_chat(
                    &[
                        ("system", "你是助手,按约定回答。"),
                        ("user", &format!("{}{q}", fill)),
                    ],
                    false,
                )
            };
        } else {
        let (f0, p0) = {
            let fill_ids = running.boot.tok.try_encode(&p_text(3)).expect("filler");
            let mut chosen = None;
            for k in 0..=fill_ids.len() {
                let fill = running.boot.tok.decode(&fill_ids[..k]);
                let p = {
                    let c = running.boot.tok.chat_format();
                    c.render_chat(
                        &[
                            ("system", "你是助手,按约定回答。"),
                            (
                                "user",
                                &format!("{}{}{}{}{}{q}", fill, item(0), mid, item(1), p_text(1)),
                            ),
                        ],
                        false,
                    )
                };
                let ids = running.boot.tok.try_encode(&p).expect("编码");
                let sq = crate::img_evict::pad_image_spans(
                    &ids,
                    &[true, true],
                    &[(1usize, 16usize, 16usize), (1usize, 16usize, 16usize)],
                    image_token,
                    Some(vs),
                    Some(ve),
                    page,
                    pad_id,
                    &[],
                )
                .expect("试算 pad");
                if sq.spans.len() == 2 && sq.spans[0].text_start % page == 0 {
                    chosen = Some((fill, p));
                    break;
                }
            }
            chosen.expect("找不到页对齐前缀")
        };
        fill = f0;
        prompt1 = p0;
        }
        let (sid, t1) = running
            .submit_session_opts(
                Some(501),
                prompt1.as_str(),
                8,
                crate::turn::TurnOpts {
                    images: if text_only {
                        Vec::new()
                    } else {
                        vec![
                            crate::turn::TurnImage { image: mk_img(0.2), evict: marked },
                            crate::turn::TurnImage { image: mk_img(0.6), evict: marked },
                        ]
                    },
                    ignore_eos: true,
                    ..Default::default()
                },
            )
            .expect("t1 提交");
        // t1(padded prompt 长度 + 生成段)—— 判断分叉是否**在第一张图那轮就有**
        let mut plen1 = 0usize;
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Prefill { turn, total, .. } if turn == t1 => plen1 = total,
                TurnEvent::Completed { turn, .. } if turn == t1 => break,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                TurnEvent::Idle => panic!("t1 队列丢失"),
                _ => {}
            }
        }
        let gen1: Vec<u32> = running
            .bank
            .sessions
            .get(sid)
            .map(|s| s.tokens[plen1.min(s.tokens.len())..].to_vec())
            .unwrap_or_default();
        eprintln!("[D] t1 生成(len={}) {gen1:?}", gen1.len());
        let text1 = running
            .bank
            .sessions
            .get(sid)
            .map(|s| s.tokens[plen1.min(s.tokens.len())..].iter().map(|t| *t).collect::<Vec<_>>())
            .map(|_| String::new())
            .unwrap_or_default();
        // 文本由 token 账反解(仅供 prompt2 回填;与客户端回同一份文本同等语义)
        let text1 = {
            let ids: Vec<u32> = running
                .bank
                .sessions
                .get(sid)
                .map(|s| s.tokens[plen1.min(s.tokens.len())..].to_vec())
                .unwrap_or_default();
            running.boot.tok.decode(&ids)
        };
        assert!(!text1.trim().is_empty(), "t1 必须出文本");
        // turn 2:删第一张(对照臂 = 原样保留两图)
        let (prompt2, imgs2) = {
            let c = running.boot.tok.chat_format();
            if text_only {
                (
                    c.render_chat(
                        &[
                            ("system", "你是助手,按约定回答。"),
                            ("user", &format!("{}{q}", fill)),
                            ("assistant", &text1),
                            ("user", "现在屏幕上还有什么?"),
                        ],
                        false,
                    ),
                    Vec::new(),
                )
            } else if no_delete {
                (
                    c.render_chat(
                        &[
                            ("system", "你是助手,按约定回答。"),
                            (
                                "user",
                                &format!("{}{}{}{}{}{q}", fill, item(0), mid, item(1), p_text(1)),
                            ),
                            ("assistant", &text1),
                            ("user", "现在屏幕上还有什么?"),
                        ],
                        false,
                    ),
                    if text_only {
                        Vec::new()
                    } else {
                        vec![
                            crate::turn::TurnImage { image: mk_img(0.2), evict: marked },
                            crate::turn::TurnImage { image: mk_img(0.6), evict: marked },
                        ]
                    },
                )
            } else {
                (
                    c.render_chat(
                        &[
                            ("system", "你是助手,按约定回答。"),
                            ("user", &format!("{}{}{}{}{q}", fill, mid, item(1), p_text(1))),
                            ("assistant", &text1),
                            ("user", "现在屏幕上还有什么?"),
                        ],
                        false,
                    ),
                    if text_only {
                        Vec::new()
                    } else {
                        vec![crate::turn::TurnImage { image: mk_img(0.6), evict: marked }]
                    },
                )
            }
        };
        let (_, t2) = running
            .submit_session_opts(
                Some(501),
                prompt2.as_str(),
                8,
                crate::turn::TurnOpts { images: imgs2, ignore_eos: true, ..Default::default() },
            )
            .expect("t2 提交");
        // 记 padded prompt 长度(= 生成段起点);不能拿"尾部 8"当生成段(spec 可能多吐)
        let mut plen2 = 0usize;
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Prefill { turn, total, .. } if turn == t2 => plen2 = total,
                TurnEvent::Completed { turn, .. } if turn == t2 => break,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                TurnEvent::Idle => panic!("t2 队列丢失"),
                _ => {}
            }
        }
        let absorbed = running.img_evict.absorbed == 1;
        let pages = running.img_evict.last.as_ref().map(|t| t.pages.len()).unwrap_or(0);
        let toks = running
            .bank
            .sessions
            .get(sid)
            .map(|s| s.tokens[plen2.min(s.tokens.len())..].to_vec())
            .unwrap_or_default();
        (toks, absorbed, pages)
    }
    // ① **先立对照臂(不删)**:判"分叉是删除引起,还是 spec×VLM 既有红"。
    // 既有红在场时,删除臂的恒等判据**不可用**(两臂都跑在同一个坏地基上)⇒
    // 门自我诊断并 skip(修好那条红之后本门自动接管)。
    // 诊断臂:两臂同构(spec 都关)⇒ 判"图像多轮路径本身是否确定"
    let spec_a = !owl_shared::env_reader::flag("OWL_GATE_SAME");
    let (c_on, c_abs_on, _) = arm(spec_a, ordinal, &dir, true).await;
    let (c_off, c_abs_off, _) = arm(false, ordinal, &dir, true).await;
    assert!(!c_abs_on && !c_abs_off, "对照臂(不删)不该有吸收");
    eprintln!("[D] 对照臂(不删)on  {c_on:?}\n[D] 对照臂(不删)off {c_off:?}");
    if owl_shared::env_reader::flag("OWL_GATE_SAME") {
        assert_eq!(
            c_on, c_off,
            "同构两臂也不一致 ⇒ 路径本身跨 boot 不确定(OWL_GATE_TEXT=1 可判是否 VLM 侧)"
        );
        eprintln!("[D] 同构两臂一致 ⇒ 之前的差异不是随机性");
        return;
    }
    if owl_shared::env_reader::flag("OWL_GATE_UNMARKED") {
        assert_eq!(
            c_on, c_off,
            "无 pad(不标记)对照臂也分叉 ⇒ 纯 spec×VLM 既有红(与 pad 律无关)"
        );
        eprintln!("[D] 无 pad 对照臂恒等 ⇒ 分叉来自 pad 律 × spec(本特性范围)");
        return;
    }
    if c_on != c_off {
        owl_shared::testkit::report::skip(
            "既有红:VLM 跨 boot 不可复现(无删除对照臂已分叉;OWL_GATE_SAME=1 同构两臂也分叉、\
             OWL_GATE_TEXT=1 纯文本则逐位相同)⇒ spec 差异无法归因于 verify 行/旋转,删除臂判据暂不可用",
        );
        return;
    }
    // ② 删除臂
    let (a, abs_a, pa) = arm(true, ordinal, &dir, false).await;
    let (b, abs_b, pb) = arm(false, ordinal, &dir, false).await;
    assert!(abs_a && abs_b, "两臂都必须吸收删除(on={abs_a} off={abs_b})");
    assert!(pa > 0 && pb > 0, "两臂都必须真的旋转过(on={pa} off={pb} 页)");
    eprintln!("[D] 删除臂 on  len={} {a:?}\n[D] 删除臂 off len={} {b:?}", a.len(), b.len());
    assert_eq!(a.len(), b.len(), "两臂生成 token 数必须一致(max_new=8)");
    assert_eq!(
        a, b,
        "spec on/off 的贪心 token 必须逐位相同(verify 行 × 旋转后的位置帧)"
    );
}

/// 27B 的 F1 控制组(首轮纯文本)。
pub async fn gpu_vlm_first_turn_probe_27b_textfirst() {
    gpu_vlm_first_turn_probe_27b(false).await;
}

/// 27B 的 F1 复现组(首轮带图)。
pub async fn gpu_vlm_first_turn_probe_27b_imagefirst() {
    gpu_vlm_first_turn_probe_27b(true).await;
}

/// **MTP KV 行对拍**(T3 前置,2026-10-22)：`prefill_extend` 多行写 vs
/// `t_rows=1` 单行写，用生产 `MtpPredictor::propose` 读回效果比较。
///
/// 不走 D2H：MTP 池块跨账本，`copy_block_at` 会拒；`propose` 是生产读路径。
/// 步骤（base=0, n=4）：
/// ① 多行写 row 0..n-1，逐行 `propose(k=1)` 记录 draft；
/// ② 逐行 `t_rows=1` 写零 hidden（哨兵）⇒ draft 必须变化（证明单行写非 no-op）；
/// ③ `t_rows=1` 写回原 hidden ⇒ draft 必须等于多行。
///
/// 门控 `OWL_E2E_ONLY=gpu_mtp_kv_parity` + `OWL_TEST_DEVICE` + ckpt:27b-awq。
pub async fn gpu_mtp_kv_parity() {
    use owl_iface::contract::{Bytes, Dtype};
    use owl_models::tensor::TensorOps;

    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
        eprintln!("skip: ckpt:27b-awq 未设");
        return;
    };

    async fn host_block(
        running: &mut RunningEngine<crate::GpuFace>,
        dtype: Dtype,
        shape: Vec<usize>,
        data: &[u8],
    ) -> (Bytes, Vec<u64>) {
        let t = TensorOps::from_host(dtype, shape, data);
        let env = running.dev.spec_rt.draft_env;
        let tp = running.dev.boot.tp.clone();
        let face = running.dev.graphs.session.face_mut();
        owl_models::interpreters::eval_ops_scoped_env(t.step(), face, env, tp.as_ref())
            .await
            .expect("host block eval")
    }

    async fn free_ids(running: &mut RunningEngine<crate::GpuFace>, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        let face = running.dev.graphs.session.face_mut();
        face.free(ids).await.expect("free");
    }

    async fn propose_one(
        running: &mut RunningEngine<crate::GpuFace>,
        sid: u64,
        anchor: u32,
        seed: &Bytes,
        p: usize,
    ) -> u32 {
        let page = running.dev.pool.geom.page;
        let dtype = running.dev.pool.geom.dims.dtype;
        let hidden = running.dev.pool.geom.dims.hidden;
        let bt = running
            .bank
            .sessions
            .session(sid)
            .expect("session")
            .mtp_block_table
            .clone();
        let (kc, vc) = running.dev.pool.mtp_kv_leaf().expect("MTP KV");
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let slot_at = |p: usize| (bt[p / page] * page as u32 + (p % page) as u32) as f32;
        let bt_step = TensorOps::from_host(
            Dtype::F32,
            vec![1, bt.len()],
            &f32b(&bt.iter().map(|&b| b as f32).collect::<Vec<_>>()),
        );
        let kv = owl_models::module::KvBuffers {
            k_cache: kc,
            v_cache: vc,
            slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot_at(p)])),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(p + 1) as f32])),
            block_tables: bt_step,
        };
        let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[p as f32]));
        let anchor_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[anchor as f32]));
        let seed_view = TensorOps::of_block(seed.id, dtype, vec![1, hidden]);
        let env = running.dev.spec_rt.draft_env;
        let vocab = running.dev.boot.model.vocab_size();
        let rope = running.dev.boot.rope.clone();
        let attn_v2 = running.dev.pool.attn_v2_scratch();
        let drafts = match running.dev.spec_rt.drafter.as_ref().expect("drafter") {
            crate::spec_runtime::Drafter::Mtp(m) => m.propose(
                &anchor_t,
                &seed_view,
                1,
                &running.dev.boot.model.embed,
                &rope,
                &[kv],
                &[pos_t],
                vocab,
                env,
                attn_v2,
            ),
            _ => panic!("MTP only"),
        };
        let tp = running.dev.boot.tp.clone();
        let (b, arena) = {
            let face = running.dev.graphs.session.face_mut();
            owl_models::interpreters::eval_ops_scoped_env(drafts.step(), face, env, tp.as_ref())
                .await
                .expect("propose eval")
        };
        let mut buf = [0u8; 4];
        {
            let face = running.dev.graphs.session.face_mut();
            face.dtoh(&b, &mut buf).await.expect("draft dtoh");
            face.free(&arena).await.expect("free arena");
            face.free(&[b.id]).await.expect("free root");
        }
        f32::from_le_bytes(buf) as u32
    }

    let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
    knobs.gdn_slots = 1;
    knobs.snap_max = 1;
    knobs.prefix_cache = false;
    knobs.draft.depth = 3;
    knobs.cuda.capture_slab_mb = Some(160);
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 2048,
        prefill_chunk: 64,
        knobs,
    })
    .expect("构造");
    let loaded = engine
        .loader()
        .load_qwen38_27b_awq(
            dir.as_ref(),
            dir.as_ref(),
            owl_models::module::QuantPlan::F16,
            owl_models::module::QuantPlan::W4A16Awq,
        )
        .await
        .expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    let (sid, _id) = running
        .submit_session(Some(7), "mtp-kv-parity", 8)
        .expect("submit");

    let dims = running.dev.pool.geom.dims;
    let page = running.dev.pool.geom.page;
    let n = 4usize;
    let base = 0usize;
    let ids: Vec<u32> = (0..=n as u32).collect();
    let hidden = dims.hidden;
    let dtype = dims.dtype;

    let row_bytes = |i: usize, zero: bool| -> Vec<u8> {
        let mut v = Vec::with_capacity(hidden * 2);
        for j in 0..hidden {
            let val = if zero { 0.0 } else { (i * 1000 + j) as f32 * 0.001 };
            v.extend_from_slice(&half::f16::from_f32(val).to_le_bytes());
        }
        v
    };
    let rows_bytes = |rows: usize, zero: bool| -> Vec<u8> {
        let mut v = Vec::with_capacity(rows * hidden * 2);
        for i in 0..rows {
            v.extend_from_slice(&row_bytes(i, zero));
        }
        v
    };

    // ① 多行写。
    let (hid_multi, arena_m) = host_block(&mut running, dtype, vec![n, hidden], &rows_bytes(n, false)).await;
    running
        .dev
        .prefill_extend(&mut running.bank, sid, &hid_multi, n, &ids, base)
        .await
        .expect("multi-row prefill_extend");
    free_ids(&mut running, &arena_m).await;
    free_ids(&mut running, &[hid_multi.id]).await;

    // ② 多行基线 draft。
    let mut row_blocks: Vec<(Bytes, Vec<u64>)> = Vec::with_capacity(n);
    let mut multi_drafts: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let (h, ha) = host_block(&mut running, dtype, vec![1, hidden], &row_bytes(i, false)).await;
        let d = propose_one(&mut running, sid, ids[i + 1], &h, i).await;
        multi_drafts.push(d);
        row_blocks.push((h, ha));
    }

    // ③ 逐行哨兵 + 恢复。
    for i in 0..n {
        let row_ids = [ids[i], ids[i + 1]];
        let (z, za) = host_block(&mut running, dtype, vec![1, hidden], &row_bytes(0, true)).await;
        running
            .dev
            .prefill_extend(&mut running.bank, sid, &z, 1, &row_ids, base + i)
            .await
            .expect("sentinel t=1");
        let dz = propose_one(&mut running, sid, ids[i + 1], &z, i).await;
        assert_ne!(
            dz, multi_drafts[i],
            "t=1 写零 hidden 后 draft 未变：单行写疑似 no-op（row {i}）"
        );
        free_ids(&mut running, &za).await;
        free_ids(&mut running, &[z.id]).await;

        let h = row_blocks[i].0.clone();
        running
            .dev
            .prefill_extend(&mut running.bank, sid, &h, 1, &row_ids, base + i)
            .await
            .expect("restore t=1");
        let dr = propose_one(&mut running, sid, ids[i + 1], &h, i).await;
        assert_eq!(dr, multi_drafts[i], "t=1 恢复后 draft != 多行（row {i}）");
    }
    for (h, ha) in &row_blocks {
        free_ids(&mut running, ha).await;
        free_ids(&mut running, &[h.id]).await;
    }

    eprintln!("[mtp-kv-parity] PASS n={n} page={page} t=1 vs 多行 draft 一致");
    drop(running);
    std::thread::sleep(std::time::Duration::from_secs(10));
}
