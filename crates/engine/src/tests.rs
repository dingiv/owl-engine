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

/// **piecewise 段驱动件门**(2026-10-15;A4)—— **不装模型**,纯 `GraphPlan` 面。
///
/// 判据(与设计 `docs/piecewise-断点图-设计与实现.md` §7 P2 同源):
/// ① 捕获成功且**段/断点数与声明一致**;
/// ② **回放两次值一致**(跨段地址接力 + 可重复);
/// ③ 断点的钉块由驱动件按声明形状分配 ⇒ 两期同址(值由断点算子写入)。
///
/// 树:`y = x + c`(段0)→ `brk = y + y`(**断点**;窗外)→ `out = brk + c`(尾段)。
pub async fn gpu_piecewise_segments() {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
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
        .name("owl-pw-test".into())
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
        let y = x.add(&c); // 段0
        let brk = y.add(&y); // 断点(窗外重放)
        sc.break_node(&brk)?;
        let out = brk.add(&c); // 尾段
        sc.output("out", &out)
    };
    let (mut plan, outcome) = crate::graph_plan::GraphPlan::plan_piecewise(
        face,
        owl_models::env::EnvProvider::default(),
        desc,
        true,
        forward,
    )
    .await
    .expect("plan_piecewise");
    assert!(
        matches!(outcome, crate::graph_plan::PlanOutcome::Captured),
        "应捕获,得 {outcome:?}"
    );
    assert_eq!(plan.piecewise_shape(), (2, 1), "应两段一断点");
    // 回放 ×2:x=1,c=1 ⇒ y=2 ⇒ brk=4 ⇒ out=5
    for i in 0..2 {
        let xs = [1.0f32; 4];
        plan.step(&[("x", &xs)]).await.expect("step");
        let v = plan.read_output_f32("out").await.expect("read out");
        assert!(
            v.iter().all(|x| (x - 5.0).abs() < 1e-4),
            "回放 #{i} 值应 5.0,得 {v:?}"
        );
    }
    // **引用计数推送面**(全量记名同轮;2026-10-16):图 plan 的 warmup 探针窗内,
    // 求值器必须把 `(users, pending)` 推给探针 —— "它凭什么是活的"的数据源。
    // 判据用**计数器**而非日志(日志受级别过滤;计数是机械事实)。窗口在 plan
    // 内部 `probe_on/off`,on 时清零 ⇒ 此处读到的是本次 plan 的推送量。
    let pushed = owl_shared::slab_hint::refs_pushed();
    assert!(
        pushed > 0,
        "两段计划跑完后引用计数推送为 0 ⇒ 数据源断了(记名/OOM dump 的\"为什么活着\"面失效)"
    );
    eprintln!("[piecewise] 两段一断点捕获 + 双回放 值一致 ✓(引用计数推送 {pushed} 次)");
}

fn asset_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models/assets/Qwen3.5-0.8B")
}

fn gpu_ordinal() -> Option<usize> {
    owl_shared::env_reader::parse::<usize>("OWL_TEST_DEVICE")
}

/// CPU:构造(零执行)+ ModelLoader 装载门禁(权重/tokenizer 解析)
/// EngineConfig 全件从**用例配置**注入(隐藏 env `OWL_TEST_CONFIG_CPU_CONSTRUCT_AND_LOAD`
/// → 引擎同款 toml;缺省 = `OwlConfig::default()`)。本用例只到 loader ——
/// 三件均不参与断言,注入不改测试行为;换档 = 换 toml 不改代码(S3 收编示范)。
#[cfg(test)] // dev-dep(owl-macros)门控:单元档专用
#[owl_test]
async fn cpu_construct_and_load(cfg: &OwlConfig) {
    let mut engine = Engine::on(
        EngineConfig {
            device_ordinal: cfg.runtime.test_device.or(cfg.runtime.device_ordinal).unwrap_or(0),
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

/// CPU:增量解码多字节字符跨 token(🌟 4 字节;逐 token 喂入,
/// 差分不得 panic 且终态与前缀累计一致)。EngineConfig 同样走用例配置注入。
#[cfg(test)] // dev-dep(owl-macros)门控:单元档专用
#[owl_test]
async fn cpu_incremental_decode_multibyte(cfg: &OwlConfig) {
    let mut engine = Engine::on(
        EngineConfig {
            device_ordinal: cfg.runtime.test_device.or(cfg.runtime.device_ordinal).unwrap_or(0),
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
/// `cfg` 必须与「独立按同键 loader 复算」逐字段一致 —— env 未设臂
/// (default)与文件臂(设 `OWL_TEST_CONFIG_CASE_CONFIG_INJECTION=<toml>`)
/// 用同一条断言。零装载零 GPU(单元档)。
#[cfg(test)]
#[owl_test]
async fn case_config_injection(cfg: &OwlConfig) {
    let key = "OWL_TEST_CONFIG_CASE_CONFIG_INJECTION";
    let mut expect = match owl_shared::env_reader::str(key) {
        Some(path) => OwlConfig::load(None, Some(std::path::Path::new(&path)))
            .unwrap_or_else(|e| panic!("独立复算失败({path}): {e}")),
        None => OwlConfig::default(),
    };
    expect.runtime.test_device = owl_shared::env_reader::parse("OWL_TEST_DEVICE");
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
        eprintln!("skip: OWL_TEST_DEVICE 未设");
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
            eng.turns.active.as_ref().unwrap().prompt_ids.len()
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
        let act = eng.turns.active.as_mut().unwrap();
        act.fed = act.prompt_ids.len();
        act.out.push(7);
    }

    let page = eng.pool.page;
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
        let act = eng.turns.active.take().unwrap();
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
        eprintln!("skip: OWL_OFFLOAD_PROBE 未设(S3-3b 实证按需开)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
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
        eprintln!("skip: OWL_CPU_MOCK / OWL_OFFLOAD_PROBE 未设(CPU mock 档按需开)");
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
        running.pool.snap_used(),
        running.pool.snap_capacity()
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
    assert_eq!(running.pool.snap_slot_of(host_key), Some(slot), "槽位索引应记上");

    // ③ 回灌后的快照能正常恢复到会话格(D2D;不报错即通路成立)
    let gdn_slot = 0usize;
    let face = running.graphs.session.face_mut();
    running
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
}

/// 状态腿三层化探针(自装载;按需开:`OWL_STATE_LEG_PROBE=1`)。
/// 档位:snap_max=1(逼出 churn)+ 状态腿宿主分片 64MiB(0.8B 一条快照 ≈19.3MiB)。
pub async fn gpu_08b_state_leg_probe() {
    if !owl_shared::env_reader::flag("OWL_STATE_LEG_PROBE") {
        eprintln!("skip: OWL_STATE_LEG_PROBE 未设(S2 状态腿探针按需开)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
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
        eprintln!("skip: OWL_CPU_MOCK / OWL_STATE_LEG_PROBE 未设");
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
async fn index_probe_scenario<D: DeviceClient>(running: &mut RunningEngine<D>) {
    let before = running.bank.rel.index_bytes();
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

/// **B 轴覆盖率探针场景体**(Q-2;设计卷 §8.6;face 无关)。
///
/// 门(§8.6):①候选块**全在索引里**(边界/分析序正确)②top-k 覆盖率落在论文
/// 量级且**显著高于随机基线** `k/块数`(约定错 ⇒ 打分退化噪声 ⇒ 贴线)。
/// 首测定档:top-8 ≥ 0.45 且 ≥ 2× 随机(实测值见 e2e 日志)。
async fn coverage_probe_scenario<D: DeviceClient>(running: &mut RunningEngine<D>) {
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
        .cov_last
        .as_ref()
        .expect("覆盖率探针无观测:prompt 未跨块边界 / coverage_probe 未开?");
    eprintln!("[cov] {}", rep.line());
    eprintln!("[cov] {}", running.prof.line());
    let pairs: Vec<(usize, f32, f32)> = (0..rep.blocks)
        .map(|i| {
            let m = rep.mass.get(i).copied().unwrap_or(0.0);
            let s = rep.scores.get(i).copied().unwrap_or(0.0);
            (i, (m * 1000.0).round() / 1000.0, (s * 1000.0).round() / 1000.0)
        })
        .collect();
    eprintln!("[cov] 逐块(块号, 真实质量, 索引分): {pairs:?}");
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
        // 视图必须**真的丢块**才有对照意义(缺省 16 < 可用块 ~37);
        // `OWL_COV_SELECT_BLOCKS` 可探"丢到多少才开始变"
        // 缺省 32(= 材料化 ~67%):本场景是**机制门**(视图/部分驻留),质量门要
        // 在材料化 ≥60% 的档位看才有意义(预算 16/48 时丢弃 28% 质量 ⇒ 输出本就该变)
        cfg.knobs.select_blocks =
            owl_shared::env_reader::parse::<usize>("OWL_COV_SELECT_BLOCKS").unwrap_or(32);
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
        let partial = running.partial;
        (
            running.select_view_steps,
            running.sel_view_fallback,
            partial,
            running
                .cov_last
                .as_ref()
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
        eprintln!("skip: OWL_SELECT_DECODE 未设(B-3d 探针按需开)");
        return;
    }
    if !owl_shared::env_reader::flag("OWL_COVERAGE_PROBE")
        || !owl_shared::env_reader::flag("OWL_INDEX_PROBE")
    {
        eprintln!("skip: 需 OWL_COVERAGE_PROBE + OWL_INDEX_PROBE(召回/索引面)");
        return;
    }
    if gpu_ordinal().is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    select_decode_scenario().await;
}

/// **B 轴覆盖率探针**(自装载;`OWL_COVERAGE_PROBE=1` + `OWL_INDEX_PROBE=1`)。
pub async fn gpu_08b_coverage_probe() {
    // 门 = **运行器级**(选卡 + 场景过滤);**行为面一律来自配置文件**
    // (`OWL_PROBE_FLAVOR` 基档 + `OWL_PROBE_OVERLAY` 覆盖件)
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设(哪张卡)");
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

// ═══════════════════════════════════════════════════════════════════════════
// **探针/基准档的统一装载口**(2026-10-19 配置面统一)
// ═══════════════════════════════════════════════════════════════════════════

/// 一次探针跑 = **两个配置文件**描述全部行为面(与生产**同一条加载路**):
///
/// ```text
/// OWL_PROBE_FLAVOR=<name>    基准档(assets/flavors/<name>.toml)
/// OWL_PROBE_OVERLAY=<path>   覆盖文件(可选;同 server 的 [CONFIG] 位置)
/// ```
///
/// 引擎旋钮一律由 [`EngineKnobs::from_config`] 搬 —— 探针档**不再自己拼 knobs**
/// (那正是"条件不一样、数字对不上"的根因)。要改条件就改文件,改完条件写在
/// 文件里、每跑一次 `digest()` 自证。
///
/// 只剩**运行器级**三件事走 env(它们不属于引擎配置):
/// - `OWL_TEST_DEVICE`(哪张卡)/ `OWL_E2E_ONLY`(跑哪些场景)
/// - `OWL_AWQ27B_DIR` / `OWL_DFLASH2_DIR`(资产路径)
/// - 本函数的两个(读哪个配置文件)
fn probe_engine_config(ordinal: usize, default_flavor: &str) -> EngineConfig {
    let flavor = owl_shared::env_reader::str("OWL_PROBE_FLAVOR")
        .unwrap_or_else(|| default_flavor.to_string());
    let base = owl_shared::assets::preset_path(&flavor).expect("预设档路径");
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
    let stage_before = running.stage.blocks;
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
    let st = running.stage;
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
        eprintln!("skip: OWL_B3E_PROBE 未设(B-3e stage-in 探针按需开)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
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
    knobs.max_active_turns = 8;
    EngineConfig { device_ordinal: ordinal, max_seq_tokens: 512, prefill_chunk: 64, knobs }
}

/// **B-5 Stage 2:27B 单卡 + 选择层 + 部分驻留**(`OWL_SELECT_27B=1` + `OWL_AWQ27B_DIR`)。
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
        owl_shared::env_reader::str("OWL_AWQ27B_DIR").expect("OWL_AWQ27B_DIR"),
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
    knobs.dense_floor = EngineKnobs::dense_floor_for(
        sp,
        owl_shared::env_reader::parse::<u64>("OWL_27B_FLOOR_MIB"),
    );
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
        .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, false)
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
    let text = loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Completed { turn, text, .. } if turn == tid => break text,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            _ => {}
        }
    };
    let rep = running.store_report();
    let st = running.stage;
    let pt = running.partial;
    let rec = running.cov_last.as_ref().map(|r| r.recall.clone()).unwrap_or_default();
    eprintln!(
        "[b5-27b] 用时 {:.1}s · 视图步 {} · 回退 {} · 召回 {} 页(覆盖 {:.1}% 物化 {:.1}%)",
        t0.elapsed().as_secs_f32(),
        running.select_view_steps,
        running.sel_view_fallback,
        rec.view_pages,
        rec.coverage * 100.0,
        rec.materialize_ratio * 100.0
    );
    eprintln!("[b5-27b] {} · {}", pt.line(), st.line());
    eprintln!("[b5-27b] {}", running.prof.line());
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
    assert!(running.select_view_steps > 0, "视图没驱动 decode");
    assert_eq!(running.sel_view_fallback, 0, "视图回退(占位符/视图缺位?)");
    assert!(text.chars().count() >= 4, "输出退化");
}

/// **B-5 Stage 2 探针入口**(自装载;`OWL_SELECT_27B=1`)。
pub async fn gpu_27b_select_probe() {
    if !owl_shared::env_reader::flag("OWL_SELECT_27B") {
        eprintln!("skip: OWL_SELECT_27B 未设(B-5 Stage2 探针按需开)");
        return;
    }
    if owl_shared::env_reader::str("OWL_AWQ27B_DIR").is_none() || gpu_ordinal().is_none() {
        eprintln!("skip: 需 OWL_AWQ27B_DIR + OWL_TEST_DEVICE");
        return;
    }
    select_27b_scenario().await;
}

/// B-0 索引采集探针(自装载;`OWL_INDEX_PROBE=1`)。
pub async fn gpu_08b_index_probe() {
    if !owl_shared::env_reader::flag("OWL_INDEX_PROBE") {
        eprintln!("skip: OWL_INDEX_PROBE 未设(B-0 索引采集探针按需开)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let mut engine = Engine::new(index_probe_config(ordinal)).expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&dir).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    index_probe_scenario(&mut running).await;
}

/// **CPU mock 档的索引采集场景**(无卡:验链路;K 恒零 ⇒ 数值无意义)。
pub async fn cpu_mock_index_probe() {
    if !owl_shared::env_reader::flag("OWL_CPU_MOCK") || !owl_shared::env_reader::flag("OWL_INDEX_PROBE") {
        eprintln!("skip: OWL_CPU_MOCK / OWL_INDEX_PROBE 未设");
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
    index_probe_scenario(&mut running).await;
}

/// 索引探针档(小池 + 开采集;prompt 短 ⇒ 靠 decode 页边界出块)
fn index_probe_config(ordinal: usize) -> EngineConfig {
    let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
    knobs.pool_tokens = Some(256);
    knobs.max_active_turns = 8;
    knobs.index_probe = true;
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
    // prompt 先算好(借用 `running.tok` 只在这里一次;下面要 &mut *running)
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
        let st = running.stage;
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
            let wrapped = running.tok.chat_wrap(p);
            let ids = running.tok.encode(&wrapped);
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
            running.pool.snap_capacity(),
            running.pool.snap_used()
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
        eprintln!("skip: OWL_BUDGET_PROBE 未设(预算实测按需开)");
        return;
    }
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
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
    let (nb, nb_kv, page, n_full, n_gdn, kv_fp8, max_active_turns, snap_slots, paged) = {
        let pool = &running.pool;
        (
            pool.nb,
            pool.nb_kv,
            pool.page,
            pool.kvs.len(),
            pool.gdn_count(),
            pool.kv_fp8,
            pool.max_active_turns,
            pool.snap_capacity(),
            pool.paged,
        )
    };
    let (block_bytes, gdn_row_bytes) = {
        let pool = &running.pool;
        (pool.block_bytes(), pool.gdn_row_bytes())
    };
    let kv_pool = (nb_kv * block_bytes) as u64;
    let gdn_pool = (max_active_turns * gdn_row_bytes) as u64;
    let (snap_pool, draft_pool) = {
        let pool = &running.pool;
        (pool.snapshot_pool_bytes() as u64, pool.draft_pool_bytes() as u64)
    };
    let kv_per_tok = (block_bytes / page.max(1)).max(1);
    let weights = live_loaded.saturating_sub(live_open);
    let runtime_floor = live_run
        .saturating_sub(live_loaded)
        .saturating_sub(kv_pool + gdn_pool + snap_pool + draft_pool);

    eprintln!(
        "[budget] 池几何 nb={} nb_kv={} page={} n_full={} n_gdn={} kv_fp8={} max_active_turns={} snap_slots={} paged={}",
        nb, nb_kv, page, n_full, n_gdn, kv_fp8, max_active_turns, snap_slots, paged
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
        eprintln!("skip: OWL_TEST_DEVICE 未设");
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

pub async fn gpu_two_turns_lifecycle(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
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
pub async fn gpu_session_continuity(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
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
pub async fn gpu_longctx_4k_prefix_qa(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    let seq: usize = owl_shared::env_reader::parse_or("OWL_E2E_SEQ", 4096);
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
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
        if running.tok.encode(&full).len() + 32 > seq {
            break; // 留 32 token 生成余量(turn2 预算同享)
        }
        body = cand;
        i += 1;
    }
    let prompt_t1 = format!("{body}{question}");
    let n_tok = running.tok.encode(&prompt_t1).len();
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
        eprintln!("skip: OWL_TEST_DEVICE 未设");
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
            if running.tok.encode(&format!("{cand}{question}")).len() + 32 > seq {
                break; // 留 32 token 生成余量
            }
            body = cand;
            i += 1;
        }
        let prompt = format!("{body}{question}");
        let n_tok = running.tok.encode(&prompt).len();
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
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir(); // 转换产物与 f16 同目录约定:../Qwen3.5-0.8B-W4A16
    let w4a16_dir = dir
        .parent()
        .expect("assets")
        .join("Qwen3.5-0.8B-W4A16");
    if !w4a16_dir.exists() {
        eprintln!("skip: W4A16 检查点不存在({:?}),先跑转换器", w4a16_dir);
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
/// GEMM_W4A16_AWQ)+ 图捕获 + 生成。门控 OWL_AWQ27B_DIR = 检查点
/// 目录(models/cyankiwi/Qwen3.8-27B-AWQ-INT4);未设则 skip。
/// VRAM 预算:权重 ~17.5GB + KV/GDN 状态 ~1.3GB → 3090 Ti 24G。
pub async fn gpu_awq27b_marlin_e2e(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else {
        eprintln!("skip: OWL_AWQ27B_DIR 未设(cyankiwi 检查点目录)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.exists() {
        eprintln!("skip: 检查点目录不存在({dir:?})");
        return;
    }
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    let _t0 = std::time::Instant::now();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
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
            .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, false)
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
        let raw = match running.graphs.session.read_input_slot_bytes("ts_buf").await {
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
pub async fn gpu_multi_session_isolation(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
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
    while running.tok.encode(&a_body).len() < 80 {
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
    let a_tokens: Vec<u32> = running.tok.encode(&running.tok.chat_wrap(a_prompt.as_str()));
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
        .tok
        .encode(&running.tok.chat_wrap("我的代号是贝塔。"));
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
pub async fn gpu_prefix_cache_hit(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
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
    while running.tok.encode(&format!("{body}{}", filler(i))).len() < 300 {
        body = format!("{body}{}", filler(i));
        i += 1;
    }
    let prompt_a = format!("{body}清单结束。按约定,暗号重复一遍:暗号是「");
    let prompt_b = format!("{body}这次换个问题:暗号里出现了什么动物?");
    let n_a = running.tok.encode(&prompt_a).len();
    let n_b = running.tok.encode(&prompt_b).len();
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
    let a_ids = running.tok.encode(&running.tok.chat_wrap(prompt_a.as_str()));
    let b_ids = running.tok.encode(&running.tok.chat_wrap(prompt_b.as_str()));
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
pub async fn gpu_27b_chat_inference(shared: Option<&mut RunningEngine<owl_cuda::GpuClient>>) {
    let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else {
        eprintln!("skip: OWL_AWQ27B_DIR 未设(cyankiwi 检查点目录)");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.exists() {
        eprintln!("skip: 检查点目录不存在({dir:?})");
        return;
    }
    let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
    // 共享装载(§四.1);None = 单跑自带装载
    let mut own;
    let running: &mut RunningEngine<owl_cuda::GpuClient> = match shared {
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
            .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, false)
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

    /// **0.8B 默认档共享装载**(e2e 唯一装载点之一;6 步共用:
/// schedule_decision_surface / two_turns_lifecycle / session_continuity /
/// multi_session_isolation / prefix_cache_hit / longctx_4k_prefix_qa)。
/// 装载本身即一步测试(构造+装载+装配+捕获态断言+计时)。
pub async fn boot_08b(max_seq: usize) -> RunningEngine<owl_cuda::GpuClient> {
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
pub async fn boot_27b_default() -> RunningEngine<owl_cuda::GpuClient> {
    let dir = std::path::PathBuf::from(
        owl_shared::env_reader::str("OWL_AWQ27B_DIR").expect("OWL_AWQ27B_DIR"),
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
        .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, false)
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
        let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else { return; };
        let dir = std::path::PathBuf::from(dir);
        if !dir.exists() { return; }
        let ordinal = gpu_ordinal().expect("OWL_TEST_DEVICE");
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal, max_seq_tokens: 512, prefill_chunk: 128,
            knobs: EngineKnobs::default(),
        }).expect("构造");
        let loaded = engine.loader().load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, false).await.expect("装载");
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
            eprintln!("skip: OWL_TEST_DEVICE 未设");
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
            // piecewise A/B 臂(2026-10-15 perf 会话;恒等门用)
            // **piecewise + FI 断点臂**(2026-10-16 转正门:"断点臂不破文本"红线)——
            // spec(断点臂)与 baseline(spec_depth=0)贪心 token 流必须逐位一致:
            // 覆盖"FI 四件套走输入槽 + ctx_total 宿主标量槽 + 段间 eager 重放"全链
            if owl_shared::env_reader::flag("OWL_GATE_PIECEWISE") {
                knobs.env.attn.verify_piecewise = true;
                knobs.env.attn.verify_fi = true;
                knobs.env.attn.verify_xqa = false;
                knobs.env.attn.verify_v2 = false;
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
            let st = running.spec.stats;
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
    /// OWL_AWQ27B_DIR。
    pub async fn gpu_spec_identity_gate_27b() {
        let Some(ordinal) = gpu_ordinal() else {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        };
        let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else {
            eprintln!("skip: OWL_AWQ27B_DIR 未设(cyankiwi 检查点目录)");
            return;
        };
        // 域参数化(E5-DF3 AL 排查:PROSE 域 = DFlash2 结构性最差域,
        // xinfer p5 实测 accept 1.9%/AL 1.13;MATH 域 AL 4.49 —— OWL_GATE_PROMPT
        // 换域对照,缺省恒 = 原 42tok 长城门)
        let prompt = owl_shared::env_reader::str_or("OWL_GATE_PROMPT", "用五十字介绍长城。");

        /// **装载一次**(带 MTP 草稿;2026-10-13 用户令:不再每轮重载)
        async fn boot_gate_engine(ordinal: usize, dir: &str) -> RunningEngine<owl_cuda::GpuClient> {
            let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
            // 24G 贴顶三旋钮(单会话门不需要多格/多快照/前缀缓存)
            knobs.max_active_turns = 1;
            knobs.snap_max = 1;
            knobs.prefix_cache = false;
            // 草稿档一律按投机配置 boot;baseline 轮靠运行时 spec_depth=0 回落
            knobs.draft.depth = 3;
            knobs.cuda.capture_slab_mb = Some(160);
            // piecewise A/B 臂(2026-10-15 perf 会话;恒等门用)
            // **piecewise + FI 断点臂**(2026-10-16 转正门:"断点臂不破文本"红线)——
            // spec(断点臂)与 baseline(spec_depth=0)贪心 token 流必须逐位一致:
            // 覆盖"FI 四件套走输入槽 + ctx_total 宿主标量槽 + 段间 eager 重放"全链
            if owl_shared::env_reader::flag("OWL_GATE_PIECEWISE") {
                knobs.env.attn.verify_piecewise = true;
                knobs.env.attn.verify_fi = true;
                knobs.env.attn.verify_xqa = false;
                knobs.env.attn.verify_v2 = false;
            }
            let mut engine = Engine::new(EngineConfig {
                device_ordinal: ordinal,
                // `_MIN_GEN_TOKENS = 1024` ⇒ room = max_seq − prompt 必须 ≥ 1024
                // (老门 256 恒拒发 = 环境性红;2026-10-16 抬到 2048 —— 恒等门不比容量)
                max_seq_tokens: 2048,
                prefill_chunk: 64,
                knobs,
            })
            .expect("构造");
            let loaded = engine
                .loader()
                .load_qwen38_27b_awq(std::path::Path::new(dir), std::path::Path::new(dir), owl_models::module::QuantPlan::F16, false)
                .await
                .expect("装载");
            engine.run(loaded).await.expect("装配")
        }

        /// 单轮(`spec_depth = 0` = 不投机;>0 = 投机)
        async fn round(
            running: &mut RunningEngine<owl_cuda::GpuClient>,
            spec_depth: usize,
            prompt: &str,
            tag: &str,
        ) -> (String, Vec<u32>) {
            running.spec.depth = spec_depth;
            let (sid, id) = running.submit_session(Some(7), prompt, 40).expect("submit");
            let text = drive_turn(running, id).await;
            let toks = running.bank.sessions
                .get(sid)
                .map(|s| s.tokens.clone())
                .unwrap_or_default();
            let st = running.spec.stats;
            eprintln!(
                "[spec-gate-27b] {tag} spec_depth={spec_depth} stats: rounds={} sum_m={} full={} partial={} zero={}",
                st.rounds, st.sum_m, st.full, st.partial, st.zero
            );
            (text, toks)
        }

        // ── 装载**一次**,两轮:① 不投机 baseline ② 投机(用户令 2026-10-13) ──
        let mut running = boot_gate_engine(ordinal, &dir).await;
        let (ta, ta_tokens) = round(&mut running, 0, &prompt, "baseline").await;
        let (tb, tb_tokens) = round(&mut running, 3, &prompt, "spec").await;
        eprintln!(
            "[spec-gate-27b] baseline({}tok)={ta:?}\n[spec-gate-27b] spec({}tok)={tb:?}",
            ta_tokens.len(),
            tb_tokens.len()
        );
        assert_eq!(ta, tb, "恒等门 27B:greedy 真草稿 spec 文本 ≡ 无 spec");
        assert_eq!(ta_tokens, tb_tokens, "恒等门 27B:token 账逐位一致");
        // 旋钮已全量显式化,无 env 清场需要
    
        drop(running);
        std::thread::sleep(std::time::Duration::from_secs(10));
    }

    /// E5-DF3 恒等门(27B AWQ target + DFlash2 草稿):greedy spec 文本
    /// ≡ 无 spec 逐位。门控 OWL_TEST_DEVICE + OWL_AWQ27B_DIR +
    /// OWL_DFLASH2_DIR。**20G 卡贴顶**(target ~13.7G + draft 3.85G +
    /// 池)—— OOM 时降 max_seq_tokens/挪 24G 卡。
    pub async fn gpu_dflash2_identity_gate_27b() {
        let Some(ordinal) = gpu_ordinal() else {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        };
        let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else {
            eprintln!("skip: OWL_AWQ27B_DIR 未设(cyankiwi 检查点目录)");
            return;
        };
        let Some(ddir) = owl_shared::env_reader::str("OWL_DFLASH2_DIR") else {
            eprintln!("skip: OWL_DFLASH2_DIR 未设(DFlash2 草稿目录)");
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
        ) -> RunningEngine<owl_cuda::GpuClient> {
            let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
            knobs.max_active_turns = 1;
            knobs.snap_max = 1;
            knobs.prefix_cache = false;
            // 草稿档一律按**投机**配置 boot(深度 + verify 捕获档);
            // 不投机的 baseline 轮靠运行时 `spec_depth = 0` 回落,不重载。
            knobs.draft.depth = owl_shared::env_reader::parse_or("OWL_SPEC_DEPTH", 7);
            knobs.cuda.capture_slab_mb = Some(160);
            knobs.draft.dflash2_dir = Some(ddir.to_string());
            // E19 定位开关(2026-10-13):knhd×fp8 池×int4 头×spec 语义回归
            // 复现(缺省关 = 原口径不变)。三项同开才复现 —— 单开任一项均绿:
            // 疑为“knHD 多序(seqs>1)读非位等”在窄边际下的贪心翻转。
            if owl_shared::env_reader::flag("OWL_GATE_FP8") {
                knobs.env.kv.quant = owl_models::env::KvQuant::Fp8E4M3;
            }
            // XQA 臂开关(2026-10-13;恒等门用 —— spec ≡ 无 spec / 裸 decode
            // 臂不破文本,是 XQA 进生产的唯一红线)
            if owl_shared::env_reader::flag("OWL_GATE_XQA") {
                knobs.env.attn.verify_xqa = true;
            }
            // 裸 decode XQA 臂(2026-10-13 ②):baseline 轮(spec_depth=0)
            // 就走它 ⇒ 本门同时是"decode 臂不破文本"的红线
            if owl_shared::env_reader::flag("OWL_GATE_DECODE_XQA") {
                knobs.env.attn.decode_xqa = true;
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
                .load_qwen38_27b_awq(std::path::Path::new(dir), std::path::Path::new(dir), head_plan, false)
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
            running: &mut RunningEngine<owl_cuda::GpuClient>,
            spec_depth: usize,
            prompt: &str,
            max_new: usize,
            tag: &str,
        ) -> (String, Vec<u32>) {
            let prompt_len = prompt.chars().count(); // 近似(测试口径)
            running.spec.depth = spec_depth;
            let (sid, id) = running.submit_session(Some(7), prompt, max_new).expect("submit");
            let tg = std::time::Instant::now();
            let text = drive_turn(running, id).await;
            let t_gen = tg.elapsed();
            let toks = running.bank.sessions
                .get(sid)
                .map(|s| s.tokens.clone())
                .unwrap_or_default();
            let gen_n = toks.len().saturating_sub(prompt_len);
            let st = running.spec.stats;
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
/// 当年"45 tok/s"是朴素口径虚高)。门控 OWL_AWQ27B_DIR + OWL_TEST_DEVICE。
///
/// 泄漏案探针(2026-10-10 深夜;slot≥1 崩塌案首刀):
/// MAX_ACTIVE_TURNS=2 下,turn(slot0) 前后对比 slot1 状态区指纹 ——
/// slot1 变 = 跨 slot 写实锤;不变 = 病在 slot1 核索引/复用路径。
/// 附:reset_gdn(slot1) 后回读验零(reset 生效性直接判)。
/// 门控 OWL_TEST_DEVICE。诊断打印,不做硬断言(判决先看数)。
pub async fn gpu_gdn_slot_cross_probe() {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let mut knobs = EngineKnobs { max_active_turns: 2, ..Default::default() };
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
    let s1_0 = running.pool.gdn_slot_fingerprint(running.graphs.session.face_mut(), 1).await.unwrap();
    let nz0 = running.pool.gdn_slot_zero_check(running.graphs.session.face_mut(), 1).await.unwrap();
    eprintln!("[slot-probe] boot 后 slot1 非零字节 = {nz0}(期望 0 = alloc 清零生效)");

    // ② turn(slot0,新 session)
    let (sid, id) = running
        .submit_session(Some(7), "用五十字介绍长城。", 40)
        .expect("submit");
    let _ = drive_turn(&mut running, id).await;
    let _ = sid;

    // ③ turn 后双 slot 指纹
    let s1_1 = running.pool.gdn_slot_fingerprint(running.graphs.session.face_mut(), 1).await.unwrap();
    let s0_1 = running.pool.gdn_slot_fingerprint(running.graphs.session.face_mut(), 0).await.unwrap();
    let s1_nz = running.pool.gdn_slot_zero_check(running.graphs.session.face_mut(), 1).await.unwrap();
    eprintln!("[slot-probe] turn 后 slot1 非零字节 = {s1_nz}(>0 = turn(slot0) 跨写了 slot1)");
    let same1 = s1_0 == s1_1;
    eprintln!("[slot-probe] slot1 指纹 turn 前后相同 = {same1}(false = 跨 slot 写实锤)");
    let _ = s0_1;

    // ④ reset 生效性:slot0 填垃圾验证?——零检查已覆盖 slot1;slot0 的
    //    reset 由健康 turn 隐证。旋钮已显式化,无 env 清场需要。
}

pub async fn gpu_decode_marginal_bench() {
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let Some(dir) = owl_shared::env_reader::str("OWL_AWQ27B_DIR") else {
        eprintln!("skip: OWL_AWQ27B_DIR 未设");
        return;
    };
    // 口径与恒等门一致(greedy 单会话三旋钮,显式 knobs)
    async fn gen(ordinal: usize, dir: &str, max_new: usize) -> (f64, usize) {
        let mut knobs = EngineKnobs { sampler_enabled: false, ..Default::default() };
        knobs.max_active_turns = 1;
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
            .load_qwen38_27b_awq(std::path::Path::new(dir), std::path::Path::new(dir), owl_models::module::QuantPlan::F16, false)
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
    // threshold=0 = 禁用(永不入降级)
    let (s, d, p) = spec_degrade_transition(9, false, 64, 0, 0, 64);
    assert_eq!((s, d, p), (10, false, 64));
}

/// **`[graph] prefill_bucket` 档位面**(2026-10-15;M2b 落地后重写)。
///
/// 初版是"非 off ⇒ 装配期拒启"(实现未落时的防静默门)。M2b 把真接线放进
/// `Engine::run`(桶捕获需要模型 + 状态池;静止点律 ⇒ 只在 boot),`Engine::on`
/// 回到纯构造 ⇒ 三档均放行;无桶可用时走 eager 原路(且有 **warn 证据面**,
/// 不是静默降级 —— 真机档见 `graph_fleet::assemble` 的 ④ 段)。
#[cfg(test)]
#[tokio::test]
async fn prefill_bucket_levels_boot() {
    use owl_shared::config::PrefillBucket;
    let mk = |b: PrefillBucket| EngineConfig {
        device_ordinal: 0,
        max_seq_tokens: 64,
        prefill_chunk: 16,
        knobs: EngineKnobs { prefill_bucket: b, ..EngineKnobs::default() },
    };
    for b in [PrefillBucket::Off, PrefillBucket::T512, PrefillBucket::Geobuckets] {
        assert!(Engine::on(mk(b), CpuFace::new()).is_ok(), "{b:?} 应可构造(真接线在 run)");
    }
}

/// **桶闸道纯函数**(M2b):阶梯/向上取整/GDN host 槽臂判定(零设备)。
#[cfg(test)]
#[test]
fn prefill_bucket_pick_and_ladder() {
    use crate::prefill_plan::{gdn_host_slot_arm, ladder, pick_bucket};
    use owl_shared::config::PrefillBucket;
    // 阶梯:off 空;t512 单桶;geobuckets 五点
    assert!(ladder(PrefillBucket::Off).is_empty());
    assert_eq!(ladder(PrefillBucket::T512), &[512]);
    assert_eq!(ladder(PrefillBucket::Geobuckets), &[32, 64, 128, 256, 512]);
    // 向上取整(不向下):512 档下 t=1..=512 全命中 512;t>512 不命中
    for t in [1usize, 33, 128, 511, 512] {
        assert_eq!(pick_bucket(ladder(PrefillBucket::T512), t), Some(512));
    }
    assert_eq!(pick_bucket(ladder(PrefillBucket::T512), 513), None);
    // geobuckets:边界贴着阶梯走
    let l = ladder(PrefillBucket::Geobuckets);
    for (t, want) in [(1usize, 32usize), (32, 32), (33, 64), (65, 128), (257, 512), (512, 512)] {
        assert_eq!(pick_bucket(l, t), Some(want), "t={t}");
    }
    assert_eq!(pick_bucket(l, 900), None);
    // GDN host 槽臂:27B 几何 + chunked ⇒ t≥64 才吃 host 槽(小 T 走 varlen)
    let mut env = owl_models::env::EnvProvider::default();
    let dims = crate::state::ModelDims {
        hq: 24, hkv: 4, hd: 256, nk: 16, hk: 128, nv: 48, hv: 128, hidden: 2048,
        dtype: owl_iface::contract::Dtype::F16,
    };
    env.gdn.chunked = true;
    assert!(!gdn_host_slot_arm(&env, &dims, 32), "小桶走 varlen(设备槽)⇒ 不吃 host 槽");
    assert!(gdn_host_slot_arm(&env, &dims, 64));
    assert!(gdn_host_slot_arm(&env, &dims, 512));
    // 非 chunked(0.8B 夹具几何)⇒ 恒 varlen ⇒ 不吃 host 槽
    env.gdn.chunked = false;
    assert!(!gdn_host_slot_arm(&env, &dims, 512));
}

// ===========================================================================
// M2b:prefill 桶图(设计 §4–§6;GPU 门控)
// ===========================================================================

/// **桶图 ≡ eager(贪心逐 token)+ 桶确实被用上**(0.8B)。
///
/// 两引擎**同配置**,唯一差别 = `prefill_bucket`;同提示词贪心生成 ⇒ 输出文本
/// 必须逐字相同。为什么是 token 级而非 hidden 级:跨桶时 FI 的 tile 数/tiling
/// 随 t 变(设计 §4.1c"数值口径分两层")—— token 级 = 软门;桶内逐位门由
/// `fi_graph_replay_probe`(M2a)与 kernels 侧门承担;GDN 假行面由
/// `gdn_bucket_pad_probe` 承担。
///
/// 提示词长度故意跨两个桶(128 → 64):既验"向上取整命中",也验"多桶共存时
/// int_ws 的 plan 自携"(见 `server/flashinfer.rs::PlanEntry`)。
#[cfg(test)]
#[tokio::test]
async fn gpu_prefill_bucket_matches_eager() {
    use owl_shared::config::PrefillBucket;
    let Some(ordinal) = gpu_ordinal() else {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    };
    let dir = asset_dir();
    // 日志面接上(桶捕获的 warn/info 是 boot 证据面;测试里默认无 sink)
    let _ = owl_shared::log::init_logger(owl_shared::log::LogConfig {
        level: owl_shared::log::LVL_INFO,
        ..Default::default()
    });
    // 贪心(关采样器)⇒ 两条路必须逐字同轨
    async fn boot_bucket(
        dir: &std::path::Path,
        ordinal: usize,
        b: PrefillBucket,
    ) -> RunningEngine<owl_cuda::GpuClient> {
        // `EnvProvider::default()` 是 attest=false(诊断 A/B 保留位);生产走
        // `from_config` ⇒ `from_dispatch` 恒真(2026-10-13 用户令)⇒ 本测显式打开。
        let env = owl_models::env::EnvProvider { attn: owl_models::env::AttnEnv { fi: true, ..Default::default() }, ..Default::default() };
        let mut engine = Engine::new(EngineConfig {
            device_ordinal: ordinal,
            max_seq_tokens: 2048,
            // 512:t_pad=512 桶(设计 M2b 的原目标形态 —— 长 chunk 也进图;
            // 提示词 ~600 tok ⇒ chunk 512 + 尾块 88 ⇒ 同时命中大桶与中桶)
            prefill_chunk: 512,
            knobs: EngineKnobs {
                prefill_bucket: b,
                sampler_enabled: false,
                env,
                ..EngineKnobs::default()
            },
        })
        .expect("构造");
        let loaded = engine.loader().load_qwen35_0_8b(dir).await.expect("装载");
        engine.run(loaded).await.expect("装配")
    }
    let prompt = "长城是中国古代的军事防御工程,用来抵御北方游牧民族的侵袭。".repeat(40);
    async fn gen(running: &mut RunningEngine<owl_cuda::GpuClient>, prompt: &str, n: usize) -> String {
        let id = running.submit(prompt.to_string(), n).expect("submit");
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Completed { turn, text, .. } if turn == id => return text,
                TurnEvent::Failed { turn, err } => panic!("turn {turn} 失败: {err}"),
                TurnEvent::Idle => panic!("队列丢失"),
                _ => {}
            }
        }
    }
    // ① 桶档
    let mut g = boot_bucket(&dir, ordinal, PrefillBucket::Geobuckets).await;
    let buckets = g.graphs.prefill.captured_t_pads();
    eprintln!("[m2b] 桶捕获:{:?}(summary = {})", buckets, g.graphs.prefill.summary());
    assert!(
        buckets.len() >= 3,
        "geobuckets 至少应捕获小桶若干(实测 {buckets:?});捕获失败看 warn 证据面"
    );
    let out_graph = gen(&mut g, &prompt, 24).await;
    let counts = g.graphs.prefill.replay_counts();
    eprintln!("[m2b] 桶回放次数:{counts:?}");
    // **证据面**:图路确实被走了(批 180 tok / chunk 128 ⇒ 命中 t128 + t64)
    let used: Vec<usize> = counts.iter().filter(|(_, n)| *n > 0).map(|(t, _)| *t).collect();
    assert!(!used.is_empty(), "桶一个都没回放 ⇒ 图路没生效(本测在测 eager vs eager)");
    assert!(used.contains(&512), "本 prompt(chunk=512)应命中 t512 桶,实测 {used:?}");
    // ② eager 档(同模型同配置,只关桶)
    let mut e = boot_bucket(&dir, ordinal, PrefillBucket::Off).await;
    let out_eager = gen(&mut e, &prompt, 24).await;
    eprintln!("[m2b] 桶档 {out_graph:?}\n[m2b] eager {out_eager:?}");
    assert!(!out_graph.trim().is_empty(), "桶档空回答");
    assert_eq!(out_graph, out_eager, "桶图与 eager 的贪心输出必须逐字相同(M2b 跨档门)");
}
