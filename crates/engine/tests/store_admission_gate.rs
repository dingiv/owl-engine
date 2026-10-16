//! **S3b 准入门(§23.6 新门)**:H1 判决在引擎真路径上的行为 —— 无卡可跑
//! (CPU mock 档;账与搬运真、数值不算)。
//!
//! 三条门(设计卷 §23.4/§23.6):
//! ① **先腾后进**:池满但可动面够 ⇒ 判 `EvictThen` ⇒ 段边界**主动下沉**冷链
//!    ⇒ 新 turn 准入成功(不是等 `ensure_for_len` 兜底驱逐);
//! ② **池墙拒**:池被**不可动面**(idle 会话持着的历史,引用 > 1)占满 ⇒
//!    `Reject` ⇒ `TurnEvent::Failed` 带数字,**且文案不带 "超预算"**
//!    (那是 context-overflow 专用标记,服务面据此触发客户端 compact;
//!    池墙拒是另一类,客户端该降并发/重试);
//! ③ **不靠兜底**:全程 `ops.ensure_evicted == 0`(需求驱动/准入判决到位时,
//!    同步兜底驱逐应为空 —— §17.4 的指纹)。
//!
//! **档位包络(必须说清)**:本档跑在 CPU mock 上,只吃**单 prefill chunk**
//! (prompt ≤ `prefill_chunk` = 32 tok)—— 多 chunk 的真前向是 §18.5.1 的
//! 已挂账缺口(dtype 口径),不在本刀范围。故池拧到最小档(2 槽)。
//!
//! 跑法(默认 skip;与 cpu_mock_smoke 同纪律):
//! ```bash
//! OWL_CPU_MOCK=1 cargo test -p owl-engine --test store_admission_gate -- --nocapture
//! ```

use owl_cpu::{CpuFace, MockLimits};
use owl_engine::{Engine, EngineConfig, EngineKnobs, TurnEvent};

fn asset_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/assets/Qwen3.5-0.8B")
}

fn skip() -> bool {
    if std::env::var_os("OWL_CPU_MOCK").is_none() {
        eprintln!("skip: OWL_CPU_MOCK 未设(CPU mock 冒烟按需开)");
        return true;
    }
    false
}

/// 数字序列 prompt。**实测锚(单 chunk 档的硬边界)**:n=4 ⇒ prompt 27 tok
/// (模板开销 ≈ 12),n=6 ⇒ 35 tok 已越 `prefill_chunk = 32`。
///
/// ⚠️ 为什么死死咬住 32:CPU mock 档的**多 chunk 真前向**炸在
/// `prefill_chunk.rs` 中间块收割(`d.hidden × esz`,esz 按模型 dtype)与
/// CPU 面"值块恒 f32"的口径差上 —— 那是 §18.5.1 的**已挂账缺口**
/// (裁决:不追),不在本刀范围。本门要验的是准入判决,故把负载压在单 chunk 内。
fn num_prompt(base: u32, n: u32) -> String {
    (0..n).map(|i| ((base + i) * 7 % 997).to_string()).collect::<Vec<_>>().join(" ")
}

/// 池 = 8 槽(下限 = 单会话容量 max_seq/32);prompt 全部 ≤ 32 tok(单 chunk)
const MAX_SEQ: usize = 256;
const POOL_TOKENS: usize = 256;
/// 数字 prompt 的实测锚(n=4 ⇒ 27 tok;模板开销 ≈ 12 ⇒ 每词 ≈ 4 tok)
const WORDS: u32 = 4;

/// 装配一个最小档(8 槽池 + 开宿主层 ⇒ 准入判决才看得到池墙)
async fn boot() -> owl_engine::RunningEngine<CpuFace> {
    boot_rows(12, owl_store::QueuePolicy::Reject).await
}

async fn boot_with(on_full: owl_store::QueuePolicy) -> owl_engine::RunningEngine<CpuFace> {
    boot_rows(12, on_full).await
}

/// `rows` = GDN 行池大小(= 并发 turn 天花板,`max_active_turns`)
async fn boot_rows(
    rows: usize,
    on_full: owl_store::QueuePolicy,
) -> owl_engine::RunningEngine<CpuFace> {
    let cfg = EngineConfig {
        device_ordinal: 0,
        max_seq_tokens: MAX_SEQ,
        prefill_chunk: 32,
        knobs: EngineKnobs {
            sampler_enabled: false,
            pool_tokens: Some(POOL_TOKENS),
            host_cache_mb: 16, // 整条链(8 块 ≈ 6.3MB)要放得下宿主层
            // 会话不再占行(S2b)⇒ 这一项现在只是"行池大小"(并发 turn 天花板)
            max_active_turns: rows,
            admission_on_full: on_full,
            ..Default::default()
        },
    };
    let face =
        CpuFace::mock(MockLimits { null_kernels: true, graph_capture: true, ..Default::default() });
    let mut engine = Engine::on(cfg, face).expect("构造(CPU face)");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    engine.run(loaded).await.expect("装配")
}

/// 提交一个 turn 并泵到终态;返回(事件, prompt token 数)
async fn run_turn(
    running: &mut owl_engine::RunningEngine<CpuFace>,
    session: Option<u64>,
    base: u32,
    n: u32,
    max_new: usize,
) -> (TurnEvent, usize) {
    let (_sid, tid) = running
        .submit_session_opts(session, num_prompt(base, n), max_new, Default::default())
        .expect("submit");
    let pt = running.turn_prompt_len(tid).unwrap_or(0);
    loop {
        match running.pump().await.expect("pump") {
            ev @ (TurnEvent::Completed { turn, .. } | TurnEvent::Failed { turn, .. })
                if turn == tid =>
            {
                return (ev, pt)
            }
            TurnEvent::Idle => panic!("泵到 Idle 而 turn {tid} 未终了(准入排队中?)"),
            _ => {}
        }
    }
}

/// **门①先腾后进**:池被**可动**缓存占满 ⇒ `EvictThen` ⇒ 主动下沉 ⇒ 准入。
#[tokio::test(flavor = "current_thread")]
async fn cold_cache_is_offloaded_before_the_new_turn_is_admitted() {
    if skip() {
        return;
    }
    let mut running = boot().await;
    let g = running.leg_stats();
    eprintln!(
        "[gate] 池: {} 槽 × {} B(page {} tok) · 宿主预算 {}B",
        g.slots_total, g.block_bytes, g.block_tokens, g.host_budget_bytes
    );

    // 轮 1:把池吃满(prompt 单 chunk;靠 ignore_eos 跑满生成长度)——
    // 临时会话 ⇒ 终了后整条链变**可动缓存**
    let (ev1, n1) = run_turn(&mut running, None, 0, WORDS, 224).await;
    let r1 = running.leg_stats();
    eprintln!(
        "[gate] 轮 1: prompt={n1} tok · 设备 {}/{} 槽 · 可动 {}B ⇒ {ev1:?}",
        (r1.slots_total - r1.slots_free),
        r1.slots_total,
        r1.movable_bytes
    );
    assert!(matches!(ev1, TurnEvent::Completed { .. }), "轮 1 应完成:{ev1:?}");
    let off_before = r1.ops.offloads;

    // 轮 2(不同前缀 ⇒ 不复用)⇒ 池不够但**可动面够** ⇒ 应先腾后进
    let (ev2, n2) = run_turn(&mut running, None, 100, WORDS, 32).await;
    let r2 = running.leg_stats();
    eprintln!(
        "[gate] 轮 2: prompt={n2} tok · 换出 {}(+{}) 条 · 回灌 {} · 兜底驱逐 {} · 宿主 {}/{} 槽",
        r2.ops.offloads,
        r2.ops.offloads - off_before,
        r2.ops.onloads,
        r2.ops.ensure_evicted,
        r2.host_blocks,
        r2.host_budget_bytes / r2.block_bytes.max(1)
    );
    assert!(matches!(ev2, TurnEvent::Completed { .. }), "轮 2 应完成:{ev2:?}");
    assert!(
        r2.ops.offloads > off_before,
        "轮 2 准入应触发**主动下沉**(EvictThen),实际未搬:{}",
        r2.ops.offloads
    );
    assert_eq!(r2.ops.ensure_evicted, 0, "应走判决腾位,而不是同步兜底驱逐");
    assert_eq!(running.store_report().active_turns, 0, "turn 终了应还保留额");
}

/// **门②idle 会话占满池 ⇒ 先腾后进(不再拒)** —— S2 引用拆分(§25.1①)的正面证据。
///
/// S3b 拍这个场景时:8 个键控会话各持一段历史(会话钉着设备表 ⇒ 不可动)⇒ 第九个
/// turn 撞**池墙被拒**。S2 之后 idle 会话**不持设备表**(历史 = 纯缓存)⇒ 同样的
/// 场景变成「**整池可动** ⇒ `EvictThen` ⇒ 下沉前八个会话的历史 ⇒ 准入成功」。
///
/// 这条门的含义:**池墙不再是"拒"的理由**(池里没有不可动的内容时,腾位总能解决)。
/// 拒/排队的**机制**仍在(store 侧单测:`budget::tests::pool_face_is_the_admission_wall`
/// / `committed_is_exactly_pool_occupancy` / `concurrency_ceiling_is_hard`),可达条件是
/// "池里确有不可动面" —— 本刀之后该条件只可能来自**多活跃 turn**(M2 批调度)或
/// 未来新腿,引擎侧单活跃态构造不出来(诚实记录,不造人工场景)。
#[tokio::test(flavor = "current_thread")]
async fn idle_sessions_hold_nothing_so_the_wall_becomes_evictable() {
    if skip() {
        return;
    }
    let mut running = boot().await;

    // 八个键控会话各留一段历史(每会话 1 整块 ⇒ 占满 8 槽池)。
    // ⚠️ 生成长度要够长:历史**只有整块进缓存**(`register_chain` 只登记整块),
    // prompt(≈27)+生成 < 32 ⇒ 一个整块都没有,表一释放内容就没了 ——
    // 这正是 S2 之后"块对齐"的直接可见后果(§25.1①)。
    for (k, sid) in (1u64..=8).enumerate() {
        let base = (k as u32) * 200;
        let (ev, _n) = run_turn(&mut running, Some(sid), base, WORDS, 12).await;
        assert!(matches!(ev, TurnEvent::Completed { .. }), "会话 {sid} 应完成:{ev:?}");
    }
    let leg = running.leg_stats();
    let rep = running.store_report();
    eprintln!(
        "[gate] 八会话后: 设备 {}B / 可动 {}B · 保留 {}B · 会话持有 {}B · 缓存 {}B",
        leg.gpu_bytes, leg.movable_bytes, rep.reserved_bytes, rep.session_held_bytes, rep.cache_bytes
    );
    // 池要**接近满**(八段历史 ⇒ 六~八槽;首块含同一份 chat 模板 ⇒ 内容寻址下
    // 部分会话共享首块,槽数略少于会话数是**正确行为**)
    let used = leg.slots_total - leg.slots_free;
    assert!(used + 1 >= leg.slots_total, "八段历史应把池占到近满(实际 {used}/{})", leg.slots_total);
    assert_eq!(rep.session_held_bytes, 0, "S2:idle 会话不再钉住设备面(会话持有面 = 0)");
    assert_eq!(
        leg.movable_bytes, leg.gpu_bytes,
        "S2:池里的历史全进**可动面**(refs = 1 的纯缓存)"
    );

    // 第九个会话:同一场景,现在是"先腾后进"而不是拒
    let off_before = leg.ops.offloads;
    let (ev, n) = run_turn(&mut running, Some(9), 1800, WORDS, 12).await;
    let leg2 = running.leg_stats();
    eprintln!(
        "[gate] 第九会话: prompt={n} tok · 换出 {}(+{}) · 兜底驱逐 {} ⇒ {ev:?}",
        leg2.ops.offloads,
        leg2.ops.offloads - off_before,
        leg2.ops.ensure_evicted
    );
    assert!(matches!(ev, TurnEvent::Completed { .. }), "应先腾后进(不再拒):{ev:?}");
    assert!(leg2.ops.offloads > off_before, "腾位应真下沉(宿主面开着)");
    assert_eq!(leg2.ops.ensure_evicted, 0, "应走判决腾位,而不是同步兜底驱逐");
    assert_eq!(running.store_report().active_turns, 0, "turn 终了应还保留额");
}

/// **门④地板语义**:本刀只实做 `Strict` —— 非 `Strict` 在两处**响亮拒**
/// (boot 的 knobs / submit 的 `TurnOpts`),不许静默当 Strict 用(§23.5)。
#[tokio::test(flavor = "current_thread")]
async fn non_strict_dense_floor_is_refused_loudly() {
    if skip() {
        return;
    }
    // ① boot 面的 knobs
    let cfg = EngineConfig {
        device_ordinal: 0,
        max_seq_tokens: 256,
        prefill_chunk: 32,
        knobs: EngineKnobs {
            sampler_enabled: false,
            dense_floor: owl_store::TurnFloor::Bytes(1 << 20),
            ..Default::default()
        },
    };
    let face = CpuFace::mock(MockLimits { null_kernels: true, graph_capture: true, ..Default::default() });
    let mut engine = Engine::on(cfg, face).expect("构造");
    let loaded = engine.loader().load_qwen35_0_8b(&asset_dir()).await.expect("装载");
    let err = match engine.run(loaded).await {
        Ok(_) => panic!("非 Strict 地板必须拒启"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    eprintln!("[gate] 非 Strict 拒启原文: {msg}");
    assert!(msg.contains("dense_floor"), "{msg}");

    // ② 请求面(per-turn)
    let mut running = boot().await;
    let err = running
        .submit_session_opts(None, num_prompt(0, WORDS), 5, owl_engine::TurnOpts {
            dense_floor: Some(owl_store::TurnFloor::Bytes(1 << 20)),
            ..Default::default()
        })
        .expect_err("请求面非 Strict 必须拒");
    let msg = format!("{err}");
    eprintln!("[gate] 非 Strict 请求拒原文: {msg}");
    assert!(msg.contains("选择层") || msg.contains("Strict"), "{msg}");
}

/// **门⑤(S2b):idle 会话不占 GDN 行** —— "能挂多少会话"与"能并发几个 turn"解耦。
///
/// 行池只给 **2 行**,却挂 **6 个键控会话**:每个 turn 起借行、止还行;
/// 会话本身只留 `tokens`(身份),状态走**边界快照**(S2 三层化)。
#[tokio::test(flavor = "current_thread")]
async fn idle_sessions_do_not_hold_gdn_rows() {
    if skip() {
        return;
    }
    let mut running = boot_rows(2, owl_store::QueuePolicy::Reject).await;
    let rows_total = running.store_report().state.rows_total;
    assert_eq!(rows_total, 2, "本档行池 = 2");

    for (k, sid) in (1u64..=6).enumerate() {
        let base = (k as u32) * 200;
        let (ev, _n) = run_turn(&mut running, Some(sid), base, WORDS, 12).await;
        assert!(matches!(ev, TurnEvent::Completed { .. }), "会话 {sid} 应完成:{ev:?}");
        let st = running.store_report().state;
        assert_eq!(st.rows_free, rows_total, "turn 止应把行还回(会话 {sid} 后)");
        eprintln!(
            "[gate] 会话 {sid}:行 {}/{} 空闲 · 快照 设备 {:?} / 宿主 {:?}",
            st.rows_free, st.rows_total, st.snap_gpu, st.snap_host
        );
    }
    // 六个会话的**状态**留在快照账里(不是留在行里)
    let st = running.store_report().state;
    assert!(
        st.snap_gpu.0 + st.snap_host.0 > 0,
        "idle 会话的状态应走快照账(行已归还)"
    );
}
