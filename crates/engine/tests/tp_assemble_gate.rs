//! **S4 ⑤-K2 装配门** —— 双 rank 引擎装配(真机两卡)。
//!
//! 判据(设计 `docs/arch/TP2-施工设计.md` §14.2 S4e / §15.6):
//! 1. **形状单源**:两侧分片 spec == 全量 spec 的一半(inter/heads/gdn 头),
//!    `hidden`/`vocab` 不切(几何闭合);
//! 2. **staging 双块与账**:两侧同额(布局单源 `StagingLayout`)+ 领主账
//!    `shared_bytes = alias_bytes = 总额`、`mappings = 1`;
//! 3. **别名连通(引擎级接线实证)**:rank0 写 own → rank1 经 peer 读回逐字节等
//!    (证明 own/peer 交叉接线没接反);
//! 4. **图族真捕到**(staging 块烘进捕获图 ⇒ 捕获窗内零分配的前提成立);
//! 5. **收尾闭合**:卸图 + `PeerRelease` 后 peer 三账归零(无泄漏)。
//!
//! # 运行(两张空闲卡 = **CUDA ordinal**;未设 = skip)
//!
//! ```bash
//! OWL_TP_TEST_DEV_A=0 OWL_TP_TEST_DEV_B=1 OWL_0_8B_DIR=<…/Qwen3.5-0.8B> \
//!   cargo test -p owl-engine --test tp_assemble_gate -- --nocapture
//! ```

use std::path::PathBuf;

use owl_engine::{Engine, EngineConfig, EngineKnobs, TpEngine};
use owl_iface::contract::{DeviceClient, Dtype, Shape};
use owl_models::formats::qwen35::qwen3_5_0_8b;
use owl_models::formats::spec::SpecSource;
use owl_models::model::ModelSpec;
use owl_models::plan::tp::TpPlan;

fn dev(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn dir_of() -> PathBuf {
    std::env::var("OWL_0_8B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../models/Qwen/Qwen3.5-0.8B"))
}

/// 抠 `"key":<digits>`(server LedgerJson 取数;与 cuda 门同法)
fn json_u64(json: &str, key: &str) -> u64 {
    let pat = format!("\"{key}\":");
    let i = json.find(&pat).unwrap_or_else(|| panic!("LedgerJson 缺键 {key}: {json}")) + pat.len();
    let rest = &json[i..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().unwrap_or_else(|e| panic!("键 {key} 数值非法: {e}"))
}

fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn cfg_of(ordinal: usize) -> EngineConfig {
    EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 4096,
        prefill_chunk: 128,
        knobs: EngineKnobs::default(),
    }
}

#[test]
fn tp_assemble_gate() {
    let (Some(oa), Some(ob)) = (dev("OWL_TP_TEST_DEV_A"), dev("OWL_TP_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_TP_TEST_DEV_A / OWL_TP_TEST_DEV_B(两张空闲卡的 CUDA ordinal)");
        return;
    };
    assert_ne!(oa, ob, "A/B 必须是两张不同卡");
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        eprintln!("skip: 0.8B 检查点不在 {dir:?}(设 OWL_0_8B_DIR)");
        return;
    }
    // 大盘装配:大栈 + current_thread rt(与 tp_load_gate 同律)
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(body(oa, ob, dir))
        })
        .expect("起测试线程")
        .join()
        .expect("装配门失败");
}

async fn body(oa: usize, ob: usize, dir: PathBuf) {
    // 标记面(K1 的 rank 打点):本门顺带验"注入确实发生在每 rank"——
    // 两 rank 的 AR 注入计数应相等且 > 0(模型对称 ⇒ 注入对称)。
    let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
    // ── ① 两 rank 引擎壳 + 分片装载(确定性分片协议:各装 1/2)────────
    let mut e0 = Engine::new(cfg_of(oa)).expect("engine(rank0)");
    let l0 = e0
        .loader()
        .slice(0, 2)
        .load_qwen35_0_8b(&dir)
        .await
        .expect("rank0 分片装载");
    eprintln!("[tp-gate] rank0 装载完成(1/2)");
    let mut e1 = Engine::new(cfg_of(ob)).expect("engine(rank1)");
    let l1 = e1
        .loader()
        .slice(1, 2)
        .load_qwen35_0_8b(&dir)
        .await
        .expect("rank1 分片装载");
    eprintln!("[tp-gate] rank1 装载完成(1/2)");

    // ── ② 装配(分档 → staging 握手 → 每 rank boot_rank)────────────
    let tp = TpEngine::assemble(vec![e0, e1], vec![l0, l1]).await.expect("TP 装配");

    // ── ③ 形状单源:分片 spec == 全量的一半(CPU 侧读 config;零 GPU)──
    let (full, _rep) = ModelSpec::from_ckpt(&dir, SpecSource::family(&qwen3_5_0_8b()))
        .expect("全量 spec(CPU 只读回退)");
    for r in 0..2 {
        let s = tp.spec(r);
        assert_eq!(tp.plan(r), TpPlan::new(r, 2).expect("plan"), "rank{r} 的分片计划");
        assert_eq!(s.inter * 2, full.inter, "rank{r}:inter 应恰半分");
        assert_eq!(s.full_heads.0 * 2, full.full_heads.0, "rank{r}:hq 半分");
        assert_eq!(s.full_heads.1 * 2, full.full_heads.1, "rank{r}:hkv 半分");
        assert_eq!(s.gdn_heads.0 * 2, full.gdn_heads.0, "rank{r}:nk 半分");
        assert_eq!(s.gdn_heads.2 * 2, full.gdn_heads.2, "rank{r}:nv 半分");
        assert_eq!(s.hidden, full.hidden, "hidden 不切(层间通道全宽)");
        assert_eq!(s.vocab, full.vocab, "v1 词表复制(AG 词表切分是将来项)");
        assert_eq!(s.dtype, full.dtype);
    }
    assert_eq!(tp.spec(0).inter, tp.spec(1).inter, "两 rank 同形(分片协议对称)");

    // ── ④ staging 账(两侧同额 + 领主账)────────────────────────────
    let layout = owl_cuda::comm::ar::StagingLayout::default_layout();
    for r in 0..2 {
        assert_eq!(tp.staging(r).total_bytes(), layout.total_bytes(), "rank{r} staging 额度");
        assert_eq!(tp.slot_bytes(), layout.slot_bytes, "slot 两侧同参");
    }
    // 账本口径:VMM 分配按**粒度上取整**(min granularity = 2MiB;
    // `vmm::round_up` 的例子就是本 staging 量 ⇒ 64MiB+64B → 66MiB)。
    // 断言取不变量(粒度无关):两侧对称 / 不小于请求 / 超出 < 2MiB /
    // 各自的 alias == 对端的 shared(交叉映射是同一笔分配)。
    let js: Vec<String> = {
        let mut v = Vec::new();
        for r in 0..2 {
            // `ledger_json` 是 GpuClient 固有面(peer 账房)→ 走可变直通口
            let mut f = tp.face_clone(r).expect("face clone");
            v.push(f.client_mut().ledger_json().await.expect("ledger_json"));
        }
        v
    };
    // ★ **2026-10-18:1-stage arena 另有 3 块/侧**(2 payload 槽 + 1 header)
    // —— 见 `TP2-施工设计.md` §35 与 `tp.rs::handshake_arena`。账本断言必须把
    //   两笔都算进来(否则门会把"多出的合法分配"当成漂移)。
    let with_arena = matches!(std::env::var("OWL_TP_1STAGE").as_deref(), Ok("1") | Ok("true"));
    let arena_bytes = if with_arena { 2 * layout.slot_bytes + owl_cuda::comm::ar::HEADER_BYTES } else { 0 };
    let per_rank_req = layout.total_bytes() + arena_bytes;
    let shared: Vec<u64> = js.iter().map(|j| json_u64(j, "shared_bytes")).collect();
    assert_eq!(shared[0], shared[1], "两侧分配额必须对称");
    for r in 0..2 {
        assert!(shared[r] >= per_rank_req, "rank{r} 账不得小于请求(staging + arena)");
        assert!(
            shared[r] - per_rank_req < (4 << 20),
            "rank{r} 超出部分应是粒度上取整(< 4MiB;staging 与 arena 各自的 2MiB 取整);实得 {}B",
            shared[r]
        );
        assert_eq!(json_u64(&js[r], "alias_bytes"), shared[1 - r], "rank{r} alias 应等于对端分配额");
        // 1 staging 分配 + 1 别名 +（2 槽 + 1 header)×(1 分配 + 1 别名)= 8
        let want_map = if with_arena { 8 } else { 2 };
        assert_eq!(json_u64(&js[r], "mappings"), want_map, "rank{r} 映射笔数 = staging 2 + arena 6(开启时)");
    }
    // ── ★ 1-stage arena:两档都断言(2026-10-18;缺省关,`OWL_TP_1STAGE=1` 开)──
    // 缺省(v1 两段):**不得**有 arena(多占 64MiB/rank 的东西不许悄悄上场)。
    // 开启:槽存在、容量合理、两个 header 不同块。
    let expect_arena = matches!(std::env::var("OWL_TP_1STAGE").as_deref(), Ok("1") | Ok("true"));
    for r in 0..2 {
        match tp.staging(r).arena() {
            Some(a) => {
                assert!(expect_arena, "rank{r} 有 arena 但 `OWL_TP_1STAGE` 没开(缺省必须关)");
                assert_eq!(a.own_slots.len(), 2, "rank{r} 应有 2 个 payload 槽(乒乓)");
                assert!(a.slot_bytes >= 1 << 20, "rank{r} 槽容量异常");
                assert_ne!(a.own_hdr.id, a.peer_hdr.id, "rank{r} header 与对端 header 必须是两块");
            }
            None => assert!(!expect_arena, "rank{r} 开了 `OWL_TP_1STAGE` 却没有 arena"),
        }
    }

    // ── ⑤ 别名连通实证:rank0 写 own → rank1 经 peer 读回 ───────────
    let pattern: [f32; 4] = [1.5, -2.25, 3.125, 42.0];
    let mut f0 = tp.face_clone(0).expect("face0");
    let src = f0
        .htod(Dtype::F32, &Shape::from(vec![4]), &le_f32(&pattern), "tp-gate:pattern")
        .await
        .expect("htod 模式块");
    let own0 = tp.staging(0).own().clone();
    f0.copy_block_at(&src, 0, &own0, 0, 16).await.expect("copy → rank0.own");
    f0.sync().await.expect("sync");
    let mut f1 = tp.face_clone(1).expect("face1");
    let mut got = vec![0u8; 16];
    f1.dtoh(tp.staging(1).peer(), &mut got).await.expect("dtoh(rank1.peer)");
    assert_eq!(got, le_f32(&pattern), "rank1 的 peer 别名应是 rank0 的 own(交叉接线错位?)");
    f0.free(&[src.id]).await.expect("free 模式块");

    // ── ⑥ 图族真捕到(staging 块烘进图 ⇒ 捕获窗零分配的前提成立)────
    for r in 0..2 {
        assert!(tp.session_captured(r), "rank{r} 主会话图应捕获成功");
        assert!(tp.session_slab_bytes(r) > 0, "rank{r} 图 slab 应 > 0");
        eprintln!("[tp-gate] rank{r} 会话图 slab = {} MiB", tp.session_slab_bytes(r) >> 20);
    }
    // 注入计数(K1 标记面):两 rank 各自有 AR 注入,且计数相等(模型对称)
    let ar_count = |tag: &str| -> u64 {
        owl_shared::metrics::query_metrics(&owl_shared::metrics::MetricsFilter::new().tag(tag))
            .iter()
            .map(|r| r.counter)
            .sum()
    };
    let (c0, c1) = (ar_count("tp.ar.r0"), ar_count("tp.ar.r1"));
    assert!(c0 > 0 && c1 > 0, "两 rank 都应有 AR 注入(实得 r0={c0} r1={c1})");
    assert_eq!(c0, c1, "对称模型 ⇒ 两 rank 注入次数相等(实得 r0={c0} r1={c1})");
    eprintln!("[tp-gate] AR 注入计数:r0={c0} r1={c1}(捕获期已烙进图)");

    // ── ⑦ 收尾:卸图 + PeerRelease ⇒ peer 三账归零 ─────────────────
    let k0 = tp.face_clone(0).expect("face0(收尾)");
    let k1 = tp.face_clone(1).expect("face1(收尾)");
    tp.shutdown().await.expect("TP 收尾");
    for (tag, mut f) in [("rank0", k0), ("rank1", k1)] {
        let j = f.client_mut().ledger_json().await.expect("ledger_json(after shutdown)");
        assert_eq!(json_u64(&j, "shared_bytes"), 0, "{tag} shared 未归零(泄漏)");
        assert_eq!(json_u64(&j, "alias_bytes"), 0, "{tag} alias 未归零(泄漏)");
        assert_eq!(json_u64(&j, "mappings"), 0, "{tag} mappings 未归零(泄漏)");
    }
    eprintln!(
        "[tp-gate] 全绿:world=2 staging={}MiB/rank(账 {}MiB)别名连通 ✓ 双图捕获 ✓ 收尾三账归零 ✓",
        layout.total_bytes() >> 20,
        shared[0] >> 20
    );
}
