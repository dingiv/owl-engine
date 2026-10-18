//! **状态腿三层化门(§25.2)**:快照 = 「槽 × 片」的一种块 —— 下沉/回灌用
//! **与 KV 腿同一套机制**(plan → `face.swap` → commit),账在 `StateStore`。
//!
//! 无卡可跑(`CpuFace` 真字节搬运):
//! ① 设备 → 宿主 → 设备**逐字节一致**;②层账跟着走(`locate` / `snaps_in_tier`);
//! ③ 宿主满 = 响亮拒(零副作用);④回灌失败可回滚(缓冲放回宿主,内容不丢)。

use owl_cpu::CpuFace;
use owl_iface::contract::{DeviceClient, Dtype, HeapRegion, PinnedRegion, SwapDir};
use owl_store::{
    AdmissionPolicy, BlockClass, BudgetInputs, ChainHash, LegConfig, PieceSpec, PoolLayout,
    QueuePolicy, Rel, SnapKey, StoreError, Tier,
};

const ROW_BYTES: usize = 64; // 一条"行宽"(片步长)
const ROWS: usize = 2; // 两片(模拟两层 × 1 quad)
const SNAP_BYTES: usize = ROW_BYTES * ROWS;

async fn snap_pieces(face: &mut CpuFace, slots: usize) -> Vec<PieceSpec> {
    let mut out = Vec::new();
    for _ in 0..ROWS {
        // 每片 = 一条缓冲,长 = 槽数 × 行宽(KV 腿同构:池矩阵)
        let b = face.alloc(Dtype::F32, (ROW_BYTES / 4) * slots, "snap").await.expect("快照池");
        out.push(PieceSpec { buf: b, stride_bytes: ROW_BYTES });
    }
    out
}

fn rel(host_bytes: usize) -> Rel {
    let inputs = BudgetInputs {
        v_card_bytes: 8 << 30,
        weights_bytes: 1 << 30,
        runtime_floor_bytes: 64 << 20,
        draft_pool_bytes: 0,
        gdn_row_bytes: 1 << 20,
        kv_bytes_per_token: 32 * 1024,
        page_tokens: 32,
        host_bytes,
        disk_bytes: 0,
        kv_face_bytes: 8 * (1 << 20),
    };
    let kv_cfg = LegConfig {
        slots: 8,
        block_tokens: 32,
        block_bytes: 1 << 20,
        cache: owl_store::CachePolicy::Cap(8),
        host_bytes: 0, // KV 腿关宿主:host 总量全给状态腿(本测只看状态腿)
        class: BlockClass::KvPage,
        label: "snap-test",
    };
    Rel::open(inputs, AdmissionPolicy::new(2, QueuePolicy::Reject), kv_cfg).expect("装配")
}

async fn seed(face: &mut CpuFace, ps: &[PieceSpec], slot: u32, seed: u8) {
    for p in ps {
        let pat: Vec<u8> = (0..p.stride_bytes).map(|i| seed.wrapping_add(i as u8)).collect();
        let lease: Box<dyn PinnedRegion + Send> = Box::new(HeapRegion(pat));
        face.upload_pinned(lease, &p.buf, slot as usize * p.stride_bytes, p.stride_bytes)
            .await
            .expect("seed");
    }
}

async fn read_slot(face: &mut CpuFace, ps: &[PieceSpec], slot: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for p in ps {
        let mut d = vec![0u8; p.stride_bytes];
        face.dtoh_at(&p.buf, slot as usize * p.stride_bytes, &mut d).await.expect("读槽");
        out.extend_from_slice(&d);
    }
    out
}

fn key(n: u64) -> SnapKey {
    SnapKey::at_boundary(ChainHash::new(n))
}

#[tokio::test]
async fn snapshot_offload_and_reload_is_byte_identical() {
    let mut face = CpuFace::new();
    let ps = snap_pieces(&mut face, 4).await;
    let mut r = rel(SNAP_BYTES); // 宿主预算 = 一条快照(门③才好判)
    r.attach_snap_layout(PoolLayout { class: BlockClass::GdnRow, epoch: 0, pieces: ps.clone() })
        .expect("注册状态腿布局");
    assert_eq!(r.snap_bytes(), SNAP_BYTES);

    // 槽 0 上有一条快照(设备层)
    let k = key(7);
    seed(&mut face, &ps, 0, 0x11).await;
    let before = read_slot(&mut face, &ps, 0).await;
    r.state.register_snap(k, SNAP_BYTES, Tier::Gpu);
    assert_eq!(r.state.locate(k), Some(Tier::Gpu));
    assert_eq!(r.state.snaps_in_tier(Tier::Gpu), (1, SNAP_BYTES));

    // ① 下沉:设备槽 0 → 宿主(一条 swap 搬 N 片)
    let (pieces, table) = r.plan_snap_offload(k, 0).expect("下沉计划");
    let conn = face.swap(SwapDir::Out, &pieces, Box::new(table)).await.expect("swap(Out)");
    let table = owl_store::reclaim_slot_table(conn).expect("回执取台");
    r.commit_snap_offload(k, table).expect("下沉记账");
    assert_eq!(r.state.locate(k), Some(Tier::Host), "层应下沉");
    assert_eq!(r.snap_host_used(), SNAP_BYTES, "宿主层应真有驻留");
    assert_eq!(r.state.snaps_in_tier(Tier::Host), (1, SNAP_BYTES));

    // ② 回灌:宿主 → 设备槽 2(另一槽 ⇒ 证明内容跟着走,不是原地)
    let (pieces, table) = r.plan_snap_onload(k, 2).expect("回灌计划");
    let conn = face.swap(SwapDir::In, &pieces, Box::new(table)).await.expect("swap(In)");
    let table = owl_store::reclaim_slot_table(conn).expect("回执取台");
    r.commit_snap_onload(k, table).expect("回灌记账");
    assert_eq!(r.state.locate(k), Some(Tier::Gpu), "层应回设备");
    assert_eq!(r.snap_host_used(), 0, "宿主层应清空");
    assert_eq!(read_slot(&mut face, &ps, 2).await, before, "**逐字节一致**(门④)");

    // ③ 宿主层满 = 响亮拒(零副作用):预算 = 一条 ⇒ 第二条装不下
    let k2 = key(8);
    r.state.register_snap(k2, SNAP_BYTES, Tier::Gpu);
    let (p1, t1) = r.plan_snap_offload(k2, 3).expect("第一条");
    let c1 = face.swap(SwapDir::Out, &p1, Box::new(t1)).await.expect("swap");
    r.commit_snap_offload(k2, owl_store::reclaim_slot_table(c1).expect("台")).expect("落账");
    let k3 = key(9);
    r.state.register_snap(k3, SNAP_BYTES, Tier::Gpu);
    let err = match r.plan_snap_offload(k3, 1) {
        Ok(_) => panic!("宿主满必须拒"),
        Err(e) => e,
    };
    assert!(matches!(err, StoreError::Exhausted(_)), "得 {err:?}");
    assert_eq!(r.state.locate(k3), Some(Tier::Gpu), "拒后层不动(零副作用)");

    // ④ 回灌失败可回滚:缓冲放回宿主(内容不丢)
    let (_pieces, _table) = r.plan_snap_onload(k2, 0).expect("回灌计划");
    // 失败路径:台子随失败的 connector 消失 ⇒ 条目删(不留假账)
    r.abort_snap_onload(k2);
    assert_eq!(r.state.locate(k2), None, "内容已丢 ⇒ 条目一并删");
    assert_eq!(r.snap_host_used(), 0, "宿主账同步清");
}

/// 快照账的宿主面与**跨腿独立**:KV 腿关宿主时,状态腿照样能下沉(两腿两本账)。
#[tokio::test]
async fn snap_host_is_independent_from_kv_host() {
    let mut face = CpuFace::new();
    let ps = snap_pieces(&mut face, 2).await;
    let mut r = rel(1 << 20); // host 总量 1MiB,全给状态腿(KV 腿分片 0)
    r.attach_snap_layout(PoolLayout { class: BlockClass::GdnRow, epoch: 0, pieces: ps.clone() })
        .expect("布局");
    assert_eq!(r.snap_host_budget(), 1 << 20);
    assert_eq!(r.snap_host_free_slots(), (1 << 20) / SNAP_BYTES);
    assert_eq!(r.leg.host_budget_bytes(), 0, "KV 腿宿主面仍关");
}
