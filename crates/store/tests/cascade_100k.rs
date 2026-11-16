//! **手工级联门（100K token 尺度）**：GPU → Host → Disk → Drop。
//!
//! 口径：**4 GiB = 100K token**（约 42.9 KB/token），`page = 1024 token`
//! ⇒ 单页约 42 MiB；三层都取 **100 页 ≈ 100K token ≈ 4 GiB**。
//! 这是手工门（`--ignored` 运行）：会真实分配约 4 GiB 宿主缓冲，**不进常规套件**。
//! 不碰真实显存/磁盘；只验证 store 的级联判决。

use owl_store::{
    plan_capacity, BlockClass, BlockLeg, CachePolicy, ChainHash, DiskTier, HostBlock, LegConfig,
    Need, NeedWhy, Room, Sink, StateStore,
};

const PAGE_TOKENS: usize = 1024;
/// 4 GiB / 100K token ≈ 42.9 KB/token。
const KV_BYTES_PER_TOKEN: usize = (4 * 1024 * 1024 * 1024) / 100_000;
const BLOCK_BYTES: usize = PAGE_TOKENS * KV_BYTES_PER_TOKEN;
const PAGES: usize = 100; // 102,400 token ≈ 100K token / 层

fn toks(base: u32) -> Vec<u32> {
    (base..base + PAGE_TOKENS as u32).collect()
}

fn add_chain(l: &mut BlockLeg, base: u32) -> ChainHash {
    let mut table = Vec::new();
    l.ensure_for_len(&mut table, PAGE_TOKENS).expect("设备页");
    let t = toks(base);
    l.register_chain(&t, &table).expect("登记链");
    l.release_table(&mut table).expect("归还表");
    l.tail_hash(&t, 1).expect("链尾")
}

#[test]
#[ignore = "手工门：真实分配约 4 GiB 宿主缓冲；按需 --ignored 运行"]
fn cascade_gpu_host_disk_drop_at_100k_tokens() {
    let mut l = BlockLeg::open(LegConfig {
        slots: PAGES,
        block_tokens: PAGE_TOKENS,
        block_bytes: BLOCK_BYTES,
        cache: CachePolicy::Cap(PAGES * 4),
        host_bytes: PAGES * BLOCK_BYTES,
        class: BlockClass::KvPage,
        label: "cascade-100k",
    })
    .expect("开腿");

    // ① 设备满：100 条 1 页链。
    let a: Vec<ChainHash> = (0..PAGES)
        .map(|i| add_chain(&mut l, 10_000 + (i as u32) * PAGE_TOKENS as u32))
        .collect();
    assert_eq!(l.slots_free(), 0, "设备池应满");

    // ② 全部搬到宿主：宿主满、设备空。
    for &tail in &a {
        l.put_host(tail, HostBlock::zeroed(BLOCK_BYTES))
            .expect("入宿主");
    }
    assert_eq!(l.host_free_slots(), 0, "宿主应满");

    let state = StateStore::new(1, 0);
    let room_disk = Room {
        host_enabled: true,
        host_kv_slots: 0,
        host_snap_slots: 0,
        disk_kv_slots: PAGES,
        disk_snap_slots: 0,
        disk_enabled: true,
        bulk_drop_when_tier_full: false,
    };
    let v = plan_capacity(
        &l,
        &state,
        &owl_store::LruPolicy,
        Need::host(1, 0),
        &[],
        &[],
        room_disk,
    );
    eprintln!("[cascade] 宿主满 + 盘有空 ⇒ {:?}", v.out);
    assert!(
        v.out.iter().any(|o| o.sink() == Sink::Disk),
        "宿主满、盘有空 ⇒ 应出 Disk 票: {v:?}"
    );

    // ③ 宿主 → 盘（模拟 engine 执行；盘层只做账，不写文件）。
    let mut disk = DiskTier::<ChainHash>::new("/tmp/owl-cascade-100k", BLOCK_BYTES, PAGES * BLOCK_BYTES)
        .expect("盘层");
    for &tail in &a {
        let buf = l.host_take(tail).expect("取宿主内容");
        let plan = disk.plan_spill(tail).expect("盘计划");
        disk.commit_spill(&plan).expect("盘落账");
        drop(buf);
    }
    assert_eq!(disk.free_slots(), 0, "盘应满");
    assert_eq!(l.host_free_slots(), PAGES, "宿主应空出来");

    // ④ 再把设备填满；此时 GPU 满、Host 空、Disk 满 ⇒ 最后一级 Drop。
    let b: Vec<ChainHash> = (0..PAGES)
        .map(|i| add_chain(&mut l, 500_000 + (i as u32) * PAGE_TOKENS as u32))
        .collect();
    assert_eq!(l.slots_free(), 0, "设备池应再次满");

    eprintln!(
        "[cascade] before drop: slots_free={} host_free={} host_candidates={} disk_free={}",
        l.slots_free(),
        l.host_free_slots(),
        l.host_chain_candidates(&[]).len(),
        disk.free_slots()
    );
    let room_drop = Room {
        host_enabled: true,
        host_kv_slots: 0,
        host_snap_slots: 0,
        disk_kv_slots: 0,
        disk_snap_slots: 0,
        disk_enabled: true,
        bulk_drop_when_tier_full: false,
    };
    let v = plan_capacity(
        &l,
        &state,
        &owl_store::LruPolicy,
        Need::kv(1, NeedWhy::Grow),
        &[],
        &[],
        room_drop,
    );
    eprintln!("[cascade] 三层都满 ⇒ {:?}", v.out);
    assert!(
        v.out.iter().any(|o| o.sink() == Sink::Drop),
        "三层都满 ⇒ 最后一级必须 Drop: {v:?}"
    );
    assert!(v.deny.is_none(), "可丢内容时不应 deny: {v:?}");
    let _ = b;
}
