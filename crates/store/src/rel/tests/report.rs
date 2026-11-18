use super::*;

#[test]
fn report_surfaces_both_legs_and_faces() {
    let mut r = rel(2);
    let tokens = toks(32);
    let mut table = Vec::new();
    r.ensure_for_len_unreserved(&mut table, 32).expect("块表");
    r.register_chain(&tokens, &table).expect("登记");
    r.admission.admit(1, 0, 0).expect("活跃 turn");
    r.admission.set_floor(1, 4 << 20).expect("实占");
    r.admission.set_elastic(1, 1 << 20).expect("弹性");
    let row = r
        .state
        .borrow_row(crate::tier::Owner::Turn(7))
        .expect("借行");
    let rep = r.report();
    let row_owner = rep
        .tiers
        .owners
        .iter()
        .find(|o| o.owner == crate::tier::Owner::Turn(7) && o.kind == AssetKind::Row)
        .expect("行归属应报 Turn(7)");
    assert_eq!(row_owner.entries, 1);
    assert_eq!(row_owner.bytes, r.state.row_bytes());
    r.state.return_row(row, None).expect("还行");
    assert_eq!(rep.active_turns, 1);
    assert_eq!(rep.reserved_bytes, 4 << 20);
    assert_eq!(rep.elastic_bytes, 1 << 20);
    assert_eq!(rep.policy, "lru");
    assert_eq!(rep.state.rows_total, 2);
    assert_eq!(rep.kv.cached_blocks, 1, "缓存面 1 块(腿报)");
    assert_eq!(rep.kv.gpu_bytes, BLOCK_BYTES, "设备绑定面 1 块");
    assert_eq!(rep.tiers.gpu_used, 0, "语义账不重复记 KV 块");
}

#[test]
fn report_host_used_includes_leg_host_blocks() {
    let mut r = rel(2);
    let chain = ChainHash::new(7);
    r.leg
        .host_put(chain, crate::host::HostBlock::zeroed(BLOCK_BYTES))
        .expect("放一块进 host");
    let rep = r.report();
    assert_eq!(rep.tiers.host_used, BLOCK_BYTES, "host 总账应含腿自报块");
    assert_eq!(rep.host_resident_bytes, BLOCK_BYTES, "腿自报同源");
    assert!(
        rep.tiers.views.iter().any(|r| r.tier == Tier::Host
            && r.kind == AssetKind::KvBlock
            && r.bytes == BLOCK_BYTES),
        "报告应有 Host/KvBlock view 行:{:?}",
        rep.tiers.views
    );
    assert_eq!(rep.tiers.gpu_used, 0, "GPU 预留面语义不因 host 行走样");
}

/// **B-0 门(store 侧)**:索引几何注册 + 登记入账(host/Index 行)+ 出账平衡。
#[test]
fn index_registers_into_the_ledger_and_balances() {
    let mut r = rel(2);
    r.attach_index(2, 1, 2).expect("注册几何");
    assert_eq!(r.index_bytes(), 0);
    let b = ChainHash::new(42);
    let entries = vec![
        crate::select::MeanKEntry {
            layer: 0,
            kv_head: 0,
            k_bar: vec![0, 0],
        },
        crate::select::MeanKEntry {
            layer: 0,
            kv_head: 1,
            k_bar: vec![0, 0],
        },
    ];
    r.index_register(b, entries.clone()).expect("登记");
    let per_block = 2 * 2 * 2; // 条目 × dim × f16
    assert_eq!(r.index_bytes(), per_block);
    let rep = r.report();
    assert_eq!(rep.tiers.host_used, per_block, "索引进 host 层账");
    assert!(
        rep.tiers
            .cells
            .iter()
            .any(|x| x.kind == AssetKind::Index && x.bytes == per_block),
        "报告应有 Index cell 行:{:?}",
        rep.tiers.cells
    );
    // 形状错 = 响亮拒(不进账)
    assert!(r
        .index_register(ChainHash::new(43), vec![entries[0].clone()])
        .is_err());
    assert_eq!(r.index_bytes(), per_block, "拒后账不动");
    // 出账
    r.index_drop(b).expect("出账");
    assert_eq!(r.index_bytes(), 0);
    assert_eq!(r.report().tiers.host_used, 0, "出账后 host 层归零");
}
