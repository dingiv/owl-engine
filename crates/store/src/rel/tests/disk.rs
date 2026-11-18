use super::*;

/// **第三层门(S6,逐层律)**:宿主满 ⇒ 把**宿主里最冷的**挪到磁盘
/// (腾宿主位,内容不丢);本轮**不出 Drop、不 deny**(留一拍)。
#[test]
fn plan_need_spills_host_kv_to_disk_when_host_full() {
    let mut i = inputs();
    i.disk_bytes = 4 << 20;
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = 1 << 20; // 宿主体量 = 单块(1MB)⇒ 满
    i.host_bytes = 1 << 20;
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(2, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
    )
    .expect("装配");
    let attach = r.attach_disk("/tmp/owl-disk-test-kv").expect("接盘");
    assert_eq!(attach.kv_slots, 4, "盘配额 4MB / 单块 1MB");
    // 两条链(各 1 块)都放进宿主 ⇒ 宿主满、设备无货
    let ta = toks(32);
    let tb: Vec<u32> = (100..132).collect();
    for t in [&ta, &tb] {
        let mut table = Vec::new();
        r.ensure_for_len_unreserved(&mut table, 32).expect("块表");
        r.register_chain(t, &table).expect("登记");
        r.release_table(&mut table).expect("归还");
    }
    let a = r.leg.tail_hash(&ta, 1).expect("A 身份");
    let _b = r.leg.tail_hash(&tb, 1).expect("B 身份");
    // A 入宿主(宿主满);B 留设备(轮一沉不动它 —— 宿主没位)
    r.leg
        .put_host(
            a,
            crate::host::HostBlock::from_bytes(vec![7u8; BLOCK_BYTES]),
        )
        .expect("A 入宿主");
    assert_eq!(r.leg.host_free_slots(), 0, "宿主体量 1 块 ⇒ 满");
    assert_eq!(r.leg.kv_stats().host_blocks, 1, "B 仍在设备(宿主只有 A)");

    let v = r.plan_need(
        Need {
            slots: 1,
            snap_bytes: 0,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Pressure,
        },
        &Protected::default(),
    );
    assert!(v.deny.is_none(), "腾过宿主 ⇒ 本轮不 deny:{:?}", v.deny);
    let disk_tickets = v
        .out
        .iter()
        .filter(|o| {
            matches!(
                o,
                EvictOut::Kv {
                    sink: Sink::Disk,
                    span: 1,
                    ..
                }
            )
        })
        .count();
    assert_eq!(disk_tickets, 1, "应出一条「宿主 → 磁盘」票:{v:?}");
    assert!(
        v.out.iter().all(|o| o.sink() != Sink::Drop),
        "内容不丢优先:不得出 Drop 票:{v:?}"
    );
}

/// **HostPressure 专用票**：宿主满时只把宿主冷面挪盘，
/// **不得**把设备块继续往宿主塞（跳过 GPU→Host）。
#[test]
fn host_pressure_skips_gpu_to_host_and_emits_disk_only() {
    let mut i = inputs();
    i.disk_bytes = 4 << 20;
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = 1 << 20; // 宿主 1 块 ⇒ 满
    i.host_bytes = 1 << 20;
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(2, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
    )
    .expect("装配");
    r.attach_disk("/tmp/owl-disk-test-host-pressure")
        .expect("接盘");
    let ta = toks(32);
    let tb: Vec<u32> = (200..232).collect();
    for t in [&ta, &tb] {
        let mut table = Vec::new();
        r.ensure_for_len_unreserved(&mut table, 32).expect("块表");
        r.register_chain(t, &table).expect("登记");
        r.release_table(&mut table).expect("归还");
    }
    let a = r.leg.tail_hash(&ta, 1).expect("A");
    let _b = r.leg.tail_hash(&tb, 1).expect("B");
    r.leg
        .put_host(
            a,
            crate::host::HostBlock::from_bytes(vec![9u8; BLOCK_BYTES]),
        )
        .expect("A 入宿主");
    assert_eq!(r.leg.host_free_slots(), 0);

    let v = r.plan_need(Need::host(1, 0), &Protected::default());
    assert!(v.deny.is_none(), "宿主有盘位 ⇒ 不拒:{:?}", v.deny);
    assert!(
        v.out.iter().all(|o| o.sink() != Sink::Host),
        "HostPressure 不得出 GPU→Host 票:{v:?}"
    );
    assert!(
        v.out.iter().any(|o| matches!(
            o,
            EvictOut::Kv {
                sink: Sink::Disk,
                span: 1,
                ..
            }
        )),
        "应把宿主冷块挪盘:{v:?}"
    );
    assert!(v.out.iter().all(|o| o.sink() != Sink::Drop), "{v:?}");
}

/// **宿主预算单位 = 链块（world × 单页）**：TP 下旧口径（除单平面页）高估
/// world 倍 ⇒ 计划按“有位”出票、`lend_slots` 半途撞 `Exhausted`
/// （双卡 1M 后第二个长 turn 的硬错，§23.11）。
#[test]
fn host_free_slots_is_counted_in_chain_slots_under_tp() {
    let mut i = inputs();
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = 4 * 2 * BLOCK_BYTES; // 4 个链块（每块 = 两平面）
    i.host_bytes = 4 * 2 * BLOCK_BYTES;
    let mut r = Rel::open_world(
        i,
        AdmissionPolicy::new(1, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
        2,
    )
    .expect("装配 world=2");
    assert_eq!(r.leg.host_slot_bytes(), 2 * BLOCK_BYTES);
    assert_eq!(r.leg.host_free_slots(), 4, "预算 8MB / 链块 2MB");
    let t = toks(32);
    let mut table = Vec::new();
    r.ensure_for_len_unreserved(&mut table, 32).expect("块表");
    r.register_chain(&t, &table).expect("登记");
    r.release_table(&mut table).expect("归还");
    let a = r.leg.tail_hash(&t, 1).expect("身份");
    r.leg
        .put_host(a, crate::host::HostBlock::zeroed(2 * BLOCK_BYTES))
        .expect("入宿主");
    assert_eq!(r.leg.host_free_slots(), 3, "一块 = 两个平面，不是两格");
}

/// **盘块 = 宿主槽（world × 单页）**：TP 档也能接盘；溢写计划字节必须等于
/// 一条链的宿主槽（否则写到盘上的只有半个链块 ⇒ 回填必截断）。
#[test]
fn disk_block_is_the_host_slot_under_tp() {
    let mut i = inputs();
    i.disk_bytes = 8 << 20;
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = 2 * 2 * BLOCK_BYTES;
    i.host_bytes = 2 * 2 * BLOCK_BYTES;
    let mut r = Rel::open_world(
        i,
        AdmissionPolicy::new(1, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
        2,
    )
    .expect("装配 world=2");
    let at = r.attach_disk("/tmp/owl-disk-test-tp").expect("TP 接盘");
    assert_eq!(at.kv_slots, 4, "4 个链块");
    let a = ChainHash::new(42);
    let p = r.disk.as_ref().expect("盘").plan_spill(a).expect("计划");
    assert_eq!(p.bytes, 2 * BLOCK_BYTES, "一个盘块 = 两个平面");
    r.disk.as_mut().expect("盘").commit_spill(&p).expect("落账");
    assert_eq!(r.disk.as_ref().expect("盘").free_slots(), 3);
}

/// **磁盘回填落宿主的兜底**：宿主满 ⇒ 先逐最冷（否则 onload 整单 `Exhausted`）。
#[test]
fn host_fill_relieves_cold_copy_when_full() {
    let mut i = inputs();
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = BLOCK_BYTES;
    i.host_bytes = BLOCK_BYTES;
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(2, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
    )
    .expect("装配");
    let ta = toks(32);
    let tb: Vec<u32> = (100..132).collect();
    for t in [&ta, &tb] {
        let mut table = Vec::new();
        r.ensure_for_len_unreserved(&mut table, 32).expect("块表");
        r.register_chain(t, &table).expect("登记");
        r.release_table(&mut table).expect("归还");
    }
    let a = r.leg.tail_hash(&ta, 1).expect("A 身份");
    let b = r.leg.tail_hash(&tb, 1).expect("B 身份");
    r.leg
        .put_host(
            a,
            crate::host::HostBlock::from_bytes(vec![7u8; BLOCK_BYTES]),
        )
        .expect("A 入宿主");
    assert_eq!(r.leg.host_free_slots(), 0, "宿主已满");
    r.host_fill(
        b,
        crate::host::HostBlock::from_bytes(vec![9u8; BLOCK_BYTES]),
    )
    .expect("先逐最冷再落");
    assert!(r.leg.host_contains(b), "回填内容已落宿主");
    assert!(!r.leg.host_contains(a), "最冷副本已让位");
    assert_eq!(r.leg.host_relief_blocks(), 1, "让位入账");
}

/// **轮零：宿主满 ⇒ 先削最冷宿主链，再沉设备链**（不许直接丢新块）。
#[test]
fn plan_need_prunes_cold_host_chain_to_make_room_for_device_chain() {
    let mut i = inputs();
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = BLOCK_BYTES; // 宿主只装一个链块
    i.host_bytes = BLOCK_BYTES;
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(2, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
    )
    .expect("装配");
    let ta = toks(32);
    let tb: Vec<u32> = (100..132).collect();
    for t in [&ta, &tb] {
        let mut table = Vec::new();
        r.ensure_for_len_unreserved(&mut table, 32).expect("块表");
        r.register_chain(t, &table).expect("登记");
        r.release_table(&mut table).expect("归还");
    }
    let a = r.leg.tail_hash(&ta, 1).expect("A 身份");
    let b = r.leg.tail_hash(&tb, 1).expect("B 身份");
    r.leg
        .put_host(
            a,
            crate::host::HostBlock::from_bytes(vec![7u8; BLOCK_BYTES]),
        )
        .expect("A 入宿主");
    assert_eq!(r.leg.host_free_slots(), 0, "宿主已满");

    let v = r.plan_need(
        Need {
            slots: 1,
            snap_bytes: 0,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Pressure,
        },
        &Protected::default(),
    );
    assert!(v.deny.is_none(), "腾过宿主 ⇒ 本轮不 deny:{:?}", v.deny);
    let drop_at = v
        .out
        .iter()
        .position(|o| matches!(o, EvictOut::Kv { sink: Sink::Drop, chain, .. } if *chain == a));
    let host_at = v
        .out
        .iter()
        .position(|o| matches!(o, EvictOut::Kv { sink: Sink::Host, chain, .. } if *chain == b));
    assert!(drop_at.is_some(), "应丢最冷宿主链:{v:?}");
    assert!(host_at.is_some(), "腾出宿主位后应下沉设备链:{v:?}");
    assert!(drop_at < host_at, "宿主让位票必须先执行:{v:?}");
}

/// **第三层门(S6)**:状态腿宿主的 Durable 挪去盘;`Transient` **不写盘**。
#[test]
fn plan_need_spills_host_snapshot_to_disk_but_not_transient() {
    let mut i = inputs();
    i.disk_bytes = 8 << 20;
    let mut cfg = kv_cfg(8, 8);
    cfg.host_bytes = i.host_bytes; // 状态腿宿主 = 0(本测试直接造"已在宿主"的账)
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(2, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
    )
    .expect("装配");
    r.attach_snap_layout(crate::leg::PoolLayout {
        class: crate::leg::BlockClass::GdnRow,
        epoch: 1,
        pieces: vec![crate::leg::PieceSpec {
            buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
            stride_bytes: BLOCK_BYTES,
        }],
    })
    .expect("状态腿布局");
    r.attach_disk("/tmp/owl-disk-test-snap").expect("接盘");

    let k_dur = SnapKey::at_boundary(ChainHash::new(0xD1));
    let k_tr = SnapKey::at_boundary(ChainHash::new(0x77));
    r.state
        .register_snap(k_dur, BLOCK_BYTES, Tier::Host, SnapDurability::Durable);
    r.state
        .register_snap(k_tr, BLOCK_BYTES, Tier::Host, SnapDurability::Transient);

    let v = r.plan_need(
        Need {
            slots: 0,
            snap_bytes: 2 * BLOCK_BYTES,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Pressure,
        },
        &Protected::default(),
    );
    assert!(v.deny.is_none(), "{:?}", v.deny);
    assert!(
        v.out
            .iter()
            .any(|o| matches!(o, EvictOut::Snap { key, sink: Sink::Disk } if *key == k_dur)),
        "Durable 应出盘票:{v:?}"
    );
    assert!(
        v.out
            .iter()
            .all(|o| !matches!(o, EvictOut::Snap { sink: Sink::Disk, key } if *key == k_tr)),
        "Transient 不写盘:{v:?}"
    );
}

/// **HostPressure 两腿同时低**：KV 票与快照票各自独立，互不折算。
#[test]
fn host_pressure_keeps_kv_and_snapshot_goals_independent() {
    let mut i = inputs();
    i.disk_bytes = 8 << 20;
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = BLOCK_BYTES; // KV 宿主 1 块
    i.host_bytes = BLOCK_BYTES;
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(2, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
    )
    .expect("装配");
    r.attach_snap_layout(crate::leg::PoolLayout {
        class: crate::leg::BlockClass::GdnRow,
        epoch: 1,
        pieces: vec![crate::leg::PieceSpec {
            buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
            stride_bytes: BLOCK_BYTES,
        }],
    })
    .expect("状态腿布局");
    r.attach_disk("/tmp/owl-disk-test-host-both").expect("接盘");
    let kv_tokens = toks(32);
    let mut table = Vec::new();
    r.ensure_for_len_unreserved(&mut table, 32).expect("块表");
    r.register_chain(&kv_tokens, &table).expect("登记");
    r.release_table(&mut table).expect("归还");
    let kv_tail = r.leg.tail_hash(&kv_tokens, 1).expect("KV 尾");
    r.leg
        .put_host(
            kv_tail,
            crate::host::HostBlock::from_bytes(vec![3u8; BLOCK_BYTES]),
        )
        .expect("KV 入宿主");
    let snap = SnapKey::at_boundary(ChainHash::new(0xF1));
    r.state
        .register_snap(snap, BLOCK_BYTES, Tier::Host, SnapDurability::Durable);

    let v = r.plan_need(Need::host(1, 1), &Protected::default());
    assert!(v.deny.is_none(), "两腿都有盘位 ⇒ 不拒:{:?}", v.deny);
    assert_eq!(
        v.out
            .iter()
            .filter(|o| matches!(
                o,
                EvictOut::Kv {
                    sink: Sink::Disk,
                    span: 1,
                    ..
                }
            ))
            .count(),
        1,
        "KV 应出 1 张前缀票:{v:?}"
    );
    assert_eq!(
        v.out
            .iter()
            .filter(|o| matches!(
                o,
                EvictOut::Snap {
                    sink: Sink::Disk,
                    ..
                }
            ))
            .count(),
        1,
        "快照应出 1 张票:{v:?}"
    );
    assert!(
        v.out
            .iter()
            .all(|o| o.sink() != Sink::Host && o.sink() != Sink::Drop),
        "HostPressure 不得 Host/Drop:{v:?}"
    );
}

/// **S6.1 候选门**:整链在宿主 ⇒ 以**叶**为锚出候选(span = 整链块数);
/// 链里有块在设备 = 不出票(否则盘上只有半链)。
#[test]
fn host_chain_candidates_anchor_at_leaf_and_require_whole_chain() {
    let mut i = inputs();
    i.disk_bytes = 8 << 20;
    let mut r = rel_with_disk(&mut i, 4 << 20, 0);
    // A = 前 1 块;B = A 的延伸(A→B)
    let ta = toks(32);
    let tb = toks(64);
    for t in [&ta, &tb] {
        let mut table = Vec::new();
        r.ensure_for_len_unreserved(&mut table, t.len())
            .expect("块表");
        r.register_chain(t, &table).expect("登记");
        r.release_table(&mut table).expect("归还");
    }
    let a = r.leg.tail_hash(&ta, 1).expect("A");
    let b = r.leg.tail_hash(&tb, 2).expect("B");
    // 只把 A 放宿主 ⇒ 整链不在宿主(B 在设备)⇒ 无候选
    r.leg
        .put_host(
            a,
            crate::host::HostBlock::from_bytes(vec![1u8; BLOCK_BYTES]),
        )
        .expect("A 入宿主");
    assert!(r.leg.host_chain_candidates(&[]).is_empty(), "半链不出票");
    // 再把 B 放宿主 ⇒ 候选 = (B, 2)
    r.leg
        .put_host(
            b,
            crate::host::HostBlock::from_bytes(vec![2u8; BLOCK_BYTES]),
        )
        .expect("B 入宿主");
    let c = r.leg.host_chain_candidates(&[]);
    assert_eq!(c.len(), 1, "{c:?}");
    assert_eq!(c[0].0, b, "锚必须是叶(B)");
    assert_eq!(c[0].1, 2, "span = 整链块数");
}

/// **S6.1 门:按盘上索引重建树**。内容在盘、索引完整 ⇒ 重建为**非设备层**
/// (`slot = None`),命中面按重建的树工作。
#[test]
fn adopt_disk_chains_rebuilds_tree_from_index() {
    let mut i = inputs();
    i.disk_bytes = 8 << 20;
    let mut r = rel_with_disk(&mut i, 4 << 20, 0);
    let tokens: Vec<u32> = (0..96).collect(); // 3 块 × 32
    let mut parent = ChainHash::new(0);
    for blk in tokens.chunks(32) {
        let h = crate::kv::hash_block(parent, blk, 0);
        parent = h;
        let plan = r.disk.as_ref().expect("盘").plan_spill(h).expect("计划");
        r.disk
            .as_mut()
            .expect("盘")
            .commit_spill(&plan)
            .expect("落账");
    }
    let idx = crate::disk_index::ChainIndex::new(32, tokens.clone()).expect("索引");
    let rep = r.adopt_disk_chains(&[idx]).expect("重建");
    assert_eq!((rep.chains, rep.blocks), (1, 3), "{rep:?}");
    assert_eq!(rep.kept.len(), 3);
    assert_eq!(r.leg.kv_stats().host_blocks, 3, "重建条目应为非设备层");
    // **tier 真相门**：盘上-only 不能被 engine 面报成 Host。
    let leg = r.leg_stats();
    assert_eq!(leg.disk_blocks, 3, "盘上链应计入 disk_blocks");
    assert_eq!(leg.host_blocks, 0, "盘上-only 不得计入 host_blocks");
    let stats = r.report();
    assert_eq!(stats.kv.disk_blocks, 3, "报告面应拆出盘块");
    assert_eq!(stats.kv.host_blocks, 0, "报告面盘上-only 不得算 Host");
    assert!(
        r.plan_checkpoint_kv(&tokens, &[], 8 << 20)
            .expect("plan")
            .is_none(),
        "整链已在盘 ⇒ 不应再产 checkpoint 写计划"
    );
    let (hit, recall) = r.match_prefix(&tokens).expect("命中");
    assert_eq!(hit.matched_blocks, 3, "命中面应按重建的树工作");
    assert_eq!(recall.kv_tier, Some(Tier::Disk), "盘上链必须报 Disk 层");
    assert_eq!(
        r.carrier_of(*hit.hashes.last().expect("尾块")),
        Ok(Carrier::Disk),
        "仅盘上有内容时必须报 Disk(不能把 slot=None 一律当 Host)"
    );
    // **会话感知的盘账**：归属来自内容树，而不是磁盘文件自己猜。
    r.set_chain_owner(7, &hit.hashes);
    let usage = r.disk_session_usage();
    assert_eq!(usage.len(), 1, "{usage:?}");
    assert_eq!(usage[0].session, 7);
    assert_eq!(usage[0].kv_blocks, 3);
    assert_eq!(usage[0].kv_bytes, 3 * (4 << 20) / 4, "盘块 = 单槽字节");
}

/// **盘层 session 感知淘汰**：活跃 session 的盘块最后才动。
#[test]
fn disk_eviction_skips_active_sessions_first() {
    let mut i = inputs();
    i.disk_bytes = 8 << 20;
    let mut r = rel_with_disk(&mut i, 4 << 20, 0);
    let ta = toks(32);
    let tb: Vec<u32> = (100..132).collect();
    let ha = r.leg.tail_hash(&ta, 1).expect("A 身份");
    let hb = r.leg.tail_hash(&tb, 1).expect("B 身份");
    // A 先落盘（更老），B 后落盘（更新）。
    for h in [ha, hb] {
        let plan = r.disk.as_ref().expect("盘").plan_spill(h).expect("计划");
        r.disk
            .as_mut()
            .expect("盘")
            .commit_spill(&plan)
            .expect("落账");
    }
    let ia = crate::disk_index::ChainIndex::new(32, ta).expect("A 索引");
    let ib = crate::disk_index::ChainIndex::new(32, tb).expect("B 索引");
    r.adopt_disk_chains(&[ia, ib]).expect("重建树");
    // A 归活跃 session 7；B 无主。按纯 LRU 会先挑 A（更老）。
    r.set_chain_owner(7, &[ha]);
    let block = r.kv_disk_block_bytes().expect("盘块");
    let plan = r.kv_disk_plan_evict_sessions(block, &[], &[7]);
    let victims: Vec<ChainHash> = plan.victims.iter().map(|(k, _)| *k).collect();
    assert!(
        victims.contains(&hb) && !victims.contains(&ha),
        "活跃 session 的盘块不应先被淘汰: {victims:?}"
    );
}

/// **S6.1 带指纹重建**：盘上内容键由 `(tokens, tags)` 算出；索引必须
/// 用同一份 tags 重建，否则图像/位置链一块都对不上（2026-11-13）。
#[test]
fn adopt_disk_chains_uses_index_tags() {
    let mut i = inputs();
    i.disk_bytes = 8 << 20;
    let mut r = rel_with_disk(&mut i, 4 << 20, 0);
    let tokens: Vec<u32> = (0..96).collect();
    let tags = vec![0xAAu64, 0, 0xCC];
    let mut parent = ChainHash::new(0);
    for (blk, tag) in tokens.chunks(32).zip(&tags) {
        let h = crate::kv::hash_block(parent, blk, *tag);
        parent = h;
        let plan = r.disk.as_ref().expect("盘").plan_spill(h).expect("计划");
        r.disk
            .as_mut()
            .expect("盘")
            .commit_spill(&plan)
            .expect("落账");
    }
    let idx =
        crate::disk_index::ChainIndex::with_tags(32, tokens.clone(), tags.clone()).expect("索引");
    let rep = r.adopt_disk_chains(&[idx]).expect("重建");
    assert_eq!((rep.chains, rep.blocks), (1, 3), "{rep:?}");
    let (hit, _) = r.match_prefix_tagged(&tokens, &tags).expect("命中");
    assert_eq!(hit.matched_blocks, 3, "带指纹链必须完整命中");
    let (plain, _) = r.match_prefix_tagged(&tokens, &[]).expect("无指纹");
    assert_eq!(plain.matched_blocks, 0, "无指纹不得误命中带指纹链");
}

/// **S6.1 截断**:索引 3 块、盘上只有前 2 块 ⇒ 重建 2 块 + 记截断。
#[test]
fn adopt_disk_chains_truncates_at_missing_content() {
    let mut i = inputs();
    i.disk_bytes = 8 << 20;
    let mut r = rel_with_disk(&mut i, 4 << 20, 0);
    let tokens: Vec<u32> = (0..96).collect();
    let mut parent = ChainHash::new(0);
    for (k, blk) in tokens.chunks(32).enumerate() {
        let h = crate::kv::hash_block(parent, blk, 0);
        parent = h;
        if k < 2 {
            let plan = r.disk.as_ref().expect("盘").plan_spill(h).expect("计划");
            r.disk
                .as_mut()
                .expect("盘")
                .commit_spill(&plan)
                .expect("落账");
        }
    }
    let idx = crate::disk_index::ChainIndex::new(32, tokens.clone()).expect("索引");
    let rep = r.adopt_disk_chains(&[idx]).expect("重建");
    assert_eq!((rep.chains, rep.blocks), (1, 2), "{rep:?}");
    assert_eq!(rep.truncated.len(), 1, "应记一条截断");
    // 只有前缀命中(第 3 块不在树里 ⇒ 断链)
    let (hit, _) = r.match_prefix(&tokens).expect("命中");
    assert_eq!(hit.matched_blocks, 2);
}
