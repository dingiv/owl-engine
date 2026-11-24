use super::*;

/// **快照键重挂**(前缀无感删除):宿主层连内容一起换身份(零搬运),
/// **准入替换的资产清理**:宿主内容 + 状态账一起消;未知键响亮拒。
#[test]
fn drop_snap_asset_clears_host_and_state() {
    let mut i = inputs();
    i.host_bytes = 16 << 20;
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = 8 << 20;
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(4, QueuePolicy::Reject),
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
    let key = SnapKey::at_boundary(ChainHash::new(0xD9));
    r.state
        .register_snap(key, BLOCK_BYTES, Tier::Host, SnapDurability::Durable);
    let buf = r.snap_host.acquire().expect("借缓冲");
    r.snap_host.put(key, buf).expect("驻留");
    assert_eq!(r.snap_host.used_bytes(), BLOCK_BYTES);

    let drop = r.drop_snap_asset(key).expect("整体丢弃");
    assert_eq!(
        drop.tier,
        Tier::Host,
        "丢弃前所在层要报出来(设备槽由引擎清)"
    );
    assert_eq!(drop.bytes, BLOCK_BYTES);
    assert!(drop.disk_path.is_none(), "宿主快照没有盘副本");
    assert!(r.state.locate(key).is_none(), "状态账已消");
    assert!(r.snap_host.get(key).is_none(), "宿主内容已消");
    assert_eq!(r.snap_host.used_bytes(), 0);
    assert!(r.drop_snap_asset(key).is_err(), "未知键必须响亮拒");
}

/// **复用面裁剪门（2026-11-13）**：`reuse_face` 是引擎复用面的**单源**
/// （末块重算 + 快照边界裁剪 + recall 快路），直接钉死三条语义：
/// ① 命中面恰好对齐 prompt ⇒ 末块留给 prefill（非空 chunk）；
/// ② 尾边界无快照 ⇒ 一路往回裁到最近在账的快照；
/// ③ recall 命中快照腿且无需重算尾块 ⇒ 不裁（快路）；
/// ④ `dev` 只数设备槽（宿主占位 `None` 不算）。
#[test]
fn reuse_face_trims_to_snapshot_boundary_and_keeps_tail_for_prefill() {
    let mut r = rel(1);
    let page = 32usize;
    let hashes: Vec<ChainHash> = (1..=5u64).map(ChainHash::new).collect();
    let hit = ChainHit {
        matched_blocks: 5,
        slots: vec![Some(0), Some(1), Some(2), None, None],
        hashes: hashes.clone(),
    };
    let recall = RecallHit {
        kv_tier: Some(Tier::Gpu),
        snap_tier: None,
        blocks: 5,
    };
    // ① 完全对齐 ⇒ 末块重算（哪怕快照就在末边界）。
    r.state.register_snap(
        SnapKey::at_boundary(hashes[3]),
        BLOCK_BYTES,
        Tier::Gpu,
        SnapDurability::Durable,
    );
    let (m, dev) = r.reuse_face(&hit, &recall, 5 * page, page);
    assert_eq!(m, 4, "完全对齐必须保留末块重算");
    assert_eq!(dev, 3, "dev 只数设备槽（块 3/4 是宿主占位）");

    // ② 尾边界无快照 ⇒ 裁到最近在账的边界：只留边界 1 的快照。
    r.drop_snap_asset(SnapKey::at_boundary(hashes[3]))
        .expect("撤边界 3");
    r.state.register_snap(
        SnapKey::at_boundary(hashes[1]),
        BLOCK_BYTES,
        Tier::Gpu,
        SnapDurability::Durable,
    );
    let (m, dev) = r.reuse_face(&hit, &recall, 5 * page + 7, page);
    assert_eq!(m, 2, "必须裁到最后一个有快照的边界（块 1 之后）");
    assert_eq!(dev, 2, "裁后的设备槽数");

    // ③ recall 命中快照腿 + 无需重算 ⇒ 快路不裁。
    let recall_snap = RecallHit {
        kv_tier: Some(Tier::Gpu),
        snap_tier: Some(Tier::Gpu),
        blocks: 5,
    };
    let (m, dev) = r.reuse_face(&hit, &recall_snap, 5 * page + 7, page);
    assert_eq!(m, 5, "recall 命中快照 + 不对齐 ⇒ 不裁");
    assert_eq!(dev, 3);

    // ④ 完全对齐时快路不生效（仍要留末块）⇒ 仍走快照裁剪。
    r.state.register_snap(
        SnapKey::at_boundary(hashes[3]),
        BLOCK_BYTES,
        Tier::Gpu,
        SnapDurability::Durable,
    );
    let (m, _) = r.reuse_face(&hit, &recall_snap, 5 * page, page);
    assert_eq!(m, 4, "对齐 ⇒ 末块重算优先于 recall 快路");

    // ⑤ 一个快照都不在账 ⇒ 复用面为 0（宁少不错）。
    let empty = ChainHit {
        matched_blocks: 5,
        slots: vec![Some(0); 5],
        hashes: (11..=15u64).map(ChainHash::new).collect(),
    };
    let (m, dev) = r.reuse_face(&empty, &recall, 5 * page + 7, page);
    assert_eq!((m, dev), (0, 0), "无快照边界 = 不可复用");
}

/// 设备层只动账(槽是池的活);目标已在账 / 账表不平 ⇒ 拒且原样不动。
#[test]
fn rekey_snap_moves_host_content_without_copy() {
    // 共享宿主池：KV 与快照同一本账（2026-11-18 裁决）
    let mut i = inputs();
    i.host_bytes = 16 << 20;
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = 8 << 20;
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(4, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
    )
    .expect("装配");
    // 宿主层块粒度 = 一条快照字节(attach 前是占位 1 字节)
    r.attach_snap_layout(crate::leg::PoolLayout {
        class: crate::leg::BlockClass::GdnRow,
        epoch: 1,
        pieces: vec![crate::leg::PieceSpec {
            buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
            stride_bytes: BLOCK_BYTES,
        }],
    })
    .expect("状态腿布局");
    let old = SnapKey::at_boundary(ChainHash::new(0xA1));
    let new = SnapKey::at_boundary(ChainHash::new(0xA2));
    r.state
        .register_snap(old, BLOCK_BYTES, Tier::Host, SnapDurability::Durable);
    let buf = r.snap_host.acquire().expect("借缓冲");
    r.snap_host.put(old, buf).expect("驻留");
    assert!(r.rekey_snap(old, new), "宿主层重挂");
    assert_eq!(r.state.locate(new), Some(Tier::Host), "账随键走");
    assert!(r.snap_host.get(old).is_none(), "旧键内容必须已换身份");
    assert!(r.snap_host.get(new).is_some(), "新键拿到同一份内容");
    assert_eq!(r.snap_host.used_bytes(), BLOCK_BYTES, "零搬运");
    // 设备层只动账
    let g_old = SnapKey::at_boundary(ChainHash::new(0xB1));
    let g_new = SnapKey::at_boundary(ChainHash::new(0xB2));
    r.state
        .register_snap(g_old, BLOCK_BYTES, Tier::Gpu, SnapDurability::Durable);
    assert!(r.rekey_snap(g_old, g_new));
    assert_eq!(r.state.locate(g_new), Some(Tier::Gpu));
    // 目标已在账 ⇒ 拒(不许覆盖)
    assert!(!r.rekey_snap(g_new, new));
    // 账说宿主、驻留表里没有 ⇒ 拒(不许猜),且账原样不动
    let h = SnapKey::at_boundary(ChainHash::new(0xC1));
    r.state
        .register_snap(h, BLOCK_BYTES, Tier::Host, SnapDurability::Durable);
    assert!(!r.rekey_snap(h, SnapKey::at_boundary(ChainHash::new(0xC2))));
    assert_eq!(r.state.locate(h), Some(Tier::Host), "拒时必须原样不动");
}

/// **两腿同表(状态腿 Host 方向)** §30:状态腿宿主有位 ⇒ 快照出
/// `Snap{Host}` 票(不再只能等 capture 撞满时被 engine 池被动逐)。
#[test]
fn plan_need_moves_durable_snapshot_to_host_when_room() {
    let i = inputs();
    // KV 腿只拿一半 host ⇒ 状态腿有 4 MiB 宿主编
    let mut cfg = kv_cfg(8, 8);
    cfg.host_bytes = 4 << 20;
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(2, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
    )
    .expect("装配");
    // 注册状态腿布局(一条快照 = 1 MiB)
    r.attach_snap_layout(crate::leg::PoolLayout {
        class: crate::leg::BlockClass::GdnRow,
        epoch: 1,
        pieces: vec![crate::leg::PieceSpec {
            buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
            stride_bytes: BLOCK_BYTES,
        }],
    })
    .expect("状态腿布局");
    assert!(r.snap_host_free_slots() >= 1, "状态腿宿主应可容至少一张");
    let key = SnapKey::at_boundary(ChainHash::new(9));
    r.state
        .register_snap(key, BLOCK_BYTES, Tier::Gpu, SnapDurability::Durable);

    let v = r.plan_need(
        Need {
            slots: 0,
            snap_bytes: BLOCK_BYTES,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Pressure,
        },
        &Protected::default(),
    );
    assert!(v.deny.is_none(), "宿主有位 ⇒ 不拒:{:?}", v.deny);
    assert!(
        matches!(v.out.as_slice(), [EvictOut::Snap { key: k, sink: Sink::Host }] if *k == key),
        "状态腿应出 Host 票:{v:?}"
    );
}

/// **Transient 只丢不沉**(事实档,§30.1):生成段快照即使宿主有位也不下沉。
#[test]
fn plan_need_never_offloads_transient_snapshot() {
    let i = inputs();
    let mut cfg = kv_cfg(8, 8);
    cfg.host_bytes = 4 << 20;
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
    let key = SnapKey::at_boundary(ChainHash::new(11));
    r.state
        .register_snap(key, BLOCK_BYTES, Tier::Gpu, SnapDurability::Transient);

    let v = r.plan_need(
        Need {
            slots: 0,
            snap_bytes: BLOCK_BYTES,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Pressure,
        },
        &Protected::default(),
    );
    assert!(
        v.out.iter().all(|o| !matches!(
            o,
            EvictOut::Snap {
                sink: Sink::Host,
                ..
            }
        )),
        "Transient 不得下沉:{v:?}"
    );
    assert!(
        v.out.iter().any(|o| matches!(
            o,
            EvictOut::Snap {
                sink: Sink::Drop,
                ..
            }
        )),
        "买不起宿主 ⇒ 仍有 Drop 票:{v:?}"
    );
}

/// 2026-11-18 共享池：快照下沉时若宿主被 **KV 副本**占满，必须让 KV 让位
/// （而不是让快照“沉不下就丢”）。
#[test]
fn snapshot_offload_evicts_kv_host_when_shared_pool_full() {
    let mut i = inputs();
    i.host_bytes = BLOCK_BYTES;
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = BLOCK_BYTES;
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(2, QueuePolicy::Reject),
        cfg,
        cell_for(&i),
    )
    .expect("装配");
    r.leg
        .attach_pool(crate::leg::PoolLayout {
            class: crate::leg::BlockClass::KvPage,
            epoch: 0,
            pieces: vec![crate::leg::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: BLOCK_BYTES,
            }],
        })
        .expect("KV 布局");
    r.attach_snap_layout(crate::leg::PoolLayout {
        class: crate::leg::BlockClass::GdnRow,
        epoch: 1,
        pieces: vec![crate::leg::PieceSpec {
            buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
            stride_bytes: BLOCK_BYTES,
        }],
    })
    .expect("状态腿布局");

    let toks = toks(32);
    let mut table = Vec::new();
    r.ensure_for_len_unreserved(&mut table, 32).expect("块表");
    r.register_chain(&toks, &table).expect("登记链");
    r.release_table(&mut table).expect("归还表");
    let kv = r.leg.tail_hash(&toks, 1).expect("KV 身份");
    r.leg
        .put_host(
            kv,
            crate::host::HostBlock::from_bytes(vec![7u8; BLOCK_BYTES]),
        )
        .expect("KV 占满宿主");
    assert_eq!(r.leg.host_free_slots(), 0, "共享池被 KV 占满");

    let k = SnapKey::at_boundary(ChainHash::new(0x41));
    r.state
        .register_snap(k, BLOCK_BYTES, Tier::Gpu, SnapDurability::Durable);
    let (_pieces, table) = r.plan_snap_offload(k, 0).expect("快照应让 KV 让位");
    r.commit_snap_offload(k, table).expect("落账");
    assert_eq!(r.state.locate(k), Some(Tier::Host));
    assert!(r.snap_host.get(k).is_some());
    assert!(!r.leg.host_contains(kv), "KV 宿主副本已被逐出");
    assert!(r.leg.host_relief_blocks() >= 1, "让位计数应入账");
}

/// 2026-11-18 TTFT 保护：宿主快照接近容量且无空转块时，下沉应**逐一条
/// 最冷宿主快照并复用其宿主块**，而不是在 prefill 关键路径上新分配大块。
#[test]
fn snap_offload_reuses_host_block_near_capacity() {
    let mut i = inputs();
    i.host_bytes = 4 * BLOCK_BYTES;
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = 4 * BLOCK_BYTES;
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

    let olds: Vec<SnapKey> = (0..3)
        .map(|n| SnapKey::at_boundary(ChainHash::new(0x50 + n)))
        .collect();
    for k in &olds {
        r.state
            .register_snap(*k, BLOCK_BYTES, Tier::Host, SnapDurability::Durable);
        let buf = r.snap_host.acquire().expect("宿主块");
        r.snap_host.put(*k, buf).expect("驻留");
    }
    assert_eq!(r.snap_host.used_bytes(), 3 * BLOCK_BYTES);
    assert_eq!(r.snap_host.free_bytes(), 0, "空转表已空");

    let new = SnapKey::at_boundary(ChainHash::new(0x60));
    r.state
        .register_snap(new, BLOCK_BYTES, Tier::Gpu, SnapDurability::Durable);
    let (_pieces, table) = r.plan_snap_offload(new, 0).expect("下沉计划");
    r.commit_snap_offload(new, table).expect("落账");

    assert_eq!(r.snap_host.used_bytes(), 3 * BLOCK_BYTES, "块数不增长");
    assert_eq!(
        r.snap_host_evicted_bytes(),
        BLOCK_BYTES as u64,
        "逐出一条旧宿主快照"
    );
    assert_eq!(r.state.locate(new), Some(Tier::Host));
    assert_eq!(
        olds.iter()
            .filter(|k| r.state.locate(**k).is_none())
            .count(),
        1,
        "只逐一条最冷旧快照"
    );
}
