use super::*;

#[test]
fn plan_need_merges_both_legs_and_respects_protect() {
    let mut r = rel(2);
    let ta = toks(32);
    let tb: Vec<u32> = (100..132).collect();
    for t in [&ta, &tb] {
        let mut table = Vec::new();
        r.ensure_for_len_unreserved(&mut table, 32).expect("块表");
        r.register_chain(t, &table).expect("登记");
        r.release_table(&mut table).expect("会话表归还");
    }
    let a = r.leg.tail_hash(&ta, 1).expect("A 身份");
    let b = r.leg.tail_hash(&tb, 1).expect("B 身份");
    let row = r
        .state
        .borrow_row(crate::tier::Owner::Turn(1))
        .expect("借行");
    r.state
        .return_row(row, Some(SnapKey::at_boundary(ChainHash::new(9))))
        .expect("快照");

    // 压力节点(= 零缺口也搬):KV 1 槽 + 状态腿 162MB(一张快照)
    let v = r.plan_need(
        Need {
            slots: 1,
            snap_bytes: 162_000_000,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Pressure,
        },
        &Protected {
            chains: vec![a],
            snaps: vec![],
        },
    );
    assert!(v.deny.is_none(), "可搬面充足:{:?}", v.deny);
    assert!(
        v.out
            .iter()
            .any(|o| matches!(o, EvictOut::Kv { chain, .. } if *chain == b)),
        "B 是可沉链(A 被 protect):{v:?}"
    );
    assert!(
        v.out.iter().any(|o| o.is_snap()),
        "状态腿必须同表出票:{v:?}"
    );
    assert!(
        v.out
            .iter()
            .all(|o| !matches!(o, EvictOut::Kv { chain, .. } if *chain == a)),
        "protect 的链绝不出票:{v:?}"
    );
}

#[test]
fn protected_active_turn_floor_is_never_evicted() {
    let mut r = rel(1);
    let tokens = toks(32);
    let mut table = Vec::new();
    r.ensure_for_len_unreserved(&mut table, 32).expect("块表");
    r.register_chain(&tokens, &table).expect("登记");
    let chain = r.leg.tail_hash(&tokens, 1).expect("身份");
    r.admission.admit(1, 0, 0).expect("登记活跃 turn");
    r.admission.set_floor(1, BLOCK_BYTES).expect("实占更新");
    let v = r.plan_need(
        Need {
            slots: 1,
            snap_bytes: 0,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Pressure,
        },
        &Protected {
            chains: vec![chain],
            snaps: vec![],
        },
    );
    assert!(
        v.out
            .iter()
            .all(|o| !matches!(o, EvictOut::Kv { chain: c, .. } if *c == chain)),
        "保留额被逐 = 不变量违反:{v:?}"
    );
}

/// **可删图锚点可淘汰门（2026-11-13）**：锚点是 `History` 角色、不进任何
/// 保护面 ⇒ 空间不足时**先于 Tail** 被逐出。这是“不搞 pin”的硬门：
/// 谁把锚点提到 `Dense` 或塞进 `Protected`，这里就红。
#[test]
fn evict_anchor_snapshot_is_extra_and_evictable() {
    use crate::state::SnapRole;
    let mut r = rel(2);
    let anchor = SnapKey::at_boundary(ChainHash::new(0xA1));
    let dense = SnapKey::at_boundary(ChainHash::new(0xA2));
    // 同一会话：锚点 = History（capture 路径的角色），尾锚点 = Tail。
    r.state.register_snap_owned_role(
        anchor,
        BLOCK_BYTES,
        Tier::Gpu,
        SnapDurability::Durable,
        7,
        5,
        SnapRole::History,
    );
    r.state.register_snap_owned_role(
        dense,
        BLOCK_BYTES,
        Tier::Gpu,
        SnapDurability::Durable,
        7,
        6,
        SnapRole::Tail,
    );
    assert_eq!(r.state.snap_role(anchor), Some(SnapRole::History));
    assert_eq!(
        r.snap_victim_order(&[]).first(),
        Some(&anchor),
        "History 角色的锚点必须先进淘汰序（不是 pin）"
    );
    // 锚点不在 `Protected` 面里；只有**正在载入**的 key 才被显式挡住。
    assert_eq!(
        r.snap_victim_order(&[anchor]),
        vec![dense],
        "显式 protect（正在载入）才挡；引擎不会把锚点塞进 protect"
    );
}

/// **状态腿出序唯一(§30.6-②)**:`snap_victim_order` 与 `plan_need` 的
/// `Snap` 票**同序**(事实档 + 策略),且 Transient 永远先走。
#[test]
fn snap_victim_order_is_the_plan_order() {
    // 本测试钉“纯 GPU 档”的出序；宿主开时 plan 会先出 Host 票，
    // 与 GPU victim 序不是同一张单（共享池语义见 plan_need 的 Host 分支）。
    let mut r = rel_no_host(4);
    let k_old = SnapKey::at_boundary(ChainHash::new(1));
    let k_tr = SnapKey::at_boundary(ChainHash::new(2));
    let k_new = SnapKey::at_boundary(ChainHash::new(3));
    r.state
        .register_snap(k_old, BLOCK_BYTES, Tier::Gpu, SnapDurability::Durable);
    r.state
        .register_snap(k_tr, BLOCK_BYTES, Tier::Gpu, SnapDurability::Transient);
    r.state
        .register_snap(k_new, BLOCK_BYTES, Tier::Gpu, SnapDurability::Durable);

    let order = r.snap_victim_order(&[]);
    assert_eq!(
        order.first(),
        Some(&k_tr),
        "Transient 是事实档第一:{order:?}"
    );
    assert_eq!(
        order,
        vec![k_tr, k_old, k_new],
        "同档内按 last_use(先登记先走)"
    );

    // 与容量调度同序:plan_need 的 Snap 票(宿主关 ⇒ 全 Drop)必须逐条相同
    let v = r.plan_need(
        Need {
            slots: 0,
            snap_bytes: 3 * BLOCK_BYTES,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: NeedWhy::Pressure,
        },
        &Protected::default(),
    );
    let planned: Vec<SnapKey> = v
        .out
        .iter()
        .filter_map(|o| match o {
            EvictOut::Snap { key, .. } => Some(*key),
            EvictOut::Kv { .. } => None,
        })
        .collect();
    assert_eq!(planned, order, "两处必须同一份判据:{v:?}");
}

/// **执行面兜底**：计划没预见（直接调 offload）时，宿主满要先逐最冷，
/// 不许 `Exhausted` 硬崩（旧行为：第二个长 turn 直接失败）。
#[test]
fn plan_kv_offload_relieves_host_before_lending_slots() {
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
        .attach_pool(crate::leg::PoolLayout {
            class: crate::leg::BlockClass::KvPage,
            epoch: 0,
            pieces: vec![crate::leg::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: BLOCK_BYTES,
            }],
        })
        .expect("布局");
    r.leg
        .put_host(
            a,
            crate::host::HostBlock::from_bytes(vec![7u8; BLOCK_BYTES]),
        )
        .expect("A 入宿主");
    assert_eq!(r.leg.host_free_slots(), 0, "宿主已满");

    let p = r.plan_kv_offload(b).expect("先逐最冷再借台");
    assert_eq!(p.blocks.len(), 1, "B 的一整块拿到票");
    assert_eq!(r.leg.host_relief_blocks(), 1, "让位块数入账（可观测）");
    assert!(!r.leg.host_contains(a), "最冷宿主链已被丢");
    assert_eq!(p.table.len(), 1, "借到 1 个宿主槽台（未提交前 B 仍在设备）");
}

/// **逐块票的同一道活性兜底**（W2;2026-10-28）：选择层的下沉是**逐块**的
/// （`exec/coverage.rs`），旧实现直呼 `plan_offload_blocks` ⇒ 宿主被别的
/// 会话填满时 `Exhausted` 直接 500。这里钉住"逐块票也能先逐最冷再出票"。
#[test]
fn plan_kv_offload_blocks_relieves_host_before_lending_slots() {
    let mut i = inputs();
    i.host_bytes = BLOCK_BYTES;
    let mut cfg = kv_cfg(64, 32);
    // 宿主只容一块 ⇒ B 下沉前必须让位
    cfg.host_bytes = BLOCK_BYTES;
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
        .attach_pool(crate::leg::PoolLayout {
            class: crate::leg::BlockClass::KvPage,
            epoch: 0,
            pieces: vec![crate::leg::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: BLOCK_BYTES,
            }],
        })
        .expect("布局");
    r.leg
        .put_host(
            a,
            crate::host::HostBlock::from_bytes(vec![7u8; BLOCK_BYTES]),
        )
        .expect("A 入宿主");
    assert_eq!(r.leg.host_free_slots(), 0, "宿主已满");

    let p = r
        .plan_kv_offload_blocks(&[b])
        .expect("逐块票:先逐最冷再借台");
    assert_eq!(p.blocks.len(), 1, "B 拿到一块票");
    assert_eq!(r.leg.host_relief_blocks(), 1, "让位块数入账（可观测）");
    assert!(!r.leg.host_contains(a), "最冷宿主链已被丢");
    assert_eq!(p.table.len(), 1, "借到 1 个宿主槽台（未提交前 B 仍在设备）");
}

/// **整枝塌缩**(2026-10-23):无宿主档下,一条 D 块冷链**一轮一张票、`span = D`**
/// —— 不再是"一轮一块"(旧行为见台账 2026-10-23)。
#[test]
fn cold_deep_branch_collapses_in_one_planning_round() {
    // **无宿主/无磁盘档**:只剩 Drop 轮(有宿主时同义:整链一张 span 票)
    let mut i = inputs();
    i.host_bytes = 0;
    let mut cfg = kv_cfg(64, 32);
    cfg.host_bytes = 0; // 两条腿的宿主预算必须一致(装配期交叉校验)
    let mut r = Rel::open(
        i,
        AdmissionPolicy::new(4, QueuePolicy::Reject),
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
        .expect("布局");

    let tokens = toks(8 * 32); // 8 块
    let mut table = Vec::new();
    r.ensure_for_len_unreserved(&mut table, tokens.len())
        .expect("块表");
    r.register_chain(&tokens, &table).expect("登记");
    r.release_table(&mut table).expect("归还");

    let need = |slots: usize| crate::hooks::Need {
        slots,
        snap_bytes: 0,
        host_kv_slots: 0,
        host_snap_slots: 0,
        why: crate::hooks::NeedWhy::Pressure,
    };
    let v = r.plan_need(need(8), &Protected::default());
    let tickets: Vec<_> = v
        .out
        .iter()
        .filter_map(|o| match o {
            crate::hooks::EvictOut::Kv {
                chain,
                sink: crate::hooks::Sink::Drop,
                span,
            } => Some((*chain, *span)),
            _ => None,
        })
        .collect();
    assert_eq!(tickets.len(), 1, "一条深链 ⇒ 一张票:{v:?}");
    assert_eq!(tickets[0].1, 8, "span = 整枝块数(一轮塌干净):{v:?}");
    // 执行
    let n = r
        .leg
        .prune_branch(tickets[0].0, tickets[0].1)
        .expect("整枝塌缩");
    assert_eq!(n, 8, "实际丢 8 块");
    // 缓存已空 ⇒ 再要也不会有 KV 票
    let v2 = r.plan_need(need(8), &Protected::default());
    assert!(
        !v2.out
            .iter()
            .any(|o| matches!(o, crate::hooks::EvictOut::Kv { .. })),
        "缓存空 ⇒ 没有 KV 票:{v2:?}"
    );
}

/// **有分层时不许整枝丢**(2026-10-23):宿主/磁盘可为内容留一条命时,Drop 票
/// 保持逐块(让链先缩小到装得进宿主,再由下沉/搬盘接走);整枝塌缩只用于
/// **没有分层可搬**的档位。
#[test]
fn drop_span_stays_one_when_a_content_preserving_tier_exists() {
    let mut r = rel(4); // host 8MiB ⇒ ~21 块;建一条 30 块的链(装不下)
    let tokens = toks(30 * 32);
    let mut table = Vec::new();
    r.ensure_for_len_unreserved(&mut table, tokens.len())
        .expect("块表");
    r.register_chain(&tokens, &table).expect("登记");
    r.release_table(&mut table).expect("归还");

    let v = r.plan_need(
        crate::hooks::Need {
            slots: 30,
            snap_bytes: 0,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: crate::hooks::NeedWhy::Pressure,
        },
        &Protected::default(),
    );
    let drops: Vec<usize> = v
        .out
        .iter()
        .filter_map(|o| match o {
            crate::hooks::EvictOut::Kv {
                sink: crate::hooks::Sink::Drop,
                span,
                ..
            } => Some(*span),
            _ => None,
        })
        .collect();
    assert!(!drops.is_empty(), "宿主装不下整链 ⇒ 应走 Drop:{v:?}");
    assert!(
        drops.iter().all(|s| *s == 1),
        "有分层时 Drop 必须逐块(内容优先保留):{v:?}"
    );
}

/// **整枝开关打开**(`bulk_drop_when_tier_full = true`):宿主装不下整链时,
/// 允许一张票丢整段(否则要 O(深度) 个调度轮)。
#[test]
fn bulk_flag_lets_a_tier_short_chain_collapse_in_one_round() {
    let mut r = rel(4); // host ✓ ~21 槽;30 块链装不下
    r.set_bulk_drop_when_tier_full(true);
    let tokens = toks(30 * 32);
    let mut table = Vec::new();
    r.ensure_for_len_unreserved(&mut table, tokens.len())
        .expect("块表");
    r.register_chain(&tokens, &table).expect("登记");
    r.release_table(&mut table).expect("归还");
    let v = r.plan_need(
        crate::hooks::Need {
            slots: 30,
            snap_bytes: 0,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why: crate::hooks::NeedWhy::Pressure,
        },
        &Protected::default(),
    );
    let drops: Vec<usize> = v
        .out
        .iter()
        .filter_map(|o| match o {
            crate::hooks::EvictOut::Kv {
                sink: crate::hooks::Sink::Drop,
                span,
                ..
            } => Some(*span),
            _ => None,
        })
        .collect();
    assert_eq!(drops, vec![30], "开关打开 + 装不下 ⇒ 一轮整枝:{v:?}");
    assert!(r
        .leg
        .prune_branch(r.leg.tail_hash(&tokens, 30).expect("尾"), 30)
        .is_ok());
}

/// **分叉点停住**:共享前缀 + 两条尾巴 ⇒ 塌缩一条尾巴**只丢它自己**,
/// 共享前缀不动(它还有另一个孩子)。
#[test]
fn prune_branch_stops_at_the_branch_point() {
    let mut r = rel(4);
    let a = toks(2 * 32); // 共享前缀(2 块)
    let mut b = a.clone();
    b.extend((900u32..932).collect::<Vec<_>>()); // 第 3 个**整块**(分支 B)
    let mut c = a.clone();
    c.extend((940u32..972).collect::<Vec<_>>()); // 第 3 个整块(分支 C)
    for t in [&a[..], &b[..], &c[..]] {
        let mut table = Vec::new();
        r.ensure_for_len_unreserved(&mut table, t.len())
            .expect("块表");
        r.register_chain(t, &table).expect("登记");
        r.release_table(&mut table).expect("归还");
    }
    let tail_b = r.leg.tail_hash(&b, 3).expect("B 尾");
    let tail_c = r.leg.tail_hash(&c, 3).expect("C 尾");
    let tail_a = r.leg.tail_hash(&a, 2).expect("A 尾");
    assert_eq!(
        r.leg.collapsible_span(tail_b),
        1,
        "B 尾的父有 2 个孩子 ⇒ 只塌 1 块"
    );
    assert_eq!(r.leg.collapsible_span(tail_a), 0, "共享前缀非叶 ⇒ 不可塌");
    let n = r.leg.prune_branch(tail_b, 4).expect("塌缩 B");
    assert_eq!(n, 1, "分叉点停住:只丢 B 的尾巴");
    // B 只剩共享前缀;C 整链不受影响
    let hb = r.match_prefix(&b).expect("B 命中");
    assert_eq!(hb.0.matched_blocks, 2, "B 剩共享前缀");
    let hc = r.match_prefix(&c).expect("C 命中");
    assert_eq!(hc.0.matched_blocks, 3, "C 完整");
    // B 已走 ⇒ C 的尾巴成了"唯一子链":塌它时共享前缀跟着一起走(合法:
    // 逐掉唯一孩子后父成为叶)。3 块 = C 尾 + 共享 2 块。
    let n = r.leg.prune_branch(tail_c, 4).expect("塌缩 C");
    assert_eq!(n, 3, "唯一枝尽 ⇒ 共享前缀可整段塌(3 块)");
    assert_eq!(r.leg.collapsible_span(tail_a), 0, "共享前缀已不在树");
    assert_eq!(
        r.match_prefix(&a).expect("A 查询").0.matched_blocks,
        0,
        "缓存已空"
    );
}

/// **KV 归属门**:`register_chain_owned` 记会话;深度 = 从根到该块的块数。
#[test]
fn kv_register_chain_owned_records_session_and_depth() {
    let mut r = rel(2);
    let tokens = toks(96); // 3 块
    let mut table = Vec::new();
    r.ensure_for_len_unreserved(&mut table, 96).expect("块表");
    r.leg
        .register_chain_owned(&tokens, &table, 7)
        .expect("带归属登记");
    let tail = r.leg.tail_hash(&tokens, 3).expect("尾");

    assert_eq!(r.leg.session_of(tail), 7, "归属应记上");
    assert_eq!(r.leg.depth_of(tail), 3, "深度 = 从根到该块");
    // 无归属的块 = 0(不会被会话档提前);必须是**另一条链**(同一链首块已归 7)
    let other_tokens: Vec<u32> = (1000..1032).collect();
    let other = r.leg.tail_hash(&other_tokens, 1).expect("另一条");
    assert_eq!(r.leg.session_of(other), crate::policy::NO_SESSION);
}
