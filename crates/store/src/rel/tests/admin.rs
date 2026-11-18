use super::*;

/// **Plan/DryRun → Commit**：plan 零状态变更；commit 收实际值改账。
///
/// 这条门锁死用户裁决（2026-11-18）：store 是状态机，不是事务；
/// plan 只描述，engine 执行后把实际值交给 commit，没有 open/abort 三态。
#[test]
fn admission_plan_is_dry_run_then_commit_changes_state() {
    let mut r = rel(1);
    let plan = r.plan_admission(7, TurnFloor::Strict, 1024, 32, 32);
    assert!(plan.is_admit(), "plan 应可提交:{:?}", plan.verdict);
    assert!(plan.floor_bytes() > 0, "Strict 地板应有字节数");

    // dry-run：三本账一字未动。
    assert_eq!(r.admission.active_turns(), 0);
    assert_eq!(r.reserved_slots(), 0);
    assert_eq!(r.in_flight_turns(), 0);
    assert_eq!(r.state.rows_free(), r.state.rows_total());

    let c = r
        .commit_admission(7, plan.floor_bytes(), plan.reserved_slots, plan.alloc_slots)
        .expect("commit");
    assert_eq!(c.floor_bytes, plan.floor_bytes());
    assert_eq!(r.admission.active_turns(), 1, "commit 后准入账在册");
    assert_eq!(
        r.admission.reserved_bytes(),
        plan.floor_bytes(),
        "floor_bytes 必须立刻进入准入账（否则同拍 plan 会重复放人）"
    );
    assert_eq!(r.reserved_slots(), 32, "ActiveFloor Grant 已落账");
    assert_eq!(r.in_flight_turns(), 1);
    assert_eq!(
        r.row_owner(c.gdn_row),
        Some(crate::tier::Owner::Turn(7)),
        "GDN 行归属 turn"
    );

    // 同一 turn 重复 commit = 过期；必须拒，且不得动已在册的 turn。
    let err = r
        .commit_admission(7, plan.floor_bytes(), plan.reserved_slots, plan.alloc_slots)
        .expect_err("重复 commit 必须拒");
    assert!(matches!(err, StoreError::Invariant(_)), "得 {err:?}");
    assert_eq!(r.admission.active_turns(), 1, "已在册的 turn 不许被误回滚");
    assert_eq!(r.row_owner(c.gdn_row), Some(crate::tier::Owner::Turn(7)));

    let rep = r.finish_turn(7).expect("finish");
    assert!(rep.admission, "准入账归还");
    assert_eq!(rep.grants, 2, "ActiveFloor + Row 两张票");
    assert_eq!(rep.row, Some(c.gdn_row), "GDN 行归还");
    assert_eq!(r.admission.active_turns(), 0);
    assert_eq!(r.reserved_slots(), 0);
    assert_eq!(r.in_flight_turns(), 0);
    assert_eq!(r.state.rows_free(), r.state.rows_total());
}

/// 非 `Admit` 的 plan 只是描述：拒绝/排队/先腾位都不改账。
///
/// 这里必须走真实 `Rel::plan_admission`，不能手工构造 `AdmissionPlan`：
/// - 第一条 = 地板很小但完整预算放不下（非超卖），旧 `decide_slots`
///   路径会漏掉；
/// - 第二条 = 超卖工作集放得下、但 ctx 溢出宿主/磁盘放不下。
#[test]
fn plan_admission_reports_non_admit_without_touching_store() {
    let r = rel(1);

    let full = r.plan_admission(7, TurnFloor::Pages(1), 100, 6000, 6000);
    assert!(
        !full.is_admit(),
        "完整预算 6000 页必须拒，不能只看 1 页地板:{:?}",
        full.verdict
    );
    assert!(
        matches!(full.verdict, AdmissionVerdict::Reject { .. }),
        "可动面为 0 ⇒ Reject，得 {:?}",
        full.verdict
    );

    let over = r.plan_admission(8, TurnFloor::Pages(0), 200_000, 64, 64);
    assert!(
        !over.is_admit(),
        "超卖溢出 6186 页但宿主只有 8 页，必须拒:{:?}",
        over.verdict
    );
    assert!(
        matches!(over.verdict, AdmissionVerdict::Reject { .. }),
        "宿主/磁盘无位 ⇒ Reject，得 {:?}",
        over.verdict
    );

    assert_eq!(r.admission.active_turns(), 0);
    assert_eq!(r.reserved_slots(), 0);
    assert_eq!(r.state.rows_free(), r.state.rows_total());
}

/// **turn 终局单一围栏**：准入账 + Grant + GDN 行一次归还，重复调用幂等。
///
/// 这条门锁死 2026-11-06 真机事故的形状：start 半态只还了 Grant，
/// 准入与行仍挂着 ⇒ 后续请求永久撞"活跃 turn 天花板(1/1)"。
#[test]
fn finish_turn_releases_admission_grants_and_row() {
    let mut r = rel(1);
    r.admission.admit(7, 0, 0).expect("准入登记");
    let g = r
        .reserve_capacity(7, 4, 8, crate::reservations::ReservationKind::ActiveFloor)
        .expect("Grant");
    let row = r
        .state
        .borrow_row(crate::tier::Owner::Turn(7))
        .expect("借行");
    assert_eq!(r.admission.active_turns(), 1);

    let rep = r.finish_turn(7).expect("终局收口");
    assert!(rep.admission, "准入账必须归还");
    assert_eq!(rep.grants, 1, "Grant 必须归还");
    assert_eq!(rep.row, Some(row), "GDN 行必须归还");
    assert_eq!(r.admission.active_turns(), 0);
    assert!(r.reservations.get(g.id).is_none());
    assert_eq!(r.state.rows_free(), r.state.rows_total());

    // 幂等：重复收口 = 全零，不再报错、不再改账。
    assert_eq!(r.finish_turn(7).expect("重复收口"), TurnRelease::default());
}

/// **孤儿收口**：准入账里有、但不在活跃集里的 turn 由围栏一次清掉。
#[test]
fn finish_orphan_turns_closes_entries_outside_live_set() {
    let mut r = rel(2);
    r.admission.admit(1, 0, 0).expect("t1");
    r.admission.admit(2, 0, 0).expect("t2");
    let row = r
        .state
        .borrow_row(crate::tier::Owner::Turn(1))
        .expect("借行");

    let reps = r.finish_orphan_turns(&[2]).expect("孤儿收口");
    assert_eq!(reps.len(), 1, "只收孤儿：{reps:?}");
    assert_eq!(reps[0].0, 1);
    assert!(reps[0].1.admission);
    assert_eq!(reps[0].1.row, Some(row));
    assert_eq!(r.admission.active_turns(), 1, "在册 turn 不得被动");
    assert_eq!(r.state.rows_free(), r.state.rows_total());
}

/// **P0-4**：`ensure_for_len` 必须吃 grant；无票/不足 = 结构化预算错。
#[test]
fn ensure_for_len_consumes_grant_and_release_returns_it() {
    let mut r = rel(1);
    let g = r
        .reserve_capacity(1, 2, 2, crate::reservations::ReservationKind::ActiveFloor)
        .expect("grant");
    assert_eq!(r.reserved_slots(), 2);
    let mut table = Vec::new();
    r.ensure_for_len(g.id, &mut table, 64).expect("2 页");
    assert_eq!(table.len(), 2);
    assert_eq!(r.reservations.remaining_slots(g.id), Some(0));
    let err = r
        .ensure_for_len(g.id, &mut table, 96)
        .expect_err("第 3 页无额度");
    assert!(matches!(err, StoreError::Budget(_)));
    assert_eq!(table.len(), 2, "失败不得部分增长");
    assert_eq!(r.release_turn_grants(1).len(), 1);
    assert_eq!(r.reserved_slots(), 0);
    assert!(r.release_turn_grants(1).is_empty(), "重复归还幂等");
    let err = r
        .ensure_for_len(g.id, &mut Vec::new(), 64)
        .expect_err("已归还的 grant 不能再消费");
    assert!(
        matches!(err, StoreError::Unknown(_)),
        "应是结构化未知票:{err:?}"
    );
}

/// 分配失败要把已消费的额度退回（grant 不能半截扣账）。
#[test]
fn ensure_for_len_refunds_grant_on_pool_exhaustion() {
    let mut r = rel(1);
    let g = r
        .reserve_capacity(
            1,
            100,
            100,
            crate::reservations::ReservationKind::ActiveFloor,
        )
        .expect("grant");
    let mut table = Vec::new();
    let err = r
        .ensure_for_len(g.id, &mut table, 100 * 32)
        .expect_err("池只有 64 槽");
    assert!(matches!(err, StoreError::Exhausted(_)));
    assert!(table.is_empty());
    assert_eq!(
        r.reservations.remaining_slots(g.id),
        Some(100),
        "额度应原样退回"
    );
}

#[test]
fn open_rejects_unmeasured_budget_and_geometry() {
    // 动态池档(无池面兜底):卡级缺项 = 拒
    let mut i = inputs();
    i.weights_bytes = 0;
    assert!(Rel::open(
        i,
        AdmissionPolicy::new(1, QueuePolicy::Reject),
        kv_cfg(8, 8),
        cell_for(&i)
    )
    .is_err());
    // 几何非法 = 拒
    assert!(Rel::open(
        inputs(),
        AdmissionPolicy::new(1, QueuePolicy::Reject),
        kv_cfg(0, 8),
        cell_for(&inputs())
    )
    .is_err());
    // 每 token 字节与腿几何矛盾 = 拒(同一件事只能有一个入口)
    let mut i2 = inputs();
    i2.kv_bytes_per_token = 4096;
    assert!(Rel::open(
        i2,
        AdmissionPolicy::new(1, QueuePolicy::Reject),
        kv_cfg(8, 8),
        cell_for(&i2)
    )
    .is_err());
    // 页宽与腿几何矛盾 = 拒(A7:地板按页算、链增长也按页 ⇒ 同一把尺)
    let mut i3 = inputs();
    i3.page_tokens = 16;
    assert!(Rel::open(
        i3,
        AdmissionPolicy::new(1, QueuePolicy::Reject),
        kv_cfg(8, 8),
        cell_for(&i3)
    )
    .is_err());
    // **账本预算不一致 = 响亮拒**(boot 的算式与 BudgetInputs 漂了)
    let wrong = crate::tier::shared_ledger(crate::tier::TierBudget {
        host_bytes: 1,
        disk_bytes: 0,
    });
    assert!(
        Rel::open(
            inputs(),
            AdmissionPolicy::new(1, QueuePolicy::Reject),
            kv_cfg(8, 8),
            wrong
        )
        .is_err(),
        "账本预算与分解不符必须拒"
    );
}

#[test]
fn recall_semantics_are_the_relation_layer_business() {
    let mut r = rel(2);
    let tokens = toks(64);
    let mut table = Vec::new();
    r.ensure_for_len_unreserved(&mut table, 64).expect("块表");
    r.register_chain(&tokens, &table).expect("登记");
    let tail = r.leg.tail_hash(&tokens, 2).expect("尾身份");
    let id = PrefixId::new(tail);

    let h = r.probe(id);
    assert_eq!(h.mode(), RecallMode::ContextOnly, "只 KV ⇒ 上下文召回");
    assert_eq!(h.blocks, 2);
    assert_eq!(
        r.probe(PrefixId::new(ChainHash::new(9999))).mode(),
        RecallMode::Miss
    );

    // 状态腿到位 ⇒ Exact
    let row = r
        .state
        .borrow_row(crate::tier::Owner::Turn(1))
        .expect("借行");
    r.state
        .return_row(row, Some(id.snap_key()))
        .expect("落快照");
    assert_eq!(r.probe(id).mode(), RecallMode::Exact);
}

/// **host 总账必须含腿自报的 Host KvBlock**(2026-10-23)。
///
/// 原 `report()` 只取 ledger 的 `host_used`(Index 行),腿的 Host KvBlock
/// 只进 rows/owners ⇒ 260k kvmem 档报 204MiB,实际约 5GiB。
/// **Grant 分用途账**：Row/Load/Snap/Temp 不污染 KV `reserved_slots`。
#[test]
fn grant_ledger_keeps_kinds_separate() {
    let mut r = rel(2);
    let floor = r
        .reserve_capacity(1, 2, 2, crate::reservations::ReservationKind::ActiveFloor)
        .expect("floor");
    assert_eq!(
        floor.kind,
        crate::reservations::ReservationKind::ActiveFloor
    );
    r.reserve_row(1).expect("row");
    r.reserve_load(1, 3).expect("load");
    r.reserve_snap(1).expect("snap");
    r.reserve_temp(1, 2, 2 << 20).expect("temp");

    assert_eq!(r.reserved_slots(), 2, "KV 准入只读 ActiveFloor");
    let report = r.report();
    let kinds: Vec<_> = report.grants.iter().map(|g| g.kind).collect();
    assert!(kinds.contains(&"active_floor"), "{report:?}");
    assert!(kinds.contains(&"row"), "{report:?}");
    assert!(kinds.contains(&"load"), "{report:?}");
    assert!(kinds.contains(&"snap"), "{report:?}");
    assert!(kinds.contains(&"temp"), "{report:?}");
    r.release_turn_grants(1);
    assert_eq!(r.reserved_slots(), 0);
}
