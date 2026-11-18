//! **互斥 VRAM 账**(2026-11-15,P0)。
//!
//! 旧 `/debug/vram` 同时展示了四套口径(plan / pools / tiers / events /
//! nvidia-smi),其中 `tiers.Pool` 已包含腿自报的 `KvBlock`/`Snap`,
//! `vram_reserve` 只是 free 内的预算阈值 —— 直接相加会重复。
//! 本模块把同一时刻的快照收敛成**逐 rank、互斥、可闭合**的一组桶:
//!
//! ```text
//! total = weight_target + weight_draft + weights_other
//!       + kv_pool + pool_fixed + pool_other
//!       + graph + temp + event_other
//!       + tp_staging + allocator_untracked
//!       + pool_cached + context + free
//! ```
//!
//! 口径:
//! - `weight_*` / `pool_*` / `graph` / `temp` 来自 DeviceHandle 事件账;
//!   权重再按块名拆 target / draft(DFlash2 草稿权重原先落在 Unclassified);
//! - `kv_pool` / `pool_fixed` 是 `Pool` 事件内部的**计划几何视图**,
//!   `pool_other` = 事件池 − 两者(表/scratch 等),三者仍互斥;
//! - `tp_staging` = 块账房有、事件账没有的 `tp.*` 块(TP2 的 staging/peer);
//! - `allocator_untracked` = 默认池 used − 事件账 − tp_staging
//!   (CUDA graph exec / 库内部等不在块账的池内占位);
//! - `pool_cached` = 默认池 reserved − used(已 free 仍被池抱着);
//! - `context` = (total − free) − reserved(CUDA ctx / 模块 / cuBLAS 等);
//! - `free` = `cuMemGetInfo` 的 free(物理真值)。
//!
//! `reserve` **不进 used**;它只是 free 内的预算阈值;prefill 瞬态由代码派生,
//! 在 JSON 的 `free_budget` 里单列。`equal` = 逐 rank 闭合误差为 0。

use std::collections::{BTreeMap, BTreeSet, HashSet};

use owl_engine::{BlockRow, DiagSnapshot};
use owl_store::{AssetKind, ComponentId, EventRow, MemoryPlan, StoreReport, Tier};
use serde_json::{json, Value};

const MIB: f64 = 1048576.0;

/// 一个 rank 的互斥分账(字节)。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RankLedger {
    pub rank: u8,
    /// `cuMemGetInfo.total`(物理可见总量)。
    pub total_bytes: u64,
    /// `cuMemGetInfo.free`。
    pub free_bytes: u64,
    /// 物理 used = total − free。
    pub used_physical_bytes: u64,

    /// 目标模型权重(权重事件按块名拆出)。
    pub weights_target_bytes: u64,
    /// DFlash2 草稿权重(原先在 Unclassified)。
    pub weights_draft_bytes: u64,
    /// 权重事件中未能按名字归类的残余。
    pub weights_other_bytes: u64,

    /// 主 KV 池几何(`MemoryPlan.pool.bytes`)。
    pub kv_pool_bytes: u64,
    /// 池内固定状态(GDN 格/快照/spec 快照/草稿 ring;计划实测值)。
    pub pool_fixed_bytes: u64,
    /// 池事件中剩余项(表/scratch 等)。
    pub pool_other_bytes: u64,

    /// 事件账 Graph 类。
    pub graph_bytes: u64,
    /// 事件账 Temp 类(空闲时为 0)。
    pub temp_bytes: u64,
    /// 事件账其余类(当前应为 0;非 0 = 新类别未接线)。
    pub event_other_bytes: u64,

    /// 块账房有、事件账没有的 `tp.*` 块。
    pub tp_staging_bytes: u64,
    /// 默认池 used 中未被事件账/tp 覆盖的部分(graph exec/库内部)。
    pub allocator_untracked_bytes: u64,
    /// 默认池 reserved − used。
    pub pool_cached_bytes: u64,
    /// 物理 used − 默认池 reserved(CUDA context/模块/cuBLAS)。
    pub context_bytes: u64,

    /// 事件账合计(用于对账)。
    pub events_bytes: u64,
    /// 默认池 used / reserved(用于对账)。
    pub pools_used_bytes: u64,
    pub pools_reserved_bytes: u64,

    /// `total − Σbuckets`(应为 0)。
    pub closure_error_bytes: i64,
    /// 本 rank 的读数/闭合错误(空 = 干净)。
    pub errors: Vec<String>,
}

impl RankLedger {
    /// 所有桶(含 `free`)之和。
    pub fn sum_bytes(&self) -> u64 {
        self.weights_target_bytes
            .saturating_add(self.weights_draft_bytes)
            .saturating_add(self.weights_other_bytes)
            .saturating_add(self.kv_pool_bytes)
            .saturating_add(self.pool_fixed_bytes)
            .saturating_add(self.pool_other_bytes)
            .saturating_add(self.graph_bytes)
            .saturating_add(self.temp_bytes)
            .saturating_add(self.event_other_bytes)
            .saturating_add(self.tp_staging_bytes)
            .saturating_add(self.allocator_untracked_bytes)
            .saturating_add(self.pool_cached_bytes)
            .saturating_add(self.context_bytes)
            .saturating_add(self.free_bytes)
    }

    /// 是否闭合。
    pub fn equal(&self) -> bool {
        self.closure_error_bytes == 0 && self.errors.is_empty()
    }
}

/// 全局账本(逐 rank + 计划审计)。
#[derive(Clone, Debug, Default)]
pub struct LedgerReport {
    pub ranks: Vec<RankLedger>,
    pub equal: bool,
    /// `MemoryPlan` 与真实池 used 的缺口(诊断,不参与闭合)。
    pub plan_audit: Option<PlanAudit>,
    /// free 内的预算阈值(不参与 used)。
    pub free_budget: Option<FreeBudget>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanAudit {
    /// 单 rank 的 plan 总账(含 reserve)。
    pub plan_total_per_rank_bytes: u64,
    /// 全部 rank 的 plan 总账(= per-rank × ranks,含 reserve)。
    pub plan_total_bytes: u64,
    /// 全部 rank 的 plan 总账去掉 reserve(与 actual_pool_used 同口径)。
    pub plan_total_ex_reserve_bytes: u64,
    /// 全部 rank 的默认池 used。
    pub actual_pool_used_bytes: u64,
    pub delta_bytes: i64,
    /// actual_pool_used − plan(ex reserve)。
    pub delta_ex_reserve_bytes: i64,
    /// 全部 rank 的草稿权重实测。
    pub draft_weights_bytes: u64,
    pub draft_weights_in_plan: bool,
    pub ranks: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FreeBudget {
    pub reserve_bytes: u64,
    /// `vram-guard` 运行时实际扣的余量(= reserve − 图/warmup 已分配部分)。
    pub runtime_headroom_bytes: u64,
    /// 各 rank `free − reserve` 的最小值。
    pub min_slack_bytes: u64,
}

/// 从 `/debug/vram` 快照构建账本。
pub fn build(snap: &DiagSnapshot) -> LedgerReport {
    let devices = if snap.devices.is_empty() {
        // 兼容旧后端/旧快照:只有 rank0 的 `device`。
        vec![(0, snap.device.clone())]
    } else {
        snap.devices.clone()
    };
    build_parts(&snap.report, &snap.pools, &devices, &snap.blocks)
}

/// 纯构建入口(单测用;不读全局态)。
pub fn build_parts(
    report: &StoreReport,
    pools: &[(u8, Result<(u64, u64, u64, u64), String>)],
    devices: &[(u8, Result<(u64, u64), String>)],
    blocks: &[(u8, Result<Vec<BlockRow>, String>)],
) -> LedgerReport {
    let mut ranks: BTreeSet<u8> = BTreeSet::new();
    for e in &report.events.rows {
        ranks.insert(e.rank);
    }
    for (r, _) in pools {
        ranks.insert(*r);
    }
    for (r, _) in devices {
        ranks.insert(*r);
    }
    for (r, _) in blocks {
        ranks.insert(*r);
    }
    if ranks.is_empty() {
        ranks.insert(0);
    }

    let events_by_rank = events_by_rank(&report.events.rows);
    let event_ids = event_ids(report);
    let pools_by_rank: BTreeMap<u8, &Result<(u64, u64, u64, u64), String>> =
        pools.iter().map(|(r, v)| (*r, v)).collect();
    let devices_by_rank: BTreeMap<u8, &Result<(u64, u64), String>> =
        devices.iter().map(|(r, v)| (*r, v)).collect();
    let blocks_by_rank: BTreeMap<u8, &Result<Vec<BlockRow>, String>> =
        blocks.iter().map(|(r, v)| (*r, v)).collect();

    let plan = report.plan.as_ref();
    let (plan_pool_bytes, plan_fixed_bytes) = plan_pool_fixed(plan);

    let mut out = Vec::new();
    for rank in ranks {
        let mut errors = Vec::new();

        let (total_bytes, free_bytes) = match devices_by_rank.get(&rank) {
            Some(Ok((free, total))) => (*total, *free),
            Some(Err(e)) => {
                errors.push(format!("device: {e}"));
                (0, 0)
            }
            None => {
                errors.push("device: 缺本 rank 读数".into());
                (0, 0)
            }
        };

        let (pools_used_bytes, pools_reserved_bytes) = match pools_by_rank.get(&rank) {
            Some(Ok((used, _used_high, reserved, _reserved_high))) => (*used, *reserved),
            Some(Err(e)) => {
                errors.push(format!("pools: {e}"));
                (0, 0)
            }
            None => {
                errors.push("pools: 缺本 rank 读数".into());
                (0, 0)
            }
        };

        let rows = events_by_rank.get(&rank).cloned().unwrap_or_default();
        let mut weight_events = 0u64;
        let mut pool_events = 0u64;
        let mut graph_events = 0u64;
        let mut temp_events = 0u64;
        let mut unclassified_events = 0u64;
        let mut events_bytes = 0u64;
        for row in &rows {
            let b = row.bytes as u64;
            events_bytes = events_bytes.saturating_add(b);
            match row.kind {
                AssetKind::Weight => weight_events = weight_events.saturating_add(b),
                AssetKind::Unclassified => {
                    unclassified_events = unclassified_events.saturating_add(b)
                }
                AssetKind::Pool => pool_events = pool_events.saturating_add(b),
                AssetKind::Graph => graph_events = graph_events.saturating_add(b),
                AssetKind::Temp => temp_events = temp_events.saturating_add(b),
                _ => {}
            }
        }

        let event_set = event_ids.get(&rank).cloned().unwrap_or_default();
        let (raw_target, raw_draft, raw_tp) = match blocks_by_rank.get(&rank) {
            Some(Ok(rows)) => weight_split_and_tp(rows, &event_set),
            Some(Err(e)) => {
                errors.push(format!("blocks: {e}"));
                (0, 0, 0)
            }
            None => {
                errors.push("blocks: 缺本 rank 读数".into());
                (0, 0, 0)
            }
        };

        // 权重事件 = Weight + Unclassified(当前 Unclassified 全是 repack 权重)。
        let weight_total = weight_events.saturating_add(unclassified_events);
        let weights_target_bytes = raw_target.min(weight_total);
        let weights_draft_bytes = raw_draft.min(weight_total - weights_target_bytes);
        let weights_other_bytes = weight_total - weights_target_bytes - weights_draft_bytes;

        let known_events = weight_total
            .saturating_add(pool_events)
            .saturating_add(graph_events)
            .saturating_add(temp_events);
        let event_other_bytes = events_bytes.saturating_sub(known_events);

        // 池事件内部三分:主 KV / 固定状态 / 其他。三者互斥且和 = Pool 事件。
        let kv_pool_bytes = plan_pool_bytes.min(pool_events);
        let pool_fixed_bytes = plan_fixed_bytes.min(pool_events - kv_pool_bytes);
        let pool_other_bytes = pool_events - kv_pool_bytes - pool_fixed_bytes;

        let tracked = events_bytes.saturating_add(raw_tp);
        let allocator_untracked_bytes = pools_used_bytes.saturating_sub(tracked);
        if tracked > pools_used_bytes {
            errors.push(format!(
                "事件账+tp({tracked}) > 默认池 used({pools_used_bytes});口径混入?"
            ));
        }
        let pool_cached_bytes = pools_reserved_bytes.saturating_sub(pools_used_bytes);
        let used_physical_bytes = total_bytes.saturating_sub(free_bytes);
        let context_bytes = used_physical_bytes.saturating_sub(pools_reserved_bytes);

        let mut rank_ledger = RankLedger {
            rank,
            total_bytes,
            free_bytes,
            used_physical_bytes,
            weights_target_bytes,
            weights_draft_bytes,
            weights_other_bytes,
            kv_pool_bytes,
            pool_fixed_bytes,
            pool_other_bytes,
            graph_bytes: graph_events,
            temp_bytes: temp_events,
            event_other_bytes,
            tp_staging_bytes: raw_tp,
            allocator_untracked_bytes,
            pool_cached_bytes,
            context_bytes,
            events_bytes,
            pools_used_bytes,
            pools_reserved_bytes,
            closure_error_bytes: 0,
            errors,
        };
        rank_ledger.closure_error_bytes = total_bytes as i64 - rank_ledger.sum_bytes() as i64;
        if rank_ledger.closure_error_bytes != 0 {
            rank_ledger.errors.push(format!(
                "闭合误差 {} B: total={} sum={}",
                rank_ledger.closure_error_bytes,
                total_bytes,
                rank_ledger.sum_bytes()
            ));
        }
        out.push(rank_ledger);
    }

    let equal = out.iter().all(RankLedger::equal);
    let actual_pool_used_bytes = out.iter().map(|r| r.pools_used_bytes).sum();
    let draft_weights_bytes = out.iter().map(|r| r.weights_draft_bytes).sum();
    let plan_audit = plan.map(|p| {
        let per_rank = p.total_bytes() as u64;
        let ranks = out.len() as u64;
        let total = per_rank.saturating_mul(ranks);
        let reserve = (p.card.reserve_bytes as u64).saturating_mul(ranks);
        let total_ex_reserve = total.saturating_sub(reserve);
        PlanAudit {
            plan_total_per_rank_bytes: per_rank,
            plan_total_bytes: total,
            plan_total_ex_reserve_bytes: total_ex_reserve,
            actual_pool_used_bytes,
            delta_bytes: actual_pool_used_bytes as i64 - total as i64,
            delta_ex_reserve_bytes: actual_pool_used_bytes as i64 - total_ex_reserve as i64,
            draft_weights_bytes,
            draft_weights_in_plan: p
                .component(ComponentId::DraftWeights)
                .is_some(),
            ranks: out.len(),
        }
    });
    let free_budget = plan.map(|p| FreeBudget {
        reserve_bytes: p.card.reserve_bytes as u64,
        runtime_headroom_bytes: (p.card.reserve_bytes as u64)
            .saturating_sub(
                owl_shared::config::POOL_GRAPH_WARMUP_RESERVE_MB << 20,
            ),
        min_slack_bytes: out
            .iter()
            .map(|r| r.free_bytes.saturating_sub(p.card.reserve_bytes as u64))
            .min()
            .unwrap_or(0),
    });

    LedgerReport {
        ranks: out,
        equal,
        plan_audit,
        free_budget,
    }
}

/// `/debug/vram` 的 `tiers` 投影:
/// - `cells` = 事件/语义互斥账，GPU 和 = `gpu_used`;
/// - `views` = 腿自报的逻辑视图(KvBlock/Snap 等在 Pool 预留块内部)，
///   **不可与 cells 相加**；同 key 的 cells 已在 store 报告中扣除。
pub fn tiers_json(report: &StoreReport) -> Value {
    let cells: Vec<Value> = report
        .tiers
        .cells
        .iter()
        .map(|row| {
            json!({
                "tier": format!("{:?}", row.tier),
                "kind": format!("{:?}", row.kind),
                "bytes": row.bytes,
                "mib": row.bytes as f64 / MIB,
                "entries": row.entries,
            })
        })
        .collect();
    let views: Vec<Value> = report
        .tiers
        .views
        .iter()
        .map(|row| {
            json!({
                "tier": format!("{:?}", row.tier),
                "kind": format!("{:?}", row.kind),
                "bytes": row.bytes,
                "mib": row.bytes as f64 / MIB,
                "entries": row.entries,
                "additive": false,
            })
        })
        .collect();
    let cell_gpu: u64 = report
        .tiers
        .cells
        .iter()
        .filter(|r| r.tier == Tier::Gpu)
        .map(|r| r.bytes as u64)
        .sum();
    json!({
        "gpu_used": report.tiers.gpu_used,
        "gpu_used_mib": report.tiers.gpu_used as f64 / MIB,
        "gpu_views_bytes": report.tiers.gpu_views_bytes,
        "gpu_views_mib": report.tiers.gpu_views_bytes as f64 / MIB,
        "host_used": report.tiers.host_used,
        "disk_used": report.tiers.disk_used,
        "host_cap": report.tiers.host_cap,
        "disk_cap": report.tiers.disk_cap,
        "cells": cells,
        "views": views,
        "views_additive": false,
        "gpu_used_matches_cells": report.tiers.gpu_used as u64 == cell_gpu,
        "note": "cells = 事件/语义互斥账;views = 腿自报逻辑视图(在 Pool 内),不可相加;vram_reserve 不在 cells",
    })
}

/// `LedgerReport` → JSON(全部带 bytes + mib)。
pub fn json_from(report: &LedgerReport) -> Value {
    let ranks: Vec<Value> = report.ranks.iter().map(rank_json).collect();
    let sum = |f: fn(&RankLedger) -> u64| -> u64 { report.ranks.iter().map(f).sum() };
    json!({
        "equal": report.equal,
        "ranks": ranks,
        "totals": {
            "total_bytes": sum(|r| r.total_bytes),
            "free_bytes": sum(|r| r.free_bytes),
            "used_physical_bytes": sum(|r| r.used_physical_bytes),
            "weights_target_bytes": sum(|r| r.weights_target_bytes),
            "weights_draft_bytes": sum(|r| r.weights_draft_bytes),
            "weights_other_bytes": sum(|r| r.weights_other_bytes),
            "kv_pool_bytes": sum(|r| r.kv_pool_bytes),
            "pool_fixed_bytes": sum(|r| r.pool_fixed_bytes),
            "pool_other_bytes": sum(|r| r.pool_other_bytes),
            "graph_bytes": sum(|r| r.graph_bytes),
            "temp_bytes": sum(|r| r.temp_bytes),
            "event_other_bytes": sum(|r| r.event_other_bytes),
            "tp_staging_bytes": sum(|r| r.tp_staging_bytes),
            "allocator_untracked_bytes": sum(|r| r.allocator_untracked_bytes),
            "pool_cached_bytes": sum(|r| r.pool_cached_bytes),
            "context_bytes": sum(|r| r.context_bytes),
            "events_bytes": sum(|r| r.events_bytes),
            "pools_used_bytes": sum(|r| r.pools_used_bytes),
            "pools_reserved_bytes": sum(|r| r.pools_reserved_bytes),
            "closure_error_bytes": report.ranks.iter().map(|r| r.closure_error_bytes).sum::<i64>(),
        },
        "plan_audit": report.plan_audit.map(|a| json!({
            "ranks": a.ranks,
            "plan_total_per_rank_bytes": a.plan_total_per_rank_bytes,
            "plan_total_per_rank_mib": a.plan_total_per_rank_bytes as f64 / MIB,
            "plan_total_bytes": a.plan_total_bytes,
            "plan_total_mib": a.plan_total_bytes as f64 / MIB,
            "plan_total_ex_reserve_bytes": a.plan_total_ex_reserve_bytes,
            "plan_total_ex_reserve_mib": a.plan_total_ex_reserve_bytes as f64 / MIB,
            "actual_pool_used_bytes": a.actual_pool_used_bytes,
            "actual_pool_used_mib": a.actual_pool_used_bytes as f64 / MIB,
            "delta_bytes": a.delta_bytes,
            "delta_mib": a.delta_bytes as f64 / MIB,
            "delta_ex_reserve_bytes": a.delta_ex_reserve_bytes,
            "delta_ex_reserve_mib": a.delta_ex_reserve_bytes as f64 / MIB,
            "draft_weights_bytes": a.draft_weights_bytes,
            "draft_weights_mib": a.draft_weights_bytes as f64 / MIB,
            "draft_weights_in_plan": a.draft_weights_in_plan,
            "note": "plan_total 含 reserve;delta_ex_reserve 与 actual_pool_used 同口径,仅诊断,不参与闭合。",
        })),
        "free_budget": report.free_budget.map(|b| json!({
            "reserve_bytes": b.reserve_bytes,
            "reserve_mib": b.reserve_bytes as f64 / MIB,
            "runtime_headroom_bytes": b.runtime_headroom_bytes,
            "runtime_headroom_mib": b.runtime_headroom_bytes as f64 / MIB,
            "reserve_already_in_free": true,
            "min_free_above_reserve_bytes": b.min_slack_bytes,
            "min_free_above_reserve_mib": b.min_slack_bytes as f64 / MIB,
            "note": "reserve 是 free 内的预算阈值(含图执行/warmup/碎片),不是已分配;runtime_headroom 是 vram-guard 实际扣的余量;prefill 瞬态由代码按 prefill_chunk 派生。",
        })),
        "unit": "bytes; 所有 *_mib = bytes / 2^20",
    })
}

fn rank_json(r: &RankLedger) -> Value {
    let b = |name: &str, bytes: u64| {
        (
            name.to_string(),
            json!({"bytes": bytes, "mib": bytes as f64 / MIB}),
        )
    };
    let buckets: serde_json::Map<String, Value> = [
        b("weights_target", r.weights_target_bytes),
        b("weights_draft", r.weights_draft_bytes),
        b("weights_other", r.weights_other_bytes),
        b("kv_pool", r.kv_pool_bytes),
        b("pool_fixed", r.pool_fixed_bytes),
        b("pool_other", r.pool_other_bytes),
        b("graph", r.graph_bytes),
        b("temp", r.temp_bytes),
        b("event_other", r.event_other_bytes),
        b("tp_staging", r.tp_staging_bytes),
        b("allocator_untracked", r.allocator_untracked_bytes),
        b("pool_cached", r.pool_cached_bytes),
        b("context", r.context_bytes),
        b("free", r.free_bytes),
    ]
    .into_iter()
    .collect();
    json!({
        "rank": r.rank,
        "total_bytes": r.total_bytes,
        "total_mib": r.total_bytes as f64 / MIB,
        "free_bytes": r.free_bytes,
        "free_mib": r.free_bytes as f64 / MIB,
        "used_physical_bytes": r.used_physical_bytes,
        "used_physical_mib": r.used_physical_bytes as f64 / MIB,
        "buckets": buckets,
        "sum_buckets_bytes": r.sum_bytes(),
        "sum_buckets_mib": r.sum_bytes() as f64 / MIB,
        "closure_error_bytes": r.closure_error_bytes,
        "equal": r.equal(),
        "events_bytes": r.events_bytes,
        "events_mib": r.events_bytes as f64 / MIB,
        "pools_used_bytes": r.pools_used_bytes,
        "pools_used_mib": r.pools_used_bytes as f64 / MIB,
        "pools_reserved_bytes": r.pools_reserved_bytes,
        "pools_reserved_mib": r.pools_reserved_bytes as f64 / MIB,
        "errors": r.errors,
    })
}

fn events_by_rank(rows: &[EventRow]) -> BTreeMap<u8, Vec<&EventRow>> {
    let mut out: BTreeMap<u8, Vec<&EventRow>> = BTreeMap::new();
    for row in rows {
        out.entry(row.rank).or_default().push(row);
    }
    out
}

fn event_ids(report: &StoreReport) -> BTreeMap<u8, HashSet<u64>> {
    report
        .events
        .ids
        .iter()
        .map(|(rank, ids)| (*rank, ids.iter().copied().collect()))
        .collect()
}

fn plan_pool_fixed(plan: Option<&MemoryPlan>) -> (u64, u64) {
    let Some(plan) = plan else { return (0, 0) };
    let fixed = plan
        .fixed
        .iter()
        .filter(|c| {
            matches!(
                c.id,
                ComponentId::GdnGrid
                    | ComponentId::GdnSnapshots
                    | ComponentId::SpecSnapshots
                    | ComponentId::ReplaySnapshots
                    | ComponentId::DraftPool
            )
        })
        .map(|c| c.bytes as u64)
        .sum();
    (plan.pool.bytes as u64, fixed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightName {
    Target,
    Draft,
}

/// 权重块名 → target/draft。
///
/// 当前 27B + DFlash2 的点名事实:
/// - target: `wt.model.*` / `repack.model.*` / `wt.lm_head*` / rope 表;
/// - draft:  `wt.candidate_selector.*` / `wt.layers.*` /
///           `repack.fc_*` / `repack.layers.*`。
///
/// 未命中 = `None`(留在 `weights_other`,不猜)。
fn weight_name(name: &str) -> Option<WeightName> {
    if name.starts_with("wt.model.")
        || name.starts_with("repack.model.")
        || name.starts_with("wt.lm_head")
        || name.starts_with("wt.cos_table")
        || name.starts_with("wt.sin_table")
    {
        Some(WeightName::Target)
    } else if owl_engine::DRAFT_WEIGHT_PREFIXES
        .iter()
        .any(|p| name.starts_with(p))
    {
        Some(WeightName::Draft)
    } else {
        None
    }
}

/// 只统计事件账内的权重块(避免把 only-block 的 staging 算进权重);
/// 同时返回 only-block 的 `tp.*` 字节。
fn weight_split_and_tp(rows: &[BlockRow], event_ids: &HashSet<u64>) -> (u64, u64, u64) {
    let mut target = 0u64;
    let mut draft = 0u64;
    let mut tp = 0u64;
    for row in rows {
        let bytes = row.bytes as u64;
        if event_ids.contains(&row.id) {
            match weight_name(&row.name) {
                Some(WeightName::Target) => target = target.saturating_add(bytes),
                Some(WeightName::Draft) => draft = draft.saturating_add(bytes),
                None => {}
            }
        } else if row.name.starts_with("tp.") {
            tp = tp.saturating_add(bytes);
        }
    }
    (target, draft, tp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use owl_engine::BlockRow;
    use owl_store::{EventBook, Owner, StoreReport, TierReport};

    fn event(rank: u8, kind: AssetKind, bytes: usize) -> EventRow {
        EventRow {
            rank,
            owner: Owner::System,
            tier: Tier::Gpu,
            kind,
            bytes,
            entries: 1,
        }
    }

    fn block(id: u64, name: &str, bytes: usize) -> BlockRow {
        BlockRow {
            id,
            bytes,
            addr: id,
            ledger: 1,
            name: name.into(),
        }
    }

    fn report(rows: Vec<EventRow>, ids: Vec<(u8, Vec<u64>)>) -> StoreReport {
        StoreReport {
            events: EventBook {
                rows,
                ids,
                ..Default::default()
            },
            tiers: TierReport::default(),
            ..Default::default()
        }
    }

    #[test]
    fn disjoint_buckets_close_and_split_weights() {
        // rank0: 1000 总量;事件 = 权重 400(其中 target 250/draft 100/other 50)
        // + Pool 300 + Graph 20 + Temp 10 + Unclassified 100 = 830;
        // pools used 900(tp 70 + untracked 0), reserved 950(cached 50),
        // physical used 1000 => context 50, free 0.
        let rows = vec![
            event(0, AssetKind::Weight, 400),
            event(0, AssetKind::Pool, 300),
            event(0, AssetKind::Graph, 20),
            event(0, AssetKind::Temp, 10),
            event(0, AssetKind::Unclassified, 100),
        ];
        let ids = vec![(0, (0..830).collect())];
        let blocks = vec![(
            0,
            Ok(vec![
                block(1, "wt.model.language_model.embed_tokens.weight", 250),
                block(2, "wt.candidate_selector.successor_codebook", 50),
                block(3, "repack.fc_0.packed_raw", 50),
                block(4, "wt.something_else", 50),
                block(900, "tp.staging.r0", 70),
            ]),
        )];
        let pools = vec![(0, Ok((900, 900, 950, 950)))];
        let devices = vec![(0, Ok((0, 1000)))];
        let plan = MemoryPlan::new(
            0,
            owl_store::CardPlan {
                total_bytes: 1000,
                free_at_boot_bytes: 1000,
                reserve_bytes: 100,
            },
            vec![
                owl_store::MemComponent::new(
                    ComponentId::Weights,
                    400,
                    owl_store::PlanSource::Measured,
                    Tier::Gpu,
                    Owner::System,
                ),
                owl_store::MemComponent::new(
                    ComponentId::GdnGrid,
                    100,
                    owl_store::PlanSource::Measured,
                    Tier::Gpu,
                    Owner::System,
                ),
            ],
            owl_store::PoolPlan {
                pages: 1,
                page_tokens: 1,
                tokens: 1,
                per_token_bytes: 200,
                block_bytes: 200,
                bytes: 200,
                gdn_slots: 1,
                gdn_row_bytes: 100,
            },
            owl_store::TierPlan::default(),
            owl_store::PlanPolicy { fixed_pool: true },
        )
        .unwrap();
        let mut rep = report(rows, ids);
        rep.plan = Some(plan);
        let led = build_parts(&rep, &pools, &devices, &blocks);
        assert!(led.equal, "{led:?}");
        let r = &led.ranks[0];
        assert_eq!(r.weights_target_bytes, 250);
        assert_eq!(r.weights_draft_bytes, 100);
        assert_eq!(r.weights_other_bytes, 150);
        assert_eq!(r.kv_pool_bytes, 200);
        assert_eq!(r.pool_fixed_bytes, 100);
        assert_eq!(r.pool_other_bytes, 0);
        assert_eq!(r.tp_staging_bytes, 70);
        assert_eq!(r.allocator_untracked_bytes, 0, "900 - 830 - 70 = 0");
        assert_eq!(r.pool_cached_bytes, 50);
        assert_eq!(r.context_bytes, 50, "used 1000 - reserved 950");
        assert_eq!(r.free_bytes, 0);
        assert_eq!(r.closure_error_bytes, 0);
        assert_eq!(r.sum_bytes(), 1000);
    }

    #[test]
    fn missing_device_is_reported_not_silently_closed() {
        let rep = report(vec![event(0, AssetKind::Weight, 10)], vec![(0, vec![1])]);
        let led = build_parts(&rep, &[], &[], &[]);
        assert!(!led.equal);
        assert!(led.ranks[0].errors.iter().any(|e| e.contains("device")));
    }

    #[test]
    fn tiers_cells_exclude_leg_views() {
        let mut rep = report(vec![event(0, AssetKind::Pool, 100)], vec![(0, vec![1])]);
        rep.tiers = TierReport {
            cells: vec![owl_store::TierRow {
                tier: Tier::Gpu,
                kind: AssetKind::Pool,
                bytes: 100,
                entries: 1,
            }],
            views: vec![owl_store::TierRow {
                tier: Tier::Gpu,
                kind: AssetKind::KvBlock,
                bytes: 40,
                entries: 1,
            }],
            gpu_used: 100,
            gpu_views_bytes: 40,
            ..Default::default()
        };
        let v = tiers_json(&rep);
        let cells = v["cells"].as_array().unwrap();
        assert_eq!(cells.len(), 1, "{v}");
        assert_eq!(cells[0]["kind"], "Pool");
        let views = v["views"].as_array().unwrap();
        assert_eq!(views.len(), 1, "{v}");
        assert_eq!(views[0]["kind"], "KvBlock");
        assert_eq!(views[0]["additive"], false);
    }
}
