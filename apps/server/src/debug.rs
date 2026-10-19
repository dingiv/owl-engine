//! 诊断分账出口(GET /debug/metrics[/<prefix>];E5-M5 spec 分相探针消费面)。
//!
//! OWL_GPU_PROF=1 时 store 非空;逐核 GPU 时总账降序。TagReport 无
//! Serialize,手工投影(总 ms 计);`/debug/metrics/<prefix>` = tag 前缀
//! 过滤(spec. / load. …),无前缀 = 全量(行数上限见 [`ROWS_LIMIT];
//! 动态 tag 增殖面已由 OWL_FACT_PROBE 门控)。

use serde_json::{json, Value};

use owl_shared::metrics::{collect_metrics, collect_since_epoch, level_name, MetricsFilter};

/// 单次查询行数上限(防动态 tag 增殖拖爆应答)
const ROWS_LIMIT: usize = 512;

/// 投影行排序键(总耗时 ms)
fn total_ms(row: &Value) -> f64 {
    row["total_ms"].as_f64().unwrap_or(0.0)
}

/// **slab 活性诊断计数**(非消费快照;见 [`owl_shared::slab_hint::live_diag::counters`])。
///
/// 为什么挂在这里:这些计数原本只有"图规划期取一次(读后清零)"的出口 ⇒
/// 之后发生的异常(实测 `live.dup_new` 一次 boot 刷 12 条)在 HTTP 面看不到。
/// `live.dup_new`/`live.alias_multi` **非 0 即异常**,可当门用。
fn slab_live_json() -> Value {
    let mut m = serde_json::Map::new();
    for (tag, v) in owl_shared::slab_hint::live_diag::counters() {
        m.insert(tag.to_string(), json!(v));
    }
    Value::Object(m)
}

/// metrics 查询投影(纯函数;HTTP 面 [&super::main] 只做路由)
pub fn metrics_json(prefix: &str) -> Value {
    let mut filter = MetricsFilter::new().limit(ROWS_LIMIT);
    if !prefix.is_empty() {
        filter = filter.tag_prefix(prefix);
    }
    let mut rows: Vec<Value> = collect_metrics(&filter)
        .into_iter()
        .map(|r| {
            json!({
                "tag": r.tag, "count": r.count,
                "counter": r.counter,
                "total_ms": r.total.as_secs_f64() * 1e3,
                "avg_ms": r.avg.as_secs_f64() * 1e3,
                "p50_ms": r.p50.as_secs_f64() * 1e3,
                "p99_ms": r.p99.as_secs_f64() * 1e3,
            })
        })
        .collect();
    rows.sort_by(|a, b| total_ms(b).partial_cmp(&total_ms(a)).unwrap_or(std::cmp::Ordering::Equal));
    json!({ "rows": rows, "slab_live": slab_live_json() })
}

// ============================================================================
// 性能框架出口(/debug/perf;docs/arch/perf-framework.md §三)
// ============================================================================

/// 收集性能账(前缀过滤 + limit)。`epoch` = 只看最近一次 mark_epoch 之后
/// (engine 每 turn 开头打标 ⇒ “本轮”视角)。
fn perf_rows(prefix: &str, epoch: bool) -> Vec<owl_shared::metrics::TagReport> {
    let mut f = MetricsFilter::new().limit(ROWS_LIMIT);
    if !prefix.is_empty() {
        f = f.tag_prefix(prefix);
    }
    if epoch {
        collect_since_epoch(f)
    } else {
        collect_metrics(&f)
    }
}

/// 参考总时:取最大 total(通常 = 阶段包络 span)。树报告的 % 相对它。
fn ref_total(rows: &[owl_shared::metrics::TagReport]) -> std::time::Duration {
    rows.iter()
        .map(|r| r.total.max(r.self_total))
        .max()
        .unwrap_or(std::time::Duration::from_millis(1))
}

/// 树状纯文本报告(人读入口:curl /debug/perf)
pub fn perf_text(prefix: &str, epoch: bool) -> String {
    let rows = perf_rows(prefix, epoch);
    let refd = ref_total(&rows);
    let mut out = format!(
        "# owl perf  level={}  prefix={:?}  epoch={}  rows={}  ref_total={:.3}ms\n",
        level_name(),
        prefix,
        if epoch { "on" } else { "off" },
        rows.len(),
        refd.as_secs_f64() * 1e3,
    );
    out.push_str("# 列:n=样本数 self=排他耗时(不含子) % =相对 ref_total;子按 self 降序\n");
    out.push_str(&owl_shared::metrics::render_tree(&rows, refd));
    out
}

/// 机器可读投影(与 /debug/metrics 同形 + self/open/orphan/ooo)
pub fn perf_json(prefix: &str, epoch: bool) -> Value {
    let rows = perf_rows(prefix, epoch);
    json!({
        "level": level_name(),
        "prefix": prefix,
        "epoch": epoch,
        "ref_total_ms": ref_total(&rows).as_secs_f64() * 1e3,
        "rows": rows
            .into_iter()
            .map(|r| json!({
                "tag": r.tag, "count": r.count, "counter": r.counter,
                "total_ms": r.total.as_secs_f64() * 1e3,
                "self_ms": r.self_total.as_secs_f64() * 1e3,
                "avg_ms": r.avg.as_secs_f64() * 1e3,
                "p50_ms": r.p50.as_secs_f64() * 1e3,
                "p99_ms": r.p99.as_secs_f64() * 1e3,
                "max_ms": r.max.as_secs_f64() * 1e3,
                "open": r.open_spans, "orphan": r.orphan_ends,
                "ooo": r.out_of_order, "drop": r.dropped,
            }))
            .collect::<Vec<Value>>(),
    })
}

/// 分级在线切换(免重启 A/B:`off` = 零成本基线臂)
pub fn perf_set_level(name: &str) -> Result<&'static str, String> {
    let l = owl_shared::metrics::level::parse(name)
        .ok_or_else(|| format!("未知 perf level {name:?}(off|counters|host|gpu)"))?;
    owl_shared::metrics::set_level(l);
    Ok(level_name())
}

/// 单步/粘滞调试门状态投影(`GET /debug/step`;见 docs/arch/perf-framework.md §五)
pub fn step_json() -> Value {
    let s = owl_shared::stepgate::state();
    json!({
        "mode": s.mode,
        "compiled": s.compiled,
        "armed": s.armed,
        "every": s.every,
        "interval_ms": s.interval_ms,
        "interval_us": s.interval_us,
        "hits": s.hits,
        "advanced": s.advanced,
        "waits": s.waits,
        "waited_ms": s.waited_ms,
        "credits": s.credits,
        "released": s.released,
        "waiting": s.waiting,
    })
}

/// **发射账投影**(`GET /debug/launches[/<n>[/<kind>]]`;见 `owl_cuda::trace` 模块头)。
/// `kind` 可匹配命令类(`launch`/`htod`/`dtoh`/`free`…)、流名(`compute`/`h2d`/`d2h`)
/// 或核名子串。`ptrs` 以十六进制给出 —— 地址复用是跨流 UAF 类缺陷的唯一钥匙。
pub fn launches_json(n: usize, kind: Option<&str>) -> Value {
    let (in_ring, total, cap) = owl_cuda::trace::stats();
    let rows: Vec<Value> = owl_cuda::trace::recent(n, kind)
        .into_iter()
        .map(|e| {
            json!({
                "seq": e.seq,
                "t_us": e.host_ns / 1000,
                "cmd": e.cmd,
                "stream": e.stream,
                "name": e.name,
                "grid": [e.grid.0, e.grid.1, e.grid.2],
                "block": [e.block.0, e.block.1, e.block.2],
                "smem": e.smem,
                "bytes": e.bytes,
                "ids": e.ids,
                "out_id": e.out_id,
                "ptrs": e.ptrs.iter().map(|p| format!("{p:#x}")).collect::<Vec<_>>(),
            })
        })
        .collect();
    json!({
        "enabled": cap > 0,
        "capacity": cap,
        "in_ring": in_ring,
        "total": total,
        "n": n,
        "kind": kind,
        "rows": rows,
    })
}

/// **块账本投影**(`GET /debug/blocks`)。
///
/// 与发射账(`/debug/launches`)互补:那张账是"每次命令的块 id + 地址",
/// 本投影是**当前账本的权威快照**。两份内联不变量直接给出判决:
/// - `dup_ids`:同一 id 出现两次(账本条目被顶掉/复用)⇒ 写读可能落在不同物理块;
/// - `overlaps`:两个**活块**的地址区间相交 ⇒ 池把重叠区间发给了两个块。
/// (仓内 2026-10-08 E5 跨流回收律 / 2026-10-03 FI 表渐进腐败案都栽在这两件事上。)
pub fn blocks_json(rows: &[owl_cuda::BlockLedgerRow]) -> Value {
    let mut by_addr: Vec<&owl_cuda::BlockLedgerRow> = rows.iter().collect();
    by_addr.sort_by_key(|r| r.addr);
    let mut overlaps: Vec<Value> = Vec::new();
    for w in by_addr.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a.addr + a.bytes as u64 > b.addr {
            overlaps.push(json!({
                "a": {"id": a.id, "addr": format!("{:#x}", a.addr), "bytes": a.bytes},
                "b": {"id": b.id, "addr": format!("{:#x}", b.addr), "bytes": b.bytes},
            }));
        }
    }
    let mut ids: Vec<u64> = rows.iter().map(|r| r.id).collect();
    ids.sort_unstable();
    let dup_ids: Vec<u64> = ids.windows(2).filter(|w| w[0] == w[1]).map(|w| w[0]).collect();
    let blocks: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "bytes": r.bytes,
                "addr": format!("{:#x}", r.addr),
                "ledger": r.ledger,
            })
        })
        .collect();
    json!({
        "count": rows.len(),
        "total_bytes": rows.iter().map(|r| r.bytes).sum::<usize>(),
        "dup_ids": dup_ids,
        "overlaps": overlaps,
        "clean": dup_ids.is_empty() && overlaps.is_empty(),
        "blocks": blocks,
    })
}

// ============================================================================
// `/debug/vram` —— store 三层账 + **owner 维明细**(第 0 刀)
// ============================================================================

/// store 报告 → JSON(纯投影,零决策)。
///
/// **判据面**:`owners[]` 里同一 owner 的 `bytes` 逐轮**单调爬坡 = 泄漏**;
/// 持平 = 正常。`system` 桶 = 尚未接线 owner 的存量调用点(随 N4/N5 排空)。
pub fn vram_json(snap: &crate::actor::VramSnapshot) -> Value {
    let r = &snap.report;
    let owners: Vec<Value> = r
        .tiers
        .owners
        .iter()
        .map(|o| {
            json!({
                "owner": o.owner.label(),
                "lifecycle": o.owner.is_lifecycle(),
                "tier": format!("{:?}", o.tier),
                "kind": format!("{:?}", o.kind),
                "bytes": o.bytes,
                "mib": (o.bytes as f64) / (1024.0 * 1024.0),
                "entries": o.entries,
            })
        })
        .collect();
    let cells: Vec<Value> = r
        .tiers
        .rows
        .iter()
        .map(|c| {
            json!({
                "tier": format!("{:?}", c.tier),
                "kind": format!("{:?}", c.kind),
                "bytes": c.bytes,
                "entries": c.entries,
            })
        })
        .collect();
    json!({
        "tiers": {
            "gpu_used": r.tiers.gpu_used,
            "host_used": r.tiers.host_used,
            "disk_used": r.tiers.disk_used,
            "host_cap": r.tiers.host_cap,
            "disk_cap": r.tiers.disk_cap,
            "cells": cells,
        },
        "owners": owners,
        "kv": {
            "slots_total": r.kv.slots_total,
            "slots_free": r.kv.slots_free,
            "cached_blocks": r.kv.cached_blocks,
            "host_blocks": r.kv.host_blocks,
            "gpu_bytes": r.kv.gpu_bytes,
            "hits": r.kv.hits,
            "misses": r.kv.misses,
            "evicted": r.kv.evicted,
        },
        "state": {
            "rows_total": r.state.rows_total,
            "rows_free": r.state.rows_free,
            "snaps": r.state.snaps,
            "snap_bytes": r.state.snap_bytes,
            "snap_gpu": r.state.snap_gpu,
            "snap_host": r.state.snap_host,
            "hits": r.state.hits,
            "misses": r.state.misses,
        },
        "faces": {
            "active_turns": r.active_turns,
            "reserved_bytes": r.reserved_bytes,
            "session_held_bytes": r.session_held_bytes,
            "elastic_bytes": r.elastic_bytes,
            "cache_bytes": r.cache_bytes,
            "host_resident_bytes": r.host_resident_bytes,
            "host_free_bytes": r.host_free_bytes,
            "policy": r.policy,
        },
        // **事件账**(DeviceHandle 钩子的出借登记;第 3 刀)—— 与腿自报/块账房
        // 互为**独立来源**:三者对不上就是 bug(不是"取其一"的问题)。
        "events": {
            "entries": r.events.entries,
            "bytes": r.events.bytes,
            "hook_fail": r.events.hook_fail,
            "rows": r.events.rows.iter().map(|e| json!({
                "rank": e.rank,
                "owner": e.owner.label(),
                "lifecycle": e.owner.is_lifecycle(),
                "tier": format!("{:?}", e.tier),
                "kind": format!("{:?}", e.kind),
                "bytes": e.bytes,
                "mib": (e.bytes as f64) / (1024.0 * 1024.0),
                "entries": e.entries,
            })).collect::<Vec<_>>(),
        },
        // **句柄计数**(非零必须能解释:记账失败 / 窗未收)
        "handle": {
            "hook_fail": snap.hook_fail,
            "window_unclosed": snap.window_unclosed,
            "open_window_blocks": snap.open_window_blocks,
        },
        // **对账(全等门)**:事件账 id 集 vs 块账房 id 集,逐 rank。
        // `only_event` = 账里已销/账外有块(或后端自管块)—— 两者都必须能解释;
        // `bytes_*` 给量级(不止块数)。
        "recon": snap.blocks.iter().map(|(rank, r)| recon_json(*rank, r, &r_events(&snap.report))).collect::<Vec<_>>(),
    })
}

/// 事件账的逐 rank id 集(对账左值)
fn r_events(rep: &owl_store::StoreReport) -> Vec<(u8, Vec<u64>)> {
    rep.events.ids.clone()
}

/// **单个 rank 的对账投影**(纯比较,零决策):
/// - `only_block`:块账房有、事件账没有 ⇒ 该块**没走 DeviceHandle**(或后端自管,如 peer 面);
/// - `only_event`:事件账有、块账房没有 ⇒ 已销但块仍在(含"被图租约拒收的 free")或**后端已回收但本层未销**;
/// - `equal`:两面 id 集逐字相等(= 全等门通过)。
fn recon_json(
    rank: u8,
    dump: &Result<Vec<owl_iface::contract::BlockLedgerRow>, String>,
    events: &[(u8, Vec<u64>)],
) -> Value {
    let ev: Vec<u64> = events
        .iter()
        .filter(|(rk, _)| *rk == rank)
        .flat_map(|(_, ids)| ids.iter().copied())
        .collect();
    match dump {
        Err(e) => json!({"rank": rank, "available": false, "error": e}),
        Ok(rows) => {
            let blk: Vec<u64> = rows.iter().map(|r| r.id).collect();
            let only_event: Vec<u64> = ev.iter().copied().filter(|id| !blk.contains(id)).collect();
            let only_block: Vec<u64> = blk.iter().copied().filter(|id| !ev.contains(id)).collect();
            let bytes_blk: usize = rows.iter().map(|r| r.bytes).sum();
            json!({
                "rank": rank,
                "available": true,
                "blocks": blk.len(),
                "events": ev.len(),
                "equal": only_event.is_empty() && only_block.is_empty(),
                "only_event": only_event.len(),
                "only_block": only_block.len(),
                "only_event_ids": only_event.iter().take(16).collect::<Vec<_>>(),
                "only_block_ids": only_block.iter().take(16).collect::<Vec<_>>(),
                "block_bytes": bytes_blk,
            })
        }
    }
}
