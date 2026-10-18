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
