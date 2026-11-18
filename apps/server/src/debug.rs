//! 诊断分账出口(GET /debug/metrics[/<prefix>];E5-M5 spec 分相探针消费面)。
//!
//! OWL_GPU_PROF=1 时 store 非空;逐核 GPU 时总账降序。TagReport 无
//! Serialize,手工投影(总 ms 计);`/debug/metrics/<prefix>` = tag 前缀
//! 过滤(spec. / load. …),无前缀 = 全量(行数上限见 [`ROWS_LIMIT];
//! 动态 tag 增殖面已由 OWL_FACT_PROBE 门控)。

use serde_json::{json, Value};

use owl_shared::metrics::{collect_metrics, collect_since_epoch, level_name, MetricsFilter};

use crate::vram_ledger;

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
    rows.sort_by(|a, b| {
        total_ms(b)
            .partial_cmp(&total_ms(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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

/// 调试门**控制回执**投影(`/debug/step/{mode,release,ms,us,every}`)。
///
/// `applied` = 控制通道有没有消费者(引擎未编入 stepgate feature / 已退出 ⇒
/// `false`,不谎报 ok);`action` = 人读的动作名。形状住这里,路由只转发。
pub fn step_action_json(applied: bool, action: &str) -> Value {
    let mut v = step_json();
    v["applied"] = json!(applied);
    v["action"] = json!(action);
    v
}

/// 日志面状态投影(`GET /debug/log`):当前级别 + tag 过滤表。
pub fn log_state_json() -> Value {
    let (allow, deny) = owl_shared::log::with_logger(|g| g.tags()).unwrap_or_default();
    json!({
        "level": owl_shared::log::level_name(),
        "log_build": owl_shared::log::LOG_BUILD,
        "tags_allow": allow,
        "tags_deny": deny,
    })
}

/// **队列/容量快照投影**(`GET /debug/queue`;每拍事实,零等待)。
///
/// 数据来自 `EngineClient::queue_facts`(泵循环每拍写 `FactsCell`),本函数
/// **纯投影** —— 旧版形状内联在 `main::handle` 里,现搬来这里供单测与复用。
pub fn queue_json(snap: &owl_engine::QueueFacts) -> Value {
    json!({
        "queue_len": snap.queue_len,
        "active": snap.active,
        "batch_steps": snap.batch_steps,
        "max_batch": snap.max_batch,
        "graph": snap.has_graph,
        "queued": snap.queued.iter().map(|(t, s, ms)| json!({
            "turn": t, "session": format!("{s:#x}"), "waited_ms": ms,
        })).collect::<Vec<_>>(),
        "rejects": snap.rejects.iter().map(|(s, code, n)| json!({
            "session": format!("{s:#x}"), "code": code, "count": n,
        })).collect::<Vec<_>>(),
        "wait_hist": {
            "lt_50ms": snap.wait_hist[0],
            "lt_200ms": snap.wait_hist[1],
            "lt_1s": snap.wait_hist[2],
            "lt_5s": snap.wait_hist[3],
            "ge_5s": snap.wait_hist[4],
        },
        "capacity": snap.capacity.map(|c| json!({
            "free_pages": c.free_pages,
            "cache_pages": c.cache_pages,
            "reserved_pages": c.reserved_pages,
            "usable_pages": c.usable_pages,
            "n_slots": c.n_slots,
            "n_mem": c.n_mem,
            "n_eff": c.n_eff,
        })),
    })
}

/// **发射账投影**(`GET /debug/launches[/<n>[/<kind>]]`)。
///
/// 数据面在 `owl_engine::launch_trace`（引擎给的稳定形状）；本函数只做 JSON。
/// `kind` 可匹配命令类(`launch`/`htod`/`dtoh`/`free`…)、流名(`compute`/`h2d`/`d2h`)
/// 或核名子串。`ptrs` 以十六进制给出 —— 地址复用是跨流 UAF 类缺陷的唯一钥匙。
pub fn launches_json(n: usize, kind: Option<&str>) -> Value {
    let t = owl_engine::launch_trace(n, kind);
    let rows: Vec<Value> = t
        .rows
        .iter()
        .map(|e| {
            json!({
                "seq": e.seq,
                "t_us": e.t_us,
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
        "enabled": t.enabled,
        "capacity": t.capacity,
        "in_ring": t.in_ring,
        "total": t.total,
        "n": n,
        "kind": kind,
        "rows": rows,
    })
}

///
/// **块账本投影**(`GET /debug/blocks`)。
///
/// 与发射账(`/debug/launches`)互补:那张账是"每次命令的块 id + 地址",
/// 本投影是**当前账本的权威快照**。两份内联不变量直接给出判决:
/// - `dup_ids`:同一 id 出现两次(账本条目被顶掉/复用)⇒ 写读可能落在不同物理块;
/// - `overlaps`:两个**活块**的地址区间相交 ⇒ 池把重叠区间发给了两个块。 (仓内 2026-10-08 E5 跨流回收律 / 2026-10-03 FI 表渐进腐败案都栽在这两件事上。)
pub fn blocks_json(rows: &[owl_engine::BlockRow]) -> Value {
    let mut by_addr: Vec<&owl_engine::BlockRow> = rows.iter().collect();
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
    let dup_ids: Vec<u64> = ids
        .windows(2)
        .filter(|w| w[0] == w[1])
        .map(|w| w[0])
        .collect();
    let blocks: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "name": r.name,
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

/// 逐 rank 的块账投影（`/debug/blocks`；单卡档输出与旧形状一致）。
pub fn blocks_ranks_json(
    ranks: &[(u8, Result<Vec<owl_engine::BlockRow>, String>)],
) -> Value {
    if ranks.len() == 1 {
        return match &ranks[0].1 {
            Ok(rows) => blocks_json(rows),
            Err(e) => json!({"ok": false, "error": e.clone()}),
        };
    }
    let list: Vec<Value> = ranks
        .iter()
        .map(|(rank, r)| match r {
            Ok(rows) => json!({"rank": rank, "ok": blocks_json(rows)}),
            Err(e) => json!({"rank": rank, "error": e.clone()}),
        })
        .collect();
    json!({"ranks": list})
}

// ============================================================================
// `/debug/vram` —— store 三层账 + **owner 维明细**(第 0 刀)
// ============================================================================

/// store 报告 → JSON(纯投影,零决策)。
///
/// **判据面**:`owners[]` 里同一 owner 的 `bytes` 逐轮**单调爬坡 = 泄漏**;
/// 持平 = 正常。`system` 桶 = 尚未接线 owner 的存量调用点(随 N4/N5 排空)。
// 结构性豁免(2026-11-02 clippy 清理):排序键是复合比较(多字段/方向混合),抽 key 会丢掉可读性
#[allow(clippy::unnecessary_sort_by)]
pub fn vram_json(snap: &owl_engine::DiagSnapshot) -> Value {
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
    // 互斥账本(2026-11-15 P0):所有桶逐 rank 闭合;`tiers.cells` 只留
    // 事件账独占项,腿自报的 KvBlock/Snap 走 `tiers.views`(不可相加)。
    let ledger_json = vram_ledger::json_from(&vram_ledger::build(snap));
    let tiers_json = vram_ledger::tiers_json(r);
    let pools: Vec<Value> = snap
        .pools
        .iter()
        .map(|(rank, r)| match r {
            Ok((used, used_high, reserved, reserved_high)) => json!({
                "rank": rank,
                "used": used,
                "used_high": used_high,
                "reserved": reserved,
                "reserved_high": reserved_high,
                "used_mib": (*used as f64) / 1048576.0,
                "used_high_mib": (*used_high as f64) / 1048576.0,
                "reserved_mib": (*reserved as f64) / 1048576.0,
                "reserved_high_mib": (*reserved_high as f64) / 1048576.0,
                // 池缓存 = reserved − used(已 free 仍占物理)
                "cached_mib": ((*reserved).saturating_sub(*used) as f64) / 1048576.0,
            }),
            Err(e) => json!({ "rank": rank, "error": e }),
        })
        .collect();
    // **块账房逐名汇总**(诊断:谁在占显存;按字节降序 top 40)
    let block_rows: Vec<Value> = {
        let mut by: std::collections::BTreeMap<&str, (usize, usize)> =
            std::collections::BTreeMap::new();
        for (_rank, rows) in &snap.blocks {
            if let Ok(rows) = rows {
                for r in rows {
                    let e = by.entry(r.name.as_str()).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += r.bytes;
                }
            }
        }
        let mut v: Vec<(&str, usize, usize)> =
            by.into_iter().map(|(n, (c, b))| (n, c, b)).collect();
        v.sort_by(|a, b| b.2.cmp(&a.2));
        v.iter()
            .take(40)
            .map(|(n, c, b)| {
                json!({"name": n, "count": c, "bytes": b,
            "mib": (*b as f64) / 1048576.0})
            })
            .collect()
    };
    // **驱动残差**(P5):物理 − free − store live − 池缓存。这一项就是
    // CUDA ctx / 模块 / cuBLAS / 图 exec / 分配器元数据 —— 块账拦不住的部分。
    // rank0 兼容面:残差按 **rank0 自己的** 事件账/池缓存算。旧实现把
    // 两 rank 的 `tiers.gpu_used` 从 rank0 的 total 里减 ⇒ 恒为 0,已修。
    let device_json = match &snap.device {
        Ok((free, total)) => {
            let (pool_cached, pool_reserved) = snap
                .pools
                .iter()
                .find(|(rank, _)| *rank == 0)
                .and_then(|(_, row)| row.as_ref().ok())
                .map(|(used, _, reserved, _)| {
                    (reserved.saturating_sub(*used), *reserved)
                })
                .unwrap_or((0, 0));
            let events_rank0: u64 = r
                .events
                .rows
                .iter()
                .filter(|e| e.rank == 0)
                .map(|e| e.bytes as u64)
                .sum();
            let residual = total
                .saturating_sub(*free)
                .saturating_sub(events_rank0)
                .saturating_sub(pool_cached);
            json!({
                "free": free,
                "total": total,
                "pool_reserved": pool_reserved,
                "pool_cached": pool_cached,
                "events_rank0": events_rank0,
                "residual": residual,
                "free_mib": (*free as f64) / 1048576.0,
                "total_mib": (*total as f64) / 1048576.0,
                "pool_cached_mib": (pool_cached as f64) / 1048576.0,
                "residual_mib": (residual as f64) / 1048576.0,
                "note": "兼容面(rank0);闭合看 ledger",
            })
        }
        Err(e) => json!({ "error": e }),
    };
    // **L1 会话资产投影**(2026-11-12 三层架构):会话 → tail 快照/消息边界。
    // 只读、不碰 store 账;用于验证"append 命中上一条消息末尾"与 pin 面。
    let sessions: Vec<Value> = snap
        .sessions
        .iter()
        .map(|s| {
            json!({
                "id": s.id,
                "history_len": s.history_len,
                "tail_snap": s.tail_snap.map(|k| format!("{k:?}")),
                "tail_at": s.tail_at,
                "msg_snaps": s.msg_snaps,
                "msg_snap_roles": s.msg_snap_roles.iter().map(|r| format!("{r:?}")).collect::<Vec<_>>(),
                "msg_snap_keys": s.msg_snap_keys.iter()
                    .map(|k| k.map(|k| format!("{k:?}")))
                    .collect::<Vec<_>>(),
                "local_chain": s.local_chain,
                "last_use": s.last_use,
            })
        })
        .collect();
    json!({
        "device": device_json,
        "ledger": ledger_json,
        "sessions": sessions,
        "pools": pools,
        "blocks_top": block_rows,
        "tiers": tiers_json,
        "owners": owners,
        "kv": {
            "slots_total": r.kv.slots_total,
            "slots_free": r.kv.slots_free,
            "cached_blocks": r.kv.cached_blocks,
            "host_blocks": r.kv.host_blocks,
            "disk_blocks": r.kv.disk_blocks,
            "gpu_bytes": r.kv.gpu_bytes,
            "hits": r.kv.hits,
            "misses": r.kv.misses,
            "evicted": r.kv.evicted,
            // 账漂指纹:非零 = 有登记入口漏清旧宿主副本(2026-10-24 案)
            "host_duplicate_landings": r.kv.host_duplicate_landings,
            // 账漂指纹:非零 = 树账说宿主、内容已不在(幂等删除已计数;2026-10-25 案)
            "host_drop_missing": r.kv.host_drop_missing,
        },
        // **槽引用 / 可下沉性诊断**(2026-11-08 池毒化立案):
        // `chain_unavailable > 0` 且设备被缓存满时，腾位会系统性不足 ——
        // 看 `chain_busy`/`stuck_refs0`/`stuck_shared` 定位是谁在持着。
        "kv_diag": {
            "refs_hist": r.kv_diag.refs_hist,
            "bound_refs_hist": r.kv_diag.bound_refs_hist,
            "chains": r.kv_diag.chains,
            "chain_unavailable": r.kv_diag.chain_unavailable,
            "chain_busy": r.kv_diag.chain_busy,
            "chain_all_host": r.kv_diag.chain_all_host,
            "chain_missing": r.kv_diag.chain_missing,
            "stuck_blocks": r.kv_diag.stuck_blocks,
            "stuck_refs0": r.kv_diag.stuck_refs0,
            "stuck_shared": r.kv_diag.stuck_shared,
        },
        "state": {
            // **淘汰序预览**(会话感知:同会话 depth 降序 = 末端先走)
            "evict_order": snap.snap_evict.iter().map(|e| json!({
                "key": format!("{:?}", e.key),
                "role": format!("{:?}", e.role),
                // 无归属 = null(会话 id 0 是合法 id,不能用 0 表示"无主")
                "session": if e.session == u64::MAX { serde_json::Value::Null } else { json!(e.session) },
                "depth": e.depth,
                "transient": e.transient,
            })).collect::<Vec<_>>(),
            "rows_total": r.state.rows_total,
            "rows_free": r.state.rows_free,
            "snaps": r.state.snaps,
            "snap_bytes": r.state.snap_bytes,
            "snap_gpu": r.state.snap_gpu,
            "snap_host": r.state.snap_host,
            "hits": r.state.hits,
            "misses": r.state.misses,
        },
        // **磁盘层**(S6;None = 关层):账上驻留 + 目录 + 配额
        "disk": {
            "kv": r.disk_kv.as_ref().map(|d| json!({
                "root": d.root, "budget_bytes": d.budget_bytes,
                "used_bytes": d.used_bytes, "entries": d.entries,
                "evicted_bytes": d.evicted_bytes,
            })),
            "snap": r.disk_snap.as_ref().map(|d| json!({
                "root": d.root, "budget_bytes": d.budget_bytes,
                "used_bytes": d.used_bytes, "entries": d.entries,
                "evicted_bytes": d.evicted_bytes,
            })),
            // 盘层归属:会话 → 盘上 KV/快照占用(0 是合法会话 id,
            // 无主用 NO_SESSION = u64::MAX,JSON 里显式 null)。
            "by_session": r.disk_sessions.iter().map(|s| json!({
                "session": if s.session == u64::MAX {
                    serde_json::Value::Null
                } else {
                    json!(s.session)
                },
                "kv_blocks": s.kv_blocks,
                "kv_bytes": s.kv_bytes,
                "snap_entries": s.snap_entries,
                "snap_bytes": s.snap_bytes,
            })).collect::<Vec<_>>(),
        },
        // **MemoryPlan**(V0/V1):固定项逐项来源 + 主池 + 预留 + 卡面闭合。
        "plan": r.plan.as_ref().map(|p| {
            let c = p.check_card();
            json!({
                "epoch": p.epoch,
                "summary": p.summary(),
                "card": {
                    "total": p.card.total_bytes,
                    "free_at_boot": p.card.free_at_boot_bytes,
                    "reserve": p.card.reserve_bytes,
                },
                "fixed": p.fixed.iter().map(|x| json!({
                    "id": x.id.label(),
                    "bytes": x.bytes,
                    "mib": (x.bytes as f64) / 1048576.0,
                    "source": x.source.label(),
                    "tier": format!("{:?}", x.tier),
                    "owner": x.owner.label(),
                })).collect::<Vec<_>>(),
                "pool": {
                    "bytes": p.pool.bytes,
                    "pages": p.pool.pages,
                    "page_tokens": p.pool.page_tokens,
                    "tokens": p.pool.tokens,
                    "per_token_bytes": p.pool.per_token_bytes,
                    "block_bytes": p.pool.block_bytes,
                    "gdn_slots": p.pool.gdn_slots,
                    "gdn_row_bytes": p.pool.gdn_row_bytes,
                },
                "check": {
                    "fits": c.fits(),
                    "total_bytes": c.total_bytes,
                    "over_free_bytes": c.over_free_bytes,
                    "over_total_bytes": c.over_total_bytes,
                },
            })
        }),
        "faces": {
            "active_turns": r.active_turns,
            // **P0-4 Grant 面**：批准槽总量 / 在册 turn 数
            "reserved_slots": r.reserved_slots,
            "in_flight_turns": r.in_flight_turns,
            "reserved_bytes": r.reserved_bytes,
            // **Grant 分用途账**：Load/Row/Snap/Temp 与 ActiveFloor 分开，
            // 不污染调度器的 KV `reserved_slots`。
            "grants": r.grants.iter().map(|g| json!({
                "kind": g.kind,
                "grants": g.grants,
                "slots": g.slots,
                "bytes": g.bytes,
                "turns": g.turns,
            })).collect::<Vec<_>>(),
            "session_held_bytes": r.session_held_bytes,
            "elastic_bytes": r.elastic_bytes,
            "cache_bytes": r.cache_bytes,
            "host_resident_bytes": r.host_resident_bytes,
            "host_free_bytes": r.host_free_bytes,
            "policy": r.policy,
        },
        // **合批事实**(B5.1):上限 / 已走批步数 / 批图在场。
        // 服务面判据:`max_batch>1` 而 `steps` 不涨 = 批门从未成立
        // (并发不足,或同会话被引擎串行 —— P4 起串行在引擎队列里)。
        "batch": {
            "max_batch": snap.batch.max_batch,
            "steps": snap.batch.steps,
            "graph": snap.batch.graph,
            // **准入容量**（物理空闲 + 可回收缓存 + 预留承诺 三者同时成立才放人）
            // **排队快照**（按等待降序，最多 16）：turn / session / waited_ms
            "queued": snap.batch.queued.iter().map(|(t, s, ms)| json!({
                "turn": t,
                "session": format!("{s:#x}"),
                "waited_ms": ms,
            })).collect::<Vec<_>>(),
            "capacity": {
                "free_pages": snap.batch.free_pages,
                "cache_pages": snap.batch.cache_pages,
                "reserved_pages": snap.batch.reserved_pages,
                "usable_pages": snap.batch.usable_pages,
                "n_slots": snap.batch.n_slots,
                "n_mem": snap.batch.n_mem,
                "n_eff": snap.batch.n_eff,
            },
        },
        // **事件账**(DeviceHandle 钩子的出借登记;第 3 刀)—— 与腿自报/块账房
        // 互为**独立来源**:三者对不上就是 bug(不是"取其一"的问题)。
        "events": {
            "entries": r.events.entries,
            "bytes": r.events.bytes,
            "hook_fail": r.events.hook_fail,
            // 片级换层跳过(口径不支持,不是失败;层账以腿自报为准)
            "partial_moves": r.events.partial_moves,
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
            // **lifecycle 在借清单**(N4/N5 未归还的逐块指纹:名字 + id)。
            // 判据:同一 `name` 的条目**逐轮增长** = 该处没在节点上归还。
            "open": r.events.open_lifecycle.iter().map(|l| json!({
                "id": l.id,
                "rank": l.rank,
                "name": l.name,
                "owner": l.owner.label(),
                "kind": format!("{:?}", l.kind),
                "bytes": l.bytes,
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
// 结构性豁免(2026-11-02 clippy 清理):排序键是复合比较(多字段/方向混合),抽 key 会丢掉可读性
#[allow(clippy::unnecessary_sort_by)]
fn recon_json(
    rank: u8,
    dump: &Result<Vec<owl_engine::BlockRow>, String>,
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
            // **未入账块的按名分账**(2026-10-20;判"该不该由 store 管"的唯一材料):
            // 名字 = 服务端分配点(`selfcheck.*` / `kv.*` / 层名 / 核名)。
            let mut hist: std::collections::BTreeMap<(&str, usize), (usize, usize)> =
                std::collections::BTreeMap::new();
            for r in rows.iter().filter(|r| !ev.contains(&r.id)) {
                let e = hist.entry((r.name.as_str(), 0)).or_insert((0, 0));
                e.0 += 1;
                e.1 += r.bytes;
            }
            let mut hist_v: Vec<(&str, usize, usize)> =
                hist.into_iter().map(|((n, _), (c, b))| (n, c, b)).collect();
            hist_v.sort_by(|a, b| b.2.cmp(&a.2));
            let only_block_names: Vec<Value> = hist_v
                .iter()
                .take(12)
                .map(|(n, c, b)| json!({"name": n, "count": c, "bytes": b}))
                .collect();
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
                "only_block_names": only_block_names,
                "block_bytes": bytes_blk,
            })
        }
    }
}
