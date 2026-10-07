//! 诊断分账出口(GET /debug/metrics[/<prefix>];E5-M5 spec 分相探针消费面)。
//!
//! OWL_GPU_PROF=1 时 store 非空;逐核 GPU 时总账降序。TagReport 无
//! Serialize,手工投影(总 ms 计);`/debug/metrics/<prefix>` = tag 前缀
//! 过滤(spec. / load. …),无前缀 = 全量(行数上限见 [`ROWS_LIMIT];
//! 动态 tag 增殖面已由 OWL_FACT_PROBE 门控)。

use serde_json::{json, Value};

use owl_shared::metrics::{collect_metrics, MetricsFilter};

/// 单次查询行数上限(防动态 tag 增殖拖爆应答)
const ROWS_LIMIT: usize = 512;

/// 投影行排序键(总耗时 ms)
fn total_ms(row: &Value) -> f64 {
    row["total_ms"].as_f64().unwrap_or(0.0)
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
    json!({ "rows": rows })
}
