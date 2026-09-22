//! 加载期计时插桩(T2-三 搬运;origin: xinfer utils/load_stats.rs,S3-P0)。
//!
//! - 门控 OWL_LOAD_DEBUG(兼容 XINFER_LOAD_DEBUG);关闭 = 近零成本;
//! - 张量级四段计时:fetch/to_cpu/repack/upload;相位级计时;
//! - dump() 汇总按 kind 聚合 + top-N 慢张量(输出走 tracing)。

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static ENABLED: OnceLock<bool> = OnceLock::new();

#[inline]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        let owl = std::env::var("OWL_LOAD_DEBUG").is_ok_and(|v| v == "1");
        let xinfer = std::env::var("XINFER_LOAD_DEBUG").is_ok_and(|v| v == "1");
        owl || xinfer
    })
}

#[inline]
pub fn now() -> Option<Instant> {
    if enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

#[inline]
fn us(d: Duration) -> u64 {
    d.as_micros() as u64
}

#[derive(Default, Clone)]
struct Rec {
    n: u64,
    bytes: u64,
    fetch_us: u64,
    to_cpu_us: u64,
    repack_us: u64,
    upload_us: u64,
}

struct TopEntry {
    cost_us: u64,
    tag: String,
    kind: &'static str,
    mb: u64,
}

static RECS: Mutex<Vec<(&'static str, Rec)>> = Mutex::new(Vec::new());
static PHASES: Mutex<Vec<(String, u64)>> = Mutex::new(Vec::new());
static TOP: Mutex<Vec<TopEntry>> = Mutex::new(Vec::new());
const TOP_KEEP: usize = 8;

#[allow(clippy::too_many_arguments)]
pub fn tensor(
    kind: &'static str,
    tag: &str,
    bytes: u64,
    fetch: Duration,
    to_cpu: Duration,
    repack: Duration,
    upload: Duration,
) {
    if !enabled() {
        return;
    }
    let (f, c, r, u) = (us(fetch), us(to_cpu), us(repack), us(upload));
    {
        let mut recs = RECS.lock().unwrap();
        let rec = recs
            .iter_mut()
            .find(|(k, _)| *k == kind)
            .map(|x| &mut x.1);
        if let Some(rec) = rec {
            rec.n += 1;
            rec.bytes += bytes;
            rec.fetch_us += f;
            rec.to_cpu_us += c;
            rec.repack_us += r;
            rec.upload_us += u;
        } else {
            recs.push((
                kind,
                Rec {
                    n: 1,
                    bytes,
                    fetch_us: f,
                    to_cpu_us: c,
                    repack_us: r,
                    upload_us: u,
                },
            ));
        }
    }
    let cost = c + r;
    if cost > 0 {
        let mut top = TOP.lock().unwrap();
        let entry = TopEntry {
            cost_us: cost,
            tag: tag.to_string(),
            kind,
            mb: bytes / (1024 * 1024),
        };
        if top.len() < TOP_KEEP {
            top.push(entry);
        } else {
            let min_idx = top
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.cost_us)
                .map(|(i, _)| i)
                .unwrap();
            if top[min_idx].cost_us < cost {
                top[min_idx] = entry;
            }
        }
    }
}

pub fn phase(name: &str, started: Option<Instant>) {
    if let (true, Some(s)) = (enabled(), started) {
        PHASES
            .lock()
            .unwrap()
            .push((name.to_string(), us(s.elapsed())));
    }
}

pub fn dump(rank: usize) {
    if !enabled() {
        return;
    }
    let recs = RECS.lock().unwrap();
    let mut total_us = 0u64;
    for (kind, r) in recs.iter() {
        tracing::info!(
            "[load-stats] r{} tensors kind={} n={} bytes={:.2}GB fetch={:.2}s to_cpu={:.2}s repack={:.2}s upload={:.2}s",
            rank,
            kind,
            r.n,
            r.bytes as f64 / 1e9,
            r.fetch_us as f64 / 1e6,
            r.to_cpu_us as f64 / 1e6,
            r.repack_us as f64 / 1e6,
            r.upload_us as f64 / 1e6,
        );
        total_us += r.fetch_us + r.to_cpu_us + r.repack_us + r.upload_us;
    }
    drop(recs);
    let phases = PHASES.lock().unwrap();
    let phase_str = phases
        .iter()
        .map(|(n, us)| format!("{}={:.2}s", n, *us as f64 / 1e6))
        .collect::<Vec<_>>()
        .join(" | ");
    if !phase_str.is_empty() {
        tracing::info!("[load-stats] r{} phases: {}", rank, phase_str);
    }
    drop(phases);
    let top = TOP.lock().unwrap();
    let mut entries: Vec<&TopEntry> = top.iter().collect();
    entries.sort_by(|a, b| b.cost_us.cmp(&a.cost_us));
    for e in entries.iter().take(TOP_KEEP) {
        tracing::info!(
            "[load-stats] r{} top [{}] {} {}MB to_cpu+repack={:.0}ms",
            rank,
            e.kind,
            e.tag,
            e.mb,
            e.cost_us as f64 / 1e3
        );
    }
    drop(top);
    tracing::info!(
        "[load-stats] r{} tensor-accounted total={:.2}s",
        rank,
        total_us as f64 / 1e6
    );
}
