//! **双会话"一 decode 一 prefill"端到端黑盒门**（2026-11-06）。
//!
//! 场景 = 真机最常见并发形状：会话 A 已经在**流式输出**（decode），
//! 会话 B 此时带着**长 prompt** 进来（prefill）。验收：
//!   ① 两边都完成（不互相打死）；
//!   ② **真并发档**：两个 turn 同时在册（`/debug/queue.active >= 2`），
//!      解码侧在预填窗口里继续出 token，且相邻 token 间隔有界；
//!   ③ **队列档**（TP2 生产 `max_batch=1`）：第二路排队，两边仍完成，
//!      解码侧不中途停摆。
//!
//! 反向形状（先长 prompt prefill、后 decode）同样覆盖。
//!
//! 前置（两条形态都覆盖）：
//! - **真并发**：单卡 + `max_batch=2`（TP 不与合批共存）+ `gdn_slots>=2` +
//!   无选择层 + spec off；要 decode 不被整段 prefill 冻结，再开
//!   `runtime.slice_every>0`（实测 8 ≈ 每 3 个 decode 步插 1 片 prefill，
//!   窗口内 max_gap ≈ 0.9s；=0 时 ≈ 3.8s ⇒ 本门红）。
//! - **队列形态**（TP2 生产档）：第二路排队，测试报 `mode=queued`。
//!
//! 用法：
//! ```text
//! tools/fixture/run-cli-decode-prefill.sh
//! tools/fixture/run-cli-decode-prefill.sh --only decode_vs_prefill_24k --json
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::client::{ChatReply, ChatRequest, Client};
use crate::workload::report::{ScenarioReport, TurnRecord};
use crate::workload::Args;

/// 随机词池（prompt 不许重复文本；AGENTS 测量纪律）。
const WORDS: &[&str] = &[
    "amber", "basin", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "iris", "jasper",
    "kelp", "lumen", "meadow", "nimbus", "onyx", "prism", "quartz", "raven", "sable", "tundra",
    "umber", "vapor", "willow", "xenon", "yarrow", "zephyr", "anchor", "beacon", "cinder",
    "drift", "echo", "fable", "glacier", "hollow", "indigo", "juniper", "kestrel", "lantern",
    "marble", "nectar", "orbit", "pebble", "quiver", "ripple", "summit", "thistle", "upland",
    "vertex", "whisper", "yonder", "zinnia", "arbor", "bramble", "cedar", "dune", "fern", "grove",
    "heather", "inlet", "jetty", "knoll", "lagoon", "moss", "nook", "orchard", "pine", "quill",
    "reed", "stone", "trail", "vale", "wren", "yew", "zenith", "alder", "birch", "cliff", "dew",
    "elm", "frost", "glen", "haze", "isle", "kite", "leaf", "mist", "nest", "oak", "path", "quay",
    "root", "sage", "tide", "vine", "wave", "yard", "zinc", "acorn", "bloom", "cloud", "creek",
    "dawn", "dusk", "field", "grain", "hill",
];

const LONG_QUESTION: &str = "Write a very long, detailed technical essay about the history of \
computer architecture, from vacuum tubes and core memory to chiplets and hardware \
accelerators. Use at least twelve sections with headings and concrete examples, and keep \
writing in full paragraphs until you reach the token limit. Do not summarize or stop early.";

/// 一个用例：`decode_max` = 解码侧生成长度；`prefill_tokens` = 长 prompt 的 token 量。
struct Case {
    name: &'static str,
    decode_max: usize,
    prefill_tokens: usize,
    /// true = decode 先起、prefill 后进；false = prefill 先起、decode 后进。
    decode_first: bool,
}

const CASES: &[Case] = &[
    Case {
        name: "decode_vs_prefill_8k",
        decode_max: 1024,
        prefill_tokens: 8_000,
        decode_first: true,
    },
    Case {
        name: "decode_vs_prefill_24k",
        decode_max: 2048,
        prefill_tokens: 24_000,
        decode_first: true,
    },
    Case {
        name: "prefill_vs_decode_24k",
        decode_max: 512,
        prefill_tokens: 24_000,
        decode_first: false,
    },
];

pub fn run(args: &Args, pid: u32) -> i32 {
    if let Some(only) = args.only.as_deref() {
        if !CASES.iter().any(|c| c.name == only) {
            eprintln!(
                "[decode-prefill] --only {only:?} 未匹配（{:?}）",
                CASES.iter().map(|c| c.name).collect::<Vec<_>>()
            );
            return 2;
        }
    }
    let mut reports = Vec::new();
    for case in CASES {
        if args.only.as_deref().is_some_and(|o| o != case.name) {
            continue;
        }
        let rep = match run_case(args, case, pid) {
            Ok(r) => r,
            Err(e) => ScenarioReport {
                name: case.name.to_string(),
                pass: false,
                detail: e,
                turns: Vec::new(),
            },
        };
        if !args.json {
            crate::workload::report::print_report(&rep);
        }
        reports.push(rep);
    }
    if args.json {
        let arr: Vec<Value> = reports.iter().map(ScenarioReport::to_json).collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&arr).unwrap_or_else(|e| format!("[{{\"error\":\"{e}\"}}]"))
        );
    }
    let failed = reports.iter().filter(|r| !r.pass).count();
    eprintln!(
        "[decode-prefill] {}/{} scenarios passed",
        reports.len() - failed,
        reports.len()
    );
    if failed == 0 { 0 } else { 1 }
}

/// 一个流式进程（请求 + 逐 delta 时刻）。
struct Streamed {
    reply: ChatReply,
    started: Instant,
    ended: Instant,
    deltas: Vec<Instant>,
}

fn spawn_stream(url: String, model: Option<String>, req: ChatRequest) -> BroadcastStream {
    let (first_tx, first_rx) = std::sync::mpsc::channel::<()>();
    let times: Arc<Mutex<Vec<Instant>>> = Arc::new(Mutex::new(Vec::new()));
    let t = Arc::clone(&times);
    let started = Instant::now();
    let handle = thread::spawn(move || {
        let client = Client::new(&url, model);
        let mut first = true;
        let r = client.chat_stream(req, |_| {
            let now = Instant::now();
            if let Ok(mut v) = t.lock() {
                v.push(now);
            }
            if first {
                first = false;
                let _ = first_tx.send(());
            }
        });
        let ended = Instant::now();
        (r, started, ended)
    });
    BroadcastStream { handle, first_rx, times }
}

struct BroadcastStream {
    handle: thread::JoinHandle<(std::io::Result<ChatReply>, Instant, Instant)>,
    first_rx: std::sync::mpsc::Receiver<()>,
    times: Arc<Mutex<Vec<Instant>>>,
}

impl BroadcastStream {
    fn into_streamed(self) -> Result<Streamed, String> {
        let deltas = self.times.lock().map(|v| v.clone()).unwrap_or_default();
        let (r, started, ended) = self.handle.join().map_err(|_| "流式线程 panic".to_string())?;
        let reply = r.map_err(|e| format!("chat 失败: {e}"))?;
        Ok(Streamed {
            reply,
            started,
            ended,
            deltas,
        })
    }
}

#[derive(Clone, Debug, Default)]
struct Monitor {
    max_active: u64,
    max_queue: u64,
    active2_samples: u64,
    /// 第二路发出后、仍观察到 `queue_len>0` 的采样数（排队档的直接证据）。
    queued_after_second: u64,
    /// 第二路发出后**连续**排队的采样数 / 最长连续段。
    queued_streak: u64,
    max_queued_streak: u64,
    free_min: Option<u64>,
    host_max: u64,
}

fn run_case(args: &Args, case: &Case, pid: u32) -> Result<ScenarioReport, String> {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tag = format!("{}-{pid}-{ms}", case.name);
    let sess_a = format!("dp-{tag}-A");
    let sess_b = format!("dp-{tag}-B");
    let decode_prompt = format!("[run {tag}]\n{LONG_QUESTION}");
    let prefill_prompt = make_prefill(case.prefill_tokens, &tag);

    // 角色 → 请求。`first` 是先起的那一路（decode-first 时 = decode；
    // 反向 = prefill），`second` 后进。
    let req = |session: &str, prompt: &str, max_tokens: usize| ChatRequest {
        messages: vec![json!({"role": "user", "content": prompt})],
        tools: None,
        max_tokens,
        temperature: 0.0,
        session: Some(session.to_string()),
        chat_template_kwargs: None,
    };
    let (first_req, first_is_decode) = if case.decode_first {
        (req(&sess_a, &decode_prompt, case.decode_max), true)
    } else {
        (req(&sess_a, &prefill_prompt, 16), false)
    };
    let second_req = if case.decode_first {
        req(&sess_b, &prefill_prompt, 16)
    } else {
        req(&sess_b, &decode_prompt, case.decode_max)
    };

    // 监控（`/debug/queue` cheap 高频；`/debug/vram` 降频）。
    let stop = Arc::new(AtomicBool::new(false));
    let second_started = Arc::new(AtomicBool::new(false));
    let stop_mon = Arc::clone(&stop);
    let second_flag = Arc::clone(&second_started);
    let mon_state = Arc::new(Mutex::new(Monitor::default()));
    let ms_clone = Arc::clone(&mon_state);
    let mon_url = args.url.clone();
    let mon_model = args.model.clone();
    let monitor = thread::spawn(move || {
        let client = Client::new(&mon_url, mon_model);
        let mut tick = 0u32;
        while !stop_mon.load(Ordering::Relaxed) {
            if let Ok(q) = client.get_json("/debug/queue") {
                let active = q.get("active").and_then(Value::as_u64).unwrap_or(0);
                let queued = q.get("queue_len").and_then(Value::as_u64).unwrap_or(0);
                if std::env::var_os("OWL_DP_DEBUG").is_some() {
                    eprintln!("[dp-mon] active={active} queue={queued}");
                }
                if let Ok(mut st) = ms_clone.lock() {
                    st.max_active = st.max_active.max(active);
                    st.max_queue = st.max_queue.max(queued);
                    if active >= 2 {
                        st.active2_samples += 1;
                    }
                    // 第二路发出后**持续**排队（连续采样）⇒ 它是被 queue 住的。
                    // 只看"有无"会把 submit→admit 之间的 50ms 瞬时排队误判成排队档。
                    if queued > 0 && second_flag.load(Ordering::Relaxed) {
                        st.queued_after_second += 1;
                        st.queued_streak += 1;
                        st.max_queued_streak = st.max_queued_streak.max(st.queued_streak);
                    } else {
                        st.queued_streak = 0;
                    }
                }
            }
            if tick.is_multiple_of(10) {
                if let Ok(v) = client.get_json("/debug/vram") {
                    let free = v
                        .get("kv")
                        .and_then(|k| k.get("slots_free"))
                        .and_then(Value::as_u64);
                    let host = v
                        .get("kv")
                        .and_then(|k| k.get("host_blocks"))
                        .and_then(Value::as_u64);
                    if let Ok(mut st) = ms_clone.lock() {
                        if let Some(f) = free {
                            st.free_min = Some(st.free_min.map_or(f, |x| x.min(f)));
                        }
                        if let Some(h) = host {
                            st.host_max = st.host_max.max(h);
                        }
                    }
                }
            }
            tick = tick.wrapping_add(1);
            thread::sleep(Duration::from_millis(50));
        }
    });

    let first_stream = spawn_stream(args.url.clone(), args.model.clone(), first_req);
    if first_is_decode {
        // 等 decode 侧真的开始出 token（= 已在 decode），再让 prefill 进来。
        first_stream
            .first_rx
            .recv_timeout(Duration::from_secs(180))
            .map_err(|_| "decode 侧 180s 内没有第一个 token".to_string())?;
    } else {
        // 反向：等 prefill 侧被准入（active>=1），再让 decode 进来。
        let until = Instant::now() + Duration::from_secs(30);
        while Instant::now() < until {
            if mon_state.lock().map(|s| s.max_active >= 1).unwrap_or(false) {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
    }

    second_started.store(true, Ordering::Relaxed);
    let second_client = Client::new(&args.url, args.model.clone());
    let second_times: Arc<Mutex<Vec<Instant>>> = Arc::new(Mutex::new(Vec::new()));
    let st = Arc::clone(&second_times);
    let second_started = Instant::now();
    let second_reply = second_client
        .chat_stream(second_req, |_| {
            if let Ok(mut v) = st.lock() {
                v.push(Instant::now());
            }
        })
        .map_err(|e| format!("第二路 chat 失败: {e}"))?;
    let second_ended = Instant::now();
    let second_deltas = second_times.lock().map(|v| v.clone()).unwrap_or_default();

    let first = first_stream.into_streamed()?;
    stop.store(true, Ordering::Relaxed);
    let _ = monitor.join();
    let mon = mon_state.lock().map(|s| s.clone()).unwrap_or_default();

    let second = Streamed {
        reply: second_reply,
        started: second_started,
        ended: second_ended,
        deltas: second_deltas,
    };
    let (decode, prefill) = if first_is_decode {
        (&first, &second)
    } else {
        (&second, &first)
    };

    // 并发窗口 = **prefill 侧的请求墙钟**；解码侧在这个窗口里的 delta = overlap。
    let (w0, w1) = (prefill.started, prefill.ended);
    let overlap = decode
        .deltas
        .iter()
        .filter(|t| **t >= w0 && **t <= w1)
        .count();
    let max_gap = max_gap(&decode.deltas, w0, w1);

    // 并发判定：第二路发出后**没有**持续排队 = 它被准入并参与调度；
    // `active>=2` 采样是补充证据（该字段在部分交错形态下会低报，实测）。
    let concurrent = mon.max_queued_streak < 3;
    let dec_turn = TurnRecord {
        scenario: case.name.to_string(),
        turn: 1,
        finish: decode.reply.finish_reason.clone(),
        prompt: decode.reply.usage.prompt_tokens,
        images: 0,
        cached: decode.reply.usage.cached_tokens,
        completion: decode.reply.usage.completion_tokens,
        reasoning: decode.reply.usage.reasoning_tokens,
        absorbed: 0,
        rotate_bytes: 0,
        rebase: 0,
        elapsed_ms: decode.reply.elapsed_ms,
        tools: Vec::new(),
        error: None,
    };
    let pre_turn = TurnRecord {
        scenario: case.name.to_string(),
        turn: 2,
        finish: prefill.reply.finish_reason.clone(),
        prompt: prefill.reply.usage.prompt_tokens,
        images: 0,
        cached: prefill.reply.usage.cached_tokens,
        completion: prefill.reply.usage.completion_tokens,
        reasoning: prefill.reply.usage.reasoning_tokens,
        absorbed: 0,
        rotate_bytes: 0,
        rebase: 0,
        elapsed_ms: prefill.reply.elapsed_ms,
        tools: Vec::new(),
        error: None,
    };

    let mut fails = Vec::new();
    let dec_out = decode.reply.usage.completion_tokens + decode.reply.usage.reasoning_tokens;
    let want_dec = (case.decode_max / 2).min(128) as u64;
    if dec_out < want_dec {
        fails.push(format!(
            "解码侧只出了 {dec_out} token（< {want_dec}；finish={}）",
            decode.reply.finish_reason
        ));
    }
    if prefill.reply.usage.prompt_tokens < (case.prefill_tokens as u64 * 8 / 10) {
        fails.push(format!(
            "预填侧 prompt_tokens={} 小于期望 {} 的 80%",
            prefill.reply.usage.prompt_tokens, case.prefill_tokens
        ));
    }
    if prefill.reply.usage.completion_tokens == 0 {
        fails.push("预填侧没有产出 token".into());
    }
    if concurrent {
        if overlap == 0 {
            fails.push("并发窗口内解码侧没有继续出 token".into());
        }
        // 3s ≈ 3 个 prefill chunk（2048 tok / ~1.5k t/s ≈ 1.3s）。
        // `slice_every=0` 时解码会被整段 prefill 冻结（实测 3.8s）⇒ 红。
        if max_gap > 3_000 {
            fails.push(format!(
                "并发窗口内解码侧最大 token 间隔 {max_gap}ms > 3000ms（疑似被 prefill 冻结）"
            ));
        }
    } else if case.decode_first && overlap == 0 {
        fails.push("排队档但解码侧在等待窗口内没有出 token".into());
    }

    let role = if case.decode_first {
        "decode→prefill"
    } else {
        "prefill→decode"
    };
    let detail = format!(
        "{role} · mode={mode} · overlap={overlap} tok · max_gap={max_gap}ms · \
         active_max={amax} queue_max={qmax} queued_after_second={qas} max_streak={qst} active2_samples={a2} · free_min={free:?} · \
         host_max={host} · 解码 {dec:.2}s / 预填 {pre:.2}s",
        mode = if concurrent {
            "concurrent"
        } else {
            "queued(max_batch=1)"
        },
        amax = mon.max_active,
        qmax = mon.max_queue,
        qas = mon.queued_after_second,
        qst = mon.max_queued_streak,
        a2 = mon.active2_samples,
        free = mon.free_min,
        host = mon.host_max,
        dec = decode.reply.elapsed_ms as f64 / 1000.0,
        pre = prefill.reply.elapsed_ms as f64 / 1000.0,
    );
    Ok(ScenarioReport {
        name: case.name.to_string(),
        pass: fails.is_empty(),
        detail: if fails.is_empty() {
            detail
        } else {
            format!("{detail} · 失败: {}", fails.join("；"))
        },
        turns: vec![dec_turn, pre_turn],
    })
}

/// 窗口内相邻 delta 的最大间隔（只看 delta 之间；窗口首尾空白不算）。
fn max_gap(times: &[Instant], t0: Instant, t1: Instant) -> u128 {
    let mut pts: Vec<Instant> = times
        .iter()
        .copied()
        .filter(|t| *t >= t0 && *t <= t1)
        .collect();
    pts.sort();
    pts.windows(2)
        .map(|w| w[1].duration_since(w[0]).as_millis())
        .max()
        .unwrap_or(0)
}

fn make_prefill(tokens: usize, tag: &str) -> String {
    // 随机短词 ≈ 1.4 tok/词（AGENTS）。
    let words = (tokens as f64 / 1.4).round() as usize;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (tokens as u64);
    let mut pick = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        WORDS[(seed as usize) % WORDS.len()]
    };
    let mut text = String::with_capacity(words * 7 + 64);
    text.push_str(&format!(
        "[run {tag}] read the following archive and answer at the end.\n"
    ));
    for _ in 0..words {
        text.push_str(pick());
        text.push(' ');
    }
    text.push_str("\nQuestion: reply with exactly one word — the first word of the archive.");
    text
}
