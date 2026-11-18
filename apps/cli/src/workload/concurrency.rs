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
//!   无选择层 + spec off；相位时间片**引擎自决**（`sched_policy::SLICE_EVERY`
//!   = 8 ⇒ 每 8 拍 decode 插 1 片 prefill，窗口内 max_gap ≈ 0.9s；没有配置键）。
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

use serde_json::{json, Value};

use crate::client::{ChatReply, ChatRequest, Client};
use crate::workload::report::{ScenarioReport, TurnRecord};
use crate::workload::Args;

/// 随机词池（prompt 不许重复文本；AGENTS 测量纪律）。
const WORDS: &[&str] = &[
    "amber", "basin", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "iris", "jasper",
    "kelp", "lumen", "meadow", "nimbus", "onyx", "prism", "quartz", "raven", "sable", "tundra",
    "umber", "vapor", "willow", "xenon", "yarrow", "zephyr", "anchor", "beacon", "cinder", "drift",
    "echo", "fable", "glacier", "hollow", "indigo", "juniper", "kestrel", "lantern", "marble",
    "nectar", "orbit", "pebble", "quiver", "ripple", "summit", "thistle", "upland", "vertex",
    "whisper", "yonder", "zinnia", "arbor", "bramble", "cedar", "dune", "fern", "grove", "heather",
    "inlet", "jetty", "knoll", "lagoon", "moss", "nook", "orchard", "pine", "quill", "reed",
    "stone", "trail", "vale", "wren", "yew", "zenith", "alder", "birch", "cliff", "dew", "elm",
    "frost", "glen", "haze", "isle", "kite", "leaf", "mist", "nest", "oak", "path", "quay", "root",
    "sage", "tide", "vine", "wave", "yard", "zinc", "acorn", "bloom", "cloud", "creek", "dawn",
    "dusk", "field", "grain", "hill",
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
            serde_json::to_string_pretty(&arr)
                .unwrap_or_else(|e| format!("[{{\"error\":\"{e}\"}}]"))
        );
    }
    let failed = reports.iter().filter(|r| !r.pass).count();
    eprintln!(
        "[decode-prefill] {}/{} scenarios passed",
        reports.len() - failed,
        reports.len()
    );
    if failed == 0 {
        0
    } else {
        1
    }
}

/// 等引擎静默：`active == 0 && queue_len == 0` 连续 3 拍（≈300ms）。
///
/// 为什么不是"读完上一用例的 Completed 就够"：终局清理在事件之后还有一拍
/// （块表归还/快照重挂），此时共享前缀仍被钉着，整链下沉会全部失效。
fn wait_quiescent(c: &Client, cap: Duration) -> Result<(), String> {
    let until = Instant::now() + cap;
    let mut quiet_since: Option<Instant> = None;
    while Instant::now() < until {
        let quiet = c
            .get_json("/debug/queue")
            .ok()
            .map(|q| {
                q.get("active").and_then(Value::as_u64).unwrap_or(1) == 0
                    && q.get("queue_len").and_then(Value::as_u64).unwrap_or(1) == 0
            })
            .unwrap_or(false);
        if quiet {
            let t0 = *quiet_since.get_or_insert_with(Instant::now);
            if t0.elapsed() >= Duration::from_millis(300) {
                return Ok(());
            }
        } else {
            quiet_since = None;
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err(format!(
        "引擎 {}s 内未静默（上一用例的 turn 未收摊？）",
        cap.as_secs()
    ))
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
    BroadcastStream {
        handle,
        first_rx,
        times,
    }
}

struct BroadcastStream {
    handle: thread::JoinHandle<(std::io::Result<ChatReply>, Instant, Instant)>,
    first_rx: std::sync::mpsc::Receiver<()>,
    times: Arc<Mutex<Vec<Instant>>>,
}

impl BroadcastStream {
    fn into_streamed(self) -> Result<Streamed, String> {
        // ⚠️ **先 join 再取时间戳**：反过来的话，线程在 clone 之后、退出之前
        // 产生的 delta 会丢（尾段 ITL/gap 统计被截断，窗口末尾假绿）。
        let (r, started, ended) = self
            .handle
            .join()
            .map_err(|_| "流式线程 panic".to_string())?;
        let deltas = self.times.lock().map(|v| v.clone()).unwrap_or_default();
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
    // **用例之间必须回到静默**（2026-11-08 第 7 轮误红）：
    // 上一用例的收尾 turn 只要还在 active，它就钉着共享前缀（refs=2）
    // ⇒ 本用例起点的整链下沉全被判"不可沉"，而 host 又刚好满 ⇒ 起点腾位
    // 只能逐块丢，长的 24k turn 必红。等 active/queue 归零（连续 3 拍）再起。
    {
        let c = Client::new(&args.url, args.model.clone());
        wait_quiescent(&c, Duration::from_secs(60))?;
    }
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

    // 监控分**两条独立线程**：
    // · queue 高频（50ms）且**绝不阻塞** —— 它是 active/queue 的唯一证词；
    // · vram 低频（1s）—— `/debug/vram` 走 executor，忙时阻塞数秒。
    // 两者同线程的后果实测过：prefill 窗口内 queue 采样被 vram 整段搭死
    // ⇒ `active_max` 永远停在 1（误报“遥测低报”）。
    let stop = Arc::new(AtomicBool::new(false));
    let second_started = Arc::new(AtomicBool::new(false));
    let mon_state = Arc::new(Mutex::new(Monitor::default()));
    // 环境变量只读一次（原来在 20Hz 采样循环里每拍查一次）。
    let dp_debug = std::env::var_os("OWL_DP_DEBUG").is_some();

    let monitor = {
        let stop = Arc::clone(&stop);
        let second_flag = Arc::clone(&second_started);
        let ms = Arc::clone(&mon_state);
        let url = args.url.clone();
        let model = args.model.clone();
        thread::spawn(move || {
            let client = Client::new(&url, model);
            while !stop.load(Ordering::Relaxed) {
                if let Ok(q) = client.get_json("/debug/queue") {
                    let active = q.get("active").and_then(Value::as_u64).unwrap_or(0);
                    let queued = q.get("queue_len").and_then(Value::as_u64).unwrap_or(0);
                    if dp_debug {
                        eprintln!("[dp-mon] active={active} queue={queued}");
                    }
                    if let Ok(mut st) = ms.lock() {
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
                thread::sleep(Duration::from_millis(50));
            }
        })
    };

    let vram_mon = {
        let stop = Arc::clone(&stop);
        let ms = Arc::clone(&mon_state);
        let url = args.url.clone();
        let model = args.model.clone();
        thread::spawn(move || {
            let client = Client::new(&url, model);
            while !stop.load(Ordering::Relaxed) {
                if let Ok(v) = client.get_json("/debug/vram") {
                    let free = v
                        .get("kv")
                        .and_then(|k| k.get("slots_free"))
                        .and_then(Value::as_u64);
                    let host = v
                        .get("kv")
                        .and_then(|k| k.get("host_blocks"))
                        .and_then(Value::as_u64);
                    if let Ok(mut st) = ms.lock() {
                        if let Some(f) = free {
                            st.free_min = Some(st.free_min.map_or(f, |x| x.min(f)));
                        }
                        if let Some(h) = host {
                            st.host_max = st.host_max.max(h);
                        }
                    }
                }
                thread::sleep(Duration::from_millis(1000));
            }
        })
    };
    // 早退（首 token 超时 / 第二路失败 / panic）也要停掉监控线程。
    let _mon_guard = StopOnDrop(Arc::clone(&stop));

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
    let _ = vram_mon.join();
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

    // 并发判定：第二路**排队等了多久**。
    //
    // `max_batch=1` 的队列档会一直排到第一路结束（秒级）；真并发档只有
    // "submit → 下一次调度" 的准入延迟（实测 0–0.5s，prefill 正忙时最长 ~0.45s）。
    // 阈值 1s 把两者分开；原先的 `streak < 3`（≈150ms）会把正常的准入延迟
    // 误判成队列档（已实测：450ms 的等待 + overlap 112 tok 被判 queued）。
    let queue_wait_ms = mon.max_queued_streak as u128 * 50;
    let concurrent = queue_wait_ms < 1_000;
    // `active>=2` 是正向证据；若判并发却从没到 2，**先查采样线程**
    // （`/debug/vram` 阻塞会把 50ms 采样整段搭死），再怀疑服务端字段。
    let tele_suspect = concurrent && mon.max_active < 2;
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
        // 相位时间片现由引擎自决（`SLICE_EVERY`=8）—— 旧的不切片行为
        // （解码被整段 prefill 冻结，实测 3.8s）已不再可达 ⇒ 本门红。
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
         active_max={amax} queue_max={qmax} qwait={qwait}ms queued_after_second={qas} max_streak={qst} active2_samples={a2} · free_min={free:?} · \
         host_max={host} · 解码 {dec:.2}s / 预填 {pre:.2}s{tele}",
        mode = if concurrent {
            "concurrent"
        } else {
            "queued(max_batch=1)"
        },
        amax = mon.max_active,
        qmax = mon.max_queue,
        qwait = queue_wait_ms,
        qas = mon.queued_after_second,
        qst = mon.max_queued_streak,
        a2 = mon.active2_samples,
        free = mon.free_min,
        host = mon.host_max,
        dec = decode.reply.elapsed_ms as f64 / 1000.0,
        pre = prefill.reply.elapsed_ms as f64 / 1000.0,
        tele = if tele_suspect {
            " · ⚠ 判并发但 active_max<2（/debug/queue 低报？）"
        } else {
            ""
        },
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

/// 监控线程存活门：任何早退路径（`?`/panic）都要把 `stop` 置位，
/// 否则 20Hz 轮询线程会跟着进程泄漏到下一个用例。
struct StopOnDrop(Arc<AtomicBool>);

impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// 窗口内相邻 delta 的最大间隔。
///
/// ⚠️ 必须带上**窗口前最后一个 delta** 作锚点：冻结往往就发生在 prefill 开始的
/// 那一瞬间，若窗口里只有一个点（`windows(2)` 为空）会算出 0 ⇒ 假绿。
fn max_gap(times: &[Instant], t0: Instant, t1: Instant) -> u128 {
    let mut pts: Vec<Instant> = times
        .iter()
        .copied()
        .filter(|t| *t >= t0 && *t <= t1)
        .collect();
    if let Some(prev) = times.iter().copied().filter(|t| *t < t0).max() {
        // 只把"紧邻窗口"的点当锚点（≤1s）：更早的发呆属于窗口前抖动，
        // 算进来会把既有停顿误判成 prefill 冻结。
        if t0.duration_since(prev) <= Duration::from_millis(1000) {
            pts.push(prev);
        }
    }
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
