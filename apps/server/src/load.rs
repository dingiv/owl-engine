//! **`POST /debug/load` —— 内源灌流**（服务面并发能力的现场测量）。
//!
//! 一次调用 = 同时向引擎灌 N 个 turn，走**与真实请求同一条** `EngineReq::Chat`
//! 通道（同一 actor / 同一调度 / 同一引擎），每 worker：
//! - **独立会话**（`session` 键带 nonce ⇒ 会话闸不互相阻塞，且能进同一个批）；
//! - **固定生成长度**（`ignore_eos`）+ 贪婪采样 ⇒ 读数不被"模型想不想停"干扰；
//! - **随机前缀**（默认开）⇒ 打掉前缀缓存，测的是真并发算力而不是缓存命中；
//! - 统计 TTFT / 生成 token 数 / 墙钟 / 引擎侧 `batch_steps` 差值。
//!
//! 为什么不去"对自己发 HTTP"：灌流请求自己也要占 `server.max_concurrency`
//! 的 permit（N 会被压到 `max_concurrency-1`，还可能自锁）。本驱动直接进
//! actor 通道，量的是同一段引擎路径，且**不占** permit —— 但 `concurrency`
//! 仍按 `max_concurrency` 校验（它就是服务面的并发上限，超了如实拒）。
//!
//! **判据**（输出里全给）：
//! - `batch.steps_delta > 0` 且 `max_batch > 1` ⇒ 真的合批；
//!   `steps_delta == 0` 而并发 >1 ⇒ "并发进引擎但批门未成立"（相位/采样档/
//!   GDN 格/单卡门任一不满足 —— 见 `docs/arch/跨会话合批-decode-调研与设计.md`）；
//! - `aggregate.decode_tps` 对 `concurrency=1` 的比值 = 并发加速比；
//! - `requests[].ttft_ms` 分布 = 排队/批内公平性（后到者的 TTFT 里含排队）。

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::AsyncWrite;
use tokio::sync::mpsc;

use owl_engine::{TurnEvent, TurnOpts, TurnSampler};
use owl_shared::config::AgentCfg;

use crate::actor::{BatchStatsSnapshot, EngineReq, SinkEvent};
use crate::chat_tpl::{ChatTpl, RenderArgs};
use crate::http;

/// 事件通道容量（灌流单 worker 的事件速率 ≪ 正常路径；256 足够）
const LOAD_SINK_CAP: usize = 256;
/// 随机词池（短词；tokenizer 实测 ≈1.4 tok/词 —— AGENTS.md 口径）
const WORDS: [&str; 24] = [
    "alpha", "bravo", "cedar", "delta", "ember", "falcon", "garden", "harbor", "indigo", "jasper",
    "kettle", "lunar", "meadow", "nectar", "orbit", "pepper", "quartz", "ripple", "saffron",
    "timber", "umber", "violet", "willow", "zephyr",
];

/// 请求体（全部可缺省；越界**响亮拒**，不夹紧）
#[derive(Debug, Deserialize)]
pub struct LoadSpec {
    /// 并发 turn 数（1..=server.max_concurrency；缺省 4）
    #[serde(default)]
    pub concurrency: Option<usize>,
    /// 每个 turn **固定生成** token 数（ignore_eos；缺省 32）
    #[serde(default)]
    pub tokens: Option<usize>,
    /// 目标 prompt 规模（token；缺省 256）
    #[serde(default)]
    pub prompt_tokens: Option<usize>,
    /// 随机前缀（打掉前缀缓存；缺省 true）
    #[serde(default)]
    pub random_prefix: Option<bool>,
    /// 整场超时（毫秒；缺省 120000）
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// 会话键前缀（缺省 `load-<nonce>`；同一前缀 = 跨 call 复用会话，
    /// **只用于对照**，正常测量请留空 —— 复用会命中前缀缓存）
    #[serde(default)]
    pub session_prefix: Option<String>,
    /// 模式：`uniform`（缺省，N 个同形请求）/ `pair`（A 先起、`delay_ms` 后 B 进来）
    #[serde(default)]
    pub mode: Option<String>,
    /// `pair`：B 的延迟（毫秒）
    #[serde(default)]
    pub delay_ms: Option<u64>,
    /// `pair`：A 的 prompt / 生成（缺省 = `prompt_tokens` / `tokens`）
    #[serde(default)]
    pub a_prompt_tokens: Option<usize>,
    #[serde(default)]
    pub a_tokens: Option<usize>,
    /// `pair`：B 的 prompt / 生成（B 的 prompt 大 = 制造 prefill 挤占）
    #[serde(default)]
    pub b_prompt_tokens: Option<usize>,
    #[serde(default)]
    pub b_tokens: Option<usize>,
}

/// 归一化后的规格（校验后；纯值）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormSpec {
    pub concurrency: usize,
    pub tokens: usize,
    pub prompt_tokens: usize,
    pub random_prefix: bool,
    pub timeout_ms: u64,
    pub session_prefix: String,
    pub nonce: u64,
    pub pair: bool,
    pub delay_ms: u64,
    pub a_prompt_tokens: usize,
    pub a_tokens: usize,
    pub b_prompt_tokens: usize,
    pub b_tokens: usize,
}

impl LoadSpec {
    /// 校验 + 归一（`max_concurrency` / `max_new_cap` 由调用方给 —— 单源在配置）
    pub fn normalize(
        self,
        max_concurrency: usize,
        max_new_cap: usize,
        nonce: u64,
    ) -> Result<NormSpec, String> {
        let concurrency = self.concurrency.unwrap_or(4);
        if concurrency == 0 {
            return Err("concurrency = 0 —— 至少 1".into());
        }
        if concurrency > max_concurrency {
            return Err(format!(
                "concurrency({concurrency}) > server.max_concurrency({max_concurrency})—— \
                 服务面并发上限就是它（要更大改配置，不是本接口夹紧）"
            ));
        }
        let tokens = self.tokens.unwrap_or(32);
        if tokens == 0 || tokens > max_new_cap {
            return Err(format!(
                "tokens({tokens}) 越界 1..={max_new_cap}([agent] max_new_cap / 池预算)"
            ));
        }
        let prompt_tokens = self.prompt_tokens.unwrap_or(256);
        if prompt_tokens == 0 || prompt_tokens > 131_072 {
            return Err(format!("prompt_tokens({prompt_tokens}) 越界 1..=131072"));
        }
        let timeout_ms = self.timeout_ms.unwrap_or(120_000);
        if timeout_ms == 0 {
            return Err("timeout_ms = 0 —— 至少 1".into());
        }
        Ok(NormSpec {
            concurrency,
            tokens,
            prompt_tokens,
            random_prefix: self.random_prefix.unwrap_or(true),
            timeout_ms,
            session_prefix: self
                .session_prefix
                .unwrap_or_else(|| format!("load-{nonce:016x}")),
            nonce,
            pair: self.mode.as_deref() == Some("pair"),
            delay_ms: self.delay_ms.unwrap_or(0),
            a_prompt_tokens: self.a_prompt_tokens.unwrap_or(prompt_tokens),
            a_tokens: self.a_tokens.unwrap_or(tokens),
            b_prompt_tokens: self.b_prompt_tokens.unwrap_or(prompt_tokens),
            b_tokens: self.b_tokens.unwrap_or(tokens),
        })
    }
}

/// 词汤长度（词数 = ceil(tokens / 1.4)；至少 1 词）
pub fn word_count_for(tokens: usize) -> usize {
    (tokens * 10 / 14).max(1)
}

/// 词汤生成（LCG；`seed` 决定内容 —— 同 seed 同文本，便于对拍）
pub fn word_soup(seed: u64, words: usize) -> String {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut out = String::with_capacity(words * 6);
    for i in 0..words {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let w = WORDS[(x >> 33) as usize % WORDS.len()];
        if i > 0 {
            out.push(' ');
        }
        out.push_str(w);
    }
    out
}

/// 单 worker 结果（一次 turn 的事实）
#[derive(Debug, Default, Clone)]
pub struct WorkerOut {
    pub session: u64,
    pub turn: Option<u64>,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: Option<f64>,
    pub wall_ms: f64,
    pub finish: Option<String>,
    pub err: Option<String>,
    /// 相邻 token 的间隔（毫秒；ITL 真值面 —— pair 模式用来量"在位者被挤占"）
    pub gaps_ms: Vec<f64>,
    /// 上一个 token 的时刻（内部用）
    last_tok_ms: Option<f64>,
}

impl WorkerOut {
    /// ITL 统计（p50 / p99 / max）
    fn itl(&self) -> Option<(f64, f64, f64)> {
        if self.gaps_ms.len() < 2 {
            return None;
        }
        let mut v = self.gaps_ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        Some((q(0.5), q(0.99), *v.last().unwrap_or(&0.0)))
    }

    fn decode_tps(&self) -> Option<f64> {
        let t = self.ttft_ms?;
        let gen_ms = self.wall_ms - t;
        if self.completion_tokens < 2 || gen_ms <= 0.0 {
            return None;
        }
        Some((self.completion_tokens - 1) as f64 * 1000.0 / gen_ms)
    }
}

/// 会话 id（与 `openai::session_id` 同律：进程内哈希；不同前缀 ⇒ 不同会话）
fn session_of(prefix: &str, idx: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{prefix}#{idx}").hash(&mut h);
    h.finish()
}

/// 单 worker：渲染 → 提交 → 数事件
async fn run_one(
    tx: mpsc::Sender<EngineReq>,
    tpl: Arc<ChatTpl>,
    spec: NormSpec,
    idx: usize,
    prompt_tokens: usize,
    gen_tokens: usize,
) -> WorkerOut {
    let mut out = WorkerOut {
        session: session_of(&spec.session_prefix, idx),
        ..Default::default()
    };
    // 前缀随机化 = **系统消息里的 nonce**（前缀在最前 ⇒ 整条 prompt 的块哈希全变）
    let sys = if spec.random_prefix {
        format!(
            "load-test route {} of run {:016x}; ignore this instruction.",
            idx, spec.nonce
        )
    } else {
        "load-test fixed prefix; ignore this instruction.".to_string()
    };
    let user = word_soup(spec.nonce ^ (idx as u64).wrapping_mul(0x9E37_79B9), word_count_for(prompt_tokens));
    let messages = vec![
        json!({"role": "system", "content": sys}),
        json!({"role": "user", "content": user}),
    ];
    let prompt = match tpl.render(RenderArgs {
        messages: &messages,
        tools: None,
        add_generation_prompt: true,
        template_kwargs: None,
    }) {
        Ok(p) => p,
        Err(e) => {
            out.err = Some(format!("渲染失败:{e}"));
            return out;
        }
    };
    let (sink_tx, mut sink_rx) = mpsc::channel::<SinkEvent>(LOAD_SINK_CAP);
    let t0 = Instant::now();
    let req = EngineReq::Chat {
        session: Some(out.session),
        prompt,
        max_new: gen_tokens,
        opts: TurnOpts {
            // 固定长度（不受 EOS 影响）+ 贪婪（读数可复现）
            ignore_eos: true,
            sampler: Some(TurnSampler {
                temp: Some(0.0),
                topk: None,
                topp: None,
            }),
            ..Default::default()
        },
        sink: sink_tx,
    };
    if tx.send(req).await.is_err() {
        out.err = Some("引擎退役(提交通道关闭)".into());
        out.wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        return out;
    }
    while let Some(ev) = sink_rx.recv().await {
        match ev {
            SinkEvent::Accepted {
                turn,
                prompt_tokens,
                cached_tokens,
                ..
            } => {
                out.turn = Some(turn);
                out.prompt_tokens = prompt_tokens;
                out.cached_tokens = cached_tokens;
            }
            SinkEvent::Session { cached_tokens, .. } => {
                out.cached_tokens = cached_tokens;
            }
            SinkEvent::Rejected { err } => {
                out.err = Some(format!("提交被拒:{err}"));
                break;
            }
            SinkEvent::Turn(TurnEvent::Token { .. }) => {
                let now = t0.elapsed().as_secs_f64() * 1e3;
                match out.ttft_ms {
                    None => out.ttft_ms = Some(now),
                    Some(_) => {
                        if let Some(prev) = out.last_tok_ms {
                            out.gaps_ms.push(now - prev);
                        }
                    }
                }
                out.last_tok_ms = Some(now);
                // 契约:1 事件 = 1 采样步 = 1 token（`usage` 同源口径）
                out.completion_tokens += 1;
            }
            SinkEvent::Turn(TurnEvent::Prefill { total, .. }) => {
                out.prompt_tokens = total;
            }
            SinkEvent::Turn(TurnEvent::Completed { finish, .. }) => {
                out.finish = Some(format!("{finish:?}"));
                break;
            }
            SinkEvent::Turn(TurnEvent::Failed { err, .. }) => {
                out.err = Some(err);
                break;
            }
            // Idle 不该出现在单 turn 的 sink 上（出现 = 引擎退役，如实记）
            SinkEvent::Turn(TurnEvent::Idle) => {
                out.err = Some("引擎 Idle(提交丢失)".into());
                break;
            }
        }
    }
    out.wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    out
}

/// 取一次引擎事实（前后各一次，算 `batch_steps` 差值）
async fn stats(tx: &mpsc::Sender<EngineReq>) -> Result<BatchStatsSnapshot, String> {
    let (reply, rx) = tokio::sync::oneshot::channel();
    tx.send(EngineReq::BatchStats { reply })
        .await
        .map_err(|_| "引擎退役(BatchStats 通道关闭)".to_string())?;
    rx.await.map_err(|_| "引擎退役(BatchStats 应答丢失)".to_string())
}

/// HTTP 入口
pub(crate) async fn load<W: AsyncWrite + Unpin>(
    w: &mut W,
    body: &[u8],
    tx: mpsc::Sender<EngineReq>,
    tpl: &Arc<ChatTpl>,
    agent: &AgentCfg,
    max_concurrency: usize,
) -> std::io::Result<()> {
    let raw: LoadSpec = match serde_json::from_slice(body) {
        Ok(s) => s,
        Err(e) => {
            return http::respond_json(w, 400, &json!({"error": format!("请求体非法: {e}")})).await
        }
    };
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);
    let spec = match raw.normalize(max_concurrency, agent.max_new_cap, nonce) {
        Ok(s) => s,
        Err(e) => return http::respond_json(w, 400, &json!({"error": e})).await,
    };
    let before = match stats(&tx).await {
        Ok(s) => s,
        Err(e) => return http::respond_json(w, 503, &json!({"error": e})).await,
    };
    // 同时起 N 个 worker（submit 全部先进 actor 通道；actor 在泵前一次排空）
    let mut handles = Vec::with_capacity(spec.concurrency);
    if spec.pair {
        assert!(spec.concurrency == 2, "pair 模式要求 concurrency = 2");
        let sp_a = spec.clone();
        let tx_a = tx.clone();
        let tpl_a = tpl.clone();
        handles.push(tokio::spawn(async move {
            run_one(tx_a, tpl_a, sp_a.clone(), 0, sp_a.a_prompt_tokens, sp_a.a_tokens).await
        }));
        if spec.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(spec.delay_ms)).await;
        }
        let sp_b = spec.clone();
        let tx_b = tx.clone();
        let tpl_b = tpl.clone();
        handles.push(tokio::spawn(async move {
            run_one(tx_b, tpl_b, sp_b.clone(), 1, sp_b.b_prompt_tokens, sp_b.b_tokens).await
        }));
    } else {
        for i in 0..spec.concurrency {
            let tx = tx.clone();
            let tpl = tpl.clone();
            let sp = spec.clone();
            handles.push(tokio::spawn(async move {
                run_one(tx, tpl, sp.clone(), i, sp.prompt_tokens, sp.tokens).await
            }));
        }
    }
    let deadline = Duration::from_millis(spec.timeout_ms);
    let joined = tokio::time::timeout(deadline, futures_join(handles)).await;
    let (outs, timed_out) = match joined {
        Ok(v) => (v, false),
        Err(_) => (Vec::new(), true),
    };
    let after = stats(&tx).await.ok();
    let body = report_json(&spec, before, after, outs, timed_out);
    // 超时 = 客户端该知道（503：这次测量无效，不是服务不可用）
    let code = if timed_out { 503 } else { 200 };
    http::respond_json(w, code, &body).await
}

/// 顺序 join（不引 futures 依赖：逐个 await —— worker 已在跑，join 顺序不影响并发）
async fn futures_join(
    handles: Vec<tokio::task::JoinHandle<WorkerOut>>,
) -> Vec<WorkerOut> {
    let mut v = Vec::with_capacity(handles.len());
    for h in handles {
        match h.await {
            Ok(o) => v.push(o),
            Err(e) => v.push(WorkerOut {
                err: Some(format!("worker panic/取消: {e}")),
                ..Default::default()
            }),
        }
    }
    v
}

/// 汇总 JSON（纯函数；单测可跑）
pub fn report_json(
    spec: &NormSpec,
    before: BatchStatsSnapshot,
    after: Option<BatchStatsSnapshot>,
    outs: Vec<WorkerOut>,
    timed_out: bool,
) -> Value {
    let wall_ms = outs.iter().map(|o| o.wall_ms).fold(0.0_f64, f64::max);
    let sum_tokens: usize = outs.iter().map(|o| o.completion_tokens).sum();
    let sum_prompt: usize = outs.iter().map(|o| o.prompt_tokens).sum();
    let sum_cached: usize = outs.iter().map(|o| o.cached_tokens).sum();
    let ok = outs.iter().filter(|o| o.err.is_none()).count();
    let decode_tps = if wall_ms > 0.0 {
        Some(sum_tokens as f64 * 1000.0 / wall_ms)
    } else {
        None
    };
    let per_req: Vec<Value> = outs
        .iter()
        .map(|o| {
            json!({
                "session": format!("{:#x}", o.session),
                "turn": o.turn,
                "prompt_tokens": o.prompt_tokens,
                "cached_tokens": o.cached_tokens,
                "completion_tokens": o.completion_tokens,
                "ttft_ms": o.ttft_ms,
                "wall_ms": o.wall_ms,
                "decode_tps": o.decode_tps(),
                "itl_p50_ms": o.itl().map(|(a, _, _)| a),
                "itl_p99_ms": o.itl().map(|(_, b, _)| b),
                "itl_max_ms": o.itl().map(|(_, _, c)| c),
                "finish": o.finish,
                "err": o.err,
            })
        })
        .collect();
    let (steps_after, steps_delta, active_after) = match after {
        Some(a) => (
            Some(a.batch_steps),
            Some(a.batch_steps.saturating_sub(before.batch_steps)),
            Some(a.active),
        ),
        None => (None, None, None),
    };
    json!({
        "ok": !timed_out && ok == outs.len() && !outs.is_empty(),
        "timed_out": timed_out,
        "spec": {
            "concurrency": spec.concurrency,
            "tokens": spec.tokens,
            "prompt_tokens": spec.prompt_tokens,
            "random_prefix": spec.random_prefix,
            "session_prefix": spec.session_prefix,
            "pair": spec.pair,
            "delay_ms": spec.delay_ms,
        },
        "wall_ms": wall_ms,
        "requests": per_req,
        "aggregate": {
            "completed": ok,
            "prompt_tokens": sum_prompt,
            "cached_tokens": sum_cached,
            "completion_tokens": sum_tokens,
            // 口径：首 token 到末 token 的聚合吞吐（含排队/TTFT ⇒ 判并发看加速比）
            "decode_tps": decode_tps,
            "per_req_tps": decode_tps.map(|t| t / spec.concurrency as f64),
        },
        "batch": {
            "max_batch": before.max_batch,
            "graph": before.has_batch_graph,
            "steps_before": before.batch_steps,
            "steps_after": steps_after,
            "steps_delta": steps_delta,
            // 判据：并发 >1 但 steps 不涨 = 并发进了引擎、批门没成立
            "batching": steps_delta.map(|d| d > 0),
            "queue_before": before.queue_len,
            "active_before": before.active,
            "active_after": active_after,
        },
        "server": { "max_concurrency": spec.concurrency.max(1) },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(steps: u64) -> BatchStatsSnapshot {
        BatchStatsSnapshot {
            max_batch: 4,
            batch_steps: steps,
            has_batch_graph: true,
            queue_len: 0,
            active: 0,
            queued: Vec::new(),
            rejects: Vec::new(),
            wait_hist: [0; 5],
        }
    }

    #[test]
    fn spec_rejects_out_of_range() {
        let mk = |c: Option<usize>, t: Option<usize>, p: Option<usize>| LoadSpec {
            concurrency: c,
            tokens: t,
            prompt_tokens: p,
            random_prefix: None,
            timeout_ms: None,
            session_prefix: None,
            mode: None,
            delay_ms: None,
            a_prompt_tokens: None,
            a_tokens: None,
            b_prompt_tokens: None,
            b_tokens: None,
        };
        assert!(mk(Some(0), None, None).normalize(8, 512, 1).is_err());
        assert!(mk(Some(9), None, None).normalize(8, 512, 1).is_err(), "超并发上限必须拒");
        assert!(mk(None, Some(0), None).normalize(8, 512, 1).is_err());
        assert!(mk(None, Some(513), None).normalize(8, 512, 1).is_err());
        assert!(mk(None, None, Some(0)).normalize(8, 512, 1).is_err());
        let n = mk(Some(8), Some(64), Some(1024)).normalize(8, 512, 7).expect("合法档");
        assert_eq!((n.concurrency, n.tokens, n.prompt_tokens), (8, 64, 1024));
        assert!(n.random_prefix, "缺省随机前缀");
        assert_eq!(n.session_prefix, "load-0000000000000007");
    }

    #[test]
    fn word_soup_is_deterministic_and_size_bounded() {
        let a = word_soup(42, 100);
        let b = word_soup(42, 100);
        assert_eq!(a, b, "同 seed 必须同文本(对拍/复现)");
        assert_ne!(a, word_soup(43, 100), "异 seed 应不同(前缀随机化)");
        assert_eq!(a.split(' ').count(), 100);
        assert_eq!(word_count_for(1), 1);
        assert_eq!(word_count_for(256), 182);
    }

    #[test]
    fn report_flags_missing_batching() {
        let spec = LoadSpec {
            concurrency: Some(2),
            tokens: Some(8),
            prompt_tokens: Some(64),
            random_prefix: None,
            timeout_ms: None,
            session_prefix: None,
            mode: None,
            delay_ms: None,
            a_prompt_tokens: None,
            a_tokens: None,
            b_prompt_tokens: None,
            b_tokens: None,
        }
        .normalize(8, 512, 1)
        .expect("norm");
        let outs = vec![
            WorkerOut {
                session: 1,
                turn: Some(1),
                prompt_tokens: 64,
                completion_tokens: 8,
                ttft_ms: Some(10.0),
                wall_ms: 100.0,
                finish: Some("Length".into()),
                ..Default::default()
            },
            WorkerOut {
                session: 2,
                turn: Some(2),
                prompt_tokens: 64,
                completion_tokens: 8,
                ttft_ms: Some(20.0),
                wall_ms: 110.0,
                finish: Some("Length".into()),
                ..Default::default()
            },
        ];
        // 批步没涨 ⇒ batching=false(判据在场)
        let v = report_json(&spec, st(10), Some(st(10)), outs.clone(), false);
        assert_eq!(v["batch"]["batching"], json!(false));
        assert_eq!(v["ok"], json!(true));
        // 批步涨了 ⇒ true
        let v = report_json(&spec, st(10), Some(st(20)), outs, false);
        assert_eq!(v["batch"]["batching"], json!(true));
        assert_eq!(v["batch"]["steps_delta"], json!(10));
        // 超时 ⇒ ok=false + 503 语义(调用方据此给 503)
        let v = report_json(&spec, st(0), Some(st(0)), Vec::new(), true);
        assert_eq!(v["ok"], json!(false));
        assert_eq!(v["timed_out"], json!(true));
    }
}
