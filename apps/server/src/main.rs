//! owl 引擎服务架子(2026-09-26):OpenAI 兼容 `/v1/chat/completions`
//! (stream SSE / 非 stream JSON)+ `/health` + `/v1/models`。
//!
//! 形态(charter A3 同步隔离 / A2.6 每卡一线程先声):**engine actor 独占
//! 专属 OS 线程** —— `RunningEngine` 非 Send(权重 Cell 内变异性 + 图闭包),
//! 构造/装载/泵全部在该线程内闭合,HTTP 层(多线程 tokio)只持提交通道;
//! 装配完成经 ready 信号回主线程,之后才开始监听(架子:引擎就绪前无
//! HTTP 面)。
//!
//! 环境:OWL_BIND(默认 127.0.0.1:8135)/ OWL_DEVICE(ordinal,默认 0)/
//! OWL_MODEL_DIR(默认 workspace 内 Qwen3.5-0.8B)/ OWL_MODEL_NAME /
//! OWL_MAX_SEQ(默认 4096;attention 窗上限,E1 后 paged 布局解除 256 顶)/ OWL_PREFILL_CHUNK(默认 32)。
//!
//! 用法:
//! ```text
//! cargo run -p server --release          # 或 OWL_DEVICE=3 cargo run -p server
//! curl 127.0.0.1:8135/health
//! curl 127.0.0.1:8135/v1/chat/completions -H 'content-type: application/json' \
//!   -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":24,"stream":true,"session":"demo"}'
//! ```
//!
//! 架子边界(挂账):chat 模板 naive 渲染(openai.rs)、usage 真账、
//! 采样参数(greedy)、keep-alive/chunked(http.rs)、turn abort
//! (actor.rs pump Err 处置)、引擎就绪前的 503 健康面。

mod actor;
mod http;
mod openai;

use std::path::PathBuf;

use owl_engine::TurnEvent;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::actor::{EngineReq, SinkEvent};
use crate::openai::ChatRequest;

#[tokio::main]
async fn main() {
    let bind = env_or("OWL_BIND", "127.0.0.1:8135".into());
    let model_name = env_or("OWL_MODEL_NAME", "qwen3.5-0.8b".into());
    let device: usize = env_or("OWL_DEVICE", "0".into()).parse().unwrap_or(0);
    // 4k 默认(E1 收尾:paged 布局 + carve 对齐落地后解除 256 顶;
    // 旧 naive 回退路径仍受核自身限制,层分派表自动选路)
    let max_seq: usize = env_or("OWL_MAX_SEQ", "4096".into()).parse().unwrap_or(4096);
    let chunk: usize = env_or("OWL_PREFILL_CHUNK", "32".into()).parse().unwrap_or(32);
    let model_dir = std::env::var("OWL_MODEL_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../crates/models/assets/Qwen3.5-0.8B")
    });
    // 模型档位选择(2026-10-01):0.8b(缺省,f16)/ awq27b(Qwen3.8-27B
    // AWQ-INT4;OWL_MODEL_DIR = 检查点目录,tokenizer 同目录)
    let model_kind = env_or("OWL_MODEL_KIND", "0.8b".into());

    // ── engine actor:专属线程(构造+装载+泵全在内;RunningEngine 非 Send)──
    let (tx, rx) = mpsc::channel::<EngineReq>(64);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<String, String>>();
    std::thread::Builder::new()
        .name("owl-engine".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("actor runtime");
            rt.block_on(actor_main(model_dir, model_kind, device, max_seq, chunk, rx, ready_tx));
        })
        .expect("actor 线程");

    // 装配结果(引擎就绪前不开 HTTP 面;架子边界)
    match ready_rx.recv().expect("actor 线程存活") {
        Ok(capture) => eprintln!("[boot] 引擎就绪(捕获态 {capture})"),
        Err(e) => {
            eprintln!("[boot] 引擎装配失败: {e}");
            std::process::exit(1);
        }
    }

    let listener = TcpListener::bind(&bind).await.expect("bind");
    eprintln!("[boot] 监听 http://{bind}(OpenAI 兼容;model={model_name})");
    loop {
        let Ok((stream, _peer)) = listener.accept().await else { continue };
        let tx = tx.clone();
        let model_name = model_name.clone();
        tokio::spawn(async move {
            if let Err(e) = serve(stream, tx, model_name).await {
                eprintln!("[conn] {e}");
            }
        });
    }
}

/// actor 线程主:引擎生命周期 facade → 事件泵(actor::run_actor)
async fn actor_main(
    model_dir: PathBuf,
    model_kind: String,
    device: usize,
    max_seq: usize,
    chunk: usize,
    rx: mpsc::Receiver<EngineReq>,
    ready: std::sync::mpsc::Sender<Result<String, String>>,
) {
    let built: owl_engine::Result<_> = async {
        let t0 = std::time::Instant::now();
        let mut engine = owl_engine::Engine::new(owl_engine::EngineConfig {
            device_ordinal: device,
            max_seq_tokens: max_seq,
            prefill_chunk: chunk,
        })?;
        eprintln!("[boot] 设备绑定 {:.2}s", t0.elapsed().as_secs_f32());
        let model = match model_kind.as_str() {
            "awq27b" => {
                engine.loader().load_qwen38_27b_awq(&model_dir, &model_dir).await?
            }
            _ => engine.loader().load_qwen35_0_8b(&model_dir).await?,
        };
        eprintln!("[boot] 权重/tokenizer/rope 装载 {:.2}s", t0.elapsed().as_secs_f32());
        let running = engine.run(model).await?;
        eprintln!("[boot] 引擎装配(run)总计 {:.2}s", t0.elapsed().as_secs_f32());
        Ok(running)
    }
    .await;
    let mut running = match built {
        Ok(r) => {
            let _ = ready.send(Ok(format!("{:?}", r.capture_outcome)));
            r
        }
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    actor::run_actor(&mut running, rx).await;
}

fn env_or(k: &str, dflt: String) -> String {
    std::env::var(k).unwrap_or(dflt)
}

/// 单连接服务(Connection: close;架子一请求一连接)
async fn serve(
    mut stream: tokio::net::TcpStream,
    tx: mpsc::Sender<EngineReq>,
    model_name: String,
) -> std::io::Result<()> {
    let Some(req) = http::read_request(&mut stream).await? else {
        return Ok(());
    };
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            http::respond_json(&mut stream, 200, &json!({"ok": true, "model": model_name})).await?;
        }
        ("GET", "/v1/models") => {
            http::respond_json(
                &mut stream,
                200,
                &json!({"object": "list", "data": [{
                    "id": model_name, "object": "model", "owned_by": "owl",
                }]}),
            )
            .await?;
        }
        ("POST", "/v1/chat/completions") => {
            chat(&mut stream, &req.body, tx, &model_name).await?;
        }
        // 诊断分账出口(OWL_GPU_PROF=1 时 store 非空)—— 逐核 GPU 时
        // 总账降序;TagReport 无 Serialize,手工投影(总 ms 计)
        ("GET", "/debug/metrics") => {
            let mut rows: Vec<serde_json::Value> = owl_shared::metrics::collect_metrics(
                &owl_shared::metrics::MetricsFilter::new().limit(128),
            )
            .into_iter()
            .map(|r| json!({
                "tag": r.tag, "count": r.count,
                "total_ms": r.total.as_secs_f64() * 1e3,
                "avg_ms": r.avg.as_secs_f64() * 1e3,
                "p50_ms": r.p50.as_secs_f64() * 1e3,
                "p99_ms": r.p99.as_secs_f64() * 1e3,
            }))
            .collect();
            rows.sort_by(|a, b| {
                let f = |v: &serde_json::Value| v["total_ms"].as_f64().unwrap_or(0.0);
                f(b).partial_cmp(&f(a)).unwrap_or(std::cmp::Ordering::Equal)
            });
            http::respond_json(&mut stream, 200, &json!({"rows": rows})).await?;
        }
        _ => {
            http::respond_json(
                &mut stream,
                404,
                &openai::error_json(format!("未知路径 {} {}", req.method, req.path)),
            )
            .await?;
        }
    }
    stream.shutdown().await
}

/// chat completions:提交 actor → 首笔 Accepted/Rejected → 事件流消费
/// (SSE 逐 Token 出帧;非流式攒全文整包回)
async fn chat<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    body: &[u8],
    tx: mpsc::Sender<EngineReq>,
    model_name: &str,
) -> std::io::Result<()> {
    let req: ChatRequest = match serde_json::from_slice::<ChatRequest>(body) {
        Ok(r) if !r.messages.is_empty() => r,
        Ok(_) => return http::respond_json(w, 400, &openai::error_json("messages 为空")).await,
        Err(e) => {
            return http::respond_json(w, 400, &openai::error_json(format!("请求体非法: {e}")))
                .await
        }
    };
    let stream_mode = req.stream;
    // 应答 model 名:请求指定优先(OpenAI 客户端习惯回显),否则服务端缺省
    let model_name = req.model.as_deref().unwrap_or(model_name);
    let (sink_tx, mut sink_rx) = mpsc::channel::<SinkEvent>(64);
    let engine_req = EngineReq::Chat {
        session: req.session.as_deref().map(openai::session_id),
        prompt: openai::render_prompt(&req.messages),
        max_new: req.max_tokens.unwrap_or(24),
        sink: sink_tx,
    };
    if tx.send(engine_req).await.is_err() {
        return http::respond_json(w, 503, &openai::error_json("引擎退役")).await;
    }
    // 首笔:受理确认 / 提交拒绝(预算越界等,submit 边界落地)
    // usage 真账:Accepted 捎带 prompt 数;Token 事件计数 = completion 数
    let mut completion_tokens = 0usize;
    let mut prompt_tokens = 0usize;
    let turn = match sink_rx.recv().await {
        Some(SinkEvent::Accepted { session, turn, prompt_tokens: pt }) => {
            eprintln!("[chat] 受理 session {session:#x} turn {turn}(stream={stream_mode})");
            prompt_tokens = pt;
            Some((session, turn))
        }
        Some(SinkEvent::Rejected { err }) => {
            return http::respond_json(w, 400, &openai::error_json(err)).await
        }
        _ => return http::respond_json(w, 503, &openai::error_json("引擎退役")).await,
    };

    let id = openai::completion_id();
    if stream_mode {
        // ── SSE:role 首帧 → Token 逐帧 → finish + [DONE] ──
        // (2026-10-02:role 帧延到首个 Token —— 提前发会让测速端把
        // HTTP 往返当 TTFT,长 prompt 的 prefill 全被藏进首帧延迟)
        http::respond_sse_head(w).await?;
        let mut role_sent = false;
        loop {
            match sink_rx.recv().await {
                Some(SinkEvent::Turn(TurnEvent::Token { delta, .. })) => {
                    if !role_sent {
                        let first = openai::chunk_json(
                            &id, model_name, json!({"role": "assistant", "content": ""}), None,
                        );
                        http::sse_data(w, &serde_json::to_string(&first).unwrap_or_default()).await?;
                        role_sent = true;
                    }
                    completion_tokens += 1;
                    let chunk =
                        openai::chunk_json(&id, model_name, json!({"content": delta}), None);
                    http::sse_data(w, &serde_json::to_string(&chunk).unwrap_or_default()).await?;
                }
                Some(SinkEvent::Turn(TurnEvent::Completed { text, .. })) => {
                    let chunk = openai::chunk_json(&id, model_name, json!({}), Some("stop"));
                    http::sse_data(w, &serde_json::to_string(&chunk).unwrap_or_default()).await?;
                    http::sse_data(w, "[DONE]").await?;
                    eprintln!("[chat] turn {:?} 完成:{}", turn.map(|(_, t)| t), preview(&text));
                    break;
                }
                Some(SinkEvent::Turn(TurnEvent::Failed { err, .. })) => {
                    // SSE 中途错误:协议无标准错误帧,error json + DONE 收口
                    let _ = http::sse_data(
                        w,
                        &serde_json::to_string(&openai::error_json(err)).unwrap_or_default(),
                    )
                    .await;
                    let _ = http::sse_data(w, "[DONE]").await;
                    break;
                }
                // Prefill 进度帧(末笔 total = prompt token 数,入 usage)
                Some(SinkEvent::Turn(TurnEvent::Prefill { total, .. })) => {
                    prompt_tokens = total;
                }
                Some(SinkEvent::Turn(_)) => {}
                _ => break,
            }
        }
    } else {
        // ── 非流式:攒全文整包 ──
        loop {
            match sink_rx.recv().await {
                Some(SinkEvent::Turn(TurnEvent::Token { .. })) => {
                    completion_tokens += 1;
                }
                Some(SinkEvent::Turn(TurnEvent::Prefill { total, .. })) => {
                    prompt_tokens = total;
                }
                Some(SinkEvent::Turn(TurnEvent::Completed { text, .. })) => {
                    return http::respond_json(
                        w,
                        200,
                        &openai::completion_json(
                            &id, model_name, &text, "stop", prompt_tokens, completion_tokens,
                        ),
                    )
                    .await;
                }
                Some(SinkEvent::Turn(TurnEvent::Failed { err, .. })) => {
                    return http::respond_json(w, 500, &openai::error_json(err)).await;
                }
                Some(SinkEvent::Turn(_)) => {}
                _ => {
                    return http::respond_json(w, 503, &openai::error_json("引擎连接中断")).await;
                }
            }
        }
    }
    Ok(())
}

/// 日志用长文截断
fn preview(s: &str) -> String {
    if s.chars().count() <= 60 {
        s.to_string()
    } else {
        let head: String = s.chars().take(60).collect();
        format!("{head}…")
    }
}
