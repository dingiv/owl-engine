//! OpenAI 兼容面(chat completions;架子覆盖本服务实际会用到的子集)。
//!
//! 会话映射(引擎 S0 接线):OpenAI 协议无状态、客户端全量重发历史 ——
//! 与引擎会话模型(全量 prompt + 前缀增量)天然同构。请求扩展字段
//! `"session"`(任意字符串)为会话键:同键跨请求 = 引擎侧同一会话,
//! 命中账本则只 prefill 新增后缀;不带 = 每请求独立(临时会话)。
//!
//! 模块布局(2026-10-10 clean-code 轮):§1 协议数据/JSON 构造;
//! §2 chat 事件流消费(自 main.rs 迁入:单折叠循环 + usage 单源 +
//! 流式/非流式出口策略)。
//!
//! 挂账:多轮 chat template 正式渲染、logprobs/n/temperature 采样参数
//! (引擎现为 greedy)、usage 模型侧真账(现 = 事件面计数)。

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use owl_engine::TurnEvent;

use crate::actor::{EngineReq, SinkEvent};
use crate::http;

// ============================================================================
// §1 协议数据与 JSON 构造
// ============================================================================

#[derive(Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    /// 扩展字段:会话键(服务端哈希为 u64 会话 id;见模块头)
    #[serde(default)]
    pub session: Option<String>,
}

#[derive(Deserialize)]
pub struct ChatMessage {
    /// 角色标签(system/user/assistant;naive 渲染暂不区分 ——
    /// 正式 chat template 落地后启用)
    #[serde(default)]
    #[allow(dead_code)]
    pub role: String,
    /// string 或分段数组([{type:"text",text:…}]);架子两种都收
    #[serde(default)]
    pub content: Value,
}

impl ChatMessage {
    pub fn text(&self) -> String {
        match &self.content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }
}

/// 历史 → prompt(naive 全拼接 + 引擎侧单轮 chat_wrap)。
/// **TODO(正式模板)**:多轮 chat template 渲染 —— 现状跨 turn 前缀
/// 一般不逐字节相等 → 引擎前缀守卫失配回退全量重算(正确性保住,
/// 缓存命中待模板落地;信息不丢,记忆可通)
pub fn render_prompt(msgs: &[ChatMessage]) -> String {
    // 多轮 Qwen chat 模板渲染(2026-10-13;§三十二:naive 拼接 + chat_wrap
    // 单轮包裹 = 多轮幻觉/指令服从崩坏根因,raw A/B 臂定谳)。
    // 挂账:家族模板串通用化(经 tokenizer ChatFormat/jinja,现 Qwen 硬编码)。
    let mut out = String::new();
    for m in msgs {
        let text = m.text();
        if text.trim().is_empty() {
            continue;
        }
        let role = m.role.as_str();
        let tag = match role {
            "system" => "system",
            "assistant" => "assistant",
            _ => "user",
        };
        out += &format!("<|im_start|>{tag}\n{text}<|im_end|>\n");
    }
    // assistant 起手 + <think> 开块(27B = thinking 模型:模型自己续写
    // 推理;空 think 预填 = 非思考调法,thinking 模型在错误上下文生成 →
    // 输出退化/答案段损坏 —— §三十三 AL 案 + §三十五 L3 同源实证)
    out + "<|im_start|>assistant\n<think>\n"
}

/// OpenAI 协议会话键 → 引擎 u64 会话 id(进程内哈希;同键同 id)
pub fn session_id(key: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

pub fn completion_json(
    id: &str,
    model: &str,
    text: &str,
    finish_reason: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        // usage 真账(2026-10-02):prompt_tokens = 引擎 Prefill 事件 total;
        // completion_tokens = Token 事件计数(1 事件 = 1 采样步 = 1 token)
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })
}

/// 流式 chunk(delta 由调用方拼;首 chunk 传 role,末 chunk 传 finish_reason)
pub fn chunk_json(id: &str, model: &str, delta: Value, finish_reason: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    })
}

pub fn error_json(msg: impl Into<String>) -> Value {
    json!({"error": {"message": msg.into(), "type": "invalid_request_error"}})
}

fn now() -> u64 {
    unix().as_secs()
}

fn unix() -> std::time::Duration {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
}

pub fn completion_id() -> String {
    format!("chatcmpl-{:x}", unix().as_nanos())
}

// ============================================================================
// §2 chat completions 事件流消费(2026-10-10 自 main.rs 迁入)
// ============================================================================

/// 引擎事件回流通道深度(与提交通道同档;R2 命名)
const SINK_CAPACITY: usize = 64;

/// 默认生成步数(请求未带 max_tokens 时)
const DEFAULT_MAX_NEW: usize = 24;

/// 日志预览截断长度
const LOG_PREVIEW_CHARS: usize = 60;

/// 应答出口策略(流式 = 逐帧发射;非流式 = 聚合整包)
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReplyMode {
    Stream,
    Aggregate,
}

/// usage 账本(单源:prompt 初值 = Accepted 捎带,Prefill 末笔覆盖;
/// completion = Token 事件计数,1 事件 = 1 采样步 = 1 token)
#[derive(Default)]
struct Usage {
    prompt: usize,
    completion: usize,
}

/// 日志用长文截断
fn preview(s: &str) -> String {
    if s.chars().count() <= LOG_PREVIEW_CHARS {
        s.to_string()
    } else {
        let head: String = s.chars().take(LOG_PREVIEW_CHARS).collect();
        format!("{head}…")
    }
}

/// chat completions:提交 actor → 首笔 Accepted/Rejected → 事件折叠循环。
/// 双 loop 已合一(原流式/非流式各自 recv→match 的重复面收编):usage
/// 累计与事件分派只写一遍,出口按 [`ReplyMode`] 分臂。None 断流行为
/// 保持原差异:流式已出帧 → 静默收口;非流式 → 503。
pub(crate) async fn chat<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    body: &[u8],
    tx: mpsc::Sender<EngineReq>,
    model_name: &str,
) -> std::io::Result<()> {
    let req: ChatRequest = match serde_json::from_slice::<ChatRequest>(body) {
        Ok(r) if !r.messages.is_empty() => r,
        Ok(_) => return http::respond_json(w, 400, &error_json("messages 为空")).await,
        Err(e) => {
            return http::respond_json(w, 400, &error_json(format!("请求体非法: {e}"))).await
        }
    };
    let mode = if req.stream { ReplyMode::Stream } else { ReplyMode::Aggregate };
    // 应答 model 名:请求指定优先(OpenAI 客户端习惯回显),否则服务端缺省
    let model_name = req.model.as_deref().unwrap_or(model_name);
    let (sink_tx, mut sink_rx) = mpsc::channel::<SinkEvent>(SINK_CAPACITY);
    let engine_req = EngineReq::Chat {
        session: req.session.as_deref().map(session_id),
        prompt: render_prompt(&req.messages),
        max_new: req.max_tokens.unwrap_or(DEFAULT_MAX_NEW),
        sink: sink_tx,
    };
    if tx.send(engine_req).await.is_err() {
        return http::respond_json(w, 503, &error_json("引擎退役")).await;
    }

    // 首笔:受理确认 / 提交拒绝(预算越界等,submit 边界落地)
    let mut usage = Usage::default();
    let turn = match sink_rx.recv().await {
        Some(SinkEvent::Accepted { session, turn, prompt_tokens }) => {
            usage.prompt = prompt_tokens;
            eprintln!("[chat] 受理 session {session:#x} turn {turn}(stream={})", mode == ReplyMode::Stream);
            Some((session, turn))
        }
        Some(SinkEvent::Rejected { err }) => {
            return http::respond_json(w, 400, &error_json(err)).await
        }
        _ => return http::respond_json(w, 503, &error_json("引擎退役")).await,
    };

    let id = completion_id();
    // role 首帧延到首个 Token(2026-10-02:提前发会让测速端把 HTTP 往返
    // 当 TTFT,长 prompt 的 prefill 全被藏进首帧延迟)
    if mode == ReplyMode::Stream {
        http::respond_sse_head(w).await?;
    }
    let mut role_sent = false;
    loop {
        match sink_rx.recv().await {
            Some(SinkEvent::Turn(TurnEvent::Token { delta, .. })) => {
                usage.completion += 1;
                if mode == ReplyMode::Stream {
                    if !role_sent {
                        role_sent = true;
                        let first = chunk_json(
                            &id, model_name, json!({"role": "assistant", "content": ""}), None,
                        );
                        http::sse_data(w, &serde_json::to_string(&first).unwrap_or_default()).await?;
                    }
                    let chunk = chunk_json(&id, model_name, json!({"content": delta}), None);
                    http::sse_data(w, &serde_json::to_string(&chunk).unwrap_or_default()).await?;
                }
            }
            // Prefill 进度帧(末笔 total = prompt token 数,入 usage)
            Some(SinkEvent::Turn(TurnEvent::Prefill { total, .. })) => {
                usage.prompt = total;
            }
            Some(SinkEvent::Turn(TurnEvent::Completed { text, .. })) => {
                return match mode {
                    ReplyMode::Stream => {
                        let chunk = chunk_json(&id, model_name, json!({}), Some("stop"));
                        http::sse_data(w, &serde_json::to_string(&chunk).unwrap_or_default()).await?;
                        http::sse_data(w, "[DONE]").await?;
                        eprintln!("[chat] turn {:?} 完成:{}", turn.map(|(_, t)| t), preview(&text));
                        Ok(())
                    }
                    ReplyMode::Aggregate => {
                        http::respond_json(
                            w,
                            200,
                            &completion_json(
                                &id, model_name, &text, "stop", usage.prompt, usage.completion,
                            ),
                        )
                        .await
                    }
                };
            }
            Some(SinkEvent::Turn(TurnEvent::Failed { err, .. })) => {
                return match mode {
                    // SSE 中途错误:协议无标准错误帧,error json + DONE 收口
                    ReplyMode::Stream => {
                        let _ = http::sse_data(
                            w,
                            &serde_json::to_string(&error_json(err)).unwrap_or_default(),
                        )
                        .await;
                        let _ = http::sse_data(w, "[DONE]").await;
                        Ok(())
                    }
                    ReplyMode::Aggregate => http::respond_json(w, 500, &error_json(err)).await,
                };
            }
            Some(SinkEvent::Turn(_)) => {}
            // 协议不变式:首笔之后不再有 Accepted/Rejected(与断流同处置,
            // 保持原 `_` 兜臂行为:流式静默收口 / 非流式 503)
            Some(SinkEvent::Accepted { .. }) | Some(SinkEvent::Rejected { .. })
                if mode == ReplyMode::Stream =>
            {
                break
            }
            Some(SinkEvent::Accepted { .. }) | Some(SinkEvent::Rejected { .. }) => {
                return http::respond_json(w, 503, &error_json("引擎连接中断")).await;
            }
            None if mode == ReplyMode::Stream => break,
            None => {
                return http::respond_json(w, 503, &error_json("引擎连接中断")).await;
            }
        }
    }
    Ok(())
}
