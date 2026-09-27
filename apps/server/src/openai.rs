//! OpenAI 兼容面(chat completions;架子覆盖本服务实际会用到的子集)。
//!
//! 会话映射(引擎 S0 接线):OpenAI 协议无状态、客户端全量重发历史 ——
//! 与引擎会话模型(全量 prompt + 前缀增量)天然同构。请求扩展字段
//! `"session"`(任意字符串)为会话键:同键跨请求 = 引擎侧同一会话,
//! 命中账本则只 prefill 新增后缀;不带 = 每请求独立(临时会话)。
//!
//! 挂账:usage 统计(prompt token 数在引擎内,事件面未回吐)、
//! 多轮 chat template 正式渲染、logprobs/n/temperature 采样参数
//! (引擎现为 greedy)。

use serde::Deserialize;
use serde_json::{json, Value};

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
    msgs.iter()
        .map(|m| m.text())
        .filter(|t| !t.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// OpenAI 协议会话键 → 引擎 u64 会话 id(进程内哈希;同键同 id)
pub fn session_id(key: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

pub fn completion_json(id: &str, model: &str, text: &str, finish_reason: &str) -> Value {
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
        // TODO: usage 真账(引擎事件面回吐 token 计数后接通)
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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
