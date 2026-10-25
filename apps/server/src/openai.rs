//! OpenAI 兼容面(chat completions;**agent 服务面专项 §3 契约**)。
//!
//! 会话映射(引擎 S0 接线):OpenAI 协议无状态、客户端全量重发历史 ——
//! 与引擎会话模型(全量 prompt + 前缀增量)天然同构。请求扩展字段
//! `"session"`(任意字符串)为会话键:同键跨请求 = 引擎侧同一会话。
//!
//! 模块布局:§1 协议数据 + 校验(不静默丢弃)/ §2 JSON 构造 /
//! §3 三通道事件流消费(思考 / 正文 / 工具调用)。
//!
//! **旧 `render_prompt` 已退役**(§4.3 渲染对称律 R2):prompt 一律经
//! [`ChatTpl`] 从模型快照 `chat_template.jinja` 渲染 —— 手写模板串
//! = 前缀复用零命中的根因。
//!
//! 挂账(M-C):取消/abort 的上游用法(`EngineReq::Abort` 已就位,pi 侧
//! 断流走路由 actor 自动弃置)。

use std::io::Write;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

use owl_engine::{TurnEvent, TurnOpts, TurnSampler};
use crate::config::AgentCfg;

use crate::actor::{EngineReq, SinkEvent};
use crate::agent_trace::{AgentTrace, OutcomeFacts, ReqFacts, TurnLedger, UsageFacts};
use crate::chat_tpl::{ChatTpl, RenderArgs};
use crate::http;
use crate::toolcall::{Chan, ChannelSplitter, Item, ToolSchemas};

// ============================================================================
// §1 协议数据与校验
// ============================================================================

/// 允许的 role(其余 = 400;不静默)
const ROLES: [&str; 5] = ["system", "developer", "user", "assistant", "tool"];

/// `stop` 串个数/长度上限(病态保护;超限 = 400 而非静默截断)
const MAX_STOP_SEQS: usize = 8;
const MAX_STOP_LEN: usize = 256;

#[derive(Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    /// **原样透传**给模板渲染器(单一权威:不做二次编码 ⇒ 不丢字段)
    pub messages: Vec<Value>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    /// 扩展字段:会话键(服务端哈希为 u64 会话 id)
    #[serde(default)]
    pub session: Option<String>,
    /// 流式 usage 开关(OpenAI 约定 `stream_options.include_usage`)
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// OpenAI 兼容扩展(部分实现通用):忽略 EOS,强制生成满 `max_tokens`
    /// —— **测速口径**(decode 吞吐要在足量生成上测)
    #[serde(default)]
    pub ignore_eos: bool,
    /// 工具声明(原样进模板 `# Tools` 块;`[]` 空数组合法)
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    /// 模板变量(pi `/thinking` 档走这里:`enable_thinking`/`preserve_thinking`)
    #[serde(default)]
    pub chat_template_kwargs: Option<Map<String, Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    // ── 采样参数(AGT-06:逐请求生效,接引擎 per-turn 采样面)──
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub seed: Option<i64>,
    /// 思考档位(OpenAI 同名协议值;**直通模板** `reasoning_effort`,
    /// 模板自持值域 none/low/medium/high/xhigh…;AGT-07)
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// 未识别字段(关键项拒启/次要项告警 —— 见 [`Self::validate`]/[`Self::warnings`])
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

fn is_image_part(p: &Value) -> bool {
    p.get("type").and_then(Value::as_str) == Some("image")
        || p.get("type").and_then(Value::as_str) == Some("image_url")
        || p.get("image_url").is_some()
}

fn is_text_part(p: &Value) -> bool {
    !is_image_part(p) && p.get("text").is_some()
}

impl ChatRequest {
    /// 结构校验:非法 → 400(不静默、不猜)
    pub fn validate(&self) -> Result<(), String> {
        if self.messages.is_empty() {
            return Err("messages 为空".into());
        }
        for (i, m) in self.messages.iter().enumerate() {
            let role = m
                .get("role")
                .and_then(|r| r.as_str())
                .ok_or_else(|| format!("messages[{i}].role 缺失或非字符串"))?;
            if !ROLES.contains(&role) {
                return Err(format!("messages[{i}].role 不认识:{role:?}"));
            }
            if role == "tool" && m.get("tool_call_id").and_then(|v| v.as_str()).is_none() {
                return Err(format!("messages[{i}]:role=tool 必须带 tool_call_id"));
            }
            // **图像输入面(V4)**:`content` 数组合法;`image_url.url` 必须是
            // data URL(预处理在 `ChatRequest::images`,真正的解码错误在那里报)。
            if let Some(arr) = m.get("content").and_then(|c| c.as_array()) {
                for (j, part) in arr.iter().enumerate() {
                    let is_image = part.get("type").and_then(|t| t.as_str())
                        == Some("image_url")
                        || part.get("image_url").is_some();
                    if is_image {
                        let url = part
                            .get("image_url")
                            .and_then(|v| v.get("url"))
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                format!(
                                    "messages[{i}].content[{j}].image_url.url 缺失或非字符串"
                                )
                            })?;
                        if !url.starts_with("data:") {
                            return Err(format!(
                                "messages[{i}].content[{j}]:图像 URL 只支持 data:image/...;base64(暂不支持 http)"
                            ));
                        }
                        // **可删标记**(前缀无感删除;2026-10-23):只认布尔
                        // (非布尔 = 协议错,必须在受理前结构化拒,不许静默当真)
                        if let Some(v) = part.get("owl_evict") {
                            if !v.is_boolean() && !v.is_null() {
                                return Err(format!(
                                    "messages[{i}].content[{j}].owl_evict 必须是布尔值(得 {v})"
                                ));
                            }
                        }
                    }
                }
            }
            if let Some(tcs) = m.get("tool_calls") {
                if role != "assistant" {
                    return Err(format!("messages[{i}]:仅 assistant 可带 tool_calls"));
                }
                let arr = tcs
                    .as_array()
                    .ok_or_else(|| format!("messages[{i}].tool_calls 必须是数组"))?;
                for (j, tc) in arr.iter().enumerate() {
                    let fd = format!("messages[{i}].tool_calls[{j}]");
                    if tc.get("id").and_then(|v| v.as_str()).is_none() {
                        return Err(format!("{fd}.id 缺失"));
                    }
                    let f = tc.get("function").ok_or_else(|| format!("{fd}.function 缺失"))?;
                    if f.get("name").and_then(|v| v.as_str()).is_none() {
                        return Err(format!("{fd}.function.name 缺失"));
                    }
                    // arguments:OpenAI 协议 = JSON 字符串(vLLM 亦收 dict,见 normalize)
                    match f.get("arguments") {
                        Some(Value::String(s)) => {
                            // 空串 = 无参调用(vLLM 归一为 {};此处只放行,不解析)
                            if !s.trim().is_empty() {
                                serde_json::from_str::<Value>(s).map_err(|e| {
                                    format!("{fd}.function.arguments 不是合法 JSON:{e}")
                                })?;
                            }
                        }
                        Some(Value::Object(_)) | Some(Value::Array(_)) => {}
                        _ => {
                            return Err(format!(
                                "{fd}.function.arguments 必须是 JSON 字符串或对象"
                            ))
                        }
                    }
                }
            }
        }
        // 未支持的关键参数:明确拒(而不是当没看见)
        if let Some(n) = self.extra.get("n") {
            if n.as_u64().unwrap_or(1) > 1 {
                return Err("n > 1 未支持(本服务单序列)".into());
            }
        }
        if self.extra.get("logprobs").and_then(|v| v.as_bool()).unwrap_or(false) {
            return Err("logprobs 未支持".into());
        }
        if let Some(rf) = self.extra.get("response_format") {
            match rf.get("type").and_then(|t| t.as_str()) {
                None | Some("text") => {}
                Some(other) => return Err(format!("response_format={other:?} 未支持(仅 text)")),
            }
        }
        match self.tool_choice.as_ref() {
            None | Some(Value::Null) => {}
            Some(Value::String(s)) if s == "auto" => {}
            Some(other) => {
                return Err(format!("tool_choice={other} 未支持(仅 auto/缺省)"));
            }
        }
        Ok(())
    }

    /// 收下但**仍未生效**的参数 → 告警(不许静默)。
    /// AGT-06 后温度/停止串已真生效,此处只剩未知字段
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.extra.is_empty() {
            out.push(format!("未知字段 {:?}", self.extra.keys().collect::<Vec<_>>()));
        }
        out
    }

    /// **多段 content 的文本边界规范化**(2026-10-25;服务端请求预处理)。
    ///
    /// 客户端删掉 image part 后,前后两个 text part 直接相邻;BPE 会在拼接
    /// 边界重新分词,导致引擎的 `detect_text_delete` 无法满足逐 token 前缀
    /// 契约(删除不被吸收、前缀缓存失效)。这里在**渲染之前**保证多段
    /// `content` 里的每个 text part 以 `\n` 结尾,给删除点留下稳定的文本边界。
    ///
    /// 影响面(**最小化**,专治“图片被删后的边界”):
    /// - plain string content 不动(pi / 普通文本请求零变化);
    /// - 只有 **user** 消息、**数组且 >=2 段** 才处理;
    /// - 单 text part 数组不动(很多客户端用 `[{"type":"text"}]` 表示纯文本);
    /// - 没有图片且 text part < 2 的数组不动;
    /// - image part / `owl_evict` / 顺序原样保留;
    /// - 幂等:已以 `\n` 结尾不重复加。
    ///
    /// 为什么不放模板/引擎层:模板是模型快照(模型无关的布局规则不应耦合进去),
    /// 引擎层已没有 content 分段(只剩 token);见 workload 回归。
    pub fn normalize_multipart_text(&mut self) -> usize {
        let mut changed = 0usize;
        for m in &mut self.messages {
            if m.get("role").and_then(Value::as_str) != Some("user") {
                continue;
            }
            let Some(arr) = m.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            if arr.len() < 2 {
                continue;
            }
            let has_image = arr.iter().any(is_image_part);
            let text_parts = arr.iter().filter(|p| is_text_part(p)).count();
            // 无图且只有一段 text = 纯文本数组,不动;其余(带图 / 多 text)规范化。
            if !has_image && text_parts < 2 {
                continue;
            }
            for part in arr.iter_mut() {
                if !is_text_part(part) {
                    continue;
                }
                let Some(text) = part.get_mut("text") else {
                    continue;
                };
                let Some(s) = text.as_str() else {
                    continue;
                };
                if s.ends_with('\n') {
                    continue;
                }
                let mut next = String::with_capacity(s.len() + 1);
                next.push_str(s);
                next.push('\n');
                *text = Value::String(next);
                changed += 1;
            }
        }
        if changed > 0 {
            owl_shared::log_debug!(
                "chat",
                "多段 content 文本边界规范化:{changed} 处补 \n(删除吸收 / 前缀稳定)"
            );
        }
        changed
    }

    /// 采样覆写(AGT-06;`None`/空 = 请求未表达采样意图)
    pub fn sampler_override(&self) -> Option<TurnSampler> {
        let s = TurnSampler {
            temp: self.temperature.map(|v| v as f32),
            topk: self.top_k.map(|v| v as usize),
            topp: self.top_p.map(|v| v as f32),
        };
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    /// **收集并预处理图像**(内容顺序 = 模板 `<|image_pad|>` 顺序)。
    ///
    /// 仅 data URL;`vision_processor` 负责 Qwen2VL 的 resize/patchify。
    /// 无图像 = 空(纯文本逐字节旧行为)。
    ///
    /// **可删标记**(前缀无感删除;2026-10-23 用户裁决):图像 content part 上
    /// 的 `owl_evict: true` = 「这张图将来可能从历史里被删」。引擎**不自作
    /// 主张**:只有打了标 + `[pool] image_evict = true` 才补页对齐 pad 并
    /// 进删除候选;没打标 = 零行为差异。
    pub fn images(&self) -> Result<Vec<owl_engine::TurnImage>, String> {
        let mut out = Vec::new();
        for (i, m) in self.messages.iter().enumerate() {
            let Some(arr) = m.get("content").and_then(|c| c.as_array()) else {
                continue;
            };
            for (j, part) in arr.iter().enumerate() {
                let is_image = part.get("type").and_then(|t| t.as_str())
                    == Some("image_url")
                    || part.get("image_url").is_some();
                if !is_image {
                    continue;
                }
                let url = part
                    .get("image_url")
                    .and_then(|v| v.get("url"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        format!("messages[{i}].content[{j}].image_url.url 缺失或非字符串")
                    })?;
                let image = crate::vision_processor::from_data_url(url)
                    .map_err(|e| format!("messages[{i}].content[{j}] 图像预处理失败: {e}"))?;
                let evict = match part.get("owl_evict") {
                    None | Some(Value::Null) => false,
                    Some(Value::Bool(b)) => *b,
                    Some(other) => {
                        return Err(format!(
                            "messages[{i}].content[{j}].owl_evict 必须是布尔值(得 {other})"
                        ))
                    }
                };
                out.push(owl_engine::TurnImage { image, evict });
            }
        }
        Ok(out)
    }

    /// 停止串归一(OpenAI 允许 string 或 string[];非法 = 结构化拒)
    pub fn stop_strings(&self) -> Result<Vec<String>, String> {
        let v = match &self.stop {
            None | Some(Value::Null) => return Ok(Vec::new()),
            Some(v) => v,
        };
        let list: Vec<String> = match v {
            Value::String(s) => vec![s.clone()],
            Value::Array(a) => a
                .iter()
                .map(|x| {
                    x.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| "stop 数组元素必须是字符串".to_string())
                })
                .collect::<Result<_, _>>()?,
            _ => return Err("stop 必须是字符串或字符串数组".into()),
        };
        let list: Vec<String> = list.into_iter().filter(|s| !s.is_empty()).collect();
        if list.len() > MAX_STOP_SEQS {
            return Err(format!("stop 串个数 {} 超过上限 {MAX_STOP_SEQS}", list.len()));
        }
        if let Some(over) = list.iter().find(|s| s.len() > MAX_STOP_LEN) {
            return Err(format!("stop 串过长({}B > {MAX_STOP_LEN}B)", over.len()));
        }
        Ok(list)
    }

    /// **vLLM 同款归一**(`vllm/entrypoints/chat_utils.py::_postprocess_messages`):
    /// 模板期望 `assistant.tool_calls[].function.arguments` 是**对象**
    /// (Transformers 维护者约定:"need to be dicts not JSON str"),而 OpenAI
    /// 协议传的是 JSON 字符串 ⇒ 渲染前转 dict。附带:空 `tool_calls` 数组剔除
    /// (回到普通 assistant 路径,保持模板分支一致)。
    ///
    /// 实证:本机 0.8B 原版模板遇字符串参数直接抛
    /// `Can only get item pairs from a mapping`(**HF 里同样失败**)⇒ 不归一是
    /// 结构性不可用,不是风格问题。
    pub fn normalize_tool_calls(&mut self) -> Result<(), String> {
        for (i, m) in self.messages.iter_mut().enumerate() {
            if m.get("role").and_then(|r| r.as_str()) != Some("assistant") {
                continue;
            }
            let empty = m
                .get("tool_calls")
                .and_then(|t| t.as_array())
                .map(|a| a.is_empty())
                == Some(true);
            if empty {
                if let Some(o) = m.as_object_mut() {
                    o.remove("tool_calls");
                }
                continue;
            }
            let Some(arr) = m.get_mut("tool_calls").and_then(|t| t.as_array_mut()) else {
                continue;
            };
            for (j, tc) in arr.iter_mut().enumerate() {
                let fd = format!("messages[{i}].tool_calls[{j}]");
                let ty = tc.get("type").and_then(|v| v.as_str()).unwrap_or("function");
                if ty != "function" {
                    return Err(format!("{fd}.type={ty:?} 未支持(仅 function)"));
                }
                let f = tc
                    .get_mut("function")
                    .and_then(|f| f.as_object_mut())
                    .ok_or_else(|| format!("{fd}.function 缺失或非对象"))?;
                let args = f
                    .get_mut("arguments")
                    .ok_or_else(|| format!("{fd}.function.arguments 缺失"))?;
                let norm = match &*args {
                    Value::String(s) if s.trim().is_empty() => Value::Object(Map::new()),
                    Value::String(s) => serde_json::from_str(s)
                        .map_err(|e| format!("{fd}.function.arguments 不是合法 JSON:{e}"))?,
                    v @ (Value::Object(_) | Value::Array(_)) => v.clone(),
                    other => {
                        return Err(format!("{fd}.function.arguments 类型非法:{other}"));
                    }
                };
                *args = norm;
            }
        }
        Ok(())
    }

    /// 生成步数(夹紧到 `agent.max_new_cap`;夹紧情况返回 true 供调用方告警)
    pub fn max_new(&self, agent: &AgentCfg) -> (usize, bool) {
        let want = self
            .max_tokens
            .filter(|v| *v > 0)
            .unwrap_or(agent.default_max_new);
        (want.min(agent.max_new_cap), want > agent.max_new_cap)
    }

    /// **模板 kwargs 的唯一组装点**(vLLM `build_chat_params` 同款)。
    ///
    /// OpenAI 面的 `reasoning_effort` 与 Qwen 模板面的 `enable_thinking`
    /// 是同一个状态的两个入口。只把前者塞进模板而不管后者 ⇒ 模板可能
    /// 关思考、`ChannelSplitter` 却按默认 true 启动，答案会被整段吞进
    /// `reasoning_content`。这里在请求层归一化：显式 `enable_thinking`
    /// 优先；否则 `reasoning_effort` 存在时同步派生。
    pub fn effective_template_kwargs(&self) -> Map<String, Value> {
        let mut kw = self.chat_template_kwargs.clone().unwrap_or_default();
        if let Some(e) = &self.reasoning_effort {
            kw.insert("reasoning_effort".into(), json!(e));
            if !kw.contains_key("enable_thinking") {
                kw.insert("enable_thinking".into(), json!(!reasoning_off(e)));
            }
        }
        kw
    }

    /// 开启思考?(模板契约:未给 = 模板缺省 true)
    ///
    /// 与 [`effective_template_kwargs`](Self::effective_template_kwargs)
    /// 同口径：显式 `enable_thinking` > `reasoning_effort` none/off > true。
    /// 不能让模板关思考而 splitter 仍按 think 态启动。
    pub fn enable_thinking(&self) -> bool {
        if let Some(v) = self
            .chat_template_kwargs
            .as_ref()
            .and_then(|k| k.get("enable_thinking"))
            .and_then(|v| v.as_bool())
        {
            return v;
        }
        match self.reasoning_effort.as_deref() {
            Some(e) if reasoning_off(e) => false,
            _ => true,
        }
    }
}

/// `reasoning_effort` 的关思考词（与快照模板同口径；大小写不敏感）。
fn reasoning_off(e: &str) -> bool {
    e.eq_ignore_ascii_case("none") || e.eq_ignore_ascii_case("off")
}

/// 隐式默认会话 id(AGT-09;`[agent] implicit_session` = true 时无 session 键的请求落此)
pub const IMPLICIT_SESSION: u64 = 0;

/// OpenAI 协议会话键 → 引擎 u64 会话 id(进程内哈希;同键同 id)
pub fn session_id(key: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

// ============================================================================
// §2 JSON 构造
// ============================================================================

pub fn completion_json(
    id: &str,
    model: &str,
    msg: &Value,
    finish_reason: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
    reasoning_tokens: usize,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
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

/// **契约溢出类**拒绝的规范短语(pi 的 `OVERFLOW_PATTERNS` 原文之一)。
///
/// 用途:引擎的预算越界拒绝(`submit: prompt N + 生成 M 超预算 K`)是**中文裸
/// 报文**——pi 的 `isContextOverflow()` **只看报文文本**与它的
/// `OVERFLOW_PATTERNS` 做正则匹配(bundle 实测:`prompt (is )?too long` /
/// `prompt exceeds max length` / `/request_too_large/` /
/// `input is too long for requested model` / `exceeds the context window` /
/// `exceeds … maximum context length…`);不在词表里 ⇒ pi 当**普通 400**
/// 原样展示,**不触发自动 compaction** —— 这正是《设计》§3.6
/// 「上下文超限须归一化为 context_length_exceeded 语义」的挂账项。
///
/// 修法:越界拒绝一律在 message 尾追加本短语 + `code` 字段;两轮重试
/// (pi 压缩后重发)即自愈。
const OVERFLOW_PHRASE: &str = "prompt exceeds max length";

/// 引擎"预算越界"的稳定标记(中文诊断是引擎的既有输出)
fn is_budget_reject(err: &str) -> bool {
    err.contains("超预算")
}

/// 拒绝面 → wire 归一化(唯一实现;两处拒绝出口共用)
pub fn error_json_for_reject(err: String) -> Value {
    if is_budget_reject(&err) {
        return json!({"error": {
            "message": format!(
                "{err}({OVERFLOW_PHRASE}: prompt + max_tokens > max_seq;请缩短上下文/压缩历史后重试)"
            ),
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }});
    }
    error_json(err)
}

fn now() -> u64 {
    unix().as_secs()
}

fn unix() -> std::time::Duration {
    std::time::SystemTime::now()
        // 系统时钟回拨兜底 → 0(非失败路径;仅用于时间戳/ID)
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
}

pub fn completion_id() -> String {
    format!("chatcmpl-{:x}", unix().as_nanos())
}

/// 日志用长文截断
fn preview(s: &str) -> String {
    const N: usize = 60;
    if s.chars().count() <= N {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(N).collect::<String>())
    }
}

// ============================================================================
// §3 chat completions 事件流消费(三通道)
// ============================================================================

/// 引擎事件回流通道深度(与提交通道同档;R2 命名)
const SINK_CAPACITY: usize = 64;

/// 应答出口策略(流式 = 逐帧发射;非流式 = 聚合整包)
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReplyMode {
    Stream,
    Aggregate,
}

/// usage 账本(单源:prompt 初值 = Accepted 捎带,Prefill 末笔覆盖;
/// completion = Token 事件计数,1 事件 = 1 采样步 = 1 token;
/// cached = Accepted 捎带的**前缀命中量**,AGT-12)
#[derive(Default)]
struct Usage {
    prompt: usize,
    completion: usize,
    /// 思考相位采样步数(`completion_tokens_details.reasoning_tokens`;
    /// 真源 = [`ChannelSplitter::reasoning_steps`],在 Completed 时取终值)
    reasoning: usize,
    cached: usize,
}

/// 非流式聚合器(三通道 → 单条 message)
#[derive(Default)]
struct Agg {
    text: String,
    reason: String,
    tools: Vec<(String, String, String)>, // (id, name, arguments)
    seed: u64,
}

impl Agg {
    /// 工具调用 id(进程内唯一;pi 按 index 归并,id 只作标识)
    fn tool_id(&self, index: usize) -> String {
        format!("call_{:x}_{index}", self.seed)
    }
}

/// 把模板里的每个 `<|image_pad|>` 展开成 `grid_h*grid_w/4` 个。
///
/// 顺序 = `images` 序(内容序)。少/多占位 = 400(不许静默错位)。
fn expand_image_pads(prompt: &str, images: &[owl_engine::TurnImage]) -> Result<String, String> {
    const PAD: &str = "<|image_pad|>";
    let mut out = String::with_capacity(prompt.len());
    let mut rest = prompt;
    for (i, img) in images.iter().enumerate() {
        let pos = rest
            .find(PAD)
            .ok_or_else(|| format!("模板缺第 {i} 张图的 <|image_pad|> 占位"))?;
        out.push_str(&rest[..pos]);
        let n = (img.image.grid.1 * img.image.grid.2) / 4;
        for _ in 0..n {
            out.push_str(PAD);
        }
        rest = &rest[pos + PAD.len()..];
    }
    if rest.contains(PAD) {
        return Err(format!(
            "模板的 <|image_pad|> 多于图像数 {}",
            images.len()
        ));
    }
    out.push_str(rest);
    Ok(out)
}

/// **原始请求 JSON 落盘**(显式诊断开关；每请求一份，便于前端联调复现)。
///
/// 文件名用纳秒 + pid + attempt；`create_new` 防并发覆盖。落盘失败只 WARN，
/// 不影响请求本身（诊断面不能把主路径打死）。
fn dump_request(dir: &std::path::Path, body: &[u8]) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    for attempt in 0..100u32 {
        let path = dir.join(format!("req-{nanos}-{pid}-{attempt}.json"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                if let Err(e) = f.write_all(body) {
                    owl_shared::log_warn!("request-dump", "写入失败 {}: {e}", path.display());
                } else {
                    owl_shared::log_debug!("request-dump", "落盘 {} ({}B)", path.display(), body.len());
                }
                return;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                owl_shared::log_warn!("request-dump", "创建失败 {}: {e}", path.display());
                return;
            }
        }
    }
    owl_shared::log_warn!("request-dump", "同一纳秒 100 次文件名冲突，放弃落盘");
}

/// chat completions:提交 actor → 首笔 Accepted/Rejected → 三通道折叠循环。
pub(crate) async fn chat<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    body: &[u8],
    tx: mpsc::Sender<EngineReq>,
    model_name: &str,
    tpl: &Arc<ChatTpl>,
    agent: &AgentCfg,
    // `/debug/agent` 诊断环(G5②):每轮必落一条(`TurnLedger` 的 Drop 是兜底网)
    trace: &Arc<AgentTrace>,
    // **会话闸**(B5.1):同会话串行(引擎在 max_batch>1 时对同会话并发硬拒)
    gates: &Arc<crate::session_gate::SessionGates>,
    // **并发 permit**(P0):必须在会话闸**之后**取 —— 见 `serve()` 的注释
    gate: &Arc<tokio::sync::Semaphore>,
    // 原始请求 JSON 落盘目录（None = 关）
    request_dump_dir: Option<&std::path::Path>,
) -> std::io::Result<()> {
    if let Some(dir) = request_dump_dir {
        dump_request(dir, body);
    }
    let mut req: ChatRequest = match serde_json::from_slice::<ChatRequest>(body) {
        Ok(r) => r,
        Err(e) => {
            return http::respond_json(w, 400, &error_json(format!("请求体非法: {e}"))).await
        }
    };
    if let Err(e) = req
        .validate()
        .and_then(|_| req.stop_strings().map(|_| ()))
        .and_then(|_| req.normalize_tool_calls())
    {
        return http::respond_json(w, 400, &error_json(e)).await;
    }
    // 图片删除的文本边界规范化(渲染前;plain string / 单 text 数组零变化)。
    let _ = req.normalize_multipart_text();
    for wmsg in req.warnings() {
        owl_shared::log_warn!("chat", "收下但暂未生效(AGT-06 后仅余未知字段):{wmsg}");
    }
    let stop = match req.stop_strings() {
        Ok(s) => s,
        Err(e) => return http::respond_json(w, 400, &error_json(e)).await,
    };
    let mode = if req.stream { ReplyMode::Stream } else { ReplyMode::Aggregate };
    let model_name = req.model.as_deref().unwrap_or(model_name);
    let (max_new, clamped) = req.max_new(agent);
    if clamped {
        owl_shared::log_warn!(
            "chat",
            "max_tokens={:?} 超出本档上限 {},已夹紧(改 [agent] max_new_cap 随池定档放开)",
            req.max_tokens,
            agent.max_new_cap
        );
    }
    // 思考档位(AGT-07):OpenAI `reasoning_effort` **直通模板变量**
    // (模板自持值域: none|low|medium|high|xhigh|…)。请求 > 配置缺省
    let kw = req.effective_template_kwargs();
    let kw_ref = if kw.is_empty() { None } else { Some(&kw) };
    // 渲染:唯一权威 = 快照模板(§4.3 R2);模板拒(如 system 含图像)= 400
    let mut prompt = match tpl.render(RenderArgs {
        messages: &req.messages,
        tools: req.tools.as_deref(),
        add_generation_prompt: true,
        template_kwargs: kw_ref,
    }) {
        Ok(p) => p,
        Err(e) => return http::respond_json(w, 400, &error_json(e.to_string())).await,
    };
    // 图像:预处理(与模板顺序一致)+ **占位 token 展开**:
    // 模板对每张图只发一个 `<|image_pad|>`;视觉塔要求 `grid_h*grid_w/4` 个。
    let images = match req.images() {
        Ok(v) => v,
        Err(e) => return http::respond_json(w, 400, &error_json(e)).await,
    };
    if !images.is_empty() {
        prompt = match expand_image_pads(&prompt, &images) {
            Ok(p) => p,
            Err(e) => return http::respond_json(w, 400, &error_json(e)).await,
        };
    }
    let (sink_tx, mut sink_rx) = mpsc::channel::<SinkEvent>(SINK_CAPACITY);
    // 请求级图像事实（日志/诊断；图像 token = 视觉占位展开量的真源）。
    let image_count = images.len();
    let image_tokens = images
        .iter()
        .map(|i| (i.image.grid.1 * i.image.grid.2) / 4)
        .sum::<usize>();
    let evict_images = images.iter().filter(|i| i.evict).count();
    // 会话身份:显式 `session` 键优先;缺省按 [agent] implicit_session 落到
    // 隐式会话 0(AGT-09:pi 每轮全量重发历史 ⇒ 前缀增量命中;守卫失配
    // 自回退全量重算,正确性不依赖客户端)
    let session = match req.session.as_deref().map(session_id) {
        Some(id) => Some(id),
        None if agent.implicit_session => Some(IMPLICIT_SESSION),
        None => None,
    };
    // **会话闸**(B5.1 服务面契约):同一会话在引擎里**只许一个 turn**
    // (引擎 `submit_session_opts` 在 max_batch>1 时对同会话并发**硬拒** ——
    // 共享 KV/GDN = 结构性互踩,调度修不了)。闸覆盖 提交 → 终态 → 响应收尾;
    // 跨会话不受影响(那正是合批的原料)。无会话(临时)= 不需要闸。
    let _session_guard = match session {
        Some(id) => Some(gates.acquire(id).await),
        None => None,
    };
    // **并发位**：闸之后再占（同会话排队不占 permit；跨会话才竞争）
    let _permit = gate.acquire().await;
    // G5②:诊断环开账(此后的每一条返回路径都可见 —— 显式 fail 或 Drop 兜底)
    let mut led = TurnLedger::new(trace);
    led.set_req(ReqFacts {
        model: model_name.to_string(),
        stream: mode == ReplyMode::Stream,
        messages: req.messages.len(),
        roles: req
            .messages
            .iter()
            .filter_map(|m| m.get("role").and_then(|r| r.as_str()))
            .collect::<Vec<_>>()
            .join(","),
        tools: req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
        images: image_count,
        image_tokens,
        evict_images,
        thinking: req.enable_thinking(),
        reasoning_effort: req.reasoning_effort.clone(),
        session,
        max_new,
        ignore_eos: req.ignore_eos,
        stop: stop.len(),
    });
    led.set_prompt(&prompt);
    let engine_req = EngineReq::Chat {
        session,
        prompt,
        max_new,
        opts: TurnOpts {
            ignore_eos: req.ignore_eos,
            sampler: req.sampler_override(),
            stop,
            seed: req.seed.map(|s| s as u64),
            // 直接显存地板(**per-turn**,设计卷 §2.4/§23.5):OpenAI 面暂无
            // 扩展字段 —— 走引擎 `knobs.dense_floor`(缺省 `Strict`)。
            // 将来加请求字段时在这里映射;非 `Strict` 由引擎**响亮拒**
            // (选择层 B 轴未落地,地板语义无从兼现)。
            dense_floor: None,
            images,
        },
        sink: sink_tx,
    };
    if tx.send(engine_req).await.is_err() {
        return http::respond_json(w, 503, &error_json("引擎退役")).await;
    }

    // 首笔:受理确认 / 提交拒绝(预算越界等,submit 边界落地)
    let mut usage = Usage::default();
    // 预算被夹紧 ⇒ 响应头如实带出（`X-Owl-Max-New`；非私有 body 字段）
    let mut clamped_max_new: Option<usize> = None;
    let turn = match sink_rx.recv().await {
        Some(SinkEvent::Accepted { session, turn, prompt_tokens, cached_tokens, max_new }) => {
            usage.prompt = prompt_tokens;
            usage.cached = cached_tokens;
            // 预算被准入夹紧 ⇒ 如实留痕（客户端会看到 finish_reason=length）
            if max_new < req.max_new(agent).0 {
                clamped_max_new = Some(max_new);
                owl_shared::log_warn!(
                    "chat",
                    "生成预算被夹紧:请求 {} → 实际 {max_new}(session {session:#x} turn {turn})",
                    req.max_new(agent).0
                );
            }
            owl_shared::log_info!(
                "chat",
                "受理 session {session:#x} turn {turn}(stream={};tools={};images={image_count}/tokens={image_tokens}/evict={evict_images};think={};prompt={prompt_tokens};cached={cached_tokens};max_new={max_new})",
                mode == ReplyMode::Stream,
                req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                req.enable_thinking(),
            );
            Some((session, turn))
        }
        Some(SinkEvent::Rejected { err }) => {
            led.fail(format!("提交被拒:{err}"));
            // **容量拒绝**（引擎 submit 边界的 `admission:<code>`）⇒ 429；
            // 其余（预算越界）仍走 pi 能认的 400 context-overflow 报文。
            if let Some(code) = admission_code(&err) {
                let retry = retry_after_secs(&err).to_string();
                return http::respond_json_hdr(
                    w,
                    429,
                    &error_json_rate_limit(code, "服务端容量不足（排队已满）—— 请按 Retry-After 退避重试"),
                    &[("Retry-After", retry.as_str())],
                )
                .await;
            }
            return http::respond_json(w, 400, &error_json_for_reject(err)).await
        }
        // 契约:首笔必为 Accepted/Rejected(actor 在 pump 前投递)⇒ 不会走到
        // 这里(真走到 = 协议序错乱,不静默吞)
        Some(SinkEvent::Session { .. }) => {
            return http::respond_json(
                w,
                503,
                &error_json("协议序错乱:缓存校正帧先于受理确认"),
            )
            .await
        }
        _ => return http::respond_json(w, 503, &error_json("引擎退役")).await,
    };

    let id = completion_id();
    let mut splitter = ChannelSplitter::new(
        req.enable_thinking(),
        ToolSchemas::from_tools(req.tools.as_deref().unwrap_or(&[])),
    )
    .with_tool_parsing(agent.parse_tool_calls);
    let mut acc = Agg {
        seed: unix().as_nanos() as u64,
        ..Default::default()
    };
    // role 首帧延到首个内容帧(2026-10-02:提前发会让测速端把 HTTP 往返
    // 当 TTFT,长 prompt 的 prefill 全被藏进首帧延迟)
    let mut role_sent = false;
    // 写失败 ⇒ 显式弃置引擎 turn(AGT-10):actor 侧虽会在下次投递失败时
    // 发现,但显式 abort 立刻停(长思考/长输出下差几十秒 GPU)。
    // 注:宏必须在 `acc`/`role_sent` 作用域内定义 —— macro_rules 卫生:
    // 宏体内标识符按**定义点**解析,定义在前就看不到(实测踩过)
    macro_rules! emit {
        ($items:expr) => {
            if let Err(e) = emit_items(w, mode, &id, model_name, &mut acc, &mut role_sent, $items)
                .await
            {
                // **先关 sink 再等回执**(2026-10-20 真机定谳):本任务若还持有
                // `sink_rx` 等 abort 回执,actor 的投递就一直成功 ⇒ 它**察觉不到**
                // 断流,会把整个 turn 跑完(实测:256 tok 的流断掉后仍烧满 5s+,
                // 回执超时、会话闸也被拖住)。`close()` 让 actor 下一笔投递失败
                // ⇒ 走既有 "断流 ⇒ 弃置"(≤1 采样步)。
                sink_rx.close();
                abort_turn(&tx, session, turn.map(|(_, t)| t)).await;
                return Err(e);
            }
        };
    }
    if mode == ReplyMode::Stream {
        match clamped_max_new {
            Some(m) => {
                let v = m.to_string();
                http::respond_sse_head_hdr(w, &[("X-Owl-Max-New", v.as_str())]).await?
            }
            None => http::respond_sse_head(w).await?,
        }
    }
    // **排队超时**(`[agent] queue_wait_ms`;0 = 关):提交后迟迟不见实质进展 ⇒
    // 取消该 turn + 429（D2 的 R 档）。`started` = 已见到 prefill/token/终态。
    let mut started = false;
    loop {
        let next = if !started && agent.queue_wait_ms > 0 {
            match tokio::time::timeout(
                std::time::Duration::from_millis(agent.queue_wait_ms as u64),
                sink_rx.recv(),
            )
            .await
            {
                Ok(v) => v,
                Err(_) => {
                    owl_shared::log_warn!(
                        "chat",
                        "排队超时 {}ms ⇒ 取消 turn 并回 429（session {:?}）",
                        agent.queue_wait_ms,
                        session
                    );
                    // 先关 sink（actor 下一笔投递失败 ⇒ 按断流弃置），再指名取消 + 等回执
                    sink_rx.close();
                    abort_turn(&tx, session, turn.map(|(_, t)| t)).await;
                    return respond_queue_timeout(w, mode).await;
                }
            }
        } else {
            sink_rx.recv().await
        };
        if matches!(
            next,
            Some(SinkEvent::Turn(
                TurnEvent::Prefill { .. }
                    | TurnEvent::Token { .. }
                    | TurnEvent::Completed { .. }
                    | TurnEvent::Failed { .. }
            ))
        ) {
            started = true;
        }
        match next {
            Some(SinkEvent::Turn(TurnEvent::Token { delta, .. })) => {
                usage.completion += 1;
                led.first_token(); // TTFT(幂等)
                let items = splitter.push(&delta);
                emit!(items);
            }
            // Prefill 进度帧(末笔 total = prompt token 数,入 usage)
            Some(SinkEvent::Turn(TurnEvent::Prefill { total, .. })) => {
                usage.prompt = total;
            }
            // 缓存校正帧(AGT-12):块级前缀缓存命中的真值(覆盖 submit 时估值)
            Some(SinkEvent::Session { turn: t, cached_tokens }) => {
                if turn.map(|(_, x)| x) == Some(t) && cached_tokens != usage.cached {
                    owl_shared::log_info!(
                        "chat",
                        "turn {t} 前缀命中校正 {}(submit 时估值 {})",
                        cached_tokens,
                        usage.cached
                    );
                    usage.cached = cached_tokens;
                }
            }
            Some(SinkEvent::Turn(TurnEvent::Completed {
                text, finish, stats, ..
            })) => {
                let items = splitter.finish();
                emit!(items);
                // 思考相位真账(在 finish 之后取:本 turn 再无 push)
                usage.reasoning = splitter.reasoning_steps();
                // 有工具调用 ⇒ finish_reason = tool_calls(pi 的 mapStopReason 只认
                // stop/length/tool_calls 三者);strays 非零 = 值得一条日志
                if splitter.strays() > 0 {
                    owl_shared::log_warn!(
                        "chat",
                        "turn {:?} 出现 {} 个游标标记(已吞掉不泄漏)",
                        turn.map(|(_, t)| t),
                        splitter.strays()
                    );
                }
                let reason = if splitter.tool_calls() > 0 {
                    "tool_calls"
                } else {
                    match finish {
                        owl_engine::turn::FinishReason::Stop => "stop",
                        owl_engine::turn::FinishReason::Length => "length",
                    }
                };
                // **单轮诊断主日志**(2026-10-25):TTFT / prefill / decode /
                // cache / spec AL 同一条,服务面日志不再只有"完成"。
                // TTFT 含排队与 HTTP;prefill/decode 来自引擎侧 TurnStats
                // (起跑→首 token / 首 token→终态),两者口径分开。
                let ttft_ms = led.ttft_ms().unwrap_or(0);
                let cache_pct = if usage.prompt > 0 {
                    usage.cached as f64 * 100.0 / usage.prompt as f64
                } else {
                    0.0
                };
                owl_shared::log_info!(
                    "chat",
                    "turn {:?} 指标: ttft={}ms; prefill={}ms/{}tok={:.1}t/s; decode={}ms/{}tok={:.1}t/s; cache={}/{} ({:.1}%); spec rounds={} accepted={} al={:.2}",
                    turn.map(|(_, t)| t),
                    ttft_ms,
                    stats.prefill_ms,
                    stats.prefill_tokens,
                    stats.prefill_tps(),
                    stats.decode_ms,
                    stats.decode_tokens,
                    stats.decode_tps(),
                    usage.cached,
                    usage.prompt,
                    cache_pct,
                    stats.spec_rounds,
                    stats.spec_accepted,
                    stats.spec_al(),
                );
                // G5②:结算诊断环(结构轨迹 + usage + 结局;与 wire 同一份账)
                led.set_events(splitter.structural_trace());
                led.commit(
                    UsageFacts {
                        prompt: usage.prompt,
                        completion: usage.completion,
                        cached: usage.cached,
                        reasoning: usage.reasoning,
                    },
                    OutcomeFacts {
                        finish: Some(reason),
                        tools_called: splitter.tool_calls(),
                        strays: splitter.strays(),
                        malformed: splitter.malformed(),
                        prefill_ms: stats.prefill_ms,
                        decode_ms: stats.decode_ms,
                        prefill_tps: stats.prefill_tps(),
                        decode_tps: stats.decode_tps(),
                        spec_rounds: stats.spec_rounds,
                        spec_accepted: stats.spec_accepted,
                        spec_al: stats.spec_al(),
                        ..Default::default()
                    },
                );
                return match mode {
                    ReplyMode::Stream => {
                        let chunk = chunk_json(&id, model_name, json!({}), Some(reason));
                        http::sse_data(w, &chunk.to_string()).await?;
                        // OpenAI 约定:请求 include_usage 时,末帧(DONE 之前)
                        // 发 choices=[] + usage 真账
                        if req.stream_options.as_ref().is_some_and(|o| o.include_usage) {
                            let mut u = chunk_json(&id, model_name, json!({}), None);
                            u["choices"] = json!([]);
                            u["usage"] = json!({
                                "prompt_tokens": usage.prompt,
                                "completion_tokens": usage.completion,
                                "total_tokens": usage.prompt + usage.completion,
                                "prompt_tokens_details": {"cached_tokens": usage.cached},
                                "completion_tokens_details": {"reasoning_tokens": usage.reasoning},
                            });
                            http::sse_data(w, &u.to_string()).await?;
                        }
                        http::sse_data(w, "[DONE]").await?;
                        // chunked 终止块(不调 = 复用连接上的客户端永远等下一 chunk)
                        http::sse_end(w).await?;
                        owl_shared::log_info!(
                            "chat",
                            "turn {:?} 完成({reason};prompt={};cached={};completion={};tools={};think={}B/text={}B):{}",
                            turn.map(|(_, t)| t),
                            usage.prompt,
                            usage.cached,
                            usage.completion,
                            splitter.tool_calls(),
                            acc.reason.len(),
                            acc.text.len(),
                            preview(&text)
                        );
                        Ok(())
                    }
                    ReplyMode::Aggregate => {
                        let msg = agg_message(&acc, splitter.tool_calls());
                        owl_shared::log_info!(
                            "chat",
                            "turn {:?} 完成({reason};prompt={};cached={};completion={};tools={};think={}B/text={}B)",
                            turn.map(|(_, t)| t),
                            usage.prompt,
                            usage.cached,
                            usage.completion,
                            splitter.tool_calls(),
                            acc.reason.len(),
                            acc.text.len()
                        );
                        let body = completion_json(
                            &id,
                            model_name,
                            &msg,
                            reason,
                            usage.prompt,
                            usage.completion,
                            usage.cached,
                            usage.reasoning,
                        );
                        match clamped_max_new {
                            Some(m) => {
                                let v = m.to_string();
                                http::respond_json_hdr(
                                    w,
                                    200,
                                    &body,
                                    &[("X-Owl-Max-New", v.as_str())],
                                )
                                .await
                            }
                            None => http::respond_json(w, 200, &body).await,
                        }
                    }
                };
            }
            Some(SinkEvent::Turn(TurnEvent::Failed { err, .. })) => {
                led.fail(format!("turn 失败:{err}"));
                // **准入拒绝**(`admission:<code>`)：用 OpenAI 协议内的标准手段
                // 表达 —— 非流式 `429 + Retry-After`;流式头已发,走既有错误帧收口
                // (不自造 SSE 事件类型)。
                if let Some(code) = admission_code(&err) {
                    let retry = retry_after_secs(&err).to_string();
                    let msg = format!(
                        "服务端容量不足（{code}）—— 在跑请求优先;请按 Retry-After 退避重试"
                    );
                    return match mode {
                        ReplyMode::Aggregate => {
                            http::respond_json_hdr(
                                w,
                                429,
                                &error_json_rate_limit(code, &msg),
                                &[("Retry-After", retry.as_str())],
                            )
                            .await
                        }
                        ReplyMode::Stream => {
                            if let Err(e) = http::sse_data(w, &error_json(msg).to_string()).await {
                                owl_shared::log_info!("chat", "准入错误帧写失败(客户端可能已断开): {e}");
                            }
                            if let Err(e) = http::sse_data(w, "[DONE]").await {
                                owl_shared::log_info!("chat", "准入收尾帧写失败(客户端可能已断开): {e}");
                            }
                            if let Err(e) = http::sse_end(w).await {
                                owl_shared::log_info!("chat", "准入终止块写失败(客户端可能已断开): {e}");
                            }
                            Ok(())
                        }
                    };
                }
                return match mode {
                    // SSE 中途错误:协议无标准错误帧,error json + DONE 收口
                    ReplyMode::Stream => {
                        // 客户端可能已断开(写失败属正常),但不许静默(律 #1)
                        if let Err(e) = http::sse_data(w, &error_json(err).to_string()).await {
                            owl_shared::log_info!("chat", "流式错误帧写失败(客户端可能已断开): {e}");
                        }
                        if let Err(e) = http::sse_data(w, "[DONE]").await {
                            owl_shared::log_info!("chat", "流式收尾帧写失败(客户端可能已断开): {e}");
                        }
                        // 错误流也要收口(chunked 终止块;失败 = 客户端已走)
                        if let Err(e) = http::sse_end(w).await {
                            owl_shared::log_info!("chat", "流式终止块写失败(客户端可能已断开): {e}");
                        }
                        Ok(())
                    }
                    ReplyMode::Aggregate => http::respond_json(w, 500, &error_json(err)).await,
                };
            }
            Some(SinkEvent::Turn(_)) => {}
            // 协议不变式:首笔之后不再有 Accepted/Rejected(与断流同处置)
            Some(SinkEvent::Accepted { .. }) | Some(SinkEvent::Rejected { .. })
                if mode == ReplyMode::Stream =>
            {
                led.fail("协议序错乱:受理帧之后再现受理/拒绝帧");
                break
            }
            Some(SinkEvent::Accepted { .. }) | Some(SinkEvent::Rejected { .. }) => {
                led.fail("引擎连接中断(受理帧之后)");
                return http::respond_json(w, 503, &error_json("引擎连接中断")).await;
            }
            None if mode == ReplyMode::Stream => {
                led.fail("客户端断开(流式;引擎 turn 由 actor 侧弃置)");
                break
            }
            None => {
                led.fail("引擎连接中断(通道关闭)");
                return http::respond_json(w, 503, &error_json("引擎连接中断")).await;
            }
        }
    }
    // 环外到达 = 流式被 break(协议序错乱 / 客户端断开):补终止块,让复用连接
    // 上的下一个请求有干净帧界(写失败 = 客户端已走,记一条但不改结局)
    if mode == ReplyMode::Stream {
        if let Err(e) = http::sse_end(w).await {
            owl_shared::log_debug!("chat", "SSE 终止块写失败(客户端可能已断开): {e}");
        }
    }
    Ok(())
}

/// **429 错误体**（OpenAI 错误 schema：`type = rate_limit_error` + 稳定 `code`）。
/// 准入拒绝用它 —— 客户端按 `Retry-After` 退避，不需要任何私有字段。
pub fn error_json_rate_limit(code: &str, msg: &str) -> Value {
    json!({
        "error": {
            "message": msg,
            "type": "rate_limit_error",
            "param": Value::Null,
            "code": code,
        }
    })
}

/// **排队超时 → 429**（`[agent] queue_wait_ms`;D2 的 R 档）。
///
/// 非流式 = `429 + Retry-After`;流式头已发 ⇒ 既有错误帧收口（不自造事件）。
async fn respond_queue_timeout<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    mode: ReplyMode,
) -> std::io::Result<()> {
    let msg = "排队超时:服务端容量不足 —— 请按 Retry-After 退避重试";
    let body = error_json_rate_limit("queue_timeout", msg);
    match mode {
        ReplyMode::Aggregate => {
            http::respond_json_hdr(w, 429, &body, &[("Retry-After", "5")]).await
        }
        ReplyMode::Stream => {
            if let Err(e) = http::sse_data(w, &body.to_string()).await {
                owl_shared::log_info!("chat", "排队超时帧写失败(客户端可能已断开): {e}");
            }
            if let Err(e) = http::sse_data(w, "[DONE]").await {
                owl_shared::log_info!("chat", "排队超时收尾帧写失败(客户端可能已断开): {e}");
            }
            http::sse_end(w).await
        }
    }
}

/// **引擎给出的退避建议**（`admission:<code> retry_ms=<n>`；无则 `None`）
pub fn admission_retry_ms(err: &str) -> Option<u64> {
    err.split_whitespace()
        .find_map(|w| w.strip_prefix("retry_ms="))
        .and_then(|v| v.parse::<u64>().ok())
}

/// 退避秒数 = `ceil(ms/1000)`，夹在 1..=30（客户端不该被估成分钟级）
pub fn retry_after_secs(err: &str) -> u64 {
    admission_retry_ms(err)
        .map(|ms| ms.div_ceil(1000).clamp(1, 30))
        .unwrap_or(5)
}

/// **准入拒绝的错误码**（引擎 `Failed{err:"admission:<code>"}` 的协议面投影）。
///
/// 单源：服务面只认这个前缀 —— 不新造 SSE 事件类型、不加私有字段；
/// 非流式翻 `429 + Retry-After`（OpenAI 协议内的标准手段），流式走既有
/// 错误帧收口。空码/非法码 = `None`（不猜，走普通错误路径）。
pub fn admission_code(err: &str) -> Option<&str> {
    // 只取**第一个词**作为码（引擎的错误串是 `admission:<code> —— 人类可读细节`；
    // 把整段当 `code` 会污染客户端可见的协议字段）。
    err.strip_prefix("admission:")
        .map(|rest| rest.trim())
        .and_then(|rest| rest.split_whitespace().next())
        .filter(|c| !c.is_empty())
}

/// 客户端写失败 ⇒ 请求引擎弃置**该 turn**(AGT-10;B5.1 起必须指名)。
///
/// 语义要点(合批档):
/// - `turn` 已知(Accepted 之后)= 精确取消该 turn —— **不许**走"弃当前执行 turn"
///   (那会杀掉同批别家会话的 turn);
/// - `turn` 未知(断在受理帧之前)= 按会话找(会话闸保证该会话至多一个 turn);
/// - **等 actor 回执再返回**:会话闸在本函数返回后才释放(函数尾),这样下一次
///   同会话提交不会撞上引擎的"同会话并发"硬拒。回执超时只告警(不静默)。
async fn abort_turn(tx: &mpsc::Sender<EngineReq>, session: Option<u64>, turn: Option<u64>) {
    // 临时会话(None)= 无键可寻;引擎会在 sink drop 后自弃(actor 下一个投递点)
    let Some(session) = session else { return };
    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
    if tx.send(EngineReq::Abort { session, turn, ack: ack_tx }).await.is_err() {
        owl_shared::log_warn!("chat", "abort 投递失败(actor 已退役,引擎随进程退出)");
        return;
    }
    match tokio::time::timeout(std::time::Duration::from_secs(5), ack_rx).await {
        Ok(Ok(())) => {}
        Ok(Err(_)) => {
            owl_shared::log_debug!("chat", "abort 回执通道关闭(actor 已退役)");
        }
        Err(_) => owl_shared::log_warn!("chat", "abort 回执超时(5s;引擎可能正忙) "),
    }
}

/// 三通道产出 → SSE 帧(流式)或聚合器(非流式);role 首帧在此处懒发
async fn emit_items<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    mode: ReplyMode,
    id: &str,
    model: &str,
    acc: &mut Agg,
    role_sent: &mut bool,
    items: Vec<Item>,
) -> std::io::Result<()> {
    for it in items {
        match it {
            Item::Delta { chan, text } => {
                if mode == ReplyMode::Stream {
                    if !*role_sent {
                        *role_sent = true;
                        let first =
                            chunk_json(id, model, json!({"role": "assistant", "content": ""}), None);
                        http::sse_data(w, &first.to_string()).await?;
                    }
                    let delta = match chan {
                        Chan::Think => json!({"reasoning_content": text}),
                        Chan::Text => json!({"content": text}),
                    };
                    let chunk = chunk_json(id, model, delta, None);
                    http::sse_data(w, &chunk.to_string()).await?;
                } else {
                    match chan {
                        Chan::Think => acc.reason.push_str(&text),
                        Chan::Text => acc.text.push_str(&text),
                    }
                }
            }
            Item::ToolCall { index, name, arguments } => {
                let cid = acc.tool_id(index);
                if mode == ReplyMode::Stream {
                    if !*role_sent {
                        *role_sent = true;
                        let first =
                            chunk_json(id, model, json!({"role": "assistant", "content": ""}), None);
                        http::sse_data(w, &first.to_string()).await?;
                    }
                    // pi 按 `index` 归并工具块(id 只作标识);一次性给全 name+arguments
                    // —— arguments 恒为合法 JSON 字符串(§3.3 约束 1)
                    let delta = json!({"tool_calls": [{
                        "index": index,
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }]});
                    let chunk = chunk_json(id, model, delta, None);
                    http::sse_data(w, &chunk.to_string()).await?;
                } else {
                    acc.tools.push((cid, name, arguments));
                }
            }
            // 畸形工具块:不发射工具调用(§3.3 约束 4),但**不静默**
            Item::Malformed { kind, raw } => {
                owl_shared::log_warn!(
                    "chat",
                    "工具块畸形({kind};{}B)已丢弃:{}",
                    raw.len(),
                    preview(&raw)
                );
            }
        }
    }
    Ok(())
}

// ============================================================================
// §3 `/tokenize`(G5③;契约见《设计》§3.7)
// ============================================================================

/// `/tokenize` 请求体(**最小面**)。
///
/// 与 `/v1/chat/completions` 不同,本端点 `deny_unknown_fields` **开着**:
/// 它是 owl 自有面(无 pi 兼容史要背),多传字段 = 调用方预期落空 ⇒ 响亮拒。
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizeRequest {
    /// 回显用(不校验与本进程模型名一致 —— 单模型服务)
    model: Option<String>,
    /// 待编码文本(**必填**;本端点不做 messages 渲染)
    prompt: Option<String>,
    /// `raw`(缺省)= 纯文本编码;`chat` = 先套家族单轮包装
    mode: Option<String>,
    /// vLLM 同名字段:**只接 false/缺省** —— true = 响亮拒
    /// (本引擎编码恒 `add_special_tokens=false`;要家族包装用 `mode="chat"`)
    add_special_tokens: Option<bool>,
}

/// 请求 → `(text, chat_wrap, mode)` 的**纯校验**(不碰网络/引擎;可单测)
fn tokenize_plan(req: &TokenizeRequest) -> Result<(String, bool, &'static str), String> {
    let text = match req.prompt.as_deref() {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => {
            return Err(
                "prompt 必填且非空(messages 渲染未做 —— 需要时先自己渲染再送本端点)".into(),
            )
        }
    };
    let mode = req.mode.as_deref().unwrap_or("raw");
    let (chat_wrap, mode_canon) = match mode {
        "raw" => (false, "raw"),
        // 家族单轮包装(与裸文本提交同一律):问“这轮用户内容实际花多少 token”
        "chat" => (true, "chat"),
        other => return Err(format!("mode 只接 raw|chat,得到 {other:?}")),
    };
    if req.add_special_tokens == Some(true) {
        return Err(
            "add_special_tokens=true 未支持(本引擎编码恒不加特殊 token;要家族单轮包装用 mode=\"chat\")"
                .into(),
        );
    }
    Ok((text, chat_wrap, mode_canon))
}

/// `POST /tokenize`:校验 → 旁路引擎 tokenize → 回兼容形状的 JSON。
///
/// ⚠️ **延迟说明**:引擎 actor 单线程且“收件与泵不相交” ⇒ 本请求排在当前
/// turn 之后(生成期间调用会等到本轮结束)。毫秒级 CPU 工,单源真理优先。
pub(crate) async fn tokenize<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    body: &[u8],
    tx: mpsc::Sender<EngineReq>,
) -> std::io::Result<()> {
    let req: TokenizeRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return http::respond_json(w, 400, &error_json(format!("请求体非法: {e}"))).await,
    };
    let (text, chat_wrap, mode) = match tokenize_plan(&req) {
        Ok(v) => v,
        Err(e) => return http::respond_json(w, 400, &error_json(e)).await,
    };
    let (rtx, rrx) = tokio::sync::oneshot::channel();
    if tx.send(EngineReq::Tokenize { text, chat_wrap, reply: rtx }).await.is_err() {
        return http::respond_json(w, 503, &error_json("引擎退役")).await;
    }
    match rrx.await {
        Ok(Ok(rep)) => {
            http::respond_json(
                w,
                200,
                &json!({
                    "count": rep.tokens.len(),
                    "max_model_len": rep.max_seq_tokens,
                    "tokens": rep.tokens,
                    "mode": mode,
                    "model": req.model,
                }),
            )
            .await
        }
        // 编码失败(显式上抛;不混淆“文本是空的”)
        Ok(Err(e)) => http::respond_json(w, 400, &error_json(e)).await,
        Err(_) => {
            http::respond_json(w, 503, &error_json("引擎连接中断(tokenize 应答通道关闭)"))
                .await
        }
    }
}

/// 非流式 message(三通道聚合;pi 的 AssistantMessage 同形)
fn agg_message(acc: &Agg, tool_calls: usize) -> Value {
    let mut msg = json!({"role": "assistant", "content": acc.text});
    if !acc.reason.is_empty() {
        msg["reasoning_content"] = json!(acc.reason);
    }
    if tool_calls > 0 {
        let tcs: Vec<Value> = acc
            .tools
            .iter()
            .map(|(id, name, args)| {
                json!({"id": id, "type": "function",
                       "function": {"name": name, "arguments": args}})
            })
            .collect();
        msg["tool_calls"] = json!(tcs);
    }
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(msgs: Value) -> ChatRequest {
        serde_json::from_value(json!({"messages": msgs})).expect("parse")
    }

    /// **可删标记只认布尔**(前缀无感删除):非布尔 = 400,不许静默当真/当假
    #[test]
    fn owl_evict_must_be_bool() {
        let base = json!({"role":"user","content":[
            {"type":"text","text":"hi"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}
        ]});
        let plain = req(json!([base.clone()]));
        assert!(plain.validate().is_ok(), "无标记应通过");
        let mut with_flag = base.clone();
        with_flag["content"][1]["owl_evict"] = json!(true);
        let flagged = req(json!([with_flag]));
        assert!(flagged.validate().is_ok(), "布尔标记应通过");
        let mut bad = base.clone();
        bad["content"][1]["owl_evict"] = json!("yes");
        let bad = req(json!([bad]));
        let e = bad.validate().expect_err("非布尔必须结构化拒");
        assert!(e.contains("owl_evict"), "错误必须点名字段(得 {e})");
    }

    /// vLLM 同款:字符串 arguments → 对象;dict/array 原样;空串 → {}
    #[test]
    fn normalize_string_arguments_to_object() {
        let mut r = req(json!([{"role":"assistant","content":"","tool_calls":[
            {"id":"c1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}},
            {"id":"c2","type":"function","function":{"name":"f2","arguments":""}},
            {"id":"c3","type":"function","function":{"name":"f3","arguments":{"a":1}}}
        ]}]));
        r.validate().unwrap();
        r.normalize_tool_calls().unwrap();
        assert_eq!(r.messages[0]["tool_calls"][0]["function"]["arguments"]["command"], "ls");
        assert_eq!(r.messages[0]["tool_calls"][1]["function"]["arguments"], json!({}));
        assert_eq!(r.messages[0]["tool_calls"][2]["function"]["arguments"], json!({"a": 1}));
    }

    /// 空 tool_calls 数组剔除(模板回普通 assistant 路径)
    #[test]
    fn normalize_drops_empty_tool_calls() {
        let mut r = req(json!([{"role":"assistant","content":"hi","tool_calls":[]}]));
        r.validate().unwrap();
        r.normalize_tool_calls().unwrap();
        assert!(r.messages[0].get("tool_calls").is_none(), "空数组应被剔除");
    }

    /// 坏 JSON / 非 function 类型 → 结构化拒绝(不静默)
    #[test]
    fn normalize_rejects_bad_shapes() {
        let r = req(json!([{"role":"assistant","content":"","tool_calls":[
            {"id":"c1","type":"function","function":{"name":"f","arguments":"{oops"}}]}]));
        assert!(r.validate().is_err(), "坏 JSON 应被 validate 拒");

        let mut r = req(json!([{"role":"assistant","content":"","tool_calls":[
            {"id":"c1","type":"custom","function":{"name":"f","arguments":{}}}]}]));
        assert!(r.normalize_tool_calls().is_err(), "非 function 类型应被拒");
    }

    /// 采样参数/未知字段 → 告警条目(AGT-06 后温度/停止串已真生效;
    /// 告警只余**未知字段**,不许静默丢弃)
    #[test]
    fn warnings_surface_only_unknown_fields() {
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}],
            "temperature": 0.7, "stop": ["x"], "seed": 1})).expect("parse");
        r.validate().unwrap();
        assert!(r.warnings().is_empty(), "已生效参数不该再告警:{:?}", r.warnings());
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}],
            "presence_penalty": 0.5})).expect("parse");
        assert_eq!(r.warnings().len(), 1, "未知字段必须可见");
    }

    /// AGT-06:采样覆写只带请求出现的字段(未带 = 继承引擎档位)
    #[test]
    fn sampler_override_partial() {
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}]})).expect("parse");
        assert!(r.sampler_override().is_none(), "未带采样参数 ⇒ 无覆写");
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}], "temperature": 0.0})).expect("parse");
        let o = r.sampler_override().expect("有覆写");
        assert_eq!(o.temp, Some(0.0));
        assert_eq!(o.topk, None);
        assert_eq!(o.topp, None);
    }

    /// AGT-06:stop 两形态(string / array)+ 非法形态结构化拒
    #[test]
    fn stop_strings_shapes() {
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}], "stop": "\n\n"})).expect("parse");
        assert_eq!(r.stop_strings().unwrap(), vec!["\n\n".to_string()]);
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}], "stop": ["a", "b"]})).expect("parse");
        assert_eq!(r.stop_strings().unwrap(), vec!["a".to_string(), "b".to_string()]);
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}], "stop": [1, 2]})).expect("parse");
        assert!(r.stop_strings().is_err(), "非字符串元素应被拒");
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"hi"}], "stop": 7})).expect("parse");
        assert!(r.stop_strings().is_err(), "非 string/array 应被拒");
    }

    /// max_tokens 夹紧(上限来自 [agent]) + enable_thinking 取值
    #[test]
    fn max_new_clamp_and_thinking_flag() {
        let agent = AgentCfg::default();
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}], "max_tokens": 32000})).expect("parse");
        assert_eq!(r.max_new(&agent), (agent.max_new_cap, true));
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}]})).expect("parse");
        assert_eq!(r.max_new(&agent), (agent.default_max_new, false));
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "chat_template_kwargs": {"enable_thinking": false}})).expect("parse");
        assert!(!r.enable_thinking());
        // `reasoning_effort=none/off` 与模板同口径：splitter 也必须按关思考启动
        for off in ["none", "off", "NONE", "Off"] {
            let r: ChatRequest = serde_json::from_value(json!({
                "messages":[{"role":"user","content":"x"}],
                "reasoning_effort": off})).expect("parse");
            assert!(!r.enable_thinking(), "reasoning_effort={off} 应关思考");
        }
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "high"})).expect("parse");
        assert!(r.enable_thinking(), "reasoning_effort=high 应开思考");
        // 显式 enable_thinking 覆盖 reasoning_effort
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": true}})).expect("parse");
        assert!(r.enable_thinking(), "显式 enable_thinking=true 应优先");

        // vLLM 同款归一化：reasoning_effort 必须同步派生 enable_thinking
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "none"})).expect("parse");
        let kw = r.effective_template_kwargs();
        assert_eq!(kw.get("reasoning_effort"), Some(&json!("none")));
        assert_eq!(kw.get("enable_thinking"), Some(&json!(false)));
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "high"})).expect("parse");
        assert_eq!(
            r.effective_template_kwargs().get("enable_thinking"),
            Some(&json!(true))
        );
        // 显式 enable_thinking 不被派生值覆盖(与 vLLM build_chat_params 一致)
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}],
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": true}})).expect("parse");
        assert_eq!(
            r.effective_template_kwargs().get("enable_thinking"),
            Some(&json!(true))
        );

        // [agent] 定制上限生效(配置面唯一旋钮)
        let agent = AgentCfg { max_new_cap: 32768, default_max_new: 2048, ..AgentCfg::default() };
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}], "max_tokens": 40000})).expect("parse");
        assert_eq!(r.max_new(&agent), (32768, true));
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}]})).expect("parse");
        assert_eq!(r.max_new(&agent), (2048, false));
    }

    /// **上下文溢出归一化**(wire 契约;设计 §3.6 挂账项)。
    ///
    /// pi 的 `isContextOverflow()` **只看报文文本**与它的 `OVERFLOW_PATTERNS`
    /// 做正则匹配(两轮门 `explicitOverflow`/`assistantUsageMatchesProjection`
    /// 都只要报文命中)⇒ 引擎的中文"超预算"必须带上词表里的规范短语,否则
    /// pi 把它当**普通 400** 原样展示、**不触发自动 compaction**。
    /// 词表 = pi-ai bundle 实测(2026-10-15;改短语就同步改这里 + 复核 bundle)。
    #[test]
    fn budget_reject_is_normalized_for_pi_compaction() {
        const PI_OVERFLOW_PATTERNS: &[&str] = &[
            "prompt (is )?too long",
            "prompt exceeds max length",
            "request_too_large",
            "input is too long for requested model",
            "exceeds the context window",
            "exceeds .* maximum context length",
        ];
        // ① 短语必须**逐字**在词表里
        assert!(
            PI_OVERFLOW_PATTERNS.contains(&OVERFLOW_PHRASE),
            "OVERFLOW_PHRASE {OVERFLOW_PHRASE:?} 不在 pi 词表里"
        );
        // ② 引擎原诊断保留 + code 落位(绝不静默改写)
        let raw = "submit: prompt 27584 + 生成 8192 超预算 32768";
        let body = error_json_for_reject(raw.to_string());
        let msg = body["error"]["message"].as_str().expect("message");
        assert!(msg.starts_with(raw), "原诊断须保留: {msg}");
        assert!(msg.contains(OVERFLOW_PHRASE), "{msg}");
        assert_eq!(body["error"]["code"], "context_length_exceeded");
        // ③ 非预算拒绝不动(无 code、报文原样)
        let plain = error_json_for_reject("messages 为空".into());
        assert!(plain["error"].get("code").is_none());
        assert_eq!(plain["error"]["message"], "messages 为空");
    }

    /// **G5③ `/tokenize` 纯校验面**:两模式 + 三类响亮拒(不静默回缺省)
    #[test]
    fn tokenize_plan_modes_and_rejects() {
        let plan = |j: &str| {
            let req: TokenizeRequest = serde_json::from_str(j).expect("请求体应可解析");
            tokenize_plan(&req)
        };
        assert_eq!(plan(r#"{"prompt":"你好"}"#).unwrap(), ("你好".into(), false, "raw"));
        assert_eq!(
            plan(r#"{"prompt":"你好","mode":"chat"}"#).unwrap(),
            ("你好".into(), true, "chat")
        );
        assert!(plan(r#"{"prompt":"你好","add_special_tokens":false}"#).is_ok(), "false 可接");
        assert!(plan(r#"{"prompt":""}"#).unwrap_err().contains("prompt 必填"), "空 prompt 拒");
        assert!(plan(r#"{"mode":"chat"}"#).unwrap_err().contains("prompt 必填"), "缺 prompt 拒");
        assert!(plan(r#"{"prompt":"x","mode":"tokens"}"#).unwrap_err().contains("raw|chat"));
        assert!(
            plan(r#"{"prompt":"x","add_special_tokens":true}"#)
                .unwrap_err()
                .contains("add_special_tokens=true 未支持")
        );
        // 未知字段:本端点开着 deny_unknown_fields(自有面,不背兼容史)
        assert!(serde_json::from_str::<TokenizeRequest>(r#"{"prompt":"x","nope":1}"#).is_err());
    }
}


#[cfg(test)]
mod admission_tests {
    use super::{admission_code, admission_retry_ms, retry_after_secs};

    #[test]
    fn admission_prefix_is_the_only_protocol_surface() {
        assert_eq!(admission_code("admission:no_memory"), Some("no_memory"));
        assert_eq!(admission_code("admission: no_slot "), Some("no_slot"));
        assert_eq!(admission_code("admission:"), None, "空码不许当拒绝语义");
        assert_eq!(admission_code("boom"), None);
        assert_eq!(admission_code("admission"), None, "缺冒号 = 普通错误");
    }

    /// 引擎的退避建议：`admission:<code> retry_ms=<n>` → 秒（1..=30）
    #[test]
    fn retry_after_comes_from_engine_estimate() {
        assert_eq!(admission_retry_ms("admission:no_memory retry_ms=12000"), Some(12_000));
        assert_eq!(retry_after_secs("admission:no_memory retry_ms=12000"), 12);
        assert_eq!(retry_after_secs("admission:no_memory retry_ms=200"), 1, "不足 1s 夹到 1");
        assert_eq!(retry_after_secs("admission:no_memory retry_ms=999999"), 30, "封顶 30");
        assert_eq!(retry_after_secs("admission:no_slot"), 5, "无建议值 = 5s 保守");
    }
}

#[cfg(test)]
mod multipart_text_tests {
    use super::ChatRequest;
    use serde_json::{json, Value};

    fn req(messages: Value) -> ChatRequest {
        serde_json::from_value(json!({"messages": messages})).expect("请求解析")
    }

    /// 最小影响面:plain string / 单 text 数组 / system / assistant 一律不动;
    /// 带图多段只给 text 补 `\n`,image 顺序与 owl_evict 原样;幂等。
    #[test]
    fn multipart_text_normalization_is_minimal_and_idempotent() {
        let mut r = req(json!([
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": "plain string"},
            {"role": "user", "content": [{"type": "text", "text": "single"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}, "owl_evict": true},
                {"type": "text", "text": "b\n"}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "asst"}]}
        ]));
        let changed = r.normalize_multipart_text();
        assert_eq!(changed, 1, "只有 a 需要补;b 已带换行");
        assert_eq!(r.messages[0]["content"][0]["text"], "sys", "system 不动");
        assert_eq!(r.messages[1]["content"], "plain string", "string 逐字节不动");
        assert_eq!(r.messages[2]["content"][0]["text"], "single", "单 text 数组不动");
        assert_eq!(r.messages[3]["content"][0]["text"], "a\n");
        assert_eq!(r.messages[3]["content"][2]["text"], "b\n");
        assert_eq!(r.messages[3]["content"][1]["owl_evict"], true, "标记保留");
        assert_eq!(
            r.messages[3]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AA",
            "image 原样"
        );
        assert_eq!(r.messages[4]["content"][0]["text"], "asst", "assistant 不动");
        let snapshot = r.messages.clone();
        assert_eq!(r.normalize_multipart_text(), 0, "幂等");
        assert_eq!(snapshot, r.messages, "第二次调用零变化");
    }

    /// 删图后的 user 消息形状:text1 + text2 相邻(无图)也必须补换行,
    /// 否则引擎的删点前缀/后缀无法对齐。
    #[test]
    fn adjacent_user_texts_after_image_removal_get_newlines() {
        let mut r = req(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "before"},
                {"type": "text", "text": "after"}
            ]}
        ]));
        assert_eq!(r.normalize_multipart_text(), 2);
        assert_eq!(r.messages[0]["content"][0]["text"], "before\n");
        assert_eq!(r.messages[0]["content"][1]["text"], "after\n");
    }

    /// 纯文本多段 content 的规范化是确定性的:同输入两次渲染结果一致。
    #[test]
    fn normalization_is_deterministic_for_cache() {
        let msg = json!([{"role": "user", "content": [
            {"type": "text", "text": "x"},
            {"type": "text", "text": "y"}
        ]}]);
        let mut a = req(msg.clone());
        let mut b = req(msg);
        a.normalize_multipart_text();
        b.normalize_multipart_text();
        assert_eq!(a.messages, b.messages);
    }
}
