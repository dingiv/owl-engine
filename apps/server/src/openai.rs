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

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

use owl_engine::{TurnEvent, TurnOpts, TurnSampler};
use owl_shared::config::AgentCfg;

use crate::actor::{EngineReq, SinkEvent};
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

    /// 开启思考?(模板契约:未给 = 模板缺省 true)
    pub fn enable_thinking(&self) -> bool {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.get("enable_thinking"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }
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

/// chat completions:提交 actor → 首笔 Accepted/Rejected → 三通道折叠循环。
pub(crate) async fn chat<W: tokio::io::AsyncWrite + Unpin>(
    w: &mut W,
    body: &[u8],
    tx: mpsc::Sender<EngineReq>,
    model_name: &str,
    tpl: &Arc<ChatTpl>,
    agent: &AgentCfg,
) -> std::io::Result<()> {
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
    let mut kw = req.chat_template_kwargs.clone().unwrap_or_default();
    if let Some(e) = &req.reasoning_effort {
        kw.insert("reasoning_effort".into(), json!(e));
    }
    let kw_ref = if kw.is_empty() { None } else { Some(&kw) };
    // 渲染:唯一权威 = 快照模板(§4.3 R2);模板拒(如 system 含图像)= 400
    let prompt = match tpl.render(RenderArgs {
        messages: &req.messages,
        tools: req.tools.as_deref(),
        add_generation_prompt: true,
        template_kwargs: kw_ref,
    }) {
        Ok(p) => p,
        Err(e) => return http::respond_json(w, 400, &error_json(e.to_string())).await,
    };
    let (sink_tx, mut sink_rx) = mpsc::channel::<SinkEvent>(SINK_CAPACITY);
    // 会话身份:显式 `session` 键优先;缺省按 [agent] implicit_session 落到
    // 隐式会话 0(AGT-09:pi 每轮全量重发历史 ⇒ 前缀增量命中;守卫失配
    // 自回退全量重算,正确性不依赖客户端)
    let session = match req.session.as_deref().map(session_id) {
        Some(id) => Some(id),
        None if agent.implicit_session => Some(IMPLICIT_SESSION),
        None => None,
    };
    let engine_req = EngineReq::Chat {
        session,
        prompt,
        max_new,
        opts: TurnOpts {
            ignore_eos: req.ignore_eos,
            sampler: req.sampler_override(),
            stop,
            seed: req.seed.map(|s| s as u64),
        },
        sink: sink_tx,
    };
    if tx.send(engine_req).await.is_err() {
        return http::respond_json(w, 503, &error_json("引擎退役")).await;
    }

    // 首笔:受理确认 / 提交拒绝(预算越界等,submit 边界落地)
    let mut usage = Usage::default();
    let turn = match sink_rx.recv().await {
        Some(SinkEvent::Accepted { session, turn, prompt_tokens, cached_tokens }) => {
            usage.prompt = prompt_tokens;
            usage.cached = cached_tokens;
            owl_shared::log_info!(
                "chat",
                "受理 session {session:#x} turn {turn}(stream={};tools={};think={};prompt={prompt_tokens};cached={cached_tokens})",
                mode == ReplyMode::Stream,
                req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
                req.enable_thinking(),
            );
            Some((session, turn))
        }
        Some(SinkEvent::Rejected { err }) => {
            return http::respond_json(w, 400, &error_json(err)).await
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
                abort_turn(&tx, turn.map(|(s, _)| s)).await;
                return Err(e);
            }
        };
    }
    if mode == ReplyMode::Stream {
        http::respond_sse_head(w).await?;
    }
    loop {
        match sink_rx.recv().await {
            Some(SinkEvent::Turn(TurnEvent::Token { delta, .. })) => {
                usage.completion += 1;
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
            Some(SinkEvent::Turn(TurnEvent::Completed { text, finish, .. })) => {
                let items = splitter.finish();
                emit!(items);
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
                            });
                            http::sse_data(w, &u.to_string()).await?;
                        }
                        http::sse_data(w, "[DONE]").await?;
                        owl_shared::log_info!(
                            "chat",
                            "turn {:?} 完成({reason};tools={};think={}B/text={}B):{}",
                            turn.map(|(_, t)| t),
                            splitter.tool_calls(),
                            acc.reason.len(),
                            acc.text.len(),
                            preview(&text)
                        );
                        Ok(())
                    }
                    ReplyMode::Aggregate => {
                        let msg = agg_message(&acc, splitter.tool_calls());
                        http::respond_json(
                            w,
                            200,
                            &completion_json(
                                &id,
                                model_name,
                                &msg,
                                reason,
                                usage.prompt,
                                usage.completion,
                                usage.cached,
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
                        // 客户端可能已断开(写失败属正常),但不许静默(律 #1)
                        if let Err(e) = http::sse_data(w, &error_json(err).to_string()).await {
                            owl_shared::log_info!("chat", "流式错误帧写失败(客户端可能已断开): {e}");
                        }
                        if let Err(e) = http::sse_data(w, "[DONE]").await {
                            owl_shared::log_info!("chat", "流式收尾帧写失败(客户端可能已断开): {e}");
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

/// 客户端写失败 ⇒ 请求引擎弃置该会话的活跃 turn(AGT-10)。
/// 投递失败本身不可恢复(actor 已退役),但**不许静默**。
async fn abort_turn(tx: &mpsc::Sender<EngineReq>, session: Option<u64>) {
    let Some(session) = session else { return };
    if tx.send(EngineReq::Abort { session }).await.is_err() {
        owl_shared::log_warn!("chat", "abort 投递失败(actor 已退役,引擎随进程退出)");
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
        // [agent] 定制上限生效(配置面唯一旋钮)
        let agent = AgentCfg { max_new_cap: 32768, default_max_new: 2048, ..AgentCfg::default() };
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}], "max_tokens": 40000})).expect("parse");
        assert_eq!(r.max_new(&agent), (32768, true));
        let r: ChatRequest = serde_json::from_value(json!({
            "messages":[{"role":"user","content":"x"}]})).expect("parse");
        assert_eq!(r.max_new(&agent), (2048, false));
    }
}
