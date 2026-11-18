//! **OpenAI 协议 DTO / wire 形状（domain）**。
//!
//! 请求校验、stop/sampler 映射、会话 id、JSON 响应构造、tokenize 请求校验，
//! 以及流式/聚合共享的 `ReplyMode`/`Usage`/`Agg`。纯数据/纯函数为主；
//! 不依赖 HTTP 帧，也不持有 `EngineClient`。
//!
//! 注：`dump_request`（原始请求落盘）暂住本文件，后续可下沉 `service`。

use std::io::Write;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use owl_engine::TurnSampler;

use crate::config::AgentCfg;

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
                    let is_image = part.get("type").and_then(|t| t.as_str()) == Some("image_url")
                        || part.get("image_url").is_some();
                    if is_image {
                        let url = part
                            .get("image_url")
                            .and_then(|v| v.get("url"))
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                format!("messages[{i}].content[{j}].image_url.url 缺失或非字符串")
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
                    let f = tc
                        .get("function")
                        .ok_or_else(|| format!("{fd}.function 缺失"))?;
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
                            return Err(format!("{fd}.function.arguments 必须是 JSON 字符串或对象"))
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
        if self
            .extra
            .get("logprobs")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
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
            out.push(format!(
                "未知字段 {:?}",
                self.extra.keys().collect::<Vec<_>>()
            ));
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
    /// 仅 data URL;`vision_processor` 只做 base64 解码 + 内容缓存（P5 起
    /// Qwen2VL 的 resize/patchify 在 `owl_models::vlm::preprocess`）。
    /// 无图像 = 空(纯文本逐字节旧行为)。
    ///
    /// **可删标记**(前缀无感删除;2026-10-23 用户裁决):图像 content part 上
    /// 的 `owl_evict: true` = 「这张图将来可能从历史里被删」。引擎**不自作
    /// 主张**:只有打了标 + `[vision] image_evict = true` 才补页对齐 pad 并
    /// 进删除候选;没打标 = 零行为差异。
    pub fn images(&self, max_pixels: usize) -> Result<Vec<owl_engine::TurnImage>, String> {
        let mut out = Vec::new();
        for (i, m) in self.messages.iter().enumerate() {
            let Some(arr) = m.get("content").and_then(|c| c.as_array()) else {
                continue;
            };
            for (j, part) in arr.iter().enumerate() {
                let is_image = part.get("type").and_then(|t| t.as_str()) == Some("image_url")
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
                let image = crate::vision_processor::from_data_url(url, max_pixels)
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
            return Err(format!(
                "stop 串个数 {} 超过上限 {MAX_STOP_SEQS}",
                list.len()
            ));
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
                let ty = tc
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("function");
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

    /// 生成步数(夹紧到 `agent.max_generation_length`;夹紧情况返回 true 供调用方告警)
    pub fn max_new(&self, agent: &AgentCfg) -> (usize, bool) {
        let want = self
            .max_tokens
            .filter(|v| *v > 0)
            .unwrap_or(agent.max_generation_length);
        (
            want.min(agent.max_generation_length),
            want > agent.max_generation_length,
        )
    }

    /// **模板 kwargs 的唯一组装点**(vLLM `build_chat_params` 同款)。
    ///
    /// OpenAI 面的 `reasoning_effort` 与 Qwen 模板面的 `enable_thinking`
    /// 是同一个状态的两个入口。只把前者塞进模板而不管后者 ⇒ 模板可能
    /// 关思考、`ChannelSplitter` 却按默认 true 启动，答案会被整段吞进
    /// `reasoning_content`。这里在请求层归一化：显式 `enable_thinking`
    /// 优先；否则 `reasoning_effort` 存在时同步派生。
    /// `defaults` = `chat_template.toml` 里的变量（2026-11-16 起模板变量
    /// 不在引擎配置里）；请求 kwargs 覆盖它。
    pub fn effective_template_kwargs(&self, defaults: &Map<String, Value>) -> Map<String, Value> {
        let mut kw = defaults.clone();
        if let Some(req) = &self.chat_template_kwargs {
            for (k, v) in req {
                kw.insert(k.clone(), v.clone());
            }
        }
        let req_enable = self
            .chat_template_kwargs
            .as_ref()
            .and_then(|k| k.get("enable_thinking"));
        if let Some(e) = &self.reasoning_effort {
            kw.insert("reasoning_effort".into(), json!(e));
            // 请求显式 `enable_thinking` 优先；否则由请求档位派生
            // （覆盖文件里的缺省 —— 请求 > 文件 > 模板自带）。
            if req_enable.is_none() {
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
    pub fn enable_thinking(&self, defaults: &Map<String, Value>, has_tools: bool) -> bool {
        let kw = self.effective_template_kwargs(defaults);
        if let Some(v) = kw.get("enable_thinking").and_then(Value::as_bool) {
            return v;
        }
        if let Some(e) = kw.get("reasoning_effort").and_then(Value::as_str) {
            if reasoning_off(e) {
                return false;
            }
        }
        // **模板对称律**：Qwen 模板里 `auto_disable_thinking_with_tools and
        // _has_tools` 会关思考 ⇒ splitter 必须同相启动，否则正文全被吞进
        // `reasoning_content`。变量现在住 `chat_template.toml`。
        if has_tools
            && kw
                .get("auto_disable_thinking_with_tools")
                .and_then(Value::as_bool)
                == Some(true)
        {
            return false;
        }
        true
    }
}

/// **SSE 静默保活间隔（毫秒；传输层策略，不是引擎配置）**。
///
/// 长 prefill（100k prompt 是分钟级）期间流上一个字节都不写 ⇒ chunked 写永远
/// 不会失败 ⇒ 对端 FIN 无人察觉 ⇒ 引擎把整段 prefill 白跑完。这里按周期写一帧
/// SSE **注释**（`: keepalive`；解析器按规范忽略），写失败 = 连接已死 ⇒ 立刻弃流。
///
/// 2026-11-16：`[runtime] keepalive_ms` 键已删 —— 这是传输层决定，操作员没有
/// 信息量去改它。
pub(crate) const SSE_KEEPALIVE_MS: u64 = 1000;

/// `reasoning_effort` 的关思考词（与快照模板同口径；大小写不敏感）。
fn reasoning_off(e: &str) -> bool {
    e.eq_ignore_ascii_case("none") || e.eq_ignore_ascii_case("off")
}

/// OpenAI 协议会话键 → 引擎 u64 会话 id(进程内哈希;同键同 id)
///
/// **只做哈希,不做缺省**：请求不带键 ⇒ `None`,引擎按 **prompt 内容身份**
/// 认会话（2026-11-16 原始算法 §32；旧 `implicit_session` 已删）。
pub fn session_id(key: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

// ============================================================================
// §2 JSON 构造
// ============================================================================

// 结构性豁免(2026-11-02 clippy 清理):派发/契约 ABI:参数表就是调用面,拆结构体会把每个调用点变成二次装箱
#[allow(clippy::too_many_arguments)]
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
pub(crate) const OVERFLOW_PHRASE: &str = "prompt exceeds max length";

/// 引擎"预算越界"的稳定标记(中文诊断是引擎的既有输出)
fn is_budget_reject(err: &str) -> bool {
    err.contains("超预算")
}

/// 拒绝面 → wire 归一化(唯一实现;两处拒绝出口共用)
pub fn error_json_for_reject(err: String) -> Value {
    if is_budget_reject(&err) {
        return json!({"error": {
            "message": format!(
                "{err}({OVERFLOW_PHRASE}: prompt + max_tokens > max_context_length;请缩短上下文/压缩历史后重试)"
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

pub(crate) fn unix() -> std::time::Duration {
    std::time::SystemTime::now()
        // 系统时钟回拨兜底 → 0(非失败路径;仅用于时间戳/ID)
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
}

pub fn completion_id() -> String {
    format!("chatcmpl-{:x}", unix().as_nanos())
}

/// 日志用长文截断
pub(crate) fn preview(s: &str) -> String {
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
pub(crate) const SINK_CAPACITY: usize = 64;

/// **取消回执的等待上限**。
///
/// 2026-11-16 P4 后它**不再是正确性依赖**（会话串行在引擎队列里，同会话后继
/// turn 自己会等前一个终了）。仍等回执是为了让取消尽快落地、别把队列位占着。
/// 引擎在**步边界**处理取消，实测通常 <1.2s；5s 是「引擎已经不正常」的兜底，
/// 超时只告警（不静默）并继续收尾。
pub(crate) const ABORT_ACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// 应答出口策略(流式 = 逐帧发射;非流式 = 聚合整包)
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplyMode {
    Stream,
    Aggregate,
}

/// usage 账本(单源:prompt 初值 = Accepted 捎带,Prefill 末笔覆盖;
/// completion = Token 事件计数,1 事件 = 1 采样步 = 1 token;
/// cached = Accepted 捎带的**前缀命中量**,AGT-12)
#[derive(Default)]
pub(crate) struct Usage {
    pub(crate) prompt: usize,
    pub(crate) completion: usize,
    /// 思考相位采样步数(`completion_tokens_details.reasoning_tokens`;
    /// 真源 = [`ChannelSplitter::reasoning_steps`],在 Completed 时取终值)
    pub(crate) reasoning: usize,
    pub(crate) cached: usize,
}

/// 非流式聚合器(三通道 → 单条 message)
#[derive(Default)]
pub(crate) struct Agg {
    pub(crate) text: String,
    pub(crate) reason: String,
    pub(crate) tools: Vec<(String, String, String)>, // (id, name, arguments)
    pub(crate) seed: u64,
}

impl Agg {
    /// 工具调用 id(进程内唯一;pi 按 index 归并,id 只作标识)
    pub(crate) fn tool_id(&self, index: usize) -> String {
        format!("call_{:x}_{index}", self.seed)
    }
}

/// 把模板里的每个 `<|image_pad|>` 展开成 `grid_h*grid_w/4` 个。
///
/// 顺序 = `images` 序(内容序)。少/多占位 = 400(不许静默错位)。
pub(crate) fn expand_image_pads(
    prompt: &str,
    images: &[owl_engine::TurnImage],
) -> Result<String, String> {
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
        return Err(format!("模板的 <|image_pad|> 多于图像数 {}", images.len()));
    }
    out.push_str(rest);
    Ok(out)
}

/// **原始请求 JSON 落盘**(显式诊断开关；每请求一份，便于前端联调复现)。
///
/// 文件名用纳秒 + pid + attempt；`create_new` 防并发覆盖。落盘失败只 WARN，
/// 不影响请求本身（诊断面不能把主路径打死）。
pub(crate) fn dump_request(dir: &std::path::Path, body: &[u8]) {
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
                    owl_shared::log_debug!(
                        "request-dump",
                        "落盘 {} ({}B)",
                        path.display(),
                        body.len()
                    );
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
