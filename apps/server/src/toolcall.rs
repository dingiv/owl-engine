//! 流式三通道解析器(**agent 服务面专项 §4.5**)。
//!
//! 引擎吐的是**增量解码文本**(`TurnEvent::Token{delta}` = 全量重解取后缀
//! 差分),而模型输出里混着四种**文本标记**:
//! `<think>` / `</think>` / `<tool_call>` / `</tool_call>`(长形态别名见常量块)。
//! 本模块把它们切成三条驱动通道(思考 / 正文 / 工具调用),并保证下述不变量
//! (§4.7):**I3** 正文通道永不出现 `<tool_call>`/`</tool_call>` 原文;
//! **I4** 工具 `arguments` 恒为合法 JSON 字符串;**I5** 工具序号连续。
//!
//! ## 为什么必须缓冲(T0 实测)
//! 标记本身是单 token,但**跨 chunk 边界仍会切半**(引擎的增量解码在标记
//! 中间态会吐出半截前缀);而 `<function=bash>` 实测是 **5 个 token**
//! (`<`/`function`/`=b`/`ash`/`>`)⇒ **解析只能在文本层做**,不能按 token
//! 边界假设。故:标记前缀挂在尾巴缓冲里(最多 hold 标记长 −1 字节)。
//!
//! ## 起始状态
//! 模板的 assistant 起手已经写进了 `<think>`(**思考开块在 prompt 里**,
//! 不在生成流里)⇒ 开思考时解析器必须**从 Think 态起步**,关思考时从 Text 态
//! 起步(见 [`ChannelSplitter::new`])。
//!
//! ## ⚠️ 标记字面量必须与快照模板一致
//! 标记是**模板资产的字面量**(本机 27B 快照写出 `<think>`/`</think>`,
//! 见 `tests/fixtures/chat_tpl/*.txt` 尾字节)。模板漂移(上游改标记名)⇒
//! 解析器静默失效,故有 `markers_match_snapshot` 测试把这对
//! 耦合钉在快照上。

use std::collections::HashMap;

use serde_json::{Map, Value as Json};

/// 通道(思考 / 正文)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chan {
    /// `reasoning_content`(pi 的思考面板)
    Think,
    /// `content`
    Text,
}

/// 解析产物(协议层按此驱动 SSE 三通道)
#[derive(Debug, PartialEq)]
pub enum Item {
    /// 通道增量
    Delta { chan: Chan, text: String },
    /// 一个**完整闭合**的工具调用(xml/json 两格式统一出口)
    ToolCall { index: usize, name: String, arguments: String },
    /// 畸形工具块(未闭合/无 `<function=`/参数不可解析)—— 不发射工具调用,
    /// 但**不静默**:调用方必须记日志(§4.6)
    Malformed { kind: &'static str, raw: String },
}

/// 文本标记(**快照模板字面量**;hex 实证:`3c7468696e6b3e` = `<think>`)
///
/// 主形态 = `<think>` / `</think>` —— 模板 `add_generation_prompt` 段与渲染
/// 产物实测(见 `tests/fixtures/chat_tpl/*.txt` 尾字节)。
/// **长形态别名** = `<thinking>` / `</thinking>`(模板在"历史 assistant 内容剥除"分支兼容它);主形态见常量块。
/// 分支兼容它(`content.startswith(...)`),模型偶尔照抄 ⇒ 解析器一并认。
const MARK_THINK_OPEN: &str = "<think>";
const MARK_THINK_CLOSE: &str = "</think>";
const MARK_TOOL_OPEN: &str = "<tool_call>";
const MARK_TOOL_CLOSE: &str = "</tool_call>";
/// 无外层 `<tool_call>` 的裸函数块（GGUF 上游模板实测会漏壳；函数名/参数
/// 形态与 XML 工具块完全一致）。只在文本/思考态遇到 `<function=` 时启用。
const MARK_FUNCTION_OPEN: &str = "<function=";
const MARK_FUNCTION_CLOSE: &str = "</function>";
/// Anthropic/Claude 风格工具块（Qwen 在长 system prompt 下实测会照抄）：
/// `<function_calls><invoke name="…"><parameter name="…">…</parameter></invoke>…</function_calls>`。
/// 外层整体缓冲到闭合后一次解析；一个 wrapper 可含多个 invoke。
const MARK_FUNCTION_CALLS_OPEN: &str = "<function_calls>";
const MARK_FUNCTION_CALLS_CLOSE: &str = "</function_calls>";
const MARK_THINK_OPEN_LONG: &str = "<thinking>";
const MARK_THINK_CLOSE_LONG: &str = "</thinking>";
// 第二组别名:模型偶尔混用 `thought`(实测 2026-10-14 出现过一次)。
// 认它 = 该标记被吞掉不进正文/思考面板,而不是当普通文本泄漏出去
const MARK_THINK_OPEN_THOUGHT: &str = "<thought>";
const MARK_THINK_CLOSE_THOUGHT: &str = "</thought>";

/// Text/Think/InToolBetween 态的识别标记(主 + 别名;游标标记也**吞掉不泄漏**,
/// 见 `next_state`)。`</function>` 不入表:它只由工具态显式 `find`,
/// 在正文态出现一律当普通文本(旧行为,见 `char_by_char`)。
const MARKS: [&str; 11] = [
    MARK_THINK_OPEN,
    MARK_THINK_CLOSE,
    MARK_TOOL_OPEN,
    MARK_TOOL_CLOSE,
    MARK_FUNCTION_OPEN,
    MARK_FUNCTION_CALLS_OPEN,
    MARK_FUNCTION_CALLS_CLOSE,
    MARK_THINK_OPEN_LONG,
    MARK_THINK_CLOSE_LONG,
    MARK_THINK_OPEN_THOUGHT,
    MARK_THINK_CLOSE_THOUGHT,
];

/// 别名归一(长形态 → 主形态);分派只看归一结果
fn canon(m: &str) -> &'static str {
    if m == MARK_THINK_OPEN_LONG || m == MARK_THINK_OPEN_THOUGHT {
        MARK_THINK_OPEN
    } else if m == MARK_THINK_CLOSE_LONG || m == MARK_THINK_CLOSE_THOUGHT {
        MARK_THINK_CLOSE
    } else if m == MARK_THINK_OPEN {
        MARK_THINK_OPEN
    } else if m == MARK_THINK_CLOSE {
        MARK_THINK_CLOSE
    } else if m == MARK_TOOL_OPEN {
        MARK_TOOL_OPEN
    } else if m == MARK_FUNCTION_OPEN {
        MARK_FUNCTION_OPEN
    } else if m == MARK_FUNCTION_CLOSE {
        MARK_FUNCTION_CLOSE
    } else if m == MARK_FUNCTION_CALLS_OPEN {
        MARK_FUNCTION_CALLS_OPEN
    } else if m == MARK_FUNCTION_CALLS_CLOSE {
        MARK_FUNCTION_CALLS_CLOSE
    } else {
        MARK_TOOL_CLOSE
    }
}

/// 工具块长度上限(病态保护:未闭合工具块不得吃满内存)
const MAX_TOOL_BYTES: usize = 1 << 20;

/// 结构轨迹**硬上限**(每 turn;超出计 `dropped`,不静默丢)
const TRACE_CAP: usize = 512;

/// 解析器**结构轨迹事件**(G5②:`/debug/agent` 的“状态机轨迹”)。
///
/// **只记结构**(状态迁移 / 吞空白 / 游标 / 工具块 / 收尾),
/// **不记 delta 文本** —— 轨迹是排查用的骨架,不做第二份输出副本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructEvent {
    /// 状态迁移(think / text / tool 三态;由标记驱动)
    State { from: &'static str, to: &'static str },
    /// 结构空白吞除量(关思考标记之后的 `\n`;模板资产,见 `swallow_lf`)
    SwallowLf(usize),
    /// 游标标记(当前态下不该出现的标记;已吞掉不泄漏)
    Stray(&'static str),
    /// 工具调用闭合(解析成功)
    Tool { index: usize, name: String, arg_bytes: usize },
    /// 模型近失格式被保守修复(只在校验通过后记;不掩盖真错误)
    Repair { tool: String, kind: &'static str },
    /// 参数类型不匹配但已软通过（保留原值，交远端工具/前端处置）
    SoftType { tool: String, prop: String, expected: String, got: String },
    /// 畸形工具块(未闭合/超限/参数不可解析;上层记日志)
    Malformed { kind: &'static str, bytes: usize },
    /// 生成流结束时的状态与尾巴余量
    Finish { state: &'static str, held_bytes: usize },
}

impl StructEvent {
    /// 单行紧凑形态(JSON 数组里的字符串;人读 + grep 两用)
    pub fn line(&self) -> String {
        match self {
            StructEvent::State { from, to } => format!("state {from}→{to}"),
            StructEvent::SwallowLf(n) => format!("swallow_lf {n}"),
            StructEvent::Stray(m) => format!("stray {m}"),
            StructEvent::Tool { index, name, arg_bytes } => {
                format!("tool #{index} {name} args={arg_bytes}B")
            }
            StructEvent::Repair { tool, kind } => format!("repair {tool} {kind}"),
            StructEvent::SoftType { tool, prop, expected, got } => {
                format!("soft_type {tool}.{prop} expect={expected} got={got}")
            }
            StructEvent::Malformed { kind, bytes } => format!("malformed {kind} {bytes}B"),
            StructEvent::Finish { state, held_bytes } => {
                format!("finish {state} held={held_bytes}B")
            }
        }
    }
}

/// 状态 → 轨迹里的短名(轨迹只存 `&'static str`,不外露 `State`)
fn state_name(s: State) -> &'static str {
    match s {
        State::Text => "text",
        State::Think => "think",
        State::InTool => "tool",
        State::InBareTool => "bare_tool",
        State::InAnthropic => "anthropic_tool",
        State::InToolBetween => "between_tool",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Text,
    Think,
    InTool,
    /// 裸 `<function=...>...</function>`（无 `<tool_call>` 外壳）
    InBareTool,
    /// Anthropic/Claude 风格 `<function_calls>…</function_calls>`（可含多个 invoke）
    InAnthropic,
    /// `</function>` 已闭合，等待可选的 `</tool_call>` / 下一个工具
    InToolBetween,
}

/// 请求侧工具 schema(取 `name → 属性名 → JSON 类型` + required;用于 xml
/// 参数类型回填与**服务层工具调用校验**)
#[derive(Debug, Default)]
pub struct ToolSchemas {
    types: HashMap<String, HashMap<String, String>>,
    required: HashMap<String, Vec<String>>,
}

/// 参数校验错误（字段级；`tool_call_arg_type` 允许软通过）。
#[derive(Debug, Clone)]
struct ArgError {
    kind: &'static str,
    tool: String,
    prop: Option<String>,
    expected: Option<String>,
    got: Option<String>,
}

impl ArgError {
    fn tool(kind: &'static str, tool: &str) -> Self {
        Self { kind, tool: tool.to_string(), prop: None, expected: None, got: None }
    }

    fn missing(tool: &str, prop: &str) -> Self {
        Self {
            kind: "tool_call_missing_required",
            tool: tool.to_string(),
            prop: Some(prop.to_string()),
            expected: Some("required".into()),
            got: Some("missing".into()),
        }
    }

    fn type_mismatch(tool: &str, prop: &str, expected: &str, got: &Json) -> Self {
        Self {
            kind: "tool_call_arg_type",
            tool: tool.to_string(),
            prop: Some(prop.to_string()),
            expected: Some(expected.to_string()),
            got: Some(short_json(got)),
        }
    }

    fn is_soft(&self) -> bool {
        self.kind == "tool_call_arg_type"
    }

    fn log_line(&self) -> String {
        match (&self.prop, &self.expected, &self.got) {
            (Some(p), Some(e), Some(g)) => {
                format!("tool={} prop={p} expect={e} got={g}", self.tool)
            }
            _ => format!("tool={} kind={}", self.tool, self.kind),
        }
    }
}

/// JSON 值是否匹配 schema 标量类型。
fn type_matches(ty: &str, v: &Json) -> bool {
    match ty {
        "number" | "float" => v.is_number(),
        "integer" => {
            v.as_i64().is_some()
                || v.as_u64().is_some()
                || v.as_f64().is_some_and(|f| f.is_finite() && f.fract() == 0.0)
        }
        "boolean" => v.is_boolean(),
        "string" => v.is_string(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        "null" => v.is_null(),
        _ => true,
    }
}

/// 取属性的标量类型；Pydantic v2 的 Optional[T] 是 `anyOf/oneOf`，
/// 取第一个非 null 分支（与 vLLM 的 qwen3 parser 类型回填同口径）。
fn json_type_of(v: &Json) -> Option<&str> {
    if let Some(t) = v.get("type").and_then(Json::as_str) {
        return Some(t);
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(items) = v.get(key).and_then(Json::as_array) {
            for item in items {
                if let Some(t) = item.get("type").and_then(Json::as_str) {
                    if t != "null" {
                        return Some(t);
                    }
                }
            }
        }
    }
    None
}

impl ToolSchemas {
    /// 从 OpenAI `tools` 数组提取(兼容嵌套 `{type,function:{…}}` 与扁平形)
    pub fn from_tools(tools: &[Json]) -> Self {
        let mut types = HashMap::new();
        let mut required = HashMap::new();
        for t in tools {
            let f = t.get("function").unwrap_or(t);
            let Some(name) = f.get("name").and_then(|v| v.as_str()) else { continue };
            let mut props = HashMap::new();
            if let Some(p) = f
                .get("parameters")
                .and_then(|p| p.get("properties"))
                .and_then(|p| p.as_object())
            {
                for (k, v) in p {
                    if let Some(ty) = json_type_of(v) {
                        props.insert(k.clone(), ty.to_string());
                    }
                }
            }
            let req: Vec<String> = f
                .get("parameters")
                .and_then(|p| p.get("required"))
                .and_then(|r| r.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            types.insert(name.to_string(), props);
            required.insert(name.to_string(), req);
        }
        Self { types, required }
    }

    /// **服务层工具调用校验**(OpenAI 语义):工具名必须在请求 tools 中;
    /// arguments 必须是 JSON 对象;required 属性必须非 null;已知属性类型必须
    /// 匹配。无 schema(空 tools)时不做推断 —— 返回 Ok,由上游决定。
    ///
    /// `tool_call_arg_type` 由调用方按**软错误**处理（保留原值，不整轮 502）。
    fn validate(&self, name: &str, args: &str) -> Result<(), ArgError> {
        if self.types.is_empty() {
            return Ok(());
        }
        if !self.types.contains_key(name) {
            return Err(ArgError::tool("tool_call_unknown_tool", name));
        }
        let v: Json = serde_json::from_str(args)
            .map_err(|_| ArgError::tool("tool_call_args_invalid_json", name))?;
        let Some(obj) = v.as_object() else {
            return Err(ArgError::tool("tool_call_args_not_object", name));
        };
        if let Some(req) = self.required.get(name) {
            for p in req {
                if obj.get(p).is_none_or(Json::is_null) {
                    return Err(ArgError::missing(name, p));
                }
            }
        }
        if let Some(props) = self.types.get(name) {
            for (k, ty) in props {
                if let Some(val) = obj.get(k) {
                    if !val.is_null() && !type_matches(ty, val) {
                        return Err(ArgError::type_mismatch(name, k, ty, val));
                    }
                }
            }
        }
        Ok(())
    }

    /// xml 参数值 → JSON 值。**按 schema 类型回填**(xml 分支把一切写成了
    /// 字符串形态;不按 schema 则 `timeout: 30` 会变成 `"30"`,pi 侧
    /// JSON-schema 校验会拒)。无 schema = 保守保持字符串(不猜)
    fn coerce(&self, tool: &str, prop: &str, raw: &str) -> Json {
        let Some(ty) = self.types.get(tool).and_then(|m| m.get(prop)) else {
            return Json::String(raw.to_string());
        };
        match ty.as_str() {
            "number" | "float" => match parse_json_or_quoted(raw) {
                Some(Json::Number(n)) => Json::Number(n),
                _ => Json::String(raw.to_string()),
            },
            "integer" => match parse_json_or_quoted(raw) {
                Some(Json::Number(n)) => normalize_integer(n),
                _ => Json::String(raw.to_string()),
            },
            "boolean" => match parse_json_or_quoted(raw) {
                Some(Json::Bool(b)) => Json::Bool(b),
                _ => match raw.trim().to_ascii_lowercase().as_str() {
                    "true" => Json::Bool(true),
                    "false" => Json::Bool(false),
                    _ => Json::String(raw.to_string()),
                },
            },
            "array" | "object" => match parse_json_or_quoted(raw) {
                Some(v @ (Json::Array(_) | Json::Object(_))) => v,
                _ => Json::String(raw.to_string()),
            },
            _ => Json::String(raw.to_string()),
        }
    }
}

/// 保守 JSON 归一：裸 JSON 或**被 JSON 引号包住的 JSON** 都取内层。
/// 例：`"512"` → 512、`"[1,2]"` → [1,2]、`"true"` → true。
fn parse_json_or_quoted(raw: &str) -> Option<Json> {
    let t = raw.trim();
    let v = serde_json::from_str::<Json>(t).ok()?;
    if let Json::String(s) = &v {
        if let Ok(inner) = serde_json::from_str::<Json>(s.trim()) {
            return Some(inner);
        }
    }
    Some(v)
}

/// 整数归一：20.0 → 20；非整数保持原 Number（由软校验兜底）。
fn normalize_integer(n: serde_json::Number) -> Json {
    if let Some(i) = n.as_i64() {
        return Json::Number(i.into());
    }
    if let Some(u) = n.as_u64() {
        return Json::Number(u.into());
    }
    if let Some(f) = n.as_f64() {
        if f.is_finite()
            && f.fract() == 0.0
            && f >= i64::MIN as f64
            && f <= i64::MAX as f64
        {
            return Json::Number((f as i64).into());
        }
    }
    Json::Number(n)
}

fn short_json(v: &Json) -> String {
    const N: usize = 160;
    let s = v.to_string();
    if s.len() <= N {
        s
    } else {
        let cut = s.char_indices().take_while(|(i, _)| *i < N).last().map_or(0, |(i, _)| i);
        format!("{}…({}B)", &s[..cut], s.len())
    }
}

/// 日志用的 raw 截断（保留完整 450B 级工具块；超长块只打头尾账）。
pub(crate) fn raw_for_log(s: &str) -> String {
    const N: usize = 8192;
    if s.len() <= N {
        return s.to_string();
    }
    let cut = s.char_indices().take_while(|(i, _)| *i < N).last().map_or(0, |(i, _)| i);
    format!("{}…(+{}B)", &s[..cut], s.len() - cut)
}

/// 三通道解析器(有状态;每 turn 新建)
pub struct ChannelSplitter {
    state: State,
    buf: String,
    schemas: ToolSchemas,
    tool_index: usize,
    /// 工具块解析开关(`[agent] parse_tool_calls`;关 = `<tool_call>` 原文
    /// 进 `content` —— 诊断臂/兼容旧客户端)
    tool_on: bool,
    /// 游标标记计数(如关思考时仍吐 `</thinking>`)—— 一律**吞掉**不泄漏
    /// 进通道,但计数交给上层记日志(不许静默)
    strays: usize,
    /// **思考相位的采样步数**(G5:`completion_tokens_details.reasoning_tokens`)。
    ///
    /// 口径:引擎 1 个 `TurnEvent::Token` = 1 次采样 = 1 个 token,而协议层
    /// 每 token 调一次 [`push`](Self::push) ⇒ 以 **push 入口时的解析器状态**
    /// 归因:Think 态 ⇒ 该 token 是思考相位产出(含 `</think>` 标记本体,
    /// 它确实由思考段生成;其后的首个正文 token 已落 Text 态)。
    /// 不用 tokenizer 反数 —— 标记文本会被解析器吞掉,反数与真值天然不等。
    reasoning: usize,
    /// 结构轨迹(G5②;有界 `TRACE_CAP`,超出计 `trace_dropped`)
    trace: Vec<StructEvent>,
    /// 轨迹超出上限而被省略的事件数(**不静默**)
    trace_dropped: usize,
    /// 畸形工具块计数(G5②`/debug/agent` 的 `outcome.malformed`;
    /// 三个入口:未闭合 / 超限 / 参数不可解析 —— 每一处都已落日志)
    malformed: usize,
    /// **关闭思考标记之后的结构空白吞除余量**(UTF-8 安全:只吞 `\n`)。
    ///
    /// 为什么要有它(§4.3 渲染对称律 R1;2026-10-19 实测定谳):模板把一轮
    /// assistant 渲染成
    /// ```text
    /// <|im_start|>assistant\n thinking\n{trim(reasoning)}\n</think>\n\n{content}
    /// ```
    /// —— `\n</think>\n\n` 是**模板的结构资产**,不属于任何通道。而生成流里
    /// 紧跟 ` response` 的那至多两个 `\n` 是**同一个结构**的落地。旧实现只吞
    /// 标记本体 ⇒ 那两个换行漏进 `content` ⇒ 客户端回填后渲染多出两个 token
    /// ⇒ **会话守卫在第一个 content token 就失配**(实测:T2 `cached_tokens=0`,
    /// 前置 83 tok 全对、分叉点正是 `\n\n`)。
    swallow_lf: u8,
}

impl ChannelSplitter {
    /// `start_in_think` = 请求是否开了思考(模板预填了 `<thinking>` ⇒ 生成流
    /// 从思考段中间起步,见模块头)
    pub fn new(start_in_think: bool, schemas: ToolSchemas) -> Self {
        Self {
            state: if start_in_think { State::Think } else { State::Text },
            buf: String::new(),
            schemas,
            tool_index: 0,
            tool_on: true,
            strays: 0,
            reasoning: 0,
            trace: Vec::new(),
            trace_dropped: 0,
            malformed: 0,
            swallow_lf: 0,
        }
    }

    /// 关工具块解析(`[agent] parse_tool_calls = false`):工具标记不再进
    /// 工具通道,而是**按普通文本放行**(诊断/回退臂)。思考通道不受影响。
    pub fn with_tool_parsing(mut self, on: bool) -> Self {
        self.tool_on = on;
        self
    }

    /// 本 drain 生效的标记集(工具解析关时剔掉工具标记)
    fn marks(&self) -> Vec<&'static str> {
        MARKS
            .iter()
            .filter(|m| {
                self.tool_on
                    || (**m != MARK_TOOL_OPEN
                        && **m != MARK_TOOL_CLOSE
                        && **m != MARK_FUNCTION_OPEN
                        && **m != MARK_FUNCTION_CALLS_OPEN
                        && **m != MARK_FUNCTION_CALLS_CLOSE)
            })
            .copied()
            .collect()
    }

    /// 本 drain 在「工具之间」态认的结构标记(只这三个;工具解析关时不会进此态)
    fn marks_structural(&self) -> Vec<&'static str> {
        vec![
            MARK_TOOL_CLOSE,
            MARK_TOOL_OPEN,
            MARK_FUNCTION_OPEN,
            MARK_FUNCTION_CALLS_OPEN,
            MARK_FUNCTION_CALLS_CLOSE,
        ]
    }

    /// 本 turn 已解析出的工具调用数(协议层据此定 `finish_reason`)
    pub fn tool_calls(&self) -> usize {
        self.tool_index
    }

    /// 游标标记计数(诊断;非零 = 值得记一条日志)
    pub fn strays(&self) -> usize {
        self.strays
    }

    /// 本 turn 思考相位的采样步数(= `completion_tokens_details.reasoning_tokens`
    /// 真源;见字段 `reasoning` 的口径注)
    pub fn reasoning_steps(&self) -> usize {
        self.reasoning
    }

    /// **结构轨迹**(G5②`/debug/agent`):只含结构事件,行序 = 发生序。
    /// 超上限的尾巴以一条显式记账行收口(`…(+N dropped)`)。
    pub fn structural_trace(&self) -> Vec<String> {
        let mut out: Vec<String> = self.trace.iter().map(StructEvent::line).collect();
        if self.trace_dropped > 0 {
            out.push(format!("…(+{} dropped)", self.trace_dropped));
        }
        out
    }

    /// 畸形工具块计数(诊断面;非零 = 上限层值得一条日志,已有)
    pub fn malformed(&self) -> usize {
        self.malformed
    }

    /// 记一条结构事件(满即计数,不静默丢)
    fn trace_event(&mut self, e: StructEvent) {
        if self.trace.len() < TRACE_CAP {
            self.trace.push(e);
        } else {
            self.trace_dropped += 1;
        }
    }

    /// 标记 → 目标态。当前状态下不该出现的标记(Text 遇 `</thinking>`/
    /// `</tool_call>`;Think 遇 `<thinking>`)一律**吞掉不泄漏**(否则原文
    /// 会漏进正文/思考面板),并计数
    fn next_state(&mut self, m: &str) -> State {
        let from = self.state;
        let to = match (self.state, canon(m)) {
            (State::Text, MARK_THINK_OPEN) => State::Think,
            (State::Text, MARK_TOOL_OPEN) => State::InTool,
            (State::Text, MARK_FUNCTION_OPEN) => State::InBareTool,
            (State::Text, MARK_FUNCTION_CALLS_OPEN) => State::InAnthropic,
            (State::Think, MARK_FUNCTION_OPEN) => State::InBareTool,
            (State::Think, MARK_FUNCTION_CALLS_OPEN) => State::InAnthropic,
            (State::InToolBetween, MARK_TOOL_CLOSE) => State::Text,
            (State::InToolBetween, MARK_TOOL_OPEN) => State::InTool,
            (State::InToolBetween, MARK_FUNCTION_OPEN) => State::InBareTool,
            (State::InToolBetween, MARK_FUNCTION_CALLS_OPEN) => State::InAnthropic,
            (State::InToolBetween, MARK_FUNCTION_CALLS_CLOSE) => State::Text,
            (State::Think, MARK_THINK_CLOSE) => {
                // 模板 `\n</think>\n\n{content}` 里的 `\n\n` 归模板(单源):
                // 后续至多两个 `\n` 从流里吞掉(不足则按实际吞尽,见 `drain`)
                self.swallow_lf = 2;
                State::Text
            }
            // Think 态遇 `<tool_call>`:模型没关思考直接开工(§4.6)—— 进工具
            // 通道,而不是把整块工具调用吞进思考面板
            (State::Think, MARK_TOOL_OPEN) => State::InTool,
            _ => {
                self.strays += 1;
                self.trace_event(StructEvent::Stray(canon(m)));
                self.state
            }
        };
        if to != from {
            self.trace_event(StructEvent::State { from: state_name(from), to: state_name(to) });
        }
        to
    }

    /// 推进一个增量文本块 → 产出事件序列
    ///
    /// 调用契约:**一个引擎 token 恰好调一次**(reasoning 计数按调用入口
    /// 状态归因;批量合并多个 token 会低估思考步数)。
    pub fn push(&mut self, chunk: &str) -> Vec<Item> {
        if self.state == State::Think {
            self.reasoning += 1;
        }
        self.buf.push_str(chunk);
        self.drain()
    }

    /// 生成流结束:冲刷尾巴(未闭合的标记前缀按**普通文本**放行 —— 模型没吐
    /// 完的最坏情况不该吞内容;未闭合工具块 → Malformed)
    pub fn finish(&mut self) -> Vec<Item> {
        let mut out = Vec::new();
        match self.state {
            State::InTool | State::InBareTool | State::InAnthropic => {
                let raw = std::mem::take(&mut self.buf);
                let kind = match self.state {
                    State::InBareTool => "tool_call_bare_unclosed",
                    State::InAnthropic => "tool_call_anthropic_unclosed",
                    _ => "tool_call_unclosed",
                };
                self.trace_event(StructEvent::Malformed { kind, bytes: raw.len() });
                self.malformed += 1;
                out.push(Item::Malformed { kind, raw });
                self.state = State::Text;
            }
            State::InToolBetween | State::Text | State::Think => {
                // 收尾:未消化的结构空白按"归模板"处理(吐给 content 会让
                // 回填渲染多出换行 —— 与 `drain` 同一律)
                if self.swallow_lf > 0 {
                    let n = self
                        .buf
                        .bytes()
                        .take(self.swallow_lf as usize)
                        .take_while(|b| *b == b'\n')
                        .count();
                    self.buf.drain(..n);
                    self.swallow_lf = 0;
                    if n > 0 {
                        self.trace_event(StructEvent::SwallowLf(n));
                    }
                }
                if !self.buf.is_empty() {
                    let text = std::mem::take(&mut self.buf);
                    out.push(Item::Delta { chan: self.chan(), text });
                }
            }
        }
        self.trace_event(StructEvent::Finish {
            state: state_name(self.state),
            held_bytes: self.buf.len(),
        });
        out
    }

    fn chan(&self) -> Chan {
        if self.state == State::Think {
            Chan::Think
        } else {
            Chan::Text
        }
    }

    fn drain(&mut self) -> Vec<Item> {
        let mut out = Vec::new();
        loop {
            // **结构空白吞除**(只在关闭思考标记之后;见 `swallow_lf` 注):
            // 缓冲空 = 不能定谳(流式,等下一个 chunk)⇒ 挂起。
            if self.swallow_lf > 0 && self.state == State::Text {
                let mut swallowed = 0usize;
                while self.swallow_lf > 0 {
                    match self.buf.as_bytes().first() {
                        Some(b'\n') => {
                            self.buf.drain(..1);
                            self.swallow_lf -= 1;
                            swallowed += 1;
                        }
                        Some(_) => self.swallow_lf = 0,
                        None => break,
                    }
                }
                if swallowed > 0 {
                    self.trace_event(StructEvent::SwallowLf(swallowed));
                }
                if self.swallow_lf > 0 {
                    break;
                }
            }
            match self.state {
                State::InAnthropic => {
                    if let Some(p) = self.buf.find(MARK_FUNCTION_CALLS_CLOSE) {
                        let body = self.buf[..p].to_string();
                        self.buf.drain(..p + MARK_FUNCTION_CALLS_CLOSE.len());
                        out.extend(self.build_anthropic_calls(&body));
                        self.state = State::Text;
                        continue;
                    }
                    if self.buf.len() > MAX_TOOL_BYTES {
                        let raw = std::mem::take(&mut self.buf);
                        self.trace_event(StructEvent::Malformed {
                            kind: "tool_call_anthropic_overflow",
                            bytes: raw.len(),
                        });
                        self.malformed += 1;
                        out.push(Item::Malformed { kind: "tool_call_anthropic_overflow", raw });
                        self.state = State::Text;
                        continue;
                    }
                    break; // 整体缓冲到 </function_calls>
                }
                State::InTool => {
                    // XML: `</function>` 就是工具体闭合（vLLM TOOL_BETWEEN 同款）；
                    // JSON: 仍以 `</tool_call>` 闭合。
                    let xml = self.buf.trim_start().starts_with(MARK_FUNCTION_OPEN);
                    let func = if xml { self.buf.find(MARK_FUNCTION_CLOSE) } else { None };
                    let tool = self.buf.find(MARK_TOOL_CLOSE);
                    let close = match (func, tool) {
                        (Some(f), Some(t)) if f < t => Some((f, MARK_FUNCTION_CLOSE, true)),
                        (_, Some(t)) => Some((t, MARK_TOOL_CLOSE, false)),
                        (Some(f), None) => Some((f, MARK_FUNCTION_CLOSE, true)),
                        _ => None,
                    };
                    if let Some((p, marker, between)) = close {
                        let body = self.buf[..p].to_string();
                        self.buf.drain(..p + marker.len());
                        out.push(self.build_tool_call(&body));
                        self.state = if between { State::InToolBetween } else { State::Text };
                        continue;
                    }
                    if self.buf.len() > MAX_TOOL_BYTES {
                        let raw = std::mem::take(&mut self.buf);
                        self.trace_event(StructEvent::Malformed {
                            kind: "tool_call_overflow",
                            bytes: raw.len(),
                        });
                        self.malformed += 1;
                        out.push(Item::Malformed { kind: "tool_call_overflow", raw });
                        self.state = State::Text;
                        continue;
                    }
                    break; // 工具块内部内容不外发;等闭合标记
                }
                State::InToolBetween => {
                    // vLLM TOOL_BETWEEN 同构:`</function>` 与下一个结构标记之间的
                    // 空白 = **模板结构资产**,既不进正文也不进工具体。
                    // 因此先找标记,再决定丢弃范围;看不到标记且尾巴可能是
                    // 标记前缀 ⇒ **挂起等下一个 chunk**(不能提前转 Text,
                    // 否则结构空白会泄进正文 —— 2026-10-24 char_by_char 翻车定案)
                    let structural = self.marks_structural();
                    let hit = structural
                        .iter()
                        .filter_map(|m| self.buf.find(m).map(|p| (p, *m)))
                        .min_by_key(|(p, _)| *p);
                    if let Some((p, m)) = hit {
                        if !self.buf[..p].chars().all(|c| c.is_ascii_whitespace()) {
                            // 标记前夹了真文本:交回 Text 照常发(不吞用户可见内容)
                            self.state = State::Text;
                            continue;
                        }
                        // 裸函数块的标记本体必须留在 buf 里(parse_tool_body 靠它定名)
                        self.buf.drain(..if canon(m) == MARK_FUNCTION_OPEN { p } else { p + m.len() });
                        self.state = match canon(m) {
                            MARK_TOOL_CLOSE => State::Text,
                            MARK_TOOL_OPEN => State::InTool,
                            MARK_FUNCTION_CALLS_OPEN => State::InAnthropic,
                            MARK_FUNCTION_CALLS_CLOSE => State::Text,
                            _ => State::InBareTool,
                        };
                        continue;
                    }
                    // 无完整标记:尾巴若是标记真前缀 → 整段挂起
                    let hold = holdback_len(&self.buf, &structural);
                    if hold == self.buf.len() {
                        break; // 全挂起,等下一 chunk
                    }
                    // 开头若是空白,先丢掉(不可能属于任何标记);剩余非结构
                    // 内容交回 Text 处理
                    let ws = self
                        .buf
                        .bytes()
                        .take_while(|b| b.is_ascii_whitespace())
                        .count();
                    if ws > 0 {
                        self.buf.drain(..ws);
                    }
                    if self.buf.is_empty() {
                        break;
                    }
                    self.state = State::Text;
                    continue;
                }
                State::InBareTool => {
                    if let Some(p) = self.buf.find(MARK_FUNCTION_CLOSE) {
                        let body = self.buf[..p].to_string();
                        self.buf.drain(..p + MARK_FUNCTION_CLOSE.len());
                        out.push(self.build_tool_call(&body));
                        self.state = State::InToolBetween;
                        continue;
                    }
                    if self.buf.len() > MAX_TOOL_BYTES {
                        let raw = std::mem::take(&mut self.buf);
                        self.trace_event(StructEvent::Malformed {
                            kind: "tool_call_bare_overflow",
                            bytes: raw.len(),
                        });
                        self.malformed += 1;
                        out.push(Item::Malformed { kind: "tool_call_bare_overflow", raw });
                        self.state = State::Text;
                        continue;
                    }
                    break; // 裸工具块内部内容不外发;等 </function>
                }
                State::Text | State::Think => {
                    // 找**最早**出现的标记(同位置取更长者,防前缀重叠)
                    let marks = self.marks();
                    let mut hit: Option<(usize, &str)> = None;
                    for m in marks.iter() {
                        if let Some(p) = self.buf.find(m) {
                            let better = match hit {
                                Some((bp, bm)) => p < bp || (p == bp && m.len() > bm.len()),
                                None => true,
                            };
                            if better {
                                hit = Some((p, m));
                            }
                        }
                    }
                    match hit {
                        Some((p, m)) => {
                            if p > 0 {
                                out.push(Item::Delta {
                                    chan: self.chan(),
                                    text: self.buf[..p].to_string(),
                                });
                            }
                            // 裸 `<function=` 的标记本体必须留在 buf 里：
                            // `parse_tool_body` 以它定位名字；普通标记照常剥掉。
                            let keep_marker = canon(m) == MARK_FUNCTION_OPEN;
                            self.buf.drain(..if keep_marker { p } else { p + m.len() });
                            self.state = self.next_state(m);
                            continue;
                        }
                        None => {
                            // 尾巴可能是标记前缀(跨 chunk 切半)⇒ 挂起不发
                            let hold = holdback_len(&self.buf, &marks);
                            let upto = self.buf.len() - hold;
                            if upto > 0 {
                                out.push(Item::Delta {
                                    chan: self.chan(),
                                    text: self.buf[..upto].to_string(),
                                });
                                self.buf.drain(..upto);
                            }
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    /// Anthropic/Claude 风格 wrapper → 一个或多个完整调用。
    ///
    /// 外层 `<function_calls>` 已由状态机剥掉；body 里是
    /// `<invoke name="…">` + 可选 `<parameter name="…">…</parameter>`。
    fn build_anthropic_calls(&mut self, body: &str) -> Vec<Item> {
        let body = body.trim();
        // 模型偶尔会在 `<function_calls>` 壳里写 Qwen 的 `<function=...>`；
        // Anthropic 解析失败时再按 Qwen 块扫一遍（只接受能过 schema 的块）。
        match parse_anthropic_calls(body, &self.schemas)
            .or_else(|_| parse_qwen_blocks(body, &self.schemas))
        {
            Ok(calls) if !calls.is_empty() => {
                // 硬错误（未知工具/缺 required/坏 JSON）整组拒；
                // arg_type 只记字段级 WARN，保留原值继续发射。
                let mut soft = Vec::new();
                for (name, args) in &calls {
                    if let Err(e) = self.schemas.validate(name, args) {
                        if e.is_soft() {
                            soft.push(e);
                        } else {
                            self.trace_event(StructEvent::Malformed {
                                kind: e.kind,
                                bytes: body.len(),
                            });
                            owl_shared::log_warn!(
                                "chat",
                                "工具调用硬拒绝({}; raw={})",
                                e.log_line(),
                                raw_for_log(body)
                            );
                            self.malformed += 1;
                            return vec![Item::Malformed { kind: e.kind, raw: body.to_string() }];
                        }
                    }
                }
                for e in soft {
                    let prop = e.prop.clone().unwrap_or_default();
                    let expected = e.expected.clone().unwrap_or_default();
                    let got = e.got.clone().unwrap_or_default();
                    self.trace_event(StructEvent::SoftType {
                        tool: e.tool.clone(),
                        prop,
                        expected,
                        got,
                    });
                    owl_shared::log_warn!(
                        "chat",
                        "工具参数类型不匹配(软通过;{}; raw={})",
                        e.log_line(),
                        raw_for_log(body)
                    );
                }
                calls
                    .into_iter()
                    .map(|(name, arguments)| {
                        let index = self.tool_index;
                        self.tool_index += 1;
                        self.trace_event(StructEvent::Tool {
                            index,
                            name: name.clone(),
                            arg_bytes: arguments.len(),
                        });
                        Item::ToolCall { index, name, arguments }
                    })
                    .collect()
            }
            Ok(_) => {
                self.trace_event(StructEvent::Malformed {
                    kind: "tool_call_anthropic_empty",
                    bytes: body.len(),
                });
                self.malformed += 1;
                vec![Item::Malformed {
                    kind: "tool_call_anthropic_empty",
                    raw: body.to_string(),
                }]
            }
            Err(kind) => {
                self.trace_event(StructEvent::Malformed { kind, bytes: body.len() });
                self.malformed += 1;
                vec![Item::Malformed { kind, raw: body.to_string() }]
            }
        }
    }

    /// 工具块体 → 完整调用(两格式;I4 保证 arguments 是合法 JSON 字符串)
    fn build_tool_call(&mut self, body: &str) -> Item {
        let body = body.trim();
        let index = self.tool_index;
        match parse_tool_body(body, &self.schemas) {
            Ok((name, args, repair)) => {
                if let Some(kind) = repair {
                    self.trace_event(StructEvent::Repair { tool: name.clone(), kind });
                    owl_shared::log_warn!(
                        "chat",
                        "工具调用近失格式已修复({kind};tool={name};args={}B)",
                        args.len()
                    );
                }
                if let Err(e) = self.schemas.validate(&name, &args) {
                    if e.is_soft() {
                        let prop = e.prop.clone().unwrap_or_default();
                        let expected = e.expected.clone().unwrap_or_default();
                        let got = e.got.clone().unwrap_or_default();
                        self.trace_event(StructEvent::SoftType {
                            tool: e.tool.clone(),
                            prop,
                            expected,
                            got,
                        });
                        owl_shared::log_warn!(
                            "chat",
                            "工具参数类型不匹配(软通过;{}; raw={})",
                            e.log_line(),
                            raw_for_log(body)
                        );
                    } else {
                        self.trace_event(StructEvent::Malformed {
                            kind: e.kind,
                            bytes: body.len(),
                        });
                        owl_shared::log_warn!(
                            "chat",
                            "工具调用硬拒绝({}; raw={})",
                            e.log_line(),
                            raw_for_log(body)
                        );
                        self.malformed += 1;
                        return Item::Malformed { kind: e.kind, raw: body.to_string() };
                    }
                }
                self.tool_index += 1;
                self.trace_event(StructEvent::Tool {
                    index,
                    name: name.clone(),
                    arg_bytes: args.len(),
                });
                Item::ToolCall { index, name, arguments: args }
            }
            Err(kind) => {
                self.trace_event(StructEvent::Malformed { kind, bytes: body.len() });
                self.malformed += 1;
                Item::Malformed { kind, raw: body.to_string() }
            }
        }
    }
}

/// 解析 Anthropic/Claude wrapper 内的一个或多个 `<invoke>`。
fn parse_anthropic_calls(
    body: &str,
    schemas: &ToolSchemas,
) -> Result<Vec<(String, String)>, &'static str> {
    let mut calls = Vec::new();
    let mut at = 0usize;
    while let Some(rel) = body[at..].find("<invoke") {
        let lt = at + rel;
        if !body[at..lt].chars().all(|c| c.is_ascii_whitespace()) {
            return Err("tool_call_anthropic_unexpected_text");
        }
        let gt = body[lt..]
            .find('>')
            .ok_or("tool_call_anthropic_bad_invoke")?
            + lt;
        let tag = &body[lt..=gt];
        let name = xml_attr(tag, "name").ok_or("tool_call_anthropic_no_name")?;
        if name.is_empty() {
            return Err("tool_call_anthropic_empty_name");
        }
        let inner_start = gt + 1;
        let close_rel = body[inner_start..]
            .find("</invoke>")
            .ok_or("tool_call_anthropic_unclosed_invoke")?;
        let inner = &body[inner_start..inner_start + close_rel];
        let args = parse_anthropic_params(&name, inner, schemas)?;
        calls.push((name, args));
        at = inner_start + close_rel + "</invoke>".len();
    }
    if !body[at..].chars().all(|c| c.is_ascii_whitespace()) {
        return Err("tool_call_anthropic_unexpected_text");
    }
    if calls.is_empty() {
        return Err("tool_call_anthropic_no_invoke");
    }
    Ok(calls)
}

/// `<invoke>` 内的 `<parameter name="…">值</parameter>` 列表 → JSON 字符串。
fn parse_anthropic_params(
    tool: &str,
    inner: &str,
    schemas: &ToolSchemas,
) -> Result<String, &'static str> {
    let mut obj = Map::new();
    let mut at = 0usize;
    while let Some(rel) = inner[at..].find("<parameter") {
        let lt = at + rel;
        if !inner[at..lt].chars().all(|c| c.is_ascii_whitespace()) {
            return Err("tool_call_anthropic_unexpected_text");
        }
        let gt = inner[lt..]
            .find('>')
            .ok_or("tool_call_anthropic_bad_param")?
            + lt;
        let tag = &inner[lt..=gt];
        // 两种拼法都收：Anthropic 的 `<parameter name="…">` 与模型混用的
        // Qwen `<parameter=…>`（只认 schema 里存在的属性，后面统一校验）。
        let name = xml_attr(tag, "name")
            .or_else(|| parameter_tag_name(tag))
            .ok_or("tool_call_anthropic_param_no_name")?;
        let value_start = gt + 1;
        let close_rel = inner[value_start..]
            .find("</parameter>")
            .ok_or("tool_call_anthropic_param_unclosed")?;
        let mut val = &inner[value_start..value_start + close_rel];
        if let Some(v) = val.strip_prefix('\n') {
            val = v;
        }
        if let Some(v) = val.strip_suffix('\n') {
            val = v;
        }
        if !name.is_empty() {
            obj.insert(name.clone(), schemas.coerce(tool, &name, val));
        }
        at = value_start + close_rel + "</parameter>".len();
    }
    if !inner[at..].chars().all(|c| c.is_ascii_whitespace()) {
        return Err("tool_call_anthropic_unexpected_text");
    }
    serde_json::to_string(&Json::Object(obj))
        .map_err(|_| "tool_call_anthropic_args_unserializable")
}

/// 从 `<invoke ...>` / `<parameter ...>` 标签里取一个带引号的属性。
fn xml_attr(tag: &str, attr: &str) -> Option<String> {
    let b = tag.as_bytes();
    let mut at = 0usize;
    while let Some(rel) = tag[at..].find(attr) {
        let i = at + rel;
        let before_ok = i == 0 || b[i - 1].is_ascii_whitespace();
        let after = i + attr.len();
        let after_ok = after >= b.len()
            || b[after].is_ascii_whitespace()
            || b[after] == b'=';
        if before_ok && after_ok {
            let mut j = skip_ws(tag, after);
            if b.get(j) != Some(&b'=') {
                at = after;
                continue;
            }
            j = skip_ws(tag, j + 1);
            let quote = *b.get(j)?;
            if quote != b'"' && quote != b'\'' {
                return None;
            }
            let start = j + 1;
            let end = tag[start..].find(quote as char)?;
            return Some(tag[start..start + end].to_string());
        }
        at = after;
    }
    None
}

/// 工具块体解析:json 格式 `{"name":…,"arguments":{…}}` 优先,其次 xml 格式
///
/// 返回值第三项 = 近失修复标记（仅当修复后的参数能通过 schema 校验才非 None）。
fn parse_tool_body(
    body: &str,
    schemas: &ToolSchemas,
) -> Result<(String, String, Option<&'static str>), &'static str> {
    // ① json 格式(template `tool_call_format="json"`)
    if body.starts_with('{') {
        let v: Json = serde_json::from_str(body).map_err(|_| "tool_call_json_invalid")?;
        let name = v.get("name").and_then(|n| n.as_str()).ok_or("tool_call_no_name")?;
        let args = v.get("arguments").cloned().unwrap_or_else(|| Json::Object(Map::new()));
        let s = serde_json::to_string(&args).map_err(|_| "tool_call_args_unserializable")?;
        return Ok((name.to_string(), s, None));
    }
    // ② `<tool_call>` 里混入 Anthropic 的 `<invoke>`（模型偶尔整块搬家）。
    if body.contains("<invoke") && !body.contains("<function=") {
        let calls = parse_anthropic_calls(body, schemas)?;
        if calls.len() == 1 {
            let (name, args) = calls.into_iter().next().ok_or("tool_call_anthropic_no_invoke")?;
            return Ok((name, args, Some("anthropic_inside_tool")));
        }
        return Err("tool_call_anthropic_multiple_in_tool");
    }
    // ③ xml 格式(template 默认):`<function=NAME> <parameter=K>V</parameter>… </function>`
    let open = body.find("<function=").ok_or("tool_call_no_function")?;
    let rest = &body[open + "<function=".len()..];
    let gt = rest.find('>').ok_or("tool_call_no_function")?;
    let name = rest[..gt].trim().to_string();
    if name.is_empty() {
        return Err("tool_call_empty_name");
    }
    let inner = &rest[gt + 1..];
    let mut obj = Map::new();
    let mut pos = 0usize;
    // 参数扫描 = vLLM qwen3_coder `_PARAM_RE` 同款口径:开标签 `<` parameter `=` NAME `>`
    // 与闭标签 `<` `/` parameter `>` 都容忍标签内空白;缺闭标签时,下一个开标签
    // 就是当前值的边界(vLLM 的前瞻交替)。
    while let Some(op) = find_param_open(inner, pos) {
        let close = find_param_close(inner, op.value_start);
        // 前瞻:下一个开标签的 **标签起点**(不是它的值起点)才是当前值的边界
        let next = find_param_open(inner, op.value_start).map(|n| n.tag_start);
        let (val_end, next_pos) = match (close, next) {
            (Some((e, x)), None) => (e, x),
            (Some((e, x)), Some(f)) => if e <= f { (e, x) } else { (f, f) },
            (None, Some(f)) => (f, f),
            // 既无闭标签也无后继开标签:与 vLLM 一样丢尾巴
            (None, None) => break,
        };
        // 模板写入形态 = 开标签 /n VALUE /n 闭标签:各去一个包裹换行
        let mut val = &inner[op.value_start..val_end];
        if let Some(v) = val.strip_prefix('\n') {
            val = v;
        }
        if let Some(v) = val.strip_suffix('\n') {
            val = v;
        }
        if !op.name.is_empty() {
            obj.insert(op.name.clone(), schemas.coerce(&name, &op.name, val));
        }
        pos = next_pos;
    }
    let s = serde_json::to_string(&Json::Object(obj.clone()))
        .map_err(|_| "tool_call_args_unserializable")?;
    if schemas.validate(&name, &s).is_ok() {
        return Ok((name, s, None));
    }
    // ④ 保守修复：模型把 `<parameter=command>` 写成 `<command>`、
    // `command="…"` 或 `<parameter name="command">`。只补 schema 里缺的属性，
    // 且修复后必须通过同一份 required/type 校验；否则原样交给上层报错。
    let mut loose = obj;
    if fill_qwen_loose(inner, &name, schemas, &mut loose) {
        let repaired = serde_json::to_string(&Json::Object(loose))
            .map_err(|_| "tool_call_args_unserializable")?;
        if schemas.validate(&name, &repaired).is_ok() {
            return Ok((name, repaired, Some("qwen_loose")));
        }
    }
    Ok((name, s, None))
}

/// `<function_calls>` 壳里写 Qwen `<function=…>` 块时的兜底解析（可多块）。
fn parse_qwen_blocks(
    body: &str,
    schemas: &ToolSchemas,
) -> Result<Vec<(String, String)>, &'static str> {
    let mut calls = Vec::new();
    let mut at = 0usize;
    while let Some(rel) = body[at..].find("<function=") {
        let start = at + rel;
        if !body[at..start].chars().all(|c| c.is_ascii_whitespace()) {
            return Err("tool_call_anthropic_unexpected_text");
        }
        let close_rel = body[start..]
            .find(MARK_FUNCTION_CLOSE)
            .ok_or("tool_call_anthropic_unclosed_invoke")?;
        let end = start + close_rel + MARK_FUNCTION_CLOSE.len();
        let (name, args, _) = parse_tool_body(&body[start..end], schemas)?;
        if let Err(e) = schemas.validate(&name, &args) {
            if !e.is_soft() {
                return Err(e.kind);
            }
        }
        calls.push((name, args));
        at = end;
    }
    if !body[at..].chars().all(|c| c.is_ascii_whitespace()) {
        return Err("tool_call_anthropic_unexpected_text");
    }
    if calls.is_empty() {
        return Err("tool_call_anthropic_no_invoke");
    }
    Ok(calls)
}

/// 从 `<parameter ...>` 标签里取 Qwen 风格的名字（`<parameter=command>`）。
fn parameter_tag_name(tag: &str) -> Option<String> {
    let rest = tag.strip_prefix('<')?.trim_start();
    let rest = rest.strip_prefix("parameter")?.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let end = rest.find('>')?;
    let name = rest[..end].trim();
    if name.is_empty() || name.chars().any(char::is_whitespace) {
        return None;
    }
    Some(name.to_string())
}

/// 只填 schema 中**缺失**的属性；不覆盖已解析出的标准参数。
fn fill_qwen_loose(
    inner: &str,
    tool: &str,
    schemas: &ToolSchemas,
    obj: &mut Map<String, Json>,
) -> bool {
    let Some(props) = schemas.types.get(tool) else { return false };
    let mut changed = false;
    for prop in props.keys() {
        if obj.get(prop).is_some_and(|v| !v.is_null()) {
            continue;
        }
        if let Some(raw) = find_loose_param(inner, prop) {
            obj.insert(prop.clone(), schemas.coerce(tool, prop, &raw));
            changed = true;
        }
    }
    changed
}

/// 在工具体里找一种近失参数写法（按 schema 属性名，保守且只认行首标签）。
fn find_loose_param(inner: &str, prop: &str) -> Option<String> {
    // ① `<parameter name="prop">value</parameter>`
    let mut at = 0usize;
    while let Some(rel) = inner[at..].find("<parameter") {
        let lt = at + rel;
        let gt = inner[lt..].find('>')? + lt;
        let tag = &inner[lt..=gt];
        if xml_attr(tag, "name").as_deref() == Some(prop)
            || parameter_tag_name(tag).as_deref() == Some(prop)
        {
            let value_start = gt + 1;
            let end = inner[value_start..]
                .find("</parameter>")
                .map(|p| value_start + p)
                .unwrap_or(inner.len());
            return Some(trim_wrapping_newline(&inner[value_start..end]).to_string());
        }
        at = gt + 1;
    }
    // ② 行首裸标签 `<prop>value</parameter>`（模型实测漏掉 `parameter=`）
    let bare = format!("<{prop}>");
    let mut at = 0usize;
    while let Some(rel) = inner[at..].find(&bare) {
        let lt = at + rel;
        let line_ok = lt == 0
            || inner.as_bytes()[lt - 1].is_ascii_whitespace()
            || inner.as_bytes()[lt - 1] == b'>';
        if line_ok {
            let value_start = lt + bare.len();
            let end = inner[value_start..]
                .find("</parameter>")
                .or_else(|| inner[value_start..].find(MARK_FUNCTION_CLOSE))
                .map(|p| value_start + p)
                .unwrap_or(inner.len());
            return Some(trim_wrapping_newline(&inner[value_start..end]).to_string());
        }
        at = lt + bare.len();
    }
    // ③ `prop="value"` / `prop='value'`（模型实测整行写成 key=value）
    let mut from = 0usize;
    while let Some(rel) = inner[from..].find(prop) {
        let i = from + rel;
        let before_ok = i == 0 || !inner.as_bytes()[i - 1].is_ascii_alphanumeric();
        let after = i + prop.len();
        let after_ok = inner.as_bytes().get(after).is_none_or(|b| !b.is_ascii_alphanumeric());
        if before_ok && after_ok {
            let mut j = after;
            while inner.as_bytes().get(j).is_some_and(|b| b.is_ascii_whitespace()) {
                j += 1;
            }
            if inner.as_bytes().get(j) == Some(&b'=') {
                j += 1;
                while inner.as_bytes().get(j).is_some_and(|b| b.is_ascii_whitespace()) {
                    j += 1;
                }
                if let Some(q @ (b'"' | b'\'')) = inner.as_bytes().get(j).copied() {
                    let start = j + 1;
                    if let Some(end) = inner[start..].find(q as char) {
                        return Some(inner[start..start + end].to_string());
                    }
                }
            }
        }
        from = after;
    }
    None
}

fn trim_wrapping_newline(s: &str) -> &str {
    let s = s.strip_prefix('\n').unwrap_or(s);
    s.strip_suffix('\n').unwrap_or(s)
}

/// 标签骨架的一跳:先跳过 ASCII 空白,再要求恰好是 `lit`。
/// 匹配 → 返回 `lit` 之后的位置;不匹配 → None(调用方自己推进游标,不会死循环)。
fn tag_step(s: &str, from: usize, lit: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut i = from;
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    if b[i..].starts_with(lit.as_bytes()) {
        Some(i + lit.len())
    } else {
        None
    }
}

/// 从 `from` 起跳过 ASCII 空白
fn skip_ws(s: &str, from: usize) -> usize {
    let b = s.as_bytes();
    let mut i = from;
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// 一个参数**开标签**的定位结果。
/// `tag_start` = 标签的 `<` 位置(前瞻边界要用它,不是值起点);
/// `value_start` = `>` 之后的值起点;`name` = `=` 与 `>` 之间的名字(已 trim)。
struct ParamOpen {
    tag_start: usize,
    value_start: usize,
    name: String,
}

/// 找下一个参数**开标签** `<` parameter `=` NAME `>`(NAME 取到 `>` 为止,可为空)。
fn find_param_open(s: &str, from: usize) -> Option<ParamOpen> {
    let mut at = from;
    while let Some(rel) = s[at..].find('<') {
        let lt = at + rel;
        if let Some(name_from) = tag_step(s, lt + 1, "parameter")
            .and_then(|k| tag_step(s, k, "="))
            .map(|k| skip_ws(s, k))
        {
            if let Some(rel_gt) = s[name_from..].find('>') {
                let gt = name_from + rel_gt;
                return Some(ParamOpen {
                    tag_start: lt,
                    value_start: gt + 1,
                    name: s[name_from..gt].trim().to_string(),
                });
            }
        }
        at = lt + 1;
    }
    None
}

/// 找参数**闭标签** `< / parameter >`。
/// 返回 (闭标签起点 = 值终点, 闭标签之后起点)。
fn find_param_close(s: &str, from: usize) -> Option<(usize, usize)> {
    let mut at = from;
    while let Some(rel) = s[at..].find('<') {
        let lt = at + rel;
        if let Some(gt_at) = tag_step(s, lt + 1, "/")
            .and_then(|k| tag_step(s, k, "parameter"))
            .map(|k| skip_ws(s, k))
        {
            if s.as_bytes().get(gt_at) == Some(&b'>') {
                return Some((lt, gt_at + 1));
            }
        }
        at = lt + 1;
    }
    None
}

/// 尾巴挂起长度:`buf` 的最长后缀,且该后缀是某个标记的**真前缀**
/// (标记最长 12 B ⇒ 最多挂 11 B,延迟 ≤ 数 token)
fn holdback_len(buf: &str, markers: &[&str]) -> usize {
    let max = markers.iter().map(|m| m.len()).max().unwrap_or(0);
    if max == 0 || buf.is_empty() {
        return 0;
    }
    let len = buf.len();
    let lo = len.saturating_sub(max - 1);
    let mut best = 0usize;
    for start in lo..len {
        if !buf.is_char_boundary(start) {
            continue;
        }
        let tail = &buf[start..];
        if markers.iter().any(|m| m.len() > tail.len() && m.starts_with(tail)) {
            best = best.max(tail.len());
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn split(items: &[Item]) -> String {
        let mut s = String::new();
        for it in items {
            if let Item::Delta { chan, text } = it {
                s.push_str(match chan {
                    Chan::Text => "T:",
                    Chan::Think => "R:",
                });
                s.push_str(text);
                s.push('|');
            }
        }
        s
    }

    fn tools() -> ToolSchemas {
        ToolSchemas::from_tools(&[
            json!({"type":"function","function":{
                "name":"bash",
                "parameters":{"type":"object","properties":{
                    "command":{"type":"string"},
                    "timeout":{"type":"number"},
                    "background":{"type":"boolean"}}}}}),
            json!({"type":"function","function":{
                "name":"read",
                "parameters":{"type":"object","properties":{
                    "path":{"type":"string"}}}}}),
        ])
    }

    /// 单块:思考 → 正文 → 工具调用(模板默认 xml 格式)
    #[test]
    fn one_shot_xml() {
        let mut s = ChannelSplitter::new(true, tools());
        let items = s.push(
            "我需要先看目录。\n</think>\n\n好的,我来执行。\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n<parameter=background>\ntrue\n</parameter>\n</function>\n</tool_call>",
        );
        // `</think>` 之后的 `\n\n` = **模板结构资产**(§4.3 R1)⇒ 不进 content:
        // 回填渲染时模板自己补回,少两个 token 才是对称的(2026-10-19 定谳)
        assert_eq!(split(&items), "R:我需要先看目录。\n|T:好的,我来执行。\n|", "通道序列");
        assert_eq!(
            items.last(),
            Some(&Item::ToolCall {
                index: 0,
                name: "bash".into(),
                arguments: r#"{"command":"ls -la","timeout":30,"background":true}"#.into()
            }),
            "xml 参数按 schema 回填类型"
        );
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.strays(), 0);
    }

    /// 逐字符推进(模拟跨 chunk 切半);I3 不得泄漏标记原文
    #[test]
    fn char_by_char() {
        let src = "想一下</think>答案<tool_call>\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n</tool_call>";
        let mut s = ChannelSplitter::new(true, tools());
        let mut items = Vec::new();
        for ch in src.chars() {
            let mut b = [0u8; 4];
            items.extend(s.push(ch.encode_utf8(&mut b)));
        }
        items.extend(s.finish());
        let text: String = items
            .iter()
            .filter_map(|i| match i {
                Item::Delta { chan: Chan::Text, text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "答案", "正文通道只有答案(标记不泄漏)");
        let think: String = items
            .iter()
            .filter_map(|i| match i {
                Item::Delta { chan: Chan::Think, text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(think, "想一下");
        assert!(items
            .iter()
            .any(|i| matches!(i, Item::ToolCall { name, .. } if name == "bash")));
    }

    /// GGUF 上游模板实测会漏 `<tool_call>` 外壳：裸 `<function=...>` 块
    /// 也必须按工具调用解析（函数名/参数形态与 XML 外壳内完全一致）。
    #[test]
    fn bare_function_block_without_tool_call_wrapper() {
        let src = "\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n";
        let mut s = ChannelSplitter::new(false, tools());
        let mut items = Vec::new();
        for ch in src.chars() {
            let mut b = [0u8; 4];
            items.extend(s.push(ch.encode_utf8(&mut b)));
        }
        items.extend(s.finish());
        let calls: Vec<(usize, &str, &str)> = items
            .iter()
            .filter_map(|i| match i {
                Item::ToolCall { index, name, arguments } => {
                    Some((*index, name.as_str(), arguments.as_str()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(calls, vec![(0, "bash", r#"{"command":"pwd"}"#)]);
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.malformed(), 0);
        assert!(!items.iter().any(|i| matches!(i, Item::Malformed { .. })));
    }

    /// vLLM qwen3_coder 同款容错：第一个参数缺 `</parameter>` 时，
    /// 下一个 `<parameter=` 是它的边界，仍应完整解析。
    #[test]
    fn missing_parameter_close_uses_next_parameter_boundary() {
        let mut s = ChannelSplitter::new(false, tools());
        let items = s.push(
            "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>",
        );
        let calls: Vec<&Item> = items.iter().filter(|i| matches!(i, Item::ToolCall { .. })).collect();
        assert_eq!(calls.len(), 1);
        assert!(matches!(
            calls[0],
            Item::ToolCall { name, arguments, .. }
                if name == "bash" && arguments == r#"{"command":"pwd","timeout":30}"#
        ));
        assert_eq!(s.malformed(), 0);
    }

    /// `</function>` 已闭合但漏了 `</tool_call>`（vLLM TOOL_BETWEEN 同款）：仍发工具。
    #[test]
    fn missing_tool_call_close_after_function_is_tolerated() {
        let mut s = ChannelSplitter::new(false, tools());
        let items = s.push(
            "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>",
        );
        assert!(items.iter().any(|i| matches!(
            i,
            Item::ToolCall { name, arguments, .. }
                if name == "bash" && arguments == r#"{"command":"pwd"}"#
        )));
        assert_eq!(s.malformed(), 0);
    }

    /// 同一 `<tool_call>` 内连续两个 `<function>`（中间没有 tool 外壳）。
    #[test]
    fn multiple_functions_inside_one_tool_call() {
        let mut s = ChannelSplitter::new(false, tools());
        let items = s.push(
            "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>",
        );
        let calls: Vec<&Item> = items.iter().filter(|i| matches!(i, Item::ToolCall { .. })).collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls.iter().map(|i| match i { Item::ToolCall { arguments, .. } => arguments.as_str(), _ => "" }).collect::<Vec<_>>(),
            vec![r#"{"command":"pwd"}"#, r#"{"command":"ls"}"#]
        );
        assert_eq!(s.malformed(), 0);
    }

    /// between 态跨 chunk 切断标记:结构空白与半截标记都不得泄进正文。
    /// (旧实现逐字符推进时会提前转 Text,把 function-close 之后的换行当正文
    /// 吐出 —— 这正是 char_by_char 翻车的机制,本用例钉死跨 chunk 形态)
    #[test]
    fn between_tools_state_survives_chunk_splits() {
        let src = "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";
        let mut s = ChannelSplitter::new(false, tools());
        let mut items = Vec::new();
        let bytes = src.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // 本串全 ASCII,3 字节一步可把任意标记切成半截
            let j = (i + 3).min(bytes.len());
            items.extend(s.push(std::str::from_utf8(&bytes[i..j]).unwrap()));
            i = j;
        }
        items.extend(s.finish());
        let text: String = items
            .iter()
            .filter_map(|it| match it {
                Item::Delta { chan: Chan::Text, text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "", "between 态结构空白不进正文: {items:?}");
        assert_eq!(s.tool_calls(), 2, "连续两个 function 块都要解析");
        assert_eq!(s.malformed(), 0);
    }

    /// function-close 之后模型接着写正文(漏了 tool-call 壳):正文照常出,
    /// 但夹在中间的结构换行归模板(与 vLLM TOOL_BETWEEN 丢空白同口径)。
    #[test]
    fn prose_after_function_block_still_streams() {
        let mut s = ChannelSplitter::new(false, tools());
        let mut items = s.push("<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>");
        items.extend(s.push("done"));
        items.extend(s.finish());
        let text: String = items
            .iter()
            .filter_map(|it| match it {
                Item::Delta { chan: Chan::Text, text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "done", "正文接得上且不泄漏标记: {items:?}");
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.malformed(), 0);
    }

    /// vLLM `_PARAM_RE` 容忍标签内空白(`< parameter = name >` 与 `闭标签`),
    /// 且参数值里的换行必须原样保留(模板只吃首尾各一个换行)。
    #[test]
    fn tolerant_param_tags_and_multiline_values() {
        let schema = ToolSchemas::from_tools(&[json!({"type":"function","function":{
            "name":"bash",
            "parameters":{"type":"object","properties":{
                "command":{"type":"string"},
                "count":{"type":"integer"}
            }}}})]);
        // 标签内故意加空白:开标签 " parameter = command "、闭标签 " / parameter "
        let body = "<tool_call>\n<function=bash>\n< parameter = command >\n\
                    echo a\nb\n< / parameter >\n<parameter=count>\n7\n</parameter>\n\
                    </function>\n</tool_call>";
        let mut s = ChannelSplitter::new(false, schema);
        let items = s.push(body);
        let call = items.iter().find_map(|it| match it {
            Item::ToolCall { arguments, .. } => Some(arguments.clone()),
            _ => None,
        });
        assert_eq!(
            call.as_deref(),
            Some(r#"{"command":"echo a\nb","count":7}"#),
            "空白标签 + 多行值 + 类型回填"
        );
        assert_eq!(s.malformed(), 0, "宽容扫描不得把合法块判成畸形");
    }

    /// Pydantic v2 的 Optional[T] 是 anyOf；类型回填取非 null 分支。
    #[test]
    fn anyof_schema_types_are_coerced() {
        let schema = ToolSchemas::from_tools(&[json!({"type":"function","function":{
            "name":"typed",
            "parameters":{"type":"object","properties":{
                "count":{"anyOf":[{"type":"integer"},{"type":"null"}]},
                "flag":{"anyOf":[{"type":"boolean"},{"type":"null"}]},
                "ratio":{"anyOf":[{"type":"float"},{"type":"null"}]}
            }}}})]);
        let mut s = ChannelSplitter::new(false, schema);
        let items = s.push(
            "<tool_call>\n<function=typed>\n<parameter=count>\n42\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n<parameter=ratio>\n3.14\n</parameter>\n</function>\n</tool_call>",
        );
        assert!(items.iter().any(|i| matches!(
            i,
            Item::ToolCall { name, arguments, .. }
                if name == "typed" && arguments == r#"{"count":42,"flag":true,"ratio":3.14}"#
        )));
    }

    /// 多工具块 + I5 序号连续
    #[test]
    fn multi_tool_calls_and_index() {
        let mut s = ChannelSplitter::new(true, tools());
        let items = s.push(
            "推理\n\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=read>\n<parameter=path>\n/a.txt\n</parameter>\n</function>\n</tool_call>",
        );
        let calls: Vec<(usize, &str)> = items
            .iter()
            .filter_map(|i| match i {
                Item::ToolCall { index, name, .. } => Some((*index, name.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(calls, vec![(0, "bash"), (1, "read")]);
    }

    /// json 格式(template `tool_call_format="json"`)= 类型原样
    #[test]
    fn json_format() {
        let mut s = ChannelSplitter::new(false, ToolSchemas::default());
        let items = s.push(
            "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\", \"timeout\": 30}}\n</tool_call>",
        );
        // 未开思考 ⇒ 首个标记就是 `<tool_call>`,正文为空
        assert_eq!(
            items,
            vec![Item::ToolCall {
                index: 0,
                name: "bash".into(),
                arguments: r#"{"command":"ls","timeout":30}"#.into()
            }]
        );
    }

    /// 未闭合工具块 ⇒ Malformed(不发射假工具调用;不静默)
    #[test]
    fn unclosed_tool_call_is_malformed() {
        let mut s = ChannelSplitter::new(false, tools());
        assert!(s
            .push("<tool_call>\n<function=bash>\n<parameter=command>\nls")
            .is_empty());
        let f = s.finish();
        assert!(matches!(f[0], Item::Malformed { kind: "tool_call_unclosed", .. }));
        assert_eq!(s.tool_calls(), 0);
    }

    /// 无 `<function=` ⇒ Malformed(而非把原文吐进正文:I3)
    #[test]
    fn malformed_body_is_not_leaked_as_text() {
        let mut s = ChannelSplitter::new(false, tools());
        let items = s.push("<tool_call>\ngarbage\n</tool_call>hello");
        assert!(matches!(items[0], Item::Malformed { kind: "tool_call_no_function", .. }));
        assert_eq!(split(&items), "T:hello|");
    }

    /// 尾巴挂起只对**真前缀**生效:裸 `<` 挂起等下一块,`</thin` 这种非前缀立即放行
    #[test]
    fn holdback_only_for_real_prefixes() {
        let mut s = ChannelSplitter::new(false, tools());
        // `<` 是 `<thinking>`/`<tool_call>` 的真前缀 ⇒ 挂起
        assert_eq!(split(&s.push("a <")), "T:a |");
        // 下一块证明它只是普通字符 ⇒ 连挂起的一起放行
        assert_eq!(split(&s.push("b")), "T:<b|");
        // `</thin` 是长形态别名的真前缀 ⇒ 继续挂起(提前放行会让标记漏出去)
        assert_eq!(split(&s.push(" and </thin")), "T: and |");
        // 下一块证明不是标记 ⇒ 连挂起部分一起放行(**不得永久挂起**)
        assert_eq!(split(&s.push("X")), "T:</thinX|");
        assert_eq!(split(&s.finish()), "");
    }

    /// 关思考起步(预填已含空 think 块)⇒ 直接正文;模型仍吐 `</think>` 时吞掉不泄漏
    #[test]
    fn start_in_text_swallows_stray_close() {
        let mut s = ChannelSplitter::new(false, tools());
        let items = s.push("直接回答\n\n</think>\n\n续答");
        assert_eq!(split(&items), "T:直接回答\n\n|T:\n\n续答|");
        assert_eq!(s.strays(), 1, "游标标记要计数(上限层记日志)");
    }

    /// **G5 `reasoning_tokens` 真源**:按 `push` 入口状态归因。
    /// ①开思考起步 ⇒ 思考 token 与 `</think>` 本体计,正文 token 不计;
    /// ②关思考起步 ⇒ 恒 0(即使模型吐游标 `</think>` 也不计);
    /// ③未关思考直接开工具 ⇒ 只计入工具前一 token,工具态不再计。
    #[test]
    fn reasoning_steps_are_phased_by_channel() {
        let mut s = ChannelSplitter::new(true, tools());
        s.push("想");        // Think ⇒ 1
        s.push("</think>"); // 入口仍 Think(标记本体属思考段)⇒ 2
        s.push("答");        // 入口 Text ⇒ 不计
        s.push("案");
        assert_eq!(s.reasoning_steps(), 2);

        let mut s = ChannelSplitter::new(false, tools());
        s.push("直接回答");
        s.push("</think>"); // 游标:Text 入口 ⇒ 不计
        assert_eq!(s.reasoning_steps(), 0);

        let mut s = ChannelSplitter::new(true, tools());
        s.push("<tool_call>"); // 入口 Think ⇒ 1(进工具态)
        s.push("<function=bash>"); // InTool ⇒ 不计
        assert_eq!(s.reasoning_steps(), 1);
    }

    /// **G5② 结构轨迹**:只记结构事件(绝不记 delta 文本),行序 = 发生序,
    /// 畸形计数与事件同源。
    #[test]
    fn structural_trace_records_structure_not_text() {
        let mut s = ChannelSplitter::new(true, tools());
        s.push("想");        // 无结构事件(纯内容)
        s.push("</think>"); // think→text
        s.push("\n\n答");    // 吞两个结构换行
        let tr = s.structural_trace();
        assert!(tr.iter().any(|l| l == "state think→text"), "{tr:?}");
        assert!(tr.iter().any(|l| l == "swallow_lf 2"), "{tr:?}");
        assert!(
            !tr.iter().any(|l| l.contains('想') || l.contains('答')),
            "轨迹不得含 delta 文本:{tr:?}"
        );

        // 工具调用:tool 事件带序号/名/参数字节;正常路径 malformed = 0
        let mut s = ChannelSplitter::new(true, tools());
        let items = s.push(
            "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>",
        );
        assert!(items.iter().any(|i| matches!(i, Item::ToolCall { .. })));
        let tr = s.structural_trace();
        assert!(tr.iter().any(|l| l.starts_with("tool #0 bash")), "{tr:?}");
        assert!(tr.iter().any(|l| l == "state think→tool"), "{tr:?}");
        assert_eq!(s.malformed(), 0);

        // 未闭合工具块:malformed 计数 + 事件(与日志同源,不静默)
        let mut s = ChannelSplitter::new(false, tools());
        let _ = s.push("<tool_call><function=bash>");
        let _ = s.finish();
        assert_eq!(s.malformed(), 1, "未闭合要计数");
        assert!(
            s.structural_trace()
                .iter()
                .any(|l| l.starts_with("malformed tool_call_unclosed")),
            "{:?}",
            s.structural_trace()
        );
    }

    /// I4:任何出口的 arguments 必须能被 JSON 解析
    #[test]
    fn arguments_always_valid_json() {
        let mut s = ChannelSplitter::new(true, tools());
        let items = s.push(
            "t\n\n</think>\n<tool_call>\n<function=bash>\n<parameter=command>\ngrep -n \"a\" f | wc -l\n</parameter>\n</function>\n</tool_call>",
        );
        let call = items
            .iter()
            .find_map(|i| match i {
                Item::ToolCall { arguments, .. } => Some(arguments),
                _ => None,
            })
            .expect("有工具调用");
        let v: Json = serde_json::from_str(call).expect("arguments 必须合法 JSON");
        assert_eq!(v["command"], "grep -n \"a\" f | wc -l");
    }

    /// 无 schema 时保守保持字符串(不猜类型 —— 路径 "123" 不能被变成数字)
    #[test]
    fn no_schema_keeps_string() {
        let mut s = ChannelSplitter::new(false, ToolSchemas::default());
        let items = s.push(
            "<tool_call>\n<function=f>\n<parameter=p>\n123\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(
            items[0],
            Item::ToolCall { index: 0, name: "f".into(), arguments: r#"{"p":"123"}"#.into() }
        );
    }

    /// **标记 × 快照耦合门**:解析器的标记字面量必须真的出现在快照渲染产物里
    /// (仓库内金标,恒跑);模板源(需 `tpl:default`)必须留长形态别名。
    /// 模板漂移(上游改标记名)会让解析器**静默失效** —— 这条是警报器。
    #[test]
    fn markers_match_snapshot() {
        let fx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat_tpl");
        let think_on = std::fs::read_to_string(fx.join("ctx1_tools.txt")).unwrap();
        let think_off = std::fs::read_to_string(fx.join("ctx3_tools_notinking.txt")).unwrap();
        assert!(think_on.contains(MARK_THINK_OPEN), "tools 档起手应含主开标记");
        assert!(think_on.contains(MARK_TOOL_OPEN), "tools 档应含工具开标记");
        assert!(think_on.contains(MARK_TOOL_CLOSE), "tools 档应含工具闭标记");
        assert!(think_off.contains(MARK_THINK_CLOSE), "关思考档应含主闭标记");
        match owl_shared::testkit::resource("tpl:default") {
            Some(dir) => {
                let src = std::fs::read_to_string(
                    std::path::Path::new(&dir).join("chat_template.jinja"),
                )
                .unwrap();
                assert!(src.contains(MARK_THINK_OPEN_LONG), "模板源应含长形态别名");
            }
            None => eprintln!("skip(部分): 资源 tpl:default 不在,跳过模板源检查"),
        }
    }

    /// 别名与主形态等价(模型偶尔混用 `thinking` / `thought` 形态;
    /// 认它 = 吞掉不泄漏,否则原文会漏进正文/思考面板)
    #[test]
    fn think_alias_forms_are_recognized() {
        for close in ["</thinking>", "</thought>"] {
            let mut s = ChannelSplitter::new(true, tools());
            let items = s.push(&(String::from("思考") + close + "正文"));
            assert_eq!(split(&items), "R:思考|T:正文|", "{close} 应被识别");
            assert_eq!(s.strays(), 0, "{close} 是别名,不是游标标记");
        }
        // 开标记同理(Text 态遇 `<thought>` 应进 Think 通道)
        let mut s = ChannelSplitter::new(false, tools());
        let items = s.push("<thought>推理</thought>答案");
        assert_eq!(split(&items), "R:推理|T:答案|");
    }

    /// `[agent] parse_tool_calls = false`(诊断/回退臂):工具标记**按普通文本
    /// 放行**,而思考通道不受影响
    #[test]
    fn tool_parsing_can_be_disabled() {
        let mut s = ChannelSplitter::new(false, tools()).with_tool_parsing(false);
        let items = s.push("<tool_call><function=bash>ls</function></tool_call>");
        assert_eq!(s.tool_calls(), 0, "关解析 ⇒ 零工具调用");
        assert_eq!(
            split(&items),
            "T:<tool_call><function=bash>ls</function></tool_call>|"
        );
        // 思考通道仍工作(两件事互不耦合)
        let mut s = ChannelSplitter::new(false, tools()).with_tool_parsing(false);
        let items = s.push("<think>思考</think>正文");
        assert_eq!(split(&items), "R:思考|T:正文|");
    }
    /// **律(§4.3 R1,2026-10-19)**:`</think>` 之后的至多两个 `\n` 是模板
    /// 结构资产(`…\n</think>\n\n{content}`),**不进任何通道** —— 否则客户端
    /// 回填后渲染会多出两个 token,会话守卫在第一个 content token 失配。
    /// 本测钉三件:①吞两个 ②只有一个时吞一个 ③没有时不吞(`答案` 原样)。
    #[test]
    fn think_close_boundary_whitespace_is_structural() {
        let f = |src: &str| {
            let mut s = ChannelSplitter::new(true, ToolSchemas::default());
            let mut all = split(&s.push(src));
            all.push_str(&split(&s.finish()));
            all
        };
        assert_eq!(f("想</think>\n\n答案"), "R:想|T:答案|");
        assert_eq!(f("想</think>\n答案"), "R:想|T:答案|", "只有一个换行时吞一个");
        assert_eq!(f("想</think>答案"), "R:想|T:答案|", "无换行 ⇒ 不吞");
        assert_eq!(f("想</think>\n\n\n答案"), "R:想|T:\n答案|", "只吞两个(其余归正文)");
    }

    /// **流式切半**:结构空白跨 chunk 到达时必须**挂起**(不许先吐给 content
    /// 再纠正 —— 通道一旦发出就收不回;客户端会把这两个换行回填)。
    #[test]
    fn think_close_boundary_survives_chunk_split() {
        let mut s = ChannelSplitter::new(true, ToolSchemas::default());
        // 逐字符切分(最坏情形):通道文本**合并后**比对 —— 分几帧无所谓,
        // 关键是内容:帧边界不得让结构空白漏进 content
        let mut merged: Vec<(Chan, String)> = Vec::new();
        let mut absorb = |items: Vec<Item>| {
            for it in items {
                if let Item::Delta { chan, text } = it {
                    match merged.last_mut() {
                        Some((c, t)) if *c == chan => t.push_str(&text),
                        _ => merged.push((chan, text)),
                    }
                }
            }
        };
        for piece in ["想", "</think>", "\n", "\n", "答", "案"] {
            absorb(s.push(piece));
        }
        absorb(s.finish());
        let joined: Vec<String> = merged
            .iter()
            .map(|(c, t)| {
                let tag = match c {
                    Chan::Text => "T",
                    Chan::Think => "R",
                };
                format!("{tag}:{t}")
            })
            .collect();
        assert_eq!(joined.join("|"), "R:想|T:答案", "切半不得漏出结构空白");
    }

    fn anthropic_tools() -> ToolSchemas {
        ToolSchemas::from_tools(&[
            json!({"type":"function","function":{
                "name":"screenshot",
                "parameters":{"type":"object","properties":{}}}}),
            json!({"type":"function","function":{
                "name":"click",
                "parameters":{"type":"object","properties":{
                    "x":{"type":"integer"},
                    "y":{"type":"integer"}}}}}),
        ])
    }

    /// Anthropic/Claude 风格：`<function_calls><invoke name="…">`。
    /// 27B 在长 system prompt 下实测会吐这个壳，而不是 Qwen 的 `<tool_call>`。
    #[test]
    fn anthropic_function_calls_single() {
        let mut s = ChannelSplitter::new(true, anthropic_tools());
        let items = s.push(
            "我先看屏幕。\n</think>\n\n<function_calls>\n<invoke name=\"screenshot\">\n</invoke>\n</function_calls>",
        );
        assert_eq!(split(&items), "R:我先看屏幕。\n|", "正文不得泄漏 wrapper");
        assert_eq!(
            items.last(),
            Some(&Item::ToolCall {
                index: 0,
                name: "screenshot".into(),
                arguments: "{}".into(),
            })
        );
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.malformed(), 0);
        assert_eq!(s.strays(), 0);
    }

    /// 一个 wrapper 内多个 invoke；参数按 schema 回填类型；wrapper 后正文继续。
    #[test]
    fn anthropic_function_calls_multiple() {
        let mut s = ChannelSplitter::new(false, anthropic_tools());
        let items = s.push(
            "<function_calls>\n\
             <invoke name=\"click\">\n\
             <parameter name=\"x\">64</parameter>\n\
             <parameter name=\"y\">64</parameter>\n\
             </invoke>\n\
             <invoke name=\"click\">\n\
             <parameter name=\"x\">192</parameter>\n\
             <parameter name=\"y\">192</parameter>\n\
             </invoke>\n\
             </function_calls>done",
        );
        let calls: Vec<(usize, &str, &str)> = items
            .iter()
            .filter_map(|i| match i {
                Item::ToolCall { index, name, arguments } => {
                    Some((*index, name.as_str(), arguments.as_str()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            calls,
            vec![
                (0, "click", r#"{"x":64,"y":64}"#),
                (1, "click", r#"{"x":192,"y":192}"#),
            ]
        );
        assert_eq!(split(&items), "T:done|", "wrapper 后正文继续");
        assert_eq!(s.tool_calls(), 2);
        assert_eq!(s.malformed(), 0);
    }

    /// 逐字符切分：wrapper/invoke/parameter 跨 chunk 时不得漏进正文。
    #[test]
    fn anthropic_function_calls_survive_chunk_split() {
        let src = "好，你会运维这个推特吗？\n<function_calls>\n\
                   <invoke name=\"screenshot\">\n\
                   </invoke>\n\
                   </function_calls>";
        let mut s = ChannelSplitter::new(false, anthropic_tools());
        let mut items = Vec::new();
        for ch in src.chars() {
            let mut b = [0u8; 4];
            items.extend(s.push(ch.encode_utf8(&mut b)));
        }
        items.extend(s.finish());
        let text: String = items
            .iter()
            .filter_map(|i| match i {
                Item::Delta { chan: Chan::Text, text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "好，你会运维这个推特吗？\n");
        assert!(items.iter().any(|i| matches!(
            i,
            Item::ToolCall { name, arguments, .. } if name == "screenshot" && arguments == "{}"
        )));
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.malformed(), 0);
    }

    /// 未闭合 wrapper ⇒ Malformed，不发射假工具调用。
    #[test]
    fn anthropic_function_calls_unclosed_is_malformed() {
        let mut s = ChannelSplitter::new(false, anthropic_tools());
        assert!(s.push("<function_calls><invoke name=\"screenshot\">").is_empty());
        let f = s.finish();
        assert!(matches!(
            f[0],
            Item::Malformed { kind: "tool_call_anthropic_unclosed", .. }
        ));
        assert_eq!(s.tool_calls(), 0);
    }

}

#[cfg(test)]
mod tool_validation_tests {
    use super::*;
    use serde_json::json;

    fn schemas() -> ToolSchemas {
        ToolSchemas::from_tools(&[
            json!({"type":"function","function":{
                "name":"bash",
                "parameters":{"type":"object",
                    "properties":{"command":{"type":"string"},"timeout":{"type":"number"}},
                    "required":["command"]}}}),
            json!({"type":"function","function":{
                "name":"screenshot",
                "parameters":{"type":"object","properties":{}}}}),
        ])
    }

    #[test]
    fn missing_required_and_unknown_tool_are_malformed() {
        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push("<tool_call><function=bash></function></tool_call>");
        assert!(matches!(
            items[0],
            Item::Malformed { kind: "tool_call_missing_required", .. }
        ));
        assert_eq!(s.tool_calls(), 0);

        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push("<tool_call><function=nope></function></tool_call>");
        assert!(matches!(
            items[0],
            Item::Malformed { kind: "tool_call_unknown_tool", .. }
        ));
        assert_eq!(s.tool_calls(), 0);
    }

    #[test]
    fn valid_required_and_anthropic_calls_pass() {
        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push(
            "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>",
        );
        assert!(matches!(
            &items[0],
            Item::ToolCall { name, arguments, .. }
                if name == "bash" && arguments == r#"{"command":"ls"}"#
        ));

        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push(
            "<function_calls><invoke name=\"bash\"><parameter name=\"command\">pwd</parameter></invoke></function_calls>",
        );
        assert!(matches!(
            &items[0],
            Item::ToolCall { name, arguments, .. }
                if name == "bash" && arguments == r#"{"command":"pwd"}"#
        ));
    }

    fn video_schemas() -> ToolSchemas {
        ToolSchemas::from_tools(&[json!({"type":"function","function":{
            "name":"video_collect",
            "parameters":{"type":"object",
                "properties":{
                    "action":{"type":"string"},
                    "url":{"type":"string"},
                    "max_download_mb":{"type":"integer"},
                    "tags":{"type":"array"},
                    "dry_run":{"type":"boolean"}
                },
                "required":["action","url"]
            }}})])
    }

    /// 自定义工具常见近失：数字/数组/布尔被 JSON 引号包住。
    /// 这些必须在 XML → JSON 回填时吃掉，否则整轮 502 的信息量只有那一个字段。
    #[test]
    fn quoted_number_array_bool_are_coerced() {
        let mut s = ChannelSplitter::new(false, video_schemas());
        let items = s.push(
            "<tool_call><function=video_collect>\n\
             <parameter=action>process</parameter>\n\
             <parameter=url>https://example.com/v</parameter>\n\
             <parameter=max_download_mb>\"512\"</parameter>\n\
             <parameter=tags>\"[\\\"ai\\\",\\\"tech\\\"]\"</parameter>\n\
             <parameter=dry_run>\"true\"</parameter>\n\
             </function></tool_call>",
        );
        let args = match &items[0] {
            Item::ToolCall { name, arguments, .. } if name == "video_collect" => {
                serde_json::from_str::<Json>(arguments).expect("参数 JSON")
            }
            other => panic!("应为 ToolCall，得到 {other:?}"),
        };
        assert_eq!(args["max_download_mb"], json!(512));
        assert_eq!(args["tags"], json!(["ai", "tech"]));
        assert_eq!(args["dry_run"], json!(true));
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.malformed(), 0);
    }

    /// 20.0 对 integer 不算错，归一成 20。
    #[test]
    fn float_integral_is_normalized_for_integer() {
        let mut s = ChannelSplitter::new(false, video_schemas());
        let items = s.push(
            "<tool_call><function=video_collect>\n\
             <parameter=action>process</parameter>\n\
             <parameter=url>https://example.com/v</parameter>\n\
             <parameter=max_download_mb>20.0</parameter>\n\
             </function></tool_call>",
        );
        assert!(matches!(
            &items[0],
            Item::ToolCall { name, arguments, .. }
                if name == "video_collect" && arguments.contains(r#""max_download_mb":20"#)
        ));
        assert_eq!(s.malformed(), 0);
    }

    /// 仍然无法归一（如 `512MB`）⇒ arg_type **软通过**，保留原值，
    /// 不再整轮 502；由远端工具/前端决定如何处置。
    #[test]
    fn arg_type_is_soft_not_whole_turn_error() {
        let mut s = ChannelSplitter::new(false, video_schemas());
        let items = s.push(
            "<tool_call><function=video_collect>\n\
             <parameter=action>process</parameter>\n\
             <parameter=url>https://example.com/v</parameter>\n\
             <parameter=max_download_mb>512MB</parameter>\n\
             </function></tool_call>",
        );
        assert!(matches!(
            &items[0],
            Item::ToolCall { name, arguments, .. }
                if name == "video_collect" && arguments.contains(r#""max_download_mb":"512MB""#)
        ));
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.malformed(), 0);
        assert!(
            s.structural_trace().iter().any(|l| l.contains("soft_type video_collect.max_download_mb")),
            "软错误必须留结构轨迹: {:?}",
            s.structural_trace()
        );
    }

    /// 软通过只限 arg_type；缺 required / 未知工具仍然硬拒。
    #[test]
    fn hard_errors_remain_hard() {
        let mut s = ChannelSplitter::new(false, video_schemas());
        let items = s.push(
            "<tool_call><function=video_collect><parameter=action>process</parameter></function></tool_call>",
        );
        assert!(matches!(
            items[0],
            Item::Malformed { kind: "tool_call_missing_required", .. }
        ));

        let mut s = ChannelSplitter::new(false, video_schemas());
        let items = s.push(
            "<tool_call><function=nope><parameter=action>process</parameter></function></tool_call>",
        );
        assert!(matches!(
            items[0],
            Item::Malformed { kind: "tool_call_unknown_tool", .. }
        ));
    }

    /// 实测近失 ①：模型把 `<parameter=command>` 写成裸 `<command>`。
    /// 只在 schema 里确有 command、且修复后能过 required/type 校验时才收。
    #[test]
    fn qwen_bare_parameter_tag_is_repaired() {
        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push(
            "<tool_call><function=bash>\n<command>\nls -la\n</parameter>\n</function></tool_call>",
        );
        assert!(matches!(
            &items[0],
            Item::ToolCall { name, arguments, .. }
                if name == "bash" && arguments == r#"{"command":"ls -la"}"#
        ));
        assert!(s.structural_trace().iter().any(|l| l.contains("repair bash qwen_loose")));
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.malformed(), 0);
    }

    /// 实测近失 ②：整行写成 `command="…"`（没有 XML 参数标签）。
    #[test]
    fn qwen_key_value_parameter_is_repaired() {
        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push(
            "<tool_call><function=bash>\ncommand=\"find . -type f\"\n</parameter>\n</function></tool_call>",
        );
        assert!(matches!(
            &items[0],
            Item::ToolCall { name, arguments, .. }
                if name == "bash" && arguments == r#"{"command":"find . -type f"}"#
        ));
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.malformed(), 0);
    }

    /// 实测近失 ③：Qwen 工具体里混用 Anthropic 的 `<parameter name="…">`。
    #[test]
    fn qwen_parameter_name_attribute_is_repaired() {
        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push(
            "<tool_call><function=bash><parameter name=\"command\">pwd</parameter></function></tool_call>",
        );
        assert!(matches!(
            &items[0],
            Item::ToolCall { name, arguments, .. }
                if name == "bash" && arguments == r#"{"command":"pwd"}"#
        ));
        assert_eq!(s.tool_calls(), 1);
        assert_eq!(s.malformed(), 0);
    }

    /// Anthropic wrapper 里混用 Qwen 的 `<parameter=…>`；以及 wrapper 里写
    /// Qwen `<function=…>` 块时的兜底。
    #[test]
    fn mixed_anthropic_qwen_tool_bodies_are_accepted() {
        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push(
            "<function_calls><invoke name=\"bash\"><parameter=command>pwd</parameter></invoke></function_calls>",
        );
        assert!(matches!(
            &items[0],
            Item::ToolCall { name, arguments, .. }
                if name == "bash" && arguments == r#"{"command":"pwd"}"#
        ));

        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push(
            "<function_calls><function=bash><parameter=command>ls</parameter></function></function_calls>",
        );
        assert!(matches!(
            &items[0],
            Item::ToolCall { name, arguments, .. }
                if name == "bash" && arguments == r#"{"command":"ls"}"#
        ));
        assert_eq!(s.malformed(), 0);
    }

    /// 修复不能把未知标签硬塞成参数：没有可匹配的 schema 属性时仍报缺参。
    #[test]
    fn unknown_bare_tag_is_not_repaired() {
        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push(
            "<tool_call><function=bash><nope>x</nope></function></tool_call>",
        );
        assert!(matches!(
            items[0],
            Item::Malformed { kind: "tool_call_missing_required", .. }
        ));
        assert_eq!(s.tool_calls(), 0);
    }

    #[test]
    fn anthropic_missing_required_is_whole_block_malformed() {
        let mut s = ChannelSplitter::new(false, schemas());
        let items = s.push(
            "<function_calls><invoke name=\"bash\"></invoke><invoke name=\"screenshot\"></invoke></function_calls>",
        );
        assert!(matches!(
            items[0],
            Item::Malformed { kind: "tool_call_missing_required", .. }
        ));
        assert_eq!(s.tool_calls(), 0, "非法组不得发射半组调用");
    }
}
