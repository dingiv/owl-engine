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
const MARK_THINK_OPEN_LONG: &str = "<thinking>";
const MARK_THINK_CLOSE_LONG: &str = "</thinking>";
// 第二组别名:模型偶尔混用 `thought`(实测 2026-10-14 出现过一次)。
// 认它 = 该标记被吞掉不进正文/思考面板,而不是当普通文本泄漏出去
const MARK_THINK_OPEN_THOUGHT: &str = "<thought>";
const MARK_THINK_CLOSE_THOUGHT: &str = "</thought>";

/// Text/Think 态的全部识别标记(主 + 别名;游标标记也**吞掉不泄漏**,见 `next_state`)
const MARKS: [&str; 8] = [
    MARK_THINK_OPEN,
    MARK_THINK_CLOSE,
    MARK_TOOL_OPEN,
    MARK_TOOL_CLOSE,
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
    } else {
        MARK_TOOL_CLOSE
    }
}

/// 工具块长度上限(病态保护:未闭合工具块不得吃满内存)
const MAX_TOOL_BYTES: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Text,
    Think,
    InTool,
}

/// 请求侧工具 schema(仅取 `name → 属性名 → JSON 类型`,用于 xml 参数类型回填)
#[derive(Debug, Default)]
pub struct ToolSchemas {
    types: HashMap<String, HashMap<String, String>>,
}

impl ToolSchemas {
    /// 从 OpenAI `tools` 数组提取(兼容嵌套 `{type,function:{…}}` 与扁平形)
    pub fn from_tools(tools: &[Json]) -> Self {
        let mut types = HashMap::new();
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
                    if let Some(ty) = v.get("type").and_then(|v| v.as_str()) {
                        props.insert(k.clone(), ty.to_string());
                    }
                }
            }
            types.insert(name.to_string(), props);
        }
        Self { types }
    }

    /// xml 参数值 → JSON 值。**按 schema 类型回填**(xml 分支把一切写成了
    /// 字符串形态;不按 schema 则 `timeout: 30` 会变成 `"30"`,pi 侧
    /// JSON-schema 校验会拒)。无 schema = 保守保持字符串(不猜)
    fn coerce(&self, tool: &str, prop: &str, raw: &str) -> Json {
        let Some(ty) = self.types.get(tool).and_then(|m| m.get(prop)) else {
            return Json::String(raw.to_string());
        };
        match ty.as_str() {
            "number" | "integer" => match serde_json::from_str::<Json>(raw.trim()) {
                Ok(Json::Number(n)) => Json::Number(n),
                _ => Json::String(raw.to_string()),
            },
            "boolean" => match raw.trim() {
                "true" => Json::Bool(true),
                "false" => Json::Bool(false),
                _ => Json::String(raw.to_string()),
            },
            "array" | "object" => match serde_json::from_str::<Json>(raw.trim()) {
                Ok(v @ (Json::Array(_) | Json::Object(_))) => v,
                _ => Json::String(raw.to_string()),
            },
            _ => Json::String(raw.to_string()),
        }
    }
}

/// 三通道解析器(有状态;每 turn 新建)
pub struct ChannelSplitter {
    state: State,
    buf: String,
    schemas: ToolSchemas,
    tool_index: usize,
    /// 游标标记计数(如关思考时仍吐 `</thinking>`)—— 一律**吞掉**不泄漏
    /// 进通道,但计数交给上层记日志(不许静默)
    strays: usize,
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
            strays: 0,
        }
    }

    /// 本 turn 已解析出的工具调用数(协议层据此定 `finish_reason`)
    pub fn tool_calls(&self) -> usize {
        self.tool_index
    }

    /// 游标标记计数(诊断;非零 = 值得记一条日志)
    pub fn strays(&self) -> usize {
        self.strays
    }

    /// 标记 → 目标态。当前状态下不该出现的标记(Text 遇 `</thinking>`/
    /// `</tool_call>`;Think 遇 `<thinking>`)一律**吞掉不泄漏**(否则原文
    /// 会漏进正文/思考面板),并计数
    fn next_state(&mut self, m: &str) -> State {
        match (self.state, canon(m)) {
            (State::Text, MARK_THINK_OPEN) => State::Think,
            (State::Text, MARK_TOOL_OPEN) => State::InTool,
            (State::Think, MARK_THINK_CLOSE) => State::Text,
            // Think 态遇 `<tool_call>`:模型没关思考直接开工(§4.6)—— 进工具
            // 通道,而不是把整块工具调用吞进思考面板
            (State::Think, MARK_TOOL_OPEN) => State::InTool,
            _ => {
                self.strays += 1;
                self.state
            }
        }
    }

    /// 推进一个增量文本块 → 产出事件序列
    pub fn push(&mut self, chunk: &str) -> Vec<Item> {
        self.buf.push_str(chunk);
        self.drain()
    }

    /// 生成流结束:冲刷尾巴(未闭合的标记前缀按**普通文本**放行 —— 模型没吐
    /// 完的最坏情况不该吞内容;未闭合工具块 → Malformed)
    pub fn finish(&mut self) -> Vec<Item> {
        let mut out = Vec::new();
        match self.state {
            State::InTool => {
                let raw = std::mem::take(&mut self.buf);
                out.push(Item::Malformed { kind: "tool_call_unclosed", raw });
                self.state = State::Text;
            }
            State::Text | State::Think => {
                if !self.buf.is_empty() {
                    let text = std::mem::take(&mut self.buf);
                    out.push(Item::Delta { chan: self.chan(), text });
                }
            }
        }
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
            match self.state {
                State::InTool => {
                    if let Some(p) = self.buf.find(MARK_TOOL_CLOSE) {
                        let body = self.buf[..p].to_string();
                        self.buf.drain(..p + MARK_TOOL_CLOSE.len());
                        out.push(self.build_tool_call(&body));
                        self.state = State::Text;
                        continue;
                    }
                    if self.buf.len() > MAX_TOOL_BYTES {
                        let raw = std::mem::take(&mut self.buf);
                        out.push(Item::Malformed { kind: "tool_call_overflow", raw });
                        self.state = State::Text;
                        continue;
                    }
                    break; // 工具块内部内容不外发;等闭合标记
                }
                State::Text | State::Think => {
                    // 找**最早**出现的标记(同位置取更长者,防前缀重叠)
                    let mut hit: Option<(usize, &str)> = None;
                    for m in MARKS.iter() {
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
                            self.buf.drain(..p + m.len());
                            self.state = self.next_state(m);
                            continue;
                        }
                        None => {
                            // 尾巴可能是标记前缀(跨 chunk 切半)⇒ 挂起不发
                            let hold = holdback_len(&self.buf, &MARKS);
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

    /// 工具块体 → 完整调用(两格式;I4 保证 arguments 是合法 JSON 字符串)
    fn build_tool_call(&mut self, body: &str) -> Item {
        let body = body.trim();
        let index = self.tool_index;
        match parse_tool_body(body, &self.schemas) {
            Ok((name, args)) => {
                self.tool_index += 1;
                Item::ToolCall { index, name, arguments: args }
            }
            Err(kind) => Item::Malformed { kind, raw: body.to_string() },
        }
    }
}

/// 工具块体解析:json 格式 `{"name":…,"arguments":{…}}` 优先,其次 xml 格式
fn parse_tool_body(body: &str, schemas: &ToolSchemas) -> Result<(String, String), &'static str> {
    // ① json 格式(template `tool_call_format="json"`)
    if body.starts_with('{') {
        let v: Json = serde_json::from_str(body).map_err(|_| "tool_call_json_invalid")?;
        let name = v.get("name").and_then(|n| n.as_str()).ok_or("tool_call_no_name")?;
        let args = v.get("arguments").cloned().unwrap_or_else(|| Json::Object(Map::new()));
        let s = serde_json::to_string(&args).map_err(|_| "tool_call_args_unserializable")?;
        return Ok((name.to_string(), s));
    }
    // ② xml 格式(template 默认):`<function=NAME> <parameter=K>V</parameter>… </function>`
    let open = body.find("<function=").ok_or("tool_call_no_function")?;
    let rest = &body[open + "<function=".len()..];
    let gt = rest.find('>').ok_or("tool_call_no_function")?;
    let name = rest[..gt].trim().to_string();
    if name.is_empty() {
        return Err("tool_call_empty_name");
    }
    let inner = &rest[gt + 1..];
    let mut obj = Map::new();
    let mut cur = inner;
    while let Some(p) = cur.find("<parameter=") {
        let after = &cur[p + "<parameter=".len()..];
        let Some(gt) = after.find('>') else { break };
        let key = after[..gt].trim().to_string();
        let tail = &after[gt + 1..];
        let end = tail.find("</parameter>").ok_or("tool_call_param_unclosed")?;
        let mut val = &tail[..end];
        // 模板写入形态 = `<parameter=k>\nVALUE\n</parameter>`:各去一个包裹换行
        if let Some(v) = val.strip_prefix('\n') {
            val = v;
        }
        if let Some(v) = val.strip_suffix('\n') {
            val = v;
        }
        if !key.is_empty() {
            obj.insert(key.clone(), schemas.coerce(&name, &key, val));
        }
        cur = &tail[end + "</parameter>".len()..];
    }
    let s = serde_json::to_string(&Json::Object(obj)).map_err(|_| "tool_call_args_unserializable")?;
    Ok((name, s))
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
        ToolSchemas::from_tools(&[json!({"type":"function","function":{
            "name":"bash",
            "parameters":{"type":"object","properties":{
                "command":{"type":"string"},
                "timeout":{"type":"number"},
                "background":{"type":"boolean"}}}}})])
    }

    /// 单块:思考 → 正文 → 工具调用(模板默认 xml 格式)
    #[test]
    fn one_shot_xml() {
        let mut s = ChannelSplitter::new(true, tools());
        let items = s.push(
            "我需要先看目录。\n</think>\n\n好的,我来执行。\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n<parameter=background>\ntrue\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(split(&items), "R:我需要先看目录。\n|T:\n\n好的,我来执行。\n|", "通道序列");
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
    /// (仓库内金标,恒跑);模板源(需 `OWL_TPL_DIR`)必须留长形态别名。
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
        match std::env::var("OWL_TPL_DIR") {
            Ok(dir) => {
                let src = std::fs::read_to_string(
                    std::path::Path::new(&dir).join("chat_template.jinja"),
                )
                .unwrap();
                assert!(src.contains(MARK_THINK_OPEN_LONG), "模板源应含长形态别名");
            }
            Err(_) => eprintln!("skip(部分): OWL_TPL_DIR 未设,跳过模板源检查"),
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
}
