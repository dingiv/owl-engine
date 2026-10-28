//! Agent-workload end-to-end harness.
//!
//! `cli --workload core --url http://127.0.0.1:8135` drives a live owl server
//! with real tool execution and real image requests. The suite is intentionally
//! functional (one scenario per feature family), not a stress test.
//!
//! Suites: `core` (all resident cases), `gui` (the real GUI-agent screenshot
//! window), `gui-capability` (ambiguous capability prompt must still call
//! screenshot), `image-evict` (image-deletion shapes), and `all` (alias of core).

pub mod gui;
pub mod png;
pub mod report;
pub mod scenarios;
pub mod tools;

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::client::{ChatReply, ChatRequest, Client, ToolCall};
use report::{ScenarioReport, TurnRecord};

/// One tool result. Coding tools keep the historical string payload; GUI tools
/// return a content array with an attached screenshot.
pub(crate) struct ToolOutput {
    pub(crate) content: Value,
}

impl ToolOutput {
    pub(crate) fn text(value: Value) -> Self {
        Self { content: Value::String(value.to_string()) }
    }

    pub(crate) fn image_result(text: String, image: String) -> Self {
        Self {
            content: json!([
                // A trailing newline keeps the text/image boundary stable when
                // the image is later removed from a tool message.
                {"type": "text", "text": format!("{text}\n")},
                {"type": "image_url", "image_url": {"url": image}, "owl_evict": true},
                {"type": "text", "text": "End of tool result."}
            ]),
        }
    }
}

pub struct Args {
    pub url: String,
    pub model: Option<String>,
    pub max_tokens: usize,
    pub suite: String,
    pub only: Option<String>,
    pub server_log: Option<String>,
    pub json: bool,
    pub keep: bool,
}

pub fn run(args: Args) -> i32 {
    let client = Client::new(&args.url, args.model.clone());
    match client.health() {
        Ok(model) => eprintln!("[workload] 已连接 {} (model={model})", client.host()),
        Err(e) => {
            eprintln!("[workload] 连接失败: {e}");
            eprintln!(
                "[workload] 8135 图像无感删除测试档示例:\n\
                 ./target/release/server -p profiles/qwen38_27b-3080_tp2.toml \\\n\
                   --set pool.image_evict=true \\\n\
                   --set pool.prefix_cache=true --set pool.snap_max=2 \\\n\
                   --set pool.snap_host_mb=1024 --set runtime.max_seq=131072 \\\n\
                   --set runtime.implicit_session=true --log-level info\n\
                 或直接:tools/run-agent-workload.sh gui"
            );
            return 2;
        }
    }

    let suite = args.suite.to_ascii_lowercase();
    if !matches!(
        suite.as_str(),
        "core" | "image-evict" | "gui" | "gui-capability" | "all"
    ) {
        eprintln!(
            "[workload] 未知 suite {suite:?}(core | image-evict | gui | gui-capability | all)"
        );
        return 2;
    }

    let pid = std::process::id();
    // Keep well above the known DFlash2 tail-cap P0 (remaining <= depth): the
    // suite is spec-on, so a short max_tokens can wedge the engine at the tail.
    let max_tokens = args.max_tokens.max(2048);
    let mut reports = Vec::new();
    let mut cases: Vec<(&str, fn(&Client, &Args, u32, usize) -> Result<ScenarioReport, String>)> = vec![
        ("gui_agent_screenshots", scenarios::gui_agent_screenshots),
        ("gui_capability_prompt", scenarios::gui_capability_prompt),
        ("text_multiturn", scenarios::text_multiturn),
        ("tool_roundtrip", scenarios::tool_roundtrip),
        ("vision_chat", scenarios::vision_chat),
        ("mixed_tool_image", scenarios::mixed_tool_image),
        ("image_delete_marked", scenarios::image_delete_marked),
        ("image_delete_middle", scenarios::image_delete_middle),
        ("image_delete_last_image", scenarios::image_delete_last_image),
        ("image_delete_with_tool_round", scenarios::image_delete_with_tool_round),
        ("image_sliding_window", scenarios::image_sliding_window),
    ];
    if suite == "gui" {
        cases.retain(|(n, _)| *n == "gui_agent_screenshots");
    } else if suite == "gui-capability" {
        cases.retain(|(n, _)| *n == "gui_capability_prompt");
    } else if suite == "image-evict" {
        cases.retain(|(n, _)| n.starts_with("image_"));
    }
    if let Some(only) = args.only.as_deref() {
        cases.retain(|(n, _)| *n == only);
        if cases.is_empty() {
            eprintln!("[workload] --only {only:?} 未匹配任何场景");
            return 2;
        }
    }
    for (name, f) in cases {
        let rep = match f(&client, &args, pid, max_tokens) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[workload] {name} 失败: {e}");
                ScenarioReport {
                    name: name.to_string(),
                    pass: false,
                    detail: e,
                    turns: Vec::new(),
                }
            }
        };
        reports.push(rep);
    }

    if args.json {
        let arr: Vec<Value> = reports.iter().map(ScenarioReport::to_json).collect();
        println!("{}", serde_json::to_string_pretty(&arr).unwrap_or_else(|e| format!("[{{\"error\":\"{e}\"}}]")));
    } else {
        for r in &reports {
            report::print_report(r);
        }
    }
    let failed = reports.iter().filter(|r| !r.pass).count();
    eprintln!(
        "[workload] {}/{} scenarios passed",
        reports.len() - failed,
        reports.len()
    );
    if failed == 0 {
        0
    } else {
        1
    }
}

/// Scenario workspace (removed on drop unless `--keep`).
pub(crate) struct Workspace {
    path: PathBuf,
    keep: bool,
}

impl Workspace {
    pub(crate) fn new(name: &str, pid: u32, keep: bool) -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!("owl-workload-{name}-{pid}"));
        std::fs::create_dir_all(&path)
            .map_err(|e| format!("创建工作区 {}: {e}", path.display()))?;
        if keep {
            eprintln!("[workload] 工作区保留:{}", path.display());
        }
        Ok(Self { path, keep })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        if !self.keep {
            if let Err(e) = std::fs::remove_dir_all(&self.path) {
                eprintln!("[workload] 清理工作区 {}: {e}", self.path.display());
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Metrics {
    absorbed: u64,
    rotate_bytes: u64,
    rebase: u64,
}

pub(crate) struct Ctx<'a> {
    pub(crate) client: &'a Client,
    pub(crate) scenario: String,
    pub(crate) session: Option<String>,
    pub(crate) tools: bool,
    pub(crate) max_tokens: usize,
    pub(crate) root: PathBuf,
    pub(crate) history: Vec<Value>,
    pub(crate) turns: Vec<TurnRecord>,
    pub(crate) log_path: Option<PathBuf>,
    /// Template kwargs forwarded to the server (e.g. `enable_thinking`).
    pub(crate) template_kwargs: Option<Value>,
    /// GUI tool runtime; `Some` also switches the exposed tool schemas.
    pub(crate) gui: Option<gui::GuiAgent>,
    /// Keep at most this many images in the conversation (GUI agent = 3).
    pub(crate) image_limit: Option<usize>,
    /// URLs removed by the image-window policy (oldest first).
    pub(crate) image_prunes: Vec<String>,
}

impl<'a> Ctx<'a> {
    pub(crate) fn new(
        client: &'a Client,
        scenario: &str,
        session: Option<String>,
        tools: bool,
        max_tokens: usize,
        root: &Path,
    ) -> Self {
        Self {
            client,
            scenario: scenario.to_string(),
            session,
            tools,
            max_tokens,
            root: root.to_path_buf(),
            history: Vec::new(),
            turns: Vec::new(),
            log_path: None,
            template_kwargs: None,
            gui: None,
            image_limit: None,
            image_prunes: Vec::new(),
        }
    }

    /// Byte offset in the server log before a request (None = no log configured).
    pub(crate) fn log_mark(&self) -> Option<u64> {
        let path = self.log_path.as_ref()?;
        std::fs::metadata(path).ok().map(|m| m.len())
    }

    /// Bytes appended to the server log since `mark`.
    pub(crate) fn log_since(&self, mark: Option<u64>) -> String {
        let (Some(path), Some(mark)) = (self.log_path.as_ref(), mark) else {
            return String::new();
        };
        use std::io::{Read, Seek, SeekFrom};
        let mut f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return String::new(),
        };
        if f.seek(SeekFrom::Start(mark)).is_err() {
            return String::new();
        }
        let mut s = String::new();
        let _ = f.read_to_string(&mut s);
        s
    }

    pub(crate) fn system(&mut self, text: &str) {
        self.history.push(json!({"role": "system", "content": text}));
    }

    /// Force the template's thinking flag for this scenario (useful when a
    /// long free-form think would evict the single suffix snapshot).
    pub(crate) fn thinking(&mut self, on: bool) {
        self.template_kwargs = Some(json!({"enable_thinking": on}));
    }

    /// Push a user message and run the model/tool loop until `stop`.
    pub(crate) fn send(&mut self, content: Value) -> Result<ChatReply, String> {
        self.history.push(json!({"role": "user", "content": content}));
        self.round()
    }

    fn round(&mut self) -> Result<ChatReply, String> {
        for _ in 0..10 {
            self.enforce_image_limit()?;
            let images = image_count(&self.history);
            let before = self.snapshot()?;
            let req = ChatRequest {
                messages: self.history.clone(),
                tools: self.tool_schemas(),
                max_tokens: self.max_tokens,
                temperature: 0.0,
                session: self.session.clone(),
                chat_template_kwargs: self.template_kwargs.clone(),
            };
            let reply = self
                .client
                .chat(req)
                .map_err(|e| format!("{}: chat 失败: {e}", self.scenario))?;
            let after = self.snapshot()?;
            let names: Vec<String> = reply.tool_calls.iter().map(|c| c.name.clone()).collect();
            // One record per model request; a user turn may contain many of
            // these (and each request may return several tool calls).
            self.turns.push(TurnRecord {
                scenario: self.scenario.clone(),
                turn: self.turns.len() + 1,
                finish: reply.finish_reason.clone(),
                prompt: reply.usage.prompt_tokens,
                cached: reply.usage.cached_tokens,
                completion: reply.usage.completion_tokens,
                reasoning: reply.usage.reasoning_tokens,
                absorbed: after.absorbed.saturating_sub(before.absorbed),
                rotate_bytes: after.rotate_bytes.saturating_sub(before.rotate_bytes),
                rebase: after.rebase.saturating_sub(before.rebase),
                elapsed_ms: reply.elapsed_ms,
                tools: names,
                images,
                error: None,
            });
            self.history.push(assistant_message(&reply));
            match reply.finish_reason.as_str() {
                "tool_calls" => {
                    if reply.tool_calls.is_empty() {
                        return Err(format!("{}: finish=tool_calls 但无 tool_calls", self.scenario));
                    }
                    for call in &reply.tool_calls {
                        let output = self
                            .execute_tool(call)
                            .map_err(|e| format!("tool {}: {e}", call.name))?;
                        self.history.push(json!({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": output.content,
                        }));
                        self.enforce_image_limit()?;
                    }
                }
                "stop" => return Ok(reply),
                "length" => {
                    return Err(format!(
                        "{}: finish=length(prompt={}, completion={}, max_tokens={}) — 给足 max_tokens 或检查尾批 P0",
                        self.scenario,
                        reply.usage.prompt_tokens,
                        reply.usage.completion_tokens,
                        self.max_tokens
                    ))
                }
                other => return Err(format!("{}: 未预期 finish_reason={other}", self.scenario)),
            }
        }
        Err(format!("{}: tool 轮数超过 10", self.scenario))
    }

    fn tool_schemas(&self) -> Option<Vec<Value>> {
        if self.gui.is_some() {
            Some(gui::GuiAgent::schemas())
        } else if self.tools {
            Some(tools::schemas())
        } else {
            None
        }
    }

    fn execute_tool(&mut self, call: &ToolCall) -> Result<ToolOutput, String> {
        if let Some(runtime) = self.gui.as_mut() {
            runtime.execute(call)
        } else {
            tools::execute(&self.root, call).map(ToolOutput::text)
        }
    }

    fn enforce_image_limit(&mut self) -> Result<(), String> {
        let Some(limit) = self.image_limit else { return Ok(()) };
        while image_count(&self.history) > limit {
            let removed = remove_oldest_image(&mut self.history).ok_or_else(|| {
                format!(
                    "{}: 图片数超过上限 {limit} 但找不到可删图片",
                    self.scenario
                )
            })?;
            self.image_prunes.push(removed);
        }
        Ok(())
    }

    fn snapshot(&self) -> Result<Metrics, String> {
        let tag = |t: &str| {
            self.client
                .metric_counter(t)
                .map_err(|e| format!("metrics {t}: {e}"))
        };
        Ok(Metrics {
            absorbed: tag("image_evict.absorbed")?,
            rotate_bytes: tag("image_evict.rotate.bytes")?,
            rebase: tag("image_evict.rebase")?,
        })
    }

    pub(crate) fn last(&self) -> &TurnRecord {
        self.turns.last().expect("round 至少记一条 turn")
    }
}

fn assistant_message(reply: &ChatReply) -> Value {
    let mut msg = json!({"role": "assistant"});
    msg["content"] = if reply.content.is_empty() && !reply.tool_calls.is_empty() {
        Value::Null
    } else {
        Value::String(reply.content.clone())
    };
    if !reply.reasoning.is_empty() {
        msg["reasoning_content"] = Value::String(reply.reasoning.clone());
    }
    if !reply.tool_calls.is_empty() {
        msg["tool_calls"] = Value::Array(
            reply
                .tool_calls
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    })
                })
                .collect(),
        );
    }
    msg
}

/// Copy a single scenario's turns into a report with a pass/fail detail.
pub(crate) fn report(name: &str, ctx: Ctx<'_>, result: Result<String, String>) -> ScenarioReport {
    match result {
        Ok(detail) => ScenarioReport {
            name: name.to_string(),
            pass: true,
            detail,
            turns: ctx.turns,
        },
        Err(detail) => ScenarioReport {
            name: name.to_string(),
            pass: false,
            detail,
            turns: ctx.turns,
        },
    }
}

/// Number of image parts in the conversation (all roles).
pub(crate) fn image_count(history: &[Value]) -> usize {
    history
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .map(|a| {
            a.iter()
                .filter(|p| p.get("image_url").is_some())
                .count()
        })
        .sum()
}

/// Remove the oldest image part (first in message/part order), returning its URL.
pub(crate) fn remove_oldest_image(history: &mut [Value]) -> Option<String> {
    for msg in history.iter_mut() {
        let Some(arr) = msg.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        if let Some(pos) = arr.iter().position(|p| p.get("image_url").is_some()) {
            let part = arr.remove(pos);
            return part["image_url"]["url"].as_str().map(str::to_string);
        }
    }
    None
}
