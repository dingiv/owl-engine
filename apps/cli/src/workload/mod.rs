//! Agent-workload end-to-end harness.
//!
//! `cli --workload core --url http://127.0.0.1:8135` drives a live owl server
//! with real tool execution and real image requests. The suite is intentionally
//! functional (one scenario per feature family), not a stress test.

pub mod png;
pub mod report;
pub mod scenarios;
pub mod tools;

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::client::{ChatReply, ChatRequest, Client};
use report::{ScenarioReport, TurnRecord};

pub struct Args {
    pub url: String,
    pub model: Option<String>,
    pub max_tokens: usize,
    pub suite: String,
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
                   --set pool.kv_fp8=false --set pool.image_evict=true \\\n\
                   --set pool.prefix_cache=true --set pool.snap_max=2 \\\n\
                   --set pool.snap_host_mb=1024 --set runtime.max_seq=131072 \\\n\
                   --set runtime.implicit_session=true --log-level info"
            );
            return 2;
        }
    }

    let suite = args.suite.to_ascii_lowercase();
    if !matches!(suite.as_str(), "core" | "image-evict" | "all") {
        eprintln!("[workload] 未知 suite {suite:?}(core | image-evict | all)");
        return 2;
    }

    let pid = std::process::id();
    // Keep well above the known DFlash2 tail-cap P0 (remaining <= depth): the
    // suite is spec-on, so a short max_tokens can wedge the engine at the tail.
    let max_tokens = args.max_tokens.max(2048);
    let mut reports = Vec::new();
    let mut cases: Vec<(&str, fn(&Client, &Args, u32, usize) -> Result<ScenarioReport, String>)> = vec![
        ("text_multiturn", scenarios::text_multiturn),
        ("tool_roundtrip", scenarios::tool_roundtrip),
        ("vision_chat", scenarios::vision_chat),
        ("mixed_tool_image", scenarios::mixed_tool_image),
        ("image_delete_marked", scenarios::image_delete_marked),
        ("image_sliding_window", scenarios::image_sliding_window),
    ];
    if suite == "image-evict" {
        cases.retain(|(n, _)| n.starts_with("image_"));
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

    /// Push a user message and run the model/tool loop until `stop`.
    pub(crate) fn send(&mut self, content: Value) -> Result<ChatReply, String> {
        self.history.push(json!({"role": "user", "content": content}));
        self.round()
    }

    fn round(&mut self) -> Result<ChatReply, String> {
        for _ in 0..10 {
            let before = self.snapshot()?;
            let req = ChatRequest {
                messages: self.history.clone(),
                tools: if self.tools { Some(tools::schemas()) } else { None },
                max_tokens: self.max_tokens,
                temperature: 0.0,
                session: self.session.clone(),
                chat_template_kwargs: None,
            };
            let reply = self
                .client
                .chat(req)
                .map_err(|e| format!("{}: chat 失败: {e}", self.scenario))?;
            let after = self.snapshot()?;
            let names: Vec<String> = reply.tool_calls.iter().map(|c| c.name.clone()).collect();
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
                error: None,
            });
            self.history.push(assistant_message(&reply));
            match reply.finish_reason.as_str() {
                "tool_calls" => {
                    if reply.tool_calls.is_empty() {
                        return Err(format!("{}: finish=tool_calls 但无 tool_calls", self.scenario));
                    }
                    for call in &reply.tool_calls {
                        let value = tools::execute(&self.root, call)
                            .map_err(|e| format!("tool {}: {e}", call.name))?;
                        self.history.push(json!({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": tools::result_string(&value),
                        }));
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
