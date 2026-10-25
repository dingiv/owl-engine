//! Local tools for the workload agent (pi-shaped read/write/edit/bash).
//!
//! Tool results are returned as JSON strings in a `role:"tool"` message. Model
//! errors (missing file, failed edit) are *results*, not harness failures, so
//! the agent can recover; malformed arguments/unknown tools are harness errors.

use std::path::{Component, Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

use crate::client::ToolCall;

pub fn schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command in the workspace and return its output and exit code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "timeout": {"type": "number", "description": "Seconds (default 10, max 60)"}
                    },
                    "required": ["command"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a UTF-8 text file. Offsets/limits are 1-based line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "number"},
                        "limit": {"type": "number"}
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "write",
                "description": "Create or overwrite a file with exact UTF-8 content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Replace exactly one occurrence of oldText with newText in a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "oldText": {"type": "string"},
                        "newText": {"type": "string"}
                    },
                    "required": ["path", "oldText", "newText"]
                }
            }
        }),
    ]
}

pub fn execute(root: &Path, call: &ToolCall) -> Result<Value, String> {
    let args: Value = serde_json::from_str(&call.arguments)
        .map_err(|e| format!("tool {} arguments 非法 JSON: {e}: {}", call.name, call.arguments))?;
    match call.name.as_str() {
        "bash" => Ok(bash(root, &args)),
        "read" => Ok(read(root, &args)),
        "write" => Ok(write(root, &args)),
        "edit" => Ok(edit(root, &args)),
        other => Err(format!("未知工具 {other}")),
    }
}

fn path_arg(root: &Path, args: &Value) -> Result<PathBuf, Value> {
    let raw = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| json!({"error": "missing path"}))?;
    safe_path(root, raw).map_err(|e| json!({"error": e}))
}

fn safe_path(root: &Path, raw: &str) -> Result<PathBuf, String> {
    if raw.trim().is_empty() {
        return Err("path 为空".into());
    }
    let p = Path::new(raw);
    if p.is_absolute() {
        return Err(format!("path 必须是工作区相对路径: {raw}"));
    }
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!("path 不允许 ..: {raw}"));
    }
    Ok(root.join(p))
}

fn read(root: &Path, args: &Value) -> Value {
    let path = match path_arg(root, args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => return json!({"error": format!("read {}: {e}", path.display())}),
    };
    let offset = args.get("offset").and_then(Value::as_u64).unwrap_or(1).max(1) as usize;
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(0) as usize;
    let lines: Vec<&str> = text.lines().collect();
    let start = (offset - 1).min(lines.len());
    let end = if limit == 0 {
        lines.len()
    } else {
        (start + limit).min(lines.len())
    };
    json!({
        "content": lines[start..end].join("\n"),
        "offset": offset,
        "lines": end - start,
        "total_lines": lines.len(),
    })
}

fn write(root: &Path, args: &Value) -> Value {
    let path = match path_arg(root, args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let Some(content) = args.get("content").and_then(Value::as_str) else {
        return json!({"error": "missing content"});
    };
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return json!({"error": format!("mkdir {}: {e}", parent.display())});
        }
    }
    match std::fs::write(&path, content.as_bytes()) {
        Ok(()) => json!({"ok": true, "bytes": content.len(), "path": path.display().to_string()}),
        Err(e) => json!({"error": format!("write {}: {e}", path.display())}),
    }
}

fn edit(root: &Path, args: &Value) -> Value {
    let path = match path_arg(root, args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let Some(old) = args.get("oldText").and_then(Value::as_str) else {
        return json!({"error": "missing oldText"});
    };
    let Some(new) = args.get("newText").and_then(Value::as_str) else {
        return json!({"error": "missing newText"});
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => return json!({"error": format!("read {}: {e}", path.display())}),
    };
    if !text.contains(old) {
        return json!({"error": "oldText not found"});
    }
    let out = text.replacen(old, new, 1);
    match std::fs::write(&path, out.as_bytes()) {
        Ok(()) => json!({"ok": true, "replacements": 1, "bytes": out.len()}),
        Err(e) => json!({"error": format!("write {}: {e}", path.display())}),
    }
}

fn bash(root: &Path, args: &Value) -> Value {
    let Some(command) = args.get("command").and_then(Value::as_str) else {
        return json!({"error": "missing command"});
    };
    let secs = args
        .get("timeout")
        .and_then(Value::as_u64)
        .unwrap_or(10)
        .clamp(1, 60);
    let out = Command::new("timeout")
        .arg(format!("{secs}s"))
        .arg("sh")
        .arg("-c")
        .arg(command)
        .current_dir(root)
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => return json!({"error": format!("spawn bash: {e}")}),
    };
    let mut text = String::from_utf8_lossy(&out.stdout).to_string();
    if !out.stderr.is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&String::from_utf8_lossy(&out.stderr));
    }
    if text.len() > 16_384 {
        text.truncate(16_384);
        text.push_str("\n[truncated]");
    }
    json!({
        "output": text,
        "exit_code": out.status.code().unwrap_or(-1),
        "command": command,
    })
}
