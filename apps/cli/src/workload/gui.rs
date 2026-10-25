//! Simulated desktop for the GUI-agent workload.
//!
//! The screen is a deterministic solid-colour PNG; actions change it so the
//! model must look at the attached screenshot to answer. Every tool result
//! carries a screenshot (the real GUI-agent shape), and the harness keeps only
//! the newest three images in the conversation.

use serde_json::{json, Value};

use crate::client::ToolCall;
use crate::workload::png::{data_url_png, BLUE, GREEN, PURPLE, RED, YELLOW};
use crate::workload::ToolOutput;

const W: u32 = 256;
const H: u32 = 256;

pub(crate) struct GuiAgent {
    screen: (u8, u8, u8),
    clicks: usize,
    typed: String,
    actions: usize,
}

impl GuiAgent {
    pub(crate) fn new() -> Self {
        Self {
            screen: PURPLE,
            clicks: 0,
            typed: String::new(),
            actions: 0,
        }
    }

    pub(crate) fn screen(&self) -> (u8, u8, u8) {
        self.screen
    }

    pub(crate) fn schemas() -> Vec<Value> {
        vec![
            json!({
                "type": "function",
                "function": {
                    "name": "screenshot",
                    "description": "Capture the current desktop screen. Returns the screen as an image.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "Click at pixel coordinates on the current screen. Returns the screen after the click as an image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "minimum": 0, "maximum": 255},
                            "y": {"type": "integer", "minimum": 0, "maximum": 255}
                        },
                        "required": ["x", "y"]
                    }
                }
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "type",
                    "description": "Type text into the focused field. Returns the screen after typing as an image.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"]
                    }
                }
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "key",
                    "description": "Press a keyboard key (ENTER, TAB, ESC). Returns the screen after the key press as an image.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"]
                    }
                }
            }),
        ]
    }

    pub(crate) fn execute(&mut self, call: &ToolCall) -> Result<ToolOutput, String> {
        let args: Value = serde_json::from_str(&call.arguments)
            .map_err(|e| format!("gui {} arguments 非法 JSON: {e}: {}", call.name, call.arguments))?;
        let text = match call.name.as_str() {
            "screenshot" => "Captured the current screen.".to_string(),
            "click" => {
                let x = coord(&args, "x")?;
                let y = coord(&args, "y")?;
                self.clicks += 1;
                self.screen = if self.clicks == 1 { RED } else { GREEN };
                format!("Clicked at ({x}, {y}).")
            }
            "type" => {
                let value = args
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "gui type 缺 text".to_string())?;
                self.typed.push_str(value);
                self.screen = BLUE;
                format!("Typed {} character(s).", value.chars().count())
            }
            "key" => {
                let value = args
                    .get("key")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "gui key 缺 key".to_string())?;
                if value.eq_ignore_ascii_case("ENTER") {
                    self.screen = YELLOW;
                }
                format!("Pressed {value}.")
            }
            other => return Err(format!("未知 GUI 工具 {other}")),
        };
        self.actions += 1;
        let image = data_url_png(W, H, self.screen);
        Ok(ToolOutput::image_result(
            format!("Step {}: {text} Current screen attached.", self.actions),
            image,
        ))
    }
}

fn coord(args: &Value, key: &str) -> Result<u64, String> {
    let value = args
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("gui click 缺 {key}(0..255 整数)"))?;
    if value > 255 {
        return Err(format!("gui click {key}={value} 越界(0..255)"));
    }
    Ok(value)
}
