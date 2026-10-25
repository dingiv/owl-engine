//! Workload result records and rendering.

use serde_json::{json, Value};

#[derive(Clone, Debug, Default)]
pub struct TurnRecord {
    pub scenario: String,
    pub turn: usize,
    pub finish: String,
    pub prompt: u64,
    pub cached: u64,
    pub completion: u64,
    pub reasoning: u64,
    /// Deltas of `image_evict.*` counters across this model call.
    pub absorbed: u64,
    pub rotate_bytes: u64,
    pub rebase: u64,
    pub elapsed_ms: u128,
    pub tools: Vec<String>,
    pub error: Option<String>,
}

impl TurnRecord {
    pub fn cached_pct(&self) -> f64 {
        if self.prompt == 0 {
            0.0
        } else {
            100.0 * self.cached as f64 / self.prompt as f64
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "scenario": self.scenario,
            "turn": self.turn,
            "finish": self.finish,
            "prompt_tokens": self.prompt,
            "cached_tokens": self.cached,
            "cached_pct": self.cached_pct(),
            "completion_tokens": self.completion,
            "reasoning_tokens": self.reasoning,
            "image_evict": {
                "absorbed": self.absorbed,
                "rotate_bytes": self.rotate_bytes,
                "rebase": self.rebase,
            },
            "elapsed_ms": self.elapsed_ms,
            "tools": self.tools,
            "error": self.error,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct ScenarioReport {
    pub name: String,
    pub pass: bool,
    pub detail: String,
    pub turns: Vec<TurnRecord>,
}

impl ScenarioReport {
    pub fn to_json(&self) -> Value {
        json!({
            "scenario": self.name,
            "pass": self.pass,
            "detail": self.detail,
            "turns": self.turns.iter().map(TurnRecord::to_json).collect::<Vec<_>>(),
        })
    }
}

pub fn print_report(r: &ScenarioReport) {
    let tag = if r.pass { "PASS" } else { "FAIL" };
    println!("[{tag}] {}{}", r.name, if r.detail.is_empty() { String::new() } else { format!(" — {}", r.detail) });
    for t in &r.turns {
        println!(
            "  turn {:<2} finish={:<10} prompt={:<6} cached={:<6} ({:5.1}%) completion={:<4} reasoning={:<4} absorbed=+{:<2} rotate=+{:<8} rebase=+{} tools=[{}] {}ms{}",
            t.turn,
            t.finish,
            t.prompt,
            t.cached,
            t.cached_pct(),
            t.completion,
            t.reasoning,
            t.absorbed,
            t.rotate_bytes,
            t.rebase,
            t.tools.join(","),
            t.elapsed_ms,
            t.error.as_ref().map(|e| format!(" error={e}")).unwrap_or_default(),
        );
    }
}
