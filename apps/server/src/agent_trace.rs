//! `/debug/agent` 诊断面(2026-10-19,**G5②**):最近 N 轮的**结构化现场**。
//!
//! 为什么需要:pi 真机排障(A7/A8 族:工具调用遵循度 / 渲染对称律 /
//! spec 臂非稳定)时,现状只有 `[probes] prompt_dump` 落盘 prompt 原文 ——
//! 缺**解析器做了什么**与**usage 真账**,只能靠日志猜。本模块补这两面:
//!
//! | 面 | 内容 |
//! |---|---|
//! | `req` | 请求协议事实(model/stream/tools/thinking/reasoning_effort/session/max_new/stop/ignore_eos) |
//! | `prompt` | **渲染后 prompt**(头 [`PROMPT_HEAD`] + 尾 [`PROMPT_TAIL`];中段省略量显式记账) |
//! | `usage` | prompt/completion/cached/**reasoning**(与 wire 同源) |
//! | `outcome` | finish_reason / 工具数 / 游标数 / 畸形数 / TTFT / 总耗时 / 错误 |
//! | `events` | **解析器结构轨迹**(`toolcall::StructEvent::line`,只记结构不记 delta) |
//!
//! ## 纪律
//! - **只在内存**(环形,不落盘、不打日志);转储 = 显式 `GET /debug/agent`
//!   —— 诊断面不主动泄漏会话内容。
//! - **不做全局变量**:`main` 建 → `serve` 传 → 路由读(同 `/debug/blocks` 的
//!   face 律;显式依赖)。
//! - **不许静默丢**(律 #1):①轨迹/环形超限都有显式计数;
//!   ②每轮结束**必须**留下一条记录 —— [`TurnLedger`] 的 `Drop` 是兜底网,
//!   提前返回(含 `?` 传播)也会记一条 `未结算` 而不是消失。
//!
//! ## 内存上界
//! `TURNS` × (`PROMPT_HEAD` + `PROMPT_TAIL` + `TRACE_CAP` 行 + 常数)
//! ≈ 8 × 32 KiB = **256 KiB** 量级(prompt 文本域);与 ctx 无关(截断)。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::{json, Value as Json};

/// 环形深度(轮)
pub const TURNS: usize = 8;
/// prompt 快照保留的**头部**字节数(系统段 + 模板起手)
pub const PROMPT_HEAD: usize = 8 * 1024;
/// prompt 快照保留的**尾部**字节数(最近对话 + assistant 起手;渲染对称律看这里)
pub const PROMPT_TAIL: usize = 24 * 1024;

// ============================================================================
// §1 事实结构(按面拆;一个 struct 只描述一个面 —— 不写万能类型)
// ============================================================================

/// 请求协议事实(`openai::ChatRequest` 的投影,不复制 messages/tools 本体)
#[derive(Debug, Clone, Default)]
pub struct ReqFacts {
    pub model: String,
    pub stream: bool,
    pub messages: usize,
    pub roles: String,
    pub tools: usize,
    pub thinking: bool,
    pub reasoning_effort: Option<String>,
    /// 会话 id(哈希后;原始键不留在诊断面 —— 含客户端本地路径信息)
    pub session: Option<u64>,
    pub max_new: usize,
    pub ignore_eos: bool,
    pub stop: usize,
}

impl ReqFacts {
    fn to_json(&self) -> Json {
        json!({
            "model": self.model,
            "stream": self.stream,
            "messages": self.messages,
            "roles": self.roles,
            "tools": self.tools,
            "thinking": self.thinking,
            "reasoning_effort": self.reasoning_effort,
            "session": self.session.map(|s| format!("{s:#x}")),
            "max_new": self.max_new,
            "ignore_eos": self.ignore_eos,
            "stop": self.stop,
        })
    }
}

/// 渲染后 prompt 的快照(头 + 尾 + 省略量)
#[derive(Debug, Clone, Default)]
pub struct PromptFacts {
    pub bytes: usize,
    pub chars: usize,
    pub head: String,
    /// 尾部(未截断时为空 —— 此时 `head` 已是全文)
    pub tail: String,
    pub omitted_bytes: usize,
}

impl PromptFacts {
    /// 渲染产物 → 快照(截断一律走 char 边界;省略量显式记账)
    pub fn clip(prompt: &str) -> Self {
        let bytes = prompt.len();
        let chars = prompt.chars().count();
        if bytes <= PROMPT_HEAD + PROMPT_TAIL {
            return Self {
                bytes,
                chars,
                head: prompt.to_string(),
                tail: String::new(),
                omitted_bytes: 0,
            };
        }
        let head_end = floor_boundary(prompt, PROMPT_HEAD);
        let tail_start = ceil_boundary(prompt, bytes - PROMPT_TAIL);
        Self {
            bytes,
            chars,
            head: prompt[..head_end].to_string(),
            tail: prompt[tail_start..].to_string(),
            omitted_bytes: tail_start - head_end,
        }
    }

    fn to_json(&self) -> Json {
        json!({
            "bytes": self.bytes,
            "chars": self.chars,
            "head": self.head,
            "tail": self.tail,
            "omitted_bytes": self.omitted_bytes,
            "truncated": self.omitted_bytes > 0,
        })
    }
}

/// usage 真账(与 wire 同源:同一个 `openai::Usage`)
#[derive(Debug, Clone, Copy, Default)]
pub struct UsageFacts {
    pub prompt: usize,
    pub completion: usize,
    pub cached: usize,
    pub reasoning: usize,
}

impl UsageFacts {
    fn to_json(&self) -> Json {
        json!({
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "cached_tokens": self.cached,
            "reasoning_tokens": self.reasoning,
        })
    }
}

/// 结局事实(正常/失败/未结算三态;`error` 非空即异常路径)
#[derive(Debug, Clone, Default)]
pub struct OutcomeFacts {
    pub finish: Option<&'static str>,
    pub tools_called: usize,
    pub strays: usize,
    pub malformed: usize,
    /// 首 token 延迟(ms;`None` = 没到首 token)
    pub ttft_ms: Option<u64>,
    pub elapsed_ms: u64,
    pub error: Option<String>,
}

impl OutcomeFacts {
    fn to_json(&self) -> Json {
        json!({
            "finish_reason": self.finish,
            "tools_called": self.tools_called,
            "strays": self.strays,
            "malformed": self.malformed,
            "ttft_ms": self.ttft_ms,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        })
    }
}

/// 一轮完整记录(五个面 + 序号)
#[derive(Debug, Clone, Default)]
pub struct TurnTrace {
    pub seq: u64,
    pub at_unix_ms: u64,
    pub req: ReqFacts,
    pub prompt: PromptFacts,
    pub usage: UsageFacts,
    pub outcome: OutcomeFacts,
    /// 解析器结构轨迹行(见 `toolcall::StructEvent::line`)
    pub events: Vec<String>,
}

impl TurnTrace {
    fn to_json(&self) -> Json {
        json!({
            "seq": self.seq,
            "at_unix_ms": self.at_unix_ms,
            "req": self.req.to_json(),
            "prompt": self.prompt.to_json(),
            "usage": self.usage.to_json(),
            "outcome": self.outcome.to_json(),
            "events": self.events,
        })
    }
}

// ============================================================================
// §2 环形仓(唯一共享态;显式构造 + 参数传递,不做全局变量)
// ============================================================================

/// 最近 `TURNS` 轮的记录仓(**进程内**;`GET /debug/agent` 读,`POST /debug/agent/clear` 清)
pub struct AgentTrace {
    ring: Mutex<VecDeque<TurnTrace>>,
    next_seq: AtomicU64,
    cap: usize,
}

impl Default for AgentTrace {
    fn default() -> Self {
        Self::with_capacity(TURNS)
    }
}

impl AgentTrace {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            ring: Mutex::new(VecDeque::with_capacity(cap)),
            next_seq: AtomicU64::new(0),
            cap,
        }
    }

    /// 落一轮(序号在**落仓时**分配 = 见证顺序;环形满则逐最旧)。
    /// `cap = 0` = 关仓(构造上只有 `with_capacity(0)`;当前调用方恒用
    /// 缺省 [`TURNS`]——留住这条分支是防御性的:0 不许变成死循环)
    pub fn record(&self, mut turn: TurnTrace) {
        if self.cap == 0 {
            return;
        }
        turn.seq = self.next_seq.fetch_add(1, Ordering::Relaxed) + 1;
        turn.at_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        // 锁中毒 = 他处 panic(诊断面不许把主流程带崩):**响亮记录**后重建仓
        let mut ring = match self.ring.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                owl_shared::log_warn!("agent-trace", "轨迹仓互斥锁中毒(他处 panic);重建仓继续");
                poisoned.into_inner()
            }
        };
        while ring.len() >= self.cap {
            ring.pop_front();
        }
        ring.push_back(turn);
    }

    /// 最近 `n` 轮(**旧→新**;`n = 0` 取全仓)—— 给 `/debug/agent[/<n>]`
    pub fn snapshot(&self, n: usize) -> Json {
        let ring = match self.ring.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let take = if n == 0 { ring.len() } else { n.min(ring.len()) };
        let turns: Vec<Json> = ring
            .iter()
            .skip(ring.len() - take)
            .map(TurnTrace::to_json)
            .collect();
        json!({
            "turns": turns,
            "kept": ring.len(),
            "capacity": self.cap,
            "prompt_head_bytes": PROMPT_HEAD,
            "prompt_tail_bytes": PROMPT_TAIL,
        })
    }

    /// 清仓(返回清掉的轮数;`POST /debug/agent/clear`)
    pub fn clear(&self) -> usize {
        let mut ring = match self.ring.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let n = ring.len();
        ring.clear();
        n
    }
}

// ============================================================================
// §3 施工期账本(`openai::chat` 持有;结束必落一条 —— Drop 是兜底网)
// ============================================================================

/// 一轮的施工期账本。
///
/// 用法:`new` → 逐面 `set_*` → 结束 `commit`。**忘了 commit 也不丢**:
/// `Drop` 会以 `未结算(提前返回)` 落一条 —— 律 #1 的兜底网
/// (设计外的提前返回也必须可见,不许静默)。
pub struct TurnLedger {
    trace: Arc<AgentTrace>,
    started: Instant,
    ttft_ms: Option<u64>,
    req: Option<ReqFacts>,
    prompt: Option<PromptFacts>,
    events: Vec<String>,
    committed: bool,
}

impl TurnLedger {
    pub fn new(trace: &Arc<AgentTrace>) -> Self {
        Self {
            trace: Arc::clone(trace),
            started: Instant::now(),
            ttft_ms: None,
            req: None,
            prompt: None,
            events: Vec::new(),
            committed: false,
        }
    }

    pub fn set_req(&mut self, f: ReqFacts) {
        self.req = Some(f);
    }

    /// 记录渲染后的 prompt(截断在 `PromptFacts::clip` 内,显式记账)
    pub fn set_prompt(&mut self, prompt: &str) {
        self.prompt = Some(PromptFacts::clip(prompt));
    }

    /// 首个生成 token(幂等:只记第一次 —— TTFT)
    pub fn first_token(&mut self) {
        if self.ttft_ms.is_none() {
            self.ttft_ms = Some(self.started.elapsed().as_millis() as u64);
        }
    }

    /// 解析器结构轨迹(结算时从 `ChannelSplitter` 取)
    pub fn set_events(&mut self, events: Vec<String>) {
        self.events = events;
    }

    /// 结算落仓(正常路径);`usage`/`outcome` 来自 wire 同一份账
    pub fn commit(mut self, usage: UsageFacts, mut outcome: OutcomeFacts) {
        outcome.ttft_ms = outcome.ttft_ms.or(self.ttft_ms);
        outcome.elapsed_ms = self.started.elapsed().as_millis() as u64;
        self.trace.record(TurnTrace {
            seq: 0, // record 时分配
            at_unix_ms: 0,
            req: self.req.take().unwrap_or_default(),
            prompt: self.prompt.take().unwrap_or_default(),
            usage,
            outcome,
            events: std::mem::take(&mut self.events),
        });
        self.committed = true;
    }

    /// 显式失败结算(错误路径;**不建议省** —— 让错误原因落进 outcome.error)
    pub fn fail(self, err: impl Into<String>) {
        self.commit(UsageFacts::default(), OutcomeFacts { error: Some(err.into()), ..Default::default() });
    }
}

impl Drop for TurnLedger {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // 兜底网:任何设计外提前返回(含 `?` 传播)都在这里显形
        let outcome = OutcomeFacts {
            ttft_ms: self.ttft_ms,
            elapsed_ms: self.started.elapsed().as_millis() as u64,
            error: Some("未结算(提前返回;见服务端日志)".to_string()),
            ..Default::default()
        };
        self.trace.record(TurnTrace {
            seq: 0,
            at_unix_ms: 0,
            req: self.req.take().unwrap_or_default(),
            prompt: self.prompt.take().unwrap_or_default(),
            usage: UsageFacts::default(),
            outcome,
            events: std::mem::take(&mut self.events),
        });
    }
}

// ============================================================================
// §4 UTF-8 边界小工具
// ============================================================================

/// 向下取整到 char 边界(≤ i,且能索引)
fn floor_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// 向上取整到 char 边界(≥ i,且能索引)
fn ceil_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_keeps_short_prompt_whole() {
        let p = "短 prompt";
        let f = PromptFacts::clip(p);
        assert_eq!(f.head, p);
        assert!(f.tail.is_empty(), "未截断 ⇒ tail 空(head 即全文)");
        assert_eq!(f.omitted_bytes, 0);
        assert_eq!(f.chars, p.chars().count());
    }

    #[test]
    fn clip_is_utf8_safe_and_accounts_omission() {
        // 每字符 3 字节(汉字)+ 4 字节(emoji)混排:截断点必落在边界上
        let unit = "汉字🙂abc";
        let p = unit.repeat(4096); // ≈ 40 KiB > HEAD+TAIL
        let f = PromptFacts::clip(&p);
        assert!(f.omitted_bytes > 0, "超长必须截断");
        assert_eq!(f.head.len() + f.tail.len() + f.omitted_bytes, p.len(), "省略量账要平");
        assert_eq!(f.head, p[..f.head.len()], "head 必须是真前缀(且 char 边界)");
        assert!(p.ends_with(&f.tail), "tail 必须是真后缀");
        assert_eq!(f.bytes, p.len());
    }

    #[test]
    fn ring_keeps_newest_and_caps() {
        let t = AgentTrace::with_capacity(2);
        for i in 0..3u64 {
            let mut turn = TurnTrace::default();
            turn.req.model = format!("m{i}");
            t.record(turn);
        }
        let snap = t.snapshot(0);
        let turns = snap["turns"].as_array().expect("turns");
        assert_eq!(turns.len(), 2, "容量 2 ⇒ 只留最新两轮");
        assert_eq!(turns[0]["req"]["model"], "m1", "旧→新序");
        assert_eq!(turns[1]["req"]["model"], "m2");
        assert_eq!(turns[1]["seq"], 3, "序号按落仓序");
        assert_eq!(t.clear(), 2);
        assert_eq!(t.snapshot(0)["turns"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn uncommitted_ledger_still_records() {
        // 兜底网:提前返回(未 commit)也必须留下一条“未结算”
        let t = Arc::new(AgentTrace::with_capacity(4));
        {
            let mut led = TurnLedger::new(&t);
            led.set_req(ReqFacts { model: "m".into(), ..Default::default() });
            // 不 commit,直接 drop
        }
        let snap = t.snapshot(0);
        let turns = snap["turns"].as_array().unwrap();
        assert_eq!(turns.len(), 1, "未结算也要落一条");
        let err = turns[0]["outcome"]["error"].as_str().unwrap_or("");
        assert!(err.contains("未结算"), "错误原因要写明:{err}");
    }

    #[test]
    fn committed_ledger_records_once() {
        let t = Arc::new(AgentTrace::with_capacity(4));
        let mut led = TurnLedger::new(&t);
        led.set_prompt("hello");
        led.first_token();
        led.commit(
            UsageFacts { prompt: 3, completion: 9, cached: 1, reasoning: 4 },
            OutcomeFacts { finish: Some("stop"), tools_called: 1, ..Default::default() },
        );
        let snap = t.snapshot(0);
        let turns = snap["turns"].as_array().unwrap();
        assert_eq!(turns.len(), 1, "commit 之后 Drop 不得重复落账");
        assert_eq!(turns[0]["usage"]["reasoning_tokens"], 4);
        assert_eq!(turns[0]["outcome"]["finish_reason"], "stop");
        assert!(turns[0]["outcome"]["ttft_ms"].is_u64());
    }
}
