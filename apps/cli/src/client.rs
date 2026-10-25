//! Minimal HTTP/SSE client for the workload harness.
//!
//! The REPL in `main.rs` predates the server's chunked SSE; this module is the
//! structured client used by `workload`: it de-chunks the body, parses
//! `content` / `reasoning_content` / `tool_calls` / `usage` / `finish_reason`,
//! and exposes the debug metrics used by assertions.

use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use serde_json::Value;

const IO_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub completion_tokens: u64,
    pub reasoning_tokens: u64,
}

#[derive(Clone, Debug)]
pub struct ToolCall {
    pub index: usize,
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Default)]
pub struct ChatReply {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    pub usage: Usage,
    pub elapsed_ms: u128,
}

impl ChatReply {
    /// Text a human/assertion can read: answer first, reasoning as fallback.
    pub fn text(&self) -> &str {
        if self.content.is_empty() {
            &self.reasoning
        } else {
            &self.content
        }
    }
}

pub struct ChatRequest {
    pub messages: Vec<Value>,
    pub tools: Option<Vec<Value>>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub session: Option<String>,
    pub chat_template_kwargs: Option<Value>,
}


pub struct Client {
    host: String,
    model: Option<String>,
}

impl Client {
    pub fn new(url: &str, model: Option<String>) -> Self {
        let host = url
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .trim_end_matches('/')
            .to_string();
        Self { host, model }
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn health(&self) -> io::Result<String> {
        let v = self.get_json("/health")?;
        Ok(v.get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("?")
            .to_string())
    }

    pub fn get_json(&self, path: &str) -> io::Result<Value> {
        let mut stream = self.connect()?;
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
            self.host
        );
        stream.write_all(req.as_bytes())?;
        let (status, mut reader, chunked) = read_head(stream)?;
        let body = read_body(&mut reader, chunked)?;
        if status != 200 {
            return Err(io::Error::other(format!(
                "GET {path} -> HTTP {status}: {body}"
            )));
        }
        serde_json::from_str(&body)
            .map_err(|e| io::Error::other(format!("GET {path} 返回非 JSON: {e}: {body}")))
    }

    /// Counter value for an exact metrics tag (`0` when the tag has no row yet).
    pub fn metric_counter(&self, tag: &str) -> io::Result<u64> {
        let v = self.get_json(&format!("/debug/metrics/{tag}"))?;
        let mut total = 0u64;
        if let Some(rows) = v.get("rows").and_then(|r| r.as_array()) {
            for row in rows {
                if row.get("tag").and_then(|t| t.as_str()) == Some(tag) {
                    total = total.saturating_add(row.get("counter").and_then(|c| c.as_u64()).unwrap_or(0));
                }
            }
        }
        Ok(total)
    }

    pub fn chat(&self, req: ChatRequest) -> io::Result<ChatReply> {
        let mut body = serde_json::json!({
            "messages": req.messages,
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        });
        if let Some(model) = &self.model {
            body["model"] = Value::String(model.clone());
        }
        if let Some(session) = &req.session {
            body["session"] = Value::String(session.clone());
        }
        if let Some(tools) = &req.tools {
            body["tools"] = Value::Array(tools.clone());
        }
        if let Some(kwargs) = &req.chat_template_kwargs {
            body["chat_template_kwargs"] = kwargs.clone();
        }
        let payload = body.to_string();
        let t0 = Instant::now();
        let mut stream = self.connect()?;
        let head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.host,
            payload.len()
        );
        stream.write_all(head.as_bytes())?;
        stream.write_all(payload.as_bytes())?;
        let (status, mut reader, chunked) = read_head(stream)?;
        let raw = read_body(&mut reader, chunked)?;
        if status != 200 {
            let msg = serde_json::from_str::<Value>(&raw)
                .ok()
                .and_then(|v| {
                    v.get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .map(str::to_string)
                })
                .unwrap_or(raw);
            return Err(io::Error::other(format!("HTTP {status}: {msg}")));
        }
        let mut reply = parse_sse(&raw)?;
        reply.elapsed_ms = t0.elapsed().as_millis();
        Ok(reply)
    }

    fn connect(&self) -> io::Result<TcpStream> {
        let s = TcpStream::connect(&self.host).map_err(|e| match e.kind() {
            io::ErrorKind::ConnectionRefused => {
                io::Error::other(format!("{} 拒绝连接(server 未起?)", self.host))
            }
            _ => e,
        })?;
        s.set_read_timeout(Some(IO_TIMEOUT))?;
        s.set_write_timeout(Some(IO_TIMEOUT))?;
        Ok(s)
    }
}

fn parse_sse(raw: &str) -> io::Result<ChatReply> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls: BTreeMap<usize, ToolCall> = BTreeMap::new();
    let mut finish: Option<String> = None;
    let mut usage = Usage::default();
    let mut saw_done = false;
    for line in raw.lines() {
        let line = line.trim_end_matches('\r');
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data.trim() == "[DONE]" {
            saw_done = true;
            break;
        }
        let v: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(err) = v.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| err.to_string());
            return Err(io::Error::other(format!("流内错误: {msg}")));
        }
        if let Some(u) = v.get("usage") {
            usage.prompt_tokens = u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
            usage.completion_tokens =
                u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0);
            usage.cached_tokens = u
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            usage.reasoning_tokens = u
                .get("completion_tokens_details")
                .and_then(|d| d.get("reasoning_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
        }
        let Some(choice) = v
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
        else {
            continue;
        };
        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            finish = Some(fr.to_string());
        }
        let Some(delta) = choice.get("delta") else { continue };
        if let Some(s) = delta.get("content").and_then(Value::as_str) {
            content.push_str(s);
        }
        for key in ["reasoning_content", "reasoning", "reasoning_text"] {
            if let Some(s) = delta.get(key).and_then(Value::as_str) {
                reasoning.push_str(s);
                break;
            }
        }
        if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
            for tc in tcs {
                let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let e = calls.entry(idx).or_insert_with(|| ToolCall {
                    index: idx,
                    id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                });
                if let Some(id) = tc.get("id").and_then(Value::as_str) {
                    e.id = id.to_string();
                }
                if let Some(f) = tc.get("function") {
                    if let Some(name) = f.get("name").and_then(Value::as_str) {
                        e.name.push_str(name);
                    }
                    if let Some(args) = f.get("arguments").and_then(Value::as_str) {
                        e.arguments.push_str(args);
                    }
                }
            }
        }
    }
    let mut tool_calls: Vec<ToolCall> = calls.into_values().collect();
    for tc in &mut tool_calls {
        if tc.arguments.trim().is_empty() {
            tc.arguments = "{}".to_string();
        }
        if tc.name.is_empty() {
            return Err(io::Error::other(format!("tool_call[{0}] 缺 name", tc.index)));
        }
    }
    if !saw_done && finish.is_none() {
        return Err(io::Error::other(
            "SSE 结束但既无 [DONE] 也无 finish_reason(服务端可能被中断)",
        ));
    }
    Ok(ChatReply {
        content,
        reasoning,
        tool_calls,
        finish_reason: finish.unwrap_or_else(|| "stop".to_string()),
        usage,
        elapsed_ms: 0,
    })
}

fn read_head(stream: TcpStream) -> io::Result<(u16, BufReader<TcpStream>, bool)> {
    let mut reader = BufReader::new(stream.try_clone()?);
    // Keep the original stream alive until the body reader is built; the clone
    // shares the same socket, and we drop `stream` after headers are read.
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut chunked = false;
    loop {
        let mut h = String::new();
        let n = reader.read_line(&mut h)?;
        if n == 0 || h.trim().is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
            chunked = true;
        }
    }
    drop(stream);
    Ok((status, reader, chunked))
}

fn read_body(reader: &mut BufReader<TcpStream>, chunked: bool) -> io::Result<String> {
    let mut out = String::new();
    if chunked {
        let mut d = BufReader::new(DechunkReader {
            inner: reader,
            remaining: 0,
            done: false,
        });
        d.read_to_string(&mut out)?;
    } else {
        reader.read_to_string(&mut out)?;
    }
    Ok(out)
}

/// Decode an HTTP/1.1 chunked body into plain bytes.
struct DechunkReader<'a> {
    inner: &'a mut BufReader<TcpStream>,
    remaining: usize,
    done: bool,
}

impl Read for DechunkReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() || self.done {
            return Ok(0);
        }
        if self.remaining == 0 {
            let mut line = String::new();
            loop {
                line.clear();
                let n = self.inner.read_line(&mut line)?;
                if n == 0 {
                    self.done = true;
                    return Ok(0);
                }
                if !line.trim().is_empty() {
                    break;
                }
            }
            let size_str = line.trim().split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size_str, 16).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("非法 chunk 长度行: {line:?}"),
                )
            })?;
            if size == 0 {
                loop {
                    let mut trailer = String::new();
                    let n = self.inner.read_line(&mut trailer)?;
                    if n == 0 || trailer.trim().is_empty() {
                        break;
                    }
                }
                self.done = true;
                return Ok(0);
            }
            self.remaining = size;
        }
        let want = buf.len().min(self.remaining);
        let n = self.inner.read(&mut buf[..want])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "chunked body 提前结束",
            ));
        }
        self.remaining -= n;
        if self.remaining == 0 {
            let mut crlf = [0u8; 2];
            self.inner.read_exact(&mut crlf)?;
        }
        Ok(n)
    }
}
