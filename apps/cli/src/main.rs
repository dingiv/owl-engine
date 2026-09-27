//! owl 交互客户端(**架子**):终端 REPL → owl server(OpenAI 兼容
//! SSE)——代替手发 HTTP 命令。
//!
//! 形态:起手探活 `/health`(报模型名)→ 阻塞终端循环:读一行 →
//! POST `/v1/chat/completions`(stream)→ 逐 delta 流式渲染 → 完了再
//! 读下一行(多轮对话)。
//!
//! 多轮语义(OpenAI 协议):客户端持有**全量历史**,每轮回传;固定
//! session 键(默认按启动时间生成)→ 引擎侧同会话,增量 prefill 命中
//! 与否由前缀守卫裁决(模板 naive 渲染期通常回退全量,正确性保住)。
//!
//! 命令:`/new` 开新会话(清历史 + 换 session 键)、`/help`、`/quit`。
//! 参数:`--url <base>`(默认 env OWL_SERVER_URL 或 127.0.0.1:8135)、
//! `--max-tokens <n>`(默认 64;须给 prompt 留量,引擎槽位 256)、
//! `--session <key>`、`--model <name>`。
//!
//! 用法:
//! ```text
//! cargo run -p cli                        # 默认连 127.0.0.1:8135
//! cargo run -p cli -- --url http://127.0.0.1:8135 --max-tokens 96
//! ```
//!
//! 架子边界(挂账):纯 std 阻塞实现(客户端零异步依赖;SSE close
//! 定界逐行读)、无超时/断线重连、Ctrl-C 即退(无优雅收尾)。

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::{SystemTime, Instant};

use serde_json::{json, Value};

const DEFAULT_URL: &str = "http://127.0.0.1:8135";
const DEFAULT_MAX_TOKENS: usize = 64;

struct Cfg {
    /// host:port(去掉 scheme 的连接目标)
    host: String,
    model: Option<String>,
    max_tokens: usize,
    session: String,
}

fn parse_args() -> Cfg {
    let base = std::env::var("OWL_SERVER_URL").unwrap_or_else(|_| DEFAULT_URL.into());
    let mut cfg = Cfg {
        host: base
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .trim_end_matches('/')
            .to_string(),
        model: None,
        max_tokens: DEFAULT_MAX_TOKENS,
        session: format!("cli-{}", SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)),
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| die(&format!("{a} 缺参数值")));
        match a.as_str() {
            "--url" => {
                cfg.host = val()
                    .trim_start_matches("http://")
                    .trim_end_matches('/')
                    .to_string();
            }
            "--max-tokens" => cfg.max_tokens = val().parse().unwrap_or_else(|_| die("--max-tokens 非数字")),
            "--session" => cfg.session = val(),
            "--model" => cfg.model = Some(val()),
            other => die(&format!("未知参数 {other}(--url/--max-tokens/--session/--model)")),
        }
    }
    cfg
}

fn die(msg: &str) -> ! {
    eprintln!("[cli] {msg}");
    std::process::exit(2);
}

fn main() {
    let cfg = parse_args();
    // 起手探活(失败不拦,允许 server 后起;首条消息时会再报错)
    match get_health(&cfg.host) {
        Ok(model) => eprintln!("[cli] 已连接 {0}(模型 {1});/help 看命令", cfg.host, model),
        Err(e) => eprintln!("[cli] 探活失败({e};server 未起?照常进入会话)"),
    }

    let mut history: Vec<Value> = Vec::new();
    let mut session = cfg.session.clone();
    let stdin = io::stdin();
    loop {
        print!("\n你› ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            eprintln!("\n[cli] EOF,退出");
            break;
        }
        match line.trim() {
            "" => continue,
            "/quit" | "/exit" | "/q" => break,
            "/help" => {
                eprintln!("[cli] 直接输字发消息;/new 新会话(清历史)/ /quit 退出");
                continue;
            }
            "/new" => {
                history.clear();
                session = format!("{0}-{1}", cfg.session, SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0));
                eprintln!("[cli] 已开新会话(session {session})");
                continue;
            }
            cmd => {
                history.push(json!({"role": "user", "content": cmd}));
            }
        }

        let body = match cfg.model {
            Some(ref m) => json!({"model": m, "messages": history, "stream": true,
                                  "max_tokens": cfg.max_tokens, "session": session}),
            None => json!({"messages": history, "stream": true,
                           "max_tokens": cfg.max_tokens, "session": session}),
        };
        match chat_stream(&cfg.host, &body) {
            Ok(text) => history.push(json!({"role": "assistant", "content": text})),
            Err(e) => {
                eprintln!("\n[cli] 请求失败: {e}");
                history.pop(); // 失败轮不入历史,可重发
            }
        }
    }
}

/// GET /health → 模型名
fn get_health(host: &str) -> std::io::Result<String> {
    let mut stream = connect(host)?;
    let req = format!("GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes())?;
    let (status, mut reader) = read_head(stream)?;
    let mut body = String::new();
    reader.read_to_string(&mut body)?;
    if status != 200 {
        return Err(io::Error::other(format!("health HTTP {status}: {body}")));
    }
    Ok(serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from))
        .unwrap_or_else(|| "?".into()))
}

/// POST /v1/chat/completions(stream):逐 delta 流式渲染,回吐全文。
/// 统计:TTFT 首 chunk / chunk 数 / 总耗时(引擎事件即 token 粒度)
fn chat_stream(host: &str, body: &Value) -> std::io::Result<String> {
    let payload = body.to_string();
    let mut stream = connect(host)?;
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(req.as_bytes())?;
    let (status, mut reader) = read_head(stream)?;
    if status != 200 {
        let mut err_body = String::new();
        reader.read_to_string(&mut err_body)?;
        let msg = serde_json::from_str::<Value>(&err_body)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(String::from)
            })
            .unwrap_or(err_body);
        return Err(io::Error::other(format!("HTTP {status}: {msg}")));
    }

    print!("owl› ");
    io::stdout().flush().ok();
    let t0 = Instant::now();
    let mut ttft = None;
    let (mut out, mut ntok) = (String::new(), 0usize);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break; // 连接关(server close 定界收口)
        }
        let Some(data) = line.trim_end().strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(err) = v.get("error") {
            eprintln!("\n[cli] 流内错误: {err}");
            break;
        }
        if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
            if delta.is_empty() {
                continue; // role 首帧(content="")不计 TTFT/chunk 数
            }
            if ttft.is_none() {
                ttft = Some(t0.elapsed());
            }
            print!("{delta}");
            io::stdout().flush().ok();
            out.push_str(delta);
            ntok += 1;
        }
    }
    let total = t0.elapsed();
    let tpot = if ntok > 0 { total.as_millis() / ntok as u128 } else { 0 };
    eprintln!(
        "\n[stat] TTFT {:?} | {} chunks | 总 {:.2}s | {:?}/chunk",
        ttft.unwrap_or(total),
        ntok,
        total.as_secs_f32(),
        tpot
    );
    Ok(out)
}

fn connect(host: &str) -> io::Result<TcpStream> {
    TcpStream::connect(host).map_err(|e| match e.kind() {
        io::ErrorKind::ConnectionRefused => {
            io::Error::other(format!("{host} 拒绝连接(server 未起?)"))
        }
        _ => e,
    })
}

/// 读响应头 → (状态码, 剩余 body 读取器)
fn read_head(stream: TcpStream) -> io::Result<(u16, BufReader<TcpStream>)> {
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    loop {
        let mut h = String::new();
        reader.read_line(&mut h)?;
        if h.trim().is_empty() {
            break;
        }
    }
    Ok((status, reader))
}
