//! 极简 HTTP/1.1 服务面(**架子**:锁文件零 HTTP 栈依赖,手写最小面)。
//!
//! 现状:每请求 `Connection: close`,SSE 用 close 定界(无 chunked);
//! 覆盖 curl / OpenAI SDK 的流式与非流式消费。**挂账**:axum/hyper 迁移
//! (keep-alive / chunked / 并发连接纪律随 M2 batching 一并立项)。

use std::io;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MAX_HEAD: usize = 64 * 1024;
const MAX_BODY: usize = 16 * 1024 * 1024;

pub struct Request {
    pub method: String,
    /// 不含 query 的路径
    pub path: String,
    pub body: Vec<u8>,
}

/// 读一个请求(对端关闭 = None)。只认 Content-Length 定界(架子不
/// 支持 chunked 请求体;OpenAI 客户端均为定长 body)
pub async fn read_request<R: AsyncReadExt + Unpin>(r: &mut R) -> io::Result<Option<Request>> {
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        let n = r.read(&mut chunk).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(p) = find(&buf, b"\r\n\r\n") {
            break p + 4;
        }
        if buf.len() > MAX_HEAD {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "头部过大"));
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]);
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "请求行非法"));
    };
    let mut content_length = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    if content_length > MAX_BODY {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "body 过大"));
    }
    let mut body = buf[head_end..].to_vec();
    while body.len() < content_length {
        let n = r.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Ok(Some(Request {
        method: method.to_string(),
        path: target.split('?').next().unwrap_or("/").to_string(),
        body,
    }))
}

fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

/// JSON 应答(Connection: close;写完由调用方关流)
pub async fn respond_json<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    code: u16,
    v: &Value,
) -> io::Result<()> {
    let body = serde_json::to_vec(v).map_err(io::Error::other)?;
    let head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_text(code),
        body.len()
    );
    w.write_all(head.as_bytes()).await?;
    w.write_all(&body).await?;
    w.flush().await
}

/// SSE 响应头(close 定界;无 Content-Length —— HTTP/1.1 连接关闭即报文完)
pub async fn respond_sse_head<W: AsyncWriteExt + Unpin>(w: &mut W) -> io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    w.write_all(head.as_bytes()).await?;
    w.flush().await
}

/// 一条 SSE data 事件(json 文本须已单行化 —— serde_json 序列化天然无换行)
pub async fn sse_data<W: AsyncWriteExt + Unpin>(w: &mut W, data: &str) -> io::Result<()> {
    w.write_all(format!("data: {data}\n\n").as_bytes()).await?;
    w.flush().await
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
