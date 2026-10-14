//! 极简 HTTP/1.1 服务面(**架子**:锁文件零 HTTP 栈依赖,手写最小面)。
//!
//! 现状:每请求 `Connection: close`,SSE 用 close 定界(无 chunked 响应);
//! 请求体支持 **Content-Length 与 `Transfer-Encoding: chunked` 两定界**
//!(AGT-11,2026-10-14)。**挂账**:keep-alive —— 非不能,而是与并发闸门
//! 语义相冲:现在"一连接 = 一个 permit"(main.rs),keep-alive 会让空闲
//! 连接长期占住 permit(pi 各连接池数个死连接 × max_concurrency=4 ⇒
//! 饿死)。正确修法是连接级 permit + 空闲超时,归 axum/hyper 迁移一并做。

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

/// 读一个请求(对端关闭 = None)。请求体定界 = Content-Length **或**
/// `Transfer-Encoding: chunked`(两者都不给 = 空体)
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
    let mut chunked = false;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            } else if k.trim().eq_ignore_ascii_case("transfer-encoding")
                && v.to_ascii_lowercase().contains("chunked")
            {
                chunked = true;
            }
        }
    }
    if content_length > MAX_BODY {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "body 过大"));
    }
    let mut body = buf[head_end..].to_vec();
    if chunked {
        body = read_chunked(r, body).await?;
    } else {
        while body.len() < content_length {
            let n = r.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            body.extend_from_slice(&chunk[..n]);
        }
        body.truncate(content_length);
    }
    if body.len() > MAX_BODY {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "body 过大"));
    }
    Ok(Some(Request {
        method: method.to_string(),
        path: target.split('?').next().unwrap_or("/").to_string(),
        body,
    }))
}

/// `Transfer-Encoding: chunked` 解码(AGT-11)。`pending` = 已读入头部之后
/// 的残字节(可能已含若干块,甚至完整体)。
///
/// 语法:`<hex>[;ext]\r\n<data>\r\n` … 终止块 `0\r\n` + 可选 trailer + `\r\n`。
async fn read_chunked<R: AsyncReadExt + Unpin>(
    r: &mut R,
    mut pending: Vec<u8>,
) -> io::Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();
    let mut read = [0u8; 4096];
    loop {
        // ① 块头行(以 CRLF 结尾)
        let line_end = loop {
            if let Some(p) = find(&pending, b"\r\n") {
                break p;
            }
            if pending.len() > 64 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "chunked 块头过长"));
            }
            let n = r.read(&mut read).await?;
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "chunked 提前结束"));
            }
            pending.extend_from_slice(&read[..n]);
        };
        let line = String::from_utf8_lossy(&pending[..line_end]).to_string();
        pending.drain(..line_end + 2);
        let hex = line.split(';').next().unwrap_or("").trim();
        let n = usize::from_str_radix(hex, 16)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("chunked 块长非法:{hex:?}")))?;
        if n == 0 {
            return Ok(out); // 终止块(不解析 trailer —— 没人在此场景发)
        }
        if out.len() + n > MAX_BODY {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "body 过大"));
        }
        // ② 块数据 + 后随 CRLF
        while pending.len() < n + 2 {
            let got = r.read(&mut read).await?;
            if got == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "chunked 数据截断"));
            }
            pending.extend_from_slice(&read[..got]);
        }
        out.extend_from_slice(&pending[..n]);
        pending.drain(..n + 2);
    }
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
/// 纯文本应答(诊断面:/debug/perf 的树状报告 —— 人读优先,不过 JSON)
pub async fn respond_text<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    code: u16,
    body: &str,
) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_text(code),
        body.len()
    );
    w.write_all(head.as_bytes()).await?;
    w.write_all(body.as_bytes()).await?;
    w.flush().await
}

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

#[cfg(test)]
mod tests {
    use super::read_request;

    /// AGT-11:chunked 请求体解码(含分块切碎到多读、扩展参数、跨读边界)
    #[tokio::test]
    async fn chunked_body_is_decoded() {
        let raw = concat!(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            "Host: x\r\n",
            "Transfer-Encoding: chunked\r\n\r\n",
            "4\r\n{\"aa\r\n",
            "4;ext=1\r\n\":1}\r\n",
            "0\r\n\r\n"
        );
        let mut cur = raw.as_bytes();
        let req = read_request(&mut cur).await.expect("read").expect("some");
        assert_eq!(req.path, "/v1/chat/completions");
        assert_eq!(req.body, br#"{"aa":1}"#);
    }

    /// 尾部残字节里已含若干块(头部与体同读命中)也要正确剥出
    #[tokio::test]
    async fn chunked_body_fully_buffered() {
        let raw = "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n";
        let mut cur = raw.as_bytes();
        let req = read_request(&mut cur).await.expect("read").expect("some");
        assert_eq!(req.body, b"abc");
    }

    /// Content-Length 路径不回归
    #[tokio::test]
    async fn content_length_still_works() {
        let raw = "POST /x HTTP/1.1\r\nContent-Length: 6\r\n\r\nabcdef";
        let mut cur = raw.as_bytes();
        let req = read_request(&mut cur).await.expect("read").expect("some");
        assert_eq!(req.body, b"abcdef");
    }

    /// 块长非法 / 数据截断 = 结构化 io 错误(不静默返回半截体)
    #[tokio::test]
    async fn chunked_malformed_is_error() {
        let raw = "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\nabc\r\n";
        let mut cur = raw.as_bytes();
        assert!(read_request(&mut cur).await.is_err(), "非法块长应报错");
        let raw = "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nab";
        let mut cur = raw.as_bytes();
        assert!(read_request(&mut cur).await.is_err(), "截断应报错");
    }
}
