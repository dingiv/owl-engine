//! 极简 HTTP/1.1 服务面(**架子**:锁文件零 HTTP 栈依赖,手写最小面)。
//!
//! 现状(**2026-10-19 G5④:keep-alive 已开**):
//! - 请求体支持 **Content-Length 与 `Transfer-Encoding: chunked` 两定界**
//!   (AGT-11,2026-10-14);
//! - 应答:JSON/文本 = `Content-Length` 定界(自定界 ⇒ 可复用连接);
//!   **SSE = `Transfer-Encoding: chunked`**(见下);
//! - 连接复用判定在 [`read_request`] 里落地(HTTP/1.1 缺省 keep-alive;
//!   `Connection: close` 尊重;HTTP/1.0 只认显式 `keep-alive`)——
//!   **连接的生命周期由 serve 循环掌握**(空闲超时/请求数上限/非 1.1 拒)。
//!
//! 为什么 SSE 必须 chunked:旧路用“连接关闭”定界 ⇒ 一次只能服务一个请求
//! (pi 的连接池会因此把所有并发位耗在空闲连接上)。chunked 让流自带终止块
//! (`0\r\n\r\n`,见 [`sse_end`]),连接随后可复用。

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
    /// 本连接**可否复用**(HTTP/1.1 缺省 true;`Connection: close` ⇒ false;
    /// HTTP/1.0 只认显式 `Connection: keep-alive` ⇒ 缺省 false)
    pub keep_alive: bool,
    /// 请求行是否为 `HTTP/1.1`(非 1.1 = 服务面拒:chunked 应答是 1.1 语法)
    pub http_11: bool,
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
    // 版本:请求行第三段(缺省/异形 = 非 1.1 ⇒ 后续 serve 拒;
    // chunked 应答是 1.1 语法,静默降级会跟复用判定撞车)
    let http_11 = request_line.split_whitespace().nth(2) == Some("HTTP/1.1");
    let mut content_length = 0usize;
    let mut chunked = false;
    let mut conn_close = false;
    let mut conn_keep_alive = false;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            } else if k.trim().eq_ignore_ascii_case("transfer-encoding")
                && v.to_ascii_lowercase().contains("chunked")
            {
                chunked = true;
            } else if k.trim().eq_ignore_ascii_case("connection") {
                let v = v.to_ascii_lowercase();
                // 逐 token 判(允许 `keep-alive, close` 这类列表形)
                for tok in v.split(',') {
                    match tok.trim() {
                        "close" => conn_close = true,
                        "keep-alive" => conn_keep_alive = true,
                        _ => {}
                    }
                }
            }
        }
    }
    let keep_alive = if http_11 { !conn_close } else { conn_keep_alive && !conn_close };
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
        keep_alive,
        http_11,
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

/// JSON 应答(`Content-Length` 自定界 ⇒ **不发 `Connection` 头**:
/// HTTP/1.1 缺省即复用;关不关由 serve 循环决定,见 [`crate::http`] 头注)
pub async fn respond_json<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    code: u16,
    v: &Value,
) -> io::Result<()> {
    let body = serde_json::to_vec(v).map_err(io::Error::other)?;
    let head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        status_text(code),
        body.len()
    );
    w.write_all(head.as_bytes()).await?;
    w.write_all(&body).await?;
    w.flush().await
}

/// 纯文本应答(诊断面:/debug/perf 的树状报告 —— 人读优先,不过 JSON)
pub async fn respond_text<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    code: u16,
    body: &str,
) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {code} {}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\n\r\n",
        status_text(code),
        body.len()
    );
    w.write_all(head.as_bytes()).await?;
    w.write_all(body.as_bytes()).await?;
    w.flush().await
}

/// SSE 响应头(**chunked 定界**:流自带终止块,连接可复用 —— G5④)
pub async fn respond_sse_head<W: AsyncWriteExt + Unpin>(w: &mut W) -> io::Result<()> {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n";
    w.write_all(head.as_bytes()).await?;
    w.flush().await
}

/// 一条 SSE data 事件(chunked 帧;json 文本须已单行化 —— serde_json 天然无换行)
pub async fn sse_data<W: AsyncWriteExt + Unpin>(w: &mut W, data: &str) -> io::Result<()> {
    let payload = format!("data: {data}\n\n");
    w.write_all(format!("{:x}\r\n", payload.len()).as_bytes()).await?;
    w.write_all(payload.as_bytes()).await?;
    w.write_all(b"\r\n").await?;
    w.flush().await
}

/// SSE 终止块(`0\r\n\r\n`)—— **每个 chunked 流都必须调**(否则客户端永远
/// 等下一个 chunk;G5④ 起 SSE 不再用连接关闭定界)
pub async fn sse_end<W: AsyncWriteExt + Unpin>(w: &mut W) -> io::Result<()> {
    w.write_all(b"0\r\n\r\n").await?;
    w.flush().await
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::{read_request, respond_sse_head, sse_data, sse_end, Request};

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

    /// **G5④ 复用判定**:HTTP/1.1 缺省复用;`Connection: close` 尊重
    /// (含列表形);HTTP/1.0 只认显式 `keep-alive` —— 且版本信号必须透出
    /// (非 1.1 的 chunked 应答是非法语法,serve 据此拒)
    #[tokio::test]
    async fn keep_alive_defaults_and_close_honored() {
        async fn read1(raw: &'static str) -> Request {
            let mut cur = raw.as_bytes();
            read_request(&mut cur).await.expect("read").expect("some")
        }
        let r = read1("GET /x HTTP/1.1\r\n\r\n").await;
        assert!(r.keep_alive && r.http_11, "1.1 缺省 = 复用");
        let r = read1("GET /x HTTP/1.1\r\nConnection: close\r\n\r\n").await;
        assert!(!r.keep_alive, "显式 close 要尊重");
        let r = read1("GET /x HTTP/1.1\r\nConnection: keep-alive, close\r\n\r\n").await;
        assert!(!r.keep_alive, "列表形里的 close 也要认");
        let r = read1("GET /x HTTP/1.0\r\n\r\n").await;
        assert!(!r.keep_alive && !r.http_11, "1.0 缺省 = 关");
        let r = read1("GET /x HTTP/1.0\r\nConnection: keep-alive\r\n\r\n").await;
        assert!(r.keep_alive && !r.http_11, "1.0 显式 keep-alive 才复用");
    }

    /// **G5④ SSE 帧**:chunked 长度行 + 后随 CRLF + 终止块 —— 不再 close 定界
    #[tokio::test]
    async fn sse_is_chunked_and_terminated() {
        let mut out: Vec<u8> = Vec::new();
        respond_sse_head(&mut out).await.expect("head");
        sse_data(&mut out, "[DONE]").await.expect("data");
        sse_end(&mut out).await.expect("end");
        let s = String::from_utf8(out).expect("utf8");
        assert!(s.starts_with("HTTP/1.1 200 OK\r\n"), "{s:?}");
        assert!(s.contains("Transfer-Encoding: chunked\r\n"), "{s:?}");
        assert!(!s.contains("Connection: close"), "SSE 不再 close 定界:{s:?}");
        assert!(
            s.contains("\r\ndata: [DONE]\n\n\r\n"),
            "chunk 帧 = 长度行 + 体 + CRLF:{s:?}"
        );
        // chunk 帧 = 十六进制长度行 + 体 + CRLF(长度按实际字节算,不写死)
        let payload = "data: [DONE]\n\n";
        assert!(
            s.contains(&format!("\r\n{:x}\r\n{payload}\r\n", payload.len())),
            "{s:?}"
        );
        assert!(s.ends_with("0\r\n\r\n"), "终止块:{s:?}");
    }
}
