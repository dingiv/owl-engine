#!/usr/bin/env python3
"""keep-alive 传输面探针(G5④;**agent 服务面专项** §9.2(4))。

对**已启动**的 owl server 用**裸 socket** 验证连接复用语义(不用 urllib ——
它会把连接生命周期藏起来,而本探针要检的正是生命周期):

  ① 同一 socket 连发两请求(JSON,`Content-Length` 定界)⇒ 两应答都完整收到
     (且连接未被关;第二次写入不报 EPIPE)。
  ② SSE 请求 ⇒ `Transfer-Encoding: chunked` + 终止块 `0\r\n\r\n`;
     随后**同一 socket** 再发 `/health` ⇒ 仍 200(流后连接可复用 —— 这是
     G5④ 的核心收益:旧实现靠连接关闭定界,一次一连接)。
  ③ `Connection: close` ⇒ 应答后服务端关连接(read 返回 EOF)。
  ④ `HTTP/1.0` ⇒ 505 + 关(服务面只接 1.1;chunked 是 1.1 语法)。

用法:
    python3 keep_alive_probe.py [--host 127.0.0.1] [--port 8135] [--model qwen3.5-0.8b]
退出码:0 全过,1 有失败(逐项打印)。
"""

import argparse
import json
import socket
import sys

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def connect(host: str, port: int) -> socket.socket:
    s = socket.create_connection((host, port), timeout=60)
    s.settimeout(300)
    return s


def send(s: socket.socket, raw: bytes) -> None:
    s.sendall(raw)


def recv_until_close(s: socket.socket, limit: int = 1 << 20) -> bytes:
    """读到对端关(EOF);用于 close 界定/拒后关的连接"""
    out = b""
    while len(out) < limit:
        try:
            b = s.recv(65536)
        except socket.timeout:
            break
        if not b:
            break
        out += b
    return out


def read_http_response(s: socket.socket):
    """读一个自带定界的应答(Content-Length 或 chunked)。

    返回 (head: str, body: bytes, complete: bool)。complete=False = 依赖
    连接关闭定界(旧行为)而对方没关 —— 本探针视之为“未终止”。
    """
    buf = b""
    while b"\r\n\r\n" not in buf:
        b = s.recv(65536)
        if not b:
            return buf.decode("utf-8", "replace"), b"", True  # EOF 定界
        buf += b
    head, rest = buf.split(b"\r\n\r\n", 1)
    head_s = head.decode("utf-8", "replace")
    hdrs = {}
    for line in head_s.split("\r\n")[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            hdrs[k.strip().lower()] = v.strip()
    if "content-length" in hdrs:
        n = int(hdrs["content-length"])
        while len(rest) < n:
            b = s.recv(65536)
            if not b:
                break
            rest += b
        return head_s, rest[:n], len(rest) >= n
    if hdrs.get("transfer-encoding", "").lower() == "chunked":
        # 解帧直到终止块
        out = b""
        data = rest
        while True:
            while b"\r\n" not in data:
                b = s.recv(65536)
                if not b:
                    return head_s, out, False
                data += b
            line, data = data.split(b"\r\n", 1)
            n = int(line.split(b";")[0], 16)
            if n == 0:
                while len(data) < 2:
                    b = s.recv(65536)
                    if not b:
                        break
                    data += b
                return head_s, out, True
            while len(data) < n + 2:
                b = s.recv(65536)
                if not b:
                    return head_s, out, False
                data += b
            out += data[:n]
            data = data[n + 2 :]
    return head_s, rest, False


def get_json(s: socket.socket) -> tuple:
    send(s, b"GET /health HTTP/1.1\r\nHost: h\r\n\r\n")
    head, body, complete = read_http_response(s)
    try:
        return head, json.loads(body), complete
    except json.JSONDecodeError:
        return head, None, complete


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8135)
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--max-tokens", type=int, default=16)
    a = ap.parse_args()
    host, port = a.host, a.port

    # ① 同一连接两请求(JSON 定界)
    s = connect(host, port)
    h1, j1, c1 = get_json(s)
    check("① 首请求 200 + JSON", h1.startswith("HTTP/1.1 200") and isinstance(j1, dict) and j1.get("ok"), h1.split("\r\n")[0])
    check("① 应答自带定界(Content-Length)", "content-length" in h1.lower(), h1[:200])
    h2, j2, c2 = get_json(s)
    check("① 同连接二请求仍 200(连接确实复用)", h2.startswith("HTTP/1.1 200") and isinstance(j2, dict), h2.split("\r\n")[0])

    # ② SSE:chunked + 终止块;然后同连接再发一次普通请求
    body = json.dumps(
        {
            "model": a.model,
            "messages": [{"role": "user", "content": "say hi"}],
            "max_tokens": a.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    send(
        s,
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: h\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body,
    )
    head, payload, complete = read_http_response(s)
    check("② SSE 用 chunked 定界", "transfer-encoding: chunked" in head.lower(), head[:200])
    check("② SSE 有终止块(读到完整体)", complete, f"complete={complete}")
    check("② SSE 体含 [DONE]", b"[DONE]" in payload, payload[-80:].decode("utf-8", "replace"))
    h3, j3, _ = get_json(s)
    check("② 流后同连接还能服务(核心收益)", h3.startswith("HTTP/1.1 200") and isinstance(j3, dict), h3.split("\r\n")[0])
    s.close()

    # ③ Connection: close 被尊重(应答后关连接)
    s = connect(host, port)
    send(s, b"GET /health HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n")
    raw = recv_until_close(s)
    check("③ Connection: close ⇒ 应答后关连接", raw.startswith(b"HTTP/1.1 200") and raw.endswith(b"}"), raw[:60])
    s.close()

    # ④ HTTP/1.0 响亮拒(505)+ 关
    s = connect(host, port)
    send(s, b"GET /health HTTP/1.0\r\nHost: h\r\n\r\n")
    raw = recv_until_close(s)
    check("④ HTTP/1.0 ⇒ 505 + 关", b"505" in raw.split(b"\r\n")[0], raw.split(b"\r\n")[0])
    s.close()

    print("✅ 全部通过" if not FAILS else f"❌ {len(FAILS)} 项失败:{FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
