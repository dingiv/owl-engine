#!/usr/bin/env python3
"""agent wire 面冒烟(G2-lite;**agent 服务面专项 §5**,无需 GPU 侧金标)。

对**已启动**的 owl server 跑一遍 OpenAI wire 契约断言(§3 口径):
三通道帧序 / finish_reason / usage 帧 / 工具历史可渲染 / 未支持字段 400 /
未知字段不静默。用 stdlib 实现,零依赖。

用法:
    python3 agent_wire_smoke.py --base-url http://127.0.0.1:8135 --model qwen3.5-0.8b \\
        [--tools 1] [--no-tool-history]
退出码:0 全过,1 有失败(逐项打印)。
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "number"},
                },
                "required": ["command"],
            },
        },
    }
]

HISTORY = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "列目录"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "先列目录。",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls", "timeout": 30}'},
            }
        ],
    },
    {"role": "tool", "content": "hello.txt", "tool_call_id": "call_1"},
    {"role": "user", "content": "现在读它"},
]

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def post(base: str, body: dict, stream: bool):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def sse_frames(raw: str):
    out = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            out.append(None if payload == "[DONE]" else json.loads(payload))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8135")
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--max-tokens", type=int, default=24)
    a = ap.parse_args()
    base, model = a.base_url.rstrip("/"), a.model

    # ① 元信息面
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        ids = [m["id"] for m in json.loads(r.read())["data"]]
    check("GET /v1/models 列出模型", model in ids, str(ids))

    # ② 流式:三通道帧序 + finish_reason + usage 帧 + [DONE]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "用一句话说明什么是目录。"}],
        "max_tokens": a.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    st, raw = post(base, body, True)
    fr = sse_frames(raw) if st == 200 else []
    check("流式 200", st == 200, f"status={st} {raw[:200]}")
    if fr:
        check("末帧 [DONE]", raw.rstrip().endswith("[DONE]"))
        check("SSE 帧含 choices[0].delta", all("choices" in f for f in fr if f))
        check("有 finish_reason(∈ stop/length/tool_calls)",
              any((f or {}).get("choices") and f["choices"][0].get("finish_reason") in ("stop", "length", "tool_calls") for f in fr))
        usage_frames = [f for f in fr if f and f.get("choices") == [] and "usage" in f]
        check("include_usage → choices=[] 的 usage 帧", len(usage_frames) == 1,
              json.dumps(usage_frames[:1]))
        if usage_frames:
            u = usage_frames[0]["usage"]
            check("usage.prompt_tokens > 0", u.get("prompt_tokens", 0) > 0, json.dumps(u))
        check("usage 帧在 [DONE] 之前", raw.rfind("usage") < raw.rfind("[DONE]"))

    # ③ 工具历史(intent:tool_calls + role=tool)可渲染 ⇒ 200 且不被静默丢弃
    body = {
        "model": model,
        "messages": HISTORY,
        "tools": TOOLS,
        "max_tokens": a.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    }
    st, raw = post(base, body, False)
    check("工具历史(assistant.tool_calls + role=tool)渲染 → 200", st == 200, f"status={st} {raw[:300]}")
    if st == 200:
        msg = json.loads(raw)["choices"][0]["message"]
        check("非流式 message 含 role=assistant", msg.get("role") == "assistant")
        check("非流式 finish_reason 合法",
              json.loads(raw)["choices"][0]["finish_reason"] in ("stop", "length", "tool_calls"))

    # ④ tools 空数组必须接受(pi 的 hasToolHistory 分支)
    body = {"model": model, "messages": HISTORY, "tools": [], "max_tokens": 8, "stream": False}
    st, raw = post(base, body, False)
    check("tools: [] 空数组接受(pi hasToolHistory 分支)", st == 200, f"status={st} {raw[:200]}")

    # ⑤ 未支持/非法字段:400(不许静默忽略)
    bad = [
        ("n=2", {"n": 2}),
        ("logprobs=true", {"logprobs": True}),
        ("tool_choice=required", {"tool_choice": "required"}),
        ("role=badguy", {"messages": [{"role": "badguy", "content": "x"}]}),
        ("tool 消息缺 tool_call_id", {"messages": [{"role": "tool", "content": "x"}]}),
        ("assistant.tool_calls 的 arguments 非 JSON",
         {"messages": [{"role": "assistant", "content": "", "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{oops"}}]}]}),
        ("messages 为空", {"messages": []}),
    ]
    for name, patch in bad:
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        body.update(patch)
        st, raw = post(base, body, False)
        check(f"非法/未支持 → 400:{name}", st == 400, f"status={st} {raw[:160]}")

    # ⑥ 采样参数:收下 + 200(暂未生效,M-A 只要求不静默;日志侧另有 warn)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False,
    }
    st, raw = post(base, body, False)
    check("temperature/top_p 收下不报错(M-B 接引擎)", st == 200, f"status={st} {raw[:160]}")

    print()
    if FAILS:
        print(f"❌ {len(FAILS)} 项失败:{FAILS}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
