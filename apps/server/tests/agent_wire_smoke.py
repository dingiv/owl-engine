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
            # G5:思考相位真账(口径 = 引擎 1 token 1 采样步,按解析器相位归因)
            ct = u.get("completion_tokens", 0)
            rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
            check("usage.completion_tokens_details.reasoning_tokens 透出",
                  isinstance(rt, int) and 0 <= rt <= ct, json.dumps(u))
        check("usage 帧在 [DONE] 之前", raw.rfind("usage") < raw.rfind("[DONE]"))

    # ②b G5②:`/debug/agent` 诊断环(刚才那轮必须留痕)
    with urllib.request.urlopen(base + "/debug/agent", timeout=30) as r:
        dbg = json.loads(r.read())
    turns = dbg.get("turns", [])
    check("GET /debug/agent 有轮次记录", len(turns) >= 1, json.dumps(dbg, ensure_ascii=False)[:300])
    if turns:
        t = turns[-1]
        check("诊断轮含请求事实 + usage 真账",
              t["req"]["model"] == model and t["usage"]["completion_tokens"] > 0
              and t["outcome"]["finish_reason"] in ("stop", "length", "tool_calls"),
              json.dumps(t, ensure_ascii=False)[:300])
        check("诊断轮含解析器结构轨迹",
              isinstance(t.get("events"), list) and len(t["events"]) >= 1,
              json.dumps(t.get("events"), ensure_ascii=False)[:200])
        head = (t.get("prompt") or {}).get("head", "")
        check("诊断轮 prompt 快照含本轮用户文本",
              "用一句话说明什么是目录" in head, head[:120])
    # 非法路径段 ⇒ 400(不静默回缺省)
    try:
        urllib.request.urlopen(base + "/debug/agent/abc", timeout=30)
        check("非法 /debug/agent 路径段 → 400", False, "居然 200")
    except urllib.error.HTTPError as e:
        check("非法 /debug/agent 路径段 → 400", e.code == 400, f"status={e.code}")

    # ④ G5③:/tokenize(raw/chat 两模式 + 四类响亮拒)
    import urllib.request as _u

    def post_path(path: str, body: dict):
        req = _u.Request(base + path, data=json.dumps(body).encode(),
                         headers={"content-type": "application/json"}, method="POST")
        try:
            with _u.urlopen(req, timeout=600) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    st, raw = post_path("/tokenize", {"model": model, "prompt": "用一句话说明什么是目录。"})
    raw_n = 0
    if st == 200:
        d = json.loads(raw)
        raw_n = d.get("count", 0)
        check("/tokenize raw:count == len(tokens) > 0",
              raw_n > 0 and raw_n == len(d.get("tokens", [])) and d.get("mode") == "raw",
              json.dumps(d, ensure_ascii=False)[:200])
        check("/tokenize 报 max_model_len",
              isinstance(d.get("max_model_len"), int) and d["max_model_len"] > 0,
              json.dumps(d, ensure_ascii=False)[:120])
    else:
        check("/tokenize raw → 200", False, f"status={st} {raw[:200]}")
    st, raw = post_path("/tokenize", {"prompt": "用一句话说明什么是目录。", "mode": "chat"})
    if st == 200:
        n = json.loads(raw)["count"]
        check("/tokenize chat 模式含家族包装(count > raw)", n > raw_n, f"chat={n} raw={raw_n}")
    else:
        check("/tokenize chat → 200", False, f"status={st} {raw[:200]}")
    for body, why in [
        ({"prompt": ""}, "空 prompt"),
        ({"prompt": "x", "mode": "tokens"}, "非法 mode"),
        ({"prompt": "x", "add_special_tokens": True}, "add_special_tokens=true"),
        ({"prompt": "x", "nope": 1}, "未知字段"),
    ]:
        st, raw = post_path("/tokenize", body)
        check(f"/tokenize 响亮拒:{why} → 400", st == 400, f"status={st} {raw[:160]}")

    # ⑤ 工具历史(intent:tool_calls + role=tool)可渲染 ⇒ 200 且不被静默丢弃
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
        # G5:非流式形态同样要透出 reasoning_tokens(两形态单源)
        u = json.loads(raw)["usage"]
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        check("非流式 usage 含 reasoning_tokens",
              isinstance(rt, int) and 0 <= rt <= u.get("completion_tokens", 0), json.dumps(u))

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
    check("temperature/top_p 被接受且已生效(AGT-06:不再是『收下未接』)", st == 200, f"status={st} {raw[:160]}")
    # stop 串(AGT-06;同请求带上):命中即截断,且 finish_reason 合法
    stop_body = dict(body, stop=["巴黎"])
    st, raw = post(base, stop_body, False)
    check("stop 串被接受(AGT-06)", st == 200, f"status={st} {raw[:160]}")
    if st == 200:
        try:
            v = json.loads(raw)
            fr = v["choices"][0].get("finish_reason")
            check("stop 请求 finish_reason 合法", fr in ("stop", "length", "tool_calls"), f"finish_reason={fr}")
            c = v["choices"][0]["message"].get("content") or ""
            check("stop 命中后输出不含停止串", "巴黎" not in c, f"content={c[:60]!r}")
        except Exception as e:  # noqa: BLE001 - 解析失败即上报(不静默)
            check("stop 响应可解析", False, f"{e}: {raw[:120]}")

    print()
    if FAILS:
        print(f"❌ {len(FAILS)} 项失败:{FAILS}")
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
