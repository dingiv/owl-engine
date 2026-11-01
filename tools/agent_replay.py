#!/usr/bin/env python3
"""**真实 agent 轨迹回放 / 长上下文性能测量**(2026-10-31 立项)。

数据来源必须是**真实前端流量**:服务端 `[server] request_dump_dir` 落盘的
每请求一份 body(`/tmp/owl-req-8135/req-*.json`)。本工具不构造 prompt。

三个子命令:

1. `report` —— 把 `/debug/agent` 的每轮事实铺成表(prompt/cached/工具/耗时),
   给"100k 上下文结构"留证据;
2. `replay` —— 拿一份真实请求原样(或**追加一用户轮**)重发,测
   TTFT / prefill / decode;`--nonce` 在首条 system 前插一个随机标记 ⇒
   前缀缓存全 miss,得到**真冷 prefill**;
3. `pick` —— 在 dump 目录里挑出"prompt 最大"的那份请求(= 达到 100k 的那轮)。

用法示例:

    python3 tools/agent_replay.py pick --dir /tmp/owl-req-8135
    python3 tools/agent_replay.py report --port 8135
    python3 tools/agent_replay.py replay --req /tmp/owl-req-8135/req-xxx.json \\
        --append "现在把窗口最大化给我看看" --max-tokens 256 --tag append-warm
    python3 tools/agent_replay.py replay --req ... --nonce --tag cold-100k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

# ---------------------------------------------------------------- 基础设施
def post_stream(url: str, body: dict, timeout: float = 1800.0):
    """SSE 请求:返回 (ttft_s, total_s, text, usage, finish)。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    text: list[str] = []
    usage = {}
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                text.append(piece)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            if d.get("usage"):
                usage = d["usage"]
    total = time.time() - t0
    return (ttft if ttft is not None else total), total, "".join(text), usage, finish


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def agent_turns(port: int):
    return get_json(f"http://127.0.0.1:{port}/debug/agent").get("turns", [])


def prompt_tokens_of(req: dict) -> int:
    """粗估:优先用 usage(服务端真账),这里只给 dump 侧排序用的近似。"""
    n = 0
    for m in req.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict):
                    n += len(str(p.get("text") or ""))
    n += len(json.dumps(req.get("tools") or [], ensure_ascii=False))
    return n // 2  # chars → tokens 量级估计(仅用于挑选,不作报告数字)


# ---------------------------------------------------------------- 子命令
def cmd_pick(args):
    files = sorted(
        (os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".json")),
        key=lambda p: os.path.getmtime(p),
    )
    if not files:
        print(f"目录 {args.dir} 里没有请求 dump", file=sys.stderr)
        return 1
    rows = []
    for p in files:
        try:
            req = json.load(open(p))
        except json.JSONDecodeError:
            continue
        rows.append((prompt_tokens_of(req), len(json.dumps(req)), p,
                     len(req.get("messages") or []), len(req.get("tools") or [])))
    rows.sort()
    print(f"{'≈tok':>8} {'bytes':>10} {'msgs':>5} {'tools':>6}  file")
    for tok, size, p, msgs, tools in rows[-args.top:]:
        print(f"{tok:>8} {size:>10} {msgs:>5} {tools:>6}  {p}")
    best = rows[-1][2]
    print(f"\n最大上下文请求 = {best}")
    return 0


def cmd_report(args):
    turns = agent_turns(args.port)
    if not turns:
        print("没有记录到 turn", file=sys.stderr)
        return 1
    print(f"{'seq':>4} {'prompt':>7} {'cached':>7} {'hit%':>5} {'out':>5} {'think':>5} "
          f"{'prefill_ms':>10} {'pf_tps':>7} {'ttft_ms':>8} {'dec_tps':>8} {'AL':>5} finish")
    for t in turns[-args.tail:]:
        u = t.get("usage") or {}
        o = t.get("outcome") or {}
        pt = u.get("prompt_tokens") or 0
        ct = u.get("cached_tokens") or 0
        hit = (100.0 * ct / pt) if pt else 0.0
        print(f"{t.get('seq'):>4} {pt:>7} {ct:>7} {hit:>5.1f} "
              f"{u.get('completion_tokens') or 0:>5} {u.get('reasoning_tokens') or 0:>5} "
              f"{o.get('prefill_ms') or 0:>10.0f} {o.get('prefill_tps') or 0:>7.1f} "
              f"{o.get('ttft_ms') or 0:>8.0f} {o.get('decode_tps') or 0:>8.1f} "
              f"{o.get('spec_al') or 0:>5.2f} {o.get('finish_reason') or o.get('error') or '-'}")
    return 0


def cmd_replay(args):
    req = json.load(open(args.req))
    msgs = list(req.get("messages") or [])
    if args.nonce:
        # 冷前缀:在第一轮 system 前插一条唯一标记 ⇒ 内容寻址缓存全 miss
        tag = f"[perf-nonce {time.time_ns()}]"
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = dict(msgs[0])
            msgs[0]["content"] = tag + "\n" + (msgs[0].get("content") or "")
        else:
            msgs.insert(0, {"role": "system", "content": tag})
    if args.drop_newest_image:
        # **归因用 ablation**(不是构造 prompt):把历史里**最后一张图**摘掉,
        # 用于分离"图片 prefill 成本"与"���本 prefill 成本"。其余逐字不动。
        for m in reversed(msgs):
            c = m.get("content")
            if isinstance(c, list):
                idx = [i for i, part in enumerate(c)
                       if isinstance(part, dict) and part.get("type") in ("image_url", "image")]
                if idx:
                    del c[idx[-1]]
                    break
    if args.append:
        msgs.append({"role": "user", "content": args.append})
    body = dict(req)
    body["messages"] = msgs
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    orig_max = req.get("max_tokens") or req.get("max_completion_tokens") or 512
    body["max_tokens"] = args.max_tokens or min(int(orig_max), 2048)
    if args.temperature is not None:
        body["temperature"] = args.temperature
    # **每次回放必须用独立会话**(2026-10-31 实测教训):沿用 implicit 会话会把
    # 回放的链拼到真机会话后面 ⇒ 污染生产会话的 KV 链,下一次真实请求直接全量重算。
    # 显式 session 键 = 服务端 `req.session` 的入口(见 openai.rs 的会话身份段)。
    body["session"] = args.session or f"perf-replay-{time.time_ns()}"

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    n_before = len(agent_turns(args.port))
    ttft, total, text, usage, finish = post_stream(url, body, timeout=args.timeout)
    time.sleep(0.3)
    turns = agent_turns(args.port)
    new = turns[n_before:] or turns[-1:]
    t_last = new[-1] if new else {}
    o = t_last.get("outcome") or {}
    # usage 以**服务端真账**为准(流式最后一帧的 usage 视客户端实现可能缺席)
    u_srv = t_last.get("usage") or {}
    if u_srv:
        usage = u_srv
    out_tok = usage.get("completion_tokens") or 0
    print(f"== {args.tag or 'replay'} ==")
    print(f"  wall={total:6.2f}s  ttft={ttft:6.2f}s  out_tok={out_tok}"
          f"  finish={o.get('finish_reason') or finish}")
    print(f"  prompt_tok={(usage.get('prompt_tokens') or 0)}"
          f"  cached={(usage.get('cached_tokens') or 0)}")
    if o:
        print(f"  prefill={o.get('prefill_ms')} ms ({o.get('prefill_tps')} tok/s)  "
              f"decode={o.get('decode_tps')} tok/s  AL={o.get('spec_al')}")
    if args.show_text:
        print("  ---")
        print("  " + text[:400].replace("\n", "\n  "))
    # 追加一行机器可读(便于多次采样后汇总)
    with open(args.csv, "a") as f:
        f.write(",".join(str(x) for x in [
            args.tag or "replay", round(total, 2), round(ttft, 2),
            usage.get("prompt_tokens") or 0, usage.get("cached_tokens") or 0,
            out_tok,
            o.get("prefill_ms") or 0, round(o.get("prefill_tps") or 0, 1),
            round(o.get("decode_tps") or 0, 1), round(o.get("spec_al") or 0, 2),
            finish,
        ]) + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pick", help="挑出 dump 里上下文最大的请求")
    p.add_argument("--dir", default="/tmp/owl-req-8135")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(fn=cmd_pick)

    p = sub.add_parser("report", help="把 /debug/agent 的每轮事实铺成表")
    p.add_argument("--port", type=int, default=8135)
    p.add_argument("--tail", type=int, default=40)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("replay", help="回放真实请求(可追加一轮)")
    p.add_argument("--req", required=True)
    p.add_argument("--port", type=int, default=8135)
    p.add_argument("--append", default="", help="追加一条 user 消息(长上下文追加单轮)")
    p.add_argument("--nonce", action="store_true", help="首 system 前插随机标记 ⇒ 冷前缀")
    p.add_argument("--drop-newest-image", action="store_true",
                   help="归因 ablation:摘掉历史里最后一张图(其余逐字不动)")
    p.add_argument("--max-tokens", type=int, default=0,
                   help="0 = 沿用原始请求的 max_tokens(缺省 512);截断会制造假的工具块错误")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--session", default="", help="回放用的独立会话键(缺省自动生成)")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--tag", default="")
    p.add_argument("--csv", default="/tmp/owl-perf.csv")
    p.add_argument("--show-text", action="store_true")
    p.set_defaults(fn=cmd_replay)

    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
