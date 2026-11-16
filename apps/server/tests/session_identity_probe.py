#!/usr/bin/env python3
"""**会话身份 = 内容身份**真机门（2026-11-16 原始算法 §32；store 会话管理）。

对**已启动**的 owl server 跑两条互不相干的对话（都不带 `session` 键）+ 一条
显式键对话，判据看**会话表**（`/debug/vram → sessions`）而不只是
`cached_tokens` —— 后者含跨会话的内容树命中，单看它判不出身份。

判据（11 条）：
  ① A1 新建 1 条会话；
  ② B1（内容不同）**另建**一条（模板前缀相同也不能串到 A）；
  ③④ A2/A3 认出 A 自己的会话（不新建）；
  ⑤ B2 认出 B 自己的会话（不新建）；
  ⑥⑦ 显式 `session` 键：C1 新建、C2 粘性（不新建）；
  ⑧⑨⑩ A2/A3/B2 的 `cached_tokens` 不倒退且随轮增长；
  ⑪ C2 增量命中。

⚠️ 两条**真机定谳**过的判据（CPU mock 抓不到，改动前必读
`crates/store/src/session.rs::match_prefix` 的注释）：
  - 只按"≥ 一页"判 ⇒ 所有请求都命中 MRU 会话（chat 模板前缀本身就超一页）；
  - 要求覆盖**整账** ⇒ 每轮都认不出会话（账里含模型生成段，客户端回填的是
    按文本重渲染的 prompt，尾部必然不同）⇒ 只要求覆盖上一轮 **prompt 段**。

用法：
    python3 apps/server/tests/session_identity_probe.py \
        --base-url http://127.0.0.1:8136 --model qwen3.8-27b
    # 每次运行都用随机 nonce 造新对话 ⇒ **不要求**引擎是刚启动的；
    # `--require-empty` 可以在"必须是干净引擎"的场合把这条收紧成硬门。

退出码：0 = 全绿；2 = 判据未过；1 = 请求/连接失败。
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# 判据里"会话表只多出 N 条"依赖内容唯一 ⇒ 每次运行造新的对话内容
NONCE = f"{time.time_ns() % 10**9:09d}"


class Fail(Exception):
    """请求层失败（退出码 1）"""


def http_json(method: str, url: str, body: dict | None = None, timeout: int = 900) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method=method, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        # 4xx/5xx 是**判据面**的事实（例如排队拒绝），照实上报
        raise Fail(f"HTTP {e.code} @ {url}: {detail}") from e
    except Exception as e:  # noqa: BLE001 - 网络层失败一律上报
        raise Fail(f"请求失败 @ {url}: {e}") from e


def sessions(base: str) -> dict:
    v = http_json("GET", f"{base}/debug/vram", timeout=60)
    return {s["id"]: s for s in (v.get("sessions") or [])}


def chat(base: str, model: str, messages: list, max_tokens: int, session: str | None):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if session is not None:
        body["session"] = session
    out = http_json("POST", f"{base}/v1/chat/completions", body)
    u = out.get("usage") or {}
    d = u.get("prompt_tokens_details") or {}
    return {
        "prompt": u.get("prompt_tokens"),
        "cached": d.get("cached_tokens", u.get("cached_tokens")),
        "out": u.get("completion_tokens"),
        "text": (out["choices"][0]["message"].get("content") or ""),
    }


def turn(base, model, history: list, user: str, max_tokens: int):
    """追加一轮（客户端视角：全量历史重发 = OpenAI 协议常态）"""
    history.append({"role": "user", "content": user})
    r = chat(base, model, history, max_tokens, None)
    history.append({"role": "assistant", "content": r["text"]})
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="会话身份 = 内容身份 真机门")
    ap.add_argument("--base-url", default="http://127.0.0.1:8137")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--max-tokens", type=int, default=24, help="每轮生成长度（门只看命中，不看质量）")
    ap.add_argument(
        "--require-empty",
        action="store_true",
        help="要求引擎刚启动（会话表必须为空）；缺省只告警并按相对增量判",
    )
    args = ap.parse_args()
    base, model = args.base_url.rstrip("/"), args.model

    try:
        st0 = sessions(base)
        if st0 and args.require_empty:
            print(f"✗ 会话表非空（{len(st0)} 条）：本档要求干净引擎（--require-empty）")
            return 2
        if st0:
            print(f"⚠️ 会话表已有 {len(st0)} 条（按相对增量判；本次对话用 nonce={NONCE} 保证内容全新）")
        print(f"启动后会话表：{len(st0)} 条")

        # 两条互不相干的对话（系统提示不同 ⇒ 只有模板前缀相同）
        a = [{"role": "system", "content": f"你是 Aquila-观测站的值班员（批次 {NONCE}），只回答一句话。"}]
        b = [{"role": "system", "content": f"你是 Borealis-船坞的调度员（批次 {NONCE}），只回答一句话。"}]
        c = [{"role": "user", "content": f"记下：C 线（{NONCE}）第 1 号，明码 9。"}]
        ckey = f"probe-c-{NONCE}"

        log: dict[str, tuple[dict, int]] = {}

        def step(tag: str, run) -> dict:
            r = run()
            n = len(sessions(base))
            log[tag] = (r, n)
            print(
                f"  {tag}: prompt={r['prompt']:>5} cached={r['cached']:>5} "
                f"out={r['out']:>3} | 会话 {n} 条"
            )
            return r

        print("[A] 无键对话 A")
        a1 = step("a1", lambda: turn(base, model, a, f"A 线（{NONCE}）第 1 号观测：云量 3 成。", args.max_tokens))
        print("[B] 无键对话 B（与 A 内容不同）")
        b1 = step("b1", lambda: turn(base, model, b, f"B 线（{NONCE}）第 1 号调度：泊位 7 号空闲。", args.max_tokens))
        print("[A] A 续写")
        a2 = step("a2", lambda: turn(base, model, a, "A 线第 2 号观测：风速 4 级。", args.max_tokens))
        a3 = step("a3", lambda: turn(base, model, a, "A 线第 3 号观测：能见度 10 公里。", args.max_tokens))
        print("[B] B 续写")
        b2 = step("b2", lambda: turn(base, model, b, "B 线第 2 号调度：3 号泊位开始装载。", args.max_tokens))

        print("[C] 显式 session 键（粘性）")
        c1 = step("c1", lambda: chat(base, model, c, args.max_tokens, ckey))
        c.append({"role": "assistant", "content": c1["text"]})
        c.append({"role": "user", "content": "C 线第 2 号，明码 10。"})
        c2 = step("c2", lambda: chat(base, model, c, args.max_tokens, ckey))
    except Fail as e:
        print(f"✗ {e}")
        return 1

    base_n = len(st0)
    ok = True

    def want(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("  ✓ " if cond else "  ✗ ") + msg)
        ok = ok and cond

    n = {k: v[1] for k, v in log.items()}
    print("\n判据：")
    want(n["a1"] == base_n + 1, f"A1 新建 1 条会话（{base_n} → {n['a1']}）")
    want(n["b1"] == base_n + 2, f"B1 是**另一条**会话（{n['a1']} → {n['b1']}，模板前缀没串）")
    want(n["a2"] == base_n + 2, f"A2 认出 A 自己的会话（不新建：{n['a2']}）")
    want(n["a3"] == base_n + 2, f"A3 仍不新建（{n['a3']}）")
    want(n["b2"] == base_n + 2, f"B2 认出 B 自己的会话（不新建：{n['b2']}）")
    want(n["c1"] == base_n + 3, f"C1（显式键）新建 1 条（{base_n + 2} → {n['c1']}）")
    want(n["c2"] == base_n + 3, f"C2 粘性不新建（{n['c2']}）")
    want(a2["cached"] >= a1["cached"], f"A2 cached 不倒退（{a1['cached']} → {a2['cached']}）")
    want(a3["cached"] > a2["cached"], f"A3 缓存随轮增长（{a2['cached']} → {a3['cached']}）")
    want(b2["cached"] >= b1["cached"], f"B2 cached 增长（{b1['cached']} → {b2['cached']}）")
    want(c2["cached"] > c1["cached"], f"C2（显式键）增量命中（{c1['cached']} → {c2['cached']}）")

    print("\n会话表（history_len = 会话账 `tokens.len()` 唯一真源；L1 前缀账已归还）：")
    for sid, s in sorted(sessions(base).items()):
        print(f"  #{sid}: history_len={s['history_len']} tail_snap={'有' if s['tail_snap'] else '无'}")

    print("\n" + ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
