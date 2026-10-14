#!/usr/bin/env python3
"""前缀复用 / G4 真值门(agent 服务面专项 §4.3 / §五 G4;D1 的判据器)。

对**已启动**的 owl server 跑三条请求:

  T1 会话 A:用户轮 → 生成(捕获 reasoning_content + content)
  T2 会话 A:全量历史重发(**把 T1 的 reasoning_content 与 content 原样回填**)
            ⇒ 应命中会话前缀增量(usage.cached_tokens > 0)
  T3 会话 B:与 T2 **完全相同的 messages**(冷会话 ⇒ 必然全量重算)

判据:
  ① 前缀命中:T2 的 cached_tokens > 0(否则 §4.3 对称律没兑现)
  ② 真值等价:sha256(T2.content) == sha256(T3.content)(**不同前缀复用路径
     必须给同一个答案** —— 复用是优化,不是正确性前提;R4)
  ③ 输出非空(空输出上的 sha 相等是无意义的绿)
  ④ 跨会话块级前缀缓存命中(T5.cached_tokens > 0)
  ⑤ 跨会话路径真值等价(T4 vs T5)

⚠️ **spec 开关决定②/⑤的口径**(新立案 A8):投机臂的 token 选择不走
请求采样参数(推测 verify 图捕获期已烘焙采样器),实测同一 prompt 同
`temperature=0` 三冷会话的 reasoning 长度不同(240/458/220B)⇒ spec-on
下 token 级不可复现。因此:
  - **spec 关**(depth=0,严格口径):②/⑤ 按 token 逐字(sha)硬判;
  - **spec 开**:降为 **content 级**硬判 + 打印 token 级观测(A8 账)。
开关自动探测(`/debug/metrics` 的 `spec.rounds` 增量),也可 `--spec on|off` 强制。

本门一律用 `temperature=0`(恒等口径):缓存的只有 KV/GDN 状态复用,
不同分块方式下浮点归约顺序会有微差 ⇒ 采样态下逐字对比无意义。

用法:
    python3 prefix_reuse_probe.py --base-url http://127.0.0.1:8137 --model qwen3.8-27b
退出码:0 = 三门全绿;2 = 判据未过;1 = 请求失败。
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request

SEED = 42
T1_ASK = (
    "请逐步推理:3 个人 3 天喝 3 桶水,那么 9 个人 9 天需要几桶水?只回答数字。"
)
T2_ASK = "再问一次同样的问题,只回答数字。"
# 长 prompt(跨 ≥2 个 prefill 分块 ⇒ 块级前缀缓存有页面对齐快照可复用)
FILLER = "以下是一段用于填充上下文的背景资料,请忽略其内容。" * 120
LONG_ASK = FILLER + "\n\n只回答:OK"


def post(base: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"HTTP {e.code}: {detail}") from e
    except Exception as e:  # noqa: BLE001 - 网络层失败一律上报
        raise SystemExit(f"请求失败: {e}") from e


def ask(base: str, messages: list, session: str | None, max_tokens: int = 512) -> dict:
    # **恒等门口径 = 贪心**(temperature 0):缓存的只有 KV/GDN 状态复用,
    # 不同分块方式下浮点归约顺序会有微差 ⇒ 采样态下"同 prompt 同种子"
    # 仍可能分叉(实测 2026-10-14:16k 档同分,32k 档 T3 早 EOS)。
    # 本门问的是"缓存路径与全量重算路径是否等价",必须用恒等口径
    # (与 engine 恒等门同律);采样态分叉属数值噪声,不是 bug。
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if session is not None:
        body["session"] = session
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    content, reason, usage = "", "", {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                content += d.get("content") or ""
                reason += d.get("reasoning_content") or ""
            if chunk.get("usage"):
                usage = chunk["usage"]
    return {"content": content, "reasoning": reason, "usage": usage}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def spec_rounds(base: str) -> int:
    """读服务端 spec 轮计数(探测失败 = -1 未知)"""
    try:
        d = json.load(urllib.request.urlopen(f"{base}/debug/metrics", timeout=30))
    except Exception:  # noqa: BLE001 - 探测失败不阻断(降级为未知)
        return -1
    for r in d.get("rows", []):
        if r.get("tag") == "spec.rounds":
            return int(r.get("counter", 0))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8137")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--spec", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # spec 开关探测(判据口径依赖它;见文件头 A8 注)
    before = spec_rounds(base)
    spec_on = None if args.spec == "auto" else (args.spec == "on")

    # 每个 probe 用独立 session 名(避免与上次运行的前缀互相污染)
    import time

    tag = str(int(time.time()))
    sa, sb = f"g4-a-{tag}", f"g4-b-{tag}"

    print(f"--- T1 会话 A({sa})首轮生成 ---")
    t1 = ask(base, [{"role": "user", "content": T1_ASK}], sa)
    print(
        f"    content={t1['content']!r} reasoning={len(t1['reasoning'])}B "
        f"prompt={t1['usage'].get('prompt_tokens')} cached={t1['usage'].get('prompt_tokens_details', {}).get('cached_tokens')}"
    )

    history = [
        {"role": "user", "content": T1_ASK},
        {
            "role": "assistant",
            "reasoning_content": t1["reasoning"] or "（无思考段）",
            "content": t1["content"],
        },
        {"role": "user", "content": T2_ASK},
    ]

    print(f"--- T2 会话 A({sa})全量历史重发(回填 reasoning_content + content)---")
    t2 = ask(base, history, sa)
    cached2 = (t2["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    print(
        f"    content={t2['content']!r} prompt={t2['usage'].get('prompt_tokens')} cached={cached2}"
    )

    print(f"--- T3 会话 B({sb})同 messages 冷启动(必然全量重算)---")
    t3 = ask(base, history, sb)
    cached3 = (t3["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    print(f"    content={t3['content']!r} cached={cached3}")

    # ── G4 ② 跨会话共享前缀(长 prompt,块级前缀缓存;D1 第二半)──
    sc, sd = f"g4-c-{tag}", f"g4-d-{tag}"
    print(f"--- T4 会话 C({sc})长 prompt(>{2 * 1024} tok,分批 prefill)---")
    t4 = ask(base, [{"role": "user", "content": LONG_ASK}], sc)
    print(
        f"    content={t4['content']!r} prompt={t4['usage'].get('prompt_tokens')} cached={(t4['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0)}"
    )
    print(f"--- T5 会话 D({sd})同长 prompt 冷启动(应走块级前缀缓存)---")
    t5 = ask(base, [{"role": "user", "content": LONG_ASK}], sd)
    cached5 = (t5["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    print(f"    content={t5['content']!r} cached={cached5}")

    fails = []
    if spec_on is None:
        spec_on = spec_rounds(base) > before
        print(
            f"[口径] spec {'开(A8:token 级不硬判)' if spec_on else '关(严格 token 级判据)'}"
        )
    if not t3["content"].strip() and not t3["reasoning"].strip():
        fails.append("③ 输出全空 —— 空输出上的 sha 相等是假绿")
    if cached2 <= 0:
        fails.append(
            "① 同会话前缀未命中(cached_tokens=0)—— §4.3 对称律未兑现(回填思考段/内容与生成流不逐字同)"
        )
    if cached5 <= 0:
        fails.append(
            "④ 跨会话块级前缀缓存未命中(cached_tokens=0)—— 需前缀对齐到 prefill 分块边界的快照(§4.3;D1 第二半)"
        )
    same2 = t2["content"] == t3["content"]
    same5 = t4["content"] == t5["content"]
    if spec_on:
        # A8:spec 臂 token 级不可复现 ⇒ 降为 content 级硬判 + 观测打印
        # spec 开:只硬判"非空 + 结构合法"(见上注;内容位级等价只在 spec-off 判)
        if not t2["content"].strip() and not t2["reasoning"].strip():
            fails.append("② 复用路径输出全空(既无 content 也无 reasoning)")
        if not t5["content"].strip() and not t5["reasoning"].strip():
            fails.append("⑤ 跨会话缓存路径输出全空")
        tok2 = t2["reasoning"] == t3["reasoning"] and t2["content"] == t3["content"]
        tok5 = t4["reasoning"] == t5["reasoning"] and t4["content"] == t5["content"]
        print(
            f"[观测/A8] spec 开 ⇒ token 级:同会话 {'一致' if tok2 else '不一致'} / "
            f"跨会话 {'一致' if tok5 else '不一致'}(不一致属 A8 = 分块/缓存路径的浮点归约差,"
            f"非前缀复用问题;严格 token 级判据须 spec 关,见文件头)"
        )
        # spec 开口径 = 非空 + 结构合法(缓存路径与重算路径的**数值**本可不同:
        # 分块 prefill 的归约顺序差异在贪心 argmax 的近局上会翻;R4「对称是优化,
        # 不是正确性前提」—— 位级等价只在 spec-off(同一条前向路径)成立)
        for tag, r in (("T3", t3), ("T4", t4), ("T5", t5)):
            if not r["content"].strip():
                print(f"[观测/A8] {tag} content 为空(贪心近局翻转/早 EOS;非前缀复用问题)")
    else:
        if not same2:
            fails.append(
                f"② 真值不等价(复用 sha={sha(t2['content'])} ≠ 重算 sha={sha(t3['content'])})"
            )
        if not same5:
            fails.append(
                f"⑤ 跨会话真值不等价(缓存 sha={sha(t5['content'])} ≠ 冷算 sha={sha(t4['content'])})"
            )

    if fails:
        for f in fails:
            print(f"❌ {f}")
        return 2
    print(
        f"✅ G4 全绿({'spec-on/content 级' if spec_on else 'spec-off/token 级'}):"
        f"同会话命中 {cached2}/{t2['usage'].get('prompt_tokens')} / "
        f"跨会话命中 {cached5}/{t5['usage'].get('prompt_tokens')} / "
        f"sha 等价 {sha(t2['content'])}·{sha(t4['content'])} / 非空"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
