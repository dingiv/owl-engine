#!/usr/bin/env python3
"""10K agent-style prefix reuse probe (multi-turn / cross-session / system branch).

Why this probe exists
---------------------
The agent workload is not "one huge 200K cold prefill"; it is a long-lived
conversation whose *new* request is typically ~10K tokens, with a short user or
tool delta appended each turn.  The useful signal is therefore
`usage.prompt_tokens_details.cached_tokens` on the **next** turn, not the cold
prefill speed.

This probe fixes a ~10K background/system prefix and drives:

  T1  session A, first turn (cold; unique salt by default)
  T2  session A, append assistant answer + short user turn
  T3  session A, one more turn
  T4  session B, exactly T1 messages (cross-session full-prefix reuse)
  T5  session C, exactly T2 messages (cross-session incremental reuse)
  T6  no `session` key (pi does not send one; implicit_session must handle it)
  T7  session D, same system prefix but a *different* user task (branch point)

T2/T4/T5/T6 require a boundary GDN snapshot to exist at the last matched block.
If `snap_max=0`, the engine structurally cannot reuse the prefix and all of
these are 0.  If `snap_max=1` and `snap_host_mb=0`, a long decode can evict the
durable prefill snapshot; same-session T2 may still hit via the newest decode
snapshot, while cross-session T4 falls to 0.

T7 measures the residual "snapshot anchor cadence" loss: with SNAP_EVERY=4
prefill chunks, a fixed system prefix that does not end on an anchor boundary
reuses only the last anchor (e.g. 8192 / ~10K = 82%).

Usage:
    python3 apps/server/tests/agent_prefix_10k_probe.py \
        --base-url http://127.0.0.1:8137 --model qwen3.8-27b

Exit code: 0 = all hard gates pass; 2 = a hard gate failed; 1 = request error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request

WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
).split()


def make_context(n_lines: int, salt: str) -> str:
    """Deterministic ~10K-token filler (for 300 lines, both tokenizers)."""
    rng = random.Random(20261024)
    lines = [
        f"Agent workspace snapshot[{salt}]. "
        "Treat this as immutable background context."
    ]
    for i in range(n_lines):
        words = " ".join(rng.choice(WORDS) for _ in range(11))
        lines.append(f"note {i:04d}: {words}; ref={rng.randrange(10**8):08d}.")
    lines.append("End of background context.")
    return "\n".join(lines)


def ask(base: str, model: str, messages: list[dict], session: str | None,
        max_tokens: int, timeout: int = 600) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if session is not None:
        body["session"] = session
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    content = reasoning = ""
    usage: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
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
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    content += delta.get("content") or ""
                    reasoning += delta.get("reasoning_content") or ""
                if chunk.get("usage"):
                    usage = chunk["usage"]
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"HTTP {e.code}: {detail}") from e
    except Exception as e:  # noqa: BLE001 - network failures must be loud
        raise SystemExit(f"request failed: {e}") from e
    return {
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "wall_s": round(time.time() - t0, 3),
    }


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def report(tag: str, result: dict) -> tuple[int, int]:
    usage = result["usage"]
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    ratio = 100.0 * cached / max(prompt, 1)
    digest = sha(result["reasoning"] + "\x00" + result["content"])
    print(
        f"{tag:14s} prompt={prompt:6d} cached={cached:6d} ({ratio:5.1f}%) "
        f"new={prompt - cached:6d} out={completion:4d} "
        f"wall={result['wall_s']:7.3f}s sha={digest} "
        f"content={result['content'][:40]!r}"
    )
    return prompt, cached


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8137")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--lines", type=int, default=300,
                    help="filler lines; 300 ≈ 10K tokens for both Qwen tokenizers")
    ap.add_argument("--max-tokens", type=int, default=160,
                    help="long enough to create decode page-boundary snapshots")
    ap.add_argument("--min-hit", type=float, default=0.80,
                    help="hard gate for T2/T4/T5/T6 cached/prompt ratio")
    ap.add_argument("--min-branch-hit", type=float, default=0.0,
                    help="optional hard gate for T7 fixed-system branch ratio")
    ap.add_argument("--salt", default=None,
                    help="context salt; default = timestamp (T1 is genuinely cold)")
    ap.add_argument("--check-equality", action="store_true",
                    help="also hard-gate cold vs cached content equality")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    salt = args.salt if args.salt is not None else str(int(time.time()))
    context = make_context(args.lines, salt)
    system = {"role": "system", "content": context}
    task1 = ("Write the integers from 1 to 80 separated by single spaces. "
             "No explanation.")
    task2 = "Continue. Reply with exactly: TURN2"
    task3 = "Continue. Reply with exactly: TURN3"
    task_alt = "Different task. Reply with exactly: ALT"

    tag = str(int(time.time()))
    sa, sb, sc, sd = (f"agent10k-{x}-{tag}" for x in "abcd")

    print(f"--- context chars={len(context)} lines={args.lines} salt={salt} ---")
    m1 = [system, {"role": "user", "content": task1}]
    t1 = ask(base, args.model, m1, sa, args.max_tokens)
    p1, c1 = report("T1 cold", t1)
    assistant1 = t1["content"].strip() or "TURN1"

    m2 = m1 + [
        {"role": "assistant", "content": assistant1},
        {"role": "user", "content": task2},
    ]
    t2 = ask(base, args.model, m2, sa, args.max_tokens)
    p2, c2 = report("T2 same", t2)
    assistant2 = t2["content"].strip() or "TURN2"

    m3 = m2 + [
        {"role": "assistant", "content": assistant2},
        {"role": "user", "content": task3},
    ]
    t3 = ask(base, args.model, m3, sa, args.max_tokens)
    report("T3 same", t3)

    t4 = ask(base, args.model, m1, sb, args.max_tokens)
    p4, c4 = report("T4 x-sess", t4)

    t5 = ask(base, args.model, m2, sc, args.max_tokens)
    p5, c5 = report("T5 x-sess", t5)

    t6 = ask(base, args.model, m2, None, args.max_tokens)
    p6, c6 = report("T6 no-session", t6)

    m_alt = [system, {"role": "user", "content": task_alt}]
    t7 = ask(base, args.model, m_alt, sd, args.max_tokens)
    p7, c7 = report("T7 sys-branch", t7)

    print("\n--- gate ---")
    failures: list[str] = []
    if p1 == 0:
        failures.append("T1 prompt_tokens=0 (request did not report usage)")
    for name, prompt, cached in (
        ("T2 same-session", p2, c2),
        ("T4 cross-session", p4, c4),
        ("T5 cross-session increment", p5, c5),
        ("T6 no-session", p6, c6),
    ):
        ratio = cached / max(prompt, 1)
        ok = ratio >= args.min_hit
        print(f"{'PASS' if ok else 'FAIL'} {name}: {cached}/{prompt} = {ratio:.1%}")
        if not ok:
            failures.append(f"{name}: {cached}/{prompt} < {args.min_hit:.0%}")
    branch_ratio = c7 / max(p7, 1)
    if args.min_branch_hit > 0:
        branch_ok = branch_ratio >= args.min_branch_hit
        print(
            f"{'PASS' if branch_ok else 'FAIL'} system-branch: "
            f"{c7}/{p7} = {branch_ratio:.1%}"
        )
        if not branch_ok:
            failures.append(
                f"system-branch: {c7}/{p7} < {args.min_branch_hit:.0%}"
            )
    else:
        print(
            f"INFO system-branch: {c7}/{p7} = {branch_ratio:.1%} "
            f"(set --min-branch-hit to gate)"
        )
    equality = (t1["content"] == t4["content"]
                and t1["reasoning"] == t4["reasoning"])
    print(f"INFO cold/cached equality: {'YES' if equality else 'NO'}")
    if args.check_equality and not equality:
        failures.append("T1 cold vs T4 cached output differs")

    if failures:
        print("\nRESULT: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 2
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
