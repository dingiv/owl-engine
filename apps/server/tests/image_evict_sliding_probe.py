#!/usr/bin/env python3
"""真实前端形态探针：图文混合 + **滑动图片窗口**（只留 N 张）+ 每轮删最老那张。

前端行为（本探针复刻）：
  1. 每轮新增一条 user 消息：`[文本, 新截图, 文本]`；
  2. 会话里图片数超过 N 时，把**最老那张图的 content part 删掉**（文字全部保留）；
  3. 每轮都发**全量历史**（OpenAI 兼容面），`session` 固定；
  4. 可删的图都带 `owl_evict: true` —— 引擎把它们补页对齐 pad 并记为删除候选。

判读：
  - `cached` = 本轮真正复用的 token（前缀页 + 删除后**保留的后缀页**）；
  - `prefill` = prompt - cached = 本轮实际重算量（应≈新增尾巴，而不是删点之后全部）；
  - 服务端日志每轮一行 `[img-evict] 前缀无感删除: … 保留后缀 N 页 …`；
  - `--no-evict` 跑同一套对话做**对照臂**（不打标记 ⇒ 删点不落页边界 ⇒ cached 塌回删点前）。

用法：
  python3 image_evict_sliding_probe.py --base http://127.0.0.1:8135 --turns 6 --keep 3
  python3 image_evict_sliding_probe.py --base http://127.0.0.1:8135 --no-evict
"""
import argparse, base64, json, struct, sys, time, urllib.request, zlib

MODEL = "qwen3.5-0.8b-vlm"


def png(w: int, h: int, rgb, stripe=None) -> str:
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            r, g, b = rgb
            if stripe:
                k = 1.0 + 0.3 * ((y // stripe) % 2)
                r, g, b = int(r * k), int(g * k), int(b * k)
            row += bytes((min(r, 255), min(g, 255), min(b, 255)))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    blob = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(blob).decode()


COLORS = [(200, 40, 40), (40, 80, 200), (40, 180, 80), (220, 180, 40),
          (160, 60, 200), (40, 190, 190), (240, 120, 40), (120, 120, 240)]


def chat(base: str, session: str, messages, max_tokens: int):
    body = json.dumps({
        "model": MODEL, "session": session, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0,
        "chat_template_kwargs": {},  # 思考默认开（答案可能在 reasoning_content）
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    return resp, (time.time() - t0) * 1000.0


def images_in(msgs):
    """(消息下标, content 下标) 按出现序。"""
    out = []
    for i, m in enumerate(msgs):
        c = m.get("content")
        if isinstance(c, list):
            for j, part in enumerate(c):
                if part.get("type") == "image_url":
                    out.append((i, j))
    return out


def drop_oldest_image(msgs):
    loc = images_in(msgs)
    if not loc:
        return None
    i, j = loc[0]
    msgs[i]["content"].pop(j)
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8135")
    ap.add_argument("--session", default="slide-demo")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--keep", type=int, default=3, help="会话里保留几张图（滑动窗口）")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--no-evict", action="store_true", help="对照臂：不打 owl_evict")
    a = ap.parse_args()

    imgs = [png(a.size, a.size, COLORS[k % len(COLORS)], stripe=64) for k in range(a.turns)]
    msgs = []
    rows = []
    t_all0 = time.time()
    for k in range(a.turns):
        # 前端：加入新图前先把最老的踢掉，保证会话里最多 keep 张
        if len(images_in(msgs)) >= a.keep:
            drop_oldest_image(msgs)
        part = {"type": "image_url", "image_url": {"url": imgs[k]}}
        if not a.no_evict:
            part["owl_evict"] = True
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": f"第 {k + 1} 步：这是当前屏幕截图。"},
            part,
            {"type": "text", "text": "请用一句话描述你看到了什么。"},
        ]})
        resp, ms = chat(a.base, a.session, msgs, a.max_tokens)
        u = resp.get("usage", {})
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        prompt = u.get("prompt_tokens", 0)
        msg = (resp.get("choices") or [{}])[0].get("message", {})
        text = (msg.get("content") or msg.get("reasoning_content") or "").replace("\n", " ")
        n_img = len(images_in(msgs))
        rows.append((k + 1, n_img, prompt, cached, prompt - cached, ms, text))
        print(f"turn {k + 1}: images={n_img} prompt={prompt:4d} cached={cached:4d} "
              f"prefill={prompt - cached:4d} ({(prompt - cached) / max(prompt, 1):5.1%}) "
              f"{ms:7.0f}ms  text={text[:48]!r}")
        # 客户端把模型的回答放回历史（真实 agent 形态）
        msgs.append({"role": "assistant", "content": msg.get("content") or ""})

    total = (time.time() - t_all0) * 1000.0
    print(f"\n合计 {a.turns} 轮 / {total / 1000:.1f}s")
    if a.no_evict:
        print("对照臂（无 owl_evict）：删点不落页边界 ⇒ 删点之后只能重算。")
    else:
        print("期望：第 1 张图被删之后（turn > keep），cached 每轮 > 0 且 prefill 只占一小截。")


if __name__ == "__main__":
    sys.exit(main())
