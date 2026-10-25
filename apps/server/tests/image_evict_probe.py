#!/usr/bin/env python3
"""LAN 联调探针:图文历史 + 删一张旧图,读 usage.cached_tokens 看引擎是否吸收删除。

关键:每轮都发**全量历史**(agent 形态);删图 = 把历史里那条 content part 拿掉,
其余文字逐字不变。
  python3 apps/server/tests/image_evict_probe.py \\
      --base http://<owl-host>:8135 --session <任意会话名>

思考默认**开**（0.8B 的答案落在 `reasoning_content`，`content` 可能为空 —— 这是
OpenAI 兼容面正常行为；缓存机制与思考无关）。诊断想看 content 时加 `--no-think`。

判读：
  - turn2 的 `cached` = 前缀页 + **保留的后缀页**(本机实测 160/193)；
  - 服务端日志出现 `[img-evict] 前缀无感删除:删 [a,b)=... 保留后缀 N 页`；
  - 去掉 `owl_evict` 再跑一遍（对照臂）= 删点不落页边界 ⇒ cached 只到删点
    之前（实测 0）。

前置：档位必须开 `[pool] image_evict = true` + `prefix_cache = true` +
分页 KV（f16 或 fp8-e4m3；FP8 需 `image_evict_delta_limit` 1..=4）；
图像建议 ≥256×256（网格过小会撞 FI ragged 的既有面）。
"""
import argparse, base64, json, struct, urllib.request, zlib

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


def chat(base: str, session: str, messages, max_tokens=48, think=True):
    body = json.dumps({
        "model": MODEL, "session": session, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0,
        # **思考默认开**（0.8B 默认思考；答案在 `reasoning_content`，content 可能为空）
        "chat_template_kwargs": {} if think else {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def show(tag, resp):
    u = resp.get("usage", {})
    det = u.get("prompt_tokens_details") or {}
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    txt = msg.get("content") or msg.get("reasoning_content", "")
    print(f"[{tag}] prompt={u.get('prompt_tokens')} "
          f"cached={det.get('cached_tokens')} completion={u.get('completion_tokens')} "
          f"text={txt[:70]!r}")
    return msg.get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8135")
    ap.add_argument("--session", default="lan-demo")
    ap.add_argument("--no-think", action="store_true", help="诊断臂:关思考(答案进 content)")
    a = ap.parse_args()
    think = not a.no_think

    img_a = png(256, 256, (200, 40, 40), stripe=64)   # 红(下一轮删掉)
    img_b = png(256, 256, (40, 80, 200), stripe=64)   # 蓝(保留)

    show("warmup", chat(a.base, a.session, [{"role": "user", "content": "你好"}], 8, think))

    # 第 1 轮:一张 user 消息里两张图(都打可删标记),图之间/之后有文字
    head, mid, tail = "这是历史开头。", "中间这段文字要留着。", "历史结尾这段也要留着。"
    content1 = [
        {"type": "text", "text": head},
        {"type": "image_url", "image_url": {"url": img_a}, "owl_evict": True},
        {"type": "text", "text": mid},
        {"type": "image_url", "image_url": {"url": img_b}, "owl_evict": True},
        {"type": "text", "text": tail},
    ]
    r1 = chat(a.base, a.session, [{"role": "user", "content": content1}], 40, think)
    reply1 = show("turn1(两图)", r1)

    # 第 2 轮:**全量历史**重发,只把红图那条 part 拿掉(文字逐字不变)
    content2 = [
        {"type": "text", "text": head},
        {"type": "text", "text": mid},
        {"type": "image_url", "image_url": {"url": img_b}, "owl_evict": True},
        {"type": "text", "text": tail},
    ]
    msgs2 = [
        {"role": "user", "content": content2},
        {"role": "assistant", "content": reply1},
        {"role": "user", "content": "用一句话描述保留的那张图。"},
    ]
    r2 = chat(a.base, a.session, msgs2, 40, think)
    show("turn2(删红图)", r2)
    print("\n提示:turn2 的 cached>0 且引擎日志出现 `img-evict 前缀无感删除` ⇒ 删除被吸收。"
          "\n     (对比臂:不带 owl_evict ⇒ cached 只到删点之前,删点之后重算。)")


if __name__ == "__main__":
    main()
