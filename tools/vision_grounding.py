#!/usr/bin/env python3
"""视觉 grounding 对拍探针(2026-11-03 立)。

用途
----
GUI agent 报「模型输出坐标偏 30~40px、点不中 UI 元素」。本工具把「感知精度」
从「链路 bug」里剥出来:

    make  : 生成合成图(纯色方块 + GT),纯 stdlib 写 PNG(无 PIL 依赖)
    probe : 打 OpenAI 兼容端点,要求模型输出 **0~1000 归一化网格**坐标,
            解析后用 GT 算偏差结构(逐轴 bias / 回归斜率 / 命中率)

判据(为什么看斜率而不是只看命中率)
-----------------------------------
模型输出的是 0~1000 归一化网格 ⇒ 归一化**天然吸收** `smart_resize` 的 32 对齐
(1280×720 → 1280×704)。所以:

    预测 = 归一化坐标 → (x/1000*W_sent, y/1000*H_sent) → 再按客户端比例回映屏幕

* 斜率 b ≈ 1、截距 ≈ 0 ⇒ 映射链正确,残差 = 感知地板(方块低于 1 个 merged
  cell 时必然发生,1 cell = 32px @ sent image)。
* 斜率 b ≠ 1 ⇒ 比例/网格口径错(例如模型按像素答、或客户端回映用了错的边长)。
* 截距 ≠ 0 ⇒ 系统性平移(位置/特征错位、pad 插入错位)。

用法
----
    # 1) 造图(1x 臂 = 客户端实际发出的尺寸;2x 臂 = 屏幕原尺寸)
    uv run tools/vision_grounding.py make --out /tmp/vg --size 1280x720 --sq 10,48

    # 2) 探针(owl 臂;带 owl_evict 标记 = 图像无感删除生效)
    uv run tools/vision_grounding.py probe --url http://127.0.0.1:8135/v1/chat/completions \
        --model qwen --fixture /tmp/vg/1280x720_sq10 --repeat 3 --evict

    # 3) 金标臂(vLLM 8133,同 checkpoint)
    uv run tools/vision_grounding.py probe --url http://127.0.0.1:8133/v1/chat/completions \
        --model Qwen3.8-27B-pi-AWQ-INT4 --fixture /tmp/vg/1280x720_sq10 --repeat 3

输出:`<fixture>/probe-<tag>.json` + stdout markdown 表。
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import re
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

# ── 调色板:高饱和、彼此可辨、在白底与深底上都清楚 ──────────────────────────
PALETTE = {
    "red": (220, 32, 44),
    "green": (24, 150, 64),
    "blue": (32, 72, 216),
    "yellow": (232, 200, 24),
    "magenta": (208, 40, 200),
    "cyan": (16, 190, 200),
    "orange": (240, 130, 20),
    "purple": (120, 52, 184),
    "brown": (140, 92, 40),
    "pink": (240, 152, 184),
    "teal": (0, 122, 112),
    "navy": (24, 32, 92),
    "olive": (128, 128, 32),
    "lime": (150, 220, 40),
    "violet": (150, 90, 220),
    "crimson": (170, 20, 70),
}

BG_LIGHT = (246, 246, 246)


# ════════════════════════════════════════════════════════════════════════════
# PNG(纯 stdlib:8bit RGB、filter=0)—— 不引入 PIL,免得为一张方块图建 venv
# ════════════════════════════════════════════════════════════════════════════
def write_png(path: Path, w: int, h: int, pixels: bytearray) -> None:
    if len(pixels) != w * h * 3:
        raise SystemExit(f"write_png: 缓冲长 {len(pixels)} ≠ {w}*{h}*3")
    raw = bytearray()
    stride = w * 3
    for y in range(h):
        raw.append(0)  # filter type 0
        raw += pixels[y * stride : (y + 1) * stride]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(blob)


def fill_rect(pixels: bytearray, w: int, x0: int, y0: int, s: int, rgb: tuple) -> None:
    for y in range(y0, y0 + s):
        base = (y * w + x0) * 3
        pixels[base : base + s * 3] = bytes(rgb) * s


# ════════════════════════════════════════════════════════════════════════════
# make:合成图 + GT
# ════════════════════════════════════════════════════════════════════════════
def cmd_make(a: argparse.Namespace) -> None:
    w, h = (int(v) for v in a.size.lower().split("x"))
    sizes = [int(v) for v in a.sq.split(",") if v.strip()]
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    for sq in sizes:
        colors = list(PALETTE)[: a.n]
        if a.n > len(PALETTE):
            raise SystemExit(f"--n {a.n} 超过调色板 {len(PALETTE)}")
        margin = max(a.margin, sq)
        min_gap = a.min_gap
        pts: list[tuple[int, int]] = []
        for _ in range(20000):
            if len(pts) == a.n:
                break
            cx = rng.randrange(margin, w - margin)
            cy = rng.randrange(margin, h - margin)
            if any(math.hypot(cx - px, cy - py) < min_gap for px, py in pts):
                continue
            pts.append((cx, cy))
        if len(pts) != a.n:
            raise SystemExit(f"放不下 {a.n} 个方块(图 {w}x{h}, min_gap={min_gap})")

        pixels = bytearray(bytes(BG_LIGHT) * (w * h))
        squares = []
        for color, (cx, cy) in zip(colors, pts):
            x0 = cx - sq // 2
            y0 = cy - sq // 2
            fill_rect(pixels, w, x0, y0, sq, PALETTE[color])
            squares.append({"color": color, "x": cx, "y": cy, "size": sq})

        name = f"{w}x{h}_sq{sq}"
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        png = d / "img.png"
        write_png(png, w, h, pixels)
        gt = {
            "image": {"path": str(png), "width": w, "height": h},
            "cell_px": 32,  # 1 merged cell(16px patch × merge2)@ 本图尺寸
            "squares": squares,
        }
        (d / "gt.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
        print(f"[make] {d}  方块={sq}px × {a.n}  文件 {png.stat().st_size / 1024:.1f} KiB")


# ════════════════════════════════════════════════════════════════════════════
# prompt(0~1000 归一化网格口径;不给模型图尺寸,避免它按像素答)
# ════════════════════════════════════════════════════════════════════════════
SYS = (
    "You are a precise visual grounding engine. You report the position of a "
    "target inside an image using a normalized grid: x and y are integers in "
    "[0, 1000], where x=0 is the left edge and x=1000 the right edge, y=0 the "
    "top edge and y=1000 the bottom edge. You never explain, never use tools, "
    "and never refuse: you always answer with one JSON object."
)


def user_prompt(colors: list[str], nonce: str = "") -> str:
    if nonce:
        return f"Run tag: {nonce}.\n" + user_prompt(colors)
    listing = ", ".join(colors)
    return (
        f"The image contains small solid-colour squares on a light background. "
        f"Report the CENTER of the {listing} square(s).\n"
        "Answer with JSON only, no prose and no code fence:\n"
        '{"points":[{"color":"<colour>","x":<0..1000>,"y":<0..1000>}]}\n'
        "Include exactly one entry per colour listed above."
    )


# ════════════════════════════════════════════════════════════════════════════
# probe
# ════════════════════════════════════════════════════════════════════════════
def post_json(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def data_url(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()


def extract_json(text: str):
    """从模型输出里挖第一个可解析的 JSON 对象(思考块/代码围栏都容忍)。"""
    for m in re.finditer(r"\{", text):
        depth = 0
        for i in range(m.start(), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[m.start() : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def parse_points(reply: str, colors: list[str]) -> tuple[dict, str]:
    """→ ({color: (x, y)}, unit) ;unit ∈ {"norm", "px", "?"}"""
    obj = extract_json(reply)
    raw: list[tuple[str | None, float, float]] = []
    if isinstance(obj, dict):
        pts = obj.get("points") or obj.get("squares") or obj.get("elements")
        if isinstance(pts, dict):
            pts = [{"color": k, "x": v[0], "y": v[1]} for k, v in pts.items() if len(v) == 2]
        if isinstance(pts, list):
            for p in pts:
                if not isinstance(p, dict):
                    continue
                name = p.get("color") or p.get("colour") or p.get("name") or p.get("label")
                try:
                    raw.append((str(name).lower() if name else None, float(p["x"]), float(p["y"])))
                except (KeyError, TypeError, ValueError):
                    continue
    if not raw:  # 兜底:逐色正则 "<color> ... x: N ... y: M"
        for c in colors:
            m = re.search(rf"{c}[^0-9]{{0,40}}?(-?\d+(?:\.\d+)?)\D+?(-?\d+(?:\.\d+)?)", reply, re.I)
            if m:
                raw.append((c, float(m.group(1)), float(m.group(2))))
    if not raw:
        return {}, "?"

    vals = [v for _, x, y in raw for v in (x, y)]
    unit = "norm" if max(abs(v) for v in vals) <= 1010 else "px"

    out: dict[str, tuple[float, float]] = {}
    for idx, (name, x, y) in enumerate(raw):
        key = name if name in colors else (colors[idx] if idx < len(colors) else None)
        if key and key not in out:
            out[key] = (x, y)
    return out, unit


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan"), float("nan")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return b, my - b * mx


def cmd_probe(a: argparse.Namespace) -> None:
    fx = Path(a.fixture)
    gt = json.loads((fx / "gt.json").read_text(encoding="utf-8"))
    png = Path(gt["image"]["path"])
    W, H = gt["image"]["width"], gt["image"]["height"]
    squares = gt["squares"]
    colors = [s["color"] for s in squares]
    url_img = data_url(png)

    tag = a.tag or f"{fx.name}{'-evict' if a.evict else ''}"
    rows, replies = [], []
    t0 = time.time()
    for rep in range(a.repeat):
        part = {"type": "image_url", "image_url": {"url": url_img}}
        if a.evict:
            part["owl_evict"] = True
        body = {
            "model": a.model,
            "messages": [
                {"role": "system", "content": SYS},
                {
                    "role": "user",
                    "content": [
                        part,
                        {"type": "text", "text": user_prompt(colors, a.nonce)},
                    ],
                },
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": a.max_tokens,
            "stream": False,
            "session": f"vg-{tag}-{rep}-{int(time.time())}",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = post_json(a.url, body, a.timeout)
        msg = resp["choices"][0]["message"]
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
        usage = resp.get("usage") or {}
        pts, unit = parse_points(text, colors)
        replies.append({"text": text, "points": pts, "unit": unit, "usage": usage})
        print(
            f"[probe {tag}] rep{rep}: 解析 {len(pts)}/{len(colors)} 点 unit={unit} "
            f"prompt_tok={usage.get('prompt_tokens')} out_tok={usage.get('completion_tokens')} "
            f"{time.time() - t0:.1f}s",
            file=sys.stderr,
        )
        if a.echo:
            print(text, file=sys.stderr)
        for s in squares:
            if s["color"] not in pts:
                rows.append({**s, "rep": rep, "hit": False})
                continue
            px, py = pts[s["color"]]
            if unit == "norm":
                x_sent, y_sent = px / 1000 * W, py / 1000 * H
            else:
                x_sent, y_sent = px, py
            rows.append(
                {
                    **s,
                    "rep": rep,
                    "hit": True,
                    "x_sent": x_sent,
                    "y_sent": y_sent,
                    "dx": x_sent - s["x"],
                    "dy": y_sent - s["y"],
                }
            )

    got = [r for r in rows if r["hit"]]
    scale = a.screen_scale  # sent-space px → screen px
    xs = [r["x"] for r in got]
    ys = [r["y"] for r in got]
    bx, ax = ols(xs, [r["x_sent"] for r in got])
    by, ay = ols(ys, [r["y_sent"] for r in got])

    def stats(key: str) -> dict:
        v = [r[key] for r in got]
        if not v:
            return {}
        return {
            "mean": sum(v) / len(v),
            "median": sorted(v)[len(v) // 2],
            "rmse": math.sqrt(sum(t * t for t in v) / len(v)),
            "max_abs": max(abs(t) for t in v),
        }

    sd = stats("dx")
    sy = stats("dy")
    errs = [math.hypot(r["dx"], r["dy"]) * scale for r in got]
    report = {
        "tag": tag,
        "url": a.url,
        "model": a.model,
        "image": gt["image"],
        "evict": a.evict,
        "repeat": a.repeat,
        "cell_px_sent": gt.get("cell_px"),
        "screen_scale": scale,
        "unit": replies[0]["unit"] if replies else "?",
        "parsed": len(got),
        "expected": len(rows),
        "dx_sent": sd,
        "dy_sent": sy,
        "dx_screen": {k: v * scale for k, v in sd.items()},
        "dy_screen": {k: v * scale for k, v in sy.items()},
        "slope_x": bx,
        "intercept_x": ax,
        "slope_y": by,
        "intercept_y": ay,
        "hit_screen": {
            f"@{k}px": (sum(1 for e in errs if e <= k) / len(errs) if errs else 0.0)
            for k in (16, 32, 64, 128)
        },
        "err_screen_px": sorted(errs),
        "rows": rows,
        "replies": replies,
    }
    out = fx / f"probe-{tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    def f(v, nd=2):
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"

    print(f"\n## {tag}  (unit={report['unit']}, 解析 {len(got)}/{len(rows)}, scale×{scale})")
    print("| 量 | mean | median | rmse | max |")
    print("|---|---|---|---|---|")
    for name, st in (("dx@sent", sd), ("dy@sent", sy)):
        if st:
            print(f"| {name} | {f(st['mean'])} | {f(st['median'])} | {f(st['rmse'])} | {f(st['max_abs'])} |")
    for name, st in (
        ("dx@screen", report["dx_screen"]),
        ("dy@screen", report["dy_screen"]),
    ):
        if st:
            print(f"| {name} | {f(st['mean'])} | {f(st['median'])} | {f(st['rmse'])} | {f(st['max_abs'])} |")
    print(f"| slope x / b | {f(bx, 4)} | intercept | {f(ax)} | — |")
    print(f"| slope y / b | {f(by, 4)} | intercept | {f(ay)} | — |")
    print("| hit@screen | " + " | ".join(f"{k} {v * 100:.0f}%" for k, v in report["hit_screen"].items()) + " | — | — |")
    print(f"\n→ {out}")


# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="视觉 grounding 对拍探针")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="生成合成图 + GT")
    m.add_argument("--out", required=True)
    m.add_argument("--size", default="1280x720")
    m.add_argument("--n", type=int, default=12)
    m.add_argument("--sq", default="10,48", help="方块边长,逗号分隔(每个边长一张图)")
    m.add_argument("--margin", type=int, default=60)
    m.add_argument("--min-gap", type=int, default=120, dest="min_gap")
    m.add_argument("--seed", type=int, default=7)
    m.set_defaults(fn=cmd_make)

    p = sub.add_parser("probe", help="打端点算偏差结构")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fixture", required=True)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--evict", action="store_true", help="图像 part 带 owl_evict(owl 删除臂)")
    p.add_argument("--screen-scale", type=float, default=1.0, dest="screen_scale",
                   help="sent→screen 像素倍率(屏幕 2560x1440 / 发出 1280x720 ⇒ 2)")
    p.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--tag", default=None)
    p.add_argument("--echo", action="store_true")
    p.add_argument("--nonce", default="",
                   help="提示词里的 run tag(空 = 不带);同 nonce + 同文本 + 不同图 = 探陈旧 KV"
                   )
    p.set_defaults(fn=cmd_probe)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
