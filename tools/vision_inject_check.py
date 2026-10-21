#!/usr/bin/env python3
"""**真图识别验收**:HF 文本模型 × **owl 塔的输出** —— 端到端"它认得出吗"。

做法:monkey-patch `model.model.visual.forward`,让它返回
`(last_hidden_state, pooler_output) = (owl 的 pre_merger, owl 的 merged)`
—— 即"文本模型吃的是 owl 塔算出来的 image embedding";然后正常 generate。

对照三档:HF 自带塔 / owl f32 塔 / owl f16 塔。答案一致 = 我们的塔对下游**可用**。

跑法(需先 `OWL_VIS_DUMP=1` 跑 Rust 的 `dump_real_tower_outputs` 落盘):
    uv run --project crates/models python tools/vision_inject_check.py [CKPT] [IMG_DIR] [OURS_DIR]
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor

CKPT = sys.argv[1] if len(sys.argv) > 1 else "./models/Qwen/Qwen3.5-4B"
IMGS = Path(sys.argv[2] if len(sys.argv) > 2 else "./assets")
OURS = Path(sys.argv[3] if len(sys.argv) > 3 else "testdata")
QUESTION = "What is in this image? Answer with one word."


def main() -> None:
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    proc = AutoProcessor.from_pretrained(CKPT)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(CKPT, dtype=torch.float32).eval()
    visual = model.model.visual
    orig_visual_forward = visual.forward

    ours = {}
    for tag in ("f32", "f16"):
        p = OURS / f"vision_ours_{tag}.safetensors"
        ours[tag] = load_file(p) if p.exists() else None

    def gen_answer(im) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": QUESTION}]}]
        try:
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[im], return_tensors="pt")
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        return proc.batch_decode(g[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    rows = []
    for name in sorted(p.stem for p in IMGS.glob("*.png")):
        im = Image.open(IMGS / f"{name}.png").convert("RGB")
        answers = {}
        for tag in ("hf", "f32", "f16"):
            if tag == "hf":
                visual.forward = orig_visual_forward
            else:
                if ours[tag] is None or f"{name}.merged" not in ours[tag]:
                    answers[tag] = "(缺 owl 输出文件)"
                    continue
                pre = ours[tag][f"{name}.pre_merger"]
                mer = ours[tag][f"{name}.merged"]

                def patched(*_a, _pre=pre, _mer=mer, **_k):
                    return BaseModelOutputWithPooling(last_hidden_state=_pre, pooler_output=_mer)

                visual.forward = patched
            answers[tag] = gen_answer(im)
        visual.forward = orig_visual_forward
        rows.append((name, answers))
        print(f"[inject] {name:12s} HF={answers['hf']!r:12} owl-f32={answers['f32']!r:12} owl-f16={answers['f16']!r}")
    print(json.dumps({"ckpt": CKPT, "rows": [{"image": n, **a} for n, a in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
