#!/usr/bin/env python3
"""M4-d owl 端到端识图**金标**:processor → 输入 + HF 贪心答案 + 位置号。

与 `vision_image_check.py` 的分工:那份只到"塔"为止(塔金标 + HF 答案);
本份给**owl 全链**(塔 + 文本)当锚:

    in.<tag>.ids / pixel_values / grid_thw   —— 真 processor 输入
    hf.<tag>.pos_ids   [3, L]                —— **真模型** `get_rope_index`
                                                (闭合 mrope 的 shim-vs-真身缺口)
    hf.<tag>.gen_ids                         —— HF 贪心生成序列
    hf.<tag>.logits_top{1,2}?                —— 末位 logits top-10(数值面)
    + OUT.json:每图答案文本 / 生成参数 / 版本

跑法(慢:4B cpu 贪心;建议后台):
    uv run --project crates/models python tools/vlm_e2e_golden.py \
        [CKPT] [IMG_DIR] [OUT.safetensors] [N_TOKENS] [PICK]
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/div/Documents/codes/models/Qwen/Qwen3.5-4B"
IMGS = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/div/Documents/codes/assets")
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "testdata/vlm_e2e.safetensors")
N_TOK = int(sys.argv[4]) if len(sys.argv) > 4 else 16
PICK = sys.argv[5] if len(sys.argv) > 5 else ""
QUESTION = "What is in this image? Answer with one word."


def main() -> None:
    from transformers import Qwen3_5ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(CKPT)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        CKPT, dtype=torch.bfloat16
    ).eval()

    out: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {
        "ckpt": CKPT,
        "question": QUESTION,
        "n_tokens": N_TOK,
        "transformers": __import__("transformers").__version__,
        "images": {},
    }
    names = sorted(p.stem for p in IMGS.glob("*.png"))
    if PICK:
        want = set(PICK.split(","))
        names = [n for n in names if n in want]
    assert names, f"{IMGS} 下没有 png"
    for name in names:
        tag = name.replace("-", "_")
        im = Image.open(IMGS / f"{name}.png").convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": QUESTION}],
            }
        ]
        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = proc(text=[text], images=[im], return_tensors="pt")
        ids = inputs["input_ids"]
        grid = inputs["image_grid_thw"]
        ttypes = inputs.get("mm_token_type_ids")
        assert ttypes is not None, "processor 未给 mm_token_type_ids(版本不符?)"
        l = ids.shape[1]

        # ① 真模型位置号(HF 正文算法;owl 侧 mrope 的对照物)
        pos_ids, deltas = model.model.get_rope_index(
            ids, ttypes, image_grid_thw=grid
        )
        # ② 末位 logits(首 token 的数值面)
        with torch.no_grad():
            fwd = model(**inputs, use_cache=False)
        logits = fwd.logits[0, -1].float()
        top = torch.topk(logits, 10)

        # ③ HF 贪心答案
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=N_TOK,
                use_cache=True,
            )
        gen_ids = gen[0, l:]

        out[f"in.{tag}.ids"] = ids[0].to(torch.float32)
        out[f"in.{tag}.mm_types"] = ttypes[0].to(torch.float32)
        out[f"in.{tag}.pixel_values"] = inputs["pixel_values"].to(torch.float32)
        out[f"in.{tag}.grid_thw"] = grid[0].to(torch.float32)
        out[f"hf.{tag}.pos_ids"] = pos_ids[:, 0, :].to(torch.float32)
        out[f"hf.{tag}.delta"] = deltas.reshape(-1).to(torch.float32)
        out[f"hf.{tag}.gen_ids"] = gen_ids.to(torch.float32)
        out[f"hf.{tag}.top_idx"] = top.indices.to(torch.float32)
        out[f"hf.{tag}.top_val"] = top.values.to(torch.float32)
        ans = proc.tokenizer.decode(gen_ids, skip_special_tokens=True)
        meta["images"][tag] = {
            "l": int(l),
            "grid": [int(x) for x in grid[0].tolist()],
            "n_patch": int(inputs["pixel_values"].shape[0]),
            "gen_ids": [int(x) for x in gen_ids.tolist()],
            "answer": ans,
            "delta": int(deltas.reshape(-1)[0].item()),
        }
        print(f"[golden] {tag}: L={l} grid={grid[0].tolist()} 答: {ans!r}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(OUT))
    OUT.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[golden] 写出 {OUT}({len(out)} 张量)")


if __name__ == "__main__":
    main()
