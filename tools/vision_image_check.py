#!/usr/bin/env python3
"""视觉塔**真图**验收:processor → 塔金标 + HF 端到端识别答案。

用途(R2 闭环):合成 randn 图只验"塔内数学";**真图**额外覆盖
① `Qwen3VLProcessor` 的真预处理(缩放/patchify/**2×2 块序**)
② 图尺寸多样性(非方形网格、patch 数非 2 的幂)
③ 端到端"这张图是什么"—— 证明塔输出对文本模型**可用**。

产出(-> OUT.safetensors,逐图一组):
    in.<tag>.pixel_values / in.<tag>.grid_thw(真 processor 口径)
    g.<tag>.merged          —— HF 塔的合并输出([n/4, out_hidden]);Rust 侧对拍锚
    g.<tag>.pre_merger      —— 塔内最后 hidden(逐层对拍可只对这两件)
    ans.<tag> = HF 端到端生成文本(字符串,存进 sidecar json 而非 safetensors)
    + OUT.json:每图答案 / 网格 / 生成参数(台账要记的东西)

跑法:crates/models venv(有 transformers+torch+torchvision+PIL):
    uv run --project crates/models python tools/vision_image_check.py [CKPT] [IMG_DIR] [OUT]
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
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "testdata/vision_real.safetensors")
PICK = sys.argv[4] if len(sys.argv) > 4 else ""  # 只跑指定图(逗号分隔;空 = 全跑)
QUESTION = "What is in this image? Answer with one word."


def main() -> None:
    from transformers import Qwen3_5ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(CKPT)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(CKPT, dtype=torch.float32).eval()
    visual = model.model.visual  # Qwen3_5VisionModel(depth 24 / hidden 1024 / out 2560)

    out: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {"ckpt": CKPT, "question": QUESTION, "images": {}}
    names = sorted(p.stem for p in IMGS.glob("*.png"))
    if PICK:
        want = set(PICK.split(","))
        names = [n for n in names if n in want]
    for name in names:
        tag = name.replace("-", "_")
        im = Image.open(IMGS / f"{name}.png").convert("RGB")
        proc_out = proc.image_processor(images=[im], return_tensors="pt")
        pv, grid = proc_out["pixel_values"], proc_out["image_grid_thw"]

        # ① 塔金标(f32;CPU)+ **表/早期层**(真网格上定位用:patch_embed/pos/h0/表)
        from transformers.vision_utils import (
            get_vision_interpolation_indices_and_weights, get_vision_position_ids,
        )
        ii, iw = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
            align_corners=visual.interpolation_align_corners, spatial_merge_size=visual.spatial_merge_size)
        pos_ids = get_vision_position_ids(grid, visual.spatial_merge_size)
        with torch.no_grad():
            pe = visual.patch_embed(pv)
            pos_emb = (visual.pos_embed(ii) * iw[:, :, None]).sum(1)
            h0 = pe + pos_emb.to(pe.dtype)
            cos, sin = visual.rotary_pos_emb(h0, pos_ids)
        out[f"g.{tag}.peak_embed"] = pe.float()
        out[f"g.{tag}.pos_embed"] = pos_emb.float()
        out[f"g.{tag}.h0"] = h0.float()
        out[f"g.{tag}.interp_indices"] = ii.to(torch.float32)
        out[f"g.{tag}.interp_weights"] = iw.to(torch.float32)
        out[f"g.{tag}.cos"] = cos.float()
        out[f"g.{tag}.sin"] = sin.float()
        with torch.no_grad():
            vis = visual(pv, grid_thw=grid)
        merged = vis.pooler_output.float()          # [n/4, out_hidden]
        pre = vis.last_hidden_state.float()          # [n, hidden]
        out[f"in.{tag}.pixel_values"] = pv
        out[f"in.{tag}.grid_thw"] = grid.to(torch.float32)
        out[f"g.{tag}.merged"] = merged
        out[f"g.{tag}.pre_merger"] = pre

        # ①a2 **块内中间量**(钩子捕获;定位用:norm1 → qkv → attn → blk0)
        cap: dict[str, torch.Tensor] = {}
        hs = [
            visual.blocks[0].norm1.register_forward_hook(lambda m, i, o: cap.__setitem__("norm1", o.detach())),
            visual.blocks[0].attn.qkv.register_forward_hook(lambda m, i, o: cap.__setitem__("qkv", o.detach())),
            visual.blocks[0].attn.register_forward_hook(lambda m, i, o: cap.__setitem__("attn", o.detach())),
            visual.blocks[0].register_forward_hook(lambda m, i, o: cap.__setitem__("blk0", o.detach())),
        ]
        with torch.no_grad():
            visual(pv, grid_thw=grid)
        for h in hs:
            h.remove()
        for k, v in cap.items():
            out[f"g.{tag}.{k}"] = v.float()

        # ①b 塔 f16 档金标(HF 自己半精度;供 owl f16 GPU 路径"同档"对拍)
        #     + **逐层 block 输出**(f32/f16 两档;定位"第一发散层"用)
        caps_g: dict[str, torch.Tensor] = {}
        hs = [
            blk.register_forward_hook(
                (lambda i: lambda m, inp, o: caps_g.__setitem__(f"blk{i:02d}", o.detach()))(bi)
            )
            for bi, blk in enumerate(visual.blocks)
        ]
        with torch.no_grad():
            visual(pv, grid_thw=grid)
        for h in hs:
            h.remove()
        for k, v in caps_g.items():
            out[f"g.{tag}.{k}"] = v.float()

        vh = visual.half()
        caps_h: dict[str, torch.Tensor] = {}
        hs = [
            blk.register_forward_hook(
                (lambda i: lambda m, inp, o: caps_h.__setitem__(f"blk{i:02d}", o.detach()))(bi)
            )
            for bi, blk in enumerate(vh.blocks)
        ]
        with torch.no_grad():
            vis16 = vh(pv.half(), grid_thw=grid)
        for h in hs:
            h.remove()
        for k, v in caps_h.items():
            out[f"h.{tag}.{k}"] = v.float()
        visual.float()
        out[f"h.{tag}.merged"] = vis16.pooler_output.float()
        out[f"h.{tag}.pre_merger"] = vis16.last_hidden_state.float()
        visual.float()  # 复位(后续生成用 f32)

        # ② 端到端识别(HF 参考)
        messages = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": QUESTION}]}]
        # 关 thinking(Qwen3.5 模板:`enable_thinking=False`;不关则前 12 token 全在"想")
        try:
            text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[im], return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        answer = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        meta["images"][tag] = {
            "size": list(im.size),
            "grid_thw": grid.tolist(),
            "n_patch": int(pv.shape[0]),
            "n_merged": int(merged.shape[0]),
            "merged_sum": round(float(merged.sum()), 4),
            "answer": answer,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
        }
        print(f"[real] {name:12s} {str(im.size):12s} grid {grid.tolist()} merged {tuple(merged.shape)} -> {answer!r}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, OUT)
    OUT.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
