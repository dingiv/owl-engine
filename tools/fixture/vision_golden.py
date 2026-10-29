#!/usr/bin/env python3
"""视觉塔(Qwen3.5-VL ViT)金标生成器 —— HF transformers 参考实现 → safetensors。

配方(可复现;设计文档 `docs/arch/vision-tower-设计与CPU对拍.md` §4.1):

    cd packages/worktrees/owl/vision-tower
    uv run --project crates/models python tools/vision_golden.py \
        [CKPT] [OUT.safetensors] [grid="1,4,6"] [seed]

默认: CKPT=models/Hcompany/Holo-3.1-9B, OUT=testdata/vision_golden.safetensors,
      grid=(t=1,h=4,w=6) ⇒ 24 patch / 6 merged token(最小可验形状), seed=0。

口径(三条硬约定):
1. **只读 `model.visual.*`**(333 张量,全在 shard 4)⇒ 不装 18.8G 文本塔;
2. 权重 **BF16 → f32**(参考实现按 f32 跑;owl CPU face 是 f32 世界);
3. 输入 = **固定种子 randn 的 pixel_values**(不跑 processor)⇒ 本份金标只验**塔内数学**,
   预处理(解码/缩放/patchify 顺序)另案(M5)。

产出键集合(设计文档 §4.1 逐条对位):
    in.pixel_values [N,3·2·16·16] · in.grid_thw [n_img,3] i64
    g.cos [N,72] · g.sin [N,72] · g.interp_indices [N,4] · g.interp_weights [N,4]
    g.patch_embed [N,1152] · g.pos_embed [N,1152] · g.h0 [N,1152]
    g.blk_00 … g.blk_{depth-1:02d}  ([N,1152],逐层)
    g.pre_merger [N,1152] · g.merged [N/4,4096]
stdout = meta JSON(版本/ckpt md5/seed/判据 —— 台账要记的东西)。
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---- 参数 -----------------------------------------------------------------
CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else "./models/Hcompany/Holo-3.1-9B")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "testdata/vision_golden.safetensors")
GRID = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "1,4,6").split(",")]
SEED = int(sys.argv[4] if len(sys.argv) > 4 else 0)

import transformers  # noqa: E402  (版本进 meta)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel  # noqa: E402
from transformers.vision_utils import (  # noqa: E402
    get_vision_interpolation_indices_and_weights, get_vision_position_ids,
)


def load_visual_weights(model, ckpt: Path) -> int:
    """按 index 找含 `model.visual.` 的 shard,只取该前缀的 333 张量(→ f32)。"""
    idx = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({s for k, s in idx.items() if k.startswith("model.visual.")})
    sd, n = {}, 0
    for shard in shards:
        with safe_open(ckpt / shard, framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual."):]] = f.get_tensor(k).to(torch.float32)
                    n += 1
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"权重键不齐:missing={missing[:4]} unexpected={unexpected[:4]}")
    return n


def md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> None:
    cfg = Qwen3_5Config.from_pretrained(CKPT)
    vc = cfg.vision_config
    model = Qwen3_5VisionModel(vc).to(torch.float32).eval()
    n_w = load_visual_weights(model, CKPT)

    grid = torch.tensor([GRID], dtype=torch.int64)
    n_patch = int(grid.prod())
    torch.manual_seed(SEED)
    # 输入口径 = **processor 归一化后的范围**((x−0.5)/0.5 ∈ [−1,1];mean/std 见 preprocessor_config)。
    # 不用 N(0,1):未归一化输入会让末层激活爆到 ~7e3(rms 105),绝对判据失去意义(实测)。
    px = torch.rand(n_patch, vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size) * 2.0 - 1.0

    # ---- host 表(HF 原样调用;Rust 侧纯函数要对的就是这两张) ----
    interp_i, interp_w = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=model.num_grid_per_side, mode=model.interpolation_mode,
        align_corners=model.interpolation_align_corners, spatial_merge_size=vc.spatial_merge_size)
    pos_ids = get_vision_position_ids(grid, vc.spatial_merge_size)
    cu = torch.repeat_interleave((grid[:, 1] * grid[:, 2]), grid[:, 0])
    cu_seqlens = torch.nn.functional.pad(cu.cumsum(0, dtype=torch.int32), (1, 0), value=0)

    out = {}
    with torch.no_grad():
        pe = model.patch_embed(px)
        out["g.patch_embed"] = pe
        pos = (model.pos_embed(interp_i) * interp_w[:, :, None]).sum(1)
        out["g.pos_embed"] = pos
        h = pe + pos.to(pe.dtype)
        out["g.h0"] = h
        cos, sin = model.rotary_pos_emb(h, pos_ids)
        out["g.cos"], out["g.sin"] = cos, sin
        for i, blk in enumerate(model.blocks):
            h = blk(h, cu_seqlens=cu_seqlens, max_seqlen=int(cu.max()),
                    position_embeddings=(cos, sin))
            out[f"g.blk_{i:02d}"] = h
        out["g.pre_merger"] = h.clone()  # 与 blk_{depth-1} 同值;clone 免 safetensors 共享内存告警
        out["g.merged"] = model.merger(h)

    # ---- f16 档金标(HF 自己半精度跑一遍;`h.` 前缀)----
    # 用途:owl 的 f16 GPU 路径与「HF f16」**同档**对比(对 f32 金标只能量 f16 漂移的绝对量,
    # 分不清是"我们的核错"还是"f16 本就该差这么多")。
    model_h = copy.deepcopy(model).half()
    px_h = px.half()
    with torch.no_grad():
        h16 = model_h.patch_embed(px_h)
        out["h.patch_embed"] = h16.float()
        pos_h = (model_h.pos_embed(interp_i) * interp_w[:, :, None]).sum(1)
        out["h.pos_embed"] = pos_h.float()
        hh = h16 + pos_h.to(h16.dtype)
        out["h.h0"] = hh.float()
        cos_h, sin_h = model_h.rotary_pos_emb(hh, pos_ids)
        for i, blk in enumerate(model_h.blocks):
            hh = blk(hh, cu_seqlens=cu_seqlens, max_seqlen=int(cu.max()),
                     position_embeddings=(cos_h, sin_h))
            out[f"h.blk_{i:02d}"] = hh.float()
        out["h.merged"] = model_h.merger(hh).float()

    out["in.pixel_values"] = px
    # 网格也存 f32(仓内契约 5:索引/形状量 f32 过线;值小,精确无损)
    out["in.grid_thw"] = grid.to(torch.float32)
    # 索引量转 f32(仓内契约 5:索引 f32 过线;值 < 2^24 精确无损)
    out["g.interp_indices"] = interp_i.to(torch.float32)
    out["g.interp_weights"] = interp_w

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, OUT)

    meta = {
        "script": "tools/vision_golden.py",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "ckpt": str(CKPT),
        "ckpt_shard_md5": {s: md5(CKPT / s) for s in
                           sorted({v for k, v in
                                   json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"].items()
                                   if k.startswith("model.visual.")})},
        "n_visual_tensors": n_w,
        "grid_thw": GRID,
        "seed": SEED,
        "input_norm": "uniform[-1,1] == processor 归一化后范围 ((x-0.5)/0.5)",

        "n_patch": n_patch,
        "n_merged": int(out["g.merged"].shape[0]),
        "vision_cfg": {k: getattr(vc, k) for k in
                       ("depth", "hidden_size", "intermediate_size", "num_heads", "out_hidden_size",
                        "num_position_embeddings", "spatial_merge_size", "temporal_patch_size",
                        "patch_size", "hidden_act", "in_channels", "deepstack_visual_indexes")},
        "head_dim": vc.hidden_size // vc.num_heads,
        "rope": {"theta": vc.rope_parameters["rope_theta"], "inv_freq_len": int(model.rotary_pos_emb.inv_freq.numel())},
        "checksums": {k: round(float(v.float().sum()), 6) for k, v in out.items() if v.is_floating_point()},
        "out": str(OUT),
    }
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
