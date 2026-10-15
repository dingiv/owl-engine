#!/usr/bin/env python3
"""HF baseline:M-RoPE(3 网格位置号 + 交错频率重组)—— **Qwen3.5-VL 文本侧**。

两段各自取 HF 真身(不是手抄公式):

1. **位置号** = `Qwen3_5Model.get_rope_index`(未绑定方法调用;`self` 用轻量
   shim —— 该方法只用到 `config.vision_config.spatial_merge_size` 与
   `self.get_vision_position_ids`(已逐行核对 `modeling_qwen3_5.py` 1368-1447);
   shim 的 `get_vision_position_ids` 绑真身,故网格展开/`max(h,w)//merge`
   推进全是 HF 原码);
2. **cos/sin** = `Qwen3_5TextRotaryEmbedding`(真类;toy `Qwen3_5TextConfig`
   带真 `rope_parameters`:sections/theta/partial/interleaved)。

输入张量:ids [L] f32 · ttypes [L] f32(0 文本 / 1 图像 / 2 视频)·
         image_grid_thw [ni, 3] f32 · video_grid_thw [nv, 3] f32(原始 patch 网格)
契约:{"merge": 2, "head_dim": 256, "rotary_dim": 64, "theta": 1e7,
       "sections": [11, 11, 10]}
输出张量:pos_ids [3, L] f32 · delta [1] f32 · cos [L, half] f32 · sin [L, half] f32
"""

import types

import torch

from common import run


def _shim(merge: int):
    """`Qwen3_5Model.get_rope_index` 的轻量 self(shim 只补该方法的两个依赖)"""
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    class _Shim:
        pass

    s = _Shim()
    s.config = types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=int(merge))
    )
    s.get_vision_position_ids = types.MethodType(m.Qwen3_5Model.get_vision_position_ids, s)
    return s


def _rotary(c: dict):
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    hd, rd = int(c["head_dim"]), int(c["rotary_dim"])
    cfg = Qwen3_5TextConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hd,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(c["theta"]),
            "partial_rotary_factor": rd / hd,
            "mrope_interleaved": True,
            "mrope_section": [int(s) for s in c["sections"]],
        },
        max_position_embeddings=512,
    )
    return m.Qwen3_5TextRotaryEmbedding(cfg)


def test(t: dict, c: dict):
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    ids = t["ids"].reshape(1, -1).long()
    ttypes = t["ttypes"].reshape(1, -1).long()
    images = t["image_grid_thw"].reshape(-1, 3).long()
    videos = t["video_grid_thw"].reshape(-1, 3).long()

    pos, delta = m.Qwen3_5Model.get_rope_index(
        _shim(c["merge"]),
        ids,
        ttypes,
        image_grid_thw=images if images.numel() else None,
        video_grid_thw=videos if videos.numel() else None,
    )
    pos = pos[:, 0, :]  # (3, L)

    rot = _rotary(c)
    x = torch.zeros(1, pos.shape[-1], int(c["head_dim"]))  # dtype/device 占位
    # position_ids 契约 = (3, bs, L)(HF forward 内部 expand(3,-1,-1))
    cos, sin = rot(x, pos.unsqueeze(1))  # (1, L, rotary_dim)—— 已 cat(f, f)
    half = int(c["rotary_dim"]) // 2
    cos, sin = cos[0, :, :half], sin[0, :, :half]

    prov = (
        "Qwen3_5Model.get_rope_index(shim self)+Qwen3_5TextRotaryEmbedding"
        "(toy Qwen3_5TextConfig:真 rope_parameters)"
    )
    return {
        "pos_ids": pos.to(torch.float32),
        "delta": delta.reshape(-1).to(torch.float32),
        "cos": cos.to(torch.float32),
        "sin": sin.to(torch.float32),
    }, prov


if __name__ == "__main__":
    run(test)
