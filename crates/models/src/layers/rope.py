#!/usr/bin/env python3
"""HF baseline:rope(Qwen3.5 text = rotate-half + partial;金标准 = HF 类直跑)。

**实证定案(2026-09-26 探针)**:config `mrope_interleaved` 指三网格(T/H/W)
在频率维交错(recomposition_frequencies slice(·,·,3)),非 GPT-J 相邻对;
文本路径三网格同 pos 退化后 = 普通 rotate-half(pair (i, i+half)),
partial 段直通。HF vs rotate-half maxdiff 6e-8 / vs 相邻对 2.5。

输入张量:x [tokens, heads*head_dim] f32 / pos [tokens] f32(数值形态)
契约:{"heads": H, "head_dim": HD, "rotary_dim": RD, "theta": 1e7,
       "max_pos": 上下限(仅配构造)}
输出张量:y [tokens, heads*head_dim] f32
"""

import torch

from common import hf_import, run

_rot_cache: dict = {}


def hf_rotary(c: dict):
    """构造 HF Qwen3_5TextRotaryEmbedding(toy config;缓存复用)"""
    key = (c["head_dim"], c["rotary_dim"], float(c["theta"]))
    if key not in _rot_cache:
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        cls, prov = hf_import([
            ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5TextRotaryEmbedding"),
        ])
        if cls is None:
            return None, prov
        cfg = Qwen3_5TextConfig(
            hidden_size=64, num_attention_heads=2, num_key_value_heads=1,
            head_dim=int(c["head_dim"]),
            rope_parameters={
                "rope_type": "default", "rope_theta": float(c["theta"]),
                "partial_rotary_factor": c["rotary_dim"] / c["head_dim"],
                "mrope_interleaved": True, "mrope_section": [1, 1, 1],
            },
            max_position_embeddings=max(int(c.get("max_pos", 4096)), 512),
        )
        _rot_cache[key] = (cls(cfg), prov)
    return _rot_cache[key]


def rotate_half_ref(x: torch.Tensor, pos: torch.Tensor, c: dict) -> torch.Tensor:
    """手工兜底:rotate-half + partial(与 HF 同式独立副本)"""
    hd, rd = int(c["head_dim"]), int(c["rotary_dim"])
    half = rd // 2
    q2 = x.reshape(*x.shape[:-1], -1, hd)
    idx = torch.arange(0, rd, 2, dtype=torch.float32)
    inv_freq = float(c["theta"]) ** (-idx / rd)
    ang = pos.reshape(-1, 1).float() * inv_freq  # [tokens, half]
    cos, sin = ang.cos(), ang.sin()
    rot, pas = q2[..., :rd], q2[..., rd:]
    r1, r2 = rot[..., :half], rot[..., half:]
    out_rot = torch.cat([r1 * cos - r2 * sin, r2 * cos + r1 * sin], dim=-1)
    return torch.cat([out_rot, pas], dim=-1).reshape(x.shape)


def test(t: dict, c: dict):
    x, pos = t["x"], t["pos"]
    rot, prov = hf_rotary(c)
    if rot is not None:
        tokens = x.shape[0]
        # 文本路径:三网格同 pos → (3, 1, tokens)
        pid = pos.reshape(1, -1).long().expand(3, 1, tokens)
        cos, sin = rot(x, pid)  # [1, tokens, rotary_dim]
        import transformers.models.qwen3_5.modeling_qwen3_5 as m

        x4 = x.reshape(1, tokens, int(c["heads"]), int(c["head_dim"])).transpose(1, 2)
        y4, _ = m.apply_rotary_pos_emb(x4, x4, cos, sin)
        y = y4[0].transpose(0, 1).reshape(x.shape)
    else:
        y = rotate_half_ref(x, pos, c)
        prov += "+manual"

    return {"y": y}, prov


if __name__ == "__main__":
    run(test)
