#!/usr/bin/env python3
"""HF baseline:rmsnorm(Qwen3.5 全系零中心 ×(1+w);×w 语义喂 w-1)。

**HF 实证(2026-09-26,parity 管道首战)**:Qwen3_5RMSNorm 全系
(主干 + q/k norm)零中心 —— weight 零初始化,forward = x×(1+w)。
×w 语义(Qwen3 旧系)= 喂 w-1(1+(w-1)=w)。

输入张量:x [rows, n] f32 / weight [n] f32
契约:{"eps": 1e-6, "w_off": 0|1}
输出张量:y [rows, n] f32
"""

import torch

from common import hf_import, run


def test(t: dict, c: dict):
    x, w = t["x"], t["weight"]
    eps = float(c.get("eps", 1e-6))
    w_off = bool(int(c.get("w_off", 0)))

    cls, prov = hf_import([
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNorm"),
        ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextRMSNorm"),
        ("transformers.models.qwen3.modeling_qwen3", "Qwen3RMSNorm"),
    ])
    if cls is not None:
        # HF 零中心类:weight = offset;×w 语义喂 w-1
        w_in = w if w_off else (w - 1.0)
        layer = cls(w.numel(), eps=eps).to(torch.float32)
        with torch.no_grad():
            layer.weight.copy_(w_in)
        y = layer(x)
    else:
        variance = x.pow(2).mean(-1, keepdim=True)
        y = x * torch.rsqrt(variance + eps) * (w if w_off else (w - 1.0) + 1.0)
        prov += "+manual"

    return {"y": y}, prov


if __name__ == "__main__":
    run(test)
