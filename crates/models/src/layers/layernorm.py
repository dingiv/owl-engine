#!/usr/bin/env python3
"""HF baseline:LayerNorm(affine)—— **Qwen3.5-VL 视觉塔**的 norm。

语义锚:`torch.nn.LayerNorm(normalized_shape, eps)` —— HF
`Qwen3_5VisionBlock.norm1/norm2` 与 `Qwen3_5VisionPatchMerger.norm` 直用它
(`transformers/models/qwen3_5/modeling_qwen3_5.py`,已核对:eps=1e-6、
elementwise_affine=True、biased variance)。视觉塔**没有任何 RMSNorm**。

口径(two-pass,与 torch 内部 Welford 同序):
    mean = Σx / n;var = Σ(x−mean)² / n;y = (x−mean)·rsqrt(var+eps)·w + b

输入张量:x [rows, n] f32 / weight [n] f32 / bias [n] f32
契约:{"eps": 1e-6}
输出张量:y [rows, n] f32
"""

import torch

from common import run


def test(t: dict, c: dict):
    x, w, b = t["x"], t["weight"], t["bias"]
    eps = float(c.get("eps", 1e-6))
    n = int(w.numel())
    ln = torch.nn.LayerNorm(n, eps=eps, elementwise_affine=True)
    with torch.no_grad():
        ln.weight.copy_(w)
        ln.bias.copy_(b)
        y = ln(x)
    return {"y": y}, "torch.nn.LayerNorm(vision-tower 直用;elementwise_affine=True)"


if __name__ == "__main__":
    run(test)
