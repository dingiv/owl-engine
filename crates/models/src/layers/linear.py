#!/usr/bin/env python3
"""HF baseline:linear(x @ W^T;W 源布局 [out,in],HF F.linear 同式)。

输入张量:x [tokens, in] f32 / w [out*in] f32(源 [out,in] 行主序)
契约:{"out": out_dim, "in": in_dim}
输出张量:y [tokens, out] f32
"""

import torch

from common import run


def test(t: dict, c: dict):
    x, w = t["x"], t["w"]
    out_dim, in_dim = int(c["out"]), int(c["in"])
    w = w.reshape(out_dim, in_dim)
    y = torch.nn.functional.linear(x, w)  # = x @ w.T(HF Linear 同式)
    return {"y": y}, "torch F.linear(HF Linear 同式)"


if __name__ == "__main__":
    run(test)
