#!/usr/bin/env python3
"""HF baseline:embedding + tied lm_head(查表 + x @ E^T;E 源 [vocab,D])。

输入张量:ids [tokens] f32(数值形态)/ w [vocab*D] f32(源 [vocab,D])/
         x [1, D] f32(lm_head 输入)
契约:{"vocab": vocab, "d": d}
输出张量:y [tokens, D] f32 / logits [1, vocab] f32
"""

import torch

from common import run


def test(t: dict, c: dict):
    ids = t["ids"].to(torch.long)
    w = t["w"].reshape(int(c["vocab"]), int(c["d"]))
    x = t["x"]

    y = torch.nn.functional.embedding(ids, w)  # HF 同式(nn.Embedding 前向)
    logits = torch.nn.functional.linear(x, w)  # tied lm_head = x @ E^T
    return {"y": y, "logits": logits}, "torch F.embedding/F.linear(HF nn.Embedding 同式)"


if __name__ == "__main__":
    run(test)
