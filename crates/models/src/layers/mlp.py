#!/usr/bin/env python3
"""HF baseline:SwiGLU MLP(Qwen3_5MLP 同式:down(silu(gate(x)) * up(x)))。

输入张量:x [tokens, hidden] f32 / gate_w / up_w / down_w(源 [out,in] 行主序)
契约:{"hidden": hidden_size, "inter": intermediate_size}
输出张量:y [tokens, hidden] f32
"""

import torch

from common import hf_import, run


def test(t: dict, c: dict):
    x = t["x"]
    h, i = int(c["hidden"]), int(c["inter"])
    gate = t["gate_w"].reshape(i, h)
    up = t["up_w"].reshape(i, h)
    down = t["down_w"].reshape(h, i)

    cls, prov = hf_import([
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5MLP"),
        ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextMLP"),
    ])
    y = None
    if cls is not None:
        try:
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
            cfg = Qwen3_5Config(hidden_size=h, intermediate_size=i)
            mlp = cls(cfg).to(torch.float32)
            with torch.no_grad():
                mlp.gate_proj.weight.copy_(gate)
                mlp.up_proj.weight.copy_(up)
                mlp.down_proj.weight.copy_(down)
                if getattr(mlp, "gate_proj", None) is not None and mlp.gate_proj.bias is not None:
                    mlp.gate_proj.bias.zero_()
                    mlp.up_proj.bias.zero_()
                    mlp.down_proj.bias.zero_()
            y = mlp(x)
        except Exception:
            y = None

    if y is None:
        # HF Qwen3_5MLP 同式(手动;F.linear/F.silu)
        y = (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T
        prov += "+manual"

    return {"y": y}, prov


if __name__ == "__main__":
    run(test)
