"""导出 HF 快照里的张量(f32 LE)到文件;对拍用(见 source.rs/transform 门)。"""
import json, sys
import numpy as np, torch
from safetensors import safe_open
HF, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(HF + "/model.safetensors.index.json"))["weight_map"]
with safe_open(HF + "/" + m[key], framework="pt") as h:
    t = h.get_tensor(key).to(torch.float32).numpy()
open(out, "wb").write(np.ascontiguousarray(t).tobytes())
print(key, t.shape, file=sys.stderr)
