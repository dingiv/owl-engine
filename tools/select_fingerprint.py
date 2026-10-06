#!/usr/bin/env python3
"""selector 指纹切分(E5-DF3 收尾):golden hidden(已证全对)+ 公式 LM 表 +
检查点码本/proj → host 复算 top16 与 walk → 与 golden drafts / owl cand 对表。
用法: .venv/bin/python tools/select_fingerprint.py [owl_cand0_16 逗号分隔]"""
import json, struct, sys

import numpy as np
import torch

ROOT = "/home/div/Documents/codes"
p = f"{ROOT}/packages/owl/testdata/dflash2_golden.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    meta = json.loads(f.read(n))
mm = np.memmap(p, dtype=np.uint8, mode="r")


def raw(key):
    v = meta[key]
    s, e = 8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1]
    dt = {"F16": np.float16, "F32": np.float32}[v["dtype"]]
    return np.frombuffer(mm[s:e].tobytes(), dtype=dt).reshape(v["shape"])


hid = torch.tensor(raw("hidden").astype(np.float32))  # [8,5120]

V, H, K = 248320, 5120, 16
i = torch.arange(V * H, dtype=torch.float64)
LM = (torch.sin((i + 21.0) * 0.23).float() * 0.7).reshape(V, H).half()

logits = (hid[1:].half() @ LM.T).float()
topv, topi = logits.topk(K, dim=-1)
print("py topi[0][:8] =", topi[0, :8].tolist())
print("py topv[0][:4] =", topv[0, :4].tolist())

ck = f"{ROOT}/models/syvai/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"
with open(ck, "rb") as f:
    n2 = struct.unpack("<Q", f.read(8))[0]
    meta2 = json.loads(f.read(n2))
mm2 = np.memmap(ck, dtype=np.uint8, mode="r")


def craw(key):
    v = meta2[key]
    s, e = 8 + n2 + v["data_offsets"][0], 8 + n2 + v["data_offsets"][1]
    dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I32": np.int32}[v["dtype"]]
    a = np.frombuffer(mm2[s:e].tobytes(), dtype=dt)
    if v["dtype"] == "BF16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return torch.tensor(a.astype(np.float32)).reshape(v["shape"])


A = craw("candidate_selector.predecessor_codebook")
B = craw("candidate_selector.successor_codebook")
Wp = craw("candidate_selector.hidden_projection.weight")
proj = (hid[1:].half() @ Wp.T.half()).float()  # [7,256]

anchor = 108820.0
toks = []
idx_prev = None
for e in range(7):
    pred = torch.full((K,), anchor, dtype=torch.long) if e == 0 else topi[e - 1]
    sc = topv[e][None, :] + ((A[pred] * proj[e][None, :]) @ B[topi[e]].T)
    pick = int(sc[0].argmax()) if e == 0 else int(sc[idx_prev].argmax())
    idx_prev = pick
    toks.append(float(topi[e][pick]))
print("py walk toks =", toks)
print("golden drafts =", raw("drafts").tolist())

if len(sys.argv) > 1 and sys.argv[1]:
    owl = [float(x) for x in sys.argv[1].split(",")]
    print("owl cand[0][:16] =", owl)
    print("owl∈py_top16 row0:", sorted(set(owl) & set(topi[0].tolist()), key=lambda v: topi[0].tolist().index(v)))
    miss = sorted(set(topi[0].tolist()) - set(owl))
    print("py_top16 row0 缺失于 owl:", miss[:8], f"(共缺 {len(miss)})")
