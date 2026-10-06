#!/usr/bin/env python3
"""DFlash2 golden 生成器(E5-DF3 排查):真实 W4A16 权重反量化 → f32 前向
(sglang dflash.py 逐式)→ 逐层激活金标存 safetensors。owl 侧同权重同输入
对拍。随机输入 = 固定种子公式(owl 测试同式生成,免存表)。"""
import json, struct, sys
import numpy as np
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/div/Documents/codes/models/syvai/Qwen3.8-27B-DFlash2-W4A16"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/div/Documents/codes/packages/owl/testdata/dflash2_golden.safetensors"
H, INTER, NQ, NKV, HD, NL, G = 5120, 17408, 32, 8, 128, 5, 320
FAN = NL * H
BLOCK, TOPK, RANK, VOCAB, MASK = 8, 16, 256, 248320, 248070
EPS = 1e-6

def read_index(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    hdr = 8 + n
    return {k: (hdr + v["data_offsets"][0], hdr + v["data_offsets"][1], v["dtype"], v["shape"]) for k, v in meta.items()}

idx = read_index(f"{CKPT}/model.safetensors")
mm = np.memmap(f"{CKPT}/model.safetensors", dtype=np.uint8, mode="r")

def bf2f(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def dequant(base):
    packed = raw(f"{base}.weight_packed").astype(np.uint32)
    # scale 读法按真实 dtype 分派(2026-10-07 定谳:本检查点 weight_scale
    # = F16 标注 + F16 字节;曾全走 bf2f 值转换 → scales ≈ 0 → 权重 ≈ 0
    # → python 参考全零 → golden 对拍零比零空转,fc/层路径 bug 全盲)
    s, e, dt, shape = idx[f"{base}.weight_scale"]
    if dt == "F16":
        scales = raw(f"{base}.weight_scale").astype(np.float32)
    else:
        scales = bf2f(raw(f"{base}.weight_scale")).astype(np.float32)
    out, k8 = packed.shape
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    q = ((packed[:, :, None] >> shifts[None, None, :]) & 0xF).reshape(out, k8 * 8).astype(np.int32) - 8
    return q.astype(np.float32) * np.repeat(scales, 128, axis=1)

def raw(key):
    s, e, dt, shape = idx[key]
    dt = {"I32": np.int32, "I64": np.int64, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}[dt]
    return np.frombuffer(mm[s:e].tobytes(), dtype=dt).reshape(shape)

WC = {}
def lin(base, x):
    if base not in WC:
        if f"{base}.weight_packed" in idx:
            WC[base] = torch.tensor(dequant(base), dtype=torch.float32)  # [out, in]
        else:
            WC[base] = torch.tensor(bf2f(raw(f"{base}.weight")).astype(np.float32))
    return x @ WC[base].T

def raww(key):
    if key not in WC:
        WC[key] = torch.tensor(bf2f(raw(key)).astype(np.float32))
    return WC[key]

def rms_add_one(x, key):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * (1 + raww(key))

def rms_plain(x, key):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * raww(key)

def gen(n, seed):
    i = torch.arange(n, dtype=torch.float64)
    return torch.sin((i + seed) * 0.23).float() * 0.7

def conv(xb, delta, base_side, block):
    T, G, Gsz = xb.shape[0], xb.shape[1], xb.shape[2]
    out = torch.zeros_like(xb)
    for tap in range(2):
        coef = base_side[tap].reshape(G, Gsz)[None] + delta[:, tap].unsqueeze(-1)
        if tap == 0:
            out += coef * xb
        else:
            src = torch.zeros_like(xb)
            src[tap:] = xb[:-tap]
            out += coef * src * (torch.arange(T)[:, None, None] >= tap)
    return out

def apply_rope(x, p, hd, nh=None):
    """x [T, nh*hd] 或 [T, nh, hd](已分头);半旋配对 (d, d+half)。"""
    T = x.shape[0]
    half = hd // 2
    inv = 1e7 ** (-(2 * torch.arange(half, dtype=torch.float32)) / hd)
    ang = p[:, None] * inv[None, :]
    c, s = ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)
    xr = x.reshape(T, -1, hd) if nh is None else x.reshape(T, nh, hd)
    a, b = xr[..., :half], xr[..., half:]
    return torch.cat([a * c - b * s, a * s + b * c], -1).reshape(x.shape)

# ── 输入(round-2 稳态:prefix = 位置 0..18(memory 前 19 行),噪声 @ 19..26)──
# ── 真实规模变体(owl AL=0 排查):深层残差规模 taps + 真实 embed/lm_head ──
import os as _os
TGT = _os.environ.get("OWL_TGT_DIR", "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4")
def read_index2(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))
    meta.pop("__metadata__", None)
    hdr = 8 + n
    return {k: (hdr + v["data_offsets"][0], hdr + v["data_offsets"][1], v["dtype"], v["shape"]) for k, v in meta.items()}
def load_bf16(path, key):
    idx2 = read_index2(path)
    s, e, dt, shape = idx2[key]
    mm2 = np.memmap(path, dtype=np.uint8, mode="r")
    b = np.frombuffer(mm2[s:e].tobytes(), dtype=np.uint16).reshape(shape)
    return torch.tensor((b.astype(np.uint32) << 16).view(np.float32), dtype=torch.float32)
import glob as _glob
shards = sorted(_glob.glob(f"{TGT}/model-*.safetensors"))
EMB = None; LMH = None
for sh in shards:
    ix = read_index2(sh)
    if "model.language_model.embed_tokens.weight" in ix:
        EMB = load_bf16(sh, "model.language_model.embed_tokens.weight")
    if "lm_head.weight" in ix:
        LMH = load_bf16(sh, "lm_head.weight")
    if EMB is not None and LMH is not None:
        break
print("EMB", tuple(EMB.shape), "LMH", tuple(LMH.shape), "emb|abs|max", float(EMB.abs().max()))
torch.manual_seed(0)
# 深层残差规模:taps 元素 ~7-50(owl 引擎实测;sglang capture 残差流)
taps = torch.stack([(torch.randn(20, H) * 20).abs().clamp_min(0) * torch.sign(torch.randn(20, H)) for i in range(NL)], 0)
taps = torch.stack([(torch.randn(20, H) * 25) for _ in range(NL)], 0)
E = EMB.to(torch.float32)
LM = LMH.to(torch.float32)
ids = torch.tensor([108820.0] + [MASK] * 7, dtype=torch.float32)
print("taps elem scale:", float(taps[0].abs().mean()))
pos_noise = torch.arange(19, 19 + BLOCK, dtype=torch.float32)
pos_kv = torch.arange(19, dtype=torch.float32)

fcw = torch.tensor(dequant("fc"), dtype=torch.float32)  # [H, FAN]
mem = rms_add_one(sum(taps[i] @ fcw[:, i * H:(i + 1) * H].T for i in range(NL)), "hidden_norm.weight")  # [20, H]

probes = {}
with torch.no_grad():
    # encode:memory 前 19 行 → 逐层 prefix k/v
    prefix_kv = []
    for li in range(NL):
        p = f"layers.{li}."
        k = lin(p + "self_attn.k_proj", mem[:19])
        v = lin(p + "self_attn.v_proj", mem[:19])
        k = apply_rope(rms_plain(k.reshape(-1, HD), p + "self_attn.k_norm.weight").reshape(19, NKV * HD), pos_kv, HD, NKV)
        prefix_kv.append((k.reshape(19, NKV, HD), v.reshape(19, NKV, HD)))
    # 噪声块前向
    x = E[ids.long()].to(torch.float16).float()
    res = None
    for li in range(NL):
        p = f"layers.{li}."
        if res is None:
            res = x
            h = rms_add_one(x, p + "input_layernorm.weight")
        else:
            res = res + x
            h = rms_add_one(res, p + "input_layernorm.weight")
        # conv prepare(input 侧)
        delta = lin(p + "attention_conv.kernel_projection", h).reshape(BLOCK, 2, 2, G)
        base = raww(p + "attention_conv.base_kernel")
        cin = conv(h.reshape(BLOCK, G, H // G), delta[:, 0], base[0], BLOCK).reshape(BLOCK, H)
        probes[f"L{li}_conv_in"] = cin.clone()
        # attention(全可见;KV = prefix + self)
        q = rms_plain(lin(p + "self_attn.q_proj", cin).reshape(-1, HD), p + "self_attn.q_norm.weight").reshape(BLOCK, -1)
        ks = rms_plain(lin(p + "self_attn.k_proj", cin).reshape(-1, HD), p + "self_attn.k_norm.weight").reshape(BLOCK, -1)
        vs = lin(p + "self_attn.v_proj", cin)
        q = apply_rope(q, pos_noise, HD, NQ).reshape(BLOCK, NQ, HD)
        ks = apply_rope(ks, pos_noise, HD, NKV).reshape(BLOCK, NKV, HD)
        vs = vs.reshape(BLOCK, NKV, HD)
        pk, pv = prefix_kv[li]
        k_all = torch.cat([pk, ks], 0)  # [27, NKV, HD]
        v_all = torch.cat([pv, vs], 0)
        qh = q.permute(1, 0, 2)                                  # [NQ, B, HD]
        rep = NQ // NKV
        kh = k_all.reshape(27, NKV, 1, HD).expand(27, NKV, rep, HD).reshape(27, NQ, HD).permute(1, 0, 2)
        vh = v_all.reshape(27, NKV, 1, HD).expand(27, NKV, rep, HD).reshape(27, NQ, HD).permute(1, 0, 2)
        w = torch.einsum("qhd,qkd->hqk", qh, kh) * (HD ** -0.5)
        w = torch.softmax(w, -1)
        ao = torch.einsum("hqs,qsd->qhd", w, vh).permute(1, 0, 2).reshape(BLOCK, -1)
        araw = lin(p + "self_attn.o_proj", ao)
        probes[f"L{li}_attn_raw"] = araw.clone()
        # conv finish(output 侧;同一 delta 的 side=1 半)
        afin = conv(araw.reshape(BLOCK, G, H // G), delta[:, 1], base[1], BLOCK).reshape(BLOCK, H)
        probes[f"L{li}_attn_fin"] = afin.clone()
        res = res + afin
        n2 = rms_add_one(res, p + "post_attention_layernorm.weight")
        # mlp(conv 双包裹)
        delta3 = lin(p + "mlp_conv.kernel_projection", n2).reshape(BLOCK, 2, 2, G)
        mbase = raww(p + "mlp_conv.base_kernel")
        cin2 = conv(n2.reshape(BLOCK, G, H // G), delta3[:, 0], mbase[0], BLOCK).reshape(BLOCK, H)
        probes[f"L{li}_mlp_conv_in"] = cin2.clone()
        g = lin(p + "mlp.gate_proj", cin2)
        u = lin(p + "mlp.up_proj", cin2)
        mo = lin(p + "mlp.down_proj", torch.nn.functional.silu(g) * u)
        mfin = conv(mo.reshape(BLOCK, G, H // G), delta3[:, 1], mbase[1], BLOCK).reshape(BLOCK, H)
        probes[f"L{li}_mlp_out"] = mfin.clone()
        x = mfin
    hid = rms_add_one(res + x, "norm.weight")
    # f16 化再 walk(2026-10-07 定谳:f32 hidden 下 logits 近平局,在 f16
    # 引擎里朗塌成大量精确平局(topk 全 9264.0),walk 取序对 f32 参考
    # 不可达 —— drafts 键必须从 f16 hidden 派生,否则对 f16 引擎是假参考)
    hid = hid.half().float()
    probes["hidden"] = hid.clone()
    # selector:logits(hidden[1:]) top16 + 格打分 + walk
    logits = hid[1:] @ LM.T
    probes["logits_head"] = logits[:, :16].clone()
    topv, topi = logits.topk(TOPK, dim=-1)
    proj = lin("candidate_selector.hidden_projection", hid[1:])
    A, B = raww("candidate_selector.predecessor_codebook"), raww("candidate_selector.successor_codebook")
    toks, idx_ch = [], torch.zeros(0, dtype=torch.long)
    for e in range(7):
        pred_ids = torch.full((TOPK,), ids[0].item(), dtype=torch.long) if e == 0 else topi[e - 1]
        sc = topv[e][None, :] + ((A[pred_ids] * proj[e][None, :]) @ B[topi[e]].T)  # [K, K]
        pick = int(sc[0].argmax()) if e == 0 else int(sc[idx_ch[-1]].argmax())
        idx_ch = torch.cat([idx_ch, torch.tensor([pick])])
        toks.append(float(topi[e][pick]))
    probes["drafts"] = torch.tensor(toks, dtype=torch.float32)
    print("hidden |abs| mean:", float(hid.abs().mean()), "max:", float(hid.abs().max()))
    print("drafts:", [int(t) for t in toks])
    print("slot0 topk ids:", topi[0][:8].tolist(), "vals:", [round(float(v),2) for v in topv[0][:8]])
    print("logit scale:", float(logits.abs().mean()), float(logits.abs().max()))

from safetensors.torch import save_file
out = {k.replace(".", "_"): (v.to(torch.float16) if k != "drafts" else v) for k, v in probes.items()}
save_file(out, OUT)
print("golden saved:", OUT)
bad = []
for k, v in out.items():
    isnan = torch.isnan(v.float()).any().item()
    print(f"  {k} {tuple(v.shape)} nan={isnan}")
    if isnan: bad.append(k)
print("NaN keys:", bad if bad else "无")
