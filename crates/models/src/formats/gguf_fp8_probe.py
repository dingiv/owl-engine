#!/usr/bin/env python3
"""**GGUF ↔ FP8 原始检查点逐张量探针**(2026-10-19 gguf 轮;**dev 工具**,非 CI 门)。

为什么是 Python 一次性工具而不是 Rust 门:参照源 = `Qwen/Qwen3.8-27B-FP8`
(block-wise fp8 + `weight_scale_inv`),Rust 侧没有 fp8 解量化臂;而本探针的
**独立价值在"三源互不共享代码"**:

  ① 变换公式 = **llama.cpp 转换器写侧**(`conversion/qwen.py::_reorder_v_heads`
     等,本脚本抄写式实现)—— 不是本仓读侧的逆公式;
  ② 解量化 = `gguf-py` 的 `quants.dequantize` —— 不是本仓的 `dequant.rs`;
  ③ 参照值 = 原始 FP8 检查点(HF 官方)。

判据与容差:变换类张量(alpha/beta/A_log/dt/conv/norm)**应 ≤ 5e-3**;
量化类张量(Q4_K/Q6_K/Q8_0 一档)**2e-2 ~ 5e-2 = 两源量化噪声**;
**错置换/错段给 O(0.5–1.0)** ⇒ 判别力充分(实测 2026-10-19:全部落在噪声档,
即 27B 权重解释面**无实质错**;见设计 §14.4)。

跑法(任一有 torch 的 venv;本机用 sglang-test 的):

    ./packages/sglang-test/.venv/bin/python \
        crates/models/src/formats/gguf_fp8_probe.py

⚠️ 需要 ~30 GB 主机内存(fp8 分片按需载 + 大张量解量化);卡不需要。
"""
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, './repos/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.quants import dequantize

FP8 = Path('./models/Qwen/Qwen3.8-27B-FP8')
GGUF = Path('./models/ggml-org/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf')
IDX = json.load(open(FP8 / 'model.safetensors.index.json'))['weight_map']
NK, NV, HD = 16, 48, 128
VPR = NV // NK
_cache = {}

def hf(key):
    """HF 张量 → f32 numpy(应用 FP8 block-wise scale_inv)"""
    try:
        return _hf(key)
    except Exception as e:
        from safetensors.torch import load_file as _lf
        sd = _lf(str(FP8 / IDX[key]))
        t = sd[key]; sc = sd.get(key+'_scale_inv')
        print(f'  !! {key}: w{tuple(t.shape)} {t.dtype} scale={None if sc is None else tuple(sc.shape)} :: {e}')
        raise

def _hf(key):
    shard = IDX[key]
    if shard not in _cache:
        from safetensors.torch import load_file
        _cache.clear()
        _cache[shard] = load_file(str(FP8 / shard))
    sd = _cache[shard]
    t = sd[key]
    if t.dtype == torch.float8_e4m3fn:
        w = t.to(torch.float32)
        sc = sd.get(key + '_scale_inv')
        if sc is not None:
            sc = sc.to(torch.float32)
            bs0 = (w.shape[0] + sc.shape[0] - 1) // sc.shape[0]
            bs1 = (w.shape[1] + sc.shape[1] - 1) // sc.shape[1] if w.dim() > 1 else 1
            sc = sc.repeat_interleave(bs0, 0)
            if w.dim() > 1:
                sc = sc.repeat_interleave(bs1, 1)
            else:
                bs1 = 1
            sc = sc[: w.shape[0]] if w.dim() == 1 else sc[: w.shape[0], : w.shape[1]]
            w = w * sc
        return w.numpy()
    return t.to(torch.float32).numpy()

def reorder(t, dim, nk=NK, vpr=VPR, hd=HD):
    """转换器 `_reorder_v_heads`:grouped → tiled([nk,vpr,hd] → [vpr,nk,hd])。"""
    shp = list(t.shape)
    if dim < 0:
        dim += len(shp)
    new = shp[:dim] + [nk, vpr, hd] + shp[dim + 1:]
    t = t.reshape(*new)
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.transpose(*perm).reshape(*shp)

r = GGUFReader(str(GGUF))
def gq(name):
    t = next(x for x in r.tensors if x.name == name)
    data = t.data if not t.data.dtype == np.uint8 else t.data.view(np.uint8)
    out = dequantize(np.asarray(t.data).reshape(-1).view(np.uint8), t.tensor_type)
    inner = t.shape[0]
    return np.asarray(out, dtype=np.float32).reshape(-1, inner)

def rowseg(w, start_row):  # 仅 V 段行重排
    return np.concatenate([w[:start_row], reorder(w[start_row:], 0)], axis=0)

CASES = []
Q = 2*NK*HD; V = NV*HD; C = 2*Q + V   # conv 通道
for l in (0, 3):
    B = f'model.language_model.layers.{l}.'
    G = f'blk.{l}.'
    if l == 0:
        CASES += [
            (B+'linear_attn.in_proj_qkv.weight', G+'attn_qkv.weight', 2e-2, lambda w: rowseg(w, Q+V and 2*NK*HD)),  # 仅 V 段
            (B+'linear_attn.in_proj_z.weight',   G+'attn_gate.weight', 2e-2, lambda w: reorder(w, 0)),
            (B+'linear_attn.in_proj_a.weight',   G+'ssm_alpha.weight', 2e-2, lambda w: reorder(w, 0, hd=1)),
            (B+'linear_attn.in_proj_b.weight',   G+'ssm_beta.weight',  2e-2, lambda w: reorder(w, 0, hd=1)),
            (B+'linear_attn.A_log',              G+'ssm_a',            2e-2, lambda w: reorder(-np.exp(w), 0, hd=1)),
            (B+'linear_attn.dt_bias',            G+'ssm_dt.bias',      2e-2, lambda w: reorder(w, 0, hd=1)),
            (B+'linear_attn.conv1d.weight',      G+'ssm_conv1d.weight',2e-2,
             lambda w: (lambda s: np.concatenate([s[:2*NK*HD], reorder(s[2*NK*HD:], 0)]))(np.squeeze(w))[..., None]),
            (B+'linear_attn.norm.weight',        G+'ssm_norm.weight',  5e-3, lambda w: w),
            (B+'linear_attn.out_proj.weight',    G+'ssm_out.weight',   2e-2, lambda w: reorder(w, 1)),
        ]
    CASES += [
        (B+'input_layernorm.weight',         G+'attn_norm.weight',  5e-3, lambda w: w + 1),
        (B+'post_attention_layernorm.weight',G+'post_attention_norm.weight',5e-3, lambda w: w + 1),
        (B+'mlp.up_proj.weight',             G+'ffn_up.weight',     3e-2, lambda w: w),
        (B+'self_attn.q_proj.weight',        G+'attn_q.weight',     2e-2, lambda w: w),
        (B+'self_attn.k_proj.weight',        G+'attn_k.weight',     2e-2, lambda w: w),
        (B+'self_attn.v_proj.weight',        G+'attn_v.weight',     2e-2, lambda w: w),
        (B+'self_attn.o_proj.weight',        G+'attn_output.weight',3e-2, lambda w: w),
        (B+'self_attn.q_norm.weight',        G+'attn_q_norm.weight',5e-3, lambda w: w + 1),
        (B+'self_attn.k_norm.weight',        G+'attn_k_norm.weight',5e-3, lambda w: w + 1),
    ]
CASES += [
    ('model.language_model.embed_tokens.weight', 'token_embd.weight', 3e-2, lambda w: w),
    ('lm_head.weight', 'output.weight', 3e-2, lambda w: w),
    ('model.language_model.norm.weight', 'output_norm.weight', 5e-3, lambda w: w + 1),
]

bad = 0
for hk, gk, tol, xf in CASES:
    ref = xf(hf(hk)).reshape(-1).astype(np.float32)
    got = gq(gk).reshape(-1)
    if ref.shape != got.shape:
        print(f'  ✗ {gk}: 形状 {ref.shape} vs {got.shape}'); bad += 1; continue
    scale = max(float(np.abs(ref).max()), 1e-6)
    d = np.abs(ref - got) / scale
    i = int(d.argmax())
    flag = '✓' if d[i] <= tol else '✗'
    if d[i] > tol: bad += 1
    print(f'  {flag} {gk:28s} {str(tuple(got.shape)):>14s} max|Δ|/max|ref| = {d[i]:.3e} (tol {tol:.0e}) @ {i}')
print(f'\n{len(CASES)-bad}/{len(CASES)} 通过')
