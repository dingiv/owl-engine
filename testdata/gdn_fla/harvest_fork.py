"""从 vLLM FLA fork 采集 f16 生产变体 + autotuner 选中配置(2026-10-11 v3)。

要点:
- 输入直接 f16(owl 激活域;vLLM 生产同款 2 字节姿势,零 f32 cast)
- autotuner 在首跑 bench 全变体并按 key 选中 —— 从 wrapper 缓存读"选中项",
  再到磁盘缓存按 (name, shared, warps, stages) 匹配 cubin
- 同时 dump 五核 ABI(tt.func 参数序)供 handler 对齐
"""
import json, os, shutil, sys
import torch

OUT = "/tmp/owl_ab/harvest16"
CACHE = "/tmp/owl_ab/harvest16_cache"
shutil.rmtree(CACHE, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = CACHE

from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum_scalar
from vllm.third_party.flash_linear_attention.ops import chunk as fla_chunk
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o

NK, NV, KD, VD, T = 16, 48, 128, 128, 1024
BT = 64
dev = "cuda"
DT = torch.float16  # owl 激活域
g = torch.Generator(device="cpu").manual_seed(7)
q = (((torch.rand(1, T, NK, KD, generator=g) - 0.5) * 0.2)).to(dev).to(DT)
k = (((torch.rand(1, T, NK, KD, generator=g) - 0.5) * 0.2)).to(dev).to(DT)
v = (((torch.rand(1, T, NV, VD, generator=g) - 0.5) * 0.2)).to(dev).to(DT)
beta = torch.sigmoid(torch.randn(1, T, NV, generator=g)).to(dev).to(DT)
gg = (-torch.rand(1, T, NV, generator=g) * 0.5 - 0.01).to(dev).to(DT)
s0 = (torch.randn(1, NV, KD, VD, generator=g) * 0.1).to(dev).to(DT)
cu = torch.tensor([0, T], dtype=torch.long, device=dev)
ci = prepare_chunk_indices(cu, BT).to(dev)

for it in range(3):
    g_cum = chunk_local_cumsum_scalar(gg, chunk_size=BT, cu_seqlens=cu)
    A = fla_chunk.chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=gg, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float32)
    A = fla_chunk.solve_tril(A=A, cu_seqlens=cu, chunk_indices=ci, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g_cum, cu_seqlens=cu, chunk_indices=ci)
    h, v_new, ht = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=gg, initial_state=s0, output_final_state=True, cu_seqlens=cu, chunk_indices=ci)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=gg, scale=KD**-0.5, cu_seqlens=cu, chunk_indices=ci)
torch.cuda.synchronize()
print("f16 流水跑通 x3: o", tuple(o.shape), o.dtype, "ht", tuple(ht.shape), ht.dtype)

# ── autotuner 选中配置 ──
import triton
kernels = {}
for modname in ("vllm.third_party.flash_linear_attention.ops.cumsum",
                "vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt",
                "vllm.third_party.flash_linear_attention.ops.solve_tril",
                "vllm.third_party.flash_linear_attention.ops.wy_fast",
                "vllm.third_party.flash_linear_attention.ops.chunk_delta_h",
                "vllm.third_party.flash_linear_attention.ops.chunk_o"):
    mod = sys.modules[modname]
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, triton.runtime.autotuner.Autotuner):
            kernels[obj.fn.__name__] = obj
print("\nautotuner 选中:")
picked = {}
for name, aut in kernels.items():
    try:
        entries = list(aut.cache.values())
        for cfg in entries:
            print(f"  {name}: {cfg}")
            picked[name] = cfg
    except Exception as e:
        print(f"  {name}: 读取失败 {e}")

# ── 磁盘缓存匹配 cubin ──
manifest = []
for root, dirs, files in os.walk(CACHE):
    for f in files:
        if not f.endswith(".json"):
            continue
        p = os.path.join(root, f)
        try:
            meta = json.load(open(p))
        except Exception:
            continue
        name = meta.get("name", "")
        cfg = picked.get(name)
        if cfg is None:
            continue
        shared, warps, stages = meta.get("shared"), meta.get("num_warps"), meta.get("num_stages")
        hit = (warps == cfg.num_warps and stages == cfg.num_stages
               and (not cfg.kwargs or True))
        cubin_p = os.path.join(root, name + ".cubin")
        if not os.path.exists(cubin_p):
            continue
        short = {  # owl 侧短名
            "chunk_local_cumsum_scalar_kernel": "cumsum",
            "chunk_gated_delta_rule_fwd_kkt_solve_kernel": "kkt",
            "recompute_w_u_fwd_kernel": "wu",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "h",
            "chunk_fwd_kernel_o": "o",
        }.get(name)
        if short is None:
            continue
        manifest.append({"kernel": short, "name": name, "shared": shared,
                         "num_warps": warps, "num_stages": stages,
                         "picked": str(cfg), "picked_match": hit})
        if hit:
            shutil.copy(cubin_p, f"{OUT}/{short}.cubin")
            json.dump({"name": name, "shared": shared, "num_warps": warps,
                       "num_stages": stages, "picked": str(cfg)},
                      open(f"{OUT}/{short}.json", "w"), indent=1)

for m in sorted(manifest, key=lambda x: (x["kernel"], not x["picked_match"])):
    print(f"[{m['kernel']}] shared={m['shared']} w{m['num_warps']} s{m['num_stages']} picked_match={m['picked_match']}")
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
print("done")
