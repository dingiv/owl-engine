# GDN chunked 深谷新金标生成器(c5;2026-10-11 M1 补课)
#
# 动机:g 语义案(补录Ⅷ/Ⅸ)教训#2 —— c1-c4 金标 g 域 [−0.51,−0.01],
#       引擎真实域深谷 −24;对拍覆盖必须含输入域极值(机器门)。
# 权威:vLLM third_party FLA fork 的**阶段函数**(与 AOT cubin 同源同变体);
#       配方 = 引擎生产配方(2026-10-11 定谳):q/k/v/beta = bf16,
#       g = f32(先 f16 量化锁定引擎输入),A/Ai = f32(engine a_buf 同款),
#       h0/ht = f32(engine state 池同款)。
# ⚠️ g 语义律(补录Ⅸ):kkt/solve_tril/wu/h/o 的 "g" 参数全部吃 **cumsum
#    产物** —— 本脚本 g_cum 生成后喂 gcum,严禁 raw g(史前 harvest_fork
#    的同错已定谳,勿复制)。
# 输出:testdata/gdn_fla/cases/c5_t128_deep.{bin,json}(schema 同 gen.py)
import json, os, sys
import torch

sys.path.insert(0, "/home/div/Documents/codes/packages/vllm")
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum_scalar
from vllm.third_party.flash_linear_attention.ops import chunk as fla_chunk
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")
TAG, T, BT = "c5_t128_deep", 128, 64
NK, NV, KD, VD = 16, 48, 128, 128
SCALE = KD ** -0.5
dev = "cuda"
DT = torch.bfloat16  # 引擎配方:q/k/v/beta bf16

torch.manual_seed(20261011)
q = ((torch.rand(1, T, NK, KD) - 0.5) * 0.2)
k = ((torch.rand(1, T, NK, KD) - 0.5) * 0.2)
v = ((torch.rand(1, T, NV, VD) - 0.5) * 0.2)
beta = torch.sigmoid(torch.randn(1, T, NV))
# 深谘 g:90% 温和域 U(−0.5,−0.01) + 10% 深谷 U(−24,−12)——层 0-4 实测
# 分布(−8~−24,补录Ⅷ)的包络;f16 量化锁定(引擎路径 g 恒经 f16)
g = torch.where(
    torch.rand(1, T, NV) < 0.10,
    -(torch.rand(1, T, NV) * 12.0 + 12.0),
    -(torch.rand(1, T, NV) * 0.5 + 0.01),
).half().float()
s0 = torch.randn(1, NV, KD, VD) * 0.1
s0_feed = s0.transpose(-1, -2).contiguous()  # [V,K] = fork h 核 h0 期望布局(引擎 kv_to_vk 同款)
cu = torch.tensor([0, T], dtype=torch.long)
assert torch.isfinite(g).all() and (g <= 0).all(), "g 必须全负(语义律)"

q, k, v, beta, g, s0_feed = (x.to(dev) for x in (q, k, v, beta, g, s0_feed))
q, k, v, beta = (x.to(DT) for x in (q, k, v, beta))  # bf16 铸造(引擎同款)
cu_d = cu.to(dev)
ci = prepare_chunk_indices(cu_d, BT).to(dev)

# 阶段 1:cumsum(g → gcum)
g_cum = chunk_local_cumsum_scalar(g, chunk_size=BT, cu_seqlens=cu_d)
assert (g_cum <= 1e-6).all(), "g_cum 全负律违反(语义律哨兵)"
# 阶段 2:kkt → solve_tril(⚠️ output_dtype=k.dtype=bf16 = 生产链同款:
# engine merge cubin 写 bf16 字节进 ai_buf(字节池),wu 读 bf16 ——
# mod.rs realloc 里 ai_buf 的 f32 视角是字节池标注,核域真实 dtype = bf16)
A = fla_chunk.chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g_cum, cu_seqlens=cu_d,
                                       chunk_indices=ci, output_dtype=torch.float32)
Ai = fla_chunk.solve_tril(A=A, cu_seqlens=cu_d, chunk_indices=ci, output_dtype=k.dtype)
# 阶段 3:wu(吃 Ai + gcum)
w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=Ai, g_cumsum=g_cum,
                         cu_seqlens=cu_d, chunk_indices=ci)
# 阶段 4:h(h0=f32 = engine state 池同款)
h, v_new, ht = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g_cum,
                                            initial_state=s0_feed.clone(), output_final_state=True,
                                            cu_seqlens=cu_d, chunk_indices=ci)
# 阶段 5:o
o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cum, scale=SCALE,
                cu_seqlens=cu_d, chunk_indices=ci)
torch.cuda.synchronize()

for name, t in [("g_cum", g_cum), ("A", Ai), ("o", o), ("ht", ht)]:
    assert torch.isfinite(t).all(), f"{name} 含非有限值(深谷下崩溃,配方/语义有错)"
print(f"[c5] g 域 [{g.min().item():.2f}, {g.max().item():.2f}] "
      f"g_cum 域 [{g_cum.min().item():.2f}, {g_cum.max().item():.2f}] "
      f"o absmax {o.abs().max().item():.4f} ht absmax {ht.abs().max().item():.4f}")

def save(name, t):
    t = t.contiguous().float().cpu()
    with open(os.path.join(OUT, name + ".bin"), "wb") as f:
        f.write(t.numpy().tobytes())
    return {"name": name, "shape": list(t.shape)}

manifest = {"tag": TAG, "T": T, "NK": NK, "NV": NV, "KD": KD, "VD": VD,
            "chunk": BT, "scale": SCALE, "cu_seqlens": cu.tolist(),
            "init_state": True, "recipe": "fork-bf16+g-f32(引擎生产配方)",
            "tensors": []}
for name, t in [(f"{TAG}.q", q), (f"{TAG}.k", k), (f"{TAG}.v", v),
                (f"{TAG}.beta", beta), (f"{TAG}.g", g), (f"{TAG}.s0", s0),  # s0 落盘仍为池 [K,V] 惯例
                (f"{TAG}.g_cum", g_cum), (f"{TAG}.A", Ai), (f"{TAG}.w", w),
                (f"{TAG}.u", u), (f"{TAG}.h", h), (f"{TAG}.v_new", v_new),
                (f"{TAG}.o", o), (f"{TAG}.final", ht)]:
    manifest["tensors"].append(save(name, t))
with open(os.path.join(OUT, f"{TAG}.manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print(f"[c5] 写入 {len(manifest['tensors'])} 张量 @ {OUT}")
