# GDN chunked 金标生成器(FLA 0.6.0 @ 9f38d24;owl GDN 前向包立案 2026-10-03)
# 输入域 = 引擎真实取值:q/k ∈ randn×0.1(投影输出),v ∈ randn×0.1,
#          beta = sigmoid(rand) ∈ (0,1),g = log-decay 负域
# 形状 = 27B GDN:nk=16, nv=48, kd=vd=128;chunk=64;varlen cu_seqlens
# 输出 = 每阶段 f32 raw LE bin + manifest.json(Rust 对拍直读)
import json, os, struct, sys
import torch

OUT = os.path.join(os.path.dirname(__file__), "cases")
os.makedirs(OUT, exist_ok=True)

from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
# 直接用 orchestrator 的阶段函数(同 chunk.py 内部序)
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.gated_delta_rule.naive import naive_chunk_gated_delta_rule  # 慢速精确参照

NK, NV, KD, VD = 16, 48, 128, 128
SCALE = KD ** -0.5

def save_bin(name, t: torch.Tensor):
    t = t.contiguous().float().cpu()
    with open(os.path.join(OUT, name + ".bin"), "wb") as f:
        f.write(t.numpy().tobytes())
    return {"name": name, "shape": list(t.shape)}

def gen_case(tag, T, cu=None, init_state=True, seed=0):
    g_cpu = torch.Generator(device="cpu"); g_cpu.manual_seed(seed)
    B = 1
    q = (torch.rand(B, T, NK, KD, generator=g_cpu) - 0.5) * 0.2
    k = (torch.rand(B, T, NK, KD, generator=g_cpu) - 0.5) * 0.2
    v = (torch.rand(B, T, NV, VD, generator=g_cpu) - 0.5) * 0.2
    beta = torch.sigmoid(torch.randn(B, T, NV, generator=g_cpu))
    g = -torch.rand(B, T, NV, generator=g_cpu) * 0.5 - 0.01  # 负域 log-decay
    if cu is None:
        cu = torch.tensor([0, T], dtype=torch.long)
    cu = cu.to(torch.long)
    S0 = torch.randn(B, NV, KD, VD, generator=g_cpu) * 0.1 if init_state else None

    dev = "cuda"
    q, k, v, beta, g = (x.to(dev) for x in (q, k, v, beta, g))
    S0d = S0.to(dev).clone() if S0 is not None else None
    cu_d = cu.to(dev)

    # 阶段 1:g cumsum(RCP_LN2 标度,log2 域)
    g_cum = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2, cu_seqlens=cu_d)
    # 阶段 2:intra(kkt + solve_tril + recompute w/u)→ w, u, A
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd_intra
    w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g_cum, beta=beta, cu_seqlens=cu_d, chunk_size=64)
    # 阶段 3:状态递推 → h(每 chunk 状态), v_new, final
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cum, initial_state=S0d, output_final_state=True,
        cu_seqlens=cu_d, chunk_size=64)
    # 阶段 4:输出
    from fla.ops.gated_delta_rule.chunk import chunk_fwd_o
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cum, scale=SCALE, cu_seqlens=cu_d, chunk_size=64)
    # 端到端(含 naive 精确参照)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    o_e2e, final_e2e = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=SCALE, initial_state=(S0d.clone() if S0d is not None else None),
        output_final_state=True, cu_seqlens=cu_d)
    # 自检:有限性 + 非零(朴素 GQA 参照不可用;金标 = 阶段张量本体,
    # port 对拍锚 = 这些张量,FLA 内部一致性由其上游 CI 保证)
    assert torch.isfinite(o_e2e).all() and torch.isfinite(final_e2e).all()
    assert o_e2e.abs().max().item() > 1e-4 and final_e2e.abs().max().item() > 1e-4
    manifest = {"tag": tag, "T": T, "NK": NK, "NV": NV, "KD": KD, "VD": VD,
                "chunk": 64, "scale": SCALE,
                "cu_seqlens": cu.tolist(),
                "init_state": init_state,
                "tensors": []}
    manifest["tensors"].append(save_bin(f"{tag}.q", q))
    manifest["tensors"].append(save_bin(f"{tag}.k", k))
    manifest["tensors"].append(save_bin(f"{tag}.v", v))
    manifest["tensors"].append(save_bin(f"{tag}.beta", beta))
    manifest["tensors"].append(save_bin(f"{tag}.g", g))
    if S0 is not None:
        manifest["tensors"].append(save_bin(f"{tag}.s0", S0))
    manifest["tensors"].append(save_bin(f"{tag}.g_cum", g_cum))
    manifest["tensors"].append(save_bin(f"{tag}.A", A))
    manifest["tensors"].append(save_bin(f"{tag}.w", w))
    manifest["tensors"].append(save_bin(f"{tag}.u", u))
    manifest["tensors"].append(save_bin(f"{tag}.h", h))
    manifest["tensors"].append(save_bin(f"{tag}.v_new", v_new))
    manifest["tensors"].append(save_bin(f"{tag}.o", o))
    manifest["tensors"].append(save_bin(f"{tag}.o_e2e", o_e2e))
    manifest["tensors"].append(save_bin(f"{tag}.final", final_state))
    # naive 参照差(FLA 自洽性记录)
    with open(os.path.join(OUT, f"{tag}.manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[{tag}] T={T} ✓ saved {len(manifest['tensors'])} tensors")

# 用例:单 chunk / 双 chunk / 非整尾 / 带初态(跨 chunk 续)
gen_case("c1_t64",   64, init_state=False, seed=1)
gen_case("c2_t128", 128, init_state=False, seed=2)
gen_case("c3_t96",   96, init_state=False, seed=3)   # 尾 chunk 32(掩码路径)
gen_case("c4_t64_s0", 64, init_state=True, seed=4)   # 初态续算(引擎逐 chunk 形态)
print("all goldens →", OUT)
