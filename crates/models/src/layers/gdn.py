#!/usr/bin/env python3
"""HF baseline:GDN 线性注意力核族(逐核小批移植;契约 op 分发)。

输入/输出张量与契约按 op 分发(契约 {"op": ...}):
  gating  : in {a_log [H], dt_bias [H], a [TH], b [TH]} / {"heads": H}
            out {g [TH], beta [TH]}
            g = -exp(A_log)·softplus(a+dt_bias)(HF Qwen3_5GatedDeltaNet 同式);
            beta = sigmoid(b)。
  l2norm  : in {x [rows·dim]} / {"rows": R, "dim": D, "eps": E}
            out {y [rows·dim]} = x · rsqrt(sum(x², dim) + eps)
            (HF l2norm 同式,qwen3_5 modeling 内嵌定义对齐 FLA)。
  conv_upd: in {x [B·d], w [d·4], state [S·d·3], slots [B]} /
            {"slots": B, "dim": d, "silu": 0|1}
            out {y [B·d], state [S·d·3](原地更新)}
            因果卷积 decode 槽更新:y = silu(x·w3 + Σ hist·w0..2),
            state[slot] 滑窗;slot<0 = padding 跳写。
  delta_dec: in {q/k [B·HK·K], v [B·HV·V], g/beta [B·HV],
                 state [S·HV·K·V], slots [B]} /
            {"batch": B, "nv": HV, "nk": HK, "kd": K, "vd": V}
            out {out [B·HV·V], state(原地更新)}
            单步门控 delta rule(HF torch_recurrent_gated_delta_rule 直跑;
            q 由 HF 内部除 √K,与核内 q_scale 同式;g log 空间)。
  norm_act : in {x [rows·vd], z [rows·vd], gamma [gs]} /
            {"rows": R, "value_dim": VD, "group_size": GS, "eps": E, "act": 0|1}
            out {y [rows·vd]} = rmsnorm(x 组内)·gamma · act(z)
            (HF Qwen3_5RMSNormGated 直跑;×w 非零中心)。
"""

import torch

from common import hf_import, run


def op_gating(t: dict, c: dict):
    heads = int(c["heads"])
    a_log, dt_bias = t["a_log"], t["dt_bias"]
    a, b = t["a"].reshape(-1, heads), t["b"].reshape(-1, heads)
    # 与旧世界 gdn_kernels fused_gating 同式(HF GatedDeltaNet 门控):
    # softplus 分支 torch 默认 threshold=20 = 核内 log1p(exp x) 分支一致
    g = -torch.exp(a_log) * torch.nn.functional.softplus(a + dt_bias)
    beta = torch.sigmoid(b)
    return {"g": g.reshape(-1), "beta": beta.reshape(-1)}, "torch softplus/sigmoid(Qwen3_5GatedDeltaNet 融合门控同式)"


def op_l2norm(t: dict, c: dict):
    rows, dim, eps = int(c["rows"]), int(c["dim"]), float(c["eps"])
    x = t["x"].reshape(rows, dim)
    # HF l2norm 同式(qwen3_5 modeling 内嵌定义,对齐 FLA)
    y = x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return {"y": y.reshape(-1)}, "torch rsqrt sum-sq(qwen3_5 l2norm 内嵌定义同式)"


def op_conv_upd(t: dict, c: dict):
    B, d, silu = int(c["slots"]), int(c["dim"]), bool(int(c.get("silu", 1)))
    x = t["x"].reshape(B, d)
    w = t["w"].reshape(d, 4)
    state = t["state"].reshape(-1, d, 3).clone()
    slots = t["slots"].reshape(B).to(torch.long)
    hist = state[slots]                     # [B, d, 3]
    y = x * w[:, 3] + (hist * w[:, :3]).sum(-1)
    if silu:
        y = torch.nn.functional.silu(y)
    state[slots] = torch.cat([hist[:, :, 1:], x.unsqueeze(-1)], dim=-1)
    return {"y": y.reshape(-1), "state": state.reshape(-1)}, "manual causal conv1d decode(HF causal_conv1d update 同式)"


def op_delta_dec(t: dict, c: dict):
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    B, HK, HV = int(c["batch"]), int(c["nk"]), int(c["nv"])
    kd, vd = int(c["kd"]), int(c["vd"])
    slots = t["slots"].reshape(B).to(torch.long)
    q = t["q"].reshape(B, 1, HK, kd)
    k = t["k"].reshape(B, 1, HK, kd)
    v = t["v"].reshape(B, 1, HV, vd)
    g = t["g"].reshape(B, 1, HV)
    beta = t["beta"].reshape(B, 1, HV)
    state = t["state"].reshape(-1, HV, kd, vd).clone()
    out, final = m.torch_recurrent_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=state[slots], output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    state[slots] = final
    return {"out": out.reshape(-1), "state": state.reshape(-1)}, f"{m.__name__}.torch_recurrent_gated_delta_rule"


def op_norm_act(t: dict, c: dict):
    rows, vd, gs = int(c["rows"]), int(c["value_dim"]), int(c["group_size"])
    eps, act = float(c["eps"]), int(c.get("act", 0))
    cls, prov = hf_import([
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNormGated"),
    ])
    x = t["x"].reshape(rows, vd)
    z = t["z"].reshape(rows, vd)
    if cls is not None:
        assert act == 0, "HF Qwen3_5RMSNormGated 恒 silu;act=1 无 HF 金标准"
        layer = cls(gs, eps=eps).to(torch.float32)
        with torch.no_grad():
            layer.weight.copy_(t["gamma"])
        # HF GatedDeltaNet 调用形态:core_attn_out.reshape(-1, head_v_dim)
        # → norm 末维 = group_size(per-(row, v_head) 归一,与核同构)
        y = layer(x.reshape(-1, gs), z.reshape(-1, gs)).reshape(-1)
    else:
        var = x.reshape(rows, -1, gs).pow(2).mean(-1, keepdim=True)
        nx = x.reshape(rows, -1, gs) * torch.rsqrt(var + eps)
        nx = nx * t["gamma"]
        gate = torch.nn.functional.silu(z.reshape(rows, -1, gs)) if act == 0 \
            else torch.sigmoid(z.reshape(rows, -1, gs))
        y = (nx * gate).reshape(rows, vd)
        prov += "+manual"
    return {"y": y.reshape(-1)}, prov


OPS = {
    "gating": op_gating,
    "l2norm": op_l2norm,
    "conv_upd": op_conv_upd,
    "delta_dec": op_delta_dec,
    "norm_act": op_norm_act,
}


def test(t: dict, c: dict):
    return OPS[c["op"]](t, c)


if __name__ == "__main__":
    run(test)
