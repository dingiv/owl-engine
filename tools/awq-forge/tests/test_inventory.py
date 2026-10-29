from awq_forge.inventory import build_targets, classify_weight, ignored_reason


def test_classify_full_attention():
    assert classify_weight("model.language_model.layers.3.self_attn.q_proj.weight") == "full-attention"
    assert classify_weight("model.language_model.layers.3.self_attn.o_proj.weight") == "full-attention"


def test_classify_gdn_and_mlp():
    assert classify_weight("model.language_model.layers.0.linear_attn.in_proj_qkv.weight") == "gdn"
    assert classify_weight("model.language_model.layers.0.linear_attn.in_proj_z.weight") == "gdn"
    assert classify_weight("model.language_model.layers.0.linear_attn.out_proj.weight") == "gdn"
    assert classify_weight("model.language_model.layers.0.mlp.down_proj.weight") == "mlp"


def test_ignore_small_gdn_and_non_text():
    assert ignored_reason("model.language_model.layers.0.linear_attn.in_proj_a.weight") == "gdn-small-proj"
    assert ignored_reason("model.language_model.layers.0.linear_attn.in_proj_b.weight") == "gdn-small-proj"
    assert ignored_reason("model.visual.blocks.0.attn.qkv.weight") == "vision"
    assert ignored_reason("mtp.fc.weight") == "mtp"
    assert ignored_reason("lm_head.weight") == "lm-head"


def test_build_targets_counts_qwen35_08b_shape():
    tensors = {}
    for layer in range(24):
        if layer % 4 == 3:
            for suffix in ("q_proj", "k_proj", "v_proj", "o_proj"):
                tensors[f"model.language_model.layers.{layer}.self_attn.{suffix}.weight"] = (1024, 1024)
        else:
            for suffix in ("in_proj_qkv", "in_proj_z", "out_proj"):
                tensors[f"model.language_model.layers.{layer}.linear_attn.{suffix}.weight"] = (1024, 1024)
        for suffix in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"model.language_model.layers.{layer}.mlp.{suffix}.weight"] = (1024, 1024)

    targets = build_targets(tensors)
    assert len(targets) == 150
    assert sum(t.role == "full-attention" for t in targets) == 24
    assert sum(t.role == "gdn" for t in targets) == 54
    assert sum(t.role == "mlp" for t in targets) == 72
    assert all(t.packed_shape == (1024, 128) for t in targets)
