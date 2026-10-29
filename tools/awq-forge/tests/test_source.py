import json
import struct

from awq_forge.source import inspect_source


def _write_checkpoint(root):
    header = {
        "__metadata__": {"format": "pt"},
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": {
            "dtype": "BF16",
            "shape": [1024, 1024],
            "data_offsets": [0, 0],
        },
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": {
            "dtype": "BF16",
            "shape": [16, 1024],
            "data_offsets": [0, 0],
        },
        "model.visual.blocks.0.attn.qkv.weight": {
            "dtype": "BF16",
            "shape": [768, 768],
            "data_offsets": [0, 0],
        },
        "mtp.fc.weight": {
            "dtype": "BF16",
            "shape": [1024, 1024],
            "data_offsets": [0, 0],
        },
    }
    encoded = json.dumps(header).encode("utf-8")
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    (root / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"})
    )


def test_inspect_source_classifies_and_ignores(tmp_path):
    _write_checkpoint(tmp_path)
    info = inspect_source(tmp_path)
    assert info.tensor_count == 4
    assert info.target_count == 1
    assert info.target_roles == {"gdn": 1}
    assert info.ignored["gdn-small-proj"] == 1
    assert info.ignored["vision"] == 1
    assert info.ignored["mtp"] == 1
    assert info.warnings == []


def test_inspect_packed_awq_targets(tmp_path):
    import json, struct
    header = {
        "__metadata__": {"format": "pt"},
        "model.language_model.layers.0.mlp.down_proj.weight_packed": {
            "dtype": "I32", "shape": [1024, 128], "data_offsets": [0, 0],
        },
        "model.language_model.layers.0.mlp.down_proj.weight_scale": {
            "dtype": "BF16", "shape": [1024, 32], "data_offsets": [0, 0],
        },
    }
    enc = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(enc)) + enc)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    info = inspect_source(tmp_path)
    assert info.target_count == 1
    assert info.target_roles == {"mlp": 1}
