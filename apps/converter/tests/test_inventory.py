from __future__ import annotations

from pathlib import Path

from owl_converter.inventory import build_inventory
from owl_converter.st_index import load_checkpoint

from conftest import write_safetensors


def _index(tmp_path: Path):
    write_safetensors(
        tmp_path / "model.safetensors",
        {
            "model.language_model.embed_tokens.weight": ("BF16", (1000, 128)),
            "lm_head.weight": ("BF16", (1000, 128)),
            "model.visual.blocks.0.attn.qkv.weight": ("BF16", (256, 128)),
            "model.language_model.layers.0.input_layernorm.weight": ("BF16", (128,)),
            "model.language_model.layers.0.q_proj.weight": ("BF16", (256, 128)),
            "model.language_model.layers.0.down_proj.weight": ("BF16", (256, 256)),
            "model.language_model.layers.0.linear_attn.in_proj_a.weight": (
                "BF16",
                (32, 128),
            ),
            "model.language_model.layers.0.linear_attn.conv1d.weight": (
                "BF16",
                (128, 1, 4),
            ),
            "model.language_model.layers.0.some3d.weight": ("BF16", (2, 2, 2)),
            "model.language_model.layers.0.linear_attn.A_log": ("BF16", (32,)),
            "mtp.fc.weight": ("BF16", (256, 128)),
        },
    )
    return load_checkpoint(tmp_path)


def test_inventory_filters_and_counts(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    inv = build_inventory(idx, "awq-g32")
    names = {c.name for c in inv.candidates}
    assert names == {
        "model.language_model.layers.0.q_proj.weight",
        "model.language_model.layers.0.down_proj.weight",
    }
    assert inv.skipped_counts["embedding"] == 1
    assert inv.skipped_counts["head"] == 1
    assert inv.skipped_counts["vision"] == 1
    assert inv.skipped_counts["norm"] == 1
    assert inv.skipped_counts["marlin-ineligible"] == 1  # in_proj_a 32x128
    assert inv.skipped_counts["non-linear"] == 1  # conv1d
    assert inv.skipped_counts["rank!=2"] == 1  # some3d
    assert inv.skipped_counts["not-weight"] == 1  # A_log
    assert inv.skipped_counts["mtp"] == 1


def test_q4_byte_estimate_awq_g32(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    inv = build_inventory(idx, "awq-g32")
    # q_proj 256x128: packed 16384 + scale 256*(128/32)*2=2048 + zp 256*4/2=512
    # down_proj 256x256: packed 32768 + scale 4096 + zp 1024
    assert inv.q4_bytes == 16384 + 2048 + 512 + 32768 + 4096 + 1024
    assert inv.symmetric is False
    assert inv.group == 32


def test_q4_byte_estimate_g128_sym(tmp_path: Path) -> None:
    idx = _index(tmp_path)
    inv = build_inventory(idx, "w4a16-g128")
    # q_proj: 16384 + 256*(128/128)*2=512；down_proj: 32768 + 256*(256/128)*2=1024
    assert inv.q4_bytes == 16384 + 512 + 32768 + 1024
    assert inv.symmetric is True
    assert inv.group == 128
