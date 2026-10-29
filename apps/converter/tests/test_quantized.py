from __future__ import annotations

import json
from pathlib import Path

import pytest

from owl_converter.errors import ConverterError
from owl_converter.quantized import normalize, read_info

from conftest import write_config, write_safetensors


def _config(group: str, *, group_size: int, symmetric: bool) -> dict:
    return {
        "model_type": "qwen3_5",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                group: {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "group_size": group_size,
                        "symmetric": symmetric,
                        "strategy": "group",
                    },
                }
            },
        },
    }


def _sym_model(tmp_path: Path, group: str = "W4A16") -> Path:
    write_config(tmp_path / "config.json", _config(group, group_size=128, symmetric=True))
    write_safetensors(
        tmp_path / "model.safetensors",
        {
            "a.weight_packed": ("I32", (256, 32)),  # in = 256
            "a.weight_scale": ("BF16", (256, 2)),
            "a.weight_shape": ("I64", (2,)),
        },
    )
    return tmp_path


def test_read_info() -> None:
    info = read_info(_config("W4A16", group_size=128, symmetric=True))
    assert info is not None
    assert info.group_name == "W4A16"
    assert info.group.owl_scheme == "w4a16-g128"


def test_normalize_renames_group(tmp_path: Path) -> None:
    src = _sym_model(tmp_path)
    report = normalize(src)
    assert report["changed"] is True
    assert report["group_from"] == "W4A16"
    assert report["group_to"] == "group_0"
    assert report["scheme"] == "w4a16-g128"
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    assert list(cfg["quantization_config"]["config_groups"]) == ["group_0"]
    # 幂等
    again = normalize(src)
    assert again["changed"] is False


def test_normalize_dry_run_keeps_file(tmp_path: Path) -> None:
    src = _sym_model(tmp_path)
    before = (src / "config.json").read_text(encoding="utf-8")
    report = normalize(src, dry_run=True)
    assert report["changed"] is True
    assert (src / "config.json").read_text(encoding="utf-8") == before


def test_unsupported_scheme_rejected(tmp_path: Path) -> None:
    src = _sym_model(tmp_path)
    cfg = _config("W4A16", group_size=128, symmetric=False)
    write_config(src / "config.json", cfg)
    with pytest.raises(ConverterError) as ei:
        normalize(src)
    assert ei.value.code == "unsupported-scheme"


def test_missing_scale_rejected(tmp_path: Path) -> None:
    write_config(
        tmp_path / "config.json", _config("W4A16", group_size=128, symmetric=True)
    )
    write_safetensors(
        tmp_path / "model.safetensors",
        {"a.weight_packed": ("I32", (256, 32))},
    )
    with pytest.raises(ConverterError) as ei:
        normalize(tmp_path)
    assert ei.value.code == "bad-layout"


def test_asymmetric_g32_accepts_zp(tmp_path: Path) -> None:
    write_config(
        tmp_path / "config.json", _config("group_0", group_size=32, symmetric=False)
    )
    write_safetensors(
        tmp_path / "model.safetensors",
        {
            "a.weight_packed": ("I32", (256, 32)),
            "a.weight_scale": ("BF16", (256, 8)),
            "a.weight_zero_point": ("I32", (32, 8)),
            "a.weight_shape": ("I64", (2,)),
        },
    )
    report = normalize(tmp_path)
    assert report["scheme"] == "awq-g32"
    assert report["counts"]["zero_point_layers"] == 1
