from __future__ import annotations

from pathlib import Path

import pytest

from owl_converter.errors import ConverterError
from owl_converter.source_profile import detect_source
from owl_converter.st_index import load_checkpoint

from conftest import write_safetensors


def _index(tmp_path: Path, tensors: dict[str, tuple[str, tuple[int, ...]]]):
    write_safetensors(tmp_path / "model.safetensors", tensors)
    return load_checkpoint(tmp_path)


def test_dense_bf16(tmp_path: Path) -> None:
    idx = _index(
        tmp_path,
        {
            "model.layers.0.q_proj.weight": ("BF16", (256, 128)),
            "model.layers.0.norm.weight": ("BF16", (128,)),
        },
    )
    profile = detect_source(idx, {"model_type": "qwen3_5"})
    assert profile.scheme == "dense-bf16"
    assert not profile.is_fp8


def test_fp8_per_tensor(tmp_path: Path) -> None:
    idx = _index(
        tmp_path,
        {
            "model.layers.0.q_proj.weight": ("F8_E4M3", (256, 128)),
            "model.layers.0.q_proj.weight_scale": ("F32", (1,)),
        },
    )
    profile = detect_source(idx, {})
    assert profile.scheme == "fp8-per-tensor"
    assert len(profile.scale_pairs) == 1
    assert profile.scale_pairs[0].kind == "per-tensor"
    assert not profile.scale_pairs[0].inverse


def test_fp8_blockwise(tmp_path: Path) -> None:
    idx = _index(
        tmp_path,
        {
            "model.layers.0.q_proj.weight": ("F8_E4M3", (256, 256)),
            "model.layers.0.q_proj.weight_scale_inv": ("F32", (2, 2)),
        },
    )
    profile = detect_source(
        idx, {"quantization_config": {"weight_block_size": [128, 128]}}
    )
    assert profile.scheme == "fp8-blockwise"
    assert profile.block_size == (128, 128)
    assert profile.scale_pairs[0].inverse


def test_fp8_mixed_scale_kinds(tmp_path: Path) -> None:
    idx = _index(
        tmp_path,
        {
            "a.weight": ("F8_E4M3", (256, 128)),
            "a.weight_scale": ("F32", (1,)),
            "b.weight": ("F8_E4M3", (256, 256)),
            "b.weight_scale_inv": ("F32", (2, 2)),
        },
    )
    profile = detect_source(idx, {})
    assert profile.scheme == "fp8-mixed"


def test_fp8_missing_scale_rejected(tmp_path: Path) -> None:
    idx = _index(tmp_path, {"a.weight": ("F8_E4M3", (256, 128))})
    with pytest.raises(ConverterError) as ei:
        detect_source(idx, {})
    assert ei.value.code == "missing-scale"


def test_fp8_unknown_scale_shape_rejected(tmp_path: Path) -> None:
    idx = _index(
        tmp_path,
        {
            "a.weight": ("F8_E4M3", (256, 128)),
            "a.weight_scale": ("F32", (7,)),
        },
    )
    with pytest.raises(ConverterError) as ei:
        detect_source(idx, {})
    assert ei.value.code == "unknown-scale-shape"


def test_fp8_fnuz_rejected(tmp_path: Path) -> None:
    idx = _index(tmp_path, {"a.weight": ("F8_E4M3FNUZ", (256, 128))})
    with pytest.raises(ConverterError) as ei:
        detect_source(idx, {})
    assert ei.value.code == "unsupported-fp8"


def test_pre_quantized_is_not_reconverted(tmp_path: Path) -> None:
    idx = _index(
        tmp_path,
        {
            "a.weight_packed": ("I32", (256, 16)),
            "a.weight_scale": ("BF16", (256, 4)),
    },
    )
    profile = detect_source(idx, {})
    assert profile.scheme == "pre-quantized-4bit"
