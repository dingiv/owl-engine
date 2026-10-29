from __future__ import annotations

from pathlib import Path

import pytest

from owl_converter.errors import ConverterError
from owl_converter.st_index import load_checkpoint, read_shard

from conftest import write_config, write_safetensors


def test_reads_complete_shard(tmp_path: Path) -> None:
    p = tmp_path / "model.safetensors"
    write_safetensors(p, {"a.weight": ("BF16", (2, 3)), "b.weight": ("F32", (4,))})
    shard = read_shard(p)
    assert shard.complete
    assert set(shard.tensors) == {"a.weight", "b.weight"}
    assert shard.tensors["a.weight"].numel == 6
    assert shard.tensors["a.weight"].nbytes == 12


def test_detects_incomplete_download(tmp_path: Path) -> None:
    p = tmp_path / "model.safetensors"
    write_safetensors(p, {"a.weight": ("BF16", (16, 16))}, truncate_data=100)
    shard = read_shard(p)
    assert not shard.complete
    assert shard.missing_bytes == 100


def test_index_json_discovers_shards(tmp_path: Path) -> None:
    write_safetensors(tmp_path / "m-00001.safetensors", {"a.weight": ("BF16", (2, 2))})
    write_safetensors(tmp_path / "m-00002.safetensors", {"b.weight": ("BF16", (2, 2))})
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{},"weight_map":{"a.weight":"m-00001.safetensors",'
        '"b.weight":"m-00002.safetensors"}}',
        encoding="utf-8",
    )
    idx = load_checkpoint(tmp_path)
    assert len(idx.shards) == 2
    assert idx.complete


def test_missing_shard_is_a_problem(tmp_path: Path) -> None:
    write_safetensors(tmp_path / "m-00001.safetensors", {"a.weight": ("BF16", (2, 2))})
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{},"weight_map":{"a.weight":"m-00001.safetensors",'
        '"b.weight":"m-00002.safetensors"}}',
        encoding="utf-8",
    )
    idx = load_checkpoint(tmp_path)
    assert not idx.complete
    assert any("m-00002" in p for p in idx.problems)


def test_duplicate_tensor_rejected(tmp_path: Path) -> None:
    write_safetensors(tmp_path / "a.safetensors", {"x.weight": ("BF16", (2, 2))})
    write_safetensors(tmp_path / "b.safetensors", {"x.weight": ("BF16", (2, 2))})
    with pytest.raises(ConverterError) as ei:
        load_checkpoint(tmp_path)
    assert ei.value.code == "duplicate-tensor"


def test_truncated_header_rejected(tmp_path: Path) -> None:
    p = tmp_path / "model.safetensors"
    p.write_bytes(b"\x01\x00")
    with pytest.raises(ConverterError) as ei:
        read_shard(p)
    assert ei.value.code == "truncated-header"


def test_config_not_required_for_index(tmp_path: Path) -> None:
    write_safetensors(tmp_path / "model.safetensors", {"a.weight": ("BF16", (2, 2))})
    write_config(tmp_path / "config.json", {"model_type": "qwen3_5"})
    idx = load_checkpoint(tmp_path)
    assert idx.root == tmp_path
