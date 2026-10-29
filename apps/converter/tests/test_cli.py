from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from owl_converter.cli import main

from conftest import write_config, write_safetensors


def _run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


def _model(tmp_path: Path, *, truncate_data: int = 0) -> Path:
    write_config(tmp_path / "config.json", {"model_type": "qwen3_5"})
    write_safetensors(
        tmp_path / "model.safetensors",
        {
            "model.language_model.embed_tokens.weight": ("BF16", (1000, 128)),
            "model.language_model.layers.0.q_proj.weight": ("BF16", (256, 128)),
            "model.language_model.layers.0.norm.weight": ("BF16", (128,)),
        },
        truncate_data=truncate_data,
    )
    return tmp_path


def test_inspect_json_complete(tmp_path: Path) -> None:
    src = _model(tmp_path)
    code, out = _run(["inspect", str(src), "--json"])
    assert code == 0
    plan = json.loads(out)
    assert plan["schema"] == "owl-convert-plan/1"
    assert plan["status"] == "ok"
    assert plan["source"]["complete"] is True
    assert plan["inventory"]["quantizable"] == 1
    assert plan["output"]["scheme"] == "awq-g32"


def test_inspect_default_allows_incomplete(tmp_path: Path) -> None:
    src = _model(tmp_path, truncate_data=100)
    code, _ = _run(["inspect", str(src)])
    assert code == 0


def test_inspect_require_complete_returns_3(tmp_path: Path) -> None:
    src = _model(tmp_path, truncate_data=100)
    code, out = _run(["inspect", str(src), "--require-complete", "--json"])
    assert code == 3
    plan = json.loads(out)
    assert plan["status"] == "blocked"
    assert any(c["code"] == "incomplete-download" and c["level"] == "error" for c in plan["checks"])


def test_inspect_writes_plan(tmp_path: Path) -> None:
    src = _model(tmp_path)
    out_path = tmp_path / "plan.json"
    code, _ = _run(["inspect", str(src), "--out", str(out_path)])
    assert code == 0
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "ok"


def test_missing_path_returns_2(tmp_path: Path) -> None:
    code, _ = _run(["inspect", str(tmp_path / "nope")])
    assert code == 2


def test_stub_command_returns_64() -> None:
    code, _ = _run(["dequant", "whatever"])
    assert code == 64
