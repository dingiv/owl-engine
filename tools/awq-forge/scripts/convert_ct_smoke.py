"""Convert the 0.8B AWQ GEMM spike output to compressed-tensors pack-quantized."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from compressed_tensors.entrypoints.convert import AutoAWQConverter, convert_checkpoint


IGNORE = [
    "lm_head",
    r"re:^model\.visual\..*",
    r"re:^visual\..*",
    r"re:^mtp\..*",
    r"re:.*\.linear_attn\.in_proj_[ab]$",
]


def normalize_config(path: Path) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    quant = config["quantization_config"]
    groups = quant.get("config_groups") or {}
    if "config_group_0" in groups and "group_0" not in groups:
        quant["config_groups"] = {"group_0": groups["config_group_0"]}
    weights = quant["config_groups"]["group_0"]["weights"]
    weights.setdefault("actorder", None)
    weights.setdefault("block_structure", None)
    weights.setdefault("observer", "mse")
    weights.setdefault("observer_kwargs", {})
    weights.setdefault("scale_dtype", None)
    weights.setdefault("zp_dtype", "torch.int8")
    quant.setdefault("global_compression_ratio", None)
    quant.setdefault("sparsity_config", {})
    quant.setdefault("transform_config", {})
    quant["ignore"] = IGNORE
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)

    converter = AutoAWQConverter(
        bits=4,
        group_size=32,
        zero_point=True,
        version="gemm",
        ignore=IGNORE,
    )
    convert_checkpoint(source, output, converter=converter, device="cpu", max_workers=1)
    normalize_config(output / "config.json")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
