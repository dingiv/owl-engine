"""Convert Qwen3.8-27B-pi-FP8 (blockwise FP8) to AWQ g32-asym CT.

The source is a compressed-tensors FP8 checkpoint (30GB). It does not fit on one
24G card, so this script:
  * loads the checkpoint on CPU via compressed-tensors auto_offload;
  * uses sequential pipeline + layerwise_decompression (FP8 -> BF16 only for the
    current decoder layer);
  * puts the activation cache on the second 3090 (cuda:1);
  * writes compressed-tensors pack-quantized, which owl AwqSource / vLLM Marlin
    can load.

Run:
  CUDA_VISIBLE_DEVICES=1,2 python scripts/convert_pi27b_awq.py \
    --source models/bytkim/Qwen3.8-27B-pi-FP8 \
    --output models/bytkim/Qwen3.8-27B-pi-AWQ-INT4
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier
from llmcompressor.utils import load_context
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from transformers import AutoModelForImageTextToText, AutoTokenizer


IGNORE = [
    "lm_head",
    r"re:.*visual.*",
    r"re:.*mtp.*",
    r"re:.*linear_attn\.in_proj_(a|b|ba)$",
]

COPY_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def load_calibration(tokenizer, samples: int) -> Dataset:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{samples}]")
    texts = [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        for row in ds
    ]
    return Dataset.from_list([{"text": text} for text in texts])


def copy_assets(source: Path, output: Path) -> None:
    for name in COPY_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-offload", default="cuda:1")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-grid", type=int, default=10)
    parser.add_argument("--cpu-memory", default="80GB")
    parser.add_argument("--pipeline", default="sequential")
    parser.add_argument("--targets-per-subgraph", type=int, default=2)
    parser.add_argument("--keep-mtp", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    dataset = load_calibration(tokenizer, args.samples)

    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    # auto_offload keeps the 30GB FP8 checkpoint on CPU; the sequential pipeline
    # materializes one decoder layer at a time. layerwise_decompression expands
    # FP8 blockwise weights only for the subgraph being calibrated.
    # Source MTP is FP8 and cannot be copied unchanged; skip it and copy the
    # raw mtp.safetensors later if the engine profile needs it.
    with load_context(model_cls=AutoModelForImageTextToText, load_mtp=False):
        model = AutoModelForImageTextToText.from_pretrained(
            source,
            device_map="auto_offload",
            max_memory={"cpu": args.cpu_memory},
            offload_folder=str(output / ".offload"),
            trust_remote_code=False,
        )
    loaded = time.perf_counter()

    weights = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=False,
        group_size=32,
        strategy=QuantizationStrategy.GROUP,
    )
    scheme = QuantizationScheme(targets=["Linear"], weights=weights)
    recipe = [
        AWQModifier(duo_scaling=True, n_grid=args.n_grid, offload_device=None),
        QuantizationModifier(config_groups={"group_0": scheme}, ignore=IGNORE),
    ]

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.samples,
        pipeline=args.pipeline,
        sequential_offload_device=args.seq_offload,
        sequential_targets_per_subgraph=args.targets_per_subgraph,
        batch_size=args.batch_size,
        layerwise_decompression=True,
        text_column="text",
    )
    quantized = time.perf_counter()

    # The source MTP is FP8; it cannot be copied unchanged. This conversion
    # targets the main backbone only. Declare no MTP so the save wrapper does
    # not try to validate/copy source-quantized MTP weights.
    if not args.keep_mtp:
        text_config = model.config.get_text_config()
        text_config.mtp_num_hidden_layers = 0
        if hasattr(model.config, "mtp_num_hidden_layers"):
            model.config.mtp_num_hidden_layers = 0

    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)
    copy_assets(source, output)
    saved = time.perf_counter()

    report = {
        "source": str(source),
        "output": str(output),
        "backend": "llm-compressor",
        "source_format": "fp8-blockwise-128x128",
        "scheme": {"bits": 4, "group_size": 32, "symmetric": False},
        "pipeline": args.pipeline,
        "layerwise_decompression": True,
        "seq_offload": args.seq_offload,
        "samples": args.samples,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "n_grid": args.n_grid,
        "targets_per_subgraph": args.targets_per_subgraph,
        "load_seconds": round(loaded - started, 2),
        "quantize_seconds": round(quantized - loaded, 2),
        "save_seconds": round(saved - quantized, 2),
        "total_seconds": round(saved - started, 2),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
