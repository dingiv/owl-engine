"""P0a reference spike: Qwen3.5-0.8B AWQ g32 via llm-compressor.

Kept separate from the GPTQModel spike because the two stacks pin different
transformers versions and both must be measured on the same calibration set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier
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
    r"re:.*linear_attn\.in_proj_[ab]$",
]


def load_calibration(tokenizer, samples: int) -> list[str]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{samples}]")
    return [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        for row in ds
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--pipeline", default="sequential")
    parser.add_argument("--seq-offload", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequential-targets", default=None)
    parser.add_argument("--n-grid", type=int, default=20)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    texts = load_calibration(tokenizer, args.samples)
    dataset = Dataset.from_list([{"text": text} for text in texts])

    if args.device != "auto":
        torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    model = AutoModelForImageTextToText.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
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
        QuantizationModifier(
            config_groups={"group_0": scheme},
            ignore=IGNORE,
        ),
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
        sequential_targets=args.sequential_targets.split(",") if args.sequential_targets else None,
        batch_size=args.batch_size,
        text_column="text",
    )
    quantized = time.perf_counter()

    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)
    saved = time.perf_counter()
    torch.cuda.synchronize()

    report = {
        "source": str(source),
        "output": str(output),
        "backend": "llm-compressor",
        "pipeline": args.pipeline,
        "seq_offload": args.seq_offload,
        "batch_size": args.batch_size,
        "sequential_targets": args.sequential_targets,
        "n_grid": args.n_grid,
        "samples": args.samples,
        "seq_len": args.seq_len,
        "load_seconds": round(loaded - started, 2),
        "quantize_seconds": round(quantized - loaded, 2),
        "save_seconds": round(saved - quantized, 2),
        "total_seconds": round(saved - started, 2),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (output / "spike-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
