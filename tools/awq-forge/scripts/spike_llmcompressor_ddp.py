"""P0a DDP spike: data-parallel AWQ calibration across N GPUs.

This is the supported multi-GPU path in llm-compressor: every rank holds one
model replica, processes a disjoint calibration shard, and AWQ all-reduces
activation sums / partial MSE before smoothing. It is NOT tensor parallel and
NOT layer split.

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/spike_llmcompressor_ddp.py --source ... --output ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier
from llmcompressor.utils import load_context
from compressed_tensors.offload import init_dist
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


def load_calibration(tokenizer, samples: int) -> Dataset:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{samples}]")
    texts = [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        for row in ds
    ]
    return Dataset.from_list([{"text": text} for text in texts])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pipeline", default="independent")
    parser.add_argument("--n-grid", type=int, default=10)
    parser.add_argument("--seq-offload", default="cpu")
    parser.add_argument("--cpu-memory", default="12GB")
    args = parser.parse_args()

    init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if rank == 0 and output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    dataset = load_calibration(tokenizer, args.samples)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    with load_context(model_cls=AutoModelForImageTextToText):
        model = AutoModelForImageTextToText.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            device_map="auto_offload",
            max_memory={"cpu": args.cpu_memory},
            offload_folder=str(output.parent / f".offload-rank{rank}"),
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
        batch_size=args.batch_size,
        text_column="text",
    )
    quantized = time.perf_counter()

    if rank == 0:
        model.save_pretrained(output, save_compressed=True)
        tokenizer.save_pretrained(output)
        for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
            src = source / name
            if src.exists():
                (output / name).write_bytes(src.read_bytes())
    saved = time.perf_counter()

    report = {
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "backend": "llm-compressor-ddp",
        "samples": args.samples,
        "samples_per_rank": args.samples // world,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "n_grid": args.n_grid,
        "seq_offload": args.seq_offload,
        "pipeline": args.pipeline,
        "load_seconds": round(loaded - started, 2),
        "quantize_seconds": round(quantized - loaded, 2),
        "save_seconds": round(saved - quantized, 2) if rank == 0 else None,
        "total_seconds": round(saved - started, 2),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    if rank == 0:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        (output / "spike-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
