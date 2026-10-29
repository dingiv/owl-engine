"""P0a spike: Qwen3.5-0.8B BF16 -> AWQ g32-asym via GPTQModel.

This is intentionally a script, not package API. It measures wall time, peak
VRAM, and the produced AutoAWQ GEMM key shapes before the CT exporter exists.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel
from gptqmodel.quantization import AWQConfig, FORMAT, METHOD
from transformers import AutoTokenizer


def patch_qwen35_awq_mapping() -> None:
    """Fix GPTQModel 7.7 Qwen3.5 GDN AWQ scaling groups.

    Upstream puts in_proj_qkv and in_proj_z in separate groups and lets
    out_proj join the AWQ scale search. Both are physically wrong for GDN:
    qkv/z share input_layernorm, while out_proj follows conv1d + gated delta
    rule and has no linear predecessor. AutoAWQ fork groups qkv+z and leaves
    out_proj as plain RTN.
    """
    from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel

    Qwen3_5QModel.module_tree_overrides = {
        METHOD.AWQ: [
            {
                "linear_attn": (
                    "norm:!",
                    "conv1d:!",
                    "in_proj_qkv:0:in=x",
                    "in_proj_z:0:in=x",
                    "in_proj_b:!:1",
                    "in_proj_a:!:1",
                    "out_proj:2",
                )
            }
        ]
    }

    original = Qwen3_5QModel.awq_get_modules_for_scaling

    def patched(self, module, input_feat, module_kwargs):
        nodes = original(self, module, input_feat, module_kwargs)
        linear_attn = getattr(module, "linear_attn", None)
        out_proj = getattr(linear_attn, "out_proj", None)
        if out_proj is None:
            return nodes
        return [node for node in nodes if out_proj not in (node.get("layers") or [])]

    Qwen3_5QModel.awq_get_modules_for_scaling = patched


def load_calibration(tokenizer, samples: int, seq_len: int) -> list[str]:
    try:
        ds = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split=f"train_sft[:{samples}]",
        )
        texts = []
        for row in ds:
            text = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        if texts:
            return texts
    except Exception as exc:
        print(f"calibration download failed: {exc}; using local fallback")

    fallback = Path(
        "/home/div/Documents/codes/packages/mistral.rs/calibration_data/"
        "calibration_datav3_small.txt"
    )
    text = fallback.read_text(encoding="utf-8")
    # Keep rows long enough to survive GPTQModel's min-length filter.
    rows = [row.strip() for row in text.splitlines() if len(row.strip()) > 200]
    if not rows:
        raise RuntimeError(f"no fallback calibration rows in {fallback}")
    return [rows[i % len(rows)] for i in range(samples)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--patch-qwen35-awq", action="store_true")
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-offload", action="store_true")
    args = parser.parse_args()

    if args.patch_qwen35_awq:
        patch_qwen35_awq_mapping()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    calibration = load_calibration(tokenizer, args.samples, args.seq_len)
    print(f"calibration rows: {len(calibration)}")

    target = torch.device(args.device)
    torch.cuda.set_device(target)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def only_target_device(devices):
        # GPTQModel expands the requested device to every CUDA device of the
        # same type. Pin it back to the one card we actually reserved.
        return [device for device in devices if device == target]

    qcfg = AWQConfig(
        bits=4,
        group_size=32,
        sym=False,
        format=FORMAT.GEMM,
        true_sequential=True,
        scale_search_chunked_activations=True,
        auto_forward_data_parallel=not args.single_gpu,
        compute_device_filter=only_target_device if args.single_gpu else None,
        offload_to_disk=not args.no_offload,
    )
    model = GPTQModel.load(str(source), quantize_config=qcfg, device=args.device)
    loaded = time.perf_counter()
    print(f"loaded in {loaded - started:.1f}s: {type(model.model).__name__}")

    model.quantize(
        calibration,
        batch_size=args.batch_size,
        calibration_concat_size=args.seq_len,
        calibration_sort="desc",
    )
    quantized = time.perf_counter()
    print(f"quantized in {quantized - loaded:.1f}s")

    model.save(str(output))
    saved = time.perf_counter()
    torch.cuda.synchronize()

    report = {
        "source": str(source),
        "output": str(output),
        "device": args.device,
        "samples": len(calibration),
        "batch_size": args.batch_size,
        "single_gpu": args.single_gpu,
        "offload_to_disk": not args.no_offload,
        "seq_len": args.seq_len,
        "load_seconds": round(loaded - started, 2),
        "quantize_seconds": round(quantized - loaded, 2),
        "save_seconds": round(saved - quantized, 2),
        "total_seconds": round(saved - started, 2),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "model_class": type(model.model).__name__,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (output / "spike-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
