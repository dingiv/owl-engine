"""BF16 -> W4A16 g32-asym RTN，导出 CT `pack-quantized`（owl `format=awq` 布局）。

与 `spike_llmcompressor.py` 的唯一差别：**不做 AWQ 的 scale 搜索**（RTN 直落）。
用途 = 速度 A/B 的对照臂 —— owl 的 NVFP4 臂本身就是装载期 RTN，
故 "RTN-g32-int4 vs RTN-nvfp4" 是同算法、不同格式的严格对照（布局与内核
才是被测变量）。要质量数字时把 `--mode awq` 打开跑同一份脚本。

输出布局与 cyankiwi/Qwen3.8-27B-AWQ-INT4 同族：`.weight_packed`(I32)、
`.weight_scale`(BF16)、`.weight_zero_point`(I32，沿 out 打包)。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier
from transformers import AutoModelForImageTextToText, AutoTokenizer

# 与 convert_ct_smoke.py / spike_llmcompressor.py 逐字一致：
# lm_head 不量化；视觉塔、MTP 头、GDN 的 in_proj_a/b（n=32，marlin 不eligible）保持 f16
IGNORE = [
    "lm_head",
    r"re:.*visual.*",
    r"re:.*mtp.*",
    r"re:.*linear_attn\.in_proj_[ab]$",
]

CALIB = "HuggingFaceH4/ultrachat_200k"


# 离线兜底校准集（RTN 只观察**权重**、不看激活 ⇒ 文本内容不参与任何数值）。
# 本机 huggingface.co 不可达（401）时用它，避免转换被网络卡死。
FALLBACK_TEXT = (
    "量化是把浮点权重映射到低比特整数域的离线变换。分组量化把一个输入维度切成长度 "
    "为 group_size 的段，每段共享一个缩放系数；对称量化没有零点是 q ∈ [-7,7]，"
    "非对称量化带零点、把零点一并烘焙进整数域，因此同样的比特数下动态范围更贴合"
    "各段实际分布。块缩放浮点格式则保留浮点指数，用 e2m1 表示尾数，用 e4m3 表示"
    "每十六个元素一个的缩放系数，好处是既能吃到原生低精度张量核，又不必担心整数"
    "域的零点重排。推理时的瓶颈由此分成两半：decode 阶段每个 token 都要把全部权重"
    "读一遍，于是只看每权重的比特数；prefill 阶段同一份权重被整批 token 复用，于是"
    "看张量核的峰值算力。The quick brown fox jumps over the lazy dog. "
)


def load_calibration(tokenizer, samples: int) -> list[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset(CALIB, split=f"train_sft[:{samples}]")
        return [
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            for row in ds
        ]
    except Exception as e:  # noqa: BLE001 - 网络/鉴权都归到这里
        print(
            f"[warn] 校准集不可达（{type(e).__name__}: {e}）⇒ 用离线兜底文本",
            flush=True,
        )
        return [FALLBACK_TEXT * 2] * samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", choices=["rtn", "awq"], default="rtn")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-grid", type=int, default=10)
    args = ap.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    texts = load_calibration(tokenizer, args.samples)
    dataset = Dataset.from_list([{"text": t} for t in texts])

    if args.device != "cpu":
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    model = AutoModelForImageTextToText.from_pretrained(
        source,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=False,
    )
    t_load = time.perf_counter()

    weights = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=False,
        group_size=32,
        strategy=QuantizationStrategy.GROUP,
    )
    scheme = QuantizationScheme(targets=["Linear"], weights=weights)
    recipe = [QuantizationModifier(config_groups={"group_0": scheme}, ignore=IGNORE)]
    if args.mode == "awq":
        recipe.insert(0, AWQModifier(duo_scaling=True, n_grid=args.n_grid))

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_context_length_length=args.seq_len,
        num_calibration_samples=args.samples,
        pipeline="sequential",
        sequential_offload_device="cpu",
        batch_size=1,
        text_column="text",
    )
    t_quant = time.perf_counter()

    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)
    torch.cuda.synchronize()
    t_save = time.perf_counter()

    # 产物字节账（= 显存面，A/B 的"每 token 权重字节"口径）
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    wbytes = sum(p.stat().st_size for p in output.glob("*.safetensors"))
    report = {
        "source": str(source),
        "output": str(output),
        "mode": args.mode,
        "group_size": 32,
        "symmetric": False,
        "num_bits": 4,
        "samples": args.samples,
        "seq_len": args.seq_len,
        "load_s": round(t_load - t0, 1),
        "quant_s": round(t_quant - t_load, 1),
        "save_s": round(t_save - t_quant, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if torch.cuda.is_available()
        else 0.0,
        "output_bytes": total,
        "weights_bytes": wbytes,
        "weights_gb": round(wbytes / 2**30, 3),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (output / "convert_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
