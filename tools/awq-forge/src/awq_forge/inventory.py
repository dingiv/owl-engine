"""Pure tensor-name classification for Qwen3.5/3.8 checkpoints.

This module deliberately knows nothing about torch. It works on names and
shapes from safetensors headers, so it can run before the quant backend is
loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


TEXT_LAYER_PREFIXES = (
    "model.language_model.layers.",
    "model.layers.",
)

FULL_ATTENTION_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
)

GDN_SUFFIXES = (
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.out_proj.weight",
)

MLP_SUFFIXES = (
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


@dataclass(frozen=True)
class Target:
    name: str
    role: str
    out_features: int
    in_features: int

    @property
    def packed_shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features // 8

    @property
    def scale_shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features // 32

    @property
    def zero_point_shape(self) -> tuple[int, int]:
        return self.out_features // 8, self.in_features // 32


def is_text_layer_weight(name: str) -> bool:
    return name.endswith(".weight") and name.startswith(TEXT_LAYER_PREFIXES)


def classify_weight(name: str) -> str | None:
    """Return full-attention / gdn / mlp for a quantizable Linear weight."""
    if not is_text_layer_weight(name):
        return None
    if name.endswith(FULL_ATTENTION_SUFFIXES):
        return "full-attention"
    if name.endswith(GDN_SUFFIXES):
        return "gdn"
    if name.endswith(MLP_SUFFIXES):
        return "mlp"
    return None


def ignored_reason(name: str) -> str | None:
    """Explain why a tensor is not an AWQ W4A16 target."""
    if not name.endswith(".weight"):
        return "not-a-weight"
    if name == "lm_head.weight":
        return "lm-head"
    if name.endswith("embed_tokens.weight"):
        return "embedding"
    if ".visual." in name or name.startswith("visual."):
        return "vision"
    if name.startswith("mtp.") or ".mtp." in name:
        return "mtp"
    if ".linear_attn.in_proj_a." in name or ".linear_attn.in_proj_b." in name:
        return "gdn-small-proj"
    if ".linear_attn.conv1d." in name:
        return "gdn-conv"
    if "norm" in name.lower():
        return "norm"
    if not is_text_layer_weight(name):
        return "outside-text-layers"
    return None


def build_targets(tensors: dict[str, tuple[int, ...]]) -> list[Target]:
    targets: list[Target] = []
    for name, shape in tensors.items():
        role = classify_weight(name)
        if role is None:
            continue
        if len(shape) != 2:
            continue
        out_features, in_features = shape
        if out_features % 8 != 0 or in_features % 32 != 0:
            # AWQ g32 + packed int32 needs these alignments. Report, don't guess.
            continue
        targets.append(Target(name, role, out_features, in_features))
    return sorted(targets, key=lambda t: t.name)


def summarize_targets(targets: Iterable[Target]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in targets:
        counts[target.role] = counts.get(target.role, 0) + 1
    return dict(sorted(counts.items()))
