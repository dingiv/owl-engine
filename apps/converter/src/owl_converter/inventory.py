"""Linear 清单 + 4bit 体积估算（不读数据区）。

判据与 owl 侧保持一致：
- `marlin_eligible` = out % 256 == 0 且 in % 128 == 0；
- AWQ g32：packed + scale(bf16) + zero_point(4bit packed)；
- W4A16 g128：packed + scale(bf16)，无 zp。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ConverterError
from .st_index import CheckpointIndex

SCHEMES = {
    "awq-g32": {"group": 32, "symmetric": False},
    "w4a16-g128": {"group": 128, "symmetric": True},
}

# 形态/角色面：这些权重保持 F16（owl 侧也通常是 F16 直读）
_NAME_SKIPS: tuple[tuple[str, str], ...] = (
    ("embed_tokens", "embedding"),
    ("lm_head", "head"),
    ("visual", "vision"),
    ("vision_tower", "vision"),
    ("mtp.", "mtp"),
    ("router", "router"),
    ("a_log", "non-linear"),
    ("dt_bias", "non-linear"),
    ("conv1d", "non-linear"),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    out_dim: int
    in_dim: int
    numel: int
    source_dtype: str
    source_bytes: int


@dataclass
class Inventory:
    scheme: str
    group: int
    symmetric: bool
    candidates: list[Candidate] = field(default_factory=list)
    skipped_counts: dict[str, int] = field(default_factory=dict)
    skipped_examples: dict[str, list[str]] = field(default_factory=dict)
    eligible_numel: int = 0
    total_numel: int = 0
    source_bytes: int = 0
    q4_bytes: int = 0
    keep_bytes: int = 0
    dense_intermediate_bytes: int = 0

    @property
    def final_bytes(self) -> int:
        return self.q4_bytes + self.keep_bytes

    @property
    def skipped_total(self) -> int:
        return sum(self.skipped_counts.values())


def _skip_reason(name: str, rank: int) -> str | None:
    lower = name.lower()
    if not name.endswith(".weight"):
        return "not-weight"
    if "norm" in lower:
        return "norm"
    for needle, reason in _NAME_SKIPS:
        if needle in lower:
            return reason
    if rank != 2:
        return "rank!=2"
    return None


def _q4_bytes(out_dim: int, in_dim: int, group: int, symmetric: bool) -> int:
    groups = in_dim // group
    packed = out_dim * in_dim // 2
    scale = out_dim * groups * 2  # bf16
    zp = 0 if symmetric else out_dim * groups // 2  # 4bit packed
    return packed + scale + zp


def build_inventory(index: CheckpointIndex, scheme: str) -> Inventory:
    spec = SCHEMES.get(scheme)
    if spec is None:
        raise ConverterError(
            "bad-scheme",
            f"未知输出方案 {scheme}",
            hint="P0 支持：" + " / ".join(SCHEMES),
        )
    group = int(spec["group"])
    symmetric = bool(spec["symmetric"])
    inv = Inventory(scheme=scheme, group=group, symmetric=symmetric)

    for name, (entry, _shard) in index.tensors.items():
        inv.total_numel += entry.numel
        inv.source_bytes += entry.nbytes
        reason = _skip_reason(name, entry.rank)
        if reason is not None:
            inv.skipped_counts[reason] = inv.skipped_counts.get(reason, 0) + 1
            inv.skipped_examples.setdefault(reason, [])
            if len(inv.skipped_examples[reason]) < 3:
                inv.skipped_examples[reason].append(name)
            continue

        out_dim, in_dim = entry.shape
        if out_dim % 256 != 0 or in_dim % 128 != 0:
            reason = "marlin-ineligible"
            inv.skipped_counts[reason] = inv.skipped_counts.get(reason, 0) + 1
            inv.skipped_examples.setdefault(reason, [])
            if len(inv.skipped_examples[reason]) < 3:
                inv.skipped_examples[reason].append(
                    f"{name} ({out_dim}x{in_dim})"
                )
            continue

        cand = Candidate(
            name=name,
            out_dim=out_dim,
            in_dim=in_dim,
            numel=entry.numel,
            source_dtype=entry.dtype,
            source_bytes=entry.nbytes,
        )
        inv.candidates.append(cand)
        inv.eligible_numel += cand.numel
        inv.q4_bytes += _q4_bytes(out_dim, in_dim, group, symmetric)

    inv.keep_bytes = (inv.total_numel - inv.eligible_numel) * 2  # BF16 保留件
    inv.dense_intermediate_bytes = inv.total_numel * 2  # FP8 反量化后的 BF16 中间态
    return inv
