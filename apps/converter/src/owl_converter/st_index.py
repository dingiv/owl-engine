"""safetensors 索引：只读文件头，不加载张量。

设计要点：
- 支持**下载中**的检查点：头已到、数据未到 ⇒ `complete=False` + `missing_bytes`，
  不把“还在下”误报成“格式损坏”；
- 头损坏/截断 = 结构化错误（响亮拒），不猜；
- 多分片按 `model.safetensors.index.json` 的 weight_map 发现；没有索引则 glob。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .dtypes import DTYPE_BYTES
from .errors import ConverterError

# safetensors 头是 JSON；正常模型 < 100KB。给足余量，但拒绝病态长度。
MAX_HEADER_BYTES = 256 << 20


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int
    numel: int
    nbytes: int
    elem_bytes: int

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass
class Shard:
    path: Path
    header_bytes: int
    file_bytes: int
    data_end: int
    tensors: dict[str, TensorEntry] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.file_bytes >= 8 + self.header_bytes + self.data_end

    @property
    def missing_bytes(self) -> int:
        need = 8 + self.header_bytes + self.data_end
        return max(0, need - self.file_bytes)


@dataclass
class CheckpointIndex:
    root: Path
    shards: list[Shard]
    tensors: dict[str, tuple[TensorEntry, Shard]]
    problems: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.problems and all(s.complete for s in self.shards)

    @property
    def missing_bytes(self) -> int:
        return sum(s.missing_bytes for s in self.shards)

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t, _ in self.tensors.values())

    @property
    def total_numel(self) -> int:
        return sum(t.numel for t, _ in self.tensors.values())

    def dtype_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t, _ in self.tensors.values():
            counts[t.dtype] = counts.get(t.dtype, 0) + 1
        return dict(sorted(counts.items()))


def _read_json(path: Path) -> object:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConverterError(
            "read-file", f"读不了 {path}: {exc}", hint="检查路径/权限；下载中的文件可能还没落盘"
        ) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConverterError(
            "bad-json",
            f"{path} 不是合法 JSON: {exc}",
            hint="文件可能只下了一半；等下载完成或重新下载该文件",
        ) from exc


def read_shard(path: Path) -> Shard:
    """读一个 safetensors 文件的头；数据区可以还没下完。"""
    try:
        file_bytes = path.stat().st_size
    except OSError as exc:
        raise ConverterError("stat-shard", f"stat {path}: {exc}") from exc

    try:
        with path.open("rb") as fh:
            head = fh.read(8)
            if len(head) < 8:
                raise ConverterError(
                    "truncated-header",
                    f"{path} 不足 8 字节，头都没下完",
                    hint="等下载完成；或检查文件是否被截断",
                )
            (header_len,) = struct.unpack("<Q", head)
            if header_len == 0 or header_len > MAX_HEADER_BYTES:
                raise ConverterError(
                    "bad-header-len",
                    f"{path} 头长度 {header_len} 不合理（上限 {MAX_HEADER_BYTES}）",
                )
            if 8 + header_len > file_bytes:
                raise ConverterError(
                    "truncated-header",
                    f"{path} 头声明 {header_len}B，但文件只有 {file_bytes - 8}B",
                    hint="等下载完成；或检查文件是否被截断",
                )
            raw_header = fh.read(header_len)
    except OSError as exc:
        raise ConverterError("read-shard", f"读 {path}: {exc}") from exc

    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise ConverterError(
            "bad-header-json", f"{path} 头 JSON 解析失败: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise ConverterError("bad-header", f"{path} 头不是 JSON object")

    metadata_raw = header.pop("__metadata__", None)
    metadata = (
        {str(k): str(v) for k, v in metadata_raw.items()}
        if isinstance(metadata_raw, dict)
        else {}
    )

    shard = Shard(path=path, header_bytes=header_len, file_bytes=file_bytes, data_end=0)
    for name, info in header.items():
        if not isinstance(info, dict):
            raise ConverterError("bad-tensor", f"{path}: {name} 的描述不是 object")
        dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if not isinstance(dtype, str):
            raise ConverterError("bad-tensor", f"{path}: {name} 缺 dtype")
        if not isinstance(shape, list) or any(
            not isinstance(x, int) or x < 0 for x in shape
        ):
            raise ConverterError("bad-tensor", f"{path}: {name} 的 shape 非法")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(x, int) or x < 0 for x in offsets)
            or offsets[0] > offsets[1]
        ):
            raise ConverterError("bad-tensor", f"{path}: {name} 的 data_offsets 非法")

        numel = 1
        for dim in shape:
            numel *= dim
        start, end = offsets
        nbytes = end - start
        elem_bytes = DTYPE_BYTES.get(dtype, 0)
        if elem_bytes and numel * elem_bytes != nbytes:
            shard.warnings.append(
                f"{name}: dtype={dtype} shape={shape} 期望 {numel * elem_bytes}B，"
                f"实际 {nbytes}B（打包格式可忽略）"
            )
        entry = TensorEntry(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            start=start,
            end=end,
            numel=numel,
            nbytes=nbytes,
            elem_bytes=elem_bytes,
        )
        shard.tensors[name] = entry
        shard.data_end = max(shard.data_end, end)

    return shard


def _shard_paths_from_index(root: Path, index_path: Path) -> tuple[list[Path], list[str]]:
    raw = _read_json(index_path)
    if not isinstance(raw, dict):
        raise ConverterError("bad-index", f"{index_path} 不是 JSON object")
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ConverterError(
            "bad-index",
            f"{index_path} 缺非空 weight_map",
            hint="索引文件可能只下了一半；等下载完成",
        )
    names = sorted({str(v) for v in weight_map.values()})
    paths = [root / name for name in names]
    missing = [p.name for p in paths if not p.is_file()]
    return paths, [f"索引引用的分片未落盘: {name}" for name in missing]


def load_checkpoint(root: Path) -> CheckpointIndex:
    """打开目录/单文件；返回索引 + 完整性问题清单。"""
    root = root.expanduser()
    if not root.exists():
        raise ConverterError(
            "no-such-path", f"{root} 不存在", hint="给模型目录或单个 .safetensors 文件"
        )

    problems: list[str] = []
    if root.is_file():
        shard_paths = [root]
    elif root.is_dir():
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            shard_paths, index_problems = _shard_paths_from_index(root, index_path)
            problems.extend(index_problems)
        else:
            shard_paths = sorted(p for p in root.glob("*.safetensors") if p.is_file())
        if not shard_paths:
            raise ConverterError(
                "no-shards",
                f"{root} 下没有 .safetensors",
                hint="模型可能还在下载（先出现 config/tokenizer，权重后到）；稍后再试",
            )
    else:
        raise ConverterError("bad-path", f"{root} 既不是目录也不是文件")

    shards: list[Shard] = []
    for path in shard_paths:
        if not path.is_file():
            continue
        try:
            shards.append(read_shard(path))
        except ConverterError as exc:
            problems.append(f"{path.name}: {exc.message}")

    if not shards:
        detail = "; ".join(problems) if problems else "没有可读分片"
        raise ConverterError(
            "no-readable-shards",
            f"{root}: 没有可读的 safetensors 头（{detail}）",
            hint="等下载完成；或检查文件是否损坏",
        )

    tensors: dict[str, tuple[TensorEntry, Shard]] = {}
    for shard in shards:
        for name, entry in shard.tensors.items():
            if name in tensors:
                other = tensors[name][1].path.name
                raise ConverterError(
                    "duplicate-tensor",
                    f"{name} 同时出现在 {other} 与 {shard.path.name}",
                    hint="分片索引冲突；删掉多余分片或重新下载",
                )
            tensors[name] = (entry, shard)

    return CheckpointIndex(root=root, shards=shards, tensors=tensors, problems=problems)
