"""FLA vendor 副本的 import 垫片(2026-10-13)。

原文件从 vllm 取两个东西:
  - `vllm.triton_utils` → 实为 `import triton` / `import triton.language as tl` /
    `triton.language.extra.libdevice as tldevice` 的**薄再导出**(见 vllm 源码);
  - `vllm.platforms`(`current_platform.is_cuda_alike()`)与
    `vllm.platforms.rocm.on_gfx1x` —— 平台判定。

本垫片给出**等价最小实现**,使 vendor 副本**不依赖 vllm**(owl 树内自足)。
语义不变:owl 只跑 CUDA/Ampere 路径。
"""

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


class _Platform:
    """`vllm.platforms.current_platform` 的最小替身(owl = CUDA-only)。"""

    @staticmethod
    def is_cuda_alike() -> bool:
        return True

    @staticmethod
    def is_rocm() -> bool:
        return False


current_platform = _Platform()


def on_gfx1x() -> bool:
    """`vllm.platforms.rocm.on_gfx1x` 的替身:owl 非 ROCm ⇒ 恒 False。"""
    return False
