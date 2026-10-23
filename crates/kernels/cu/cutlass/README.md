# CUTLASS 4.8.0 — shared header family

**Upstream:** [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass), workspace
`repos/cutlass` commit `78e46228` (v4.8.0), BSD-3-Clause. See `LICENSE.txt`.

This is a **separate kernel-family directory**, not a subdirectory of any one
operator vendor. FlashAttention-2 (`../flash_attn`) and future CUTLASS-based
kernels include it via:

```text
-Icrates/kernels/cu/cutlass/include
```

## Scope

The directory contains only the **146-header compile closure** required by the
current FA2 forward adapter (6.2 MiB), not the full 28 MiB CUTLASS tree.
`closure.txt` is the exact list.

If a future kernel needs more CUTLASS/cute headers, add the required files from
the same pinned revision and regenerate `closure.txt` with `nvcc -M` under the
same include paths/flags. The full CUTLASS tests/examples/tools are intentionally
not vendored.
