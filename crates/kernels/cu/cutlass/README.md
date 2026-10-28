# CUTLASS 4.8.0 — shared header family (sm86 closure)

**Upstream:** [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass), workspace
`repos/cutlass` commit `78e46228` (v4.8.0), BSD-3-Clause. See `LICENSE.txt`.

This is a **separate kernel-family directory**, not a subdirectory of any one
operator vendor. FlashAttention-2 (`../flash_attn`) includes it via:

```text
-Icrates/kernels/cu/cutlass/include
```

## Scope

The directory contains only the **95-header compile closure** required by the
current FA2 forward adapter (1.9 MiB), not the full 28 MiB CUTLASS tree.
`closure.txt` is the exact list. It is **sm86-only** on purpose: the FA2 adapter
instantiates only the SM80 MMA atom and SM75/SM80 copy atoms used by
`run_mha_fwd_hdim64/96`.

Two trim passes:

1. **146 → 107**: removed 33 SM90+/SM100/SM107/SM120 headers plus 6 headers
   reached only through those paths.
2. **107 → 95**: removed SM61/SM70/SM89 MMA atom pairs (6 files), unused
   algorithms (`axpby`, `cooperative_copy`, `cooperative_gemm`, `prefetch`),
   and the two print utilities (`print_tensor`, `print_latex`).

`owl_fa2.o` SASS is identical before/after both passes.

## Local changes (trim-only, no semantic edits)

- `include/cute/atom/mma_atom.hpp` — keep SM75/SM80 traits; drop SM61/SM70/
  SM89 and all SM90+ traits.
- `include/cute/atom/copy_atom.hpp` — keep SM50/SM75/SM80 copy traits; drop
  the SM90/SM100 trait and TMA includes.
- `include/cute/tensor.hpp` — keep only the algorithms FA2 calls:
  `tensor_algorithms`, `clear`, `copy`, `gemm`, plus `swizzle_layout`
  (needed for `composition(Swizzle, Layout)`). `fill/prefer` remain because
  `clear.hpp` includes `fill.hpp`.

## Left for a later pass

- `cutlass/layout/**` (7 headers) plus `matrix_coord/pitch_linear_coord/
  tensor_coord` are still in the closure because the FROZEN FA2
  `kernel_traits.h` includes `<cutlass/layout/layout.h>`. FA2 itself never
  references `cutlass::layout::*`; removing that single include would drop
  ~150 KiB but requires editing a FROZEN vendor header, so it is not part of
  this trim.

If a future kernel needs Hopper/Blackwell/SM120 atoms or the removed
algorithms, restore the include blocks from the pinned CUTLASS revision,
re-vendor the required headers, and regenerate `closure.txt` with `nvcc -M`
under the same include paths/flags. The full CUTLASS tests/examples/tools are
intentionally not vendored.
