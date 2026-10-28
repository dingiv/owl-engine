# vLLM FlashAttention-2 forward (vendored)

**Upstream:** vLLM in-tree `vllm-flash-attn` (`csrc/flash_attn/src` only),
taken from this workspace's vLLM checkout at commit `e20bccc642` (vLLM `dev`).
CUTLASS is **not** taken from vLLM; it is the separate shared family
`crates/kernels/cu/cutlass/` (workspace CUTLASS 4.8.0).

**License:** BSD-3-Clause (Tri Dao / vLLM contributors). See `vendor/LICENSE`
(FlashAttention). CUTLASS lives in the separate shared family
`../cutlass/` (see its README/LICENSE).

**Trimmed closure:** the vendor is **not** the full CUTLASS tree. It contains only
the 16-file FlashAttention/c10 closure of this directory (`closure.txt`).
CUTLASS is a **separate shared family** at `crates/kernels/cu/cutlass/`
(95-header sm86 closure, 1.9 MiB). Regenerate both closure lists with `nvcc -M`
(same include paths/flags) after changing the adapter or upstream source.

**Class:** FROZEN. The kernel/body is **verbatim upstream**; only the host C ABI
adapter `owl_fa2.cu` (ours) and the `compat/` c10 CUDA-check shim are local.

## What is built

`owl_fa2.cu` includes the upstream `flash_fwd_launch_template.h` and calls:

```cpp
run_mha_fwd_hdim96<cutlass::half_t, false>(params, stream);
```

This is exactly vLLM's Qwen3.5-ViT path for `head_dim=72`:

- `params.d = 72`, `params.d_rounded = 96` (`round_multiple(72, 32)`);
- `HEADDIM_SWITCH(72)` selects `kHeadDim=96`;
- `Is_even_K=false` makes the kernel load/store only the first 72 columns;
- `causal=false`, `p_dropout=1.0`, no alibi/softcap/local/split-kv.

No Python-side head padding. The adapter only fills `Flash_fwd_params` and
allocates nothing; `lse` and `cu_seqlens` scratch are owned by owl's runtime.

## Build flags

`-O3 -std=c++17 -arch=sm_86 --expt-relaxed-constexpr` plus:
`-DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_SOFTCAP
 -DFLASHATTENTION_DISABLE_LOCAL -DFLASHATTENTION_DISABLE_ALIBI`.

The source closure is compiled at build time (no prebuilt `.a` in git); the
incremental gate lives in `crates/kernels/build.rs`.

## Contract

```text
q/k/v/out: [total_q, heads, 72] f16 contiguous
cu_seqlens: int32 [batch+1], patch-level
lse: f32 [heads, total_q] (unpadded)
max_seqlen: max patches
scale: 1/sqrt(72)
causal: false
```

owl currently routes only single-image (`n_seq==1`) hd=72 to this path; multi-image
remains native until the model side grows a multi-image `cu_seqlens` contract.
