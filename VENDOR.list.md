# VENDOR map — third-party dependencies and reference sources

> **Authoritative inventory** of third-party code *inside this repository*.
> Read these two first before changing dependencies or adding a vendor:
> - [`crates/backends/README.md`](crates/backends/README.md) — **CUDA admission rule**
>   (any CUDA-touching library must be source-reviewed for untracked allocations;
>   if it calls `cudaMalloc` behind your back, don't use it — or copy it in and fix it);
> - [`docs/devlog/canon/charter.md`](docs/devlog/canon/charter.md) A4 — the four ownership classes.
>
> 2026-10-21: this file used to be a placeholder; it is now materialized.

## 1. Ownership classes (from charter A4)

| Class | Meaning | Requirement here |
|---|---|---|
| **FROZEN** | Read-only snapshot of upstream code; we do not change its semantics | Keep verbatim + upstream copyright headers + record any in-tree changes |
| **OWNED** | Code we drive and evolve (may originate upstream, but is ours now) | Free to change; still keep upstream attribution |
| **REFERENCE** | Used for comparison/spec only, not built | Not committed; cite the source in docs |
| **BINARY** | Prebuilt binaries | Not committed by default; exceptions must be registered with a reproduction recipe |

## 2. Vendored code (participates in the build)

| Path | Upstream | Pinned | License | Class | In-tree changes |
|---|---|---|---|---|---|
| `crates/kernels/cu/flashinfer/include/**` (22 files) | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | nightly-v0.6.18-20260819-317-gfdc56343 | **Apache-2.0** | **FROZEN** | Only the paged-prefill header closure (trimmed, no cutlass); closure list in [`cu/flashinfer/README.md`](crates/kernels/cu/flashinfer/README.md) + `closure.txt` |
| `crates/kernels/cu/xqa/**` (16 files) | [NVIDIA XQA](https://github.com/flashinfer-ai/flashinfer) via avesed/vllm-ampere-optimized @ f2e081d (Ampere closure) | f2e081d | **Apache-2.0** | **FROZEN** | Removed TVM-FFI binding and prebuilt `.a`; host launcher moved to Rust — only device sources remain. See [README](crates/kernels/cu/xqa/README.md) |
| `crates/kernels/cu/marlin/vllm_marlin/**` (16 files) | [vLLM](https://github.com/vllm-project/vllm) (`csrc/libtorch_stable/quantization/marlin`) | local vLLM fork | **Apache-2.0** | **OWNED** (derived) | Host launcher rewritten to a raw-pointer C ABI; `torch::stable` unbound; selector trimmed to 354 instantiated variants; `shim/torch/` is ours |
| `crates/kernels/cu/gguf/vendor/**` (llama.cpp MMQ, dp4a arm) | [llama.cpp](https://github.com/ggml-org/llama.cpp) | `1b1e27cb49` (2024-05-23) | **MIT** (`LICENSE-llama.cpp`) | **FROZEN** | 2 registered changes (`dst` → `__half*`; `OWL_MMQ_TENSOR_CORES` switch) — see [README](crates/kernels/cu/gguf/vendor/README.md) |
| `crates/kernels/cu/gguf/vendor2/**` (head MMQ, tensor-core arm) | same | `dfcf18193` (2026-09-13) | **MIT** (`LICENSE`) | **OWNED** | 4 registered changes P1/P2/P2b/P3 — see [README](crates/kernels/cu/gguf/vendor2/README.md) |

**Common admission checks** (verified per tree):

- no cutlass, no PyTorch runtime, no TVM dependency;
- **no untracked `cudaMalloc`** (everything goes through our pool/arena — the first
  iron rule in `crates/backends/README.md`);
- sources are committed; prebuilt `.a` files are not (nvcc generates them at build
  time, with an incremental gate in `crates/kernels/build.rs`).

## 3. External references (REFERENCE: not committed)

| Source | Used for | Where |
|---|---|---|
| vLLM (production incumbent) | Baseline comparison / piecewise and degradation semantics | `docs/devlog/perf/`, construction volumes |
| candle / attention-rs / mistral.rs | Early material evaluation | `docs/devlog/canon/charter.md` §3 (historical; **not** dependencies today) |
| llama.cpp (head) | MMQ / GGUF semantics comparison | 外部存档(llama.cpp 卷宗) |
| FlashInfer upstream | plan semantics / variant comparison | `docs/devlog/kernels/prefill-attention-swap.md` |

> None of these are in this repo's dependency graph (`Cargo.toml` contains only
> workspace-local `path` dependencies).

## 4. Maintenance discipline

1. **Upgrading a vendor** = bump the pinned revision + update the table above + update
   the version line in `NOTICE` + re-run the corresponding numerical gates;
2. **Changing a FROZEN tree** = a deliberate ownership transfer: register it (README
   table) + name the gate that proves it + list the diff; no drive-by edits;
3. **Adding a vendor** = pass the admission rule first (copy-and-fix vs use as-is),
   then add a row here and attribution in `NOTICE`;
4. **Never relicense** vendored files: our dual license covers only our own code.
