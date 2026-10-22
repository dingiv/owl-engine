# Third-party inventory — vendored, ported and referenced code

> **Authoritative inventory of the third-party code that Cargo does not declare** —
> vendored sources, ported kernels, referenced upstreams (Cargo crates are covered by
> `Cargo.lock` + `cargo deny`, see §4).
> Read these two first before changing dependencies or adding a vendor:
> - [`crates/backends/README.md`](crates/backends/README.md) — **CUDA admission rule**
>   (any CUDA-touching library must be source-reviewed for untracked allocations;
>   if it calls `cudaMalloc` behind your back, don't use it — or copy it in and fix it);
> - [`docs/devlog/canon/charter.md`](docs/devlog/canon/charter.md) A4 — the four ownership classes.
>
> Copyright lines and upstream license texts live in [`NOTICE`](NOTICE); this file is
> the inventory that points at them.

## 1. Ownership classes (charter A4)

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
| `crates/kernels/triton/fla/**` (GDN chunked, 6 Triton kernels + import closure) | [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) | pinned via `triton/gdn_chunked_aot.json` | **MIT** | **FROZEN** | `triton_shim.py` is ours; AOT-compiled at build time by `triton/gdn_aot_compile.py` (no runtime compile) |

**Common admission checks** (verified per tree):

- no cutlass, no PyTorch runtime, no TVM dependency;
- **no untracked `cudaMalloc`** (everything goes through our pool/arena — the first
  iron rule in `crates/backends/README.md`);
- sources are committed; prebuilt `.a` files are not (nvcc generates them at build
  time, with an incremental gate in `crates/kernels/build.rs`).

## 3. Ported / derived code (in our own files)

Upstream code **ported into our own files** — rewritten to our ABI/calling
convention, but semantically derived. The bulk is `crates/kernels/`; a few live in
`crates/models/`. Every such file carries its own provenance header
(`溯源:` / `ported from` / `设计来源`); this table is the index of them.

| Upstream | License | Landed in | How |
|---|---|---|---|
| **attention.rs** (frozen snapshot) | **MIT** — except the paged-attention files, which are file-level from **vLLM / FasterTransformer** ⇒ **Apache-2.0** | `cu/attn/{pagedattention_f16,prefill_paged_attn_f16,reshape_and_cache,reshape_and_cache_fp8kv}.cu` · `cu/gdn/*.cu` (8 kernels) · `cu/gguf/{gguf.cuh,gguf_gemm.cu}` · `cu/load/marlin_repack_ct.cu` | file-level reuse (attn, `gguf.cuh`; headers kept) + explicit template expansion (GDN `gdn.cu`); dtype narrowed to our contract |
| **vLLM** | Apache-2.0 | `cu/ln/{fused_add_rmsnorm,silu_and_mul}.cu` · `cu/fused/qknorm_insert.cu` · `cu/elems/quant_int8_row.cu` · `cu/marlin/marlin_host.cu` | port: semantics preserved, host side rewritten to our raw-pointer ABI |
| **sglang** (DFlash2 branch) | Apache-2.0 | `cu/attn/naive_attn_nc.cu` · `cu/dflash/{dflash_conv,dflash_select}.cu` · `cu/elems/dflash_topk.cu` | port with **per-formula parity** against the Python reference (`models/dflash.py` + `kernels/ops/speculative/dflash.py`) |
| **lmdeploy / TurboMind** | Apache-2.0 | `cu/gdn/gdn_chunk_scalar.cu` | port of `kernels/linear_attn/…/pre_sm90/chunked.cu` |
| **IST-DASLab/marlin** | Apache-2.0 | `cu/marlin/marlin_ffi_c.cu` | port of `marlin_cuda.cpp` (C-ABI entry point) |
| **NInfer** (`ninfer_duo_3080`) | Apache-2.0 | `cu/elems/sigmoid_gate_mul_f16.cu` | port (`bf16x8` → f16) |
| **xinfer** (upstream `guoqingbao/xinfer`, MIT; developed in our fork) | MIT (the Marlin kernel inside it: Apache-2.0, see below) | `cu/marlin/**` + `src/family/marlin/**` — vendored from its `crates/kernels/cuda/csrc/marlin` (vLLM-enhanced Marlin ⊂ **IST-DASLab/marlin + Neural Magic**, Apache-2.0; selector is a trimmed derivative) · engine crate layout transplanted from its `crates/core/src` · attention kernels arrived through its vendored `attention.rs` copy | **predecessor project**: vendored kernels + structural transplant + shared discipline (parity gates, build flags) |
| **llama.cpp / ggml** | MIT | `cu/gguf/gguf_mmq*.cu` · `cu/gguf/gguf_quant.cu` · `cu/gguf/gguf_rows.cu` · `cu/gguf/vendor{,2}/**` (§2) · `models/src/formats/gguf/iq_tables.rs` (IQ codebooks, verbatim transcription) · `models/src/interpreters/planner.rs` (port of `ggml_dyn_tallocr`) | verbatim vendoring + thin adapters (registered H1/H2/H3 deviations) |

> **Audit note (2026-10-22)**: every file under `crates/kernels/cu/` was read for
> provenance. Files **without** a provenance header are our own implementations
> (the `elems/` · `ln/` · `comm/` · `probe/` · `vis/` families and the remaining
> `attn/`/`gdn/`/`gguf/` kernels). A few follow upstream *semantics* without
> copying code (rope rotate-half layout, RMSNorm block reduction, flash-decoding
> partition) — those upstreams are listed in §5.

**Obligation**: a ported file keeps its upstream attribution header **and** a row
here. Rewriting to our ABI does **not** relicense it — the upstream license still
applies to the derived file. All upstreams above are MIT/Apache-2.0 (compatible with
our `MIT OR Apache-2.0`), but the notice must stay.

## 4. Rust dependencies (declared in Cargo — not listed here)

Crates pulled through Cargo **are not inventoried in this file**: the single source
of truth is `Cargo.toml` (direct deps) + `Cargo.lock` (the full resolved tree), and
the license/source gate is `cargo deny check licenses bans sources` (`deny.toml` is
the allow-list; CI runs it on every change).

To materialise the per-crate table on demand (nothing is committed):

```bash
cargo metadata --format-version 1 --all-features \
  | python3 tools/gen_third_party_licenses.py
```

Audit result (reproducible with the command above): **no copyleft (GPL/AGPL)
dependency**; all crates are permissively licensed (MIT / Apache-2.0 / BSD / ISC /
Zlib / Unlicense / Unicode-3.0).

## 5. Referenced upstreams (no code in this repo)

Projects we read, benchmarked against, or took conventions from. **Nothing beyond
what §2/§3 already list was copied from them** — if that ever changes, the code gets
a provenance header and moves to §3.

| Source | Used for | Note |
|---|---|---|
| **candle-gb** (fork snapshot) | tensor-op conventions: block-reduction RMSNorm (同构), `[out,in]` weight layout for nt-matmul | MIT OR Apache-2.0; **not** a dependency today |
| **cudarc-gb** (fork snapshot) | driver-API evaluation | MIT OR Apache-2.0; superseded by official `cudarc` 0.19 |
| **kvmem** (KVMem 0.18.0 reference implementation) | upstream **prototype** of the KV-virtualization design (B axis): attention-aware LRFU eviction, GQA-max scoring, 32-token block granularity | our implementation (`crates/store`) is an independent Rust rewrite — **no code copied**; the wrapper source carries no explicit license (its bundled backends keep theirs) |
| **llm-compressor** | model-conversion toolchain that produced the W4A16 / `compressed-tensors` checkpoints this engine loads (recipes run outside this repo) | Apache-2.0 |
| **vllm-ampere-optimized** (avesed) | the W4A8 int8-tensor-core **route** we followed (a named copy target in `requirements.md` REQ-DESIGN); its Ampere XQA closure is in-tree → **§2** | Apache-2.0 |
| **club-3090** | 2×3090 serving recipes (fp8-KV / 256k) | Apache-2.0; its three patches were ported into the **vLLM production stack** (a different repository), not into owl |
| **mistral.rs** | early material evaluation; ISQ / quantized-loading semantics | MIT |
| **HyperQwen** | single-24G-card 27B serving benchmark (levers to borrow) | Apache-2.0 |
| vLLM (production incumbent) | baseline comparison; piecewise / degradation semantics | Apache-2.0 |
| llama.cpp (head) | MMQ / GGUF semantics comparison | MIT |
| FlashInfer upstream | plan semantics / variant comparison | Apache-2.0 |

> None of these are in this repo's dependency graph (`Cargo.toml` contains only
> workspace-local `path` dependencies). Reference material for the wider workspace
> (not this repo) is kept in an external archive.

## 6. Maintenance discipline

1. **Upgrading a vendor** = bump the pinned revision + update the §2 table + update
   the version line in `NOTICE` + re-run the corresponding numerical gates;
2. **Changing a FROZEN tree** = a deliberate ownership transfer: register it (§2
   table) + name the gate that proves it + list the diff; no drive-by edits;
3. **Adding a vendor** = pass the admission rule first (copy-and-fix vs use as-is),
   then add a row to §2 and attribution to `NOTICE`;
4. **Porting upstream kernel code** = keep the provenance header + add a §3 row
   (upstream, license, landing path, what was taken); a rewrite does not erase
   attribution;
5. **Adding/upgrading a Rust dependency** = nothing to update in this file (Cargo is
   the source of truth); just make sure `cargo deny check licenses bans sources`
   stays green (`deny.toml` is the allow-list);
6. **Never relicense** vendored or ported files: our dual license covers only our own code.
