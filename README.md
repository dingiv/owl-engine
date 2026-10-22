# owl

**Run the models you want fast, on the GPUs you already own — and keep the context.**

| | What you get | Status |
|---|---|---|
| 🧮 | **VRAM-efficient local deployment** — a budget declared at boot that the process never exceeds; every allocation is ledgered by name and reconciled against the driver | ✅ |
| 🚀 | **High performance** — CUDA-graph capture, block-level prefix cache, fused kernels (measured numbers to be published) | ✅ |
| 📦 | **GGUF + SafeTensors** — AWQ-INT4, W4A16, Q4_K, IQ family are all supported | ✅ |
| 🔗 | **Tensor parallelism (TP=2)** — one-shot allreduce over IPC / P2P | ✅ |
| 🗄 | **Three-tier KV offload** — GPU pool → pinned host RAM → disk; blocks demote and come back on demand instead of being dropped | ✅ |
| 🧠 | **KV-cache virtualization** — each turn materializes only its working set on the device; the full KV stays virtual and is recalled by attention-aware scoring | ✅ |
| ⚡ | **Speculative decoding** — DFlash2 draft (d7) and MTP head, with spec-aware graph capture | ✅ |
| 👁 | **Vision** — Qwen3.8-VL vision tower (m-RoPE, image scatter) served by the same engine | ✅ |
| 🖥 | **CUDA + AMD (ROCm) ecosystems** — the backend is a vendor-agnostic HAL today; the second implementation is planned | 🎯 planned |
| 🖼 | **Image KV transparent deletion** — drop an old screenshot from the history and keep the prefix cache hit (for GUI agents) | 🚧 upcoming |
| 🧩 | **LoRA adapters + dynamic LoRA fine-tuning** — adapter registration, hot activation and rollback, with the full lifecycle (data → train → eval → serve) scoped in the design log | 🎯 planned |
| 🔄 | **Model hot-swap** — drain in-flight turns, park the current model's weights (host RAM or disk) and bring another one up **on the same GPU**, so one deployment switches models without a restart | 🎯 planned |

`✅ implemented` · `🎯 planned` · `🚧 in design`

owl is a from-scratch LLM inference engine in Rust, built on one conviction: **the
first hard limit of local AI is VRAM, so memory is a contract, not a gamble.** Declare
the budget at boot, ledger every allocation, virtualize the KV cache across tiers, and
treat quantization as the first lever — then **the context you can actually afford
becomes the product**: long agent sessions, 100k+ windows, no OOM.

> **Scope, honestly** — this is alpha, and narrow on purpose.
>
> - **Models: today one family — Qwen3.8 / Qwen3.5.** Production is the 27B hybrid
>   (GDN linear + full attention), with AWQ-INT4 / W4A16 / GGUF checkpoints, a vision
>   tower, and DFlash2 / MTP draft heads. A new architecture lands as a spec + layer
>   set rather than a fork — but **if your checkpoint is not this family, owl cannot
>   serve it yet.**
> - **Status: single machine, alpha.** Tuned on RTX 30-series (sm_86); production
>   inference still runs on vLLM. This project is grown milestone by milestone.
>
> The design log — axioms, measurements, verdicts and an explicit *retracted
> conclusions* list — and is the most useful part of this repo.
> Start at [`docs/devlog/canon/charter.md`](docs/devlog/canon/charter.md); the
> English index is [`docs/README.md`](docs/README.md).
>
> Internally the project is organized around two first-class ideas — **CUDA-graph
> capture as the model seam** and **tensor-parallel communication as a first-class
> citizen** ([charter](docs/devlog/canon/charter.md) A1/A2). Those are architectural
> choices, not the pitch.

## 1. Capabilities

- **Model family**: Qwen3.8 / Qwen3.5 — dense hybrid (GDN linear + full attention),
  27B production / 0.8B dev, plus the VL vision tower — [`crates/models/src/formats/qwen35.rs`](crates/models/src/formats/qwen35.rs)
- **Model formats**: AWQ-INT4 (pack-quantized, gs32 asymmetric), W4A16 safetensors,
  F16/BF16, GGUF (Q4_K / Q5_K / Q6_K / IQ family) — [`crates/models`](crates/models)
- **Graph capture & replay**: decode / verify / fold / propose captured per family;
  prefill runs eager with chunk buckets; `[cuda] graph = false` flips everything to
  eager for A/B — [`docs/devlog/graph/graph-model-seam.md`](docs/devlog/graph/graph-model-seam.md)
- **TP=2**: one-shot allreduce, fused collective entry, two-tier peer modes;
  `--disable-custom-all-reduce` is a hard requirement on this hardware —
  [`docs/devlog/comm/tp-overview.md`](docs/devlog/comm/tp-overview.md)
- **Speculative decoding**: DFlash2 draft (d7) and MTP head; spec-aware graph capture —
  [`docs/devlog/spec/spec-decode-survey.md`](docs/devlog/spec/spec-decode-survey.md)
- **KV plane**: fp8 KV, paged kNHD layout, prefix cache, long context (the 1M study
  lives in the external archive; the 27B operator inventory is
  [`docs/devlog/kernels/qwen38-operator-inventory.md`](docs/devlog/kernels/qwen38-operator-inventory.md))
- **Hybrid attention**: GDN / mamba linear layers with chunked and scalar arms —
  [`docs/devlog/kernels/attention-port.md`](docs/devlog/kernels/attention-port.md)
- **Serving**: OpenAI-compatible `/v1/chat/completions` (SSE), Jinja2-compatible chat
  templates, tool calling, multi-session concurrency —
  [`docs/devlog/serving/agent-surface.md`](docs/devlog/serving/agent-surface.md)
- **Diagnostics**: unified logging (level + tag filtering + tee to file), in-memory
  metrics (printed only on query), probe families (`[probes]`: booleans gate
  *collection*, logging gates *visibility*) —
  [`docs/devlog/perf/log-framework.md`](docs/devlog/perf/log-framework.md)

### How VRAM and context are handled

| Mechanism | What it buys you |
|---|---|
| **Declared VRAM budget (hard contract)** | Budget declared at boot, held for the whole process life. The decomposition is closed — weights / KV pool / graph allowance / activation peak / workspace / comm buffers / runtime floor / margin — and **a missing measured number means refuse-to-start**. Every allocation is ledgered *by name* and periodically reconciled against the driver; overspend is fail-fast with a ledger snapshot. **The process never OOMs itself.** |
| **Three-tier KV offload** | The KV pool is not a hard ceiling: blocks demote GPU → pinned host RAM → disk and return on demand, tracked by a content-addressed ledger (`crates/store` tiers). Running out of VRAM costs latency, not a failed request. |
| **KV-cache virtualization (selection layer)** | A turn only materializes its working set on the device (bounded by `workspace_per_req`); everything else stays virtual and is recalled by attention-aware scoring — this is the path to 262k on a single consumer card. |
| **Quantization as the first lever** | W4A16 / W4A8 / AWQ-INT4 / GGUF (Q4_K · IQ family) — every GB saved on weights is a GB handed back to context. |
| **KV dtype knob** | fp16 (default) / fp8 / int8 — fp8 doubles the pool, at a measured accuracy cost. |
| **Graph budget up front** | Capture memory is registered *before* any pool arithmetic (axiom A1.1), so capture OOM is a boot-time decision, not a runtime surprise. |
| **Prefix cache** | 88% hit on same-session multi-turn, 89% on cross-session shared prefixes (1536/1732 blocks) — long agent sessions stop recomputing. |

> The arithmetic is written down: single-card 262k KV needs 26.1 GB dense at fp16 /
> 13.0 GB at fp8 for the 27B model — i.e. **262k is a virtualization (or TP) problem,
> not a tuning problem** (`docs/devlog/memory/store.md` §2).

## 2. Quick start

### Prerequisites

| Item | Requirement |
|---|---|
| GPU | NVIDIA sm_86 (RTX 30 series) first; single card or 2-card TP |
| CUDA | Local CUDA Toolkit with `nvcc` (sm_86 support); driver matched to toolkit |
| Rust | Stable, edition 2021 |
| Memory | Plan ~20 GB host RAM per GPU for weights + caches |
| Models | Bring your own (AWQ-INT4 / safetensors / GGUF; **not** distributed here) |

> ⚠️ **No authentication.** The server binds `127.0.0.1` by default. If you expose it to
> a LAN, put your own reverse proxy and auth in front — see [`SECURITY.md`](SECURITY.md).

### Build and run

```bash
# 1) Build (release without `perf` = byte-for-byte zero cost for timers)
cargo build --release -p server
#    Benchmark binary (timers stay enabled in release):
#    cargo build --release -p server --features perf

# 2) Pick a profile and point it at your checkpoint
cp profiles/qwen38_27b-3090-w4a16.toml my.toml
$EDITOR my.toml          # [model] path = "/path/to/Qwen3.8-27B-AWQ-INT4"

# 3) Preflight (no weights, no GPU; prints resolved config + memory estimate)
./target/release/server -p my.toml --check
./target/release/server -p my.toml --print-config            # TOML
./target/release/server -p my.toml --print-config --format json
./target/release/server -p my.toml --explain runtime.max_seq  # per-key provenance

# 4) Serve
./target/release/server -p my.toml --log-level info

# 5) Talk to it
curl -s 127.0.0.1:8135/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

Configuration is **fully explicit**: `profiles/full.toml` is the annotated template
(every key has a comment); other profiles use `extends = ["full.toml"]` and list only
their differences. See [`docs/devlog/dev/config.md`](docs/devlog/dev/config.md).

## 3. Diagnostics and measurement

The API and the diagnostics endpoints share **one port** (`[server] bind`, default
`127.0.0.1:8135`).

```bash
# Logs: level + tag allow/deny lists + tee to file (adjustable at runtime)
curl 127.0.0.1:8135/debug/log
curl 127.0.0.1:8135/debug/log/level/debug
curl "127.0.0.1:8135/debug/log/tags/owl-fi,-kernels"

# Numeric ledger: held in memory, emitted only on query.
# Take two snapshots and diff them to get a per-phase mean.
curl -s 127.0.0.1:8135/debug/metrics/prefill
curl -s 127.0.0.1:8135/debug/perf

# Dump a table when the run ends (CSV + JSON)
./target/release/server -p my.toml --perf-dump /tmp/owl-perf
```

Measurement discipline (and its traps) is documented in
[`docs/devlog/dev/debug-tools.md`](docs/devlog/dev/debug-tools.md) and
[`docs/devlog/dev/profiling.md`](docs/devlog/dev/profiling.md).

## 4. Testing

```bash
cargo test --workspace --lib          # pure-CPU unit tests
cargo test -p owl-kernels --lib       # operator face (oracle parity included)
```

- GPU tests are gated by `#[owl_test(gpu)]`: without `OWL_TEST_DEVICE` (a BDF such as
  `01:00.0`) they print `skip:` and return — **"did not run" is never disguised as
  "passed"**.
- Test configuration has a single entry point, `OWL_TEST_CONF` (an environment profile)
  plus convention-based overlays; asset paths come from `testdata/resources.toml`.
  See [`docs/devlog/dev/testing.md`](docs/devlog/dev/testing.md).

## 5. Repository layout

```
crates/
  sink       Log/metric sink (leaf crate, zero dependencies)
  store      KV/state three-tier virtualization + block ledger (pure state machine)
  iface      HAL contract (DeviceClient) + DeviceHandle (alloc ledger + command face)
  kernels    Operator dispatch + device-code assets (AOT cubins; Marlin/FlashInfer/XQA/GDN/GGUF)
  backends/  cuda (GpuServer actor / GpuClient), cpu (CpuFace reference impl)
  models     Loaders / interpreters / layers (including the VLM vision tower)
  engine     Scheduling / graph families / speculative decoding / sessions
  shared     Config surface / logging / metrics / testkit
  macros     #[owl_test] and friends
apps/
  server     OpenAI-compatible SSE server (engine actor owns the device)
  cli        Terminal REPL client
docs/        Index + devlog/ (design volumes and post-mortems, mostly Chinese)
profiles/    Config profiles (annotated full template + diffs)
tools/       Probe / bench / golden-generation scripts
testdata/    Reproducible fixtures (weights and large data are not committed)
```

## 6. Documentation

| Entry | Content |
|---|---|
| [`docs/README.md`](docs/README.md) | **Index** (English nav) + per-document review status |
| [`docs/devlog/canon/charter.md`](docs/devlog/canon/charter.md) | **Charter**: axioms, boundaries, ownership, borrowed materials |
| [`docs/devlog/dev/`](docs/devlog/dev/) | Required for contributors: config / testing / debugging / profiling / fixtures |
| [`docs/devlog/`](docs/devlog/) | Design volumes + construction post-mortems, grouped by theme (81 files, incl. retractions) |
| [`docs/devlog/perf/`](docs/devlog/perf/) | Performance & engineering: framework, optimization candidates, device/pitfall ledgers, bench reports |
| [`docs/devlog/kernels/`](docs/devlog/kernels/) | Operator kernels: admission gates, contract, quantization families, operator inventory |

> Body documents are in **Chinese** — they are the project's design log.
> Meta documents (this README, `CONTRIBUTING`, `SECURITY`, `docs/README.md`,
> `THIRD_PARTY.md`, `NOTICE`) are in **English**.

## 7. Third party and licensing

- Code in this repository: **MIT OR Apache-2.0** (see `LICENSE-MIT` / `LICENSE-APACHE`).
- Vendored CUDA sources are **not relicensed**: FlashInfer / NVIDIA XQA / vLLM-marlin
  are Apache-2.0, llama.cpp·ggml is MIT — attribution and pinned revisions are in
  [`NOTICE`](NOTICE) and [`THIRD_PARTY.md`](THIRD_PARTY.md).
- Third-party **non-Cargo** inventory (vendored / ported / referenced): [`THIRD_PARTY.md`](THIRD_PARTY.md); Cargo deps live in `Cargo.lock`.

## 8. Disclaimer

Provided **as is**, without warranty of any kind. It is built for single-machine
experiments: no authentication, no multi-tenant isolation. Driver/kernel-level patches
mentioned in the docs (e.g. GPU P2P) are **not** part of this repository; evaluate and
deploy them yourself if you need them.

