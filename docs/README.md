# owl documentation index

> Navigation **and** per-document review ledger for the public repository.
>
> Status legend: `⬜ pending` / `✅ reviewed`. Reviewing one document means three things:
> ① de-identify (`/home/...` absolute paths, hardware identifiers) →
> ② reword out-of-repo references as "external archive" →
> ③ align conclusions with the current code (mark stale claims `> ⚠️ retracted`).
>
> **Code comments are out of scope** (maintainer decision, 2026-10-21).
>
> Layout (2026-10-22): every body document lives under `devlog/`, grouped by theme;
> filenames are ASCII slugs (no numbering, no dates). Cross-references inside
> documents are **name-only** by convention — no directory paths.
>
> **No hard-coded paths to things a reader cannot see.** References to anything
> outside this repository, or to git-ignored locations, are written with one of
> these markers instead of a path:
> `外部存档` (external archive) · `本地台账` (local work ledger) ·
> `本地实测档` (local measurement archive) · `本地原始数据` (local raw data) ·
> `/path/to/models/…` (weights you supply) · `<worktree>/…` (git worktree).

Progress: **0 / 99** reviewed.

Body documents are written in **Chinese**; this index and all meta documents
(`README`, `CONTRIBUTING`, `SECURITY`, `THIRD_PARTY.md`, `NOTICE`) are in **English**.

## Newcomer reading path (five documents, in order)

1. [`devlog/canon/charter.md`](devlog/canon/charter.md) — axioms and boundaries
2. [`devlog/dev/README.md`](devlog/dev/README.md) — how to change things, how to test
3. [`devlog/graph/graph-model-seam.md`](devlog/graph/graph-model-seam.md) — the graph/model seam
4. [`devlog/memory/store.md`](devlog/memory/store.md) — where state lives
5. [`devlog/dev/debug-tools.md`](devlog/dev/debug-tools.md) — where to look when it breaks

## A. Canon: charter, requirements, runtime model

- ⬜ [`devlog/canon/charter.md`](devlog/canon/charter.md) — **Charter**: axioms A1/A2, ownership boundaries, borrowed materials
- ⬜ [`devlog/canon/requirements.md`](devlog/canon/requirements.md) — Requirements specification (inherited from the frozen xinfer baseline)
- ⬜ [`devlog/canon/async-runtime.md`](devlog/canon/async-runtime.md) — Async model: four-layer architecture + declarative tensors

## B. Graph execution and capture

- ⬜ [`devlog/graph/graph-model-seam.md`](devlog/graph/graph-model-seam.md) — Signal-Graph × model-layer seam
- ⬜ [`devlog/graph/prefill-graph.md`](devlog/graph/prefill-graph.md) — prefill CUDA Graph design and implementation
- ⬜ [`devlog/graph/fi-single-graph.md`](devlog/graph/fi-single-graph.md) — Graph-profile FI single-graph (boot plan / capture families)
- ⬜ [`devlog/graph/piecewise-graph.md`](devlog/graph/piecewise-graph.md) — Piecewise (breakable) capture graphs
- ⬜ [`devlog/graph/decode-out-of-graph.md`](devlog/graph/decode-out-of-graph.md) — decode out-of-graph round trip (large GPU-server change)
- ⬜ [`devlog/graph/prefill-bucket-qslot.md`](devlog/graph/prefill-bucket-qslot.md) — prefill buckets × KVMem: Q-slot blocking analysis

## C. Communication and tensor parallelism

- ⬜ [`devlog/comm/tp-overview.md`](devlog/comm/tp-overview.md) — TP from the math skeleton to each engine's reality
- ⬜ [`devlog/comm/tp2-prestudy.md`](devlog/comm/tp2-prestudy.md) — TP2 communication pre-study (constraints × contract × sequence)
- ⬜ [`devlog/comm/tp2-design.md`](devlog/comm/tp2-design.md) — TP2 construction design (commit slicing)
- ⬜ [`devlog/comm/tp2-log.md`](devlog/comm/tp2-log.md) — TP2 war log (full construction and battle accounts)
- ⬜ [`devlog/comm/tp2-roadmap.md`](devlog/comm/tp2-roadmap.md) — TP2 prefill/decode speedup roadmap

## D. Operators and kernels

- ⬜ [`devlog/kernels/operator-matrix.md`](devlog/kernels/operator-matrix.md) — Quantization-oriented operator family matrix
- ⬜ [`devlog/kernels/operator-contract.md`](devlog/kernels/operator-contract.md) — Operator contract authority (kernels dual-plane infrastructure)
- ⬜ [`devlog/kernels/attention-port.md`](devlog/kernels/attention-port.md) — attention-rs kernel port initiative
- ⬜ [`devlog/kernels/long-ctx-attention.md`](devlog/kernels/long-ctx-attention.md) — Long-context attention: paged-family port route
- ⬜ [`devlog/kernels/prefill-attention-swap.md`](devlog/kernels/prefill-attention-swap.md) — prefill attention kernel swap: candidates and unit tests
- ⬜ [`devlog/kernels/w4a8-family.md`](devlog/kernels/w4a8-family.md) — W4A8 operator family
- ⬜ [`devlog/kernels/kernel-tag-catalog.md`](devlog/kernels/kernel-tag-catalog.md) — Operator tag catalog + build-time master table
- ⬜ [`devlog/kernels/kernel-admission.md`](devlog/kernels/kernel-admission.md) — Operator admission gates and block governance
- ⬜ [`devlog/kernels/kernel-error-ledger.md`](devlog/kernels/kernel-error-ledger.md) — Operator error ledger (history × prevention)
- ⬜ [`devlog/kernels/qwen3-mini-demo.md`](devlog/kernels/qwen3-mini-demo.md) — Qwen3.5-0.8B mini-demo implementation list
- ⬜ [`devlog/kernels/qwen38-operator-inventory.md`](devlog/kernels/qwen38-operator-inventory.md) — Qwen3.8-27B production operator inventory

## E. Models, formats, loading, vision

- ⬜ [`devlog/models/candle-api-mapping.md`](devlog/models/candle-api-mapping.md) — candle Tensor API → owl translation map
- ⬜ [`devlog/models/interpreter-env.md`](devlog/models/interpreter-env.md) — Interpreter EnvProvider design and parameter enumeration
- ⬜ [`devlog/models/interpreter-tap.md`](devlog/models/interpreter-tap.md) — Interpreter node-level tap
- ⬜ [`devlog/models/loader-interpreter.md`](devlog/models/loader-interpreter.md) — Loader as model interpreter
- ⬜ [`devlog/models/load-speedup.md`](devlog/models/load-speedup.md) — Load-time speedup verdict round (27B AWQ)
- ⬜ [`devlog/models/load-speedup-tp2-handoff-20261024.md`](devlog/models/load-speedup-tp2-handoff-20261024.md) — Load-speed handoff: TP2 parallel load + per-device vram account (P0 done)
- ⬜ [`devlog/models/gguf-format.md`](devlog/models/gguf-format.md) — GGUF format support
- ⬜ [`devlog/models/new-format-checklist.md`](devlog/models/new-format-checklist.md) — New-format onboarding checklist
- ⬜ [`devlog/models/vision-tower.md`](devlog/models/vision-tower.md) — Vision tower integration + CPU parity harness
- ⬜ [`devlog/models/vision-wiring.md`](devlog/models/vision-wiring.md) — Vision tower × engine/serving wiring
- ⬜ [`devlog/models/vision-multi-image-handoff-20261024.md`](devlog/models/vision-multi-image-handoff-20261024.md) — Vision multi-image handoff: FA2 convergence, 4B MTP restore_spec_snap blocker
- ⬜ [`devlog/models/model-hot-swap.md`](devlog/models/model-hot-swap.md) — Model hot-swap proposal
- ⬜ [`devlog/models/lora-lifecycle.md`](devlog/models/lora-lifecycle.md) — LoRA lifecycle proposal

## F. Speculative decoding

- ⬜ [`devlog/spec/spec-decode-survey.md`](devlog/spec/spec-decode-survey.md) — Survey of four speculative-decoding approaches
- ⬜ [`devlog/spec/dflash2-design.md`](devlog/spec/dflash2-design.md) — DFlash2 semantic verdict and landing
- ⬜ [`devlog/spec/mtp-design.md`](devlog/spec/mtp-design.md) — MTP speculative decoding
- ⬜ [`devlog/spec/mtp-confidence-cutoff.md`](devlog/spec/mtp-confidence-cutoff.md) — MTP confidence cutoff (T2 design and construction)
- ⬜ [`devlog/spec/mtp-draft-kv-catchup.md`](devlog/spec/mtp-draft-kv-catchup.md) — MTP draft KV catch-up (T3 pre-requisite design and construction)
- ⬜ [`devlog/spec/mtp-longctx-windowed.md`](devlog/spec/mtp-longctx-windowed.md) — MTP long-context anti-decay: Windowed-MTP + adaptive depth (T1a/T1b)
- ⬜ [`devlog/spec/mtp-longctx-handoff.md`](devlog/spec/mtp-longctx-handoff.md) — MTP long-context anti-decay: next-stage construction and handoff
- ⬜ [`devlog/spec/gguf-dflash2-survey.md`](devlog/spec/gguf-dflash2-survey.md) — DFlash2 speculative decoding on the GGUF line
- ⬜ [`devlog/spec/gguf-dflash2-roadmap.md`](devlog/spec/gguf-dflash2-roadmap.md) — GGUF DFlash2 speedup roadmap
- ⬜ [`devlog/spec/longctx-spec-survey.md`](devlog/spec/longctx-spec-survey.md) — Long-context spec-decode: our decay vs community approaches

## G. Memory, KV and long context

- ⬜ [`devlog/memory/store.md`](devlog/memory/store.md) — store = GPU pool → host → disk three-tier virtualization
- ⬜ [`devlog/memory/kvmem-selection.md`](devlog/memory/kvmem-selection.md) — KVMem selection layer (B axis)
- ⬜ [`devlog/memory/longctx-attribution.md`](devlog/memory/longctx-attribution.md) — Long-context performance attribution, round one
- ⬜ [`devlog/memory/prefix-deletion.md`](devlog/memory/prefix-deletion.md) — Prefix-transparent deletion
- 🟡 [`devlog/memory/image-evict-leave-gap-交接.md`](devlog/memory/image-evict-leave-gap-交接.md) — Image eviction without delta-limit rebuilds: leave-gap positions, position ledger, consolidation, phased handoff（P0-P2 核心 + 0.8B spec 门已实现；P3 27B/TP2 与 P4 consolidation 未做）
- ⬜ [`devlog/memory/store-hook-contract.md`](devlog/memory/store-hook-contract.md) — Store hook contract: 19-point audit, 3-hook convergence (Acquire/Release/Maintain), Phase 1–4 complete
- ✅ [`devlog/memory/2026-10-27-vram-scheduling-review.md`](devlog/memory/2026-10-27-vram-scheduling-review.md) — store × engine VRAM scheduling review: MemoryPlan / Grant / Reconciliation, code findings and V0–V5 migration（V0–V5 已落地，见 `store.md` §33–§36）
- ⬜ [`devlog/memory/kvarn-k4v4-施工文档.md`](devlog/memory/kvarn-k4v4-施工文档.md) — KVarN k4v4/g128 handoff: layout, kernels, page128 integration, phase gates
- ⬜ [`devlog/memory/kvarn-k4v4-落地可行性复核.md`](devlog/memory/kvarn-k4v4-落地可行性复核.md) — KVarN feasibility review: sink/tail lifecycle, store RTN/pack gap, page128 blast radius, 3090/3080 capacity math, k4v4 vs k4v2 verdict
- ⬜ [`devlog/memory/kv-quant-deep-dive.md`](devlog/memory/kv-quant-deep-dive.md) — KV quant deep dive: FP8 vs TQ k8v4 vs KVarN (quality/capacity/integration)
- ⬜ [`devlog/memory/fp8-baseline-2026-10-25.md`](devlog/memory/fp8-baseline-2026-10-25.md) — FP8 KV baseline: GSM8K 200, MMLU-Pro smoke, needle 240k, std (spec-off)
- ⬜ [`devlog/memory/2026-10-27-memory-plan-v0-v1.md`](devlog/memory/2026-10-27-memory-plan-v0-v1.md) — MemoryPlan V0/V1 + first schedulers refactor: single budget source, shared-prefix dedup, state-slot disk spill, Grant/Reservation ledger, active-snapshot guard, PressureSignals + idle-off-by-default, HostPressure disk-only tickets with low/high host watermarks and per-request KV/snapshot quotas, measured graph-slab backfill, GDN row owner attribution, multi-kind Grant ledger

## H. Serving, sessions and configuration

- ⬜ [`devlog/serving/agent-surface.md`](devlog/serving/agent-surface.md) — Agent serving surface (pi integration)
- ⬜ [`devlog/serving/constrained-tool-grammar/README.md`](devlog/serving/constrained-tool-grammar/README.md) — Constrained tool grammar project: prototype design, phases, decode-cost budget (design only; not implemented)
- ⬜ [`devlog/serving/agent-prefix-10k.md`](devlog/serving/agent-prefix-10k.md) — Agent 10K multi-turn prefix hit rate: snapshot verdict, message-boundary anchors, probe
- ⬜ [`devlog/serving/multi-session.md`](devlog/serving/multi-session.md) — Multi-session and concurrency
- ⬜ [`devlog/serving/cross-session-batch.md`](devlog/serving/cross-session-batch.md) — Cross-session batched decode (B5.1)
- ⬜ [`devlog/serving/session-plan.md`](devlog/serving/session-plan.md) — Session orchestration surface
- ⬜ [`devlog/serving/config-ux.md`](devlog/serving/config-ux.md) — Configuration UX (fully explicit fields)
- ⬜ [`devlog/serving/spec-tailcap-gpu-spin-立案.md`](devlog/serving/spec-tailcap-gpu-spin-立案.md) — DFlash2 尾批裁剪 ⇒ verify 图档形状错配 ⇒ **设备核自旋整台钉死**(已修;含根因、复现与回归门)
- ⬜ [`devlog/serving/2026-10-26-leave-gap-frontend-handoff.md`](devlog/serving/2026-10-26-leave-gap-frontend-handoff.md) — leave-gap 前端联调交接:KV 引用解除、turn35 重复输出、turn38 压缩全量 prefill、视觉塔缓存复用
- ⬜ [`devlog/serving/2026-10-26-prefill-f16-shadow-3080实测.md`](devlog/serving/2026-10-26-prefill-f16-shadow-3080实测.md) — prefill FP8→F16 shadow:3080 TP2 +3~9%、1.265× attention、cap 预留修 OOM、相对数值门
- ✅ [`devlog/serving/2026-10-26-topk32-device-rep-penalty-handoff.md`](devlog/serving/2026-10-26-topk32-device-rep-penalty-handoff.md) — top-32 设备端 rep_penalty:H=256 committed history、固定图槽、先惩罚后 top-32、3080 A/B 与 kernel 对拍
- ⬜ [`devlog/serving/2026-10-26-prefill-longctx-attention-kv-thrash-handoff.md`](devlog/serving/2026-10-26-prefill-longctx-attention-kv-thrash-handoff.md) — 长上下文 prefill 交接:100k 内 FlashInfer attention 逼近硬件成本；200k 的 13s TTFT 主要是整链 KV 下沉/回灌（9s 级），准入/复用账待修；**2026-10-26 追加：下一会话只做 P0（准入/复用账）+ P2（f16 CTA64/换核），P1（capped f16 shadow）暂缓**

## I. Performance and engineering (framework · device · pitfalls · bench reports)

- ⬜ [`devlog/perf/perf-framework.md`](devlog/perf/perf-framework.md) — Performance framework: metrics + interpreter hooks
- ⬜ [`devlog/perf/optimization-candidates.md`](devlog/perf/optimization-candidates.md) — Optimization candidate ledger: prefill/decode candidates, verdicts, closed doors
- ⬜ [`devlog/perf/decode-cycle.md`](devlog/perf/decode-cycle.md) — decode cycle breakthrough (vLLM/SGLang comparison)
- ⬜ [`devlog/perf/hyperqwen-survey.md`](devlog/perf/hyperqwen-survey.md) — Single-24G-card 27B serving benchmark (gap verdict)
- ⬜ [`devlog/perf/device-command-ledger.md`](devlog/perf/device-command-ledger.md) — DeviceHandle: allocation ledger + command face
- ⬜ [`devlog/perf/stream-ordering.md`](devlog/perf/stream-ordering.md) — GPU-server cross-stream ordering
- ⬜ [`devlog/perf/alloc-naming.md`](devlog/perf/alloc-naming.md) — Named allocations + OOM dump
- ⬜ [`devlog/perf/log-framework.md`](devlog/perf/log-framework.md) — logger: event-stream outlet (level + tag + tee)
- ⬜ [`devlog/perf/test-architecture.md`](devlog/perf/test-architecture.md) — Test architecture (two classes, single entry point)
- ⬜ [`devlog/perf/test-macro.md`](devlog/perf/test-macro.md) — #[owl_test] procedural macro
- ⬜ [`devlog/perf/pitfalls-general.md`](devlog/perf/pitfalls-general.md) — Common development pitfalls
- ⬜ [`devlog/perf/pitfalls-cuda.md`](devlog/perf/pitfalls-cuda.md) — CUDA/Graph pitfall ledger
- ⬜ [`devlog/perf/owl-phase1-nn.md`](devlog/perf/owl-phase1-nn.md) — owl speed compilation (phase-1 nn path)
- ⬜ [`devlog/perf/qwen38-single-3090.md`](devlog/perf/qwen38-single-3090.md) — 27B single-card engine battle
- ⬜ [`devlog/perf/v1-fi-switch.md`](devlog/perf/v1-fi-switch.md) — All three attention phases moved to FI
- ⬜ [`devlog/perf/session-2026-10-14.md`](devlog/perf/session-2026-10-14.md) — 2026-10-14 prefill attention kernel swap: closing notes
- ⬜ [`devlog/perf/w4a16-prefill-bucket-ab.md`](devlog/perf/w4a16-prefill-bucket-ab.md) — W4A16 × prefill buckets A/B (no benefit, verdict)
- ⬜ [`devlog/perf/w4a8-prefill-bucket-ab.md`](devlog/perf/w4a8-prefill-bucket-ab.md) — W4A8 × prefill buckets A/B
- ⬜ [`devlog/perf/gguf-kvmem-decode-nsys.md`](devlog/perf/gguf-kvmem-decode-nsys.md) — GGUF × KVMem decode verdict (bandwidth wall)
- ⬜ [`devlog/perf/tp2-decode-fast.md`](devlog/perf/tp2-decode-fast.md) — TP2 decode fast path + W4A8 attribution + draft-pool ring (2026-10-22)
- ⬜ [`devlog/perf/dflash-optimization-handoff.md`](devlog/perf/dflash-optimization-handoff.md) — DFlash2 优化交接：lookup / W8A16 TC / 质量门 / NInfer K=1..15 (2026-10-23)

## J. Developer guide (how to change things)

- ⬜ [`devlog/dev/README.md`](devlog/dev/README.md) — Developer entry point (navigation for this directory)
- ⬜ [`devlog/dev/config.md`](devlog/dev/config.md) — Config chain: intent → derive → preflight → explain
- ⬜ [`devlog/dev/testing.md`](devlog/dev/testing.md) — Testing workflow v2
- ⬜ [`devlog/dev/fixtures.md`](devlog/dev/fixtures.md) — Fixtures and gating
- ⬜ [`devlog/dev/debug-tools.md`](devlog/dev/debug-tools.md) — Debugging quick reference
- ⬜ [`devlog/dev/profiling.md`](devlog/dev/profiling.md) — Profiler and measurement discipline
