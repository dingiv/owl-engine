# owl documentation index

> Navigation **and** per-document review ledger for the public repository.
>
> Status legend: `⬜ pending` / `✅ reviewed`. Reviewing one document means three things:
> ① de-identify (`/home/...` absolute paths, hardware identifiers) →
> ② reword out-of-repo references as "external archive" →
> ③ align conclusions with the current code (mark stale claims `> ⚠️ retracted`).
>
> **Code comments are out of scope** (maintainer decision, 2026-10-21).

Progress: **0 / 75** reviewed.

Body documents are written in **Chinese**; this index and all meta documents
(`README`, `CONTRIBUTING`, `SECURITY`, `VENDOR.list.md`, `NOTICE`) are in **English**.

## Newcomer reading path (five documents, in order)

1. [`arch/charter.md`](arch/charter.md) — axioms and boundaries
2. [`dev/README.md`](dev/README.md) — how to change things, how to test
3. [`arch/graph-model-seam.md`](arch/graph-model-seam.md) — the graph/model seam
4. [`arch/store-状态存储-设计与迁移.md`](arch/store-状态存储-设计与迁移.md) — where state lives
5. [`dev/debug-tools.md`](dev/debug-tools.md) — where to look when it breaks

## A. Charter and axioms (read these first)

- ⬜ [`arch/charter.md`](arch/charter.md) — **Charter**: axioms A1/A2, ownership boundaries, borrowed materials
- ⬜ [`arch/requirements.md`](arch/requirements.md) — Requirements specification (inherited from the frozen xinfer baseline)
- ⬜ [`arch/operator-matrix.md`](arch/operator-matrix.md) — Quantization-oriented operator family matrix
- ⬜ [`arch/hyperqwen-对标调研.md`](arch/hyperqwen-对标调研.md) — Single-24G-card 27B serving benchmark (gap verdict)
- ⬜ [`arch/async-runtime.md`](arch/async-runtime.md) — Async model: four-layer architecture + declarative tensors
- ⬜ [`arch/candle-api-mapping.md`](arch/candle-api-mapping.md) — candle Tensor API → owl translation map

## B. Graphs, scheduling, sessions

- ⬜ [`arch/graph-model-seam.md`](arch/graph-model-seam.md) — Signal-Graph × model-layer seam
- ⬜ [`arch/prefill-graph-设计与实现.md`](arch/prefill-graph-设计与实现.md) — prefill CUDA Graph design and implementation
- ⬜ [`arch/图档-FI-单图-设计与施工.md`](arch/图档-FI-单图-设计与施工.md) — Graph-profile FI single-graph (boot plan / capture families)
- ⬜ [`piecewise-断点图-设计与实现.md`](piecewise-断点图-设计与实现.md) — Piecewise (breakable) capture graphs
- ⬜ [`arch/decode-图外往返-设计与施工.md`](arch/decode-图外往返-设计与施工.md) — decode out-of-graph round trip (large GPU-server change)
- ⬜ [`arch/三流事件序-设计与迁移.md`](arch/三流事件序-设计与迁移.md) — GPU-server cross-stream ordering
- ⬜ [`arch/跨会话合批-decode-调研与设计.md`](arch/跨会话合批-decode-调研与设计.md) — Cross-session batched decode (B5.1)
- ⬜ [`arch/session-plan.md`](arch/session-plan.md) — Session orchestration surface
- ⬜ [`arch/device-命令面与分配账-设计与施工.md`](arch/device-命令面与分配账-设计与施工.md) — DeviceHandle: allocation ledger + command face
- ⬜ [`arch/decode-cycle-突破方案.md`](arch/decode-cycle-突破方案.md) — decode cycle breakthrough (vLLM/SGLang comparison)

## C. Operators and kernels

- ⬜ [`arch/operator-contract-施工设计.md`](arch/operator-contract-施工设计.md) — Operator contract authority (kernels dual-plane infrastructure)
- ⬜ [`arch/attention-kernel-port.md`](arch/attention-kernel-port.md) — attention-rs kernel port initiative
- ⬜ [`arch/long-ctx-attention.md`](arch/long-ctx-attention.md) — Long-context attention: paged-family port route
- ⬜ [`arch/prefill-attention-换核-候选单测.md`](arch/prefill-attention-换核-候选单测.md) — prefill attention kernel swap: candidates and unit tests
- ⬜ [`arch/W4A8-算子家族-设计与施工.md`](arch/W4A8-算子家族-设计与施工.md) — W4A8 operator family
- ⬜ [`arch/kernel-tag-catalog-设计与施工.md`](arch/kernel-tag-catalog-设计与施工.md) — Operator tag catalog + build-time master table
- ⬜ [`arch/kernel-准入门槛-施工设计.md`](arch/kernel-准入门槛-施工设计.md) — Operator admission gates and block governance
- ⬜ [`arch/kernel-error-ledger.md`](arch/kernel-error-ledger.md) — Operator error ledger (history × prevention)
- ⬜ [`arch/全量记名-与-OOM-dump.md`](arch/全量记名-与-OOM-dump.md) — Named allocations + OOM dump
- ⬜ [`arch/新格式接入-改动点清单.md`](arch/新格式接入-改动点清单.md) — New-format onboarding checklist
- ⬜ [`arch/qwen3-mini-demo.md`](arch/qwen3-mini-demo.md) — Qwen3.5-0.8B mini-demo implementation list
- ⬜ [`qwen38/算子盘点.md`](qwen38/算子盘点.md) — Qwen3.8-27B production operator inventory

## D. Communication and TP

- ⬜ [`arch/TP-实现方案综述.md`](arch/TP-实现方案综述.md) — TP from the math skeleton to each engine's reality
- ⬜ [`arch/TP2-通信-预研.md`](arch/TP2-通信-预研.md) — TP2 communication pre-study (constraints × contract × sequence)
- ⬜ [`arch/TP2-施工设计.md`](arch/TP2-施工设计.md) — TP2 construction design (commit slicing)
- ⬜ [`arch/TP2-实录卷.md`](arch/TP2-实录卷.md) — TP2 war log (full construction and battle accounts)
- ⬜ [`arch/TP2-下一步路线图.md`](arch/TP2-下一步路线图.md) — TP2 prefill/decode speedup roadmap

## E. Storage, KV and long context

- ⬜ [`arch/store-状态存储-设计与迁移.md`](arch/store-状态存储-设计与迁移.md) — store = GPU pool → host → disk three-tier virtualization
- ⬜ [`arch/kvmem-选择层-设计与施工.md`](arch/kvmem-选择层-设计与施工.md) — KVMem selection layer (B axis)
- ⬜ [`arch/prefill-桶-KVMem-Q槽-分析-2026-10-21.md`](arch/prefill-桶-KVMem-Q槽-分析-2026-10-21.md) — prefill buckets × KVMem: Q-slot blocking analysis
- ⬜ [`arch/前缀无感删除-设计与施工.md`](arch/前缀无感删除-设计与施工.md) — Prefix-transparent deletion
- ⬜ [`长ctx归因-首轮.md`](长ctx归因-首轮.md) — Long-context performance attribution, round one

## F. Models, formats and speculative decoding

- ⬜ [`arch/装载器-解释器化-设计与迁移.md`](arch/装载器-解释器化-设计与迁移.md) — Loader as model interpreter
- ⬜ [`arch/装载提速-定谳轮-2026-10-16.md`](arch/装载提速-定谳轮-2026-10-16.md) — Load-time speedup verdict round (27B AWQ)
- ⬜ [`arch/gguf-格式支持-设计与施工.md`](arch/gguf-格式支持-设计与施工.md) — GGUF format support
- ⬜ [`arch/gguf-dflash2-投机-调研与设计.md`](arch/gguf-dflash2-投机-调研与设计.md) — DFlash2 speculative decoding on the GGUF line
- ⬜ [`arch/gguf-dflash2-验证提速-路线图.md`](arch/gguf-dflash2-验证提速-路线图.md) — GGUF DFlash2 speedup roadmap
- ⬜ [`arch/dflash2-施工方案.md`](arch/dflash2-施工方案.md) — DFlash2 semantic verdict and landing
- ⬜ [`arch/mtp-施工方案.md`](arch/mtp-施工方案.md) — MTP speculative decoding
- ⬜ [`arch/spec-decode-调研与立项.md`](arch/spec-decode-调研与立项.md) — Survey of four speculative-decoding approaches
- ⬜ [`arch/模型热替换-提案.md`](arch/模型热替换-提案.md) — Model hot-swap proposal
- ⬜ [`arch/LoRA-动态微调-提案.md`](arch/LoRA-动态微调-提案.md) — LoRA lifecycle proposal
- ⬜ [`arch/vision-tower-设计与CPU对拍.md`](arch/vision-tower-设计与CPU对拍.md) — Vision tower integration + CPU parity harness
- ⬜ [`arch/vision-引擎与服务面-接线-设计与施工.md`](arch/vision-引擎与服务面-接线-设计与施工.md) — Vision tower × engine/serving wiring
- ⬜ [`arch/interpreter-env.md`](arch/interpreter-env.md) — Interpreter EnvProvider design and parameter enumeration
- ⬜ [`arch/interpreter-tap.md`](arch/interpreter-tap.md) — Interpreter node-level tap
- ⬜ [`长ctx投机-社区方案调研.md`](长ctx投机-社区方案调研.md) — Long-context spec-decode: our decay vs community approaches

## G. Serving and configuration

- ⬜ [`arch/agent-服务面-设计与施工.md`](arch/agent-服务面-设计与施工.md) — Agent serving surface (pi integration)
- ⬜ [`arch/server-多会话并发设计.md`](arch/server-多会话并发设计.md) — Multi-session and concurrency
- ⬜ [`arch/配置面-体验优化-设计与施工.md`](arch/配置面-体验优化-设计与施工.md) — Configuration UX (fully explicit fields)

## H. Frameworks (logging / performance)

- ⬜ [`arch/log-framework.md`](arch/log-framework.md) — logger: event-stream outlet (level + tag + tee)
- ⬜ [`arch/perf-framework.md`](arch/perf-framework.md) — Performance framework: metrics + interpreter hooks

## I. Testing and development discipline

- ⬜ [`arch/test-architecture.md`](arch/test-architecture.md) — Test architecture (two classes, single entry point)
- ⬜ [`arch/owl-test-macro.md`](arch/owl-test-macro.md) — #[owl_test] procedural macro
- ⬜ [`arch/common-dev-pitfall.md`](arch/common-dev-pitfall.md) — Common development pitfalls
- ⬜ [`arch/cuda-pitfalls.md`](arch/cuda-pitfalls.md) — CUDA/Graph pitfall ledger
- ⬜ [`dev/README.md`](dev/README.md) — Developer entry point (navigation for this directory)
- ⬜ [`dev/config.md`](dev/config.md) — Config chain: intent → derive → preflight → explain
- ⬜ [`dev/testing.md`](dev/testing.md) — Testing workflow v2
- ⬜ [`dev/fixtures.md`](dev/fixtures.md) — Fixtures and gating
- ⬜ [`dev/debug-tools.md`](dev/debug-tools.md) — Debugging quick reference
- ⬜ [`dev/profiling.md`](dev/profiling.md) — Profiler and measurement discipline

## J. Post-mortems and bench reports

- ⬜ [`arch/session-2026-10-14-结案纪要.md`](arch/session-2026-10-14-结案纪要.md) — 2026-10-14 prefill attention kernel swap: closing notes
- ⬜ [`bench/owl.md`](bench/owl.md) — owl speed compilation (phase-1 nn path)
- ⬜ [`bench/qwen38-3090单卡-引擎会战.md`](bench/qwen38-3090单卡-引擎会战.md) — 27B single-card engine battle
- ⬜ [`bench/v1-fi换装-三段战报.md`](bench/v1-fi换装-三段战报.md) — All three attention phases moved to FI
- ⬜ [`bench/w4a16-prefill-bucket-ab-2026-10-21.md`](bench/w4a16-prefill-bucket-ab-2026-10-21.md) — W4A16 × prefill buckets A/B (no benefit, verdict)
- ⬜ [`bench/w4a8-prefill-bucket-ab-2026-10-21.md`](bench/w4a8-prefill-bucket-ab-2026-10-21.md) — W4A8 × prefill buckets A/B
- ⬜ [`bench/gguf-kvmem-decode-nsys-2026-10-21.md`](bench/gguf-kvmem-decode-nsys-2026-10-21.md) — GGUF × KVMem decode verdict (bandwidth wall)
