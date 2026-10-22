# Security Policy

## Supported versions

This project is in **alpha**. Fixes land on the tip of `main` only.
No backports, no LTS.

## Reporting a vulnerability

**Do not open a public issue.** Email **dingiv456@163.com** with a subject starting
with `[owl-security]`, and include:

- affected version (commit hash) and reproduction steps;
- impact class (read/write, denial of service, information disclosure, escape, …);
- if GPU driver/kernel code is involved: driver version and whether custom kernel
  modules are loaded.

You will get an acknowledgement within **7 days**. Once a fix is released the issue is
disclosed with credit (unless you ask to stay anonymous).

## Known exposure (by design, not vulnerabilities)

| # | Fact | Notes |
|---|---|---|
| 1 | **No HTTP authentication** | The server binds `127.0.0.1` by default. Binding `0.0.0.0` means "anyone on the network can use your GPUs and read your diagnostics" — add a reverse proxy and auth if you need that. |
| 2 | Diagnostics share the **same port** as the API | `/debug/log`, `/debug/metrics`, `/debug/vram` are readable by any client on that port. They contain **no prompt text**, but they do contain runtime ledgers and paths. |
| 3 | No multi-tenant isolation | One engine per process; all sessions share the process and the same GPU memory. |
| 4 | No per-request budget | Bounds come from `runtime.max_seq` / `max_concurrency`; there is no per-user quota. |
| 5 | Configuration is trusted | Paths such as `[model] path` are read as configured. Anyone who can edit the config already has host access. |
| 6 | Diagnostic dumps may write to disk | `prompt_dump`, `pf_stages`, `--perf-dump` can write prompts or ledgers to the paths you give them. Control directory permissions accordingly. |
| 7 | Driver/kernel patches are **not** in this repo | GPU P2P patches and custom kernel modules mentioned in the docs are deployment-environment concerns, out of scope here. |

## Out of scope

- **Vendored upstream code** (FlashInfer / NVIDIA XQA / vLLM-marlin / llama.cpp) —
  report to those projects; this repo only follows them (see `THIRD_PARTY.md`);
- Model behaviour (harmful or incorrect generations);
- OOM / boot rejection caused by misconfiguration — that is the *intended* loud failure;
- Physical access, host root, or driver defects.

## Supply chain

- Dependency inventory and licenses: `THIRD_PARTY.md` (Rust deps section is generated) and
  `Cargo.lock` (committed);
- Vendored CUDA sources are pinned per `THIRD_PARTY.md`; every change must be recorded;
- Admission rule for third-party CUDA libraries: `crates/backends/README.md`
  (**zero tolerance for untracked `cudaMalloc`**).
