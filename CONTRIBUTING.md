# Contributing

First, the frame: this is a **single-node, hardware-specific research engine** in alpha.
Issues and PRs are welcome, but the maintainer works against *design volumes and gates*,
not against a feature backlog.

## 0. License

Released under **MIT OR Apache-2.0**. By submitting a PR you agree that your
contribution is dual-licensed under the same terms (no CLA to sign).

## 1. Before you start

```bash
cargo build -p server                 # needs a local CUDA Toolkit (nvcc)
cargo check --workspace --tests       # zero warnings is a hard requirement
cargo test -p owl-shared --lib        # pure-CPU unit tests (fast)
cargo test -p owl-kernels --lib       # operator face (oracle parity included)
```

GPU tests are gated by `#[owl_test(gpu)]`: without `OWL_TEST_DEVICE` (a BDF such as
`01:00.0`) they print `skip:` and return. **Never let "did not run" look like "passed".**

No GPU / no CUDA toolkit? Two build-script switches make the whole workspace compile
(device paths then refuse at runtime, which is the documented behaviour):

```bash
OWL_CUBIN_SKIP=1 GDN_TRITON_BUILD=0 cargo check --workspace --tests --locked
# asset-free unit tests (the four crates that need no cubins):
OWL_CUBIN_SKIP=1 GDN_TRITON_BUILD=0 cargo test -p owl-sink -p owl-iface -p owl-shared -p owl-store --lib
```

This is exactly what CI does (`.github/workflows/ci.yml`).

## 2. Code style (the three rules that actually matter here)

1. **Fail loudly.** Anything fallible returns `Result` and propagates with `?`.
   Do not write fallible logic as a `()`-returning function; do not swallow errors in a
   silent fallback. Internal invariants in long-running services must not `panic`:
   expose an explicit accessor, bubble up to the main loop, print the error, drop that
   request, and shut down gracefully.
2. **One interface per concept.** No mirrored types, no bypass APIs, no second source
   of truth for the same fact.
3. **Go through the sanctioned I/O doors.** File I/O via `owl_shared::file_loader`;
   logging via `log_*!` (lower layers: `klog_*!` / `OWL_LOG_*` on the C/C++ side);
   timings and counters via `metrics`.

> Long form (in Chinese): `docs/arch/charter.md`, `docs/arch/common-dev-pitfall.md`.

## 3. Diagnostics and probes (read before adding a print)

- **Compile-time**: the `log` feature (on by default). Without it both the Rust macros
  and the C macros vanish — zero cost, byte for byte.
- **Runtime**: `--log-level` plus `/debug/log/tags/...` decide visibility.
- **Division of labour**: booleans under `[probes]` gate **collection / path selection**
  (do we pay an extra D2H, do we insert a stream fence). Do **not** add a boolean just
  to hide a print — that is the log gate's job.
- **C/C++ side**: only `cu/owl_log.h` (`OWL_LOG_*`); device code only `OWL_DEV_PRINTF`.
  No new bare `fprintf`; error-code paths use `OWL_LOG_ERROR`.

The whitelist of allowed bare `eprintln!` (boot rejection, CLI output, pre-logger) is in
`docs/arch/log-framework.md` §8.6.

## 4. Third-party code and CUDA libraries

The iron rule (`crates/backends/README.md`): any library that touches CUDA must be
source-reviewed for **untracked allocations**. A library that calls `cudaMalloc` behind
your back is either not used, or copied in and fixed.

When adding a vendor:

1. add a row to `VENDOR.list.md` (upstream, pinned revision, license, A4 class);
2. add the attribution to `NOTICE`;
3. if you change upstream files, record *what* and *which gate proves it* in that
   directory's `README.md`;
4. **never relicense** vendored files — keep the upstream copyright headers.

## 5. Tests and gates

| Change surface | Minimum verification |
|---|---|
| Pure functions / config / parsing | `cargo test -p <crate> --lib` |
| Operators (kernels) | that family's oracle parity + `cargo test -p owl-kernels --lib` |
| Engine / server | `cargo check --workspace --tests` + relevant gated e2e |
| Docs | links resolve (`docs/README.md` index) + commands are copy-pasteable |

In a PR description, state the **gate / measurement** that justifies the change, not
only what changed.

## 6. Commit messages

- Chinese is fine (the project's body docs are Chinese);
- prefixes: `feat:` / `fix:` / `perf:` / `docs:` / `refactor:` / `chore:`;
- if a change settles or retracts a previous conclusion, update the status section of
  the corresponding design volume in the same commit.

## 7. Code of conduct

This project follows [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Report violations to **dingiv456@163.com**.

## 8. Security

Please report vulnerabilities privately — see [`SECURITY.md`](SECURITY.md).
