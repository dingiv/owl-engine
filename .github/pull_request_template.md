## What changed

<!-- One paragraph. If it touches behaviour, say which seam (config / graph / kernels / store / serving). -->

## Why

<!-- The problem, and why this shape of solution. -->

## Gate (how we know it works)

<!-- Required. Name the test / numeric threshold / A-B arm / oracle parity check. -->

- [ ] `cargo check --workspace --tests` passes with zero warnings
- [ ] Relevant unit tests pass (`cargo test -p <crate> --lib`)
- [ ] GPU-gated tests: skipped locally (`skip:` line) **or** run with `OWL_TEST_DEVICE`
- [ ] Docs updated (design volume status section / `docs/README.md` index) if conclusions changed

## Third-party impact

- [ ] No vendored code touched
- [ ] Vendored code touched → registered in `VENDOR.list.md` + direction recorded in that
      directory's `README.md` + `NOTICE` updated if the revision changed

## Notes for reviewers

<!-- Anything subtle: borrow-checker driven shapes, hardware quirks, measurement caveats. -->
