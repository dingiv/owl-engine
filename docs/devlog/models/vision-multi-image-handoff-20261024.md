# Vision multi-image handoff (2026-10-24)

> Status: multi-image + FA2 convergence is merged into `main`.
> Remaining blocker: **4B Q4 + MTP + multi-image + long generation** fails in
> `restore_spec_snap` after the main merge. This document is the starting point
> for the next worktree.

## 1. Current state

- `main` tip at handoff: `f8476999`
- Worktree used for the vision work: `<worktree>` (`fix/vision-stability`,
  fast-forwarded to the same tip)
- Merged commits:
  - `b5ca7ebb` — single-request multi-image (gate, m-RoPE, chunk-local rope pos)
  - `fcaa0cdb` — vision attention unified on vLLM FA2 (hd64/hdim64, hd72/hdim96);
    FI ragged retained but no longer selected
  - `b9292295` — vision docs consolidated
  - `9608f21b` — merge current `main` into the vision branch
  - `f8476999` — post-merge 4B MTP blocker note
- No push has been made (`main` is ahead of `origin/main`).

The vision path is **single-request multi-image**; it is **not** full
multi-image batching. `max_batch > 1` with images is still rejected. Image
prefix caching / deduplication of repeated history images is also not
implemented.

## 2. What is already verified

| Config | Request | Result |
|---|---|---|
| 4B Q4 / 3080 / spec off | 2×1024, 4×1024, 2×1408 | pass; 2×1408 ~2.2s, 4×1024 ~0.7s |
| 4B Q4 / 3080 / spec off | single 1024/1408 | pass |
| 27B Q4 / 3090 / spec off | 2×1024, 2×1408 | pass; 2×1408 ~4.5s |
| 27B AWQ / 3090 / MTP d3 | 2×1024, 64 tok | pass |
| 27B AWQ / 3090 / DFlash2 d7 | 2×1024, 64 tok | pass (windowed-spec may degrade to bare decode) |
| 4B Q4 / 3080 / MTP d3 | 2×1024, 16 tok | pass |
| 4B Q4 / 3080 / MTP d3 | 2×1024, 64 tok | **fails after main merge** |

The 4B MTP failure is recorded in `vision-wiring.md` §10.5. Do not claim the
4B MTP multi-image long-output case is fixed.

## 3. The blocker

### Symptom

Profile: `profiles/qwen35_4b-3090-q4km-mtp.toml` with `vision=true` and the
4B mmproj. Request: two 1024px images, `max_tokens=64`, temperature 0.

- 16 token: passes.
- 64 token: fails after several spec rounds with:

```text
engine: copy_batch: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, ...)
```

- Subsequent requests on the same CUDA context may fail with
  `memset_zero: ... ILLEGAL_ADDRESS` (sticky context error).
- Single-image 64-token: passes.
- 4B spec-off multi-image: passes.
- 27B MTP multi-image 64-token: passes.
- 27B DFlash2 multi-image 64-token: passes.
- Text-only 617-token MTP 64-token: passes.

Temporary diagnostics showed the last operation was `restore_spec_snap`
(96 copy ranges; 24 GDN layers × 4 state tensors). The failure did not occur
in the FA2 vision kernel itself.

### Important history

- Before merging current `main`, the same 4B MTP + 2-image + 64-token request
  passed 3/3 on the vision branch with FA2.
- After merging `main` (`b9a88841` plus image-evict / message-boundary changes),
  it fails 3/3.

The current `main` changes that are candidates for the interaction:

- message-boundary snapshots (`b9a88841`)
- image eviction / `TurnImage` / `image_pad_law` (`0d354d9f` and related)
- KVMem/DFlash2 snapshot changes merged in the same wave

The 4B MTP test profile has `snap_msg_boundary=false`, `image_evict=false`, and
`snap_max=0`, so these paths should be no-ops. Verify rather than assuming.

## 4. Code map

| File | Relevant entry |
|---|---|
| `crates/engine/src/state.rs` | `capture_spec_snap`, `restore_spec_snap`, `capture_snap`, `restore_snap` |
| `crates/engine/src/exec/spec.rs` | `spec` round; `ensure_spec_snap`; snapshot capture/restore around fold/rollback |
| `crates/engine/src/running.rs` | image gate, `TurnImage`, `image_pad_law` |
| `crates/engine/src/img_evict.rs` | `pad_image_spans`, `image_runs`, text delete detection |
| `crates/engine/src/exec/phases.rs` | `prepare_vlm_rank` (multi-image tower preparation) |
| `crates/engine/src/msg_boundary.rs` | message-boundary snap points (main change) |
| `crates/engine/src/turn.rs` | `TurnImage`, `ActiveTurn.img_spans/img_pads` |
| `crates/models/src/vlm.rs` | multi-grid `Vlm::inputs`; chunk-local pos |
| `crates/models/src/layers/mrope.rs` | adjacent image/video grid segmentation |
| `crates/kernels/src/server/flashinfer.rs` | FA2 varlen, cu cache, legacy FI ragged |
| `crates/kernels/src/driver.rs` | `vis.attn_full` selects FA2 for hd64/72 |

`restore_spec_snap` builds `96` `(src, src_off, dst, dst_off, bytes)` tuples and
calls one `face.copy_batch`; `capture_spec_snap` is the reverse. The copy sizes
are based on `gdn_slots` and per-layer tensor element counts; `gdn_slots=1` for
this profile.

## 5. First experiments (in order)

Run only the 4B MTP profile; do not start the full suite. Stop the server by
the exact PID from `nvidia-smi --query-compute-apps`; never `pkill -f`.

### 5.1 Check whether it is spec-specific

Use the same two-image request and vary one knob at a time:

```text
--set spec.depth=1
--set spec.depth=2
--set spec.depth=3
--set spec.adaptive_depth=false
--set spec.mtp_window=0
--set spec.mtp_window_min_ctx=0
--set spec.lookup=false          # only if applicable
--set spec.depth=0               # control: should pass
```

Question: does the failure begin at a particular round / depth / accepted-token
pattern, or only after enough rounds?

### 5.2 Check main's no-op assumptions

For the failing profile verify:

```text
--print-config | grep -E 'image_evict|snap_msg|prefix_cache|snap_max|snap_decode'
```

Then compare the prompt token IDs before/after `image_pad_law` for the two-image
request. `image_evict=false` should leave `ids` byte-identical and `pads` empty.
If not, the multi-image `TurnImage` path is the regression.

Also test:

```text
--set pool.image_evict=false
--set pool.snap_msg_boundary=false
--set pool.prefix_cache=false
--set pool.snap_max=0
```

Do not assume the profile's inherited values; `--set` one at a time.

### 5.3 Instrument the snapshot boundary

Temporarily add tag logging around the four `copy_batch` call sites:

```text
capture_snap / restore_snap / capture_spec_snap / restore_spec_snap
```

Log `copies.len()`, `src.id`, `dst.id`, `src.byte_offset`, `dst.byte_offset`,
`bytes`, and `gdn_slot`. Remove the logs before commit. The existing temporary
diagnostic (removed) showed `capture_spec_snap` / `restore_spec_snap` at
`n=96`; the failure is on a restore after multiple rounds.

Check:

- Are source/destination block IDs still live? (`resolve` / ledger)
- Are the source/destination offsets and lengths inside the block accounting?
- Does the `spec_snap` buffer get freed or reused by any `Turn` cleanup?
- Does a multi-image request allocate larger `merged` / intermediate blocks that
  change the pool layout, exposing a stale pointer?

### 5.4 Compare 4B and 27B geometry

Both use MTP d3, but 4B has 24 GDN layers and 27B has 64; spec snapshot sizes
and `gdn_slots` differ. Check `spec_snap` allocation and `gdn` row sizes in
both. A 4B-only failure may be a size/rounding bug rather than a multi-image
bug.

## 6. Reproduction recipe

Use a clean worktree and an empty GPU. The worktree has `models` symlinked and
`testdata` prepared; see the worktree README.

```bash
cd <worktree>
./target/release/server \
  -p profiles/qwen35_4b-3090-q4km-mtp.toml \
  --set 'parallel.device=["<3080-bdf>"]' \
  --set 'model.vision=true' \
  --set 'model.vision_path="../models/bartowski/Qwen_Qwen3.5-4B-GGUF/mmproj-Qwen_Qwen3.5-4B-f16.gguf"' \
  --set 'server.bind="127.0.0.1:8142"' \
  --log-level info
```

Wait for `就绪`; do not sleep more than 30 seconds.

Minimal client (two images, 64 tokens):

```python
import base64, json, urllib.request

def data_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()

body = {
    "model": "qwen3.5-4b-gguf-vlm",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url("assets/images/bird.png")}},
        {"type": "image_url", "image_url": {"url": data_url("assets/images/cat.png")}},
        {"type": "text", "text": "What is in these images? Answer briefly."},
    ]}],
    "max_tokens": 64,
    "temperature": 0,
}
req = urllib.request.Request(
    "http://127.0.0.1:8142/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    print(r.read().decode())
```

Expected current result: 500 with `copy_batch` / `CUDA_ERROR_ILLEGAL_ADDRESS`.
Use the same body with `max_tokens=16` as a control.

For 27B comparison, use `profiles/qwen38_27b-3090-w4a16-mtp.toml` (3090 BDF,
separate port) with `vision=true`; the 4B failure should not appear.

## 7. Likely fix directions

Do not start with a broad refactor.

1. **If `image_pad_law` is non-identical for multi-image + `image_evict=false`**:
   early-return `ids`/empty spans/empty pads before `pad_image_spans`, or make
   the no-evict path explicitly identity. This keeps multi-image requests from
   entering image-eviction code.
2. **If `spec_snap` lifecycle is the issue**: make the spec snapshot own a
   stable buffer for the whole turn; verify no `Turn`-owned block is kept by
   mistake. Compare against DFlash2, which passes.
3. **If main's message-boundary snapshot is the trigger**: isolate by disabling
   the snapshot path for image/MTP turns (only if it is truly a no-op for this
   profile); do not change text-only behavior.
4. **If the fault is a 4B MTP geometry bug**: fix the `spec_snap` sizing/offset
   and add a 64/128-token regression; do not paper over it by disabling MTP.

A guard that rejects multi-image + MTP is only a last-resort safety net. The
preferred outcome is a working 4B MTP multi-image path.

## 8. Definition of done

- 4B Q4 + MTP d3 + 2 images:
  - 64 tokens × 3 runs pass;
  - 128 tokens × 1 run pass;
  - no `copy_batch` / `memset_zero` / Xid 31.
- 4B single-image 64 tokens passes.
- 4B spec-off multi-image 2×1408 still passes.
- 27B MTP and DFlash2 multi-image 64 tokens still pass.
- Add the smallest regression test/probe that would have caught the failure;
  keep the full suite out of the loop until the single case is green.
- Update `vision-wiring.md` §10.5 and the index; commit only after the user asks.

## 9. Guardrails

- Do not touch text decode/propose/verify operators for this bug; the current
  evidence points to the spec snapshot / multi-image integration.
- Do not remove the FA2 hdim64/96 routing; it is the selected vision path.
- FI ragged is legacy and unselected; do not re-enable it as a workaround
  without measurement.
- Keep `max_batch > 1` image rejection.
- Use exact PIDs from `nvidia-smi --query-compute-apps`; no `pkill -f`.
- Check `nvidia-smi` and `/proc/vmstat` before starting; the workspace is shared.
- Do not run `cargo test --workspace` as a first step.
