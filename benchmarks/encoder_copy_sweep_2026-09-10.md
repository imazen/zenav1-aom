# KB-PERF-34/35 — the copy audit: a per-lane store and a borrow-checker `to_vec()`, **−0.48 %** together

**2026-09-10.** Byte-identical; **−0.48 %** at 1024x1024 cq27 `--cpu-used 3`
(−0.492 % at 28/30, p<0.0001, and −0.472 % at 26/30, p=0.0001, against the two
base copies), null +0.058 % (12/30, p=0.36).

Two changes, and the split is measured rather than assumed because the first was
banded on its own before the second existed:

| | effect |
|---|---:|
| `two_tap_run`: per-lane copy → one `copy_from_slice` | −0.22 % (−0.190 % / −0.258 %, p=0.0014 / 0.0003) |
| `txfm_rd_in_plane_uv_p`: two `to_vec()` → two `[i8; 32]` | ≈ **−0.23 pp** (the remainder) |
| **together** | **−0.48 %** |

## KB-PERF-34 — the per-lane store, and it is the direct converse of a rejection

`two_tap_run_impl` computed 16 lanes, stored once into an `i16` buffer, then
copied out **per lane with an `as u16` cast**. It now stores into a `u16` buffer
via `bitcast_u16x16` and does one `copy_from_slice`.

**The bitcast is exact, not convenient**: every output is
`((a0 * (32 - shift) + a1 * shift + 16) >> 5)` with taps bounded by
`I16_TAP_MAX = 1023`, so it is non-negative and `<= 1023` — its `i16` bit
pattern *is* its `u16` value, which is the same equality the `buf[k] as u16`
cast relied on.

**This is the converse of the experiment rejected hours earlier.**
`encoder_dir_pred_reach_audit_2026-09-10.md` doubled this kernel's ARITHMETIC to
avoid a gather and measured **+0.34 %**, diagnosing it as store-bound because
the doubling doubled the per-lane copy. Removing that copy now measures
**−0.22 %**. Same kernel, same box, same day, opposite signs — the diagnosis was
right, and the rejection paid for the fix.

## KB-PERF-35 — the borrow-checker copy KB-PERF-13 left in the chroma twin

    let mut t_above: Vec<i8> = env.above_ctx[pi][..max_blocks_wide].to_vec();
    let mut t_left:  Vec<i8> = env.left_ctx[pi][..max_blocks_high].to_vec();

Two heap allocations per chroma transform-search call, in a **48.9 ms** symbol.
KB-PERF-13 removed exactly this from the LUMA walk and did not carry the fix
across. `MI_SIZE_WIDE_B` / `MI_SIZE_HIGH_B` top out at **32**, so these are at
most 32 bytes each and never needed a heap; every use slices to
`max_blocks_wide` / `max_blocks_high`, so the lengths handed downstream are the
ones the `Vec` form passed.

**~0.23 pp for deleting two allocations** is the largest per-line return this
cycle has produced, and it argues the audit was worth running.

## What the audit found, and what it did NOT

The sweep was measured, not grepped — a frame-pointer profile
(`RUSTFLAGS="-C force-frame-pointers=yes"`, `perf --call-graph fp`), the same
instrument that found KB-PERF-15.

* **`malloc` is 0.25 % of the profile**, and its only two named callers were
  `txfm_rd_in_plane_uv_p` (fixed here) and `txfm_rd_in_plane_intra` (whose
  remaining allocation is the `winners` Vec it RETURNS). **Allocator
  bookkeeping is effectively closed** — KB-PERF-2 and KB-PERF-13 did the rest.
* `__memmove` 2.04 % and `__memset` 1.34 % are the remaining memory cost, and
  their callers are legitimate buffer fills inside transform and txb kernels —
  `av1_fwd_txfm2d_into`, `optimize_txb_core`, `txb_init_levels`,
  `assemble_dir_edges` — not borrow-checker artifacts. KB-PERF-2 explicitly
  built and rejected skipping the scratch re-zero, because the zero-fill is
  what makes scratch reuse byte-identical *by construction*.
* **A third site was found, attempted, and reverted**: `encode_intra.rs` has the
  same `vec![0i8; n]` pattern twice, but those arrays are **returned** inside
  `EncodeIntraPlaneOutcome`, so removing the allocation needs a struct change
  and its consumers. Out of proportion to a 0.25 % class; left as is, named
  here.

## A process failure worth recording

The `encode_intra.rs` half **never compiled**, and three builds silently shipped
a stale binary because the failure was invisible: `cargo build … | grep -E
"^error"` matches nothing, since cargo's error lines begin with ANSI colour
codes. This repo's own memory note (`strip-ansi-before-grepping-cargo`)
documents exactly this, and it was still hit.

It was caught only by a sha256 comparison that came out **identical across three
different source states** — impossible for real codegen. **Compare binary hashes
between arms before trusting a band**, and pipe cargo through
`sed -r 's/\x1b\[[0-9;]*m//g'` before any anchored grep. The consequence here
was benign (one band re-measured a comparison already made, and agreed with it),
but a stale arm in an A/B is a silent wrong answer.

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6) — two sha256-**distinct**
  binaries, checked.
* `-p zenav1-aom-dsp` 446/446 in both dispatch modes.

## Not covered

One box, one content class, `--cpu-used 3`, x86-64. The two changes are banded
together; the split above rests on the earlier two_tap-only band being the same
comparison on the same box the same day, not on a three-arm decomposition.
