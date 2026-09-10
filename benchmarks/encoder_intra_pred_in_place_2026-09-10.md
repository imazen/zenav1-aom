# The intra predictor wrote into a scratch and the caller copied the block back — removed at five call sites: −0.50 % at the shipping preset, −0.79 % at speed 0

**2026-09-10.** Byte-identical; `just gate-landing` 1507/1507 twice.

## The defect

libaom's `av1_predict_intra_block_facade` hands the predictor `pd->dst` — the
prediction lands in the reconstruction plane directly (reconintra.c:1622, the
`ref == dst` form already documented by KB-34). The port's `predict_intra_high`
takes the reference plane and the destination as two slices, so every encoder
call site spelled out:

```rust
walk.pred.clear(); walk.pred.resize(txw * txh, 0);   // a memset
predict_intra_high(recon, txb_off, ref_stride, pred, txw, ..);
for r in 0..txh {                                     // the memcpy
    recon[txb_off + r*ref_stride ..][..txw].copy_from_slice(&pred[r*txw ..][..txw]);
}
```

`benchmarks/encoder_lever_map_s3_2026-09-10.md` annotated `intra_model_rd_y`
(2.45 % of the profile) and found **the hottest single instruction in it is that
copy** — an indirect `callq` with `(dst_ptr, src_ptr, len*2)`, i.e. a per-row
`memcpy` of u16, at the encoder's highest trip count: **per txb PER CANDIDATE
MODE**.

## Why it is sound, and why that is a property of the DSP and not of the caller

Each of the three highbd builders is now split into

* a `plan_*` half taking `&[u16]` — it performs **every** read of the reference
  plane (the above/left/corner neighbours through `assemble_nd_edges` /
  `assemble_dir_edges`, plus the two corner reads of the directional degenerate
  early-out) into owned local arrays, and returns a small descriptor; and
* a `write_*` half taking `&mut [u16]` — it touches no reference pixel.

So all reads complete before the first write and aliasing the two slices cannot
change a value. `predict_intra_high_in_place(buf, off, stride, ..)` is
plan-then-write on one buffer. **The two-slice entry points are unchanged**, so
the decoder — which predicts into a tight scratch and blends — is outside the
blast radius.

## Converted, and deliberately not

| site | converted | why |
|---|---|---|
| `tx_search::intra_model_rd_y` | yes | scratch, memset AND copy all gone; the subtract reads the plane at `ref_stride` |
| `intra_uv_rd::intra_model_rd_uv` | yes | the recon→tight snapshot deleted; subtract reads the plane |
| `intra_uv_rd::predict_uv_txb` (CfL DC arm + plain arm) | yes | the `pred` PARAMETER is gone; both callers already re-read the plane |
| `nonrd_pickmode::nonrd_pick_intra_mode` | yes | SAD prune and subtract both read the plane strided |
| `tx_search::txfm_rd_in_plane_intra` | **no** | its `pred` feeds `dist_block_px_domain` per TX TYPE, which needs a tight `w*h` buffer — in place would move the copy, not remove it |
| `encode_intra::encode_intra_block_plane_{y,uv}` | **no** | same reason (`tight`/reconstruction base) |

That distinction is the interesting half: the win is only available where the
prediction is consumed **strided or not at all**. Where a tight buffer is
genuinely needed downstream, predicting in place just relocates the memcpy.

## Measured

Two sha256-distinct binaries from one tree, arms ROTATED per round, a
same-binary null arm in every band. Byte-length identity checked on four cells
before timing: 1024x1024 s3 **40,237**, 1024x1024 s0 39,694, 512x512 s0 10,912,
512x512 s6 11,961 — identical on both arms.

| cell | base | new | paired | rounds | p | null |
|---|---:|---:|---:|---:|---:|---:|
| 1024x1024 cq27 **s3** (the shipping preset) | 3009.33 ms | 2992.94 ms | **−0.497 %** | 28/30 | 8.7e-07 | +0.080 % (p=0.36) |
| 512x512 cq27 s0 | 2976.28 ms | 2955.30 ms | **−0.793 %** | 24/24 | 1.2e-07 | +0.042 % (p=0.84) |

The s3 base arm reads 3009.33 against the recorded HEAD median of 3011.65
(0.08 % apart), so the delta is read against a live control.

**It pays MORE at speed 0 than at s3, and the mechanism predicts that**: the
removed copy is per txb per CANDIDATE mode, and speed 0 evaluates far more
candidates per transform block than `--cpu-used 3` does. That is the opposite
of the speed behaviour of the per-transform-call setup class
(`encoder_ratio_vs_speed_2026-09-09.md`), so it is a genuinely different
overhead class, not more of the same one.

## Where it sits in the session's rule

`perf-removal-pays-reshaping-does-not`: this **removes** work on a hot inner
path — one `memset` and one strided row copy per prediction, plus the scratch
`Vec` traffic — and adds none. It is the second-largest landing of the cycle
after KB-PERF-53's −0.827 %, and it is the third consecutive confirmation that
removal is what pays.

It also sharpens the necessary-not-sufficient caveat with a POSITIVE case:
KB-PERF-51 removed two copies on a MINORITY of txbs and measured null
(−0.147 % against a −0.077 % null). Same class of change, same file; the
difference is entirely trip count. **Cost the trip count, not the bytes.**

## Gate

New differential `predict_intra_in_place_diff` (aom-dsp `tests/all`). The claim
under test is aliasing-freedom, so the assertion is on the **whole plane**, not
the block: for every mode x angle-delta x filter-intra mode x 19 tx sizes x 6
availability combinations x bd {8,10,12}, predicting in place must leave the
plane byte-identical to predict-then-copy, and the block must still equal the
REAL exported C predictor (`ref_hbd_predict_intra`). Two of the six availability
combinations exist to reach the directional degenerate early-out — the only arm
that reads `recon` outside the edge assembly.

`just gate-landing`: `test-next` **1507/1507**, `test-next-scalar` **1507/1507**
(1506 + this file's one test), `census-gate` 4/4, `test-whereat` 4/4.
