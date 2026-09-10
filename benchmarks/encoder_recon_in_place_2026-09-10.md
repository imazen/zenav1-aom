# The intra search reconstructed through a scratch buffer it did not need — removed, and it measures NULL

**2026-09-10.** Byte-identical, strictly less work, and **not significant**:
**−0.147 % against a same-binary null of −0.077 %, both 15/24, p = 0.31.**
Reported as a null, not a win.

## The defect, and it is real

Both intra transform-search walks — `tx_search::txfm_rd_in_plane_intra` (luma)
and `intra_uv_rd::txfm_rd_in_plane_uv_p` (chroma) — reconstructed a winning txb
like this:

```rust
tight.clear();
tight.extend_from_slice(pred);              // whole-block copy IN
av1_inverse_transform_add(&dqcoeff, tight, txw, ...);
for r in 0..txh {                           // strided row copy OUT
    recon[txb_off + r*stride .. +txw].copy_from_slice(&tight[r*txw .. +txw]);
}
```

Both copies are avoidable, because **`recon[txb_off..]` already holds exactly
that txb's prediction** and `av1_inverse_transform_add` already takes a
destination STRIDE:

* luma writes the prediction into `recon` a few lines above, under its own
  comment *"The C facade writes the prediction into dst (the recon plane)"*;
* chroma writes it into `recon` **directly** in both arms (the palette fill and
  `predict_uv_txb`), and its `pred` buffer is a copy taken back OUT of `recon`
  for the subtract.

C does the in-place form: `av1_inverse_transform_block` reconstructs straight
into `pd->dst.buf` at `dst_stride`. So this is a fidelity alignment as well as a
copy removal.

**Safety was checked, not assumed:** inside each loop body `recon` is touched at
exactly two points — the prediction store and this reconstruct — and
`search_tx_type_intra_into` does not take `recon` at all.

## Why it was expected to pay, and why that was wrong

A frame-pointer profile at the shipping preset (1024x1024 cq27 `--cpu-used 3`)
attributes **100 %** of `__memmove_avx512` to these two functions:

| caller | share of memmove | of profile | ms |
|---|---:|---:|---:|
| `txfm_rd_in_plane_intra` | 72.1 % | 1.01 % | ~30.7 |
| `txfm_rd_in_plane_uv_p` | 27.9 % | 0.39 % | ~11.9 |

**That is a CALLER attribution, not a copy attribution — and the difference is
the whole result.** The 1.01 % contains three different copies:

1. the prediction store into `recon`, which runs on **every** txb and is
   load-bearing (C does it too);
2. `winners` growth;
3. the `tight` round trip — which runs only when `best_eob > 0 && not_last_txb`,
   a minority of txbs.

Only (3) was removable, and it is evidently a small fraction of the 1.01 %.

**This is playbook §14 in a form the log has not recorded before.** Every
previous instance was a ranked STAGE or CLASS table being optimistic about a
lever inside it. This one is a *frame-pointer caller attribution* — the very
instrument introduced to make memory work actionable (KB-PERF-15) and used
successfully twice since (KB-PERF-35, KB-PERF-37) — being optimistic in exactly
the same way, because a function can contain several copies with different trip
counts and the profile names only the function.

**Before costing a lever off a frame-pointer caller share, count how many
distinct copies live in that caller and which of them your change actually
removes.**

## The measurement

Two sha256-distinct binaries from one tree, arms rotated each round, same-binary
null, 1024x1024 cq27 `--cpu-used 3`, 4 reps per invocation.

| band | effect | rounds | p |
|---|---:|---:|---:|
| luma only (n=15, cut short) | −0.107 % | 9/15 | 0.61 |
| **luma + chroma (n=24)** | **−0.147 %** | **15/24** | **0.31** |
| null, same binary both sides | −0.077 % | 15/24 | 0.31 |

The effect and the null are the same size and the same sign. Nothing is resolved
here.

**Byte-identical on all four standard cells** — 40,237 (1024 s3), 39,694
(1024 s0), 10,912 (512 s0), 11,961 (512 s6).

## Why it is KEPT anyway, and on what precedent

KB-PERF-44's rule: a change that **strictly removes work and adds none** is kept
even at a non-significant band, where a change that ADDS a mechanism (the
`with_capacity` hint's two divisions, the first TLS pool's lookup) is not. This
removes one whole-block copy and one strided row copy per reconstructed txb and
adds nothing — it cannot be slower in principle — and it moves the port onto C's
own in-place structure. It is recorded as a **null**, not as a −0.147 % win, and
nothing downstream should quote it as one.

## Not covered

One box, one content class, one quantizer, `--cpu-used 3`, x86-64. The two
larger memmove consumers inside the same callers — the per-txb prediction store
and `winners` growth — are untouched, and the prediction store is the one that
runs on every txb. Removing IT means letting `predict_intra_high` write straight
into `recon` at `ref_stride` (C's `ref == dst` form, which KB-34 already
documents), a DSP signature change rather than a caller change.
