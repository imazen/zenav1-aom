# The directional-predictor gate: the tap bound never bites, and the decline is `up == 1`

**2026-09-10.** A reach census of KB-PERF-4's three runtime gates, taken with
throwaway counters on the shipping-preset cell (`av1-1-b8-01-size-196x196`
mirror-tiled to 1024x1024, cq27, `--cpu-used 3`, bd8 4:2:0). Instrumentation
added, run, and reverted; nothing here changes a line of the encoder.

**Why it was taken.** The s3 re-profile
(`encoder_x86_reprofile_1024_s3_2026-09-10.md`) put **27.7 ms in
`z3_high_scalar`** — genuinely scalar work inside a kernel KB-PERF-4
vectorized — and named the question it could not answer: *is the runtime tap
bound declining, or the shape gate?*

## The census

| kernel | calls | `up == 1` | run too short | **tap bound** | scalar px / total px |
|---|---:|---:|---:|---:|---:|
| z1 | 391,644 | 215,084 (54.9 %) | 35,652 (9.1 %) | **0** | 9.24 M / 24.20 M = 38.2 % |
| z2 (above half) | 1,323,240 | 182,586 (13.8 %) | — | **0** | — |
| z3 | 1,169,710 | 759,520 (64.9 %) | 15,094 (1.3 %) | **0** | 33.60 M / 98.52 M = 34.1 % |

**The tap bound declines NOTHING — 0 of 2,884,594 calls.** KB-PERF-4 derived
`M* = 1023` (every edge tap must satisfy `32M + 16 <= 32767`) and pays an
`O(bw + bh)` scan per block to check it. On bd8 content that scan is a pure cost
with no decline behind it. It cannot simply be deleted — the gate is on the
DATA, not on `bd`, and that is exactly what makes the kernel safe for a bd10
caller and for any future caller of the public entry point — but its price is
now known to be paid for nothing at bd8, which is the only depth the encoder's
own profile reaches.

**The decline is `up == 1`**, the upsampled edge, whose taps sit at stride 2 —
a gather the vector kernel does not do. It is 55 %, 14 % and 65 % of calls.

## A published reach figure is corrected

KB-PERF-4's entry records *"z1 `up==0` 87.4 %, z2 above-half 49.8 %, z3 `up==0`
85.1 %"*. Measured at this preset:

| | recorded | measured at s3 |
|---|---:|---:|
| z1 `up == 0` | 87.4 % | **45.1 %** |
| z2 above-half admitted | 49.8 % | **86.2 %** |
| z3 `up == 0` | 85.1 % | **35.1 %** |

Two of the three are wrong by roughly a factor of two, in **opposite**
directions. Nothing was falsified about the kernel — those figures were measured
on that landing's own corpus and were true of it. What they are not is a
statement about the shipping preset, and they were being read as one. Same shape
as §14 for cost figures: **a reach figure carries its regime too.**

## The lever this names, and its honest size

Addressable = `up == 1` **and** the vectorized dimension already >= 8 (below
that the run is too short for the kernel whatever the stride):

| | calls | pixels | share of that kernel's pixels |
|---|---:|---:|---:|
| z1 | 109,528 | 6.70 M | 27.7 % |
| z3 | 436,556 | 27.33 M | 27.7 % |

**And it needs no gather instruction.** At `up == 1` the taps within a column
are `left[base], left[base+2], left[base+4], …`, each output reading its own
tap and the next — i.e. the *even* outputs of the SAME two-tap run over the
contiguous span, at twice the length. So running `2n - 1` elements through the
existing `two_tap_run` and keeping every second result is bit-exact by
construction, needs no de-interleave, and stays inside the tier-agnostic
vocabulary (no raw AVX2, so NEON and wasm128 get it too).

**It is not 2x the work at the sizes that carry the mass**: a 16-lane kernel
does `bh = 8` in one vector either way (8 live lanes vs 15), so the doubling
only costs from `bh >= 16` up.

Expected value, stated with §14's discount in mind rather than without it: z3 is
40.7 ms and z1 is below the profile's 0.5 % cut, so ~28 % of z3 plus a smaller
z1 share is **~12-16 ms addressable, and the last five levers came in 1.7x to
4.8x optimistic against exactly this kind of estimate.** Treat it as a ~0.2 %
lever — the same class as KB-PERF-32 — not a 0.5 % one.

## THE LEVER WAS BUILT, MEASURED TWICE, AND REJECTED

It is not enough that the identity is correct. Both variants were implemented in
full, gated, bite-proved and banded, and **both are slower**:

| variant | vs base | vs baseB | rounds faster | p |
|---|---:|---:|---:|---:|
| z1 + z3 (`.up1_both.rejected.tsv`, n=30) | **+0.206 %** | +0.239 % | 10/30 | 0.0987 |
| **z3 alone** (`.up1_z3only.rejected.tsv`, n=29) | **+0.342 %** | +0.275 % | **6-7/29** | **0.0023** |

Nulls flat in both bands (−0.03 % and +0.002 %, p=0.86 and p=1.00). The z3-only
band is significant, so this is a measured loss and not an inconclusive one.

**Both were correct.** Output was byte-identical on four cells, the reach pin was
updated to assert the new admission, and the bite proof landed: selecting the
ODD element instead of the even one fails `dr_predict_high_matches_c` and
`highbd_dr_predictors_identical` — both against the **real exported C** — plus
two more, while 442 stay green. So the `up == 1` path was genuinely reached and
genuinely right. It is simply slower than the scalar loop it replaced.

**Why, and it is the third time this project has hit it: the kernel is
STORE-bound, not arithmetic-bound.** `two_tap_run_impl` computes 16 lanes, does
one `res.store(&mut buf)`, and then copies out **per lane**. Running `2n - 1`
elements to keep `n` of them leaves the vector op count unchanged at `bh == 8`
(15 lanes and 8 lanes are both one 16-lane vector) — but it **doubles the
per-lane copy**, and then adds a second strided read to pick the even results.
The arithmetic that the doubling was supposed to be paying for was never the
bottleneck.

The precedents agree and were already on record: KB-PERF-15 found
`txb_init_levels` spending its time in eight single-byte stores around SIMD that
was already correct, and KB-PERF-5 measured PAETH as store-bound and rejected
its half on the same evidence. **Before doubling work to avoid a shuffle, check
whether the kernel's cost is in the shuffle at all.**

**What would actually pay** is removing the per-lane copy — a whole-vector store
straight into the destination — which for z3 means transposing, i.e. libaom's
own `av1_dr_prediction_z3_avx2` structure (compute z1-style into a transposed
buffer, then transpose 8x8 blocks out). That is a real rewrite, not a gate
change, and it is the honest form of this lever.

## Not covered

One box, one content class, one quantizer, `--cpu-used 3`, bd8, x86-64. The
`up == 1` share is a property of `av1_use_intra_edge_upsample`, which fires on
small blocks with small angle deltas, so it is expected to move with speed and
with content; only s3 was measured.
