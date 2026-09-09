# Re-profile after KB-PERF-6, and the lever the old ranking could not see

**2026-09-08, second x86-64 profile.** Same box, cell and method as
`benchmarks/encoder_x86_profile_2026-09-08.md` (AMD Ryzen 9 7900X, Linux, glibc,
AVX2/AVX-512; `av1-1-b8-01-size-196x196` cropped 192x192, cq27, bd8 4:2:0,
speed 0; both arms emit the same 1177 bytes). Taken because that profile's
ranked table predates KB-PERF-6's three loop-restoration landings, and
DIFFERENTIAL_PLAYBOOK §14 says re-profile before trusting a ranked lever.

```
perf record -F 999 -g -- ./target/release/examples/eprof_x86 <port|c> 192 192 27 0 9
```

## Where the ratio is

**port 453.75 ms vs libaom 178.52 ms = 2.542x**, gap **+275.2 ms** (was 2.69x /
+304.5 ms before KB-PERF-6). Against Gate 3's <= 1.5x bar, closing this needs
~178 ms of the 275 — i.e. several more landings, not one.

## Stage table — full-tail attribution

**Corrected 2026-09-08, same day, before this record was relied on.** The first
version of this table read symbols off a `--percent-limit 0.8` report and
estimated the C side by eye. That underestimated C badly: libaom's inverse
transform is a long tail of size-specialised symbols
(`lowbd_inv_txfm2d_add_*_ssse3/avx2`, `idct4_w4_sse2`, ...) each under 0.2 %,
summing to 5.5 % — where the first pass guessed 1.7 %. Both tables are below so
the error is legible; **the corrected one is the one to rank from.** It
categorises 99.7 % (port) and 99.5 % (C) of samples at `--percent-limit 0.01`.

| stage | port ms | C ms | ratio | **gap ms** | **% of gap** |
|---|---:|---:|---:|---:|---:|
| **other (RD drivers, tx_search, ctx)** | 113.8 | 48.0 | 2.4 | **+65.8** | **23.9 %** |
| **transform FORWARD** | 72.7 | 16.4 | 4.4 | **+56.3** | **20.4 %** |
| **transform INVERSE** | 66.7 | 15.7 | 4.2 | **+51.0** | **18.5 %** |
| loop-restoration search | 57.8 | 8.8 | 6.6 | +49.0 | 17.8 % |
| intra predictors | 36.8 | 16.2 | 2.3 | +20.6 | 7.5 % |
| coeff trellis (`optimize_txb`) | 71.9 | 61.0 | **1.18** | +10.9 | 4.0 % |
| memset/memcpy | 12.5 | 1.9 | 6.6 | +10.6 | 3.8 % |
| allocation | 8.9 | 1.1 | 8.2 | +7.8 | 2.8 % |
| quantize | 11.2 | 8.5 | 1.3 | +2.7 | 1.0 % |

Gaps sum to 274.7 ms against a measured 275.2 — the attribution is complete, not
a top-N sample.

**Two things the first pass got wrong and this one fixes:** the FORWARD
transform is larger than the inverse (+56.3 vs +51.0), and the largest single
bucket is the RD drivers, which the earlier ranked table never surfaced at all.

`optimize_txb_core` is STILL the port's largest single symbol (12.0 %) at a
**1.18x** stage ratio. The original profile's finding #1 holds and has
strengthened: a session that optimised the port's top symbol would spend itself
on the one stage that is already at parity.

## The structural root under half the gap: bd8 runs on the highbd path

The three transform/variance rows are not three problems. **The encoder holds
its planes as `u16` at every bit depth including bd8, and therefore runs the
highbd (u16 / i32-lane) kernels where libaom runs a fully specialised lowbd
(u8 / i16-lane) path** — `av1_lowbd_fwd_txfm_avx2`,
`lowbd_inv_txfm2d_add_no_identity_avx2`, `aom_variance*_sse2`. Measured
consequences, all one cause:

| item | gap ms |
|---|---:|
| transform forward | +56.3 |
| transform inverse | +51.0 |
| `highbd_variance` + `highbd_variance64_impl` at bd8 | +7.6 |

**~115 ms, or 42 % of the whole gap, from one design decision.** KB-PERF-1's
Darwin profile named the same root ("the port runs the highbd forward transform
+ intra predictors where libaom runs `lowbd_fwd_txfm2d_*_neon`"), and KB-PERF-3
and KB-PERF-5 are both partial unwindings of it. It is the largest coherent
programme available, and no single landing closes it.

## The `other` bucket, decomposed

Not one lever — the port's drivers cost 2-10x C's across many small functions,
which is the price of a generic driver where libaom has a specialised one:

| port symbol | ms | nearest C | ms |
|---|---:|---|---:|
| `tx_search::search_tx_type_intra_into` | 18.9 | `search_tx_type` | 10.9 |
| `tx_search::txfm_rd_in_plane_intra` | 12.7 | `av1_txfm_rd_in_plane` | 1.6 |
| `tx_search::intra_model_rd_y` | 9.2 | (`block_rd_txfm` etc.) | ~1.2 |
| `xform_quant_into` + `..._optimize_split_into` | 11.1 | (below cut) | — |
| `txb::simd::txb_init_levels` | 6.5 | `av1_txb_init_levels_avx2` | 2.0 |
| `dist::highbd_variance` + `..._variance64_impl` | 7.6 | (lowbd variance) | — |
| `dist::hadamard::hadamard_col8` | 5.0 | (below cut) | — |
| `txb::entropy_ctx::get_txb_ctx` | 4.8 | `av1_get_txb_entropy_context` | 3.2 |

## The finding: the i16 inverse is wired only to the DECODER's path

KB-PERF-3 built and audited i16-lane inverse kernels for the DCT family, and
KB-PERF-3's own residual note reads *"of the port's encode-side inverse, 79 %
still runs the wide i32 path — only the DCT family passed the i16 audit"*. That
attributes the gap to the AUDIT. **Measured here, the attribution is wrong.**

A reach probe on `try_inv_col_pass_u8` — the entry that consults
`lowbd16::inv_kernel_i16` — records **zero calls** on this cell. Not "declines
on tx type": never reached. The reason is structural rather than numeric:

* `av1_inv_txfm2d_add_u8_into` (`inv_txfm2d.rs:385`) is the **u8** output path
  and is the only caller of `try_inv_col_pass_u8`, hence the only path that can
  reach the i16 column kernels;
* `av1_inv_txfm2d_add_into` (`:235`) is the **u16** output path, and it is what
  the encoder uses — the encoder holds planes as `u16` at every bit depth,
  including bd8 (the same fact KB-PERF-4 records when it scopes the directional
  predictors as an encoder-only lever).

So the encode-side inverse column pass runs `inv_col_pass_core` at **i32 lanes,
100 % of the time**, and not because of the audit. The ROW pass does not have
this gap — `try_inv_row_pass` already reaches the i16 kernels and is
pixel-type-agnostic, which is why only the column side shows.

## What the existing audited kernels would already cover

Counted on the encoder's own `try_inv_col_pass`, by column kernel, over 400,001
calls (probe removed before landing):

| family | calls | % calls | pixels | **% pixels** | i16-audited today |
|---|---:|---:|---:|---:|---|
| dct4 | 104,804 | 26.2 | 2,092,912 | 8.9 | yes |
| dct8 | 71,936 | 18.0 | 4,782,848 | 20.4 | yes |
| dct16 | 22,991 | 5.7 | 4,667,264 | 19.9 | yes |
| dct32 | 2,424 | 0.6 | 1,641,984 | 7.0 | yes |
| dct64 | 131 | 0.0 | 242,688 | 1.0 | yes |
| **DCT total** | | **50.5** | | **58.2** | **yes** |
| adst4 | 89,371 | 22.3 | 1,767,104 | 7.5 | no |
| adst8 | 55,383 | 13.8 | 3,594,784 | 15.3 | no |
| adst16 | 16,969 | 4.2 | 3,190,592 | 13.6 | no |
| **ADST total** | | **40.3** | | **36.4** | no |
| identity 4/8/16 | | 9.1 | | 6.5 | no |

**58.2 % of the column pass's pixel work is already audited i16-safe.** Adding a
u16-output twin of `inv_col_pass_u8_i16` needs no new audit — the kernels are
the same, the bd8 preconditions the audit rests on (`col_clamp == 16`,
`shift1_bit == 4`, `stage_range == 16`) are the same at bd8 whatever the pixel
type, and only the final `highbd_clip_pixel_add` store differs. The row pass is
the worked precedent: it sign-extend-stores into the same row-major i32 `buf`,
so an i16 column pass composes with it unchanged.

Extending the audit to ADST would add a further 36.4 %; that is a separate and
larger piece of work (the forward audit REJECTED `fadst4` outright, so an ADST
inverse result cannot be assumed).

## Honest bound on the lever

The column side is ~6.5 % of port time (`inv_col_pass_core` self 4.06 % plus its
share of `run_inv1d` 4.22 %), so ~29 ms of the inverse's 66.7. At 58 % coverage
and roughly double the lane width, the arithmetic suggests **-8 to -10 ms, i.e.
~3-4 % of the gap** — worth landing, not a step change, and NOT the whole of the
+51 ms inverse row. Stated up front because §14 has fired four times on this
codebase (18x, 5x, 13x optimistic) and once in the other direction (KB-PERF-5
held to 8 %).

**A redundancy hypothesis was tested and REFUTED, which is why the lane-width
reading survives.** The first corrected reading put the inverse at 53 ms port vs
~3 ms C — a 17x ratio that lane width cannot explain, and which would have meant
the port simply performs more inverse transforms (the KB-PERF-1 shape). Summing
C's full symbol tail instead gives 15.7 ms and a ratio of **4.2x**, which is
what a generic i32-lane driver against libaom's size-specialised i16 path should
cost. No redundancy is indicated. The hypothesis was worth one measurement
because a redundancy finding would have been worth an order of magnitude more
than a lane-width one.

## THE LEVER WAS BUILT AND MEASURED NULL — do not rebuild it

Same day, after the analysis above. **Built in full, measured, reverted.**

`lowbd16::inv_col_pass_u16_i16` — the u16-pixel twin of `inv_col_pass_u8_i16`,
derived mechanically from the u8 source so the gather logic could not drift,
with the store widened to `u16` and clamped at 255 — plus the arm in
`try_inv_col_pass` gated on `bd == 8` and the same structural constants the u8
arm asserts.

**It is CORRECT.** `self_contained_key_frame` is 9/9 with it live, i.e. all
**427 cells still byte-identical to real aomenc**, and the whole `aom-dsp`
suite is green (25 binaries). So this is a null result, not a broken one.

**It is not FASTER.** Interleaved A/B, two binaries from one tree, the profile
cell:

| variant | paired median vs base | rounds faster |
|---|---:|---:|
| as first written | **+1.98 %** | 0 / 8 |
| gated `col_n >= 16` | +0.4 % | 2 / 6 |
| store restructured (full-group / tail split) | **-0.03 %** | 5 / 8 |

The +1.98 % was a self-inflicted store: a per-lane `if j < active` branch across
all 16 lanes on every row, where the i32 core splits the full and tail groups
instead. Fixing it recovered the loss and delivered nothing.

**Why, and it is KB-PERF-3's own lesson arriving from the other side.** That
entry measured its half-batch forward extension at -0.006 % and recorded the
mechanism: *"an `i16x16` and an `i32x8` are both 256 bits, so the op count is
equal."* The same holds here. The i16 win in KB-PERF-3's shipped work came from
`fbtf16` being cheaper than `prims::hb` (a widening madd instead of a widen-to-
i64 round trip), and the INVERSE kernels do not get that: their 17-bit
butterfly transients live in `prims16`'s two-domain `P32` representation, which
costs about what staying in i32 costs. The lane count changes; the work does
not.

**What this retires.** The bound stated above (-8 to -10 ms, ~3-4 % of the gap)
is REFUTED — measured 0. The 58.2 %-coverage figure is still correct and still
irrelevant, because coverage was never the binding constraint. **The transform
inverse row (+51 ms) needs a different idea than lane width**, and the same
doubt now attaches to the forward row (+56.3 ms), whose i16 work KB-PERF-3
measured at -2.0 to -2.6 % — real, but an order of magnitude short of what the
row would need.

To re-derive rather than re-invent: the twin is a mechanical transform of
`inv_col_pass_u8_i16` (replace `u8` -> `u16` in the output type, and the
`add_store_u8` full-group store with a `from_array`/`clamp`/`to_array` at
ceiling 255), plus a `bd == 8` arm in `try_inv_col_pass` ahead of the
`inv_kernel` lookup. It took under an hour; the measurement is the expensive
part and it is above.

## SIZE CONTROL: does any of this survive real image sizes?

Everything above was measured on a **192x192** cell. At bd8 4:2:0 held as `u16`
that is ~108 KiB of planes — **L2-resident on this box** (L1d 32 KiB/core, L2
1 MiB/core, L3 64 MiB shared). Memory traffic is therefore nearly free there,
and a lever that halves traffic — any lane-width change — cannot show its win in
that regime. So the null above could have been an artefact of the cell, and the
STAGE RANKING could have been mis-ordered for the same reason.

`eprof_x86` now mirror-tiles its 196x196 source (the recipe every HD gate here
uses) so it can be driven at real sizes. Measured:

### 1. The i16 lever is null at every size — the cache regime does not rescue it

| cell | planes | regime | base | i16 | delta | faster |
|---|---:|---|---:|---:|---:|---:|
| 192x192 s6 | 108 KiB | L2 | 18.99 ms | 18.54 ms | -2.37 % | — |
| 512x512 s6 | 768 KiB | L2 | 129.56 ms | 129.92 ms | +0.28 % | — |
| 1024x1024 s6 | 3 MiB | L3 | 482.82 ms | 483.71 ms | +0.18 % | — |
| **512x512 s0** | 768 KiB | L2 | 3862.3 ms | 3886.4 ms | **+0.64 %** | 1/5 |
| **1024x1024 s0** | 3 MiB | L3 | 10927.7 ms | 10956.7 ms | **+0.22 %** | 1/5 |

Both arms emit identical bytes at every size (1473 at 192 s6, 11961 at 512 s6,
39694 at 1024 s0). If anything the sign runs the WRONG way for the cache
hypothesis — the only negative reading is the smallest, most L2-resident cell,
at n=3. **The revert stands, now on evidence spanning a 28x pixel count rather
than one cell.**

### 2. The stage ranking DOES transfer — the cheap cell is a valid ranking tool

Full-tail attribution at 1024x1024 speed 0 (port 10904.0 ms vs C 4156.4 ms),
against the 192x192 table:

| stage | 192² gap | % | 1024² gap | % | shift |
|---|---:|---:|---:|---:|---:|
| other (RD drivers) | 65.8 | 23.9 % | 1532.2 | 22.7 % | -1.2 pp |
| loop-restoration | 49.0 | 17.8 % | 1405.5 | 20.8 % | **+3.0 pp** |
| transform forward | 56.3 | 20.4 % | 1251.9 | 18.6 % | -1.9 pp |
| transform inverse | 51.0 | 18.5 % | 1053.2 | 15.6 % | -2.9 pp |
| intra predictors | 20.6 | 7.5 % | 572.9 | 8.5 % | +1.0 pp |
| memset/memcpy | 10.6 | 3.8 % | 286.4 | 4.2 % | +0.4 pp |
| trellis | 10.9 | 4.0 % | 265.3 | 3.9 % | -0.0 pp |
| allocation | 7.8 | 2.8 % | 216.7 | 3.2 % | +0.4 pp |
| quantize | 2.7 | 1.0 % | 136.3 | 2.0 % | +1.0 pp |

**Maximum shift 3.0 pp, no reordering of the top four.** The two transform rows
shrink slightly at scale and loop-restoration grows — consistent with the
transform working on small tiles that stay cached however big the frame is,
while restoration sweeps whole planes. This licenses ranking from the 192x192
cell, which matters: **0.45 s per speed-0 rep against 11 s, a 24x iteration
difference.**

### 3. The small cell is mildly OPTIMISTIC about the ratio — quote both

**2.542x at 192x192 vs 2.624x at 1024x1024**, same cq, same speed, byte-identical
output. The headline ratio should be quoted with its cell; the gap to Gate 3's
<= 1.5x bar is slightly larger on real images than the profile cell suggests.

## Loop-restoration: the record said it was nearly done; it is the #2 lever

KB-PERF-6's summary line reads *"Stage total: 90.6 -> ~19.8 ms"*, which would
make loop-restoration ~4 % of the encode and effectively finished. **It is
wrong, and that entry's own A/B table disproves it**: the table shows the whole
encode moving 483.01 -> 452.65 ms (-30.4 ms), and a stage cannot shed 70.8 ms
while the encode sheds 30.4.

Re-measured here by summing every `aom_dsp::restore::*` symbol at
`--percent-limit 0.01`:

| cell | port LR | C LR | ratio | gap | % of total gap |
|---|---:|---:|---:|---:|---:|
| 192x192 s0 | 57.8 ms | 8.8 ms | 6.6x | +49.0 | 17.8 % |
| 1024x1024 s0 | 1635 ms | 229 ms | 7.1x | **+1406** | **20.8 %** |

So the three KB-PERF-6 landings took the stage 90.6 -> ~58 ms, a real -33 ms,
and left **the #2 gap item — one that GROWS with frame size** while the two
transform rows shrink. The per-lever numbers in that entry are sound; only the
roll-up is not.

Largest remaining LR symbols at 1024x1024: `acc_stat_line_impl_v3` **446 ms**,
`calculate_intermediate` + `{closure#0}` **416 ms**, `pixel_proj_error_impl_v3`
222 ms, `wiener_impl_v3` 201 ms, `selfguided_restoration` 136 ms. C's side is
`av1_selfguided_restoration_avx2` 128 ms, `av1_lowbd_pixel_proj_error_avx2`
53 ms, `av1_wiener_convolve_add_src_avx2` 32 ms.

**`calculate_intermediate` was sized at 8.5 ms and dismissed** ("a gather the
current vector vocabulary does not handle well, and not worth forcing for
8 ms"). With its closure it is **16.3 ms at 192x192 and 416 ms at 1024x1024** —
the conclusion was drawn on half the number, and the closure it omits is one
that entry itself flags as an inlining sink.

**`acc_stat_line_impl_v3` is still the largest LR symbol** after being
vectorized, which is what KB-PERF-6 predicted when it said the next step is
"register blocking over pixels — libaom's own structure — not wider lanes".
That prediction stands and is now sized: 446 ms on a 1 MP frame.

## What is left on the transform rows

With lane width refuted for the inverse and known-small for the forward, the
+107 ms those two rows carry needs a different mechanism. The one the profile
points at is not lane width but SPECIALISATION: libaom's lowbd inverse is a
family of size-specific whole-transform functions
(`lowbd_inv_txfm2d_8x8_no_identity_avx2`, `lowbd_inv_txfm2d_add_8x4_ssse3`,
`idct4_w4_sse2`, ...), each fusing row pass, column pass and reconstruction with
no generic driver between them, while the port runs one generic driver over
per-kernel function pointers. That is a much larger piece of work than a lane
swap and should not be started without first measuring what fraction of the
+107 ms is driver overhead rather than arithmetic — the `other` bucket above
suggests driver overhead is substantial across the whole encoder.
