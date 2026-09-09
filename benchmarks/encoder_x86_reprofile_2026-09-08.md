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

## Not attempted here, and why

The implementation is deliberately left for its own landing. It sits on the
reconstruction path, where a wrong sample is a wrong bitstream, and the protocol
it needs — differential against the i32 pass, a reach pin, a bite proof, and
both dispatch modes — is what KB-42 exists to enforce. Landing it at the end of
a long session without room for that protocol is how KB-42 happened.
