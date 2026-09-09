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

## The old #1 lever is no longer #1, and the port's own top symbol is still not one

| stage | port % | port ms | C % | C ms | gap ms | % of gap |
|---|---:|---:|---:|---:|---:|---:|
| **transform INVERSE** | 11.7 | 53.2 | ~1.7 | ~3 | **+50** | **~18 %** |
| loop-restoration search | 10.9 | 49.5 | ~5.1 | ~9 | +40 | ~15 % |
| transform forward | 9.2 | 41.6 | ~2.2 | ~4 | +38 | ~14 % |
| coeff trellis (`optimize_txb`) | 12.0 | 54.6 | **32.3** | 57.7 | **-3** | — |

`optimize_txb_core` is STILL the port's largest single symbol (12.0 %) and the
port is now marginally FASTER than C on that stage. The original profile's
finding #1 holds and has strengthened: a session that optimised the port's top
symbol would spend itself on the one stage that is already at parity.

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
share of `run_inv1d` 4.22 %), so ~29 ms. At 58 % coverage and roughly double the
lane width, the arithmetic suggests **-8 to -10 ms, i.e. ~3-4 % of the gap** —
worth landing, not a step change. Stated up front because §14 has fired four
times on this codebase (18x, 5x, 13x optimistic) and once in the other
direction (KB-PERF-5 held to 8 %).

## Not attempted here, and why

The implementation is deliberately left for its own landing. It sits on the
reconstruction path, where a wrong sample is a wrong bitstream, and the protocol
it needs — differential against the i32 pass, a reach pin, a bite proof, and
both dispatch modes — is what KB-42 exists to enforce. Landing it at the end of
a long session without room for that protocol is how KB-42 happened.
