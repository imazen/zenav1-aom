# The u16-plane root is **24 %** of the shipping-preset gap, not ~42 % — and 53 % is coefficient-domain, where it cannot help

**2026-09-09. No code changed.** `CLAUDE.md` has carried this since the first
1 MP profile:

> *"The structural root under **~42 %** of the gap is that the encoder holds
> planes as `u16` at every bit depth and so runs the HIGHBD kernels at bd8,
> where libaom runs a specialised lowbd path ... one cause, ~115 ms, no single
> landing closes it."*

That number is from a **192x192 aarch64** profile. This session's audits kept
concluding "the residual is the u16-at-bd8 root" — for wiener, for
pixel_proj_error, for directional intra — so it was worth checking what that
root is actually worth **at the preset zenavif ships**. It is worth much less
than the ranking assumes, and the mass is somewhere else.

## The split

Every symbol of both flat profiles (`encoder_shipping_preset_1mp_2026-09-09.*.tsv`)
bucketed by whether it touches plane SAMPLES (so `u8` planes at bd8 would halve
its data) or COEFFICIENTS (where the plane type is irrelevant):

| domain | port | C | **gap** | **% of gap** | ratio |
|---|---:|---:|---:|---:|---:|
| RD drivers — **inlining sinks, unattributable** | 497.9 ms | 186.1 | +311.8 | 16.0 % | 2.68x |
| **PIXEL — `u8` planes would halve the data** | 743.8 | 269.0 | **+474.9** | **24.4 %** | 2.77x |
| **COEFFICIENT — `u8` cannot help** | 1862.7 | 833.9 | **+1028.8** | **52.8 %** | 2.23x |
| other / unclassified | 381.4 | 251.9 | +129.4 | 6.6 % | 1.51x |

Total gap 1949 ms at 1024x1024 cq27 `--cpu-used 3`.

## What follows, and it re-ranks the whole programme

* **The plane refactor's ceiling is 24.4 % of the gap, and its realistic value is
  about half that.** `u8` halves the DATA, not the time — libaom's pixel-domain
  advantage is 2.77x, of which lane width is one part and its size-specialised
  kernels another. Assuming `u8` recovers half of +474.9 gives ~237 ms, **~12 %
  of the gap**, for an encoder-wide change to `KeyFramePlanes`, `ReconPlane` and
  every predictor / filter / restoration call site.
* **53 % of the gap is coefficient-domain, where the plane type is irrelevant** —
  and the transform class alone is **+789 ms, 40.5 %** (`encoder_shipping_preset`
  §3). That is the programme, and it is the one the census already named:
  libaom's **size-specialised whole-transform entry points** against the port's
  generic driver.
* **The two are not close.** Coefficient-domain is 2.2x the size of the pixel
  domain's ceiling and 4x its realistic value.

**So the next structural landing is NOT the plane representation.** It is a
SIMD-PRESERVING size-specialised transform — which
`encoder_txfm_size_census_2026-09-09.md`'s own correction already anticipated:
*"Do not re-attempt an 8x8 fusion against a SIMD-capable generic path. A future
one would have to be vectorised itself to beat what is already there."* 4x4 is
40.5 % of forwards at this preset and is already fused; 8x8 is 25.7 % and is the
target, as a VECTOR kernel rather than the scalar fusion that measured +7.07 %.

## Limits, stated because the buckets are hand-made

* **16 % sits in RD-driver inlining sinks** (`search_tx_type_intra_into`,
  `intra_model_rd_y`, `txfm_rd_in_plane_*` and their C counterparts) that contain
  BOTH domains and cannot be split without instrumentation. KB-PERF-10's rule
  applies: do not cost a lever off them. Both headline figures are therefore
  shares of the attributable ~84 %.
* The classification is regex-over-symbol-names and hand-made; two
  misclassifications were caught and fixed while building it (`dist::hadamard`
  belongs to coefficient, `simd_variance`/`highbd_variance` to pixel), so treat
  it as a ranking instrument with ~±5 pp, not a measurement to three digits.
* One cell, one content class, one box, bd8 4:2:0, cq27, `--cpu-used 3`.
* This does not say the plane refactor is worthless — ~12 % of the gap is more
  than this whole cycle's three landings combined. It says it is **not the
  largest item**, and the ranking has been treating it as though it were.
