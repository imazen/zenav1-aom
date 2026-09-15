# highbd_variance64 v3 width specialization — 2026-09-15

Profile cell: `port 196x196 cq27 s3 reps=5`, callgrind `Ir`, release binary,
same protocol both arms (vbase = HEAD, var2 = this change). Shipping witness:
`port 1024x1024 cq27 s3` byte-identical 40,237 B.

## Gap

`__arcane_highbd_variance64_impl_v3` was **137.9M self Ir over 438,420 calls
(~315 Ir/call)** — 94 % of calls from `dist_block_px_domain_into` (the per-txb
px-domain SSE). C pays ~60 Ir/call across its per-size kernels
(`aom_variance4x4_sse2`, `aom_highbd_8_bit_variance8x8_sse2`, ... — libaom
compiles one kernel per block size). The port's single generic `(w, h)` body
kept the whole bounds-check apparatus alive: every row's
`a[ra+c..ra+c+16].try_into()` re-derived `min(len, ra+c+16)` through
`cmova`/`cmovb` chains — for an 8x8 call, ~250 Ir of overhead on ~60 Ir of
real madd work.

## Change

`var_w_v3::<W>` const-generic bodies for `W ∈ {8, 16, 32, 64, 128}` — the
shape C already specializes. With `W` literal:

- the row's chunk loop unrolls to a constant trip count;
- one checked `&a[ra..ra+W]` slice per row makes `ar.len() == W` known, so
  every inner `ar[c..c+16]` index is statically in range and the checks fold
  entirely;
- the `w%16==8` xmm tail const-folds in/out per width.

Other multiples of 8 (only reachable via the frame-edge
`pixel_dist_visible_only` clip, never a real block width) keep the generic
body — which also got the per-row slice hoist, dropping its per-chunk checks
to one per row.

## Measured

Same-protocol A/B (`/tmp/cg196_vbase.out` → `/tmp/cg196_var2.out`),
identical 438,420-call arcs, `bytes=1352` both arms:

- kernel chain: `highbd_variance`+v3 inclusive **165.2M → 143.1M Ir**
- total: **8,972.30M → 8,950.51M (−21.8M, −0.24 %)**

−0.24 % of the encode cell — above the documented-only landings (boxsum
0.07 %, pixel_proj 0.10 %) and below the band-measured threshold (~0.3 %);
landed on the strict-Ir precedent. Per-call ~315 → ~265 Ir; still ~4x C's
per-call on the small sizes — the residual is the `incant!` summon +
prologue amortized over 16-256 pixels of work, which C's static dispatch
doesn't pay.

Green: `hbd_variance_simd_diff` (every tier, incl. the generic-arm widths),
`hbd_dist_diff`, `obmc_dist_diff` x4, `dist_diff::sad_variance`, encode-side
`pixel_distortion`/`uniform_txfm`/`var_tx`/`intra_model`/`encode_intra_plane`
differentials — all vs the real C oracles.
