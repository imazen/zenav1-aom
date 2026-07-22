# bd8 lowbd decode pipeline — Phase-3 integration reconciliation + Gate-3 measurement (2026-07-22)

Integration-agent (Phase 3) pass over the second, parallel bd8 "lowbd" (u8 pixel /
i16-narrowable coefficient) decode pipeline. Base: `origin/main` `d00b07ad`
(the `cdef_frame_u8` delegate finding). Nothing here is projected or estimated;
every number is a fresh callgrind Ir measurement on this box.

## 1. Reconciliation — which families are LIVE-lowbd in the decode path

**0 of 4 ported families are wired into the `aom-decode` tile driver.** The
Phase-2 agents landed byte-identity-proven lowbd *leaf kernels* in `aom-dsp`, but
the decode driver (`crates/aom-decode/src/lib.rs`, `frame.rs`) still runs the
highbd `Vec<u16>` / `i32` pipeline for **every** bit depth, including bd8.

| family | lowbd kernel exists (aom-dsp) | differential green | **wired into aom-decode?** | status |
|---|---|---|---|---|
| inverse transform | `av1_inv_txfm2d_add_u8[_into]`, `av1_iwht4x4_add_u8`, `av1_inverse_transform_add_u8` | yes (`inv_txfm2d_lowbd_diff`, re-run green) | **NO** | delegating (highbd) |
| recon (dequant+itx+add) | `reconstruct_txb_u8_into` | yes | **NO** | delegating (highbd) |
| intra prediction | `predict_intra_u8`, `build_{non_directional,directional,filter}_intra_u8`, `dr_predict_u8` | yes | **NO** | delegating (highbd) |
| CDEF | `cdef_frame_u8` | yes | **NO** (deliberate: measured +6.61% Ir, delegate recommended) | delegating (highbd) |

Evidence (grep of `origin/main`): every `_u8` lowbd entry point is referenced
ONLY from within `aom-dsp` (kernels calling their own helpers) and from
`crates/aom-dsp/tests/*` + `crates/aom-bench/*`. `git grep` for
`reconstruct_txb_u8 | av1_inv_txfm2d_add_u8 | cdef_frame_u8 | predict_intra_u8`
under `crates/aom-decode/src/` returns **zero call sites**. The tile driver's
recon planes are `Vec<u16>` (`TileKf.recon/recon_u/recon_v`,
`KfTileDecode.recon*`, `RefFrame.{y,u,v}`) and all 4 recon call sites go to
`reconstruct_txb_into` (highbd).

The `aom_dsp::lowbd` module that the scaffold landed is a **documentation /
contract module** (the dispatch *pattern*), not a live dispatcher. The
`ReconPlanes { LowBd(Vec<u8>) / HighBd(Vec<u16>) }` carrier it describes as the
"intended carrier" is **not implemented**.

### The blocker (precisely located)

Routing a bd8 frame to the lowbd kernels requires the tile working-plane storage
to be `u8`. Today it is hardwired `Vec<u16>`:

* `struct TileKf<'c>` — `recon: Vec<u16>`, `recon_u`, `recon_v`
  (`crates/aom-decode/src/lib.rs:1330-1338`).
* Every `self.recon[..]` / `self.recon_u[..]` / `self.recon_v[..]` access in the
  ~4000-line `impl TileKf` body: intra-neighbour reads, `predict_intra` writes,
  `reconstruct_txb_into` add-back (`:3649,:3747,:4539`), `reconstruct_txb_wht`
  (`:4764`), the intrabc plane copies (`:4528-4535`), the inter predictor writes
  (`:2229..:3356`), and the interintra combine (`:3543-3560`).
* Downstream (`frame.rs`): the post-recon filter chain — deblock, CDEF, loop
  restoration, superres, film grain — all consume the `Vec<u16>` recon planes,
  and **none of those 5 stages has a wired lowbd kernel** (CDEF's `cdef_frame_u8`
  exists but is a measured delegate; the other 4 have no `_u8` entry at all).

None of the three ways to close this is a small chunk:
1. **Generic `TileKf<P: Pixel>`** — monomorphizes the entire ~5000-line tile
   driver twice; the `aom_dsp::lowbd` doc explicitly rejects this (defeats SIMD
   `#[target_feature]` inlining under dyn, or 2x code under generic).
2. **`ReconPlanes` enum + `match` at every access site** — the doc's intended
   carrier; touches every `self.recon*` line above.
3. **A second parallel `TileKf8` u8 driver** — code duplication of the hottest,
   most bit-exact-sensitive loop in the codebase.

## 2. Whole-pipeline byte-identity — PASS (highbd path; lowbd not exercised)

`cargo test -p zenav1-aom-decode --release conformance` →
`conformance_single_frame_intra_byte_identical_to_c_and_golden`:

* **235 in-scope vectors, 274 frames byte-identical (port == C == golden MD5)**,
  5 out-of-envelope (inter) skipped. Corpus at 240 `.ivf`.
* Green on **default SIMD** AND **`AOM_FORCE_SCALAR=1`**.
* Coverage witnesses satisfied (bd10, CDEF `cdef_bits>0`, LR, intrabc, odd size,
  monochrome, film grain).

`inv_txfm2d_lowbd_diff` (the transform lowbd foundation differential) re-run
green (3/3), confirming the leaf kernels ARE byte-identical to the real C
`av1_inv_txfm2d_add_*_c` at bd8.

**Interpretation:** this gate proves the *highbd* pipeline is byte-identical to
C. It does **not** exercise the lowbd pipeline, because the lowbd pipeline is not
wired. There is therefore no "lowbd end-to-end byte-identity" result to report —
the lowbd kernels are byte-identical at the unit level, but the decoder never
calls them. No divergence to bisect.

## 3. Gate-3 measurement — the lowbd lever closed 0% of the decode gap

Fresh callgrind Ir on `origin/main` `d00b07ad`, method identical to
`gate3_decode_profile_2026-07-19.meta` (port = inclusive Ir of
`decode_frame_obus` ÷ port-decode-count; C = inclusive Ir of `shim_decode_av1_kf`
÷ 1; no `-C target-cpu=native`; each cell byte-verified in setup).

| cell | port Ir/decode | C Ir/decode | **current Ir ratio** | 2026-07-19 @`93228daf` |
|---|---:|---:|---:|---:|
| `dec_352x288_q00` (lossless, entropy-dominated) | 103,896,282 | 87,572,319 | **1.186x** | 1.431x |
| `dec_352x288_q32` (filter-dominated) | 53,136,560 | 24,223,274 | **2.194x** | 2.470x |

The improvement vs the 2026-07-19 baseline (q00 1.431x → 1.186x; q32 2.470x →
2.194x) is entirely the DECODE-PERF **entropy 64-bit window + BCE levers 1+2**
(`a00aa51`, `879e24b`, `046b897`) that landed since `93228daf` — **not** the
lowbd pipeline, which contributes nothing because it is unwired.

**How much of the ~50% highbd-pipeline gap did lowbd close: 0%.** The port still
runs the wide `u16`/`i32` pipeline for 8-bit exactly where C uses lowbd. The 4K
wall-clock headline is unchanged at **1.45x / 1.39x** (the decode path is
byte-for-byte the same code that produced the 2026-07-19 wall-clock baseline plus
the Ir-only levers, which that meta states "moved instruction counts, not the
gate"). No fresh 4K wall-clock re-measure was taken this pass (byte-identical
decode path ⇒ nothing to re-measure; extrapolating would violate the
no-extrapolation rule).

### Why a partial wiring would REGRESS (measured reasoning, not shipped)

The primary lowbd win is **memory bandwidth** (u8 vs u16 planes), which callgrind
Ir does not capture and which only materializes when the *whole* plane pipeline
is u8. A partial u8 tile that delegates the 6 un-ported downstream stages
(deblock/CDEF/LR/superres/inter/film-grain) pays a widen→highbd→narrow round-trip
at each boundary. The landed CDEF measurement (`cdef_lowbd_ir_2026-07-22.md`)
already quantifies one such boundary: direct-u8 CDEF is **+6.61% Ir** worse than
delegating, and even delegation costs an amortized whole-plane widen (~44M Ir).
With 6 un-ported stages, a partial u8 tile would add conversion churn at every
one — net Ir regression, uncertain/negative wall-clock — while opening a large
new bit-exactness surface. Shipping that violates the no-regression +
bit-exactness rules, so it was NOT taken.

Additionally, the transform lowbd **safe step is Ir-neutral** by its own
foundation measurement (`lowbd_txfm_foundation_2026-07-22.md`: +0.70%); the
Ir-measurable win is the **i16 SIMD-lane narrowing**, which is *also* gated behind
the same TileKf plane refactor (the i16-narrowed transform is a bd8-specific
kernel — it must be *called*, which requires the u8 recon plane to route to it).

## What remains for ≤ 1.20x (the honest path)

The lowbd lever is real but the FIRST, irreducible step is the TileKf plane-type
refactor (blocker above). Recommended sequencing for the next agent:

1. **Land the `ReconPlanes` enum carrier** (option 2) on `TileKf` — bd8 tiles
   hold `Vec<u8>` working planes, bd10/12 hold `Vec<u16>`; the tile→`KfTileDecode`
   crop widens `u8→u16` once at the boundary so `frame.rs` and the public
   `FrameDecode` surface stay untouched. This is the enabling refactor; on its own
   it is byte-identity-preserving and ~Ir-neutral (prove with the 240-vector
   conformance gate). It unblocks everything below.
2. **Route transform + recon + intra to the `_u8` kernels** inside the bd8 tile
   (these three already have green differentials). Delegate CDEF (measured) and
   the other downstream stages via the single tile-boundary widen.
3. **Take the i16 SIMD-lane narrowing** in the transform column/row passes — the
   Ir-measurable win, now reachable because the u8 recon plane routes bd8 to the
   dedicated kernel. `av1_gen_inv_stage_range` gives `opt_range==16` at bd8, so it
   is byte-identity-safe (proven by the foundation doc).
4. **Only after 1-3 land byte-identical**, re-measure 4K wall-clock — that is
   where the u8-plane bandwidth win appears, and it is the number that moves Gate 3.

Coverage this pass: **verification + measurement complete; 0/4 families newly
wired** (the blocker is the enabling refactor, not the leaf kernels, which are
done). No decode code was changed — the tree is byte-identical to `d00b07ad`.

## Provenance

Box: dedicated aom-rs workstation. Corpus: `AOM_CONFORMANCE_DIR` = the 240-vector
`conformance/data`. Oracle: `upstream/` libaom v3.14.1 pin `03087864`,
linked in-process via `aom-sys-ref shim_decode_av1_kf`. Method: exact callgrind
Ir, load-independent, no `-C target-cpu=native`. Measured, not extrapolated.
