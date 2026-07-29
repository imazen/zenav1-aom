# SIMD reach audit — does every tier list actually EXECUTE on aarch64? (2026-07-28)

**Audited at repo sha `101784b`** on `aarch64-apple-darwin` (Apple M4 Pro, 12 cores).
Read-only audit: no kernel, test, or CI file was changed. All instrumentation used to
gather evidence was reverted before commit.

## Headline — is anything both dead AND vacuously tested?

**No. Not one family.** The transform's failure mode (a `neon` tier that was carried in
the attribute list for months while being unreachable, guarded by a differential that
compared the scalar path against itself) **does not repeat anywhere else in `aom-dsp`.**

Every one of the 27 `#[magetypes]` kernel bodies in the crate was **observed executing on
`NeonToken` at runtime** on this aarch64 host — 27/27, zero unreached — and every one of
the eight per-tier differentials was **observed entering the NEON tier inside the test
that guards it**. The structural reason is that the transform was the *only* family built
from hand-written per-architecture `#[arcane]` bodies and x86-gated call sites; every
other family was written as one generic magetypes body dispatched by an
`incant!(..., [v3, neon, wasm128, scalar])` with **no `#[cfg(target_arch)]` anywhere on
the path**. A grep of the whole crate for the two poison patterns finds them *only* under
`transform/` (where they are now `any(x86_64, aarch64)`, plus one deliberately x86-only
lane-doubling specialization, see F2).

The correctness gap that remains is smaller and different in kind: **two bd8-primary
kernels are validated at only one tier per test process** (F3), and **seven of the eight
differentials lack the non-vacuity assertion that would catch a regression back into the
transform's failure mode** (F4). Neither is a live hole today — both were measured
non-vacuous — but F4 is precisely the guard whose absence let the transform bug survive.

> **Status since the audit was written.** F4 is **fixed** (`33bb8a6` — all seven now assert
> a per-architecture `simd_perms >= 1`). F3's CDEF half is **fixed**
> (`cdef_lowbd_simd_diff.rs`); its transform half (`av1_inv_txfm2d_add_u8`) is still open.
> F7 is **fixed**. Everything else below stands as measured at `101784b`.

## Method

Static reading alone is how the transform bug survived, so every "reachable" cell below
cites runtime or object-code evidence, classified:

* **(a)** an existing per-kernel differential that reports a permutation counter,
  checked for a non-zero *vector* permutation count on ARM.
* **(b)** temporary instrumentation inside the tier body — a one-shot-per-generated-variant
  `eprintln!` of `core::any::type_name_of_val(&token)` — run under the family's real tests,
  then reverted. Because `#[magetypes]` duplicates the body per tier, each generated
  variant gets its **own** `static`, so the report is exact per (kernel, tier) and costs
  one relaxed atomic load per call.
* **(c)** `objdump -d` of a `--release` build, counting instructions whose mnemonic carries
  an Apple-syntax vector suffix (`.16b`/`.8h`/`.4s`/…) inside the `*_neon` symbol.

Raw artifacts (committed):

* `benchmarks/simd_reach_tier_log_2026-07-28.tsv` — evidence (b): every
  (dispatch-mode, test binary, kernel body, token tier) tuple observed across two full
  `cargo test -p zenav1-aom-dsp` runs (`AOM_FORCE_SCALAR` unset, and `=1`). Both runs green.
* `benchmarks/simd_reach_neon_census_2026-07-28.tsv` — evidence (c): per-symbol NEON
  instruction census of a `--release` build.

Hotspot shares are reused from `benchmarks/gate3_profile_ranking_2026-07-16.md` (encode
`enc_s0_128_cq32`, decode `dec_352x288_q32`) and `benchmarks/gate3_filters_2026-07-22.md`
(the 2K/4K mosaic gate cells); nothing was re-profiled.

> Reading note on `transform/`: Agent B is actively editing that subtree. Everything here
> about it was read and measured at **`101784b`** and may have moved since.

## The table

`reachable on aarch64?` means: does a **vector** tier actually execute there, on the path
a real decode/encode takes — not "does the attribute list mention `neon`".

| family | tier list | reachable on aarch64? | evidence | differential non-vacuous on ARM? |
|---|---|---|---|---|
| **transform** — 2-D pass drivers `inv_row_pass`, `inv_col_pass`, `inv_col_pass_u8`, `fwd_col_pass`, `fwd_row_pass` (+ `run_inv1d`/`run_fwd1d` and 25 1-D lane kernels) | `define(i32x8), v3, neon, -scalar`; `incant!(…, [v3, neon, scalar])` | **YES** (fixed `fd7efe1`; call sites `any(x86_64, aarch64)`) | **(b)** all 7 driver/dispatcher bodies + both `_core`s printed `arm::NeonToken` under `inv_txfm2d_diff`, `inv_txfm2d_lowbd_diff`, `recon_lowbd_diff`, `txfm2d_diff`, `txfm2d_simd_perm_diff`. **(c)** `___arcane_av1_idct64_impl_neon` 4616 NEON ops / 9913 instrs; `run_fwd1d_neon` 3639; `av1_fdct64_impl_neon` 3580; `run_inv1d_neon` 2649; `av1_idct32_impl_neon` 1920 | **YES, and asserted.** `txfm2d_simd_perm_diff` is the only test in the crate that asserts `simd_perms >= 1` + `scalar_perms >= 1`. Measured 25 permutations. **Caveat:** it drives `av1_inv_txfm2d_add` (u16) and `av1_fwd_txfm2d` only — **not** `av1_inv_txfm2d_add_u8`, see F3 |
| **cdef** — `cdef_filter_16_w8`, `cdef_filter_16_w4` (u16) | `define(i16x8,u16x8), v3, neon, wasm128, -scalar` | **YES** | **(b)** both printed `arm::NeonToken` in `cdef_filter_diff`, `cdef_filter_simd_diff`, `cdef_frame_diff`, `cdef_lowbd_diff`. **(c)** `cdef_filter_16_w8` 667 NEON ops / 953 instrs; `_w4` 668 / 1075 | **YES** — `cdef_filter_simd_diff` entered NEON, 25 permutations. Asserts only `permutations_run >= 2` (F4) |
| **cdef** — `cdef_filter_8_w8`, `cdef_filter_8_w4` (**u8, the bd8 decode walk**) | same | **YES** | **(b)** both printed `arm::NeonToken` under `cdef_lowbd_diff`; under `AOM_FORCE_SCALAR=1` the same binary printed **nothing** — the two tiers are directly A/B-observed. **(c)** `cdef_filter_8_w8` 667 / 988; `_w4` 667 / 1101 | **YES, and asserted (since 2026-07-28).** `cdef_lowbd_simd_diff.rs` drives the dispatching `cdef_filter_block_u8` against the never-dispatched scalar `cdef_filter_block`, 25 permutations × 3200 cases per width, with the same `simd_perms >= 1` guard. Frame-level `cdef_lowbd_diff` (vs REAL C lowbd + vs the u16 port) still pins the walk. See **F3** |
| **loopfilter** — `lpf_impl` (u16 highbd) | `define(i32x4), v3, neon, wasm128, -scalar` | **YES** | **(b)** printed `arm::NeonToken` in `hbd_lpf_diff`, `lf_apply_diff`, `loopfilter_lowbd_diff`, `lpf_simd_diff`. **(c)** `___arcane_lpf_impl_neon` 597 NEON ops / 2294 instrs | **YES** — `lpf_simd_diff::hbd` entered NEON, 25 permutations. `permutations_run >= 2` only (F4) |
| **loopfilter** — `lpf_impl_u8` (**u8, the bd8 deblock walk**) | same | **YES** | **(b)** printed `arm::NeonToken` in `lpf_diff`, `lf_apply_diff`, `loopfilter_lowbd_diff`, `lpf_simd_diff`. **(c)** `___arcane_lpf_impl_u8_neon` 834 NEON ops / 3831 instrs | **YES** — `lpf_simd_diff::lowbd` entered NEON, 25 permutations. `permutations_run >= 2` only (F4) |
| **intra** — `smooth_impl`, `smooth_v_impl`, `smooth_h_impl`, `paeth_impl` | `define(i32x8), v3, neon, wasm128, -scalar` | **YES — but only from `predict_highbd` (bd10/12).** The bd8 `u8` predictor never calls them | **(b)** all four printed `arm::NeonToken` in `intra_simd_diff`, `highbd_diff`, `predict_intra_diff`, `build_nd_diff`, `intra_lowbd_diff`. A probe calling `intra::predict` (u8) directly printed **nothing**; the same probe on `intra::predict_highbd` printed all four. **(c)** `intra::simd::paeth` 88 NEON ops, `smooth` 83, `smooth_v` 73, `smooth_h` 62 | **YES** — `intra_simd_diff` entered NEON, 25 permutations. `permutations_run >= 2` only (F4). Covers the **u16** entry only — nothing to cover on u8, there is no u8 tier. See **F1** |
| **quant** — `quantize_fp_impl` | `define(i32x8), v3, neon, wasm128, -scalar` | **YES** | **(b)** printed `arm::NeonToken` in `quantize_fp_simd_diff` and from a probe calling `av1_quantize_fp_no_qmatrix_dispatch` (the entry `aom-encode/src/lib.rs:384` actually uses). **(c)** `av1_quantize_fp_no_qmatrix_dispatch` 150 NEON ops / 346 instrs | **YES** — entered NEON, 25 permutations. `permutations_run >= 2` only (F4) |
| **txb** — `txb_init_levels_impl` | `define(i32x8), v3, neon, wasm128, -scalar` | **YES** | **(b)** printed `arm::NeonToken` in `txb_init_levels_simd_diff`, `txb_diff`, `read_coeffs_diff`, `optimize_diff`, `cost_coeffs_diff`, `write_txb_full_diff` and 3 more. **(c)** `txb_init_levels` 18 NEON ops / 237 instrs | **YES** — entered NEON, 25 permutations. `permutations_run >= 2` only (F4) |
| **dist** — `highbd_variance64_impl` | `define(i32x8), v3, neon, wasm128, -scalar` | **YES** | **(b)** printed `arm::NeonToken` in `hbd_variance_simd_diff` and `hbd_dist_diff`. **(c)** `highbd_variance` 58 NEON ops / 296 instrs | **YES** — entered NEON, 25 permutations. `permutations_run >= 2` only (F4) |
| **restore** — `wiener_impl` | `define(i32x8), v3, neon, wasm128, -scalar` | **YES** | **(b)** printed `arm::NeonToken` in `wiener_simd_diff`, `frame_walk_diff`, `kernels_diff`, `pick_search`. **(c)** `wiener_convolve_add_src_into` 163 NEON ops / 745 instrs | **YES** — entered NEON, 25 permutations. `permutations_run >= 2` only (F4) |
| **dist** — `sad_simd` (`#[autoversion]`, not magetypes) | AVX-512/AVX2/NEON/WASM/scalar, generated by `#[autoversion]` | **Vectorized in `--release`, but NOT on any production path — it has no caller outside `tests/`** | **(c)** release: 37 NEON ops / 144 instrs (`movi.2d`, `uaddw.4s`, `add.4s`). The tiers are inlined arms selected off a cached detection byte, not separate functions — in the `test-fast` build the dispatcher's only non-panic `bl` callee is `archmage::…::arm::neon_detect`, and in release even that is folded in. **Tier identity NOT ESTABLISHED** — `#[autoversion]` emits no per-tier symbols to attribute against, and it is not a `#[magetypes]` body so the (b) instrumentation does not apply | **N/A** — `tests/sad_simd.rs` is a vs-C + vs-scalar equality test with no token permutation at all. See **F5**, **F9** |
| **convolve**, **inter**, **recon**, **entropy** | *(none)* | N/A — these modules contain **no** `#[magetypes]`, `#[arcane]`, `#[autoversion]`, `incant!`, or `target_feature` of any kind | grep over `crates/aom-dsp/src/{convolve,inter,recon,entropy}` returns zero hits | N/A |
| **cdef** — `cdef_find_dir`; **dist** — `hadamard_*` / `highbd_hadamard_*` (SATD) | *(none)* | N/A — pure scalar, no SIMD tier | source read: `cdef/mod.rs:401`, `dist/hadamard.rs` | N/A. See **F6** |

Also verified as **not** an ARM gap: `AOM_FORCE_SCALAR=1` genuinely pins NEON off in test
builds. Across a full pinned suite run, the *only* vector-tier entries came from the three
binaries that call `for_each_token_permutation` (which re-enables tokens per permutation by
design); every non-permuting binary showed zero SIMD entry. This is the first whole-suite
confirmation of the claim in `aom_dsp::dispatch`'s module docs.

## Findings, ranked by expected value (hotspot share × confidence)

### F2 — HIGHEST. The bd8 i16-lane transform specialization is x86-64 only; ARM runs half the lane width
`transform/simd/mod.rs:80-83` gates `inv1d_v3_i16_gen` and `lowbd16` behind
`#[cfg(target_arch = "x86_64")]`, and the two entry arms that select them
(`try_inv_row_pass` ~line 364, `try_inv_col_pass_u8` ~line 931) are likewise x86-only. On
aarch64 the bd8 inverse transform therefore takes the generic i32x8 path — **one 8-lane
batch where x86 takes two 16-lane i16 batches**. This is *documented and deliberate*
("no magetypes expression and no NEON twin yet"), and it is byte-identical either way, so
it is a pure perf gap, not a correctness one. Confidence: **certain** (explicit cfg,
confirmed by absence of any `*_i16*` symbol in the aarch64 release binary).
Share: transforms are **~12% of decode Ir** and **~33% of encode Ir**
(`gate3_profile_ranking_2026-07-16.md`); the i16 lane doubling was worth measurable wall
time on x86 (`benchmarks/bd8_i16_transform_2026-07-22.md`, `bd8_i16_rows_2026-07-23.md`).
**Next chunk:** write NEON twins for the `lowbd16` helpers (`unpk16`/`pack16`/`madd`
butterflies) — they are the only thing blocking the arm, and the `prims.rs` x86/neon
twin-module pattern (`prims.rs:535-545`) is the template.

### F3 — HIGH (correctness posture). Two bd8-primary kernels are pinned at one tier per test process
**CDEF half FIXED 2026-07-28** (`crates/aom-dsp/tests/cdef_lowbd_simd_diff.rs`); the
transform half is still open. Original finding:

`cdef_filter_8_w{4,8}` and the transform's `inv_col_pass_u8`/`inv_col_pass_u8_core` are
**the** kernels the primary bd8 decode configuration runs, and neither has a
`for_each_token_permutation` differential:

* ~~`cdef_filter_simd_diff` drives `cdef_filter_block_16` only; the u8 kernels' sole guard is
  `cdef_lowbd_diff` (a genuine REAL-C frame-level oracle — good — but at whatever tier
  happens to be live in that process).~~ **CLOSED:** `cdef_lowbd_simd_diff.rs` now drives
  the dispatching `cdef_filter_block_u8` (the entry the frame walk calls) against the
  never-dispatched scalar `cdef_filter_block` under `for_each_token_permutation`, for both
  the w8 and w4 arms, over the u16 differential's structural domain (`CDEF_VERY_LARGE`
  border sentinel + boundary flavours, header strength/damping ranges, all 8 directions,
  all four `en_pri`/`en_sec` combos) plus strided/offset stores and heights `{2,4,5,6,8}`;
  `coeff_shift` is pinned to 0 because `cdef_frame_u8` is the bd8 walk and passes 0. Both
  tests carry the F4 `simd_perms >= 1` guard. Proven to have teeth: off-by-one in the
  SIMD-only `eight` constant of each kernel makes the matching test fail on a
  single-pixel diff (`63` vs `62`), and disabling the counter fires the non-vacuity
  message — instrumentation reverted.
* `txfm2d_simd_perm_diff` drives `av1_inv_txfm2d_add` (u16) and `av1_fwd_txfm2d`; the u8
  column pass is only reached by `inv_txfm2d_lowbd_diff` / `recon_lowbd_diff`, again at
  the live tier. **STILL OPEN** — `crates/aom-dsp/src/transform/**` was under active
  rewrite when the CDEF half landed, so `av1_inv_txfm2d_add_u8` was left alone.

Today both tiers *are* exercised, because CI runs the aarch64 aom-dsp suite twice — default
dispatch and `AOM_FORCE_SCALAR=1` (`.github/workflows/ci.yml`, `test-macos-aarch64` matrix)
— and I A/B-confirmed that pairing locally on `cdef_lowbd_diff` (NEON in default, silent
under the pin). But that coverage lives in the *workflow file*, not in the test, so it is
one CI edit away from silently vanishing, and it never covers the intermediate x86 tiers
(v3-off/v4-on style permutations) at all. Confidence: **certain** (source-read + observed).
Share: CDEF **~27% of decode Ir** on the conformance-style q32 cell; the u8 inverse column
pass rides the ~12% transform share.

### F4 — HIGH (correctness posture). Seven of eight differentials lack the non-vacuity assertion
Only `txfm2d_simd_perm_diff` asserts `simd_perms >= 1`. `cdef_filter_simd_diff`,
`lpf_simd_diff` (×2), `intra_simd_diff`, `quantize_fp_simd_diff`,
`txb_init_levels_simd_diff`, `hbd_variance_simd_diff` and `wiener_simd_diff` all assert
only `report.permutations_run >= 2`. That predicate is satisfied by two permutations in
which **no vector tier ran at all** — exactly the state the transform differential was in
before `d3feb5d`, where it reported `simd_perms=0` and passed. I measured all seven as
non-vacuous today (each printed `arm::NeonToken` from inside its own test binary, 25
permutations each), so this is a *missing guard*, not a live hole. Confidence: **certain**.
**Next chunk:** lift the transform's `simd_perms`/`scalar_perms` counters into a tiny shared
test helper and assert them in all eight — mechanical, ~1 line per test.

### F1 — MEDIUM. The bd8 intra predictor never enters the SIMD tier (on any architecture)
`intra::predict` (the `u8` entry, reached from `predict_intra_u8` →
`build_non_directional_intra_u8`, `intra/mod.rs:1338`) implements SMOOTH / SMOOTH_V /
SMOOTH_H / PAETH as plain scalar loops. The four magetypes kernels in `intra/simd.rs` are
called *only* from `predict_highbd` (`intra/mod.rs:272-294`), i.e. bd10/12. This is
**architecture-independent** — it is not the ARM bug class — and it is partly mitigated:
LLVM autovectorizes the u8 loops anyway (`intra::predict` carries 161 NEON ops in the
release census, vs 86 for `predict_highbd`). Share is modest: the encode profile ranks
"intra predictors (`z2_high` + family)" at **~2%**, and `z2` is *directional* — the
SMOOTH/PAETH family the SIMD covers does not appear in the ranking at all. Confidence:
**certain** for the reach claim; **low** that closing it is worth much. Rank it below F2/F3/F4.

### F6 — MEDIUM (perf). Two named hotspots have no SIMD tier at all
`cdef_find_dir` is **2.49 M Ir/decode = ~4.7% of the q32 decode** and C uses
`cdef_find_dir_avx2` (`gate3_filters_2026-07-22.md` already names porting it as attack
item 3). The `hadamard_*` / `highbd_hadamard_*` SATD family is likewise fully scalar.
Neither is a *reach* failure — there is nothing to reach — but they belong on the same
worklist. Confidence: **certain**.

### F5 — LOW. `sad_simd` has no production call site
`aom_dsp::dist::simd::sad_simd` is referenced only by `tests/sad_simd.rs`. Its own module
doc says so ("that entry point does not exist in this kernel-only crate yet"). The encoder
calls the scalar `dist::sad`. So the SAD family is "dead" in a third sense — reachable code
that nothing reaches. Confidence: **certain** (workspace-wide grep).

### F7 — LOW (doc rot, but it is the kind that misleads an audit) — FIXED 2026-07-28
`crates/aom-dsp/tests/cdef_lowbd_diff.rs:28` stated the u8 filter "reuses the u16 SIMD+scalar
dispatch via a per-block scratch — see `cdef_filter_block_u8`". That was superseded when
dedicated `cdef_filter_8_*` u8-store kernels landed (`cdef/simd.rs:463-474`, which explicitly
says it "avoids the per-block u16 scratch round-trip the first `cdef_filter_block_u8` used").
The stale sentence is what made the u8 CDEF kernels look already-covered by the u16
differential. Same class of hazard as the transform's tier list. **Fixed** in the same
landing as the F3-CDEF differential: the comment now names the dedicated u8-store kernels,
says explicitly that the frame-level harness does NOT cover their per-tier parity, and
points at `cdef_lowbd_simd_diff.rs`.

### F8 — LOW (bench fidelity). The quant bench row measures the scalar wrapper
`crates/aom-dsp-bench/benches/dsp_kernels.rs` `bench_quant` calls
`quant::av1_quantize_fp`, which is a thin wrapper over the scalar
`av1_quantize_fp_no_qmatrix` — **not** `quant::simd::av1_quantize_fp_no_qmatrix_dispatch`,
which is what `aom-encode/src/lib.rs:384` actually calls. Verified directly: a probe through
`av1_quantize_fp` printed no tier; through the dispatch entry it printed `arm::NeonToken`.
So the `quant/fp_*` bench rows time the scalar path on every architecture. (`bench_cdef`,
`bench_loopfilter`, `bench_intra`, and the three transform groups were all verified to reach
the SIMD entry.)

### F9 — Methodological note for whoever does the fixes
**Take object-code evidence from `--release`, never `--profile test-fast`.** `test-fast`
sets `overflow-checks = true`, whose panicking `adds`/`b.lo` chain blocks LLVM's
autovectorizer: `sad_simd` disassembles to a byte-at-a-time scalar loop under `test-fast`
and to `movi.2d`/`uaddw.4s`/`add.4s` under `--release`. This affects `#[autoversion]`
kernels only — the `#[magetypes]` kernels' vector ops are structural, not
autovectorizer-dependent — but it is an easy way to "discover" a nonexistent dead tier.
Also: on Darwin, `objdump` prints Apple syntax (`smin.4s v0, v1, v2`), so a
`v[0-9]+\.[0-9]+[bhsd]` operand regex finds **nothing**; match the mnemonic suffix instead.

## Prioritized worklist

1. **F2** — NEON twins for `lowbd16`'s helpers, then drop the `#[cfg(target_arch = "x86_64")]`
   on the i16 lane path (transform: ~12% decode / ~33% encode Ir; ARM currently at half lane width).
2. **F4** — add `simd_perms >= 1` to the seven differentials that only check
   `permutations_run >= 2`. Cheap, and it is the exact guard whose absence hid the transform bug.
3. **F3** — ~~give `cdef_filter_8_w{4,8}`~~ (DONE 2026-07-28, `cdef_lowbd_simd_diff.rs`)
   and `av1_inv_txfm2d_add_u8` (still open) real `for_each_token_permutation`
   differentials, so their tier coverage stops depending on the CI workflow's two
   dispatch-mode legs (CDEF ~27% of q32 decode Ir).
4. **F6** — port `cdef_find_dir` to magetypes (~4.7% of q32 decode Ir, C has an AVX2 version);
   then the SATD/hadamard family.
5. ~~**F7** — fix the stale `cdef_lowbd_diff.rs:28` comment (5 minutes; it actively
   misleads).~~ DONE 2026-07-28.
6. **F8** — point `bench_quant` at `av1_quantize_fp_no_qmatrix_dispatch`.
7. **F1** — decide whether the bd8 intra SMOOTH/PAETH path is worth an explicit tier at all;
   the profile says ~2% for the whole intra cluster and LLVM already autovectorizes it. Lowest.
