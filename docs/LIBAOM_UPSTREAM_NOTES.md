# libaom upstream notes — ISA divergences, UB, and surprising behaviour

**What this file is.** A running catalogue of things in *libaom itself* that a
port has to model, work around, or refuse to reproduce. Every entry cost this
project real investigation time; the point of writing them down is that the next
session does not re-derive them.

**What this file is NOT.** A list of our bugs. Port defects live in CLAUDE.md's
Known Bugs (KB-*) ledger. An entry belongs here only if the surprising behaviour
is on libaom's side of the line.

**Ground rule.** Every claim carries a `file:line` citation into the pinned
`upstream/` submodule (libaom v3.14.1, `03087864`) and a provenance tag:

| tag | meaning |
|---|---|
| **MEASURED** | we observed it running, with numbers in this repo |
| **SOURCE** | read from libaom source and reasoned; not executed |
| **CI-CONFIRMED** | predicted on one target, confirmed by CI on another |

Line numbers drift across libaom versions. If one does not match, re-find the
symbol before assuming the entry is wrong — and update the line.

**Citation audit, 2026-07-31.** Every `file:line` in this file was independently
re-checked against `upstream/` at `03087864` (libaom v3.14.1) by a session that
did not write the original entries. Line drift was corrected silently. What did
*not* check out, and is now fixed inline:

- **A2** — the "SSE2 is a third variant *because of its threshold*" claim was
  **wrong**: SSE2's `abs >= dequant>>1` and AVX2's `abs > (dequant>>1) - 1` are
  the same integer predicate, and libaom says so in a comment. Rewritten around
  the difference that *is* real (eob source). "per-8-lane gating" was also wrong
  — both tiers gate per 16 coefficients.
- **A2** — `vmovn_s32` is not in `quantize_neon.c` at all; it is in the
  `load_tran_low_to_s16q` helper. Re-cited.
- **A3** — "no libaom TU enables FMA on x86" is true only in the default
  configuration; `-march=skylake-avx512` (which implies FMA) is applied to the
  Highway AVX-512 TUs when `CONFIG_HIGHWAY=1`. Qualified.
- **A5** — `config/av1_rtcd.h:186` is **not verifiable**: that header is
  generated at build time and exists nowhere in the repo or the submodule. The
  claim is sound and is now cited against the generator instead.
- **C2** — the screen-content threshold is a strict `>` 10%, not `≥`.
- **C3** — `encodeframe.c:276` was a wrong line (it is delta-q bookkeeping); the
  `all_lossless` assignment is at `encodeframe.c:2276`.
- **A1** — the entry was missing its single most load-bearing citation: the
  libaom call site that actually feeds a high-bit-depth residual to the *lowbd*
  kernel (`nonrd_opt.c:202-212`). Added.

Claims that are *measurements made in this repo* (the `MEASURED` tag) were
checked for a recorded source, not re-run; where no record was found the entry
says so.

---

## Category A — ISA divergences: same source, different results per target

These are the dangerous ones. libaom's own cross-tier tests do not catch them
because the tests exercise the domain where the tiers happen to agree.

### A1. `aom_hadamard_16x16`'s 4-way combine is int16-with-wrapping on x86, int32 elsewhere

The three specialised tiers are exactly `avx2 sse2 neon`
(`aom_dsp/aom_dsp_rtcd_defs.pl:1282`).

| tier | combine type | citation (fn / the combine itself) |
|---|---|---|
| `_c` | `tran_low_t` = `int32_t` (`aom_dsp/aom_dsp_common.h:68`) | `aom_dsp/avg.c:249` / `:261-269` |
| `_neon` | `int32x4_t` | `aom_dsp/arm/hadamard_neon.c:188` / `:205-218` (first of four unrolled quarters) |
| `_avx2` | **int16, wrapping** (`_mm256_add_epi16` + `_mm256_srai_epi16`) | `aom_dsp/x86/avg_intrin_avx2.c:144` / `:163-176` |
| `_sse2` | **int16, wrapping** (`_mm_add_epi16` + `_mm_srai_epi16`) | `aom_dsp/x86/avg_intrin_sse2.c:442` / `:466-479` |

The x86 tiers sign-extend the wrapped int16 back to `tran_low_t` on store —
AVX2 via `store_tran_low` (`aom_dsp/x86/bitdepth_conversion_avx2.h:24-32`),
SSE2 via `store_tran_low_offset_4` (`bitdepth_conversion_sse2.h:44`, "sign
extend the values by multiplying by 1", `:33-34`) — so the wrap is invisible in
the signature.

**Why libaom never sees it:** libaom bounds the input at 9-bit `src_diff` and
the output at `[-32640, 32640]` — inside int16, and `_c` says so in its own
comments (`aom_dsp/avg.c:266-269`) — so at bd8 every tier agrees. The
high-bit-depth nonrd estimate feeds an 11-/13-bit residual, the combine reaches
±65534, and x86 wraps where `_c` and NEON do not.

**The call site that does it** (added 2026-07-31 — the entry previously asserted
this without citing it): `av1_block_yrd`, the nonrd estimate, dispatches
`TX_16X16` under `use_hbd` to the **lowbd** `aom_hadamard_16x16` and then
straight into `av1_quantize_fp` — `av1/encoder/nonrd_opt.c:202-212` (function at
`:126`). Note this is the *nonrd* path specifically; libaom's regular
`wht_fwd_txfm` correctly routes highbd to `aom_highbd_hadamard_16x16`
(`av1/encoder/hybrid_fwd_txfm.c:326` vs `:341`). **SOURCE.**

**Tier-independent on x86**: AVX2 and SSE2 make the same int16 choice and SSE2
is baseline, so a compile-time model is valid here (unlike A2).

- **MEASURED / CI-CONFIRMED.** Predicted from aarch64 by counting out-of-int16
  hadamard outputs per cell: 7/7 recall on the x86-divergent cells, 22/24
  overall; the 2 misses are false negatives by construction (a numeric estimate
  difference need not flip the winning mode). Confirmed by CI run 30599200826
  (green, head `8f2e3ac`; re-checked 2026-07-31 — the run exists and passed).
  The recall figures are also recorded in CLAUDE.md KB-20.
- **Our handling:** `hadamard_16x16_dispatched`
  (`crates/aom-encode/src/nonrd_pickmode.rs:641`, call sites `:699`/`:932`),
  KB-20 root #4.
- **Blast radius checked:** `block_yrd_hbd`
  (`crates/aom-encode/src/nonrd_pickmode.rs:891`, call at `:932`) is the only
  production call site — the other caller, `hadamard_16x16_models` at `:696`, is
  the exposition/differential helper. `aom_highbd_hadamard_16x16` is 32-bit in
  every tier — `_c`
  (`aom_dsp/avg.c:444`), `_avx2` (`aom_dsp/x86/avg_intrin_avx2.c:419`), `_neon`
  (`aom_dsp/arm/highbd_hadamard_neon.c:141`), and those are the only tiers
  (`aom_dsp/aom_dsp_rtcd_defs.pl:1301`).
- **Likely a latent libaom bug** for any high-bit-depth caller of the nonrd
  path. Not reported upstream; would need a minimal repro first.
- **The LOW-PRECISION twin is NOT ISA-conditional, and that was worth
  measuring** (added 2026-08-02). `aom_hadamard_lp_16x16`'s 4-way combine is
  `int16` in `_c` too (`aom_dsp/avg.c:305-317`), and the x86 tiers spell it
  slightly differently: `_mm_add_epi16` + `_mm_srai_epi16`
  (`aom_dsp/x86/avg_intrin_sse2.c:442`, AVX2 twin at
  `avg_intrin_avx2.c` `aom_hadamard_lp_16x16_avx2`) truncate to int16 BEFORE
  the shift, where `_c`'s `int16_t b0 = (a0 + a1) >> 1` promotes to `int`,
  shifts, and narrows only the result. That is the same A1 hazard one width
  down — but the *lowbd* arm is bd8-only by construction (`av1_block_yrd`'s
  `use_hbd == 0` branch), the residual is 9-bit (`src - pred`, both u8), so the
  8x8 stage peaks at `255 * 64 = 16320` and `|a0 + a1| <= 32640 < 32768`. Every
  tier agrees over that whole domain, MEASURED on aarch64 by
  `crates/aom-encode/tests/nonrd_block_yrd_lp_diff.rs::lp_hadamard_tiers_agree_over_the_reachable_range`,
  which also asserts the grid actually drives `|coeff|` past 16000 so the
  agreement is not vacuous. **Do not port that agreement to any hbd or
  synthetic-input caller of the lp kernels** — it is a statement about the
  input range, not about the kernels.

  Separately, and it is the reason this note exists: `aom_hadamard_lp_8x8_c`
  and `aom_hadamard_8x8_c` both END with a transpose the SIMD tiers get for
  free (`avg.c:232-236` and `:201-205`, *"Extra transpose to match SSE2
  behavior"*). It is part of the kernel's contract, not a C implementation
  detail, and dropping it is invisible to every order-invariant consumer —
  which is exactly how it cost this project sixteen days as KB-12.

### A2. `av1_quantize_fp`'s SIMD tiers disagree with `_c` and with each other outside int16

Every SIMD tier is a 16-bit kernel that narrows `tran_low_t` on load and
multiplies `dqcoeff` in 16 bits.

- **NEON** narrows with `vmovn_s32` — **truncating**. The narrowing is not in
  the quantizer file at all: it is `load_tran_low_to_s16q`
  (`aom_dsp/arm/mem_neon.h:1505-1511`), called from the lane kernel
  `quantize_fp_8` (`av1/encoder/arm/quantize_neon.c:57`, load at `:61`; entry
  point `av1_quantize_fp_neon` at `:76`). *(Corrected 2026-07-31: the entry
  previously cited `quantize_neon.c:76` for `vmovn_s32`, which does not appear
  in that file.)* `dqcoeff` is `vmulq_s16` at `:68`.
- **x86 AVX2** narrows with `_mm256_packs_epi32` — **saturating** —
  in `load_coefficients_avx2` (`av1/encoder/x86/av1_quantize_avx2.c:64-68`),
  called from lane kernel `quantize_fp_16` (`:198`, load at `:202`; entry point
  `av1_quantize_fp_avx2` at `:224`). `dqcoeff` is `_mm256_mullo_epi16` at `:211`.
  *(`_mm_packs_epi32` — the spelling the entry used to carry — is the SSE2
  variant, `av1/encoder/x86/av1_quantize_sse2.c:28-29`. Also saturating.)*
- **The largest divergence in this entry — read this one first.** The SIMD
  gate is per **16-coefficient
  group** — if any lane passes, all 16 are quantized (`av1_quantize_avx2.c:204-205`;
  `av1_quantize_sse2.c:91-95`, where two 8-lane masks are OR-reduced into one
  `nzflag`) — while `_c` gates each coefficient individually
  (`av1_quantize.c:57`). A sub-threshold coefficient in a passing group can come
  out nonzero in SIMD and is forced to zero in `_c`.
- **Secondary, an off-by-one at odd `dequant`.** The SIMD group threshold
  `abs > (dequant>>1) - 1`
  (`av1_quantize_avx2.c:51-54`, applied at `:204`) is one *more* permissive
  than `_c`'s `abs*2 >= dequant` (`av1/encoder/av1_quantize.c:57`, inside
  `av1_quantize_fp_no_qmatrix` at `:38`) whenever `dequant` is odd — `floor`
  vs `ceil` of `dequant/2`.
- **SSE2 is a third variant, but not for the reason this entry used to give.**
  *(Corrected 2026-07-31.)* Its threshold `abs >= dequant>>1`
  (`av1_quantize_sse2.c:162-163`, compared at `:91-94`) is the *same integer
  predicate* as AVX2's `abs > (dequant>>1) - 1`; libaom says so in a comment —
  the `- 1` exists purely to save a `_mm256_cmpeq_epi16`
  (`av1_quantize_avx2.c:52-53`). The gating granularity is also identical (16
  coefficients, not 8). The difference that **is** real is the eob source: SSE2
  scans the **dequantized** value for nonzero (`av1_quantize_sse2.c:116-121`,
  fed by `coeff0 = _mm_mullo_epi16(qcoeff0, *dequant0)` at `:111`), whereas AVX2
  (`av1_quantize_avx2.c:212`, on `abs_q` from `:209`) and `_c`
  (`av1_quantize.c:60`/`:67`, on `tmp32`) test the **quantized** magnitude. A
  `q * dequant` that wraps to exactly 0 in int16 therefore yields a different
  eob on SSE2. **SOURCE** — not measured.

**MEASURED** on aarch64 over 12 cells: NEON model 12/12 byte-identical,
saturating model 9/12, `_c` 9/12. Recorded in-code at
`crates/aom-encode/src/nonrd_pickmode.rs:735-736` and in CLAUDE.md KB-20.

**Caution — a compile-time `cfg` cannot model this.** libaom dispatches
`av1_quantize_fp` through RTCD at *runtime*, so the correct tier depends on the
host CPU, not the target triple. (Narrowed 2026-07-31: within x86 the AVX2/SSE2
gap is smaller than this entry used to claim — same narrowing, same threshold,
same group size, differing only in the eob source above. The
runtime-vs-compile-time caution still stands because the ARM/x86 gap is a real
truncate-vs-saturate difference.) In our case A1's fix makes every coefficient
reaching the quantizer int16-valued on x86, so the divergence is moot on that
path — but do not generalise that.

### A3. Floating-point contraction differs by target, in libaom's own builds

Clang defaults to `-ffp-contract=on` for C. On **aarch64** `fmadd` is baseline,
so `a*b + c` fuses and rounds once. On **x86-64**, in libaom's default
configuration, no TU enables FMA — AVX2 object libraries get `-mavx2` only
(`cmake/aom_optimization.cmake:57` — that branch adds nothing but
`-mno-avx256-split-unaligned-{load,store}`; `av1/av1.cmake:812-817`,
`aom_dsp/aom_dsp.cmake:482-485`) and neither clang nor gcc lets `-mavx2` imply
`-mfma` — so the same source cannot contract there.

**Qualification added 2026-07-31.** "No libaom TU enables FMA on x86" is a
statement about the *default* build, not about the tree. `-march=skylake-avx512`
— which does imply FMA — is applied to the Highway AVX-512 object libraries at
`av1/av1.cmake:822-832`. Those are gated on `HAVE_AVX512 AND CONFIG_HIGHWAY`,
and `CONFIG_HIGHWAY` defaults to `0` (`cmake/aom_config_defaults.cmake:185`), so
they are not built unless someone asks. Every other flag in the tree
(`-msse2`/`-mssse3`/`-msse4.1`/`-msse4.2`/`-mavx`/`-mavx2`) is FMA-free. If you
ever compare against a Highway-enabled libaom, this entry does not apply to it.

**Consequence:** a production **libaom-on-ARM differs from libaom-on-x86** by a
few ULP in every multiply-accumulate-heavy kernel — NN inference, curve fitting,
FFT, denoise. That is a property of libaom, not of any port.

- **MEASURED.** `av1_nn_predict_c` carried 2 `fmadd`s → 0 with the flag; an
  isolated probe showed 1 `fmadd` on aarch64 under two clang versions, 0 with
  `-ffp-contract=off`, 0 on x86-64 at the default.
- **Our handling:** the oracle is pinned to `-ffp-contract=off`
  (`ORACLE_FP_CFLAGS`, `crates/aom-sys-ref/build.rs:44`; passed to libaom's
  cmake at `:260` and to every shim TU at `:356`) so "bit-exact vs libaom"
  means one thing regardless of host. See `reference/BUILD_CONFIG.md:17-29`
  and CLAUDE.md KB-ARM-FLOAT root #1. *(Note `reference/BUILD_CONFIG.md` says
  nothing about `CONFIG_HIGHWAY` or AVX-512 — do not cite it for the
  qualification above.)*

### A4. `av1_nn_output_prec_reduce` is a *near*-equaliser across SIMD variants, not a guarantee

libaom states the equalisation outright: *"Applies a precision reduction to
output of `av1_nn_predict` to prevent mismatches between C and SIMD
implementations"* (`av1/encoder/ml.h:77-78`). The reduction quantises to
`prec_bits = 9`, i.e. 1/512 buckets (`av1/encoder/ml.c:19-26`). Measured, it
nearly equalises.

- **MEASURED** at the setting the production caller uses — `reduce_prec = 1`,
  hard-coded at the HOG prune's `av1_nn_predict` call
  (`av1/encoder/intra_mode_search_utils.h:446`): **56 of 160,000 lanes differ,
  worst |Δ| exactly one 1/512 bucket.** About 0.035% of evaluations sit on a
  bucket boundary where one ULP flips the whole bucket — which can flip the
  `scores[...] <= th` prune mask (`intra_mode_search_utils.h:449`, with `th`
  from the `-1.2f`-headed threshold tables at `intra_mode_search.c:1321`
  and `:1505`) and therefore the encode. Pinned as an upper bound in
  `crates/aom-encode/tests/hog_prune_diff.rs:207` (`MAX_ONE_QUANTUM_LANES = 56`;
  20,000 cases × 8 lanes = 160,000, `:204`).
- At `reduce_prec = false` the disagreement is much wider: 49,918/160,000 lanes,
  worst |Δ| 1.53e-5. **NOT ESTABLISHED as a pinned measurement** — checked
  2026-07-31: `hog_prune_diff.rs` does not measure or assert anything at
  `reduce_prec = false` beyond `reduce(raw) == reduced` on both sides, and these
  two numbers appear nowhere in the repo except this file and CLAUDE.md prose.
  Treat as an unreproduced observation until someone re-derives it.
- **Implication for porting:** do not treat prec-reduce as licence to ignore
  accumulation order. Our resolution was to make the x86 accumulation-order
  assertion explicitly x86-only and assert lattice+mask parity elsewhere
  (CLAUDE.md KB-ARM-FLOAT, `crates/aom-encode/tests/hog_prune_diff.rs`).

### A5. arm64 RTCD binds some kernels at compile time, so there is no pointer to swap

On arm64, NEON is baseline, so libaom's generated `config/av1_rtcd.h` emits
`#define av1_nn_predict av1_nn_predict_neon` — a macro, not the
runtime-swappable `RTCD_EXTERN` pointer x86 gets.

*Citation corrected 2026-07-31.* This used to cite `config/av1_rtcd.h:186`.
**That header is generated at build time and exists nowhere in the repo or the
`upstream/` submodule** — the citation was uncheckable from a fresh checkout.
The claim is sound and is checkable against the generator instead:

- `cmake/rtcd.pl:455-460` — on `arch == arm64`, `neon` is unconditionally
  `require`d, which sets it as the `_default` and marks the `_c` variant
  `_link = 'false'`.
- `cmake/rtcd.pl:146-168` (`determine_indirection`) — with only one linkable
  variant left, `_indirect` is set to `'false'`.
- `cmake/rtcd.pl:182-186` — `_indirect == 'false'` emits `#define ${fn} ${dfn}`;
  otherwise `RTCD_EXTERN $rtyp (*${fn})($args)`.
- `av1/common/av1_rtcd_defs.pl:467` — `specialize qw/av1_nn_predict sse3 avx2
  neon/`: `neon` is the only arm-side variant, so the count really is one.

**Consequence:** on ARM you cannot force the `_c` variant at runtime, so a
differential that wants a scalar reference has to obtain one another way. Our
workaround compiles libaom's own `av1/encoder/cnn.c` into the shim with the one
RTCD-dispatched primitive — `av1_cnn_convolve_no_maxpool_padding_valid` —
rebound to `_c` and its exports renamed `shim_cscalar_*` / `shim_cnn_*_cscalar`
(`crates/aom-sys-ref/shim/cnn_cscalar.c`; per-TU flags at
`crates/aom-sys-ref/build.rs:87`). **SOURCE + MEASURED.**

Related: **`av1_nn_predict_avx2` does not exist in an ARM libaom at all** —
`ml_avx2.c` is x86-intrinsic source, listed only in `AOM_AV1_ENCODER_INTRIN_AVX2`
(`av1/av1.cmake:387`) and so never compiled on that target. Any contract phrased
against the AVX2 kernel is therefore inexpressible on ARM, which is a fact about
the contract, not a defect to fix. (It is also dropped on x86 under
`CONFIG_EXCLUDE_SIMD_MISMATCH=1`, `av1/av1.cmake:384`.)

### A6. `aom_fdct4x4`'s SIMD tiers are int16-only, so they diverge from `_c` at every bit depth above 8 — while `aom_fdct4x4_lp`'s do not

`aom_fdct4x4_c` (`aom_dsp/fwd_txfm.c:16-82`) carries `tran_high_t` (int64)
butterflies into `tran_low_t` (int32) intermediates and output. Both specialised
tiers (`specialize qw/aom_fdct4x4 neon sse2/`,
`aom_dsp/aom_dsp_rtcd_defs.pl:680-681`) hold **every** value in int16 registers:

- `aom_fdct4x4_neon` (`aom_dsp/arm/fwd_txfm_neon.c:85-99`) shares
  `aom_fdct4x4_helper` (`:21-83`) with the `_lp` variant — `vshl_n_s16(.., 4)`
  for the ×16 (wraps), `vrshrn_n_s32(.., 14)` for `fdct_round_shift` (a
  **truncating** narrow), `vaddq_s16`/`vsubq_s16` butterflies (wrap);
- `aom_fdct4x4_sse2` (`aom_dsp/x86/fwd_txfm_sse2.c:23` →
  `aom_dsp/x86/fwd_txfm_impl_sse2.h:34-203`) uses `_mm_slli_epi16` for the ×16
  (wraps), `_mm_add_epi16`/`_mm_sub_epi16` (`DCT_HIGH_BIT_DEPTH 0`, so wrapping)
  and `_mm_packs_epi32` at each stage boundary — a **saturating** narrow.

So the three implementations agree only while every intermediate fits int16.
That bound holds for a 9-bit (lowbd) residual — pass 1's largest term is
`((step0+step1) * cospi_16_64) >> 14 <= 32654` — and **not** above it:

| residual | first-pass peak | `_c` | NEON | SSE2 |
|---|---|---|---|---|
| bd8, ±255 | 32654 | 32654 | 32654 | 32654 |
| bd10, ±1023 | 46296 | 46296 | wraps negative | clamps to 32767 |
| bd12, ±4095 | the `<< 4` itself wraps (`4095 * 16 = 65520` → `-16`) | — | — | — |

The call site that matters is `av1_block_yrd`'s TX_4X4 arm
(`av1/encoder/nonrd_opt.c:246-263`), reached whenever `cm->features.coded_lossless`
forces `select_tx_mode` to `ONLY_4X4` (`av1/encoder/rdopt_utils.h:392`). Its hbd
half calls `aom_fdct4x4`; its lowbd half calls `aom_fdct4x4_lp`, whose tiers are
therefore **not** ISA-conditional over their reachable domain.

Same shape as A1, and the same evidence: on aarch64 the binding is at compile
time (A5), and `nm -go upstream/build/libaom.a` reports
`nonrd_opt.c.o: U _aom_fdct4x4_neon` — `av1_block_yrd` is linked directly
against the NEON tier, so `_c` is not the function real aomenc runs.

**Port model:** `nonrd_pickmode::fdct4x4_dispatched` (cfg-selected NEON / SSE2 /
`_c` models) for the hbd arm; `nonrd_pickmode::fdct4x4_lp` (`_c`-shaped) for the
lowbd one. **MEASURED** — `nonrd_block_yrd_hbd_diff::fdct4x4_dispatched_matches_
the_real_specialised_symbol` (6,000 blocks vs the exported tier, bd8/10/12),
`..._dispatch_is_inert_at_bd8_and_load_bearing_above_it`, and
`nonrd_block_yrd_lp_diff::fdct4x4_lp_tiers_agree_over_the_reachable_range`.
Substituting `_c` for the dispatched model at the hbd call site diverges
**4 of 8** hbd lossless estimate-arm cells end-to-end
(`kb5_lossless_speed_axis`, teeth run 2026-08-03).

---

## Category B — undefined behaviour in libaom

### B1. `av1_block_error_c` has signed-overflow UB, and the two targets exploit it differently

`av1/encoder/rdopt.c:892` (`av1_block_error_c`) accumulates `error += diff * diff`
with `const int diff` and the multiply in `int` (`:898-899`). The function is
therefore defined only where the product fits int32 (`|diff| <= 46340`). The
same shape is in `av1_block_error_lp_c` (`:907`, `:912-913`).

**How the UB manifests:** aarch64 clang vectorises as `mul.4s` (32-bit wrapping,
same as x86) but widens with **`uaddw`/`uaddw2` — zero-extension** — because it
proved the product non-negative *from the absence of UB*. x86-64 sign-extends. A
wrapped-negative product therefore gains exactly 2³² on ARM.

**Signature:** every affected `dist` differs by exactly N·2³² pre-shift.

**Is it reachable in practice?** Not from real streams. The defined domain,
derived over all 19 tx sizes × all 1-D type pairs
(`av1_gen_fwd_stage_range`, `av1_fwd_txfm2d.c:41`): at bd8 the worst final width
is 16 signed bits, so `|coeff| <= 2^15`; `dqcoeff` is spec-clamped to
`[-(1<<(7+bd)), (1<<(7+bd))-1]` (`av1/decoder/decodetxb.c:116-117`) and written
with `coeff`'s sign; hence `|diff| <= 2^15` and `diff*diff <= 2^30 < INT_MAX`.
bd>8 is structurally immune (`av1_highbd_block_error_c`, `rdopt.c:920`, declares
`const int64_t diff` at `:929`).

**libaom knows the bound** — `test/error_block_test.cc` declares
`const int msb = bit_depth_ + 8 - 1` (`:93`, also `:136`, `:191`) and comments
that coeff and dqcoeff "will always have at least the same sign, and this can be
used for optimization, so generate test input precisely" (`:98-99`) — but the
kernel neither documents nor enforces it.

- **MEASURED.** Our harness reached the UB by drawing AOM_QUANT_B's
  `(quant, quant_shift)` independently when libaom consumes them as one
  reciprocal (`invert_quant`, `av1_quantize.c:582`), inflating `dqcoeff` ~400×
  past the clamp. See CLAUDE.md KB-ARM-FLOAT root #3.
- **Reportable upstream?** Arguably yes as a hardening request; unreachable from
  conforming input, so low priority.

---

## Category C — surprising-but-intended behaviour

Not bugs. Recorded because each one cost time to discover and would cost it
again.

### C1. libaom will not segment a KEY frame in one pass

`av1_vaq_frame_setup` has exactly one call site in `av1/`, inside
`encode_with_recode_loop` (`av1/encoder/encoder.c:3495`; function at `:3260`;
definition at `aq_variance.c:43`). `speed_features.c:2784-2785` sets
`DISALLOW_RECODE` when `oxcf->pass == AOM_RC_ONE_PASS && has_no_stats_stage(cpi)`
— note the second conjunct, which the entry used to omit — and that routes the
frame to `encode_without_recode` instead (`encoder.c:3719-3722`), which never
reaches the setup. **MEASURED:** `--aq-mode=1` one-pass yields `seg.enabled=0`;
two-pass yields 1. If you need a segmented KEY frame from the C encoder, you
must ask for two passes.

### C2. The screen-content detector is a colour-count statistic, and flat content does not trip it

`estimate_screen_content` (`av1/encoder/encoder.c:2042-2100`, reached via
`av1_set_screen_content_options`, `:2439`, dispatch at `:2472-2477`): count the
full 16×16 luma blocks having 2..4 distinct `pix >> (bd-8)` values
(`n_colors > 1 && n_colors <= kColorThresh`, `kColorThresh = 4`, `:2057`,
`:2078`); `allow_screen_content_tools` is
`counts_1 * 256 * 10 > width * height` (`:2091`) — i.e. a **strictly greater
than 10%** area fraction. *(Corrected 2026-07-31: this entry said "≥ 10%"; the
comparison is `>`.)* `allow_intrabc` additionally needs `counts_2 * 256 * 12 >
area` over the same blocks with per-pixel variance `> 0` (`:2083-2085`, `:2094`).

- It is **bit-depth independent by construction** — the highbd path asks
  `av1_count_colors_highbd` for `num_color_bins`, not `num_colors`
  (call at `encoder.c:2071-2073`), and that count is built after a down-convert
  to the 8-bit domain (`av1/encoder/intra_mode_search.c:351-357`, function at
  `:338`, bin loop `:366-369`), so bd8/bd10 twins of the same clip classify
  identically.
- **Counter-intuitive, MEASURED:** DC-flat content scores **0/16 blocks**.
  At source level an exactly-flat block has `n_colors == 1` and is excluded by
  the `n_colors > 1` half of the test (`:2078`); on decoded material,
  quantisation ringing pushes the count the other way, past 4. Either way,
  perceptual flatness is not this statistic.
- The default detector is `AOM_SCREEN_DETECTION_STANDARD`
  (`av1/av1_cx_iface.c:405`, in the non-realtime `default_extra_cfg`); the
  AA-aware variant is set only under `AOM_TUNE_IQ || AOM_TUNE_SSIMULACRA2`
  (gate at `:1939-1940`, assignment at `:1969`).

### C3. Control combinations libaom forbids or silently drops

| combination | libaom's behaviour | citation |
|---|---|---|
| `--enable-tx-size-search=0` + `--enable-tx64=0` | hard assert — `assert(oxcf->txfm_cfg.enable_tx64 \|\| tx_search_type != USE_LARGESTALL)` | `av1/encoder/encodeframe.c:2461` |
| superres + lossless | superres dropped (`all_lossless = coded_lossless && !av1_superres_scaled`) | `av1/encoder/encodeframe.c:2276` (decoder twin `av1/decoder/decodeframe.c:5173`; recomputed post-superres at `av1/encoder/encoder.c:2646-2647`) |
| superres + intrabc | spec-forbidden — `allow_intrabc` is only read when `!av1_superres_scaled(cm)` | `av1/decoder/decodeframe.c:4933-4934` (KEY), `:4944-4945` (INTRA_ONLY) |

*Row 2's citation was `encodeframe.c:276` and was wrong* — that line is delta-q
bookkeeping (`td->deltaq_used |= ...`). Corrected 2026-07-31. Row 3 previously
cited "AV1 spec" with no line; libaom's own read sites are given above.

**Subtlety, MEASURED:** `--enable-tx-size-search=0` stops being a configuration
at speed ≥ 8 — `speed_features.c:2726-2729` is gated on `use_nonrd_pick_mode==0`
(set at `:579`, inside `set_allintra_speed_features_framesize_independent`
at `:345`, under `if (speed >= 8)` at `:577`) — so the forbidden pair above
*lapses* there, and a harness that asserts `TX_MODE_LARGEST` will panic on a
stream aomenc produced happily.

### C4. Several speed features are framesize-gated *and* speed-gated

A framesize threshold alone does not make a feature live. `prune_tx_type_using_stats`
needs ≥480p **and** speed ≥ 2 (level 1) / ≥ 4 (level 2) — both inside
`set_allintra_speed_feature_framesize_dependent`
(`av1/encoder/speed_features.c:166`): `if (speed >= 2)` at `:236` +
`if (is_480p_or_larger)` at `:261` → level 1 at `:262`; `if (speed >= 4)` at
`:292` + `is_480p_or_larger` at `:299` → level 2 at `:300`. We wasted analysis
on the assumption that adding ≥480p contexts would exercise it; at speed 0 it
stays at its init value 0 (`:2464`) in real aomenc too. The full framesize table
is in `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md:678-699`.

### C5. Superres doubles the minimum tile width, and libaom's encoder ignores it

`av1_is_min_tile_width_satisfied` (`av1/common/tile_common.c:200-206`) requires
every INNER (non-rightmost) tile column to satisfy
`min_inner_width << MI_SIZE_LOG2 >= 64 << av1_superres_scaled(cm)` — 64 luma
pixels normally, **128 under superres**, measured on the CODED (downscaled)
frame. The decoder enforces it immediately after `read_tile_info`
(`av1/decoder/decodeframe.c:5114-5118`) and reports `AOM_CODEC_CORRUPT_FRAME`.

Without superres the bound is unreachable — a tile column is at least one
superblock, i.e. ≥ 64 px — so it looks like dead code until superres is in play.

**The encoder checks it only in an `assert`.** `av1_get_tile_limits` /
`av1_calculate_tile_cols` (`tile_common.c:32`, `:52`) clamp tile columns against
`MAX_TILE_WIDTH`, `MAX_TILE_AREA` and the superblock grid — none of which knows
about superres. The one encode-side consultation is
`assert(av1_is_min_tile_width_satisfied(cm))` at the tail of `av1_setup_frame_size`
(`av1/encoder/superres_scale.c:397`), i.e. libaom *states* the invariant but
enforces it only when assertions are live. Our pinned oracle build is
`CMAKE_BUILD_TYPE=Release` → `-DNDEBUG` (`upstream/build/CMakeCache.txt:84`,
`:118`), so the assert is compiled out and aomenc silently produces streams its
own decoder refuses. There is no clamp that would have avoided the shape; a debug
libaom would abort instead.

**MEASURED (2026-07-31).** `--tile-columns=2` with `rc_superres_kf_denominator=16`
on a 512×96 frame: the coded frame is 256 px = 4 superblocks, split into 4 columns
of 1 superblock each = 64 px inner columns. The encode succeeds; `aomdec` then
says `Failed to decode frame 1: Corrupt frame detected / Additional information:
Minimum tile width requirement not satisfied`. Same at denom 12 on 512×96 and
denom 16 on 768×128. Halving the request (`--tile-columns=1`, 2-superblock
columns) is accepted and byte-decodes normally — the asymmetric control.

**How we handle it:** `parse_frame_header_ext`
(`crates/aom-decode/src/frame.rs`) applies the same predicate via
`min_inner_tile_width_mi` and returns `DecodeError::Malformed`, so the port
refuses exactly what the reference decoder refuses. Pinned by
`superres_multitile_below_min_tile_width_is_rejected`
(`crates/aom-decode/tests/superres_tiles_diff.rs`), which asserts BOTH halves
(C rejects, port rejects) against a conformant control at half the column count.

**Why libaom's own tests do not catch it:** the constraint only binds when
superres and multiple tile columns are combined, and libaom's encode/decode
round-trip tests do not cross those two axes — the same gap that left the AV1
intra conformance corpus with zero superres and zero `tiles > 1` vectors
(`benchmarks/decoder_corpus_feature_tuples_2026-07-30.tsv`). It also means a
harness that generates its own superres multi-tile vectors must pick
(width, denominator, tile-columns) triples on the legal side, or it will be
comparing against a C decoder that never ran.

---

## How to add an entry

1. Cite `file:line` into `upstream/`, and say which tag applies (MEASURED /
   SOURCE / CI-CONFIRMED).
2. State how *we* handle it and where (KB number, module, test).
3. If libaom's own tests do not catch it, say why — that is usually the most
   useful sentence in the entry.
4. Prefer "NOT ESTABLISHED" to a plausible story. Two claims in this project's
   history were confidently wrong and had to be retracted (a "17 invalid AV1
   streams" finding that was a harness bug — `STATUS.md:213-214`,
   `benchmarks/intra_tiebreak_deltas_2026-07-23.md:3`; and a root-cause
   attribution corrected mid-investigation, CLAUDE.md KB-1 bug #2). Both
   retractions are recorded rather than deleted, which is the standard here.
5. Cite something a fresh checkout can open. Build-generated headers
   (`config/*_rtcd.h`), build directories, and anything under `target/` are not
   citable — cite the generator or the `.pl`/`.cmake` that produces them.

### C6. libaom's quality-vs-quantizer curve is NOT monotone on screen content at cpu-used 1

**Why it matters:** every bracketed target-quality search assumes monotonicity —
raise the quantizer, quality falls. On libaom's cpu-used 1 screen-content path
that assumption is false, and a search that relies on it maintains a "bracket"
that is not one.

**MEASURED** (`benchmarks/xbench_2026-08-01.md` stage 3, 260 searches over five
encoders): on `sc_codec_wiki`, SSIMULACRA2 across adjacent quantizers runs
**73.819 (q49) → 70.610 (q50) → 74.445 (q51)** — going from q50 to q51 *raises*
quality by 3.8 points **while lowering the rate**. `sc_imac_dark` shows the same
sign flip, smaller (58.700 at q50 → 58.735 at q51).

**Mechanism** (SOURCE, reasoned not instrumented): the deep palette/IntraBC
search re-decides block-level tooling between adjacent quantizers, so two
neighbouring q values can select materially different tool sets. It is a property
of that search path, not a measurement artifact — it reproduces, and the rate
moves the *expected* direction while the quality does not.

**Consequences for anything that targets a quality level against libaom:**

- It is a **distinct failure mode from quantizer granularity**, and the two are
  easy to conflate. Granularity means the target band is *narrower than one
  step* (on a 0..63 scale one step is ~1.9 SSIMULACRA2, wider than a ±1 band —
  see the same document). Non-monotonicity means the band can be **stepped over
  in both directions**. A finer quantizer scale fixes the first and not the
  second.
- It is why libaom-C converges lowest of the five encoders measured (83.1 %)
  despite an unremarkable 1.911 SSIMULACRA2/step, and it is the ONLY miss above
  target 70 anywhere in that study.
- A search that assumes monotonicity should at minimum detect the inversion and
  fall back (e.g. sample the neighbourhood rather than bisect), not iterate until
  it exhausts a budget.

**Not established:** whether this extends to cpu-used 0, to photographic content
at any speed, or to the port's own screen path — only libaom-C at cpu-used 1 on
screen content was measured. Our `encode_rgb8_with_target` work should not assume
it is libaom-specific without checking.


### C7. `aom_once` performs no synchronisation in our oracle build, and two of the tables it guards are cleared before they are published

**Why it matters:** it makes every lazily-initialised libaom table a data race
under the concurrency libtest gives us for free, and the observable failure is a
bare SIGSEGV with no attribution. It cost an afternoon of bisection and a false
"runner flake" conclusion (`docs/DIFFERENTIAL_PLAYBOOK.md` §11).

**SOURCE.** `upstream/aom_ports/aom_once.h:41-81` has three implementations. We
build the oracle `-DCONFIG_MULTITHREAD=0` (`crates/aom-sys-ref/build.rs:263`,
deliberate — it is the determinism definition), so neither the Win32 `INIT_ONCE`
arm nor the `pthread_once` arm compiles; we get the fallback at `:70-80`, whose
own comment says *"Default version that performs no synchronization"*:

```c
static void aom_once(void (*func)(void)) {
  static volatile int done;
  if (!done) { func(); done = 1; }
}
```

`done` is set **after** `func()` returns, so N threads can be inside `func()` at
once. That is harmless for an idempotent fill of *distinct* storage, and is
presumably why libaom ships it — single-threaded builds do not race, and
libaom's own multithreaded builds get the real once.

**It is not harmless for two of the seven**, because they publish through
storage they first clear:

- `av1_init_wedge_masks` (`upstream/av1/common/reconinter.c:600`) →
  `init_wedge_masks` (`:494-519`) opens with
  `memset(wedge_masks, 0, sizeof(wedge_masks))`. For the whole duration of the
  init, every `av1_wedge_params_lookup[bsize].masks[sign][index]` is NULL, and
  `av1_get_contiguous_soft_mask` (`reconinter.h:456-460`) hands that pointer to
  the caller with no check. A second thread's `memset` re-NULLs entries the
  first thread already published;
- `av1_init_intra_predictors` (`upstream/av1/common/reconintra.c:1894`) has the
  same shape over a **function-pointer** table — the same fault, one indirection
  worse.

The other five (`av1_rtcd`, `aom_dsp_rtcd`, `aom_scale_rtcd`,
`av1_init_me_luts`, `av1_rc_init_minq_luts`) fill their tables without a
clearing pass, so a double entry is benign there.

**libaom's own tests do not catch it** because libaom never runs in this
configuration: `CONFIG_MULTITHREAD=0` is a single-threaded build, and their test
suite's threading is inside the encoder/decoder, not across two independent
first-touch call sites. We are the ones who put a single-threaded-configured
libaom under a multi-threaded test harness.

**MEASURED** (Apple Silicon, `--profile test-fast`, 2026-08-03):
`crates/aom-dsp/tests/interintra_diff.rs` faulted **16/400 (4.0%)** —
`EXC_BAD_ACCESS ... at 0x0` in `_platform_memmove` ← `shim_ii_wedge_mask`; the
isolating harness `crates/aom-sys-ref/tests/wedge_init_race.rs` (8 threads
released together) faulted **133/200 (66.5%)**. Both are 0 after the fix
(0/1000 and 0/300).

**How we handle it:** `aom_sys_ref::ref_init` forces **all seven** initialisers
under one Rust `Once`, so every `done` flag is set before a second thread can
reach C; `shim_ii_wedge_mask` additionally returns `-1` on a NULL table so a
regression names itself instead of faulting. **Any new shim entry point that can
reach a lazy libaom table must call `ref_init()` first** — including anything
that runs a real encoder, since `av1_initialize_enc`
(`upstream/av1/encoder/encoder.c:305-315`) calls six of the seven and is itself
unguarded.
