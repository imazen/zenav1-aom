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

---

## Category A — ISA divergences: same source, different results per target

These are the dangerous ones. libaom's own cross-tier tests do not catch them
because the tests exercise the domain where the tiers happen to agree.

### A1. `aom_hadamard_16x16`'s 4-way combine is int16-with-wrapping on x86, int32 elsewhere

| tier | combine type | citation |
|---|---|---|
| `_c` | `tran_low_t` (int32) | `aom_dsp/avg.c:249` |
| `_neon` | `int32x4_t` | `aom_dsp/arm/hadamard_neon.c:188` |
| `_avx2` | **int16, wrapping** (`_mm256_add_epi16` + `_mm256_srai_epi16`) | `aom_dsp/x86/avg_intrin_avx2.c:144` |
| `_sse2` | **int16, wrapping** | `aom_dsp/x86/avg_intrin_sse2.c:442` |

The x86 tiers' `store_tran_low` sign-extends the wrapped int16 back to
`tran_low_t`, so the wrap is invisible in the signature.

**Why libaom never sees it:** libaom bounds the input at 9-bit `src_diff` and
the output at `[-32640, 32640]` — inside int16 — so at bd8 every tier agrees.
The high-bit-depth nonrd estimate feeds an 11-/13-bit residual, the combine
reaches ±65534, and x86 wraps where `_c` and NEON do not.

**Tier-independent on x86**: AVX2 and SSE2 make the same int16 choice and SSE2
is baseline, so a compile-time model is valid here (unlike A2).

- **MEASURED / CI-CONFIRMED.** Predicted from aarch64 by counting out-of-int16
  hadamard outputs per cell: 7/7 recall on the x86-divergent cells, 22/24
  overall; the 2 misses are false negatives by construction (a numeric estimate
  difference need not flip the winning mode). Confirmed by CI run 30599200826.
- **Our handling:** `nonrd_pickmode::hadamard_16x16_dispatched`, KB-20 root #4.
- **Blast radius checked:** `block_yrd_hbd` is the only exposed call site;
  `aom_highbd_hadamard_16x16` is 32-bit in every tier
  (`aom_dsp/x86/avg_intrin_avx2.c:419`).
- **Likely a latent libaom bug** for any high-bit-depth caller of the nonrd
  path. Not reported upstream; would need a minimal repro first.

### A2. `av1_quantize_fp`'s SIMD tiers disagree with `_c` and with each other outside int16

Every SIMD tier is a 16-bit kernel that narrows `tran_low_t` on load and
multiplies `dqcoeff` in 16 bits.

- **NEON** narrows with `vmovn_s32` — **truncating**
  (`av1/encoder/arm/quantize_neon.c:76`, lane kernel `:57`).
- **x86 AVX2** narrows with `_mm_packs_epi32` — **saturating**
  (`av1/encoder/x86/av1_quantize_avx2.c:224`, lane kernel `:194`).
- The AVX2 group threshold `abs > (dequant>>1) - 1` (`:30`) is one *more*
  permissive than `_c`'s `abs*2 >= dequant` at odd `dequant`.
- **SSE2 differs again** (`thr = dequant >> 1`, per-8-lane gating), so an x86
  build resolving to SSE2 is a third variant.

**MEASURED** on aarch64 over 12 cells: NEON model 12/12 byte-identical,
saturating model 9/12, `_c` 9/12.

**Caution — a compile-time `cfg` cannot model this.** libaom dispatches
`av1_quantize_fp` through RTCD at *runtime*, so the correct tier depends on the
host CPU, not the target triple. In our case A1's fix makes every coefficient
reaching the quantizer int16-valued on x86, so the divergence is moot on that
path — but do not generalise that.

### A3. Floating-point contraction differs by target, in libaom's own builds

Clang defaults to `-ffp-contract=on` for C. On **aarch64** `fmadd` is baseline,
so `a*b + c` fuses and rounds once. On **x86-64** no libaom TU enables FMA —
AVX2 object libraries get `-mavx2` only (`cmake/aom_optimization.cmake:57`,
`av1/av1.cmake:812-817`) and neither clang nor gcc lets `-mavx2` imply `-mfma`
— so the same source cannot contract there.

**Consequence:** a production **libaom-on-ARM differs from libaom-on-x86** by a
few ULP in every multiply-accumulate-heavy kernel — NN inference, curve fitting,
FFT, denoise. That is a property of libaom, not of any port.

- **MEASURED.** `av1_nn_predict_c` carried 2 `fmadd`s → 0 with the flag; an
  isolated probe showed 1 `fmadd` on aarch64 under two clang versions, 0 with
  `-ffp-contract=off`, 0 on x86-64 at the default.
- **Our handling:** the oracle is pinned to `-ffp-contract=off`
  (`ORACLE_FP_CFLAGS`, `crates/aom-sys-ref/build.rs`) so "bit-exact vs libaom"
  means one thing regardless of host. See `reference/BUILD_CONFIG.md` and
  CLAUDE.md KB-ARM-FLOAT root #1.

### A4. `av1_nn_output_prec_reduce` is a *near*-equaliser across SIMD variants, not a guarantee

Comments around the NN path imply the precision reduction makes SIMD variants
agree. Measured, it nearly does.

- **MEASURED** at the setting the production caller uses (`reduce_prec = true`):
  **56 of 160,000 lanes differ, worst |Δ| exactly one 1/512 bucket.** About
  0.035% of evaluations sit on a bucket boundary where one ULP flips the whole
  bucket — which can flip the `score <= -1.2f` prune mask and therefore the
  encode.
- At `reduce_prec = false` the disagreement is much wider: 49,918/160,000 lanes,
  worst |Δ| 1.53e-5.
- **Implication for porting:** do not treat prec-reduce as licence to ignore
  accumulation order. Our resolution was to make the x86 accumulation-order
  assertion explicitly x86-only and assert lattice+mask parity elsewhere
  (CLAUDE.md KB-ARM-FLOAT, `crates/aom-encode/tests/hog_prune_diff.rs`).

### A5. arm64 RTCD binds some kernels at compile time, so there is no pointer to swap

On arm64, NEON is baseline, so libaom's generated `config/av1_rtcd.h` emits
`#define av1_nn_predict av1_nn_predict_neon` — a macro, not the
runtime-swappable `RTCD_EXTERN` pointer x86 gets (`config/av1_rtcd.h:186`).

**Consequence:** on ARM you cannot force the `_c` variant at runtime, so a
differential that wants a scalar reference has to obtain one another way. Our
workaround compiles libaom's own `av1/encoder/cnn.c` into the shim with the one
RTCD-dispatched primitive rebound to `_c` and its exports renamed
(`crates/aom-sys-ref/shim/cnn_cscalar.c`). **SOURCE + MEASURED.**

Related: **`av1_nn_predict_avx2` does not exist in an ARM libaom at all** —
`ml_avx2.c` is x86-intrinsic source, never compiled on that target. Any contract
phrased against the AVX2 kernel is therefore inexpressible on ARM, which is a
fact about the contract, not a defect to fix.

---

## Category B — undefined behaviour in libaom

### B1. `av1_block_error_c` has signed-overflow UB, and the two targets exploit it differently

`av1/encoder/rdopt.c:892` accumulates `error += diff * diff` with the multiply
in `int`. The function is therefore defined only where the product fits int32
(`|diff| <= 46340`).

**How the UB manifests:** aarch64 clang vectorises as `mul.4s` (32-bit wrapping,
same as x86) but widens with **`uaddw`/`uaddw2` — zero-extension** — because it
proved the product non-negative *from the absence of UB*. x86-64 sign-extends. A
wrapped-negative product therefore gains exactly 2³² on ARM.

**Signature:** every affected `dist` differs by exactly N·2³² pre-shift.

**Is it reachable in practice?** Not from real streams. The defined domain,
derived over all 19 tx sizes × all 1-D type pairs
(`av1_gen_fwd_stage_range`, `av1_fwd_txfm2d.c:41`): at bd8 the worst final width
is 16 signed bits, so `|coeff| <= 2^15`; `dqcoeff` is spec-clamped to
`[-(1<<(7+bd)), (1<<(7+bd))-1]` (`decodetxb.c:116`) and written with `coeff`'s
sign; hence `|diff| <= 2^15` and `diff*diff <= 2^30 < INT_MAX`. bd>8 is
structurally immune (`av1_highbd_block_error_c`, `rdopt.c:919`, uses `int64_t`).

**libaom knows the bound** — `test/error_block_test.cc` declares
`msb = bit_depth + 8 - 1` and comments that coeff and dqcoeff "always have at
least the same sign" — but the kernel neither documents nor enforces it.

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

`av1_vaq_frame_setup` is only reached from `encode_with_recode_loop`
(`encoder.c:3495`), and `speed_features.c:2784` sets `DISALLOW_RECODE` for
`AOM_RC_ONE_PASS`. **MEASURED:** `--aq-mode=1` one-pass yields `seg.enabled=0`;
two-pass yields 1. If you need a segmented KEY frame from the C encoder, you
must ask for two passes.

### C2. The screen-content detector is a colour-count statistic, and flat content does not trip it

`estimate_screen_content` (`encoder.c:2042-2100`, reached via
`av1_set_screen_content_options`, `:2439`): the fraction of full 16×16 luma
blocks having 2..4 distinct `pix >> (bd-8)` values must be ≥ 10%.

- It is **bit-depth independent by construction** — `av1_count_colors_highbd`
  (`intra_mode_search.c:352-357`) down-converts to the 8-bit domain before
  binning, so bd8/bd10 twins of the same clip classify identically.
- **Counter-intuitive, MEASURED:** DC-flat content scores **0/16 blocks**.
  Quantisation ringing keeps colour counts high, so perceptual flatness is not
  this statistic.
- The default detector is `AOM_SCREEN_DETECTION_STANDARD`
  (`av1_cx_iface.c:405`); the AA-aware variant is TUNE_IQ/SSIMULACRA2-only
  (`:1969`).

### C3. Control combinations libaom forbids or silently drops

| combination | libaom's behaviour | citation |
|---|---|---|
| `--enable-tx-size-search=0` + `--enable-tx64=0` | hard assert | `encodeframe.c:2461` |
| superres + lossless | superres dropped (`all_lossless = coded_lossless && !av1_superres_scaled`) | `encodeframe.c:276` |
| superres + intrabc | spec-forbidden (`allow_intrabc` only read when `!superres_scaled`) | AV1 spec |

**Subtlety, MEASURED:** `--enable-tx-size-search=0` stops being a configuration
at speed ≥ 8 — `speed_features.c:2726-2729` is gated on `use_nonrd_pick_mode==0`
(set at `:579`) — so the forbidden pair above *lapses* there, and a harness that
asserts `TX_MODE_LARGEST` will panic on a stream aomenc produced happily.

### C4. Several speed features are framesize-gated *and* speed-gated

A framesize threshold alone does not make a feature live. `prune_tx_type_using_stats`
needs ≥480p **and** speed ≥ 2 (level 1) / ≥ 4 (level 2)
(`speed_features.c:261`, `:299`). We wasted analysis on the assumption that
adding ≥480p contexts would exercise it; at speed 0 it is 0 in real aomenc too.
The full framesize table is in `docs/CONFIG_PERMUTATION_DESIGN_2026-07-30.md`.

---

## How to add an entry

1. Cite `file:line` into `upstream/`, and say which tag applies (MEASURED /
   SOURCE / CI-CONFIRMED).
2. State how *we* handle it and where (KB number, module, test).
3. If libaom's own tests do not catch it, say why — that is usually the most
   useful sentence in the entry.
4. Prefer "NOT ESTABLISHED" to a plausible story. Two claims in this project's
   history were confidently wrong and had to be retracted (a "17 invalid AV1
   streams" finding that was a harness bug, and a root-cause attribution
   corrected mid-investigation). Both retractions are recorded rather than
   deleted, which is the standard here.
