# KB-PERF-22 — the wiener convolve: `vpmulld` -> `vpmaddwd`, 16 columns per iteration — LANDED, −0.40 %

**2026-09-09.** Byte-identical; **−0.40 %** at 1024x1024 cq27 `--cpu-used 3`
(mean of two base copies: **−0.247 %** and **−0.553 %**, both significant), over
a 48-round rotated band. The third and last of the levers the intrinsics probe
unblocked.

**It landed exactly where it was costed.** `encoder_wiener_uop_analysis_2026-09-09.md`
predicted **−0.3 to −0.4 %** for the 16-column shape *before any code was
written*, and warned that the obvious 8-column shape would measure null. That
prediction is what this landing tests, and it held.

## The defect

`restore::wiener::wiener_impl_v3` was **40.9 ms against
`av1_wiener_convolve_add_src_avx2`'s 6.2 = 6.60x, +34.7 ms** — the worst ratio
in `encoder_simd_lane_width_audit_2026-09-09.md`. The disassembly said why:
**16 `vpmulld` and ZERO `vpmaddwd`**, where `vpmulld` is ~10 cycles / 2 uops for
one multiply per lane and `vpmaddwd` is ~5 / 1 for two multiply-accumulates per
lane pair.

## What was built

Both passes share one shape — `out[c] = clamp(round(bias + (in[c+3*step] << 7)
+ sum_k in[c + k*step] * tap[k]))` — with `step` = one sample horizontally and
one row vertically, so ONE helper serves both:

* two source loads `step` apart, `unpacklo`/`unpackhi` to interleave them into
  `(in[c+2j], in[c+2j+1])` pairs, `vperm2i128` to straighten AVX2's
  per-128-lane result, then **`vpmaddwd` against a broadcast tap pair**;
* four tap pairs x two halves = **8 `vpmaddwd` for 16 output columns**;
* `vpackusdw` + `vpermq` to narrow and store 16 `u16` directly.

Emitted, from the shipped binary: `4 vpunpcklwd, 4 vpunpckhwd, 4 vperm2i128,
8 vpmaddwd, 2 vpmovzxwd, 2 vpsrad, 2 vpmaxsd, 2 vpminsd, 1 vpackusdw, 1 vpermq`
— and **no `vpmulld`**.

## No range obligation — proven, not gated

`_mm256_madd_epi16` takes i16 inputs and accumulates in **i32**; nothing is
narrowed. Horizontal inputs are samples `<= (1<<bd)-1 <= 4095`. Vertical inputs
are the intermediate, whose clamp ceiling `conv_params_wiener` pins at **exactly
32767 = i16::MAX** at every bit depth — it raises `round_0` precisely when
`bd + FILTER_BITS - round_0 + 2 > 16`. Four madd results sum to ~33.5 M, inside
i32. So this runs at bd 8/10/12 with **no runtime gate and no audit**, which is
a real simplification over what the first wiener record claimed.

## Correctness

* Encoder output **byte-identical** on four cells: 40,237 B (1024x1024 s3),
  39,694 (1024 s0), 10,912 (512 s0), 11,961 (512 s6).
* `-p zenav1-aom-dsp` **395/395 in both dispatch modes**; the
  `AOM_FORCE_SCALAR` pin and any `w < 16` decline to the untouched i32x8 loop.
* **Bite proof, asymmetric**: swapping the two halves of one madd accumulate
  fails **4** tests — `kernels_diff::wiener_convolve_matches_c` (real exported
  C), `wiener_simd_diff::wiener_simd_bit_identical_to_scalar_at_every_tier`,
  `frame_walk_diff::lr_filter_frame_matches_c` and
  `pick_search::noisy_recon_search_improves_sse_and_is_deterministic` — while
  **391 stay green**.

## The band, and why it is quoted as a range

48 rounds (sized by the uop analysis, which called n=24 marginal for a ~0.35 %
effect), rotated arms, TWO copies of the base binary:

| comparison | paired median | rounds faster | p |
|---|---:|---:|---:|
| `baseB` vs `base` (same binary) | **+0.286 %** | 4/48 | <0.0001 |
| wiener vs `base` | **−0.247 %** | 32/48 | 0.0293 |
| wiener vs `baseB` | **−0.553 %** | 37/48 | 0.0002 |

**Both comparisons are negative and significant, and the spread between them IS
the copy systematic** — `encoder_rotate_reverify_2026-08-03.md` measures that at
~0.27 pp and records that rotation cannot remove it (cyclic rotation fixes each
arm's predecessor). So the honest number is **−0.40 % +- 0.15 pp**, the mean of
the two, and NOT the more flattering −0.553 %.

## Not covered

* **`w < 16` keeps the i32x8 path** — restoration units narrower than 16 px get
  nothing.
* **aarch64 and wasm get nothing**: `cfg(target_arch = "x86_64")`.
* **The remaining ~2x against libaom is the plane representation, not this
  kernel.** libaom's source is `u8`, so a 256-bit load holds 32 samples and its
  pairs land in the right halves with no `vperm2i128`; the port holds `u16` at
  every bit depth. That is the u16-at-bd8 structural root, and it is not
  reachable from inside a convolve.

## Found while here: `cdef_find_dir_simd_diff` is FLAKY, and that is a gate hole

It failed once during this work, on a tree whose only change was in
`restore/wiener.rs`. Investigated rather than assumed: the full suite then passed
**395/395 three times**, and the cdef test **run ALONE, three times, with no code
change, failed 1 of 3**. So it is genuinely non-deterministic.

This also corrects KB-PERF-20's note (which called it "reproducible") and
KB-PERF-21's (which downgraded it to "two of three perturbations"): the real
explanation is neither — it is flaky on its own. **A non-deterministic
differential is a hole in the gate**: it can mask a real regression, and it cost
time here by implicating an unrelated change. Worth its own fix.
