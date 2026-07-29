# `cdef_find_dir` — i16-lane magetypes partial sums (SIMD_REACH_AUDIT F6), measured

Finding **F6** of `docs/SIMD_REACH_AUDIT_2026-07-28.md`: `cdef_find_dir` is
**2,486,880 Ir/decode = ~4.7 % of the q32 decode**
(`gate3_filters_2026-07-22.md`), C reaches `cdef_find_dir_avx2`, and the port
had **no SIMD tier at all**. This landing gives its partial-sum half one
`#[magetypes(define(i16x8, u16x8), v3, neon, wasm128, -scalar)]` kernel.

Baseline is **current `origin/main` 47aba59**, i.e. *after* `ea61406f`'s per-row
slice-add restructure (1570 → 442 Ir/call). The pre-restructure 1570 is NOT the
starting point for anything below.

## Headline (wall clock, Apple M4 Pro, two-build alternation)

| cdef-group row | touched? | before (median of 5) | after (median of 5) | Δ |
|---|---|---:|---:|---:|
| `cdef::find_dir_08x08` | **yes** | 47,435 ns | 35,899 ns | **−24.32 %** (1.32×) |
| `cdef::filter_u8_08x08` | no (control) | 92,547 ns | 92,299 ns | −0.27 % |
| `cdef::filter_u8_04x04` | no (control) | 132,987 ns | 133,084 ns | +0.07 % |

Min-of-5 cross-check: 47,167 → 35,576 ns = **−24.6 %** — 0.3 pp from the median
estimate. Raw per-run numbers: `cdef_find_dir_simd_2026-07-28.tsv`; method,
host, load and exact commands: `.meta`.

**Read the control band, not the single number.** The two untouched CDEF rows
are timed in the SAME interleaved rounds as `find_dir_08x08` and both land
inside ±0.3 % on medians-of-5, so a −24 % move on the touched row is signal.
That is only true because of the alternation: within a single run the spread on
*every* row (touched or not) is 22–40 %, which is what the whole-suite pair
below shows.

## Why a two-BUILD alternation and not `AOM_FORCE_SCALAR`

`AOM_FORCE_SCALAR=1` is a **no-op for the NEON tier** on aarch64 (neon is a
compile-time-guaranteed baseline feature; archmage refuses to disable it — see
the `dsp_kernels.rs` module docs, measured 2026-07-25). A pinned-vs-unpinned
pair on this box measures the same code twice. So the baseline is a separate
BUILD of the same bench crate, staged to a file, and the two binaries are run
strictly alternating.

## Secondary evidence: the whole-suite `--save-baseline` pair (noisy)

Two whole-suite runs against the same saved baseline, taken while a concurrent
agent held/released the zenbench exclusive lock:

| run | `cdef::find_dir_08x08` | widest untouched row (control band) |
|---|---:|---|
| 1 | **−25.08 %** | `intra::h_04x04` +12.05 % … `dist::sse_32x32` −6.90 % |
| 2 | **−10.17 %** | `intra::h_04x04` +11.38 % … `cdef::filter_u8_08x08` **+11.11 %** |

Run 2 is the honest illustration of why this method alone cannot carry the
claim: an **untouched** CDEF row moved +11.11 % in it. Both runs are reported
rather than the flattering one. The alternation above is the number to quote.

## What changed (shape)

`cdef_find_dir` is now a router:

* `cdef_find_dir_simd_eligible(img, stride, cs)` — a **checked value
  predicate**, `(px >> cs) <= 255` for all 64 pixels, evaluated per call.
* eligible → `cdef::simd::cdef_find_dir_partials` (the i16 kernel), then the
  shared cost fold.
* otherwise, and under the scalar pin → `cdef_find_dir_scalar`, the transcribed
  i32 port, unchanged.

Bit-identity is therefore unconditional and needs **no range assumption**: in
the eligible domain every partial is bounded by `|.| <= 2048` (lines
`[-128, 127]`; ≤ 8 line contributions per slot for d0/d2/d4/d5/d6/d7, ≤ 8 pair
folds of `|pf| <= 256` for d1/d3), so each i16 partial holds the **same
integer** as the scalar port's i32 partial and the shared fold — which keeps the
C's `wrapping_*` ops and the normative `>`-scan tie-break in ONE copy — returns
the same `(dir, var)`. The eligible domain is also everything the decoder can
produce (`cs == bd - 8`, the find_dir window is interior plane data), so the
fallback is a safety net, not the common path.

Two rewrites drop work the scalar port pays for, both exact regroupings of a
commuting wrapping-add set:

1. **No lane reversals.** d3/d4 accumulate in reversed slot order
   (`q4[7-i+j] += row[j]`, `q3[7-i+k] += pf[k]`), so the `rev`/`pfr` arrays
   disappear; the entry's accessor undoes the flip.
2. **Pair folding for d5/d6/d7**, whose per-row offsets (`3-i/2`, `0`, `i/2`)
   are shared by rows `2k`/`2k+1` — add the row pair once, slice-add once.

## Gates

* `cdef_diff::cdef_find_dir_matches_c` — the REAL C `cdef_find_dir_c`, 600 k
  cases across `coeff_shift` 0/2/4. Every one of those cases is **eligible**, so
  this now gates the SIMD path directly against C.
* `tests/cdef_find_dir_simd_diff.rs` (new) — dispatching entry vs the
  never-dispatched `cdef_find_dir_scalar` under `for_each_token_permutation`,
  25 permutations × 10,800 cases (201,600 vector-kernel accepts per run), with:
  * the `simd_perms >= 1` non-vacuity guard (F4);
  * a **deeper** non-vacuity guard — `cdef_find_dir_took_simd_path` reports
    whether the kernel actually ACCEPTED, and the routing must equal
    `cdef_find_dir_simd_eligible` in every live vector permutation. Without it
    the F4 guard is not enough here: the entry is bit-identical whichever route
    it takes, so a kernel that declined everything would pass a `simd_perms`-only
    check while comparing the scalar port against itself;
  * an all-8-directions-reached floor and a `var != 0` floor;
  * both sides of the eligibility predicate — border sentinel, and full-`u16`
    values ≥ 0x8000 which is a live check that the guard's `simd_gt` is an
    UNSIGNED compare.
* Suite: **361/361** in BOTH dispatch modes (default and `AOM_FORCE_SCALAR=1`);
  base `origin/main` was 359/359.

**A liveness check in the LIBRARY test binary was tried first and reverted.**
It was flaky: `for_each_token_permutation` mutates process-global token
availability, and a fourth permutation sweeper in `--lib` widened the existing
race with `dispatch::tests::disable_sweep_kills_summon_and_reenables` (base:
6/6 pinned lib runs green; with the lib test: intermittent failures in *that*
pre-existing test). Reading the kernel's decision from inside the differential's
own binary is race-free and strictly stronger. Related measured fact, recorded
because it will bite the next agent: under `AOM_FORCE_SCALAR=1` the permutation
set a test binary sees is **not deterministic** — both 2-permutation and
25-permutation pinned runs were observed for the same binary, depending on which
test thread latches the pin first. The aggregate non-vacuity floor is therefore
asserted in the unpinned CI leg (`force_scalar: "0"`), with the per-permutation
routing contract asserted unconditionally.

### Teeth (each perturbation reverted, `git diff` clean afterwards)

All four re-run against the FINAL code; each reverted immediately, `git diff`
verified clean afterwards.

| # | perturbation | result |
|---|---|---|
| 1 | `add8(&mut pa[5], 3 - k, tmp)` → `2 - k.min(2)` (d5 slot offset) | **FAILS**: `[all enabled] cdef_find_dir divergence: cs=0 stride=8 flavour=4 eligible=true img=[17, 17, …]  left: (5, 74118)  right: (0, 0)` |
| 2 | `if vec_live { simd_perms += 1 }` → `&& false` | **FAILS**: `the SIMD permutation (neon) must run at least once — a passing run with zero vector permutations compares the scalar path against itself.` |
| 3 | guard limit `(256 << cs) - 1` → `u16::MAX` (widen the domain) | **FAILS**: `[all enabled] routing diverged from cdef_find_dir_simd_eligible: cs=0 stride=8 flavour=7  left: true  right: false` |
| 4 | kernel returns `false` unconditionally (declines every window) | **FAILS**: `[all enabled] routing diverged from cdef_find_dir_simd_eligible: cs=0 stride=8 flavour=0  left: false  right: true` |

**A finding worth recording: the `- 128` DC offset is provably INERT** and is
therefore *useless as a teeth perturbation* (the first one tried; the test
passed and the SIMD path had to be proven live another way). `DIV_TABLE[n] =
840/n` is exactly the reciprocal of a slot's contribution count, so
`cost[d] = 840 · Σ_k n_k · mean_k²`; adding a constant `c` to every pixel sends
`mean_k → mean_k + c` and shifts **every** direction's cost by the same
`840·(2cS + 64c²)` (`S` = the block sum, `Σ n_k = 64` for all `d`). The argmax
and `var = best − opposite` are both invariant. Perturb a **slot offset** or the
**guard**, not the offset constant.

## Chunk 2 — what is left on the table

The kernel keeps its eight accumulators in a stack table and shifts the
**address**; `cdef_find_dir_avx2` keeps them in registers and shifts the
**lanes** (`v128_shl_n_byte`), which removes ~40 load/add/store triples per
call and their store-to-load-forwarding chains. Blocking op: magetypes 0.9.27
has **no lane shift, no cross-lane permute, no integer widen/narrow and no
madd** for i16 lanes (verified against the 0.9.28 public-API snapshot at
`~/work/archmage/docs/public-api/magetypes.txt`; `interleave` / `transpose_8x8`
/ `cross_width` are **f32-only**). Register residency therefore needs per-tier
`#[rite]` primitives in the `transform/simd/prims.rs` Pattern-C shape:

1. `shift_lanes_left::<N>` / `shift_lanes_right::<N>` on `i16x8`
   (x86: `_mm_slli_si128`/`_mm_srli_si128`; NEON: `vextq_s16` against a zero
   vector). This alone converts the eight `add8` triples per direction into
   shift+add pairs.
2. `reverse_transpose_8x8` on `i16x8` (x86: `unpack`/`shuffle` ladder; NEON:
   `vtrn`/`vzip`), which would let ONE `compute_directions` body serve d0-d3 and
   d4-d7 the way libaom's does, halving the accumulation source.
3. i16→i32 widening for the cost fold (x86 `_mm_cvtepi16_epi32`, NEON
   `vmovl_s16`) if the fold is vectorised too.

Estimated ceiling from libaom's own instruction count: `cdef_find_dir_avx2`
lands near ~240 128-bit ops for the whole function, versus 442 Ir/call for the
port's scalar and (unmeasured in Ir) something between for this kernel — so
another ~1.3–1.5× on the row, i.e. ~1–1.5 % of q32 decode. **Ir was not
measured here**: callgrind is not available on this aarch64 box, so the q32
Ir share of this landing is *not* claimed. The next agent with the Linux/x86
profiling box should re-run the `gate3_filters` q32 cell to convert the −24 %
wall number into an Ir/decode delta before quoting one.

The `hadamard_*` / `highbd_hadamard_*` SATD family (F6's other half) is
untouched and still fully scalar.
