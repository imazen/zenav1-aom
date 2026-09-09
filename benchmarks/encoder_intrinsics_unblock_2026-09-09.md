# Three "blocked on magetypes vocabulary" levers are NOT blocked — raw intrinsics compile under `forbid(unsafe_code)`

**2026-09-09. Measured by compiling, not argued.** `CLAUDE.md` and three
KB-PERF records state that `wiener`, `hadamard` and `quantize_fp` are blocked on
primitives magetypes does not have. The vocabulary half of that is **confirmed
true**; the CONCLUSION is **false**.

## The vocabulary claim, re-verified against the pinned version

`Cargo.lock` pins **magetypes 0.9.29** (not 0.9.28, as several entries say —
that is itself a correction). Searching the whole crate:

| op | present? |
|---|---|
| `interleave_lo` / `interleave_hi` / `interleave` / `transpose_4x4` / `transpose_8x8` | **f32x4 / f32x8 ONLY** (`block_ops_f32x*`) |
| the same for ANY integer type (`i16x8/16`, `i32x4/8`, `u8x32`) | **absent** |
| `mul_high` | **absent everywhere** |
| `madd_adjacent` | present (20 sites) |
| `narrow_saturating_i32_to_i16` / `_i16_to_u8` | present |

So the vocabulary notes are RIGHT: there is no integer interleave, no integer
transpose, and no `mul_high` at 0.9.29.

## But the levers are not blocked, because archmage re-exports `core::arch`

`archmage::intrinsics::x86_64` glob-imports `core::arch::x86_64::*` (value
intrinsics) and shadows the pointer-based memory ops with
`safe_unaligned_simd`'s reference-based ones.

**Probe, compiled inside `zenav1-aom-dsp`, which is `#![forbid(unsafe_code)]`:**

```rust
#[rite(v3)]
pub fn probe_unpack(a: i32, b: i32) -> i32 {
    use archmage::intrinsics::x86_64::*;
    let va = _mm256_set1_epi16(a as i16);
    let vb = _mm256_set1_epi16(b as i16);
    let lo = _mm256_unpacklo_epi16(va, vb);   // the "missing" integer interleave
    let m  = _mm256_mulhi_epi16(lo, vb);      // the "missing" mul_high
    _mm256_extract_epi16::<0>(m) as i32
}
```

**It compiles.** rustc 1.98.1, release, no `unsafe` block, `forbid(unsafe_code)`
in force — because a `#[rite(v3)]` body carries `target_feature(avx2)` and
target-feature-11 makes matching-feature intrinsic calls safe from inside it.

The probe was removed after the check; nothing shipped.

## What that unblocks, with the specific instruction each needs

| lever | gap at the shipping preset | "blocked on" | the intrinsic |
|---|---:|---|---|
| `quantize_fp` | **+54.6 ms** (3.23x) | `mul_high` | `_mm256_mulhi_epi16` |
| `wiener` | **+34.7 ms** (6.61x) | integer interleave | `_mm256_unpacklo_epi16` / `_hi_` |
| `hadamard` | **+26.5 ms** (3.04x) | integer 8x8 transpose | the `unpack`/`permute` family |
| | **+115.8 ms = 5.9 % of the gap** | | |

Each is the SAME instruction libaom's own AVX2 kernel uses, so this is the
directive "wherever C references AVX/SSE, port the same" applied literally
rather than routed around.

## Risk, stated per lever, because they are not equal

* **`hadamard` is the safe one to do first.** KB-PERF-10 already established the
  exactness argument (the trailing transpose is FUSED, and KB-12 is the standing
  proof that losing it moves only the `eob`); it needs a transpose and nothing
  numerically new.
* **`wiener` is next.** Its i16 convolution is a straight lane-width narrowing
  with a bound that `xtask/audit_i16_fwd.py`'s method already covers.
* **`quantize_fp` is the biggest and the MOST DANGEROUS, and should not be done
  casually.** KB-20 is the standing warning: `av1_quantize_fp`'s SIMD tiers
  **disagree with `_c` and with each other outside `int16`** — NEON truncates
  (`vmovn_s32`), AVX2 saturates (`_mm_packs_epi32`) — and the port already has to
  model the DISPATCHED variant for the nonrd arm
  (`nonrd_pickmode::quantize_fp_dispatched`). This kernel feeds the bitstream, so
  moving it to i16 changes an ISA-conditional domain that only some cells catch.
  It needs its own landing with the KB-20 analysis redone for this call site.

## Also corrected here

`archmage` also exposes `#[rite(v3)]` (a tier-based `target_feature` body with no
token needed) and `#[autoversion]`. The lane-width audit's **loopfilter row was
overstated**, the same asymmetric-pairing error as its filter_intra row: it
counted the port's `loop_filter_frame_stop` (12.6 ms) but no C frame driver.
Like for like the kernel is **44.8 ms vs 12.5 = 3.58x, +32.3 ms**.

**And the loopfilter's `define(i32x4)` is NOT wasted width**, which the audit
implied: one `aom_lpf_*` call filters exactly **4 edge positions** and those are
the 4 lanes. The real gap is that libaom leans on `_dual` (8 positions) and
`_quad` (16) variants — visible in the C profile, `aom_lpf_vertical_14_dual_sse2`
being its largest at 2.3 ms — on **u8** data. Closing it needs the frame driver
to batch adjacent segments AND a narrower type, i.e. a programme, not a lever.
