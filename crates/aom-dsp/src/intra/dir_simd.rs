//! **i16-lane** vector kernel for the two-tap directional intra interpolation —
//! the inner arithmetic of `av1_dr_prediction_z{1,2,3}` — plus the runtime bound
//! that admits it.
//!
//! # Why this exists
//!
//! [`super::dir`]'s `z1_high` / `z2_high` / `z3_high` were **pure scalar at every
//! bit depth**, while libaom dispatches `av1_dr_prediction_z{1,2,3}_neon` (and
//! `_avx2` / `_sse4_1`) on the lowbd path. That is the same structural gap the
//! forward transform had (`transform::simd::lowbd16_fwd`), one stage over: the
//! port ran the wide/scalar path where libaom runs a narrow-lane vector one.
//! Measured at the profile cell, the three kernels were **3.50 ms of a 150 ms
//! encode against libaom's ~0.37 ms** — see
//! `benchmarks/encoder_intra_dir_i16_2026-08-03.md`.
//!
//! The body is ONE `#[magetypes(define(i16x16), v3, neon, -scalar)]` function,
//! so it serves NEON and AVX2 from the same source (the cross-platform half of
//! the bd8 lane-width programme — `benchmarks/winperf_windows_2026-08-02.md`
//! §CROSS-PLATFORM SCOPING).
//!
//! # The identity, and why i16 lanes are exact
//!
//! The scalar kernel computes, per output,
//!
//! ```text
//! (a0 * (32 - shift) + a1 * shift + 16) >> 5      with a0 = edge[b], a1 = edge[b+1]
//! ```
//!
//! The vector form uses libaom's re-association (`intrapred_neon.c:1307-1308`)
//!
//! ```text
//! a0 * (32 - shift) + a1 * shift  ==  (a0 << 5) + (a1 - a0) * shift
//! ```
//!
//! which is an identity over the integers, so the two agree **exactly** provided
//! no i16 lane wraps. With `shift ∈ [0, 31]` (it is `((x << up) & 0x3F) >> 1`)
//! and every tap `0 <= v <= M` the three intermediates are bounded by
//!
//! * `a0 << 5` — `<= 32 * M`,
//! * `a1 - a0` — `|.| <= M`, and `(a1 - a0) * shift` — `|.| <= 31 * M`,
//! * the sum, which equals `a0*(32-shift) + a1*shift` — a convex combination
//!   scaled by 32, so `∈ [0, 32 * M]`,
//! * `+ 16` — `<= 32 * M + 16`.
//!
//! Every one of those is inside `i16` iff `32 * M + 16 <= 32767`, i.e.
//! **`M <= 1023`**. That is [`I16_TAP_MAX`], and it is the whole audit: it is
//! **tight** (at `M = 1024`, `a0 << 5` is exactly `-32768` and the result is
//! wrong — pinned by `gate_bite::the_tap_bound_is_load_bearing`), and it is
//! taken at RUNTIME on the actual edge span, so the path is sound for any
//! caller of the public predictors and not only for bd8. In bit-depth terms it
//! admits **bd8 and bd10** (samples `<= 1023`) and declines bd12, which is the
//! honest statement of its reach — the gate is on the data, not on `bd`.
//!
//! The final `>> 5` is an arithmetic shift of a value in `[0, 32752]`, which is
//! the scalar `>>` on the same non-negative value, so the narrowing `as u16` is
//! exact.
//!
//! # Scope — what runs vector and what does not
//!
//! Only **contiguous** tap runs (`base_inc == 1`, i.e. `upsample == 0`) take the
//! vector path, because then the two operand vectors are plain unaligned loads
//! of `edge[b..b+16]` and `edge[b+1..b+17]` with no staging array at all. The
//! `upsample == 1` runs are a stride-2 gather and stay scalar; they are
//! **12.6 % of z1 and 14.9 % of z3 pixels** at the profile cell (upsampling is
//! only ever enabled for `bw + bh <= 16`, `edge::use_upsample`), and the census
//! is in the writeup. `z2`'s left-hand half is a genuine gather (`base_y` is not
//! affine in `c`) and likewise stays scalar — it is 50.2 % of z2's pixels.
//!
//! Runs shorter than [`MIN_VEC_RUN`] stay scalar: a 4-wide block cannot fill
//! enough of a 16-lane vector to pay for the round trip through the stack array.

use archmage::prelude::*;

/// The largest edge sample for which every i16 lane intermediate is exact.
/// `32 * 1023 + 16 = 32752 <= i16::MAX`; `32 * 1024 = 32768` is not.
pub(crate) const I16_TAP_MAX: u16 = 1023;

/// Shortest run given to the vector kernel. Below this the array round trip
/// costs more than the 16 scalar multiply-adds it replaces.
pub(crate) const MIN_VEC_RUN: usize = 8;

/// `true` if every sample in `edge[lo..=hi]` is inside the i16 lane bound.
/// `O(hi - lo)` — the caller's spans are `O(bw + bh)` against `O(bw * bh)` of
/// predictor work, so this is a per-block scan, never a per-pixel one.
#[inline]
pub(crate) fn span_fits_i16(edge: &[u16], lo: usize, hi: usize) -> bool {
    hi < edge.len() && lo <= hi && edge[lo..=hi].iter().all(|&v| v <= I16_TAP_MAX)
}

/// The scalar two-tap run — the differential reference AND the tail/decline
/// path. Byte-identical to the expression in [`super::dir`] by construction.
#[inline]
pub(crate) fn two_tap_run_scalar(out: &mut [u16], edge: &[u16], start: usize, shift: i32, n: usize) {
    for (i, o) in out.iter_mut().take(n).enumerate() {
        let a0 = edge[start + i] as i32;
        let a1 = edge[start + i + 1] as i32;
        *o = ((a0 * (32 - shift) + a1 * shift + 16) >> 5) as u16;
    }
}

/// Dispatch entry: write `n` outputs of the contiguous two-tap run starting at
/// `edge[start]` into `out[..n]`.
///
/// PRECONDITIONS (the caller's, and all three are what the scalar kernel already
/// requires plus the bound): `start + n < edge.len()`, `shift ∈ [0, 31]`, and
/// every sample in `edge[start ..= start + n]` `<= I16_TAP_MAX`. Callers take
/// the last one with [`span_fits_i16`] once per block.
pub(crate) fn two_tap_run(out: &mut [u16], edge: &[u16], start: usize, shift: i32, n: usize) {
    let _ = crate::dispatch::scalar_forced(); // one-time AOM_FORCE_SCALAR pin
    incant!(
        two_tap_run_impl(out, edge, start, shift, n),
        [v3, neon, scalar]
    )
}

fn two_tap_run_impl_scalar(
    _t: archmage::ScalarToken,
    out: &mut [u16],
    edge: &[u16],
    start: usize,
    shift: i32,
    n: usize,
) {
    two_tap_run_scalar(out, edge, start, shift, n);
}

#[magetypes(define(i16x16, u16x16), v3, neon, -scalar)]
fn two_tap_run_impl(
    token: Token,
    out: &mut [u16],
    edge: &[u16],
    start: usize,
    shift: i32,
    n: usize,
) {
    if n < MIN_VEC_RUN {
        two_tap_run_scalar(out, edge, start, shift, n);
        return;
    }
    let sv = i16x16::splat(token, shift as i16);
    let round = i16x16::splat(token, 16);
    let mut buf = [0i16; 16];
    let mut i = 0;
    // A chunk needs 17 in-range samples. For a FULL chunk that is implied by the
    // caller's `start + n < edge.len()`; the guard binds only on a partial tail
    // (n == 8 is the common one — an 8-wide block).
    while i < n && start + i + 17 <= edge.len() {
        let m = (n - i).min(16);
        let idx = start + i;
        let v0 = u16x16::from_slice(token, &edge[idx..idx + 16]).bitcast_i16x16();
        let v1 = u16x16::from_slice(token, &edge[idx + 1..idx + 17]).bitcast_i16x16();
        let res = (v0.shl_const::<5>() + (v1 - v0) * sv + round).shr_arithmetic_const::<5>();
        res.store(&mut buf);
        for (k, o) in out[i..i + m].iter_mut().enumerate() {
            *o = buf[k] as u16;
        }
        i += m;
    }
    if i < n {
        two_tap_run_scalar(&mut out[i..], edge, start + i, shift, n - i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every token permutation, against the scalar core, over the full admitted
    /// domain. Probes are asymmetric (a flat edge is invariant under the
    /// re-association being tested — playbook §1 / KB-12).
    #[test]
    fn two_tap_matches_scalar_at_every_tier() {
        let mut edge = vec![0u16; 200];
        let mut s = 0x1234_5678u32;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let mut vector_cells = 0usize;
        for rep in 0..8 {
            for (i, e) in edge.iter_mut().enumerate() {
                *e = match rep {
                    0 => (next() % (I16_TAP_MAX as u32 + 1)) as u16, // dense random
                    1 => {
                        if i % 2 == 0 {
                            I16_TAP_MAX
                        } else {
                            0
                        }
                    } // max sawtooth
                    2 => (i as u16) % 256,                        // ramp
                    3 => I16_TAP_MAX,                             // flat max
                    4 => (next() % 256) as u16,                   // bd8 range
                    5 => 255 - (i as u16 % 256),                  // reverse ramp
                    6 => {
                        if i < 100 {
                            0
                        } else {
                            I16_TAP_MAX
                        }
                    } // step
                    _ => (next() % 1024) as u16,
                };
            }
            for &n in &[1usize, 4, 7, 8, 9, 15, 16, 17, 31, 32, 64] {
                for shift in 0..32i32 {
                    for &start in &[0usize, 1, 3, 16, 100] {
                        if start + n + 1 > edge.len() {
                            continue;
                        }
                        let mut got = vec![0u16; n];
                        let mut want = vec![0u16; n];
                        two_tap_run(&mut got, &edge, start, shift, n);
                        two_tap_run_scalar(&mut want, &edge, start, shift, n);
                        assert_eq!(got, want, "n={n} shift={shift} start={start} rep={rep}");
                        if n >= MIN_VEC_RUN {
                            vector_cells += 1;
                        }
                    }
                }
            }
        }
        // Non-vacuity: the vector body must actually have been reachable.
        assert!(vector_cells > 1000, "vector arm unreached ({vector_cells})");
    }

    /// Playbook §2 — the bound must BITE. One tap over `I16_TAP_MAX` and the
    /// i16 lanes must genuinely diverge from the scalar reference, else the
    /// gate is decorative.
    ///
    /// The divergence half is necessarily conditional on a VECTOR tier actually
    /// dispatching: under `AOM_FORCE_SCALAR=1` `two_tap_run` routes to
    /// `two_tap_run_scalar`, so it cannot diverge from itself, and asserting
    /// otherwise fails the scalar-pinned CI leg (it did, on the first run). The
    /// gate's own rejection is asserted UNconditionally — that half is pure
    /// arithmetic on the span and has no tier.
    #[test]
    fn the_tap_bound_is_load_bearing() {
        let n = 16;
        let mut edge = vec![I16_TAP_MAX; 64];
        // At exactly the bound, every shift agrees. True at every tier.
        for shift in 0..32i32 {
            let (mut got, mut want) = (vec![0u16; n], vec![0u16; n]);
            two_tap_run(&mut got, &edge, 0, shift, n);
            two_tap_run_scalar(&mut want, &edge, 0, shift, n);
            assert_eq!(got, want, "at the bound, shift={shift}");
        }
        // The gate rejects one over the bound, and accepts the bound itself.
        edge[3] = I16_TAP_MAX + 1;
        assert!(!span_fits_i16(&edge, 0, 16), "1024 must be rejected");
        edge[3] = I16_TAP_MAX;
        assert!(span_fits_i16(&edge, 0, 16), "1023 must be accepted");

        if crate::dispatch::scalar_forced() {
            return; // no vector tier to diverge; the half above still ran
        }
        // One over the bound, and the vector path is wrong for at least one
        // shift — else the gate guards nothing.
        edge[3] = I16_TAP_MAX + 1;
        let mut diverged = false;
        for shift in 0..32i32 {
            let (mut got, mut want) = (vec![0u16; n], vec![0u16; n]);
            two_tap_run(&mut got, &edge, 0, shift, n);
            two_tap_run_scalar(&mut want, &edge, 0, shift, n);
            if got != want {
                diverged = true;
            }
        }
        assert!(
            diverged,
            "the i16 tap bound never bites — the gate would be decorative"
        );
    }
}
