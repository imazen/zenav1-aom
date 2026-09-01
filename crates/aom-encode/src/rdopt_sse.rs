//! The two `av1/encoder/rdopt.c` decisions that read the encoder's VARIANCE
//! FUNCTION TABLE: `get_sse` (`:868`) and `prune_zero_mv_with_sse` (`:2809`).
//!
//! Tier 1c. Gate: `crates/aom-encode/tests/rdopt_sse_diff.rs`.
//!
//! # Why these were blocked until now
//!
//! Both read `aom_variance_fn_ptr_t::vf`, and libaom fills that table with a
//! `BFP()` macro cascade INLINE inside `av1_create_primary_compressor` — there
//! is no separately callable initialiser a shim could invoke. The shim now
//! rebuilds the table from the same exported `aom_variance<W>x<H>` entry
//! points libaom itself assigns, so the oracle dispatches exactly as the
//! encoder would.
//!
//! The port calls [`aom_dsp::dist::variance`], which is separately gated
//! against `aom_variance*_c`. Variance is an integer sum, so every SIMD tier
//! agrees with the scalar one bit-for-bit; that is what makes composing the
//! two ports legitimate here rather than a second transcription.

use aom_dsp::dist::variance;

/// `AOM_PLANE_Y`.
pub const AOM_PLANE_Y: usize = 0;

/// One plane's source and prediction, as `get_sse` reads them.
pub struct SsePlane<'a> {
    /// `x->plane[p].src`.
    pub src: &'a [u8],
    /// `x->plane[p].src.stride`.
    pub src_stride: usize,
    /// `xd->plane[p].dst`.
    pub dst: &'a [u8],
    /// `xd->plane[p].dst.stride`.
    pub dst_stride: usize,
    /// The plane's block dimensions in pixels (`get_plane_block_size` applied
    /// to the luma bsize, then `block_size_{wide,high}`).
    pub w: usize,
    /// See [`Self::w`].
    pub h: usize,
}

/// `get_sse` (rdopt.c:868): the summed prediction SSE over every coded plane,
/// scaled by 16, plus the luma-only value the caller also wants.
///
/// Returns `(total_sse, sse_y)`.
///
/// The `<< 4` is applied to the TOTAL after summing; that is where C puts it,
/// though it is distributive over the sum and so not observable. Likewise C's
/// `if (plane && !xd->is_chroma_ref) break;` is a `break` rather than a
/// `continue`, which is also not observable because the condition does not
/// depend on the plane beyond `plane != 0`. Both are written C's way, and both
/// were confirmed INERT by perturbation rather than assumed to matter.
pub fn get_sse(planes: &[SsePlane<'_>], is_chroma_ref: bool) -> (i64, i64) {
    let mut total = 0i64;
    let mut sse_y = 0i64;
    for (p, plane) in planes.iter().enumerate() {
        if p > 0 && !is_chroma_ref {
            break;
        }
        let (_var, sse) = variance(
            plane.src,
            plane.src_stride,
            plane.dst,
            plane.dst_stride,
            plane.w,
            plane.h,
        );
        total += i64::from(sse);
        if p == 0 {
            sse_y = i64::from(sse);
        }
    }
    (total << 4, sse_y)
}

/// `IDENTITY` (`warped_motion.h`).
pub const IDENTITY: i32 = 0;
/// `INT32_MAX`, C's "no valid single-reference SSE recorded" sentinel.
pub const NO_SINGLE_SSE: u32 = i32::MAX as u32;

/// `prune_zero_mv_with_sse` (rdopt.c:2809): skip a zero-MV candidate whose
/// prediction is measurably worse than the best NEWMV already found.
///
/// Three preconditions, each of which makes the answer `false` outright:
/// every reference must use IDENTITY global motion (C's own comment explains
/// that TRANSLATION would work in theory but is not coded, due to a spec bug
/// it points at `gm_get_motion_vector`), and every reference must have a
/// recorded single-reference SSE.
///
/// The final comparison is done in `f64` — C's, and it matters: the threshold
/// is 1.25 at level 1, which is not representable as an integer ratio without
/// changing the rounding at the boundary.
pub fn prune_zero_mv_with_sse(
    ref_frames: [i32; 2],
    gm_wmtype: &[i32; 8],
    best_single_sse_in_refs: &[u32; 8],
    zero_mv_sse: &[u32],
    level: i32,
) -> bool {
    let is_comp_pred = ref_frames[1] > 0;
    let n = usize::from(is_comp_pred) + 1;
    for &r in ref_frames.iter().take(n) {
        if gm_wmtype[r as usize] != IDENTITY {
            return false;
        }
        // C's "don't prune if we have invalid data" guard. Note it is
        // OBSERVATIONALLY INERT on the reachable input domain and the
        // differential cannot distinguish it: an SSE is at most
        // 255^2 * 128 * 128 < INT32_MAX, so a sum containing the sentinel
        // never wraps and the final comparison is false either way. It is
        // kept because it is C's guard, not because the test proves it.
        if best_single_sse_in_refs[r as usize] == NO_SINGLE_SSE {
            return false;
        }
    }
    // C accumulates both sums in `unsigned int`, so they WRAP rather than
    // saturate; reproduced with wrapping adds.
    let mut this_sse_sum = 0u32;
    let mut best_sse_sum = 0u32;
    for (idx, &r) in ref_frames.iter().take(n).enumerate() {
        this_sse_sum = this_sse_sum.wrapping_add(zero_mv_sse[idx]);
        best_sse_sum = best_sse_sum.wrapping_add(best_single_sse_in_refs[r as usize]);
    }
    let mul = if level > 1 { 1.00f64 } else { 1.25f64 };
    f64::from(this_sse_sum) > mul * f64::from(best_sse_sum)
}
