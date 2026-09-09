//! `(int)double` cast semantics on the `--deltaq-mode=3` wiener-variance chain
//! (`allintra_vis.c`) — the ISA-dependent hole under `WeberVarMap`.
//!
//! **What this file exists to pin.** `get_window_wiener_var`
//! (allintra_vis.c:209-210) ends in `(int)(((base_num + base_reg) /
//! (base_den + base_reg)) / mb_count)`, and the frame normalizer ends in
//! `(int64_t)(exp(sb_wiener_log / sb_count))` (`:509`, `:678`). All three
//! operands are `double`s with no bound of their own, so the conversion can
//! leave the destination range — and out-of-range float->integer conversion is
//! **undefined behaviour in C** (C11 6.3.1.4 p1). "What libaom does" is
//! therefore a property of the ISA:
//!
//! | | x86-64 `cvttsd2si` | aarch64 `fcvtzs` | Rust `as` |
//! |---|---|---|---|
//! | `> INT_MAX` | `INT_MIN` | `INT_MAX` | `INT_MAX` |
//! | `< INT_MIN` | `INT_MIN` | `INT_MIN` | `INT_MIN` |
//! | `NaN` | `INT_MIN` | `0` | `0` |
//!
//! So **the port already agrees with an aarch64 libaom and disagrees with an
//! x86-64 one**, and the disagreement is not small: `.max(1)` turns C's
//! `INT_MIN` into `1`, giving `beta = norm / 1` (clamped to the 4.0 ceiling),
//! where the port's `INT_MAX` gives `beta ~ 0` (clamped to the 0.25 floor) —
//! opposite ends of the clamp, hence an opposite-signed per-SB qindex delta.
//!
//! This is the same class as the `-ffp-contract` split recorded in
//! `docs/LIBAOM_UPSTREAM_NOTES.md` A3 and KB-ARM-FLOAT: libaom's own answer
//! differs by target, so a port cannot match both. **Nothing here changes the
//! port's behaviour** — these tests pin the semantics, bound the reachability,
//! and make the consequence visible, so the choice is made deliberately rather
//! than by whichever `as` someone typed first.
//!
//! History: found 2026-09-08 on the abandoned branch
//! `preserve/2026-07-25-agent-a788dbb3aaec3a1dc`, whose WIP commit proposed an
//! unconditional x86-64 model (`c_int`) without noticing the ARM half.

use aom_encode::allintra_vis::{self, DELTA_Q_RES_PERCEPTUAL, WeberStats, WeberVarMap};
use aom_sys_ref as c;

/// Values that straddle every boundary the chain can produce: in range, exactly
/// on the limits, past them, both infinities and NaN.
fn probe_values() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2147483646.5,
        2147483647.0,
        2147483647.9,
        2147483648.0,
        3e9,
        1e12,
        1e18,
        1e300,
        -2147483648.0,
        -2147483649.0,
        -3e9,
        -1e18,
        9.223_372_036_854_775e18,
        1e19,
        -1e19,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]
}

/// The primitive, against the REAL C cast as this host's compiler emits it.
///
/// Asserts the *contract*, both directions: inside the destination range the
/// two must agree exactly (that half is well-defined C and must never drift),
/// and outside it the divergence set must be exactly what this host's ISA
/// prescribes — so a host whose behaviour changed fails here rather than
/// silently re-defining what "matches libaom" means.
#[test]
fn c_double_to_int_cast_matches_rust_only_inside_the_destination_range() {
    let mut in_range_checked = 0usize;
    let mut out_of_range = Vec::new();

    for &d in &probe_values() {
        let c_i32 = c::ref_c_cast_double_to_int(d);
        let r_i32 = d as i32;
        let representable_i32 = d.is_finite() && d > -2147483649.0 && d < 2147483648.0;
        if representable_i32 {
            assert_eq!(
                c_i32, r_i32,
                "in-range (int) cast must agree exactly: d={d} c={c_i32} rust={r_i32}"
            );
            in_range_checked += 1;
        } else if c_i32 != r_i32 {
            out_of_range.push((d, c_i32, r_i32));
        }
    }
    assert!(
        in_range_checked >= 8,
        "the probe grid must actually exercise the well-defined range (got {in_range_checked})"
    );

    // The ISA contract. x86-64 returns INT_MIN for every out-of-range value and
    // for NaN; aarch64 saturates and maps NaN to 0 — which is Rust's own rule,
    // so there is nothing to diverge.
    if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
        assert!(
            !out_of_range.is_empty(),
            "x86-64 must diverge from Rust above INT_MAX / on NaN; it did not — \
             has the oracle been optimised into a Rust-equivalent form?"
        );
        for &(d, ci, _) in &out_of_range {
            assert_eq!(
                ci,
                i32::MIN,
                "x86-64 cvttsd2si must return the integer indefinite for d={d}"
            );
        }
    } else {
        assert!(
            out_of_range.is_empty(),
            "on a saturating-convert ISA Rust `as` IS the C behaviour; \
             unexpected divergences: {out_of_range:?}"
        );
    }

    eprintln!(
        "cast contract: arch={} in-range agreed={} out-of-range divergences={}",
        std::env::consts::ARCH,
        in_range_checked,
        out_of_range.len()
    );
}

/// Same contract for the `(int64_t)` cast used by both `exp()` normalizer
/// sites (allintra_vis.c:509, :678).
#[test]
fn c_double_to_int64_cast_matches_rust_only_inside_the_destination_range() {
    let mut diverged = 0usize;
    for &d in &probe_values() {
        let c_i64 = c::ref_c_cast_double_to_int64(d);
        let r_i64 = d as i64;
        let representable = d.is_finite() && d.abs() < 9.223_372_036_854_775e18;
        if representable {
            assert_eq!(
                c_i64, r_i64,
                "in-range (int64_t) cast must agree exactly: d={d}"
            );
        } else if c_i64 != r_i64 {
            diverged += 1;
            if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
                assert_eq!(c_i64, i64::MIN, "x86-64 must return INT64_MIN for d={d}");
            }
        }
    }
    if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
        assert!(diverged > 0, "x86-64 must diverge above INT64_MAX / on NaN");
    } else {
        assert_eq!(diverged, 0, "saturating ISA: Rust `as` is the C behaviour");
    }
}

// ---------------------------------------------------------------------------
// Reachability. The cast only matters if the chain can actually produce a value
// outside i32 — so bound it from the field ranges rather than guessing.
// ---------------------------------------------------------------------------

/// The largest `get_window_wiener_var` numerator a single 8x8 block can
/// contribute at a given bit depth, and the smallest denominator, using the
/// physical bounds of every `WeberStats` field:
///
/// * `src_pix_max` / `rec_pix_max` <= `2^bd - 1` (a pixel value),
/// * `src_variance` / `rec_variance` <= `(2^bd - 1)^2` (variance of 8-bit-to-
///   12-bit samples about their mean),
/// * `distortion` <= `64 * (2^bd - 1)^2` (SSE over the 8x8 block).
///
/// `base_den` can approach its floor of 1.0 independently of `distortion`:
/// it measures a *peak x sqrt(variance)* mismatch, which stays near zero
/// whenever reconstruction preserves peak and variance, even when the error
/// energy is large (spread, zero-mean error). So the extremes are jointly
/// attainable, not mutually exclusive.
fn window_var_upper_bound(bd: u32) -> f64 {
    let peak = f64::from((1u32 << bd) - 1);
    let var = peak * peak;
    let distortion = 64.0 * var;
    // One block, mb_count = 1: the ratio is largest when the window has a
    // single contributing block (the /mb_count divide only shrinks it).
    let base_num = 1.0 + distortion * var.sqrt() * peak;
    let base_reg = 1.0 + distortion.sqrt() * peak.sqrt() * 0.1;
    let base_den = 1.0; // its floor, attained when rec preserves peak+variance
    (base_num + base_reg) / (base_den + base_reg)
}

/// bd8 is structurally immune; bd10 and bd12 are not.
///
/// This is the load-bearing reachability claim, and it is an *arithmetic*
/// bound, not an observation of a real encode: it says the cast is unreachable
/// at bd8 no matter the content, and that bd10/bd12 have no such guarantee.
#[test]
fn window_wiener_var_can_leave_i32_only_above_bd8() {
    let max_i32 = f64::from(i32::MAX);
    let b8 = window_var_upper_bound(8);
    let b10 = window_var_upper_bound(10);
    let b12 = window_var_upper_bound(12);

    eprintln!(
        "window-var upper bound: bd8={b8:.3e} bd10={b10:.3e} bd12={b12:.3e}  (i32::MAX={max_i32:.3e})"
    );
    eprintln!(
        "  headroom: bd8={:.1}x under, bd10={:.1}x over, bd12={:.1}x over",
        max_i32 / b8,
        b10 / max_i32,
        b12 / max_i32
    );

    assert!(
        b8 < max_i32,
        "bd8 must be structurally immune: bound {b8:.3e} >= i32::MAX"
    );
    assert!(
        b10 > max_i32,
        "bd10 must be able to exceed i32::MAX: bound {b10:.3e}"
    );
    assert!(
        b12 > max_i32,
        "bd12 must be able to exceed i32::MAX: bound {b12:.3e}"
    );
}

/// What the divergence costs once it fires, measured through the PUBLIC chain
/// with the REAL `av1_get_deltaq_offset` oracle on both legs.
///
/// Constructs a map whose window value overflows, then compares the port's
/// answer with the answer an x86-64 libaom reaches for the same map. Neither
/// leg is a model of the other: the port leg is the port, and the C leg is
/// `sb_wiener_var = AOMMAX(1, INT_MIN) = 1` fed through the exported C offset
/// function.
#[test]
fn i32_overflow_flips_beta_to_the_opposite_end_of_the_clamp() {
    let mi = 16i32; // one BLOCK_64X64 SB
    let bd = 12u8;
    let base_qindex = 128i32;

    // bd12 extremes, chosen to drive base_den to its floor while keeping the
    // error energy maximal: identical peak and variance on both sides.
    let peak = ((1i64 << bd) - 1) as i16;
    let var = i64::from(peak) * i64::from(peak);
    let blk = WeberStats {
        src_variance: var,
        rec_variance: var,
        src_pix_max: peak,
        rec_pix_max: peak,
        distortion: 64 * var,
        satd: 1 << 20,
        max_scale: 1.0,
    };
    let map = WeberVarMap {
        stats: vec![blk; (mi * mi) as usize],
        mi_rows: mi,
        mi_cols: mi,
        norm_wiener_variance: 1_000_000,
    };

    let port_q =
        map.av1_get_sbq_perceptual_ai(base_qindex, bd, DELTA_Q_RES_PERCEPTUAL, mi, mi, 0, 0);

    // The x86-64 leg. `(int)` of the overflowing ratio is INT_MIN, `AOMMAX(1,
    // .)` makes it 1, so beta = norm / 1 -> the 4.0 ceiling.
    let c_beta = (map.norm_wiener_variance as f64).min(4.0).max(0.25);
    let c_off = c::ref_av1_get_deltaq_offset(bd, base_qindex, c_beta);
    let c_off = c_off
        .min(DELTA_Q_RES_PERCEPTUAL * 20 - 1)
        .max(-DELTA_Q_RES_PERCEPTUAL * 20 + 1);
    let c_q = (base_qindex + c_off).clamp(0, 255).max(1);

    eprintln!(
        "overflow consequence @bd{bd} base_q={base_qindex}: port qindex={port_q}, \
         x86-64-libaom qindex={c_q} (beta {c_beta})"
    );

    // Sanity: the constructed map really does overflow, else this proves nothing.
    let bound = window_var_upper_bound(u32::from(bd));
    assert!(
        bound > f64::from(i32::MAX),
        "the constructed map must be in the overflow regime"
    );

    assert_ne!(
        port_q, c_q,
        "the two ISA readings must disagree here — if they agree, either the \
         overflow stopped being reachable or the port changed cast semantics; \
         re-read this file's header before editing the assertion"
    );
}

/// The real-content search: how close does an actual encode get to the cliff?
///
/// The theoretical bound above is a *single-block* limit — `mb_count == 1` with
/// `base_den` simultaneously at its 1.0 floor. A full 64x64 window averages up
/// to 64 blocks and `base_den` accumulates with them, so the divide by
/// `mb_count` and the growing denominator both damp the ratio. This sweeps the
/// three axes that fight that damping — bit depth, content energy, and
/// `mb_count` (driven to its minimum by frames small enough that a window has
/// one in-frame block) — and reports the closest approach found.
///
/// **This is a search, and it reports a negative result.** Nothing here asserts
/// that an overflow is unreachable — absence of evidence over one sweep is not
/// proof — only that these cells do not reach it, and by how far.
#[test]
fn real_content_search_for_an_i32_overflow_reports_the_closest_approach() {
    use aom_dsp::quant::{Dequants, Quants};

    /// `BLOCK_64X64`'s ordinal in `BLOCK_SIZES_ALL` — the argument is a
    /// block-size enum, not a pixel count (SB128 would be 15).
    const BLOCK_64X64: usize = 12;
    const WEBER_MI_STEP: i32 = 2;

    /// Worst (largest) value reaching the `(int)` cast over every 64x64 window,
    /// reproducing `get_window_wiener_var`'s own accumulation
    /// (allintra_vis.c:187-210) rather than a per-block proxy.
    fn worst_window_value(map: &WeberVarMap) -> (f64, usize, usize) {
        let (mi_wide, mi_high) = (16i32, 16i32);
        let (mut worst, mut over, mut windows) = (0.0f64, 0usize, 0usize);
        let mut sb_row = 0i32;
        while sb_row < map.mi_rows {
            let mut sb_col = 0i32;
            while sb_col < map.mi_cols {
                let (mut num, mut den, mut reg) = (1.0f64, 1.0f64, 1.0f64);
                let mut mb_count = 0i32;
                let mut row = sb_row;
                while row < sb_row + mi_high {
                    let mut col = sb_col;
                    while col < sb_col + mi_wide {
                        if row < map.mi_rows && col < map.mi_cols {
                            let w = &map.stats[((row / WEBER_MI_STEP) * map.mi_cols
                                + col / WEBER_MI_STEP)
                                as usize];
                            num += (w.distortion as f64)
                                * (w.src_variance as f64).sqrt()
                                * f64::from(w.rec_pix_max);
                            den += (f64::from(w.rec_pix_max) * (w.src_variance as f64).sqrt()
                                - f64::from(w.src_pix_max) * (w.rec_variance as f64).sqrt())
                            .abs();
                            reg += (w.distortion as f64).sqrt()
                                * f64::from(w.src_pix_max).sqrt()
                                * 0.1;
                            mb_count += 1;
                        }
                        col += WEBER_MI_STEP;
                    }
                    row += WEBER_MI_STEP;
                }
                if mb_count > 0 {
                    let r = ((num + reg) / (den + reg)) / f64::from(mb_count);
                    windows += 1;
                    if r > worst {
                        worst = r;
                    }
                    if r > f64::from(i32::MAX) {
                        over += 1;
                    }
                }
                sb_col += mi_wide;
            }
            sb_row += mi_high;
        }
        (worst, over, windows)
    }

    // Content generators, chosen for what they do to the two competing terms:
    // `checker` maximises per-block variance and distortion; `impulse` puts all
    // the energy in one sample (large distortion, peak preserved -> small den);
    // `ramp` is a smooth control.
    let contents: [(&str, fn(usize, usize, u16) -> u16); 3] = [
        (
            "checker",
            |x, y, m| if (x / 4 + y / 4) % 2 == 0 { m } else { 0 },
        ),
        (
            "impulse",
            |x, y, m| if x % 8 == 0 && y % 8 == 0 { m } else { 0 },
        ),
        ("ramp", |x, _y, m| {
            ((x as u32 * u32::from(m) / 191) as u16).min(m)
        }),
    ];

    let mut closest = 0.0f64;
    let mut closest_desc = String::new();
    let mut any_overflow = false;
    let mut cells = 0usize;

    // Small frames drive `mb_count` toward 1 (the single-block limit the bound
    // is stated in); large ones are the realistic case. qindex spans the range,
    // since distortion grows with the quantizer.
    for &bd in &[10u8, 12] {
        let peak = ((1u32 << bd) - 1) as u16;
        for &(w, h) in &[(8usize, 8usize), (16, 16), (64, 64), (192, 192)] {
            for &qindex in &[8i32, 128, 255] {
                for &(cname, make) in &contents {
                    let stride = w;
                    let mut src = vec![0u16; stride * h];
                    for y in 0..h {
                        for x in 0..w {
                            src[y * stride + x] = make(x, y, peak);
                        }
                    }
                    let mut quants = Quants::zeroed();
                    let mut deq = Dequants::zeroed();
                    aom_dsp::quant::av1_build_quantizer(
                        bd,
                        0,
                        0,
                        0,
                        0,
                        0,
                        &mut quants,
                        &mut deq,
                        0,
                    );
                    let map = allintra_vis::av1_set_mb_wiener_variance(
                        &src,
                        0,
                        stride,
                        h as i32 / 4,
                        w as i32 / 4,
                        qindex,
                        bd,
                        &quants,
                        &deq,
                        BLOCK_64X64,
                        16,
                        false,
                    );
                    let (worst, over, _wins) = worst_window_value(&map);
                    cells += 1;
                    if over > 0 {
                        any_overflow = true;
                        eprintln!(
                            "  OVERFLOW bd{bd} {w}x{h} q{qindex} {cname}: {over} windows, worst={worst:.4e}"
                        );
                    }
                    if worst > closest {
                        closest = worst;
                        closest_desc = format!("bd{bd} {w}x{h} q{qindex} {cname}");
                    }
                }
            }
        }
    }

    let cliff = f64::from(i32::MAX);
    eprintln!(
        "searched {cells} cells; closest approach: {closest_desc} -> {closest:.4e} \
         = {:.3e} of i32::MAX ({:.0}x under the cliff)",
        closest / cliff,
        cliff / closest
    );

    assert!(closest > 0.0, "the sweep produced no statistics — vacuous");
    assert!(
        !any_overflow,
        "an encode-reachable i32 overflow was FOUND — this changes the finding \
         from theoretical to live; see this file's header and record it in the \
         coverage queue before relaxing this assertion"
    );
}
