//! **KB-32 — the two nonrd (`--cpu-used` 8/9) size bands (issue #7).**
//!
//! Reported shape: at `--cpu-used=8` real content diverged at every size from
//! 512² up with a delta growing roughly linearly in area; at `--cpu-used=9` it
//! was byte-exact through 896² (196 SBs) and diverged from 1024² (256 SBs) up.
//! `--cpu-used` 0..7 were byte-exact at every size measured, including 9.7 MP.
//!
//! **Both bands were ONE speed feature libaom resolves from the frame's
//! dimensions and this port did not model at all** —
//! `rt_sf.force_large_partition_blocks_intra` (speed_features.c:326-328,
//! `speed >= 8 && is_720p_or_larger`), whose only consumer anywhere in libaom
//! is `set_vbp_thresholds_key_frame` (var_based_part.c:535-560), the KEY
//! variance partitioner this port runs from speed 7 up. It has TWO arms, and
//! the two bands are the two arms:
//!
//! * `threshold_base <<= (var_part_split_threshold_shift - 7)` (:539-544).
//!   The shift is **8** at speed 8 and **7** at speed 9
//!   (speed_features.c:581 / :601 — *"intentionally lower than speed 8's"*),
//!   so the scaling is live at speed 8 and a no-op at speed 9. That is the
//!   **cpu8** band, and it is why that band has no threshold: every frame at
//!   least 720 px on its short side takes it;
//! * `shift_val = 1` instead of 2 inside the `num_pixels >= RESOLUTION_720P`
//!   arm (:552-554). `RESOLUTION_720P` is `1280 * 720` **pixels of AREA**
//!   (rd.h:65), which falls between 896² = 802,816 and 1024² = 1,048,576 —
//!   the exact reported cpu9 threshold. That arm is live at BOTH speeds.
//!
//! A second, independent root sits above it at 4k: at allintra `speed >= 9`
//! the framesize-INdependent cascade sets the coeff/mode cost-update level to
//! `INTERNAL_COST_UPD_SBROW` (speed_features.c:593-594) and the
//! framesize-dependent pass demotes it to `INTERNAL_COST_UPD_OFF` **only below
//! 4k** (:648-651). The port hardcoded OFF at speed 9, so every frame at least
//! 2160 px on its short side lost the per-SB-row cost refresh. `pack.rs`
//! carried that as a written HANDOFF; these cells are what made it reachable.
//!
//! **What is left after both fixes is NOT a size band.** Decode-both
//! localization on every surviving cell puts the partition trees in exact
//! agreement — 45,780 nodes at 2176², 3,496 at 512² — with the first
//! divergence at a leaf whose `y_mode` differs inside
//! `av1_nonrd_pick_intra_mode`'s four-mode `intra_mode_list`
//! (DC / V / H / SMOOTH), same `tx_size`, same `uv_mode`, no angle delta, no
//! filter-intra. See [`estimate_arm_residual_is_a_leaf_mode_near_tie`].
//!
//! Run the on-demand tier:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb32_nonrd_size_bands -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

/// Mirror-tile (the `kb31_mandatory_tiles` / `kb22_hd_arms` recipe).
fn mirror_tile(
    base: &EncodeCell,
    label: &str,
    w: usize,
    h: usize,
    cq: i32,
    speed: i32,
) -> EncodeCell {
    let mir = |i: usize, n: usize| {
        let m = i % (2 * n);
        if m < n { m } else { 2 * n - 1 - m }
    };
    let (bw, bh) = (base.w, base.h);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = base.y[mir(r, bh) * bw + mir(col, bw)];
        }
    }
    let (bcw, bch) = ((bw + base.ss_x) >> base.ss_x, (bh + base.ss_y) >> base.ss_y);
    let (cw, ch) = ((w + base.ss_x) >> base.ss_x, (h + base.ss_y) >> base.ss_y);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            u[r * cw + col] = base.u[mir(r, bch) * bcw + mir(col, bcw)];
            v[r * cw + col] = base.v[mir(r, bch) * bcw + mir(col, bcw)];
        }
    }
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono: base.mono,
        ss_x: base.ss_x,
        ss_y: base.ss_y,
        usage: base.usage,
        cq_level: cq,
        speed,
        bd: base.bd,
        y,
        u,
        v,
    }
}

/// One cell: real aomenc (`c_encode_defaults`) vs `EncodeCell::port_encode_with`.
/// Returns `(port_len - real_len, byte_identical)` and prints the row.
fn run(base: &EncodeCell, w: usize, h: usize, cq: i32, speed: i32) -> (i64, bool) {
    let cell = mirror_tile(base, &format!("kb32_{w}x{h}_s{speed}"), w, h, cq, speed);
    let c_tu = cell.c_encode_defaults();
    assert!(!c_tu.is_empty(), "{w}x{h} s{speed}: C encode failed");
    let ours = cell.port_encode_with(&c_tu, &ToggleKnobs::default());
    let real = EncodeCell::frame_obu_payload(&c_tu);
    let d = ours.len() as i64 - real.len() as i64;
    println!(
        "  {w:>5}x{h:<5} px={:>9} min={:>4} s{speed} cq{cq}: {:>9} vs {:>9} delta {:+} {}",
        w * h,
        w.min(h),
        ours.len(),
        real.len(),
        d,
        if ours == real { "MATCH" } else { "DIVERGE" }
    );
    (d, ours == real)
}

// ---------------------------------------------------------------------------
// The cpu9 band — `RESOLUTION_720P` is an AREA threshold
// ---------------------------------------------------------------------------

/// `(w, h, why, pre-fix delta)` — four cells chosen so the AREA predicate
/// (`num_pixels >= 1280 * 720`) and the SHORT-SIDE predicate
/// (`min(w, h) >= 720`, which is what arms the speed feature at all) are
/// separable by the RESULT PATTERN rather than by intuition: the first pair
/// holds the short side at 768 and straddles the area threshold, the second
/// holds it at 704 — below 720, so the arm must NOT fire — and straddles the
/// same area threshold.
///
/// A port that keyed the threshold arm off the SHORT SIDE would pass rows 1-2
/// and fail row 3; one that ignored the short side and keyed off area alone
/// would fail row 3 as well. Both are excluded by this grid.
///
/// MEASURED 2026-08-01 (aarch64-apple-darwin, `--profile test-fast`, cq30,
/// mirror-tiled `av1-1-b8-00-quantizer-00`, bd8 4:2:0).
const SPEED9_GRID: &[(usize, usize, &str, i64)] = &[
    (768, 1152, "884,736 px < RESOLUTION_720P; short side 768 >= 720", 0),
    (768, 1216, "933,888 px >= RESOLUTION_720P; short side 768 >= 720", 498),
    (704, 1280, "901,120 px < RESOLUTION_720P; short side 704 < 720", 0),
    (
        1024,
        1024,
        "1,048,576 px >= RESOLUTION_720P; the issue's own threshold cell",
        613,
    ),
];

/// **GATE — the cpu9 band, closed.** Every cell in [`SPEED9_GRID`] byte-matches
/// real aomenc at `--cpu-used=9`.
///
/// Bite proof (playbook §1, per root): dropping the `shift_val` arm from
/// `var_part::set_vbp_thresholds_key` — i.e. restoring the unconditional
/// `>> 2` — fails rows 2 and 4 (+498 and +613) while rows 1 and 3 stay green,
/// because those two are below the area threshold where the arm is
/// unreachable. Reverting the *other* root (the 4k cost-update arm) leaves
/// this whole gate green: none of its cells reach 2160 px.
#[test]
fn nonrd_speed9_area_threshold_byte_identical() {
    c::ref_init();
    let base = EncodeCell::real_content("kb32", "av1-1-b8-00-quantizer-00", None, 30, 9);
    println!("KB-32 cpu9 band (RESOLUTION_720P = 1280*720 = 921,600 px):");
    let (mut below, mut above, mut short_lt_720) = (0, 0, 0);
    for &(w, h, why, pre) in SPEED9_GRID {
        let (d, exact) = run(&base, w, h, 30, 9);
        assert!(
            exact,
            "{w}x{h} ({why}): delta {d:+} — the pre-fix delta here was {pre:+}"
        );
        if (w * h) as i64 >= 1280 * 720 {
            above += 1
        } else {
            below += 1
        }
        if w.min(h) < 720 {
            short_lt_720 += 1
        }
    }
    // Non-vacuity (playbook §2): the grid must straddle BOTH predicates, or it
    // proves nothing about which one governs.
    assert!(
        below >= 1 && above >= 2 && short_lt_720 >= 1,
        "the grid must straddle the area threshold AND contain a cell below \
         720 on the short side (below={below} above={above} short<720={short_lt_720})"
    );
}

/// **The 4k cost-update arm** (`is_4k_or_larger`, root #2), with its negative
/// control one SB-column below the boundary.
///
/// 2112² has `min(w, h) == 2112 < 2160`, so speed 9 keeps
/// `INTERNAL_COST_UPD_OFF` and the frame must byte-match — that is what proves
/// the fix did not disturb sub-4k frames. 2176² is the cheapest frame at or
/// above 2160 on both sides; before the fix it was **-2,599 B**, and modelling
/// `INTERNAL_COST_UPD_SBROW` takes it to the residual pinned below.
///
/// Bite proof: reverting the cost-upd arm alone (hardcoding OFF at speed 9)
/// leaves [`nonrd_speed9_area_threshold_byte_identical`] green and moves ONLY
/// the 2176² row, -184 -> -2,599.
#[test]
#[ignore = "two ~4.5 MP encode pairs; on-demand tier"]
fn nonrd_speed9_4k_cost_upd_sbrow() {
    c::ref_init();
    let base = EncodeCell::real_content("kb32", "av1-1-b8-00-quantizer-00", None, 30, 9);
    println!("KB-32 cpu9 x is_4k_or_larger (min(w,h) >= 2160):");
    let (_, exact_2112) = run(&base, 2112, 2112, 30, 9);
    assert!(
        exact_2112,
        "2112x2112 is BELOW is_4k_or_larger — INTERNAL_COST_UPD_OFF must still \
         be exact there, or the fix changed a sub-4k frame"
    );
    let (d, exact) = run(&base, 2176, 2176, 30, 9);
    // Pinned OPEN in both directions (playbook §5): the residual is the
    // estimate-arm class, not the cost-upd arm.
    assert!(
        !exact,
        "2176x2176 cpu9 is now BYTE-IDENTICAL — the KB-12 estimate-arm residual \
         closed. Promote this to a hard byte gate and re-pin."
    );
    assert!(
        d.abs() < 1_000,
        "2176x2176 cpu9 delta {d:+} is far outside the estimate-arm residual \
         band (-2,599 before the INTERNAL_COST_UPD_SBROW arm, -184 after) — \
         the cost-update level regressed"
    );
}

// ---------------------------------------------------------------------------
// The cpu8 band — the `shift_steps` arm, and what remains
// ---------------------------------------------------------------------------

/// **The cpu8 band's mechanism, and the shape of what survives.** At speed 8
/// the arm ALSO doubles `threshold_base` (`shift_steps = 8 - 7 = 1`), which is
/// why cpu8 diverged at every size from 512² up while cpu9 had a clean
/// threshold.
///
/// MEASURED 2026-08-01, before -> after the fix, same box and same cells:
///
/// | cell | SBs | pre-fix | post-fix |
/// |---|---|---|---|
/// | 512² | 64 | +61 | +61 (arm unreachable: 512 < 720) |
/// | 768² | 144 | +152 | -50 |
/// | 896² | 196 | +253 | -23 |
/// | 1024² | 256 | +581 | -168 |
/// | 2048² | 1,024 | +2,576 | **+21** |
///
/// Per superblock, pre-fix -> post-fix: 512² 0.95 -> **0.95** (unchanged —
/// 512 < 720, the arm is unreachable there), 768² 1.06 -> 0.35, 896²
/// 1.29 -> 0.12, 1024² 2.27 -> 0.66, 2048² 2.52 -> **0.02**. The pre-fix
/// column RISES monotonically with area; the post-fix column does not, and on
/// the four cells the arm actually reaches it falls by 3-100x. That is what is
/// pinned — a shape property, not byte counts — so the gate does not fire on
/// every unrelated encoder landing but does fire the moment a size-scaling
/// root returns.
#[test]
#[ignore = "5 cells up to 4.2 MP; on-demand tier"]
fn nonrd_speed8_size_ladder_residual_is_bounded() {
    c::ref_init();
    let base = EncodeCell::real_content("kb32", "av1-1-b8-00-quantizer-00", None, 30, 8);
    println!("KB-32 cpu8 ladder:");
    // (w, h, superblocks at SB64, the PRE-fix delta)
    const LADDER: &[(usize, usize, i64, i64)] = &[
        (512, 512, 64, 61),
        (768, 768, 144, 152),
        (896, 896, 196, 253),
        (1024, 1024, 256, 581),
        (2048, 2048, 1024, 2576),
    ];
    // Worst per-SB residual among the cells the arm actually REACHES
    // (min(w, h) >= 720). 512² is excluded on purpose: the arm is unreachable
    // there, so its 0.95 B/SB is untouched by this fix and including it would
    // make the bound vacuous.
    let mut worst_armed_per_sb = 0.0f64;
    for &(w, h, sbs, pre) in LADDER {
        let (d, _) = run(&base, w, h, 30, 8);
        let per_sb = (d.abs() as f64) / (sbs as f64);
        if w.min(h) >= 720 {
            worst_armed_per_sb = worst_armed_per_sb.max(per_sb);
        }
        assert!(
            d.abs() <= pre.abs().max(200),
            "{w}x{h} cpu8: delta {d:+} exceeds its pre-fix value {pre:+} — the \
             force_large_partition_blocks_intra arm regressed"
        );
    }
    println!("  worst armed-cell residual: {worst_armed_per_sb:.3} B per superblock");
    assert!(
        worst_armed_per_sb < 1.0,
        "the worst >=720p cpu8 residual is {worst_armed_per_sb:.3} B/SB — those \
         four cells ran at 1.06 / 1.29 / 2.27 / 2.52 B/SB and RISING before the \
         force_large_partition_blocks_intra arm was modelled, so a size-scaling \
         root is back"
    );
}

// ---------------------------------------------------------------------------
// What is left: the KB-12 estimate-arm leaf-mode class
// ---------------------------------------------------------------------------

const MI_SIZE_WIDE_B: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const PARTITION_NAMES: [&str; 10] = [
    "NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B", "VERT_A", "VERT_B", "HORZ_4", "VERT_4",
];
/// `av1_nonrd_pick_intra_mode`'s `intra_mode_list` — RTC_INTRA_MODES
/// (nonrd_pickmode.c): DC_PRED, V_PRED, H_PRED, SMOOTH_PRED.
const RTC_INTRA_MODES: [i32; 4] = [0, 1, 2, 9];

#[allow(clippy::too_many_arguments)]
fn replay_tree(
    tree: &[i8],
    cursor: &mut usize,
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    mi_rows: i32,
    mi_cols: i32,
    out: &mut Vec<(i32, i32, usize, i8)>,
) {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return;
    }
    let p = tree[*cursor];
    out.push((mi_row, mi_col, bsize, p));
    *cursor += 1;
    if p == 3 {
        let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
        let sub = aom_dsp::entropy::partition::get_partition_subsize(bsize, 3) as usize;
        for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
            replay_tree(
                tree,
                cursor,
                mi_row + dr,
                mi_col + dc,
                sub,
                mi_rows,
                mi_cols,
                out,
            );
        }
    }
}

/// Decode-both localization (playbook §10): encode with real aomenc and with
/// the port, splice the port payload back into the reference stream, decode
/// BOTH with the (bit-exact vs C) decoder, replay both partition trees and
/// report the FIRST divergent decision.
///
/// Returns `Some((y_mode_real, y_mode_port))` when the trees agree and the
/// first divergence is a leaf `y_mode`; `None` when the cell is byte-exact.
/// PANICS when the trees disagree — that is the signature of a partition-side
/// root (KB-32's two roots were exactly that), and it must not come back
/// unnoticed.
fn localize_leaf_mode(w: usize, h: usize, cq: i32, speed: i32) -> Option<(i32, i32)> {
    c::ref_init();
    let base = EncodeCell::real_content("kb32l", "av1-1-b8-00-quantizer-00", None, cq, speed);
    let cell = mirror_tile(&base, &format!("kb32l_{w}x{h}_s{speed}"), w, h, cq, speed);
    let c_tu = cell.c_encode_defaults();
    let port = cell.port_encode_with(&c_tu, &ToggleKnobs::default());
    let real = EncodeCell::frame_obu_payload(&c_tu);
    println!(
        "== {w}x{h} cq{cq} s{speed}: port {} B vs real {} B (delta {:+})",
        port.len(),
        real.len(),
        port.len() as i64 - real.len() as i64
    );
    if port == real {
        println!("   byte-identical");
        return None;
    }
    let ours_tu = aom_bench::rd_close::splice_frame_obu(&c_tu, &port);
    let (t_real, _c1, _h1) =
        aom_decode::frame::decode_frame_obus_prefilter(&c_tu).expect("decode real");
    let (t_ours, _c2, _h2) =
        aom_decode::frame::decode_frame_obus_prefilter(&ours_tu).expect("decode port");
    let mi_rows = ((h as i32 + 7) & !7) >> 2;
    let mi_cols = ((w as i32 + 7) & !7) >> 2;
    let sb_mi = 16i32; // allintra always selects BLOCK_64X64 (av1_select_sb_size)
    let seq = |tree: &[i8]| {
        let mut cur = 0usize;
        let mut out = Vec::new();
        for r in 0..(mi_rows + sb_mi - 1) / sb_mi {
            for cc in 0..(mi_cols + sb_mi - 1) / sb_mi {
                replay_tree(
                    tree,
                    &mut cur,
                    r * sb_mi,
                    cc * sb_mi,
                    12,
                    mi_rows,
                    mi_cols,
                    &mut out,
                );
            }
        }
        out
    };
    let (rs, os) = (seq(&t_real.tree), seq(&t_ours.tree));
    assert_eq!(
        rs.len(),
        os.len(),
        "{w}x{h} s{speed}: the two frames have DIFFERENT partition-tree shapes \
         — that is a partition-side root (both KB-32 roots were), not the \
         estimate-arm residual"
    );
    for (r, o) in rs.iter().zip(os.iter()) {
        assert_eq!(
            (r.0, r.1, r.2),
            (o.0, o.1, o.2),
            "{w}x{h} s{speed}: the partition walks desynced"
        );
        assert_eq!(
            r.3, o.3,
            "{w}x{h} s{speed}: PARTITION DIVERGENCE at mi({},{}) bsize={} \
             (real PARTITION_{}, port PARTITION_{}) — a partition-side root is \
             back; the variance partitioner's thresholds are the first suspect",
            r.0,
            r.1,
            r.2,
            PARTITION_NAMES[r.3 as usize],
            PARTITION_NAMES[o.3 as usize]
        );
    }
    println!("   partition trees AGREE ({} nodes); scanning leaves", rs.len());
    for rb in &t_real.blocks {
        if let Some(ob) = t_ours
            .blocks
            .iter()
            .find(|b| b.mi_row == rb.mi_row && b.mi_col == rb.mi_col)
        {
            if ob.bsize != rb.bsize
                || ob.info.y_mode != rb.info.y_mode
                || ob.info.angle_delta_y != rb.info.angle_delta_y
                || ob.info.use_filter_intra != rb.info.use_filter_intra
                || ob.tx_size != rb.tx_size
                || ob.info.uv_mode != rb.info.uv_mode
                || ob.txbs != rb.txbs
                || ob.txbs_uv != rb.txbs_uv
            {
                println!(
                    ">>> FIRST LEAF MISMATCH at mi({},{}): real bsize={} y_mode={} \
                     adly={} fi={} tx={} uv={} | port bsize={} y_mode={} adly={} \
                     fi={} tx={} uv={}",
                    rb.mi_row,
                    rb.mi_col,
                    rb.bsize,
                    rb.info.y_mode,
                    rb.info.angle_delta_y,
                    rb.info.use_filter_intra,
                    rb.tx_size,
                    rb.info.uv_mode,
                    ob.bsize,
                    ob.info.y_mode,
                    ob.info.angle_delta_y,
                    ob.info.use_filter_intra,
                    ob.tx_size,
                    ob.info.uv_mode,
                );
                assert_eq!(
                    (
                        rb.bsize,
                        rb.info.angle_delta_y,
                        rb.info.use_filter_intra,
                        rb.tx_size,
                        rb.info.uv_mode
                    ),
                    (
                        ob.bsize,
                        ob.info.angle_delta_y,
                        ob.info.use_filter_intra,
                        ob.tx_size,
                        ob.info.uv_mode
                    ),
                    "{w}x{h} s{speed}: the first leaf mismatch moved something \
                     OTHER than y_mode — the residual is no longer purely the \
                     estimate arm's mode choice"
                );
                return Some((rb.info.y_mode, ob.info.y_mode));
            }
        }
    }
    panic!("{w}x{h} s{speed}: payloads differ but every shared leaf agrees");
}

/// **What the residual IS, asserted rather than asserted-about.**
///
/// Every surviving KB-32 cell — at BOTH speeds, at three quality points, and
/// at 4k where root #2 lived — has (a) partition trees in EXACT agreement and
/// (b) a first divergence that is a leaf `y_mode`, both sides inside
/// `av1_nonrd_pick_intra_mode`'s four-mode `intra_mode_list`
/// {DC, V, H, SMOOTH}, with `tx_size`, `uv_mode`, angle delta and filter-intra
/// all equal. That is KB-12's pinned speed-8 estimate-arm class — the
/// attribution KB-32 recorded as a hypothesis and explicitly flagged as NOT
/// established. It is established now, and it is BROADER than KB-12's entry
/// says: KB-12 localized V-vs-H, and the pairs below include DC-vs-SMOOTH,
/// V-vs-SMOOTH, DC-vs-H and DC-vs-V.
///
/// It also fires at `--cpu-used=9` (KB-12 recorded speed 9 as 64/64 canon, at
/// 64² and 128²), so speed 9's three estimate-loop prunes MASK the class on
/// small frames rather than removing it.
///
/// MEASURED 2026-08-01: 512² cq30 s8 -> trees agree (3,496 nodes), first
/// mismatch mi(4,108) BLOCK_8X8 real SMOOTH vs port DC; 2176² cq30 s9 -> trees
/// agree (45,780 nodes), first mismatch mi(108,174) BLOCK_8X8 real DC vs port V.
#[test]
#[ignore = "includes a 4.7 MP pair; on-demand tier"]
fn estimate_arm_residual_is_a_leaf_mode_near_tie() {
    let cells: &[(usize, usize, i32, i32)] = &[
        (512, 512, 30, 8),
        (512, 512, 48, 8),
        (512, 512, 63, 8),
        (256, 256, 30, 8),
        (512, 512, 48, 9),
        (2176, 2176, 30, 9),
    ];
    let mut seen = 0usize;
    for &(w, h, cq, speed) in cells {
        if let Some((r, o)) = localize_leaf_mode(w, h, cq, speed) {
            assert!(
                RTC_INTRA_MODES.contains(&r) && RTC_INTRA_MODES.contains(&o),
                "{w}x{h} cq{cq} s{speed}: the diverging leaf modes are real={r} \
                 port={o}, and at least one is OUTSIDE \
                 av1_nonrd_pick_intra_mode's intra_mode_list {RTC_INTRA_MODES:?} \
                 — this is no longer the estimate-arm class"
            );
            seen += 1;
        }
    }
    // Non-vacuity: if every cell became byte-exact the class closed, and this
    // test must be re-pinned rather than silently passing on zero evidence.
    assert!(
        seen >= 5,
        "only {seen} of {} cells still diverge — the estimate-arm class is \
         closing. Re-pin this test (and KB-12).",
        cells.len()
    );
}

