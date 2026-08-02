//! **KB-28 — the framesize predicates key on the TRUE CROP, not the mi grid.**
//!
//! `av1_get_MBs` (alloccommon.c:30-33) aligns the mi grid UP to 8 px, so
//! `mi_cols * 4` runs up to **7 px larger** than `cm->width`. Every framesize
//! predicate libaom evaluates reads `cm->width` / `cm->height`:
//!
//! | consumer | C site | predicate |
//! |---|---|---|
//! | `set_vbp_thresholds_key_frame` (speed >= 7 VAR_BASED partitioner) | var_based_part.c:667 -> :547 | `cm->width * cm->height < RESOLUTION_720P` |
//! | `force_large_partition_blocks_intra` (speed >= 8) | speed_features.c:326-328 | `AOMMIN(cm->width, cm->height) >= 720` |
//! | `use_square_partition_only_threshold` (speed >= 1) | speed_features.c:175-316 | `>= 480` / `>= 720` |
//! | `intra_mode_cnn_partition` res-tier thresholds (speed >= 1) | partition_strategy.c:311-312 | `>= 480` / `>= 720` |
//! | `av1_ml_prune_4_partition`'s `res_idx` | partition_strategy.c:1349-1352 | `>= 480` / `>= 720` |
//! | `ext_partition_eval_thresh` (speed 5) | speed_features.c:510-511 | `>= 480` |
//!
//! The port re-derived all six from `env.mi_cols * 4` / `env.mi_rows * 4`.
//! For the VBP one that produced a **refusal** — `pack.rs` asserted whenever
//! the mi-aligned area and an "up to 3 px smaller" crop could land on opposite
//! sides of `1280 * 720`, and an exactly 1280x720 frame is inside that window
//! (`mi_px == 921600` is not `< 921600` while `1277 * 717 == 915609` is), so the
//! most ordinary HD frame refused to encode at `--cpu-used` 7, 8 AND 9. For the
//! other five it produced a **silently wrong** framesize tier on any crop whose
//! mi-rounding crosses 480 or 720 (`partition_pick.rs`'s `res_idx` comment named
//! that gap and left it open).
//!
//! Two things the pre-fix guard got wrong beyond refusing a legal size, both
//! pinned below by [`refusal_window_is_characterised`]:
//! * the crop can be up to **7** px smaller per axis, not 3 — the guard used
//!   `mi - 3`;
//! * because of that, the guard has a **hole**: crops with
//!   `(mi_w - 3) * (mi_h - 3) >= 921600 > crop_w * crop_h` take the wrong
//!   threshold arm with no refusal at all. 8,776 such crops exist with both mi
//!   extents in 8..=4096 (`1274x722` is one, and it is gated below).
//!
//! **Is there an analogous window at another `set_vbp_thresholds` bucket?
//! No — checked in source, not assumed.** `set_vbp_thresholds` reads
//! `num_pixels` against `RESOLUTION_288P` / `480P` / `720P` / `1080P` /
//! `1440P`, but on the KEY path it delegates to `set_vbp_thresholds_key_frame`
//! and **returns** (var_based_part.c:660-664) before
//! `tune_base_thresh_content`, `tune_thresh_based_on_resolution` and
//! `tune_thresh_based_on_qindex` — which is where every other bucket lives.
//! `set_vbp_thresholds_key_frame`'s only bucket is `RESOLUTION_720P` (:547).
//! The file's other `cm->width * cm->height` reads are `chroma_check` (:1004,
//! returns immediately on a key frame) and two `!is_key_frame`-gated arms
//! (:1344/:1358 in `do_int_pro_motion_estimation`, :1821's
//! `is_360p_or_smaller`). So the AREA axis has exactly one boundary here.
//! The MIN-DIM axis has three (480 / 720 / 2160, speed_features.c:169-172) and
//! all three are fixed by the same change — the frame-level SF resolver
//! already read the true crop (`aom-bench/src/lib.rs`'s
//! `apply_allintra_framesize_dependent(w, h, speed)`); it was only the
//! re-derivations *inside* the walk that were wrong.
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb28_crop_dims -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

/// `ALIGN_POWER_OF_TWO(px, 3)` — the mi grid's pixel extent (alloccommon.c:30).
const fn mi_aligned(px: i32) -> i32 {
    (px + 7) & !7
}

const RESOLUTION_720P: i64 = 1280 * 720;

/// Mirror-tile (same recipe as `s4cov_hd_speed_axis::mirror_tile`).
fn mirror_tile(base: &EncodeCell, label: &str, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
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

/// What the mi-aligned extent and the true crop each say about a frame. Used
/// both for the reach assertions and for labelling the printed rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Verdicts {
    /// `cm->width * cm->height >= RESOLUTION_720P` from the true crop.
    crop_area_720p: bool,
    /// The same predicate evaluated on the mi-aligned extent (what the port
    /// did before this fix).
    mi_area_720p: bool,
    /// `AOMMIN(cm->width, cm->height) >= 480` / `>= 720`, crop vs mi.
    crop_min_480: bool,
    mi_min_480: bool,
    crop_min_720: bool,
    mi_min_720: bool,
    /// Did the pre-fix `pack.rs` VBP guard fire on this shape? (It compared
    /// `mi_px < R720` with `(mi_w - 3) * (mi_h - 3) < R720`.)
    old_guard_fires: bool,
}

fn verdicts(w: i32, h: i32) -> Verdicts {
    let (mw, mh) = (mi_aligned(w), mi_aligned(h));
    let mi_px = i64::from(mw) * i64::from(mh);
    let old_min_crop_px = i64::from(mw - 3) * i64::from(mh - 3);
    Verdicts {
        crop_area_720p: i64::from(w) * i64::from(h) >= RESOLUTION_720P,
        mi_area_720p: mi_px >= RESOLUTION_720P,
        crop_min_480: w.min(h) >= 480,
        mi_min_480: mw.min(mh) >= 480,
        crop_min_720: w.min(h) >= 720,
        mi_min_720: mw.min(mh) >= 720,
        old_guard_fires: (mi_px < RESOLUTION_720P) != (old_min_crop_px < RESOLUTION_720P),
    }
}

/// **The refusal window, characterised exactly — and the hole in it.**
///
/// Pure arithmetic, no encodes: this is the map that says which sizes the
/// heavy gates below have to cover, and it is also the non-vacuity proof for
/// them (playbook §2). Every cell the byte gates encode is asserted here to
/// sit where the docs claim it sits.
///
/// The guard fired iff `mi_w * mi_h >= 921600` and
/// `(mi_w - 3) * (mi_h - 3) < 921600`, i.e. iff the mi area lies in
/// `[921600, 921591 + 3 * (mi_w + mi_h))` — a band whose width grows with the
/// perimeter, **369 distinct mi-aligned shapes** with both extents in
/// 8..=4096. The window is a range, not a point; 1280x720 is simply its most
/// ordinary member.
#[test]
fn refusal_window_is_characterised() {
    // 1. The window's extent over every mi-aligned shape up to 4096x4096.
    let mut fires = 0usize;
    let mut fires_examples: Vec<(i32, i32)> = Vec::new();
    let mut min_mi_w = i32::MAX;
    let mut max_mi_w = 0;
    for mw in (8..=4096).step_by(8) {
        for mh in (8..=4096).step_by(8) {
            let mi_px = i64::from(mw) * i64::from(mh);
            if mi_px >= RESOLUTION_720P
                && i64::from(mw - 3) * i64::from(mh - 3) < RESOLUTION_720P
            {
                fires += 1;
                min_mi_w = min_mi_w.min(mw);
                max_mi_w = max_mi_w.max(mw);
                if fires_examples.len() < 12 {
                    fires_examples.push((mw, mh));
                }
            }
        }
    }
    println!(
        "  refusal window: {fires} mi-aligned shapes in 8..=4096 (mi_w {min_mi_w}..{max_mi_w}); \
         first: {fires_examples:?}"
    );
    assert_eq!(
        fires, 369,
        "the refusal window's extent moved — it is every (mi_w, mi_h) with \
         921600 <= mi_w*mi_h < 921591 + 3*(mi_w+mi_h)"
    );
    assert!(
        verdicts(1280, 720).old_guard_fires,
        "1280x720 must be inside the window (it is the reported KB-28 shape)"
    );

    // 2. The HOLE. The crop can be 7 px smaller per axis, not 3, so there are
    //    crops that take the WRONG arm with no refusal at all. That is the
    //    silent-corruption case the guard was written to prevent, and it is
    //    strictly worse than the refusal it did produce.
    let mut caught = 0usize;
    let mut missed: Vec<(i32, i32, i32, i32)> = Vec::new();
    let mut missed_n = 0usize;
    for mw in (8..=4096).step_by(8) {
        for mh in (8..=4096).step_by(8) {
            let mi_px = i64::from(mw) * i64::from(mh);
            if mi_px < RESOLUTION_720P {
                continue; // mi area >= crop area always; no disagreement possible
            }
            let guard = i64::from(mw - 3) * i64::from(mh - 3) < RESOLUTION_720P;
            for w in (mw - 7)..=mw {
                for h in (mh - 7)..=mh {
                    if i64::from(w) * i64::from(h) < RESOLUTION_720P {
                        if guard {
                            caught += 1;
                        } else {
                            missed_n += 1;
                            if missed.len() < 8 && w > 900 && h > 600 {
                                missed.push((w, h, mw, mh));
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "  crops taking the WRONG VBP arm pre-fix: {caught} refused loudly, \
         {missed_n} SILENT; silent examples (crop -> mi): {missed:?}"
    );
    assert_eq!(caught, 19071, "the loudly-refused set moved");
    assert_eq!(
        missed_n, 8776,
        "the pre-fix guard's hole moved. It used `mi - 3` where the mi grid \
         aligns to 8 px (`mi - 7`), so these crops took the >=720p threshold \
         arm silently."
    );

    // 3. Reach assertions for every cell the byte gates below encode
    //    (playbook §8): each row must actually exercise the axis its comment
    //    claims, and the mi-aligned answer must DIFFER from the crop answer
    //    wherever the row is billed as a divergence.
    for &(w, h, note) in CELLS {
        let v = verdicts(w, h);
        let (mw, mh) = (mi_aligned(w), mi_aligned(h));
        println!(
            "  {w}x{h} -> mi {mw}x{mh} | area crop {} mi {} | min480 crop {} mi {} | \
             min720 crop {} mi {} | old guard {} | {note}",
            v.crop_area_720p as u8,
            v.mi_area_720p as u8,
            v.crop_min_480 as u8,
            v.mi_min_480 as u8,
            v.crop_min_720 as u8,
            v.mi_min_720 as u8,
            if v.old_guard_fires { "FIRES" } else { "-" },
        );
    }
    let find = |w: i32, h: i32| verdicts(w, h);
    // 1280x720: mi == crop, so the arm was never actually wrong — the guard
    // refused a frame it did not need to. Post-fix it must simply encode.
    let hd = find(1280, 720);
    assert!(hd.old_guard_fires && hd.crop_area_720p == hd.mi_area_720p);
    // 1272x724: guard fires AND the AREA arm was wrong.
    let v = find(1272, 724);
    assert!(v.old_guard_fires, "1272x724 must be inside the refusal window");
    assert!(
        !v.crop_area_720p && v.mi_area_720p,
        "1272x724 must straddle RESOLUTION_720P (crop below, mi at/above)"
    );
    // 1274x722 / 954x962: the HOLE — wrong arm, no refusal.
    for &(w, h) in &[(1274, 722), (954, 962)] {
        let v = find(w, h);
        assert!(
            !v.old_guard_fires,
            "{w}x{h} must sit in the guard's HOLE (no refusal) — that is what \
             makes it the silent-corruption witness"
        );
        assert!(
            !v.crop_area_720p && v.mi_area_720p,
            "{w}x{h} must straddle RESOLUTION_720P"
        );
    }
    // 1288x716 ISOLATES the MIN-DIM arm (`force_large_partition_blocks_intra`,
    // speed >= 8) from the AREA arm: the crop and mi extents agree that the
    // area is >= RESOLUTION_720P, and disagree only about `AOMMIN(w, h) >= 720`.
    // The pre-fix guard still refused it, because the guard's `mi - 3` window
    // is about the area alone.
    let v = find(1288, 716);
    assert!(v.old_guard_fires, "1288x716 must be inside the refusal window");
    assert_eq!(
        v.crop_area_720p, v.mi_area_720p,
        "1288x716 must NOT straddle the area threshold — it is the min-dim isolator"
    );
    assert!(
        !v.crop_min_720 && v.mi_min_720,
        "1288x716 must straddle AOMMIN(w,h) >= 720"
    );
    // Controls: no disagreement on any axis, so this fix is a literal no-op on
    // them (`frame_width == mi_cols * 4` makes every changed expression
    // evaluate to what it evaluated to before). Two are partial-SB and two are
    // SB-exact, which is what separates "crop-dependent" from "partial
    // superblock x nonrd" in the speed-8/9 rows.
    for &(w, h) in &[(1280, 712), (1280, 728), (1280, 704), (1216, 768)] {
        let v = find(w, h);
        assert!(!v.old_guard_fires);
        assert_eq!(v.crop_area_720p, v.mi_area_720p);
        assert_eq!(v.crop_min_720, v.mi_min_720);
        assert_eq!(mi_aligned(w), w, "{w}x{h} must be an exact mi extent");
        assert_eq!(mi_aligned(h), h, "{w}x{h} must be an exact mi extent");
    }
    assert!(
        (1280 % 64 == 0 && 704 % 64 == 0) && (1216 % 64 == 0 && 768 % 64 == 0),
        "the SB-exact controls must have no partial superblock"
    );
    assert!(
        712 % 64 != 0 && 728 % 64 != 0,
        "the partial-SB controls must have one"
    );
    // RD-band rows: min-dim only, no VBP involvement.
    let v = find(474, 480);
    assert!(!v.crop_min_480 && v.mi_min_480, "474x480 must straddle 480");
    let v = find(714, 720);
    assert!(
        v.crop_min_480 && !v.crop_min_720 && v.mi_min_720,
        "714x720 must straddle 720 while staying >= 480 on both readings"
    );
}

/// The cells the byte gates encode. `(w, h, note)`.
const CELLS: &[(i32, i32, &str)] = &[
    (1280, 720, "the reported KB-28 shape: guard fired, arm was correct"),
    (1272, 724, "guard fired, arm WAS wrong (crop area below 720p)"),
    (1288, 716, "guard fired; isolates the min-dim arm (area agrees)"),
    (1274, 722, "guard's HOLE: wrong arm, no refusal (silent)"),
    (954, 962, "guard's HOLE at a different aspect ratio"),
    (1280, 712, "control: below the area threshold on both readings"),
    (1280, 728, "control: above the area threshold on both readings"),
    (1280, 704, "SB-EXACT control below the area threshold"),
    (1216, 768, "SB-EXACT control above the area threshold"),
    (474, 480, "RD band: straddles AOMMIN(w,h) >= 480"),
    (714, 720, "RD band: straddles AOMMIN(w,h) >= 720"),
];

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Verdict {
    Ok,
    Diverge,
    Panic,
}

fn measure(cell: &EncodeCell) -> (Verdict, i64, String) {
    let c_tu = cell.c_encode();
    let real = EncodeCell::frame_obu_payload(&c_tu);
    let msg = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = std::sync::Arc::clone(&msg);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let s = info.to_string();
        *sink.lock().unwrap() = s
            .lines()
            .find(|l| l.contains("assertion") || l.contains("not implemented"))
            .unwrap_or_else(|| s.lines().last().unwrap_or(""))
            .to_string();
    }));
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cell.port_encode_with(&c_tu, &ToggleKnobs::default())
    }));
    std::panic::set_hook(hook);
    match got {
        Ok(p) if p == real => (Verdict::Ok, 0, String::new()),
        Ok(p) => (
            Verdict::Diverge,
            p.len() as i64 - real.len() as i64,
            String::new(),
        ),
        Err(_) => (
            Verdict::Panic,
            0,
            format!("PANIC: {}", msg.lock().unwrap()),
        ),
    }
}

/// **The VBP band: `--cpu-used` 7, 8 and 9 across the refusal window.**
///
/// Speed 7 is where `set_vbp_thresholds_key_frame` first runs (the VAR_BASED
/// partitioner, speed_features.c:571); 8 and 9 add
/// `force_large_partition_blocks_intra`'s two arms (KB-32), one of which IS
/// the `num_pixels >= RESOLUTION_720P` arm — so the area predicate is
/// load-bearing at all three. KB-28's title named speed 7 only; the refusal
/// fired at 7, 8 and 9, and this grid runs all three.
///
/// **MEASURED 2026-08-02** (aarch64-apple-darwin, `--profile test-fast`,
/// `av1-1-b8-00-quantizer-00` mirror-tiled, bd8 4:2:0):
///
/// | | before | after |
/// |---|---|---|
/// | speed 7, all 9 cells | 4 PANIC + 2 DIVERGE + 3 MATCH | **9/9 MATCH** |
/// | speeds 8/9, SB-EXACT controls | MATCH | MATCH |
/// | speeds 8/9, PARTIAL-SB cells | (masked by the refusal) | DIVERGE — pinned |
///
/// **The speed-8/9 divergence was NOT this bug** — it was KB-12's nonrd
/// estimate-arm class, and this grid was the evidence that named it: a
/// partial-superblock explanation was *refuted* here (`1280x704` and
/// `1216x768` are exact multiples of 64 and diverged at 8/9 too, -1/-6 and
/// +1/-8), and the five shapes with `cm->width == mi_cols * 4` on both axes —
/// where KB-28's fix is a literal no-op — carried byte-identical deltas with
/// the fix reverted.
///
/// **CLOSED 2026-08-02**: `hadamard_lp_8x8` omitted the trailing transpose at
/// `aom_dsp/avg.c:232-236`, so the nonrd estimate arm's coefficients were the
/// transpose of libaom's and its `eob` — the only order-sensitive output —
/// drifted. Every row below is byte-exact now, so "1280x720 is byte-identical
/// at every speed" is finally true. See KB-12 and `nonrd_block_yrd_lp_diff.rs`.
///
/// **The last two rows CLOSED 2026-08-02 as well (KB-34).** They were the two
/// speed-9 cells reaching KB-32's non-square-leaf refusal ("HANDOFF: nonrd
/// estimate arm at non-square leaf") — which KB-32 had measured as reachable
/// only on its 108 MP cell, so finding it at 0.9 MP here was the first
/// contradiction of that claim. The estimate arm now runs C's real per-txb
/// walk and both rows are byte-exact, so this grid is **20/20** and
/// `NONRD_ESTIMATE_ARM_OPEN` is empty.
#[test]
#[ignore = "large-frame encode pairs at cpu 7/8/9; nightly / on-demand tier"]
fn vbp_band_crop_dims_byte_match() {
    c::ref_init();
    let base = EncodeCell::real_content("kb28base", "av1-1-b8-00-quantizer-00", None, 24, 0);
    // What is still not byte-exact on this grid, pinned EXACTLY and
    // self-promoting in both directions. It is now EMPTY: KB-12's estimate-arm
    // class held 18 of these 20 rows until 2026-08-02 (closed by the
    // aom_hadamard_lp_8x8 transpose) and KB-34's non-square-leaf refusal held
    // the other 2 (closed the same day by the per-txb walk). Nothing at speed 7
    // may appear either — speed 7 is KB-28's own band and is asserted clean.
    const NONRD_ESTIMATE_ARM_OPEN: &[(i32, i32, i32, i32, Verdict)] = &[];
    let mut observed: Vec<(i32, i32, i32, i32, Verdict)> = Vec::new();
    let mut worst_b_per_sb = 0.0f64;
    let mut worst_cell = String::new();
    let mut rows = 0usize;
    for &(w, h, note) in CELLS.iter().filter(|(_, _, n)| !n.starts_with("RD band")) {
        for speed in 7..=9 {
            // cq24 everywhere; the headline shape also at cq40, the other side
            // of the KB-22 qindex arm, because that is how KB-28 was pinned.
            let cqs: &[i32] = if (w, h) == (1280, 720) { &[24, 40] } else { &[24] };
            for &cq in cqs {
                let cell = mirror_tile(
                    &base,
                    &format!("kb28_{w}x{h}_cq{cq}_s{speed}"),
                    w as usize,
                    h as usize,
                    cq,
                    speed,
                );
                let t0 = std::time::Instant::now();
                let (v, delta, note2) = measure(&cell);
                rows += 1;
                println!(
                    "  {w}x{h} cq{cq} cpu{speed}: {v:?} delta {delta:+} [{} ms] {} {note}{}",
                    t0.elapsed().as_millis(),
                    if w % 64 == 0 && h % 64 == 0 {
                        "SB-exact "
                    } else {
                        "PARTIAL-SB"
                    },
                    if note2.is_empty() {
                        String::new()
                    } else {
                        format!("  [{note2}]")
                    }
                );
                if v != Verdict::Ok {
                    observed.push((w, h, cq, speed, v));
                }
                if speed == 8 && v == Verdict::Diverge {
                    let sbs = f64::from(((w + 63) / 64) * ((h + 63) / 64));
                    let bps = (delta.abs() as f64) / sbs;
                    if bps > worst_b_per_sb {
                        worst_b_per_sb = bps;
                        worst_cell = format!("{w}x{h} cq{cq} {delta:+} over {sbs} SB");
                    }
                }
            }
        }
    }
    println!("  KB-28 VBP band: {}/{rows} byte-exact", rows - observed.len());

    // --- KB-28's own result: speed 7 is CLEAN on every shape in the window. ---
    let s7_bad: Vec<String> = observed
        .iter()
        .filter(|(_, _, _, s, _)| *s == 7)
        .map(|(w, h, cq, s, v)| format!("{w}x{h} cq{cq} cpu{s} {v:?}"))
        .collect();
    assert!(
        s7_bad.is_empty(),
        "a speed-7 cell stopped matching real aomenc — that is a KB-28 \
         regression. `Panic` means the crop-ambiguity refusal is back \
         (`pack.rs` must take `num_pixels` from \
         `SbEncodeEnv::frame_num_pixels`, not from the mi-aligned extent). \
         `Diverge` on 1274x722 / 954x962 means the mi-aligned area is being \
         read again, in the band where the old guard did not even fire: \
         {s7_bad:?}"
    );

    // --- The speed-8 band, closed 2026-08-02. It used to be pinned on a SHAPE
    //     (worst residual < 1.0 B/SB, sign-random) because KB-12's estimate-arm
    //     class sat on 18 of these rows; that class is fixed, so the band is a
    //     hard byte gate. The B/SB figure is still reported because it is the
    //     falsifier if a row comes back: sub-byte-per-SB and sign-random is the
    //     estimate arm, a residual that GROWS with area is a systematic
    //     search-configuration difference (playbook §10 — never infer the
    //     mechanism from the SIZE of the delta).
    println!("  worst speed-8 residual: {worst_b_per_sb:.3} B/SB ({worst_cell})");
    let s8_bad: Vec<String> = observed
        .iter()
        .filter(|(_, _, _, s, v)| *s == 8 && *v == Verdict::Diverge)
        .map(|(w, h, cq, s, v)| format!("{w}x{h} cq{cq} cpu{s} {v:?}"))
        .collect();
    assert!(
        s8_bad.is_empty(),
        "the speed-8 VBP band diverged again ({s8_bad:?}; worst \
         {worst_b_per_sb:.3} B/SB at {worst_cell}). Under 1 B/SB and \
         sign-random is KB-12's estimate arm — run \
         `nonrd_block_yrd_lp_diff.rs` first. Growing with area is a \
         size-scaling speed feature, KB-32's shape."
    );

    let pinned: Vec<(i32, i32, i32, i32, Verdict)> = NONRD_ESTIMATE_ARM_OPEN.to_vec();
    assert_eq!(
        observed, pinned,
        "the nonrd map moved, and it is pinned EMPTY — every row is a \
         regression. `Panic` at speed 9 is KB-34's non-square-leaf arm coming \
         back (2 rows lived there until 2026-08-02); a sub-byte-per-SB \
         sign-random `Diverge` is KB-12's estimate arm (18 rows, closed the \
         same day by the aom_hadamard_lp_8x8 transpose); a delta that grows \
         with area is a size-scaling speed feature, KB-32's shape."
    );
}

/// **The RD band: the min-dim tiers at `--cpu-used` 1..6.**
///
/// `use_square_partition_only_threshold` (speeds 1-2 move BOTH tiers), the
/// intra-CNN res-tier thresholds (speeds 1-6), `ext_partition_eval_thresh`
/// (speed 5) and `av1_ml_prune_4_partition`'s `res_idx` all read
/// `AOMMIN(cm->width, cm->height)`. 474x480 and 714x720 are the two crops
/// whose mi-rounding crosses 480 and 720 respectively; both were resolved one
/// tier too high before this fix.
///
/// **MEASURED 2026-08-02**: before the fix both sizes DIVERGE at every speed
/// 1..6; after, all 12 cells are byte-identical.
#[test]
#[ignore = "12 encode pairs up to 714x720; nightly / on-demand tier"]
fn rd_band_min_dim_tiers_byte_match() {
    c::ref_init();
    let base = EncodeCell::real_content("kb28rdbase", "av1-1-b8-00-quantizer-00", None, 24, 0);
    let mut bad: Vec<String> = Vec::new();
    let mut rows = 0usize;
    for &(w, h, note) in CELLS.iter().filter(|(_, _, n)| n.starts_with("RD band")) {
        for speed in 1..=6 {
            let cell = mirror_tile(
                &base,
                &format!("kb28rd_{w}x{h}_s{speed}"),
                w as usize,
                h as usize,
                24,
                speed,
            );
            let t0 = std::time::Instant::now();
            let (v, delta, note2) = measure(&cell);
            rows += 1;
            println!(
                "  {w}x{h} cq24 cpu{speed}: {v:?} delta {delta:+} [{} ms] {note}{}",
                t0.elapsed().as_millis(),
                if note2.is_empty() {
                    String::new()
                } else {
                    format!("  [{note2}]")
                }
            );
            if v != Verdict::Ok {
                bad.push(format!("{w}x{h} cpu{speed} {v:?} {delta:+} {note2}"));
            }
        }
    }
    println!("  KB-28 RD band: {}/{rows} byte-exact", rows - bad.len());
    assert!(
        bad.is_empty(),
        "a min-dim-tier cell stopped matching real aomenc — a framesize \
         predicate in `partition_pick.rs` is reading `mi_cols/mi_rows * 4` \
         again instead of `SbEncodeEnv::frame_{{width,height}}`: {bad:?}"
    );
}

/// **KB-23's 250x250 row, kept here as a control with the reason stated.**
///
/// KB-28's ledger entry said 250x250 "names the same gap". Half of that is
/// right and half is not, and the difference is worth pinning:
/// * the CNN *res-tier threshold* (partition_strategy.c:311-312) IS the same
///   root, and 250x250 cannot reach it — `min(250, 250) = 250` and
///   `min(256, 256) = 256` are both below 480, so crop and mi agree;
/// * the CNN *window* (`extract_intra_cnn_window`) is what 250x250 does
///   exercise, and it is **inert** either way: C reads the border-extended
///   source without clamping at all (partition_strategy.c:205-220), so every
///   read between the crop and the mi extent returns the replicated edge
///   pixel, which is exactly what a clamp to either bound produces.
///
/// So 250x250 was byte-identical before this change and stays byte-identical
/// after it — and this test asserts the second half, which is the part a
/// regression could break. The rows that DO carry KB-28's root are 474x480 and
/// 714x720 in [`rd_band_min_dim_tiers_byte_match`].
#[test]
#[ignore = "2 encode pairs; nightly / on-demand tier"]
fn cnn_window_clamp_is_replication_inert() {
    c::ref_init();
    let base = EncodeCell::real_content("kb28cnn", "av1-1-b8-00-quantizer-00", None, 24, 0);
    // Reach: 250x250 must be a crop whose mi rounding is strictly larger AND
    // whose framesize tier is unchanged by that rounding — otherwise this is
    // not the window-only cell it claims to be.
    let v = verdicts(250, 250);
    assert_eq!(mi_aligned(250), 256);
    assert_eq!(v.crop_min_480, v.mi_min_480);
    assert_eq!(v.crop_min_720, v.mi_min_720);
    for speed in [1, 2] {
        let cell = mirror_tile(&base, &format!("kb28cnn_250_s{speed}"), 250, 250, 24, speed);
        let (v, delta, note) = measure(&cell);
        println!("  250x250 cq24 cpu{speed}: {v:?} delta {delta:+} {note}");
        assert_eq!(
            v,
            Verdict::Ok,
            "250x250 cpu{speed} diverged. The intra-CNN window clamp moved from \
             the mi-aligned extent to the true crop in KB-28; those agree only \
             while the source plane is edge-replicated from the crop \
             (`extend_plane`, aom-bench/src/lib.rs). delta {delta:+} {note}"
        );
    }
}
