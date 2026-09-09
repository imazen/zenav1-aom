//! **S4 coverage extension — KB-28's crop axis at the formats its own ledger
//! entry named as unmeasured.**
//!
//! KB-28 (`SbEncodeEnv::frame_{width,height}`: six consumers re-derived a
//! framesize predicate from the **mi-aligned** extent instead of
//! `cm->width`/`cm->height`, so a crop whose mi-aligned area lands on the other
//! side of a 480 / 720 / 2160 boundary took the wrong speed-feature arm — 19,071
//! crops refused loudly and **8,776 took the wrong arm silently**) was found and
//! closed on one grid, and its entry says so:
//!
//! > *"Still unmeasured: the crop axis at bd10/12, 4:2:2/4:4:4, monochrome,
//! > SB128 and multi-tile; crops straddling the `is_4k_or_larger` (2160)
//! > predicate; and the 480/720 straddle at `--cpu-used 0` (where
//! > `use_square_partition_only_threshold`'s base tier still moves)."*
//!
//! **Four of those seven are swept here** (`--cpu-used 0`, monochrome,
//! 4:4:4/4:2:2, SB128, plus bd10 where it is interpretable). The two left are
//! recorded at the bottom of this file with what each would cost.
//!
//! **Why the formats are not a rerun of KB-28's grid.** The root is shared —
//! one crop-vs-mi read — but the six consumers are not. Three of them
//! (`use_square_partition_only_threshold`, the intra-CNN res tier,
//! `av1_ml_prune_4_partition`'s `res_idx`) key off `AOMMIN(w, h)` and are
//! chroma-blind; the VBP `num_pixels` arm keys off `w * h`. What subsampling
//! changes is not the predicate but everything it then gates: the partition
//! shapes each arm admits interact with the chroma-reference rules
//! (`max_block_units`, the frame-edge entropy-stamp tail-zero at `blockd.c:29`)
//! whose footprint depends on `ss_x`/`ss_y`. **Monochrome removes that
//! machinery entirely and is the negative control for it.** SB128 is a
//! different question again: the crop-vs-mi gap is up to 7 px either way
//! (`av1_get_MBs` aligns to 8), so the two extents can straddle a boundary
//! identically at both superblock sizes — but `av1_select_sb_size`
//! (encoder_utils.c:958) picks SB64 at `min(w,h) <= 480` regardless, so the
//! **474x480 row is an SB128 request that C DOWNGRADES**, and the gate asserts
//! that it does (otherwise the row is an SB64 test wearing an SB128 label).
//!
//! **MEASURED 2026-08-03** (`benchmarks/s4cov_crop_format_2026-08-03.tsv`).
//! Every arm below is byte-exact; the per-arm counts are in each test's doc.
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test s4cov_crop_format_axis -- --ignored --nocapture
//! ```


use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;
use aom_sys_ref::cx_ctrl::{AOM_SUPERBLOCK_SIZE_128X128, AV1E_SET_SUPERBLOCK_SIZE};

/// cq24 -> `base_qindex` 96 — the quality point `kb28_crop_dims`'s own RD-band
/// rows were measured at, so every row here has a directly comparable 4:2:0
/// SB64 twin there.
const CQ: i32 = 24;

/// Mirror-tile a decoded cell up to `w x h`. Same recipe as
/// `kb22_hd_arms::mirror_tile` (and `kb19_min_partition_4k`, and the size axis
/// of the config-permutation gate), extended to carry monochrome cells through
/// unchanged — mirroring keeps the seam continuous so the enlarged frame stays
/// photographic instead of acquiring a synthetic edge grid every tile period.
fn mirror_tile(base: &EncodeCell, label: &str, w: usize, h: usize, speed: i32) -> EncodeCell {
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
    let (mut u, mut v) = (Vec::new(), Vec::new());
    if !base.mono {
        let (bcw, bch) = (
            (bw + base.ss_x) >> base.ss_x,
            (bh + base.ss_y) >> base.ss_y,
        );
        let (cw, ch) = ((w + base.ss_x) >> base.ss_x, (h + base.ss_y) >> base.ss_y);
        u = vec![0u16; cw * ch];
        v = vec![0u16; cw * ch];
        for r in 0..ch {
            for col in 0..cw {
                u[r * cw + col] = base.u[mir(r, bch) * bcw + mir(col, bcw)];
                v[r * cw + col] = base.v[mir(r, bch) * bcw + mir(col, bcw)];
            }
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
        cq_level: CQ,
        speed,
        bd: base.bd,
        y,
        u,
        v,
    }
}

/// Drop the chroma planes (4:0:0).
fn to_mono(base: &EncodeCell, label: &str) -> EncodeCell {
    EncodeCell {
        label: label.to_string(),
        mono: true,
        ss_x: 1,
        ss_y: 1,
        u: Vec::new(),
        v: Vec::new(),
        ..base.clone()
    }
}

/// Re-render a 4:2:0 cell at `(ss_x, ss_y)` by nearest-neighbour chroma
/// upsampling — see `s4cov_qm_axis.rs::to_ss` for why nearest-neighbour.
fn to_ss(base: &EncodeCell, label: &str, ss_x: usize, ss_y: usize) -> EncodeCell {
    assert!(!base.mono);
    assert_eq!((base.ss_x, base.ss_y), (1, 1));
    let bcw = (base.w + 1) >> 1;
    let (cw, ch) = ((base.w + ss_x) >> ss_x, (base.h + ss_y) >> ss_y);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        let sr = (r << ss_y) >> 1;
        for col in 0..cw {
            let sc = (col << ss_x) >> 1;
            u[r * cw + col] = base.u[sr * bcw + sc];
            v[r * cw + col] = base.v[sr * bcw + sc];
        }
    }
    EncodeCell {
        label: label.to_string(),
        ss_x,
        ss_y,
        u,
        v,
        ..base.clone()
    }
}

fn base_b8() -> EncodeCell {
    c::ref_init();
    EncodeCell::real_content("s4cropbase", "av1-1-b8-00-quantizer-00", None, CQ, 0)
}

fn base_b10() -> EncodeCell {
    c::ref_init();
    EncodeCell::real_content("s4cropbase10", "av1-1-b10-00-quantizer-00", None, CQ, 0)
}

/// KB-28's two RD-band crops, verbatim from `kb28_crop_dims::CELLS`:
/// * 474x480 — mi-aligned to 480x480, so `AOMMIN(crop) = 474 < 480 <=
///   AOMMIN(mi) = 480`: the crop and the mi reading DISAGREE across
///   `is_480p_or_larger` (speed_features.c:169);
/// * 714x720 — mi-aligned to 720x720, same disagreement across
///   `is_720p_or_larger` (:170).
///
/// These are the only two sizes in the file, so every arm below varies exactly
/// one thing against KB-28's own measured rows.
const CROPS: [(usize, usize); 2] = [(474, 480), (714, 720)];

/// `av1_get_MBs` (alloccommon.c:30-33) aligns to **8** px, not 4.
const fn mi_aligned(px: usize) -> usize {
    (px + 7) & !7
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Verdict {
    Ok,
    Diverge,
    Panic,
}

struct Row {
    tag: String,
    size: (usize, usize),
    speed: i32,
    verdict: Verdict,
    delta: i64,
    note: String,
}

/// Encode one cell against real aomenc with the given extra C controls and
/// classify. A PANIC is a distinct outcome, not a test abort (the KB-20 /
/// KB-28 lesson: a refusal sitting between two green rows is exactly what a
/// coverage extension is looking for, and aborting on the first hides the
/// shape of the rest).
fn measure(cell: &EncodeCell, ctrls: &[(i32, i32)], tag: &str) -> (Row, Vec<u8>) {
    let c_tu = cell.c_encode_ctrls(ctrls);
    assert!(!c_tu.is_empty(), "{}: C encode failed", cell.label);
    let real = EncodeCell::frame_obu_payload(&c_tu);
    let msg = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = std::sync::Arc::clone(&msg);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        *sink.lock().unwrap() = info.to_string().lines().last().unwrap_or("").to_string();
    }));
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cell.port_encode_with(&c_tu, &ToggleKnobs::default())
    }));
    std::panic::set_hook(hook);
    let (verdict, delta, note) = match got {
        Ok(p) if p == real => (Verdict::Ok, 0, String::new()),
        Ok(p) => (
            Verdict::Diverge,
            p.len() as i64 - real.len() as i64,
            String::new(),
        ),
        Err(_) => (Verdict::Panic, 0, msg.lock().unwrap().clone()),
    };
    let row = Row {
        tag: tag.to_string(),
        size: (cell.w, cell.h),
        speed: cell.speed,
        verdict,
        delta,
        note,
    };
    (row, c_tu)
}

fn report(rows: &[Row]) -> Vec<String> {
    let mut bad = Vec::new();
    for r in rows {
        println!(
            "  {} {}x{} cpu{}: {:?} delta {:+}{}",
            r.tag,
            r.size.0,
            r.size.1,
            r.speed,
            r.verdict,
            r.delta,
            if r.note.is_empty() {
                String::new()
            } else {
                format!("  [{}]", r.note)
            }
        );
        if r.verdict != Verdict::Ok {
            bad.push(format!(
                "{} {}x{} cpu{} {:?} ({:+}) {}",
                r.tag, r.size.0, r.size.1, r.speed, r.verdict, r.delta, r.note
            ));
        }
    }
    println!(
        "  {}/{} byte-exact",
        rows.len() - bad.len(),
        rows.len()
    );
    bad
}

/// The message every arm here fails with. Named once because the diagnosis is
/// the same wherever it fires.
const KB28_HINT: &str = "a crop-axis cell stopped byte-matching real aomenc. The root KB-28 \
     fixed is a framesize predicate re-derived from `env.mi_cols/mi_rows * 4` instead of \
     `SbEncodeEnv::frame_{width,height}` — there are six such consumers (pack.rs's VBP \
     `num_pixels`, and partition_pick.rs's `use_square_partition_only_threshold`, intra-CNN \
     res tier, `ext_partition_eval_thresh` and `av1_ml_prune_4_partition` `res_idx`). Compare \
     the failing row against its 4:2:0 SB64 twin in \
     `kb28_crop_dims::rd_band_min_dim_tiers_byte_match`: if THAT is green, the crop read is \
     right and the divergence is the format's own";

// ---------------------------------------------------------------------------
// 1. `--cpu-used 0` — the straddle at the speed KB-28's grid skipped.
// ---------------------------------------------------------------------------

/// **The `--cpu-used 0` arm, named explicitly by KB-28's residual.**
///
/// `use_square_partition_only_threshold` is `BLOCK_128X128` at speed 0 for
/// `is_720p_or_larger`, `BLOCK_64X64` for `is_480p_or_larger` and
/// `BLOCK_32X32` below (speed_features.c:175-316), so BOTH crops sit on a tier
/// boundary that moves at speed 0 — which is why KB-28's `1..=6` band could
/// not answer for it.
///
/// **MEASURED 2026-08-03: 4:2:0 is 2/2 byte-exact. MONOCHROME DIVERGES** at
/// both crops (-150 at 474x480, +160 at 714x720) — and that is **not this
/// axis's finding**. It is `s4cov_partial_sb_axis::MONO_S0_OPEN`, the
/// pre-existing `(monochrome, base_qindex 96, speed 0)` near-tie whose own
/// localizer reduces it to a **single 64x64 superblock** at cq24, i.e. to a
/// frame with no crop straddle at all. The attribution is not argued here, it
/// is measured: each mono crop is run beside an **SB-exact mono control at its
/// own mi-aligned extent** (480x480, 720x720), where the crop and mi readings
/// AGREE and KB-28's root is a literal no-op. Those controls diverge too, so
/// the divergence does not depend on the straddle.
///
/// Pinned self-promoting in both directions rather than moved to a quality
/// point where it does not fire (which is the banned form).
#[test]
#[ignore = "6 encode pairs at cpu 0, up to 720x720 (~4 min); nightly / on-demand tier"]
fn crop_straddle_speed0_byte_matches() {
    let b8 = base_b8();
    let mono = to_mono(&b8, "mono");
    // (tag, w, h) -> byte delta, for the rows that are NOT byte-exact.
    // Measured 2026-08-03; every entry is monochrome and every entry has a
    // diverging SB-exact control, i.e. none of them is the crop straddle.
    const MONO_S0_OPEN: &[(&str, usize, usize, i64)] =
        &[("mono", 474, 480, -150), ("mono", 714, 720, 160)];
    let mut rows = Vec::new();
    let mut ctl_rows = Vec::new();
    for &(w, h) in &CROPS {
        // Reach assertion (playbook §2): the row is only about KB-28's root if
        // the crop and the mi-aligned extent DISAGREE about the tier.
        let (tier_crop, tier_mi) = (w.min(h), mi_aligned(w).min(mi_aligned(h)));
        let boundary = if h == 480 { 480 } else { 720 };
        assert!(
            tier_crop < boundary && tier_mi >= boundary,
            "{w}x{h}: crop min-dim {tier_crop} and mi min-dim {tier_mi} must straddle \
             {boundary}, else this row does not exercise KB-28's root"
        );
        for (tag, src) in [("420 ", &b8), ("mono", &mono)] {
            let cell = mirror_tile(src, &format!("{tag}_{w}x{h}_s0"), w, h, 0);
            let (row, _) = measure(&cell, &[], tag);
            rows.push(row);
        }
        // The SB-exact control at the crop's own mi-aligned extent, monochrome:
        // crop and mi READ THE SAME TIER there, so KB-28's fix is a literal
        // no-op on it. Whatever this row does is not the straddle's.
        let (cw, ch) = (mi_aligned(w), mi_aligned(h));
        let ctl = mirror_tile(&mono, &format!("monoctl_{cw}x{ch}_s0"), cw, ch, 0);
        let (row, _) = measure(&ctl, &[], "monoctl");
        ctl_rows.push(row);
    }
    let ctl_bad = report(&ctl_rows);
    let bad = report(&rows);
    let observed: Vec<(&str, usize, usize, i64)> = rows
        .iter()
        .filter(|r| r.verdict != Verdict::Ok)
        .map(|r| (r.tag.trim(), r.size.0, r.size.1, r.delta))
        .collect();
    // Every 4:2:0 row must be exact — that is this arm's positive result and
    // the teeth on `use_square_partition_only_threshold`'s speed-0 tiers.
    let ss_bad: Vec<&String> = bad.iter().filter(|b| b.starts_with("420")).collect();
    assert!(ss_bad.is_empty(), "{KB28_HINT}: {ss_bad:?}");
    // The monochrome rows are pinned, and pinned WITH their attribution: the
    // SB-exact controls must diverge too, or the divergence really is the
    // straddle and belongs to KB-28 after all.
    assert!(
        !ctl_bad.is_empty(),
        "the SB-exact monochrome speed-0 controls (480x480, 720x720) are now byte-exact \
         while the CROPS diverge — that makes the divergence depend on the straddle, i.e. \
         it IS KB-28's root and not `s4cov_partial_sb_axis::MONO_S0_OPEN`. Re-localize \
         before touching anything else"
    );
    assert_eq!(
        observed,
        MONO_S0_OPEN.to_vec(),
        "the speed-0 crop map moved. A row that started MATCHING means the monochrome \
         cq24 speed-0 near-tie closed (re-pin here AND in \
         `s4cov_partial_sb_axis::partial_sb_speed_axis_chroma_formats_byte_match`, which \
         pins the same class at 132x132 / 192x192 / 196x196 / 256x256). A 4:2:0 row \
         appearing here is KB-28's root returning"
    );
}

// ---------------------------------------------------------------------------
// 2. Chroma format x the straddle, across KB-28's own speed band.
// ---------------------------------------------------------------------------

/// **The subsampling arm.** KB-28's grid is 4:2:0 only. 4:4:4 and 4:2:2 change
/// the chroma footprint every partition decision then has to code; monochrome
/// deletes it. Speeds 1..6 are KB-28's own band, so each row has a directly
/// comparable 4:2:0 twin in `rd_band_min_dim_tiers_byte_match`.
///
/// **MEASURED 2026-08-03: 36/36 byte-exact** (3 formats x 2 crops x cpu 1..6).
#[test]
#[ignore = "36 encode pairs up to 714x720 (~4 min); nightly / on-demand tier"]
fn crop_straddle_chroma_formats_byte_match() {
    let b8 = base_b8();
    let formats: Vec<(&str, EncodeCell)> = vec![
        ("mono", to_mono(&b8, "mono")),
        ("444 ", to_ss(&b8, "444", 0, 0)),
        ("422 ", to_ss(&b8, "422", 1, 0)),
    ];
    let mut rows = Vec::new();
    for (tag, src) in &formats {
        for &(w, h) in &CROPS {
            for speed in 1..=6 {
                let cell = mirror_tile(src, &format!("{tag}_{w}x{h}_s{speed}"), w, h, speed);
                assert_eq!(
                    (cell.mono, cell.ss_x, cell.ss_y),
                    (src.mono, src.ss_x, src.ss_y),
                    "the mirror-tile must preserve the chroma format"
                );
                let (row, _) = measure(&cell, &[], tag);
                rows.push(row);
            }
        }
    }
    let bad = report(&rows);
    assert!(bad.is_empty(), "{KB28_HINT}: {bad:?}");
}

// ---------------------------------------------------------------------------
// 3. SB128 x the straddle.
// ---------------------------------------------------------------------------

/// **The SB128 arm.** `--sb-size=128` at both crops, speeds 1..6.
///
/// **A predicate written here was FALSE and the first run refuted it, which is
/// worth recording** (playbook §9's "try to break it before writing it down").
/// The first version asserted that 474x480 would come back byte-identical to
/// its SB64 encode, reasoning from `av1_select_sb_size`'s
/// `AOMMIN(w, h) <= 480 -> BLOCK_64X64` rule — the rule KB-34's ledger entry
/// quotes. That rule is in the **`AOM_SUPERBLOCK_SIZE_DYNAMIC` branch**: an
/// explicit `--sb-size=128` returns `BLOCK_128X128` from the top of the
/// function (encoder_utils.c:961-963) before any size test runs. So BOTH crops
/// really are SB128 here, and the row that was going to be a stated
/// non-result is a genuine second data point.
///
/// The anti-vacuity check stays, in the form the measurement supports: the C
/// stream must CHANGE versus the same cell at `--sb-size=64` on every row, or
/// the row is an SB64 test wearing an SB128 label (playbook §8).
///
/// **MEASURED 2026-08-03: 12/12 byte-exact**, C stream changed on 12/12.
#[test]
#[ignore = "12 encode pairs up to 714x720 (~1 min); nightly / on-demand tier"]
fn crop_straddle_sb128_byte_matches() {
    let b8 = base_b8();
    let sb128 = [(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_128X128)];
    let mut rows = Vec::new();
    let mut inert: Vec<String> = Vec::new();
    for &(w, h) in &CROPS {
        for speed in 1..=6 {
            let cell = mirror_tile(&b8, &format!("sb128_{w}x{h}_s{speed}"), w, h, speed);
            let (row, c_tu) = measure(&cell, &sb128, "sb128");
            if c_tu == cell.c_encode_ctrls(&[]) {
                inert.push(format!("{w}x{h} cpu{speed}"));
            }
            rows.push(row);
        }
    }
    assert!(
        inert.is_empty(),
        "`--sb-size=128` did not change the C stream vs `--sb-size=64` on these rows, so they \
         prove nothing about the 128-superblock geometry. An explicit superblock size is \
         honoured unconditionally (encoder_utils.c:961-963) — if a row went inert, the \
         control stopped being applied: {inert:?}"
    );
    let bad = report(&rows);
    assert!(bad.is_empty(), "{KB28_HINT}: {bad:?}");
}

// ---------------------------------------------------------------------------
// 4. High bit depth x the straddle — only where it is INTERPRETABLE.
// ---------------------------------------------------------------------------

/// **The bd10 arm, run only at the speeds where it can be read.**
///
/// bd10 diverges from real aomenc at `--cpu-used` 1..6 on SB-EXACT content
/// already — the pinned `b10_64` band of
/// `config_permutations.rs::speed_envelope_stock_map_is_pinned`, widened by
/// `s4cov_qm_axis.rs` (it reaches 4:4:4, 12-bit, monochrome and cq5, and is
/// LUMA-borne). A crop cell at those speeds cannot answer "does the crop-vs-mi
/// read hold at high bit depth?", because both explanations predict a
/// divergence. Speeds **0 and 7** are clean at bd10 on SB-exact content, so
/// they are where the question is well-posed — and each is asked here with its
/// own SB-EXACT control at the identical speed and bit depth, so the answer
/// does not rest on the other file's measurement.
///
/// The controls are 480x480 and 720x720: the mi-aligned extents of the two
/// crops, i.e. the frames the buggy read would have been reading. That makes
/// the pair the sharpest possible A/B — same tier under the mi reading,
/// different tier under the crop reading.
///
/// **MEASURED 2026-08-03: 8/8 byte-exact** (2 crops + 2 controls x speeds
/// {0, 7}).
#[test]
#[ignore = "8 bd10 encode pairs up to 720x720 (~2 min); nightly / on-demand tier"]
fn crop_straddle_high_bitdepth_byte_matches_where_interpretable() {
    let b10 = base_b10();
    assert_eq!(b10.bd, 10);
    let mut crop_rows = Vec::new();
    let mut ctl_rows = Vec::new();
    for speed in [0, 7] {
        for &(w, h) in &CROPS {
            let cell = mirror_tile(&b10, &format!("b10_{w}x{h}_s{speed}"), w, h, speed);
            assert_eq!(cell.bd, 10);
            let (row, _) = measure(&cell, &[], "b10crop");
            crop_rows.push(row);
            // The SB-exact control at the crop's own mi-aligned extent.
            let (cw, ch) = (mi_aligned(w), mi_aligned(h));
            let ctl = mirror_tile(&b10, &format!("b10ctl_{cw}x{ch}_s{speed}"), cw, ch, speed);
            let (row, _) = measure(&ctl, &[], "b10ctl ");
            ctl_rows.push(row);
        }
    }
    let ctl_bad = report(&ctl_rows);
    assert!(
        ctl_bad.is_empty(),
        "the bd10 SB-EXACT control diverged at speed 0 or 7. Those are the only speeds where \
         bd10 is byte-exact on SB-exact content, and they are what make the crop rows \
         interpretable — so this is a bd10 regression (or a spread of the pinned speed-1..6 \
         band), not a crop-axis result: {ctl_bad:?}"
    );
    let bad = report(&crop_rows);
    assert!(bad.is_empty(), "{KB28_HINT}: {bad:?}");
}

/// **The bd12 arm — the same four cells as the bd10 one, one bit depth up.**
///
/// `s4cov_qm_axis.rs` shows the `b10_64` band reaches 12-bit identically, so
/// the interpretable speeds are the same {0, 7} and the SB-exact controls play
/// the same role. The bd12 source is BIT-REPLICATED from the genuine 10-bit
/// one (`v << 2 | v >> 8`), not left-shifted: a plain shift leaves the low two
/// bits zero, which is the regime KB-4 calls out as the easy one — it never
/// produces a coefficient whose low bits matter.
///
/// **MEASURED 2026-08-04: 8/8 byte-exact** (53 s), and re-run byte-exact under
/// `AOM_FORCE_SCALAR=1`.
///
/// **BITE PROOF.** Making `SbEncodeEnv::frame_min_dim()` return the MI extent
/// (`((w+7)&!7).min((h+7)&!7)`) — KB-28's root — fails this test on
/// **`b12crop 474x480 cpu0` (+7 B) with all four SB-exact controls byte-exact**.
/// The bite is narrow (1 of the 4 crop rows; at 714x720 and at cpu 7 the
/// affected predicates do not change a decision on this content at bd12), and
/// it is stated that way rather than dressed up — what it establishes is that
/// the crop read is genuinely live on this arm's cells and that the controls
/// are insensitive to it, which is the asymmetry the arm is for.
#[test]
#[ignore = "8 bd12 encode pairs up to 720x720 (~2 min); nightly / on-demand tier"]
fn crop_straddle_bd12_byte_matches_where_interpretable() {
    let b10 = base_b10();
    assert_eq!(b10.bd, 10);
    // `to_bd`, inline: widen by bit replication.
    let widen = |v: &u16| -> u16 { (v << 2) | (v >> 8) };
    let b12 = EncodeCell {
        label: "s4cropbase12".to_string(),
        bd: 12,
        y: b10.y.iter().map(widen).collect(),
        u: b10.u.iter().map(widen).collect(),
        v: b10.v.iter().map(widen).collect(),
        ..b10.clone()
    };
    // Non-vacuity (playbook §2): a "bd12" cell whose samples all fit in 10 bits
    // is a bd10 cell wearing a bd12 label.
    assert!(
        b12.y.iter().any(|&s| s > 1023) && b12.y.iter().any(|&s| s & 3 != 0),
        "the bd12 cell must use the extra two bits"
    );
    let mut crop_rows = Vec::new();
    let mut ctl_rows = Vec::new();
    for speed in [0, 7] {
        for &(w, h) in &CROPS {
            let cell = mirror_tile(&b12, &format!("b12_{w}x{h}_s{speed}"), w, h, speed);
            assert_eq!(cell.bd, 12, "the mirror-tile must preserve the bit depth");
            let (row, _) = measure(&cell, &[], "b12crop");
            crop_rows.push(row);
            // The SB-exact control at the crop's own mi-aligned extent — same
            // tier under the mi reading, different tier under the crop reading.
            let (cw, ch) = (mi_aligned(w), mi_aligned(h));
            let ctl = mirror_tile(&b12, &format!("b12ctl_{cw}x{ch}_s{speed}"), cw, ch, speed);
            let (row, _) = measure(&ctl, &[], "b12ctl ");
            ctl_rows.push(row);
        }
    }
    let ctl_bad = report(&ctl_rows);
    assert!(
        ctl_bad.is_empty(),
        "the bd12 SB-EXACT control diverged at speed 0 or 7. Those are the only speeds where \
         high bit depth is byte-exact on SB-exact content, and they are what make the crop \
         rows interpretable — so this is a bd12 regression (or a spread of the pinned \
         speed-1..6 band), not a crop-axis result: {ctl_bad:?}"
    );
    let bad = report(&crop_rows);
    assert!(bad.is_empty(), "{KB28_HINT}: {bad:?}");
}

// ---------------------------------------------------------------------------
// What this file does NOT cover, and what each would cost.
// ---------------------------------------------------------------------------
//
// * **~~bd12 x the straddle~~** — DONE 2026-08-04,
//   `crop_straddle_bd12_byte_matches_where_interpretable`, 8/8 byte-exact.
// * **Crops straddling `is_4k_or_larger` (2160)** — KB-19's arm. **DONE
//   2026-08-04**, in `s4cov_hd_format_axis::crop_straddling_4k_arm_byte_matches`:
//   2154x2160 vs 2160x2160, **2/2 byte-exact in 35 s**. The predicate is the
//   same one arm 4 uses — 2154 mi-aligns UP to 2160, so
//   `default_min_partition_size`'s `BLOCK_8X8` arm (speed_features.c:187-189)
//   fires under the mi reading and must not under the crop reading.
//   The **25-30 minute** estimate this comment used to carry was ~50x high, and
//   the reason is worth keeping: it costed the arm from KB-19's **speed-0** cell
//   (C ~26 s / port ~195 s), but `is_4k_or_larger` is speed-UNconditional, so
//   the arm is observable at every speed 0..5 — at 6 KB-36's `is_1080p_or_larger`
//   arm sets the same field to the same value for a frame this large, and from 7
//   `set_allintra` sets it framesize-independently (:570). Cost a razor at the
//   cheapest speed the predicate is OBSERVABLE at, not the speed it was FOUND at.
// * **Multi-tile x the straddle.** KB-31's file is bd8 4:2:0 SB64 throughout
//   and its own residual (c) says so. A frame big enough to REQUIRE a tile
//   split is >4096 px wide or ~9.44 MP, so it cannot be combined with a
//   474x480 / 714x720 crop at all — the axis needs its own crop pair straddling
//   a boundary at that size, e.g. 4090x2154 vs 4096x2160. Same cost class as
//   the 2160 arm.
