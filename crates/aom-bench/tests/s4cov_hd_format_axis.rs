//! **S4 coverage extension — the >=1080p band at every format axis except the
//! one KB-36 swept, plus the band 1440..2160, plus the crop straddle at the
//! `is_4k_or_larger` boundary.**
//!
//! KB-36 found and closed `default_min_partition_size`'s `is_1080p_or_larger`
//! arm (`speed_features.c:311-313`), and its own closing note states exactly
//! how wide the grid it closed on was:
//!
//! > *"Still unmeasured on this axis: above 1440p at speed >= 1 other than
//! > KB-19's single 2160p speed-0 cell ...; the >=1080p band at bit depths
//! > above 8, at 4:2:2/4:4:4/mono, and at SB128; and any quantizer other than
//! > cq24 in this band."*
//!
//! That is one cell wide in format: **bd8 4:2:0 SB64 cq24**. KB-36 is itself the
//! precedent for why that matters — its arm is ONE SPEED wide, so a speed sweep
//! at any size under 1080 and a size sweep at any speed but 6 were both green
//! with the arm missing. A format axis hides an arm the same way.
//!
//! **What each test can and cannot decide, stated up front** (playbook §1 — a
//! row whose two explanations both predict its verdict decides nothing):
//!
//! * The KB-36 arm's observable window is **exactly `--cpu-used 6`**: below 6
//!   the enclosing `if (speed >= 6)` block does not run, and from speed 7
//!   `set_allintra` sets the same field framesize-independently
//!   (`speed_features.c:570`). So a format arm of this axis must be run at
//!   speed 6 against its own `<1080p` twin to be an ARM crossing.
//! * High bit depth cannot be an arm crossing here, and that is a measured
//!   constraint rather than a budget cut: bd10/bd12 content diverges from real
//!   aomenc at `--cpu-used` 1..6 on SB-EXACT 64x64 content already (the pinned
//!   `b10_64` band of `config_permutations.rs::speed_envelope_stock_map_is_pinned`,
//!   widened to bd12/mono/4:4:4 by `s4cov_qm_axis.rs`), and speed 6 is inside
//!   it. The interpretable hbd speeds are {0, 7}, which is precisely the
//!   complement of the arm's window. The hbd rows below therefore measure
//!   **byte-identity of the >=1080p band at high bit depth**, at the speeds
//!   where that question is well-posed — not the arm — and they say so.
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test s4cov_hd_format_axis -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;
use aom_sys_ref::cx_ctrl::{AOM_SUPERBLOCK_SIZE_128X128, AV1E_SET_SUPERBLOCK_SIZE};

/// KB-36's quality point, so the bd8 4:2:0 SB64 rows here are directly
/// comparable to `kb36_above_720p_speed_axis.rs` row for row.
const CQ: i32 = 24;

/// Mirror-tile a decoded cell up to `w x h` — same recipe as
/// `kb36_above_720p_speed_axis::mirror_tile`, extended to carry monochrome
/// cells through unchanged (a mono cell has no chroma planes to tile).
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
    let (mut u, mut v) = (Vec::new(), Vec::new());
    if !base.mono {
        let (bcw, bch) = ((bw + base.ss_x) >> base.ss_x, (bh + base.ss_y) >> base.ss_y);
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
        cq_level: cq,
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

/// Re-render a cell at a higher bit depth by BIT REPLICATION — `s4cov_qm_axis
/// ::to_bd` verbatim, and for its reason (a plain shift leaves the low bits
/// zero, which is the regime KB-4 calls out as the easy one).
fn to_bd(base: &EncodeCell, label: &str, bd: u8) -> EncodeCell {
    assert!(bd > base.bd, "{label}: to_bd only widens");
    let k = u32::from(bd - base.bd);
    let src_bits = u32::from(base.bd);
    let widen = |v: &u16| -> u16 { (v << k) | (v >> (src_bits - k)) };
    EncodeCell {
        label: label.to_string(),
        bd,
        y: base.y.iter().map(widen).collect(),
        u: base.u.iter().map(widen).collect(),
        v: base.v.iter().map(widen).collect(),
        ..base.clone()
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Verdict {
    Ok,
    Diverge,
    Panic,
}

fn measure(cell: &EncodeCell, ctrls: &[(i32, i32)]) -> (Verdict, i64, String, u128, Vec<u8>) {
    let t0 = std::time::Instant::now();
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
    let ms = t0.elapsed().as_millis();
    let (v, d, note) = match got {
        Ok(p) if p == real => (Verdict::Ok, 0, String::new()),
        Ok(p) => (Verdict::Diverge, p.len() as i64 - real.len() as i64, String::new()),
        Err(_) => (Verdict::Panic, 0, msg.lock().unwrap().clone()),
    };
    (v, d, note, ms, c_tu)
}

fn base_b8() -> EncodeCell {
    c::ref_init();
    EncodeCell::real_content("hdfmt8", "av1-1-b8-00-quantizer-00", None, CQ, 0)
}

fn base_b10() -> EncodeCell {
    c::ref_init();
    EncodeCell::real_content("hdfmt10", "av1-1-b10-00-quantizer-00", None, CQ, 0)
}

const HINT: &str = "an above-1080p cell stopped byte-matching real aomenc. If the row is at \
     `--cpu-used 6`, >= 1080 on its SHORT side, and its <1080p twin at the same speed and \
     format MATCHES, that is KB-36's arm shape at a format it was never measured at: \
     `SpeedFeatures::apply_allintra_framesize_dependent` must carry \
     `speed >= 6 && is_1080p_or_larger -> default_min_partition_size = BLOCK_8X8` \
     (speed_features.c:311-313) and it must not be format-conditioned. If BOTH twins diverge \
     it is not a framesize arm at all";

// ---------------------------------------------------------------------------
// 1. The >=1080p band at every format axis KB-36 did not sweep.
// ---------------------------------------------------------------------------

/// **The format axis of KB-36's band, run at the arm's own speed against the
/// arm's own razor.**
///
/// Every format is measured at BOTH 1920x1072 and 1920x1080 — the same
/// mirror-tiled content, eight pixels apart, on opposite sides of
/// `AOMMIN(w, h) >= 1080` — at `--cpu-used 6`, the only speed where the arm is
/// observable. That pairing is what makes a divergence attributable: a row that
/// diverges at 1080 while its 1072 twin matches is the arm; a pair that
/// diverges together is not a framesize arm at all.
///
/// The formats:
///
/// | arm | what it crosses that KB-36's grid does not |
/// |---|---|
/// | 4:4:4 / 4:2:2 | the chroma footprint of every frame-edge clip and of the partition walk's `max_block_units` |
/// | monochrome | removes the chroma machinery entirely — the negative control for it |
/// | SB128 | four independent 64x64 CNN roots per superblock, and `av1_select_sb_size`'s explicit-size path |
/// | cq5 / cq40 / cq63 | `default_min_partition_size` interacts with the qindex-dependent speed pass (KB-22's arm), which KB-36 held fixed at base_qindex 96 |
///
/// **MEASURED 2026-08-04: 14/14 byte-exact** (26 s).
#[test]
#[ignore = "14 encode pairs at 1920x1072/1080 (~30 s); nightly / on-demand tier"]
fn above_1080p_format_axis_byte_matches() {
    let b8 = base_b8();
    // (tag, source cell, extra C controls). SB128 is the same 4:2:0 source
    // with a control pair, not a different source.
    let sb128 = [(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_128X128)];
    let arms: Vec<(&str, EncodeCell, &[(i32, i32)], i32)> = vec![
        ("444   ", to_ss(&b8, "444", 0, 0), &[], CQ),
        ("422   ", to_ss(&b8, "422", 1, 0), &[], CQ),
        ("mono  ", to_mono(&b8, "mono"), &[], CQ),
        ("sb128 ", b8.clone(), &sb128, CQ),
        ("cq5   ", b8.clone(), &[], 5),
        ("cq40  ", b8.clone(), &[], 40),
        ("cq63  ", b8.clone(), &[], 63),
    ];

    /// Divergent `(tag, h)` rows at speed 6, pinned in both directions.
    /// **Empty as measured 2026-08-04** — 14/14 byte-exact.
    const HD_FORMAT_OPEN: &[(&str, usize)] = &[];

    let mut observed: Vec<(String, usize)> = Vec::new();
    let mut panicked: Vec<String> = Vec::new();
    let mut inert: Vec<String> = Vec::new();
    for (tag, src, ctrls, cq) in &arms {
        for h in [1072usize, 1080usize] {
            // Reach assertion (playbook §2): the pair is worthless unless it
            // straddles the predicate on the SHORT side.
            assert_eq!(
                1920usize.min(h) >= 1080,
                h == 1080,
                "1920x{h}: the pair must sit on opposite sides of AOMMIN(w,h) >= 1080"
            );
            let cell = mirror_tile(src, &format!("hd_{}_{h}", tag.trim()), 1920, h, *cq, 6);
            assert_eq!(
                (cell.mono, cell.ss_x, cell.ss_y, cell.bd),
                (src.mono, src.ss_x, src.ss_y, src.bd),
                "the mirror-tile must preserve the format"
            );
            let (v, d, note, ms, c_tu) = measure(&cell, ctrls);
            // Anti-vacuity for the SB128 arm only (playbook §8 — derive
            // coverage from artefacts): `--sb-size=128` must CHANGE the C
            // stream, else the row is an SB64 test wearing an SB128 label.
            if !ctrls.is_empty() && c_tu == cell.c_encode_ctrls(&[]) {
                inert.push(format!("{} 1920x{h}", tag.trim()));
            }
            println!(
                "  {tag} 1920x{h} ({}) cq{cq} cpu6: {v:?} delta {d:+} [{ms} ms]{}",
                if h >= 1080 { ">=1080p" } else { "<1080p " },
                if note.is_empty() { String::new() } else { format!("  [{note}]") }
            );
            if v == Verdict::Panic {
                panicked.push(format!("{} 1920x{h}: {note}", tag.trim()));
            }
            if v != Verdict::Ok {
                observed.push((tag.trim().to_string(), h));
            }
        }
    }
    assert!(
        panicked.is_empty(),
        "the port PANICKED instead of encoding an above-1080p cell at a non-default format. \
         That is an unported arm, not a near-tie: {panicked:?}"
    );
    assert!(
        inert.is_empty(),
        "--sb-size=128 did not change the C stream vs --sb-size=64, so those rows prove \
         nothing about the 128-superblock geometry: {inert:?}"
    );
    let pinned: Vec<(String, usize)> = HD_FORMAT_OPEN
        .iter()
        .map(|(t, h)| ((*t).to_string(), *h))
        .collect();
    assert_eq!(observed, pinned, "{HINT}: {observed:?}");
}

/// **The >=1080p band at high bit depth, at the two speeds where it can be
/// read.**
///
/// Split out from the test above because it is a WEAKER claim and must not be
/// read as the same one. bd10/bd12 diverge from real aomenc at `--cpu-used`
/// 1..6 on SB-exact 64x64 content already (`b10_64` / `HBD_OPEN`), and the
/// KB-36 arm is observable at speed 6 alone — so at high bit depth this axis
/// **cannot** cross the arm. What it can do, and does, is establish that the
/// >=1080p band is byte-identical at bd10 and bd12 at speeds 0 and 7, which
/// nothing had measured: the largest high-bit-depth frame in the tree before
/// this was 256x256 (`s4cov_partial_sb_axis.rs`).
///
/// Both sides of the razor are still run, so if the hbd band ever DOES become
/// readable at speed 6 the pair is already in place.
///
/// **MEASURED 2026-08-04: 6/8 byte-exact** — see `HD_HBD_OPEN`; the two open
/// rows are KB-38's speed-0 band, which reaches bd8 too.
#[test]
#[ignore = "8 high-bit-depth encode pairs at 1920x1072/1080 (~2 min); nightly / on-demand tier"]
fn above_1080p_high_bitdepth_byte_matches_where_interpretable() {
    let b10 = base_b10();
    assert_eq!(b10.bd, 10, "bd10 source");
    let b12 = to_bd(&b10, "b12", 12);
    // Non-vacuity: a bd12 cell that fits in 10 bits is a bd10 cell relabelled.
    assert!(
        b12.y.iter().any(|&s| s > 1023) && b12.y.iter().any(|&s| s & 3 != 0),
        "bd12 cell must use the extra two bits"
    );

    /// Divergent `(tag, h, speed)` rows, pinned in both directions.
    ///
    /// **MEASURED 2026-08-04: 6/8 byte-exact.** The two speed-0 rows at
    /// 1920x1080 are **KB-38's band, not this axis's finding** — measured, not
    /// argued: `speed0_1080p_band_map_is_pinned` reproduces them at **bd8** as
    /// well (-726 B), and their `1920x1072` twins at the same bit depths and
    /// quantizer are byte-exact. Every speed-7 row here matches, which is what
    /// this test set out to establish (the >= 1080p band at high bit depth had
    /// never been encoded at all — the largest hbd frame in the tree before
    /// this was 256x256).
    const HD_HBD_OPEN: &[(&str, usize, i32)] = &[("bd10", 1080, 0), ("bd12", 1080, 0)];

    let mut observed: Vec<(String, usize, i32)> = Vec::new();
    let mut panicked: Vec<String> = Vec::new();
    for (tag, src) in [("bd10", &b10), ("bd12", &b12)] {
        for h in [1072usize, 1080usize] {
            for speed in [0, 7] {
                // Speed 0 at 1920x1080 costs minutes; only speed 7 is run on
                // both sides of the razor, and speed 0 only at >= 1080p (the
                // side that has never been encoded at high bit depth at all).
                if speed == 0 && h != 1080 {
                    continue;
                }
                let cell = mirror_tile(src, &format!("hdhbd_{tag}_{h}_s{speed}"), 1920, h, CQ, speed);
                assert_eq!(cell.bd, src.bd, "the mirror-tile must preserve the bit depth");
                let (v, d, note, ms, _) = measure(&cell, &[]);
                println!(
                    "  {tag} 1920x{h} cq{CQ} cpu{speed}: {v:?} delta {d:+} [{ms} ms]{}",
                    if note.is_empty() { String::new() } else { format!("  [{note}]") }
                );
                if v == Verdict::Panic {
                    panicked.push(format!("{tag} 1920x{h} cpu{speed}: {note}"));
                }
                if v != Verdict::Ok {
                    observed.push((tag.to_string(), h, speed));
                }
            }
        }
    }
    assert!(
        panicked.is_empty(),
        "the port PANICKED on an above-1080p high-bit-depth cell — an unported hbd arm \
         (KB-20's shape) at a frame size nothing had reached: {panicked:?}"
    );
    let pinned: Vec<(String, usize, i32)> = HD_HBD_OPEN
        .iter()
        .map(|(t, h, s)| ((*t).to_string(), *h, *s))
        .collect();
    assert_eq!(
        observed, pinned,
        "the above-1080p high-bit-depth map moved. Speeds 0 and 7 are the only ones where high \
         bit depth is byte-exact on small SB-exact content, so a divergence here is either a \
         spread of the pinned `b10_64` / `HBD_OPEN` band into the frame sizes, or a \
         framesize-dependent hbd arm nothing has modelled: {observed:?}"
    );
}

/// **Localizer for the speed-0 >=1080p divergence the test above found.**
/// Diagnostic, not a gate — the gate is
/// [`speed0_1080p_qindex108_arm_byte_matches`] below.
///
/// The finding: `1920x1080 cq24 --cpu-used 0` diverges at bd10 (+483 B) and
/// bd12 (+181 B) while every speed-7 row matches. The candidate mechanism is
/// the **sub-block nothing in the tree had ever entered**
/// (speed_features.c:2926-2935):
///
/// ```text
/// if (speed == 0) {
///   if (is_720p_or_larger && base_qindex <= 128) {           // KB-22's arm
///     ...
///     if (is_1080p_or_larger && base_qindex <= 108) {        // THIS one
///       sf->rd_sf.tx_domain_dist_level      = boosted ? 1 : 2;
///       sf->rd_sf.tx_domain_dist_thres_level = 1;
///       sf->tx_sf.tx_type_search.ml_tx_split_thresh  = 4000;
///       sf->tx_sf.tx_type_search.prune_2d_txfm_mode  = TX_TYPE_PRUNE_2;
///       sf->tx_sf.tx_type_search.skip_tx_search      = 1;
///       ...
/// ```
///
/// **The predicate, stated so it can be falsified on one cell** (playbook §9):
/// the arm fires exactly when `speed == 0 && AOMMIN(w, h) >= 1080 &&
/// base_qindex <= 108`. cq24 is `base_qindex` 96 (fires); cq32 is 128 (does
/// not). So the grid below must diverge on **exactly** the two
/// `1920x1080 cq24` rows and match on the other ten — a size that is >= 720 but
/// < 1080 cannot fire it at any quantizer, and a >= 1080p frame at cq32 cannot
/// either.
///
/// That three-term window is why nothing caught it: KB-36's >= 1080p grid runs
/// `--cpu-used` 1..9 (the whole block is `speed == 0`), and KB-19/KB-22's
/// speed-0 2160p cell runs **cq32**, whose `base_qindex` 128 is above this
/// sub-block's 108 while still inside KB-22's own 128.
#[test]
#[ignore = "12 speed-0 encode pairs up to 1920x1080; diagnostic, run explicitly"]
fn speed0_1080p_qindex_arm_localize() {
    let b8 = base_b8();
    let b10 = base_b10();
    let mut fired: Vec<String> = Vec::new();
    let mut rows: Vec<(String, usize, usize, i32, Verdict, i64)> = Vec::new();
    for (tag, src) in [("bd8 ", &b8), ("bd10", &b10)] {
        for &(w, h) in &[(1280usize, 720usize), (1920, 1072), (1920, 1080)] {
            for cq in [24, 32] {
                let cell = mirror_tile(src, &format!("loc_{tag}_{w}x{h}_{cq}"), w, h, cq, 0);
                let (v, d, note, ms, _) = measure(&cell, &[]);
                // The C mapping is `base_qindex = 4 * cq` on this path; both
                // sides of the sub-block's 108 are exercised.
                let predicted = w.min(h) >= 1080 && 4 * cq <= 108;
                println!(
                    "  {tag} {w}x{h} cq{cq} (qindex {}) cpu0: {v:?} delta {d:+} \
                     [predicted {}] [{ms} ms]{}",
                    4 * cq,
                    if predicted { "ARM" } else { "no-arm" },
                    if note.is_empty() { String::new() } else { format!("  [{note}]") }
                );
                if v != Verdict::Ok {
                    fired.push(format!("{tag} {w}x{h} cq{cq} ({d:+})"));
                }
                rows.push((tag.to_string(), w, h, cq, v, d));
            }
        }
    }
    println!("  divergent rows: {fired:?}");
    assert!(
        !rows.is_empty(),
        "the localizer must encode something to be worth running"
    );
}

/// **The gate over the same 12-cell grid — a self-promoting pin (playbook §5),
/// because the speed-0 >=1080p band is NOT closed.**
///
/// KB-38's arm is ported, and it is load-bearing: it moved
/// `bd8 1920x1080 cq24` from **-536 B to -726 B** and `bd10` from **+483 to
/// +408**. It did not close either cell, and one more row diverges that the
/// arm's predicate does not explain at all (`bd10 1920x1080 cq32`, **-8 B** —
/// `base_qindex` 128 is above this sub-block's 108, and its bd8 twin at the
/// same size and quantizer is byte-exact). So there is at least one further
/// root at `(speed 0, min(w,h) >= 1080)`, and this pin records the map rather
/// than asserting a closure that has not happened.
///
/// **The nine matching rows are the load-bearing part of the pin.** They are
/// one size step (1072 vs 1080) and one quantizer step (cq24 vs cq32) on each
/// side of the predicate, at two bit depths, and every one of them is
/// byte-exact — which is what makes ">= 1080p at speed 0" the shape rather than
/// "large frames at speed 0" or "high bit depth at speed 0". 1920x1072 is eight
/// pixels of the same mirror-tiled content.
///
/// Fails in BOTH directions: a new divergent row is a regression, and a row
/// that starts MATCHING means a root closed and the pin must be re-cut.
#[test]
#[ignore = "12 speed-0 encode pairs up to 1920x1080 (~11 min); nightly / on-demand tier"]
fn speed0_1080p_band_map_is_pinned() {
    let b8 = base_b8();
    let b10 = base_b10();
    // (bit-depth tag, w, h, cq) — the CURRENT divergent set, measured
    // 2026-08-04 WITH KB-38's arm ported.
    const SPEED0_1080P_OPEN: &[(&str, usize, usize, i32)] = &[
        ("bd8 ", 1920, 1080, 24),
        ("bd10", 1920, 1080, 24),
        ("bd10", 1920, 1080, 32),
    ];
    let mut observed: Vec<(String, usize, usize, i32)> = Vec::new();
    let mut fired_side = 0usize;
    for (tag, src) in [("bd8 ", &b8), ("bd10", &b10)] {
        for &(w, h) in &[(1280usize, 720usize), (1920, 1072), (1920, 1080)] {
            for cq in [24, 32] {
                // Reach assertion (playbook §2): the grid is worthless unless it
                // lands on BOTH sides of BOTH terms of KB-38's predicate.
                if w.min(h) >= 1080 && 4 * cq <= 108 {
                    fired_side += 1;
                }
                let cell = mirror_tile(src, &format!("q108_{tag}_{w}x{h}_{cq}"), w, h, cq, 0);
                let (v, d, note, ms, _) = measure(&cell, &[]);
                println!(
                    "  {tag} {w}x{h} cq{cq} (qindex {}) cpu0: {v:?} delta {d:+} [{ms} ms]{}",
                    4 * cq,
                    if note.is_empty() { String::new() } else { format!("  [{note}]") }
                );
                assert_ne!(
                    v,
                    Verdict::Panic,
                    "the port PANICKED at {tag} {w}x{h} cq{cq} cpu0 — that is an unported arm, \
                     never a pinnable near-tie: {note}"
                );
                if v != Verdict::Ok {
                    observed.push((tag.to_string(), w, h, cq));
                }
            }
        }
    }
    assert_eq!(
        fired_side, 2,
        "exactly the two `1920x1080 cq24` rows must satisfy \
         `min(w,h) >= 1080 && base_qindex <= 108`; if that count moves the grid no longer \
         straddles KB-38's predicate and proves nothing"
    );
    let pinned: Vec<(String, usize, usize, i32)> = SPEED0_1080P_OPEN
        .iter()
        .map(|(t, w, h, q)| ((*t).to_string(), *w, *h, *q))
        .collect();
    assert_eq!(
        observed, pinned,
        "the speed-0 >=1080p map moved. A NEW row (especially a 1920x1072 or a 1280x720 one) \
         is a regression and means the band is wider than >= 1080p. A row that started \
         MATCHING means one of KB-38's remaining roots closed — re-pin `SPEED0_1080P_OPEN` \
         and say which. The arm already ported is \
         `is_1080p_or_larger && base_qindex <= 108` (speed_features.c:2926-2935); what is \
         still missing is whatever else moves between 1920x1072 and 1920x1080 at speed 0"
    );
}

// ---------------------------------------------------------------------------
// 2. The band 1440..2160 at speed >= 1.
// ---------------------------------------------------------------------------

/// **`--cpu-used` 1..9 between 1440p and 4k** — the gap KB-36 left above its
/// own 2560x1440 ceiling, where the only cell that ever existed is KB-19's
/// single 2160x2160 **speed-0** one.
///
/// Two sizes, both > 1440 and < 2160 on the short side, so both sit strictly
/// inside the gap between the two `default_min_partition_size` framesize arms:
/// `is_1080p_or_larger` (covered by KB-36, and live here at speed 6) and
/// `is_4k_or_larger` (2160, the next test's subject). 1920x1920 and 2560x1600
/// differ in aspect and in which dimension is the short one, because every
/// framesize predicate in `set_allintra_speed_feature_framesize_dependent`
/// reads `AOMMIN(w, h)` and a square cell cannot distinguish "short side" from
/// "either side".
///
/// **MEASURED 2026-08-04: 18/18 byte-exact** (431 s).
#[test]
#[ignore = "18 encode pairs at up to 4.1 MP (~7 min); nightly / on-demand tier"]
fn band_1440_to_2160_speed_axis_byte_matches() {
    let b8 = base_b8();

    /// Divergent `(w, h, speed)` rows, pinned in both directions.
    /// **Empty as measured 2026-08-04** — 18/18 byte-exact.
    const BAND_OPEN: &[(usize, usize, i32)] = &[];

    let mut observed: Vec<(usize, usize, i32)> = Vec::new();
    let mut panicked: Vec<String> = Vec::new();
    for &(w, h) in &[(1920usize, 1920usize), (2560, 1600)] {
        // Reach assertion (playbook §2): strictly inside the gap, on the SHORT
        // side, or the row is a re-run of a band that is already gated.
        let short = w.min(h);
        assert!(
            short > 1440 && short < 2160,
            "{w}x{h}: short side {short} must be strictly between the 1440 ceiling KB-36 \
             reached and the 2160 `is_4k_or_larger` boundary"
        );
        for speed in 1..=9 {
            let cell = mirror_tile(&b8, &format!("band_{w}x{h}_s{speed}"), w, h, CQ, speed);
            let (v, d, note, ms, _) = measure(&cell, &[]);
            println!(
                "  {w}x{h} cq{CQ} cpu{speed}: {v:?} delta {d:+} [{ms} ms]{}",
                if note.is_empty() { String::new() } else { format!("  [{note}]") }
            );
            if v == Verdict::Panic {
                panicked.push(format!("{w}x{h} cpu{speed}: {note}"));
            }
            if v != Verdict::Ok {
                observed.push((w, h, speed));
            }
        }
    }
    assert!(
        panicked.is_empty(),
        "the port PANICKED between 1440p and 4k — an unported arm at a frame size no gate had \
         encoded at any speed other than 0: {panicked:?}"
    );
    let pinned: Vec<(usize, usize, i32)> = BAND_OPEN.to_vec();
    assert_eq!(observed, pinned, "{HINT}: {observed:?}");
}

// ---------------------------------------------------------------------------
// 3. The crop straddle at `is_4k_or_larger` (2160).
// ---------------------------------------------------------------------------

/// **KB-28's root at KB-19's boundary: 2154x2160 vs 2160x2160.**
///
/// `is_4k_or_larger` is `AOMMIN(cm->width, cm->height) >= 2160`
/// (`speed_features.c:172`), and KB-28's root is that the in-walk predicates
/// once re-derived framesize from the **mi-aligned** extent instead of the
/// crop. 2154 is exactly the shape that separates the two readings:
/// `ALIGN_POWER_OF_TWO(2154, 3) == 2160`, so under the mi reading 2154x2160
/// takes the 4k arm and under the crop reading it does not. 2160x2160 takes it
/// under both, and is the positive control.
///
/// **Run at `--cpu-used 5`, and that is not a cost dodge — it is the only band
/// where the arm is observable at all.** The `is_4k_or_larger` assignment is
/// unconditional on speed inside `set_allintra_speed_feature_framesize_dependent`
/// (KB-19), but from speed 6 the `is_1080p_or_larger` arm sets the same field to
/// the same value for any frame this large (KB-36), and from speed 7
/// `set_allintra` sets it framesize-independently (`speed_features.c:570`). So
/// the arm is distinguishable only at speeds 0..5, and speed 5 is the cheapest
/// of those by a wide margin (KB-19's speed-0 2160² cell is C ~26 s / port
/// ~195 s).
///
/// **MEASURED 2026-08-04: 2/2 byte-exact, in 35 s** — and that number is itself a
/// finding. The coverage queue costed this arm at *25-30 minutes of port encode*
/// from KB-19's speed-0 cell (C ~26 s / port ~195 s). At speed 5 the same razor
/// costs **20 s and 15 s** for the two encodes: the estimate was ~50x high
/// because it assumed the speed the arm was FOUND at rather than the cheapest
/// speed the arm is OBSERVABLE at.
#[test]
#[ignore = "2 encode pairs at 4.6 MP (~35 s); nightly / on-demand tier"]
fn crop_straddling_4k_arm_byte_matches() {
    let b8 = base_b8();
    let mut bad: Vec<String> = Vec::new();
    for &(w, h) in &[(2154usize, 2160usize), (2160, 2160)] {
        let short = w.min(h);
        // Reach assertions (playbook §2), both directions of the razor:
        // the pair must disagree under the CROP reading and AGREE under the
        // mi reading, else it does not separate them at all.
        let mi_w = (w + 7) & !7usize;
        let mi_h = (h + 7) & !7usize;
        assert_eq!(
            short >= 2160,
            w == 2160,
            "{w}x{h}: the pair must sit on opposite sides of AOMMIN(crop w, crop h) >= 2160"
        );
        assert!(
            mi_w.min(mi_h) >= 2160,
            "{w}x{h}: mi extent {mi_w}x{mi_h} must be >= 2160 on BOTH sides — that is what \
             makes this a crop-vs-mi razor rather than an ordinary size pair"
        );
        let cell = mirror_tile(&b8, &format!("k4_{w}x{h}"), w, h, CQ, 5);
        let (v, d, note, ms, _) = measure(&cell, &[]);
        println!(
            "  {w}x{h} (mi {mi_w}x{mi_h}, {}) cq{CQ} cpu5: {v:?} delta {d:+} [{ms} ms]{}",
            if short >= 2160 { ">=4k" } else { "<4k " },
            if note.is_empty() { String::new() } else { format!("  [{note}]") }
        );
        if v != Verdict::Ok {
            bad.push(format!("{w}x{h} {v:?} ({d:+}) {note}"));
        }
    }
    assert!(
        bad.is_empty(),
        "a cell straddling `is_4k_or_larger` stopped byte-matching. If 2154x2160 diverges while \
         2160x2160 matches, some consumer re-derived the 4k predicate from the MI extent \
         instead of the crop — KB-28's root at KB-19's boundary; `SbEncodeEnv::frame_min_dim()` \
         is the value it must read. If BOTH diverge it is KB-19's arm itself: {bad:?}"
    );
}
