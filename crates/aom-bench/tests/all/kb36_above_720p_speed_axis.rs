//! **KB-36 — the RD speed band ABOVE 1280x720, and the
//! `is_1080p_or_larger` speed-feature arm nothing had ever reached.**
//!
//! `s4cov_hd_speed_axis.rs` opens by naming its own ceiling:
//!
//! > *"Nothing had run `--cpu-used >= 1` above 640x640 except KB-22's two
//! > 1280x720 speed-0/1 cells and KB-19's 2160p speed-0 cell."*
//!
//! It then raised that ceiling to **1280x720**, and stopped there. Between
//! 1280x720 and 2160p sits `is_1080p_or_larger` (speed_features.c:171), and it
//! has exactly one arm in the all-intra framesize-dependent pass:
//!
//! ```text
//! if (speed >= 6) {
//!   ...
//!   if (is_1080p_or_larger) sf->part_sf.default_min_partition_size = BLOCK_8X8;
//!   ...
//! }                                        // speed_features.c:304-316
//! ```
//!
//! **It was unmodelled**, so every >= 1080p frame at `--cpu-used 6` searched
//! 4x4 partitions that C had already stopped at 8x8. Measured against real
//! aomenc before the fix: **1920x1080 -127 B and 2560x1440 +79 B**, with
//! **1920x1072 — eight pixels shorter, the same content, the same speed —
//! byte-exact**.
//!
//! **The window is ONE SPEED WIDE, and that is why nothing caught it.** Speed 7
//! sets the same field framesize-independently (`speed_features.c:570`), so
//! speeds 7, 8 and 9 cannot show it; below speed 6 the enclosing block does not
//! run. A speed sweep at any size under 1080 is green, and a size sweep at any
//! speed other than 6 is green. It needs the crossing, which is the same shape
//! KB-19, KB-22, KB-26 and KB-28 all had.
//!
//! This is the SECOND arm found on `default_min_partition_size`. KB-19 modelled
//! the `is_4k_or_larger` one and its ledger entry said the new method held
//! *"currently just the `is_4k_or_larger` one"* — accurate, and worth reading as
//! a queue rather than as a statement of completeness.
//!
//! **MEASURED 2026-08-03** (`benchmarks/kb36_above_720p_2026-08-03.tsv`): with
//! the arm modelled, **28/28 byte-exact** across 1280x720 / 1920x1072 /
//! 1920x1080 / 2560x1440 x `--cpu-used` 1..7, and the gate below extends that
//! to speeds 8 and 9.
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb36_above_720p_speed_axis -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

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

const CQ: i32 = 24;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Verdict {
    Ok,
    Diverge,
    Panic,
}

fn measure(cell: &EncodeCell) -> (Verdict, i64, String, u128) {
    let t0 = std::time::Instant::now();
    let c_tu = cell.c_encode();
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
    match got {
        Ok(p) if p == real => (Verdict::Ok, 0, String::new(), ms),
        Ok(p) => (
            Verdict::Diverge,
            p.len() as i64 - real.len() as i64,
            String::new(),
            ms,
        ),
        Err(_) => (Verdict::Panic, 0, msg.lock().unwrap().clone(), ms),
    }
}

fn base() -> EncodeCell {
    c::ref_init();
    EncodeCell::real_content("kb36", "av1-1-b8-00-quantizer-00", None, CQ, 0)
}

const KB36_HINT: &str = "an above-720p cell stopped byte-matching real aomenc. If the row is at \
     `--cpu-used 6` and >= 1080 on its SHORT side, suspect KB-36 first: \
     `SpeedFeatures::apply_allintra_framesize_dependent` must carry \
     `speed >= 6 && is_1080p_or_larger -> default_min_partition_size = BLOCK_8X8` \
     (speed_features.c:311-313). Otherwise compare the row against its <1080p twin at the \
     same speed: only->=1080p means a framesize arm; both means it is not";

// ---------------------------------------------------------------------------
// 1. The razor: 8 pixels across `is_1080p_or_larger`, at speed 6.
// ---------------------------------------------------------------------------

/// **The bite pair.** 1920x1080 and 1920x1072 are the same mirror-tiled
/// content at the same quantizer and the same speed; the only difference is
/// which side of `AOMMIN(w, h) >= 1080` they fall on. Speeds 5, 6 and 7 are
/// all run, because the arm's window is exactly speed 6 and the neighbours
/// are what prove that:
///
/// * **speed 5** — the enclosing `if (speed >= 6)` block does not run, so the
///   1080p row must match with the arm absent;
/// * **speed 6** — the only speed where the arm is observable. This is the
///   pair that measured -127 B before the fix;
/// * **speed 7** — `set_allintra` already sets `default_min_partition_size =
///   BLOCK_8X8` framesize-independently (speed_features.c:570), so the arm is
///   a no-op and the row must match either way.
///
/// A gate that ran only speed 6 would pass with a wrongly-gated fix (e.g. one
/// that fired at every speed); this triple does not.
///
/// **MEASURED 2026-08-03: 6/6 byte-exact.**
#[test]
#[ignore = "6 encode pairs at 1920x1072/1080 (~30 s); nightly / on-demand tier"]
fn is_1080p_arm_straddle_byte_matches() {
    let b = base();
    let mut rows: Vec<(usize, usize, i32, Verdict, i64)> = Vec::new();
    for &(w, h) in &[(1920usize, 1072usize), (1920, 1080)] {
        // Reach assertion (playbook §2): the pair is worthless unless it
        // straddles the predicate, and it must straddle it on the SHORT side.
        assert_eq!(
            w.min(h) >= 1080,
            h == 1080,
            "{w}x{h}: the pair must sit on opposite sides of AOMMIN(w,h) >= 1080"
        );
        for speed in [5, 6, 7] {
            let cell = mirror_tile(&b, &format!("kb36_{w}x{h}_s{speed}"), w, h, CQ, speed);
            let (v, d, note, ms) = measure(&cell);
            println!(
                "  {w}x{h} ({}) cq{CQ} cpu{speed}: {v:?} delta {d:+} [{ms} ms]{}",
                if w.min(h) >= 1080 { ">=1080p" } else { "<1080p " },
                if note.is_empty() {
                    String::new()
                } else {
                    format!("  [{note}]")
                }
            );
            rows.push((w, h, speed, v, d));
        }
    }
    let bad: Vec<String> = rows
        .iter()
        .filter(|r| r.3 != Verdict::Ok)
        .map(|r| format!("{}x{} cpu{} {:?} ({:+})", r.0, r.1, r.2, r.3, r.4))
        .collect();
    assert!(bad.is_empty(), "{KB36_HINT}: {bad:?}");
}

// ---------------------------------------------------------------------------
// 2. The band the ceiling was hiding: the whole RD speed axis above 720p.
// ---------------------------------------------------------------------------

/// **`--cpu-used` 1..9 at 1920x1080 and 2560x1440** — the sizes above
/// `s4cov_hd_speed_axis`'s 1280x720 ceiling and below KB-19's 2160p cell,
/// which no gate encoded at any speed other than 0.
///
/// Both sizes are >= 1080 on the short side, so both cross the KB-36 arm; the
/// `<1080p` control for them is the 1920x1072 row of
/// `is_1080p_arm_straddle_byte_matches`, and the `<720p` control is
/// `s4cov_hd_speed_axis::hd_speed_axis_byte_matches`' 640x640 column.
///
/// Speeds 8 and 9 are included even though the RD search does not run there:
/// they cross `speed >= 8 && is_720p_or_larger ->
/// force_large_partition_blocks_intra` (KB-32 root #1) at a size no gate had
/// used, and `av1_select_sb_size`'s allintra-speed-9-below-4k SB64 rule.
///
/// **MEASURED 2026-08-03: 18/18 byte-exact** (~5 min).
#[test]
#[ignore = "18 encode pairs up to 2560x1440 (~5 min); nightly / on-demand tier"]
fn above_720p_speed_axis_byte_matches() {
    let b = base();
    let mut bad: Vec<String> = Vec::new();
    let mut rows = 0usize;
    for &(w, h) in &[(1920usize, 1080usize), (2560, 1440)] {
        assert!(
            w.min(h) >= 1080,
            "{w}x{h} must be >= 1080 on the short side — that is what this grid is for"
        );
        for speed in 1..=9 {
            let cell = mirror_tile(&b, &format!("kb36band_{w}x{h}_s{speed}"), w, h, CQ, speed);
            let (v, d, note, ms) = measure(&cell);
            rows += 1;
            println!(
                "  {w}x{h} cq{CQ} cpu{speed}: {v:?} delta {d:+} [{ms} ms]{}",
                if note.is_empty() {
                    String::new()
                } else {
                    format!("  [{note}]")
                }
            );
            if v != Verdict::Ok {
                bad.push(format!("{w}x{h} cpu{speed} {v:?} ({d:+}) {note}"));
            }
        }
    }
    println!("  KB-36 above-720p band: {}/{rows} byte-exact", rows - bad.len());
    assert!(bad.is_empty(), "{KB36_HINT}: {bad:?}");
}
