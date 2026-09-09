//! **bd12 x dispatch tier — the gate that did not exist, which is why the
//! disagreement it covers sat open for a month.**
//!
//! The standing goal names TWO classes that must CLOSE rather than ship under
//! the "measured, attributed, bounded and documented" cap. One of them is
//! *"anything where the port disagrees with ITSELF across dispatch tiers (the
//! bd12 `1920x1080 cq24 cpu0` cell, +181 B default vs +55 B scalar), because a
//! kernel whose tiers disagree is a differential hole (playbook §1)"*.
//!
//! **MEASURED 2026-09-09: that disagreement is CLOSED.** At the exact cell
//! KB-38 named, both tiers now emit the byte-identical stream
//! (`len 158731`, fnv-1a-64 `4b5f1efec0345f57`), against KB-38's 2026-08-04
//! reading of +181 default / +55 scalar. The port no longer disagrees with
//! itself; the residual `+59 B` against real aomenc is a plain DIVERGENCE
//! (KB-38's own open root), not a differential hole, so it falls under the cap
//! instead of the must-close list.
//!
//! It closed silently, because **nothing in the tree encodes bd12 at all**:
//! `s4cov_hd_format_axis::speed0_1080p_band_map_is_pinned` sweeps bd8 and bd10
//! only. That is the hole this file fills.
//!
//! # How this gates a TIER disagreement without running two dispatch modes
//!
//! `AOM_FORCE_SCALAR` is read once per process (`dispatch::scalar_forced` is a
//! one-time pin), so no single test can compare the tiers directly. It does not
//! need to: **`just gate-landing` runs the whole suite TWICE**, once as
//! `test-next` and once as `test-next-scalar`. A byte-identity assertion
//! checked under both modes is therefore exactly a tier-agreement assertion —
//! if the tiers ever disagree, one of the two runs fails and names the cell.
//! That is why the default-tier test below asserts byte-identity to real
//! aomenc rather than merely pinning a hash: it gets C-parity and
//! tier-agreement from one measurement.

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

/// Widen to `bd`, replicating the top bits into the new low bits. Widening and
/// mirror-tiling are both per-sample maps, so they commute — this is the same
/// content `s4cov_hd_format_axis::to_bd` produces.
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

fn mirror_tile(base: &EncodeCell, label: &str, w: usize, h: usize, cq: i32) -> EncodeCell {
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
    EncodeCell { label: label.to_string(), w, h, cq_level: cq, speed: 0, y, u, v, ..base.clone() }
}

fn bd12_cell(w: usize, h: usize, cq: i32) -> EncodeCell {
    c::ref_init();
    let b10 = EncodeCell::real_content("tier10", "av1-1-b10-00-quantizer-00", None, cq, 0);
    let tiled = mirror_tile(&b10, &format!("bd12_{w}x{h}_cq{cq}"), w, h, cq);
    to_bd(&tiled, &format!("bd12_{w}x{h}_cq{cq}"), 12)
}

/// `(port_len - c_len)`, or `None` if the port panicked.
fn delta(cell: &EncodeCell) -> Option<i64> {
    let c_tu = cell.c_encode_ctrls(&[]);
    assert!(!c_tu.is_empty(), "{}: C encode failed", cell.label);
    let real = EncodeCell::frame_obu_payload(&c_tu);
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cell.port_encode_with(&c_tu, &ToggleKnobs::default())
    }));
    match got {
        Ok(p) if p == real => Some(0),
        Ok(p) => Some(p.len() as i64 - real.len() as i64),
        Err(_) => None,
    }
}

/// **The default-tier bd12 gate: 18 cells, ~20 s, byte-identical to real
/// aomenc — and, run under both dispatch modes by `just gate-landing`, a
/// bd12 tier-agreement gate.**
///
/// Sizes deliberately span SB-exact (64/128/192/256) and partial-superblock
/// (100/196) shapes, because several past roots (KB-23, KB-34) were reachable
/// only through a partial superblock. Every one of these is below the
/// `is_1080p_or_larger` predicate, so none of them can reach KB-38's open arm —
/// which is what makes a hard byte-identity assertion the right shape here
/// rather than a pinned divergence map.
#[test]
fn bd12_small_grid_byte_matches_real_aomenc() {
    let mut bad: Vec<String> = Vec::new();
    for &(w, h) in &[(64usize, 64usize), (100, 100), (128, 128), (196, 196), (192, 192), (256, 256)]
    {
        for cq in [24i32, 32, 48] {
            let cell = bd12_cell(w, h, cq);
            match delta(&cell) {
                Some(0) => println!("  {} OK", cell.label),
                Some(d) => bad.push(format!("{} DIVERGE {d:+} B", cell.label)),
                None => bad.push(format!("{} PANIC", cell.label)),
            }
        }
    }
    assert!(
        bad.is_empty(),
        "bd12 cells below 1080p must be byte-identical to real aomenc at every dispatch tier. \
         If this fails ONLY under `AOM_FORCE_SCALAR=1` (or only WITHOUT it), the port disagrees \
         with itself across dispatch tiers — that is a differential hole (playbook §1) and a \
         must-close item under the standing goal, NOT a pinnable near-tie. Failures: {bad:#?}"
    );
}

/// **The >=1080p bd12 band, pinned — including the cell the standing goal
/// named.** `#[ignore]`d: ~3.5 min (measured 194.7 s), dominated by the 1920x1080
/// speed-0 encode.
///
/// The razor is the point. `1080x1080 cq24` diverges while BOTH of its
/// neighbours are byte-exact — `1072x1072 cq24` is eight pixels under the
/// `AOMMIN(w,h) >= 1080` term and `1080x1080 cq32` has `base_qindex` 128 above
/// the `<= 108` term — so the divergence is attributable to KB-38's predicate
/// rather than to frame size or to bit depth.
#[test]
#[ignore = "~3.5 min (measured 194.7 s): dominated by the 1920x1080 bd12 speed-0 encode"]
fn bd12_1080p_band_map_is_pinned() {
    // (w, h, cq, expected delta vs real aomenc). MEASURED 2026-09-09.
    const MAP: &[(usize, usize, i32, i64)] = &[
        (1072, 1072, 24, 0),
        (1072, 1072, 32, 0),
        (1080, 1080, 32, 0),
        (1080, 1080, 24, 138),
        // The cell the standing goal names. KB-38 measured +181 default /
        // +55 scalar on 2026-08-04; both tiers now agree here.
        (1920, 1080, 24, 59),
    ];
    let mut fired = 0usize;
    let mut observed: Vec<(usize, usize, i32, i64)> = Vec::new();
    for &(w, h, cq, _) in MAP {
        if w.min(h) >= 1080 && 4 * cq <= 108 {
            fired += 1;
        }
        let cell = bd12_cell(w, h, cq);
        let d = delta(&cell).unwrap_or_else(|| {
            panic!("{} PANICKED — an unported arm, never a pinnable divergence", cell.label)
        });
        println!("  {} delta {d:+}", cell.label);
        observed.push((w, h, cq, d));
    }
    assert_eq!(
        fired, 2,
        "the grid must straddle BOTH terms of KB-38's predicate \
         (`min(w,h) >= 1080 && base_qindex <= 108`); if this count moves the grid proves nothing"
    );
    let expected: Vec<(usize, usize, i32, i64)> = MAP.to_vec();
    assert_eq!(
        observed, expected,
        "the bd12 >=1080p map moved. A row that started MATCHING means one of KB-38's roots \
         closed — re-pin and say which. A NEW divergent row below 1080p, or at cq32, means the \
         band is wider than KB-38's predicate. And if THIS test disagrees between \
         `test-next` and `test-next-scalar`, the bd12 dispatch-tier disagreement has REOPENED: \
         that is the must-close class, not a divergence."
    );
}
