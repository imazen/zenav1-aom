//! **KB-34 — the nonrd (`--cpu-used` 8/9) estimate arm codes a NON-SQUARE leaf.**
//!
//! Until 2026-08-02 it refused to: `nonrd_pickmode::nonrd_leaf_tx_size` panicked
//! with *"HANDOFF: nonrd estimate arm at non-square leaf bsize {bsize} —
//! max_txsize_lookup gives a tx smaller than the leaf, so
//! `av1_foreach_transformed_block_in_plane` visits more than one txb and
//! `nonrd_pick_intra_mode`'s single-txb invariant does not hold (KB-32)"*, so
//! `--cpu-used 9` could not encode a frame that reached one **at all**.
//!
//! ## The refusal's own reachability claim was false
//!
//! It read: *"REACHABILITY, MEASURED 2026-08-01: of 18 large cells probed at
//! speeds 8 and 9 (768² through 5472x3648), NONE reach a non-square leaf. The
//! only cell in the tree that does is issue #6's 12000x9000 at cpu9."*
//!
//! Playbook §9 in its purest form — a statement true of the cells that happened
//! to be probed, written as though it were general. It was already contradicted
//! twice before this landing (KB-28 found two 0.9 MP cells; the encoder hotspot
//! profile found 1024² cq44), and the sweep that replaced it
//! (`benchmarks/nonsquare_leaf_reach_2026-08-02.tsv`, 2,012 cells) found **627
//! reaching cells**, the smallest a **100x100** frame. Even inside the claim's
//! own stated range it is wrong at the very first size: 768² reaches at cq32
//! cpu9.
//!
//! ## The measured shape — **there is no size floor at all**
//!
//! The thing that predicts reaching is not size and not quality; it is whether
//! the frame has a **partial superblock**. `set_vt_partitioning` fits a
//! candidate by `mi_col + bs_width_check <= tile->mi_col_end`, and at the
//! frame's right/bottom edge (SB64 only) it relaxes the checks *asymmetrically*
//! — `bs_width_check` to `(block_width >> 1) + 1` but `bs_width_vert_check` to
//! `(block_width >> 2) + 1` (var_based_part.c:164-173). So at an edge node the
//! NONE candidate stops fitting while the VERT/HORZ pair still does, and a rect
//! is stamped. Whether the mi-aligned extent is a whole number of superblocks
//! is therefore the dominant variable, and `av1_select_sb_size`
//! (encoder_utils.c:958) decides which superblock: 64x64 at
//! `min(w, h) <= 480`, 64x64 again at allintra speed 9 below 4k, 128x128
//! otherwise.
//!
//! Measured over the sweep's 1,972 speed-8/9 rows
//! (`benchmarks/nonsquare_leaf_reach_2026-08-02.tsv`), classifying each row by
//! that predicate:
//!
//! | frame class | rows reaching a non-square leaf | smallest reaching |
//! |---|---|---|
//! | **partial superblock on either axis** | **609 / 884 = 68.9 %** | **100x100 = 10,000 px** |
//! | mi-aligned extent is a whole number of SBs | 18 / 1088 = 1.7 % | 589,824 px (768²) |
//!
//! So the refusal's "only 12000x9000" was wrong by **four orders of magnitude
//! in area** and by ~600 cells, and it was wrong in a way no size threshold
//! would have caught: **an ordinary 100x100 thumbnail reaches it, while 512²,
//! 1024² and 2176² mostly do not.** Any frame whose dimensions are not a whole
//! number of superblocks — 1920x1080, 1280x720, and essentially every crop
//! that is not a multiple of 64 — is in the reaching class.
//!
//! The 1.7 % SB-exact column is the second, rarer route: a genuine interior
//! variance win (the node over threshold with both halves under it), which
//! needs locally flat content and in this sweep occurred only on the
//! photographic source.
//!
//! All four shapes the KEY tree can stamp occur — BLOCK_8X16, 16X8, 16X32,
//! 32X16 — and no others, which is what `set_vt_partitioning` predicts:
//! `bsize > BLOCK_32X32` returns 0 on a key frame (:205-209) so 64X32/32X64
//! never appear, and `bsize == bsize_min` offers only NONE-or-split (:186-199)
//! so nothing below 8x8 does.
//!
//! *(A `min(w, h) >= 720` hypothesis — `force_large_partition_blocks_intra`,
//! speed_features.c:326-328, which is what KB-32 root #1 was about — fitted the
//! first sweep exactly and is FALSE: 1272x716 reaches 22 leaves at cq24 cpu9.
//! It is recorded here because it is the same mistake the refusal made, one
//! sweep later.)*
//!
//! **No RD speed reaches it.** `nonrd_pick_intra_mode` is dispatched from one
//! place, `pack_tile`'s `allintra && pick_cfg.speed >= 8` branch (pack.rs:1917);
//! [`rd_speeds_never_reach_the_estimate_arm`] measures that rather than
//! asserting it.
//!
//! ## What the port does now
//!
//! `nonrd_pick_intra_mode` runs C's real `av1_foreach_transformed_block_in_plane`
//! walk. Landing it exposed one more refusal immediately behind it —
//! `partition_pick.rs`'s `unimplemented!("frame-edge single-strip nonrd rect")`,
//! the exact claim KB-25 had already removed from the speed-7 walk in the same
//! words — which is fixed the same way (a poisoned slot-1 clone).
//!
//! Run:
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb34_nonsquare_nonrd_leaf \
//!     -- --ignored --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_encode::nonrd_pickmode::{multi_txb_leaf_counts, reset_multi_txb_leaf_counts};
use aom_sys_ref as c;

/// Mirror-tile (the `kb31_mandatory_tiles` / `kb32_nonrd_size_bands` /
/// `kb28_crop_dims` recipe — the same function, so rows here are directly
/// comparable with those gates').
fn mirror_tile(base: &EncodeCell, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
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
        label: format!("kb34_{w}x{h}_cq{cq}_s{speed}"),
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

/// Encode one cell both ways. Returns `(byte_identical, delta, per-bsize
/// multi-txb leaf counts)` and prints the row.
fn run(cell: &EncodeCell, tag: &str) -> (bool, i64, [u64; 22]) {
    let c_tu = cell.c_encode_defaults();
    assert!(
        !c_tu.is_empty(),
        "{}x{} cq{} s{}: C encode failed",
        cell.w,
        cell.h,
        cell.cq_level,
        cell.speed
    );
    let real = EncodeCell::frame_obu_payload(&c_tu);
    reset_multi_txb_leaf_counts();
    let ours = cell.port_encode_with(&c_tu, &ToggleKnobs::default());
    let counts = multi_txb_leaf_counts();
    let nsq: u64 = counts.iter().sum();
    let by: Vec<String> = counts
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 0)
        .map(|(b, n)| format!("{}:{n}", BSIZE_NAMES[b]))
        .collect();
    let d = ours.len() as i64 - real.len() as i64;
    println!(
        "  {:>4}x{:<4} cq{:<2} cpu{} {tag:<9} {:>8} vs {:>8} delta {:+} {}  nsq={nsq} [{}]",
        cell.w,
        cell.h,
        cell.cq_level,
        cell.speed,
        ours.len(),
        real.len(),
        d,
        if ours == real { "MATCH  " } else { "DIVERGE" },
        by.join(",")
    );
    (ours == real, d, counts)
}

/// BLOCK_SIZES_ALL names in the port's ordering (`MI_W`/`MI_H` in
/// `nonrd_pickmode.rs` carry the same order).
const BSIZE_NAMES: [&str; 22] = [
    "4X4", "4X8", "8X4", "8X8", "8X16", "16X8", "16X16", "16X32", "32X16", "32X32", "32X64",
    "64X32", "64X64", "64X128", "128X64", "128X128", "4X16", "16X4", "8X32", "32X8", "16X64",
    "64X16",
];

/// The four shapes `set_vt_partitioning` can stamp on a KEY frame — and the
/// only four the estimate arm may ever see as a multi-txb leaf. Anything else
/// appearing means the partitioner changed and this gate's reasoning is stale.
const KEY_RECT_SHAPES: [usize; 4] = [4, 5, 7, 8]; // 8X16, 16X8, 16X32, 32X16

/// **GATE — every newly-encodable cell is byte-identical to real aomenc.**
///
/// Cells, and why each one is here:
///
/// * `1272x724 cq24 cpu9` and `954x962 cq24 cpu9` — **KB-28's two pinned
///   rows.** `kb28_crop_dims::vbp_band_crop_dims_byte_match` carried them in
///   `NONRD_ESTIMATE_ARM_OPEN` as `Verdict::Panic`; they are the cells whose
///   discovery proved the refusal's "only at 108 MP" claim wrong at 0.9 MP.
///   Both encode and both are byte-exact here, and that pin is now empty;
/// * `1920x1080` at cpu 8 and 9 — the largest reaching size in the sweep on
///   in-repo content, and the one with the highest leaf counts (up to 66);
/// * `1280x720 diag` at cpu9 — the only in-repo content measured producing
///   **BLOCK_32X16**, so the gate covers a shape whose txbs sit side by side
///   rather than stacked (the `blk_col` half of the walk);
/// * `1024x1024 cq44 cpu-used 9` — **the encoder hotspot profile's own cell**
///   (`benchmarks/encoder_hotspot_profile_2026-08-02.md` "Two incidental
///   findings" #1, which reported it refusing). Its source there is an
///   out-of-repo photograph, so this row substitutes the one in-repo content
///   that reaches at that exact size/quantizer/speed: the mirror-tiled
///   `av1-1-b8-00-quantizer-58` decode. 1024² is SB-exact, so this is an
///   INTERIOR-rect cell — it needs the txb walk and NOT the frame-edge rect
///   constructor, which is the pair that separates the two roots in the bite
///   proof. Found by the `NSQ_VECTOR_SCAN=1` pass of
///   `examples/nonsquare_leaf_reach.rs` (10 quantizer decodes x 4 cq x 2
///   speeds; `-58` and `-40` are the two that reach, `-00`..`-30` never do —
///   the flat decodes are the ones with interior variance wins).
///
/// **Coverage, stated as a fraction rather than implied:** this grid reaches
/// **3 of the 4** shapes the KEY tree can stamp — BLOCK_8X16, 16X8 and 32X16.
/// The fourth, **BLOCK_16X32, is NOT covered by any in-repo content** here; the
/// sweep saw it 40 times, all on the out-of-repo photographic source
/// (768²/896²/954x962/1024²/1272x724, cpu9). It is the transpose of 32X16,
/// which IS covered, so the walk's `blk_row` and `blk_col` arms are both
/// exercised — but that is an argument, not a measurement, and it is the
/// honest gap in this gate.
///
/// MEASURED 2026-08-02 (aarch64-apple-darwin, `--profile test-fast`,
/// `av1-1-b8-00-quantizer-00` and `-58` mirror-tiled + `synthetic_diag`, bd8
/// 4:2:0); every cell PANICKED on the pristine tree with the message quoted at
/// the top of this file. Full sweep:
/// `benchmarks/nonsquare_leaf_reach_2026-08-02.tsv`; teeth and per-root bite
/// proof: `..._bite_2026-08-02.tsv`.
#[test]
#[ignore = "large-frame encode pairs at cpu 8/9; nightly / on-demand tier"]
fn nonsquare_leaf_cells_byte_match() {
    c::ref_init();
    let base = EncodeCell::real_content("kb34", "av1-1-b8-00-quantizer-00", None, 24, 9);
    // The heavily-quantized decode of the same clip — flat enough that
    // `set_vt_partitioning`'s pair arms win in the frame INTERIOR, which is the
    // only route on an SB-exact frame like 1024².
    let flat = EncodeCell::real_content("kb34f", "av1-1-b8-00-quantizer-58", None, 44, 9);
    println!("KB-34 — non-square nonrd estimate-arm leaves:");

    // (w, h, cq, speed, content) — 0 = mirror-tiled `-quantizer-00`,
    // 1 = `synthetic_diag`, 2 = mirror-tiled `-quantizer-58` (the flat decode).
    const CELLS: &[(usize, usize, i32, i32, u8)] = &[
        (1272, 724, 24, 9, 0),  // KB-28 pin #1
        (954, 962, 24, 9, 0),   // KB-28 pin #2
        (954, 962, 56, 8, 0),   // the same shape at speed 8
        (1920, 1080, 24, 8, 0), // 2 MP, cpu8
        (1920, 1080, 48, 9, 0), // 2 MP, cpu9 — the densest in-repo cell
        (1280, 720, 24, 9, 1),  // diag: the BLOCK_32X16 carrier
        (1280, 720, 48, 9, 1),
        (1024, 1024, 44, 9, 2), // the encoder hotspot profile's cell, SB-exact
    ];
    let mut total = [0u64; 22];
    let mut bad: Vec<String> = Vec::new();
    for &(w, h, cq, speed, content) in CELLS {
        let cell = match content {
            0 => mirror_tile(&base, w, h, cq, speed),
            1 => EncodeCell::synthetic_diag("kb34diag", w, h, cq, speed),
            _ => mirror_tile(&flat, w, h, cq, speed),
        };
        let tag = ["real", "diag", "flat"][content as usize];
        let (exact, d, counts) = run(&cell, tag);
        for (t, c) in total.iter_mut().zip(counts) {
            *t += c;
        }
        if !exact {
            bad.push(format!("{w}x{h} cq{cq} cpu{speed} {tag} {d:+}"));
        }
        assert!(
            counts.iter().sum::<u64>() > 0,
            "{w}x{h} cq{cq} cpu{speed} {tag} reaches NO non-square leaf — it is in this \
             gate because it did on 2026-08-02, so either the partitioner moved or the \
             counter stopped counting. A cell that does not reach the arm proves nothing \
             about it (playbook §1)."
        );
    }
    assert!(
        bad.is_empty(),
        "cells that used to REFUSE now encode, but not byte-identically: {bad:?}. \
         A frame that merely stops panicking is not a fix — drive to the first divergent \
         block (playbook §10), starting at nonrd_pick_intra_mode's txb walk: the \
         per-visit `av1_block_yrd` clamp is the TXB's num_4x4 plus the LEAF's \
         mb_to_*_edge, the skippable flag is the LAST txb's (assigned, not ANDed), and \
         each txb's prediction must land in the recon plane before the next one reads \
         its neighbours out of it."
    );

    // Shape coverage (playbook §8: derive coverage from artefacts, not names).
    let seen: Vec<usize> = (0..22).filter(|&b| total[b] > 0).collect();
    println!(
        "  shapes reached: {:?}",
        seen.iter().map(|&b| BSIZE_NAMES[b]).collect::<Vec<_>>()
    );
    for b in seen.iter() {
        assert!(
            KEY_RECT_SHAPES.contains(b),
            "the estimate arm coded a multi-txb leaf at BLOCK_{} — the KEY VBP tree can \
             only stamp {:?} (var_based_part.c:186-249), so either the partitioner \
             changed or a square leaf is taking the multi-txb path",
            BSIZE_NAMES[*b],
            KEY_RECT_SHAPES.map(|b| BSIZE_NAMES[b])
        );
    }
    // Self-promoting: 3 of the 4 KEY rect shapes on 2026-08-02, and if in-repo
    // content ever starts producing BLOCK_16X32 the count should be RAISED
    // rather than left at the floor.
    assert_eq!(
        seen.len(),
        3,
        "this grid covered exactly 3 of the 4 KEY rect shapes (8X16, 16X8, 32X16) on          2026-08-02 and now covers {:?}. FEWER is a regression or a partitioner move;          MORE means in-repo content started reaching BLOCK_16X32 — re-pin to 4 and          delete the honest-gap paragraph in this test's doc comment",
        seen.iter().map(|&b| BSIZE_NAMES[b]).collect::<Vec<_>>()
    );
}

/// **The shape, as a contrast, on cells small enough for the default tier.**
///
/// `100x100` and `196x196` have partial superblocks (their mi-aligned extents,
/// 104 and 200 px, are not multiples of the 64 px superblock these frames use);
/// `128x128` and `250x250` do not (250 rounds UP to a mi extent of 256 px,
/// which is exactly 4 superblocks — so it is SB-exact despite not looking it,
/// and that is why it is here rather than a rounder number). Same content, same
/// quantizer, same speed, sizes within 2.6x of each other: the partial ones
/// reach non-square leaves and the exact ones reach none.
///
/// Two things this pins that a size threshold cannot:
///
/// * **the smallest reaching frame in the whole sweep is 100x100 = 10,000 px**,
///   which is what makes "the only cell in the tree that reaches this is
///   12000x9000" a four-orders-of-magnitude error rather than a near miss;
/// * the counter can read ZERO. Without that, every non-zero reading elsewhere
///   in this file is compatible with a stuck instrument (playbook §2).
///
/// MEASURED 2026-08-02: 100x100 cq24 cpu9 = 1 leaf, 196x196 = 5, 128x128 = 0,
/// 250x250 = 0; all four byte-identical.
#[test]
fn partial_superblocks_are_what_reaches_it_not_frame_size() {
    c::ref_init();
    let base = EncodeCell::real_content("kb34s", "av1-1-b8-00-quantizer-00", None, 24, 9);
    println!("KB-34 — partial-SB vs SB-exact, at 10k..62k px:");
    // (w, h, mi-aligned px, superblock px, expect_reaching)
    const GRID: &[(usize, usize, usize, usize, bool)] = &[
        (100, 100, 104, 64, true),
        (196, 196, 200, 64, true),
        (128, 128, 128, 64, false),
        (250, 250, 256, 64, false),
    ];
    for &(w, h, mi_px, sb, want) in GRID {
        let (exact, d, counts) = run(
            &mirror_tile(&base, w, h, 24, 9),
            if want { "partial" } else { "sb-exact" },
        );
        assert!(exact, "{w}x{h} cq24 cpu9: {d:+} B");
        let n: u64 = counts.iter().sum();
        assert_eq!(
            mi_px % sb != 0,
            want,
            "{w}x{h}: this row's own arithmetic is wrong — mi-aligned {mi_px} px vs \
             {sb} px superblocks"
        );
        assert_eq!(
            n > 0,
            want,
            "{w}x{h} (mi-aligned {mi_px} px, {sb} px superblocks, partial={}) reached \
             {n} non-square leaves. The predicate that governs this is \
             `set_vt_partitioning`'s frame-edge fit-check relaxation \
             (var_based_part.c:164-173), not the frame's size — if this row flipped, \
             re-derive it before trusting anything else in this file",
            mi_px % sb != 0
        );
    }
}

/// **`RD_BAND_OPEN` — a divergence class this sweep found, which is NOT this
/// landing's and is pinned exactly rather than smoothed over.**
///
/// 1272x724 cq24 (mi-aligned 1272x728, so partial superblocks on both axes)
/// diverges at `--cpu-used` 2, 3, 4 and 5 by -14 / -104 / -167 / -189 B, and
/// matches at 0, 1, 6, 7, 8 and 9. **Measured identical with this landing's two
/// hunks stashed** (`benchmarks/nonsquare_leaf_reach_bite_2026-08-02.tsv`,
/// the `rd` rows of the before and after arms are byte-for-byte the same), and
/// it cannot be this landing's by construction either: the counter reads 0 at
/// every one of those speeds, i.e. `av1_nonrd_pick_intra_mode` is never
/// entered, and `nonrd_use_partition_real` is dispatched only at speed >= 8.
///
/// It is adjacent to KB-28's `rd_band_min_dim_tiers_byte_match` (474x480 and
/// 714x720 at cpu 1..6, 12/12) but at a size neither that gate nor
/// `hd_speed_axis_byte_matches` covers — a new cell for an old band. Pinned
/// self-promoting in both directions.
const RD_BAND_OPEN: &[(i32, i64)] = &[(2, -14), (3, -104), (4, -167), (5, -189)];

/// **No RD speed reaches the estimate arm** — measured, not argued.
/// `nonrd_pick_intra_mode` has exactly one dispatch site, `pack_tile`'s
/// `allintra && pick_cfg.speed >= 8` branch (pack.rs:1917), so speeds 0..7 pick
/// every leaf — square or rect — through the full RD search, which has coded
/// non-square blocks since speed 1. The asymmetry is the point (playbook §1):
/// the same frame at speed 9 must reach it, or "speeds 0-7 reach zero" is a
/// statement about the frame rather than about the speeds.
#[test]
#[ignore = "9 encode pairs at ~0.9 MP, one per speed; on-demand tier"]
fn rd_speeds_never_reach_the_estimate_arm() {
    c::ref_init();
    let base = EncodeCell::real_content("kb34r", "av1-1-b8-00-quantizer-00", None, 24, 9);
    println!("KB-34 — speed axis (1272x724 cq24, the KB-28 cell):");
    let mut observed: Vec<(i32, i64)> = Vec::new();
    for speed in 0..=7 {
        let (exact, d, counts) = run(&mirror_tile(&base, 1272, 724, 24, speed), "rd");
        if !exact {
            observed.push((speed, d));
        }
        assert_eq!(
            counts.iter().sum::<u64>(),
            0,
            "cpu-used {speed} entered av1_nonrd_pick_intra_mode. It is dispatched only \
             from pack_tile's `allintra && speed >= 8` branch; if that moved, every \
             speed-{speed} byte gate is now exercising a different search"
        );
    }
    assert_eq!(
        observed,
        RD_BAND_OPEN.to_vec(),
        "the RD band's open set moved. FEWER/smaller entries => something closed it: \
         re-pin RD_BAND_OPEN and say which KB. MORE => a regression, and note this \
         cell reaches ZERO non-square leaves at every RD speed, so it is not the nonrd \
         estimate arm's"
    );
    let (exact, d, counts) = run(&mirror_tile(&base, 1272, 724, 24, 9), "nonrd");
    assert!(exact, "1272x724 cq24 cpu9: {d:+} B");
    assert!(
        counts.iter().sum::<u64>() > 0,
        "the speed-9 control reached no non-square leaf, so the eight zeroes above are \
         a property of this frame, not of the RD speeds"
    );
}
