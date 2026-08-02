//! **KB-PERF-1 — the per-64x64 intra-CNN cache returns what a recomputation
//! returns, on a real encode.**
//!
//! `intra_mode_cnn_partition` (`partition_strategy.c:160`) runs the 5-layer CNN
//! cascade only at a `BLOCK_64X64` node and only when
//! `!part_info->cnn_output_valid`; every 32x32 / 16x16 / 8x8 node inside that
//! 64x64 reads `part_info->cnn_buffer`. The port used to re-run the cascade at
//! every node — 2558 times per 1 MP frame against libaom's 256 — which is
//! **74.7 % of the port's entire encode**
//! (`benchmarks/encoder_hotspot_profile_2026-08-02.md`).
//!
//! The correctness claim for caching is that all ~10 runs per superblock
//! convolve the identical window, because `extract_intra_cnn_window` snaps its
//! origin to the containing 64x64. This test does not take that on argument:
//! with `set_cnn_cache_verify` armed, **every cache read re-extracts its window,
//! re-runs the cascade and asserts bit-identity with what was cached**, and the
//! resulting frame is byte-compared against real `aomenc`.
//!
//! It also pins the two counts that make the latch *structural* rather than
//! incidental:
//!
//! * cascades computed == the number of whole-in-frame 64x64 nodes — one each,
//!   never two;
//! * on a PARTIAL-superblock frame (196x196 -> mi 50x50) that count is **9, not
//!   16**: the seven frame-edge 64x64 roots never reach the compute branch, the
//!   latch stays 0, and every smaller node inside them returns early at
//!   `partition_strategy.c:227` — pruning nothing. That is **KB-23's** result,
//!   now produced by the cache instead of by a separate `cnn_root_whole_in_frame`
//!   predicate, which is why it is asserted here.
//!
//! The check is off in every shipping build (`AtomicBool`, default false); this
//! is its only consumer besides `examples/eprof_cnn_verify.rs`. It lives in its
//! own test binary because the toggle is process-wide.
//!
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test cnn_cache_identity -- --nocapture
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_encode::cnn_partition::decision as cnn_dec;
use aom_sys_ref as c;
use aom_sys_ref::cx_ctrl::{AOM_SUPERBLOCK_SIZE_128X128, AV1E_SET_SUPERBLOCK_SIZE};

/// Mirror-tile a base cell up/down to `w x h` (same recipe as
/// `kb28_crop_dims::mirror_tile` / `s4cov_hd_speed_axis::mirror_tile`).
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

/// The number of `BLOCK_64X64` nodes that are WHOLE-in-frame, i.e. the number of
/// cascades C computes. `av1_get_MBs` aligns the mi grid up to 8 px
/// (alloccommon.c:30), and a 64x64 node at mi origin `r` is whole iff
/// `r + 16 <= mi_rows`.
fn whole_64x64_nodes(w: usize, h: usize) -> u64 {
    let mi = |px: usize| ((px + 7) & !7) / 4;
    let (mr, mc) = (mi(h), mi(w));
    let n = |m: usize| (0..m).step_by(16).filter(|r| r + 16 <= m).count() as u64;
    n(mr) * n(mc)
}

#[test]
fn cache_reads_equal_a_recomputation_and_the_frame_is_byte_exact() {
    c::ref_init();
    let base = EncodeCell::real_content("cnncache", "av1-1-b8-00-quantizer-00", None, 24, 0);

    let sb128 = [(AV1E_SET_SUPERBLOCK_SIZE, AOM_SUPERBLOCK_SIZE_128X128)];
    // (w, h, speed, sb128, why)
    let cells: &[(usize, usize, i32, bool, &str)] = &[
        (256, 256, 2, false, "SB-EXACT: every 64x64 whole in frame"),
        (
            196,
            196,
            2,
            false,
            "PARTIAL-SB (KB-23's shape): 9 of 16 64x64 roots whole in frame",
        ),
        (
            256,
            256,
            6,
            false,
            "speed 6 — the mode the KB-PERF-1 profile measured",
        ),
        // THE cell that makes this test able to fail on the invalidation
        // (playbook §1). Under SB64 the per-superblock `PartitionSearchInfo`
        // is itself a per-64x64 reset (one 64x64 per SB), so deleting
        // `invalidate_cnn()` is inert there — every SB64 row above stays green.
        // Under `--sb-size=128` one superblock holds FOUR 64x64 roots, so the
        // reset is the only thing separating them: without it this row's
        // computes fall 16 -> 4, the verify assert fires on the first read of
        // a sibling 64x64's cascade, and the frame diverges. Measured both
        // ways — see the bite proof in
        // `benchmarks/encoder_cnn_cache_2026-08-02.md`.
        (
            256,
            256,
            2,
            true,
            "--sb-size=128: four 64x64 roots per superblock",
        ),
    ];

    let mut armed = 0usize;
    for &(w, h, speed, use_sb128, why) in cells {
        let tag = if use_sb128 { "_sb128" } else { "" };
        let cell = mirror_tile(
            &base,
            &format!("cnncache_{w}x{h}_s{speed}{tag}"),
            w,
            h,
            24,
            speed,
        );
        let c_tu = if use_sb128 {
            cell.c_encode_ctrls(&sb128)
        } else {
            cell.c_encode()
        };
        let real = EncodeCell::frame_obu_payload(&c_tu);

        cnn_dec::reset_cnn_cache_stats();
        cnn_dec::set_cnn_cache_verify(true);
        // Every cache read inside this call re-runs the cascade and asserts
        // bit-identity; a mismatch panics here rather than returning bad bytes.
        let got = cell.port_encode_with(&c_tu, &ToggleKnobs::default());
        cnn_dec::set_cnn_cache_verify(false);
        let (computes, reads) = cnn_dec::cnn_cache_stats();

        println!(
            "  {w}x{h} cq24 cpu{speed}{tag}: computes {computes} reads {reads} \
             bytes {} vs {} — {why}",
            got.len(),
            real.len()
        );

        assert_eq!(
            got,
            real,
            "{w}x{h} cpu{speed}{tag} diverged from real aomenc ({} vs {} bytes)",
            got.len(),
            real.len()
        );
        // Non-vacuity (playbook §2): if nothing READ the cache, the identity
        // above was verified over an empty set and proves nothing.
        assert!(
            reads > 0,
            "{w}x{h} cpu{speed}{tag}: the cache was never read — this cell does not \
             reach the intra-CNN prune, so it cannot gate it"
        );
        // The latch, structurally: one cascade per whole-in-frame 64x64, and
        // none for the frame-edge roots (KB-23, now emergent from the cache).
        assert_eq!(
            computes,
            whole_64x64_nodes(w, h),
            "{w}x{h} cpu{speed}{tag}: expected exactly one cascade per whole-in-frame \
             64x64 node"
        );
        armed += 1;
    }
    assert_eq!(armed, cells.len());
}
