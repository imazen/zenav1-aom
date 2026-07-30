//! KB-19 — the `is_4k_or_larger` arm of
//! `set_allintra_speed_feature_framesize_dependent` (speed_features.c:187-189).
//!
//! libaom raises `part_sf.default_min_partition_size` to `BLOCK_8X8` for any
//! frame with `AOMMIN(cm->width, cm->height) >= 2160`, at every speed. The port
//! left the field at `BLOCK_4X4` for every frame size, so above 2160p its
//! partition search could descend a level deeper than C's.
//!
//! The derivation itself is gated cheaply and in the default tier by
//! `aom_encode::speed_features`'s
//! `framesize_dependent_min_partition_size_4k_arm` unit test. THIS file is the
//! end-to-end companion: one real encode at 2160x2160 compared byte-for-byte
//! against real aomenc. It measures the arm as worth **8,623 bytes (2.00%)**
//! on that cell — and also pins the 150-byte residual that survives it
//! (**KB-22**, a separate open >=2160p divergence). See the test's own doc
//! comment for the measured table.
//!
//! It is `#[ignore]`d because a 2160x2160 speed-0 cell costs minutes per
//! encode pair — it belongs in a nightly / on-demand tier, not in the default
//! one. Run it with:
//!
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb19_min_partition_4k -- --ignored --nocapture
//! ```

use aom_bench::EncodeCell;
use aom_sys_ref as c;

/// Mirror-tile a decoded cell up to `w x h`. Mirroring (rather than wrapping)
/// keeps the seam continuous, so the enlarged frame stays photographic content
/// instead of acquiring a synthetic edge grid every tile period. Same recipe as
/// the size axis of the config-permutation gate.
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
    EncodeCell {
        label: label.to_string(),
        w,
        h,
        mono: base.mono,
        ss_x: base.ss_x,
        ss_y: base.ss_y,
        usage: base.usage,
        cq_level: cq,
        speed: base.speed,
        bd: base.bd,
        y,
        u,
        v,
    }
}

/// 2160x2160 bd8 4:2:0 real content, speed-0 ALLINTRA KEY, stock knobs.
///
/// 2160x2160 (not 3840x2160) because the predicate is on the SHORT side —
/// `AOMMIN(w, h) >= 2160` — and a square frame is the cheapest shape that
/// satisfies it.
///
/// **MEASURED 2026-07-30 (KB-19 landing), reference box aarch64-apple-darwin,
/// `--profile test-fast`, mirror-tiled `av1-1-b8-00-quantizer-00` at cq32:**
///
/// | port build | port bytes | C bytes | delta |
/// |---|---|---|---|
/// | WITHOUT the `is_4k_or_larger` arm | 440,347 | 431,724 | **+8,623 (+2.00%)** |
/// | WITH it (shipped) | 431,574 | 431,724 | **-150 (-0.035%)** |
///
/// So the arm is heavily load-bearing at this frame size — it closes 98.3% of
/// the byte gap — and that is the end-to-end bite proof for KB-19. Encode
/// wall on that box: C ~26 s, port ~195 s.
///
/// **It does NOT close the cell.** The remaining 150 bytes are a SEPARATE,
/// still-open >=2160p divergence (KB-22), which is why this test pins DIVERGE
/// rather than asserting byte-identity: pinning the true state is the repo's
/// pattern for an open finding (`mono_vector_open_divergences_pinned`,
/// `size_axis_open_divergences_pinned`). It is self-promoting in both
/// directions — it fails if the cell starts matching (KB-22 closed: promote
/// this to a hard byte gate) and it fails if the port's byte count moves away
/// from the pinned value (a regression, most likely of the arm itself, whose
/// absence costs +8.6 KB).
#[test]
#[ignore = "2160x2160 speed-0 encode pair costs minutes; nightly / on-demand tier"]
fn min_partition_4k_arm_e2e_pinned() {
    /// Port payload size measured with the shipped KB-19 arm (see the table
    /// in this test's doc comment).
    const PINNED_PORT_LEN: usize = 431_574;
    /// Real aomenc's payload size for the same cell.
    const PINNED_C_LEN: usize = 431_724;

    c::ref_init();
    let base = EncodeCell::real_content("kb19base", "av1-1-b8-00-quantizer-00", None, 32, 0);
    let cell = mirror_tile(&base, "kb19_2160sq", 2160, 2160, 32);
    assert_eq!((cell.w, cell.h), (2160, 2160));
    assert!(cell.w.min(cell.h) >= 2160, "the cell must reach the is_4k_or_larger arm");

    let t0 = std::time::Instant::now();
    let tu = cell.c_encode();
    let c_ms = t0.elapsed().as_millis();
    let real = EncodeCell::frame_obu_payload(&tu);

    let t1 = std::time::Instant::now();
    let port = cell.port_encode(&tu);
    let port_ms = t1.elapsed().as_millis();

    println!(
        "  kb19 2160x2160 cq32 speed0: port {} B ({port_ms} ms) vs C {} B ({c_ms} ms) -> {}",
        port.len(),
        real.len(),
        if port == real { "MATCH" } else { "DIVERGE" }
    );
    assert_eq!(
        real.len(),
        PINNED_C_LEN,
        "the C reference itself moved — the oracle build or the cell recipe changed, \
         so nothing below is comparable to the pinned measurement"
    );
    assert_ne!(
        port, real,
        "KB-22 HAS CLOSED: the 2160x2160 cell is now byte-identical to real \
         aomenc. Promote this test to a hard `assert_eq!` byte gate and close \
         KB-22 in CLAUDE.md."
    );
    assert_eq!(
        port.len(),
        PINNED_PORT_LEN,
        "the >=2160p port output MOVED (pinned {PINNED_PORT_LEN} B, C is \
         {PINNED_C_LEN} B). If it grew by roughly 8.6 KB the KB-19 \
         `is_4k_or_larger` arm regressed \
         (`SpeedFeatures::apply_allintra_framesize_dependent` must set \
         default_min_partition_size = BLOCK_8X8 at min(w,h) >= 2160, \
         speed_features.c:187-189, and `min_partition_bsize` must AOMMAX it \
         with the CLI floor, partition_strategy.h:224-226). Any other move is \
         a change in the open KB-22 residual — re-measure and re-pin."
    );
}
