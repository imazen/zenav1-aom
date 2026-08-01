//! KB-19 — the `is_4k_or_larger` arm of
//! `set_allintra_speed_feature_framesize_dependent` (speed_features.c:187-189),
//! and KB-22 — the `speed == 0` >=720p arm of
//! `av1_set_speed_features_qindex_dependent` (speed_features.c:2914).
//!
//! libaom raises `part_sf.default_min_partition_size` to `BLOCK_8X8` for any
//! frame with `AOMMIN(cm->width, cm->height) >= 2160`, at every speed. The port
//! left the field at `BLOCK_4X4` for every frame size, so above 2160p its
//! partition search could descend a level deeper than C's (KB-19). Landing that
//! arm left a 150-byte residual, which was the SECOND unmodelled framesize arm:
//! at speed 0, `min(w, h) >= 720` and `base_qindex <= 128` raise
//! `rd_sf.perform_coeff_opt` to `2 + is_1080p_or_larger` and
//! `tx_sf.intra_tx_size_search_init_depth_rect` to 1 (KB-22).
//!
//! Both derivations are gated cheaply and in the default tier by
//! `aom_encode::speed_features`'s `framesize_dependent_min_partition_size_4k_arm`
//! and `qindex_dependent_speed0_hd_arm` unit tests. THIS file is the end-to-end
//! companion: one real encode at 2160x2160 compared byte-for-byte against real
//! aomenc, with a decode-both localizer on the failure path.
//!
//! It is `#[ignore]`d because a 2160x2160 speed-0 cell costs minutes per
//! encode pair — it belongs in a nightly / on-demand tier, not in the default
//! one. Run it with:
//!
//! ```text
//! cargo test --profile test-fast -p zenav1-aom-bench --test kb19_min_partition_4k -- --ignored --nocapture
//! ```

use aom_bench::EncodeCell;
use aom_dsp::entropy::obu::read_obu_header;
use aom_dsp::entropy::partition::get_partition_subsize;
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


/// 2160x2160 bd8 4:2:0 real content, speed-0 ALLINTRA KEY, stock knobs —
/// a HARD byte-identity gate vs real aomenc.
///
/// 2160x2160 (not 3840x2160) because the `is_4k_or_larger` predicate is on the
/// SHORT side — `AOMMIN(w, h) >= 2160` — and a square frame is the cheapest
/// shape that satisfies it. cq32 (base_qindex 128) additionally sits exactly on
/// the KB-22 arm's `base_qindex <= 128` boundary.
///
/// **MEASURED, reference box aarch64-apple-darwin, `--profile test-fast`,
/// mirror-tiled `av1-1-b8-00-quantizer-00` at cq32** (C reference: 431,724 B,
/// ~26 s):
///
/// | port build | port bytes | delta vs C | port wall |
/// |---|---|---|---|
/// | 2026-07-30, no `is_4k_or_larger` arm | 440,347 | **+8,623 (+2.00%)** | — |
/// | 2026-07-30, KB-19 arm only | 431,574 | **-150 (-0.035%)** | 125 s |
/// | 2026-07-31, + the KB-22 qindex arm | 431,724 | **0 — BYTE IDENTICAL** | 83 s |
///
/// So both arms are load-bearing at this frame size and neither is a paper fix:
/// KB-19's closes 98.3% of the byte gap, KB-22's closes the rest (and takes 34%
/// off the port's wall time, because a higher `perform_coeff_opt` row and a
/// deeper rectangular tx-size init depth both cut search work).
///
/// On a mismatch this gate does NOT just report a byte count: it decodes both
/// streams with the (bit-exact vs C) decoder and prints the first divergent
/// partition node, then the first divergent leaf record, then the first
/// divergent reconstruction pixel — the KB-6 recipe
/// (`aom-encode/tests/decode_diff_multisb.rs`), which is how KB-22 was
/// localized to node 1 of SB(0,0).
#[test]
#[ignore = "2160x2160 speed-0 encode pair costs minutes; nightly / on-demand tier"]
fn min_partition_4k_arm_e2e_byte_match() {
    /// Real aomenc's payload size for the same cell. Pinned so that an oracle
    /// or cell-recipe change is reported as such, instead of silently
    /// re-baselining the comparison below.
    const PINNED_C_LEN: usize = 431_724;

    c::ref_init();
    let base = EncodeCell::real_content("kb19base", "av1-1-b8-00-quantizer-00", None, 32, 0);
    let cell = mirror_tile(&base, "kb19_2160sq", 2160, 2160, 32);
    assert_eq!((cell.w, cell.h), (2160, 2160));
    assert!(cell.w.min(cell.h) >= 2160, "the cell must reach the is_4k_or_larger arm");
    // KB-22 reach: that arm additionally needs `base_qindex <= 128`.
    // `av1_quantizer_to_qindex` (av1_quantize.c:1033) maps --cq-level 32 -> 128,
    // i.e. this cell sits exactly ON the boundary.
    assert_eq!(cell.cq_level, 32, "the cell must reach the KB-22 base_qindex <= 128 arm");

    let t0 = std::time::Instant::now();
    let tu = cell.c_encode();
    let c_ms = t0.elapsed().as_millis();
    let real = EncodeCell::frame_obu_payload(&tu);

    let t1 = std::time::Instant::now();
    let port = cell.port_encode(&tu);
    let port_ms = t1.elapsed().as_millis();

    println!(
        "  kb19/kb22 2160x2160 cq32 speed0: port {} B ({port_ms} ms) vs C {} B ({c_ms} ms) -> {}",
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
    if port != real {
        localize(&cell, &tu, &real, &port);
    }
    // `assert!` rather than `assert_eq!`: these are ~432 KB vectors and the
    // localization above is the diagnosis, not a 432 KB left/right dump.
    assert!(
        port == real,
        "the >=2160p cell is no longer byte-identical to real aomenc (port {} B vs C {} B). \
         The localization above names the first divergent decision. A ~+8.6 KB regression \
         means the KB-19 `is_4k_or_larger` arm broke \
         (`SpeedFeatures::apply_allintra_framesize_dependent`, speed_features.c:187-189, \
         plus the `min_partition_bsize` AOMMAX, partition_strategy.h:224-226); a ~-150 B \
         regression means the KB-22 arm broke \
         (`SpeedFeatures::apply_allintra_qindex_dependent`, speed_features.c:2914).",
        port.len(),
        real.len()
    );
}

/// Decode BOTH streams and print, in order: the first divergent partition node,
/// the first divergent leaf record, the first divergent reconstruction pixel.
/// Called only when the byte gate above has already failed.
fn localize(cell: &EncodeCell, tu: &[u8], real_payload: &[u8], port_payload: &[u8]) {
    let port_stream = rewrap_frame_obu(tu, port_payload);
    let (t_real, _, _) = aom_decode::frame::decode_frame_obus_prefilter(tu)
        .unwrap_or_else(|e| panic!("decode of REAL aomenc bytes failed: {e}"));
    let (t_ours, _, _) = aom_decode::frame::decode_frame_obus_prefilter(&port_stream)
        .unwrap_or_else(|e| panic!("decode of the port's own rewrapped bytes failed: {e}"));

    let (mi_rows, mi_cols) = (mi_dim(cell.h as i32), mi_dim(cell.w as i32));
    println!(
        "  LOCALIZE: real {} B tree={} blocks={} | port {} B tree={} blocks={} | mi {mi_rows}x{mi_cols}",
        real_payload.len(),
        t_real.tree.len(),
        t_real.blocks.len(),
        port_payload.len(),
        t_ours.tree.len(),
        t_ours.blocks.len(),
    );

    let mut real_seq = Vec::new();
    let mut ours_seq = Vec::new();
    replay_tree(&t_real.tree, &mut 0, 0, 0, SB, mi_rows, mi_cols, &mut real_seq);
    replay_tree(&t_ours.tree, &mut 0, 0, 0, SB, mi_rows, mi_cols, &mut ours_seq);

    let mut first_div = None;
    for (i, (r, o)) in real_seq.iter().zip(ours_seq.iter()).enumerate() {
        assert_eq!(
            (r.0, r.1, r.2),
            (o.0, o.1, o.2),
            "positions must stay locked until the first partition divergence (node {i})"
        );
        if r.3 != o.3 {
            first_div = Some(*r);
            println!(
                ">>> FIRST PARTITION DIVERGENCE at node {i} (mi_row={}, mi_col={}, bsize={}) \
                 SB(mi {},{}): real=PARTITION_{} port=PARTITION_{}",
                r.0,
                r.1,
                r.2,
                (r.0 / SB_MI) * SB_MI,
                (r.1 / SB_MI) * SB_MI,
                PARTITION_NAMES[r.3 as usize],
                PARTITION_NAMES[o.3 as usize]
            );
            break;
        }
    }

    if first_div.is_none() {
        println!(
            "  partition trees agree on the shared prefix (real={} port={} nodes); scanning leaves",
            real_seq.len(),
            ours_seq.len()
        );
        let mut found = false;
        for rb in &t_real.blocks {
            if let Some(ob) = t_ours
                .blocks
                .iter()
                .find(|b| b.mi_row == rb.mi_row && b.mi_col == rb.mi_col)
            {
                let modes_differ = ob.bsize != rb.bsize
                    || ob.partition != rb.partition
                    || ob.info.y_mode != rb.info.y_mode
                    || ob.info.angle_delta_y != rb.info.angle_delta_y
                    || ob.info.use_filter_intra != rb.info.use_filter_intra
                    || ob.tx_size != rb.tx_size
                    || ob.info.uv_mode != rb.info.uv_mode;
                let txbs_differ = ob.txbs != rb.txbs || ob.txbs_uv != rb.txbs_uv;
                if modes_differ || txbs_differ {
                    println!(
                        ">>> FIRST LEAF MISMATCH at (mi_row={}, mi_col={}) SB(mi {},{}) \
                         [modes_differ={modes_differ} txbs_differ={txbs_differ}]\n     \
                         real bsize={} part={} y_mode={} adly={} use_fi={} tx_size={} uv_mode={} txbs={:?} txbs_uv={:?}\n     \
                         port bsize={} part={} y_mode={} adly={} use_fi={} tx_size={} uv_mode={} txbs={:?} txbs_uv={:?}",
                        rb.mi_row, rb.mi_col,
                        (rb.mi_row / SB_MI) * SB_MI, (rb.mi_col / SB_MI) * SB_MI,
                        rb.bsize, rb.partition, rb.info.y_mode, rb.info.angle_delta_y,
                        rb.info.use_filter_intra, rb.tx_size, rb.info.uv_mode, rb.txbs, rb.txbs_uv,
                        ob.bsize, ob.partition, ob.info.y_mode, ob.info.angle_delta_y,
                        ob.info.use_filter_intra, ob.tx_size, ob.info.uv_mode, ob.txbs, ob.txbs_uv,
                    );
                    found = true;
                    break;
                }
            }
        }
        if !found {
            println!(
                "  no partition/leaf-field/txb divergence — the byte delta is in coefficient \
                 VALUES for identical modes+tx_type"
            );
        }
    }

    let mut recon_div = None;
    'rec: for row in 0..t_real.height.min(t_ours.height) {
        for col in 0..t_real.width.min(t_ours.width) {
            let rv = t_real.recon.px(row * t_real.stride + col);
            let ov = t_ours.recon.px(row * t_ours.stride + col);
            if rv != ov {
                recon_div = Some((row, col, rv, ov));
                break 'rec;
            }
        }
    }
    match recon_div {
        Some((row, col, rv, ov)) => println!(
            ">>> FIRST RECON PIXEL DIVERGENCE at luma (row={row}, col={col}) -> SB(mi {},{}): \
             real={rv} port={ov}",
            (row / 64) * 16,
            (col / 64) * 16
        ),
        None => println!("  reconstruction planes are IDENTICAL"),
    }
}

// ---------------------------------------------------------------------------
// Decode-both localizer, run on the byte gate's FAILURE path (playbook §10:
// diagnose to the decision, not to the byte count). This is what localized
// KB-22 to SB(0,0)'s first 32x32 node.
// ---------------------------------------------------------------------------

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_FRAME: u32 = 6;
/// `mi` extent of each `BLOCK_*` ordinal (`mi_size_wide[]`, blockd.h).
const MI_SIZE_WIDE_B: [usize; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const PARTITION_NAMES: [&str; 10] = [
    "NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B", "VERT_A", "VERT_B", "HORZ_4", "VERT_4",
];
const SB: usize = 12; // BLOCK_64X64
const SB_MI: i32 = 16; // 64 px / 4

fn mi_dim(px: i32) -> i32 {
    ((px + 7) & !7) >> 2
}

/// Splice `payload` in as the stream's `OBU_FRAME` payload, leaving every other
/// OBU (notably the sequence header) byte-verbatim. This is how the port's
/// frame-OBU payload — the byte-compared unit — becomes a decodable stream.
fn rewrap_frame_obu(stream: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(stream.len() + payload.len());
    let mut pos = 0usize;
    let mut spliced = false;
    while pos < stream.len() {
        let h = read_obu_header(&stream[pos..]).expect("valid OBU header");
        let after_header = pos + h.header_len;
        let (size, size_bytes) = aom_dsp::entropy::leb128::uleb_decode(&stream[after_header..])
            .expect("valid leb128 size");
        let end = after_header + size_bytes + size as usize;
        if h.obu_type == OBU_FRAME {
            out.extend_from_slice(&stream[pos..after_header]);
            out.extend_from_slice(
                &aom_dsp::entropy::leb128::uleb_encode(payload.len() as u64, 8)
                    .expect("leb128 encode"),
            );
            out.extend_from_slice(payload);
            spliced = true;
        } else {
            out.extend_from_slice(&stream[pos..end]);
        }
        pos = end;
    }
    assert!(spliced, "no OBU_FRAME to splice into");
    assert!(
        out.len() > payload.len(),
        "rewrap dropped the sequence header — the port stream must carry the \
         reference seq header verbatim (OBU type {OBU_SEQUENCE_HEADER})"
    );
    out
}

/// Replay the decoder's pre-order partition sequence into
/// `(mi_row, mi_col, bsize, partition)` records — the same walk
/// `decode_diff_multisb.rs::replay_tree` uses, so positions stay locked across
/// the two streams until the first partition-VALUE divergence.
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
    if mi_row >= mi_rows || mi_col >= mi_cols || *cursor >= tree.len() {
        return;
    }
    let p = tree[*cursor];
    out.push((mi_row, mi_col, bsize, p));
    *cursor += 1;
    if p as usize == 3 {
        let hbs = (MI_SIZE_WIDE_B[bsize] / 2) as i32;
        let subsize = get_partition_subsize(bsize, p as i32) as usize;
        for (dr, dc) in [(0, 0), (0, hbs), (hbs, 0), (hbs, hbs)] {
            replay_tree(
                tree,
                cursor,
                mi_row + dr,
                mi_col + dc,
                subsize,
                mi_rows,
                mi_cols,
                out,
            );
        }
    }
}
