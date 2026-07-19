//! Byte-identity of `KfFrameContext::default_for_qindex` (the generated
//! default-CDF tables + arena packing) against the COMPILED C defaults: the
//! real `av1_setup_past_independence` over a minimal AV1_COMMON, dumped flat
//! by `shim_dump_default_kf_fc` in KfFrameContext field order. Covers all four
//! `TOKEN_CDF_Q_CTXS` coefficient bands and both sides of each band boundary
//! (`get_q_ctx`: <=20 / <=60 / <=120 / >120).

use aom_dsp::entropy::partition::{coeff_cdf_q_ctx, KfFrameContext};
use aom_sys_ref as c;

/// Flatten a KfFrameContext in the dump's field order (see dec_shim.c).
fn flatten(fc: &KfFrameContext) -> Vec<u16> {
    let mut v = Vec::with_capacity(c::DUMP_KF_FC_LEN);
    for a in &fc.kf_y {
        for b in a {
            v.extend_from_slice(b);
        }
    }
    for a in &fc.uv_mode {
        for b in a {
            v.extend_from_slice(b);
        }
    }
    for a in &fc.angle_delta {
        v.extend_from_slice(a);
    }
    for a in &fc.skip {
        v.extend_from_slice(a);
    }
    for a in &fc.seg_spatial {
        v.extend_from_slice(a);
    }
    for a in &fc.partition {
        v.extend_from_slice(a);
    }
    for a in &fc.palette_y_mode {
        for b in a {
            v.extend_from_slice(b);
        }
    }
    for a in &fc.palette_uv_mode {
        v.extend_from_slice(a);
    }
    for a in &fc.palette_y_size {
        v.extend_from_slice(a);
    }
    for a in &fc.palette_uv_size {
        v.extend_from_slice(a);
    }
    for a in &fc.palette_y_color_index {
        for b in a {
            v.extend_from_slice(b);
        }
    }
    for a in &fc.palette_uv_color_index {
        for b in a {
            v.extend_from_slice(b);
        }
    }
    for a in &fc.filter_intra {
        v.extend_from_slice(a);
    }
    v.extend_from_slice(&fc.filter_intra_mode);
    v.extend_from_slice(&fc.cfl_sign);
    for a in &fc.cfl_alpha {
        v.extend_from_slice(a);
    }
    v.extend_from_slice(&fc.delta_q);
    for a in &fc.delta_lf_multi {
        v.extend_from_slice(a);
    }
    v.extend_from_slice(&fc.delta_lf);
    v.extend_from_slice(&fc.intrabc);
    v.extend_from_slice(&fc.ndvc_joints);
    v.extend_from_slice(&fc.ndvc_comp0);
    v.extend_from_slice(&fc.ndvc_comp1);
    for a in &fc.tx_size {
        for b in a {
            v.extend_from_slice(b);
        }
    }
    for a in &fc.ext_tx_1ddct {
        for b in a {
            v.extend_from_slice(b);
        }
    }
    for a in &fc.ext_tx_dtt4 {
        for b in a {
            v.extend_from_slice(b);
        }
    }
    v.extend_from_slice(&fc.switchable_restore);
    v.extend_from_slice(&fc.wiener_restore);
    v.extend_from_slice(&fc.sgrproj_restore);
    v.extend_from_slice(&fc.coeff);
    v
}

/// Field names by dump offset, for failure localization.
const FIELDS: [(&str, usize); 30] = [
    ("kf_y", 350),
    ("uv_mode", 390),
    ("angle_delta", 64),
    ("skip", 9),
    ("seg_spatial", 27),
    ("partition", 220),
    ("palette_y_mode", 63),
    ("palette_uv_mode", 6),
    ("palette_y_size", 56),
    ("palette_uv_size", 56),
    ("palette_y_color_index", 315),
    ("palette_uv_color_index", 315),
    ("filter_intra", 66),
    ("filter_intra_mode", 6),
    ("cfl_sign", 9),
    ("cfl_alpha", 102),
    ("delta_q", 5),
    ("delta_lf_multi", 20),
    ("delta_lf", 5),
    ("intrabc", 3),
    ("ndvc_joints", 5),
    ("ndvc_comp0", 69),
    ("ndvc_comp1", 69),
    ("tx_size", 48),
    ("ext_tx_1ddct", 416),
    ("ext_tx_dtt4", 312),
    ("switchable_restore", 4),
    ("wiener_restore", 3),
    ("sgrproj_restore", 3),
    ("coeff", 4045),
];

fn field_at(off: usize) -> String {
    let mut base = 0;
    for &(name, len) in &FIELDS {
        if off < base + len {
            return format!("{name}[{}]", off - base);
        }
        base += len;
    }
    format!("past-end[{off}]")
}

#[test]
fn default_kf_fc_matches_compiled_c_all_bands() {
    // Both sides of every band boundary + extremes.
    for q in [0, 1, 20, 21, 60, 61, 120, 121, 200, 255] {
        let rust = flatten(&KfFrameContext::default_for_qindex(q));
        let cd = c::ref_dump_default_kf_fc(q);
        assert_eq!(rust.len(), cd.len(), "dump length");
        if rust != cd {
            let bad = rust.iter().zip(&cd).position(|(a, b)| a != b).unwrap();
            panic!(
                "q={q}: first mismatch at {} — rust {} vs C {}",
                field_at(bad),
                rust[bad],
                cd[bad]
            );
        }
    }
}

#[test]
fn coeff_band_selection_matches_get_q_ctx() {
    for q in 0..256 {
        let expect = if q <= 20 {
            0
        } else if q <= 60 {
            1
        } else if q <= 120 {
            2
        } else {
            3
        };
        assert_eq!(coeff_cdf_q_ctx(q), expect, "q={q}");
    }
    // Adjacent bands really are different tables (non-vacuity).
    use aom_dsp::entropy::default_cdfs::DEFAULT_COEFF_ARENA;
    for w in DEFAULT_COEFF_ARENA.windows(2) {
        assert_ne!(w[0], w[1]);
    }
}

/// `inter_ext_tx_cdf` is the intrabc tx-type CDF region. It is not part of the
/// `shim_dump_default_kf_fc` KF dump (that stays append-only), so verify it
/// against a dedicated dump of the real `av1_setup_past_independence`
/// `fc->inter_ext_tx_cdf`. The table is qindex-independent
/// (`av1_init_mode_probs`), so any band reproduces it — sweep a few anyway.
#[test]
fn default_inter_ext_tx_matches_compiled_c() {
    for q in [0, 20, 21, 120, 121, 255] {
        let fc = KfFrameContext::default_for_qindex(q);
        // Row-major flatten matching the contiguous [4][4][17] C memcpy.
        let rust: Vec<u16> = fc
            .inter_ext_tx
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        let cd = c::ref_dump_default_inter_ext_tx(q);
        assert_eq!(rust.len(), c::DUMP_INTER_EXT_TX_LEN, "dump length");
        assert_eq!(rust.len(), cd.len(), "length parity");
        if rust != cd {
            let bad = rust.iter().zip(&cd).position(|(a, b)| a != b).unwrap();
            let (set, sz, slot) = (bad / (4 * 17), (bad / 17) % 4, bad % 17);
            panic!(
                "q={q}: inter_ext_tx mismatch at set {set} sz {sz} slot {slot} \
                 — rust {} vs C {}",
                rust[bad], cd[bad]
            );
        }
    }
    // Non-vacuity: set 0 is DCT-only (all-zero), sets 1-3 are populated.
    let fc = KfFrameContext::default_for_qindex(120);
    assert!(
        fc.inter_ext_tx[0].iter().flatten().all(|&x| x == 0),
        "set 0 zero"
    );
    assert!(
        fc.inter_ext_tx[1].iter().flatten().any(|&x| x != 0),
        "set 1 nonzero"
    );
}

/// The default CDFs the INTRA-block-inside-an-INTER-frame path and the
/// inter-intra prediction path code with. None of them are part of the
/// `shim_dump_default_kf_fc` KF dump (that stays append-only), so verify them
/// against a dedicated dump of the real `av1_setup_past_independence` frame
/// context. All five are qindex-independent (`av1_init_mode_probs`), so any
/// band reproduces them — sweep a few anyway.
///
/// `DEFAULT_Y_MODE` is the one the KEY path never touches: an intra block in an
/// inter frame codes its Y mode on `y_mode_cdf[size_group_lookup[bsize]]`, NOT
/// the KEY frame's neighbour-context `kf_y_cdf` (decodemv.c:1077 vs :815).
#[test]
fn default_intra_in_inter_cdfs_match_compiled_c() {
    use aom_dsp::entropy::default_cdfs as d;
    let rust: Vec<u16> = d::DEFAULT_Y_MODE
        .iter()
        .flatten()
        .chain(d::DEFAULT_INTERINTRA.iter().flatten())
        .chain(d::DEFAULT_INTERINTRA_MODE.iter().flatten())
        .chain(d::DEFAULT_WEDGE_INTERINTRA.iter().flatten())
        .chain(d::DEFAULT_WEDGE_IDX.iter().flatten())
        .copied()
        .collect();
    assert_eq!(rust.len(), c::DUMP_INTRA_IN_INTER_LEN, "dump length");
    let names = [
        "y_mode_cdf",
        "interintra_cdf",
        "interintra_mode_cdf",
        "wedge_interintra_cdf",
        "wedge_idx_cdf",
    ];
    for q in [0, 20, 21, 120, 121, 255] {
        let cd = c::ref_dump_default_intra_in_inter_cdfs(q);
        assert_eq!(rust.len(), cd.len(), "length parity");
        if rust != cd {
            let bad = rust.iter().zip(&cd).position(|(a, b)| a != b).unwrap();
            let (mut which, mut base) = (0usize, 0usize);
            for (i, len) in c::DUMP_INTRA_IN_INTER_LENS.iter().enumerate() {
                if bad < base + len {
                    which = i;
                    break;
                }
                base += len;
            }
            panic!(
                "q={q}: {} mismatch at dump offset {bad} (slot {} within table) \
                 — rust {} vs C {}",
                names[which],
                bad - base,
                rust[bad],
                cd[bad]
            );
        }
    }
    // Non-vacuity: every table carries real (non-zero, non-uniform) content, so
    // a silently-zeroed dump on either side cannot pass.
    assert!(d::DEFAULT_Y_MODE.iter().flatten().any(|&x| x != 0));
    assert_ne!(d::DEFAULT_Y_MODE[0][0], d::DEFAULT_Y_MODE[3][0]);
    assert_ne!(
        d::DEFAULT_INTERINTRA_MODE[0][0],
        d::DEFAULT_INTERINTRA_MODE[1][0]
    );
    assert!(d::DEFAULT_WEDGE_IDX.iter().flatten().any(|&x| x != 0));
    // The interintra bsize range (BLOCK_8X8=3 ..= BLOCK_32X32=9) must carry a
    // trained wedge flag; the out-of-range rows stay at the 16384 init.
    assert_ne!(d::DEFAULT_WEDGE_INTERINTRA[3][0], 16384);
    assert_eq!(d::DEFAULT_WEDGE_INTERINTRA[0][0], 16384);
}
