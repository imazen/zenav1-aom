//! INTER-ENCODE chunk 2, sub-step 2f (pack half): the INTER-frame block
//! mode-info writer.
//!
//! C: `pack_inter_mode_mvs` (`av1/encoder/bitstream.c:1092`) — the inter-frame
//! analogue of `write_mb_modes_kf` (`:1266`), which the port's `pack_leaf`
//! already implements for KEY frames. **Every** block of an inter frame goes
//! through this writer, intra blocks included (they take the `is_inter == 0`
//! branch and then write intra prediction modes).
//!
//! # Symbol order — VERIFIED against C, and against a real stream
//!
//! `pack_inter_mode_mvs` writes, in this exact order:
//!
//! 1. `write_inter_segment_id(..., preskip)` — nothing when segmentation is off
//! 2. `write_skip_mode` — nothing when `skip_mode_allowed == 0`
//! 3. **`write_skip`**
//! 4. `write_inter_segment_id(..., postskip)` — nothing when segmentation is off
//! 5. `write_cdef` — nothing when CDEF is off
//! 6. `write_delta_q_params` — nothing when delta-q is off
//! 7. **`write_is_inter`**
//! 8. inter branch: `write_ref_frames` → `write_inter_mode` → [`write_drl_idx`]
//!    → [MV] → [interintra] → `write_motion_mode` → [compound] →
//!    `write_mb_interp_filter`
//!
//! Note that **skip precedes is_inter**, with cdef and delta-q between them —
//! the prologue is shape-identical to the KEY writer's, and only step 7 onward
//! differs. (An earlier handoff note in `INTER-CHUNK2-HANDOFF.md` gave the order
//! as "is_inter → ref → mode → skip"; that is wrong, and this module follows
//! the C source.)
//!
//! Independently confirmed on a real `aomenc` stream with the instrumented
//! libaom decoder's per-symbol accounting
//! (`/root/aom-inspect/examples/inspect -a`): for the §3 zero-MV P the frame-1
//! symbol sequence is exactly `read_partition`, `read_skip_txfm`,
//! `read_is_inter_block`, `read_ref_frames` (3 binary symbols),
//! `read_inter_mode` (3 binary symbols).
//!
//! # Envelope
//!
//! This writer covers the INTER-ENCODE-ROADMAP §3 envelope: segmentation off,
//! `skip_mode_allowed = 0`, single reference (`reference_mode` SINGLE_REFERENCE),
//! `switchable_motion_mode = 0`, non-switchable frame `interp_filter`, no
//! compound, no interintra. Under those the tail (steps 8's bracketed items)
//! writes NOTHING:
//! - `write_drl_idx` is gated on `NEWMV || NEW_NEWMV || have_nearmv_in_inter_mode`
//!   — `NEARESTMV` and `GLOBALMV` code no DRL index;
//! - the MV coder runs only for the NEW\* modes;
//! - `write_motion_mode` collapses to `SIMPLE_TRANSLATION` and writes nothing
//!   when `switchable_motion_mode == 0` (bitstream.c:280-287);
//! - `write_mb_interp_filter` returns early unless the FRAME filter is
//!   `SWITCHABLE` (bitstream.c:638) — the §3 config pins a fixed filter.
//!
//! Anything outside that envelope is rejected by an explicit assertion rather
//! than silently mis-written.

use aom_dsp::entropy::enc::OdEcEnc;
use aom_dsp::entropy::partition::{write_inter_mode, write_is_inter, write_ref_frames, write_skip};

use crate::inter_costs::{InterFrameCdfs, SingleRefCtx, NEARESTMV, NEWMV};

/// One inter leaf's coded mode info, reduced to the §3 envelope.
#[derive(Clone, Copy, Debug)]
pub struct InterLeafSyntax {
    /// `mbmi->skip_txfm`.
    pub skip_txfm: i32,
    /// `is_inter_block(mbmi)` — 0 for an intra block inside the inter frame
    /// (whose intra prediction modes the caller writes after this).
    pub is_inter: i32,
    /// `mbmi->ref_frame[0]`. Only `LAST_FRAME` (1) is supported here.
    pub ref_frame0: i8,
    /// `mbmi->mode` (`NEARESTMV`/`NEARMV`/`GLOBALMV`/`NEWMV`).
    pub inter_mode: i32,
    /// `av1_mode_context_analyzer`'s result for this block+reference.
    pub mode_context: i32,
}

/// The per-block prediction contexts this writer selects CDF rows with.
#[derive(Clone, Copy, Debug, Default)]
pub struct InterLeafCtx {
    /// `av1_get_skip_txfm_context(xd)`.
    pub skip_ctx: usize,
    /// `av1_get_intra_inter_context(xd)`.
    pub intra_inter_ctx: i32,
    /// `av1_get_pred_context_single_ref_p1..p6`.
    pub single_ref: SingleRefCtx,
}

/// Write one inter-frame leaf's mode info — steps 3, 7 and 8 of
/// `pack_inter_mode_mvs` (the segmentation, skip-mode, CDEF and delta-q steps
/// are the caller's, exactly as in `pack_leaf`'s KEY path).
///
/// Mutates `inter_cdfs` and `skip_cdf` in place: each written symbol adapts its
/// CDF (`disable_cdf_update == 0` in the §3 config), so the next block sees the
/// updated distribution.
///
/// # Panics
///
/// On anything outside the §3 envelope — a non-LAST reference, a compound
/// reference, or a mode that would require DRL/MV syntax this writer does not
/// emit. Failing loudly is deliberate: silently omitting a symbol desyncs the
/// decoder for the rest of the tile.
pub fn write_inter_leaf_mode_info(
    enc: &mut OdEcEnc,
    inter_cdfs: &mut InterFrameCdfs,
    skip_cdf: &mut [u16],
    ctx: &InterLeafCtx,
    info: &InterLeafSyntax,
) {
    // --- step 3: write_skip ---
    write_skip(enc, skip_cdf, false, info.skip_txfm);

    // --- step 7: write_is_inter ---
    write_is_inter(
        enc,
        &mut inter_cdfs.intra_inter[ctx.intra_inter_ctx as usize],
        false,
        false,
        info.is_inter,
    );
    if info.is_inter == 0 {
        // An intra block inside the inter frame: the caller writes
        // `write_intra_prediction_modes` next.
        return;
    }

    assert_eq!(
        info.ref_frame0, 1,
        "inter pack envelope is single-reference LAST_FRAME only (got ref_frame0 {})",
        info.ref_frame0
    );
    assert!(
        (NEARESTMV..=NEWMV).contains(&info.inter_mode),
        "inter pack envelope covers the single-reference inter modes only (got mode {})",
        info.inter_mode
    );
    assert_ne!(
        info.inter_mode, NEWMV,
        "NEWMV needs the MV coder + DRL syntax, which this writer does not emit yet"
    );
    assert_ne!(
        info.inter_mode, crate::inter_costs::NEARMV,
        "NEARMV needs write_drl_idx, which this writer does not emit yet"
    );

    // --- step 8: write_ref_frames (single-reference cascade) ---
    // The blob carries each single-ref slot pre-selected by its own prediction
    // context; the writer adapts the rows it takes, so absorb them back.
    let mut blob = inter_cdfs.single_ref_blob(&ctx.single_ref);
    write_ref_frames(
        enc,
        &mut blob,
        false, // seg_ref_active
        false, // seg_skipgmv_active
        false, // reference_mode_is_select — SINGLE_REFERENCE in the §3 config
        false, // is_comp_ref_allowed
        false, // is_compound
        0,     // comp_ref_type (dead)
        i32::from(info.ref_frame0),
        -1, // ref_frame[1] = NONE
    );
    inter_cdfs.absorb_single_ref_blob(&ctx.single_ref, &blob);

    // --- step 8: write_inter_mode ---
    // Split-borrow the three CDF tables (write_inter_mode indexes each by its
    // own slice of mode_context).
    let InterFrameCdfs {
        newmv,
        zeromv,
        refmv,
        ..
    } = inter_cdfs;
    write_inter_mode(enc, newmv, zeromv, refmv, info.inter_mode, info.mode_context);

    // The remaining tail items (DRL, MV, interintra, motion mode, compound,
    // interp filter) write nothing in this envelope — see the module docs.
}
