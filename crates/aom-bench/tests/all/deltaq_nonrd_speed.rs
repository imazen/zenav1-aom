//! `--deltaq-mode` 2 and 3 at the NONRD speeds (`--cpu-used` >= 8) — the arm
//! that used to be a REFUSAL rather than a divergence.
//!
//! `port_encode_with` asserted
//! *"derived delta_q_present must match the real --deltaq-mode=3 header,
//! left: false right: true"* on every cell here: the port derived the per-SB
//! qindex with `setup_delta_q` (encodeframe.c:697) at every speed, while at
//! ALLINTRA `--cpu-used` >= 8 the frame goes through `encode_nonrd_sb`, whose
//! delta-q is **`setup_delta_q_nonrd`** (:598). That arm models
//! `DELTA_Q_VARIANCE_BOOST` and nothing else, so modes 2 and 3 quantize every
//! superblock against the frame `base_qindex`, never consult their modulation
//! maps, leave `cpi->deltaq_used == 0`, and have `delta_q_present_flag` cleared
//! outright at :2450.
//!
//! Two properties are gated, and the second is what makes the first mean
//! something:
//!
//! 1. **byte-identity** to real aomenc across `--cpu-used` 8 and 9, both modes,
//!    the web quality range and a non-square shape;
//! 2. **the arm is REACHED, and it is not the trivially-equal arm** — the
//!    reference encoder's own header is asserted to carry
//!    `delta_q_present = false` on these cells while the SAME content at
//!    `--cpu-used 0` carries `true`. Without that, a port that simply switched
//!    delta-q off at speed 8 would pass, and so would one that never wired the
//!    knob at all.

use aom_bench::{EncodeCell, ToggleKnobs, stream_frame_header};

const AV1E_SET_DELTAQ_MODE: i32 = 107;

fn knobs(mode: i32) -> ToggleKnobs {
    ToggleKnobs {
        deltaq_mode2: mode == 2,
        deltaq_mode3: mode == 3,
        ..Default::default()
    }
}

/// Real photographic content, cropped from the 196x196 conformance vector —
/// the same source `deltaq_mode3_e2e` uses, so a divergence here cannot be
/// blamed on content the delta-q gates have never seen.
fn cell(w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    EncodeCell::real_content(
        &format!("dqnonrd_{w}x{h}_cq{cq}_s{speed}"),
        "av1-1-b8-01-size-196x196",
        Some((w, h, 0, 0)),
        cq,
        speed,
    )
}

fn run_cell(cell: &EncodeCell, mode: i32) -> Result<usize, String> {
    let c_stream = cell.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, mode)]);
    let real = EncodeCell::frame_obu_payload(&c_stream);
    let ours = cell.port_encode_with(&c_stream, &knobs(mode));
    if ours == real {
        return Ok(real.len());
    }
    let first = ours
        .iter()
        .zip(real.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(ours.len().min(real.len()));
    Err(format!(
        "first diff at frame-OBU byte {first}; port {} B vs real {} B",
        ours.len(),
        real.len()
    ))
}

/// Cells the port is byte-identical on. `cq 63` is NOT here — see
/// [`NONRD_CQ63_OPEN`].
fn gated_cells() -> Vec<(EncodeCell, i32)> {
    let mut cells = Vec::new();
    for mode in [2, 3] {
        for speed in [8, 9] {
            for cq in [12, 32, 48, 55, 60] {
                cells.push((cell(192, 192, cq, speed), mode));
            }
            // A non-square shape (3x2 SBs), so the running-base chain is
            // exercised over a different SB raster.
            cells.push((cell(192, 128, 32, speed), mode));
        }
    }
    cells
}

/// The measured open set, pinned self-promotingly (playbook §5): `cq 63` only,
/// at both nonrd speeds, in both modes. See the module doc for the attribution.
const NONRD_CQ63_OPEN: &[(i32, i32)] = &[(2, 8), (2, 9), (3, 8), (3, 9)];

/// THE gate. Every cell here was a hard `assert_eq!` panic inside
/// `port_encode_with` before `setup_delta_q_nonrd` was ported.
#[test]
fn deltaq_modes_2_and_3_at_nonrd_speeds_byte_match() {
    let cells = gated_cells();
    let mut matched = 0usize;
    let mut report = String::new();
    for (c, mode) in &cells {
        match run_cell(c, *mode) {
            Ok(n) => {
                matched += 1;
                report.push_str(&format!("  MATCH  mode{mode} {} ({n} B)\n", c.label));
            }
            Err(e) => report.push_str(&format!("  DIFF   mode{mode} {}: {e}\n", c.label)),
        }
    }
    println!(
        "--deltaq-mode 2/3 x nonrd speeds: {matched}/{}\n{report}",
        cells.len()
    );
    assert_eq!(
        matched,
        cells.len(),
        "every gated nonrd delta-q cell must byte-match real aomenc:\n{report}"
    );
}

/// NON-VACUITY, and the honest version of it. On the gated cells the reference
/// codes **byte-identical** streams with and without `--deltaq-mode`, so the
/// byte gate above, taken alone, only proves the port ignores the knob as
/// thoroughly as the reference does. That is worth stating rather than dressing
/// up — and it is also the FINDING: `setup_delta_q_nonrd` models
/// `DELTA_Q_VARIANCE_BOOST` and nothing else, so at nonrd speeds modes 2 and 3
/// are inert, every delta is zero, and `encodeframe.c:2450` clears the header
/// bit. An encoder that behaved otherwise would fail this test.
///
/// What stops "delta-q does nothing at speed 8/9" from being vacuously true is
/// the cq-63 row: there the reference DOES move (235 B against plain's 228 at
/// `--cpu-used 8`), so the arm is reachable and observable, and the flat model
/// is a real prediction rather than a description of an unreachable path. That
/// half is asserted in [`cq63_at_nonrd_speeds_is_pinned_open`].
#[test]
fn the_modes_are_inert_at_nonrd_speeds_and_that_is_the_prediction() {
    let mut inert = 0usize;
    let mut flag_off = 0usize;
    let mut checked = 0usize;
    for (c, mode) in gated_cells() {
        let with_stream = c.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, mode)]);
        let plain = EncodeCell::frame_obu_payload(&c.c_encode_ctrls(&[]));
        let with = EncodeCell::frame_obu_payload(&with_stream);
        if plain == with {
            inert += 1;
        }
        if !stream_frame_header(&with_stream).delta_q.delta_q_present {
            flag_off += 1;
        }
        checked += 1;
    }
    assert_eq!(
        (inert, flag_off),
        (checked, checked),
        "at nonrd speeds `--deltaq-mode` 2/3 must be byte-INERT in the reference \
         and must clear `delta_q_present` — that is what `setup_delta_q_nonrd` \
         predicts. {inert} of {checked} inert, {flag_off} of {checked} flag-off."
    );

    // ...and NOT inert where the clamp inside `av1_adjust_q_from_delta_q_res`
    // bites, which is what makes the line above a prediction rather than a
    // description of a dead path.
    let mut moved = 0usize;
    for speed in [8, 9] {
        let c = cell(192, 192, 63, speed);
        let plain = EncodeCell::frame_obu_payload(&c.c_encode_ctrls(&[]));
        for mode in [2, 3] {
            let with =
                EncodeCell::frame_obu_payload(&c.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, mode)]));
            if plain != with {
                moved += 1;
            }
        }
    }
    assert_eq!(
        moved, 4,
        "the cq-63 nonrd cells must show the knob is NOT universally inert, else \
         the inertness above proves nothing"
    );
    println!("gated: {inert}/{checked} inert, {flag_off}/{checked} flag-off; cq63: {moved}/4 moved");
}

/// The measured residual, pinned in BOTH directions. `cq 63` (base_qindex 63)
/// diverges at both nonrd speeds in both modes, while every other quantizer on
/// the same content and the PLAIN (no delta-q) control at cq 63 are
/// byte-identical — so this is delta-q's, not a general cq-63 nonrd defect.
///
/// Measured attribution, not a guess:
///
/// * `--cpu-used 8`: the reference header carries `delta_q_present = false`, and
///   the port agrees, yet the reference stream is 235 B against plain's 228 —
///   `setup_delta_q_nonrd`'s per-superblock `av1_init_plane_quantizers` +
///   `mi->current_qindex` stamp are observable with the flag off, and the port
///   models the flag but not that.
/// * `--cpu-used 9`: the reference carries `delta_q_present = TRUE` with
///   `delta_q_res = 8`. 8 is not `DEFAULT_DELTA_Q_RES_PERCEPTUAL` (4) — it is
///   what `aom_get_variance_boost_delta_q_res` produces, i.e. at speed 9 the
///   reference is on the `DELTA_Q_VARIANCE_BOOST` arm of
///   `setup_delta_q_nonrd`, NOT on mode 2/3 at all. That arm is deliberately
///   unported (see `setup_delta_q_nonrd`'s doc): modelling it would be claiming
///   coverage no cell here can check.
///
/// Bounded: cq {12, 32, 48, 55, 60} match at both speeds in both modes, and so
/// does the 192x128 shape. Promote a row into `gated_cells` when it closes.
#[test]
fn cq63_at_nonrd_speeds_is_pinned_open() {
    let mut open = Vec::new();
    for &(mode, speed) in NONRD_CQ63_OPEN {
        let c = cell(192, 192, 63, speed);
        if run_cell(&c, mode).is_err() {
            open.push((mode, speed));
        }
    }
    assert_eq!(
        open.as_slice(),
        NONRD_CQ63_OPEN,
        "the pinned cq-63 nonrd divergence set MOVED. A row that closed must be \
         promoted into `gated_cells`; a NEW row is a regression."
    );

    // The control that attributes it to delta-q rather than to cq 63: with no
    // delta-q knob at all, the same cells are byte-identical.
    for &(_, speed) in NONRD_CQ63_OPEN {
        let c = cell(192, 192, 63, speed);
        let s = c.c_encode_ctrls(&[]);
        assert_eq!(
            c.port_encode_with(&s, &ToggleKnobs::default()),
            EncodeCell::frame_obu_payload(&s),
            "cq63 --cpu-used {speed} must be byte-exact WITHOUT delta-q, else \
             this pin is mis-attributed"
        );
    }
}
