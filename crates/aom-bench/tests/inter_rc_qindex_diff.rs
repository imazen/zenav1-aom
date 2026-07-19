//! INTER-ENCODE chunk 2b GATE — fixed-Q inter rate control (the low-delay P-frame
//! qindex), byte-verified vs real `aomenc`.
//!
//! `crates/aom-encode/src/rc.rs` derives the `base_qindex` a frame gets from
//! `--cq-level` under `--end-usage=q`. The lone-KEY value was already locked
//! (`qindex_from_cq_diff`); this gate locks the **multi-frame low-delay P (inter
//! leaf) frame** value — the qindex frame 1 of a `--lag-in-frames=0 --limit=2`
//! `[KEY, P]` clip is coded at (INTER-ENCODE-ROADMAP.md §3, chunk 2b).
//!
//! Method: encode the 2-frame clip with real `aomenc` at the §3 simplest inter
//! config ([`MultiFrameEncodeCell::c_encode_inter`]), decode both frames with the
//! port decoder, and assert frame 1's coded `base_qindex` equals the port's
//! [`aom_encode::rc::base_qindex_lowdelay_p_from_cq`] — proving the port derives
//! the inter-leaf CQ qindex byte-identically, without reading it off the stream.
//!
//! The cq sweep is bounded to the port inter DECODER's byte-exact envelope
//! (monochrome, translational P; see chunk 0's finding) so decoding frame 1's
//! header is reliable. cq 0 is excluded — the coded-lossless P is outside the
//! decoder's inter skeleton (it panics in recon), and its qindex (0) is the
//! degenerate floor case, not the RC path this gate covers.

use aom_bench::{EncodeCell, MultiFrameEncodeCell};
use aom_encode::rc::base_qindex_lowdelay_p_from_cq;

/// A textured mono base frame (frame 0), usage = GOOD (the inter context).
fn mono_base(w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for c in 0..w {
            y[r * w + c] = (40 + ((r * 3 + c * 5) % 160)) as u16;
        }
    }
    EncodeCell {
        label: format!("rc_mono{w}x{h}_cq{cq}"),
        w,
        h,
        mono: true,
        ss_x: 1,
        ss_y: 1,
        usage: 0, // GOOD_QUALITY (inter context)
        cq_level: cq,
        speed,
        bd: 8,
        y,
        u: Vec::new(),
        v: Vec::new(),
    }
}

#[test]
fn lowdelay_p_qindex_byte_matches_aomenc() {
    // Bounded to the port inter decoder's byte-exact envelope (mono, translational
    // P), which lets us read frame 1's coded header reliably.
    let cqs = [8, 12, 20, 32, 48, 60, 63];
    let mut anti_vacuous = false;
    for &cq in &cqs {
        let cell = MultiFrameEncodeCell::translational(&mono_base(64, 64, cq, 0), 3, 0);
        let stream = cell.c_encode_inter(false, false);
        let frames = aom_decode::frame::decode_frames(&stream)
            .unwrap_or_else(|e| panic!("cq{cq}: decode 2-frame stream failed: {e}"));
        assert_eq!(frames.len(), 2, "cq{cq}: expected KEY + P");

        let want_p = base_qindex_lowdelay_p_from_cq(cq);
        let got_p = frames[1].base_qindex;
        assert_eq!(
            got_p, want_p,
            "cq{cq}: port low-delay P qindex {want_p} != aomenc coded frame-1 base_qindex {got_p}"
        );

        // Anti-vacuity: the KEY frame of the SAME clip is boosted LOWER (kf_boost,
        // frames_to_key > 1), so KEY qindex < P qindex. This proves we are locking
        // the leaf-P CQ path, not a value that trivially equals the KEY qindex.
        let key_q = frames[0].base_qindex;
        if key_q != got_p {
            assert!(
                key_q < got_p,
                "cq{cq}: KEY qindex {key_q} should be boosted below P qindex {got_p}"
            );
            anti_vacuous = true;
        }
    }
    assert!(
        anti_vacuous,
        "anti-vacuity: no cq exercised a KEY-vs-P qindex difference — the gate would \
         pass even if the P path collapsed onto the KEY value"
    );
}
