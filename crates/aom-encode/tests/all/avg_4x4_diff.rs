//! Differential: [`aom_encode::var_part::avg_4x4`] vs the REAL exported
//! `aom_avg_4x4_c` (aom_dsp/avg.c:32) — the variance-based partitioner's
//! KEY-frame 4x4 downsampling kernel (`fill_variance_4x4avg`,
//! var_based_part.c:390). The port computes on u16 pixel buffers (this
//! port's universal plane type); the reference is the lowbd u8 kernel — for
//! 8-bit sample values the arithmetic is identical (and
//! `aom_highbd_avg_4x4_c` is the same expression at u16 width, so the one
//! differential covers both dispatches).

use aom_encode::var_part::avg_4x4;
use aom_sys_ref as c;

#[test]
fn avg_4x4_matches_real_c() {
    c::ref_init();
    // Deterministic LCG (no external rand dep, matching the established
    // diff-test pattern).
    let mut state = 0x2456_1a3du32;
    let mut next = move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        state
    };
    for case in 0..4000 {
        let stride = 4 + (next() % 13) as usize;
        let mut buf8 = vec![0u8; 3 * stride + 4 + 16];
        for b in buf8.iter_mut() {
            *b = (next() >> 13) as u8;
        }
        // Edge-heavy cases: saturate some blocks to extremes.
        if case % 7 == 0 {
            for b in buf8.iter_mut() {
                *b = if case % 14 == 0 { 255 } else { 0 };
            }
        }
        let buf16: Vec<u16> = buf8.iter().map(|&b| u16::from(b)).collect();
        let ours = avg_4x4(&buf16, 0, stride);
        let real = c::ref_avg_4x4(&buf8, stride);
        assert_eq!(
            ours as u32, real,
            "avg_4x4 mismatch at case {case} (stride {stride})"
        );
    }
}
