//! Differential harness for the intra edge-filter DSP vs C libaom:
//! intra_edge_filter_strength, av1_use_intra_edge_upsample,
//! av1_filter_intra_edge_c, av1_upsample_intra_edge_c.

use aom_dsp::intra::edge::{edge_filter_strength, filter_intra_edge, upsample_intra_edge, use_upsample};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo) as u64) as i32
    }
}

#[test]
fn strength_and_upsample_decisions_exhaustive() {
    // block dims 4..64, delta -90..90, both filter types.
    for &bs0 in &[4, 8, 16, 32, 64] {
        for &bs1 in &[4, 8, 16, 32, 64] {
            for delta in -90..=90 {
                for ty in 0..2 {
                    assert_eq!(
                        edge_filter_strength(bs0, bs1, delta, ty),
                        c::ref_intra_edge_strength(bs0, bs1, delta, ty),
                        "strength {bs0}+{bs1} d={delta} ty={ty}"
                    );
                    assert_eq!(
                        use_upsample(bs0, bs1, delta, ty),
                        c::ref_use_intra_edge_upsample(bs0, bs1, delta, ty),
                        "upsample {bs0}+{bs1} d={delta} ty={ty}"
                    );
                }
            }
        }
    }
}

#[test]
fn filter_intra_edge_byte_identical() {
    let mut rng = Rng(0x_ed6e_f117_0000_1111);
    for sz in 2..=65usize {
        for strength in 0..=3 {
            for _ in 0..200 {
                let base: Vec<u8> = (0..sz).map(|_| rng.range(0, 256) as u8).collect();
                let mut a = base.clone();
                let mut b = base.clone();
                filter_intra_edge(&mut a, sz, strength);
                c::ref_filter_intra_edge(&mut b, 0, sz, strength);
                assert_eq!(a, b, "filter sz={sz} strength={strength}");
            }
        }
    }
}

#[test]
fn upsample_intra_edge_byte_identical() {
    let mut rng = Rng(0x_c057_9a1e_0000_2222);
    const OFF: usize = 4;
    for sz in 1..=16usize {
        for _ in 0..300 {
            // buffer: OFF pad bytes + (2*sz + a few) working region.
            let n = OFF + 2 * sz + 4;
            let base: Vec<u8> = (0..n).map(|_| rng.range(0, 256) as u8).collect();
            let mut a = base.clone();
            let mut b = base.clone();
            upsample_intra_edge(&mut a, OFF, sz);
            c::ref_upsample_intra_edge(&mut b, OFF, sz);
            assert_eq!(a, b, "upsample sz={sz}");
        }
    }
}

#[test]
fn highbd_filter_intra_edge_byte_identical() {
    let mut rng = Rng(0x_86bd_ed6e_0000_3333);
    for &bd in &[8u8, 10, 12] {
        let max = (1u32 << bd) - 1;
        for sz in 2..=65usize {
            for strength in 0..=3 {
                for _ in 0..120 {
                    let base: Vec<u16> =
                        (0..sz).map(|_| (rng.next() as u32 & max) as u16).collect();
                    let mut a = base.clone();
                    let mut b = base.clone();
                    aom_dsp::intra::edge::highbd_filter_intra_edge(&mut a, sz, strength);
                    c::ref_highbd_filter_intra_edge(&mut b, 0, sz, strength);
                    assert_eq!(a, b, "hbd filter bd={bd} sz={sz} s={strength}");
                }
            }
        }
    }
}

#[test]
fn highbd_filter_intra_edge_at_byte_identical() {
    // The slack-buffered entry (C's SSE4 sliding-window shape): the edge
    // region must match the scalar C oracle byte-for-byte, and the kernel's
    // C-faithful side-effect writes land at buf[off-1] / buf[off+sz..+8].
    let mut rng = Rng(0x_5a1d_1ab1_e000_7777);
    const OFF: usize = 15; // production DIR_PAD - 1
    for &bd in &[8u8, 10, 12] {
        let max = (1u32 << bd) - 1;
        for sz in 2..=129usize {
            for strength in 0..=3 {
                for _ in 0..40 {
                    let n = OFF + sz + 16;
                    let mut a: Vec<u16> =
                        (0..n).map(|_| (rng.next() as u32 & max) as u16).collect();
                    let orig = a.clone();
                    let mut b: Vec<u16> = a[OFF..OFF + sz].to_vec();
                    aom_dsp::intra::edge::highbd_filter_intra_edge_at(&mut a, OFF, sz, strength);
                    c::ref_highbd_filter_intra_edge(&mut b, 0, sz, strength);
                    assert_eq!(
                        &a[OFF..OFF + sz],
                        b.as_slice(),
                        "hbd filter_at edge bd={bd} sz={sz} s={strength}"
                    );
                    if strength != 0 {
                        // C-SSE4 side effects: p[-1] = p[0], p[sz..sz+8] splat.
                        assert_eq!(a[OFF - 1], orig[OFF], "p[-1] write");
                        assert!(
                            a[OFF + sz..OFF + sz + 8]
                                .iter()
                                .all(|&v| v == orig[OFF + sz - 1]),
                            "tail splat"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn highbd_upsample_intra_edge_byte_identical() {
    let mut rng = Rng(0x_86bd_c057_0000_4444);
    const OFF: usize = 4;
    for &bd in &[8u8, 10, 12] {
        let max = (1u32 << bd) - 1;
        for sz in 1..=16usize {
            for _ in 0..200 {
                let n = OFF + 2 * sz + 4;
                let base: Vec<u16> = (0..n).map(|_| (rng.next() as u32 & max) as u16).collect();
                let mut a = base.clone();
                let mut b = base.clone();
                aom_dsp::intra::edge::highbd_upsample_intra_edge(&mut a, OFF, sz, bd);
                c::ref_highbd_upsample_intra_edge(&mut b, OFF, sz, bd);
                assert_eq!(a, b, "hbd upsample bd={bd} sz={sz}");
            }
        }
    }
}

/// Tier sweep for the i32-lane kernels behind `highbd_filter_intra_edge` and
/// `highbd_upsample_intra_edge`: the C differentials above pin the default
/// tier; this runs every archmage token permutation so the NEON / wasm128 /
/// scalar tiers are held to the same byte-identity against the real C
/// kernels. The filter's `sz` sweep reaches 129 — the production maximum
/// (`n_top_px + 1 + txhpx` for 64-wide blocks), beyond the default-tier
/// test's 65 — and the upsample sweep covers its full `sz <= 16` domain.
#[test]
fn highbd_edge_kernels_match_c_at_every_tier() {
    use archmage::SimdToken;
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
    // Serialise: this sweep permutes PROCESS-GLOBAL dispatch state.
    let _serial = crate::dispatch_serial::dispatch_serial();
    let mut simd_perms = 0usize;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |_tier| {
        if if cfg!(target_arch = "aarch64") {
            archmage::NeonToken::summon().is_some()
        } else {
            archmage::X64V3Token::summon().is_some()
        } {
            simd_perms += 1;
        }
        let mut rng = Rng(0x_ED6E_51B1_7E12_2026 ^ 0x9E37_79B9_7F4A_7C15);
        for &bd in &[8u8, 10, 12] {
            let max = (1u32 << bd) - 1;
            for sz in 2..=129usize {
                for strength in 1..=3 {
                    for _ in 0..8 {
                        let base: Vec<u16> =
                            (0..sz).map(|_| (rng.next() as u32 & max) as u16).collect();
                        let mut a = base.clone();
                        let mut b = base.clone();
                        aom_dsp::intra::edge::highbd_filter_intra_edge(&mut a, sz, strength);
                        c::ref_highbd_filter_intra_edge(&mut b, 0, sz, strength);
                        assert_eq!(a, b, "tier filter bd={bd} sz={sz} s={strength}");
                    }
                }
            }
            for sz in 1..=16usize {
                for _ in 0..40 {
                    const OFF: usize = 4;
                    let n = OFF + 2 * sz + 4;
                    let base: Vec<u16> =
                        (0..n).map(|_| (rng.next() as u32 & max) as u16).collect();
                    let mut a = base.clone();
                    let mut b = base.clone();
                    aom_dsp::intra::edge::highbd_upsample_intra_edge(&mut a, OFF, sz, bd);
                    c::ref_highbd_upsample_intra_edge(&mut b, OFF, sz, bd);
                    assert_eq!(a, b, "tier upsample bd={bd} sz={sz}");
                }
            }
        }
    });
    eprintln!("{report}");
    assert!(simd_perms >= 1, "no vector tier ever ran — sweep was vacuous");
}
