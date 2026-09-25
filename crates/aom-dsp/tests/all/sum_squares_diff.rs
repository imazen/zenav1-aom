//! Differential harness for the residual-energy kernels (aom_sum_squares_i16 +
//! aom_sum_squares_2d_i16) vs C libaom: ss = sum(v*v) over i16 values, 1-D and
//! 2-D-strided. i16 magnitudes cover the residual range (up to 12-bit).
//!
//! Tier note: the port's v3 kernel is an instruction-level mirror of the REAL
//! `aom_sum_squares_2d_i16_avx2` dispatcher — whose madd_epi16/add_epi32
//! accumulation wraps in i32 where `aom_sum_squares_2d_i16_c` stays exact
//! (reachable only via adjacent `i16::MIN` pairs, never from encoder
//! residuals). When v3 is live the oracle is therefore the exported AVX2
//! symbol; every other tier keeps `_c`.

use aom_dsp::dist::{sum_squares_2d_i16, sum_squares_i16};
use aom_sys_ref as c;
#[cfg(target_arch = "x86_64")]
use archmage::SimdToken;

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
    fn v16(&mut self) -> i16 {
        (self.next() % 65536) as i16 // full i16 (wraps); covers residual magnitudes
    }
    fn range(&mut self, hi: u32) -> u32 {
        (self.next() % hi as u64) as u32
    }
}

#[test]
fn sum_squares_i16_differential() {
    let mut rng = Rng(0x0055_c0de_9e37_79b9);
    for &n in &[1usize, 16, 64, 256, 1024, 4096] {
        for _ in 0..2000 {
            let src: Vec<i16> = (0..n).map(|_| rng.v16()).collect();
            assert_eq!(
                sum_squares_i16(&src),
                c::ref_sum_squares_i16(&src),
                "1d n={n}"
            );
        }
    }
}

/// True when the dispatched `sum_squares_2d_i16` runs the v3 (C-avx2 mirror)
/// tier: a v3-capable x86-64 host and no scalar pin. The token-testing lock is
/// held by the caller for the whole comparison, so a concurrent
/// `for_each_token_permutation` cannot flip availability mid-test.
#[cfg(target_arch = "x86_64")]
fn v3_live() -> bool {
    archmage::X64V3Token::summon().is_some() && std::env::var_os("AOM_FORCE_SCALAR").is_none()
}

/// Buffer with a guaranteed 16-byte-aligned base — C's w=8 arm
/// (`aom_sum_squares_2d_i16_nxn_sse2`) uses `movdqa` row loads and faults on
/// anything less, a contract its real callers satisfy implicitly.
#[repr(align(16))]
struct Aligned<const N: usize>([i16; N]);

fn aligned_src<const N: usize>(fill: impl FnMut(usize) -> i16) -> Aligned<N> {
    let mut fill = fill;
    let mut a = Aligned([0i16; N]);
    for (i, v) in a.0.iter_mut().enumerate() {
        *v = fill(i);
    }
    a
}

#[test]
fn sum_squares_2d_i16_differential() {
    // Held for the test body: pins the v3-live observation against
    // process-wide token permutation by sibling tests.
    #[cfg(target_arch = "x86_64")]
    let _token_guard = archmage::testing::lock_token_testing();
    #[cfg(target_arch = "x86_64")]
    let avx2 = v3_live();
    let mut rng = Rng(0x0055_c057_0000_b111);
    const DIMS: [(usize, usize); 8] = [
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (4, 16),
        (16, 4),
        (8, 32),
    ];
    for &(w, h) in &DIMS {
        for _ in 0..2000 {
            // The C-avx2 dispatcher's w=8 arm needs stride % 8 == 0 (its row
            // loads are aligned 16-byte ops); other arms take any stride.
            let stride = if w == 8 {
                w + 8 * rng.range(2) as usize
            } else {
                w + rng.range(5) as usize
            };
            let src = aligned_src::<8192>(|i| if i < h * stride { rng.v16() } else { 0 });
            #[cfg(target_arch = "x86_64")]
            let want = if avx2 {
                c::ref_sum_squares_2d_i16_avx2(&src.0, stride, w, h)
            } else {
                c::ref_sum_squares_2d_i16(&src.0, stride, w, h)
            };
            #[cfg(not(target_arch = "x86_64"))]
            let want = c::ref_sum_squares_2d_i16(&src.0, stride, w, h);
            assert_eq!(
                sum_squares_2d_i16(&src.0, stride, w, h),
                want,
                "2d w={w} h={h} stride={stride}"
            );
        }
    }
}

/// The v3 kernel against the REAL `aom_sum_squares_2d_i16_avx2` on the full
/// adversarial domain — including `i16::MIN` pairs that wrap its i32
/// accumulation differently per shape arm (4x4 sign-extends the i32 total,
/// 4xn wraps across all groups, nxn wraps within each 4-row group). Shapes
/// covering every dispatch branch plus non-multiple fallbacks.
#[cfg(target_arch = "x86_64")]
#[test]
fn sum_squares_2d_i16_v3_matches_real_avx2_dispatcher() {
    let _token_guard = archmage::testing::lock_token_testing();
    if !v3_live() {
        eprintln!("no v3 token on this host; skipping");
        return;
    }
    let mut rng = Rng(0xa11c_e55e_0000_d1ff);
    const DIMS: [(usize, usize); 16] = [
        (4, 4),
        (4, 8),
        (4, 16),
        (4, 64),
        (8, 4),
        (8, 8),
        (8, 32),
        (16, 4),
        (16, 16),
        (32, 8),
        (32, 32),
        (64, 64),
        (128, 4),
        (16, 5), // h % 4 != 0 -> C-c arm
        (12, 8), // w % 16 != 0 -> C-c arm
        (6, 10), // fully odd -> C-c arm
    ];
    for &(w, h) in &DIMS {
        for rep in 0..500 {
            let stride = if w == 8 {
                w + 8 * rng.range(2) as usize
            } else {
                w + rng.range(5) as usize
            };
            let mut src = aligned_src::<8192>(|i| if i < h * stride { rng.v16() } else { 0 });
            match rep % 4 {
                // Salt the block with i16::MIN so adjacent-pair wraps fire.
                1 => {
                    for v in src.0.iter_mut().take(h * stride).step_by(3) {
                        *v = i16::MIN;
                    }
                }
                2 => {
                    for v in src.0.iter_mut().take(h * stride) {
                        *v = i16::MIN;
                    }
                }
                _ => {}
            }
            assert_eq!(
                sum_squares_2d_i16(&src.0, stride, w, h),
                c::ref_sum_squares_2d_i16_avx2(&src.0, stride, w, h),
                "v3 w={w} h={h} stride={stride} rep={rep}"
            );
        }
    }

    // The w=8 arm is unreachable to C-avx2 without a 16-byte-aligned base and
    // stride % 8 == 0 (movdqa). The port falls back to the exact scalar loop
    // there — which is C-c's result — and must not fault.
    let src = aligned_src::<256>(|i| i as i16 * 7 - 300);
    let misaligned = &src.0[1..41]; // 8x4 block, stride 8, base +2B
    assert_eq!(
        sum_squares_2d_i16(misaligned, 8, 8, 4),
        c::ref_sum_squares_2d_i16(misaligned, 8, 8, 4),
        "w=8 misaligned base -> scalar (C-c) result"
    );
    let odd_stride = &src.0[..40]; // 8x4 block, stride 10
    assert_eq!(
        sum_squares_2d_i16(odd_stride, 10, 8, 4),
        c::ref_sum_squares_2d_i16(odd_stride, 10, 8, 4),
        "w=8 stride%8!=0 -> scalar (C-c) result"
    );
}
