//! Interleaved runtime SIMD versus forced-scalar DSP kernels.
//! The bench crate enables testable_dispatch so ARM baseline NEON can be
//! disabled. Compiler auto-vectorization of the scalar fallback is allowed.
//! Each size/kernel is its own paired group. This measures hot L2-resident
//! batches, not end-to-end codec throughput.

use std::time::Duration;

use aom_dsp::transform::{inv_txfm2d, txfm2d};
use aom_dsp::{cdef, dist, intra, loopfilter, quant};
use zenbench::prelude::*;

/// Pixels of kernel work per timed call (see the module docs on batching).
const WORK_PX: usize = 1 << 16;
/// Cap on a cell's cycled working set, so cells stay L2-resident.
const WORK_BYTES_CAP: usize = 256 << 10;

/// `tx_size_wide` / `tx_size_high` (`common_data.h`), indexed by `TX_SIZE`.
const TX_W: [usize; 19] = [
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
];
const TX_H: [usize; 19] = [
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
];

/// Transform cells spanning the size axis: squares 4x4..64x64 plus the extreme
/// aspect ratios (1:4 / 4:1), which take different driver paths.
///
/// `04x08` / `08x04` (TX_4X8 / TX_8X4) were added 2026-07-31 specifically to
/// measure the `kernel_points == 8` rung of
/// `aom_dsp::transform::simd::half_batch_pays`, whose aarch64 threshold was
/// INTERPOLATED between the measured 4-point and 16-point cells because this
/// grid had no 4x8 cell. `04x08` is the 4-wide (half-batch) column at
/// `kernel_points = 8` — i.e. the exact rung the threshold sits on; `08x04` is
/// the complementary shape (full-width lanes, 4-point kernel).
const TX_CELLS: &[(usize, &str)] = &[
    (0, "04x04"),
    (1, "08x08"),
    (2, "16x16"),
    (3, "32x32"),
    (4, "64x64"),
    (5, "04x08"),
    (6, "08x04"),
    (8, "16x08"),
    (13, "04x16"),
    (16, "32x08"),
];

const DCT_DCT: usize = 0;
const ADST_ADST: usize = 3;

/// How many kernel calls one timed batch makes, and how many distinct block
/// slots the cycled working set holds.
fn batch(w: usize, h: usize, bytes_per_px: usize) -> (usize, usize) {
    let reps = (WORK_PX / (w * h)).max(1);
    let slots = (WORK_BYTES_CAP / (w * h * bytes_per_px)).clamp(1, reps);
    (reps, slots)
}

/// Deterministic pseudo-random fill — an LCG, so cells are reproducible across
/// runs and machines (a fixed seed is what makes saved baselines comparable).
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// Coefficient-shaped values: small, signed, with a heavy DC — the
    /// distribution the transform's eob-driven paths actually see.
    fn coeff(&mut self, i: usize) -> i32 {
        let v = (self.next_u32() % 512) as i32 - 256;
        if i == 0 {
            v * 8
        } else {
            v
        }
    }
    fn pixel(&mut self) -> u8 {
        (self.next_u32() >> 8) as u8
    }
}

fn tune(g: &mut BenchGroup) {
    g.config()
        .min_rounds(12)
        .max_rounds(200)
        .warmup_time(Duration::from_millis(250))
        .max_time(Duration::from_secs(10))
        .max_wall_time(Duration::from_secs(60));
}

struct TierGroups<'a> {
    suite: &'a mut Suite,
    prefix: &'static str,
}

fn set_simd(enabled: bool) {
    #[cfg(target_arch = "aarch64")]
    archmage::NeonToken::dangerously_disable_token_process_wide(!enabled)
        .expect("testable ARM dispatch");
    #[cfg(target_arch = "x86_64")]
    archmage::X64V2Token::dangerously_disable_token_process_wide(!enabled)
        .expect("testable x86 dispatch");
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    panic!("tier comparison requires ARM64 or x86-64, enabled={enabled}");
}

impl TierGroups<'_> {
    fn bench<F>(&mut self, name: impl Into<String>, f: F)
    where
        F: FnMut(&mut Bencher) + Clone + Send + 'static,
    {
        self.suite
            .compare(format!("{}/{}", self.prefix, name.into()), |g| {
                g.throughput_unit(if self.prefix.starts_with("quant") {
                    "coeff"
                } else {
                    "px"
                });
                g.throughput(Throughput::Elements(WORK_PX as u64));
                tune(g);
                for (label, enabled) in [("native_simd", true), ("forced_scalar", false)] {
                    let mut run = f.clone();
                    g.bench(label, move |b| {
                        set_simd(enabled);
                        run(b);
                    });
                }
            });
    }
}

fn tier_group(suite: &mut Suite, prefix: &'static str, f: impl FnOnce(&mut TierGroups<'_>)) {
    f(&mut TierGroups { suite, prefix });
}

/// bd8 inverse transform (`av1_inv_txfm2d_add_u8`) — the decode hot path.
fn bench_inv_txfm_u8(suite: &mut Suite) {
    tier_group(suite, "inv_txfm_u8", |g| {
        for &(tx_size, name) in TX_CELLS {
            for (tx_type, tname) in [(DCT_DCT, "dct"), (ADST_ADST, "adst")] {
                if !inv_txfm2d::inv_txfm_valid(tx_type, tx_size) {
                    continue;
                }
                let (w, h) = (TX_W[tx_size], TX_H[tx_size]);
                let (reps, slots) = batch(w, h, 1);
                g.bench(format!("{name}_{tname}"), move |b| {
                    let mut rng = Rng(0x5EED_0001 ^ (tx_size as u64) << 8 ^ tx_type as u64);
                    let n = inv_txfm2d::inv_input_len(tx_size);
                    let input: Vec<i32> = (0..n * slots).map(|i| rng.coeff(i % n)).collect();
                    let dst: Vec<u8> = (0..w * h * slots).map(|_| rng.pixel()).collect();
                    b.with_input(move || (input.clone(), dst.clone())).run(
                        move |(input, mut dst)| {
                            for r in 0..reps {
                                let s = r % slots;
                                inv_txfm2d::av1_inv_txfm2d_add_u8(
                                    &input[s * n..][..n],
                                    &mut dst[s * w * h..][..w * h],
                                    w,
                                    tx_type,
                                    tx_size,
                                );
                            }
                            dst
                        },
                    );
                });
            }
        }
    });
}

/// High-bit-depth inverse transform (`av1_inv_txfm2d_add`, bd10).
fn bench_inv_txfm_hbd(suite: &mut Suite) {
    tier_group(suite, "inv_txfm_hbd10", |g| {
        for &(tx_size, name) in TX_CELLS {
            if !inv_txfm2d::inv_txfm_valid(DCT_DCT, tx_size) {
                continue;
            }
            let (w, h) = (TX_W[tx_size], TX_H[tx_size]);
            let (reps, slots) = batch(w, h, 2);
            g.bench(name, move |b| {
                let mut rng = Rng(0x5EED_0002 ^ (tx_size as u64) << 8);
                let n = inv_txfm2d::inv_input_len(tx_size);
                let input: Vec<i32> = (0..n * slots).map(|i| rng.coeff(i % n)).collect();
                let dst: Vec<u16> = (0..w * h * slots)
                    .map(|_| (rng.next_u32() % 1024) as u16)
                    .collect();
                b.with_input(move || (input.clone(), dst.clone()))
                    .run(move |(input, mut dst)| {
                        for r in 0..reps {
                            let s = r % slots;
                            inv_txfm2d::av1_inv_txfm2d_add(
                                &input[s * n..][..n],
                                &mut dst[s * w * h..][..w * h],
                                w,
                                DCT_DCT,
                                tx_size,
                                10,
                            );
                        }
                        dst
                    });
            });
        }
    });
}

/// Forward transform (`av1_fwd_txfm2d`) — the encode hot path.
fn bench_fwd_txfm(suite: &mut Suite) {
    tier_group(suite, "fwd_txfm", |g| {
        for &(tx_size, name) in TX_CELLS {
            for (tx_type, tname) in [(DCT_DCT, "dct"), (ADST_ADST, "adst")] {
                if !txfm2d::fwd_txfm_valid(tx_type, tx_size) {
                    continue;
                }
                let (w, h) = (TX_W[tx_size], TX_H[tx_size]);
                let (reps, slots) = batch(w, h, 2 + 4);
                g.bench(format!("{name}_{tname}"), move |b| {
                    let mut rng = Rng(0x5EED_0003 ^ (tx_size as u64) << 8 ^ tx_type as u64);
                    // Residual domain: signed, |v| < 2^9 for 8-bit input.
                    let input: Vec<i16> = (0..w * h * slots)
                        .map(|_| (rng.next_u32() % 512) as i16 - 256)
                        .collect();
                    let out = vec![0i32; w * h * slots];
                    b.with_input(move || (input.clone(), out.clone())).run(
                        move |(input, mut out)| {
                            for r in 0..reps {
                                let s = r % slots;
                                txfm2d::av1_fwd_txfm2d(
                                    &input[s * w * h..][..w * h],
                                    &mut out[s * w * h..][..w * h],
                                    w,
                                    tx_type,
                                    tx_size,
                                );
                            }
                            out
                        },
                    );
                });
            }
        }
    });
}

/// CDEF — the in-loop directional filter (bd8 block filter + direction search).
fn bench_cdef(suite: &mut Suite) {
    // The CDEF input buffer is the decode driver's bordered u16 staging plane.
    // The stride MUST be `CDEF_BSTRIDE` — the direction tap tables
    // (`cdef_directions`) bake that stride into their offsets, so any other
    // value reads the wrong neighbours. Origin is (CDEF_VBORDER=2,
    // CDEF_HBORDER=8), matching `cdef::frame`'s staging buffer.
    const STRIDE: usize = cdef::CDEF_BSTRIDE;
    const VB: usize = 2;
    const HB: usize = 8;
    tier_group(suite, "cdef", |g| {
        for (bw, bh, name) in [(4usize, 4usize, "04x04"), (8, 8, "08x08")] {
            let reps = WORK_PX / (bw * bh);
            g.bench(format!("filter_u8_{name}"), move |b| {
                let mut rng = Rng(0x5EED_0004 ^ (bw as u64) << 8);
                // Rows of slack past the borders: the secondary taps reach 2
                // rows out and the directional offsets add up to 2 more.
                let in_buf: Vec<u16> = (0..STRIDE * (bh + 2 * VB + 4))
                    .map(|_| (rng.next_u32() % 256) as u16)
                    .collect();
                let dst = vec![0u8; bw * bh + 64];
                let in_off = VB * STRIDE + HB;
                b.with_input(move || (in_buf.clone(), dst.clone())).run(
                    move |(in_buf, mut dst)| {
                        for r in 0..reps {
                            cdef::cdef_filter_block_u8(
                                &mut dst,
                                0,
                                bw,
                                &in_buf,
                                in_off,
                                15,
                                8,
                                (r % 8) as i32,
                                3,
                                3,
                                0,
                                bw,
                                bh,
                                true,
                                true,
                            );
                        }
                        dst
                    },
                );
            });
        }
        let dir_reps = WORK_PX / 64;
        g.bench("find_dir_08x08", move |b| {
            let mut rng = Rng(0x5EED_0005);
            let img: Vec<u16> = (0..9 * STRIDE)
                .map(|_| (rng.next_u32() % 256) as u16)
                .collect();
            b.with_input(move || img.clone()).run(move |img| {
                let mut acc = 0i32;
                for _ in 0..dir_reps {
                    let (d, v) = cdef::cdef_find_dir(&img, STRIDE, 0);
                    acc = acc.wrapping_add(d).wrapping_add(v);
                }
                acc
            });
        });
    });
}

/// Deblocking loop filter — horizontal + vertical edges at every filter width.
fn bench_loopfilter(suite: &mut Suite) {
    const STRIDE: usize = 64;
    const ROWS: usize = 64;
    tier_group(suite, "loopfilter", |g| {
        for width in [4u32, 8, 14] {
            // One call filters a `width`-tap edge across a 4-sample run.
            let reps = WORK_PX / (width as usize * 4);
            for (dir, is_h) in [("h", true), ("v", false)] {
                g.bench(format!("{dir}_w{width:02}"), move |b| {
                    let mut rng = Rng(0x5EED_0006 ^ width as u64);
                    let buf: Vec<u8> = (0..STRIDE * ROWS).map(|_| rng.pixel()).collect();
                    let p = if is_h { STRIDE } else { 1 };
                    b.with_input(move || buf.clone()).run(move |mut buf| {
                        for r in 0..reps {
                            // Walk the edge position so the batch touches the
                            // whole plane instead of one hot cache line.
                            let center = STRIDE * (16 + (r % 32)) + 32;
                            if is_h {
                                loopfilter::horizontal(width, &mut buf, center, p, 20, 12, 8);
                            } else {
                                loopfilter::vertical(width, &mut buf, center, p, 20, 12, 8);
                            }
                        }
                        buf
                    });
                });
            }
        }
    });
}

/// Distortion metrics — SAD / SSE, the RD search's highest-call-count kernels.
fn bench_dist(suite: &mut Suite) {
    const STRIDE: usize = 64;
    tier_group(suite, "dist_scalar_control", |g| {
        for (w, h, name) in [
            (4usize, 4usize, "04x04"),
            (8, 8, "08x08"),
            (16, 16, "16x16"),
            (32, 32, "32x32"),
            (64, 64, "64x64"),
        ] {
            let reps = WORK_PX / (w * h);
            g.bench(format!("sad_{name}"), move |b| {
                let mut rng = Rng(0x5EED_0007 ^ (w as u64) << 8);
                let a: Vec<u8> = (0..STRIDE * 64).map(|_| rng.pixel()).collect();
                let c: Vec<u8> = (0..STRIDE * 64).map(|_| rng.pixel()).collect();
                b.with_input(move || (a.clone(), c.clone()))
                    .run(move |(a, c)| {
                        let mut acc = 0u64;
                        for _ in 0..reps {
                            acc += dist::sad(&a, STRIDE, &c, STRIDE, w, h) as u64;
                        }
                        acc
                    });
            });
            g.bench(format!("sse_{name}"), move |b| {
                let mut rng = Rng(0x5EED_0008 ^ (w as u64) << 8);
                let a: Vec<u8> = (0..STRIDE * 64).map(|_| rng.pixel()).collect();
                let c: Vec<u8> = (0..STRIDE * 64).map(|_| rng.pixel()).collect();
                b.with_input(move || (a.clone(), c.clone()))
                    .run(move |(a, c)| {
                        let mut acc = 0i64;
                        for _ in 0..reps {
                            acc += dist::sse(&a, STRIDE, &c, STRIDE, w, h);
                        }
                        acc
                    });
            });
            g.bench(format!("highbd_sse_{name}"), move |b| {
                let mut rng = Rng(0x5EED_0009 ^ (w as u64) << 8);
                let a: Vec<u16> = (0..STRIDE * 64)
                    .map(|_| (rng.next_u32() % 1024) as u16)
                    .collect();
                let c: Vec<u16> = (0..STRIDE * 64)
                    .map(|_| (rng.next_u32() % 1024) as u16)
                    .collect();
                b.with_input(move || (a.clone(), c.clone()))
                    .run(move |(a, c)| {
                        let mut acc = 0i64;
                        for _ in 0..reps {
                            acc += dist::highbd_sse(&a, STRIDE, &c, STRIDE, w, h);
                        }
                        acc
                    });
            });
        }
    });
}

/// Quantization — `av1_quantize_fp` across the block-size axis.
// The plain dist::{sad,sse,highbd_sse} entries are scalar control functions.
// Measure the separate runtime-dispatched SAD entry explicitly as well.
fn bench_sad_dispatch(suite: &mut Suite) {
    for w in [4usize, 8, 16, 32, 64] {
        const STRIDE: usize = 64;
        let mut rng = Rng(0x5EED_0007 ^ (w as u64) << 8);
        let a: Vec<u8> = (0..STRIDE * 64).map(|_| rng.pixel()).collect();
        let c: Vec<u8> = (0..STRIDE * 64).map(|_| rng.pixel()).collect();
        let expected = dist::sad(&a, STRIDE, &c, STRIDE, w, w);
        for enabled in [false, true] {
            set_simd(enabled);
            assert_eq!(dist::simd::sad_simd(&a, STRIDE, &c, STRIDE, w, w), expected);
        }
        suite.compare(format!("dist_dispatch/sad_{w:02}x{w:02}"), |g| {
            tune(g);
            g.throughput(Throughput::Elements(WORK_PX as u64));
            for (label, enabled, reference) in [
                ("runtime_enabled", true, false),
                ("forced_scalar", false, false),
                ("scalar_reference", true, true),
            ] {
                let a = a.clone();
                let c = c.clone();
                g.bench(label, move |b| {
                    set_simd(enabled);
                    b.iter(|| {
                        let mut sum = 0u64;
                        for _ in 0..WORK_PX / (w * w) {
                            sum += if reference {
                                dist::sad(black_box(&a), STRIDE, black_box(&c), STRIDE, w, w)
                            } else {
                                dist::simd::sad_simd(
                                    black_box(&a),
                                    STRIDE,
                                    black_box(&c),
                                    STRIDE,
                                    w,
                                    w,
                                )
                            } as u64;
                        }
                        sum
                    });
                });
            }
        });
    }
}

fn bench_quant(suite: &mut Suite) {
    tier_group(suite, "quant_scalar_control", |g| {
        for &(tx_size, name) in &[(0usize, "04x04"), (1, "08x08"), (2, "16x16"), (3, "32x32")] {
            let n = TX_W[tx_size] * TX_H[tx_size];
            let reps = WORK_PX / n;
            g.bench(format!("fp_{name}"), move |b| {
                let mut rng = Rng(0x5EED_000A ^ (tx_size as u64) << 8);
                let coeff: Vec<i32> = (0..n).map(|i| rng.coeff(i) * 4).collect();
                let scan: Vec<i16> = (0..n as i16).collect();
                let q = vec![0i32; n];
                let dq = vec![0i32; n];
                b.with_input(move || (coeff.clone(), scan.clone(), q.clone(), dq.clone()))
                    .run(move |(coeff, scan, mut q, mut dq)| {
                        let mut acc = 0u32;
                        for _ in 0..reps {
                            acc += quant::av1_quantize_fp(
                                &coeff,
                                &[13, 13],
                                &[0x4000, 0x4000],
                                &[16, 16],
                                &mut q,
                                &mut dq,
                                &scan,
                            ) as u32;
                        }
                        (acc, q, dq)
                    });
            });
        }
    });
}

/// Intra prediction — the compute-heavy predictors across the size axis.
fn bench_intra(suite: &mut Suite) {
    tier_group(suite, "intra_scalar_control", |g| {
        // SMOOTH* / PAETH are the compute-heavy predictors; V/H are
        // memory-bound copies, kept as the control.
        for (mode, mname) in [
            (intra::V, "v"),
            (intra::H, "h"),
            (intra::PAETH, "paeth"),
            (intra::SMOOTH, "smooth"),
            (intra::SMOOTH_V, "smooth_v"),
        ] {
            for (bw, bh, name) in [
                (4usize, 4usize, "04x04"),
                (16, 16, "16x16"),
                (32, 32, "32x32"),
            ] {
                let reps = WORK_PX / (bw * bh);
                g.bench(format!("{mname}_{name}"), move |b| {
                    let mut rng = Rng(0x5EED_000B ^ (bw as u64) << 8 ^ mode as u64);
                    let above: Vec<u8> = (0..bw + 2 * bh + 2).map(|_| rng.pixel()).collect();
                    let left: Vec<u8> = (0..bh + bw).map(|_| rng.pixel()).collect();
                    let dst = vec![0u8; bw * bh];
                    b.with_input(move || (above.clone(), left.clone(), dst.clone()))
                        .run(move |(above, left, mut dst)| {
                            for _ in 0..reps {
                                intra::predict(
                                    mode,
                                    &mut dst,
                                    bw,
                                    bw,
                                    bh,
                                    &intra::AboveRef(&above),
                                    &left,
                                );
                            }
                            dst
                        });
                });
            }
        }
    });
}

fn bench_quant_dispatch(suite: &mut Suite) {
    tier_group(suite, "quant_dispatch", |g| {
        for &(tx_size, name) in &[(0usize, "04x04"), (1, "08x08"), (2, "16x16"), (3, "32x32")] {
            let n = TX_W[tx_size] * TX_H[tx_size];
            let reps = WORK_PX / n;
            g.bench(format!("fp_{name}"), move |b| {
                let mut rng = Rng(0x5EED_000A ^ (tx_size as u64) << 8);
                let coeff: Vec<i32> = (0..n).map(|i| rng.coeff(i) * 4).collect();
                let scan: Vec<i16> = (0..n as i16).collect();
                let q = vec![0i32; n];
                let dq = vec![0i32; n];
                let mut want_q = q.clone();
                let mut want_dq = dq.clone();
                let want_eob = quant::av1_quantize_fp(
                    &coeff,
                    &[13, 13],
                    &[0x4000, 0x4000],
                    &[16, 16],
                    &mut want_q,
                    &mut want_dq,
                    &scan,
                );
                let mut got_q = q.clone();
                let mut got_dq = dq.clone();
                let got_eob = quant::simd::av1_quantize_fp_no_qmatrix_dispatch(
                    &[0x4000, 0x4000],
                    &[16, 16],
                    &[13, 13],
                    0,
                    &scan,
                    &scan,
                    &coeff,
                    &mut got_q,
                    &mut got_dq,
                );
                assert_eq!((got_eob, got_q, got_dq), (want_eob, want_q, want_dq));
                b.with_input(move || (coeff.clone(), scan.clone(), q.clone(), dq.clone()))
                    .run(move |(coeff, scan, mut q, mut dq)| {
                        let mut acc = 0u32;
                        for _ in 0..reps {
                            acc += quant::simd::av1_quantize_fp_no_qmatrix_dispatch(
                                &[0x4000, 0x4000],
                                &[16, 16],
                                &[13, 13],
                                0,
                                &scan,
                                &scan,
                                &coeff,
                                &mut q,
                                &mut dq,
                            ) as u32;
                        }
                        (acc, q, dq)
                    });
            });
        }
    });
}

fn bench_intra_dispatch(suite: &mut Suite) {
    tier_group(suite, "intra_dispatch", |g| {
        // SMOOTH* / PAETH are the compute-heavy predictors; V/H are
        // memory-bound copies, kept as the control.
        for (mode, mname) in [
            (intra::V, "v"),
            (intra::H, "h"),
            (intra::PAETH, "paeth"),
            (intra::SMOOTH, "smooth"),
            (intra::SMOOTH_V, "smooth_v"),
        ] {
            for (bw, bh, name) in [
                (4usize, 4usize, "04x04"),
                (16, 16, "16x16"),
                (32, 32, "32x32"),
            ] {
                let reps = WORK_PX / (bw * bh);
                g.bench(format!("{mname}_{name}"), move |b| {
                    let mut rng = Rng(0x5EED_000B ^ (bw as u64) << 8 ^ mode as u64);
                    let above: Vec<u16> = (0..bw + 2 * bh + 2)
                        .map(|_| u16::from(rng.pixel()))
                        .collect();
                    let left: Vec<u16> = (0..bh + bw).map(|_| u16::from(rng.pixel())).collect();
                    let dst = vec![0u16; bw * bh];
                    let mut want = dst.clone();
                    let mut got = dst.clone();
                    intra::predict_highbd_scalar(
                        mode,
                        &mut want,
                        bw,
                        bw,
                        bh,
                        &intra::AboveRef16(&above),
                        &left,
                        8,
                    );
                    intra::predict_highbd(
                        mode,
                        &mut got,
                        bw,
                        bw,
                        bh,
                        &intra::AboveRef16(&above),
                        &left,
                        8,
                    );
                    assert_eq!(got, want);
                    b.with_input(move || (above.clone(), left.clone(), dst.clone()))
                        .run(move |(above, left, mut dst)| {
                            for _ in 0..reps {
                                intra::predict_highbd(
                                    mode,
                                    &mut dst,
                                    bw,
                                    bw,
                                    bh,
                                    &intra::AboveRef16(&above),
                                    &left,
                                    8,
                                );
                            }
                            dst
                        });
                });
            }
        }
    });
}

fn main() {
    assert!(
        !aom_dsp::dispatch::scalar_forced(),
        "remove AOM_FORCE_SCALAR for paired tiers"
    );
    set_simd(false);
    set_simd(true);
    let group_filter: Option<String> =
        std::env::args().find_map(|a| a.strip_prefix("--group=").map(String::from));
    let result = zenbench::run(|suite: &mut Suite| {
        if let Some(f) = group_filter {
            suite.set_group_filter(f);
        }
        bench_inv_txfm_u8(suite);
        bench_inv_txfm_hbd(suite);
        bench_fwd_txfm(suite);
        bench_cdef(suite);
        bench_loopfilter(suite);
        bench_dist(suite);
        bench_sad_dispatch(suite);
        bench_quant_dispatch(suite);
        bench_intra_dispatch(suite);
        bench_quant(suite);
        bench_intra(suite);
    });
    zenbench::postprocess_result(&result);
}
