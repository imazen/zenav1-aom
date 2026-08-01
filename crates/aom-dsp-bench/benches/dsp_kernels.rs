//! Port-only DSP kernel benchmarks — the SIMD-lever measurement harness.
//!
//! Unlike `aom-bench`'s `gate3` (port vs the REAL libaom C oracle), this bench
//! needs NO C reference: it times the port's own public dispatch entry points.
//! That makes it the harness for **per-architecture SIMD work**, where the
//! question is "what does this kernel gain from its vector path on THIS CPU",
//! not "how do we compare to C". It runs unchanged on x86-64, aarch64, and
//! wasm32 — which the Gate-3 oracle harness cannot do (`aom-sys-ref` needs the
//! pinned libaom submodule + cmake, and on a dev box it may not be built).
//!
//! # How to compare two builds
//!
//! The intended use is a **before/after baseline around your own change**:
//!
//! ```text
//! # on the commit before the optimization
//! cargo bench -p zenav1-aom-dsp-bench --bench dsp_kernels -- --save-baseline=before
//! # after the optimization
//! cargo bench -p zenav1-aom-dsp-bench --bench dsp_kernels -- --baseline=before
//! ```
//!
//! Within a single run, rows are interleaved, so same-run comparisons (e.g.
//! across tx sizes) are paired and thermally sound.
//!
//! ## `AOM_FORCE_SCALAR=1` is NOT a valid scalar baseline on aarch64
//!
//! MEASURED 2026-07-25 on an Apple M4 Pro: `AOM_FORCE_SCALAR=1` (the
//! `aom_dsp::dispatch` pin) is a **no-op for the NEON tier**, so a
//! pinned-vs-unpinned pair on ARM measures the SAME code twice. `neon` is a
//! compile-time-guaranteed baseline feature of `aarch64-apple-darwin` (and of
//! aarch64 generally), and archmage refuses to disable compile-time-guaranteed
//! tokens: `NeonToken::dangerously_disable_token_process_wide(true)` returns
//! `Err`, and `NeonToken::summon()` keeps returning `Some` afterwards (verified
//! directly). A full 77-row pinned-vs-unpinned pair on this box came back with
//! every row inside ±3% — noise, not a scalar/SIMD delta.
//!
//! So on aarch64 do NOT read `AOM_FORCE_SCALAR` rows as "the scalar baseline".
//! The pin still works as documented on x86-64, where every tier above the
//! `sse2` baseline is runtime-detected and therefore disableable.
//!
//! A second consequence worth keeping in mind when reading ARM numbers: because
//! NEON is baseline, the `_scalar` variants are themselves compiled with NEON
//! available, so LLVM auto-vectorizes them. On ARM the interesting question is
//! not "scalar vs vector" but "does this kernel exploit structure LLVM cannot
//! find on its own" — chiefly batching independent work across lanes (e.g. the
//! transform's 8-columns-at-once passes, which the per-column scalar driver
//! loop structurally prevents LLVM from forming).
//!
//! # Why every cell batches to a fixed pixel budget
//!
//! A single 4x4 inverse transform is tens of ns — below useful resolution once
//! the timer (41ns on an M4 Pro) and per-call overhead are accounted for; an
//! unbatched first cut of this bench measured CV ~50%. Each cell therefore runs
//! [`WORK_PX`] pixels' worth of back-to-back kernel calls, which puts every row
//! in the tens-of-µs range and makes the batch a fair stand-in for the
//! frame-level loops that call these kernels thousands of times per tile.
//! Throughput is reported over the whole batch, so `px/s` is comparable across
//! cells of different block sizes.
//!
//! The working set is capped at [`WORK_BYTES_CAP`] and cycled, keeping cells
//! L2-resident: this measures kernel COMPUTE (the thing SIMD changes), not DRAM
//! bandwidth. Kernels that are memory-bound at frame scale will show a smaller
//! end-to-end win than their row here suggests — read this harness as a
//! per-kernel lever and `gate3` as the end-to-end truth.
//!
//! # Size sweep
//!
//! Every group sweeps the transform/block-size axis from tiny (4x4) to large
//! (64x64), including the extreme 1:4 / 4:1 aspect ratios, so per-call fixed
//! overhead is separable from per-pixel work per the sweep discipline in
//! CLAUDE.md. Bit depth is swept where the kernel has an hbd path.

use std::time::Duration;

use aom_dsp::transform::{inv_txfm2d, txfm2d};
use aom_dsp::{cdef, dist, intra, loopfilter, quant};
use zenbench::prelude::*;

/// Pixels of kernel work per timed call (see the module docs on batching).
const WORK_PX: usize = 1 << 16;
/// Cap on a cell's cycled working set, so cells stay L2-resident.
const WORK_BYTES_CAP: usize = 256 << 10;

/// `tx_size_wide` / `tx_size_high` (`common_data.h`), indexed by `TX_SIZE`.
const TX_W: [usize; 19] = [4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64];
const TX_H: [usize; 19] = [4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16];

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
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// Coefficient-shaped values: small, signed, with a heavy DC — the
    /// distribution the transform's eob-driven paths actually see.
    fn coeff(&mut self, i: usize) -> i32 {
        let v = (self.next_u32() % 512) as i32 - 256;
        if i == 0 { v * 8 } else { v }
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

/// bd8 inverse transform (`av1_inv_txfm2d_add_u8`) — the decode hot path.
fn bench_inv_txfm_u8(suite: &mut Suite) {
    suite.group("inv_txfm_u8", |g| {
        g.throughput_unit("px");
        g.throughput(Throughput::Elements(WORK_PX as u64));
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
        tune(g);
    });
}

/// High-bit-depth inverse transform (`av1_inv_txfm2d_add`, bd10).
fn bench_inv_txfm_hbd(suite: &mut Suite) {
    suite.group("inv_txfm_hbd10", |g| {
        g.throughput_unit("px");
        g.throughput(Throughput::Elements(WORK_PX as u64));
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
                let dst: Vec<u16> =
                    (0..w * h * slots).map(|_| (rng.next_u32() % 1024) as u16).collect();
                b.with_input(move || (input.clone(), dst.clone())).run(move |(input, mut dst)| {
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
        tune(g);
    });
}

/// Forward transform (`av1_fwd_txfm2d`) — the encode hot path.
fn bench_fwd_txfm(suite: &mut Suite) {
    suite.group("fwd_txfm", |g| {
        g.throughput_unit("px");
        g.throughput(Throughput::Elements(WORK_PX as u64));
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
                    let input: Vec<i16> =
                        (0..w * h * slots).map(|_| (rng.next_u32() % 512) as i16 - 256).collect();
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
        tune(g);
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
    suite.group("cdef", |g| {
        g.throughput_unit("px");
        g.throughput(Throughput::Elements(WORK_PX as u64));
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
            let img: Vec<u16> = (0..9 * STRIDE).map(|_| (rng.next_u32() % 256) as u16).collect();
            b.with_input(move || img.clone()).run(move |img| {
                let mut acc = 0i32;
                for _ in 0..dir_reps {
                    let (d, v) = cdef::cdef_find_dir(&img, STRIDE, 0);
                    acc = acc.wrapping_add(d).wrapping_add(v);
                }
                acc
            });
        });
        tune(g);
    });
}

/// Deblocking loop filter — horizontal + vertical edges at every filter width.
fn bench_loopfilter(suite: &mut Suite) {
    const STRIDE: usize = 64;
    const ROWS: usize = 64;
    suite.group("loopfilter", |g| {
        g.throughput_unit("px");
        g.throughput(Throughput::Elements(WORK_PX as u64));
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
        tune(g);
    });
}

/// Distortion metrics — SAD / SSE, the RD search's highest-call-count kernels.
fn bench_dist(suite: &mut Suite) {
    const STRIDE: usize = 64;
    suite.group("dist", |g| {
        g.throughput_unit("px");
        g.throughput(Throughput::Elements(WORK_PX as u64));
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
                b.with_input(move || (a.clone(), c.clone())).run(move |(a, c)| {
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
                b.with_input(move || (a.clone(), c.clone())).run(move |(a, c)| {
                    let mut acc = 0i64;
                    for _ in 0..reps {
                        acc += dist::sse(&a, STRIDE, &c, STRIDE, w, h);
                    }
                    acc
                });
            });
            g.bench(format!("highbd_sse_{name}"), move |b| {
                let mut rng = Rng(0x5EED_0009 ^ (w as u64) << 8);
                let a: Vec<u16> =
                    (0..STRIDE * 64).map(|_| (rng.next_u32() % 1024) as u16).collect();
                let c: Vec<u16> =
                    (0..STRIDE * 64).map(|_| (rng.next_u32() % 1024) as u16).collect();
                b.with_input(move || (a.clone(), c.clone())).run(move |(a, c)| {
                    let mut acc = 0i64;
                    for _ in 0..reps {
                        acc += dist::highbd_sse(&a, STRIDE, &c, STRIDE, w, h);
                    }
                    acc
                });
            });
        }
        tune(g);
    });
}

/// Quantization — `av1_quantize_fp` across the block-size axis.
fn bench_quant(suite: &mut Suite) {
    suite.group("quant", |g| {
        g.throughput_unit("coeff");
        g.throughput(Throughput::Elements(WORK_PX as u64));
        for &(tx_size, name) in &[(0usize, "04x04"), (1, "08x08"), (2, "16x16"), (3, "32x32")] {
            let n = TX_W[tx_size] * TX_H[tx_size];
            let reps = WORK_PX / n;
            g.bench(format!("fp_{name}"), move |b| {
                let mut rng = Rng(0x5EED_000A ^ (tx_size as u64) << 8);
                let coeff: Vec<i32> = (0..n).map(|i| rng.coeff(i) * 4).collect();
                let scan: Vec<i16> = (0..n as i16).collect();
                let q = vec![0i32; n];
                let dq = vec![0i32; n];
                b.with_input(move || (coeff.clone(), scan.clone(), q.clone(), dq.clone())).run(
                    move |(coeff, scan, mut q, mut dq)| {
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
                    },
                );
            });
        }
        tune(g);
    });
}

/// Intra prediction — the compute-heavy predictors across the size axis.
fn bench_intra(suite: &mut Suite) {
    suite.group("intra", |g| {
        g.throughput_unit("px");
        g.throughput(Throughput::Elements(WORK_PX as u64));
        // SMOOTH* / PAETH are the compute-heavy predictors; V/H are
        // memory-bound copies, kept as the control.
        for (mode, mname) in [
            (intra::V, "v"),
            (intra::H, "h"),
            (intra::PAETH, "paeth"),
            (intra::SMOOTH, "smooth"),
            (intra::SMOOTH_V, "smooth_v"),
        ] {
            for (bw, bh, name) in [(4usize, 4usize, "04x04"), (16, 16, "16x16"), (32, 32, "32x32")]
            {
                let reps = WORK_PX / (bw * bh);
                g.bench(format!("{mname}_{name}"), move |b| {
                    let mut rng = Rng(0x5EED_000B ^ (bw as u64) << 8 ^ mode as u64);
                    let above: Vec<u8> = (0..bw + 2 * bh + 2).map(|_| rng.pixel()).collect();
                    let left: Vec<u8> = (0..bh + bw).map(|_| rng.pixel()).collect();
                    let dst = vec![0u8; bw * bh];
                    b.with_input(move || (above.clone(), left.clone(), dst.clone())).run(
                        move |(above, left, mut dst)| {
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
                        },
                    );
                });
            }
        }
        tune(g);
    });
}

fn main() {
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
        bench_quant(suite);
        bench_intra(suite);
    });
    zenbench::postprocess_result(&result);
}
