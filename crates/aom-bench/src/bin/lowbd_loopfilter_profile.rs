//! Callgrind microbench for the bd8 deblock lowbd lever: run a fixed
//! representative whole-frame loop-filter workload N times through either the
//! LOWBD (u8) path (`loop_filter_frame_u8`) or the HIGHBD (u16, bd=8) path
//! (`loop_filter_frame`), on IDENTICAL inputs, so a callgrind Ir profile
//! compares the two entry points directly.
//!
//! Usage: lowbd_loopfilter_profile <u8|u16> <iters>
//!
//! Before profiling, iter 0's output is cross-checked u8-vs-u16 byte-identical
//! (a corrupt build must never be profiled). Compare inclusive Ir of
//! `loop_filter_frame_u8` (u8 side) vs `loop_filter_frame` (u16 side) across the
//! two runs. NB: the loop filter's SIMD width is fixed at 4 (the 4 edge
//! positions of one kernel call), independent of pixel width — so unlike the
//! transform (i32->i16 lane doubling) the lowbd win here is memory bandwidth
//! (u8 vs u16 plane traffic) + the eliminated widen/narrow round-trip once the
//! recon plane is u8, NOT a lane-count change; expect a small Ir delta.

use aom_dsp::loopfilter::frame::{
    loop_filter_frame, loop_filter_frame_u8, LfFrameBuf, LfFrameBufU8, LfMi, LfMiGrid, LfParams,
};

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
}

const BLOCK_16X16: u8 = 6;

struct Workload {
    mi: Vec<LfMi>,
    mi_rows: i32,
    mi_cols: i32,
    p: LfParams,
    // original (pre-filter) planes; the runs reset to these each iter
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    uv_stride: usize,
    w: u32,
    h: u32,
}

/// A fixed 256x256 4:2:0 all-intra KEY frame: BLOCK_16X16 coding blocks with a
/// rotating luma tx_size in {TX_4X4, TX_8X8, TX_16X16} per 16px block (so the
/// walk fires luma filter widths {4,8,14} and chroma {4,6}), a nonzero luma +
/// chroma filter level, near-flat + noisy pixel regions (triggers both the
/// `filter4` and the wide flat/flat2 arms). The per-edge derivation is identical
/// for the u8 and u16 paths — only the kernel storage width differs.
fn workload() -> Workload {
    let (w, h) = (256u32, 256u32);
    let mi_cols = (w / 4) as i32;
    let mi_rows = (h / 4) as i32;
    let mut mi = vec![LfMi::default(); (mi_rows * mi_cols) as usize];
    for r in 0..mi_rows as usize {
        for c in 0..mi_cols as usize {
            let tx = ((r / 4 + c / 4) % 3) as u8; // TX_4X4 / TX_8X8 / TX_16X16
            mi[r * mi_cols as usize + c] = LfMi {
                bsize: BLOCK_16X16,
                tx_size: tx,
                segment_id: 0,
                ref0: 0,
                mode_lf: 0,
                is_inter: false,
                skip_txfm: false,
                delta_lf_from_base: 0,
                delta_lf: [0; 4],
            };
        }
    }
    let p = LfParams {
        filter_level: [20, 20],
        filter_level_u: 20,
        filter_level_v: 20,
        ..Default::default()
    };
    let mut rng = Rng(0x_deb1_0c47_2026);
    let genp = |rng: &mut Rng, n: usize| -> Vec<u8> {
        // Alternate near-flat (128 + small dither -> wide flat/flat2 arm) and
        // fully-random (filter4 arm) regions in 32-sample bands so both fire.
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let dither = (rng.next() % 5) as i32 - 2;
            let base = if (i / 32) % 2 == 0 { 128i32 } else { (rng.next() & 0xff) as i32 };
            out.push((base + dither).clamp(0, 255) as u8);
        }
        out
    };
    let y_stride = (mi_cols * 4) as usize;
    let uv_stride = y_stride >> 1;
    let y = genp(&mut rng, y_stride * (mi_rows * 4) as usize);
    let u = genp(&mut rng, uv_stride * ((mi_rows * 4) >> 1) as usize);
    let v = genp(&mut rng, uv_stride * ((mi_rows * 4) >> 1) as usize);
    Workload { mi, mi_rows, mi_cols, p, y, u, v, y_stride, uv_stride, w, h }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: lowbd_loopfilter_profile <u8|u16> <iters>");
        std::process::exit(2);
    }
    let side = args[1].as_str();
    let iters: usize = args[2].parse().expect("iters must be a number");
    let wl = workload();
    let grid = LfMiGrid { mi: &wl.mi, stride: wl.mi_cols as usize, mi_rows: wl.mi_rows, mi_cols: wl.mi_cols };

    // Byte-identity cross-check (u8 path vs u16 path) before profiling.
    {
        let (mut y8, mut u8v, mut v8) = (wl.y.clone(), wl.u.clone(), wl.v.clone());
        let mut buf8 = LfFrameBufU8 {
            y: &mut y8, y_stride: wl.y_stride, u: &mut u8v, v: &mut v8, uv_stride: wl.uv_stride,
            crop_width: wl.w, crop_height: wl.h, ss_x: 1, ss_y: 1,
        };
        loop_filter_frame_u8(&mut buf8, &grid, &wl.p, 0, 3);

        let (mut y16, mut u16v, mut v16): (Vec<u16>, Vec<u16>, Vec<u16>) = (
            wl.y.iter().map(|&x| x as u16).collect(),
            wl.u.iter().map(|&x| x as u16).collect(),
            wl.v.iter().map(|&x| x as u16).collect(),
        );
        let mut buf16 = LfFrameBuf {
            y: &mut y16, y_stride: wl.y_stride, u: &mut u16v, v: &mut v16, uv_stride: wl.uv_stride,
            crop_width: wl.w, crop_height: wl.h, ss_x: 1, ss_y: 1, bd: 8,
        };
        loop_filter_frame(&mut buf16, &grid, &wl.p, 0, 3);

        for (i, (&a, &b)) in y8.iter().zip(y16.iter()).enumerate() {
            assert_eq!(a as u16, b, "u8 vs u16 luma divergence @ {i}");
        }
        for (i, (&a, &b)) in u8v.iter().zip(u16v.iter()).enumerate() {
            assert_eq!(a as u16, b, "u8 vs u16 U divergence @ {i}");
        }
        for (i, (&a, &b)) in v8.iter().zip(v16.iter()).enumerate() {
            assert_eq!(a as u16, b, "u8 vs u16 V divergence @ {i}");
        }
    }

    let mut sink = 0u64;
    match side {
        "u8" => {
            let (mut y, mut u, mut v) = (wl.y.clone(), wl.u.clone(), wl.v.clone());
            for _ in 0..iters {
                y.copy_from_slice(&wl.y);
                u.copy_from_slice(&wl.u);
                v.copy_from_slice(&wl.v);
                let mut buf = LfFrameBufU8 {
                    y: &mut y, y_stride: wl.y_stride, u: &mut u, v: &mut v, uv_stride: wl.uv_stride,
                    crop_width: wl.w, crop_height: wl.h, ss_x: 1, ss_y: 1,
                };
                loop_filter_frame_u8(&mut buf, &grid, &wl.p, 0, 3);
                sink = sink.wrapping_add(buf.y[0] as u64);
            }
        }
        "u16" => {
            let y0: Vec<u16> = wl.y.iter().map(|&x| x as u16).collect();
            let u0: Vec<u16> = wl.u.iter().map(|&x| x as u16).collect();
            let v0: Vec<u16> = wl.v.iter().map(|&x| x as u16).collect();
            let (mut y, mut u, mut v) = (y0.clone(), u0.clone(), v0.clone());
            for _ in 0..iters {
                y.copy_from_slice(&y0);
                u.copy_from_slice(&u0);
                v.copy_from_slice(&v0);
                let mut buf = LfFrameBuf {
                    y: &mut y, y_stride: wl.y_stride, u: &mut u, v: &mut v, uv_stride: wl.uv_stride,
                    crop_width: wl.w, crop_height: wl.h, ss_x: 1, ss_y: 1, bd: 8,
                };
                loop_filter_frame(&mut buf, &grid, &wl.p, 0, 3);
                sink = sink.wrapping_add(buf.y[0] as u64);
            }
        }
        other => panic!("side must be u8|u16, got {other}"),
    }
    eprintln!("{side} x{iters}: sink={sink}");
}
