//! Whole-frame loop restoration vs the REAL C walk: random frame pairs
//! (a "deblocked" pre-CDEF state and a DIFFERENT "current" post-CDEF state —
//! more adversarial than production, where they differ only where CDEF
//! fired) + random per-RU parameters, through `ref_lr_filter_frame` (real
//! `av1_loop_restoration_save_boundary_lines` x2 + real
//! `av1_loop_restoration_filter_frame` on bordered YV12 buffers) and
//! `aom_restore::frame::loop_restoration_filter_frame`. All planes
//! byte-identical, both the boundary-swapped (`optimized=false`, the CDEF
//! decoder path) and the optimized (no-CDEF) arm.

use aom_entropy::lr::{LrFrameConfig, LrUnitInfo, SgrprojInfoLr, WienerInfoLr};
use aom_restore::frame::{loop_restoration_filter_frame, LrPlaneInput};
use aom_sys_ref as c;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u64) as i32
    }
}

/// One random unit for `frame_rtype`: `(packed 10-i32 for C, LrUnitInfo)`.
fn gen_unit(rng: &mut Rng, frame_rtype: u8, chroma: bool) -> ([i32; c::LRF_WORDS], LrUnitInfo) {
    let rtype = match frame_rtype {
        3 => rng.below(3) as u8, // SWITCHABLE: NONE/WIENER/SGRPROJ per unit
        t => {
            if rng.below(4) == 0 {
                0
            } else {
                t
            }
        }
    };
    let mut p = [0i32; c::LRF_WORDS];
    p[0] = rtype as i32;
    let mut info = LrUnitInfo {
        restoration_type: rtype,
        wiener: WienerInfoLr {
            vfilter: [0; 8],
            hfilter: [0; 8],
        },
        sgrproj: SgrprojInfoLr { ep: 0, xqd: [0; 2] },
    };
    match rtype {
        1 => {
            for d in 0..2 {
                let t0 = if chroma { 0 } else { rng.range(-5, 10) };
                let t1 = rng.range(-23, 8);
                let t2 = rng.range(-17, 46);
                p[1 + 3 * d] = t0;
                p[2 + 3 * d] = t1;
                p[3 + 3 * d] = t2;
                let f = if d == 0 {
                    &mut info.wiener.vfilter
                } else {
                    &mut info.wiener.hfilter
                };
                *f = [
                    t0 as i16,
                    t1 as i16,
                    t2 as i16,
                    (-2 * (t0 + t1 + t2)) as i16,
                    t2 as i16,
                    t1 as i16,
                    t0 as i16,
                    0,
                ];
            }
        }
        2 => {
            let ep = rng.below(16) as i32;
            let r = aom_entropy::lr::SGR_PARAMS_R[ep as usize];
            let xqd = if r[0] == 0 {
                [0, rng.range(-32, 95)]
            } else if r[1] == 0 {
                let x0 = rng.range(-96, 31);
                [x0, (128 - x0).clamp(-32, 95)]
            } else {
                [rng.range(-96, 31), rng.range(-32, 95)]
            };
            p[7] = ep;
            p[8] = xqd[0];
            p[9] = xqd[1];
            info.sgrproj = SgrprojInfoLr { ep, xqd };
        }
        _ => {}
    }
    (p, info)
}

fn rand_plane(rng: &mut Rng, len: usize, bd: i32) -> Vec<u16> {
    let mask = (1u64 << bd) - 1;
    (0..len).map(|_| (rng.next() & mask) as u16).collect()
}

#[test]
fn lr_filter_frame_matches_c() {
    c::ref_init();
    let mut rng = Rng(0xF17E_57A1_1CE5_0FAD);
    let dims: &[(usize, usize)] = &[
        (64, 64),
        (65, 65),
        (128, 96),
        (176, 144),
        (250, 127),
        (36, 20),
        (320, 240),
        (448, 232),
        (64, 260),
        (129, 57),
    ];
    let mut applied = [0usize; 4]; // per frame_rtype population
    let mut case = 0usize;
    for &(w, h) in dims {
        for &(ss_x, ss_y, mono) in &[
            (0usize, 0usize, false),
            (1, 0, false),
            (1, 1, false),
            (1, 1, true),
        ] {
            for &bd in &[8, 10, 12] {
                // Two cases per axis point: one boundary-swapped (the CDEF
                // decoder path), one optimized (no-CDEF path).
                for &optimized in &[false, true] {
                    case += 1;
                    // Thin the full product to keep runtime sane, but keep
                    // every (dims x ss) at bd 8 + a rotating bd 10/12 slice.
                    if bd != 8 && (case % 3 != (bd as usize / 2) % 3) {
                        continue;
                    }
                    let num_planes = if mono { 1 } else { 3 };
                    // Frame types: at least one non-NONE plane.
                    let mut frame_rtype = [0u8; 3];
                    loop {
                        for (p, t) in frame_rtype.iter_mut().enumerate().take(num_planes) {
                            *t = rng.below(4) as u8;
                            let _ = p;
                        }
                        if frame_rtype[..num_planes].iter().any(|&t| t != 0) {
                            break;
                        }
                    }
                    // Unit sizes: random luma 64/128/256; chroma same or
                    // halved when both axes subsample.
                    let ys = 64 << rng.below(3);
                    let cs = if ss_x.min(ss_y) == 1 && ys > 64 && rng.below(2) == 1 {
                        ys >> 1
                    } else {
                        ys
                    };
                    let unit_size = [ys, cs, cs];
                    let lr = LrFrameConfig {
                        frame_restoration_type: frame_rtype,
                        unit_size,
                        crop_width: w as i32,
                        crop_height: h as i32,
                        superres_denom: 0, // unscaled
                    };

                    // Units per plane.
                    let mut packed: [Vec<i32>; 3] = Default::default();
                    let mut infos: [Vec<LrUnitInfo>; 3] = Default::default();
                    for p in 0..num_planes {
                        if frame_rtype[p] == 0 {
                            continue;
                        }
                        let (hu, vu) = lr.plane_units(p, ss_x, ss_y);
                        for _ in 0..hu * vu {
                            let (pk, inf) = gen_unit(&mut rng, frame_rtype[p], p > 0);
                            packed[p].extend_from_slice(&pk);
                            infos[p].push(inf);
                        }
                        applied[frame_rtype[p] as usize] += 1;
                    }

                    // Random current + deblocked planes (deliberately
                    // unrelated content).
                    let (cw, ch) = ((w + ss_x) >> ss_x, (h + ss_y) >> ss_y);
                    let (y_stride, uv_stride) = (w + 13, cw + 7); // odd strides
                    let mut y = rand_plane(&mut rng, y_stride * h, bd);
                    let (mut u, mut v, du, dv);
                    if mono {
                        u = Vec::new();
                        v = Vec::new();
                        du = Vec::new();
                        dv = Vec::new();
                    } else {
                        u = rand_plane(&mut rng, uv_stride * ch, bd);
                        v = rand_plane(&mut rng, uv_stride * ch, bd);
                        du = rand_plane(&mut rng, uv_stride * ch, bd);
                        dv = rand_plane(&mut rng, uv_stride * ch, bd);
                    }
                    let dy = rand_plane(&mut rng, y_stride * h, bd);

                    // C reference.
                    let mut y_c = y.clone();
                    let mut u_c = u.clone();
                    let mut v_c = v.clone();
                    c::ref_lr_filter_frame(
                        &mut y_c,
                        &mut u_c,
                        &mut v_c,
                        &dy,
                        &du,
                        &dv,
                        w,
                        h,
                        y_stride,
                        uv_stride,
                        num_planes,
                        ss_x,
                        ss_y,
                        bd,
                        optimized,
                        frame_rtype.map(i32::from),
                        unit_size,
                        [&packed[0], &packed[1], &packed[2]],
                    );

                    // Rust.
                    {
                        let mut planes = Vec::new();
                        planes.push(LrPlaneInput {
                            cur: &mut y,
                            deblocked: &dy,
                            stride: y_stride,
                            units: &infos[0],
                        });
                        if !mono {
                            planes.push(LrPlaneInput {
                                cur: &mut u,
                                deblocked: &du,
                                stride: uv_stride,
                                units: &infos[1],
                            });
                            planes.push(LrPlaneInput {
                                cur: &mut v,
                                deblocked: &dv,
                                stride: uv_stride,
                                units: &infos[2],
                            });
                        }
                        loop_restoration_filter_frame(&mut planes, &lr, ss_x, ss_y, bd, optimized);
                    }

                    let tag = format!(
                        "{w}x{h} ss({ss_x},{ss_y}) mono={mono} bd{bd} opt={optimized} \
                         rt{frame_rtype:?} us{unit_size:?}"
                    );
                    assert_eq!(y, y_c, "Y: {tag}");
                    assert_eq!(u, u_c, "U: {tag}");
                    assert_eq!(v, v_c, "V: {tag}");
                }
            }
        }
    }
    // Population floors: every frame type + the switchable mix exercised.
    assert!(
        applied[1] >= 10 && applied[2] >= 10 && applied[3] >= 10,
        "type populations {applied:?}"
    );
}
