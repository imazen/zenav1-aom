//! Strided `KeyFramePlanes` code to the bytes of tight ones: padding holds
//! samples out of range and is never read, and each buffer is the exact minimum
//! `(rows - 1) * stride + width`. Covers 8, 10 and 12 bits, 4:2:0 and 4:4:4 where the
//! envelope allows, monochrome, odd sizes and superres, plus the refusals.

use aom_encode::key_frame::{KeyFrameConfig, KeyFrameError, KeyFramePlanes, encode_key_frame};

fn sample(x: usize, y: usize, c: usize, depth: u8) -> u16 {
    let max = (1u32 << depth) - 1;
    (((x * 3 + y * 5 + (x * y) / 7 + c * 37) % 200 + 20) as u32 * max / 255) as u16
}

fn plane(w: usize, h: usize, c: usize, depth: u8) -> Vec<u16> {
    (0..w * h).map(|i| sample(i % w, i / w, c, depth)).collect()
}

fn padded(tight: &[u16], w: usize, stride: usize) -> Vec<u16> {
    let h = tight.len() / w;
    let mut out = vec![0xFFFF; (h - 1) * stride + w];
    for (row, line) in tight.chunks_exact(w).enumerate() {
        out[row * stride..row * stride + w].copy_from_slice(line);
    }
    out
}

fn assert_identical(cfg: &KeyFrameConfig, label: &str) {
    let (w, h, d) = (cfg.width, cfg.height, cfg.bit_depth);
    let (cw, ch) = cfg.chroma_dims();
    let y = plane(w, h, 0, d);
    let (u, v) = if cfg.monochrome {
        (Vec::new(), Vec::new())
    } else {
        (plane(cw, ch, 1, d), plane(cw, ch, 2, d))
    };
    let tight = encode_key_frame(KeyFramePlanes::new(&y, w, &u, cw, &v, cw), cfg).expect(label);
    let (ys, us, vs) = (w + 7, cw + 3, cw + 11);
    let py = padded(&y, w, ys);
    let (pu, pv) = if cfg.monochrome {
        (Vec::new(), Vec::new())
    } else {
        (padded(&u, cw, us), padded(&v, cw, vs))
    };
    let strided =
        encode_key_frame(KeyFramePlanes::new(&py, ys, &pu, us, &pv, vs), cfg).expect(label);
    assert!(!tight.is_empty(), "{label}");
    assert!(tight == strided, "{label}: strided stream differs");
}

fn base(w: usize, h: usize, depth: u8) -> KeyFrameConfig {
    let mut cfg = KeyFrameConfig::allintra_speed0(w, h, depth, false, 1, 1, 32);
    cfg.cpu_used = 9;
    cfg
}

#[test]
fn strided_planes_code_to_the_tight_bytes() {
    for (w, h) in [(64usize, 64usize), (65, 67), (33, 27)] {
        for depth in [8u8, 10, 12] {
            assert_identical(&base(w, h, depth), &format!("{w}x{h} bd{depth} 420"));
            let mut cfg = base(w, h, depth);
            (cfg.ss_x, cfg.ss_y) = (0, 0);
            assert_identical(&cfg, &format!("{w}x{h} bd{depth} 444"));
        }
    }
}

#[test]
fn monochrome_ignores_chroma_strides() {
    for (w, h) in [(64usize, 64usize), (65, 67)] {
        let mut cfg = KeyFrameConfig::allintra_speed0(w, h, 8, true, 1, 1, 32);
        cfg.cpu_used = 9;
        assert_identical(&cfg, &format!("{w}x{h} mono"));
    }
}

#[test]
fn superres_packs_a_strided_source_once() {
    for denom in [9u8, 16] {
        let mut cfg = base(65, 67, 8);
        cfg.superres_denom = denom;
        cfg.enable_cdef = false;
        cfg.enable_restoration = false;
        assert_identical(&cfg, &format!("superres {denom}"));
    }
}

#[test]
fn image_samples_out_of_range_are_still_refused() {
    let cfg = base(33, 27, 8);
    let (cw, ch) = cfg.chroma_dims();
    let (ys, us, vs) = (40, cw + 3, cw + 11);
    let y = padded(&plane(33, 27, 0, 8), 33, ys);
    let u = padded(&plane(cw, ch, 1, 8), cw, us);
    let v = padded(&plane(cw, ch, 2, 8), cw, vs);
    for idx in 0..3 {
        let (mut by, mut bu, mut bv) = (y.clone(), u.clone(), v.clone());
        match idx {
            0 => by[ys + 1] = 300,
            1 => bu[us + 1] = 300,
            _ => bv[vs + 1] = 300,
        }
        let e = encode_key_frame(KeyFramePlanes::new(&by, ys, &bu, us, &bv, vs), &cfg).unwrap_err();
        assert!(
            matches!(e, KeyFrameError::SampleRange { plane, got: 300, .. } if plane == idx),
            "{e:?}"
        );
    }
}

#[test]
fn bad_geometry_is_refused() {
    let (w, h) = (33usize, 27usize);
    let cfg = base(w, h, 8);
    let (cw, ch) = cfg.chroma_dims();
    let (ys, us, vs) = (w + 7, cw + 3, cw + 11);
    let y = padded(&plane(w, h, 0, 8), w, ys);
    let u = padded(&plane(cw, ch, 1, 8), cw, us);
    let v = padded(&plane(cw, ch, 2, 8), cw, vs);
    let run = |p: KeyFramePlanes<'_>| encode_key_frame(p, &cfg).unwrap_err();

    for (idx, e) in [
        run(KeyFramePlanes::new(&y, w - 1, &u, us, &v, vs)),
        run(KeyFramePlanes::new(&y, ys, &u, cw - 1, &v, vs)),
        run(KeyFramePlanes::new(&y, ys, &u, us, &v, cw - 1)),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(
            matches!(e, KeyFrameError::PlaneStride { plane, .. } if plane == idx),
            "{e:?}"
        );
    }
    for (idx, e) in [
        run(KeyFramePlanes::new(&y[..y.len() - 1], ys, &u, us, &v, vs)),
        run(KeyFramePlanes::new(&y, ys, &u[..u.len() - 1], us, &v, vs)),
        run(KeyFramePlanes::new(&y, ys, &u, us, &v[..v.len() - 1], vs)),
        run(KeyFramePlanes::new(&y, usize::MAX, &u, us, &v, vs)),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(
            matches!(e, KeyFrameError::PlaneSize { plane, .. } if plane == idx % 3),
            "{e:?}"
        );
    }
    // A tight plane's length is checked by the same rule: too short is refused, and
    // samples past the last row are not read.
    let (ty, tu, tv) = (plane(w, h, 0, 8), plane(cw, ch, 1, 8), plane(cw, ch, 2, 8));
    assert!(matches!(
        run(KeyFramePlanes::new(
            &ty[..ty.len() - 1],
            w,
            &tu,
            cw,
            &tv,
            cw
        )),
        KeyFrameError::PlaneSize { plane: 0, .. }
    ));
    let mut long = ty.clone();
    long.push(0xFFFF);
    assert_eq!(
        encode_key_frame(KeyFramePlanes::new(&long, w, &tu, cw, &tv, cw), &cfg).unwrap(),
        encode_key_frame(KeyFramePlanes::new(&ty, w, &tu, cw, &tv, cw), &cfg).unwrap()
    );
}
