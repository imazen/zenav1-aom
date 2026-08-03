//! `winperf` — the pure-Rust half of the encode harness, so the port's own
//! before/after wall time can be measured on a runner that has never built
//! libaom.
//!
//! # Why this exists
//!
//! `benchmarks/encoder_alloc_scratch_2026-08-02.md` measured the KB-PERF-2
//! allocation levers at **−1.34 ms** (lever 3) on an Apple M4 Pro. That number
//! is **Darwin's allocator's**, and the encoder ships on Windows, whose default
//! heap is a different implementation. Nothing in this repo had ever run the
//! encoder on Windows at all: the `windows-11-arm` CI job is `portability` — it
//! *builds* the published crates and runs neither tests nor benches. So the
//! allocator's cost there was unmeasured, not merely unreported.
//!
//! Measuring it needs an encode harness with **no C oracle**, because
//! `drv-aom` (and `eprof_alloc`) take their sequence-header bootstrap from a
//! live `c_encode_defaults()` call, and building libaom on the Windows runners
//! is a separate sub-project (`.github/workflows/ci.yml`, the `portability`
//! job's comment). Two substitutions make the same encode reachable without it:
//!
//! * **the source image** is [`synth_i420`], generated in-process from integer
//!   arithmetic only, so it is bit-identical on every target; and
//! * **the bootstrap** is a committed fixture ([`BOOTSTRAP_DETAIL_HEX`] /
//!   [`BOOTSTRAP_SMOOTH_HEX`]) — the bytes a real `aomenc --allintra` emitted
//!   for this exact cell on the dev box, so the frame header the port parses is
//!   a genuine one.
//!
//! Everything between those two is the port's ordinary
//! [`EncodeCell::port_encode`] path, unchanged.
//!
//! # What this is NOT
//!
//! It is not a differential. No C encoder runs, so nothing here proves
//! byte-exactness; the byte count is reported per invocation purely as a
//! *did-both-arms-do-the-same-work* check. The differential gates are the ~40
//! integration tests in this crate, which are unaffected and still require the
//! `c-oracle` feature.
//!
//! It is also not the dev-box study's image. That study used a real photograph
//! (`~/tmp/xb/src/photo_1024.yuv`), which is 1.5 MB and cannot be committed.
//! [`synth_i420`] is a fractal-noise stand-in with a broadly photographic
//! spectrum — see its docs for exactly what is and is not claimed. The
//! cross-platform comparison this harness supports is **the same harness on
//! both boxes**, never "Windows synthetic vs Darwin photo".

use crate::EncodeCell;

/// The cell the whole KB-PERF-2 record is written against: 1024x1024 (1 MP),
/// `--cq-level 44`, `--cpu-used 6`, ALLINTRA (`usage = 2`), 8-bit 4:2:0, single
/// tile, single thread.
pub const CELL: (usize, usize, i32, i32) = (1024, 1024, 44, 6);

/// Which synthetic content to encode.
///
/// TWO of them exist, and the reason is a measurement, not a preference. On the
/// dev box, at this cell, **lever 3's sign depends on the content**: on the
/// study's photograph it is −2.21 ms, on [`Content::Detail`] it is +2.49 ms.
/// Reporting a platform comparison off one image would therefore be reporting
/// one image. The two variants bracket the photograph's coded size (2 471 /
/// 8 734 bytes against its 4 472) and are run as separate bands that are never
/// averaged together.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Content {
    /// Shallow amplitude ladder (26/22/18/15/12/10/8/7 over periods 256..2),
    /// i.e. strong detail all the way down to pixel-scale grain. Tuned so its
    /// allocator-call count matches the study photograph's to 95 %.
    Detail,
    /// Classic halving ladder (32/16/8/4/2/1 over periods 256..8) at 3/4
    /// contrast — markedly smoother than a photograph, and the low-work end of
    /// the bracket.
    Smooth,
    /// **Oriented** content: value noise summed with a *streak* field whose
    /// orientation rotates smoothly across the frame, fitted to the study
    /// photograph's INTRA MODE DISTRIBUTION rather than to its allocator call
    /// count. See [`PHOTO`] for the fitted parameters and
    /// `benchmarks/winperf_content_census_2026-08-03.md` for the fit.
    ///
    /// [`Detail`](Content::Detail) and [`Smooth`](Content::Smooth) are
    /// *isotropic* — fractional-Brownian value noise has no preferred direction
    /// at any scale — so the encoder's directional intra modes essentially never
    /// win on them (`detail`: `z1` fires **six times** in a 1 MP frame). This
    /// variant exists so a lever scoped to a mode family has content that
    /// reaches it.
    Photo,
}

impl Content {
    /// `"detail"` / `"smooth"` / `"photo"`, for a command line.
    pub fn parse(s: &str) -> Self {
        match s {
            "detail" => Content::Detail,
            "smooth" => Content::Smooth,
            "photo" => Content::Photo,
            _ => panic!("unknown winperf content {s:?}; want `detail`, `smooth` or `photo`"),
        }
    }

    /// The lower-case name `parse` accepts.
    pub fn label(self) -> &'static str {
        match self {
            Content::Detail => "detail",
            Content::Smooth => "smooth",
            Content::Photo => "photo",
        }
    }

    /// Every content, in the order the census and the workflow list them.
    pub const ALL: [Content; 3] = [Content::Detail, Content::Smooth, Content::Photo];
}

/// The sequence-header bootstrap for [`CELL`] at [`Content::Detail`], as hex.
///
/// These are the bytes a real `aomenc --allintra --cq-level=44 --cpu-used=6`
/// (via `EncodeCell::c_encode_defaults`, the true-ALLINTRA-defaults path)
/// produced for [`synth_i420`] on the dev box. The port reads the sequence
/// header and the uncompressed frame header out of it and produces every coded
/// byte itself — the same arrangement `drv-aom` documents, with the one
/// difference that the encode which produced these bytes happened once, at
/// commit time, instead of once per process start.
///
/// Stored as hex rather than as a binary blob so it diffs, greps and reviews
/// like source. Regenerate with `examples/winperf_bootstrap_gen.rs` (which
/// needs the `c-oracle` feature, i.e. a box that can build libaom).
pub const BOOTSTRAP_DETAIL_HEX: &str =
    include_str!("../fixtures/winperf_bootstrap_1024x1024_cq44_s6_detail.hex");

/// [`BOOTSTRAP_DETAIL_HEX`]'s twin for [`Content::Smooth`].
pub const BOOTSTRAP_SMOOTH_HEX: &str =
    include_str!("../fixtures/winperf_bootstrap_1024x1024_cq44_s6_smooth.hex");

/// [`BOOTSTRAP_DETAIL_HEX`]'s twin for [`Content::Photo`].
pub const BOOTSTRAP_PHOTO_HEX: &str =
    include_str!("../fixtures/winperf_bootstrap_1024x1024_cq44_s6_photo.hex");

/// The committed bootstrap for `content`, decoded. Panics on any non-hex byte —
/// the fixture is checked in, so a failure here is a corrupted tree, not a
/// runtime condition.
pub fn bootstrap(content: Content) -> Vec<u8> {
    let hex = match content {
        Content::Detail => BOOTSTRAP_DETAIL_HEX,
        Content::Smooth => BOOTSTRAP_SMOOTH_HEX,
        Content::Photo => BOOTSTRAP_PHOTO_HEX,
    };
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut hi: Option<u8> = None;
    for c in hex.bytes() {
        let v = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => panic!("winperf bootstrap fixture: non-hex byte {c:#x}"),
        };
        match hi.take() {
            None => hi = Some(v),
            Some(h) => out.push((h << 4) | v),
        }
    }
    assert!(hi.is_none(), "winperf bootstrap fixture: odd hex digit count");
    out
}

/// One round of a 32-bit integer hash (a finalizer in the murmur/`splitmix`
/// family). Pure integer arithmetic with wrapping semantics, so the value is
/// identical on aarch64, x86-64 and 32-bit targets alike.
fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Lattice value at integer grid point `(gx, gy)` for octave `seed`, in 0..=255.
fn lattice(gx: i32, gy: i32, seed: u32) -> i32 {
    let h = hash32(
        (gx as u32)
            .wrapping_mul(0x9e37_79b9)
            ^ (gy as u32).wrapping_mul(0x85eb_ca6b)
            ^ seed.wrapping_mul(0xc2b2_ae35),
    );
    (h >> 24) as i32
}

/// Smooth (cubic, `3t^2 - 2t^3`) interpolation weight in Q8 for `t` in Q8.
fn smooth_q8(t: i32) -> i32 {
    // 3t^2 - 2t^3 with t in [0, 256): keep everything in i64-free i32 range by
    // shifting down between multiplies. Max intermediate is 256*256 = 65536.
    let t2 = (t * t) >> 8; // Q8
    let t3 = (t2 * t) >> 8; // Q8
    3 * t2 - 2 * t3
}

/// One octave of bilinear-interpolated integer value noise at period `period`
/// (a power of two), returning 0..=255.
fn octave(x: usize, y: usize, period: usize, seed: u32) -> i32 {
    let p = period as i32;
    let (xi, yi) = (x as i32 / p, y as i32 / p);
    let (xf, yf) = (x as i32 % p, y as i32 % p);
    // Fractional position in Q8.
    let tx = smooth_q8(xf * 256 / p);
    let ty = smooth_q8(yf * 256 / p);
    let v00 = lattice(xi, yi, seed);
    let v10 = lattice(xi + 1, yi, seed);
    let v01 = lattice(xi, yi + 1, seed);
    let v11 = lattice(xi + 1, yi + 1, seed);
    let a = v00 * (256 - tx) + v10 * tx; // Q8
    let b = v01 * (256 - tx) + v11 * tx; // Q8
    ((a * (256 - ty) + b * ty) >> 16).clamp(0, 255)
}

/// One round of 1-D integer value noise: lattice point `i` for octave `seed`.
fn lattice1(i: i32, seed: u32) -> i32 {
    (hash32((i as u32).wrapping_mul(0x9e37_79b9) ^ seed.wrapping_mul(0xc2b2_ae35)) >> 24) as i32
}

/// One octave of 1-D smooth-interpolated value noise at `period`, 0..=255.
///
/// `u` may be negative (it is a projection onto a direction with a negative
/// component), which is why this uses `div_euclid`/`rem_euclid` rather than `/`
/// and `%`: truncating division would mirror the lattice about the origin and
/// put a visible seam down the frame.
fn octave1(u: i32, period: i32, seed: u32) -> i32 {
    let i = u.div_euclid(period);
    let f = u.rem_euclid(period);
    let t = smooth_q8(f * 256 / period);
    let a = lattice1(i, seed);
    let b = lattice1(i + 1, seed);
    ((a * (256 - t) + b * t) >> 8).clamp(0, 255)
}

/// The eight streak directions, as Q8 `(cos, sin)` at `k * 22.5°`. Together
/// they span a half-turn, so index 7 (157.5°) wraps continuously into index 0
/// (180° ≡ 0°) and the orientation field below has no discontinuity.
const DIR_Q8: [(i32, i32); 8] = [
    (256, 0),
    (237, 98),
    (181, 181),
    (98, 237),
    (0, 256),
    (-98, 237),
    (-181, 181),
    (-237, 98),
];

/// Knobs for [`synth_photo_luma`], the oriented generator.
///
/// Public and `Copy` because the fit is a **sweep over this struct**
/// (`crates/aom-bench/examples/content_fit.rs`) and the shipped content
/// ([`PHOTO`]) is one point of it — so the parameter choice is re-derivable
/// rather than a set of magic numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhotoParams {
    /// Period of the low-frequency field that selects the local streak
    /// ORIENTATION. Larger = orientation turns more slowly = longer coherent
    /// edges = more directional-mode wins.
    pub orient_period: usize,
    /// Coarse and fine 1-D noise periods ALONG the local gradient direction.
    pub streak_p: (i32, i32),
    /// Their amplitudes.
    pub streak_a: (i32, i32),
    /// Isotropic value-noise ladder over periods 128/64/32/16/8/4 — the part
    /// that keeps DC / SMOOTH / PAETH alive and supplies the small-transform
    /// mass.
    pub iso_a: [i32; 6],
    /// Mixture weights: streak field vs isotropic field. Their ratio is the
    /// single knob that moves the directional share monotonically.
    pub mix: (i32, i32),
    /// Q8 contrast about mid-grey, applied to the mixture.
    pub contrast: i32,
}

/// The FITTED parameters `Content::Photo` ships.
///
/// Chosen by `examples/content_fit.rs` as the grid point minimising L1 distance
/// to the study photograph's intra-class share vector — **not** by looking at
/// any lever's delta. Provenance, grid and the runner-up rows:
/// `benchmarks/winperf_content_census_2026-08-03.md`.
pub const PHOTO: PhotoParams = PhotoParams {
    orient_period: 128,
    streak_p: (96, 24),
    streak_a: (40, 4),
    iso_a: [26, 22, 18, 15, 12, 10], // `Detail`'s ladder, which already matches the work level
    mix: (12, 8),
    contrast: 208,
};

/// Luma sample at `(x, y)` for the oriented generator, 0..=255 before the
/// illumination ramp.
///
/// **The mechanism, and why the isotropic generators cannot do this.** Value
/// noise summed over octaves is isotropic at every scale: its expected
/// structure is the same in all directions, so a block's best intra predictor
/// is almost always DC / SMOOTH / PAETH and the directional predictors never
/// win. Here a *streak* field is added: 1-D noise evaluated on the projection
/// of `(x, y)` onto a direction, which makes it **exactly constant along the
/// perpendicular** — the structure a directional predictor exists to extrapolate.
/// A low-frequency field selects that direction and blends the two nearest of
/// the eight, so orientation rotates smoothly across the frame and all of `z1`,
/// `z2`, `z3`, `V` and `H` get regions where they are the right answer.
///
/// Two 1-D octaves along the direction (coarse + fine) rather than one, so a
/// block sees both a long ramp (which a 32x32 directional predictor wins) and
/// pixel-scale detail (which keeps the small transforms and the residual alive).
pub fn synth_photo_luma(x: usize, y: usize, p: &PhotoParams) -> i32 {
    // Orientation field: 0..255 mapped onto the eight directions, with a smooth
    // blend between the two nearest so there is no seam where it crosses.
    let sel = octave(x, y, p.orient_period, 401);
    let kq = sel * 8;
    let k = ((kq >> 8) as usize).min(7);
    let k1 = (k + 1) & 7;
    let t = smooth_q8(kq & 255);

    let proj = |d: usize| -> i32 {
        let (cx, cy) = DIR_Q8[d];
        (x as i32 * cx + y as i32 * cy) >> 8
    };
    let streak_at = |d: usize| -> i32 {
        let u = proj(d);
        (p.streak_a.0 * octave1(u, p.streak_p.0, 300 + d as u32)
            + p.streak_a.1 * octave1(u, p.streak_p.1, 320 + d as u32))
            / (p.streak_a.0 + p.streak_a.1)
    };
    let streak = (streak_at(k) * (256 - t) + streak_at(k1) * t) >> 8;

    let mut acc = 0i32;
    let mut den = 0i32;
    for (i, &a) in p.iso_a.iter().enumerate() {
        acc += a * octave(x, y, 128 >> i, 1 + i as u32);
        den += a;
    }
    let iso = acc / den;

    let mix = (p.mix.0 * streak + p.mix.1 * iso) / (p.mix.0 + p.mix.1);
    (((mix - 128) * p.contrast) >> 8) + 128
}

/// A deterministic, bit-identical-on-every-target I420 test image.
///
/// **What it is:** eight octaves of integer value noise (periods 256 down to 2)
/// summed into the luma plane over a low-frequency illumination ramp, and four
/// octaves at half resolution and a third of the contrast into each chroma
/// plane. The amplitude ladder is deliberately SHALLOW (26/22/18/15/12/10/8/7,
/// not a halving 1/f-per-octave one), because a halving ladder is much smoother
/// than a photograph.
///
/// **What is claimed, and it was tuned until it was true:** that the amount of
/// *encoder work* it provokes matches the dev box's 1 MP photograph at this
/// cell. Measured on the post-change build, Darwin: **488 750 allocator calls
/// against the photo's 512 557 (95 %)** and **158.5 ms against 154.95 ms
/// (102 %)**. The halving ladder this replaced came in at 406 184 calls (79 %).
/// The coded size does NOT match (8 734 bytes against the photo's 4 472 — this
/// content is harder), and that is fine: coded size is not what drives the
/// per-txb allocation churn the KB-PERF-2 levers touched.
///
/// **What is NOT claimed:** that it is the dev box's photograph, or that any
/// absolute millisecond here is comparable to a number measured on that image.
/// Only same-harness-to-same-harness comparisons are meaningful, which is why
/// the Windows study re-runs this generator on Darwin rather than quoting the
/// photo numbers.
///
/// Integer-only by construction: no floating point appears anywhere above, so
/// there is no room for an x87/FMA/rounding difference to hand two targets
/// different pixels and therefore different encoder work.
pub fn synth_i420(w: usize, h: usize, content: Content) -> Vec<u8> {
    if content == Content::Photo {
        return synth_i420_photo(w, h, &PHOTO);
    }
    assert!(w % 2 == 0 && h % 2 == 0, "even dimensions only");
    let (cw, ch) = (w / 2, h / 2);
    let mut out = vec![0u8; w * h + 2 * cw * ch];

    for y in 0..h {
        for x in 0..w {
            // Low-frequency illumination ramp, so large flat-ish areas exist
            // for the partition search to actually merge.
            let ramp = (x * 40 / w + y * 40 / h) as i32;
            out[y * w + x] = match content {
                // fBm over periods 256..2 with a SHALLOW amplitude ladder,
                // chosen so the allocator census lands near the dev box's 1 MP
                // photograph: a steep 1/2-per-octave ladder is markedly
                // smoother than real photos and returns only 79 % of its calls.
                Content::Detail => {
                    let mut acc = 0i32;
                    acc += 26 * octave(x, y, 256, 1);
                    acc += 22 * octave(x, y, 128, 2);
                    acc += 18 * octave(x, y, 64, 3);
                    acc += 15 * octave(x, y, 32, 4);
                    acc += 12 * octave(x, y, 16, 5);
                    acc += 10 * octave(x, y, 8, 6);
                    acc += 8 * octave(x, y, 4, 7);
                    acc += 7 * octave(x, y, 2, 8);
                    ((acc / 118) + ramp - 20).clamp(0, 255) as u8
                }
                // The classic halving ladder at 3/4 contrast — the low-work end
                // of the bracket.
                Content::Smooth => {
                    let mut acc = 0i32;
                    acc += 32 * octave(x, y, 256, 1);
                    acc += 16 * octave(x, y, 128, 2);
                    acc += 8 * octave(x, y, 64, 3);
                    acc += 4 * octave(x, y, 32, 4);
                    acc += 2 * octave(x, y, 16, 5);
                    acc += octave(x, y, 8, 6);
                    ((acc / 63) * 3 / 4 + 16 + ramp).clamp(0, 255) as u8
                }
                Content::Photo => unreachable!("handled above"),
            };
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            let mut au = 0i32;
            au += 32 * octave(x, y, 128, 11);
            au += 16 * octave(x, y, 64, 12);
            au += 8 * octave(x, y, 32, 13);
            au += 4 * octave(x, y, 16, 14);
            let u = au / 60;
            let mut av = 0i32;
            av += 32 * octave(x, y, 128, 21);
            av += 16 * octave(x, y, 64, 22);
            av += 8 * octave(x, y, 32, 23);
            av += 4 * octave(x, y, 16, 24);
            let v = av / 60;
            out[w * h + y * cw + x] = (128 + (u - 128) / 3).clamp(0, 255) as u8;
            out[w * h + cw * ch + y * cw + x] = (128 + (v - 128) / 3).clamp(0, 255) as u8;
        }
    }
    out
}

/// [`synth_i420`] for [`Content::Photo`], with the parameters spelled out — the
/// entry point `examples/content_fit.rs` sweeps. `synth_i420(w, h, Photo)` is
/// exactly this called with [`PHOTO`].
///
/// The chroma planes are the SAME generator the isotropic variants use, on
/// purpose: the axis under study is luma structure, and holding chroma fixed
/// across all three contents keeps the census comparison one-dimensional.
pub fn synth_i420_photo(w: usize, h: usize, p: &PhotoParams) -> Vec<u8> {
    assert!(w % 2 == 0 && h % 2 == 0, "even dimensions only");
    let (cw, ch) = (w / 2, h / 2);
    let mut out = vec![0u8; w * h + 2 * cw * ch];
    for y in 0..h {
        for x in 0..w {
            let ramp = (x * 40 / w + y * 40 / h) as i32;
            out[y * w + x] = (synth_photo_luma(x, y, p) + ramp - 20).clamp(0, 255) as u8;
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            let mut au = 0i32;
            au += 32 * octave(x, y, 128, 11);
            au += 16 * octave(x, y, 64, 12);
            au += 8 * octave(x, y, 32, 13);
            au += 4 * octave(x, y, 16, 14);
            let u = au / 60;
            let mut av = 0i32;
            av += 32 * octave(x, y, 128, 21);
            av += 16 * octave(x, y, 64, 22);
            av += 8 * octave(x, y, 32, 23);
            av += 4 * octave(x, y, 16, 24);
            let v = av / 60;
            out[w * h + y * cw + x] = (128 + (u - 128) / 3).clamp(0, 255) as u8;
            out[w * h + cw * ch + y * cw + x] = (128 + (v - 128) / 3).clamp(0, 255) as u8;
        }
    }
    out
}

/// The [`EncodeCell`] for `(w, h, cq, speed)` over [`synth_i420`].
pub fn cell(w: usize, h: usize, cq: i32, speed: i32, content: Content) -> EncodeCell {
    let buf = synth_i420(w, h, content);
    let (cw, ch) = (w / 2, h / 2);
    let up = |s: &[u8]| s.iter().map(|&b| u16::from(b)).collect::<Vec<u16>>();
    EncodeCell {
        label: format!("winperf-{}", content.label()),
        w,
        h,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        usage: 2, // ALLINTRA
        cq_level: cq,
        speed,
        bd: 8,
        y: up(&buf[..w * h]),
        u: up(&buf[w * h..w * h + cw * ch]),
        v: up(&buf[w * h + cw * ch..]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generator is a pinned constant, not just "some noise": if anyone
    /// edits it, every previously recorded winperf millisecond stops being
    /// comparable, so the change has to be deliberate. Checksums pin the
    /// content of all three planes at the study cell.
    ///
    /// Non-vacuity (playbook §2): the three sums are distinct, none is 0 or a
    /// saturated 255*n, and the luma variance assertion below fails outright
    /// for a constant plane — so this cannot pass against a blank image.
    #[test]
    fn synth_source_is_pinned_and_not_blank() {
        let (w, h, _, _) = CELL;
        for (content, want, lo_below, hi_above) in [
            (Content::Detail, (159_114_483u64, 34_350_299u64, 33_351_188u64), 90u8, 200u8),
            // Smooth's observed luma range is 80..=224 (Detail's is 66..=232),
            // and the two variants share a chroma generator by design — the
            // content axis under test is luma detail.
            (Content::Smooth, (167_367_154, 34_350_299, 33_351_188), 100, 190),
            // Photo's observed luma range is 56..=251 — wider than either
            // isotropic variant, because the streak field adds long coherent
            // ramps on top of the noise rather than averaging into it.
            (Content::Photo, (146_304_204, 34_350_299, 33_351_188), 70, 235),
        ] {
            let buf = synth_i420(w, h, content);
            let (cw, ch) = (w / 2, h / 2);
            assert_eq!(buf.len(), w * h + 2 * cw * ch);
            let sum = |s: &[u8]| s.iter().map(|&b| u64::from(b)).sum::<u64>();
            let y = sum(&buf[..w * h]);
            let u = sum(&buf[w * h..w * h + cw * ch]);
            let v = sum(&buf[w * h + cw * ch..]);
            assert_eq!(
                (y, u, v),
                want,
                "winperf {content:?} source changed — every recorded winperf \
                 timing was taken against the old one and is no longer comparable"
            );
            // Not blank, and not saturated. The bounds are strictly inside each
            // variant's observed range, so a flat or low-contrast plane fails.
            let mean = (y / (w * h) as u64) as i64;
            let var = buf[..w * h]
                .iter()
                .map(|&b| {
                    let d = i64::from(b) - mean;
                    d * d
                })
                .sum::<i64>()
                / (w * h) as i64;
            assert!(var > 400, "{content:?} luma variance {var} too low");
            assert!(buf[..w * h].iter().any(|&b| b < lo_below));
            assert!(buf[..w * h].iter().any(|&b| b > hi_above));
        }
        // The variants must actually be different content, or the bracket is
        // one point several times.
        for (a, b) in [
            (Content::Detail, Content::Smooth),
            (Content::Detail, Content::Photo),
            (Content::Smooth, Content::Photo),
        ] {
            assert_ne!(synth_i420(64, 64, a), synth_i420(64, 64, b), "{a:?} == {b:?}");
        }
    }

    /// Mean absolute luma difference at radius 4 in each of eight directions,
    /// over one `bs x bs` block. The census's own statistic, computed without
    /// the encoder.
    fn block_dir_grad(y: &[u8], w: usize, h: usize, bx: usize, by: usize, bs: usize) -> [f64; 8] {
        const DIRS: [(i32, i32); 8] =
            [(4, 0), (4, 2), (3, 3), (2, 4), (0, 4), (-2, 4), (-3, 3), (-4, 2)];
        let mut g = [0.0; 8];
        for (i, (dx, dy)) in DIRS.iter().enumerate() {
            let (mut acc, mut n) = (0u64, 0u64);
            for yy in by * bs + 4..by * bs + bs + 4 {
                for xx in bx * bs + 4..bx * bs + bs + 4 {
                    if yy + 4 >= h || xx + 4 >= w {
                        continue;
                    }
                    let a = i64::from(y[yy * w + xx]);
                    let b = i64::from(y[((yy as i32 + dy) as usize) * w + (xx as i32 + dx) as usize]);
                    acc += (a - b).unsigned_abs();
                    n += 1;
                }
            }
            g[i] = acc as f64 / n.max(1) as f64;
        }
        g
    }

    /// **`Photo` is ORIENTED and `Detail` is not** — the one property `Photo`
    /// exists for, pinned without needing an encoder.
    ///
    /// Isotropic value noise has, by construction, the same expected structure
    /// in every direction, so within any block its eight directional gradients
    /// are near-equal and their max/min ratio sits just above 1. `Photo`'s
    /// streak field makes one direction locally dominant, which is exactly what
    /// gives the directional intra predictors something to win on.
    ///
    /// Non-vacuity (playbook §2): `Detail` is measured in the SAME test and is
    /// the comparator, so this cannot pass by measuring nothing — if the streak
    /// field were dropped, `Photo` would collapse onto `Detail`'s value and the
    /// ratio assertion would fail. The gradient-magnitude floor separately
    /// rejects a degenerate "oriented because it is nearly flat" source, which
    /// is how `Smooth` scores a high ratio on near-zero gradients.
    #[test]
    fn photo_is_locally_oriented_and_detail_is_not() {
        let (w, h, _, _) = CELL;
        let bs = 32;
        let stat = |c: Content| -> (f64, f64) {
            let buf = synth_i420(w, h, c);
            let y = &buf[..w * h];
            let mut ratios = Vec::new();
            let mut gsum = 0.0;
            for by in 0..(h - 8) / bs {
                for bx in 0..(w - 8) / bs {
                    let g = block_dir_grad(y, w, h, bx, by, bs);
                    let lo = g.iter().cloned().fold(f64::MAX, f64::min).max(0.01);
                    let hi = g.iter().cloned().fold(0.0, f64::max);
                    ratios.push(hi / lo);
                    gsum += g.iter().sum::<f64>() / 8.0;
                }
            }
            let n = ratios.len();
            ratios.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
            (ratios[n / 2], gsum / n as f64)
        };
        let (aniso_photo, grad_photo) = stat(Content::Photo);
        let (aniso_detail, grad_detail) = stat(Content::Detail);
        assert!(
            aniso_photo > 1.4 * aniso_detail,
            "Photo's local direction anisotropy ({aniso_photo:.3}) is not \
             materially above Detail's ({aniso_detail:.3}) — the streak field \
             is not doing anything, and the content is back to isotropic"
        );
        assert!(
            grad_photo > 3.0 && grad_detail > 3.0,
            "gradient magnitudes too low (photo {grad_photo:.3}, detail \
             {grad_detail:.3}) — an oriented but nearly FLAT source scores a \
             high ratio while giving the encoder nothing to predict"
        );
    }

    /// Every committed bootstrap is a real section-5 stream for the study cell,
    /// and they are all different.
    #[test]
    fn bootstrap_fixtures_decode_and_are_streams() {
        for c in Content::ALL {
            let b = bootstrap(c);
            assert!(b.len() > 64, "{c:?} bootstrap fixture is {} bytes", b.len());
            // First OBU must be a temporal delimiter (type 2) with obu_has_size.
            assert_eq!((b[0] >> 3) & 0xf, 2, "{c:?}: first OBU is not a TD");
            assert_eq!((b[0] >> 1) & 1, 1, "{c:?}: obu_has_size_field not set");
        }
        for (i, a) in Content::ALL.iter().enumerate() {
            for b in &Content::ALL[i + 1..] {
                assert_ne!(bootstrap(*a), bootstrap(*b), "{a:?} == {b:?}");
            }
        }
    }
}
