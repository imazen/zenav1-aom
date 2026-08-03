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
}

impl Content {
    /// `"detail"` / `"smooth"`, for a command line.
    pub fn parse(s: &str) -> Self {
        match s {
            "detail" => Content::Detail,
            "smooth" => Content::Smooth,
            _ => panic!("unknown winperf content {s:?}; want `detail` or `smooth`"),
        }
    }

    /// The lower-case name `parse` accepts.
    pub fn label(self) -> &'static str {
        match self {
            Content::Detail => "detail",
            Content::Smooth => "smooth",
        }
    }
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

/// The committed bootstrap for `content`, decoded. Panics on any non-hex byte —
/// the fixture is checked in, so a failure here is a corrupted tree, not a
/// runtime condition.
pub fn bootstrap(content: Content) -> Vec<u8> {
    let hex = match content {
        Content::Detail => BOOTSTRAP_DETAIL_HEX,
        Content::Smooth => BOOTSTRAP_SMOOTH_HEX,
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
        // The two variants must actually be different content, or the bracket
        // is one point twice.
        assert_ne!(synth_i420(64, 64, Content::Detail), synth_i420(64, 64, Content::Smooth));
    }

    /// Both committed bootstraps are real section-5 streams for the study cell.
    #[test]
    fn bootstrap_fixtures_decode_and_are_streams() {
        for c in [Content::Detail, Content::Smooth] {
            let b = bootstrap(c);
            assert!(b.len() > 64, "{c:?} bootstrap fixture is {} bytes", b.len());
            // First OBU must be a temporal delimiter (type 2) with obu_has_size.
            assert_eq!((b[0] >> 3) & 0xf, 2, "{c:?}: first OBU is not a TD");
            assert_eq!((b[0] >> 1) & 1, 1, "{c:?}: obu_has_size_field not set");
        }
        assert_ne!(bootstrap(Content::Detail), bootstrap(Content::Smooth));
    }
}
