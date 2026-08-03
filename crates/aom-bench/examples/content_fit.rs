//! `content_fit` — pick [`aom_bench::winperf::PHOTO`]'s parameters by
//! **measuring the reference distribution**, not by eye.
//!
//! # Why this is a separate tool from `content_census`
//!
//! `winperf`'s existing sources were fitted on ONE axis (allocator call count)
//! and were then quoted for a lever on a DIFFERENT axis (intra mode family),
//! where `detail` turns out to fire `z1` six times in a whole 1 MP frame. The
//! fix is not "a better guess at a synthetic image"; it is to make the content
//! choice a fit against a measured target, with the grid and the objective
//! written down.
//!
//! So: this sweeps [`PhotoParams`], censuses every candidate through the real
//! encoder, and reports each one's distance to the reference. The winner is
//! pasted into `winperf::PHOTO` and the whole table is committed
//! (`benchmarks/winperf_content_census_2026-08-03.*`).
//!
//! # The objective, fixed BEFORE the sweep runs
//!
//! `L1_class` — the L1 distance, in percentage points, between the candidate's
//! and the reference's **intra-class predicted-pixel share** vector
//! (`filter / non-dir / z1 / z2 / z3 / V / H`). Lower is better; 0 is identical.
//!
//! It is deliberately NOT any lever's delta. Fitting content until a lever's
//! number looks good is how a harness stops measuring anything
//! (`docs/DIFFERENTIAL_PLAYBOOK.md` §14); fitting it to the reference
//! distribution and then measuring once is not. `L1_bsize` (partition depth)
//! and `L1_tx` (transform type x size) are reported alongside as secondary
//! evidence and are NOT part of the objective — they are there so the record
//! can say honestly what the winner still gets wrong.
//!
//! # The three passes
//!
//! 1. **the mixture axis** — `mix` swept whole, everything else coarse. This is
//!    the axis that moves the directional share, and it has to be shown to have
//!    an INTERIOR optimum or the "fit" is just a saturated edge.
//! 2. **shape** — the streak periods, the fine-octave amplitude and the
//!    contrast, around pass 1's basin.
//! 3. **refinement** — a finer grid around pass 2's leader, on the axes where
//!    pass 2's best sat on a grid EDGE (`contrast` low, `streak_a.1` low). An
//!    edge optimum is not an optimum, and a fit that stops at one is a fit that
//!    stopped when the number looked good.
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --features census,c-oracle \
//!     --example content_fit -- <reference.yuv> <w> <h> [1|2|3|all]
//! ```

//! # The SCREEN fit (`--screen`), added 2026-08-03
//!
//! Same discipline, different objective, because the families it exists to
//! reach are different ones.
//!
//! **`L1_screen`, declared before the sweep ran**: the L1 distance, in
//! percentage points, between the candidate's and the reference's **coded-leaf
//! screen-tool class share** vector. The classes partition coded leaves — each
//! leaf falls in exactly one, tested in this fixed order —
//! `[intrabc, palette_y, filter_intra, directional_y, other]`.
//!
//! It is a LEAF distribution rather than the photo fit's predicted-pixel one
//! because palette and intraBC leaves issue no intra prediction at all: a
//! pixel-share objective would be blind to precisely the two tools this content
//! exists to reach. `L1_bsize`, `L1_class`, the UV-palette share and the CFL
//! share are reported alongside and are NOT part of the objective.
//!
//! Both tools need `allow_screen_content_tools` in the frame header, which real
//! `aomenc` signals from its OWN screen detection. Every candidate is
//! bootstrapped through `c_encode_screen(true, true)` and the `scdet` column
//! reports whether detection fired; a candidate where it did not is not a
//! candidate at all, however good its other columns look.
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --features census,c-oracle \
//!     --example content_fit -- --screen <reference.yuv> <w> <h> [1|2|all]
//! ```

use aom_bench::winperf::{PhotoParams, ScreenParams};
use aom_bench::{EncodeCell, ToggleKnobs, winperf};
use aom_dsp::census::{self, Counts, INTRA_CLASS, N_BSIZE, N_TX_SIZE, N_TX_TYPE};

fn main() {
    assert!(census::enabled(), "built without --features census");
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    if a.first().map(String::as_str) == Some("--screen") {
        a.remove(0);
        return screen_fit(&a);
    }
    assert!((3..=4).contains(&a.len()), "usage: content_fit <reference.yuv> <w> <h> [1|2|3|all]");
    let (w, h): (usize, usize) = (a[1].parse().unwrap(), a[2].parse().unwrap());
    let pass = a.get(3).map_or("all", |s| s.as_str());
    let want = |n: &str| pass == "all" || pass == n;

    // ---- reference --------------------------------------------------------
    let refc = census_of(&yuv_cell(&a[0], w, h));
    eprintln!(
        "reference {}: dir_px {:.2} % dir_calls {:.2} % intra_calls {} coded {}",
        a[0],
        pct(refc.0.directional_px(), refc.0.intra_total_px()),
        pct(refc.0.directional_calls(), refc.0.intra_total_calls()),
        refc.0.intra_total_calls(),
        refc.1,
    );

    println!(
        "orient_p\tstreak_p0\tstreak_p1\tstreak_a0\tstreak_a1\tmix_streak\tmix_iso\tcontrast\t\
         dir_pct_px\tdir_pct_calls\tnondir_pct_px\tL1_class\tL1_bsize\tL1_tx\t\
         intra_calls\tintra_px\tfwd_tx\tleaves\tcoded_bytes"
    );

    // ---- the grid ---------------------------------------------------------
    //
    // Coarse and small on purpose. `mix` is the knob that moves the directional
    // share monotonically (it is a linear blend between a purely oriented field
    // and a purely isotropic one), so a 1-D-dominant grid is the right shape;
    // the other axes set HOW oriented the oriented half is.
    let iso = [26, 22, 18, 15, 12, 10]; // `detail`'s ladder, which already matches the work level
    for &orient_period in if want("1") { &[64usize, 128, 256][..] } else { &[][..] } {
        for &streak_p in &[(32i32, 8i32), (48, 12), (96, 24)] {
            for &streak_a in &[(40i32, 14i32)] {
                for &mix in &[(2i32, 8i32), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2)] {
                    for &contrast in &[256i32] {
                        let p = PhotoParams {
                            orient_period,
                            streak_p,
                            streak_a,
                            iso_a: iso,
                            mix,
                            contrast,
                        };
                        row(&p, w, h, &refc.0);
                    }
                }
            }
        }
    }
    // A second pass on the best-shaped axes, varying the streak amplitude
    // ladder and the contrast — held fixed above so the first pass reads as one
    // clean axis.
    for &orient_period in if want("2") { &[128usize, 256][..] } else { &[][..] } {
        for &streak_p in &[(48i32, 12i32), (64, 16)] {
            for &streak_a in &[(40i32, 8i32), (40, 14), (40, 22), (40, 30)] {
                for &mix in &[(4i32, 6i32), (5, 5), (6, 4)] {
                    for &contrast in &[208i32, 256, 304] {
                        let p = PhotoParams {
                            orient_period,
                            streak_p,
                            streak_a,
                            iso_a: iso,
                            mix,
                            contrast,
                        };
                        row(&p, w, h, &refc.0);
                    }
                }
            }
        }
    }
    // Pass 3. Pass 2's leader was `orient 128 / streak_p (64,16) / streak_a
    // (40,8) / mix (6,4) / contrast 208`, and TWO of those (`streak_a.1` and
    // `contrast`) were the smallest value on their axis — i.e. the grid ran out
    // before the optimum did. This pass extends both downward and refines `mix`
    // on a /20 denominator so the ratio can move by less than a whole step.
    for &orient_period in if want("3") { &[96usize, 128, 192][..] } else { &[][..] } {
        for &streak_p in &[(56i32, 14i32), (64, 16), (80, 20)] {
            for &streak_a in &[(40i32, 4i32), (40, 8), (40, 14)] {
                for &mix in &[(11i32, 9i32), (12, 8), (13, 7)] {
                    for &contrast in &[160i32, 184, 208] {
                        let p = PhotoParams {
                            orient_period,
                            streak_p,
                            streak_a,
                            iso_a: iso,
                            mix,
                            contrast,
                        };
                        row(&p, w, h, &refc.0);
                    }
                }
            }
        }
    }
    // Pass 4. Pass 3's leader sat on the LONG end of `streak_p` (80,20), so the
    // same edge objection applies again; this walks it out past the point where
    // L1 starts rising. Small on purpose — everything else is held at pass 3's
    // winner.
    for &orient_period in if want("4") { &[128usize][..] } else { &[][..] } {
        for &streak_p in &[(80i32, 20i32), (96, 24), (112, 28), (128, 32)] {
            for &streak_a in &[(40i32, 4i32), (40, 8)] {
                for &mix in &[(12i32, 8i32)] {
                    for &contrast in &[184i32, 208] {
                        let p = PhotoParams {
                            orient_period,
                            streak_p,
                            streak_a,
                            iso_a: iso,
                            mix,
                            contrast,
                        };
                        row(&p, w, h, &refc.0);
                    }
                }
            }
        }
    }
}

fn row(p: &PhotoParams, w: usize, h: usize, refc: &Counts) {
    let (c, bytes) = census_of(&photo_cell(p, w, h));
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{}\t{}\t{}\t{}\t{}",
        p.orient_period,
        p.streak_p.0,
        p.streak_p.1,
        p.streak_a.0,
        p.streak_a.1,
        p.mix.0,
        p.mix.1,
        p.contrast,
        pct(c.directional_px(), c.intra_total_px()),
        pct(c.directional_calls(), c.intra_total_calls()),
        pct(c.intra_px[1].iter().sum::<u64>(), c.intra_total_px()),
        l1_class(&c, refc),
        l1(&c.leaf_bsize[..N_BSIZE], &refc.leaf_bsize[..N_BSIZE]),
        l1(&flat_tx(&c), &flat_tx(refc)),
        c.intra_total_calls(),
        c.intra_total_px(),
        c.fwd_tx.iter().flatten().sum::<u64>(),
        c.leaf_bsize.iter().sum::<u64>(),
        bytes,
    );
}

fn flat_tx(c: &Counts) -> Vec<u64> {
    (0..N_TX_TYPE * N_TX_SIZE).map(|i| c.fwd_tx[i / N_TX_SIZE][i % N_TX_SIZE]).collect()
}

fn pct(n: u64, d: u64) -> f64 {
    if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 }
}

fn l1_class(a: &Counts, b: &Counts) -> f64 {
    (0..INTRA_CLASS.len())
        .map(|i| {
            (pct(a.intra_px[i].iter().sum::<u64>(), a.intra_total_px())
                - pct(b.intra_px[i].iter().sum::<u64>(), b.intra_total_px()))
            .abs()
        })
        .sum()
}

fn l1(a: &[u64], b: &[u64]) -> f64 {
    let (sa, sb) = (a.iter().sum::<u64>(), b.iter().sum::<u64>());
    a.iter().zip(b).map(|(x, y)| (pct(*x, sa) - pct(*y, sb)).abs()).sum()
}

/// One encode's census, plus the coded byte count. A warm-up encode's counts
/// are subtracted so nothing built lazily on first use lands in the census.
fn census_of(cell: &EncodeCell) -> (Box<Counts>, usize) {
    let boot = cell.c_encode_defaults();
    assert!(!boot.is_empty(), "the C bootstrap encode produced nothing");
    census::reset();
    let _ = cell.port_encode(&boot);
    let base = census::snapshot();
    let out = cell.port_encode(&boot);
    let c = census::snapshot().since(&base);
    assert!(!c.is_empty(), "empty census — is the `census` feature on?");
    (c, out.len())
}

fn cell_from(label: String, w: usize, h: usize, buf: &[u8]) -> EncodeCell {
    let (cw, ch) = (w / 2, h / 2);
    assert_eq!(buf.len(), w * h + 2 * cw * ch);
    let (_, _, q, s) = winperf::CELL;
    let up = |x: &[u8]| x.iter().map(|&b| u16::from(b)).collect::<Vec<u16>>();
    EncodeCell {
        label,
        w,
        h,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        usage: 2, // ALLINTRA, the study cell
        cq_level: q,
        speed: s,
        bd: 8,
        y: up(&buf[..w * h]),
        u: up(&buf[w * h..w * h + cw * ch]),
        v: up(&buf[w * h + cw * ch..]),
    }
}

fn yuv_cell(path: &str, w: usize, h: usize) -> EncodeCell {
    let buf = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    cell_from(format!("ref:{path}"), w, h, &buf)
}

fn photo_cell(p: &PhotoParams, w: usize, h: usize) -> EncodeCell {
    cell_from("photo-candidate".to_string(), w, h, &winperf::synth_i420_photo(w, h, p))
}

// ---------------------------------------------------------------------------
// The screen fit
// ---------------------------------------------------------------------------

/// Screen-tool leaf classes, in the priority order a leaf is tested against.
/// They partition coded leaves, so the share vector is a distribution and its
/// L1 is bounded by 200 pp like the photo fit's.
const SCREEN_CLASS: [&str; 5] = ["intrabc", "palette_y", "filter_intra", "dir_y", "other"];

fn screen_shares(c: &Counts) -> [f64; 5] {
    let leaves = c.leaves();
    let ibc = c.leaf_intrabc;
    let pal = c.palette_y_leaves();
    let fi = c.filter_intra_leaves();
    // Directional LUMA leaves that are none of the above. `leaf_angle_delta_y`
    // is only incremented for a directional leaf that is not palette / intraBC
    // / filter-intra (see `census::note_coded_leaf_impl`), so its total is
    // exactly the disjoint directional class.
    let dir = c.leaf_angle_delta_y.iter().sum::<u64>();
    let other = leaves.saturating_sub(ibc + pal + fi + dir);
    [
        pct(ibc, leaves),
        pct(pal, leaves),
        pct(fi, leaves),
        pct(dir, leaves),
        pct(other, leaves),
    ]
}

/// One screen candidate's census: the SCREEN bootstrap (so the frame header can
/// carry `allow_screen_content_tools`) and both screen knobs on.
fn screen_census_of(cell: &EncodeCell) -> (Box<Counts>, usize, bool) {
    let boot = cell.c_encode_screen(true, true);
    assert!(!boot.is_empty(), "the C screen bootstrap encode produced nothing");
    let scdet = aom_bench::stream_allows_screen_content_tools(&boot);
    let knobs =
        ToggleKnobs { enable_palette: true, enable_intrabc: true, ..Default::default() };
    census::reset();
    let _ = cell.port_encode_with(&boot, &knobs);
    let base = census::snapshot();
    let out = cell.port_encode_with(&boot, &knobs);
    let c = census::snapshot().since(&base);
    assert!(!c.is_empty(), "empty census — is the `census` feature on?");
    (c, out.len(), scdet)
}

fn screen_cell(p: &ScreenParams, w: usize, h: usize) -> EncodeCell {
    cell_from("screen-candidate".to_string(), w, h, &winperf::synth_i420_screen(w, h, p))
}

fn screen_fit(a: &[String]) {
    // The reference is a VECTOR, not an image. `--target ibc,pal,fi,dir,other`
    // takes it directly, which is how the shipped fit is run: a single screen
    // screenshot is a poor target (`gb82-sc/imac_dark` at 512x512 censuses at
    // 69.2 % intraBC / 4.7 % directional, a profile dominated by large flat
    // black regions), so the DECLARED objective is the per-class MEDIAN over
    // the screen corpus. Stating the rule rather than the image is what stops
    // the reference itself from being a choice made after seeing the answer.
    // A raw `.yuv` reference is still accepted, and is what produced the
    // corpus rows the median is taken over.
    let (target, label, w, h, pass) = if a[0] == "--target" {
        let v: Vec<f64> = a[1].split(',').map(|x| x.parse().expect("f64")).collect();
        assert_eq!(v.len(), 5, "--target wants 5 comma-separated percentages");
        let mut t = [0.0f64; 5];
        t.copy_from_slice(&v);
        (
            t,
            format!("target:{}", a[1]),
            a[2].parse::<usize>().unwrap(),
            a[3].parse::<usize>().unwrap(),
            a.get(4).map_or("all", |s| s.as_str()).to_string(),
        )
    } else {
        let (w, h): (usize, usize) = (a[1].parse().unwrap(), a[2].parse().unwrap());
        let (refc, refb, refs) = screen_census_of(&yuv_cell(&a[0], w, h));
        assert!(
            refs,
            "{}: the REFERENCE is not detected as screen content — palette and \
             intraBC are unreachable on it and it cannot be a screen reference",
            a[0]
        );
        eprintln!("reference {} coded {refb}", a[0]);
        (
            screen_shares(&refc),
            a[0].clone(),
            w,
            h,
            a.get(3).map_or("all", |s| s.as_str()).to_string(),
        )
    };
    let want = |n: &str| pass == "all" || pass == n;
    eprintln!(
        "reference {label}: ibc {:.2} % pal {:.2} % fi {:.2} % dir {:.2} % other {:.2} %",
        target[0], target[1], target[2], target[3], target[4],
    );

    println!(
        "n_levels\tglyph_px\tn_glyphs\tpanel_px\ttext_q8\tink_q8\taa\tn_chroma\timage_q8\tscdet\t\
         ibc_pct\tpal_pct\tfi_pct\tdir_pct\tother_pct\tL1_screen\t\
         pal_uv_pct\tcfl_pct\tleaves\tcoded_bytes"
    );

    // Pass 1 — the two axes that decide whether the tools fire AT ALL: the
    // colour alphabet (palette's precondition) and the glyph alphabet size
    // (intraBC's). Everything else coarse.
    if want("1") {
        for &n_levels in &[4usize, 6, 10, 16] {
            for &n_glyphs in &[8usize, 24, 64] {
                for &glyph_px in &[8usize, 16] {
                    screen_row(
                        &ScreenParams {
                            n_levels,
                            glyph_px,
                            n_glyphs,
                            panel_px: 128,
                            text_q8: 154,
                            ink_q8: 96,
                            aa: 1,
                            n_chroma: 4,
                            image_q8: 0,
                        },
                        w,
                        h,
                        &target,
                    );
                }
            }
        }
    }
    // Pass 2 — the axis pass 1 did not have.
    //
    // Every pass-1 candidate came back with 6-11 % of leaves in the `other`
    // class against the target's 47.07 %, and 1-2 % directional against 8.55 %:
    // the grid could not reach the target from ANY point, because a frame that
    // is wall-to-wall flat UI and glyphs has no ordinary intra blocks in it.
    // Real screenshots carry photos, icons and gradients. `image_q8` is that
    // missing degree of freedom (a share of panels filled with the oriented
    // photographic field), and it is swept here against the layout knobs.
    //
    // An optimum outside the grid is the same objection as an optimum on a grid
    // EDGE; the fix is to widen the generator, not to pick a friendlier target.
    if want("2") {
        for &image_q8 in &[64i32, 102, 128, 154, 179] {
            for &text_q8 in &[102i32, 154] {
                for &n_levels in &[10usize, 16] {
                    for &aa in &[1i32] {
                        screen_row(
                            &ScreenParams {
                                n_levels,
                                glyph_px: 16,
                                n_glyphs: 24,
                                panel_px: 128,
                                text_q8,
                                ink_q8: 96,
                                aa,
                                n_chroma: 4,
                                image_q8,
                            },
                            w,
                            h,
                            &target,
                        );
                    }
                }
            }
        }
    }
    // Pass 3 — walking the `n_levels` EDGE out.
    //
    // Pass 1's winner sat at the LARGEST `n_levels` on its grid (16), and an
    // optimum on a grid edge is not an optimum
    // (`benchmarks/winperf_content_census_2026-08-03.md` §3, where the photo
    // fit needed two extra passes for the same reason). This walks the colour
    // alphabet past the point where L1 stops improving, at pass 2's leading
    // `image_q8`.
    if want("3") {
        for &n_levels in &[16usize, 24, 32, 48] {
            for &image_q8 in &[0i32, 64, 128] {
                screen_row(
                    &ScreenParams {
                        n_levels,
                        glyph_px: 16,
                        n_glyphs: 8,
                        panel_px: 128,
                        text_q8: 154,
                        ink_q8: 96,
                        aa: 1,
                        n_chroma: 4,
                        image_q8,
                    },
                    w,
                    h,
                    &target,
                );
            }
        }
    }
    println!(
        "# columns: the 8 ScreenParams fields, scdet, the 5 declared classes, \
         L1_screen (the OBJECTIVE), then reported-but-not-optimised columns"
    );
}

fn screen_row(p: &ScreenParams, w: usize, h: usize, target: &[f64; 5]) {
    let (c, bytes, scdet) = screen_census_of(&screen_cell(p, w, h));
    let s = screen_shares(c.as_ref());
    let l1_screen: f64 = (0..SCREEN_CLASS.len()).map(|i| (s[i] - target[i]).abs()).sum();
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t\
         {:.2}\t{:.2}\t{}\t{}",
        p.n_levels,
        p.glyph_px,
        p.n_glyphs,
        p.panel_px,
        p.text_q8,
        p.ink_q8,
        p.aa,
        p.n_chroma,
        p.image_q8,
        u8::from(scdet),
        s[0],
        s[1],
        s[2],
        s[3],
        s[4],
        l1_screen,
        pct(c.palette_uv_leaves(), c.leaves()),
        pct(c.cfl_leaves(), c.leaf_chroma_ref),
        c.leaves(),
        bytes,
    );
}
