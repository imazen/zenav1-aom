//! `nonsquare_leaf_reach` — **what content actually reaches a NON-SQUARE leaf
//! in the nonrd (`--cpu-used` 8/9) estimate arm**, and does the port encode it
//! byte-identically when it does.
//!
//! Until 2026-08-02 `nonrd_pickmode::nonrd_leaf_tx_size` PANICKED on such a
//! leaf ("HANDOFF: nonrd estimate arm at non-square leaf bsize …", KB-32), so
//! `--cpu-used 9` could not encode frames it reached at all. That refusal
//! carried a measured reachability claim — *"of 18 large cells probed at speeds
//! 8 and 9 (768² through 5472x3648), NONE reach a non-square leaf. The only
//! cell in the tree that does is issue #6's 12000x9000 at cpu9"* — which was
//! already contradicted twice (KB-28 found two 0.9 MP cells; the encoder
//! hotspot profile found 1024² cq44) and is playbook §9 in its purest form: a
//! statement true of the cells that happened to be probed, written as though it
//! were general. This sweep replaces it with a measured SHAPE.
//!
//! Per cell it reports the exact number of multi-txb leaves the estimate arm
//! coded, broken down by `bsize`
//! (`aom_encode::nonrd_pickmode::multi_txb_leaf_counts`, which counts only the
//! non-square path), together with byte-identity against real aomenc. A cell
//! with `nsq > 0` is a cell the PRE-FIX port refused; a cell with `nsq == 0`
//! is one it could always encode. So the TSV is simultaneously the reachable
//! set and the byte gate over it.
//!
//! ```text
//! cargo run --profile test-fast -p zenav1-aom-bench --example nonsquare_leaf_reach \
//!     -- > ~/tmp/nsq_reach.tsv
//! ```
//!
//! Axes (the CLAUDE.md sweep rules: size tiny→large, quality dense across the
//! WHOLE range, every mode axis, several content classes):
//!
//! * **size** 64² … 2176², plus non-SB-multiple and non-square shapes;
//! * **quality** `--cq-level` 0..63 step 4 plus 63;
//! * **speed** 0..9 — the RD speeds are in the grid on purpose, because
//!   "can an RD speed reach it?" is a question the sweep should answer rather
//!   than a structural claim the reader has to take on trust;
//! * **content** the mirror-tiled conformance vector (real photographic
//!   content), a smooth synthetic gradient, and any raw I420 handed on the
//!   command line — which is how the encoder-profile cell
//!   (`~/tmp/xb/src/photo_1024.yuv`, 1024² cq44 cpu9) is reproduced.
//!
//! Columns: `content size cq speed port_bytes c_bytes delta verdict nsq
//! nsq_by_bsize`.
//!
//! `NSQ_VECTOR_SCAN=1` runs a different, small pass instead: 1024x1024 (which
//! is SB-EXACT, so the only route to a rect leaf is an INTERIOR variance win)
//! across ten decodes of the same conformance clip at different quantizers.
//! The flat — heavily quantized — decodes are the ones that reach; that is how
//! `kb34_nonsquare_nonrd_leaf.rs` gets an IN-REPO cell at the encoder hotspot
//! profile's exact 1024x1024 / cq44 / cpu-used 9 cell, whose own source is an
//! out-of-repo photograph.

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_encode::nonrd_pickmode::{multi_txb_leaf_counts, reset_multi_txb_leaf_counts};
use aom_sys_ref as c;

/// Mirror-tile a base cell up to `w x h` (the `kb31_mandatory_tiles` /
/// `kb32_nonrd_size_bands` / `kb28_crop_dims` recipe — same function, so the
/// cells here are comparable with those gates' rows).
fn mirror_tile(base: &EncodeCell, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let mir = |i: usize, n: usize| {
        let m = i % (2 * n);
        if m < n { m } else { 2 * n - 1 - m }
    };
    let (bw, bh) = (base.w, base.h);
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = base.y[mir(r, bh) * bw + mir(col, bw)];
        }
    }
    let (bcw, bch) = ((bw + base.ss_x) >> base.ss_x, (bh + base.ss_y) >> base.ss_y);
    let (cw, ch) = ((w + base.ss_x) >> base.ss_x, (h + base.ss_y) >> base.ss_y);
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            u[r * cw + col] = base.u[mir(r, bch) * bcw + mir(col, bcw)];
            v[r * cw + col] = base.v[mir(r, bch) * bcw + mir(col, bcw)];
        }
    }
    EncodeCell {
        label: format!("nsq_{w}x{h}"),
        w,
        h,
        mono: base.mono,
        ss_x: base.ss_x,
        ss_y: base.ss_y,
        usage: base.usage,
        cq_level: cq,
        speed,
        bd: base.bd,
        y,
        u,
        v,
    }
}

/// Crop / mirror-extend a raw I420 source to `w x h`.
fn from_i420(buf: &[u8], sw: usize, sh: usize, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let mir = |i: usize, n: usize| {
        let m = i % (2 * n);
        if m < n { m } else { 2 * n - 1 - m }
    };
    let (scw, sch) = (sw / 2, sh / 2);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let sy = &buf[..sw * sh];
    let su = &buf[sw * sh..sw * sh + scw * sch];
    let sv = &buf[sw * sh + scw * sch..];
    let mut y = vec![0u16; w * h];
    for r in 0..h {
        for col in 0..w {
            y[r * w + col] = u16::from(sy[mir(r, sh) * sw + mir(col, sw)]);
        }
    }
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for r in 0..ch {
        for col in 0..cw {
            u[r * cw + col] = u16::from(su[mir(r, sch) * scw + mir(col, scw)]);
            v[r * cw + col] = u16::from(sv[mir(r, sch) * scw + mir(col, scw)]);
        }
    }
    EncodeCell {
        label: format!("i420_{w}x{h}"),
        w,
        h,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        usage: 2,
        cq_level: cq,
        speed,
        bd: 8,
        y,
        u,
        v,
    }
}

fn row(content: &str, cell: &EncodeCell) {
    let c_tu = cell.c_encode_defaults();
    assert!(
        !c_tu.is_empty(),
        "{content} {}x{} cq{} s{}: C encode failed",
        cell.w,
        cell.h,
        cell.cq_level,
        cell.speed
    );
    let real = EncodeCell::frame_obu_payload(&c_tu);
    reset_multi_txb_leaf_counts();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cell.port_encode_with(&c_tu, &ToggleKnobs::default())
    }));
    std::panic::set_hook(hook);
    let msg = got.as_ref().err().map_or_else(String::new, |e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "<non-string panic payload>".to_string())
            .replace('\n', " ")
    });
    let counts = multi_txb_leaf_counts();
    let nsq: u64 = counts.iter().sum();
    let by: Vec<String> = counts
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 0)
        .map(|(b, n)| format!("b{b}:{n}"))
        .collect();
    let (verdict, plen, delta) = match &got {
        Ok(p) if p == &real => ("MATCH", p.len() as i64, 0),
        Ok(p) => ("DIVERGE", p.len() as i64, p.len() as i64 - real.len() as i64),
        Err(_) => ("PANIC", -1, 0),
    };
    println!(
        "{content}\t{}x{}\t{}\t{}\t{}\t{}\t{:+}\t{}\t{}\t{}\t{}",
        cell.w,
        cell.h,
        cell.w * cell.h,
        cell.cq_level,
        cell.speed,
        plen,
        delta,
        real.len(),
        verdict,
        nsq,
        if by.is_empty() {
            "-".to_string()
        } else {
            by.join(",")
        }
    );
    if verdict == "PANIC" {
        println!("# PANIC MESSAGE: {msg}");
    }
}

fn main() {
    c::ref_init();
    println!("# nonsquare_leaf_reach — multi-txb (non-square) leaves in the nonrd estimate arm");
    println!("content\tsize\tpx\tcq\tspeed\tport_bytes\tdelta\tc_bytes\tverdict\tnsq\tnsq_by_bsize");

    let args: Vec<String> = std::env::args().collect();
    // Optional raw I420 source: `-- <file.yuv> <w> <h>`.
    let raw = if args.len() >= 4 {
        let sw: usize = args[2].parse().unwrap();
        let sh: usize = args[3].parse().unwrap();
        Some((std::fs::read(&args[1]).expect("read .yuv"), sw, sh))
    } else {
        None
    };

    let base = EncodeCell::real_content("nsq", "av1-1-b8-00-quantizer-00", None, 30, 9);

    // Pass 0 (`NSQ_VECTOR_SCAN=1`) — which in-repo conformance vectors reach a
    // non-square leaf at an SB-EXACT size? 1024² is 16x16 whole superblocks, so
    // the only route there is an INTERIOR variance win, which needs locally
    // flat content; the intra corpus is one clip at 60 quantizers, and the
    // heavily-quantized decodes are the flat ones. This is how the durable gate
    // gets a 1024x1024 cq44 cpu-used 9 cell (the encoder hotspot profile's) out
    // of in-repo content instead of an out-of-repo photograph.
    if std::env::var_os("NSQ_VECTOR_SCAN").is_some() {
        for q in ["00", "10", "20", "30", "40", "50", "58", "60", "62", "63"] {
            let v = format!("av1-1-b8-00-quantizer-{q}");
            let b = EncodeCell::real_content("nsqv", &v, None, 44, 9);
            for &(w, h) in &[(1024usize, 1024usize)] {
                for &cq in &[24, 36, 44, 52] {
                    for speed in [8, 9] {
                        row(&format!("vec{q}"), &mirror_tile(&b, w, h, cq, speed));
                    }
                }
            }
        }
        return;
    }

    // (w, h) — tiny / small / medium / large, SB-exact and not, square and not.
    const SIZES: &[(usize, usize)] = &[
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (768, 768),
        (896, 896),
        (954, 962),
        (1024, 1024),
        (1272, 724),
        (1280, 720),
        (1920, 1080),
        (2176, 2176),
    ];
    // `--cq-level` 2..63. cq 0 is LOSSLESS (base_qindex 0) and is out of this
    // harness's scope entirely (`aom-bench/src/lib.rs:1134`), which has nothing
    // to do with the non-square arm; it is excluded rather than reported as 54
    // identical unrelated refusals.
    let cqs: Vec<i32> = std::iter::once(2)
        .chain((1..=15).map(|i| i * 4))
        .chain(std::iter::once(63))
        .collect();

    // Pass 1 — the whole quality range at the two nonrd speeds, on every size.
    for &(w, h) in SIZES {
        for &cq in &cqs {
            for speed in [8, 9] {
                row("real", &mirror_tile(&base, w, h, cq, speed));
                row("diag", &EncodeCell::synthetic_diag("nsq", w, h, cq, speed));
                if let Some((buf, sw, sh)) = &raw {
                    row("photo", &from_i420(buf, *sw, *sh, w, h, cq, speed));
                }
            }
        }
    }

    // Pass 3 — SMALL frames, PARTIAL-superblock vs SB-exact. Pass 1's sizes
    // below 954x962 are all multiples of 64 AND of 128, so "small never
    // reaches" there is confounded with "SB-exact rarely reaches": every one
    // of `set_vt_partitioning`'s frame-edge fit-check relaxations
    // (var_based_part.c:164-173) is dead on an SB-exact frame. This pass
    // separates the two by pairing each partial-SB size with the SB-exact size
    // just below it.
    for &(w, h) in &[
        (100usize, 100usize),
        (128, 128),
        (196, 196),
        (256, 256),
        (250, 250),
        (300, 260),
        (456, 328),
        (520, 520),
        (512, 512),
        (600, 600),
        (712, 716),
    ] {
        for &cq in &cqs {
            for speed in [8, 9] {
                row("smallreal", &mirror_tile(&base, w, h, cq, speed));
                row("smalldiag", &EncodeCell::synthetic_diag("nsq", w, h, cq, speed));
            }
        }
    }

    // Pass 2 — the RD speeds. `nonrd_pick_intra_mode` is dispatched only from
    // `pack_tile`'s `allintra && pick_cfg.speed >= 8` branch (pack.rs:1917), so
    // the estimate arm cannot run below 8; this is the measurement that says so
    // rather than the claim. Kept at 256² across every RD speed (cpu-used 0 at
    // 1 MP is minutes) plus the two speeds nearest the boundary at 1024².
    for &cq in &[8, 24, 44, 60] {
        for speed in 0..=7 {
            row("real", &mirror_tile(&base, 256, 256, cq, speed));
        }
        for speed in [6, 7] {
            row("real", &mirror_tile(&base, 1024, 1024, cq, speed));
        }
    }
}
