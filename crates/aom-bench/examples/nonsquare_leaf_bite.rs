//! `nonsquare_leaf_bite` — KB-34's **per-root bite proof and teeth harness**
//! (playbook §1). Byte-compares one fixed cell list against real aomenc and
//! reports PANIC / DIVERGE / MATCH per cell.
//!
//! It deliberately uses **no API that KB-34 added**, so the same binary builds
//! on the pristine tree as on the patched one. That is what lets the four arms
//! be produced by `git stash push -- <one file>` rather than by an in-code
//! toggle, and it is what makes the "before" arm a real before:
//!
//! | arm | stashed | result |
//! |---|---|---|
//! | before | both hunks | 14 PANIC |
//! | after | — | 0 |
//! | revert root 1 | `aom-encode/src/nonrd_pickmode.rs` | 14 PANIC (the same 14) |
//! | revert root 2 | `aom-encode/src/partition_pick.rs` | 9 PANIC (a strict subset) |
//!
//! The cell list is chosen so the two roots are SEPARABLE: the `gate` rows are
//! frame-edge rects (both roots), the `sbexact` / `profile` rows are SB-exact
//! frames reaching an INTERIOR rect (root 1 only), and `250x250` never reaches
//! the arm at all (the negative control — it must MATCH in every arm). The `rd`
//! and `photo8` rows carry pre-existing divergences whose byte counts must be
//! identical across all four arms, which is how "this landing moved nothing
//! else" is measured rather than argued.
//!
//! Output: `benchmarks/nonsquare_leaf_reach_bite_2026-08-02.tsv`.
//!
//! ```text
//! cargo run --profile test-fast -p zenav1-aom-bench --example nonsquare_leaf_bite \
//!     -- [<photo.yuv> <w> <h>]
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

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
        label: format!("probe_{w}x{h}"),
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
        label: format!("photo_{w}x{h}"),
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

fn row(tag: &str, cell: &EncodeCell) {
    let c_tu = cell.c_encode_defaults();
    let real = EncodeCell::frame_obu_payload(&c_tu);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cell.port_encode_with(&c_tu, &ToggleKnobs::default())
    }));
    std::panic::set_hook(hook);
    match got {
        Ok(p) if p == real => println!(
            "{tag}\t{}x{}\tcq{}\tcpu{}\tMATCH\t{}\t+0",
            cell.w,
            cell.h,
            cell.cq_level,
            cell.speed,
            p.len()
        ),
        Ok(p) => println!(
            "{tag}\t{}x{}\tcq{}\tcpu{}\tDIVERGE\t{}\t{:+}",
            cell.w,
            cell.h,
            cell.cq_level,
            cell.speed,
            p.len(),
            p.len() as i64 - real.len() as i64
        ),
        Err(e) => {
            let m = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_default()
                .replace('\n', " ");
            println!(
                "{tag}\t{}x{}\tcq{}\tcpu{}\tPANIC\t-1\t+0\t{m}",
                cell.w, cell.h, cell.cq_level, cell.speed
            );
        }
    }
}

fn main() {
    c::ref_init();
    let args: Vec<String> = std::env::args().collect();
    let raw = if args.len() >= 4 {
        let sw: usize = args[2].parse().unwrap();
        let sh: usize = args[3].parse().unwrap();
        Some((std::fs::read(&args[1]).expect("read .yuv"), sw, sh))
    } else {
        None
    };
    let base = EncodeCell::real_content("probe", "av1-1-b8-00-quantizer-00", None, 24, 9);

    // --- KB-34's gate cells (the teeth: these must PANIC before the fix) ---
    for &(w, h, cq, sp, real_c) in &[
        (1272usize, 724usize, 24i32, 9i32, true),
        (954, 962, 24, 9, true),
        (954, 962, 56, 8, true),
        (1920, 1080, 24, 8, true),
        (1920, 1080, 48, 9, true),
        (1280, 720, 24, 9, false),
        (1280, 720, 48, 9, false),
        (196, 196, 24, 9, true),
        (196, 196, 24, 8, true),
        (250, 250, 24, 9, true),
    ] {
        let cell = if real_c {
            mirror_tile(&base, w, h, cq, sp)
        } else {
            EncodeCell::synthetic_diag("probe", w, h, cq, sp)
        };
        row("gate", &cell);
    }

    // --- the RD-speed ladder on the KB-28 cell ---
    for speed in 0..=7 {
        row("rd", &mirror_tile(&base, 1272, 724, 24, speed));
    }

    // --- the cpu-8 photo divergences the sweep turned up (nsq == 0 rows) ---
    if let Some((buf, sw, sh)) = &raw {
        for &(w, h, cq) in &[
            (512usize, 512usize, 32i32),
            (512, 512, 44),
            (768, 768, 52),
            (896, 896, 52),
            (1024, 1024, 52),
            (1024, 1024, 63),
        ] {
            row("photo8", &from_i420(buf, *sw, *sh, w, h, cq, 8));
        }
        // and the encoder-profile cell itself
        row("profile", &from_i420(buf, *sw, *sh, 1024, 1024, 44, 9));
        // SB-EXACT frames that reach an INTERIOR rect (both strips in frame),
        // i.e. cells that need the txb walk and NOT the frame-edge constructor.
        for &(w, h, cq) in &[
            (768usize, 768usize, 32i32),
            (896, 896, 28),
            (1024, 1024, 24),
            (1024, 1024, 36),
        ] {
            row("sbexact", &from_i420(buf, *sw, *sh, w, h, cq, 9));
        }
    }
}
