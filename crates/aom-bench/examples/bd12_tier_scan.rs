//! Hunt a CHEAP reproducer for the bd12 dispatch-tier disagreement.
//!
//! The standing goal names this as one of two classes that must CLOSE (it is
//! not a divergence): at `bd12 1920x1080 cq24 cpu0` the port emits +181 B on
//! default dispatch and +55 B under `AOM_FORCE_SCALAR=1` — i.e. the port
//! disagrees with ITSELF across dispatch tiers, which is a differential hole
//! (playbook §1), not a byte.
//!
//! That cell is ~85 s per port encode, far too slow to bisect against. This
//! sweeps SMALL bd12 cells and prints `label len hash` for the port's own
//! payload; run it once with and once without `AOM_FORCE_SCALAR=1` and diff.
//! Any row whose hash moves is a reproducer, and the smallest one is the cell
//! to localize on.
//!
//! ```text
//! cargo build --release -p zenav1-aom-bench --example bd12_tier_scan
//! ./target/release/examples/bd12_tier_scan > /tmp/simd.txt
//! AOM_FORCE_SCALAR=1 ./target/release/examples/bd12_tier_scan > /tmp/scal.txt
//! diff /tmp/simd.txt /tmp/scal.txt
//! ```

use aom_bench::{EncodeCell, ToggleKnobs};
use aom_sys_ref as c;

/// Widen a cell's samples to `bd`, replicating the top bits into the new low
/// bits so the result is genuinely bd12 content (not bd10 values in a bd12
/// container, which would leave the low bits zero and hide bit-depth-sensitive
/// kernels). Same recipe as `s4cov_hd_format_axis::to_bd`.
fn to_bd(base: &EncodeCell, label: &str, bd: u8) -> EncodeCell {
    assert!(bd > base.bd, "{label}: to_bd only widens");
    let k = u32::from(bd - base.bd);
    let src_bits = u32::from(base.bd);
    let widen = |v: &u16| -> u16 { (v << k) | (v >> (src_bits - k)) };
    EncodeCell {
        label: label.to_string(),
        bd,
        y: base.y.iter().map(widen).collect(),
        u: base.u.iter().map(widen).collect(),
        v: base.v.iter().map(widen).collect(),
        ..base.clone()
    }
}


/// Mirror-tile a small cell up to `w`x`h` — the recipe every >=1080p gate in
/// this repo uses (`kb28_crop_dims::mirror_tile`), since no corpus vector is
/// that large.
fn mirror_tile(base: &EncodeCell, label: &str, w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
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
    EncodeCell { label: label.to_string(), w, h, cq_level: cq, speed, y, u, v, ..base.clone() }
}

fn fnv64(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn main() {
    c::ref_init();
    // Crops of the real bd10 vector, widened to bd12. Sizes deliberately span
    // SB-exact (64/128/192/256) and partial-superblock (100/196) shapes, since
    // several past roots were reachable only through a partial SB.
    // `WxH,WxH` and `cq,cq` override the default small grid, so the same
    // binary can be pointed at the >=1080p band (which needs mirror-tiling,
    // since the source vector is far smaller than 1080p).
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sizes: Vec<(usize, usize)> = match args.first() {
        Some(a) => a
            .split(',')
            .map(|t| {
                let (w, h) = t.split_once('x').expect("WxH");
                (w.parse().unwrap(), h.parse().unwrap())
            })
            .collect(),
        None => vec![(64, 64), (100, 100), (128, 128), (196, 196), (192, 192), (256, 256)],
    };
    let cqs: Vec<i32> = match args.get(1) {
        Some(a) => a.split(',').map(|t| t.parse().unwrap()).collect(),
        None => vec![24, 32, 48],
    };
    for &(w, h) in &sizes {
        for &cq in &cqs {
            let src = EncodeCell::real_content(
                "tier10",
                "av1-1-b10-00-quantizer-00",
                None,
                cq,
                0,
            );
            let b10 = mirror_tile(&src, "tier10m", w, h, cq, 0);
            let cell = to_bd(&b10, &format!("bd12_{w}x{h}_cq{cq}"), 12);
            let c_tu = cell.c_encode_ctrls(&[]);
            if c_tu.is_empty() {
                println!("{:<22} C-ENCODE-FAILED", cell.label);
                continue;
            }
            let real = EncodeCell::frame_obu_payload(&c_tu);
            let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cell.port_encode_with(&c_tu, &ToggleKnobs::default())
            }));
            match got {
                Ok(p) => println!(
                    "{:<22} port_len {:>7} hash {:016x}  c_len {:>7} delta {:+}",
                    cell.label,
                    p.len(),
                    fnv64(&p),
                    real.len(),
                    p.len() as i64 - real.len() as i64
                ),
                Err(_) => println!("{:<22} PANIC", cell.label),
            }
        }
    }
}
