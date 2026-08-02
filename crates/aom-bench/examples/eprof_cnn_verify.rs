//! `eprof_cnn_verify` — prove the per-64x64 intra-CNN cache (KB-PERF-1) returns
//! exactly what a recomputation would, on a REAL encode of the profile cell.
//!
//! `intra_mode_cnn_partition` runs the 5-layer cascade once per `BLOCK_64X64`
//! and every 32x32 / 16x16 / 8x8 node inside that 64x64 reads the cached buffer
//! instead. The claim that this cannot move a byte rests on "every one of those
//! nodes would have convolved the identical 65x65 window" — so this turns the
//! claim into a measurement: with the check armed, EVERY cache read re-extracts
//! its window, re-runs the cascade, and asserts the result is bit-identical to
//! what was cached (`decision::set_cnn_cache_verify`). A single mismatch panics.
//!
//! It also prints the byte length of the coded frame, so the same run shows the
//! stream is unaffected, and the compute/read split — which is the 2558 -> 256
//! call-count claim from `benchmarks/encoder_hotspot_profile_2026-08-02.md`
//! measured from the other side (that profile counted allocator call sites;
//! this counts the cascade itself).
//!
//! Roughly 2x the wall time of a plain encode: every avoided cascade is run
//! after all, plus the one that was cached.
//!
//! ```text
//! cargo run --release -p zenav1-aom-bench --example eprof_cnn_verify -- \
//!     1024 1024 44 6 ~/tmp/xb/src/photo_1024.yuv
//! ```

use aom_bench::EncodeCell;
use aom_encode::cnn_partition::decision as cnn_dec;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 6 {
        eprintln!("usage: eprof_cnn_verify <w> <h> <cq> <cpu-used> <in.yuv>");
        std::process::exit(2);
    }
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let q: i32 = a[3].parse().unwrap();
    let speed: i32 = a[4].parse().unwrap();
    let buf = std::fs::read(&a[5]).expect("read .yuv");
    let (cw, ch) = (w / 2, h / 2);
    assert_eq!(buf.len(), w * h + 2 * cw * ch, "I420 size mismatch");
    let up = |s: &[u8]| s.iter().map(|&b| u16::from(b)).collect::<Vec<u16>>();
    let cell = EncodeCell {
        label: "eprof_cnn_verify".to_string(),
        w,
        h,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        usage: 2,
        cq_level: q,
        speed,
        bd: 8,
        y: up(&buf[..w * h]),
        u: up(&buf[w * h..w * h + cw * ch]),
        v: up(&buf[w * h + cw * ch..]),
    };

    let bootstrap = cell.c_encode_defaults();
    assert!(!bootstrap.is_empty());
    let real = EncodeCell::frame_obu_payload(&bootstrap);

    // Unverified pass: the compute counter alone (this is what ships).
    cnn_dec::reset_cnn_cache_stats();
    let plain = cell.port_encode(&bootstrap);
    let (computes_plain, reads_plain) = cnn_dec::cnn_cache_stats();

    // Verified pass: every cache read re-runs the cascade and asserts equality.
    cnn_dec::reset_cnn_cache_stats();
    cnn_dec::set_cnn_cache_verify(true);
    let checked = cell.port_encode(&bootstrap);
    cnn_dec::set_cnn_cache_verify(false);
    let (computes, reads) = cnn_dec::cnn_cache_stats();

    let sb = w.div_ceil(64) * h.div_ceil(64);
    println!("cell               {w}x{h} cq{q} cpu-used {speed}");
    println!("64x64 superblocks  {sb}");
    println!(
        "cascade COMPUTES   {computes}  ({:.2} per 64x64)",
        computes as f64 / sb as f64
    );
    println!("cache READS        {reads}  (all re-verified bit-identical)");
    println!(
        "nodes total        {}  ({:.2} per 64x64 — what the uncached port ran)",
        computes + reads,
        (computes + reads) as f64 / sb as f64
    );
    println!(
        "unverified pass    computes {computes_plain} reads {reads_plain} \
         (reads are only counted while armed)"
    );
    println!(
        "coded frame bytes  port {} / libaom {}",
        checked.len(),
        real.len()
    );
    assert_eq!(
        checked, plain,
        "the verification pass changed the bitstream"
    );
    assert_eq!(
        checked, real,
        "port frame differs from libaom's at this cell"
    );
    println!(
        "VERIFIED: {reads} cache reads bit-identical to a recomputation; \
         frame byte-identical to libaom."
    );
}
