//! drv-aom — timed still-picture encode driver for the **zenav1-aom** pure-Rust
//! libaom port.
//!
//! Uniform driver contract (identical for all four encoders in this harness):
//!
//!   drv <w> <h> <q> <speed> <in.yuv> <out.obu> <warmup> <reps>
//!   stdout: `NS=<n> NS=<n> ... BYTES=<m>`   (one NS per timed rep)
//!
//! Timed region = the port's own frame encode only. Untimed setup: reading the
//! `.yuv`, and the C-libaom **bootstrap** encode. The bootstrap is a structural
//! requirement of this port, not a shortcut: per the project's Gate-2 scope
//! note, `zenav1-aom` never AUTHORS a sequence header — every encode path
//! parses one out of a real aomenc stream and emits only the `OBU_FRAME`. So a
//! matched-config C encode runs first (untimed) purely to supply the sequence
//! header; the frame bytes we measure and score are entirely the port's.
//!
//! `q` is `--cq-level` (0..63), `speed` is `--cpu-used` (0..9), ALLINTRA
//! (`usage = 2`), 8-bit 4:2:0, single tile, single thread — the port's primary
//! configuration. The bootstrap uses aomenc's true ALLINTRA defaults
//! (`c_encode_defaults`: CDEF off, loop-restoration ON), and the port's
//! restoration-aware default path is what runs.

//! `XBENCH_AOM_SCREEN_TOOLS=1` additionally turns ON the port's palette and
//! IntraBC RD searches. They are OFF in `ToggleKnobs::default()` (the path
//! `port_encode` takes), which is what the default runs measure — so this env
//! switch exists to QUANTIFY that gap on screen content rather than leave it
//! as an unmeasured caveat. The frame still has to signal
//! `allow_screen_content_tools` / `allow_intrabc` for either search to run.

use aom_bench::{EncodeCell, ToggleKnobs};
use std::time::Instant;

const OBU_FRAME: u8 = 6;

/// Minimal section-5 OBU walk: `(type, header_bytes, payload)` per OBU.
fn walk(mut s: &[u8]) -> Vec<(u8, Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    while !s.is_empty() {
        let mut hdr = vec![s[0]];
        let ty = (s[0] >> 3) & 0xf;
        let ext = (s[0] >> 2) & 1;
        let has_size = (s[0] >> 1) & 1;
        assert_eq!(has_size, 1, "obu_has_size_field must be set");
        let mut i = 1;
        if ext == 1 {
            hdr.push(s[1]);
            i = 2;
        }
        // leb128 size
        let mut size = 0usize;
        let mut shift = 0;
        loop {
            let b = s[i];
            i += 1;
            size |= ((b & 0x7f) as usize) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        out.push((ty, hdr, s[i..i + size].to_vec()));
        s = &s[i + size..];
    }
    out
}

fn leb128(mut v: usize) -> Vec<u8> {
    let mut o = Vec::new();
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        o.push(b);
        if v == 0 {
            return o;
        }
    }
}

/// Rebuild `bootstrap` with its OBU_FRAME payload replaced by the port's.
fn reassemble(bootstrap: &[u8], frame_payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for (ty, hdr, payload) in walk(bootstrap) {
        let p: &[u8] = if ty == OBU_FRAME { frame_payload } else { &payload };
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&leb128(p.len()));
        out.extend_from_slice(p);
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 9 {
        eprintln!("usage: drv-aom <w> <h> <cq 0..63> <cpu-used 0..9> <in.yuv> <out.obu> <warmup> <reps>");
        std::process::exit(2);
    }
    let w: usize = a[1].parse().unwrap();
    let h: usize = a[2].parse().unwrap();
    let q: i32 = a[3].parse().unwrap();
    let speed: i32 = a[4].parse().unwrap();
    let warmup: usize = a[7].parse().unwrap();
    let reps: usize = a[8].parse().unwrap();
    assert!(w % 2 == 0 && h % 2 == 0, "even dims only");

    let buf = std::fs::read(&a[5]).expect("read .yuv");
    let (cw, ch) = (w / 2, h / 2);
    assert_eq!(buf.len(), w * h + 2 * cw * ch, "I420 size mismatch");
    let up = |s: &[u8]| s.iter().map(|&b| u16::from(b)).collect::<Vec<u16>>();
    let cell = EncodeCell {
        label: "xbench".to_string(),
        w,
        h,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        usage: 2, // ALLINTRA
        cq_level: q,
        speed,
        bd: 8,
        y: up(&buf[..w * h]),
        u: up(&buf[w * h..w * h + cw * ch]),
        v: up(&buf[w * h + cw * ch..]),
    };

    // UNTIMED: the C sequence-header bootstrap (see the module note).
    let bootstrap = cell.c_encode_defaults();
    assert!(!bootstrap.is_empty(), "C bootstrap encode failed");

    let on = |k: &str| std::env::var(k).as_deref() == Ok("1");
    let both = on("XBENCH_AOM_SCREEN_TOOLS");
    let knobs = ToggleKnobs {
        enable_palette: both || on("XBENCH_AOM_PALETTE"),
        enable_intrabc: both || on("XBENCH_AOM_INTRABC"),
        ..ToggleKnobs::default()
    };
    for _ in 0..warmup {
        let _ = cell.port_encode_with(&bootstrap, &knobs);
    }
    let mut samples = Vec::with_capacity(reps);
    let mut last = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let payload = cell.port_encode_with(&bootstrap, &knobs);
        samples.push(t.elapsed().as_nanos());
        last = payload;
    }
    let stream = reassemble(&bootstrap, &last);
    std::fs::write(&a[6], &stream).expect("write .obu");
    let mut line = String::new();
    for s in &samples {
        line.push_str(&format!("NS={s} "));
    }
    // BYTES = the whole section-5 OBU stream, so it is directly comparable
    // with the other three drivers (all four emit exactly TD + 7-byte sequence
    // header + frame OBU = 14 bytes of non-frame overhead — verified). The
    // frame payload the port itself produced is reported separately.
    println!("{line}BYTES={} FRAMEBYTES={}", stream.len(), last.len());
}
