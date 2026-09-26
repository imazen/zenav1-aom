#![allow(clippy::too_many_arguments)] // C-shaped harness signatures
//! Scratch probe (KB-65): find cells where C's IntraBC arm actually
//! ENGAGES at coded-lossless — `palette=0 intrabc=1` must differ from
//! `palette=0 intrabc=0` — and check port-vs-C byte parity there.

use aom_bench::winperf::{Content, synth_i420};
use aom_encode::key_frame::{KeyFrameConfig, KeyFramePlanes, encode_key_frame};
use aom_sys_ref as c;

fn planes(w: usize, h: usize, ct: Content) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let i420 = synth_i420(w, h, ct);
    let (cw, ch) = (w / 2, h / 2);
    let y: Vec<u16> = i420[..w * h].iter().map(|&b| u16::from(b)).collect();
    let u: Vec<u16> = i420[w * h..w * h + cw * ch]
        .iter()
        .map(|&b| u16::from(b))
        .collect();
    let v: Vec<u16> = i420[w * h + cw * ch..]
        .iter()
        .map(|&b| u16::from(b))
        .collect();
    (y, u, v)
}

fn csc(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    cq: i32,
    sp: i32,
    pal: bool,
    ibc: bool,
) -> Vec<u8> {
    c::ref_encode_av1_kf_screen_content(
        y, u, v, w, h, 8, false, 1, 1, cq, sp, false, true, 2, 0, false, pal, ibc,
    )
}

fn port(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    w: usize,
    h: usize,
    cq: i32,
    sp: i32,
    pal: bool,
    ibc: bool,
) -> Vec<u8> {
    let mut cfg = KeyFrameConfig::allintra_speed0(w, h, 8, false, 1, 1, cq);
    cfg.cpu_used = sp;
    cfg.enable_restoration = true;
    cfg.enable_palette = pal;
    cfg.enable_intrabc = ibc;
    encode_key_frame(KeyFramePlanes::new(y, u, v), &cfg).expect("encode")
}

fn main() {
    c::ref_init();
    // `ibc_lossless_probe dump <w> <h> <cq> <speed> <prefix>` — write the
    // intrabc-on streams for both sides to <prefix>.{port,c}.
    let a: Vec<String> = std::env::args().collect();
    if a.len() > 1 && a[1] == "dump" {
        let (w, h, cq, sp, prefix) = (
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
            a[4].parse().unwrap(),
            a[5].parse().unwrap(),
            a[6].clone(),
        );
        let (y, u, v) = planes(w, h, Content::Screen);
        for (tag, pal, ibc) in [("off", false, false), ("ibc", false, true)] {
            let cs = csc(&y, &u, &v, w, h, cq, sp, pal, ibc);
            let ps = port(&y, &u, &v, w, h, cq, sp, pal, ibc);
            std::fs::write(format!("{prefix}.{tag}.c"), &cs).unwrap();
            std::fs::write(format!("{prefix}.{tag}.port"), &ps).unwrap();
            let fd = cs.iter().zip(&ps).position(|(a, b)| a != b);
            let cdec_port = std::panic::catch_unwind(|| c::ref_decode_av1_kf(&ps, w, h).y.len());
            let pdec_port = aom_decode::frame::decode_frame_obus(&ps)
                .map(|f| f.y.len())
                .map_err(|e| format!("{e:?}"));
            eprintln!(
                "{tag}: c={} port={} first_diff={:?} c_dec(port)={:?} port_dec(port)={:?}",
                cs.len(),
                ps.len(),
                fd,
                cdec_port,
                pdec_port
            );
        }
        return;
    }
    // `ibc_lossless_probe dec <file>` — port-decode one stream with whatever
    // env-gated tracing is set (AOM_DBG_BLOCKS etc.).
    if a.len() > 1 && a[1] == "dec" {
        let bytes = std::fs::read(&a[2]).unwrap();
        match aom_decode::frame::decode_frame_obus(&bytes) {
            Ok(f) => eprintln!("decoded ok {}x{}", f.y.len(), f.u.len()),
            Err(e) => eprintln!("decode failed: {e:?}"),
        }
        return;
    }
    for ct in [Content::Screen, Content::Detail] {
        for &(w, h) in &[(64usize, 64usize), (128, 128), (256, 256), (512, 384)] {
            for &cq in &[0i32, 32] {
                for &sp in &[0i32, 3, 6] {
                    let (y, u, v) = planes(w, h, ct);
                    let off = csc(&y, &u, &v, w, h, cq, sp, false, false);
                    let ibc = csc(&y, &u, &v, w, h, cq, sp, false, true);
                    let engaged = ibc != off;
                    let (pc, pp) = (
                        port(&y, &u, &v, w, h, cq, sp, false, true),
                        port(&y, &u, &v, w, h, cq, sp, false, false),
                    );
                    let parity = pc == ibc;
                    println!(
                        "{ct:?} {w}x{h} cq{cq} s{sp}: c_off={} c_ibc={} engaged={engaged} \
                         port_ibc={} port_off={} parity={parity}",
                        off.len(),
                        ibc.len(),
                        pc.len(),
                        pp.len()
                    );
                }
            }
        }
    }
}
