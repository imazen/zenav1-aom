//! **How much does the SHIPPING path leave on the table for screen content?**
//!
//! `aom_encode::key_frame::encode_key_frame` — the entry zenavif calls — builds
//! its `PickFrameCfg` with `palette_costs: None` and `intrabc: None`, i.e. it
//! runs NEITHER screen-content RD search, on every frame, including frames
//! whose own header it writes with `allow_screen_content_tools = 1` from its
//! own detector. That is documented in the module as "matching `aom-sys-ref`'s
//! `shim_encode_av1_kf`, whose `--enable-palette=0`" — true, and the reason the
//! 427/427 byte gate holds: **its oracle is a palette-DISABLED libaom**
//! (`dec_shim.c:612`, `/*enable_palette=*/0, /*enable_intrabc=*/0`). So that
//! gate says nothing about screen content at aomenc's real ALLINTRA defaults,
//! where both tools are ON and gated only on the screen flag.
//!
//! This probe measures the size of that gap with libaom on BOTH sides, so the
//! number is a property of the TOOLS and not of the port's RD:
//!
//! * `palette=0 intrabc=0` — the envelope the shell actually implements;
//! * `palette=1 intrabc=0` — what pass 1 of `av1_determine_sc_tools_with_encoding`
//!   would enable (that trial leaves intrabc off; encoder_utils.c:1204);
//! * `palette=1 intrabc=1` — aomenc's ALLINTRA default on a screen-detected frame.
//!
//! Content is the committed, integer-only, census-gated `Content::Screen`
//! generator (`winperf::synth_i420`), because the screen corpus
//! (`codec-corpus/gb82-sc`) is not present on every box. Its palette reach is
//! pinned by `content_family_census`, so "this content reaches palette" is an
//! asserted property rather than an assumption.
//!
//! It runs the SHIPPING path (`encode_key_frame`) in the same three
//! configurations alongside, so the table answers both questions at once: what
//! the tools are worth, and whether this port captures it.
//!
//! Usage: `cargo run --release -p zenav1-aom-bench --example screen_tools_gap`

use aom_bench::winperf::{synth_i420, Content};
use aom_bench::EncodeCell;
use aom_encode::key_frame::{encode_key_frame, KeyFrameConfig, KeyFramePlanes};

fn cell(w: usize, h: usize, cq: i32, speed: i32) -> EncodeCell {
    let i420 = synth_i420(w, h, Content::Screen);
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
    EncodeCell {
        label: format!("screen_{w}x{h}_cq{cq}_s{speed}"),
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

fn main() {
    println!(
        "cell\tw\th\tcq\tspeed\tc_off\tc_pal\tc_both\tport_off\tport_pal\tport_both\t\
         pal_gain_pct\tboth_gain_pct\tport_vs_c_both"
    );
    for &(w, h) in &[(512usize, 384usize), (256, 256), (1024, 768)] {
        for &cq in &[20i32, 32, 44, 55] {
            for &speed in &[3i32, 6] {
                let c = cell(w, h, cq, speed);
                let a = c.c_encode_screen(false, false).len() as f64;
                let b = c.c_encode_screen(true, false).len() as f64;
                let d = c.c_encode_screen(true, true).len() as f64;
                let port = |pal: bool, ibc: bool| -> usize {
                    let mut cfg =
                        KeyFrameConfig::allintra_speed0(w, h, 8, false, 1, 1, cq);
                    cfg.cpu_used = speed;
                    cfg.enable_restoration = true;
                    cfg.enable_palette = pal;
                    cfg.enable_intrabc = ibc;
                    encode_key_frame(
                        KeyFramePlanes {
                            y: &c.y,
                            u: &c.u,
                            v: &c.v,
                        },
                        &cfg,
                    )
                    .expect("encode_key_frame")
                    .len()
                };
                let (po, pp, pb) = (port(false, false), port(true, false), port(true, true));
                println!(
                    "{}\t{w}\t{h}\t{cq}\t{speed}\t{}\t{}\t{}\t{po}\t{pp}\t{pb}\t{:+.2}\t{:+.2}\t{:+}",
                    c.label,
                    a as usize,
                    b as usize,
                    d as usize,
                    (b - a) / a * 100.0,
                    (d - a) / a * 100.0,
                    pb as i64 - d as i64
                );
            }
        }
    }
}
