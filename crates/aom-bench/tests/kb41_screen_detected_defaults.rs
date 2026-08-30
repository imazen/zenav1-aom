//! **KB-41 localizer — real-content cells that the datagen arm REFUSED at
//! `--cpu-used` 6/8 (zensim's byte-verified `aom-rs` sweep, zenav1-aom#14).**
//!
//! Hypothesis under test (playbook §10: drive to the decision, never infer
//! the mechanism from the delta's shape): every refused cell is a frame that
//! real `aomenc`'s ALLINTRA defaults SCREEN-DETECT (`allow_screen_content_tools`
//! = 1 in the oracle's own frame header), and the divergence is the port being
//! driven with `ToggleKnobs::default()` — palette + IntraBC OFF — against an
//! oracle whose defaults have both ON. If so, `port_encode_with` with the two
//! screen knobs on must reproduce the oracle's payload on those cells, and the
//! cells whose header bit is 0 must already match with the default knobs.
//!
//! Content: the exact u8 4:2:0 planes the datagen arm fed both encoders
//! (`ZEN_AOMRS_DUMP_PLANES`, zenmetrics `sweep/encode.rs::encode_avif_aom_rs`),
//! read from `$ZENAV1_PLANES_DIR/<w>x<h>_cq<cq>_s<speed>.{y,u,v,json}`. The
//! renditions are NOT in-repo (screenshots + product photos from the imazen-26
//! train set), so this runs on demand; the pinned in-repo gate follows once the
//! mechanism is known.
use aom_bench::rd_close::{port_decode_tu, splice_frame_obu};
use aom_bench::{EncodeCell, ToggleKnobs, stream_allows_screen_content_tools};
use aom_sys_ref as c;
use std::path::Path;

/// First differing byte between two payloads, or None when equal.
fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y).or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

/// Decode both temporal units with the port's decoder and report the first
/// divergent reconstructed sample as (plane, x, y, sb64 col/row), or None.
fn first_recon_diff(oracle_tu: &[u8], port_tu: &[u8]) -> Option<String> {
    let a = port_decode_tu("oracle", oracle_tu);
    let b = port_decode_tu("port", port_tu);
    for (plane, (pa, pb, w)) in [(&a.y, &b.y, a.width), (&a.u, &b.u, a.width_uv), (&a.v, &b.v, a.width_uv)]
        .into_iter()
        .enumerate()
    {
        if let Some(i) = pa.iter().zip(pb).position(|(x, y)| x != y) {
            let (x, y) = (i % w, i / w);
            let sh = if plane == 0 { 0 } else { 1 };
            return Some(format!(
                "plane {plane} @({x},{y}) sb64 ({},{})",
                (x << sh) / 64,
                (y << sh) / 64
            ));
        }
    }
    None
}

fn load(dir: &Path, stem: &str) -> EncodeCell {
    let json = std::fs::read_to_string(dir.join(format!("{stem}.json"))).unwrap();
    let num = |k: &str| -> usize {
        let pat = format!("\"{k}\":");
        let i = json.find(&pat).unwrap() + pat.len();
        json[i..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap()
    };
    let (w, h, cq, speed) = (num("w"), num("h"), num("cq_level"), num("speed"));
    let rd = |ext: &str| -> Vec<u16> {
        std::fs::read(dir.join(format!("{stem}.{ext}")))
            .unwrap()
            .into_iter()
            .map(u16::from)
            .collect()
    };
    EncodeCell {
        label: stem.to_string(),
        w,
        h,
        mono: false,
        ss_x: 1,
        ss_y: 1,
        usage: 2,
        cq_level: cq as i32,
        speed: speed as i32,
        bd: 8,
        y: rd("y"),
        u: rd("u"),
        v: rd("v"),
    }
}

#[test]
#[ignore = "on-demand: needs ZENAV1_PLANES_DIR (planes dumped by the zensim datagen arm)"]
fn kb41_screen_detected_cells_match_with_screen_tools_on() {
    let dir = std::env::var("ZENAV1_PLANES_DIR").expect("ZENAV1_PLANES_DIR");
    let dir = Path::new(&dir);
    let mut stems: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.strip_suffix(".json").map(str::to_string)
        })
        .collect();
    stems.sort();
    if let Ok(f) = std::env::var("ZENAV1_PLANES_FILTER") {
        stems.retain(|s| s.contains(&f));
    }
    c::ref_init();
    let screen = ToggleKnobs { enable_palette: true, enable_intrabc: true, ..ToggleKnobs::default() };
    let mut rows = Vec::new();
    let mut unexplained = Vec::new();
    for stem in &stems {
        let cell = load(dir, stem);
        let oracle = cell.c_encode_defaults();
        assert!(!oracle.is_empty(), "{stem}: oracle encode failed");
        let sct = stream_allows_screen_content_tools(&oracle);
        let real = EncodeCell::frame_obu_payload(&oracle);
        let port_default = cell.port_encode(&oracle);
        let port_screen = cell.port_encode_with(&oracle, &screen);
        let d_ok = port_default == real;
        let s_ok = port_screen == real;
        // Where does the screen-knobs encode first differ — in the bytes and in
        // the reconstruction (decode-both, playbook §10)?
        let detail = if s_ok {
            String::new()
        } else {
            let fd = first_diff(&port_screen, &real).unwrap();
            let recon = if std::env::var_os("ZENAV1_DECODE_BOTH").is_some() {
                let port_tu = splice_frame_obu(&oracle, &port_screen);
                first_recon_diff(&oracle, &port_tu).unwrap_or_else(|| "recon IDENTICAL".into())
            } else {
                "(set ZENAV1_DECODE_BOTH=1 for recon diff)".into()
            };
            format!("  first_byte_diff={fd}  {recon}")
        };
        rows.push(format!(
            "{stem:>22}  sct={}  default={}({:+})  screen={}({:+})  oracle={}B{detail}",
            sct as u8,
            if d_ok { "OK " } else { "DIV" },
            port_default.len() as i64 - real.len() as i64,
            if s_ok { "OK " } else { "DIV" },
            port_screen.len() as i64 - real.len() as i64,
            real.len()
        ));
        // The hypothesis' two predictions: sct=0 cells match with default knobs;
        // sct=1 cells match once the screen tools are on.
        let explained = if sct { s_ok } else { d_ok };
        if !explained {
            unexplained.push(stem.clone());
        }
    }
    for r in &rows {
        eprintln!("{r}");
    }
    eprintln!("unexplained by the screen-tools hypothesis: {unexplained:?}");
    // Report-only on the on-demand path: the assertion is the printed table.
}
