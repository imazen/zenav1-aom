//! `svt_interop_probe` — decode a set of AV1 streams with BOTH the REAL C
//! libaom decoder (`aom_codec_av1_dx`, the authority) and the port's decoder,
//! and report per-stream accept/reject plus pixel identity.
//!
//! Written for GitHub issue #5: "37 of 100 real SVT-AV1 v4.2.0 screen-content
//! encodes are rejected by aom-decode ... while real libaom decodes every one of
//! them clean". Which streams libaom ITSELF rejects is a separate fact from
//! which ones the port rejects, so this prints both rather than assuming: a
//! stream both reject is a bad stream, not an interop bug.
//!
//! Usage:
//!   svt_interop_probe <manifest.tsv>
//!
//! Manifest columns (tab-separated, `#` comments and blank lines skipped):
//!   `name  w  h  path`
//!
//! Output (stdout, TSV): `name  bytes  c_dec  port_dec  pixels  detail`
//! where `c_dec`/`port_dec` are OK/REJECT and `pixels` is EQ/DIFF/-.

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 2 {
        eprintln!("usage: svt_interop_probe <manifest.tsv>   # name\\tw\\th\\tpath");
        std::process::exit(2);
    }
    let manifest = std::fs::read_to_string(&a[1]).unwrap_or_else(|e| panic!("read {}: {e}", a[1]));

    // `ref_decode_av1_kf` asserts on a non-zero shim rc, so a C rejection
    // arrives as a panic. Silence the hook so the TSV stays readable.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    println!("name\tbytes\tc_dec\tport_dec\tpixels\ttiles\tdetail");
    let (mut n, mut c_rej, mut p_rej, mut pix_diff) = (0usize, 0usize, 0usize, 0usize);
    for line in manifest.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 4, "manifest row must be name\\tw\\th\\tpath: {line}");
        let (name, path) = (f[0], f[3]);
        let w: usize = f[1].parse().expect("w");
        let h: usize = f[2].parse().expect("h");
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));

        let c = std::panic::catch_unwind(|| aom_sys_ref::ref_decode_av1_kf(&data, w, h)).ok();
        let p = aom_decode::frame::decode_frame_obus(&data);

        let c_ok = c.is_some();
        let p_ok = p.is_ok();
        let mut detail = String::new();
        let mut pixels = "-";
        // The tile grid as the port parsed it. `av1_is_dv_valid` reads
        // `tile->mi_col_start/end` directly, so this is a coverage fact worth
        // printing, not decoration: a corpus that is 1x1 everywhere has not
        // exercised the multi-tile DV bounds at all.
        let tiles = match &p {
            Ok(d) => format!("{}x{}", d.tile_cols, d.tile_rows),
            Err(_) => "-".to_string(),
        };
        if let Err(e) = &p {
            detail = format!("port: {e}");
        }
        if let (Some(c), Ok(p)) = (&c, &p) {
            let ydiff = p.y.iter().zip(&c.y).filter(|(x, y)| x != y).count();
            let udiff = p.u.iter().zip(&c.u).filter(|(x, y)| x != y).count();
            let vdiff = p.v.iter().zip(&c.v).filter(|(x, y)| x != y).count();
            if p.y.len() == c.y.len() && ydiff + udiff + vdiff == 0 {
                pixels = "EQ";
            } else {
                pixels = "DIFF";
                pix_diff += 1;
                detail = format!("y{ydiff} u{udiff} v{vdiff}");
            }
        }
        n += 1;
        if !c_ok {
            c_rej += 1;
        }
        if !p_ok {
            p_rej += 1;
        }
        println!(
            "{name}\t{}\t{}\t{}\t{pixels}\t{tiles}\t{detail}",
            data.len(),
            if c_ok { "OK" } else { "REJECT" },
            if p_ok { "OK" } else { "REJECT" },
        );
    }
    std::panic::set_hook(prev);
    eprintln!(
        "=== {n} streams: C rejected {c_rej}, port rejected {p_rej}, pixel-diff {pix_diff} ==="
    );
}
