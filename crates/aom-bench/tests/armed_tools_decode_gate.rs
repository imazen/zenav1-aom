//! **Decode-side conformance gate for ARMED encoder tools** (KB-29).
//!
//! # Why this file exists
//!
//! Every other encoder gate in this repo proves *byte-identity against a real
//! aomenc stream*. That proves conformance **only where a reference exists and
//! is asserted equal**. A `ToggleKnobs` arm that aomenc cannot be driven into
//! from `ToggleKnobs::c_ctrls` — or one whose port stream is only compared for
//! SIZE / RD-closeness rather than asserted byte-equal — is a configuration the
//! port can *produce* and that **nothing ever decodes**. The port could emit
//! arbitrary garbage there and the whole suite would stay green.
//!
//! That is exactly how KB-29 shipped: `port_encode` runs `ToggleKnobs::default()`
//! (IntraBC off), `rd_close_intrabc` compares sizes rather than decoding, and the
//! IntraBC-armed encode emitted a stream both `aomdec` and `dav1d` rejected
//! (*"Invalid intrabc dv"*). Three separate defects hid behind that hole (two
//! encoder-side, one decoder-side); see CLAUDE.md KB-29.
//!
//! # What this gate asserts
//!
//! For every armed configuration:
//!
//! 1. the port's frame OBU, reassembled into the bootstrap's OBU stream, is
//!    **accepted by the REAL C decoder** (`aom_codec_av1_dx`) — the authority;
//! 2. the port's own decoder accepts it too, and
//! 3. **both decoders produce identical pixels** on every plane.
//!
//! Optionally (4) `dav1d` accepts it, when `AOM_DAV1D_BIN` names the binary.
//! That leg is caller-controlled — the decision is in the justfile
//! (`just gate-armed-decode`), never inside the test body. `dav1d` is worth
//! having because it caught KB-29 independently of libaom.
//!
//! # Coverage is DERIVED, not asserted by name
//!
//! [`single_knob_arms`] lists every one-knob-off-default `ToggleKnobs` setting.
//! A knob whose `c_ctrls()` is EMPTY cannot be handed to `c_encode_ctrls`, so the
//! generic byte-match harness has no reference for it — it is **unguarded by
//! byte-identity** by construction. `armed_arm_coverage_is_complete` recomputes
//! that set from `ToggleKnobs` itself and fails if any member lacks a decode arm
//! here. Add a knob to `ToggleKnobs` without a C control and this test tells you
//! to gate it, rather than silently leaving a second KB-29 shaped hole.

use aom_bench::{EncodeCell, ToggleKnobs};

const OBU_FRAME: u8 = 6;

/// `AV1E_SET_DELTAQ_MODE` (`aomcx.h`) — mirrored the same way
/// `deltaq_mode2_e2e.rs` does.
const AV1E_SET_DELTAQ_MODE: i32 = 107;

/// The screen-content conformance vector — the only corpus content that makes
/// `estimate_screen_content` fire, so the only content on which the palette and
/// IntraBC searches are non-vacuous.
const SCREEN_VEC: &str = "av1-1-b8-16-intra_only-intrabc-extreme-dv";

/// `(label, w, h, off_x, off_y, cq, speed)`. Small crops keep the gate cheap:
/// the IntraBC DV search is the expensive part (KB-29 measured 45x encode time
/// at 1 MP), so this exercises the path at ~200² instead of sweeping.
/// The speeds bracket the two encoder roots KB-29 closed — the chroma chunk
/// extent (visible at every speed) and the stale txfm-partition context (only
/// reachable once the RD search picks a BLOCK_4X4 IntraBC coeff leaf, which it
/// does at speed 2 cq12 and not at speed 0).
const CELLS: &[(&str, usize, usize, usize, usize, i32, i32)] = &[
    ("scc196_cq48_s0", 196, 196, 480, 180, 48, 0),
    ("scc196_cq12_s2", 196, 196, 480, 180, 12, 2),
    ("scc196_cq32_s6", 196, 196, 480, 180, 32, 6),
    // The decoder-side root: this cell is where the leaf-vs-raster var-tx walk
    // gate bites (C accepts, the port decoder rejects, if the walk is selected
    // on leaf-size uniformity instead of "the quadtree was read").
    ("scc196_cq12_s6", 196, 196, 480, 180, 12, 6),
];

// ---------------------------------------------------------------------------
// OBU reassembly (the port emits only the frame OBU; the bootstrap supplies the
// temporal-delimiter + sequence header — see `EncodeCell::port_encode_with`).
// ---------------------------------------------------------------------------

fn walk(mut s: &[u8]) -> Vec<(u8, Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    while !s.is_empty() {
        let mut hdr = vec![s[0]];
        let ty = (s[0] >> 3) & 0xf;
        let ext = (s[0] >> 2) & 1;
        assert_eq!((s[0] >> 1) & 1, 1, "obu_has_size_field must be set");
        let mut i = 1;
        if ext == 1 {
            hdr.push(s[1]);
            i = 2;
        }
        let (mut size, mut shift) = (0usize, 0u32);
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

// ---------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------

/// Which C bootstrap an arm needs. The port never authors a sequence header
/// (CLAUDE.md Gate-2 scope note), so the arm's frame-level tool bits
/// (`allow_screen_content_tools`, `allow_intrabc`) have to come from a matched
/// C encode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Boot {
    /// `c_encode_screen(enable_palette, enable_intrabc)`.
    Screen(bool, bool),
    /// `c_encode_qm(qm_min, qm_max)`.
    Qm(i32, i32),
    /// `c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, n)])`.
    DeltaQ(i32),
    /// `c_encode()` — the plain KEY reference.
    Plain,
}

struct Arm {
    /// Must match the `single_knob_arms` name of the knob it covers, so the
    /// coverage check can pair them.
    knob: &'static str,
    knobs: ToggleKnobs,
    boot: Boot,
}

fn arms() -> Vec<Arm> {
    let d = ToggleKnobs::default();
    vec![
        Arm {
            knob: "enable_intrabc",
            knobs: ToggleKnobs {
                enable_intrabc: true,
                ..d
            },
            boot: Boot::Screen(false, true),
        },
        Arm {
            knob: "enable_palette",
            knobs: ToggleKnobs {
                enable_palette: true,
                ..d
            },
            boot: Boot::Screen(true, false),
        },
        // Both screen tools at once — the combination the xbench run measured
        // and the one a real screen-content encode would use.
        Arm {
            knob: "enable_palette+enable_intrabc",
            knobs: ToggleKnobs {
                enable_palette: true,
                enable_intrabc: true,
                ..d
            },
            boot: Boot::Screen(true, true),
        },
        Arm {
            knob: "qm",
            knobs: ToggleKnobs {
                qm: Some((5, 9)),
                ..d
            },
            boot: Boot::Qm(5, 9),
        },
        Arm {
            knob: "delta_lf_mode",
            knobs: ToggleKnobs {
                delta_lf_mode: true,
                ..d
            },
            boot: Boot::Plain,
        },
        // `--deltaq-mode=2/3` DOES have an aomenc control
        // (`AV1E_SET_DELTAQ_MODE`), but `c_ctrls` cannot emit it — the port
        // knob is a bool per mode while the control is one tri-state int — so
        // the byte-match harness reaches it only through the two bespoke e2e
        // tests. Gate the decode side here as well.
        Arm {
            knob: "deltaq_mode2",
            knobs: ToggleKnobs {
                deltaq_mode2: true,
                ..d
            },
            boot: Boot::DeltaQ(2),
        },
        Arm {
            knob: "deltaq_mode3",
            knobs: ToggleKnobs {
                deltaq_mode3: true,
                ..d
            },
            boot: Boot::DeltaQ(3),
        },
        Arm {
            knob: "disable_tx_stats_prune",
            knobs: ToggleKnobs {
                disable_tx_stats_prune: true,
                ..d
            },
            boot: Boot::Plain,
        },
    ]
}

/// Every single-knob-off-default `ToggleKnobs` value, by field name. Kept
/// exhaustive by hand because `ToggleKnobs` has no reflection; the destructure
/// in `single_knob_arms_are_exhaustive` makes omitting a field a COMPILE error.
fn single_knob_arms() -> Vec<(&'static str, ToggleKnobs)> {
    let d = ToggleKnobs::default();
    vec![
        ("enable_rect_partitions", ToggleKnobs { enable_rect_partitions: !d.enable_rect_partitions, ..d }),
        ("enable_ab_partitions", ToggleKnobs { enable_ab_partitions: !d.enable_ab_partitions, ..d }),
        ("enable_1to4_partitions", ToggleKnobs { enable_1to4_partitions: !d.enable_1to4_partitions, ..d }),
        ("min_partition_size_px", ToggleKnobs { min_partition_size_px: 16, ..d }),
        ("max_partition_size_px", ToggleKnobs { max_partition_size_px: 32, ..d }),
        ("enable_intra_edge_filter", ToggleKnobs { enable_intra_edge_filter: !d.enable_intra_edge_filter, ..d }),
        ("enable_filter_intra", ToggleKnobs { enable_filter_intra: !d.enable_filter_intra, ..d }),
        ("enable_smooth_intra", ToggleKnobs { enable_smooth_intra: !d.enable_smooth_intra, ..d }),
        ("enable_paeth_intra", ToggleKnobs { enable_paeth_intra: !d.enable_paeth_intra, ..d }),
        ("enable_cfl_intra", ToggleKnobs { enable_cfl_intra: !d.enable_cfl_intra, ..d }),
        ("enable_directional_intra", ToggleKnobs { enable_directional_intra: !d.enable_directional_intra, ..d }),
        ("enable_diagonal_intra", ToggleKnobs { enable_diagonal_intra: !d.enable_diagonal_intra, ..d }),
        ("enable_angle_delta", ToggleKnobs { enable_angle_delta: !d.enable_angle_delta, ..d }),
        ("enable_tx64", ToggleKnobs { enable_tx64: !d.enable_tx64, ..d }),
        ("enable_rect_tx", ToggleKnobs { enable_rect_tx: !d.enable_rect_tx, ..d }),
        ("enable_flip_idtx", ToggleKnobs { enable_flip_idtx: !d.enable_flip_idtx, ..d }),
        ("use_intra_dct_only", ToggleKnobs { use_intra_dct_only: !d.use_intra_dct_only, ..d }),
        ("use_intra_default_tx_only", ToggleKnobs { use_intra_default_tx_only: !d.use_intra_default_tx_only, ..d }),
        ("reduced_tx_type_set", ToggleKnobs { reduced_tx_type_set: !d.reduced_tx_type_set, ..d }),
        ("enable_tx_size_search", ToggleKnobs { enable_tx_size_search: !d.enable_tx_size_search, ..d }),
        ("cdf_update_mode", ToggleKnobs { cdf_update_mode: 0, ..d }),
        ("disable_trellis_quant", ToggleKnobs { disable_trellis_quant: 1, ..d }),
        ("coeff_cost_upd_freq", ToggleKnobs { coeff_cost_upd_freq: 1, ..d }),
        ("mode_cost_upd_freq", ToggleKnobs { mode_cost_upd_freq: 1, ..d }),
        ("enable_palette", ToggleKnobs { enable_palette: !d.enable_palette, ..d }),
        ("enable_intrabc", ToggleKnobs { enable_intrabc: !d.enable_intrabc, ..d }),
        ("disable_tx_stats_prune", ToggleKnobs { disable_tx_stats_prune: !d.disable_tx_stats_prune, ..d }),
        ("delta_lf_mode", ToggleKnobs { delta_lf_mode: !d.delta_lf_mode, ..d }),
        ("qm", ToggleKnobs { qm: Some((5, 9)), ..d }),
        ("deltaq_mode2", ToggleKnobs { deltaq_mode2: !d.deltaq_mode2, ..d }),
        ("deltaq_mode3", ToggleKnobs { deltaq_mode3: !d.deltaq_mode3, ..d }),
    ]
}

/// A knob is UNREFERENCED when flipping it emits no `aome_enc_control_id`, i.e.
/// `EncodeCell::c_encode_ctrls` cannot build a matched aomenc stream for it and
/// the generic byte-match harness therefore has no reference to compare against.
fn unreferenced_knobs() -> Vec<&'static str> {
    single_knob_arms()
        .into_iter()
        .filter(|(_, k)| k.c_ctrls().is_empty())
        .map(|(n, _)| n)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Compile-time exhaustiveness: destructuring `ToggleKnobs` with no `..` means
/// a NEW field is a hard error here until it is added to `single_knob_arms`.
#[test]
fn single_knob_arms_are_exhaustive() {
    let ToggleKnobs {
        enable_rect_partitions: _,
        enable_ab_partitions: _,
        enable_1to4_partitions: _,
        min_partition_size_px: _,
        max_partition_size_px: _,
        enable_intra_edge_filter: _,
        enable_filter_intra: _,
        enable_smooth_intra: _,
        enable_paeth_intra: _,
        enable_cfl_intra: _,
        enable_directional_intra: _,
        enable_diagonal_intra: _,
        enable_angle_delta: _,
        enable_tx64: _,
        enable_rect_tx: _,
        enable_flip_idtx: _,
        use_intra_dct_only: _,
        use_intra_default_tx_only: _,
        reduced_tx_type_set: _,
        enable_tx_size_search: _,
        cdf_update_mode: _,
        enable_palette: _,
        disable_trellis_quant: _,
        coeff_cost_upd_freq: _,
        mode_cost_upd_freq: _,
        disable_tx_stats_prune: _,
        delta_lf_mode: _,
        enable_intrabc: _,
        qm: _,
        deltaq_mode2: _,
        deltaq_mode3: _,
    } = ToggleKnobs::default();
    let names = single_knob_arms();
    assert_eq!(
        names.len(),
        31,
        "single_knob_arms must list every ToggleKnobs field exactly once — the \
         destructure above is the compile-time half of that contract, this count \
         is the runtime half"
    );
}

/// The coverage claim, DERIVED. Every knob with no aomenc control — i.e. every
/// knob the byte-match harness structurally cannot guard — must have a decode
/// arm here.
#[test]
fn armed_arm_coverage_is_complete() {
    let unref = unreferenced_knobs();
    eprintln!("=== knobs with NO aomenc control (unguarded by byte-identity) ===");
    for n in &unref {
        eprintln!("  {n}");
    }
    let armed: Vec<&str> = arms().iter().map(|a| a.knob).collect();
    let missing: Vec<&str> = unref
        .iter()
        .copied()
        .filter(|n| !armed.contains(n))
        .collect();
    assert!(
        missing.is_empty(),
        "these ToggleKnobs arms have NO aomenc reference (so no byte-match gate \
         can ever see them) and NO decode arm in this file: {missing:?}. Add an \
         `Arm` for each — an armed configuration nothing decodes is exactly the \
         KB-29 hole."
    );
    // Anti-vacuity: the set must be non-empty, or the filter is broken and this
    // test would pass while proving nothing.
    assert!(
        !unref.is_empty(),
        "no unreferenced knob found — `c_ctrls()` filtering is broken, this gate \
         would be vacuous"
    );
}

/// The gate proper.
#[test]
fn armed_tools_round_trip_through_the_c_decoder() {
    let dav1d = std::env::var("AOM_DAV1D_BIN").ok();
    eprintln!(
        "=== armed-tool decode gate (dav1d leg: {}) ===",
        dav1d.as_deref().unwrap_or("OFF — set AOM_DAV1D_BIN to enable")
    );
    let mut cells_run = 0usize;
    for &(label, w, h, ox, oy, cq, speed) in CELLS {
        let cell = EncodeCell::real_content(label, SCREEN_VEC, Some((w, h, ox, oy)), cq, speed);
        for arm in arms() {
            let bootstrap = match arm.boot {
                Boot::Screen(p, i) => cell.c_encode_screen(p, i),
                Boot::Qm(lo, hi) => cell.c_encode_qm(lo, hi),
                Boot::DeltaQ(n) => {
                    cell.c_encode_ctrls(&[(AV1E_SET_DELTAQ_MODE, n)])
                }
                Boot::Plain => cell.c_encode(),
            };
            assert!(
                !bootstrap.is_empty(),
                "{label}/{}: C bootstrap encode failed",
                arm.knob
            );
            let frame = cell.port_encode_with(&bootstrap, &arm.knobs);
            let stream = reassemble(&bootstrap, &frame);

            // (1) THE AUTHORITY: the real `aom_codec_av1_dx`. `ref_decode_av1_kf`
            // asserts on a non-zero shim rc, so a rejection surfaces as a panic
            // we catch and re-raise with the arm named.
            let c_dec = std::panic::catch_unwind(|| {
                aom_sys_ref::ref_decode_av1_kf(&stream, cell.w, cell.h)
            });
            let c_dec = match c_dec {
                Ok(d) => d,
                Err(_) => panic!(
                    "{label}/{}: the REAL C decoder REJECTED the port's armed \
                     stream ({} B frame payload). The port emitted a \
                     non-conformant bitstream on a configuration no byte-match \
                     gate covers — this is the KB-29 class.",
                    arm.knob,
                    frame.len()
                ),
            };

            // (2) the port's own decoder must accept it too.
            let p_dec = aom_decode::frame::decode_frame_obus(&stream).unwrap_or_else(|e| {
                panic!(
                    "{label}/{}: the C decoder accepted the port's armed stream \
                     but the PORT decoder rejected it: {e}. One of the two is \
                     non-conformant; the C decoder is the authority, so this is \
                     a decoder-side defect.",
                    arm.knob
                )
            });

            // (3) identical pixels on every plane.
            assert_eq!(
                (p_dec.width, p_dec.height),
                (cell.w, cell.h),
                "{label}/{}: port decode geometry",
                arm.knob
            );
            assert!(
                p_dec.y == c_dec.y,
                "{label}/{}: luma differs between the C decoder and the port \
                 decoder ({} of {} samples)",
                arm.knob,
                p_dec
                    .y
                    .iter()
                    .zip(&c_dec.y)
                    .filter(|(a, b)| a != b)
                    .count(),
                p_dec.y.len()
            );
            let dbg = |a: &[u16], b: &[u16], w: usize| -> String {
                let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
                let first = a
                    .iter()
                    .zip(b)
                    .position(|(x, y)| x != y)
                    .map(|i| format!("first at ({}, {}) {} vs {}", i / w, i % w, a[i], b[i]))
                    .unwrap_or_default();
                format!("{n} of {} differ; {first}", a.len())
            };
            assert!(
                p_dec.u == c_dec.u && p_dec.v == c_dec.v,
                "{label}/{}: chroma differs between the C decoder and the port \
                 decoder — U: {}  V: {}",
                arm.knob,
                dbg(&p_dec.u, &c_dec.u, p_dec.width_uv),
                dbg(&p_dec.v, &c_dec.v, p_dec.width_uv)
            );

            // (4) optional dav1d leg — an INDEPENDENT implementation. It caught
            // KB-29 on its own, so it is worth running whenever the caller wires
            // it in.
            if let Some(bin) = &dav1d {
                let dir = std::env::temp_dir().join(format!(
                    "aom_armed_gate_{}_{}",
                    std::process::id(),
                    label
                ));
                std::fs::create_dir_all(&dir).expect("scratch dir");
                let path = dir.join(format!("{}.obu", arm.knob.replace('+', "_")));
                std::fs::write(&path, &stream).expect("write obu");
                let out = std::process::Command::new(bin)
                    .args(["-i".as_ref(), path.as_os_str(), "-o".as_ref(), "/dev/null".as_ref()])
                    .output()
                    .unwrap_or_else(|e| panic!("running {bin}: {e}"));
                assert!(
                    out.status.success(),
                    "{label}/{}: dav1d REJECTED the port's armed stream: {}",
                    arm.knob,
                    String::from_utf8_lossy(&out.stderr)
                );
                let _ = std::fs::remove_file(&path);
            }

            eprintln!("  {label}/{:<32} {} B  OK", arm.knob, frame.len());
            cells_run += 1;
        }
    }
    assert!(
        cells_run >= CELLS.len() * arms().len(),
        "the gate ran {cells_run} cells — fewer than the arm table declares, so \
         some arm silently did not run"
    );
}
