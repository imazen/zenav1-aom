//! Differential harness for the reference / GOP management layer
//! (`av1/encoder/encode_strategy.c`) vs the REAL exported C libaom v3.14.1.
//! **Tier 1** — every oracle call below reaches the exported C symbol through
//! `crates/aom-sys-ref/shim/refgop_shim.c`; nothing is compared against a
//! second transcription.
//!
//! | test | C oracle |
//! |---|---|
//! | `refresh_ref_frame_map_matches_c` | `av1_get_refresh_ref_frame_map` |
//! | `configure_buffer_updates_matches_c` | `av1_configure_buffer_updates` (+ static `set_refresh_frame_flags`) |
//! | `calc_refresh_idx_for_intnl_arf_matches_c` | `av1_calc_refresh_idx_for_intnl_arf` (+ statics `get_free_ref_map_index`, `get_refresh_idx`) |
//! | `refresh_frame_flags_matches_c` | `av1_get_refresh_frame_flags` |
//! | `refresh_frame_flags_ext_override_matches_c` | ditto, the external-override arm |
//! | `get_ref_frames_matches_c` | `av1_get_ref_frames` (+ statics `is_in_ref_map`, `add_ref_to_slot`, `set_unmapped_ref`, `compare_map_idx_pair_asc`) |
//! | `get_ref_frames_parallel_skip_matches_c` | ditto, the frame-parallel exclusion arm |
//! | `get_ref_frames_ext_map_matches_c` | ditto, the `use_ext_ref_frame_map` arm |
//!
//! # Why the generator is shaped the way it is
//! `av1_get_ref_frames` is a *permutation* problem: eight buffers, seven named
//! slots, and a cascade of tie-breaks. A generator that only produces tidy
//! pyramids never reaches `set_unmapped_ref`, never produces a
//! `disp_order == cur_frame_disp` buffer (the show-existing BWDREF map), and
//! never produces an all-future or all-past buffer set (the two places C's
//! cursor walks run off their ends). The sweep below deliberately covers:
//! empty slots, duplicate display orders (C's `is_in_ref_map` dedup), buffer
//! sets of every size 0..=8, all-past / all-future / straddling sets, a
//! single pyramid level (which disables the GOLDEN/ALTREF mapping entirely),
//! and both frame-parallel exclusion forms.

use aom_encode::ref_gop::{
    ExtRefreshFrameFlags, FrameType, FrameUpdateType, ParallelSkip, REF_FRAMES, RefFrameMapPair,
    RefbufState, calc_refresh_idx_for_intnl_arf, configure_buffer_updates, get_ref_frames,
    get_ref_frames_from_ext_map, get_refresh_frame_flags, refresh_ref_frame_map,
};
use aom_sys_ref::{
    RefMapPair, ref_calc_refresh_idx_for_intnl_arf, ref_configure_buffer_updates,
    ref_get_ref_frames, ref_get_refresh_frame_flags, ref_get_refresh_ref_frame_map,
};

/// A tiny xorshift so the sweep is reproducible without a dev-dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u32) -> i32 {
        (self.next() % u64::from(n)) as i32
    }
}

fn to_ffi(pairs: &[RefFrameMapPair; REF_FRAMES]) -> [RefMapPair; 8] {
    core::array::from_fn(|i| (pairs[i].pyr_level, pairs[i].disp_order))
}

/// The seven `FRAME_UPDATE_TYPE`s, port-side and as C discriminants.
const UPDATE_TYPES: [(FrameUpdateType, i32); 7] = [
    (FrameUpdateType::Kf, 0),
    (FrameUpdateType::Lf, 1),
    (FrameUpdateType::Gf, 2),
    (FrameUpdateType::Arf, 3),
    (FrameUpdateType::Overlay, 4),
    (FrameUpdateType::IntnlOverlay, 5),
    (FrameUpdateType::IntnlArf, 6),
];

/// What C's `return 1 << refresh_idx` produces when `get_refresh_idx` found no
/// candidate and returned -1: the shift count masks to 31, giving `INT_MIN`.
/// No legitimate refresh mask (`0`, `1 << 0..8`, `SELECT_ALL_BUF_SLOTS`) can
/// collide with it.
const C_UB_SHIFT: i32 = i32::MIN;

const FRAME_TYPES: [(FrameType, i32); 4] = [
    (FrameType::Key, 0),
    (FrameType::Inter, 1),
    (FrameType::IntraOnly, 2),
    (FrameType::Switch, 3),
];

#[test]
fn refresh_ref_frame_map_matches_c() {
    for flags in 0u32..256 {
        let want = ref_get_refresh_ref_frame_map(flags as i32);
        let got = refresh_ref_frame_map(flags).map_or(-1, |i| i as i32);
        assert_eq!(got, want, "refresh_frame_flags = {flags:#010b}");
    }
}

#[test]
fn configure_buffer_updates_matches_c() {
    let mut cells = 0;
    for (ty, c_ty) in UPDATE_TYPES {
        for (state, c_state) in [(RefbufState::Reset, 0), (RefbufState::Update, 1)] {
            for force in [false, true] {
                // `None` plus all eight external-override combinations.
                let exts: Vec<Option<(bool, bool, bool)>> = core::iter::once(None)
                    .chain((0..8).map(|m| Some((m & 1 != 0, m & 2 != 0, m & 4 != 0))))
                    .collect();
                for ext in exts {
                    let want = ref_configure_buffer_updates(c_ty, c_state, force, ext);
                    let (got, new_ty) = configure_buffer_updates(
                        ty,
                        state,
                        force,
                        ext.map(|(g, b, a)| ExtRefreshFrameFlags {
                            golden_frame: g,
                            bwd_ref_frame: b,
                            alt_ref_frame: a,
                            ..Default::default()
                        }),
                    );
                    // C's gf_group entry keeps the incoming update type when
                    // the override rewrites nothing.
                    let got_ty = new_ty.map_or(c_ty, |t| t as i32);
                    assert_eq!(
                        (
                            got.refresh.golden_frame,
                            got.refresh.bwd_ref_frame,
                            got.refresh.alt_ref_frame,
                            got.is_src_frame_alt_ref,
                            got_ty,
                        ),
                        want,
                        "update_type={ty:?} refbuf={state:?} force={force} ext={ext:?}"
                    );
                    cells += 1;
                }
            }
        }
    }
    assert_eq!(cells, 7 * 2 * 2 * 9, "the sweep lost cells");
}

/// Build a reference-map state: `n_occupied` slots filled, display orders drawn
/// from `disp_lo..disp_hi`, pyramid levels from `1..=max_level`.
fn random_pairs(
    rng: &mut Rng,
    n_occupied: usize,
    disp_lo: i32,
    disp_hi: i32,
    max_level: u32,
) -> [RefFrameMapPair; REF_FRAMES] {
    let mut pairs = [RefFrameMapPair::EMPTY; REF_FRAMES];
    let span = (disp_hi - disp_lo).max(1) as u32;
    for p in pairs.iter_mut().take(n_occupied) {
        *p = RefFrameMapPair {
            pyr_level: 1 + rng.below(max_level),
            disp_order: disp_lo + rng.below(span),
        };
    }
    pairs
}

#[test]
fn calc_refresh_idx_for_intnl_arf_matches_c() {
    let mut rng = Rng(0x5eed_1234_9abc_def0);
    let mut cells = 0;
    for n_occupied in 0..=REF_FRAMES {
        for max_level in [1, 2, 4] {
            for one_pass_rt in [false, true] {
                for _ in 0..24 {
                    let pairs = random_pairs(&mut rng, n_occupied, 0, 40, max_level);
                    let cur = rng.below(48);
                    // A skip list of 0..3 display orders, -1 terminated.
                    let mut skip = [-1i32; REF_FRAMES];
                    for k in 0..(rng.below(4) as usize) {
                        skip[k] = rng.below(40);
                    }
                    let want = ref_calc_refresh_idx_for_intnl_arf(
                        &to_ffi(&pairs),
                        &skip,
                        one_pass_rt,
                        cur,
                    );
                    let got = calc_refresh_idx_for_intnl_arf(&pairs, &skip, !one_pass_rt, cur)
                        .map_or(-1, |i| i as i32);
                    assert_eq!(
                        got, want,
                        "n={n_occupied} max_level={max_level} rt={one_pass_rt} \
                         cur={cur} pairs={pairs:?} skip={skip:?}"
                    );
                    cells += 1;
                }
            }
        }
    }
    assert!(cells >= 1000, "sweep too small: {cells}");
}

#[test]
fn refresh_frame_flags_matches_c() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut cells = 0;
    let mut ub_cells = 0;
    for n_occupied in 0..=REF_FRAMES {
        for (ty, c_ty) in UPDATE_TYPES {
            for (ft, c_ft) in FRAME_TYPES {
                for (state, c_state) in [(RefbufState::Reset, 0), (RefbufState::Update, 1)] {
                    for show_existing in [false, true] {
                        let pairs = random_pairs(&mut rng, n_occupied, 0, 40, 3);
                        let cur = rng.below(48);
                        let mut skip = [-1i32; REF_FRAMES];
                        for k in 0..(rng.below(3) as usize) {
                            skip[k] = rng.below(40);
                        }
                        let one_pass_rt = rng.below(2) == 1;
                        let want = ref_get_refresh_frame_flags(
                            &to_ffi(&pairs),
                            c_state,
                            c_ft,
                            show_existing,
                            c_ty,
                            &skip,
                            one_pass_rt,
                            cur,
                            None,
                        );
                        let got = get_refresh_frame_flags(
                            &pairs,
                            state,
                            ft,
                            show_existing,
                            ty,
                            &skip,
                            !one_pass_rt,
                            cur,
                            None,
                        );
                        if want == C_UB_SHIFT {
                            // C reached `return 1 << refresh_idx` with
                            // refresh_idx == -1 (get_refresh_idx found no
                            // candidate). That is UB; on this ISA the shift
                            // count masks to 31 and yields INT_MIN. libaom
                            // asserts the state cannot arise
                            // (`assert(0 && "No valid refresh index found")`,
                            // encode_strategy.c:588), so the generator is
                            // producing a reference map the encoder would not.
                            // The port declines instead of shifting by -1.
                            assert_eq!(
                                got, None,
                                "C hit its no-valid-refresh-index UB but the port \
                                 produced a mask: n={n_occupied} update={ty:?} \
                                 pairs={pairs:?} skip={skip:?}"
                            );
                            ub_cells += 1;
                        } else {
                            assert_eq!(
                                got.map_or(-1, |m| m as i32),
                                want,
                                "n={n_occupied} update={ty:?} frame={ft:?} refbuf={state:?} \
                                 show_existing={show_existing} rt={one_pass_rt} cur={cur} \
                                 pairs={pairs:?} skip={skip:?}"
                            );
                        }
                        cells += 1;
                    }
                }
            }
        }
    }
    assert!(cells >= 500, "sweep too small: {cells}");
    // The UB path must stay a rounding error, not the sweep. If it grew to
    // dominate, this test would be mostly asserting "the port returns None".
    assert!(
        ub_cells * 20 < cells,
        "{ub_cells} of {cells} cells hit C's no-valid-refresh-index UB — the \
         generator has drifted away from states the encoder can produce"
    );
}

#[test]
fn refresh_frame_flags_ext_override_matches_c() {
    let mut rng = Rng(0x0bad_c0de_1111_2222);
    let mut cells = 0;
    for mask in 0u32..32 {
        for (ty, c_ty) in UPDATE_TYPES {
            for (ft, c_ft) in FRAME_TYPES {
                let flags = ExtRefreshFrameFlags {
                    last_frame: mask & 1 != 0,
                    golden_frame: mask & 2 != 0,
                    bwd_ref_frame: mask & 4 != 0,
                    alt_ref_frame: mask & 8 != 0,
                    alt2_ref_frame: mask & 16 != 0,
                };
                // Every named reference points somewhere, including the
                // INVALID_IDX case that the override arm must skip.
                let map: [i32; REF_FRAMES] = core::array::from_fn(|_| {
                    let v = rng.below(9);
                    if v == 8 { -1 } else { v }
                });
                let pairs = random_pairs(&mut rng, 8, 0, 40, 3);
                let skip = [-1i32; REF_FRAMES];
                let cur = rng.below(48);
                let want = ref_get_refresh_frame_flags(
                    &to_ffi(&pairs),
                    1,
                    c_ft,
                    false,
                    c_ty,
                    &skip,
                    false,
                    cur,
                    Some((
                        (
                            flags.last_frame,
                            flags.golden_frame,
                            flags.bwd_ref_frame,
                            flags.alt_ref_frame,
                            flags.alt2_ref_frame,
                        ),
                        map,
                    )),
                );
                let got = get_refresh_frame_flags(
                    &pairs,
                    RefbufState::Update,
                    ft,
                    false,
                    ty,
                    &skip,
                    true,
                    cur,
                    Some((flags, &map)),
                );
                assert_eq!(
                    got.map_or(-1, |m| m as i32),
                    want,
                    "mask={mask:#07b} update={ty:?} frame={ft:?} map={map:?}"
                );
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 32 * 7 * 4);
}

#[test]
fn get_ref_frames_matches_c() {
    let mut rng = Rng(0xfeed_face_dead_beef);
    let mut cells = 0;
    // The four display-order regimes: all past, all future, straddling, and
    // one buffer exactly at cur_frame_disp (the show-existing BWDREF map).
    for regime in 0..4 {
        for n_occupied in 0..=REF_FRAMES {
            for max_level in [1, 2, 3, 6] {
                for _ in 0..40 {
                    let cur = 24;
                    let mut pairs = match regime {
                        0 => random_pairs(&mut rng, n_occupied, 0, 24, max_level),
                        1 => random_pairs(&mut rng, n_occupied, 25, 60, max_level),
                        _ => random_pairs(&mut rng, n_occupied, 0, 60, max_level),
                    };
                    if regime == 3 && n_occupied > 0 {
                        pairs[rng.below(n_occupied as u32) as usize].disp_order = cur;
                    }
                    let want = ref_get_ref_frames(&to_ffi(&pairs), cur, false, 0, 0, None);
                    let got = get_ref_frames(&pairs, cur, None);
                    assert_eq!(
                        got, want,
                        "regime={regime} n={n_occupied} max_level={max_level} \
                         cur={cur} pairs={pairs:?}"
                    );
                    cells += 1;
                }
            }
        }
    }
    assert!(cells >= 5000, "sweep too small: {cells}");
}

#[test]
fn get_ref_frames_parallel_skip_matches_c() {
    let mut rng = Rng(0xa5a5_5a5a_1234_4321);
    let mut cells = 0;
    for kind in [1, 2] {
        for n_occupied in 1..=REF_FRAMES {
            for _ in 0..64 {
                let cur = 24;
                let pairs = random_pairs(&mut rng, n_occupied, 0, 60, 3);
                // Aim the exclusion at a buffer that actually exists half the
                // time, and at nothing the other half.
                let skip_value = if rng.below(2) == 0 {
                    match kind {
                        1 => rng.below(n_occupied as u32),
                        _ => pairs[rng.below(n_occupied as u32) as usize].disp_order,
                    }
                } else {
                    999
                };
                let want = ref_get_ref_frames(&to_ffi(&pairs), cur, false, kind, skip_value, None);
                let skip = if kind == 1 {
                    ParallelSkip::MapIdx(skip_value)
                } else {
                    ParallelSkip::DispOrder(skip_value)
                };
                let got = get_ref_frames(&pairs, cur, Some(skip));
                assert_eq!(
                    got, want,
                    "kind={kind} n={n_occupied} skip={skip_value} pairs={pairs:?}"
                );
                cells += 1;
            }
        }
    }
    assert!(cells >= 500, "sweep too small: {cells}");
}

#[test]
fn get_ref_frames_ext_map_matches_c() {
    let mut rng = Rng(0x3141_5926_5358_9793);
    for _ in 0..400 {
        // C indexes ref_frame_list[LAST_FRAME..REF_FRAMES], i.e. entries 1..=7;
        // entry 0 is never read, so it is filled with a value that would be
        // visible if the port were off by one.
        let mut list = [0i32; 8];
        list[0] = 7;
        for e in list.iter_mut().skip(1) {
            let v = rng.below(9);
            *e = if v == 8 { -1 } else { v };
        }
        let pairs = random_pairs(&mut rng, 8, 0, 40, 3);
        let want = ref_get_ref_frames(&to_ffi(&pairs), 24, false, 0, 0, Some(list));
        let named: [i32; 7] = core::array::from_fn(|i| list[i + 1]);
        let got = get_ref_frames_from_ext_map(&named);
        assert_eq!(got, want, "list={list:?}");
    }
}
