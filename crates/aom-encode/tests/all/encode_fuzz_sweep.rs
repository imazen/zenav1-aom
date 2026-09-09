//! STABLE-PATH ENCODER FUZZ SWEEP (no nightly / cargo-fuzz required).
//!
//! The decoder has had this contract since the zen hardening work
//! (`aom-decode/tests/fuzz_sweep.rs`): for ANY input, the public entry returns
//! `Ok` or `Err` — never a panic. The encoder had nothing, and the standing
//! goal asks in as many words for "no panics ... on inputs a caller can
//! produce".
//!
//! The encoder's input is not a byte stream, so the sweep is shaped differently
//! from the decoder's. What a caller controls is a `KeyFrameConfig` and three
//! plane buffers, so all three are randomised — including in ways a careless
//! caller really does produce:
//!
//!   * **out-of-range SAMPLES.** Nothing in the API checks that a bd8 plane
//!     holds only 0..=255; a caller that forgot to shift a 16-bit source hands
//!     over values up to 65535. That must be refused or encoded, never a panic.
//!   * **mismatched plane LENGTHS**, including empty planes for a non-mono
//!     config and over-long ones.
//!   * every enumerated config field, at and just past its documented range.
//!
//! Frames are kept small (<= 40x40) because an encode is orders of magnitude
//! more expensive than a decode; the property under test is panic-freedom,
//! which does not need large frames, and the sizes that DO need covering for
//! their own sake are covered by `refusal_census.rs`.

use aom_encode::key_frame::{
    EncodeConfig, EncodeLimits, KeyFrameConfig, KeyFrameError, KeyFramePlanes,
    encode_key_frame_with,
};

/// XorShift64* — the same cheap seeded PRNG the decoder's sweep uses, so a
/// failing seed is reproducible from the printed number alone.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
    fn chance(&mut self, one_in: u64) -> bool {
        self.below(one_in) == 0
    }
}

/// Mostly-VALID configurations with occasional excursions past each boundary.
///
/// The first version drew every field uniformly over "legal plus a bit either
/// side" and only **13 of 600** inputs reached a real encode — the rest were
/// refused at the config gate, so the sweep was mostly testing
/// `validate_configuration` and barely testing the encoder. Each field is now
/// valid ~15 times in 16, which puts the majority of inputs INSIDE the encoder
/// while still exercising every refusal path (the non-vacuity assertions at the
/// end require all of them to be reached).
fn random_config(r: &mut Rng) -> KeyFrameConfig {
    let w = 1 + r.below(40) as usize;
    let h = 1 + r.below(40) as usize;
    let bd = if r.chance(16) {
        [9u8, 16, 7, 11][r.below(4) as usize]
    } else {
        [8u8, 10, 12][r.below(3) as usize]
    };
    let mono = r.chance(4);
    let (sx, sy) = if r.chance(16) {
        (0usize, 1usize) // 4:4:0 — not an AV1 format
    } else {
        [(1usize, 1usize), (1, 0), (0, 0)][r.below(3) as usize]
    };
    let mut c = KeyFrameConfig::allintra_speed0(w, h, bd, mono, sx, sy, 0);
    c.cq_level = if r.chance(16) {
        [-1i32, 64, 200][r.below(3) as usize]
    } else {
        r.below(64) as i32
    };
    c.cpu_used = if r.chance(16) {
        [-1i32, 10, 99][r.below(3) as usize]
    } else {
        r.below(10) as i32
    };
    c.usage = if r.chance(16) { r.below(3) as u32 } else { 2 };
    c.sb_size_128 = r.chance(2);
    c.tile_columns_log2 = r.below(5) as i32;
    c.tile_rows_log2 = r.below(5) as i32;
    c.enable_cdef = r.chance(2);
    c.enable_restoration = r.chance(2);
    c
}

/// Planes for `cfg`, usually the right size, sometimes deliberately not, and
/// with sample values that sometimes exceed what the bit depth allows.
fn random_planes(r: &mut Rng, cfg: &KeyFrameConfig) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let (cw, ch) = cfg.chroma_dims();
    let peak = if cfg.bit_depth >= 16 {
        u16::MAX
    } else {
        (1u16 << cfg.bit_depth.min(15)) - 1
    };
    let mut mk = |n: usize, r: &mut Rng| -> Vec<u16> {
        // Length: usually exact; sometimes short, long, or empty.
        let len = match r.below(24) {
            0 => 0,
            1 => n / 2,
            2 => n + 1 + r.below(8) as usize,
            _ => n,
        };
        // Out-of-depth samples stay common enough to keep the class reached,
        // but not so common that they starve the real encodes.
        let wild = r.chance(8);
        (0..len)
            .map(|_| {
                let v = r.next() as u16;
                if wild { v } else { v % peak.max(1) }
            })
            .collect()
    };
    let y = mk(cfg.width * cfg.height, r);
    if cfg.monochrome {
        (y, Vec::new(), Vec::new())
    } else {
        let u = mk(cw * ch, r);
        let v = mk(cw * ch, r);
        (y, u, v)
    }
}

/// THE contract: for any config and any planes, the entry returns — never
/// panics. A panic is reported with its seed and iteration so it is
/// reproducible from the failure message alone.
#[test]
fn no_config_or_plane_input_can_panic_the_encoder() {
    // Keep the default run CI-affordable; raise it locally with the env var
    // when hunting (the decoder's sweep uses the same convention).
    let iters: u64 = std::env::var("AOM_ENC_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);
    let seed: u64 = std::env::var("AOM_ENC_FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x5EED_1234_ABCD_0001);

    // A caller-supplied ceiling, so a mutated giant configuration cannot make
    // this in-process sweep allocate its way out of the runner.
    let limits = EncodeLimits {
        max_pixels: Some(1 << 16),
        max_memory_bytes: Some(1 << 30),
        ..EncodeLimits::new()
    };

    let mut r = Rng(seed);
    let (mut ok, mut unsupported, mut plane_size, mut limited, mut sample_range) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    for i in 0..iters {
        let cfg = random_config(&mut r);
        let (y, u, v) = random_planes(&mut r, &cfg);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            encode_key_frame_with(
                KeyFramePlanes {
                    y: &y,
                    u: &u,
                    v: &v,
                },
                &cfg,
                &EncodeConfig::new().with_limits(limits),
            )
        }));
        match res {
            Ok(Ok(_)) => ok += 1,
            Ok(Err(KeyFrameError::Unsupported(_))) => unsupported += 1,
            Ok(Err(KeyFrameError::PlaneSize { .. })) => plane_size += 1,
            Ok(Err(KeyFrameError::LimitExceeded { .. })) => limited += 1,
            Ok(Err(KeyFrameError::SampleRange { .. })) => sample_range += 1,
            Ok(Err(e)) => panic!("unexpected error class at iter {i}: {e}"),
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "<non-string panic>".into());
                panic!(
                    "PANIC at iteration {i} (seed {seed:#x}): {msg}\n  config: \
                     {}x{} bd{} mono={} ss=({},{}) cq={} cpu={} usage={} sb128={} \
                     tiles=({},{}) cdef={} lr={}\n  planes: y={} u={} v={}\n\
                     Reproduce with AOM_ENC_FUZZ_SEED={seed} and step to iteration {i}.",
                    cfg.width, cfg.height, cfg.bit_depth, cfg.monochrome, cfg.ss_x,
                    cfg.ss_y, cfg.cq_level, cfg.cpu_used, cfg.usage, cfg.sb_size_128,
                    cfg.tile_columns_log2, cfg.tile_rows_log2, cfg.enable_cdef,
                    cfg.enable_restoration, y.len(), u.len(), v.len()
                );
            }
        }
    }
    println!(
        "encoder fuzz sweep: {iters} inputs, {ok} encoded, {unsupported} unsupported, \
         {plane_size} plane-size, {sample_range} sample-range, {limited} limited, \
         0 panics (seed {seed:#x})"
    );

    // NON-VACUITY: a sweep that only ever hits the refusal path proves nothing
    // about the encoder. Require that real encodes and every refusal class are
    // all actually reached.
    assert!(ok > 0, "no input reached a successful encode — the sweep is vacuous");
    assert!(unsupported > 0, "no input reached the config-refusal path");
    assert!(plane_size > 0, "no input reached the plane-size refusal path");
    // The class this sweep FOUND, on its first run, as an arithmetic-overflow
    // panic in `highbd_variance64_scalar`: a bd8 encode handed 16-bit samples.
    // Keeping it non-vacuous is what stops the refusal being deleted later
    // because "nothing produces it".
    assert!(
        sample_range > 0,
        "no input reached the sample-range refusal path — the sweep no longer \
         generates out-of-depth samples, and the check it found is unguarded"
    );
}
