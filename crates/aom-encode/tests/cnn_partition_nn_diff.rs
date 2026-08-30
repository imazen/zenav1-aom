//! Differential: the ported DNN forward pass (`cnn_partition::nn::nn_predict`,
//! a transcription of `av1/encoder/ml.c av1_nn_predict_c` + prec-reduce) vs the
//! REAL `av1_nn_predict_c` (via `ref_nn_predict` / `rd_shim.c shim_nn_predict`),
//! over randomised shapes + weights + features. Covers `reduce_prec` on (the
//! path `intra_mode_cnn_partition` uses: `av1_nn_predict(.., 1, ..)`) and off.
//!
//! Includes the four ACTUAL intra-CNN branch DNN shapes (features 37/25/25/41,
//! hidden 16→24, 1 logit) so the exact model geometry is exercised.

use aom_encode::cnn_partition::nn::{nn_predict, nn_predict_dispatched};
use aom_sys_ref as c;

struct XorShift(u64);
impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// A float in [-range, range], quantised to a modest grid so values are
    /// representative of trained weights (not denormal noise).
    fn f(&mut self, range: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        (u * 2.0 - 1.0) * range
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }
}

/// Run both implementations for one config and assert bit-identical logits.
fn check(
    rng: &mut XorShift,
    num_inputs: usize,
    hidden: &[usize],
    num_outputs: usize,
    reduce_prec: bool,
    tag: &str,
) {
    // Build per-layer weight/bias tables.
    let mut dims = Vec::new();
    let mut prev = num_inputs;
    for &h in hidden {
        dims.push((prev, h));
        prev = h;
    }
    dims.push((prev, num_outputs));

    let mut weights: Vec<Vec<f32>> = Vec::new();
    let mut biases: Vec<Vec<f32>> = Vec::new();
    for &(nin, nout) in &dims {
        weights.push((0..nin * nout).map(|_| rng.f(1.5)).collect());
        biases.push((0..nout).map(|_| rng.f(0.8)).collect());
    }
    let features: Vec<f32> = (0..num_inputs).map(|_| rng.f(3.0)).collect();

    // Rust port.
    let w_refs: Vec<&[f32]> = weights.iter().map(|v| v.as_slice()).collect();
    let b_refs: Vec<&[f32]> = biases.iter().map(|v| v.as_slice()).collect();
    let mut got = vec![0.0f32; num_outputs];
    nn_predict(
        &features,
        hidden,
        &w_refs,
        &b_refs,
        num_outputs,
        reduce_prec,
        &mut got,
    );

    // C oracle: flatten weights/biases in NN_CONFIG order.
    let w_flat: Vec<f32> = weights.iter().flatten().copied().collect();
    let b_flat: Vec<f32> = biases.iter().flatten().copied().collect();
    let hidden_i32: Vec<i32> = hidden.iter().map(|&h| h as i32).collect();
    let want = c::ref_nn_predict(
        &features,
        num_inputs,
        num_outputs,
        &hidden_i32,
        &w_flat,
        &b_flat,
        reduce_prec,
    );

    for (o, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{tag}: logit[{o}] mismatch: rust={g} ({:#010x}) c={w} ({:#010x}) \
             [num_inputs={num_inputs} hidden={hidden:?} num_outputs={num_outputs} \
             reduce_prec={reduce_prec}]",
            g.to_bits(),
            w.to_bits()
        );
    }
}

#[test]
fn nn_predict_matches_c_random_shapes() {
    let mut rng = XorShift(0x1234_5678_9abc_def1);
    let mut n = 0usize;
    for _ in 0..4000 {
        let num_inputs = rng.range(1, 48);
        let num_hidden = rng.range(1, 4);
        let hidden: Vec<usize> = (0..num_hidden).map(|_| rng.range(1, 40)).collect();
        let num_outputs = rng.range(1, 8);
        for &rp in &[false, true] {
            check(&mut rng, num_inputs, &hidden, num_outputs, rp, "random");
            n += 1;
        }
    }
    eprintln!("nn_predict_matches_c_random_shapes: {n} configs bit-identical");
}

#[test]
fn nn_predict_matches_c_intra_cnn_branch_shapes() {
    // The four real intra-CNN branch DNN geometries (partition_cnn_weights.h):
    // features 37/25/25/41, two hidden layers 16 -> 24, a single logit; run with
    // reduce_prec=1 (exactly how intra_mode_cnn_partition calls it).
    let mut rng = XorShift(0xdead_beef_0bad_f00d);
    let branch_features = [37usize, 25, 25, 41];
    let mut n = 0usize;
    for _ in 0..500 {
        for &nf in &branch_features {
            check(&mut rng, nf, &[16, 24], 1, true, "intra-cnn-branch");
            // also exercise the raw (pre-prec-reduce) path.
            check(&mut rng, nf, &[16, 24], 1, false, "intra-cnn-branch-raw");
            n += 2;
        }
    }
    eprintln!("nn_predict_matches_c_intra_cnn_branch_shapes: {n} configs bit-identical");
}


// ---------------------------------------------------------------------------
// KB-41 root #26 — the DISPATCHED variant
// ---------------------------------------------------------------------------

/// `av1_nn_predict` is RTCD-specialized (`av1_rtcd_defs.pl:467`), so a real
/// encode runs `av1_nn_predict_avx2` on any AVX2 host, NOT the `_c` function
/// the test above pins. Those reassociate the dot product, and the intra-CNN
/// prune's 1/512 output quantisation does not always hide the difference: on
/// `2765x4096 cq6 --cpu-used 6` mi(0,352) the two land on adjacent quanta
/// either side of `no_split_thresh`, flipping `do_square_split`.
///
/// So the SHIPPING entry point (`nn_predict_dispatched`, what
/// `finish_decision` calls) is gated against the DISPATCHED C here — the same
/// four real intra-CNN branch geometries plus randomised shapes.
#[test]
fn nn_predict_dispatched_matches_dispatched_c() {
    // `av1_nn_predict` is an RTCD pointer — it is NULL until the tables are set
    // up (`palette_shim.c`'s note: "Requires ref_init() (rtcd)").
    c::ref_init();
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    // The four real intra-CNN branch DNN shapes + a randomised sweep.
    let real_shapes: [usize; 4] = [37, 25, 25, 41];
    let mut cases: Vec<(usize, Vec<usize>, usize)> = real_shapes
        .iter()
        .map(|&n| (n, vec![16usize, 24usize], 1usize))
        .collect();
    for _ in 0..200 {
        let num_in = rng.range(1, 48);
        let h0 = rng.range(1, 32);
        let h1 = rng.range(1, 32);
        cases.push((num_in, vec![h0, h1], rng.range(1, 4)));
    }

    let mut mismatches = Vec::new();
    for (num_in, hidden, num_out) in cases {
        let mut weights_flat = Vec::new();
        let mut bias_flat = Vec::new();
        let mut sizes = vec![num_in];
        sizes.extend(hidden.iter().copied());
        sizes.push(num_out);
        for w in sizes.windows(2) {
            for _ in 0..w[0] * w[1] {
                weights_flat.push(rng.f(0.5));
            }
            for _ in 0..w[1] {
                bias_flat.push(rng.f(0.5));
            }
        }
        let features: Vec<f32> = (0..num_in).map(|_| rng.f(2.0)).collect();

        // Slice the flat tables the way the port's API wants them.
        let mut wsl: Vec<&[f32]> = Vec::new();
        let mut bsl: Vec<&[f32]> = Vec::new();
        let (mut wo, mut bo) = (0usize, 0usize);
        for w in sizes.windows(2) {
            wsl.push(&weights_flat[wo..wo + w[0] * w[1]]);
            bsl.push(&bias_flat[bo..bo + w[1]]);
            wo += w[0] * w[1];
            bo += w[1];
        }

        let mut got = vec![0.0f32; num_out];
        nn_predict_dispatched(&features, &hidden, &wsl, &bsl, num_out, true, &mut got);
        let hidden_i32: Vec<i32> = hidden.iter().map(|&h| h as i32).collect();
        let want = c::ref_nn_predict_dispatched(
            &features,
            num_in,
            num_out,
            &hidden_i32,
            &weights_flat,
            &bias_flat,
            true,
        );
        if got != want {
            mismatches.push(format!(
                "in={num_in} hidden={hidden:?} out={num_out}: port {got:?} vs dispatched C {want:?}"
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of the sweep disagree with the DISPATCHED av1_nn_predict:\n  {}",
        mismatches.len(),
        mismatches.join("\n  ")
    );
}
