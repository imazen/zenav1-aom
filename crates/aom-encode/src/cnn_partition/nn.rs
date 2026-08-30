//! Port of `av1/encoder/ml.c` `av1_nn_predict_c` + `av1_nn_output_prec_reduce`
//! — the fully-connected DNN forward pass that `intra_mode_cnn_partition`
//! (the speed>=1 intra CNN partition prune) runs on its assembled features to
//! produce the split/no-split logits.
//!
//! Bit-exact transcription of the C: sequential accumulation (`val = bias;
//! val += w*in` in source order, no reassociation), ReLU on hidden layers, a
//! linear output layer, then — when `reduce_prec` — the 1/512 output
//! quantisation libaom uses to keep C and SIMD agreeing. The single caller in
//! this port (the CNN partition prune) passes `reduce_prec = true`, exactly as
//! `av1_nn_predict(dnn_features, dnn_config, 1, logits)` does.
//!
//! Validated against the REAL `av1_nn_predict_c` via `ref_nn_predict`
//! (`rd_shim.c` `shim_nn_predict`) over randomised shapes + weights in
//! `tests/cnn_partition_nn_diff.rs`.

/// `NN_MAX_NODES_PER_LAYER` (ml.h:22). Hidden layers never exceed this.
const NN_MAX_NODES_PER_LAYER: usize = 128;

/// `av1_nn_output_prec_reduce` (ml.c) — quantise each output to `prec_bits = 9`
/// fractional bits. Transcribed with the C's exact float/double promotion:
/// `output[i] * prec` is `float * int` (→ f32), `+ 0.5` promotes to double, the
/// `(int)` cast truncates toward zero, and `* inv_prec` is `int * float` (→ f32).
fn nn_output_prec_reduce(output: &mut [f32]) {
    const PREC: f32 = 512.0; // 1 << 9
    // inv_prec = (float)(1.0 / prec); 1/512 is exactly representable in f32.
    const INV_PREC: f32 = (1.0f64 / 512.0f64) as f32;
    for o in output.iter_mut() {
        // (int)(output[i] * prec + 0.5): the multiply is f32, the +0.5 is f64.
        let q = (f64::from(*o * PREC) + 0.5f64) as i32;
        *o = (q as f32) * INV_PREC;
    }
}

/// One ReLU-activated fully-connected layer: `output[node] = relu(bias[node] +
/// sum_i weights[node*num_in + i] * input[i])`. Accumulation order matches C.
fn relu_layer(input: &[f32], weights: &[f32], bias: &[f32], num_out: usize, output: &mut [f32]) {
    let num_in = input.len();
    for node in 0..num_out {
        let mut val = bias[node];
        let row = &weights[node * num_in..node * num_in + num_in];
        for i in 0..num_in {
            val += row[i] * input[i];
        }
        // ReLU: `val > 0.0f ? val : 0.0f` (ml.c). NaN -> 0.0 either way.
        output[node] = if val > 0.0 { val } else { 0.0 };
    }
}

/// Port of `av1_nn_predict_c` (ml.c). `hidden_nodes[l]` = the node count of
/// hidden layer `l`; `weights[l]` / `biases[l]` are that layer's tables
/// (`weights[l][node*num_in + i]`, `biases[l][node]`), and the final entry
/// (index `hidden_nodes.len()`) is the linear output layer producing
/// `num_outputs` values into `output`. `reduce_prec` applies the 1/512 output
/// quantisation. `weights.len() == biases.len() == hidden_nodes.len() + 1`.
pub fn nn_predict(
    features: &[f32],
    hidden_nodes: &[usize],
    weights: &[&[f32]],
    biases: &[&[f32]],
    num_outputs: usize,
    reduce_prec: bool,
    output: &mut [f32],
) {
    let num_hidden = hidden_nodes.len();
    debug_assert_eq!(weights.len(), num_hidden + 1);
    debug_assert_eq!(biases.len(), num_hidden + 1);

    // Ping-pong buffers (C uses `float buf[2][NN_MAX_NODES_PER_LAYER]`). Two
    // distinct locals so each layer reads one and writes the other without a
    // borrow conflict.
    let mut a = [0.0f32; NN_MAX_NODES_PER_LAYER];
    let mut b = [0.0f32; NN_MAX_NODES_PER_LAYER];
    a[..features.len()].copy_from_slice(features);
    let mut cur_len = features.len();
    let mut input_in_a = true;

    for layer in 0..num_hidden {
        let n_out = hidden_nodes[layer];
        debug_assert!(n_out <= NN_MAX_NODES_PER_LAYER);
        if input_in_a {
            let (inp, out) = (&a, &mut b);
            relu_layer(&inp[..cur_len], weights[layer], biases[layer], n_out, out);
        } else {
            let (inp, out) = (&b, &mut a);
            relu_layer(&inp[..cur_len], weights[layer], biases[layer], n_out, out);
        }
        cur_len = n_out;
        input_in_a = !input_in_a;
    }

    // Final (linear, no ReLU) output layer.
    let input = if input_in_a {
        &a[..cur_len]
    } else {
        &b[..cur_len]
    };
    let final_w = weights[num_hidden];
    let final_b = biases[num_hidden];
    for node in 0..num_outputs {
        let mut val = final_b[node];
        let row = &final_w[node * cur_len..node * cur_len + cur_len];
        for i in 0..cur_len {
            val += row[i] * input[i];
        }
        output[node] = val;
    }

    if reduce_prec {
        nn_output_prec_reduce(&mut output[..num_outputs]);
    }
}

// ===========================================================================
// KB-41 root #26 — the DISPATCHED `av1_nn_predict`, not `av1_nn_predict_c`
// ===========================================================================
//
// `av1_nn_predict` is an RTCD-specialized symbol (`av1_rtcd_defs.pl:467`,
// `specialize qw/av1_nn_predict sse3 avx2 neon/`), so a real aomenc encode
// runs `av1_nn_predict_avx2` on any AVX2 host — NOT the `_c` transcription
// above. The SIMD variants reassociate the dot product (pairwise `hadd`
// trees instead of C's sequential `val += w*in`) and add the bias LAST, so
// their f32 result differs from `_c` in the last ulps.
//
// That normally does not matter: `av1_nn_output_prec_reduce` quantises the
// logit to 1/512, which hides a few-ulp difference — the comment above used
// to claim it hid ALL of them. It does not. When the raw logit sits within
// a few ulps of a 1/512 boundary the two variants land on DIFFERENT quanta,
// and the intra-CNN partition prune compares that quantum against
// `no_split_thresh` (`partition_strategy.c:341`).
//
// MEASURED (KB-41 root #26), `2765x4096 cq6 --cpu-used 6`, mi(0,352),
// BLOCK_32X32 (bsize_idx 2, quad_tree_idx 1), identical DNN features on both
// sides: the dispatched C gives logit **-3.857421875** (= -1975/512) and the
// `_c` order gives **-3.859375** (= -1976/512), against
// `no_split_thresh = -3.858222961`. One quantum, and it straddles the
// threshold: C keeps `do_square_split = 1` and splits the 32x32; the port
// took `av1_disable_square_split_partition` and coded 32x32 NONE. Proven by
// re-running the SAME oracle under `AOM_SIMD_CAPS=0`, which reproduces the
// port's -3.859375 exactly.
//
// So this module models BOTH: `nn_predict` dispatches on the HOST's CPU
// capability exactly as libaom's RTCD does — deliberately NOT on the port's
// own `AOM_FORCE_SCALAR` tier, because that env only forces the PORT's
// kernels; the linked C encoder still runs its AVX2 variant, and the two
// must agree in both CI legs.
//
// **Honest scope:** only the AVX2 order is ported. On a host without AVX2 —
// aarch64 (where libaom dispatches `av1_nn_predict_neon`) and any pre-AVX2
// x86 (where it dispatches `_sse3`) — this falls back to the `_c` order,
// which is what the port did everywhere before this landing, so no platform
// regresses. Registered in PARITY as the remaining half of root #26.

/// `_mm256_hadd_ps(a, b)` — the AVX2 in-lane horizontal add, as f32 lanes.
#[inline]
fn hadd8(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    [
        a[0] + a[1],
        a[2] + a[3],
        b[0] + b[1],
        b[2] + b[3],
        a[4] + a[5],
        a[6] + a[7],
        b[4] + b[5],
        b[6] + b[7],
    ]
}

#[inline]
fn mul8(inputs: &[f32], weights: &[f32]) -> [f32; 8] {
    let mut m = [0.0f32; 8];
    for (k, m) in m.iter_mut().enumerate() {
        *m = inputs[k] * weights[k];
    }
    m
}

/// `nn_propagate_8to8` (ml_avx2.c) — 8 outputs at a time, 8 inputs at a time.
fn nn_propagate_8to8(
    inputs: &[f32],
    weights: &[f32],
    bias: &[f32],
    n_proc: usize,
    tot: usize,
    num_out: usize,
    out: &mut [f32],
    clip: bool,
) {
    let mut base = 0usize;
    while base < num_out {
        let mut in_result = [0.0f32; 8];
        let mut inx = 0usize;
        while inx < n_proc {
            let inputs256 = &inputs[inx..inx + 8];
            let weight_idx = inx + base * tot;
            let mut hadd = [[0.0f32; 8]; 4];
            for (i, h) in hadd.iter_mut().enumerate() {
                let index = weight_idx + 2 * i * tot;
                let mul0 = mul8(inputs256, &weights[index..index + 8]);
                let mul1 = mul8(inputs256, &weights[index + tot..index + tot + 8]);
                *h = hadd8(&mul0, &mul1);
            }
            let hh0 = hadd8(&hadd[0], &hadd[1]);
            let hh1 = hadd8(&hadd[2], &hadd[3]);
            // _mm256_permute2f128_ps(hh0, hh1, 0x20) / (.., 0x31).
            for k in 0..4 {
                in_result[k] += hh0[k] + hh0[k + 4];
                in_result[k + 4] += hh1[k] + hh1[k + 4];
            }
            inx += 8;
        }
        for k in 0..8 {
            let mut v = in_result[k] + bias[base + k];
            if clip && !(v > 0.0) {
                v = 0.0;
            }
            out[base + k] = v;
        }
        base += 8;
    }
}

/// `nn_propagate_8to4` (ml_avx2.c) — 4 outputs at a time.
fn nn_propagate_8to4(
    inputs: &[f32],
    weights: &[f32],
    bias: &[f32],
    n_proc: usize,
    tot: usize,
    num_out: usize,
    out: &mut [f32],
    clip: bool,
) {
    let mut base = 0usize;
    while base < num_out {
        let mut in_result = [0.0f32; 4];
        let mut inx = 0usize;
        while inx < n_proc {
            let inputs256 = &inputs[inx..inx + 8];
            let weight_idx = inx + base * tot;
            let mut hadd = [[0.0f32; 8]; 2];
            for (i, h) in hadd.iter_mut().enumerate() {
                let index = weight_idx + 2 * i * tot;
                let mul0 = mul8(inputs256, &weights[index..index + 8]);
                let mul1 = mul8(inputs256, &weights[index + tot..index + tot + 8]);
                *h = hadd8(&mul0, &mul1);
            }
            let sum_par = hadd8(&hadd[0], &hadd[1]);
            for k in 0..4 {
                in_result[k] += sum_par[k] + sum_par[k + 4];
            }
            inx += 8;
        }
        for k in 0..4 {
            let mut v = in_result[k] + bias[base + k];
            if clip && !(v > 0.0) {
                v = 0.0;
            }
            out[base + k] = v;
        }
        base += 4;
    }
}

/// `nn_propagate_8to1` (ml_avx2.c) — one output row at a time.
fn nn_propagate_8to1(
    inputs: &[f32],
    weights: &[f32],
    bias: &[f32],
    n_proc: usize,
    tot: usize,
    num_out: usize,
    out: &mut [f32],
    clip: bool,
) {
    for o in 0..num_out {
        let mut in_result = [0.0f32; 8];
        let mut inx = 0usize;
        while inx < n_proc {
            let weight_idx = inx + o * tot;
            let mul0 = mul8(&inputs[inx..inx + 8], &weights[weight_idx..weight_idx + 8]);
            for k in 0..8 {
                in_result[k] += mul0[k];
            }
            inx += 8;
        }
        // low + high, then _mm_hadd_ps(s, s) and the 0x99 shuffle-add: the
        // scalar result is `(s2+s3) + (s0+s1)` — that ADD ORDER is load-bearing.
        let s: [f32; 4] = [
            in_result[0] + in_result[4],
            in_result[1] + in_result[5],
            in_result[2] + in_result[6],
            in_result[3] + in_result[7],
        ];
        let sum_par_1 = [s[0] + s[1], s[2] + s[3]];
        let mut val = bias[o] + (sum_par_1[1] + sum_par_1[0]);
        if clip && !(val > 0.0) {
            val = 0.0;
        }
        out[o] = val;
    }
}

/// `nn_propagate_input_multiple_of_8` (ml_avx2.c) — the shape selector.
#[allow(clippy::too_many_arguments)]
fn nn_propagate_input_multiple_of_8(
    inputs: &[f32],
    weights: &[f32],
    bias: &[f32],
    n_proc: usize,
    tot: usize,
    is_output_layer: bool,
    num_out: usize,
    out: &mut [f32],
) {
    // "The saturation of output is considered for hidden layer which is not
    // equal to final hidden layer" — i.e. only when this call consumed EVERY
    // input, else the remainder loop below applies the ReLU.
    let clip = !is_output_layer && n_proc == tot;
    if num_out % 8 == 0 {
        nn_propagate_8to8(inputs, weights, bias, n_proc, tot, num_out, out, clip);
    } else if num_out % 4 == 0 {
        nn_propagate_8to4(inputs, weights, bias, n_proc, tot, num_out, out, clip);
    } else {
        nn_propagate_8to1(inputs, weights, bias, n_proc, tot, num_out, out, clip);
    }
}

/// `av1_nn_propagate_4to8_sse3` / `_4to4_` / `_4to1_` (ml_sse3.c) as scalar
/// lane arithmetic — the AVX2 path's remainder helpers, reached when the
/// leftover input count is a multiple of 4.
#[inline]
fn propagate_4to1(inputs: &[f32], weights: &[f32], total: &mut [f32; 4]) {
    // `_mm_mul_ps` then two `_mm_hadd_ps(m, zero)` folds, then add to lane 0.
    let m: [f32; 4] = [
        inputs[0] * weights[0],
        inputs[1] * weights[1],
        inputs[2] * weights[2],
        inputs[3] * weights[3],
    ];
    let h0 = [m[0] + m[1], m[2] + m[3]];
    let v = h0[0] + h0[1];
    total[0] += v;
}

#[inline]
fn propagate_4to4(inputs: &[f32], weights: &[f32], num_inputs: usize, outputs: &mut [f32; 4]) {
    for (r, o) in outputs.iter_mut().enumerate() {
        let w = &weights[r * num_inputs..r * num_inputs + 4];
        let m: [f32; 4] = [
            inputs[0] * w[0],
            inputs[1] * w[1],
            inputs[2] * w[2],
            inputs[3] * w[3],
        ];
        *o += (m[0] + m[1]) + (m[2] + m[3]);
    }
}

/// Port of the DISPATCHED `av1_nn_predict_avx2` (ml_avx2.c).
fn nn_predict_avx2_order(
    features: &[f32],
    hidden_nodes: &[usize],
    weights: &[&[f32]],
    biases: &[&[f32]],
    num_outputs: usize,
    reduce_prec: bool,
    output: &mut [f32],
) {
    let num_hidden = hidden_nodes.len();
    let mut a = [0.0f32; NN_MAX_NODES_PER_LAYER];
    let mut b = [0.0f32; NN_MAX_NODES_PER_LAYER];
    a[..features.len()].copy_from_slice(features);
    let mut num_inputs = features.len();
    let mut input_in_a = true;

    for layer in 0..=num_hidden {
        let is_output_layer = layer == num_hidden;
        let num_out = if is_output_layer {
            num_outputs
        } else {
            hidden_nodes[layer]
        };
        let lw = weights[layer];
        let lb = biases[layer];
        let mut scratch = [0.0f32; NN_MAX_NODES_PER_LAYER];
        {
            let inp: &[f32] = if input_in_a {
                &a[..num_inputs]
            } else {
                &b[..num_inputs]
            };
            if num_inputs % 8 == 0 {
                nn_propagate_input_multiple_of_8(
                    inp,
                    lw,
                    lb,
                    num_inputs,
                    num_inputs,
                    is_output_layer,
                    num_out,
                    &mut scratch,
                );
            } else {
                let in_mul_8 = num_inputs / 8;
                let processed = in_mul_8 * 8;
                let mut bias_is_considered = false;
                if in_mul_8 != 0 {
                    nn_propagate_input_multiple_of_8(
                        inp,
                        lw,
                        lb,
                        processed,
                        num_inputs,
                        is_output_layer,
                        num_out,
                        &mut scratch,
                    );
                    bias_is_considered = true;
                }
                // `out_temp = bias_is_considered ? output_nodes : layer_bias`.
                let remaining = num_inputs % 8;
                if remaining % 4 == 0 && num_out % 4 == 0 {
                    // The 4to8 / 4to4 arms: both accumulate `(m0+m1)+(m2+m3)`
                    // per row, so one helper covers the pair.
                    let mut base = 0usize;
                    while base < num_out {
                        let mut acc = [0.0f32; 4];
                        for k in 0..4 {
                            acc[k] = if bias_is_considered {
                                scratch[base + k]
                            } else {
                                lb[base + k]
                            };
                        }
                        let mut inx = processed;
                        while inx < num_inputs {
                            propagate_4to4(
                                &inp[inx..inx + 4],
                                &lw[base * num_inputs + inx..],
                                num_inputs,
                                &mut acc,
                            );
                            inx += 4;
                        }
                        for k in 0..4 {
                            let mut v = acc[k];
                            if !is_output_layer && !(v > 0.0) {
                                v = 0.0;
                            }
                            scratch[base + k] = v;
                        }
                        base += 4;
                    }
                } else if remaining % 4 == 0 {
                    for o in 0..num_out {
                        let mut total = [0.0f32; 4];
                        total[0] = if bias_is_considered { scratch[o] } else { lb[o] };
                        let mut inx = processed;
                        while inx < num_inputs {
                            propagate_4to1(
                                &inp[inx..inx + 4],
                                &lw[o * num_inputs + inx..o * num_inputs + inx + 4],
                                &mut total,
                            );
                            inx += 4;
                        }
                        let mut v = total[0];
                        if !is_output_layer && !(v > 0.0) {
                            v = 0.0;
                        }
                        scratch[o] = v;
                    }
                } else {
                    // The scalar-in-SSE tail: plain sequential mul-add.
                    for o in 0..num_out {
                        let mut v = if bias_is_considered { scratch[o] } else { lb[o] };
                        for in_node in processed..num_inputs {
                            v += inp[in_node] * lw[num_inputs * o + in_node];
                        }
                        if !is_output_layer && !(v > 0.0) {
                            v = 0.0;
                        }
                        scratch[o] = v;
                    }
                }
            }
        }
        if is_output_layer {
            output[..num_outputs].copy_from_slice(&scratch[..num_outputs]);
        } else if input_in_a {
            b[..num_out].copy_from_slice(&scratch[..num_out]);
        } else {
            a[..num_out].copy_from_slice(&scratch[..num_out]);
        }
        num_inputs = num_out;
        input_in_a = !input_in_a;
    }

    if reduce_prec {
        nn_output_prec_reduce(&mut output[..num_outputs]);
    }
}

/// Does the LINKED libaom dispatch `av1_nn_predict_avx2` on this host? RTCD
/// picks the highest available specialization, so this asks the same question
/// libaom's own `x86_simd_caps()` does. Deliberately independent of the port's
/// `AOM_FORCE_SCALAR` (that env forces the PORT's kernels; the C encoder it is
/// compared against still runs AVX2).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn host_dispatches_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn host_dispatches_avx2() -> bool {
    false
}

/// THE entry point the intra-CNN prune uses: whichever `av1_nn_predict`
/// variant the linked C encoder dispatches on this host. Falls back to the
/// `_c` order ([`nn_predict`]) where the AVX2 variant is not the dispatched
/// one — see the module note on the remaining NEON/SSE3 half of root #26.
#[allow(clippy::too_many_arguments)]
pub fn nn_predict_dispatched(
    features: &[f32],
    hidden_nodes: &[usize],
    weights: &[&[f32]],
    biases: &[&[f32]],
    num_outputs: usize,
    reduce_prec: bool,
    output: &mut [f32],
) {
    if host_dispatches_avx2() {
        nn_predict_avx2_order(
            features,
            hidden_nodes,
            weights,
            biases,
            num_outputs,
            reduce_prec,
            output,
        );
    } else {
        nn_predict(
            features,
            hidden_nodes,
            weights,
            biases,
            num_outputs,
            reduce_prec,
            output,
        );
    }
}

/// Test-only handle on the AVX2-order transcription, so the differential gate
/// can compare it against the DISPATCHED C on an AVX2 host regardless of what
/// [`nn_predict_dispatched`] selects.
#[allow(clippy::too_many_arguments)]
pub fn nn_predict_avx2_order_for_test(
    features: &[f32],
    hidden_nodes: &[usize],
    weights: &[&[f32]],
    biases: &[&[f32]],
    num_outputs: usize,
    reduce_prec: bool,
    output: &mut [f32],
) {
    nn_predict_avx2_order(
        features,
        hidden_nodes,
        weights,
        biases,
        num_outputs,
        reduce_prec,
        output,
    );
}
