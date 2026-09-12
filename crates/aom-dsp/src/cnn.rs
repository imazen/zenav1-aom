//! AVX2 port of `av1_cnn_convolve_no_maxpool_padding_valid_avx2`
//! (`upstream/av1/encoder/x86/cnn_avx2.c`) — the convolve the **dispatched**
//! libaom build runs inside `av1_cnn_predict_img_multi_out`.
//!
//! # Why this exists
//!
//! The encoder-side caller (`aom_encode::cnn_partition::cnn::conv_valid`) is a
//! verbatim transcription of `av1_cnn_convolve_no_maxpool_padding_valid_c` —
//! bit-exact against the C-SCALAR engine, but real libaom runs the AVX2 kernel,
//! which accumulates each output through `hadd` trees, not the scalar
//! sequential order. That difference is measured to flip partition decisions
//! (KB-41 root #27: branch logits differ in the 7th digit, the prec-reduce
//! quantum is 1/512, `no_split_thresh` comparisons land on either side of a
//! boundary). The fix is not "tighten the tolerance" — it is to run the same
//! arithmetic the dispatched C runs, which is what this file does: a 1:1 port
//! of the AVX2 layer kernels, every `mul_ps`/`add_ps`/`hadd_ps` in C's order,
//! so the result is bit-identical to `ref_intra_cnn_run(_, false)`.
//!
//! # Dispatch shape
//!
//! [`conv_valid`] returns `true` when a v3 kernel wrote `output`. It returns
//! `false` for everything C's own dispatch hands back to `_c` — the layer-3/4
//! shapes (4x4, 2x2 inputs), non-x86 builds, absent v3, and the
//! `AOM_FORCE_SCALAR` pin — and the caller's scalar transcription runs, which
//! is itself bit-exact against `_c`. So every reachable configuration is
//! bit-exact against the function real libaom would dispatch for it.
//!
//! # Bounds
//!
//! All loads are unaligned vector reads inside the input planes. For the gated
//! shapes every access is provably in-bounds (see each kernel); slice indexing
//! keeps the same panic surface as the scalar recipe.

use archmage::magetypes;
use archmage::prelude::*;

/// One conv layer of `av1_cnn_convolve_no_maxpool_padding_valid_avx2`.
///
/// `input` is the flat `[channel][row][col]` plane (`channel k` at
/// `k * in_h * in_stride`), `kernel`/`bias` the layer's weight tables with C's
/// `off = k * out_ch + i`, `off += in_ch * out_ch` per tap indexing. `output`
/// is the flat `[channel][row][col]` destination (`out_ch * out_h * out_w`).
///
/// Returns `true` if `output` was written by the v3 kernels.
#[allow(clippy::too_many_arguments)]
pub fn conv_valid(
    input: &[f32],
    in_ch: usize,
    in_w: usize,
    in_h: usize,
    in_stride: usize,
    kernel: &[f32],
    bias: &[f32],
    out_ch: usize,
    filter: usize,
    skip: usize,
    output: &mut [f32],
    out_w: usize,
    out_h: usize,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = crate::dispatch::scalar_forced();
        return archmage::incant!(
            conv_valid_impl(
                input, in_ch, in_w, in_h, in_stride, kernel, bias, out_ch, filter, skip, output,
                out_w, out_h,
            ),
            [v3, scalar]
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (
            input, in_ch, in_w, in_h, in_stride, kernel, bias, out_ch, filter, skip, output, out_w,
            out_h,
        );
        false
    }
}

/// Reach probe for tests: `true` when [`conv_valid`] will run the v3 kernels
/// in this process (x86-64 v3 token present and the `AOM_FORCE_SCALAR` pin not
/// applied). The first call applies the pin if set — the same ordering rule
/// every dispatch entry point follows — so the answer always describes the
/// tier that will actually run.
pub fn v3_tier_active() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = crate::dispatch::scalar_forced();
        return X64V3Token::summon().is_some();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Force-scalar / no-token arm: decline so the caller's `_c` transcription
/// runs — exactly the arm the test pins against the C-scalar oracle.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn conv_valid_impl_scalar(
    _t: ScalarToken,
    _input: &[f32],
    _in_ch: usize,
    _in_w: usize,
    _in_h: usize,
    _in_stride: usize,
    _kernel: &[f32],
    _bias: &[f32],
    _out_ch: usize,
    _filter: usize,
    _skip: usize,
    _output: &mut [f32],
    _out_w: usize,
    _out_h: usize,
) -> bool {
    false
}

/// `av1_cnn_convolve_no_maxpool_padding_valid_avx2`: 5x5/4-stride layers go
/// through the layer-0 kernel; 2x2/2-stride layers at 16x16 or 8x8 inputs go
/// through the layer-1/layer-2 kernels; every other shape declines (C calls
/// `_c` for those too).
#[cfg(target_arch = "x86_64")]
#[magetypes(define(f32x8), v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn conv_valid_impl(
    _token: Token,
    input: &[f32],
    in_ch: usize,
    in_w: usize,
    in_h: usize,
    in_stride: usize,
    kernel: &[f32],
    bias: &[f32],
    out_ch: usize,
    filter: usize,
    skip: usize,
    output: &mut [f32],
    out_w: usize,
    out_h: usize,
) -> bool {
    use archmage::intrinsics::x86_64::*;

    let cstep = in_ch * out_ch;
    let in_ch_len = in_h * in_stride;
    let out_ch_len = out_h * out_w;

    let l256 = |s: &[f32]| _mm256_loadu_ps(s[..8].try_into().unwrap());
    let l128 = |s: &[f32]| _mm_loadu_ps(s[..4].try_into().unwrap());

    if filter == 5 && skip == 4 {
        // `cnn_convolve_no_maxpool_padding_valid_5x5_avx2`. The source shuffle
        // masks fold three 5x5 windows (starts w, w+4, w+8) into two 8-lane
        // loads; the weight masks place each row's five taps accordingly.
        let block0_1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 4, 5, 6);
        let block1_2 = _mm256_setr_epi32(0, 1, 1, 2, 3, 4, 5, 0);
        let wm0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 0, 1, 2);
        let wm1 = _mm256_setr_epi32(3, 4, 0, 1, 2, 3, 4, 0);
        for i in 0..out_ch {
            let ob = bias[i];
            for k in 0..in_ch {
                // prepare_weights_for_5x5_convolve.
                let mut w = [[0f32; 8]; 5];
                let mut off = k * out_ch + i;
                for row in w.iter_mut() {
                    for c in row.iter_mut().take(5) {
                        *c = kernel[off];
                        off += cstep;
                    }
                }
                let mut sw = [_mm256_setzero_ps(); 10];
                for (r, s) in sw.iter_mut().enumerate().take(5) {
                    *s = _mm256_permutevar8x32_ps(_mm256_loadu_ps(&w[r]), wm0);
                }
                for r in 0..5 {
                    sw[5 + r] = _mm256_permutevar8x32_ps(sw[r], wm1);
                }

                let cbase = k * in_ch_len;
                let mut u = 0usize;
                let mut h = 0usize;
                while h + 5 <= in_h {
                    let orow = i * out_ch_len + u * out_w;
                    let mut v = 0usize;
                    let mut wcol = 0usize;
                    let mut rem = in_w;
                    // PERFORM_CONVOLVE_FOR_3_5X5_BLOCKS — three blocks while
                    // `rem >= 2*skip + 5` (= 13).
                    while rem >= 13 {
                        let base = cbase + h * in_stride + wcol;
                        let mut a0 = _mm256_setzero_ps();
                        let mut a1 = _mm256_setzero_ps();
                        for row in 0..5 {
                            let p = base + row * in_stride;
                            let s0 = _mm256_permutevar8x32_ps(l256(&input[p..]), block0_1);
                            let s1 = _mm256_permutevar8x32_ps(l256(&input[p + 7..]), block1_2);
                            a0 = _mm256_add_ps(_mm256_mul_ps(s0, sw[row]), a0);
                            a1 = _mm256_add_ps(_mm256_mul_ps(s1, sw[5 + row]), a1);
                        }
                        let acc = _mm256_hadd_ps(a0, a1);
                        let t0 = _mm256_extractf128_ps::<1>(a0);
                        let t1 = _mm256_extractf128_ps::<1>(a1);
                        let al = _mm256_castps256_ps128(acc);
                        let ah = _mm256_extractf128_ps::<1>(acc);
                        let t2 = _mm_add_ps(al, t0);
                        let t3 = _mm_add_ps(t0, ah);
                        let t4 = _mm_add_ps(t1, ah);
                        let (mut ala, mut t2a, mut t3a, mut t4a) =
                            ([0f32; 4], [0f32; 4], [0f32; 4], [0f32; 4]);
                        _mm_storeu_ps(&mut ala, al);
                        _mm_storeu_ps(&mut t2a, t2);
                        _mm_storeu_ps(&mut t3a, t3);
                        _mm_storeu_ps(&mut t4a, t4);
                        output[orow + v] = ob + t2a[0] + ala[1];
                        output[orow + v + 1] = ob + t3a[1] + ala[2];
                        output[orow + v + 2] = ob + t4a[2] + ala[3];
                        v += 3;
                        wcol += 12;
                        rem -= 12;
                    }
                    // PERFORM_CONVOLVE_FOR_1_5X5_BLOCK — one block while
                    // `rem >= 5`. `w` here is the UNSHUFFLED `weight[5][8]`
                    // (the macro reads the outer `weight`, not its `w`
                    // parameter) and `w[r]`'s low 128 is `weight[r][0..4]`.
                    while rem >= 5 {
                        let base = cbase + h * in_stride + wcol;
                        let mut lcs = 0f32;
                        let mut ls = [_mm_setzero_ps(); 5];
                        for row in 0..5 {
                            let p = base + row * in_stride;
                            ls[row] = l128(&input[p..]);
                            lcs += input[p + 4] * w[row][4];
                        }
                        for (r, l) in ls.iter_mut().enumerate() {
                            *l = _mm_mul_ps(*l, _mm256_castps256_ps128(sw[r]));
                        }
                        let l13 =
                            _mm_add_ps(_mm_add_ps(ls[1], ls[2]), _mm_add_ps(ls[3], ls[4]));
                        let acc = _mm_hadd_ps(_mm_add_ps(ls[0], l13), _mm_add_ps(ls[0], l13));
                        let mut a4 = [0f32; 4];
                        _mm_storeu_ps(&mut a4, acc);
                        output[orow + v] = ob + lcs + a4[0] + a4[1];
                        v += 1;
                        wcol += 4;
                        rem -= 4;
                    }
                    h += 4;
                    u += 1;
                }
            }
        }
        return true;
    }

    if filter == 2 && skip == 2 {
        // `load_shuffle_masks_for_2x2_convolve`.
        let om = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        let wm0 = _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1);
        let wm1 = _mm256_setr_epi32(2, 3, 2, 3, 2, 3, 2, 3);
        // prepare_weights_for_2x2_convolve.
        let prep = |i: usize, k: usize| -> [__m256; 2] {
            let mut w4 = [0f32; 4];
            let mut off = k * out_ch + i;
            for t in w4.iter_mut() {
                *t = kernel[off];
                off += cstep;
            }
            let wv = _mm256_castps128_ps256(_mm_loadu_ps(&w4));
            [
                _mm256_permutevar8x32_ps(wv, wm0),
                _mm256_permutevar8x32_ps(wv, wm1),
            ]
        };

        if in_w == 16 && in_h == 16 {
            // `cnn_convolve_no_maxpool_padding_valid_layer1_avx2`: eight
            // horizontal 2x2 blocks per output row; `out_accum[u]` holds the
            // u-th output row's 8 outputs, accumulated across in_channels.
            for i in 0..out_ch {
                let bv = _mm256_set1_ps(bias[i]);
                let mut acc = [bv; 8];
                for k in 0..in_ch {
                    let sw = prep(i, k);
                    let cb = k * in_ch_len;
                    let mut u = 0usize;
                    let mut h = 0usize;
                    while h + 2 <= 16 {
                        let p = cb + h * in_stride;
                        let m0 = _mm256_mul_ps(l256(&input[p..]), sw[0]);
                        let m1 = _mm256_mul_ps(l256(&input[p + 8..]), sw[0]);
                        let m2 = _mm256_mul_ps(l256(&input[p + in_stride..]), sw[1]);
                        let m3 = _mm256_mul_ps(l256(&input[p + in_stride + 8..]), sw[1]);
                        let r = _mm256_hadd_ps(_mm256_add_ps(m0, m2), _mm256_add_ps(m1, m3));
                        acc[u] = _mm256_add_ps(acc[u], _mm256_permutevar8x32_ps(r, om));
                        h += 2;
                        u += 1;
                    }
                }
                for (j, a) in acc.iter().enumerate() {
                    let o = i * out_ch_len + j * out_w;
                    _mm256_storeu_ps((&mut output[o..o + 8]).try_into().unwrap(), *a);
                }
            }
            return true;
        }

        if in_w == 8 && in_h == 8 {
            // `cnn_convolve_no_maxpool_padding_valid_layer2_avx2`: each
            // `out_accum[j]` covers 4 horizontal x 2 vertical 2x2 blocks =
            // two output rows of 4, stored 8 floats at `j * out_stride * 2`.
            for i in 0..out_ch {
                let bv = _mm256_set1_ps(bias[i]);
                let mut acc = [bv; 2];
                for k in 0..in_ch {
                    let sw = prep(i, k);
                    let cb = k * in_ch_len;
                    let mut u = 0usize;
                    let mut h = 0usize;
                    while h + 2 <= 8 {
                        let p = cb + h * in_stride;
                        let m0 = _mm256_mul_ps(l256(&input[p..]), sw[0]);
                        let m1 = _mm256_mul_ps(l256(&input[p + in_stride..]), sw[1]);
                        let m2 = _mm256_mul_ps(l256(&input[p + 2 * in_stride..]), sw[0]);
                        let m3 = _mm256_mul_ps(l256(&input[p + 3 * in_stride..]), sw[1]);
                        let r = _mm256_hadd_ps(_mm256_add_ps(m0, m1), _mm256_add_ps(m2, m3));
                        acc[u] = _mm256_add_ps(acc[u], _mm256_permutevar8x32_ps(r, om));
                        h += 4;
                        u += 1;
                    }
                }
                for (j, a) in acc.iter().enumerate() {
                    let o = i * out_ch_len + j * out_w * 2;
                    _mm256_storeu_ps((&mut output[o..o + 8]).try_into().unwrap(), *a);
                }
            }
            return true;
        }

        // Layers 3/4 (4x4, 2x2 inputs): C calls `_c` — decline so the caller's
        // scalar transcription runs, bit-exact against the same oracle.
        return false;
    }

    false
}
