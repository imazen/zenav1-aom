//! CDF adaptation + symbol coding, bit-exact port of libaom v3.14.1
//! `update_cdf` (`aom_dsp/prob.h`) and the `aom_write_symbol`/`aom_read_symbol`
//! composition (`aom_dsp/bitwriter.h`/`bitreader.h`).
//!
//! AV1 CDF arrays are stored in *inverse* form (Q15), length `nsymbs+1`:
//! entries `0..nsymbs-1` are the icdf (`cdf[nsymbs-1] == 0`), and `cdf[nsymbs]`
//! is the adaptation counter.

use crate::{OdEcDec, OdEcEnc};

const CDF_PROB_TOP: i32 = 1 << 15;

/// Bit-exact port of `update_cdf`.
pub fn update_cdf(cdf: &mut [u16], val: i32, nsymbs: usize) {
    let count = cdf[nsymbs] as i32;
    let rate = 4 + (count >> 4) + (nsymbs > 3) as i32;
    let mut i = 0usize;
    loop {
        let c = cdf[i] as i32;
        if (i as i32) < val {
            cdf[i] = (c + ((CDF_PROB_TOP - c) >> rate)) as u16;
        } else {
            cdf[i] = (c - (c >> rate)) as u16;
        }
        i += 1;
        if i >= nsymbs - 1 {
            break;
        }
    }
    cdf[nsymbs] += (count < 32) as u16;
}

/// `aom_write_symbol` with CDF adaptation enabled.
pub fn write_symbol(enc: &mut OdEcEnc, symb: i32, cdf: &mut [u16], nsymbs: usize) {
    enc.encode_cdf_q15(symb, &cdf[..nsymbs], nsymbs as i32);
    update_cdf(cdf, symb, nsymbs);
}

/// `aom_read_symbol` with CDF adaptation enabled.
pub fn read_symbol(dec: &mut OdEcDec, cdf: &mut [u16], nsymbs: usize) -> i32 {
    let ret = dec.decode_cdf_q15(&cdf[..nsymbs], nsymbs as i32);
    update_cdf(cdf, ret, nsymbs);
    ret
}
