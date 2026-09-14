//! CRC-32C (Castagnoli, poly `0x82F63B78`) for the IntraBC hash machinery —
//! `aom_encode::intrabc_search` is the consumer. The CRC *value* is
//! load-bearing there (bucket key + full-hash equality gate + the 64-candidate
//! cap), so only a bit-identical acceleration is admissible — the `crc32q`
//! instruction is exactly that.
//!
//! `safe_unaligned_simd` is deliberately not the vehicle: that crate wraps
//! only pointer-taking unaligned load/store intrinsics, and `crc32` takes
//! register values. `#[arcane]` carries `target_feature(sse4.2)` on the v2
//! tier, which makes `_mm_crc32_u64` a safe call inside the body (Rust 1.87+);
//! `incant!` owns the one `unsafe` at the dispatch boundary, so
//! `#![forbid(unsafe_code)]` stays in force.

/// CRC-32C of a 16-byte input on the SSE4.2 tier, or `None` when no hardware
/// tier is available (or `AOM_FORCE_SCALAR` is pinned) — the caller keeps its
/// software-table fallback either way. Bit-identical to the standard
/// init-`0xffffffff` / final-xor-`0xffffffff` Castagnoli CRC.
#[inline]
pub fn crc32c16_hw(b: &[u8; 16]) -> Option<u32> {
    let _ = crate::dispatch::scalar_forced();
    archmage::incant!(crc32c16(b), [v2, scalar])
}

/// Two chained `crc32q`s over the LE halves — the `mcomp.c`/`hash_sse42.c`
/// shape. Serial latency ~6 cycles, but ~4 uops total versus the table
/// variant's 16 loads + 16 XORs, so throughput wins at IntraBC's call volume.
#[archmage::arcane]
fn crc32c16_v2(_token: archmage::X64V2Token, b: &[u8; 16]) -> Option<u32> {
    use core::arch::x86_64::*;
    let lo = u64::from_le_bytes(b[..8].try_into().unwrap());
    let hi = u64::from_le_bytes(b[8..].try_into().unwrap());
    Some((_mm_crc32_u64(_mm_crc32_u64(0xffff_ffff, lo), hi) as u32) ^ 0xffff_ffff)
}

/// Decline: no hardware tier → the caller's table path. Also what the
/// `AOM_FORCE_SCALAR` pin selects.
fn crc32c16_scalar(_t: archmage::ScalarToken, _b: &[u8; 16]) -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Software CRC-32C reference — slicing-by-1, the `hash.c` semantics
    /// (init/final-xor `0xffffffff`).
    fn table_crc32c16(b: &[u8; 16]) -> u32 {
        const P: u32 = 0x82f63b78;
        let mut t = [0u32; 256];
        for n in 0..256u32 {
            let mut c = n;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ P } else { c >> 1 };
            }
            t[n as usize] = c;
        }
        let mut crc = 0xffff_ffffu32;
        for &x in b {
            crc = t[((crc ^ u32::from(x)) & 0xff) as usize] ^ (crc >> 8);
        }
        crc ^ 0xffff_ffff
    }

    #[test]
    fn hw_matches_table_crc32c() {
        let mut s = 0x9e3779b97f4a7c15u64;
        for _ in 0..4096 {
            let mut b = [0u8; 16];
            for x in b.iter_mut() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                *x = s as u8;
            }
            if let Some(v) = crc32c16_hw(&b) {
                assert_eq!(v, table_crc32c16(&b), "input {b:02x?}");
            }
        }
    }
}
