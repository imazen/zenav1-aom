//! LEB128 unsigned varint codec (libaom `aom/src/aom_integer.c`): the
//! variable-length integer encoding used for OBU sizes. 7 payload bits per byte,
//! MSB = continuation. Values are capped at `UINT32_MAX` (as libaom does, for
//! 32/64-bit consistency). Byte-identical to C libaom.

const MAX_LEB128_SIZE: usize = 8;
const LEB128_BYTE_MASK: u8 = 0x7f;
const MAX_LEB128_VALUE: u64 = u32::MAX as u64;

/// `aom_uleb_size_in_bytes`: number of bytes to encode `value` (>= 1).
pub fn uleb_size_in_bytes(value: u64) -> usize {
    let mut size = 0;
    let mut v = value;
    loop {
        size += 1;
        v >>= 7;
        if v == 0 {
            break;
        }
    }
    size
}

/// `aom_uleb_encode`: LEB128-encode `value` if it fits `available` bytes and the
/// 32-bit cap. Returns the coded bytes, or `None` on failure (matching C's `-1`).
pub fn uleb_encode(value: u64, available: usize) -> Option<Vec<u8>> {
    let leb_size = uleb_size_in_bytes(value);
    if value > MAX_LEB128_VALUE || leb_size > MAX_LEB128_SIZE || leb_size > available {
        return None;
    }
    let mut out = Vec::with_capacity(leb_size);
    let mut v = value;
    for _ in 0..leb_size {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80; // more bytes follow
        }
        out.push(byte);
    }
    Some(out)
}

/// `aom_uleb_decode`: decode a LEB128 value from `buffer`. Returns `(value,
/// bytes_consumed)`, or `None` on failure (overrun, or a value exceeding the
/// 32-bit cap) — matching C's `-1`.
pub fn uleb_decode(buffer: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for (i, &byte) in buffer.iter().take(MAX_LEB128_SIZE).enumerate() {
        let decoded = (byte & LEB128_BYTE_MASK) as u64;
        value |= decoded << (i * 7);
        if byte >> 7 == 0 {
            if value > MAX_LEB128_VALUE {
                return None;
            }
            return Some((value, i + 1));
        }
    }
    None
}
