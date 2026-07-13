//! `aom_read_bit_buffer` (libaom `aom_dsp/bitreader_buffer.c`): the byte-aligned,
//! MSB-first bit reader used for the uncompressed headers (sequence / frame / tile
//! group / OBU) — the exact inverse of [`crate::wb::WriteBitBuffer`]. Distinct from
//! the `od_ec` arithmetic coder used for coefficients. Reading past the end returns
//! `0` and sets `error` (mirroring libaom's error handler).

/// A borrowed MSB-first bit reader.
#[derive(Clone, Debug)]
pub struct ReadBitBuffer<'a> {
    buf: &'a [u8],
    bit_offset: usize,
    /// Set once a read runs past the end of the buffer.
    pub error: bool,
}

impl<'a> ReadBitBuffer<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_offset: 0, error: false }
    }

    /// `aom_rb_read_bit`: one bit at the current MSB-first position.
    pub fn read_bit(&mut self) -> u32 {
        let off = self.bit_offset;
        let p = off >> 3;
        if p < self.buf.len() {
            let q = 7 - (off & 7);
            self.bit_offset = off + 1;
            ((self.buf[p] >> q) & 1) as u32
        } else {
            self.error = true;
            0
        }
    }

    /// `aom_rb_read_literal`: `bits` MSB-first bits into a signed int (`bits <= 31`).
    pub fn read_literal(&mut self, bits: u32) -> i32 {
        let mut v = 0i32;
        for _ in 0..bits {
            v = (v << 1) | self.read_bit() as i32;
        }
        v
    }

    /// `aom_rb_read_unsigned_literal`: `bits` MSB-first bits into a u32 (`bits <= 32`).
    pub fn read_unsigned_literal(&mut self, bits: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..bits {
            v = (v << 1) | self.read_bit();
        }
        v
    }

    /// `aom_rb_read_inv_signed_literal`: inverse of
    /// [`crate::wb::WriteBitBuffer::write_inv_signed_literal`] — read `bits+1` bits and
    /// sign-extend from bit `bits`.
    pub fn read_inv_signed_literal(&mut self, bits: u32) -> i32 {
        let nbits = 32 - bits - 1;
        let value = self.read_unsigned_literal(bits + 1) << nbits;
        (value as i32) >> nbits
    }

    /// `aom_rb_read_uvlc`: inverse of [`crate::wb::WriteBitBuffer::write_uvlc`] — the
    /// Exp-Golomb unsigned variable-length code. Returns `u32::MAX` on 32+ leading zeros.
    pub fn read_uvlc(&mut self) -> u32 {
        let mut leading_zeros = 0u32;
        while self.read_bit() == 0 {
            leading_zeros += 1;
            if leading_zeros >= 32 {
                return u32::MAX;
            }
        }
        let base = (1u32 << leading_zeros) - 1;
        base + self.read_unsigned_literal(leading_zeros)
    }

    /// Current bit position.
    pub fn bit_position(&self) -> usize {
        self.bit_offset
    }

    /// `aom_rb_bytes_read` (rounds up to whole bytes).
    pub fn bytes_read(&self) -> usize {
        self.bit_offset.div_ceil(8)
    }

    /// Whether the read cursor sits on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_offset.is_multiple_of(8)
    }

    /// Advance to the next byte boundary (consumes the padding/stop bits).
    pub fn byte_align(&mut self) {
        while !self.is_byte_aligned() {
            self.read_bit();
        }
    }
}
