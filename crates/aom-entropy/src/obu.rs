//! OBU header packing (libaom `av1/encoder/bitstream.c` `av1_write_obu_header`):
//! the 1-2 byte header that prefixes every Open Bitstream Unit. Byte-identical to
//! C libaom. (The level-stats side effect in the C function does not affect the
//! bytes and is out of scope here.)

/// `av1_write_obu_header` (byte output): `obu_type` in bits 6..3, the extension
/// flag in bit 2 (= `has_nonzero_operating_point_idc && is_layer_specific_obu`),
/// `obu_has_size_field` (always 1) in bit 1; then the extension byte if flagged.
pub fn write_obu_header(
    obu_type: u32,
    has_nonzero_operating_point_idc: bool,
    is_layer_specific_obu: bool,
    obu_extension: u8,
) -> Vec<u8> {
    let obu_extension_flag = has_nonzero_operating_point_idc && is_layer_specific_obu;
    let obu_has_size_field = 1u32;
    let mut out = Vec::with_capacity(2);
    out.push(((obu_type << 3) | ((obu_extension_flag as u32) << 2) | (obu_has_size_field << 1)) as u8);
    if obu_extension_flag {
        out.push(obu_extension);
    }
    out
}
