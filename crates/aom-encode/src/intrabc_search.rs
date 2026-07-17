//! IntraBC (intra block copy) SEARCH machinery — chunk 3a of the
//! screen-content stills family: the source-frame hash table
//! (`av1/encoder/hash_motion.c` + the CRC-32C calculator, `hash.c`), the
//! block hash query, the hash candidate search (`av1_intrabc_hash_search`,
//! mcomp.c:1908), and the DV signalling cost table
//! (`av1_build_nmv_cost_table` / `av1_fill_dv_costs`, encodemv.c / rd.c:708).
//!
//! Faithfulness notes:
//! - CRC input serialization: the C feeds `uint32_t[4]` buffers to CRC32C as
//!   raw bytes (acknowledged arch-dependent, bug aomedia:433531610); the
//!   oracle build is x86-64 little-endian, so this port serializes with
//!   `to_le_bytes` — bit-identical to the reference behaviour.
//! - The hash table is bucket-keyed by `hash_value1` (block-size tag + CRC
//!   low 16 bits); buckets keep INSERTION order (the C `aom_vector`) capped
//!   at 256 entries — candidate iteration order is part of the tie-breaking
//!   semantics (`<` keeps the earliest best).
//! - `av1_get_crc32c_value` dispatches to a SSE4.2 hardware version in the
//!   reference build; CRC-32C is a bit-exact function of the input either
//!   way, so the table-driven software port is equivalent.
//!
//! Pixel planes are this port's u16-at-any-depth convention; the `bd > 8`
//! flag selects the C hbd hash arms (xor-fold) exactly as
//! `is_cur_buf_hbd(xd)` does.

use std::collections::HashMap;

/// `kSrcBits` (hash_motion.c).
const K_SRC_BITS: u32 = 16;
/// `kMaxCandidatesPerHashBucket`.
const K_MAX_CANDIDATES_PER_BUCKET: usize = 256;
/// `AOM_BUFFER_SIZE_FOR_BLOCK_HASH` (hash.h).
const BLOCK_HASH_BUF: usize = 4096;

// ---------------------------------------------------------------------------
// CRC-32C (hash.c — iSCSI polynomial, table-driven software version)
// ---------------------------------------------------------------------------

/// `CRC32C` + `av1_crc32c_calculator_init` (hash.c): the 8x256 slicing table.
pub struct Crc32c {
    table: [[u32; 256]; 8],
}

impl Crc32c {
    pub fn new() -> Self {
        const POLY: u32 = 0x82f63b78;
        let mut table = [[0u32; 256]; 8];
        for n in 0..256u32 {
            let mut crc = n;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ POLY
                } else {
                    crc >> 1
                };
            }
            table[0][n as usize] = crc;
        }
        for n in 0..256usize {
            let mut crc = table[0][n];
            for k in 1..8 {
                crc = table[0][(crc & 0xff) as usize] ^ (crc >> 8);
                table[k][n] = crc;
            }
        }
        Crc32c { table }
    }

    /// `av1_get_crc32c_value_c` (hash.c): little-endian 8-byte slicing.
    pub fn value(&self, buf: &[u8]) -> u32 {
        let t = &self.table;
        let mut crc: u64 = 0xffffffff;
        let mut i = 0usize;
        // The C aligns to 8; our buffers are 16-byte Vec-backed slices whose
        // base alignment Rust doesn't pin, but the RESULT is
        // alignment-independent (the prefix loop consumes bytes identically).
        while i < buf.len() && (buf.len() - i) >= 8 {
            let chunk = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap());
            crc ^= chunk;
            crc = u64::from(t[7][(crc & 0xff) as usize])
                ^ u64::from(t[6][((crc >> 8) & 0xff) as usize])
                ^ u64::from(t[5][((crc >> 16) & 0xff) as usize])
                ^ u64::from(t[4][((crc >> 24) & 0xff) as usize])
                ^ u64::from(t[3][((crc >> 32) & 0xff) as usize])
                ^ u64::from(t[2][((crc >> 40) & 0xff) as usize])
                ^ u64::from(t[1][((crc >> 48) & 0xff) as usize])
                ^ u64::from(t[0][(crc >> 56) as usize]);
            i += 8;
        }
        while i < buf.len() {
            crc = u64::from(t[0][((crc ^ u64::from(buf[i])) & 0xff) as usize]) ^ (crc >> 8);
            i += 1;
        }
        (crc as u32) ^ 0xffffffff
    }
}

impl Default for Crc32c {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// hash table build (hash_motion.c)
// ---------------------------------------------------------------------------

/// `block_hash` (hash_motion.h): a candidate block position + its full CRC.
#[derive(Clone, Copy, Debug)]
pub struct BlockHash {
    pub x: i16,
    pub y: i16,
    pub hash_value2: u32,
}

/// `hash_table` + `IntraBCHashInfo`: bucket map keyed by `hash_value1`
/// (`(crc & 0xffff) + (size_index << 16)`), insertion-ordered buckets capped
/// at 256 candidates.
pub struct IntrabcHashTable {
    pub buckets: HashMap<u32, Vec<BlockHash>>,
    pub crc: Crc32c,
}

/// `hash_block_size_to_index` (hash_motion.c).
fn hash_block_size_to_index(block_size: usize) -> i32 {
    match block_size {
        4 => 0,
        8 => 1,
        16 => 2,
        32 => 3,
        64 => 4,
        128 => 5,
        _ => -1,
    }
}

/// `get_identity_hash_value` (lbd 2x2 base hash).
#[inline]
fn identity_hash(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (u32::from(a) << 24) + (u32::from(b) << 16) + (u32::from(c) << 8) + u32::from(d)
}

/// `get_xor_hash_value_hbd` (hbd 2x2 base hash).
#[inline]
fn xor_hash_hbd(a: u16, b: u16, c: u16, d: u16) -> u32 {
    let mut result = (u32::from(a & 0x00ff) << 24)
        + (u32::from(b & 0x00ff) << 16)
        + (u32::from(c & 0x00ff) << 8)
        + u32::from(d & 0x00ff);
    result ^= (u32::from(a & 0xff00) << 16)
        + (u32::from(b & 0xff00) << 8)
        + u32::from(c & 0xff00)
        + (u32::from(d & 0xff00) >> 8);
    result
}

/// `av1_generate_block_2x2_hash_value` over this port's u16 planes: the 2x2
/// base layer, one hash per (x, y) with `x < w-1, y < h-1` (stored at
/// `y * width + x`; the +1 borders are never read downstream).
fn generate_block_2x2_hash(
    src: &[u16],
    off: usize,
    stride: usize,
    width: usize,
    height: usize,
    is_hbd: bool,
    out: &mut [u32],
) {
    let x_end = width - 1;
    let y_end = height - 1;
    for y in 0..y_end {
        for x in 0..x_end {
            let p = off + y * stride + x;
            let (a, b, c, d) = (src[p], src[p + 1], src[p + stride], src[p + stride + 1]);
            out[y * width + x] = if is_hbd {
                xor_hash_hbd(a, b, c, d)
            } else {
                identity_hash(a as u8, b as u8, c as u8, d as u8)
            };
        }
    }
}

/// `av1_generate_block_hash_value`: compose 4 half-size hashes into the
/// `block_size` layer via CRC-32C (LE-serialized `u32[4]`).
fn generate_block_hash_layer(
    crc: &Crc32c,
    width: usize,
    height: usize,
    block_size: usize,
    src_hash: &[u32],
    dst_hash: &mut [u32],
) {
    // C: x_end = width - block_size + 1 (int; negative -> the loops skip).
    let x_end = (width + 1).saturating_sub(block_size);
    let y_end = (height + 1).saturating_sub(block_size);
    let src_size = block_size >> 1;
    let mut bytes = [0u8; 16];
    for y in 0..y_end {
        for x in 0..x_end {
            let pos = y * width + x;
            let p = [
                src_hash[pos],
                src_hash[pos + src_size],
                src_hash[pos + src_size * width],
                src_hash[pos + src_size * width + src_size],
            ];
            for (i, v) in p.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            dst_hash[pos] = crc.value(&bytes);
        }
    }
}

/// `av1_add_to_hash_map_by_row_with_precal_data`: the hierarchical
/// (coarse-to-fine, no-two-adjacent) exploration that inserts every candidate
/// block position for one `block_size` layer.
fn add_to_hash_map_by_row(
    buckets: &mut HashMap<u32, Vec<BlockHash>>,
    pic_hash: &[u32],
    pic_width: usize,
    pic_height: usize,
    block_size: usize,
) {
    // C: int x_end/y_end go negative for layers larger than the frame.
    let x_end = (pic_width + 1).saturating_sub(block_size);
    let y_end = (pic_height + 1).saturating_sub(block_size);
    if x_end == 0 || y_end == 0 {
        return;
    }
    let add_value = (hash_block_size_to_index(block_size) as u32) << K_SRC_BITS;
    let crc_mask = (1u32 << K_SRC_BITS) - 1;
    let mut step = block_size;
    let mut x_offset = 0usize;
    let mut y_offset = 0usize;

    while step > 1 {
        let mut x_pos = x_offset;
        while x_pos < x_end {
            let mut y_pos = y_offset;
            while y_pos < y_end {
                let pos = y_pos * pic_width + x_pos;
                let hash_value1 = (pic_hash[pos] & crc_mask) + add_value;
                let bucket = buckets.entry(hash_value1).or_default();
                if bucket.len() < K_MAX_CANDIDATES_PER_BUCKET {
                    bucket.push(BlockHash {
                        x: x_pos as i16,
                        y: y_pos as i16,
                        hash_value2: pic_hash[pos],
                    });
                }
                y_pos += step;
            }
            x_pos += step;
        }
        // The offset/step state machine (hash_motion.c:318-338).
        if x_offset == 0 && y_offset == 0 {
            x_offset = step / 2;
        } else if x_offset == step / 2 && y_offset == 0 {
            x_offset = 0;
            y_offset = step / 2;
        } else if x_offset == 0 && y_offset == step / 2 {
            x_offset = step / 2;
        } else {
            debug_assert!(x_offset == step / 2 && y_offset == step / 2);
            step /= 2;
            x_offset = step / 2;
            y_offset = 0;
        }
    }
}

/// The `encodeframe.c:2199-2255` intrabc hash-table build: the 2x2 base layer,
/// then sizes 4..=`max_size` ping-pong composed and (from `min_alloc_size`=4,
/// `mi_alloc_bsize` BLOCK_4X4 in this envelope) inserted into the table.
/// `src` is the SOURCE luma plane (`cpi->source`), crop `width x height`.
#[allow(clippy::too_many_arguments)]
pub fn build_intrabc_hash_table(
    src: &[u16],
    off: usize,
    stride: usize,
    width: usize,
    height: usize,
    is_hbd: bool,
    sb_px: usize,
) -> IntrabcHashTable {
    let crc = Crc32c::new();
    let mut buckets = HashMap::new();
    let mut buf0 = vec![0u32; width * height];
    let mut buf1 = vec![0u32; width * height];

    generate_block_2x2_hash(src, off, stride, width, height, is_hbd, &mut buf0);
    let max_size = 64.min(sb_px);
    let min_alloc_size = 4usize; // block_size_wide[BLOCK_4X4]

    let mut size = 4usize;
    let mut src_is_0 = true;
    while size <= max_size {
        {
            let (s, d): (&[u32], &mut [u32]) = if src_is_0 {
                (&buf0, &mut buf1)
            } else {
                (&buf1, &mut buf0)
            };
            generate_block_hash_layer(&crc, width, height, size, s, d);
        }
        let d: &[u32] = if src_is_0 { &buf1 } else { &buf0 };
        if size >= min_alloc_size {
            add_to_hash_map_by_row(&mut buckets, d, width, height, size);
        }
        size *= 2;
        src_is_0 = !src_is_0;
    }
    IntrabcHashTable { buckets, crc }
}

/// `av1_get_block_hash_value`: the query block's `(hash_value1, hash_value2)`
/// — 2x2 base + iterated CRC composition over the block only.
pub fn get_block_hash_value(
    table: &IntrabcHashTable,
    src: &[u16],
    off: usize,
    stride: usize,
    block_size: usize,
    is_hbd: bool,
) -> (u32, u32) {
    let add_value = (hash_block_size_to_index(block_size) as u32) << K_SRC_BITS;
    let crc_mask = (1u32 << K_SRC_BITS) - 1;
    let mut buf = [vec![0u32; BLOCK_HASH_BUF], vec![0u32; BLOCK_HASH_BUF]];

    // 2x2 sub-block hashes.
    let mut sub_block_in_width = block_size >> 1;
    for y in (0..block_size).step_by(2) {
        for x in (0..block_size).step_by(2) {
            let pos = (y >> 1) * sub_block_in_width + (x >> 1);
            let p = off + y * stride + x;
            let (a, b, c, d) = (src[p], src[p + 1], src[p + stride], src[p + stride + 1]);
            buf[0][pos] = if is_hbd {
                xor_hash_hbd(a, b, c, d)
            } else {
                identity_hash(a as u8, b as u8, c as u8, d as u8)
            };
        }
    }

    let mut src_sub_block_in_width = sub_block_in_width;
    sub_block_in_width >>= 1;
    let mut src_idx = 0usize;
    let mut dst_idx = 1usize;
    let mut bytes = [0u8; 16];

    let mut sub_width = 4usize;
    while sub_width <= block_size {
        dst_idx = src_idx ^ 1;
        let (lo, hi) = buf.split_at_mut(1);
        let (s, d): (&[u32], &mut [u32]) = if src_idx == 0 {
            (&lo[0], &mut hi[0])
        } else {
            (&hi[0], &mut lo[0])
        };
        let mut dst_pos = 0usize;
        for y in 0..sub_block_in_width {
            for x in 0..sub_block_in_width {
                let src_pos = (y << 1) * src_sub_block_in_width + (x << 1);
                let p = [
                    s[src_pos],
                    s[src_pos + 1],
                    s[src_pos + src_sub_block_in_width],
                    s[src_pos + src_sub_block_in_width + 1],
                ];
                for (i, v) in p.iter().enumerate() {
                    bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                d[dst_pos] = table.crc.value(&bytes);
                dst_pos += 1;
            }
        }
        src_sub_block_in_width = sub_block_in_width;
        sub_block_in_width >>= 1;
        sub_width *= 2;
        src_idx ^= 1;
    }

    let h = buf[dst_idx][0];
    ((h & crc_mask) + add_value, h)
}

// ---------------------------------------------------------------------------
// DV signalling costs (av1_fill_dv_costs -> av1_build_nmv_cost_table)
// ---------------------------------------------------------------------------

/// `MV_MAX` / `MV_VALS` (entropymv.h): 14-bit magnitudes.
pub const MV_MAX: i32 = (1 << 14) - 1;
pub const MV_VALS: usize = (MV_MAX as usize) * 2 + 1;
const MV_CLASSES: usize = 11;
const MV_OFFSET_BITS: usize = 10;
const CLASS0_BITS: usize = 1;
const CLASS0_SIZE: usize = 1 << CLASS0_BITS;
const MV_FP_SIZE: usize = 4;

/// `IntraBCMVCosts` (block.h): the joint-type costs + the per-component
/// magnitude cost tables, index-centred at `MV_MAX` (`dv_costs[c][MV_MAX + v]`
/// = C's `dv_costs[c][v]` through the centred pointer).
pub struct DvCosts {
    pub joint_mv: [i32; 4],
    pub dv_costs: [Vec<i32>; 2],
}

/// `av1_cost_tokens_from_cdf` on a padded row (delegates to aom-txb's port).
fn cost_tokens(out: &mut [i32], cdf: &[u16]) {
    aom_txb::cost_tokens_from_cdf(out, cdf, None);
}

/// `av1_build_nmv_component_cost_table` (encodemv.c:124) at
/// `MV_SUBPEL_NONE` precision — the DV case (`av1_fill_dv_costs`). `mvcost`
/// is the centred view (`[MV_MAX + v]`). The component CDF blob layout is
/// the port's 69-u16 packing (aom-entropy partition.rs:452-461):
/// sign 0..3, classes 3..15, class0 15..18, bits[10] 18..48,
/// class0_fp[2] 48..58, fp 58..63, class0_hp 63..66, hp 66..69.
fn build_nmv_component_cost_table_none(mvcost: &mut [i32], comp_cdf: &[u16; 69]) {
    let mut sign_cost = [0i32; 2];
    let mut class_cost = [0i32; MV_CLASSES];
    let mut class0_cost = [0i32; CLASS0_SIZE];
    let mut bits_cost = [[0i32; 2]; MV_OFFSET_BITS];
    // MV_SUBPEL_NONE: fractional/hp cost arrays stay ZERO (the C `= { 0 }`
    // initializers; the precision gates skip their fills).
    let class0_fp_cost = [[0i32; MV_FP_SIZE]; CLASS0_SIZE];
    let fp_cost = [0i32; MV_FP_SIZE];
    let class0_hp_cost = [0i32; 2];
    let hp_cost = [0i32; 2];

    cost_tokens(&mut sign_cost, &comp_cdf[0..3]);
    cost_tokens(&mut class_cost, &comp_cdf[3..15]);
    cost_tokens(&mut class0_cost, &comp_cdf[15..18]);
    for i in 0..MV_OFFSET_BITS {
        cost_tokens(&mut bits_cost[i], &comp_cdf[18 + i * 3..18 + i * 3 + 3]);
    }

    let c = MV_MAX as usize; // centre index

    let mut cost_swap = [0i32; MV_OFFSET_BITS];
    let negate_sign = sign_cost[1] - sign_cost[0];
    for i in 1..MV_OFFSET_BITS {
        cost_swap[i] = bits_cost[i - 1][1];
        if i > CLASS0_BITS {
            cost_swap[i] -= class_cost[i - CLASS0_BITS];
        }
    }

    // Seed the fractional costs (fp/hp are zero at MV_SUBPEL_NONE, so this
    // seeds sign_cost[0] over the first 2*MV_FP_SIZE magnitudes).
    for o in 0..MV_FP_SIZE {
        for hp in 0..2usize {
            let v = 2 * o + hp + 1;
            mvcost[c + v] = fp_cost[o] + hp_cost[hp] + sign_cost[0];
        }
    }

    mvcost[c] = 0;
    // Per-exponent fill from the previous exponents.
    for i in 0..MV_OFFSET_BITS {
        let exponent = (2 * MV_FP_SIZE) << i;
        let class = if i >= CLASS0_BITS {
            class_cost[i - CLASS0_BITS + 1]
        } else {
            0
        };
        let mut mantissa = 0usize;
        for j in 0..=i {
            while mantissa < ((2 * MV_FP_SIZE) << j) {
                let cost = mvcost[c + mantissa + 1] + class + cost_swap[j];
                let v = exponent + mantissa + 1;
                mvcost[c + v] = cost;
                mvcost[c - v] = cost + negate_sign;
                mantissa += 1;
            }
            cost_swap[j] += bits_cost[i][0];
        }
    }

    // The last-exponent special case (buffer-overrun guard in C).
    {
        let exponent = (2 * MV_FP_SIZE) << MV_OFFSET_BITS;
        let class = class_cost[MV_CLASSES - 1];
        let mut mantissa = 0usize;
        for j in 0..MV_OFFSET_BITS {
            while mantissa < ((2 * MV_FP_SIZE) << j) {
                let cost = mvcost[c + mantissa + 1] + class + cost_swap[j];
                let v = exponent + mantissa + 1;
                mvcost[c + v] = cost;
                mvcost[c - v] = cost + negate_sign;
                mantissa += 1;
            }
        }
        let cost_swap_hi = bits_cost[MV_OFFSET_BITS - 1][1] - class_cost[MV_CLASSES - 2];
        while mantissa < exponent - 1 {
            let cost = mvcost[c + mantissa + 1] + class + cost_swap_hi;
            let v = exponent + mantissa + 1;
            mvcost[c + v] = cost;
            mvcost[c - v] = cost + negate_sign;
            mantissa += 1;
        }
    }

    // Class-0 vectors overwrite the placeholders.
    for i in 0..CLASS0_SIZE {
        let top = i * 2 * MV_FP_SIZE;
        for o in 0..MV_FP_SIZE {
            let cost = class0_fp_cost[i][o] + class_cost[0] + class0_cost[i];
            for hp in 0..2usize {
                let v = top + 2 * o + hp + 1;
                mvcost[c + v] = cost + class0_hp_cost[hp] + sign_cost[0];
                let neg = cost + class0_hp_cost[hp] + sign_cost[1];
                mvcost[c - v] = neg;
            }
        }
    }
}

/// `av1_fill_dv_costs` (rd.c:708): joint costs + both component tables from
/// the live ndvc CDFs, at `MV_SUBPEL_NONE`.
pub fn fill_dv_costs(
    ndvc_joints: &[u16],
    ndvc_comp0: &[u16; 69],
    ndvc_comp1: &[u16; 69],
) -> DvCosts {
    let mut joint_mv = [0i32; 4];
    cost_tokens(&mut joint_mv, ndvc_joints);
    let mut c0 = vec![0i32; MV_VALS];
    let mut c1 = vec![0i32; MV_VALS];
    build_nmv_component_cost_table_none(&mut c0, ndvc_comp0);
    build_nmv_component_cost_table_none(&mut c1, ndvc_comp1);
    DvCosts {
        joint_mv,
        dv_costs: [c0, c1],
    }
}

/// `mv_cost` (mcomp.c:329): `joint + comp0[row] + comp1[col]` on the centred
/// tables.
#[inline]
pub fn mv_cost(diff_row: i32, diff_col: i32, dv: &DvCosts) -> i32 {
    let joint = aom_entropy::partition::get_mv_joint(diff_row, diff_col) as usize;
    dv.joint_mv[joint]
        + dv.dv_costs[0][(MV_MAX + diff_row) as usize]
        + dv.dv_costs[1][(MV_MAX + diff_col) as usize]
}

/// `av1_mv_bit_cost` (mcomp.c:334): the RD-search DV rate —
/// `ROUND_POWER_OF_TWO(mv_cost(diff * 8) * weight, 7)` with
/// `MV_COST_WEIGHT_SUB` = 120 at the intrabc call site (rdopt.c:3606).
#[inline]
pub fn mv_bit_cost_sub(dv_row: i32, dv_col: i32, ref_row: i32, ref_col: i32, dv: &DvCosts) -> i32 {
    const MV_COST_WEIGHT_SUB: i64 = 120;
    let c = i64::from(mv_cost(dv_row - ref_row, dv_col - ref_col, dv));
    ((c * MV_COST_WEIGHT_SUB + 64) >> 7) as i32
}

/// `mv_err_cost` (mcomp.c:341), `MV_COST_ENTROPY` arm: the full-pel search's
/// variance-metric MV cost. `error_per_bit` = `AOMMAX(rdmult >> 6, 1)`
/// (av1_set_error_per_bit); the shift is `RDDIV_BITS(7) +
/// AV1_PROB_COST_SHIFT(9) - RD_EPB_SHIFT(6) + PIXEL_TRANSFORM_ERROR_SCALE(4)`
/// = 14, 64-bit rounded.
#[inline]
pub fn mv_err_cost(
    diff_row_subpel: i32,
    diff_col_subpel: i32,
    dv: &DvCosts,
    error_per_bit: i32,
) -> i32 {
    let c = i64::from(mv_cost(diff_row_subpel, diff_col_subpel, dv)) * i64::from(error_per_bit);
    ((c + (1 << 13)) >> 14) as i32
}

/// `mvsad_err_cost` (mcomp.c:372), `MV_COST_ENTROPY` arm: the SAD-metric MV
/// cost over FULL-PEL diffs promoted to subpel (`GET_MV_SUBPEL` = `*8`);
/// `sad_per_bit` from `av1_set_sad_per_bit`.
#[inline]
pub fn mvsad_err_cost(
    full_diff_row: i32,
    full_diff_col: i32,
    dv: &DvCosts,
    sad_per_bit: i32,
) -> i32 {
    let c = u64::from(mv_cost(full_diff_row * 8, full_diff_col * 8, dv) as u32)
        * u64::from(sad_per_bit as u32);
    ((c + (1 << 8)) >> 9) as i32
}

// ---------------------------------------------------------------------------
// pixel metrics (aom_dsp variance/sad over u16 planes)
// ---------------------------------------------------------------------------

/// `aom_variance{w}x{h}` (variance.c): returns the variance
/// (`sse - sum^2 / (w*h)`) — the hash search's `get_mvpred_var_cost` metric.
/// Arithmetic on u16 pixel values matches C's lowbd path at bd 8 and the
/// highbd path above (both accumulate the same integers).
pub fn variance_wxh(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut sum: i64 = 0;
    let mut sse: u64 = 0;
    for r in 0..h {
        for c in 0..w {
            let d = i64::from(src[src_off + r * src_stride + c])
                - i64::from(refb[ref_off + r * ref_stride + c]);
            sum += d;
            sse += (d * d) as u64;
        }
    }
    (sse - ((sum * sum) as u64) / (w as u64 * h as u64)) as u32
}

/// `aom_sad{w}x{h}`: the diamond search's SAD metric.
pub fn sad_wxh(
    src: &[u16],
    src_off: usize,
    src_stride: usize,
    refb: &[u16],
    ref_off: usize,
    ref_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut sad: u64 = 0;
    for r in 0..h {
        for c in 0..w {
            let d = i64::from(src[src_off + r * src_stride + c])
                - i64::from(refb[ref_off + r * ref_stride + c]);
            sad += d.unsigned_abs();
        }
    }
    sad as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CRC-32C known-answer test (iSCSI polynomial): "123456789" ->
    /// 0xE3069283 (the canonical check value).
    #[test]
    fn crc32c_known_answer() {
        let crc = Crc32c::new();
        assert_eq!(crc.value(b"123456789"), 0xE3069283);
        // 16-byte (u32[4]) shape the hash layers feed.
        assert_eq!(crc.value(&[0u8; 16]), crc.value(&[0u8; 16]));
        assert_ne!(crc.value(&[0u8; 16]), crc.value(&[1u8; 16]));
    }

    /// The hierarchical exploration must visit every candidate exactly once
    /// (the C doc-comment's 8x8/block-4 example: 25 candidates).
    #[test]
    fn exploration_visits_all_once() {
        let mut buckets: HashMap<u32, Vec<BlockHash>> = HashMap::new();
        // 8x8 picture, block 4 -> x_end = y_end = 5 -> 25 candidates. Use a
        // constant hash so they all land in one bucket.
        let pic_hash = vec![7u32; 8 * 8];
        add_to_hash_map_by_row(&mut buckets, &pic_hash, 8, 8, 4);
        let bucket = buckets.values().next().unwrap();
        assert_eq!(buckets.len(), 1);
        assert_eq!(bucket.len(), 25);
        let mut seen = std::collections::HashSet::new();
        for b in bucket {
            assert!(seen.insert((b.x, b.y)), "duplicate visit ({},{})", b.x, b.y);
            assert!(b.x <= 4 && b.y <= 4);
        }
    }

    /// Identical source blocks must produce identical (hash1, hash2) through
    /// both the table build and the query path (the match invariant the
    /// search relies on).
    #[test]
    fn build_and_query_hashes_agree() {
        // 32x32 source with an exact 8x8 repeat at (0,0) and (16, 8).
        let w = 32usize;
        let mut src = vec![0u16; w * w];
        for y in 0..8 {
            for x in 0..8 {
                let v = (y * 8 + x) as u16;
                src[y * w + x] = v;
                src[(y + 8) * w + (x + 16)] = v;
            }
        }
        let table = build_intrabc_hash_table(&src, 0, w, w, w, false, 64);
        let (h1a, h2a) = get_block_hash_value(&table, &src, 0, w, 8, false);
        let (h1b, h2b) = get_block_hash_value(&table, &src, 8 * w + 16, w, 8, false);
        assert_eq!((h1a, h2a), (h1b, h2b), "identical blocks must hash equal");
        // And the table must contain candidates for that bucket, including
        // both repeat positions.
        let bucket = table.buckets.get(&h1a).expect("bucket exists");
        let mut found = [false; 2];
        for b in bucket.iter().filter(|b| b.hash_value2 == h2a) {
            if (b.x, b.y) == (0, 0) {
                found[0] = true;
            }
            if (b.x, b.y) == (16, 8) {
                found[1] = true;
            }
        }
        assert!(found[0] && found[1], "both repeat positions in the table");
    }
}
