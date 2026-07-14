/* Shim over has_top_right / has_bottom_left (av1/common/reconintra.c). Verbatim
 * paste of the static tables + functions (renamed `sh_`), exposed as two
 * non-static entry points so the Rust port can be diffed against the real C
 * logic. Size tables (mi_size_wide etc.) come from the libaom headers, so this is
 * the pristine reference recompiled, not a re-derivation. */
#include <stdint.h>
#include <assert.h>
#include "av1/common/av1_common_int.h"

static uint8_t sh_has_tr_4x4[128] = {
  255, 255, 255, 255, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  255, 127, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  255, 255, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  255, 127, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
  127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
};
static uint8_t sh_has_tr_4x8[64] = {
  255, 255, 255, 255, 119, 119, 119, 119, 127, 127, 127, 127, 119,
  119, 119, 119, 255, 127, 255, 127, 119, 119, 119, 119, 127, 127,
  127, 127, 119, 119, 119, 119, 255, 255, 255, 127, 119, 119, 119,
  119, 127, 127, 127, 127, 119, 119, 119, 119, 255, 127, 255, 127,
  119, 119, 119, 119, 127, 127, 127, 127, 119, 119, 119, 119,
};
static uint8_t sh_has_tr_8x4[64] = {
  255, 255, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
  127, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
  255, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
  127, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
};
static uint8_t sh_has_tr_8x8[32] = {
  255, 255, 85, 85, 119, 119, 85, 85, 127, 127, 85, 85, 119, 119, 85, 85,
  255, 127, 85, 85, 119, 119, 85, 85, 127, 127, 85, 85, 119, 119, 85, 85,
};
static uint8_t sh_has_tr_8x16[16] = {
  255, 255, 119, 119, 127, 127, 119, 119,
  255, 127, 119, 119, 127, 127, 119, 119,
};
static uint8_t sh_has_tr_16x8[16] = {
  255, 0, 85, 0, 119, 0, 85, 0, 127, 0, 85, 0, 119, 0, 85, 0,
};
static uint8_t sh_has_tr_16x16[8] = {
  255, 85, 119, 85, 127, 85, 119, 85,
};
static uint8_t sh_has_tr_16x32[4] = { 255, 119, 127, 119 };
static uint8_t sh_has_tr_32x16[4] = { 15, 5, 7, 5 };
static uint8_t sh_has_tr_32x32[2] = { 95, 87 };
static uint8_t sh_has_tr_32x64[1] = { 127 };
static uint8_t sh_has_tr_64x32[1] = { 19 };
static uint8_t sh_has_tr_64x64[1] = { 7 };
static uint8_t sh_has_tr_64x128[1] = { 3 };
static uint8_t sh_has_tr_128x64[1] = { 1 };
static uint8_t sh_has_tr_128x128[1] = { 1 };
static uint8_t sh_has_tr_4x16[32] = {
  255, 255, 255, 255, 127, 127, 127, 127, 255, 127, 255,
  127, 127, 127, 127, 127, 255, 255, 255, 127, 127, 127,
  127, 127, 255, 127, 255, 127, 127, 127, 127, 127,
};
static uint8_t sh_has_tr_16x4[32] = {
  255, 0, 0, 0, 85, 0, 0, 0, 119, 0, 0, 0, 85, 0, 0, 0,
  127, 0, 0, 0, 85, 0, 0, 0, 119, 0, 0, 0, 85, 0, 0, 0,
};
static uint8_t sh_has_tr_8x32[8] = {
  255, 255, 127, 127, 255, 127, 127, 127,
};
static uint8_t sh_has_tr_32x8[8] = {
  15, 0, 5, 0, 7, 0, 5, 0,
};
static uint8_t sh_has_tr_16x64[2] = { 255, 127 };
static uint8_t sh_has_tr_64x16[2] = { 3, 1 };

static const uint8_t *const sh_has_tr_tables[BLOCK_SIZES_ALL] = {
  sh_has_tr_4x4, sh_has_tr_4x8, sh_has_tr_8x4, sh_has_tr_8x8, sh_has_tr_8x16,
  sh_has_tr_16x8, sh_has_tr_16x16, sh_has_tr_16x32, sh_has_tr_32x16,
  sh_has_tr_32x32, sh_has_tr_32x64, sh_has_tr_64x32, sh_has_tr_64x64,
  sh_has_tr_64x128, sh_has_tr_128x64, sh_has_tr_128x128, sh_has_tr_4x16,
  sh_has_tr_16x4, sh_has_tr_8x32, sh_has_tr_32x8, sh_has_tr_16x64, sh_has_tr_64x16
};
static uint8_t sh_has_tr_vert_8x8[32] = {
  255, 255, 0, 0, 119, 119, 0, 0, 127, 127, 0, 0, 119, 119, 0, 0,
  255, 127, 0, 0, 119, 119, 0, 0, 127, 127, 0, 0, 119, 119, 0, 0,
};
static uint8_t sh_has_tr_vert_16x16[8] = {
  255, 0, 119, 0, 127, 0, 119, 0,
};
static uint8_t sh_has_tr_vert_32x32[2] = { 15, 7 };
static uint8_t sh_has_tr_vert_64x64[1] = { 3 };
static const uint8_t *const sh_has_tr_vert_tables[BLOCK_SIZES] = {
  NULL, sh_has_tr_4x8, NULL, sh_has_tr_vert_8x8, sh_has_tr_8x16, NULL,
  sh_has_tr_vert_16x16, sh_has_tr_16x32, NULL, sh_has_tr_vert_32x32,
  sh_has_tr_32x64, NULL, sh_has_tr_vert_64x64, sh_has_tr_64x128, NULL,
  sh_has_tr_128x128
};
static const uint8_t *sh_get_has_tr_table(PARTITION_TYPE partition, BLOCK_SIZE bsize) {
  if (partition == PARTITION_VERT_A || partition == PARTITION_VERT_B)
    return sh_has_tr_vert_tables[bsize];
  return sh_has_tr_tables[bsize];
}

int shim_has_top_right(int sb_size, int bsize, int mi_row, int mi_col,
                       int top_available, int right_available, int partition,
                       int txsz, int row_off, int col_off, int ss_x, int ss_y) {
  if (!top_available || !right_available) return 0;
  const int bw_unit = mi_size_wide[bsize];
  const int plane_bw_unit = AOMMAX(bw_unit >> ss_x, 1);
  const int top_right_count_unit = tx_size_wide_unit[txsz];
  if (row_off > 0) {
    if (block_size_wide[bsize] > block_size_wide[BLOCK_64X64]) {
      if (row_off == mi_size_high[BLOCK_64X64] >> ss_y &&
          col_off + top_right_count_unit == mi_size_wide[BLOCK_64X64] >> ss_x)
        return 1;
      const int plane_bw_unit_64 = mi_size_wide[BLOCK_64X64] >> ss_x;
      const int col_off_64 = col_off % plane_bw_unit_64;
      return col_off_64 + top_right_count_unit < plane_bw_unit_64;
    }
    return col_off + top_right_count_unit < plane_bw_unit;
  } else {
    if (col_off + top_right_count_unit < plane_bw_unit) return 1;
    const int bw_in_mi_log2 = mi_size_wide_log2[bsize];
    const int bh_in_mi_log2 = mi_size_high_log2[bsize];
    const int sb_mi_size = mi_size_high[sb_size];
    const int blk_row_in_sb = (mi_row & (sb_mi_size - 1)) >> bh_in_mi_log2;
    const int blk_col_in_sb = (mi_col & (sb_mi_size - 1)) >> bw_in_mi_log2;
    if (blk_row_in_sb == 0) return 1;
    if (((blk_col_in_sb + 1) << bw_in_mi_log2) >= sb_mi_size) return 0;
    const int this_blk_index =
        ((blk_row_in_sb + 0) << (MAX_MIB_SIZE_LOG2 - bw_in_mi_log2)) + blk_col_in_sb + 0;
    const int idx1 = this_blk_index / 8;
    const int idx2 = this_blk_index % 8;
    const uint8_t *has_tr_table = sh_get_has_tr_table(partition, bsize);
    return (has_tr_table[idx1] >> idx2) & 1;
  }
}

static uint8_t sh_has_bl_4x4[128] = {
  84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0,  1,  1,  1,  84, 85, 85,
  85, 16, 17, 17, 17, 84, 85, 85, 85, 0,  0,  1,  0,  84, 85, 85, 85, 16, 17,
  17, 17, 84, 85, 85, 85, 0,  1,  1,  1,  84, 85, 85, 85, 16, 17, 17, 17, 84,
  85, 85, 85, 0,  0,  0,  0,  84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85,
  0,  1,  1,  1,  84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0,  0,  1,
  0,  84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0,  1,  1,  1,  84, 85,
  85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0,  0,  0,  0,
};
static uint8_t sh_has_bl_4x8[64] = {
  16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 1, 0,
  16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 0, 0,
  16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 1, 0,
  16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 0, 0,
};
static uint8_t sh_has_bl_8x4[64] = {
  254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 1,
  254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 0,
  254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 1,
  254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 0,
};
static uint8_t sh_has_bl_8x8[32] = {
  84, 85, 16, 17, 84, 85, 0, 1, 84, 85, 16, 17, 84, 85, 0, 0,
  84, 85, 16, 17, 84, 85, 0, 1, 84, 85, 16, 17, 84, 85, 0, 0,
};
static uint8_t sh_has_bl_8x16[16] = {
  16, 17, 0, 1, 16, 17, 0, 0, 16, 17, 0, 1, 16, 17, 0, 0,
};
static uint8_t sh_has_bl_16x8[16] = {
  254, 84, 254, 16, 254, 84, 254, 0, 254, 84, 254, 16, 254, 84, 254, 0,
};
static uint8_t sh_has_bl_16x16[8] = {
  84, 16, 84, 0, 84, 16, 84, 0,
};
static uint8_t sh_has_bl_16x32[4] = { 16, 0, 16, 0 };
static uint8_t sh_has_bl_32x16[4] = { 78, 14, 78, 14 };
static uint8_t sh_has_bl_32x32[2] = { 4, 4 };
static uint8_t sh_has_bl_32x64[1] = { 0 };
static uint8_t sh_has_bl_64x32[1] = { 34 };
static uint8_t sh_has_bl_64x64[1] = { 0 };
static uint8_t sh_has_bl_64x128[1] = { 0 };
static uint8_t sh_has_bl_128x64[1] = { 0 };
static uint8_t sh_has_bl_128x128[1] = { 0 };
static uint8_t sh_has_bl_4x16[32] = {
  0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
  0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
};
static uint8_t sh_has_bl_16x4[32] = {
  254, 254, 254, 84, 254, 254, 254, 16, 254, 254, 254, 84, 254, 254, 254, 0,
  254, 254, 254, 84, 254, 254, 254, 16, 254, 254, 254, 84, 254, 254, 254, 0,
};
static uint8_t sh_has_bl_8x32[8] = {
  0, 1, 0, 0, 0, 1, 0, 0,
};
static uint8_t sh_has_bl_32x8[8] = {
  238, 78, 238, 14, 238, 78, 238, 14,
};
static uint8_t sh_has_bl_16x64[2] = { 0, 0 };
static uint8_t sh_has_bl_64x16[2] = { 42, 42 };
static const uint8_t *const sh_has_bl_tables[BLOCK_SIZES_ALL] = {
  sh_has_bl_4x4, sh_has_bl_4x8, sh_has_bl_8x4, sh_has_bl_8x8, sh_has_bl_8x16,
  sh_has_bl_16x8, sh_has_bl_16x16, sh_has_bl_16x32, sh_has_bl_32x16,
  sh_has_bl_32x32, sh_has_bl_32x64, sh_has_bl_64x32, sh_has_bl_64x64,
  sh_has_bl_64x128, sh_has_bl_128x64, sh_has_bl_128x128, sh_has_bl_4x16,
  sh_has_bl_16x4, sh_has_bl_8x32, sh_has_bl_32x8, sh_has_bl_16x64, sh_has_bl_64x16
};
static uint8_t sh_has_bl_vert_8x8[32] = {
  254, 255, 16, 17, 254, 255, 0, 1, 254, 255, 16, 17, 254, 255, 0, 0,
  254, 255, 16, 17, 254, 255, 0, 1, 254, 255, 16, 17, 254, 255, 0, 0,
};
static uint8_t sh_has_bl_vert_16x16[8] = {
  254, 16, 254, 0, 254, 16, 254, 0,
};
static uint8_t sh_has_bl_vert_32x32[2] = { 14, 14 };
static uint8_t sh_has_bl_vert_64x64[1] = { 2 };
static const uint8_t *const sh_has_bl_vert_tables[BLOCK_SIZES] = {
  NULL, sh_has_bl_4x8, NULL, sh_has_bl_vert_8x8, sh_has_bl_8x16, NULL,
  sh_has_bl_vert_16x16, sh_has_bl_16x32, NULL, sh_has_bl_vert_32x32,
  sh_has_bl_32x64, NULL, sh_has_bl_vert_64x64, sh_has_bl_64x128, NULL,
  sh_has_bl_128x128
};
static const uint8_t *sh_get_has_bl_table(PARTITION_TYPE partition, BLOCK_SIZE bsize) {
  if (partition == PARTITION_VERT_A || partition == PARTITION_VERT_B)
    return sh_has_bl_vert_tables[bsize];
  return sh_has_bl_tables[bsize];
}

int shim_has_bottom_left(int sb_size, int bsize, int mi_row, int mi_col,
                         int bottom_available, int left_available, int partition,
                         int txsz, int row_off, int col_off, int ss_x, int ss_y) {
  if (!bottom_available || !left_available) return 0;
  if (block_size_wide[bsize] > block_size_wide[BLOCK_64X64] && col_off > 0) {
    const int plane_bw_unit_64 = mi_size_wide[BLOCK_64X64] >> ss_x;
    const int col_off_64 = col_off % plane_bw_unit_64;
    if (col_off_64 == 0) {
      const int plane_bh_unit_64 = mi_size_high[BLOCK_64X64] >> ss_y;
      const int row_off_64 = row_off % plane_bh_unit_64;
      const int plane_bh_unit = AOMMIN(mi_size_high[bsize] >> ss_y, plane_bh_unit_64);
      return row_off_64 + tx_size_high_unit[txsz] < plane_bh_unit;
    }
  }
  if (col_off > 0) {
    return 0;
  } else {
    const int bh_unit = mi_size_high[bsize];
    const int plane_bh_unit = AOMMAX(bh_unit >> ss_y, 1);
    const int bottom_left_count_unit = tx_size_high_unit[txsz];
    if (row_off + bottom_left_count_unit < plane_bh_unit) return 1;
    const int bw_in_mi_log2 = mi_size_wide_log2[bsize];
    const int bh_in_mi_log2 = mi_size_high_log2[bsize];
    const int sb_mi_size = mi_size_high[sb_size];
    const int blk_row_in_sb = (mi_row & (sb_mi_size - 1)) >> bh_in_mi_log2;
    const int blk_col_in_sb = (mi_col & (sb_mi_size - 1)) >> bw_in_mi_log2;
    if (blk_col_in_sb == 0) {
      const int blk_start_row_off =
          blk_row_in_sb << (bh_in_mi_log2 + MI_SIZE_LOG2 - MI_SIZE_LOG2) >> ss_y;
      const int row_off_in_sb = blk_start_row_off + row_off;
      const int sb_height_unit = sb_mi_size >> ss_y;
      return row_off_in_sb + bottom_left_count_unit < sb_height_unit;
    }
    if (((blk_row_in_sb + 1) << bh_in_mi_log2) >= sb_mi_size) return 0;
    const int this_blk_index =
        ((blk_row_in_sb + 0) << (MAX_MIB_SIZE_LOG2 - bw_in_mi_log2)) + blk_col_in_sb + 0;
    const int idx1 = this_blk_index / 8;
    const int idx2 = this_blk_index % 8;
    const uint8_t *has_bl_table = sh_get_has_bl_table(partition, bsize);
    return (has_bl_table[idx1] >> idx2) & 1;
  }
}
