//! The RD mode-threshold machinery — the `THR_MODES` mode ordering, the
//! baseline `thresh_mult` table, and the adaptive per-block-size factor update.
//!
//! This is the seed data the inter RD brain (`av1_rd_pick_inter_mode_sb`) ranks
//! and prunes modes with. The port had none of it, and none of it exists in a
//! form that can be checked by reading: `THR_MODES` is a 169-entry enum whose
//! ORDER is load-bearing, and `av1_set_rd_speed_thresholds` is 169 hand-written
//! assignments keyed by that enum.
//!
//! | Rust | C |
//! |---|---|
//! | [`THR_NEARESTMV`] .. [`THR_D45_PRED`], [`MAX_MODES`] | the `THR_MODES` enum (`av1/encoder/enc_enums.h:24`) |
//! | [`set_rd_speed_thresholds`] | `av1_set_rd_speed_thresholds` (`av1/encoder/rd.c`) |
//! | [`update_rd_thresh_fact`] | `av1_update_rd_thresh_fact` (rd.c:1468) |
//! | [`update_thr_fact`] | `update_thr_fact` (rd.c:1451, static — gated through the above) |
//!
//! # How this was transcribed, and why that is safe
//! The enum ordering and the 169 table entries were extracted from the C source
//! mechanically rather than typed, because a hand transcription of 169 indexed
//! constants is exactly the input a reviewer cannot check. What makes the
//! result trustworthy is not the extraction: it is that
//! `tests/rd_thresh_diff.rs` compares the WHOLE `thresh_mult` array against the
//! array the real `av1_set_rd_speed_thresholds` writes. A wrong enum index
//! puts a right value in a wrong slot, so one array comparison covers the
//! ordering and the constants together.
//!
//! # Differential coverage
//! `tests/rd_thresh_diff.rs`, tier 1 against the real exported C.

/// `MAX_MODES` (`av1/encoder/enc_enums.h`) — the number of RD mode slots.
pub const MAX_MODES: usize = 169;

/// `BLOCK_SIZES_ALL` (`av1/common/enums.h:122`).
pub const BLOCK_SIZES_ALL: usize = 22;

/// `RD_THRESH_FAC_FRAC_BITS` (`av1/encoder/rd.h:50`).
const RD_THRESH_FAC_FRAC_BITS: i32 = 5;
/// `RD_THRESH_FAC_FRAC_VAL` (rd.h:51).
const RD_THRESH_FAC_FRAC_VAL: i32 = 1 << RD_THRESH_FAC_FRAC_BITS;
/// `RD_THRESH_MAX_FACT` (rd.h:53).
pub const RD_THRESH_MAX_FACT: i32 = RD_THRESH_FAC_FRAC_VAL << 1;
/// `RD_THRESH_LOG_DEC_FACTOR` (rd.h:54).
const RD_THRESH_LOG_DEC_FACTOR: i32 = 4;
/// `RD_THRESH_INC` (rd.h:55).
const RD_THRESH_INC: i32 = 1;

/// `BLOCK_4X4` (`av1/common/enums.h`).
const BLOCK_4X4: i32 = 0;

// ===================================================================
// The THR_MODES ordering (enc_enums.h). The ORDER is the contract; the
// numbers are a consequence of it.
// ===================================================================

/// `THR_NEARESTMV` = 0.
pub const THR_NEARESTMV: usize = 0;
/// `THR_NEARESTL2` = 1.
pub const THR_NEARESTL2: usize = 1;
/// `THR_NEARESTL3` = 2.
pub const THR_NEARESTL3: usize = 2;
/// `THR_NEARESTB` = 3.
pub const THR_NEARESTB: usize = 3;
/// `THR_NEARESTA2` = 4.
pub const THR_NEARESTA2: usize = 4;
/// `THR_NEARESTA` = 5.
pub const THR_NEARESTA: usize = 5;
/// `THR_NEARESTG` = 6.
pub const THR_NEARESTG: usize = 6;
/// `THR_NEWMV` = 7.
pub const THR_NEWMV: usize = 7;
/// `THR_NEWL2` = 8.
pub const THR_NEWL2: usize = 8;
/// `THR_NEWL3` = 9.
pub const THR_NEWL3: usize = 9;
/// `THR_NEWB` = 10.
pub const THR_NEWB: usize = 10;
/// `THR_NEWA2` = 11.
pub const THR_NEWA2: usize = 11;
/// `THR_NEWA` = 12.
pub const THR_NEWA: usize = 12;
/// `THR_NEWG` = 13.
pub const THR_NEWG: usize = 13;
/// `THR_NEARMV` = 14.
pub const THR_NEARMV: usize = 14;
/// `THR_NEARL2` = 15.
pub const THR_NEARL2: usize = 15;
/// `THR_NEARL3` = 16.
pub const THR_NEARL3: usize = 16;
/// `THR_NEARB` = 17.
pub const THR_NEARB: usize = 17;
/// `THR_NEARA2` = 18.
pub const THR_NEARA2: usize = 18;
/// `THR_NEARA` = 19.
pub const THR_NEARA: usize = 19;
/// `THR_NEARG` = 20.
pub const THR_NEARG: usize = 20;
/// `THR_GLOBALMV` = 21.
pub const THR_GLOBALMV: usize = 21;
/// `THR_GLOBALL2` = 22.
pub const THR_GLOBALL2: usize = 22;
/// `THR_GLOBALL3` = 23.
pub const THR_GLOBALL3: usize = 23;
/// `THR_GLOBALB` = 24.
pub const THR_GLOBALB: usize = 24;
/// `THR_GLOBALA2` = 25.
pub const THR_GLOBALA2: usize = 25;
/// `THR_GLOBALA` = 26.
pub const THR_GLOBALA: usize = 26;
/// `THR_GLOBALG` = 27.
pub const THR_GLOBALG: usize = 27;
/// `THR_COMP_NEAREST_NEARESTLA` = 28.
pub const THR_COMP_NEAREST_NEARESTLA: usize = 28;
/// `THR_COMP_NEAREST_NEARESTL2A` = 29.
pub const THR_COMP_NEAREST_NEARESTL2A: usize = 29;
/// `THR_COMP_NEAREST_NEARESTL3A` = 30.
pub const THR_COMP_NEAREST_NEARESTL3A: usize = 30;
/// `THR_COMP_NEAREST_NEARESTGA` = 31.
pub const THR_COMP_NEAREST_NEARESTGA: usize = 31;
/// `THR_COMP_NEAREST_NEARESTLB` = 32.
pub const THR_COMP_NEAREST_NEARESTLB: usize = 32;
/// `THR_COMP_NEAREST_NEARESTL2B` = 33.
pub const THR_COMP_NEAREST_NEARESTL2B: usize = 33;
/// `THR_COMP_NEAREST_NEARESTL3B` = 34.
pub const THR_COMP_NEAREST_NEARESTL3B: usize = 34;
/// `THR_COMP_NEAREST_NEARESTGB` = 35.
pub const THR_COMP_NEAREST_NEARESTGB: usize = 35;
/// `THR_COMP_NEAREST_NEARESTLA2` = 36.
pub const THR_COMP_NEAREST_NEARESTLA2: usize = 36;
/// `THR_COMP_NEAREST_NEARESTL2A2` = 37.
pub const THR_COMP_NEAREST_NEARESTL2A2: usize = 37;
/// `THR_COMP_NEAREST_NEARESTL3A2` = 38.
pub const THR_COMP_NEAREST_NEARESTL3A2: usize = 38;
/// `THR_COMP_NEAREST_NEARESTGA2` = 39.
pub const THR_COMP_NEAREST_NEARESTGA2: usize = 39;
/// `THR_COMP_NEAREST_NEARESTLL2` = 40.
pub const THR_COMP_NEAREST_NEARESTLL2: usize = 40;
/// `THR_COMP_NEAREST_NEARESTLL3` = 41.
pub const THR_COMP_NEAREST_NEARESTLL3: usize = 41;
/// `THR_COMP_NEAREST_NEARESTLG` = 42.
pub const THR_COMP_NEAREST_NEARESTLG: usize = 42;
/// `THR_COMP_NEAREST_NEARESTBA` = 43.
pub const THR_COMP_NEAREST_NEARESTBA: usize = 43;
/// `THR_COMP_NEAR_NEARLB` = 44.
pub const THR_COMP_NEAR_NEARLB: usize = 44;
/// `THR_COMP_NEW_NEWLB` = 45.
pub const THR_COMP_NEW_NEWLB: usize = 45;
/// `THR_COMP_NEW_NEARESTLB` = 46.
pub const THR_COMP_NEW_NEARESTLB: usize = 46;
/// `THR_COMP_NEAREST_NEWLB` = 47.
pub const THR_COMP_NEAREST_NEWLB: usize = 47;
/// `THR_COMP_NEW_NEARLB` = 48.
pub const THR_COMP_NEW_NEARLB: usize = 48;
/// `THR_COMP_NEAR_NEWLB` = 49.
pub const THR_COMP_NEAR_NEWLB: usize = 49;
/// `THR_COMP_GLOBAL_GLOBALLB` = 50.
pub const THR_COMP_GLOBAL_GLOBALLB: usize = 50;
/// `THR_COMP_NEAR_NEARLA` = 51.
pub const THR_COMP_NEAR_NEARLA: usize = 51;
/// `THR_COMP_NEW_NEWLA` = 52.
pub const THR_COMP_NEW_NEWLA: usize = 52;
/// `THR_COMP_NEW_NEARESTLA` = 53.
pub const THR_COMP_NEW_NEARESTLA: usize = 53;
/// `THR_COMP_NEAREST_NEWLA` = 54.
pub const THR_COMP_NEAREST_NEWLA: usize = 54;
/// `THR_COMP_NEW_NEARLA` = 55.
pub const THR_COMP_NEW_NEARLA: usize = 55;
/// `THR_COMP_NEAR_NEWLA` = 56.
pub const THR_COMP_NEAR_NEWLA: usize = 56;
/// `THR_COMP_GLOBAL_GLOBALLA` = 57.
pub const THR_COMP_GLOBAL_GLOBALLA: usize = 57;
/// `THR_COMP_NEAR_NEARL2A` = 58.
pub const THR_COMP_NEAR_NEARL2A: usize = 58;
/// `THR_COMP_NEW_NEWL2A` = 59.
pub const THR_COMP_NEW_NEWL2A: usize = 59;
/// `THR_COMP_NEW_NEARESTL2A` = 60.
pub const THR_COMP_NEW_NEARESTL2A: usize = 60;
/// `THR_COMP_NEAREST_NEWL2A` = 61.
pub const THR_COMP_NEAREST_NEWL2A: usize = 61;
/// `THR_COMP_NEW_NEARL2A` = 62.
pub const THR_COMP_NEW_NEARL2A: usize = 62;
/// `THR_COMP_NEAR_NEWL2A` = 63.
pub const THR_COMP_NEAR_NEWL2A: usize = 63;
/// `THR_COMP_GLOBAL_GLOBALL2A` = 64.
pub const THR_COMP_GLOBAL_GLOBALL2A: usize = 64;
/// `THR_COMP_NEAR_NEARL3A` = 65.
pub const THR_COMP_NEAR_NEARL3A: usize = 65;
/// `THR_COMP_NEW_NEWL3A` = 66.
pub const THR_COMP_NEW_NEWL3A: usize = 66;
/// `THR_COMP_NEW_NEARESTL3A` = 67.
pub const THR_COMP_NEW_NEARESTL3A: usize = 67;
/// `THR_COMP_NEAREST_NEWL3A` = 68.
pub const THR_COMP_NEAREST_NEWL3A: usize = 68;
/// `THR_COMP_NEW_NEARL3A` = 69.
pub const THR_COMP_NEW_NEARL3A: usize = 69;
/// `THR_COMP_NEAR_NEWL3A` = 70.
pub const THR_COMP_NEAR_NEWL3A: usize = 70;
/// `THR_COMP_GLOBAL_GLOBALL3A` = 71.
pub const THR_COMP_GLOBAL_GLOBALL3A: usize = 71;
/// `THR_COMP_NEAR_NEARGA` = 72.
pub const THR_COMP_NEAR_NEARGA: usize = 72;
/// `THR_COMP_NEW_NEWGA` = 73.
pub const THR_COMP_NEW_NEWGA: usize = 73;
/// `THR_COMP_NEW_NEARESTGA` = 74.
pub const THR_COMP_NEW_NEARESTGA: usize = 74;
/// `THR_COMP_NEAREST_NEWGA` = 75.
pub const THR_COMP_NEAREST_NEWGA: usize = 75;
/// `THR_COMP_NEW_NEARGA` = 76.
pub const THR_COMP_NEW_NEARGA: usize = 76;
/// `THR_COMP_NEAR_NEWGA` = 77.
pub const THR_COMP_NEAR_NEWGA: usize = 77;
/// `THR_COMP_GLOBAL_GLOBALGA` = 78.
pub const THR_COMP_GLOBAL_GLOBALGA: usize = 78;
/// `THR_COMP_NEAR_NEARL2B` = 79.
pub const THR_COMP_NEAR_NEARL2B: usize = 79;
/// `THR_COMP_NEW_NEWL2B` = 80.
pub const THR_COMP_NEW_NEWL2B: usize = 80;
/// `THR_COMP_NEW_NEARESTL2B` = 81.
pub const THR_COMP_NEW_NEARESTL2B: usize = 81;
/// `THR_COMP_NEAREST_NEWL2B` = 82.
pub const THR_COMP_NEAREST_NEWL2B: usize = 82;
/// `THR_COMP_NEW_NEARL2B` = 83.
pub const THR_COMP_NEW_NEARL2B: usize = 83;
/// `THR_COMP_NEAR_NEWL2B` = 84.
pub const THR_COMP_NEAR_NEWL2B: usize = 84;
/// `THR_COMP_GLOBAL_GLOBALL2B` = 85.
pub const THR_COMP_GLOBAL_GLOBALL2B: usize = 85;
/// `THR_COMP_NEAR_NEARL3B` = 86.
pub const THR_COMP_NEAR_NEARL3B: usize = 86;
/// `THR_COMP_NEW_NEWL3B` = 87.
pub const THR_COMP_NEW_NEWL3B: usize = 87;
/// `THR_COMP_NEW_NEARESTL3B` = 88.
pub const THR_COMP_NEW_NEARESTL3B: usize = 88;
/// `THR_COMP_NEAREST_NEWL3B` = 89.
pub const THR_COMP_NEAREST_NEWL3B: usize = 89;
/// `THR_COMP_NEW_NEARL3B` = 90.
pub const THR_COMP_NEW_NEARL3B: usize = 90;
/// `THR_COMP_NEAR_NEWL3B` = 91.
pub const THR_COMP_NEAR_NEWL3B: usize = 91;
/// `THR_COMP_GLOBAL_GLOBALL3B` = 92.
pub const THR_COMP_GLOBAL_GLOBALL3B: usize = 92;
/// `THR_COMP_NEAR_NEARGB` = 93.
pub const THR_COMP_NEAR_NEARGB: usize = 93;
/// `THR_COMP_NEW_NEWGB` = 94.
pub const THR_COMP_NEW_NEWGB: usize = 94;
/// `THR_COMP_NEW_NEARESTGB` = 95.
pub const THR_COMP_NEW_NEARESTGB: usize = 95;
/// `THR_COMP_NEAREST_NEWGB` = 96.
pub const THR_COMP_NEAREST_NEWGB: usize = 96;
/// `THR_COMP_NEW_NEARGB` = 97.
pub const THR_COMP_NEW_NEARGB: usize = 97;
/// `THR_COMP_NEAR_NEWGB` = 98.
pub const THR_COMP_NEAR_NEWGB: usize = 98;
/// `THR_COMP_GLOBAL_GLOBALGB` = 99.
pub const THR_COMP_GLOBAL_GLOBALGB: usize = 99;
/// `THR_COMP_NEAR_NEARLA2` = 100.
pub const THR_COMP_NEAR_NEARLA2: usize = 100;
/// `THR_COMP_NEW_NEWLA2` = 101.
pub const THR_COMP_NEW_NEWLA2: usize = 101;
/// `THR_COMP_NEW_NEARESTLA2` = 102.
pub const THR_COMP_NEW_NEARESTLA2: usize = 102;
/// `THR_COMP_NEAREST_NEWLA2` = 103.
pub const THR_COMP_NEAREST_NEWLA2: usize = 103;
/// `THR_COMP_NEW_NEARLA2` = 104.
pub const THR_COMP_NEW_NEARLA2: usize = 104;
/// `THR_COMP_NEAR_NEWLA2` = 105.
pub const THR_COMP_NEAR_NEWLA2: usize = 105;
/// `THR_COMP_GLOBAL_GLOBALLA2` = 106.
pub const THR_COMP_GLOBAL_GLOBALLA2: usize = 106;
/// `THR_COMP_NEAR_NEARL2A2` = 107.
pub const THR_COMP_NEAR_NEARL2A2: usize = 107;
/// `THR_COMP_NEW_NEWL2A2` = 108.
pub const THR_COMP_NEW_NEWL2A2: usize = 108;
/// `THR_COMP_NEW_NEARESTL2A2` = 109.
pub const THR_COMP_NEW_NEARESTL2A2: usize = 109;
/// `THR_COMP_NEAREST_NEWL2A2` = 110.
pub const THR_COMP_NEAREST_NEWL2A2: usize = 110;
/// `THR_COMP_NEW_NEARL2A2` = 111.
pub const THR_COMP_NEW_NEARL2A2: usize = 111;
/// `THR_COMP_NEAR_NEWL2A2` = 112.
pub const THR_COMP_NEAR_NEWL2A2: usize = 112;
/// `THR_COMP_GLOBAL_GLOBALL2A2` = 113.
pub const THR_COMP_GLOBAL_GLOBALL2A2: usize = 113;
/// `THR_COMP_NEAR_NEARL3A2` = 114.
pub const THR_COMP_NEAR_NEARL3A2: usize = 114;
/// `THR_COMP_NEW_NEWL3A2` = 115.
pub const THR_COMP_NEW_NEWL3A2: usize = 115;
/// `THR_COMP_NEW_NEARESTL3A2` = 116.
pub const THR_COMP_NEW_NEARESTL3A2: usize = 116;
/// `THR_COMP_NEAREST_NEWL3A2` = 117.
pub const THR_COMP_NEAREST_NEWL3A2: usize = 117;
/// `THR_COMP_NEW_NEARL3A2` = 118.
pub const THR_COMP_NEW_NEARL3A2: usize = 118;
/// `THR_COMP_NEAR_NEWL3A2` = 119.
pub const THR_COMP_NEAR_NEWL3A2: usize = 119;
/// `THR_COMP_GLOBAL_GLOBALL3A2` = 120.
pub const THR_COMP_GLOBAL_GLOBALL3A2: usize = 120;
/// `THR_COMP_NEAR_NEARGA2` = 121.
pub const THR_COMP_NEAR_NEARGA2: usize = 121;
/// `THR_COMP_NEW_NEWGA2` = 122.
pub const THR_COMP_NEW_NEWGA2: usize = 122;
/// `THR_COMP_NEW_NEARESTGA2` = 123.
pub const THR_COMP_NEW_NEARESTGA2: usize = 123;
/// `THR_COMP_NEAREST_NEWGA2` = 124.
pub const THR_COMP_NEAREST_NEWGA2: usize = 124;
/// `THR_COMP_NEW_NEARGA2` = 125.
pub const THR_COMP_NEW_NEARGA2: usize = 125;
/// `THR_COMP_NEAR_NEWGA2` = 126.
pub const THR_COMP_NEAR_NEWGA2: usize = 126;
/// `THR_COMP_GLOBAL_GLOBALGA2` = 127.
pub const THR_COMP_GLOBAL_GLOBALGA2: usize = 127;
/// `THR_COMP_NEAR_NEARLL2` = 128.
pub const THR_COMP_NEAR_NEARLL2: usize = 128;
/// `THR_COMP_NEW_NEWLL2` = 129.
pub const THR_COMP_NEW_NEWLL2: usize = 129;
/// `THR_COMP_NEW_NEARESTLL2` = 130.
pub const THR_COMP_NEW_NEARESTLL2: usize = 130;
/// `THR_COMP_NEAREST_NEWLL2` = 131.
pub const THR_COMP_NEAREST_NEWLL2: usize = 131;
/// `THR_COMP_NEW_NEARLL2` = 132.
pub const THR_COMP_NEW_NEARLL2: usize = 132;
/// `THR_COMP_NEAR_NEWLL2` = 133.
pub const THR_COMP_NEAR_NEWLL2: usize = 133;
/// `THR_COMP_GLOBAL_GLOBALLL2` = 134.
pub const THR_COMP_GLOBAL_GLOBALLL2: usize = 134;
/// `THR_COMP_NEAR_NEARLL3` = 135.
pub const THR_COMP_NEAR_NEARLL3: usize = 135;
/// `THR_COMP_NEW_NEWLL3` = 136.
pub const THR_COMP_NEW_NEWLL3: usize = 136;
/// `THR_COMP_NEW_NEARESTLL3` = 137.
pub const THR_COMP_NEW_NEARESTLL3: usize = 137;
/// `THR_COMP_NEAREST_NEWLL3` = 138.
pub const THR_COMP_NEAREST_NEWLL3: usize = 138;
/// `THR_COMP_NEW_NEARLL3` = 139.
pub const THR_COMP_NEW_NEARLL3: usize = 139;
/// `THR_COMP_NEAR_NEWLL3` = 140.
pub const THR_COMP_NEAR_NEWLL3: usize = 140;
/// `THR_COMP_GLOBAL_GLOBALLL3` = 141.
pub const THR_COMP_GLOBAL_GLOBALLL3: usize = 141;
/// `THR_COMP_NEAR_NEARLG` = 142.
pub const THR_COMP_NEAR_NEARLG: usize = 142;
/// `THR_COMP_NEW_NEWLG` = 143.
pub const THR_COMP_NEW_NEWLG: usize = 143;
/// `THR_COMP_NEW_NEARESTLG` = 144.
pub const THR_COMP_NEW_NEARESTLG: usize = 144;
/// `THR_COMP_NEAREST_NEWLG` = 145.
pub const THR_COMP_NEAREST_NEWLG: usize = 145;
/// `THR_COMP_NEW_NEARLG` = 146.
pub const THR_COMP_NEW_NEARLG: usize = 146;
/// `THR_COMP_NEAR_NEWLG` = 147.
pub const THR_COMP_NEAR_NEWLG: usize = 147;
/// `THR_COMP_GLOBAL_GLOBALLG` = 148.
pub const THR_COMP_GLOBAL_GLOBALLG: usize = 148;
/// `THR_COMP_NEAR_NEARBA` = 149.
pub const THR_COMP_NEAR_NEARBA: usize = 149;
/// `THR_COMP_NEW_NEWBA` = 150.
pub const THR_COMP_NEW_NEWBA: usize = 150;
/// `THR_COMP_NEW_NEARESTBA` = 151.
pub const THR_COMP_NEW_NEARESTBA: usize = 151;
/// `THR_COMP_NEAREST_NEWBA` = 152.
pub const THR_COMP_NEAREST_NEWBA: usize = 152;
/// `THR_COMP_NEW_NEARBA` = 153.
pub const THR_COMP_NEW_NEARBA: usize = 153;
/// `THR_COMP_NEAR_NEWBA` = 154.
pub const THR_COMP_NEAR_NEWBA: usize = 154;
/// `THR_COMP_GLOBAL_GLOBALBA` = 155.
pub const THR_COMP_GLOBAL_GLOBALBA: usize = 155;
/// `THR_DC` = 156.
pub const THR_DC: usize = 156;
/// `THR_PAETH` = 157.
pub const THR_PAETH: usize = 157;
/// `THR_SMOOTH` = 158.
pub const THR_SMOOTH: usize = 158;
/// `THR_SMOOTH_V` = 159.
pub const THR_SMOOTH_V: usize = 159;
/// `THR_SMOOTH_H` = 160.
pub const THR_SMOOTH_H: usize = 160;
/// `THR_H_PRED` = 161.
pub const THR_H_PRED: usize = 161;
/// `THR_V_PRED` = 162.
pub const THR_V_PRED: usize = 162;
/// `THR_D135_PRED` = 163.
pub const THR_D135_PRED: usize = 163;
/// `THR_D203_PRED` = 164.
pub const THR_D203_PRED: usize = 164;
/// `THR_D157_PRED` = 165.
pub const THR_D157_PRED: usize = 165;
/// `THR_D67_PRED` = 166.
pub const THR_D67_PRED: usize = 166;
/// `THR_D113_PRED` = 167.
pub const THR_D113_PRED: usize = 167;
/// `THR_D45_PRED` = 168.
pub const THR_D45_PRED: usize = 168;

/// The baseline `thresh_mult` table `av1_set_rd_speed_thresholds` writes,
/// indexed by `THR_MODES`.
///
/// C builds it by `av1_zero(rd->thresh_mult)` followed by one assignment per
/// mode; every mode is in fact assigned, so no entry is left at zero — which
/// `tests/rd_thresh_diff.rs` asserts, since a silently-missed assignment would
/// otherwise read as a legitimate 0.
#[rustfmt::skip]
static THRESH_MULT: [i32; MAX_MODES] = [
    300, 300, 300, 300, 300, 300,
    300, 1000, 1000, 1000, 1000, 1100,
    1000, 1000, 1000, 1000, 1000, 1000,
    1000, 1000, 1000, 2200, 2000, 2000,
    2400, 2000, 2400, 2000, 1100, 1000,
    800, 900, 1000, 1000, 1000, 1000,
    1000, 1000, 1000, 1000, 2000, 2000,
    2000, 2000, 1200, 2400, 1500, 1500,
    1700, 1360, 2250, 1200, 2400, 1500,
    1500, 1870, 1530, 2750, 1200, 1800,
    1500, 1500, 1700, 1870, 2500, 1200,
    2000, 1500, 1500, 1700, 1700, 3000,
    1320, 2000, 1500, 1500, 1700, 2040,
    2250, 1200, 2000, 1500, 1500, 1700,
    1700, 2500, 1200, 2000, 1500, 1500,
    1700, 1870, 2500, 1200, 2000, 1500,
    1500, 1700, 1700, 2500, 1200, 2000,
    1500, 1800, 1700, 1700, 2500, 1200,
    2000, 1500, 1500, 1700, 1700, 2500,
    1440, 2000, 1500, 1500, 1700, 1700,
    2500, 1200, 2000, 1500, 1500, 1700,
    1700, 2750, 1600, 2400, 2000, 2000,
    2200, 2640, 3200, 1600, 2400, 1800,
    2000, 2200, 2200, 3200, 1760, 2400,
    2000, 2400, 2640, 1760, 3200, 1600,
    2640, 2000, 2000, 1980, 2200, 3200,
    1000, 1000, 2200, 2000, 2000, 2000,
    1800, 2500, 2000, 2500, 2000, 2500,
    2500,
];

/// `av1_set_rd_speed_thresholds` (`av1/encoder/rd.c`): the baseline per-mode RD
/// threshold multipliers.
#[must_use]
pub fn set_rd_speed_thresholds() -> [i32; MAX_MODES] {
    THRESH_MULT
}

/// `update_thr_fact` (rd.c:1451, `static`): nudge the adaptive threshold factor
/// for one mode range across a block-size range.
///
/// The winning mode's factor DECAYS (`fact -= fact >> 4`) while every other
/// mode's grows by one, clamped. Note the asymmetry: the decay is
/// multiplicative and the growth additive, so a mode that stops winning climbs
/// back slowly and a mode that starts winning drops fast.
#[allow(clippy::too_many_arguments)]
pub fn update_thr_fact(
    factor_buf: &mut [[i32; MAX_MODES]],
    best_mode_index: usize,
    mode_start: usize,
    mode_end: usize,
    min_size: usize,
    max_size: usize,
    max_rd_thresh_factor: i32,
) {
    // Indexed rather than iterated on purpose: `mode` is compared against
    // `best_mode_index` and `bs` selects a row, so both are values in C's loop,
    // not just cursors.
    #[allow(clippy::needless_range_loop)]
    for mode in mode_start..mode_end {
        for bs in min_size..=max_size {
            let fact = &mut factor_buf[bs][mode];
            if mode == best_mode_index {
                *fact -= *fact >> RD_THRESH_LOG_DEC_FACTOR;
            } else {
                *fact = (*fact + RD_THRESH_INC).min(max_rd_thresh_factor);
            }
        }
    }
}

/// `av1_update_rd_thresh_fact` (rd.c:1468).
///
/// The block-size window is NOT symmetric around `bsize`: a 1:4 or 4:1 shape
/// (which C detects as `bsize > sb_size`, exploiting that those sizes sort
/// after the square ones) updates only its own row, while every other size
/// updates `[bsize - 2, bsize + 2]` clamped to `[BLOCK_4X4, sb_size]`.
/// Reproducing that as a plain +/-2 window would touch the wrong rows for the
/// extreme aspect ratios.
#[allow(clippy::too_many_arguments)]
pub fn update_rd_thresh_fact(
    sb_size: usize,
    factor_buf: &mut [[i32; MAX_MODES]],
    use_adaptive_rd_thresh: i32,
    bsize: usize,
    best_mode_index: usize,
    inter_mode_start: usize,
    inter_mode_end: usize,
    intra_mode_start: usize,
    intra_mode_end: usize,
) {
    assert!(use_adaptive_rd_thresh > 0);
    let max_rd_thresh_factor = use_adaptive_rd_thresh * RD_THRESH_MAX_FACT;

    let (min_size, max_size) = if bsize > sb_size {
        (bsize, bsize)
    } else {
        (
            (bsize as i32 - 2).max(BLOCK_4X4) as usize,
            (bsize + 2).min(sb_size),
        )
    };

    update_thr_fact(
        factor_buf,
        best_mode_index,
        inter_mode_start,
        inter_mode_end,
        min_size,
        max_size,
        max_rd_thresh_factor,
    );
    update_thr_fact(
        factor_buf,
        best_mode_index,
        intra_mode_start,
        intra_mode_end,
        min_size,
        max_size,
        max_rd_thresh_factor,
    );
}
