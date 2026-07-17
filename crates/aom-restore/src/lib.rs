//! aom-restore — AV1 loop restoration (libaom v3.14.1 `av1/common/restoration.c`):
//! the Wiener 7-tap separable filter (`av1_wiener_convolve_add_src` /
//! `av1_highbd_wiener_convolve_add_src`), self-guided restoration
//! (`av1_selfguided_restoration` + `av1_apply_selfguided_restoration`), and
//! the whole-frame restoration-unit walk with striped boundary handling
//! (`av1_loop_restoration_filter_frame` +
//! `av1_loop_restoration_save_boundary_lines`).
//!
//! Pixels are `u16` at every bit depth. For 8-bit streams the C decoder runs
//! its lowbd (u8) kernels; these ports replicate that arithmetic exactly on
//! u16 values ≤ 255 (differentially verified against the real lowbd C
//! kernels), so one u16 path serves all depths — same discipline as
//! aom-loopfilter and aom-cdef. Parameter types (`LrUnitInfo` etc.) come from
//! `aom_entropy::lr` (the tile-parse side).

#![forbid(unsafe_code)]

pub mod frame;
pub mod pick;
pub mod sgr;
pub mod wiener;
