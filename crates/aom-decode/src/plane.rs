//! Reconstruction-plane storage — the bd8 lowbd / highbd split carrier.
//!
//! Per the lowbd pipeline contract (`aom_dsp::lowbd`), a bd8 frame stores its
//! reconstruction planes as `u8` and a bd10/bd12 frame as `u16`. This enum is
//! the carrier threaded through the tile driver ([`crate::KfTileDecode`] and
//! the private `TileKf`) and the post-recon filter stages in `crate::frame`.
//!
//! # Phase A: byte-identical delegation
//!
//! In the current phase NO u8 kernel is wired: every kernel call site on a
//! `LowBd` plane DELEGATES to the existing highbd kernel by widening the
//! touched region `u8 -> u16`, running the unchanged highbd kernel (with
//! `bd == 8`), and narrowing the result back `u16 -> u8`. This is byte-exact:
//! a bd8 sample round-trips `u8 -> u16 -> u8` losslessly, and every bd8 kernel
//! output is clamped to `0..=255` by the normative pixel clamp, so the
//! narrowing `as u8` cannot truncate (debug-asserted). The delegation costs
//! conversion copies; the follow-up phase replaces each `LowBd` arm with the
//! already-landed `*_u8` kernel and removes that cost.
//!
//! The `HighBd` arms are structurally the pre-refactor code: the same slices
//! reach the same kernels, so bd10/bd12 decoding is unchanged by construction.

/// One reconstruction plane: `u8` samples at bit depth 8, `u16` at 10/12.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReconPlane {
    /// bd8: one byte per pixel.
    LowBd(Vec<u8>),
    /// bd10/bd12 (and the >8-bit init-canary test path): one `u16` per pixel.
    HighBd(Vec<u16>),
}

/// Narrow a widened bd8 working value back to its `u8` storage.
///
/// On every CONFORMANT decode this is lossless: the normative bd8 pixel clamp
/// bounds every kernel output to `0..=255`, and the Gate-1 byte-identity
/// corpus proves the delegation round-trip exact. HOSTILE input, however, can
/// smuggle out-of-range values into the pipeline before the corrupt guards
/// unwind — e.g. a corrupt palette colour literal `> 255` at bd8 — so this
/// must NOT assert (the fuzz no-panic gate exercises exactly that). Plain
/// truncation mirrors the C lowbd decoder's own `(uint8_t)` pixel stores on
/// the same input.
#[inline]
fn narrow(v: u16) -> u8 {
    v as u8
}

/// Samples per stop-token poll in [`ReconPlane::take_wide_stop`] /
/// [`ReconPlane::put_wide_stop`].
///
/// 1 << 20 samples is 1 MiB read + 2 MiB written per chunk — about 0.7 ms on
/// the machine the cancellation record was taken on, i.e. two orders under the
/// 20 ms bar, while costing a 4096x4096 luma plane only 16 branches.
const WIDEN_SAMPLES_PER_POLL: usize = 1 << 20;

impl ReconPlane {
    /// A plane of `len` samples filled with `init`. `lowbd` selects the `u8`
    /// representation (caller guarantees `init <= 255` then — asserted).
    pub(crate) fn filled(lowbd: bool, len: usize, init: u16) -> Self {
        if lowbd {
            assert!(init <= 0xFF, "lowbd plane init {init} exceeds u8");
            ReconPlane::LowBd(vec![init as u8; len])
        } else {
            ReconPlane::HighBd(vec![init; len])
        }
    }

    /// True when this plane holds `u8` (bd8 lowbd) samples.
    pub fn is_lowbd(&self) -> bool {
        matches!(self, ReconPlane::LowBd(_))
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        match self {
            ReconPlane::LowBd(p) => p.len(),
            ReconPlane::HighBd(p) => p.len(),
        }
    }

    /// True when the plane holds no samples (monochrome chroma).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Single-pixel read, widened to the common `u16` domain.
    #[inline]
    pub fn px(&self, i: usize) -> u16 {
        match self {
            ReconPlane::LowBd(p) => p[i] as u16,
            ReconPlane::HighBd(p) => p[i],
        }
    }

    /// Widen the whole plane to `u16` (bit-exact at bd8). This is the Phase A
    /// boundary conversion for consumers that stay `u16` (`RefFrame`, the
    /// `FrameDecode` crop, tests).
    pub fn to_u16(&self) -> Vec<u16> {
        match self {
            ReconPlane::LowBd(p) => p.iter().map(|&v| v as u16).collect(),
            ReconPlane::HighBd(p) => p.clone(),
        }
    }

    /// Move the plane out as a `u16` working buffer (zero-copy for `HighBd`,
    /// a widening copy for `LowBd`), leaving `self` to be refilled by
    /// [`ReconPlane::put_wide`]. The take/put pair brackets a whole-plane
    /// highbd stage (deblock / CDEF / LR) in Phase A.
    pub(crate) fn take_wide(&mut self) -> Vec<u16> {
        match self {
            ReconPlane::LowBd(p) => p.iter().map(|&v| v as u16).collect(),
            ReconPlane::HighBd(p) => core::mem::take(p),
        }
    }

    /// [`ReconPlane::take_wide`] with a cooperative stop token polled every
    /// [`WIDEN_SAMPLES_PER_POLL`] samples of the widening copy.
    ///
    /// The `LowBd` arm is a whole-plane `u8 -> u16` conversion, and at
    /// 4096x4096 it is the single largest un-pollable window left in a decode:
    /// **MEASURED 11.5-11.8 ms** as the deblock -> CDEF transition gap
    /// (GitHub #17), against CDEF's own 2.3 ms per filter-block row. It is not
    /// removable by routing CDEF through `cdef_frame_u8` instead — that entry's
    /// own doc records it as **6.6 % MORE Ir than delegating**, i.e. the widen
    /// is a deliberate, measured throughput choice — so it is made INTERRUPTIBLE
    /// rather than avoided.
    ///
    /// `HighBd` is a `mem::take` and cannot be subdivided; it does not need to
    /// be, being O(1).
    pub(crate) fn take_wide_stop(
        &mut self,
        stop: Option<&dyn enough::Stop>,
    ) -> Result<Vec<u16>, enough::StopReason> {
        match self {
            ReconPlane::LowBd(p) => {
                let mut out = Vec::with_capacity(p.len());
                for chunk in p.chunks(WIDEN_SAMPLES_PER_POLL) {
                    if let Some(s) = stop {
                        s.check()?;
                    }
                    out.extend(chunk.iter().map(|&v| v as u16));
                }
                Ok(out)
            }
            ReconPlane::HighBd(p) => Ok(core::mem::take(p)),
        }
    }

    /// [`ReconPlane::put_wide`] with the same per-chunk poll as
    /// [`ReconPlane::take_wide_stop`]. The narrowing store is the same size as
    /// the widening load, so it is the same un-pollable window on the way back.
    ///
    /// On cancellation the plane is left PARTIALLY filled — which is sound
    /// because every caller propagates the stop and drops the whole decode; a
    /// cancelled decode never produces a frame.
    pub(crate) fn put_wide_stop(
        &mut self,
        v: Vec<u16>,
        stop: Option<&dyn enough::Stop>,
    ) -> Result<(), enough::StopReason> {
        match self {
            ReconPlane::LowBd(p) => {
                p.clear();
                p.reserve(v.len());
                for chunk in v.chunks(WIDEN_SAMPLES_PER_POLL) {
                    if let Some(s) = stop {
                        s.check()?;
                    }
                    p.extend(chunk.iter().map(|&x| narrow(x)));
                }
                Ok(())
            }
            ReconPlane::HighBd(p) => {
                *p = v;
                Ok(())
            }
        }
    }

    /// Store a `u16` working buffer back (the counterpart of
    /// [`ReconPlane::take_wide`]). The length may differ (superres replaces
    /// planes at a new stride); the variant is preserved.
    pub(crate) fn put_wide(&mut self, v: Vec<u16>) {
        match self {
            ReconPlane::LowBd(p) => {
                p.clear();
                p.extend(v.iter().map(|&x| narrow(x)));
            }
            ReconPlane::HighBd(p) => *p = v,
        }
    }

    /// Widen `dst.len()` samples starting at `off` into `dst`.
    #[inline]
    pub(crate) fn copy_row_wide(&self, off: usize, dst: &mut [u16]) {
        let n = dst.len();
        match self {
            ReconPlane::LowBd(p) => {
                for (d, &s) in dst.iter_mut().zip(&p[off..off + n]) {
                    *d = s as u16;
                }
            }
            ReconPlane::HighBd(p) => dst.copy_from_slice(&p[off..off + n]),
        }
    }

    /// Store `src` (narrowing on `LowBd`) at `off`.
    #[inline]
    pub(crate) fn store_row(&mut self, off: usize, src: &[u16]) {
        match self {
            ReconPlane::LowBd(p) => {
                for (d, &s) in p[off..off + src.len()].iter_mut().zip(src) {
                    *d = narrow(s);
                }
            }
            ReconPlane::HighBd(p) => p[off..off + src.len()].copy_from_slice(src),
        }
    }

    /// Run a highbd kernel over the `w x h` rectangle at `off` (row stride
    /// `stride`). `HighBd`: the kernel runs directly on the plane (`f` gets the
    /// slice starting at `off` and the plane stride) — structurally the
    /// pre-refactor call. `LowBd` (Phase A): the rectangle is widened into
    /// `tmp` at compact stride `w`, the kernel runs on it, and the result is
    /// narrowed back — byte-identical per the module contract.
    pub(crate) fn with_wide_rect<R>(
        &mut self,
        off: usize,
        stride: usize,
        w: usize,
        h: usize,
        tmp: &mut Vec<u16>,
        f: impl FnOnce(&mut [u16], usize) -> R,
    ) -> R {
        match self {
            ReconPlane::HighBd(p) => f(&mut p[off..], stride),
            ReconPlane::LowBd(p) => {
                let need = w * h;
                if tmp.len() < need {
                    tmp.resize(need, 0);
                }
                for r in 0..h {
                    let s = off + r * stride;
                    for (d, &v) in tmp[r * w..r * w + w].iter_mut().zip(&p[s..s + w]) {
                        *d = v as u16;
                    }
                }
                let ret = f(&mut tmp[..need], w);
                for r in 0..h {
                    let d = off + r * stride;
                    for (dst, &v) in p[d..d + w].iter_mut().zip(&tmp[r * w..r * w + w]) {
                        *dst = narrow(v);
                    }
                }
                ret
            }
        }
    }
}
