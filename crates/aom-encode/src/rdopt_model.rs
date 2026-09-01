//! libaom's inter-mode RD MODEL (`av1/encoder/rdopt.c:353-467`): the running
//! least-squares fit that lets the inter search estimate a candidate's
//! rate and distortion from its prediction SSE, instead of running a full
//! transform search on every candidate.
//!
//! Two of these four functions — `av1_inter_mode_data_init` and
//! `av1_inter_mode_data_fit` — are among rdopt.c's ten EXPORTED symbols, so
//! their gate is **tier 1 proper** (the real symbol out of the archive).
//! `get_est_rate_dist` and `inter_mode_data_push` are `static` and go through
//! the rdopt shim TU (tier 1c). Gate:
//! `crates/aom-encode/tests/rdopt_model_diff.rs`.
//!
//! # How the model works
//!
//! Each block size keeps running sums over observed `(sse, dist, rate)`
//! triples. `ld` ("lambda-distortion") is `(sse - dist) / residue_cost`: the
//! distortion bought per bit at that block. Once enough samples are in,
//! [`InterModeRdModel::fit`] solves a one-variable least-squares line
//! `ld = a * sse + b`, and [`InterModeRdModel::est_rate_dist`] inverts it to
//! predict a candidate's residual cost from its SSE.
//!
//! # Floating point
//!
//! This is `f64` arithmetic that has to match C bit-for-bit. It does, for
//! three reasons the differential checks rather than assumes: the oracle is
//! built `-ffp-contract=off` so C cannot fuse a multiply-add where Rust
//! cannot; `sqrt` is IEEE-exact in both; and C's `round()` and Rust's
//! `f64::round` both round half AWAY from zero (`f64::round_ties_even` is the
//! other one, and is NOT what C's `round` does).

/// `INTER_MODE_RD_DATA_OVERALL_SIZE` (`encoder.h:1246`).
pub const INTER_MODE_RD_DATA_OVERALL_SIZE: i32 = 6400;

/// `BLOCK_4X4`, `BLOCK_4X8`, `BLOCK_8X4`, `BLOCK_4X16`, `BLOCK_16X4` — the
/// five shapes `inter_mode_data_block_idx` excludes from the model.
const NO_MODEL_BSIZES: [usize; 5] = [0, 1, 2, 16, 17];

/// `inter_mode_data_block_idx` (`rdopt_utils.h:298`): whether this block size
/// participates in the RD model at all.
///
/// C returns `-1` or `1` and every caller only tests for `-1`, so the port
/// returns the predicate directly.
pub fn block_uses_rd_model(bsize: usize) -> bool {
    !NO_MODEL_BSIZES.contains(&bsize)
}

/// `InterModeRdModel` (`encoder.h:1248`): one block size's running fit.
///
/// `ready == false` means the means and `a`/`b` are meaningless — C leaves
/// them at whatever the allocation held, and [`Self::init`] does not clear
/// them either. That is reproduced rather than tidied: a "fixed" init would
/// diverge from C on the first `fit` after a re-init, because `fit`'s
/// `ready == 1` arm blends the OLD means into the new ones.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InterModeRdModel {
    /// `ready`: the fit is usable.
    pub ready: bool,
    /// `num`: samples accumulated since the last fit.
    pub num: i32,
    /// `a`, `b`: the fitted line `ld = a * sse + b`.
    pub a: f64,
    /// See [`Self::a`].
    pub b: f64,
    /// `dist_mean`.
    pub dist_mean: f64,
    /// `ld_mean`.
    pub ld_mean: f64,
    /// `sse_mean`.
    pub sse_mean: f64,
    /// `sse_sse_mean`.
    pub sse_sse_mean: f64,
    /// `sse_ld_mean`.
    pub sse_ld_mean: f64,
    /// `dist_sum`.
    pub dist_sum: f64,
    /// `ld_sum`.
    pub ld_sum: f64,
    /// `sse_sum`.
    pub sse_sum: f64,
    /// `sse_sse_sum`.
    pub sse_sse_sum: f64,
    /// `sse_ld_sum`.
    pub sse_ld_sum: f64,
}

impl InterModeRdModel {
    /// `av1_inter_mode_data_init` (rdopt.c:353), for one block size.
    ///
    /// Resets `ready`, `num` and the five running sums — and DELIBERATELY not
    /// the means or `a`/`b`, exactly as C does.
    pub fn init(&mut self) {
        self.ready = false;
        self.num = 0;
        self.dist_sum = 0.0;
        self.ld_sum = 0.0;
        self.sse_sum = 0.0;
        self.sse_sse_sum = 0.0;
        self.sse_ld_sum = 0.0;
    }

    /// `inter_mode_data_push` (rdopt.c:450): record one observed
    /// `(sse, dist, residue_cost)` triple.
    ///
    /// A zero residual cost, or `sse == dist`, carries no rate information and
    /// is dropped — which is also what keeps `ld` from dividing by zero.
    pub fn push(&mut self, bsize: usize, sse: i64, dist: i64, residue_cost: i32) {
        if residue_cost == 0 || sse == dist {
            return;
        }
        if !block_uses_rd_model(bsize) {
            return;
        }
        if self.num >= INTER_MODE_RD_DATA_OVERALL_SIZE {
            return;
        }
        let ld = (sse - dist) as f64 / f64::from(residue_cost);
        self.num += 1;
        self.dist_sum += dist as f64;
        self.ld_sum += ld;
        self.sse_sum += sse as f64;
        self.sse_sse_sum += sse as f64 * sse as f64;
        self.sse_ld_sum += sse as f64 * ld;
    }

    /// `av1_inter_mode_data_fit` (rdopt.c:400), for one block size.
    ///
    /// Needs 200 samples for the first fit and 64 for a refresh; below that it
    /// is a no-op. A refresh blends the previous means in at weight 3, so the
    /// model has a memory of about four fits.
    ///
    /// `rdmult` is C's parameter and C's own `(void)rdmult` says it is unused;
    /// it is not taken here.
    pub fn fit(&mut self, bsize: usize) {
        if !block_uses_rd_model(bsize) {
            return;
        }
        if (!self.ready && self.num < 200) || (self.ready && self.num < 64) {
            return;
        }
        let n = f64::from(self.num);
        if !self.ready {
            self.dist_mean = self.dist_sum / n;
            self.ld_mean = self.ld_sum / n;
            self.sse_mean = self.sse_sum / n;
            self.sse_sse_mean = self.sse_sse_sum / n;
            self.sse_ld_mean = self.sse_ld_sum / n;
        } else {
            const FACTOR: f64 = 3.0;
            let blend = |old: f64, sum: f64| (old * FACTOR + (sum / n)) / (FACTOR + 1.0);
            self.dist_mean = blend(self.dist_mean, self.dist_sum);
            self.ld_mean = blend(self.ld_mean, self.ld_sum);
            self.sse_mean = blend(self.sse_mean, self.sse_sum);
            self.sse_sse_mean = blend(self.sse_sse_mean, self.sse_sse_sum);
            self.sse_ld_mean = blend(self.sse_ld_mean, self.sse_ld_sum);
        }

        // The least-squares line through the accumulated moments. `dx` is
        // C's `sqrt(sse_sse_mean)`, immediately squared again in the
        // denominator — kept verbatim because `sqrt` then `*` is not the
        // identity in binary64 and dropping it changes the last bits.
        let my = self.ld_mean;
        let mx = self.sse_mean;
        let dx = self.sse_sse_mean.sqrt();
        let dxy = self.sse_ld_mean;
        self.a = (dxy - mx * my) / (dx * dx - mx * mx);
        self.b = my - self.a * mx;
        self.ready = true;

        self.num = 0;
        self.dist_sum = 0.0;
        self.ld_sum = 0.0;
        self.sse_sum = 0.0;
        self.sse_sse_sum = 0.0;
        self.sse_ld_sum = 0.0;
    }

    /// `get_est_rate_dist` (rdopt.c:366): predict `(residue_cost, dist)` from
    /// a candidate's prediction SSE.
    ///
    /// `None` is C's `return 0` — the model is not ready and the caller must
    /// run a real transform search.
    pub fn est_rate_dist(&self, sse: i64) -> Option<(i32, i64)> {
        if !self.ready {
            return None;
        }
        // Below the mean distortion the residual is not worth coding: report
        // the SSE as the distortion and charge nothing.
        if (sse as f64) < self.dist_mean {
            return Some((0, sse));
        }
        let mut est_dist = self.dist_mean.round() as i64;
        let est_ld = self.a * sse as f64 + self.b;
        // C clamps at INT_MAX / 2 rather than INT_MAX; its own TODO calls this
        // a stopgap.
        const CLAMP: i64 = (i32::MAX / 2) as i64;
        let mut est_residue_cost = if est_ld.abs() < 1e-2 {
            CLAMP as i32
        } else {
            let v = (sse as f64 - self.dist_mean) / est_ld;
            if v < 0.0 {
                0
            } else {
                (v.round() as i64).min(CLAMP) as i32
            }
        };
        if est_residue_cost <= 0 {
            est_residue_cost = 0;
            est_dist = sse;
        }
        Some((est_residue_cost, est_dist))
    }
}
