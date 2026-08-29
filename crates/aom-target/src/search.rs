//! Bracketed qindex search toward a quality-score target.
//!
//! Score is assumed MONOTONE NON-INCREASING in qindex (higher qindex =
//! coarser quantization = lower perceptual score) — the AV1 direction.
//! The search brackets the target, bisecting in qindex, spending at most
//! `max_encodes` trials; it returns the best trial seen (never an
//! un-encoded interpolation). Mirrors `svtav1-target::search` with the
//! qp [1,63] domain widened to qindex [0,255].

/// Options for [`search_target_qindex`].
#[derive(Debug, Clone, Copy)]
pub struct TargetOptions {
    /// Inclusive qindex bounds (AV1 qindex domain, 0..=255).
    pub min_qindex: u8,
    pub max_qindex: u8,
    /// Stop early when a trial lands within `target ± tolerance`.
    /// `0.0` spends the full budget (census mode).
    pub tolerance: f64,
    /// Hard cap on trials (encode→judge cycles). The census k.
    pub max_encodes: u8,
    /// First trial qindex. `None` = midpoint of the bounds (the
    /// content-blind control). Fitted seeds land as a phase-B
    /// `TargetOptions::seeded` constructor once the census exists.
    pub qindex_start: Option<u8>,
}

impl Default for TargetOptions {
    fn default() -> Self {
        Self {
            min_qindex: 1,
            max_qindex: 255,
            tolerance: 0.5,
            max_encodes: 3,
            qindex_start: None,
        }
    }
}

/// Outcome of a search: the best (closest-scoring) trial actually encoded.
#[derive(Debug, Clone, Copy)]
pub struct TargetSearchResult {
    pub qindex: u8,
    pub score: f64,
    /// Trials spent (= encodes performed).
    pub encodes_used: u8,
    /// `true` iff some trial landed inside the tolerance band.
    pub converged: bool,
}

/// Bracketed search: trial the seed, then bisect toward the target within
/// the shrinking qindex bracket. `trial(qindex)` performs one
/// encode→decode→judge cycle and returns the achieved score; errors abort
/// the search. The trial closure is the WHOLE dependency-injection
/// surface — encoder, decoder, and judge all live behind it.
pub fn search_target_qindex<E, Err>(
    target: f64,
    options: &TargetOptions,
    mut trial: E,
) -> Result<TargetSearchResult, Err>
where
    E: FnMut(u8) -> Result<f64, Err>,
{
    let mut lo = options.min_qindex.min(options.max_qindex);
    let mut hi = options.max_qindex.max(options.min_qindex);
    let budget = options.max_encodes.max(1);
    let mut qi = options
        .qindex_start
        .unwrap_or(lo + (hi - lo) / 2)
        .clamp(lo, hi);
    let mut best: Option<TargetSearchResult> = None;
    let mut used = 0u8;
    while used < budget {
        let score = trial(qi)?;
        used += 1;
        let cand = TargetSearchResult {
            qindex: qi,
            score,
            encodes_used: used,
            converged: (score - target).abs() <= options.tolerance,
        };
        let better = best
            .map(|b| (score - target).abs() < (b.score - target).abs())
            .unwrap_or(true);
        if better {
            best = Some(cand);
        }
        if cand.converged && options.tolerance > 0.0 {
            break;
        }
        // Monotone non-increasing in qindex: too-low score => go finer
        // (lower qindex); too-high => coarser (higher qindex).
        if score < target {
            hi = qi.saturating_sub(1).max(lo);
        } else {
            lo = qi.saturating_add(1).min(hi);
        }
        if lo >= hi && used > 0 {
            qi = lo;
            if best.map(|b| b.qindex == qi).unwrap_or(false) {
                break;
            }
        } else {
            qi = lo + (hi - lo) / 2;
        }
    }
    Ok(best.map(|mut b| { b.encodes_used = used; b }).expect("budget >= 1 ran at least one trial"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(qi: u8) -> f64 {
        // monotone non-increasing: 100 at qindex 0 -> 20 at 255
        100.0 - (qi as f64) * (80.0 / 255.0)
    }

    #[test]
    fn converges_on_synthetic_curve() {
        let opts = TargetOptions { tolerance: 0.5, max_encodes: 8, ..Default::default() };
        let r = search_target_qindex::<_, ()>(70.0, &opts, |q| Ok(synthetic(q))).unwrap();
        assert!(r.converged, "should converge within 8 trials, got {r:?}");
        assert!((r.score - 70.0).abs() <= 0.5);
    }

    #[test]
    fn census_mode_spends_full_budget() {
        let opts = TargetOptions { tolerance: 0.0, max_encodes: 3, ..Default::default() };
        let r = search_target_qindex::<_, ()>(80.0, &opts, |q| Ok(synthetic(q))).unwrap();
        assert_eq!(r.encodes_used, 3);
    }

    #[test]
    fn seed_is_first_trial() {
        let mut first = None;
        let opts = TargetOptions { qindex_start: Some(42), max_encodes: 1, tolerance: 0.0, ..Default::default() };
        let _ = search_target_qindex::<_, ()>(70.0, &opts, |q| {
            first.get_or_insert(q);
            Ok(synthetic(q))
        });
        assert_eq!(first, Some(42));
    }

    #[test]
    fn error_aborts() {
        let opts = TargetOptions::default();
        let r = search_target_qindex::<_, &str>(70.0, &opts, |_| Err("boom"));
        assert!(r.is_err());
    }
}
