# Pre-registered reading rule — written 2026-08-03T20:45Z, BEFORE any lever band was read

Declared while `p4_rot.tsv` was at round 17 of 50 and the other two bands had not
started, so that the decision rule cannot be fitted to the outcome
(`DIFFERENTIAL_PLAYBOOK.md` §14).

## When a rotated band is allowed to be called "resolving"

Both must hold:

1. the paired-**mean** MDE at 95 % (`1.96·sd/√n`, the column
   `scripts/eprof_ab_position.py` prints) is **≤ half the published Darwin
   headline** for that lever —
   KB-PERF-2 ≤ 1.455 pp, KB-PERF-3 ≤ 1.245 pp, KB-PERF-4 ≤ 0.375 pp;
2. **both** same-binary null arms in that same band have |paired mean| **smaller
   than the measured post-vs-pre effect** in that band.

## If it does not hold

Extend ROUNDS on the same arms, same binaries, same cell, until it does, or
report the lever **unresolved on this box in this window**. Nothing else may
change: not the content, not the cell, not the arm set, not the statistic.
Extra rounds are a *new, longer band*, not rounds appended to a short one that
already read badly.

## The headline statistic

The paired **mean** is the headline, with the paired median and the sign test
beside it. Reason (from `encoder_intra_smooth_paeth_2026-08-03.md` §3): at these
n, a sub-1 % effect against a ~0.5-3 sd moves the median far more than the mean,
and the two post-side copies there agreed to 0.005 pp on the mean while sitting
0.25 pp apart on the median. The originals published paired **medians**, so both
are reported side by side and the comparison of like with like is stated
explicitly.
