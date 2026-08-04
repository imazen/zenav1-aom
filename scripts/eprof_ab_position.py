#!/usr/bin/env python3
"""Position + paired statistics for an `eprof_ab.sh` TSV.

`eprof_ab_stats.py` reports per-arm medians and one paired-median column.  This
adds the three things a ROTATE-era band needs:

* **the POSITION table** — each arm's mean by round position, normalised to that
  arm's own mean, plus the pooled per-position gradient.  A fixed-order
  interleave confounds arm with position (`docs/DIFFERENTIAL_PLAYBOOK.md` §6:
  two copies of ONE binary at positions 5 and 6 came out 0.34 pp apart), and
  this table is what makes the confound visible instead of assumed.  Under
  `ROTATE=1` each arm should occupy every position equally often, which the
  occupancy column asserts;
* **paired mean + sd + MDE** alongside the paired median, because a 0.4 %
  effect against a 0.55 sd moves the median far more than the mean at n=60;
* **the sign test**, which the contended bands need because the raw
  distribution has a heavy tail that makes a parametric MDE meaningless.

  eprof_ab_position.py <ab.tsv> [--vs ARM ...] [--ref libaom-c]

`--vs` may be repeated as `POST:PRE` pairs; with no `--vs` every arm is
differenced against the first arm in the file.
"""

import sys
from collections import OrderedDict
from math import comb, sqrt


def med(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def sign_p(k, n):
    """Two-sided exact binomial p at p0 = 0.5 for k successes of n."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def load(path):
    """Rows as (arm, round, position, ms, bytes).

    Pre-ROTATE bands have no `position` column, but `eprof_ab.sh` has always
    written one line per arm per round in the order the arms ran, so position is
    RECOVERABLE from the row order within each round — which is what makes the
    confound measurable retrospectively on bands recorded before the column
    existed.  `derived` says which of the two it is.
    """
    rows = []
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        cols = {name: i for i, name in enumerate(hdr)}
        has_pos = "position" in cols
        seen = {}
        for line in f:
            g = line.rstrip("\n").split("\t")
            rnd = int(g[cols["round"]])
            if has_pos:
                pos = int(g[cols["position"]])
            else:
                pos = seen.get(rnd, 0) + 1
                seen[rnd] = pos
            rows.append(
                (g[cols["arm"]], rnd, pos, int(g[cols["median_ns"]]) / 1e6, int(g[cols["bytes"]]))
            )
    return rows, has_pos


def main():
    path = sys.argv[1]
    argv = sys.argv[2:]
    pairs, ref = [], None
    for i, tok in enumerate(argv):
        if tok == "--vs":
            pairs.append(argv[i + 1])
        if tok == "--ref":
            ref = argv[i + 1]

    rows, has_pos = load(path)
    arms = OrderedDict()
    byround = {}
    bybytes = {}
    for arm, rnd, pos, ms, by in rows:
        arms.setdefault(arm, []).append((rnd, pos, ms))
        byround.setdefault(arm, {})[rnd] = ms
        bybytes.setdefault(arm, set()).add(by)

    n_rounds = len({r for _, r, _, _, _ in rows})
    print(f"# {path}   arms={len(arms)}  rounds={n_rounds}  "
          f"position: {'from the column' if has_pos else 'DERIVED from row order (pre-ROTATE band)'}")
    print()
    print(f"{'arm':<12} {'median':>10} {'min':>10} {'max':>10} {'spread':>8} {'n':>4}  bytes")
    for arm, xs in arms.items():
        ms = [m for _, _, m in xs]
        lo, hi = min(ms), max(ms)
        print(f"{arm:<12} {med(ms):>10.3f} {lo:>10.3f} {hi:>10.3f} "
              f"{100 * (hi - lo) / lo:>7.2f}% {len(ms):>4}  "
              f"{','.join(str(b) for b in sorted(bybytes[arm]))}")

    positions = sorted({p for _, _, p, _, _ in rows})
    single = all(len({p for _, p, _ in xs}) == 1 for xs in arms.values())
    if single:
        print()
        print("## POSITION TABLE — NOT SEPARABLE: every arm occupies exactly ONE position")
        print("   This is a FIXED-ORDER band.  Position and arm are perfectly aliased, so the")
        print("   position effect cannot be estimated from it at all — it is entirely inside")
        print("   each arm's number, with no residual to look at.  The only way to see it is a")
        print("   ROTATE=1 band, or two copies of one binary at different positions.")
        for arm, xs in arms.items():
            print(f"     {arm:<12} position {next(iter({p for _, p, _ in xs}))}")
    else:
        print()
        print("## POSITION TABLE — each arm's mean by round position, as % of that arm's own mean")
        print(f"{'arm':<12} " + " ".join(f"{'p' + str(p):>8}" for p in positions) + f" {'occupancy':>28}")
        pooled = {p: [] for p in positions}
        pooled_med = {p: [] for p in positions}
        for arm, xs in arms.items():
            ms = [m for _, _, m in xs]
            mu = sum(ms) / len(ms)
            md = med(ms)
            cells, occ = [], []
            for p in positions:
                v = [m for _, pp, m in xs if pp == p]
                occ.append(len(v))
                if v:
                    rel = 100 * (sum(v) / len(v) - mu) / mu
                    cells.append(f"{rel:>+8.3f}")
                    pooled[p].append(rel)
                    pooled_med[p].append(100 * (med(v) - md) / md)
                else:
                    cells.append(f"{'-':>8}")
            print(f"{arm:<12} " + " ".join(cells) + "   " + "/".join(str(o) for o in occ))
        print()
        vals = [sum(pooled[p]) / len(pooled[p]) if pooled[p] else 0.0 for p in positions]
        print("POOLED       " + " ".join(f"{v:>+8.3f}" for v in vals))
        print(f"  pooled position gradient (max-min) = {max(vals) - min(vals):.3f} pp   [MEANS]")
        # The mean-based row is what a Gaussian reader wants, and it is unusable on a
        # contended band: ONE 300 ms round out of 20 at a position moves that cell by
        # several pp, so the "gradient" it reports is an outlier count, not a drift.
        # Measured 2026-08-03 over six rotated bands: the mean gradient read 0.35-1.31 pp
        # and the median gradient over the SAME invocations read 0.17-0.59 pp. Worst
        # disagreement was the contended KB-PERF-4 n=150 band, 1.309 vs 0.166 -- 7.9x on
        # one dataset; the quiet 720-invocation control_rot120 read 1.206 vs 0.383, 3.1x.
        # Quote the MEDIAN row as the position effect and the mean row only as a tail
        # indicator; when they disagree by more than ~3x the band had outliers, not drift.
        vmed = [med(pooled_med[p]) if pooled_med[p] else 0.0 for p in positions]
        print("POOLED-med   " + " ".join(f"{v:>+8.3f}" for v in vmed))
        print(f"  pooled position gradient (max-min) = {max(vmed) - min(vmed):.3f} pp   [MEDIANS — quote this one]")

    if ref and ref in arms:
        print()
        print(f"## ratio against `{ref}` (paired per round)")
        for arm in arms:
            if arm == ref:
                continue
            pr = [byround[arm][r] / byround[ref][r] for r in byround[arm] if r in byround[ref]]
            print(f"{arm:<12} median-of-paired {med(pr):.4f}x   min {min(pr):.4f}  max {max(pr):.4f}")

    print()
    print("## paired deltas  (POST vs PRE, per round)")
    hdr = (f"{'post':<10} {'pre':<10} {'p-med%':>9} {'p-mean%':>9} {'sd':>7} "
           f"{'MDE95%':>8} {'faster':>9} {'p(sign)':>9}")
    print(hdr)
    if not pairs:
        first = next(iter(arms))
        pairs = [f"{a}:{first}" for a in arms if a != first]
    for spec in pairs:
        post, pre = spec.split(":")
        if post not in byround or pre not in byround:
            print(f"{post:<10} {pre:<10}   MISSING ARM")
            continue
        rs = sorted(set(byround[post]) & set(byround[pre]))
        d = [100 * (byround[post][r] / byround[pre][r] - 1) for r in rs]
        n = len(d)
        mu = sum(d) / n
        sd = sqrt(sum((x - mu) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
        k = sum(1 for x in d if x < 0)
        mde = 1.96 * sd / sqrt(n) if n else 0.0
        print(f"{post:<10} {pre:<10} {med(d):>+9.3f} {mu:>+9.3f} {sd:>7.3f} "
              f"{mde:>8.3f} {str(k) + '/' + str(n):>9} {sign_p(k, n):>9.4f}")


if __name__ == "__main__":
    main()
