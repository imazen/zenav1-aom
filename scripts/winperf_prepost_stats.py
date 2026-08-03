#!/usr/bin/env python3
"""winperf_prepost_stats.py — read a `arms: prepost` band the way its shape allows.

`.github/workflows/winperf.yml`'s `prepost` mode builds FIVE arms from TWO
binaries: `pre`/`l3a`/`l3` are three copies of the base and `post`/`postB` are
two copies of the change.  `eprof_ab_stats.py --vs pre` reports every arm against
ONE of them, which throws away most of that: it compares a 2-copy side with a
1-copy side and inherits whatever `pre` alone happened to do.

This reads the same TSV with the copies pooled:

  effect        per round, median(post-side) / median(pre-side) - 1, then the
                median over rounds.  Both sides get the same treatment, so a
                within-round position drift affects both and largely cancels.
  null_pre      the same statistic computed between two disjoint halves of the
                PRE side alone, over every way of splitting its copies.  It is a
                measurement of this band's noise, from binaries that are
                identical by construction.
  null_post     ditto on the post side.
  noise floor   max(|null_pre|, |null_post|) — what `effect` has to clear.
  sign          how many rounds have post-side < pre-side, out of n, with the
                two-sided binomial p.  A median can be dragged by one bad round;
                a sign count cannot.

  winperf_prepost_stats.py <band.tsv> --pre pre,l3a,l3 --post post,postB
"""

import itertools
import math
import sys
from collections import OrderedDict


def med(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def binom_two_sided(k, n):
    """P(|X - n/2| >= |k - n/2|) for X ~ Binomial(n, 1/2)."""
    c = lambda a, b: math.comb(a, b)
    d = abs(k - n / 2)
    return sum(c(n, i) for i in range(n + 1) if abs(i - n / 2) >= d - 1e-9) / 2**n


def load(path):
    byround = OrderedDict()
    with open(path) as f:
        hdr = f.readline()
        assert hdr.split("\t")[0] == "arm", hdr
        for line in f:
            arm, rnd, m, _ns, _by = line.rstrip("\n").split("\t")
            byround.setdefault(int(rnd), {})[arm] = int(m) / 1e6
    return byround


def ratio(byround, a, b):
    """Per-round median(a)/median(b) - 1, in percent; returns the list."""
    out = []
    for r in sorted(byround):
        row = byround[r]
        if not all(x in row for x in a + b):
            continue
        out.append(100.0 * (med([row[x] for x in a]) / med([row[x] for x in b]) - 1.0))
    return out


def within_side_null(byround, arms):
    """Largest |median ratio| over every way of splitting `arms` in two."""
    worst = 0.0
    n = len(arms)
    for k in range(1, n):
        for left in itertools.combinations(arms, k):
            right = [x for x in arms if x not in left]
            if not right:
                continue
            worst = max(worst, abs(med(ratio(byround, list(left), right))))
    return worst


def main():
    path = sys.argv[1]
    pre = ["pre", "l3a", "l3"]
    post = ["post", "postB"]
    a = sys.argv[2:]
    for i, tok in enumerate(a):
        if tok == "--pre":
            pre = a[i + 1].split(",")
        if tok == "--post":
            post = a[i + 1].split(",")

    byround = load(path)
    present = set().union(*(set(v) for v in byround.values()))
    pre = [x for x in pre if x in present]
    post = [x for x in post if x in present]
    rs = ratio(byround, post, pre)
    n = len(rs)
    wins = sum(1 for x in rs if x < 0)
    null_pre = within_side_null(byround, pre) if len(pre) > 1 else float("nan")
    null_post = within_side_null(byround, post) if len(post) > 1 else float("nan")
    floor = max(x for x in (null_pre, null_post) if x == x)

    print(f"file          {path}")
    print(f"pre  side     {','.join(pre)}")
    print(f"post side     {','.join(post)}")
    print(f"rounds        {n}")
    print(f"null_pre      {null_pre:+.3f} %   (worst split of identical copies)")
    print(f"null_post     {null_post:+.3f} %")
    print(f"NOISE FLOOR   {floor:.3f} %")
    print(f"effect        {med(rs):+.3f} %   (per-round paired median)")
    print(f"vs floor      {abs(med(rs)) / floor:.2f}x")
    print(f"sign          {wins}/{n} rounds post-side faster, p = {binom_two_sided(wins, n):.4f}")
    print(f"per-round     {' '.join(f'{x:+.2f}' for x in rs)}")


if __name__ == "__main__":
    main()
