#!/usr/bin/env python3
"""Summarise an eprof_ab.sh TSV: per-arm median/min/max/spread + paired ratios.

Reports the control band FIRST (docs/DIFFERENTIAL_PLAYBOOK.md §6): the spread of
each arm's per-invocation medians, so a delta can be read against the noise it
has to clear.  `--ref` names the denominator arm for the paired ratio column
(normally the C driver); `--vs` names a second port arm to difference against.

  eprof_ab_stats.py <ab.tsv> [--ref libaom-c] [--vs base]
"""

import sys
from collections import OrderedDict


def med(xs):
    s = sorted(xs)
    return s[(len(s) + 1) // 2 - 1]


def main():
    path = sys.argv[1]
    ref = None
    vs = None
    a = sys.argv[2:]
    for i, tok in enumerate(a):
        if tok == "--ref":
            ref = a[i + 1]
        if tok == "--vs":
            vs = a[i + 1]

    arms = OrderedDict()
    byround = {}
    bytecounts = {}
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        assert hdr[0] == "arm", hdr
        # `eprof_ab.sh` gained a `position` column when it gained ROTATE mode;
        # bands recorded before that have five columns and must still parse.
        cols = {name: i for i, name in enumerate(hdr)}
        for line in f:
            f_ = line.rstrip("\n").split("\t")
            arm, rnd, m, by = f_[cols["arm"]], f_[cols["round"]], f_[cols["median_ns"]], f_[cols["bytes"]]
            arms.setdefault(arm, []).append(int(m) / 1e6)
            byround.setdefault(arm, {})[int(rnd)] = int(m) / 1e6
            bytecounts.setdefault(arm, set()).add(int(by))

    print(f"{'arm':<14} {'median':>10} {'min':>10} {'max':>10} {'spread':>8} {'n':>3}  bytes")
    for arm, xs in arms.items():
        m, lo, hi = med(xs), min(xs), max(xs)
        print(
            f"{arm:<14} {m:>10.3f} {lo:>10.3f} {hi:>10.3f} "
            f"{100 * (hi - lo) / lo:>7.2f}% {len(xs):>3}  "
            f"{','.join(str(b) for b in sorted(bytecounts[arm]))}"
        )

    if ref and ref in arms:
        print()
        rm = med(arms[ref])
        for arm, xs in arms.items():
            if arm == ref:
                continue
            pair = sorted(
                byround[arm][r] / byround[ref][r]
                for r in byround[arm]
                if r in byround[ref]
            )
            print(
                f"{arm:<14} vs {ref}: median-of-medians ratio {med(xs) / rm:.4f}x  "
                f"paired {' '.join(f'{p:.3f}' for p in pair)}  "
                f"spread {100 * (pair[-1] - pair[0]) / pair[0]:.2f}%"
            )

    if vs and vs in arms:
        print()
        bm = med(arms[vs])
        for arm, xs in arms.items():
            if arm == vs:
                continue
            m = med(xs)
            pair = sorted(
                byround[arm][r] / byround[vs][r]
                for r in byround[arm]
                if r in byround[vs]
            )
            print(
                f"{arm:<14} vs {vs}: {m - bm:+.3f} ms ({100 * (m - bm) / bm:+.2f}%)  "
                f"paired-median {100 * (med(pair) - 1):+.2f}%"
            )


if __name__ == "__main__":
    main()
