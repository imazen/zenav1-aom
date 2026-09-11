#!/usr/bin/env python3
"""perf_band_stats.py — paired statistics for a perf_band.sh TSV.

Prints, per arm, median / min / n; then the two comparisons that decide a
landing: `new vs base` (the lever) and `baseB vs base` (the same-binary null the
lever has to clear). Paired per-round percentage deltas, median, count of rounds
faster, and a two-sided exact sign test. Also checks the `bytes` column agrees
across arms — a byte difference is an RD change and disqualifies the band as a
pure timing comparison.

  python3 scripts/perf_band_stats.py ~/tmp/arms/band.tsv [--new NAME] [--base NAME]
"""
import csv, sys
from collections import defaultdict
from math import comb

args = sys.argv[1:]
path = args[0]
def opt(flag, default):
    return args[args.index(flag) + 1] if flag in args else default
NEW, BASE, NULL = opt('--new', 'new'), opt('--base', 'base'), opt('--null', 'baseB')

rows = list(csv.DictReader(open(path), delimiter='\t'))
by = defaultdict(dict); bytes_by_arm = defaultdict(set)
for r in rows:
    by[int(r['round'])][r['arm']] = float(r['ms'])
    if r.get('bytes'): bytes_by_arm[r['arm']].add(r['bytes'])

def med(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

def signtest(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

arms = sorted({a for r in by.values() for a in r})
for a in arms:
    v = [by[r][a] for r in by if a in by[r]]
    print(f"{a:8s} median {med(v):9.2f} ms  min {min(v):9.2f}  n={len(v)}  bytes={sorted(bytes_by_arm[a])}")
allb = {b for s in bytes_by_arm.values() for b in s}
if len(allb) > 1:
    print(f"WARNING: arms emit different byte counts {sorted(allb)} — this is an RD change, not a timing band")

def cmp(a, b, label):
    d = [(by[r][b] - by[r][a]) / by[r][a] * 100 for r in by if a in by[r] and b in by[r]]
    if not d:
        print(f"{label}: no paired rounds"); return
    n = len(d); faster = sum(1 for x in d if x < 0)
    print(f"{label}: paired median {med(d):+.3f} %   {faster}/{n} rounds faster   p={signtest(faster, n):.4g}")

cmp(BASE, NEW, f'{NEW} vs {BASE}   ')
cmp(BASE, NULL, f'null ({NULL} vs {BASE})')
print("read: a lever counts only if |new| clears the null AND p < 0.05; two base copies routinely differ by ~0.25 pp")
