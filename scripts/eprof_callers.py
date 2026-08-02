#!/usr/bin/env python3
"""eprof_callers.py — attribute a class of leaf cost (allocator, memset,
memcpy, ...) back to the call sites that CAUSE it.

The stage rollup answers "how much time is in malloc"; it does not answer "whose
malloc". This walks the same `sample` call graph and charges every node whose
symbol matches `--leaf` to its nearest ancestor that does NOT match, which is
the code responsible for the traffic. It is the sampling analogue of the
callgrind "attribute shared libc leaves through their caller edges" step in
`benchmarks/gate3_decode_profile_2026-07-19.md`.

Usage:
    eprof_callers.py <sample.txt> --leaf 'malloc|free|realloc|memset|memcpy' [--top 25]
"""
import argparse
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eprof_rollup as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample")
    ap.add_argument("--leaf", required=True)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--tsv")
    a = ap.parse_args()
    pat = re.compile(a.leaf)

    nodes, total = R.parse(a.sample)
    # Rebuild the ancestor chain the same way parse() did.
    charged = defaultdict(int)
    stack = []   # (depth, pretty_name, is_leafclass)
    for d, c, sym, _img, _src, _slf, _outer in nodes:
        while stack and stack[-1][0] >= d:
            stack.pop()
        name = R.pretty(sym)
        leafy = bool(pat.search(name))
        if leafy:
            # charge this node's INCLUSIVE cost to the nearest non-matching
            # ancestor, but only when we are ENTERING the leaf class (the
            # parent is not itself in the class), so nested libc frames are
            # not double counted.
            if not (stack and stack[-1][2]):
                blame = next((n for _dd, n, l in reversed(stack) if not l), "<root>")
                charged[blame] += c
        stack.append((d, name, leafy))

    tot = sum(charged.values())
    print(f"# {a.sample}: {total} samples; leaf class /{a.leaf}/ = {tot} "
          f"inclusive samples ({100 * tot / total:.2f}% of the window)")
    print(f"{'samples':>9}{'% window':>10}{'% of class':>12}  caller")
    rows = sorted(charged.items(), key=lambda kv: -kv[1])[: a.top]
    for name, v in rows:
        print(f"{v:>9}{100 * v / total:>9.2f}%{100 * v / tot:>11.2f}%  {name}")
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("caller\tsamples\tpct_window\tpct_of_class\n")
            for name, v in sorted(charged.items(), key=lambda kv: -kv[1]):
                f.write(f"{name}\t{v}\t{100 * v / total:.4f}\t{100 * v / tot:.4f}\n")
        print(f"# wrote {a.tsv}")


if __name__ == "__main__":
    main()
