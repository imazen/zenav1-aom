#!/usr/bin/env python3
"""eprof_align.py — put the two arms' stage rollups side by side in ABSOLUTE
milliseconds per encode, which is the only form in which they are comparable.

A share is a share of a different denominator on each side (the port's encode is
~10x longer), so "stage X is 5% of the port and 10% of libaom" says nothing on
its own.  Multiplying each side's self-share by ITS OWN measured ms/encode puts
both on the same axis, and the per-stage port/C ratio and absolute gap fall out
— the same shape as the module rollup in
`benchmarks/gate3_decode_profile_2026-07-19.md`.

Usage:
    eprof_align.py --port port.tsv --port-ms 511.69 \
                   --c libaom.tsv  --c-ms 48.171 [--out aligned.tsv]
"""
import argparse
import csv


def stages(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["kind"] == "stage":
                out[r["stage"]] = float(r["self_pct"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--c", required=True)
    ap.add_argument("--port-ms", type=float, required=True)
    ap.add_argument("--c-ms", type=float, required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    p, c = stages(a.port), stages(a.c)
    rows = []
    for st in sorted(set(p) | set(c)):
        pms = p.get(st, 0.0) / 100.0 * a.port_ms
        cms = c.get(st, 0.0) / 100.0 * a.c_ms
        rows.append((st, pms, p.get(st, 0.0), cms, c.get(st, 0.0), pms - cms,
                     (pms / cms) if cms > 1e-9 else float("inf")))
    rows.sort(key=lambda r: -r[5])

    tot_gap = a.port_ms - a.c_ms
    print(f"# port {a.port_ms:.2f} ms/encode   libaom-c {a.c_ms:.2f} ms/encode   "
          f"ratio {a.port_ms / a.c_ms:.2f}x   absolute gap {tot_gap:.2f} ms")
    print()
    hdr = (f"{'stage':<24}{'port ms':>9}{'%':>7}{'C ms':>9}{'%':>7}"
           f"{'gap ms':>9}{'% of gap':>10}{'port/C':>9}")
    print(hdr)
    print("-" * len(hdr))
    for st, pms, pp, cms, cp, gap, ratio in rows:
        rs = "inf" if ratio == float("inf") else f"{ratio:.2f}x"
        print(f"{st:<24}{pms:>9.2f}{pp:>7.2f}{cms:>9.2f}{cp:>7.2f}"
              f"{gap:>9.2f}{100 * gap / tot_gap:>9.1f}%{rs:>9}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<24}{a.port_ms:>9.2f}{100:>7.2f}{a.c_ms:>9.2f}{100:>7.2f}"
          f"{tot_gap:>9.2f}{100.0:>9.1f}%{a.port_ms / a.c_ms:>8.2f}x")

    if a.out:
        with open(a.out, "w") as f:
            f.write("stage\tport_ms\tport_self_pct\tc_ms\tc_self_pct\tgap_ms\t"
                    "pct_of_total_gap\tport_over_c\n")
            for st, pms, pp, cms, cp, gap, ratio in rows:
                rs = "" if ratio == float("inf") else f"{ratio:.4f}"
                f.write(f"{st}\t{pms:.4f}\t{pp:.4f}\t{cms:.4f}\t{cp:.4f}\t"
                        f"{gap:.4f}\t{100 * gap / tot_gap:.4f}\t{rs}\n")
            f.write(f"TOTAL\t{a.port_ms:.4f}\t100\t{a.c_ms:.4f}\t100\t"
                    f"{tot_gap:.4f}\t100\t{a.port_ms / a.c_ms:.4f}\n")
        print(f"\n# wrote {a.out}")


if __name__ == "__main__":
    main()
