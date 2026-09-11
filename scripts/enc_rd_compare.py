#!/usr/bin/env python3
"""enc_rd_compare — the RATE/QUALITY half of the cross-encoder comparison.

`benchmarks/xbench/encbench` (zenbench) answers "how fast"; this answers "how
many bytes, at what quality", which is the half a speed chart cannot carry.

Four encoders, each swept along ITS OWN quality ladder, every output decoded by
ONE decoder and scored in RGB:

  zenav1-aom   this repo's `encode_key_frame` -- the SHIPPING path
  zenav1-svt   the pure-Rust SVT-AV1 port, sibling repo at main
  ravif        CRATES.IO `ravif` (the cavif engine over upstream rav1e)
  zenrav1e     the in-house rav1e fork

WHY THE SCORING GOES THROUGH ONE DECODER. The arms do not agree on chroma
format or bit depth: `ravif` always codes 4:4:4 (av1encoder.rs:414, "subsampling
is a bad idea for AVIF anyway") and defaults to 10-bit, while every other arm
codes 8-bit 4:2:0. A YUV-plane comparison cannot even be written between them.
`xtool decode` reads each stream's OWN signalled matrix coefficients and range,
converts to 8-bit RGB, and `xtool score-rgb` scores SSIMULACRA2 + butteraugli
against the same reference RGB. Format differences then show up where they
belong -- in the rate needed for a given quality -- instead of as a scoring
artifact.

WHAT THE REFERENCE IS, stated because it bounds every number here: the source
is a PNG downsampled to an I420 `.yuv` by `xtool prep`, and the reference RGB
is that `.yuv` converted back. So all four arms are asked the same question --
"given this 4:2:0 source, what do you produce" -- and ravif's 4:4:4 cannot
recover chroma detail the source no longer has. That is the normal still-image
pipeline, and it is the convention every other driver in this harness already
follows; it is NOT a claim about ravif on a 4:4:4 source.

Usage:
  scripts/enc_rd_compare.py sweep  --out benchmarks/enc_rd_<date>.tsv [--size 512]
  scripts/enc_rd_compare.py chart  --tsv <file> [--out <file.md>]
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XB = ROOT / "benchmarks" / "xbench"
BIN = XB / "target" / "release"
ZEN = Path(os.environ.get("ZEN", Path.home() / "work" / "zen"))
PNG = ZEN / "imazen-26-png-v3" / "png-v3"
WORK = Path(os.environ.get("RD_WORK", Path.home() / "tmp" / "encrd"))

# The corpus split the comparison is reported over. Photographic and
# screen-shaped content are kept SEPARATE rather than averaged: they are the
# two regimes where AV1 encoders differ most (palette and IntraBC exist for the
# second and do nothing for the first), and a single pooled curve would hide
# exactly the effect worth showing.
CLASSES = {
    "photo": ["1000-lilith-photos-general", "1400-lilith-nature"],
    "screen": ["8000-lilith-mobile-screenshots", "9000-lilith-ai-clipart"],
}

# Each arm's own QUALITY ladder and its own SPEED ladder. Neither axis is
# interchangeable between encoders -- ravif takes a 1..100 quality, rav1e a
# 0..255 quantizer, the other two a 0..63 index; and "preset 6" means something
# different in each codebase. That is exactly why this produces CURVES and a
# Pareto frontier rather than a table of matched numbers.
#
# **The speed axis is not optional, and leaving it out is the single easiest way
# to publish a misleading encoder chart.** Measured on the 512x512 study image:
# zenav1-aom at its shipping `--cpu-used 3` takes 1.39 s while zenav1-svt at
# preset 6 takes 60 ms. Charted side by side that reads as a 23x gap; it is
# mostly a statement about which operating point each was asked for. Only a
# (time, quality-at-matched-rate) frontier compares them honestly.
LADDERS = {
    # Quality points chosen to span roughly SSIMULACRA2 50..90 -- the band a
    # still-image pipeline actually ships in. A ladder that runs off the bottom
    # produces points nobody would deploy and drags the interpolation with it.
    "zenav1-aom": ("drv-aom", [10, 18, 26, 34], [3, 6, 8]),
    "zenav1-svt": ("drv-svtrs", [10, 18, 26, 34], [2, 6, 10]),
    "ravif": ("drv-ravif", [92, 84, 74, 62], [4, 6, 8]),
    "zenrav1e": ("drv-rav1e", [50, 80, 110, 140], [4, 6, 8]),
}

# zenbench's `phosphor` scheme, so the two charts read as one set.
SCHEME = {
    "bkg": "080808", "title": "#33ff66", "tick": "#999999",
    "grid": "#1a1a1a", "zero": "#333333", "legend": "#cccccc",
}
ARM_COLOR = {
    "zenav1-aom": "#00ff41",
    "zenav1-svt": "#2196f3",
    "ravif": "#ff9800",
    "zenrav1e": "#bb9af7",
}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def images(limit_per_class):
    """One deterministic sample per class: sorted, first N. The choice is
    recorded in the TSV so a re-run is reproducible without this function."""
    out = {}
    for cls, dirs in CLASSES.items():
        got = []
        for d in dirs:
            p = PNG / d
            if not p.is_dir():
                continue
            # LFS pointers are ~130 bytes; a real render is megabytes. Skipping
            # by SIZE rather than by existence is what keeps a partially-pulled
            # checkout from silently shrinking the corpus.
            for f in sorted(p.glob("*.png")):
                if f.stat().st_size > 100_000:
                    got.append(f)
        out[cls] = got[:limit_per_class]
    return out


def sweep(args):
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    picks = images(args.per_class)
    for cls, files in picks.items():
        if not files:
            print(f"warning: no local renders for class {cls} "
                  f"(git lfs pull the imazen-26-png-v3 subset)", file=sys.stderr)
        for img in files:
            stem = img.stem.split("_")[0]
            ref_yuv = WORK / f"{stem}.yuv"
            ref_rgb = WORK / f"{stem}.rgb"
            run([str(BIN / "xtool"), "prep", str(img), str(ref_yuv),
                 f"square:{args.size}"])
            run([str(BIN / "xtool"), "yuv2rgb", str(ref_yuv), str(ref_rgb),
                 str(args.size), str(args.size)])
            for arm, (drv, ladder, speeds) in LADDERS.items():
                for speed in speeds:
                    for q in ladder:
                        tag = f"{arm}.s{speed}.{q}"
                        out = WORK / f"{stem}.{tag}.bin"
                        line = run([str(BIN / drv), str(args.size), str(args.size),
                                    str(q), str(speed), str(ref_yuv), str(out),
                                    "0", "1"]).strip()
                        ns = [int(t[3:]) for t in line.split() if t.startswith("NS=")]
                        by = next(int(t[6:]) for t in line.split()
                                  if t.startswith("BYTES="))
                        dec = WORK / f"{stem}.{tag}.rgb"
                        info = run([str(BIN / "xtool"), "decode", str(out),
                                    str(dec)]).strip()
                        sc = run([str(BIN / "xtool"), "score-rgb", str(ref_rgb),
                                  str(dec), str(args.size), str(args.size)]).strip()
                        kv = dict(t.split("=") for t in (info + " " + sc).split())
                        rows.append({
                            "class": cls, "image": stem, "arm": arm, "q": q,
                            "speed": speed, "bytes": by,
                            "ms": min(ns) / 1e6,
                            "bpp": by * 8 / (args.size * args.size),
                            "ssim2": float(kv["SSIM2"]),
                            "ba_max": float(kv["BA_MAX"]),
                            "bd": kv["BD"], "ss": kv["SS"], "mc": kv["MC"],
                        })
                        print(f"{cls}\t{stem}\t{arm}\ts{speed}\tq{q}\t{by}B\t"
                              f"{rows[-1]['ms']:.1f}ms\t"
                              f"ssim2={rows[-1]['ssim2']:.2f}", flush=True)
    cols = list(rows[0].keys())
    with open(args.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {args.out} ({len(rows)} rows)")


def _chart_url(title, series, xlabel, ylabel):
    cfg = {
        "type": "line",
        "data": {"datasets": [
            {"label": name,
             "data": [{"x": round(x, 4), "y": round(y, 3)} for x, y in pts],
             "borderColor": ARM_COLOR.get(name, "#666666"),
             "backgroundColor": ARM_COLOR.get(name, "#666666"),
             "fill": False, "borderWidth": 3, "pointRadius": 4}
            for name, pts in series.items()]},
        "options": {
            "plugins": {"datalabels": {"display": False}},
            "scales": {
                "xAxes": [{"type": "linear", "position": "bottom",
                           "scaleLabel": {"display": True, "labelString": xlabel,
                                          "fontColor": SCHEME["tick"], "fontSize": 16},
                           "ticks": {"fontColor": SCHEME["tick"], "fontSize": 14},
                           "gridLines": {"color": SCHEME["grid"],
                                         "zeroLineColor": SCHEME["zero"]}}],
                "yAxes": [{"scaleLabel": {"display": True, "labelString": ylabel,
                                          "fontColor": SCHEME["tick"], "fontSize": 16},
                           "ticks": {"fontColor": SCHEME["tick"], "fontSize": 14},
                           "gridLines": {"color": SCHEME["grid"]}}]},
            "legend": {"display": True, "position": "bottom",
                       "labels": {"fontColor": SCHEME["legend"], "fontSize": 16}},
            "title": {"display": True, "text": title, "fontColor": SCHEME["title"],
                      "fontSize": 20, "padding": 8},
        },
    }
    q = urllib.parse.urlencode({
        "w": 760, "h": 420, "bkg": "#" + SCHEME["bkg"], "f": "png",
        "c": json.dumps(cfg, separators=(",", ":")),
    })
    return "https://quickchart.io/chart?" + q


def _interp_ssim_at(pts, bpp):
    """Linear interpolation of SSIMULACRA2 at a target bits-per-pixel.

    Returns None outside the arm's measured range rather than extrapolating —
    an extrapolated point on a cross-encoder chart is a fabricated claim about
    an operating point nobody measured.
    """
    pts = sorted(pts)
    if len(pts) < 2 or bpp < pts[0][0] or bpp > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= bpp <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (bpp - x0) / (x1 - x0)
    return None


def chart(args):
    rows = []
    with open(args.tsv) as f:
        cols = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(dict(zip(cols, line.rstrip("\n").split("\t"))))
    out = []
    for cls in sorted({r["class"] for r in rows}):
        crows = [r for r in rows if r["class"] == cls]
        n_img = len({r["image"] for r in crows})

        # --- Chart 1: RD curves, each arm at its MIDDLE speed preset. -------
        # Mean over the class's images at each ladder point, so one image
        # cannot carry a curve.
        rd = {}
        for arm, (_, _, speeds) in LADDERS.items():
            mid = speeds[len(speeds) // 2]
            byq = {}
            for r in crows:
                if r["arm"] != arm or int(r["speed"]) != mid:
                    continue
                byq.setdefault(r["q"], []).append((float(r["bpp"]), float(r["ssim2"])))
            pts = [(sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v))
                   for _, v in sorted(byq.items(), key=lambda kv: float(kv[0]))]
            if pts:
                rd[f"{arm} s{mid}"] = sorted(pts)
        t1 = f"AV1 still RD — {cls} ({n_img} images; up and left is better)"
        out.append(f"![{t1}]({_chart_url(t1, rd, 'bits per pixel', 'SSIMULACRA2')})\n")

        # --- Chart 2: the SPEED/QUALITY Pareto at a matched rate. ----------
        # For every (arm, speed) the curve is interpolated to ONE common
        # bits-per-pixel, so the axis is "quality per millisecond at equal
        # rate" rather than "whatever each preset happened to spend". The
        # target bpp is the median of every arm's own measured range, so no
        # arm is extrapolated to reach it.
        lo, hi = [], []
        percell = {}
        for arm, (_, _, speeds) in LADDERS.items():
            for sp in speeds:
                byq = {}
                for r in crows:
                    if r["arm"] != arm or int(r["speed"]) != sp:
                        continue
                    byq.setdefault(r["q"], []).append(
                        (float(r["bpp"]), float(r["ssim2"]), float(r["ms"])))
                pts = [(sum(x for x, _, _ in v) / len(v),
                        sum(y for _, y, _ in v) / len(v),
                        sum(m for _, _, m in v) / len(v))
                       for _, v in sorted(byq.items(), key=lambda kv: float(kv[0]))]
                if len(pts) >= 2:
                    percell[(arm, sp)] = sorted(pts)
                    lo.append(min(x for x, _, _ in pts))
                    hi.append(max(x for x, _, _ in pts))
        if percell:
            target = (max(lo) + min(hi)) / 2 if max(lo) <= min(hi) else \
                sorted(x for v in percell.values() for x, _, _ in v)[len(percell) // 2]
            pareto = {}
            for (arm, sp), pts in percell.items():
                q = _interp_ssim_at([(x, y) for x, y, _ in pts], target)
                if q is None:
                    continue
                ms = _interp_ssim_at([(x, m) for x, _, m in pts], target)
                pareto.setdefault(arm, []).append((ms, q))
            for k in pareto:
                pareto[k] = sorted(pareto[k])
            t2 = (f"AV1 still speed/quality frontier — {cls} "
                  f"at {target:.2f} bpp (up and LEFT is better)")
            out.append(
                f"![{t2}]({_chart_url(t2, pareto, 'encode ms (log-ish, lower better)', 'SSIMULACRA2')})\n"
            )
            out.append(f"<!-- matched-rate target {target:.4f} bpp; "
                       f"{n_img} images; arms outside their measured bpp range are "
                       f"omitted rather than extrapolated -->\n")
    md = "\n".join(out)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    else:
        print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--out", required=True)
    s.add_argument("--size", type=int, default=512)
    s.add_argument("--per-class", type=int, default=3)
    s.set_defaults(fn=sweep)
    c = sub.add_parser("chart")
    c.add_argument("--tsv", required=True)
    c.add_argument("--out")
    c.set_defaults(fn=chart)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
