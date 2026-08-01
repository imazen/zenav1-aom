#!/usr/bin/env python3
"""xbench — cross-encoder AV1 still-picture benchmark orchestrator.

Five encoders, one input byte stream, one timing contract, one scorer:

  zenav1-aom  the pure-Rust libaom port   (this repo)  cq 0..63, cpu-used 0..9
  libaom-c    libaom, upstream C (the port's oracle)   cq 0..63, cpu-used 0..9
  svt-c       SVT-AV1, upstream C                      qp 0..63, preset 0..13
  svt-rust    zenav1-svt, pure-Rust SVT port           qp 0..63, preset 0..13
  zenrav1e    the rav1e fork                           quantizer 0..255, speed 0..10

Every driver implements the SAME contract (see benchmarks/xbench/*/src/main.rs
and benchmarks/xbench/csrc/*.c):

    drv <w> <h> <q> <speed> <in.yuv> <out.obu> <warmup> <reps>
    stdout: NS=<n> NS=<n> ... BYTES=<m>

and times ONLY its own frame-encode call — never process startup, never file
I/O, never its own constructor/init.  All five are SINGLE-THREADED.

Subcommands:
  prep     build the source .yuv ladders from the corpus
  stage1   throughput calibration: MP/s per encoder per preset + the
           alpha (fixed per-call cost) / beta (per-pixel cost) fit
  stage2   RD sweep at the qualifying modes
  stage3   quality-target accuracy (identical external search for all five)
  byteid   whole-stream sha256 of two encoders over the RD corpus
"""

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XB = ROOT / "benchmarks" / "xbench"
BIN = XB / "target" / "release"
CBIN = XB / "target"
ZEN = Path(os.environ.get("ZEN", Path.home() / "work" / "zen"))
CORPUS = ZEN / "codec-corpus"
WORK = Path(os.environ.get("XBENCH_WORK", Path.home() / "tmp" / "xb"))

XTOOL = BIN / "xtool"

# label -> (driver path, q-range, speed-range, q-scale note)
ENCODERS = {
    "zenav1-aom": (BIN / "drv-aom", (1, 63), list(range(0, 10))),
    # The C encoder zenav1-aom is a port of, driven at the IDENTICAL config the
    # port's own bootstrap uses (drv_libaom.c transcribes
    # shim_encode_av1_kf_defaults). Added 2026-08-01 as the decomposition arm
    # the original four-encoder study listed as its biggest gap.
    "libaom-c": (CBIN / "drv_libaom", (1, 63), list(range(0, 10))),
    "svt-c": (CBIN / "drv_svtc", (1, 63), list(range(0, 14))),
    "svt-rust": (BIN / "drv-svtrs", (1, 63), list(range(0, 14))),
    "zenrav1e": (BIN / "drv-rav1e", (1, 255), list(range(0, 11))),
}

NS_RE = re.compile(rb"NS=(\d+)")
# `(?<![A-Za-z])` so drv-aom's extra `FRAMEBYTES=` field can never be mistaken
# for the comparable whole-stream `BYTES=`.
BYTES_RE = re.compile(rb"(?<![A-Za-z])BYTES=(\d+)")


def run_encode(enc, w, h, q, speed, yuv, out, warmup, reps, timeout=1800):
    """One driver invocation. Returns (list_of_ns, coded_bytes) or None on failure."""
    drv, _, _ = ENCODERS[enc]
    cmd = [str(drv), str(w), str(h), str(q), str(speed), str(yuv), str(out),
           str(warmup), str(reps)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    if p.returncode != 0:
        err = (p.stderr or b"")[-400:].decode("utf-8", "replace").strip()
        return None, None, f"rc={p.returncode} {err}"
    blob = p.stdout + p.stderr
    ns = [int(m) for m in NS_RE.findall(blob)]
    by = BYTES_RE.findall(blob)
    if not ns or not by:
        return None, None, "no NS/BYTES in output"
    return ns, int(by[-1]), None


# --------------------------------------------------------------------- prep --

# The size ladder. 64-aligned squares so NO encoder is handicapped by a
# partial-superblock path, and so the only thing changing across the ladder is
# pixel count.
PHOTO_SRC = CORPUS / "clic2025" / "final-test" / \
    "8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png"   # 2048x2048
SCREEN_SRC = CORPUS / "gb82-sc" / "imac_dark.png"                             # 2940x1912
LADDERS = {
    "photo": (PHOTO_SRC, [64, 256, 1024, 2048]),
    "screen": (SCREEN_SRC, [64, 256, 1024, 1856]),
}


def cmd_prep(_args):
    (WORK / "src").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (src, sizes) in LADDERS.items():
        for n in sizes:
            out = WORK / "src" / f"{name}_{n}.yuv"
            r = subprocess.run([str(XTOOL), "prep", str(src), str(out), f"square:{n}"],
                               capture_output=True, check=True)
            rows.append((name, n, out, r.stdout.decode().strip()))
            print(f"{name:8s} {n:5d}  {out}  {r.stdout.decode().strip()}")
    return rows


# ------------------------------------------------------------------- stage1 --

# Reps per size: enough samples that the median is stable without the tiny
# sizes taking a whole second of wall clock each.
def reps_for(px):
    if px <= 8192:
        return 60
    if px <= 100_000:
        return 20
    if px <= 1_200_000:
        return 7
    return 5


def median_ns(ns):
    return statistics.median(ns)


def selected(args):
    """The encoders this invocation touches: all of ENCODERS unless --only.

    `--only` exists so a NEW arm can be added to a published study without
    re-running (and thereby perturbing) the arms already measured.
    """
    if not getattr(args, "only", ""):
        return list(ENCODERS)
    want = [e.strip() for e in args.only.split(",") if e.strip()]
    for e in want:
        if e not in ENCODERS:
            raise SystemExit(f"--only: unknown encoder {e!r}; known: {', '.join(ENCODERS)}")
    return want


def cmd_stage1(args):
    """Screen every preset at 1 MP, then fit alpha+beta over the full ladder."""
    out_tsv = Path(args.out)
    raw_tsv = out_tsv.with_suffix(".raw.tsv")
    qmap = json.loads(args.q)
    only = selected(args)
    rawf = open(raw_tsv, "w")
    rawf.write("phase\tencoder\tpreset\tcontent\tsize\tpixels\tq\trep\tns\tbytes\n")

    # ---- phase A: screen at the medium (1024x1024 = 1.05 MP) photo cell.
    screen = {}
    for enc, (_drv, _qr, presets) in ((e, ENCODERS[e]) for e in only):
        q = qmap[enc]
        yuv = WORK / "src" / "photo_1024.yuv"
        for preset in sorted(presets, reverse=True):   # fastest first
            t0 = time.time()
            ns, by, err = run_encode(enc, 1024, 1024, q, preset, yuv,
                                     WORK / "s1.obu", 1, 3, timeout=args.timeout)
            if err:
                print(f"  {enc:11s} preset {preset:2d}  FAILED: {err}")
                screen[(enc, preset)] = (None, None, err)
                continue
            m = median_ns(ns)
            mps = (1024 * 1024) / (m / 1e9) / 1e6
            screen[(enc, preset)] = (m, by, None)
            for i, v in enumerate(ns):
                rawf.write(f"screen\t{enc}\t{preset}\tphoto\t1024\t{1024*1024}\t{q}\t{i}\t{v}\t{by}\n")
            rawf.flush()
            print(f"  {enc:11s} preset {preset:2d}  {m/1e6:9.2f} ms  {mps:8.3f} MP/s  "
                  f"{by:8d} B   (wall {time.time()-t0:.1f}s)")
            # Descend only while there is any hope of clearing the 1 MP/s bar.
            if mps < args.floor:
                print(f"  {enc:11s} preset {preset:2d} is below {args.floor} MP/s — "
                      f"stopping the descent (slower presets are strictly slower)")
                break

    # ---- phase B: full size ladder for every preset that CLEARED 1 MP/s at
    # 1 MP, plus the first preset below it (so the frontier is bracketed).
    fits = {}
    for enc, (_drv, _qr, presets) in ((e, ENCODERS[e]) for e in only):
        q = qmap[enc]
        ok = [p for p in presets if screen.get((enc, p), (None,))[0] is not None
              and (1024 * 1024) / (screen[(enc, p)][0] / 1e9) / 1e6 >= 1.0]
        if not ok:
            print(f"{enc}: NO preset cleared 1 MP/s at 1 MP")
            continue
        frontier = min(ok)
        below = [p for p in presets if p < frontier and screen.get((enc, p), (None,))[0]]
        chosen = sorted(set(ok + ([max(below)] if below else [])))
        if args.only_frontier:
            chosen = [frontier] + ([max(below)] if below else [])
        for preset in chosen:
            for content, (_src, sizes) in LADDERS.items():
                pts = []
                for n in sizes:
                    yuv = WORK / "src" / f"{content}_{n}.yuv"
                    px = n * n
                    ns, by, err = run_encode(enc, n, n, q, preset, yuv,
                                             WORK / "s1.obu", 2, reps_for(px),
                                             timeout=args.timeout)
                    if err:
                        print(f"  {enc} p{preset} {content} {n}: FAILED {err}")
                        continue
                    m = median_ns(ns)
                    for i, v in enumerate(ns):
                        rawf.write(f"ladder\t{enc}\t{preset}\t{content}\t{n}\t{px}\t{q}\t{i}\t{v}\t{by}\n")
                    rawf.flush()
                    pts.append((px, m / 1e9, by, n))
                    print(f"  {enc:11s} p{preset:2d} {content:6s} {n:5d}  "
                          f"{m/1e6:9.3f} ms  {px/(m/1e9)/1e6:8.3f} MP/s  {by} B")
                if len(pts) >= 2:
                    fits[(enc, preset, content)] = fit_alpha_beta(pts)
    rawf.close()

    with open(out_tsv, "w") as f:
        f.write("encoder\tpreset\tcontent\talpha_ms\tbeta_ms_per_MP\tmps_at_1MP\t"
                "mps_tiny_64\tmps_large\tr2\tpoints\n")
        for (enc, preset, content), fit in sorted(fits.items()):
            f.write(f"{enc}\t{preset}\t{content}\t{fit['alpha_ms']:.6f}\t"
                    f"{fit['beta_ms_per_mp']:.4f}\t{fit['mps_1mp']:.4f}\t"
                    f"{fit['mps_tiny']:.4f}\t{fit['mps_large']:.4f}\t{fit['r2']:.6f}\t"
                    f"{fit['points']}\n")
    print(f"\nwrote {out_tsv}\nwrote {raw_tsv}")


def fit_alpha_beta(pts):
    """Least-squares total_seconds = alpha + beta * pixels over the ladder."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    beta = sxy / sxx if sxx else 0.0
    alpha = my - beta * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (alpha + beta * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    small = min(pts, key=lambda p: p[0])
    big = max(pts, key=lambda p: p[0])
    return {
        "alpha_ms": alpha * 1e3,
        "beta_ms_per_mp": beta * 1e6 * 1e3,
        "mps_1mp": 1e6 / (alpha + beta * 1e6) / 1e6,
        "mps_tiny": small[0] / small[1] / 1e6,
        "mps_large": big[0] / big[1] / 1e6,
        "r2": r2,
        "points": ";".join(f"{p[3]}:{p[1]*1e3:.4f}ms" for p in sorted(pts)),
    }


# ------------------------------------------------------- stage2 / stage3 ----

# The RD / target corpus. Real content across classes, 64-aligned so no encoder
# takes a partial-superblock penalty. Screen content is CENTER-CROPPED, never
# resampled — downscaling destroys exactly the text/AA statistics that make it
# screen content.  NOT Kodak, no synthetic gradients (both banned).
CID22 = CORPUS / "CID22" / "CID22-512" / "validation"
CLIC = CORPUS / "clic2025" / "final-test"
SC = CORPUS / "gb82-sc"
RD_CORPUS = [
    # (label, class, source png, prep mode, w, h)
    ("cid22_1025469", "photo", CID22 / "1025469.png", "square:512", 512, 512),
    ("cid22_1420710", "photo", CID22 / "1420710.png", "square:512", 512, 512),
    ("cid22_1624487", "photo", CID22 / "1624487.png", "square:512", 512, 512),
    ("cid22_225228", "photo", CID22 / "225228.png", "square:512", 512, 512),
    ("cid22_2775196", "photo", CID22 / "2775196.png", "square:512", 512, 512),
    ("cid22_3316926", "photo", CID22 / "3316926.png", "square:512", 512, 512),
    ("cid22_382297", "photo", CID22 / "382297.png", "square:512", 512, 512),
    ("cid22_6292444", "photo", CID22 / "6292444.png", "square:512", 512, 512),
    ("clic_8426ed22", "photo-hr", CLIC / "8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png", "square:1024", 1024, 1024),
    ("clic_ebfd571f", "photo-hr", CLIC / "ebfd571f1c6824316047a29cb5f376eec15f56dd51821119c1842be068a8b950.png", "square:1024", 1024, 1024),
    ("sc_imac_dark", "screen", SC / "imac_dark.png", "crop:1024x1024", 1024, 1024),
    ("sc_terminal", "screen", SC / "terminal.png", "crop:1024x1024", 1024, 1024),
    ("sc_codec_wiki", "screen", SC / "codec_wiki.png", "crop:1024x1024", 1024, 1024),
]

# 14-point quality grids. Uniform over each encoder's FULL native quantizer
# range, so the aggressive (low-quality / high-quantizer) end is sampled at
# least as densely as the high-quality end — this is web-compression work.
QGRID = {
    "zenav1-aom": [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62],
    "svt-c": [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62],
    "svt-rust": [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62],
    "zenrav1e": [40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200, 216, 232, 248],
}


def prep_rd_corpus():
    d = WORK / "rd"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for label, cls, src, mode, w, h in RD_CORPUS:
        yuv = d / f"{label}.yuv"
        if not yuv.exists():
            r = subprocess.run([str(XTOOL), "prep", str(src), str(yuv), mode],
                               capture_output=True, check=True)
            got = r.stdout.decode()
            assert f"W={w} H={h}" in got, f"{label}: prep gave {got.strip()}, want {w}x{h}"
        out.append((label, cls, yuv, w, h))
    return out


def encode_score(enc, w, h, q, speed, yuv, tag, timeout=1800):
    """Encode one cell, decode it, score it. Returns dict or None."""
    tmp = WORK / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    obu = tmp / f"{tag}.obu"
    ivf = tmp / f"{tag}.ivf"
    dec = tmp / f"{tag}.yuv"
    ns, by, err = run_encode(enc, w, h, q, speed, yuv, obu, 0, 1, timeout=timeout)
    if err:
        return {"error": err}
    subprocess.run([str(XTOOL), "ivf", str(obu), str(ivf), str(w), str(h)],
                   capture_output=True, check=True)
    d = subprocess.run(["aomdec", "--rawvideo", "-o", str(dec), str(ivf)],
                       capture_output=True)
    if d.returncode != 0 or not dec.exists():
        return {"error": "DECODE_FAIL " + d.stderr[-200:].decode("utf-8", "replace")}
    s = subprocess.run([str(XTOOL), "score", str(yuv), str(dec), str(w), str(h)],
                       capture_output=True)
    if s.returncode != 0:
        return {"error": "SCORE_FAIL " + s.stderr[-200:].decode("utf-8", "replace")}
    o = s.stdout.decode()
    g = lambda k: float(re.search(rf"{k}=([-\d.]+)", o).group(1))
    return {"bytes": by, "ns": ns[0], "ssim2": g("SSIM2"),
            "ba_max": g("BA_MAX"), "ba_3n": g("BA_3N")}


def cmd_stage2(args):
    modes = json.loads(args.modes)          # {"encoder": preset}
    corpus = prep_rd_corpus()
    if args.classes:
        keep = set(args.classes.split(","))
        corpus = [c for c in corpus if c[1] in keep]
    qgrid = json.loads(args.qgrid) if args.qgrid else None
    out = Path(args.out)
    f = open(out, "w")
    f.write("encoder\tpreset\timage\tclass\tw\th\tq\tbytes\tbpp\tssim2\tba_max\tba_3n\tenc_ms\n")
    for label, cls, yuv, w, h in corpus:
        for enc, preset in modes.items():
            for q in (qgrid or QGRID[enc]):
                r = encode_score(enc, w, h, q, preset, yuv, "s2", timeout=args.timeout)
                if "error" in r:
                    print(f"  FAIL {enc} p{preset} {label} q{q}: {r['error']}", flush=True)
                    continue
                bpp = r["bytes"] * 8 / (w * h)
                f.write(f"{enc}\t{preset}\t{label}\t{cls}\t{w}\t{h}\t{q}\t{r['bytes']}\t"
                        f"{bpp:.6f}\t{r['ssim2']:.4f}\t{r['ba_max']:.4f}\t{r['ba_3n']:.4f}\t"
                        f"{r['ns']/1e6:.3f}\n")
                f.flush()
                print(f"  {enc:11s} p{preset:2d} {label:15s} q{q:3d}  "
                      f"{bpp:7.4f} bpp  ssim2 {r['ssim2']:7.3f}  ba3n {r['ba_3n']:6.3f}",
                      flush=True)
    f.close()
    print(f"wrote {out}")


# ---- Bjontegaard-delta rate, PCHIP (monotone cubic) over log10(rate) --------

def _pchip_slopes(x, y):
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    d = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0] = d[0]
    m[-1] = d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    return m


def _pchip_eval(x, y, m, t):
    lo, hi = 0, len(x) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= t:
            lo = mid
        else:
            hi = mid
    h = x[hi] - x[lo]
    s = (t - x[lo]) / h
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return h00 * y[lo] + h10 * h * m[lo] + h01 * y[hi] + h11 * h * m[hi]


def bd_rate(r_ref, m_ref, r_tst, m_tst):
    """% bitrate change of `tst` vs `ref` at equal quality. Negative = better."""
    import math
    def clean(r, m):
        pts = sorted({(mm, math.log10(rr)) for rr, mm in zip(r, m) if rr > 0})
        # strictly increasing metric
        out = []
        for mm, lr in pts:
            if not out or mm > out[-1][0]:
                out.append((mm, lr))
        return [p[0] for p in out], [p[1] for p in out]
    xa, ya = clean(r_ref, m_ref)
    xb, yb = clean(r_tst, m_tst)
    if len(xa) < 4 or len(xb) < 4:
        return None
    lo, hi = max(xa[0], xb[0]), min(xa[-1], xb[-1])
    if hi - lo <= 1e-9:
        return None
    ma, mb = _pchip_slopes(xa, ya), _pchip_slopes(xb, yb)
    N = 1000
    ia = ib = 0.0
    for i in range(N + 1):
        t = lo + (hi - lo) * i / N
        wt = 0.5 if i in (0, N) else 1.0
        ia += wt * _pchip_eval(xa, ya, ma, t)
        ib += wt * _pchip_eval(xb, yb, mb, t)
    ia /= N
    ib /= N
    return (10 ** (ib - ia) - 1) * 100


def cmd_bdrate(args):
    rows = []
    with open(args.tsv) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    metric = args.metric
    higher_better = metric == "ssim2"
    encs = sorted({r["encoder"] for r in rows})
    imgs = sorted({r["image"] for r in rows})
    cls = {r["image"]: r["class"] for r in rows}
    print(f"BD-rate vs {args.ref}  (metric = {metric}; negative = fewer bits at equal quality)")
    per = {e: [] for e in encs}
    for img in imgs:
        line = f"  {img:16s} {cls[img]:9s}"
        ref = [(float(r["bpp"]), float(r[metric])) for r in rows
               if r["image"] == img and r["encoder"] == args.ref]
        for e in encs:
            tst = [(float(r["bpp"]), float(r[metric])) for r in rows
                   if r["image"] == img and r["encoder"] == e]
            if e == args.ref or not tst or not ref:
                line += f"  {'--':>9s}"
                continue
            sgn = 1.0 if higher_better else -1.0
            v = bd_rate([p[0] for p in ref], [sgn * p[1] for p in ref],
                        [p[0] for p in tst], [sgn * p[1] for p in tst])
            if v is None:
                line += f"  {'n/a':>9s}"
            else:
                per[e].append((cls[img], v))
                line += f"  {v:+8.2f}%"
        print(line)
    print(f"\n  {'':26s}" + "".join(f"  {e:>9s}" for e in encs))
    print(f"  {'MEDIAN (all)':26s}" + "".join(
        f"  {statistics.median([v for _, v in per[e]]):+8.2f}%" if per[e] else f"  {'--':>9s}"
        for e in encs))
    for c in sorted({cls[i] for i in imgs}):
        print(f"  {'MEDIAN ' + c:26s}" + "".join(
            f"  {statistics.median([v for cc, v in per[e] if cc == c]):+8.2f}%"
            if [v for cc, v in per[e] if cc == c] else f"  {'--':>9s}" for e in encs))


# ------------------------------------------------------------------ stage3 --

def cmd_rdtable(args):
    """bpp needed to reach fixed SSIMULACRA2 levels — the BD-rate numbers in
    concrete units. Monotone-cubic interpolation of log10(bpp) vs the metric,
    per image; the table reports the median across images in each class."""
    rows = []
    for t in args.tsv:
        with open(t) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            for line in f:
                rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    keys = sorted({(r["encoder"], r["preset"]) for r in rows})
    levels = [float(x) for x in args.levels.split(",")]
    classes = sorted({r["class"] for r in rows})
    print(f"bpp to reach each SSIMULACRA2 level (median over images in the class; "
          f"'--' = level outside the measured range for >half the images)")
    print("class\tencoder\tpreset\t" + "\t".join(f"ss2_{int(l)}" for l in levels))
    for c in classes:
        imgs = sorted({r["image"] for r in rows if r["class"] == c})
        for enc, preset in keys:
            out = []
            for L in levels:
                vals = []
                for img in imgs:
                    pts = sorted((float(r["ssim2"]), math.log10(float(r["bpp"])))
                                 for r in rows if r["image"] == img
                                 and r["encoder"] == enc and r["preset"] == preset)
                    ded = []
                    for m, lb in pts:
                        if not ded or m > ded[-1][0]:
                            ded.append((m, lb))
                    if len(ded) < 4 or not (ded[0][0] <= L <= ded[-1][0]):
                        continue
                    xs = [p[0] for p in ded]
                    ys = [p[1] for p in ded]
                    vals.append(10 ** _pchip_eval(xs, ys, _pchip_slopes(xs, ys), L))
                out.append(f"{statistics.median(vals):.4f}" if len(vals) > len(imgs) // 2
                           else "--")
            print(f"{c}\t{enc}\t{preset}\t" + "\t".join(out))


def cmd_stage3(args):
    """Quality-target accuracy.

    NONE of the four encoders has a native "encode to SSIMULACRA2 = T" mode, so
    all four are driven by the SAME external search implemented here — a
    bracketed secant over the encoder's own quantizer scale, seeded from an
    anchor line fitted on the fly from the two bracket ends. Any difference
    between encoders is therefore a property of THEIR quality-vs-quantizer
    curve (smoothness, monotonicity, reachability), not of a different search.
    """
    modes = json.loads(args.modes)
    corpus = prep_rd_corpus()
    targets = [float(t) for t in args.targets.split(",")]
    band = args.band
    out = Path(args.out)
    f = open(out, "w")
    f.write("encoder\tpreset\timage\tclass\ttarget\tachieved\tabs_err\tencodes\t"
            "converged\tq_final\tbytes\tbpp\treason\n")
    for label, cls, yuv, w, h in corpus:
        for enc, preset in modes.items():
            qlo, qhi = ENCODERS[enc][1]
            for T in targets:
                r = target_search(enc, preset, yuv, w, h, qlo, qhi, T, band,
                                  args.max_encodes, args.timeout)
                f.write(f"{enc}\t{preset}\t{label}\t{cls}\t{T}\t{r['achieved']}\t"
                        f"{r['abs_err']}\t{r['encodes']}\t{int(r['converged'])}\t"
                        f"{r['q']}\t{r['bytes']}\t{r['bpp']:.6f}\t{r['reason']}\n")
                f.flush()
                print(f"  {enc:11s} {label:15s} T={T:5.1f} -> {r['achieved']:7.3f} "
                      f"|e|={r['abs_err']:6.3f} n={r['encodes']} "
                      f"{'OK' if r['converged'] else 'MISS:' + r['reason']}", flush=True)
    f.close()
    print(f"wrote {out}")


def target_search(enc, preset, yuv, w, h, qlo, qhi, T, band, max_encodes, timeout):
    """Seeded secant on quantizer -> SSIMULACRA2. IDENTICAL for all four.

    Deliberately shaped like a real target-quality implementation (cf. zenavif's
    `encode_rgb8_with_target`), not like an oracle:
      1. seed from a fixed anchor line in NORMALIZED quantizer space
         t = (q - qlo) / (qhi - qlo), so the seed is the same "position in the
         encoder's own range" for every encoder;
      2. take a second probe offset from the seed to get a secant slope;
      3. secant, maintaining a bracket once one exists, clamped to the range;
      4. stop on |achieved - target| <= band, on the bracket collapsing to one
         quantizer step, or at `max_encodes`.
    Boundary saturation (the target is off the top/bottom of what the encoder
    can produce on this image) is reported as an honest failure, not hidden.
    """
    seen = {}

    def probe(qf):
        q = max(qlo, min(qhi, int(round(qf))))
        if q not in seen:
            r = encode_score(enc, w, h, q, preset, yuv, "s3", timeout=timeout)
            seen[q] = None if "error" in r else r
        return q, seen[q]

    span = qhi - qlo
    # Anchor: SSIMULACRA2 ~100 at the low-quantizer end, ~40 at the high end.
    t = min(max((100.0 - T) / 60.0, 0.0), 1.0)
    q1, r1 = probe(qlo + t * span)
    if r1 is None:
        return dict(achieved=float("nan"), abs_err=float("nan"), encodes=len(seen),
                    converged=False, q=-1, bytes=0, bpp=0.0, reason="encode_error")
    # Second probe on the side the first one missed.
    step = max(2.0, span / 8.0)
    q2, r2 = probe(q1 - step if r1["ssim2"] < T else q1 + step)
    if r2 is None:
        return dict(achieved=r1["ssim2"], abs_err=abs(T - r1["ssim2"]), encodes=len(seen),
                    converged=abs(T - r1["ssim2"]) <= band, q=q1, bytes=r1["bytes"],
                    bpp=r1["bytes"] * 8 / (w * h), reason="encode_error")

    pts = {q1: r1["ssim2"], q2: r2["ssim2"]}
    best_q, best_r = min(((q1, r1), (q2, r2)), key=lambda p: abs(p[1]["ssim2"] - T))
    while len(seen) < max_encodes and abs(best_r["ssim2"] - T) > band:
        # bracket if we have one, else secant-extrapolate from the two nearest.
        above = [(q, s) for q, s in pts.items() if s >= T]
        below = [(q, s) for q, s in pts.items() if s < T]
        if above and below:
            a, fa = max(above, key=lambda p: p[0])   # largest q still >= T
            b, fb = min(below, key=lambda p: p[0])   # smallest q already < T
            if abs(b - a) <= 1:
                break
        else:
            srt = sorted(pts.items(), key=lambda p: abs(p[1] - T))[:2]
            (a, fa), (b, fb) = srt[0], srt[1]
            if abs(a - b) < 1e-9:
                break
        if abs(fb - fa) < 1e-9:
            break
        qn = a + (T - fa) * (b - a) / (fb - fa)
        qn = min(max(qn, qlo), qhi)
        if int(round(qn)) in seen:
            # nudge toward the target side rather than re-probing a known point
            qn = int(round(qn)) + (-1 if fa < T else 1)
            if int(round(qn)) in seen or not (qlo <= qn <= qhi):
                break
        qi, r = probe(qn)
        if r is None:
            break
        pts[qi] = r["ssim2"]
        if abs(r["ssim2"] - T) < abs(best_r["ssim2"] - T):
            best_q, best_r = qi, r
    conv = abs(best_r["ssim2"] - T) <= band
    reason = ""
    if not conv:
        if best_q == qlo and best_r["ssim2"] < T:
            reason = "above_max_quality"          # even the best quantizer misses
        elif best_q == qhi and best_r["ssim2"] > T:
            reason = "below_min_quality"
        else:
            reason = "band_not_reached"
    return dict(achieved=best_r["ssim2"], abs_err=abs(best_r["ssim2"] - T),
                encodes=len(seen), converged=conv, q=best_q, bytes=best_r["bytes"],
                bpp=best_r["bytes"] * 8 / (w * h), reason=reason)


def cmd_byteid(args):
    """Whole-stream sha256 of TWO encoders over the RD corpus x q grid.

    Written for the `zenav1-aom` vs `libaom-c` decomposition: the port is a
    byte-exactness port of libaom, so wherever the two streams hash equal their
    BD-rate is identical BY CONSTRUCTION and no RD comparison at that cell can
    say anything about coding. Reporting the identical FRACTION (and naming the
    cells that diverge) is what makes the matched-preset arm interpretable —
    the same role xbench_svt_byteidentity_2026-08-01.tsv plays for the SVT pair,
    but per RD cell rather than per preset on one calibration image.
    """
    import hashlib
    a, b = args.a, args.b
    pa, pb = args.preset_a, args.preset_b
    corpus = prep_rd_corpus()
    if args.classes:
        keep = set(args.classes.split(","))
        corpus = [c for c in corpus if c[1] in keep]
    qs = [int(x) for x in args.qgrid.split(",")] if args.qgrid else QGRID[a]
    tmp = WORK / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    n_ok = n_tot = 0
    with open(out, "w") as f:
        f.write("image\tclass\tw\th\tq\tenc_a\tpreset_a\tbytes_a\tsha_a\t"
                "enc_b\tpreset_b\tbytes_b\tsha_b\tidentical\n")
        for label, cls, yuv, w, h in corpus:
            for q in qs:
                shas, bys, bad = [], [], None
                for enc, preset, tag in ((a, pa, "bidA"), (b, pb, "bidB")):
                    obu = tmp / f"{tag}.obu"
                    ns, by, err = run_encode(enc, w, h, q, preset, yuv, obu, 0, 1,
                                             timeout=args.timeout)
                    if err:
                        bad = f"{enc}: {err}"
                        break
                    shas.append(hashlib.sha256(obu.read_bytes()).hexdigest())
                    bys.append(by)
                if bad:
                    print(f"  FAIL {label} q{q}: {bad}", flush=True)
                    continue
                same = int(shas[0] == shas[1])
                n_tot += 1
                n_ok += same
                f.write(f"{label}\t{cls}\t{w}\t{h}\t{q}\t{a}\t{pa}\t{bys[0]}\t{shas[0]}\t"
                        f"{b}\t{pb}\t{bys[1]}\t{shas[1]}\t{same}\n")
                f.flush()
                print(f"  {label:15s} q{q:3d}  {bys[0]:8d} vs {bys[1]:8d}  "
                      f"{'IDENTICAL' if same else 'DIFFER'}", flush=True)
    print(f"\n{n_ok}/{n_tot} cells byte-identical "
          f"({a}@{pa} vs {b}@{pb})\nwrote {out}")


def cmd_fitreport(args):
    """Re-derive the stage-1 fit from the RAW samples and report how well the
    `total = alpha + beta*pixels` model actually holds.

    Reporting a bare alpha/beta would be misleading where the model misfits, so
    this also emits the per-size residual of the fit and the LOG-LOG exponent
    (t ~ pixels^k). k == 1 means the linear model is the right shape; k < 1
    means per-pixel cost falls with frame size and the intercept is doing work
    the model cannot honestly attribute to a fixed per-call cost."""
    import math
    cells = {}
    with open(args.raw) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(hdr, line.rstrip("\n").split("\t")))
            if r["phase"] != "ladder":
                continue
            cells.setdefault((r["encoder"], int(r["preset"]), r["content"]), {}) \
                 .setdefault((int(r["pixels"]), int(r["size"])), []).append(int(r["ns"]))
    print("encoder\tpreset\tcontent\talpha_ms\tbeta_ms_per_MP\tmax_resid_pct\t"
          "loglog_k\tmps_64\tmps_256\tmps_1024\tmps_large\tlarge_px")
    for (enc, preset, content), by_px in sorted(cells.items()):
        pts = sorted((px, statistics.median(v) / 1e9, n) for (px, n), v in by_px.items())
        if len(pts) < 3:
            continue
        fit = fit_alpha_beta([(p[0], p[1], 0, p[2]) for p in pts])
        a, b = fit["alpha_ms"] / 1e3, fit["beta_ms_per_mp"] / 1e3 / 1e6
        resid = max(abs((a + b * px) - t) / t * 100 for px, t, _ in pts)
        ks = [math.log(pts[i + 1][1] / pts[i][1]) / math.log(pts[i + 1][0] / pts[i][0])
              for i in range(len(pts) - 1)]
        mps = {n: px / t / 1e6 for px, t, n in pts}
        print(f"{enc}\t{preset}\t{content}\t{fit['alpha_ms']:.4f}\t"
              f"{fit['beta_ms_per_mp']:.3f}\t{resid:.1f}\t{statistics.mean(ks):.4f}\t"
              + "\t".join(f"{mps.get(n, float('nan')):.4f}" for n in (64, 256, 1024))
              + f"\t{pts[-1][0]/pts[-1][1]/1e6:.4f}\t{pts[-1][0]}")


def cmd_control(args):
    """Run-to-run control band: re-run the SAME cell as N INDEPENDENT process
    invocations and report the spread of the per-invocation median. Anything
    smaller than this band in the tables below is noise, not a result."""
    modes = json.loads(args.modes)
    qmap = json.loads(args.q)
    print("encoder\tpreset\tsize\tn\tmedian_ms\tmin_ms\tmax_ms\tspread_pct\tstdev_pct\tbytes_unique")
    for enc, preset in modes.items():
        yuv = WORK / "src" / f"photo_{args.size}.yuv"
        meds, bys = [], set()
        for _ in range(args.n):
            ns, by, err = run_encode(enc, args.size, args.size, qmap[enc], preset, yuv,
                                     WORK / "ctl.obu", 2, reps_for(args.size ** 2),
                                     timeout=args.timeout)
            if err:
                print(f"{enc}: FAILED {err}")
                break
            meds.append(median_ns(ns) / 1e6)
            bys.add(by)
        if not meds:
            continue
        m = statistics.median(meds)
        sd = statistics.stdev(meds) if len(meds) > 1 else 0.0
        print(f"{enc}\t{preset}\t{args.size}\t{len(meds)}\t{m:.4f}\t{min(meds):.4f}\t"
              f"{max(meds):.4f}\t{(max(meds)-min(meds))/m*100:.2f}\t{sd/m*100:.2f}\t{len(bys)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    s1 = sub.add_parser("stage1")
    s1.add_argument("--out", required=True)
    s1.add_argument("--q", default='{"zenav1-aom":40,"libaom-c":40,"svt-c":40,'
                                   '"svt-rust":40,"zenrav1e":160}')
    s1.add_argument("--floor", type=float, default=0.30,
                    help="stop descending presets below this MP/s")
    s1.add_argument("--timeout", type=int, default=1800)
    s1.add_argument("--only-frontier", action="store_true")
    s1.add_argument("--only", default="",
                    help="comma list of encoders to run (default: all)")
    s2 = sub.add_parser("stage2")
    s2.add_argument("--out", required=True)
    s2.add_argument("--modes", required=True, help='{"svt-c":1,...} encoder->preset')
    s2.add_argument("--timeout", type=int, default=1800)
    s2.add_argument("--classes", default="", help="comma list of content classes to keep")
    s2.add_argument("--qgrid", default="", help="JSON list overriding the default q grid")
    bd = sub.add_parser("bdrate")
    bd.add_argument("--tsv", required=True)
    bd.add_argument("--ref", default="svt-c")
    bd.add_argument("--metric", default="ssim2", choices=["ssim2", "ba_3n", "ba_max"])
    s3 = sub.add_parser("stage3")
    s3.add_argument("--out", required=True)
    s3.add_argument("--modes", required=True)
    s3.add_argument("--targets", default="50,60,70,80,90")
    s3.add_argument("--band", type=float, default=1.0)
    s3.add_argument("--max-encodes", type=int, default=10)
    s3.add_argument("--timeout", type=int, default=1800)
    ct = sub.add_parser("control")
    ct.add_argument("--modes", required=True)
    ct.add_argument("--q", default='{"zenav1-aom":44,"libaom-c":44,"svt-c":40,'
                                   '"svt-rust":40,"zenrav1e":187}')
    ct.add_argument("--size", type=int, default=1024)
    ct.add_argument("--n", type=int, default=9)
    ct.add_argument("--timeout", type=int, default=1800)
    bi = sub.add_parser("byteid")
    bi.add_argument("--out", required=True)
    bi.add_argument("--a", required=True, help="encoder A label")
    bi.add_argument("--b", required=True, help="encoder B label")
    bi.add_argument("--preset-a", required=True, type=int)
    bi.add_argument("--preset-b", required=True, type=int)
    bi.add_argument("--qgrid", default="", help="comma list; default = QGRID[a]")
    bi.add_argument("--classes", default="")
    bi.add_argument("--timeout", type=int, default=1800)
    rt = sub.add_parser("rdtable")
    rt.add_argument("--tsv", required=True, nargs="+")
    rt.add_argument("--levels", default="50,60,70,80,90")
    fr = sub.add_parser("fitreport")
    fr.add_argument("--raw", required=True)
    a = ap.parse_args()
    if a.cmd == "rdtable":
        return cmd_rdtable(a)
    if a.cmd == "fitreport":
        return cmd_fitreport(a)
    if a.cmd == "control":
        return cmd_control(a)
    if a.cmd == "byteid":
        return cmd_byteid(a)
    if a.cmd == "prep":
        cmd_prep(a)
    elif a.cmd == "stage1":
        cmd_stage1(a)
    elif a.cmd == "stage2":
        cmd_stage2(a)
    elif a.cmd == "bdrate":
        cmd_bdrate(a)
    elif a.cmd == "stage3":
        cmd_stage3(a)


if __name__ == "__main__":
    main()
