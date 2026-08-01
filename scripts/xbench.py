#!/usr/bin/env python3
"""xbench — cross-encoder AV1 still-picture benchmark orchestrator.

Four encoders, one input byte stream, one timing contract, one scorer:

  zenav1-aom  the pure-Rust libaom port   (this repo)  cq 0..63, cpu-used 0..9
  svt-c       SVT-AV1, upstream C                      qp 0..63, preset 0..13
  svt-rust    zenav1-svt, pure-Rust SVT port           qp 0..63, preset 0..13
  zenrav1e    the rav1e fork                           quantizer 0..255, speed 0..10

Every driver implements the SAME contract (see benchmarks/xbench/*/src/main.rs):

    drv <w> <h> <q> <speed> <in.yuv> <out.obu> <warmup> <reps>
    stdout: NS=<n> NS=<n> ... BYTES=<m>

and times ONLY its own frame-encode call — never process startup, never file
I/O, never its own constructor/init.  All four are SINGLE-THREADED.

Subcommands:
  prep     build the source .yuv ladders from the corpus
  stage1   throughput calibration: MP/s per encoder per preset + the
           alpha (fixed per-call cost) / beta (per-pixel cost) fit
  stage2   RD sweep at the qualifying modes
  stage3   quality-target accuracy (identical external search for all four)
"""

import argparse
import json
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
    "svt-c": (CBIN / "drv_svtc", (1, 63), list(range(0, 14))),
    "svt-rust": (BIN / "drv-svtrs", (1, 63), list(range(0, 14))),
    "zenrav1e": (BIN / "drv-rav1e", (1, 255), list(range(0, 11))),
}

NS_RE = re.compile(rb"NS=(\d+)")
BYTES_RE = re.compile(rb"BYTES=(\d+)")


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


def cmd_stage1(args):
    """Screen every preset at 1 MP, then fit alpha+beta over the full ladder."""
    out_tsv = Path(args.out)
    raw_tsv = out_tsv.with_suffix(".raw.tsv")
    qmap = json.loads(args.q)
    rawf = open(raw_tsv, "w")
    rawf.write("phase\tencoder\tpreset\tcontent\tsize\tpixels\tq\trep\tns\tbytes\n")

    # ---- phase A: screen at the medium (1024x1024 = 1.05 MP) photo cell.
    screen = {}
    for enc, (_drv, _qr, presets) in ENCODERS.items():
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
    for enc, (_drv, _qr, presets) in ENCODERS.items():
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
    out = Path(args.out)
    f = open(out, "w")
    f.write("encoder\tpreset\timage\tclass\tw\th\tq\tbytes\tbpp\tssim2\tba_max\tba_3n\tenc_ms\n")
    for label, cls, yuv, w, h in corpus:
        for enc, preset in modes.items():
            for q in QGRID[enc]:
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
    """Bracketed secant on quantizer -> SSIMULACRA2. Identical for all four."""
    seen = {}

    def probe(q):
        q = max(qlo, min(qhi, int(round(q))))
        if q in seen:
            return q, seen[q]
        r = encode_score(enc, w, h, q, preset, yuv, "s3", timeout=timeout)
        if "error" in r:
            seen[q] = None
        else:
            seen[q] = r
        return q, seen[q]

    # Bracket ends first (they also give the anchor line and prove reachability).
    q_hi_qual, r_hi = probe(qlo)          # lowest quantizer  = highest quality
    q_lo_qual, r_lo = probe(qhi)          # highest quantizer = lowest quality
    if r_hi is None or r_lo is None:
        return dict(achieved=float("nan"), abs_err=float("nan"), encodes=len(seen),
                    converged=False, q=-1, bytes=0, bpp=0.0, reason="encode_error")
    if T > r_hi["ssim2"] + band:
        best = r_hi
        return dict(achieved=best["ssim2"], abs_err=abs(T - best["ssim2"]),
                    encodes=len(seen), converged=False, q=q_hi_qual,
                    bytes=best["bytes"], bpp=best["bytes"] * 8 / (w * h),
                    reason="above_max_quality")
    if T < r_lo["ssim2"] - band:
        best = r_lo
        return dict(achieved=best["ssim2"], abs_err=abs(T - best["ssim2"]),
                    encodes=len(seen), converged=False, q=q_lo_qual,
                    bytes=best["bytes"], bpp=best["bytes"] * 8 / (w * h),
                    reason="below_min_quality")
    a, fa = float(q_hi_qual), r_hi["ssim2"]
    b, fb = float(q_lo_qual), r_lo["ssim2"]
    best_q, best_r = (q_hi_qual, r_hi) if abs(r_hi["ssim2"] - T) < abs(r_lo["ssim2"] - T) \
        else (q_lo_qual, r_lo)
    while len(seen) < max_encodes:
        if abs(best_r["ssim2"] - T) <= band:
            break
        if abs(fb - fa) < 1e-9:
            break
        q = a + (T - fa) * (b - a) / (fb - fa)          # secant
        q = min(max(q, min(a, b) + 1e-9), max(a, b) - 1e-9)
        qi, r = probe(q)
        if r is None:
            break
        if abs(r["ssim2"] - T) < abs(best_r["ssim2"] - T):
            best_q, best_r = qi, r
        # keep the bracket: quality DEcreases as quantizer increases
        if r["ssim2"] > T:
            a, fa = float(qi), r["ssim2"]
        else:
            b, fb = float(qi), r["ssim2"]
        if abs(b - a) <= 1.0:
            break
    conv = abs(best_r["ssim2"] - T) <= band
    return dict(achieved=best_r["ssim2"], abs_err=abs(best_r["ssim2"] - T),
                encodes=len(seen), converged=conv, q=best_q, bytes=best_r["bytes"],
                bpp=best_r["bytes"] * 8 / (w * h),
                reason="" if conv else "band_not_reached")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    s1 = sub.add_parser("stage1")
    s1.add_argument("--out", required=True)
    s1.add_argument("--q", default='{"zenav1-aom":40,"svt-c":40,"svt-rust":40,"zenrav1e":160}')
    s1.add_argument("--floor", type=float, default=0.30,
                    help="stop descending presets below this MP/s")
    s1.add_argument("--timeout", type=int, default=1800)
    s1.add_argument("--only-frontier", action="store_true")
    s2 = sub.add_parser("stage2")
    s2.add_argument("--out", required=True)
    s2.add_argument("--modes", required=True, help='{"svt-c":1,...} encoder->preset')
    s2.add_argument("--timeout", type=int, default=1800)
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
    a = ap.parse_args()
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
