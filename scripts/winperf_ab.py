#!/usr/bin/env python3
"""winperf_ab.py — INTERLEAVED control band over N `winperf` binaries, anywhere.

`scripts/eprof_ab.sh` does this on the dev box, but it is bash + `nice` + `seq`
+ `sed` + `awk` and takes a `.yuv` path; the Windows CI runners have none of
that guaranteed and the `winperf` driver generates its own source. This is the
same protocol in stdlib Python so it runs identically on Windows, macOS and
Linux, and it writes the SAME TSV columns so `scripts/eprof_ab_stats.py` reads
its output unchanged.

  winperf_ab.py <rounds> <detail|smooth|photo> <out.tsv> label=/path/to/winperf ...

One invocation of EVERY arm per round, round after round, so runner drift lands
on all arms equally (docs/DIFFERENTIAL_PLAYBOOK.md §6 — comparing medians taken
in separate time windows is the mistake this exists to prevent, and a shared CI
VM is a far worse offender than the dev box).

Each invocation is `WARM` warm-up + `REPS` timed encodes and contributes its own
median. Defaults 2/7, matching `eprof_ab.sh` exactly.

Pass the SAME binary twice under two labels (e.g. `post=... postB=...`) to get a
null arm: its measured "delta" is the runner's noise floor, which is the only
honest thing to read a real delta against on hardware you do not own.
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    rounds = int(sys.argv[1])
    content = sys.argv[2]
    # `photo` is the ORIENTED content (winperf::Content::Photo), fitted to the
    # study photograph's intra MODE distribution rather than to its allocator
    # call count — the axis `detail`/`smooth` get wrong. See
    # benchmarks/winperf_content_census_2026-08-03.md.
    if content not in ("detail", "smooth", "photo"):
        sys.exit(f"content must be detail|smooth|photo, got {content!r}")
    out = sys.argv[3]
    arms = []
    for spec in sys.argv[4:]:
        label, _, path = spec.partition("=")
        if not path:
            sys.exit(f"bad arm spec {spec!r}; want label=/path/to/winperf")
        arms.append((label, path))

    warm = os.environ.get("WARM", "2")
    reps = os.environ.get("REPS", "7")

    with open(out, "w", encoding="ascii") as f:
        f.write("arm\tround\tmedian_ns\tsamples_ns\tbytes\n")
        f.flush()
        for r in range(1, rounds + 1):
            for label, path in arms:
                res = subprocess.run(
                    [path, content, warm, reps],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                toks = res.stdout.split()
                ns = sorted(
                    int(t[3:]) for t in toks if t.startswith("NS=")
                )
                if not ns:
                    sys.exit(f"{label}: no NS= samples in {res.stdout!r}")
                med = ns[(len(ns) + 1) // 2 - 1]
                by = next(t[6:] for t in toks if t.startswith("BYTES="))
                f.write(
                    "%s\t%d\t%d\t%s\t%s\n"
                    % (label, r, med, ",".join(str(x) for x in ns), by)
                )
                f.flush()
                print(f"{label} #{r} median={med}ns bytes={by}", flush=True)


if __name__ == "__main__":
    main()
