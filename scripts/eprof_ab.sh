#!/usr/bin/env bash
# eprof_ab.sh — INTERLEAVED control band over N arbitrary encoder binaries.
#
# eprof_control.sh interleaves exactly two fixed arms (the release port build and
# the C driver).  A perf landing needs to compare SEVERAL port builds — baseline,
# each lever alone, both together — and comparing their medians across separate
# time windows is exactly the mistake docs/DIFFERENTIAL_PLAYBOOK.md §6 warns
# about: on this box a same-binary re-run has moved a row by +16.5 %, and the
# background load is not stationary.
#
# So: take every arm as a label=path pair, and run ONE invocation of each per
# round, in order, round after round.  Every arm therefore sees the same drift.
# Each invocation is 2 warm-up + 7 timed encodes with its own median, identical
# to eprof_control.sh.
#
#   eprof_ab.sh <rounds> <out.tsv> label=/path/to/drv ...
#
# The C arm is just another label (label=.../drv_libaom) — the driver CLI is the
# same on both sides.
set -euo pipefail
YUV=${YUV:-$HOME/tmp/xb/src/photo_1024.yuv}
W=${W:-1024}; H=${H:-1024}; Q=${Q:-44}; S=${S:-6}
WARM=${WARM:-2}; REPS=${REPS:-7}

N=$1; shift
OUT=$1; shift

printf 'arm\tround\tmedian_ns\tsamples_ns\tbytes\n' > "$OUT"
for i in $(seq 1 "$N"); do
  for spec in "$@"; do
    arm=${spec%%=*}
    drv=${spec#*=}
    line=$(nice -n 19 "$drv" "$W" "$H" "$Q" "$S" "$YUV" "$HOME/tmp/eprof_ab_$arm.obu" "$WARM" "$REPS")
    ns=$(echo "$line" | tr ' ' '\n' | sed -n 's/^NS=//p' | sort -n | tr '\n' ',')
    med=$(echo "$line" | tr ' ' '\n' | sed -n 's/^NS=//p' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    by=$(echo "$line" | tr ' ' '\n' | sed -n 's/^BYTES=//p' | head -1)
    printf '%s\t%d\t%s\t%s\t%s\n' "$arm" "$i" "$med" "${ns%,}" "$by" >> "$OUT"
    echo "$arm #$i median=${med}ns bytes=$by"
  done
done
