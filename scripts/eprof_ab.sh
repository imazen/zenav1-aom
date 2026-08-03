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

# ROTATE=1 rotates the arm order by one position each round, so that over N
# rounds every arm spends 1/k of them in each of the k positions.  This matters:
# on 2026-08-03 a 44-round fixed-order band put TWO COPIES OF ONE BINARY at
# positions 5 and 6 and they came out 0.34 pp apart, while the copies at
# positions 1 and 2 agreed to 0.11 pp -- i.e. the position an arm occupies
# inside a round is worth as much as the effect being measured
# (benchmarks/encoder_intra_smooth_paeth_2026-08-03.md sec 3; the same drift is
# recorded for windows-11-arm in benchmarks/winperf_content_census_2026-08-03.md
# sec 5, where it is handled by pooling instead).  Rotating makes it cancel by
# construction rather than by a correction, and costs nothing.
#
# Default OFF so previously recorded bands stay reproducible command-for-command.
N=$1; shift
OUT=$1; shift
ARMS=("$@")
K=${#ARMS[@]}

printf 'arm\tround\tposition\tmedian_ns\tsamples_ns\tbytes\n' > "$OUT"
for i in $(seq 1 "$N"); do
  ORDER=()
  for j in $(seq 0 $((K - 1))); do
    if [ "${ROTATE:-0}" = "1" ]; then
      ORDER+=("${ARMS[$(( (j + i - 1) % K ))]}")
    else
      ORDER+=("${ARMS[$j]}")
    fi
  done
  POS=0
  for spec in "${ORDER[@]}"; do
    POS=$((POS + 1))
    arm=${spec%%=*}
    drv=${spec#*=}
    line=$(nice -n 19 "$drv" "$W" "$H" "$Q" "$S" "$YUV" "$HOME/tmp/eprof_ab_$arm.obu" "$WARM" "$REPS")
    ns=$(echo "$line" | tr ' ' '\n' | sed -n 's/^NS=//p' | sort -n | tr '\n' ',')
    med=$(echo "$line" | tr ' ' '\n' | sed -n 's/^NS=//p' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    by=$(echo "$line" | tr ' ' '\n' | sed -n 's/^BYTES=//p' | head -1)
    printf '%s\t%d\t%d\t%s\t%s\t%s\n' "$arm" "$i" "$POS" "$med" "${ns%,}" "$by" >> "$OUT"
    echo "$arm #$i pos=$POS median=${med}ns bytes=$by"
  done
done
