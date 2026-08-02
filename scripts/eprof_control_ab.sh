#!/usr/bin/env bash
# eprof_control_ab.sh — the A/B form of eprof_control.sh: an interleaved control
# band over an ARBITRARY set of driver binaries, so a before/after pair and the
# libaom-c reference are all measured under the same background load.
#
# Same protocol as eprof_control.sh (which this does not replace — that one is
# the committed 2-arm form the 2026-08-02 profile used): N INDEPENDENT process
# invocations per arm, each 2 warm-up + 7 timed encodes with its own median
# taken, and the arms INTERLEAVED (a1, a2, a3, a1, a2, a3, ...) so drift lands
# on every arm equally.  Report the per-arm spread FIRST — on this box a
# same-binary re-run has moved a row by double digits (playbook §6).
#
# usage: ARMS="label=/path/to/drv ..." eprof_control_ab.sh <n_invocations> <out.tsv>
#   W/H/Q/S/YUV override the cell (defaults = the 2026-08-02 profile cell).
#
# example:
#   ARMS="port-base=$XB/target/release/drv-aom-base \
#         port-cache=$XB/target/release/drv-aom \
#         libaom-c=$XB/target/drv_libaom" \
#     scripts/eprof_control_ab.sh 9 ~/tmp/ab.tsv
set -euo pipefail
YUV=${YUV:-$HOME/tmp/xb/src/photo_1024.yuv}
N=${1:-9}
OUT=${2:-$HOME/tmp/eprof_control_ab.tsv}
W=${W:-1024}; H=${H:-1024}; Q=${Q:-44}; S=${S:-6}
: "${ARMS:?set ARMS=\"label=path ...\"}"

printf 'arm\tinvocation\tmedian_ns\tsamples_ns\tbytes\n' > "$OUT"
for i in $(seq 1 "$N"); do
  for spec in $ARMS; do
    arm=${spec%%=*}
    drv=${spec#*=}
    line=$(nice -n 19 "$drv" "$W" "$H" "$Q" "$S" "$YUV" "$HOME/tmp/eprof_ab_$arm.obu" 2 7)
    ns=$(echo "$line" | tr ' ' '\n' | sed -n 's/^NS=//p' | sort -n | tr '\n' ',')
    med=$(echo "$line" | tr ' ' '\n' | sed -n 's/^NS=//p' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    by=$(echo "$line" | tr ' ' '\n' | sed -n 's/^BYTES=//p' | head -1)
    printf '%s\t%d\t%s\t%s\t%s\n' "$arm" "$i" "$med" "${ns%,}" "$by" >> "$OUT"
    echo "$arm #$i median=${med}ns bytes=$by"
  done
done
echo "# wrote $OUT"
