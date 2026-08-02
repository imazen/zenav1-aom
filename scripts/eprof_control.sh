#!/usr/bin/env bash
# eprof_control.sh — control band for the encoder-profile cell.
#
# N INDEPENDENT process invocations per arm (each: 2 warmup + 7 timed encodes,
# its own median taken), the protocol of benchmarks/xbench_2026-08-01.md
# "Control band".  The two arms are INTERLEAVED (port, C, port, C, ...) so that
# any drift in the box's background load lands on both arms equally and the
# port/C ratio stays readable even when the box is not idle.
#
# usage: eprof_control.sh <n_invocations> <out.tsv>
set -euo pipefail
XB=$(cd "$(dirname "$0")/../benchmarks/xbench" && pwd)
YUV=${YUV:-$HOME/tmp/xb/src/photo_1024.yuv}
N=${1:-9}
OUT=${2:-$HOME/tmp/eprof_control.tsv}
W=${W:-1024}; H=${H:-1024}; Q=${Q:-44}; S=${S:-6}

printf 'arm\tinvocation\tmedian_ns\tsamples_ns\tbytes\n' > "$OUT"
for i in $(seq 1 "$N"); do
  for arm in port libaom-c; do
    case $arm in
      port)     DRV="$XB/target/release/drv-aom" ;;
      libaom-c) DRV="$XB/target/drv_libaom" ;;
    esac
    line=$(nice -n 19 "$DRV" "$W" "$H" "$Q" "$S" "$YUV" "$HOME/tmp/eprof_ctl_$arm.obu" 2 7)
    ns=$(echo "$line" | tr ' ' '\n' | sed -n 's/^NS=//p' | sort -n | tr '\n' ',')
    med=$(echo "$line" | tr ' ' '\n' | sed -n 's/^NS=//p' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    by=$(echo "$line" | tr ' ' '\n' | sed -n 's/^BYTES=//p' | head -1)
    printf '%s\t%d\t%s\t%s\t%s\n' "$arm" "$i" "$med" "${ns%,}" "$by" >> "$OUT"
    echo "$arm #$i median=${med}ns bytes=$by"
  done
done
