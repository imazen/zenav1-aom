#!/usr/bin/env bash
# perf_band.sh — rotated interleaved timing band over the arms perf_arms.sh built.
#
#   N=24 OUT=~/tmp/arms/band.tsv scripts/perf_band.sh
#
# Arms: base, new, baseB (a second invocation of the SAME base binary — the
# same-binary NULL; no null, no claim). The arm order rotates by one position
# every round, so over N rounds every arm spends N/3 rounds in each position —
# a fixed order confounds arm with position, measured at up to 1.7 % within a
# round on this box, i.e. the size of a typical lever.
#
# Each encode is ONE eprof_x86 invocation at reps=1 under `nice -n 19` (the
# protocol of every 2026-09-09/10 band). Cell defaults to the shipping preset.
# Output columns: round position arm ms bytes.
set -euo pipefail
N=${N:-24}
ARMS_DIR=${ARMS:-$HOME/tmp/arms}
OUT=${OUT:-$ARMS_DIR/band.tsv}
W=${W:-1024}; H=${H:-1024}; CQ=${CQ:-27}; SPEED=${SPEED:-3}
ARMS=(base new baseB)
declare -A BIN=([base]="$ARMS_DIR/base" [new]="$ARMS_DIR/new" [baseB]="$ARMS_DIR/base")
K=${#ARMS[@]}
[ $((N % K)) -eq 0 ] || echo "WARNING: N=$N is not a multiple of $K arms — position occupancy will be unequal" >&2
echo -e "round\tposition\tarm\tms\tbytes" > "$OUT"
for ((i=1;i<=N;i++)); do
  for ((j=0;j<K;j++)); do
    arm=${ARMS[$(( (j + i - 1) % K ))]}
    out=$(nice -n 19 "${BIN[$arm]}" port "$W" "$H" "$CQ" "$SPEED" 1 2>&1 | tail -1)
    ms=$(echo "$out" | grep -oE 'per=[0-9.]+' | cut -d= -f2)
    by=$(echo "$out" | grep -oE 'bytes=[0-9]+' | cut -d= -f2)
    echo -e "$i\t$j\t$arm\t$ms\t$by" | tee -a "$OUT" >/dev/null
    printf '\rround %d/%d %-5s %s ms' "$i" "$N" "$arm" "$ms" >&2
  done
done
echo >&2
echo "wrote $OUT"
