#!/usr/bin/env bash
# eprof_breadth.sh — is the profile cell representative?  Re-measures the
# port/libaom-c wall ratio, and the port's exact CNN-cascade call count, across
# the size and cpu-used axes, so the single-cell hotspot finding is not
# over-read.  Both arms are run back to back on each cell.
#
#   eprof_breadth.sh <out.tsv>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
XB=$ROOT/benchmarks/xbench
SRC=${SRC:-$HOME/tmp/xb/src}
OUT=${1:-$HOME/tmp/eprof_breadth.tsv}
PORT=$XB/target/release/drv-aom
CDRV=$XB/target/drv_libaom
ALLOC=$ROOT/target/release/examples/eprof_alloc

med() { tr ' ' '\n' | sed -n 's/^NS=//p' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }

printf 'cell\tw\th\tcq\tcpu_used\tport_ms\tc_ms\tratio\tport_bytes\tc_bytes\tcnn_calls\tsb64\tcnn_per_sb\n' > "$OUT"

run() {  # <label> <src.yuv> <w> <h> <cq> <speed> <reps>
  local lbl=$1 yuv=$2 w=$3 h=$4 q=$5 s=$6 n=$7
  local pl cl pms cms pb cb cnn sb
  # A cell the port refuses (an open HANDOFF panic, e.g. KB-32 at cpu-used 9)
  # is RECORDED as a refusal, not silently skipped and not fatal to the sweep.
  if ! pl=$(nice -n 19 "$PORT" "$w" "$h" "$q" "$s" "$yuv" "$HOME/tmp/eb_p.obu" 1 "$n" 2>"$HOME/tmp/eb_err.txt"); then
    printf '%s\t%d\t%d\t%d\t%d\tPORT_REFUSED\t\t\t\t\t\t%d\t\t%s\n' \
      "$lbl" "$w" "$h" "$q" "$s" $(( ((w+63)/64) * ((h+63)/64) )) \
      "$(head -c 160 "$HOME/tmp/eb_err.txt" | tr '\n' ' ')" | tee -a "$OUT"
    return 0
  fi
  cl=$(nice -n 19 "$CDRV" "$w" "$h" "$q" "$s" "$yuv" "$HOME/tmp/eb_c.obu" 1 "$n")
  pms=$(echo "$pl" | med); cms=$(echo "$cl" | med)
  pb=$(echo "$pl" | tr ' ' '\n' | sed -n 's/^BYTES=//p' | head -1)
  cb=$(echo "$cl" | tr ' ' '\n' | sed -n 's/^BYTES=//p' | head -1)
  # exact CNN-cascade call count = the count of the 20480-byte layer-0 output
  # allocation, which has exactly one call site (eprof_alloc).
  cnn=$(nice -n 19 "$ALLOC" "$w" "$h" "$q" "$s" "$yuv" \
        | awk '/^  size 20480/ {print $5}')
  sb=$(( ((w+63)/64) * ((h+63)/64) ))
  printf '%s\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%d\t%.2f\n' \
    "$lbl" "$w" "$h" "$q" "$s" \
    "$(echo "$pms" | awk '{print $1/1e6}')" "$(echo "$cms" | awk '{print $1/1e6}')" \
    "$(echo "$pms $cms" | awk '{print $1/$2}')" "$pb" "$cb" "$cnn" "$sb" \
    "$(echo "$cnn $sb" | awk '{print $1/$2}')" | tee -a "$OUT"
}

# size axis at the profile's cpu-used / quantizer
run size256   "$SRC/photo_256.yuv"   256   256 44 6 5
run size1024  "$SRC/photo_1024.yuv" 1024 1024 44 6 5
run size2048  "$SRC/photo_2048.yuv" 2048 2048 44 6 3
# cpu-used axis at 1 MP
for s in 9 8 7 6 5 4 3; do
  case $s in 3|4) n=3 ;; *) n=5 ;; esac
  run "speed$s" "$SRC/photo_1024.yuv" 1024 1024 44 "$s" "$n"
done
# quantizer axis at the profile's size/speed
for q in 10 26 58; do
  run "cq$q" "$SRC/photo_1024.yuv" 1024 1024 "$q" 6 5
done
echo "# wrote $OUT"
