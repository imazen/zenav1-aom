#!/usr/bin/env bash
# eprof_sample.sh — take a /usr/bin/sample profile of ONE xbench encoder arm on
# the profile cell, with the untimed setup (yuv read, C bootstrap, warmup)
# excluded by delaying the start of sampling.
#
#   eprof_sample.sh <arm: port|libaom-c|port-dbg> <reps> <sample_seconds> <out.txt> [skip_seconds]
#
# `sample` is macOS's built-in sampling profiler: it walks the target's stacks
# every <interval> ms and reports an inclusive call graph plus a self ("top of
# stack") table.  It is a WALL-CLOCK sampler, not an instruction counter, so
# shares are shares of elapsed time in the sampled window.  See the writeup's
# Method section for what that does and does not buy.
set -euo pipefail
XB=$(cd "$(dirname "$0")/../benchmarks/xbench" && pwd)
YUV=${YUV:-$HOME/tmp/xb/src/photo_1024.yuv}
W=${W:-1024}; H=${H:-1024}; Q=${Q:-44}; S=${S:-6}
INTERVAL=${INTERVAL:-1}

ARM=$1; REPS=$2; SECS=$3; OUT=$4; SKIP=${5:-3}
case $ARM in
  port)      DRV="$XB/target/release/drv-aom" ;;
  port-dbg)  DRV="$HOME/tmp/eprof-dbg-target/release/drv-aom" ;;
  libaom-c)  DRV="$XB/target/drv_libaom" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

echo "# arm=$ARM drv=$DRV reps=$REPS sample=${SECS}s interval=${INTERVAL}ms skip=${SKIP}s"
nice -n 19 "$DRV" "$W" "$H" "$Q" "$S" "$YUV" "$HOME/tmp/eprof_prof_$ARM.obu" 1 "$REPS" \
    > "$HOME/tmp/eprof_prof_$ARM.stdout" 2>&1 &
PID=$!
sleep "$SKIP"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "target exited before sampling started — raise REPS" >&2; exit 1
fi
sample "$PID" "$SECS" "$INTERVAL" -mayDie -f "$OUT"
wait "$PID" || true
echo "# target stdout: $(cat "$HOME/tmp/eprof_prof_$ARM.stdout" | tr ' ' '\n' | grep -c '^NS=') timed reps"
echo "# wrote $OUT"
