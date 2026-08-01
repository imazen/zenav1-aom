#!/usr/bin/env bash
# xbench_build.sh — build every encoder driver for the cross-encoder AV1
# still-picture benchmark (benchmarks/xbench).
#
# NOTHING here modifies a sibling repo: the SVT-AV1 C tree is configured with an
# OUT-OF-TREE cmake build dir, and the Rust siblings are consumed as path
# dependencies (their sources are read, their build output lands in this
# harness's own target/).
#
# Builds are serialized and `nice -n 19 -j ${JOBS:-4}` so a concurrent agent's
# build is never starved.
#
# Env:
#   ZEN            sibling-repo root         (default ~/work/zen)
#   SVTC_BUILD     SVT-AV1 C cmake build dir (default ~/tmp/svtc-build)
#   SVTC_BIN       SVT-AV1 C lib output dir  (default ~/tmp/svtc-bin)
#   JOBS           parallelism               (default 4)
set -euo pipefail

ZEN=${ZEN:-$HOME/work/zen}
SVTC_SRC=$ZEN/zenav1-svt-c
SVTC_BUILD=${SVTC_BUILD:-$HOME/tmp/svtc-build}
SVTC_BIN=${SVTC_BIN:-$HOME/tmp/svtc-bin}
JOBS=${JOBS:-4}
HERE=$(cd "$(dirname "$0")/.." && pwd)
XB=$HERE/benchmarks/xbench
LOGS=${LOGS:-$HOME/tmp}
mkdir -p "$LOGS"

echo "== 1/3  SVT-AV1 C static lib (out-of-tree: $SVTC_BUILD)"
if [ ! -f "$SVTC_BIN/libSvtAv1Enc.a" ]; then
  nice -n 19 cmake -S "$SVTC_SRC" -B "$SVTC_BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OUTPUT_DIRECTORY="$SVTC_BIN/" \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_APPS=OFF -DBUILD_TESTING=OFF \
    -DSVT_AV1_LTO=OFF >"$LOGS/svtc-cmake-config.log" 2>&1
  nice -n 19 cmake --build "$SVTC_BUILD" -j "$JOBS" >"$LOGS/svtc-build.log" 2>&1
fi
ls -l "$SVTC_BIN/libSvtAv1Enc.a"

echo "== 2/3  drv_svtc (C driver against that lib)"
nice -n 19 cc -O2 -o "$XB/target/drv_svtc" "$XB/csrc/drv_svtc.c" \
  -I"$SVTC_SRC/Source/API" \
  -L"$SVTC_BIN" -lSvtAv1Enc -lm -lpthread -lstdc++
echo "  -> $XB/target/drv_svtc"

echo "== 3/3  Rust drivers + xtool"
cd "$XB"
# NO -C target-cpu=native: runtime SIMD dispatch is what users get.
nice -n 19 cargo build --release -j "$JOBS" \
  -p xtool -p drv-svtrs -p drv-rav1e -p drv-aom 2>&1 | tail -5
ls -l "$XB/target/release/xtool" "$XB/target/release/drv-svtrs" \
      "$XB/target/release/drv-rav1e" "$XB/target/release/drv-aom"
