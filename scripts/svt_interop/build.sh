#!/usr/bin/env bash
# build.sh — build svt_scc_encode against the out-of-tree SVT-AV1 C static lib.
#
# Reuses the exact lib scripts/xbench_build.sh produces (~/tmp/svtc-bin by
# default); builds it here only when absent. NOTHING modifies the SVT-AV1 C
# tree: the cmake build dir and the lib output dir are both outside it.
set -euo pipefail

ZEN=${ZEN:-$HOME/work/zen}
SVTC_SRC=$ZEN/zenav1-svt-c
SVTC_BUILD=${SVTC_BUILD:-$HOME/tmp/svtc-build}
SVTC_BIN=${SVTC_BIN:-$HOME/tmp/svtc-bin}
JOBS=${JOBS:-4}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/tmp/svt_interop}
LOGS=${LOGS:-$HOME/tmp}
mkdir -p "$OUT" "$LOGS"

if [ ! -f "$SVTC_BIN/libSvtAv1Enc.a" ]; then
  echo "== SVT-AV1 C static lib (out-of-tree: $SVTC_BUILD)"
  nice -n 19 cmake -S "$SVTC_SRC" -B "$SVTC_BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OUTPUT_DIRECTORY="$SVTC_BIN/" \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_APPS=OFF -DBUILD_TESTING=OFF \
    -DSVT_AV1_LTO=OFF >"$LOGS/svtc-cmake-config.log" 2>&1
  nice -n 19 cmake --build "$SVTC_BUILD" -j "$JOBS" >"$LOGS/svtc-build.log" 2>&1
fi
ls -l "$SVTC_BIN/libSvtAv1Enc.a"

nice -n 19 cc -O2 -o "$OUT/svt_scc_encode" "$HERE/svt_scc_encode.c" \
  -I"$SVTC_SRC/Source/API" \
  -L"$SVTC_BIN" -lSvtAv1Enc -lm -lpthread -lstdc++
echo "  -> $OUT/svt_scc_encode"
