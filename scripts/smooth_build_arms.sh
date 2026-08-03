#!/usr/bin/env bash
# smooth_build_arms.sh — build the timing arms for the smooth+paeth i16-lane band.
#
# Each arm is a SEPARATE release binary built from this one worktree by forcing
# `intra::simd16`'s three gate predicates to `false` and restoring the file from
# a pristine copy afterwards. `base` forces all three, so it is the pre-lever
# baseline INCLUDING the absence of the gate's per-block scan.
#
# The PAETH arms this script used to build are gone with the PAETH kernel; see
# benchmarks/encoder_intra_smooth_paeth_2026-08-03.md sec 4 for why it was
# reverted, and .ab*.tsv for the four bands that measured it.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$HOME/tmp/smooth/arms}
SRC=crates/aom-dsp/src/intra/simd16.rs
mkdir -p "$OUT"
cd "$ROOT"
# Pristine copy: simd16.rs may be untracked at build time, so `git checkout --`
# is not a reliable restore. Snapshot it once and restore from that.
PRISTINE="$OUT/simd16.rs.pristine"
cp "$SRC" "$PRISTINE"
trap 'cp "$PRISTINE" "$ROOT/$SRC"' EXIT

patch_off() { # $@ = which gates to force false: smooth smooth_v smooth_h paeth
  python3 - "$@" <<'PY'
import sys
p='crates/aom-dsp/src/intra/simd16.rs'
s=open(p).read()
bodies={
 'smooth':("pub(crate) fn smooth_applies(bw: usize, bh: usize, above_row: &[u16], left: &[u16]) -> bool {\n    span_le(above_row, bw, U16_SMOOTH_MAX) && span_le(left, bh, U16_SMOOTH_MAX)\n}",
           "pub(crate) fn smooth_applies(bw: usize, bh: usize, above_row: &[u16], left: &[u16]) -> bool {\n    let _ = (bw, bh, above_row, left);\n    false\n}"),
 'smooth_v':("pub(crate) fn smooth_v_applies(bw: usize, above_row: &[u16], below: i32) -> bool {\n    span_le(above_row, bw, U16_SMOOTH_MAX) && (0..=i32::from(U16_SMOOTH_MAX)).contains(&below)\n}",
           "pub(crate) fn smooth_v_applies(bw: usize, above_row: &[u16], below: i32) -> bool {\n    let _ = (bw, above_row, below);\n    false\n}"),
 'smooth_h':("pub(crate) fn smooth_h_applies(bh: usize, left: &[u16], right: i32) -> bool {\n    span_le(left, bh, U16_SMOOTH_MAX) && (0..=i32::from(U16_SMOOTH_MAX)).contains(&right)\n}",
           "pub(crate) fn smooth_h_applies(bh: usize, left: &[u16], right: i32) -> bool {\n    let _ = (bh, left, right);\n    false\n}"),
}
for k in sys.argv[1:]:
    old,new=bodies[k]
    assert old in s, f"gate {k} not found verbatim"
    s=s.replace(old,new,1)
open(p,'w').write(s)
PY
}

build() { # $1 = arm name, rest = gates to force false
  local name=$1; shift
  cp "$PRISTINE" "$SRC"
  if [ $# -gt 0 ]; then patch_off "$@"; fi
  (cd benchmarks/xbench && nice -n 19 cargo build --release -j 4 -p drv-aom >/dev/null 2>&1)
  cp benchmarks/xbench/target/release/drv-aom "$OUT/drv-$name"
  cp "$PRISTINE" "$SRC"
  echo "$name  $(shasum -a 256 "$OUT/drv-$name" | cut -c1-16)"
}

build base   smooth smooth_v smooth_h
cp "$OUT/drv-base" "$OUT/drv-baseB"
build all
cp "$OUT/drv-all" "$OUT/drv-allB"
echo "--- distinctness ---"
shasum -a 256 "$OUT"/drv-* | sort
