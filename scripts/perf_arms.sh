#!/usr/bin/env bash
# perf_arms.sh — build the two timing arms for a clause-(4) band.
#
#   scripts/perf_arms.sh <BASE_SHA>      # base = that commit, new = the working tree
#
# `base` is built in a throwaway git worktree at BASE_SHA (so the working tree is
# untouched and the two builds cannot see each other's source); `new` is built
# from the current working tree. Both are `eprof_x86` release builds. Output:
# $ARMS/base and $ARMS/new (default ~/tmp/arms).
#
# TWO CHECKS THAT EARNED THEIR KEEP (see docs/ITERATION_PLAYBOOK.md):
#  * the build's exit status is checked — never `| grep error`, cargo's error
#    lines begin with ANSI escapes and an anchored grep matches nothing, which is
#    how three consecutive builds once shipped a STALE binary into a band;
#  * the arms must differ by sha256. Identical hashes across two source states is
#    impossible for real codegen and means one of them is stale.
set -euo pipefail
BASE_SHA=${1:?usage: perf_arms.sh <BASE_SHA>}
ARMS=${ARMS:-$HOME/tmp/arms}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ARMS"
cd "$ROOT"

echo "== new arm (working tree, HEAD=$(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty'))"
cargo build --release -p zenav1-aom-bench --example eprof_x86
cp target/release/examples/eprof_x86 "$ARMS/new"

WT=$(mktemp -d "$HOME/tmp/perf_base_wt.XXXX")
trap 'git worktree remove --force "$WT" 2>/dev/null || true' EXIT
echo "== base arm ($BASE_SHA) in worktree $WT"
git worktree add --detach "$WT" "$BASE_SHA" >/dev/null
# Share the target dir so the base build reuses every unchanged artifact.
( cd "$WT" && CARGO_TARGET_DIR="$ROOT/target" cargo build --release -p zenav1-aom-bench --example eprof_x86 )
cp "$ROOT/target/release/examples/eprof_x86" "$ARMS/base"

echo "== sha256"
sha256sum "$ARMS/base" "$ARMS/new"
if cmp -s "$ARMS/base" "$ARMS/new"; then
  echo "ERROR: base and new are byte-identical — one build is stale or the change is not in the binary" >&2
  exit 1
fi
echo "== byte-length identity on the shipping cell (both arms must agree; a differing count is an RD change, not a perf change)"
for a in base new; do printf '%-5s ' $a; "$ARMS/$a" port "${W:-1024}" "${H:-1024}" "${CQ:-27}" "${SPEED:-3}" 1 | grep -oE 'bytes=[0-9]+'; done
