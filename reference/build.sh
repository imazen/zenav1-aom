#!/usr/bin/env bash
# Build the C libaom reference oracle that aom-rs measures bit-exactness against.
# Idempotent: if the artifacts already exist it does nothing, so it is safe to
# run locally (won't clobber a working build) and in CI (does the full build on
# a fresh checkout). See reference/BUILD_CONFIG.md for the authoritative config.
set -euo pipefail

# Pinned libaom: tag v3.14.1. The exact commit the shims are written against.
LIBAOM_TAG="v3.14.1"
LIBAOM_SHA="03087864cf4bea6abb0d28f95cf7843511413d8f"
LIBAOM_URL="https://aomedia.googlesource.com/aom"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/libaom"
BUILD="$SRC/build"

if [[ -f "$BUILD/libaom.a" && -x "$BUILD/aomenc" && -x "$BUILD/aomdec" ]]; then
    echo "reference libaom already built ($BUILD/libaom.a) — nothing to do."
    exit 0
fi

# Fetch the pinned source if it isn't present (reference/libaom is gitignored).
if [[ ! -d "$SRC/.git" ]]; then
    echo "cloning libaom $LIBAOM_TAG into $SRC ..."
    git clone --no-checkout "$LIBAOM_URL" "$SRC"
fi
git -C "$SRC" fetch --tags origin
git -C "$SRC" checkout -q "$LIBAOM_SHA"

# Configure + build with the bit-exactness config from BUILD_CONFIG.md.
#   CONFIG_MULTITHREAD=0  -> deterministic single-thread encoder output target.
#   DECODER + ENCODER + TESTS/EXAMPLES/TOOLS  -> aomenc/aomdec/libaom.a shipped.
mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCONFIG_MULTITHREAD=0 \
    -DENABLE_TESTS=1 -DENABLE_EXAMPLES=1 -DENABLE_TOOLS=1 \
    -DCONFIG_AV1_DECODER=1 -DCONFIG_AV1_ENCODER=1
cmake --build "$BUILD" --target aom aomenc aomdec -j "$(nproc)"

echo "built: $BUILD/{libaom.a, aomenc, aomdec}"
