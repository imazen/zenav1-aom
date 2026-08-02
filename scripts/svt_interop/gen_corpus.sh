#!/usr/bin/env bash
# gen_corpus.sh — build the issue-#5 SVT-AV1 screen-content interop corpus.
#
# Tiles every gb82-sc source into fixed-size crops at explicit offsets, encodes
# each with the REAL C SVT-AV1 (screen-content mode ON, so IntraBC + palette are
# armed), and writes a manifest the `svt_interop_probe` example consumes.
#
# The issue's named repro is a 512^2 crop of gb82-sc/windows95.png at preset 3 /
# qp 48 -- but windows95.png is 640x480, so no 512^2 crop of it exists. The
# corpus therefore uses per-source tile sizes (512^2 where the source allows it,
# 448^2 for windows95) and sweeps offsets/presets/quantizers to cover the
# per-neighbour-configuration nature of the bug: one stream may not reach it.
#
# Env:
#   ZEN     sibling-repo root      (default ~/work/zen)
#   OUT     output dir             (default ~/tmp/i5)
#   ENC     svt_scc_encode binary  (default ~/tmp/svt_interop/svt_scc_encode)
#   SCM     SVT screen_content_mode (default 1 = forced on)
#   PRESETS space-separated preset list (default "3")
#   QPS     space-separated qp list     (default "48")
set -euo pipefail

ZEN=${ZEN:-$HOME/work/zen}
OUT=${OUT:-$HOME/tmp/i5}
ENC=${ENC:-$HOME/tmp/svt_interop/svt_scc_encode}
SCM=${SCM:-1}
PRESETS=${PRESETS:-3}
QPS=${QPS:-48}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
XTOOL=${XTOOL:-$HERE/benchmarks/xbench/target/release/xtool}
SRC=$ZEN/codec-corpus/gb82-sc

[ -x "$ENC" ] || { echo "missing $ENC -- run scripts/svt_interop/build.sh" >&2; exit 1; }
[ -x "$XTOOL" ] || { echo "missing $XTOOL -- cargo build --release -p xtool in benchmarks/xbench" >&2; exit 1; }

mkdir -p "$OUT/yuv" "$OUT/obu"
MAN=$OUT/manifest.tsv
: >"$MAN"

# `name:WxH:offsets` -- offsets are space-separated `X,Y` into the source PNG.
# TILESET=square (default) uses square crops sized to fit each source; TILESET=geom
# sweeps the frame GEOMETRY instead (SB-size selection, tile grid, wavefront /
# `total_sb64_per_row` -- the quantities `av1_is_dv_valid` and the INTRA_FRAME
# ref-mv scan actually depend on), which a fixed 512^2 crop cannot vary.
TILESET=${TILESET:-square}
if [ "$TILESET" = square ]; then
TILES=(
"codec_wiki:512x512:0,0 512,256 1024,512 1600,900 2048,1152"
"gmessages:512x512:0,0 256,512 512,1024 900,1600 928,2576"
"graph:448x448:0,0 348,0 0,33 174,16 348,33"
"gui:512x512:0,0 400,300 844,620 200,200 600,100"
"imac_dark:512x512:0,0 800,400 1600,800 2428,1400 1200,600"
"imac_g3:512x512:0,0 800,400 1600,800 2428,1400 1200,600"
"imessage:512x512:0,0 200,600 400,1200 694,2110 100,1800"
"terminal:512x512:0,0 400,200 800,400 1134,550 200,300"
"windows:512x512:0,0 700,300 1400,600 2048,880 1000,400"
"windows95:448x448:0,0 192,0 0,32 96,16 192,32"
)
else
# Geometry sweep: widths that cross SB64/SB128 column counts, non-square shapes,
# non-multiple-of-SB extents (partial edge SBs), and full-HD/native sizes.
TILES=(
"codec_wiki:2560x1664:0,0"
"codec_wiki:1920x1080:0,0 320,300"
"codec_wiki:1280x720:0,0 640,400"
"codec_wiki:2048x256:0,0"
"codec_wiki:256x1536:0,0"
"codec_wiki:1284x716:0,0"
"terminal:1646x1062:0,0"
"terminal:1600x1024:0,0"
"terminal:1092x772:0,0"
"terminal:832x64:0,0"
"windows:2560x1392:0,0"
"windows:1920x1080:0,0 400,200"
"windows:1156x644:0,0"
"windows95:640x480:0,0"
"windows95:640x128:0,0"
"windows95:128x480:0,0"
"gui:1356x1132:0,0"
"gui:1024x1024:0,0"
"gui:772x516:0,0"
"imac_dark:2940x1912:0,0"
"imac_dark:1920x1080:0,0"
"imessage:1206x2622:0,0"
"imessage:1200x2048:0,0"
"gmessages:1440x3088:0,0"
"graph:796x480:0,0"
)
fi

n=0
for row in "${TILES[@]}"; do
  name=${row%%:*}; rest=${row#*:}
  dims=${rest%%:*}; offs=${rest#*:}
  cw=${dims%%x*}; ch=${dims##*x}
  png=$SRC/$name.png
  [ -f "$png" ] || { echo "missing $png" >&2; exit 1; }
  for off in $(echo "$offs" | tr ' ' '\n'); do
    ox=${off%%,*}; oy=${off##*,}
    yuv=$OUT/yuv/${name}_${cw}x${ch}_${ox}_${oy}.yuv
    [ -f "$yuv" ] || "$XTOOL" prep "$png" "$yuv" "at:${cw}x${ch}+${ox}+${oy}" >/dev/null
    for p in $PRESETS; do
      for q in $QPS; do
        cell=${name}_${cw}x${ch}_${ox}_${oy}_p${p}q${q}_scm${SCM}
        obu=$OUT/obu/$cell.obu
        [ -f "$obu" ] || "$ENC" "$cw" "$ch" "$q" "$p" "$SCM" "$yuv" "$obu" >/dev/null 2>&1
        printf '%s\t%s\t%s\t%s\n' "$cell" "$cw" "$ch" "$obu" >>"$MAN"
        n=$((n + 1))
      done
    done
  done
done
echo "$n cells -> $MAN"
