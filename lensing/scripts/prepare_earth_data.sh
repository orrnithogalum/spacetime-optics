#!/usr/bin/env bash
# Convert the downloaded imagery to PNG (ffmpeg) and pack it into the renderer's textures.
#   DEST=data/earth WIDTH=21600 scripts/prepare_earth_data.sh
# On a cluster, run it on a compute node: sbatch scripts/slurm/prepare.slurm
set -euo pipefail

DEST=${DEST:-data/earth}
WIDTH=${WIDTH:-}
RAW="$DEST/raw"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
FFMPEG=${FFMPEG:-ffmpeg}

command -v "$FFMPEG" >/dev/null || { echo "ffmpeg not found (set FFMPEG=...)"; exit 1; }

topng() {   # in out
    [ -s "$2" ] && { echo "have $2"; return; }
    "$FFMPEG" -hide_banner -loglevel error -y -i "$1" -pix_fmt gray -frames:v 1 "$2"
}
topng "$RAW/clouds.tif" "$RAW/clouds.png"
topng "$RAW/lights.jpg" "$RAW/lights.png"

args=(--albedo "$RAW/albedo.png" --clouds "$RAW/clouds.png" --lights "$RAW/lights.png"
      --elev "$RAW/elev.png" --bathy "$RAW/bathy.png" --out "$DEST")
[ -n "$WIDTH" ] && args+=(--width "$WIDTH")
"$HERE/lensim-prep" "${args[@]}"
