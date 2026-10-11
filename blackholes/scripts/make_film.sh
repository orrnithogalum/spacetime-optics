#!/usr/bin/env bash
# Concatenate the encoded shots into the film:  scripts/make_film.sh <render_root> [out.mp4]
set -euo pipefail
ROOT=${1:?usage: make_film.sh <render_root> [out.mp4]}
OUT=${2:-$ROOT/photon_ring_film.mp4}
FFMPEG=${FFMPEG:-ffmpeg}
command -v "$FFMPEG" >/dev/null || { echo "ffmpeg not found (set FFMPEG=...)"; exit 1; }
LIST=$(mktemp)
for s in 00_title 01_m87_eht 02_photon_ring_zoom 03_gaia_bh3 04_quasar_disk 05_end; do
    [ -f "$ROOT/$s/$s.mp4" ] && echo "file '$ROOT/$s/$s.mp4'" >> "$LIST"
done
"$FFMPEG" -hide_banner -loglevel error -y -f concat -safe 0 -i "$LIST" -c copy "$OUT"
rm -f "$LIST"
echo "wrote $OUT"
