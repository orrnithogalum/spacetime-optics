#!/usr/bin/env bash
# Encode a rendered shot (frames/frame_%05d.png) to <dir>/<name>.mp4 (H.264, libx264).
set -euo pipefail
DIR=${1:?usage: encode_shot.sh <shot_dir> [fps]}
FPS=${2:-30}
FFMPEG=${FFMPEG:-ffmpeg}
command -v "$FFMPEG" >/dev/null || { echo "ffmpeg not found (set FFMPEG=...)"; exit 1; }
NAME=$(basename "$(cd "$DIR" && pwd)")
"$FFMPEG" -hide_banner -loglevel error -y -framerate "$FPS" -i "$DIR/frames/frame_%05d.png" \
    -c:v libx264 -preset slow -crf ${CRF:-15} -pix_fmt yuv420p -movflags +faststart "$DIR/$NAME.mp4"
echo "wrote $DIR/$NAME.mp4"
