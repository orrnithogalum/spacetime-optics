#!/usr/bin/env bash
# Encode a render directory into videos and build the interactive viewer.
#
#   scripts/make_video.sh <render_dir> [fps]
#
# Produces
#   <render_dir>/<name>.mp4               normal playback (H.264, GOP 2 s)
#   <render_dir>/viewer/index.html        scrub the telescope back and forth (open in a browser)
#   <render_dir>/viewer/sweep.mp4         all-intra copy used by the viewer for instant seeking
#   <render_dir>/viewer/sweep_telescope.mp4   (if a telescope view was rendered)
#
# Env: FFMPEG (default ffmpeg), SCRUB_HEIGHT (viewer video height, default 1080), CRF (default 16)
set -euo pipefail
DIR=${1:?usage: make_video.sh <render_dir> [fps]}
FPS=${2:-30}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
FFMPEG=${FFMPEG:-ffmpeg}
SCRUB_HEIGHT=${SCRUB_HEIGHT:-1080}
CRF=${CRF:-16}
NAME=$(basename "$(cd "$DIR" && pwd)")

command -v "$FFMPEG" >/dev/null || { echo "ffmpeg not found (set FFMPEG=...)"; exit 1; }
ENCODERS=$("$FFMPEG" -hide_banner -encoders 2>/dev/null || true)
if grep -q libx264 <<<"$ENCODERS"; then
    CODEC=(-c:v libx264 -preset slow -crf "$CRF" -pix_fmt yuv420p)
else
    echo "note: libx264 unavailable, falling back to mpeg4"
    CODEC=(-c:v mpeg4 -q:v 2 -pix_fmt yuv420p)
fi

encode() {   # frames_dir out.mp4 gop scale_height
    local vf="scale=-2:$4:flags=lanczos"
    "$FFMPEG" -hide_banner -loglevel error -stats -y -framerate "$FPS" -i "$1/frame_%05d.png" \
        -vf "$vf" "${CODEC[@]}" -g "$3" -movflags +faststart "$2"
}

mkdir -p "$DIR/viewer"
# frame height straight from the PNG header (IHDR height = bytes 20..23, big-endian)
H=$(od -An -tu1 -j20 -N4 "$DIR/ideal/frame_00000.png" | awk '{print $1*16777216 + $2*65536 + $3*256 + $4}')
echo "encoding $DIR (frames ${H}p, $FPS fps)"
encode "$DIR/ideal" "$DIR/$NAME.mp4" $((FPS * 2)) "$H"
encode "$DIR/ideal" "$DIR/viewer/sweep.mp4" 1 "$SCRUB_HEIGHT"
if [ -d "$DIR/telescope" ]; then
    encode "$DIR/telescope" "$DIR/$NAME-telescope.mp4" $((FPS * 2)) "$H"
    encode "$DIR/telescope" "$DIR/viewer/sweep_telescope.mp4" 1 "$SCRUB_HEIGHT"
fi
{ printf 'window.LENSIM = '; cat "$DIR/meta.json"; printf ';\nwindow.LENSIM.fps = %s;\n' "$FPS"; } > "$DIR/viewer/meta.js"
cp "$HERE/viewer/index.html" "$DIR/viewer/index.html"
echo "done: $DIR/$NAME.mp4 and $DIR/viewer/index.html"
