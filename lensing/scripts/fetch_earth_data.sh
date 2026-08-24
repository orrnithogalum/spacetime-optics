#!/usr/bin/env bash
# Download the NASA imagery (public domain) used for the Earth model.
# Run on a machine with internet access (on a cluster: usually the login node).
#
#   MONTH=05 RES=21600 DEST=data/earth scripts/fetch_earth_data.sh
#
#   MONTH : Blue Marble Next Generation month 01..12 (snow cover / vegetation of that season;
#           June is not published, July is used instead)
#   RES   : 21600 (500 m/px at the equator, ~125 MB) or 5400 (quick tests)
set -euo pipefail

MONTH=${MONTH:-05}
RES=${RES:-21600}
DEST=${DEST:-data/earth}
RAW="$DEST/raw"
mkdir -p "$RAW"

base=https://eoimages.gsfc.nasa.gov/images/imagerecords
declare -A BMNG=([01]=73938 [02]=73967 [03]=73992 [04]=74017 [05]=74042 [06]=74092
                 [07]=74092 [08]=74117 [09]=74142 [10]=74167 [11]=74192 [12]=74218)
m=$MONTH
[ "$m" = 06 ] && m=07
id=${BMNG[$MONTH]}
if [ "$RES" = 21600 ]; then size=21600x10800; else size=5400x2700; fi

fetch() {   # url dest
    if [ -s "$2" ]; then echo "have $2"; return; fi
    echo "downloading $1"
    curl -fL --retry 3 --retry-delay 5 -o "$2.part" "$1"
    mv "$2.part" "$2"
}

fetch "$base/$((id / 1000 * 1000))/$id/world.2004$m.3x$size.png"          "$RAW/albedo.png"
fetch "$base/57000/57747/cloud_combined_8192.tif"                         "$RAW/clouds.tif"
fetch "$base/144000/144898/BlackMarble_2016_3km.jpg"                      "$RAW/lights.jpg"
fetch "$base/73000/73934/gebco_08_rev_elev_21600x10800.png"               "$RAW/elev.png"
fetch "$base/73000/73963/gebco_08_rev_bath_21600x10800.png"               "$RAW/bathy.png"
echo "raw data in $RAW -- now run scripts/prepare_earth_data.sh (needs ffmpeg + lensim-prep)"
