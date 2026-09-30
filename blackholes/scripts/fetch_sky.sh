#!/usr/bin/env bash
# Download NASA SVS "Deep Star Maps 2020" (Gaia DR2 + Hipparcos-2 + Tycho-2, 1.7 billion stars),
# galactic coordinates, OpenEXR (ZIP, half float). Run where there is internet access.
#   RES=32k scripts/fetch_sky.sh          # 16k (0.4 GB), 32k (1.3 GB) or 64k (4 GB)
# Credit: NASA/Goddard Space Flight Center Scientific Visualization Studio; Gaia DR2: ESA/Gaia/DPAC.
set -euo pipefail
RES=${RES:-32k}
DEST=${DEST:-data/sky}
mkdir -p "$DEST"
url=https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_${RES}_gal.exr
out="$DEST/starmap_2020_${RES}_gal.exr"
if [ -s "$out" ]; then echo "have $out"; exit 0; fi
curl -fL --retry 5 --retry-delay 10 -C - -o "$out.part" "$url"
mv "$out.part" "$out"
echo "downloaded $out"
