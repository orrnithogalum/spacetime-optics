# Spacetime Optics

Spacetime Optics is a physically based GPU renderer for light bent by gravity, written in C++ and CUDA.  
It shows what a telescope would see from Earth's gravitational focal point.

## Earth as a gravitational lens

A telescope 4,000 to 90,000 AU behind Earth, lined up with an exoplanet (here Proxima b), sees the planet's light bent into an Einstein ring around Earth.

### Features

- General relativity deflection, with Earth's oblateness (J2) and the Sun's tidal field
- Atmospheric refraction up to 1000 km, which beats gravity below ~107 km
- Earth rendered from NASA data:
  - Blue Marble surface, clouds and city lights
  - Ocean sun glint
  - Atmosphere scattering, ozone and airglow
- Spectral rendering (16 bands)
- Telescope view with diffraction through a real aperture
- Web viewer to move the telescope along and across the focal line

## Dependencies
Before compiling, make sure you have:

- an nvidia gpu (sm_80 or newer recommended)
- cuda 12 or newer and zlib
- mpi if you want to render on several gpus
- ffmpeg for the videos

## Installation

Build manually from source:
```bash
git clone https://github.com/orrnithogalum/spacetime-optics.git
cd spacetime-optics

cd lensing && make
```

Use `make NOMPI=1` for a single gpu build, or `make ARCH=sm_89` for other gpus.

## Usage

```bash
cd lensing
scripts/fetch_earth_data.sh
scripts/prepare_earth_data.sh

mpirun -np 8 ./lensim scenes/proxima_b_explorer.ini --out out/explorer
scripts/make_video.sh out/explorer
```

Then open `out/explorer/viewer/index.html`:

- `LEFT / RIGHT` arrows or `SCROLL` → move the telescope along the focal line
- `UP / DOWN` arrows → move it off the axis
- `SPACE` → play
- `T` → switch to the telescope view
