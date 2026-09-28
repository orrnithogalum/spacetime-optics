# Spacetime Optics

Spacetime Optics is a set of physically based GPU renderers for light bent by gravity, written in C++ and CUDA.  
It renders the photon rings of black holes, and what a telescope would see from Earth's gravitational focal point.

## Earth as a gravitational lens

<p align="center">
  <img src="docs/images/earth.jpg" alt="Earth" width="270">
  <img src="docs/images/earth_einstein_ring.jpg" alt="Einstein ring" width="270">
  <img src="docs/images/earth_arcs.jpg" alt="Off axis arcs" width="270">
</p>

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

> [!TIP]
> Run `./lensim scenes/proxima_b.ini --report` to print the refraction and gravity tables.

## Black holes

### Features

- Kerr geodesics traced in double precision (Carter-Mino equations, adaptive Dormand-Prince)
- M87* at 230 GHz:
  - Thermal synchrotron emission from a hot accretion flow
  - Flux normalised to the 0.6 Jy seen by the Event Horizon Telescope
  - Animated turbulence that keeps its light travel delays
- Thin accretion disk in true colours
- Self test against the analytic shadow edge
- Built with **CUDA** and **MPI**, frames are spread across all the gpus of one or more machines

## Dependencies
Before compiling, make sure you have:

- an nvidia gpu (sm_80 or newer recommended, the geodesics use fp64)
- cuda 12 or newer and zlib
- mpi if you want to render on several gpus
- ffmpeg for the videos

## Installation

Build manually from source:
```bash
git clone https://github.com/orrnithogalum/spacetime-optics.git
cd spacetime-optics

cd blackholes && make && cd ..
cd lensing && make && cd ..
```

Use `make NOMPI=1` for a single gpu build, or `make ARCH=sm_89` for other gpus.

## Usage

### Earth lens

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

### Black holes

```bash
cd blackholes
./bhrt --selftest
```

### Slurm clusters

> [!NOTE]
> Set `SBATCH_ACCOUNT` and `SBATCH_PARTITION` first, and `MODULES` if your cuda / mpi modules have other names.

```bash
source scripts/slurm/env.sh
sbatch --nodes=4 scripts/slurm/render.slurm scenes/proxima_b_cinematic.ini
```

Use one MPI rank per gpu.
