#!/usr/bin/env bash
# Submit every shot of the film at 4K (each is encoded when done), then: scripts/make_film.sh $OUT_ROOT
set -euo pipefail
cd "$(dirname "$0")/../.."
export VIDEO=1
sbatch -N1 -t 00:20:00 scripts/slurm/render.slurm shots/00_title.ini
sbatch -N3 -t 01:00:00 scripts/slurm/render.slurm shots/01_m87_eht.ini
sbatch -N4 -t 02:00:00 scripts/slurm/render.slurm shots/02_photon_ring_zoom.ini
sbatch -N3 -t 01:00:00 scripts/slurm/render.slurm shots/03_gaia_bh3.ini
sbatch -N3 -t 01:00:00 scripts/slurm/render.slurm shots/04_quasar_disk.ini
sbatch -N1 -t 00:20:00 scripts/slurm/render.slurm shots/05_end.ini
