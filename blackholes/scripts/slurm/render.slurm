#!/bin/bash
# Render one shot: one MPI rank per GPU, frames handed out dynamically.
#   sbatch --nodes=8 scripts/slurm/render.slurm shots/02_photon_ring_zoom.ini [extra bhrt args]
# mpirun is used because many Open MPI builds lack Slurm PMI support.
#SBATCH --job-name=bhrt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/env.sh
SHOT=${1:?shot file}
shift || true
OUT=${OUT:-$OUT_ROOT/$(basename "$SHOT" .ini)}
mkdir -p "$OUT"
mpirun -np "$SLURM_NTASKS" --map-by ppr:1:socket:PE=${SLURM_CPUS_PER_TASK} --bind-to core \
       -x OMPI_MCA_coll_hcoll_enable -x LD_LIBRARY_PATH ./bhrt "$SHOT" --out "$OUT" "$@"
if [ "${VIDEO:-0}" = 1 ]; then scripts/encode_shot.sh "$OUT"; fi
