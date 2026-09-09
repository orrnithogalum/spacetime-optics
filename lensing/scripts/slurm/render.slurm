#!/bin/bash
# Multi-node render: one MPI rank per GPU, frames handed out dynamically.
#   sbatch --nodes=8 scripts/slurm/render.slurm scenes/proxima_b.ini [extra lensim args]
# One rank per CPU socket keeps each rank's memory and threads next to its GPU (e.g. GH200 nodes,
# 4 sockets x 1 GPU). mpirun is used because many Open MPI builds lack Slurm PMI support.
#SBATCH --job-name=lensim
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/env.sh
SCENE=${1:-scenes/proxima_b.ini}
shift || true
OUT=${OUT:-$OUT_ROOT/$(basename "$SCENE" .ini)}
mkdir -p "$OUT"
mpirun -np "$SLURM_NTASKS" --map-by ppr:1:socket:PE=${SLURM_CPUS_PER_TASK} --bind-to core \
       -x OMPI_MCA_coll_hcoll_enable -x LD_LIBRARY_PATH ./lensim "$SCENE" --out "$OUT" "$@"
if [ "${VIDEO:-0}" = 1 ]; then scripts/make_video.sh "$OUT" "${FPS:-30}"; fi
