#!/bin/bash
# Convert the downloaded star map into the renderer's sky texture (~10 GB RAM, a few minutes).
#   sbatch scripts/slurm/prepare_sky.slurm
#SBATCH --job-name=bh-sky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
RES=${RES:-32k}
./bh-prep-sky data/sky/starmap_2020_${RES}_gal.exr data/sky/sky.bhsk
