#!/bin/bash
# Pack the downloaded NASA imagery into GPU textures (~6 GB RAM, a few minutes).
#   sbatch scripts/slurm/prepare.slurm
#SBATCH --job-name=lensim-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/slurm/env.sh
DEST=${DEST:-data/earth} scripts/prepare_earth_data.sh
