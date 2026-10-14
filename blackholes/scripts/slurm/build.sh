#!/usr/bin/env bash
# Build on a GPU node (sm_90 by default; pass ARCH=sm_80 etc. through the environment).
set -euo pipefail
cd "$(dirname "$0")/../.."
srun --gres=gpu:1 -c 32 --mem=32G -t 00:20:00 bash -lc "source scripts/slurm/env.sh && make -j32 ${ARCH:+ARCH=$ARCH}"
