# Source on the cluster before building or running:  source scripts/slurm/env.sh
# Module names differ between sites: set MODULES to whatever provides CUDA >= 12 and a CUDA-aware
# Open MPI. Slurm account/partition come from the standard SBATCH_ACCOUNT / SBATCH_PARTITION
# (and SLURM_ACCOUNT / SLURM_PARTITION for srun) environment variables.
module load ${MODULES:-cuda openmpi} >/dev/null 2>&1 || true
export OMPI_MCA_coll_hcoll_enable=0
export OUT_ROOT=${OUT_ROOT:-$HOME/renders}
