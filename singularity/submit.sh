#!/bin/bash

#SBATCH -A col_seales_uksr
#SBATCH --mail-type=END
#SBATCH --job-name=pgs_recon
#SBATCH --output=out/pgs_recon_%A_%a.out

# Make rclone available on the container and tell system where to look
#SBATCH --export=SINGULARITY_BIND='/share/singularity/bin',SINGULARITYENV_PREPEND_PATH='/share/singularity/bin'

module load ccs/singularity

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    time singularity run ${PROJECT}/seales_uksr/containers/pgs_recon.sif "$@"
else
    time singularity run ${PROJECT}/seales_uksr/containers/pgs_recon.sif "$@" -k $SLURM_ARRAY_TASK_ID
fi
