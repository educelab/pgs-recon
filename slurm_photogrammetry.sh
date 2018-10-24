#!/bin/bash

#SBATCH --job-name=slurm_photogrammetry
#SBATCH --output=out/slurm_photogrammetry_%j.out

#SBATCH -pFatComp

# cat /proc/meminfo | grep MemTotal
time python3 photogrammetry.py "$@"
