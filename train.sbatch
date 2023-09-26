#!/bin/bash

#SBATCH --time=0-15:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1

source ~/.bashrc
source $PREAMBLE
conda activate wb

srun --mpi=pmix "$@"