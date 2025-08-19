#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate thesis

python3 ssddpm.py