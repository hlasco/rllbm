#!/bin/bash
#SBATCH --partition=milan7513_A100
#SBATCH --error=slurm_log.err
#SBATCH --output=slurm_log.out
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus=1



eval "$(conda shell.bash hook)"
conda activate rllbm

python train_sac.py
