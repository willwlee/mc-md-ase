#!/bin/bash
#SBATCH --partition=granite-gpu
#SBATCH --qos=granite-gpu-freecycle
#SBATCH --account=prameela
#SBATCH --gres=gpu:l40s:4

#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=6
#SBATCH --output=mc.out
#SBATCH --error=mc.err

#Usage: sbatch montecarlo.sbatch <temp> <nsteps> <disl: True/False>
temp=$1
steps=$2
disl=$3

module load miniforge3
source activate base
conda activate ase_new
module load cuda

python3 montecarlo.py "$@"
