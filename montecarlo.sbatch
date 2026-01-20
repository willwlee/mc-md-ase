#!/bin/bash
#SBATCH --partition=granite-gpu
#SBATCH --qos=granite-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=prameela
#SBATCH -o mc.out
#SBATCH -e mc.err
#SBATCH --time=24:00:00
#SBATCH --mem=50G

#Usage: sbatch montecarlo.sbatch <nsteps> <temp> <disl: True/False>
steps=$1
temp=$2
disl=$3

module load miniforge3
source activate base
conda activate ase_new
module load cuda

python3 montecarlo.py "$@"
