#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=besteffort              # choix de la priorité gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G                    # memoire par coeur cpu
#SBATCH --gres=gpu:a100-80:1                 # nombre de gpu par noeud
#SBATCH --time=30:00:00
#SBATCH --output=./%j.out                       ###changed!!!
#SBATCH --error=./%j.err

module purge
ml conda
conda activate jsalt2025

# Exit on error
set -e
set -o pipefail


export PYTHONPATH=/home/lis.isabella.gidi/jsalt2025:$PYTHONPATH

srun python3 -u exploratory_analysis.py
