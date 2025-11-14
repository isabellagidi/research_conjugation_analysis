#!/bin/bash
#SBATCH --job-name=compare_incorrect
#SBATCH --partition=besteffort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=02:00:00
#SBATCH -D /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited
#SBATCH -o /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited/logs/compare_incorrect_%j.out
#SBATCH -e /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited/logs/compare_incorrect_%j.err

set -euo pipefail
set -x

module purge
ml conda
conda activate jsalt2025

BASEDIR="/home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited"
cd "$BASEDIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/lis.isabella.gidi/jsalt2025/src/experiments:${PYTHONPATH:-}"

# Inputs (adjust if your tree differs)
CORRECT_BASE="results"
INCORRECT_BASE="incorrect/results"
OUTDIR="compare_outputs"

# Run (CPU is fine; GPU not required, but we keep the same template)
srun -u python -u compare_incorrect.py \
  --base_dir "$CORRECT_BASE" \
  --incorrect_base "$INCORRECT_BASE" \
  --output_dir "$OUTDIR" \
  --bins 30
