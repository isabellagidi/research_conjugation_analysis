#!/bin/bash
#SBATCH --job-name=xling_spa_por_1sg_3sg
#SBATCH --partition=besteffort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=24:00:00
#SBATCH -D /home/lis.isabella.gidi/jsalt2025/src/experiments/cross_lingual
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
ml conda
conda activate jsalt2025
set -euo pipefail
mkdir -p logs

export PYTHONPATH="/home/lis.isabella.gidi/jsalt2025/src/experiments:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="bigscience/bloom-1b1"

srun python3 -u cross_lingual_patching.py \
  --model_name "${MODEL_NAME}" \
  --lang_a "spa" \
  --lang_b "por" \
  --slot_x "first singular" \
  --slot_y "first plural" \
  --max_verbs 1300 \
  --max_prompts_head 50 \
  --topk_groups 3 \
  --seed 0
