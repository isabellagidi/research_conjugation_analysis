#!/bin/bash
#SBATCH --job-name=actpatch_repair
#SBATCH --partition=besteffort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=06:00:00
#SBATCH -D /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited
#SBATCH -o /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited/logs/repair_%j.out
#SBATCH -e /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited/logs/repair_%j.err

module purge
ml conda
conda activate jsalt2025
set -euo pipefail
set -x

BASEDIR="/home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited"
cd "$BASEDIR"

export PYTHONPATH="/home/lis.isabella.gidi/jsalt2025/src/experiments:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MODEL_NAME="bigscience/bloom-1b1"
MORPHY="/home/lis.isabella.gidi/jsalt2025/src/datasets/morphy_net/MorphyNet_all_present_conjugations.json"

python3 -u repair_with_top_heads.py \
  --model_name "$MODEL_NAME" \
  --results_root results \
  --morphynet_path "$MORPHY" \
  --max_verbs 1300 \
  --max_prompts_head 50 \
  --batch_size 16 \
  --device cuda \
  --out_dir results_summary
