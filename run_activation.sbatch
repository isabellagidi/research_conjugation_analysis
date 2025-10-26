#!/bin/bash
#SBATCH -J actpatch
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-12%4   # 13 langs, run at most 4 concurrently

# ---- user config -------------------------------------------------------
MODEL_NAME=${MODEL_NAME:-bigscience/bloom-1b1}

# Languages (indices 0..12)
LANGS=(cat ces deu eng fin fra hun ita mon por rus spa swe)

# Map array index -> ISO3
LANG_ISO3=${LANGS[$SLURM_ARRAY_TASK_ID]}

# PERSON_A and PERSON_B are substituted via --export from submit_all.sh
if [[ -z "$PERSON_A" || -z "$PERSON_B" ]]; then
  echo "ERROR: PERSON_A and PERSON_B must be set (e.g., first singular / second singular)."
  exit 1
fi

mkdir -p logs

# (Optional) modules / env
# module load anaconda
# conda activate yourenv

# ---- run ---------------------------------------------------------------
python your_script.py \
  --model_name "$MODEL_NAME" \
  --lang_iso3 "$LANG_ISO3" \
  --person_a "$PERSON_A" \
  --person_b "$PERSON_B"
