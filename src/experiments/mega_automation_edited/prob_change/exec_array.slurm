#!/bin/bash
#SBATCH --job-name=actpatch
#SBATCH --partition=besteffort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=30:00:00
#SBATCH --array=0-7%4
#SBATCH -D /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited/prob_change
# prevent duplicate SLURM-captured files since we manage our own logs:
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module purge
ml conda
conda activate jsalt2025
set -euo pipefail
set -x

BASEDIR="/home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited/prob_change" # EDIT HERE (and above) (and below!!)
cd "$BASEDIR"
mkdir -p "$BASEDIR/logs"

export PYTHONPATH="/home/lis.isabella.gidi/jsalt2025/src/experiments:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- single knob to change the model ----
MODEL_NAME="bigscience/bloom-1b1"   # <-- change this ONE line to switch models (NOTE: if switch from BLOOMz need to change function)

# Map array index to language
LANGS=(cat eng fra ita por rus spa swe)
LANG_ISO3=${LANGS[$SLURM_ARRAY_TASK_ID]}

# ---- model-first per-task logs (separate .out / .err) ----
LOGDIR="$BASEDIR/logs"
MODEL_SAFE="$(echo "$MODEL_NAME" | tr '/:' '__')"
PAIR_SAFE="$(echo "${PERSON_A}_to_${PERSON_B}" | tr ' /' '__')"
LOGBASE="${LOGDIR}/${MODEL_SAFE}-${PAIR_SAFE}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-${LANG_ISO3}"

# Stream stdout to .out and stderr to .err (no SLURM-captured files since we set /dev/null)
exec > >(tee -a "${LOGBASE}.out")
exec 2> >(tee -a "${LOGBASE}.err" >&2)

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} -> ${LANG_ISO3}"
echo "Pair: ${PERSON_A} -> ${PERSON_B}"
echo "Model: ${MODEL_NAME}"

srun python3 -u mega_automation_prob_change.py \ 
  --model_name "${MODEL_NAME}" \
  --lang_iso3 "${LANG_ISO3}" \
  --person_a "${PERSON_A}" \
  --person_b "${PERSON_B}"
