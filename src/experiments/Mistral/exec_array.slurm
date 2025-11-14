#!/bin/bash
#SBATCH --job-name=actpatch
#SBATCH --partition=besteffort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# If your scheduler supports total memory, prefer this next line:
#SBATCH --mem=32G
# (If you must use per-CPU: comment the above and use the next two lines)
##SBATCH --mem-per-cpu=8G
##SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=30:00:00
#SBATCH --array=0-4%4
#SBATCH -D /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited/Mistral
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module purge
ml conda
conda activate jsalt2025
set -euo pipefail
set -x

# -------- Env that saves time/memory ----------
export HF_HOME=/home/lis.isabella.gidi/.cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------- Paths ----------
BASEDIR="/home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited/Mistral"
cd "$BASEDIR"
mkdir -p "$BASEDIR/logs"

export PYTHONPATH="/home/lis.isabella.gidi/jsalt2025/src/experiments:${PYTHONPATH:-}"

# Model name can be injected from submit_all; default kept
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-7B-v0.1}"

# ---- languages for this array (German, French, Spanish, Italian, Portuguese) ----
LANGS=(eng swe deu fra spa ita por)
LANG_ISO3=${LANGS[$SLURM_ARRAY_TASK_ID]}
export LANG_ISO3   

# Logs
LOGDIR="$BASEDIR/logs"
MODEL_SAFE="$(echo "$MODEL_NAME" | tr '/:' '__')"
PAIR_SAFE="$(echo "${PERSON_A}_to_${PERSON_B}" | tr ' /' '__')"
LOGBASE="${LOGDIR}/${MODEL_SAFE}-${PAIR_SAFE}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-${LANG_ISO3}"

exec > >(tee -a "${LOGBASE}.out")
exec 2> >(tee -a "${LOGBASE}.err" >&2)

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} -> ${LANG_ISO3}"
echo "PHASE: ${PHASE:-select}"
echo "Pair: ${PERSON_A} -> ${PERSON_B}"
echo "Model: ${MODEL_NAME}"

# I/O
DATASET_JSON="/home/lis.isabella.gidi/jsalt2025/src/datasets/morphy_net/MorphyNet_all_present_conjugations.json"
OUT_BASE="$BASEDIR/results"
SEL_DIR="$OUT_BASE/selections"
PATCH_DIR="$OUT_BASE/patches"
mkdir -p "$SEL_DIR" "$PATCH_DIR"

SEL_PATH="${SEL_DIR}/${MODEL_SAFE}-${PAIR_SAFE}-${LANG_ISO3}.json"
OUT_DIR_LANG="${PATCH_DIR}/${MODEL_SAFE}-${PAIR_SAFE}-${LANG_ISO3}"

# Phase logic
if [[ "${PHASE:-select}" == "select" ]]; then
  srun python3 -u select_examples.py \
    --model_name "${MODEL_NAME}" \
    --lang_iso3 "${LANG_ISO3}" \
    --lang_name "$(python - <<'PY'
LANG_MAP={'cat':'catalan','ces':'czech','deu':'german','eng':'english','fin':'finnish','fra':'french','hbs':'serbo-croatian','hun':'hungarian','ita':'italian','mon':'mongolian','pol':'polish','por':'portuguese','rus':'russian','spa':'spanish','swe':'swedish'}
import os; print(LANG_MAP[os.environ["LANG_ISO3"]])
PY
)" \
    --person_a "${PERSON_A}" \
    --person_b "${PERSON_B}" \
    --max_verbs 1300 \
    --keep_k 50 \
    --dataset_path "${DATASET_JSON}" \
    --out_path "${SEL_PATH}"

elif [[ "${PHASE}" == "patch" ]]; then
  # (defensive) wait until selection file exists; protects against race in dependency scheduling
  for _try in {1..60}; do
    [[ -s "${SEL_PATH}" ]] && break
    echo "Waiting for ${SEL_PATH} ..."; sleep 30
  done
  srun python3 -u patch_selected.py \
    --tl_model_name "${MODEL_NAME}" \
    --selections_path "${SEL_PATH}" \
    --out_dir "${OUT_DIR_LANG}" \
    --chunk_size 4 \
    --max_seq_len 64
fi
