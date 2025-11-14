#!/bin/bash
set -euo pipefail

A="first singular"
B="second singular"
MODEL_NAME="mistralai/Mistral-7B-v0.1"
JBASE="actpatch_${A// /_}_to_${B// /_}_$(basename "$MODEL_NAME" | tr '/:' '__')"

echo "Submitting array: $JBASE"

# Phase 1: selection
jid_sel=$(sbatch --job-name="sel_${JBASE}" \
  --export=ALL,PERSON_A="$A",PERSON_B="$B",PHASE=select,MODEL_NAME="$MODEL_NAME" \
  exec_array.slurm | awk '{print $4}')

echo "Selection array submitted as $jid_sel"

# Phase 2: patch depends on phase 1 success
sbatch --job-name="patch_${JBASE}" \
  --dependency=afterok:$jid_sel \
  --export=ALL,PERSON_A="$A",PERSON_B="$B",PHASE=patch,MODEL_NAME="$MODEL_NAME" \
  exec_array.slurm

echo "Patch array submitted with dependency on $jid_sel"

