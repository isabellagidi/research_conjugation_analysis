#!/bin/bash
set -euo pipefail

PAIRS=(
  "first plural|second plural"
  "first singular|first plural"
  "first singular|second singular"
  "first singular|third singular"
)

for pair in "${PAIRS[@]}"; do
  IFS='|' read -r A B <<< "$pair"
  JNAME="actpatch_$(echo "$A" | tr ' ' '_')_to_$(echo "$B" | tr ' ' '_')"
  echo "Submitting array: $JNAME"
  sbatch --job-name="$JNAME" \
         --export=ALL,PERSON_A="$A",PERSON_B="$B" \
         exec_array.slurm
done

