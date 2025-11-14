#!/bin/bash
#SBATCH --job-name=plot_cosine_hist
#SBATCH --partition=besteffort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH -D /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited
#SBATCH -o /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited/logs/plot_cosine_%j.out
#SBATCH -e /home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited/logs/plot_cosine_%j.err

set -euo pipefail
set -x

# ---- env ----
module purge
ml conda
conda activate jsalt2025
export PYTHONUNBUFFERED=1

# ---- inputs (edit as needed) ----
BASEDIR="/home/lis.isabella.gidi/jsalt2025/src/experiments/mega_automation_edited"
CSV="${BASEDIR}/compare_outputs/cosine_similarities.csv"         # path to your CSV
OUTPNG="${BASEDIR}/compare_outputs/cosine_histogram_-1_to_1.png" # output plot path
COL="cosine_similarity"                                          # column name in CSV
BINS=30                                                          # number of bins

mkdir -p "$(dirname "$OUTPNG")"

# Pass variables to Python via env
export CSV OUTPNG COL BINS

# ---- run plotting (no GPU needed) ----
python - <<'PY'
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for SLURM nodes
import matplotlib.pyplot as plt
from pathlib import Path

csv   = os.environ["CSV"]
out   = os.environ["OUTPNG"]
col   = os.environ.get("COL", "cosine_similarity")
bins  = int(os.environ.get("BINS", "30"))

print(f"[info] reading: {csv}")
df = pd.read_csv(csv)
if col not in df.columns:
    raise SystemExit(f"Column '{col}' not found in {csv}. Available: {list(df.columns)}")

vals = df[col].to_numpy()
vals = vals[np.isfinite(vals)]
if vals.size == 0:
    raise SystemExit("[info] No finite cosine values to plot.")

edges = np.linspace(-1.0, 1.0, bins + 1)

plt.figure(figsize=(8, 5))
plt.hist(vals, bins=edges)      # full range [-1, 1]
plt.title("Cosine similarities (full range [-1, 1])")
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.xlim(-1.0, 1.0)
plt.tight_layout()

Path(out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=200)
print(f"[info] wrote {out}")
PY
