# compare_logit_methods.py
# Compare heatmap tensors from two pipelines (mega_automation_edited vs mega_automation)
# by cosine similarity / Pearson correlation per (pair, language, direction).
#
# Expected layout for BOTH roots:
# <ROOT>/results/bigscience/bloom-1b1/<pair>/<language>/
#     attn_head_out_all_pos_<direction>_<language>.pt
#
# Example direction: 1sgto2sg, 2sgto1sg, etc.

import os
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
import json

# ---------------- Config ----------------
EDITED_ROOT   = Path("mega_automation_edited") / "results" / "bigscience" / "bloom-1b1"
BASELINE_ROOT = Path("mega_automation")        / "results" / "bigscience" / "bloom-1b1"

OUT_DIR       = Path("logit_comparisons")  # all outputs will go here
SAVE_CSV      = True                       # write a summary csv
SAVE_JSON     = True                       # write a summary json
PLOT_COSINE   = True                       # make per-direction cosine bar charts
PLOT_PEARSON  = True                       # make per-direction pearson bar charts
PLOT_HIST     = True                       # global histograms for cosine/pearson
# ----------------------------------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_direction_files(pair_dir: Path, language: str) -> List[Path]:
    """
    Return all matching files in <pair_dir>/<language> that follow:
    attn_head_out_all_pos_<direction>_<language>.pt
    (We parse <direction> from the filename.)
    """
    lang_dir = pair_dir / language
    if not lang_dir.is_dir():
        return []
    pattern = re.compile(rf"^attn_head_out_all_pos_(.+)_{re.escape(language)}\.pt$")
    files = []
    for f in lang_dir.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                files.append(f)
    return sorted(files)


def extract_direction_from_filename(fname: str, language: str) -> str:
    """
    Given 'attn_head_out_all_pos_1sgto2sg_spanish.pt' and language='spanish',
    return '1sgto2sg'.
    """
    prefix = "attn_head_out_all_pos_"
    suffix = f"_{language}.pt"
    assert fname.startswith(prefix) and fname.endswith(suffix), f"Unexpected filename: {fname}"
    return fname[len(prefix) : -len(suffix)]


def load_tensor(path: Path) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor at {path}, got {type(t)}")
    return t.to(dtype=torch.float32, device="cpu")


def align_tensors(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """
    If shapes differ, crop to overlapping top-left region per dimension.
    If ranks differ, flatten and crop to min length.
    Returns (a_aligned, b_aligned, used_shape)
    """
    if a.shape == b.shape:
        return a, b, tuple(a.shape)

    if len(a.shape) != len(b.shape):
        af = a.flatten()
        bf = b.flatten()
        n = min(af.numel(), bf.numel())
        return af[:n], bf[:n], (n,)

    used_dims = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
    sl = tuple(slice(0, d) for d in used_dims)
    return a[sl], b[sl], used_dims


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.reshape(-1)
    bf = b.reshape(-1)
    denom = (af.norm() * bf.norm()).item()
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(af, bf).item() / denom)


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.reshape(-1).numpy()
    bf = b.reshape(-1).numpy()
    sa = float(np.std(af))
    sb = float(np.std(bf))
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(af, bf)[0, 1])


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a.reshape(-1) - b.reshape(-1)
    return float((diff * diff).mean().item())


def collect_all_comparisons() -> List[Dict[str, Any]]:
    """
    Walk EDITED_ROOT and for each (pair, language, direction) found there,
    try to load the exact same file from BASELINE_ROOT, then compute metrics.
    """
    rows: List[Dict[str, Any]] = []

    if not EDITED_ROOT.exists():
        print(f"❌ Edited root not found: {EDITED_ROOT}")
        return rows
    if not BASELINE_ROOT.exists():
        print(f"❌ Baseline root not found: {BASELINE_ROOT}")
        return rows

    for pair_dir in sorted(d for d in EDITED_ROOT.iterdir() if d.is_dir()):
        pair = pair_dir.name  # e.g., "1sg-2sg"
        for lang_dir in sorted(d for d in pair_dir.iterdir() if d.is_dir()):
            language = lang_dir.name  # e.g., "spanish"
            edited_files = list_direction_files(pair_dir, language)
            if not edited_files:
                continue

            for edited_path in edited_files:
                direction = extract_direction_from_filename(edited_path.name, language)
                # Construct baseline path
                baseline_path = BASELINE_ROOT / pair / language / edited_path.name
                if not baseline_path.exists():
                    print(f"⚠️ Missing baseline file: {baseline_path}")
                    continue

                try:
                    t_edit = load_tensor(edited_path)
                    t_base = load_tensor(baseline_path)
                except Exception as e:
                    print(f"⚠️ Failed loading tensors for {pair}/{language}/{direction}: {e}")
                    continue

                tE, tB, used_shape = align_tensors(t_edit, t_base)
                cos = cosine_similarity(tE, tB)
                pcc = pearson_corr(tE, tB)
                err = mse(tE, tB)

                rows.append({
                    "pair": pair,
                    "language": language,
                    "direction": direction,
                    "edited_path": str(edited_path),
                    "baseline_path": str(baseline_path),
                    "edited_shape": tuple(t_edit.shape),
                    "baseline_shape": tuple(t_base.shape),
                    "used_shape": used_shape,
                    "cosine": cos,
                    "pearson": pcc,
                    "mse": err,
                })
                print(f"[{pair} | {language} | {direction}] cos={cos:.4f} pearson={pcc:.4f} mse={err:.6g} used={used_shape}")

    return rows


def plot_bars_per_direction(rows: List[Dict[str, Any]], metric: str, out_root: Path):
    """
    For each (pair, direction), make a bar chart over languages for the given metric.
    """
    # group by (pair, direction)
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["pair"], r["direction"])
        by_key.setdefault(key, []).append(r)

    for (pair, direction), items in by_key.items():
        langs = [r["language"] for r in items]
        vals  = [r[metric] for r in items]

        # Sort by language for consistent ordering
        langs, vals = zip(*sorted(zip(langs, vals), key=lambda kv: kv[0]))

        fig = plt.figure(figsize=(10, 5))
        plt.bar(langs, vals)
        plt.title(f"{metric.capitalize()} — Edited vs Baseline\npair={pair} • dir={direction}")
        plt.xlabel("Language")
        plt.ylabel(metric.capitalize())
        # annotate
        for x, v in enumerate(vals):
            if v == v:  # not NaN
                plt.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        out_dir = out_root / "plots" / pair / direction
        ensure_dir(out_dir)
        out_path = out_dir / f"{metric}_bars_{pair}_{direction}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"✅ Saved {out_path}")


def plot_global_hists(rows: List[Dict[str, Any]], out_root: Path):
    metrics = ["cosine", "pearson"]
    for metric in metrics:
        vals = [r[metric] for r in rows if r[metric] == r[metric]]  # drop NaNs
        if not vals:
            continue
        fig = plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=30)
        plt.title(f"Distribution of {metric} across all (pair, lang, dir)")
        plt.xlabel(metric.capitalize())
        plt.ylabel("Count")
        plt.tight_layout()
        out_path = out_root / f"hist_{metric}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"✅ Saved {out_path}")


def save_table(rows: List[Dict[str, Any]], out_root: Path):
    if SAVE_CSV:
        out_csv = out_root / "logit_comparison_summary.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"💾 Wrote {out_csv}")

    if SAVE_JSON:
        out_json = out_root / "logit_comparison_summary.json"
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"💾 Wrote {out_json}")


def main():
    ensure_dir(OUT_DIR)
    rows = collect_all_comparisons()
    if not rows:
        print("No comparisons computed. Check roots and filenames.")
        return

    # Save tables
    save_table(rows, OUT_DIR)

    # Plots
    if PLOT_COSINE:
        plot_bars_per_direction(rows, metric="cosine", out_root=OUT_DIR)
    if PLOT_PEARSON:
        plot_bars_per_direction(rows, metric="pearson", out_root=OUT_DIR)
    if PLOT_HIST:
        plot_global_hists(rows, out_root=OUT_DIR)


if __name__ == "__main__":
    main()
