#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # headless plotting for SLURM nodes
import matplotlib.pyplot as plt


def collect_files(base_dir: Path) -> dict:
    """
    Walk base_dir and collect .pt files into a dict keyed by:
      (org, model, pair, language, direction_label) -> file_path

    Expected layout:
      <base>/<org>/<model>/<pair>/<language>/attn_head_out_all_pos_patch_results_<direction_label>_<language>.pt
    """
    base_dir = base_dir.resolve()
    mapping = {}
    for f in base_dir.rglob("attn_head_out_all_pos_patch_results_*.pt"):
        try:
            rel = f.relative_to(base_dir)
        except Exception:
            continue

        parts = rel.parts
        # Expect: <org>/<model>/<pair>/<language>/<filename>
        if len(parts) < 5:
            continue

        org, model, pair, language = parts[0], parts[1], parts[2], parts[3]
        filename = parts[-1]

        # Parse filename tail robustly:
        # attn_head_out_all_pos_patch_results_<direction_label>_<file_language>.pt
        if not filename.endswith(".pt"):
            continue
        stem = filename[:-3]  # drop ".pt"
        prefix = "attn_head_out_all_pos_patch_results_"
        if not stem.startswith(prefix) or "_" not in stem[len(prefix):]:
            continue
        direction_label, file_language = stem[len(prefix):].rsplit("_", 1)

        if language != file_language:
            print(f"[warn] language mismatch dir='{language}' file='{file_language}' for {f}")

        key = (org, model, pair, language, direction_label)
        mapping[key] = f
    return mapping


def to_numpy(x):
    """Convert a torch tensor or compatible object to numpy array, else None."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, dict) and "data" in x:
        return np.array(x["data"])
    return None


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity on flattened arrays; returns NaN for zero-norm/empty."""
    u = u.ravel().astype(np.float64)
    v = v.ravel().astype(np.float64)
    mask = np.isfinite(u) & np.isfinite(v)
    if not np.any(mask):
        return float("nan")
    u = u[mask]
    v = v[mask]
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def main():
    p = argparse.ArgumentParser(
        description="Compare incorrect vs correct activation patching heatmaps by cosine similarity and plot a histogram."
    )
    p.add_argument("--base_dir", default="results", help="Base for correct results.")
    p.add_argument("--incorrect_base", default="incorrect/results", help="Base for incorrect results.")
    p.add_argument("--output_dir", default="compare_outputs", help="Directory to write CSV/PNG.")
    p.add_argument("--bins", type=int, default=20, help="Histogram bins between 0 and 1.")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    incorrect_base = Path(args.incorrect_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] scanning correct:   {base_dir.resolve()}")
    print(f"[info] scanning incorrect: {incorrect_base.resolve()}")
    print(f"[info] output directory:   {output_dir.resolve()}")

    correct_map = collect_files(base_dir)
    incorrect_map = collect_files(incorrect_base)

    keys_correct = set(correct_map.keys())
    keys_incorrect = set(incorrect_map.keys())
    common_keys = sorted(keys_correct & keys_incorrect)

    print(f"[info] found {len(keys_correct)} correct files, {len(keys_incorrect)} incorrect files")
    print(f"[info] comparing {len(common_keys)} matching pairs")

    records = []
    for key in common_keys:
        f_correct = correct_map[key]
        f_incorrect = incorrect_map[key]

        try:
            t_correct = torch.load(f_correct, map_location="cpu")
            t_incorrect = torch.load(f_incorrect, map_location="cpu")
        except Exception as e:
            print(f"[warn] failed to load pair {key}: {e}")
            continue

        tc = to_numpy(t_correct)
        ti = to_numpy(t_incorrect)
        if tc is None or ti is None:
            print(f"[warn] unexpected types for {key}; skipping")
            continue
        if tc.shape != ti.shape:
            print(f"[warn] shape mismatch {tc.shape} vs {ti.shape} for {key}; skipping")
            continue

        cos = cosine_similarity(tc, ti)
        org, model, pair, language, direction_label = key
        records.append({
            "org": org,
            "model": model,
            "pair": pair,
            "language": language,
            "direction_label": direction_label,
            "cosine_similarity": cos,
            "path_correct": str(f_correct),
            "path_incorrect": str(f_incorrect),
        })

    df = pd.DataFrame.from_records(records)
    csv_path = output_dir / "cosine_similarities.csv"
    df.to_csv(csv_path, index=False)
    print(f"[info] wrote CSV: {csv_path} ({len(df)} rows)")

    vals = df["cosine_similarity"].to_numpy() if not df.empty else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        print("[info] no finite cosine values; skipping plot")
        return

    vals = np.clip(vals, 0.0, 1.0)  # display range 0..1
    edges = np.linspace(0.0, 1.0, args.bins + 1)

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=edges)
    plt.title("Cosine similarities (incorrect vs correct) across matching heatmaps")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.tight_layout()
    png_path = output_dir / "cosine_histogram.png"
    plt.savefig(png_path, dpi=200)
    print(f"[info] wrote histogram: {png_path}")


if __name__ == "__main__":
    main()
