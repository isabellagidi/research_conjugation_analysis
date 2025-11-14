#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless nodes
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Plot cosine similarity histogram on [-1, 1].")
    p.add_argument("--csv", required=True, help="Path to cosine_similarities.csv")
    p.add_argument("--col", default="cosine_similarity", help="CSV column name with cosine values")
    p.add_argument("--bins", type=int, default=20, help="Number of bins across [-1, 1]")
    p.add_argument("--out", default="cosine_histogram_-1_to_1.png", help="Output PNG")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.col not in df.columns:
        raise SystemExit(f"Column '{args.col}' not found in {args.csv}. Available: {list(df.columns)}")

    vals = df[args.col].to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise SystemExit("No finite cosine values to plot.")

    edges = np.linspace(-1.0, 1.0, args.bins + 1)

    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=edges)  # no clipping; full [-1,1]
    plt.title("Cosine similarities (full range [-1, 1])")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.xlim(-1.0, 1.0)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[info] wrote {args.out}")

if __name__ == "__main__":
    main()
