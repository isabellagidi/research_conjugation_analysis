#!/usr/bin/env python3
# rsa_english_bars.py
import os
import re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CHOOSE HERE ----------------
MODEL_NAME = "Qwen/Qwen2-1.5B"   # <-- set the HF model id you want
PAIR       = "1sg-2sg"                # <-- one of: 1pl-2pl, 1sg-1pl, 1sg-2sg, 1sg-3sg
# ---------------------------------------------

# Folder layout (matches your results)
RESULTS_BASE = Path("results")                  # run from mega_automation/
OUTPUT_BASE  = Path("RSA_language_bars")        # where the bar charts go
FOCUS_LANG   = "english"                        # compare English vs others

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def model_safe_name(model_name: str) -> str:
    return re.sub(r"[/: ]", "__", model_name)

def pair_to_file_tag(pair: str) -> str:
    # "1sg-2sg" -> "1sgto2sg"
    return pair.replace("-", "to")

def list_languages_with_pt(results_root: Path, model_hf: str, pair: str) -> list[str]:
    """Find language folders that actually contain the expected .pt file."""
    if "/" not in model_hf:
        raise SystemExit(f"MODEL_NAME must be 'org/name', got '{model_hf}'")
    family, name = model_hf.split("/", 1)
    pair_dir = results_root / family / name / pair
    if not pair_dir.exists():
        return []
    tag = pair_to_file_tag(pair)
    langs = []
    for d in sorted(p.name for p in pair_dir.iterdir() if p.is_dir()):
        expected = pair_dir / d / f"attn_head_out_all_pos_patch_results_{tag}_{d}.pt"
        if expected.exists():
            langs.append(d)
    return langs

def load_language_vectors(results_root: Path, model_hf: str, pair: str, langs: list[str]) -> torch.Tensor:
    """Load flattened per-language tensors → matrix (L, D)."""
    family, name = model_hf.split("/", 1)
    pair_dir = results_root / family / name / pair
    tag = pair_to_file_tag(pair)
    vecs = []
    for lang in langs:
        pt_path = pair_dir / lang / f"attn_head_out_all_pos_patch_results_{tag}_{lang}.pt"
        t = torch.load(pt_path, map_location="cpu")
        vecs.append(t.flatten().float())
    return torch.stack(vecs, dim=0)  # (L, D)

def cosine_rsa_from_vectors(V: torch.Tensor) -> torch.Tensor:
    """Compute cosine RSA (L, L) from (L, D) language vectors."""
    Vn = V / (V.norm(dim=1, keepdim=True) + 1e-8)
    return Vn @ Vn.T

def plot_english_bars(langs: list[str], english_row: np.ndarray, model_hf: str, pair: str, out_png: Path):
    """Bar chart of English vs each other language (sorted)."""
    try:
        e_idx = langs.index(FOCUS_LANG)
    except ValueError:
        raise RuntimeError(f"'{FOCUS_LANG}' not in languages: {langs}")

    vals = [(lang, float(english_row[i])) for i, lang in enumerate(langs) if i != e_idx]
    vals.sort(key=lambda kv: kv[1], reverse=True)

    labels = [v[0] for v in vals]
    scores = [v[1] for v in vals]

    plt.figure(figsize=(max(10, 0.7 * len(labels)), 6))
    sns.barplot(x=labels, y=scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cosine similarity (RSA row)")
    plt.xlabel("Language")
    plt.title(f"English vs Others — {pair}\n{model_hf}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"📊 saved {out_png}")
    plt.close()

def main():
    model_hf = MODEL_NAME
    pair     = PAIR

    langs = list_languages_with_pt(RESULTS_BASE, model_hf, pair)
    if len(langs) < 2:
        raise SystemExit(f"Not enough languages for {model_hf} / {pair}. Found: {langs}")
    if FOCUS_LANG not in langs:
        raise SystemExit(f"'{FOCUS_LANG}' not available for {model_hf} / {pair}. "
                         f"Languages present: {langs}")

    print(f"✅ {model_hf} / {pair}: languages = {langs}")

    V   = load_language_vectors(RESULTS_BASE, model_hf, pair, langs)
    RSA = cosine_rsa_from_vectors(V)

    e_idx = langs.index(FOCUS_LANG)
    english_row = RSA[e_idx].cpu().numpy()

    out_dir = OUTPUT_BASE / model_safe_name(model_hf) / pair
    ensure_dir(out_dir)
    out_png = out_dir / f"english_similarity_bars_{model_safe_name(model_hf)}_{pair}.png"

    plot_english_bars(langs, english_row, model_hf, pair, out_png)

    # Also print values to stdout
    print("English vs others (cosine RSA):")
    for i, lang in enumerate(langs):
        if i == e_idx:
            continue
        print(f"  {FOCUS_LANG:8s} — {lang:>12s} : {english_row[i]:.4f}")

if __name__ == "__main__":
    main()
