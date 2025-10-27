# compare_models_second_order_rsa.py
import os
import re
from pathlib import Path
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- config --------------------
RESULTS_BASE = Path("results")                           # run from mega_automation/ (where results live)
OUTPUT_BASE  = Path("RSA_model_comparisons_reduced")     # reduced outputs folder

# Models to compare (HF names = path split into <family>/<name>)
MODELS = [
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloomz-1b1",
    "bigscience/bloomz-1b7",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2-1.5B",
]

# Conjugation pairs (directory names); file names use "to" (1sgto2sg)
PAIRS = ["1pl-2pl", "1sg-1pl", "1sg-2sg", "1sg-3sg"]

# Toggle: if True, force a global language intersection across *all* models (per pair).
# Default False = use pairwise intersections (recommended).
STRICT_GLOBAL_INTERSECTION = False

# ===== Reduced language set =====
# Codes: cat, eng, rus, fra, por, spa, swe, ita
REDUCED_LANG_CODES = ["cat", "eng", "rus", "fra", "por", "spa", "swe", "ita"]
LANG_CODE_TO_FULL = {
    "cat": "catalan",
    "eng": "english",
    "rus": "russian",
    "fra": "french",
    "por": "portuguese",
    "spa": "spanish",
    "swe": "swedish",
    "ita": "italian",
}
REDUCED_LANG_FULL = {LANG_CODE_TO_FULL[c] for c in REDUCED_LANG_CODES}
# ------------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def model_safe_name(model_name: str) -> str:
    return re.sub(r"[/: ]", "__", model_name)

def pretty_axis_label(model_name: str) -> str:
    # show only the part after the first slash, if present
    return model_name.split("/", 1)[-1] if "/" in model_name else model_name

def pair_to_file_tag(pair: str) -> str:
    # "1sg-2sg" -> "1sgto2sg"
    return pair.replace("-", "to")

def list_available_languages(results_root: Path, model_hf: str, pair: str):
    """
    Return list of language folder names (full names) that:
      1) are in the reduced set, and
      2) actually contain the expected .pt file.
    """
    family, name = model_hf.split("/", 1)
    pair_dir = results_root / family / name / pair
    if not pair_dir.exists():
        return []

    tag = pair_to_file_tag(pair)
    langs = []
    # scan subfolders; keep only those in REDUCED_LANG_FULL
    for d in sorted(p.name for p in pair_dir.iterdir() if p.is_dir()):
        if d not in REDUCED_LANG_FULL:
            continue
        expected = pair_dir / d / f"attn_head_out_all_pos_patch_results_{tag}_{d}.pt"
        if expected.exists():
            langs.append(d)
    return langs

def load_language_vectors(results_root: Path, model_hf: str, pair: str, langs: list[str]) -> torch.Tensor:
    """
    For a (model, pair), load flattened patch tensors per language into a matrix
    of shape (len(langs), D).
    """
    family, name = model_hf.split("/", 1)
    pair_dir = results_root / family / name / pair
    tag = pair_to_file_tag(pair)

    vecs = []
    for lang in langs:
        pt_path = pair_dir / lang / f"attn_head_out_all_pos_patch_results_{tag}_{lang}.pt"
        t = torch.load(pt_path, map_location="cpu")
        vecs.append(t.flatten().float())
    V = torch.stack(vecs, dim=0)  # (L, D)
    return V

def cosine_rsa_from_vectors(V: torch.Tensor) -> torch.Tensor:
    """
    Given (L, D) language vectors, return cosine RSA (L, L).
    """
    Vn = V / (V.norm(dim=1, keepdim=True) + 1e-8)
    return Vn @ Vn.T

def vec_upper_tri(mat: torch.Tensor) -> torch.Tensor:
    """Vectorize upper triangle (exclude diagonal)."""
    n = mat.size(0)
    iu = torch.triu_indices(n, n, offset=1)
    return mat[iu[0], iu[1]]

def second_order_similarity(model_rsa_a, langs_a, model_rsa_b, langs_b, use_global=None, global_langs=None):
    """
    Compute Pearson correlation between upper-tri vectors of two RSA matrices,
    aligned on either:
      - pairwise intersection (default), or
      - a provided 'global_langs' list if use_global=True.
    """
    if use_global and global_langs is not None:
        common = [l for l in global_langs if (l in langs_a) and (l in langs_b)]
    else:
        common = sorted(set(langs_a) & set(langs_b))

    if len(common) < 2:
        return np.nan

    idx_a = [langs_a.index(l) for l in common]
    idx_b = [langs_b.index(l) for l in common]

    sub_a = model_rsa_a[idx_a][:, idx_a]
    sub_b = model_rsa_b[idx_b][:, idx_b]

    va = vec_upper_tri(sub_a).numpy()
    vb = vec_upper_tri(sub_b).numpy()

    if np.std(va) < 1e-8 or np.std(vb) < 1e-8:
        return np.nan

    corr = np.corrcoef(va, vb)[0, 1]
    return float(corr)

def plot_heatmap(arr: np.ndarray, labels, title, out_png):
    n = len(labels)
    plt.figure(figsize=(2.0 + 1.0*n, 2.0 + 1.0*n))
    ax = sns.heatmap(arr, xticklabels=labels, yticklabels=labels,
                     cmap="coolwarm", center=0.0, annot=True, fmt=".2f")
    plt.title(title)
    ax.tick_params(axis='both', labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"📈 saved {out_png}")
    plt.close()

def main():
    ensure_dir(OUTPUT_BASE)

    for pair in PAIRS:
        print(f"\n=== Building second-order RSA (reduced langs) for pair: {pair} ===")
        model_langs = {}
        model_rsas  = {}
        kept_models = []

        # 1) discover reduced-set languages per model, and build each model's RSA
        for m in MODELS:
            langs = list_available_languages(RESULTS_BASE, m, pair)
            if len(langs) < 2:
                print(f"⚠️  {m} has <2 reduced languages for {pair}; skipping.")
                continue
            V = load_language_vectors(RESULTS_BASE, m, pair, langs)
            rsa = cosine_rsa_from_vectors(V)  # (L, L)
            model_langs[m] = langs
            model_rsas[m]  = rsa
            kept_models.append(m)
            print(f"✅ {m}: {len(langs)} reduced languages for {pair}: {langs}")

        if len(kept_models) < 2:
            print(f"❌ Not enough models with reduced-language data for {pair}; skipping.")
            continue

        # 2) optional: global intersection (per pair)
        global_langs = None
        if STRICT_GLOBAL_INTERSECTION:
            sets = [set(model_langs[m]) for m in kept_models]
            inter = set.intersection(*sets) if sets else set()
            global_langs = sorted(inter)
            print(f"Global intersection for {pair}: {len(global_langs)} languages — {global_langs}")

        # 3) model×model similarities (pairwise language intersections by default)
        M = len(kept_models)
        sim = np.full((M, M), np.nan, dtype=np.float32)
        for i, ma in enumerate(kept_models):
            sim[i, i] = 1.0
            for j in range(i+1, M):
                mb = kept_models[j]
                val = second_order_similarity(
                    model_rsa_a=model_rsas[ma], langs_a=model_langs[ma],
                    model_rsa_b=model_rsas[mb], langs_b=model_langs[mb],
                    use_global=STRICT_GLOBAL_INTERSECTION, global_langs=global_langs
                )
                sim[i, j] = sim[j, i] = val

        # 4) save
        out_dir = OUTPUT_BASE / pair
        ensure_dir(out_dir)
        labels = [pretty_axis_label(m) for m in kept_models]

        torch.save(torch.tensor(sim), out_dir / f"model_similarity_{pair}.pt")
        np.save(out_dir / f"model_similarity_{pair}.npy", sim)
        plot_heatmap(sim, labels,
                     f"Second-order RSA (reduced langs) — {pair}",
                     out_dir / f"model_similarity_{pair}.png")

if __name__ == "__main__":
    main()
