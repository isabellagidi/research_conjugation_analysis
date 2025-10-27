# RSA.py
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# --- base dirs (relative to this file) ---
BASEDIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(BASEDIR, "results")             # where patch .pt live
RSA_ROOT     = os.path.join(BASEDIR, "RSA_plots_reduced")   # <<< save reduced outputs here
os.makedirs(RSA_ROOT, exist_ok=True)

# === Languages (reduced set) ===
# Only evaluate these 8 languages:
# cat, eng, rus, fra, por, spa, swe, ita
lang_codes = ["cat", "eng", "rus", "fra", "por", "spa", "swe", "ita"]

lang_labels = {
    "cat": "catalan",
    "ces": "czech",
    "deu": "german",
    "eng": "english",
    "fin": "finnish",
    "fra": "french",
    "hbs": "serbo-croatian",
    "hun": "hungarian",
    "ita": "italian",
    "mon": "mongolian",
    "pol": "polish",
    "por": "portuguese",
    "rus": "russian",
    "spa": "spanish",
    "swe": "swedish"
}

# === Friendly model names -> folders (family, name) ===
MODEL_MAP = {
    "Bloom-560m":   ("bigscience", "bloom-560m"),
    "Bloom-1b1":    ("bigscience", "bloom-1b1"),
    "Bloom-1b7":    ("bigscience", "bloom-1b7"),
    "Bloomz-1b1":   ("bigscience", "bloomz-1b1"),
    "Qwen2-0.5B":   ("Qwen",       "Qwen2-0.5B"),
    "Qwen2-1.5B":   ("Qwen",       "Qwen2-1.5B"),
}

# === Which models to run (unchanged; runs all listed) ===
MODELS = [
    "Bloom-1b1",
    "Bloom-1b7",
    "Bloomz-1b1",
    "Bloom-560m",
    "Qwen2-0.5B",
    "Qwen2-1.5B",
]

# === Conjugation pairs (folder uses hyphen, filename uses "to") ===
PAIRS_HYPHEN = ["1pl-2pl", "1sg-1pl", "1sg-2sg", "1sg-3sg"]

def pair_to_filename(pair_hyphen: str) -> str:
    """Convert '1sg-2sg' -> '1sgto2sg' (also strips stray spaces)."""
    a, b = [p.strip() for p in pair_hyphen.split("-")]
    return f"{a}to{b}"

# --- plotting helper ---
def plot_rsa(matrix: torch.Tensor, labels, title, filename, cmap="coolwarm", center=None):
    plt.figure(figsize=(1.2 * len(labels), 1 + 0.8 * len(labels)))
    sns.heatmap(
        matrix.numpy(),
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        center=center,
        annot=True,
        fmt=".2f"
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"📈 Saved {filename}")
    plt.close()

def load_patch_tensor(path: str):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"⚠️  Failed to load {path}: {e}")
        return None

def compute_rsa(heatmaps: dict[str, torch.Tensor]):
    """Return (labels, mse, corr, cos).  MSE/Corr code is commented out; we only compute cosine."""
    langs = list(heatmaps.keys())
    n = len(langs)
    if n < 2:
        return langs, None, None, None

    # mse = torch.zeros((n, n))     # ← commented out
    # corr = torch.zeros((n, n))    # ← commented out
    cos  = torch.zeros((n, n))

    for i, li in enumerate(langs):
        hi = heatmaps[li].flatten().float()
        for j, lj in enumerate(langs):
            hj = heatmaps[lj].flatten().float()
            # --- MSE (commented) ---
            # mse[i, j] = torch.nn.functional.mse_loss(hi, hj)
            # --- Pearson corr (commented) ---
            # try:
            #     cc = torch.corrcoef(torch.stack([hi, hj]))[0, 1]
            # except Exception:
            #     cc = torch.tensor(float("nan"))
            # corr[i, j] = cc
            # --- Cosine (keep) ---
            cos[i, j] = torch.nn.functional.cosine_similarity(hi, hj, dim=0)

    return langs, None, None, cos

def main():
    for friendly_model in MODELS:
        if friendly_model not in MODEL_MAP:
            print(f"❌ Unknown model key: {friendly_model} (skipping)")
            continue

        family, model_name = MODEL_MAP[friendly_model]
        model_dir = os.path.join(RESULTS_ROOT, family, model_name)

        # Safe folder name for RSA outputs (no slashes)
        model_sub = f"{family}__{model_name}"
        model_rsa_dir = os.path.join(RSA_ROOT, model_sub)
        os.makedirs(model_rsa_dir, exist_ok=True)

        print(f"\n================= MODEL: {friendly_model} ({family}/{model_name}) =================")
        print(f"Looking under: {model_dir}")

        for pair_hyphen in PAIRS_HYPHEN:
            pair_fname = pair_to_filename(pair_hyphen)   # e.g., "1sgto2sg"
            pair_dir   = os.path.join(model_dir, pair_hyphen)

            # load per-language patch heatmaps (restricted set)
            heatmaps: dict[str, torch.Tensor] = {}
            for code in lang_codes:
                lang_full = lang_labels[code]
                fdir = os.path.join(pair_dir, lang_full)
                fname = f"attn_head_out_all_pos_patch_results_{pair_fname}_{lang_full}.pt"
                fpath = os.path.join(fdir, fname)

                if os.path.exists(fpath):
                    tens = load_patch_tensor(fpath)
                    if tens is not None:
                        heatmaps[lang_full] = tens
                        print(f"✅ Loaded {lang_full} from {fpath}")
                else:
                    print(f"⚠️ Missing: {fpath}")

            # compute (only cosine is active)
            labels, mse, corr, cos = compute_rsa(heatmaps)
            if cos is None:
                print(f"↪️  Not enough languages for RSA: model={friendly_model}, pair={pair_hyphen}")
                continue

            # create pair subdir under this model’s RSA folder
            outdir = os.path.join(model_rsa_dir, pair_hyphen)
            os.makedirs(outdir, exist_ok=True)

            # save PT bundle (MSE/Corr lines commented out)
            bundle = {
                "languages": labels,
                # "mse": mse,
                # "corr": corr,
                "cos": cos,
                "model_family": family,
                "model_name": model_name,
                "pair": pair_hyphen,
            }
            pt_out = os.path.join(outdir, f"rsa_matrices_{pair_fname}.pt")
            torch.save(bundle, pt_out)
            print(f"💾 Saved {pt_out}")

            # save plots (only cosine)
            plot_rsa(
                cos, labels,
                f"{friendly_model} • {pair_hyphen} • RSA (Cosine)",
                os.path.join(outdir, f"rsa_cosine_{pair_fname}.png"),
                cmap="coolwarm", center=0.0
            )

if __name__ == "__main__":
    main()
