# RSA.py (drop-in for your Mistral paths)
import os
import glob
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ---- roots (relative to this file) ----
BASEDIR      = os.path.dirname(os.path.abspath(__file__))
PATCHES_ROOT = os.path.join(BASEDIR, "results", "patches")           # where patch .pt files live
RSA_ROOT     = os.path.join(BASEDIR, "RSA_plots_reduced_double")     # where RSA outputs go
os.makedirs(RSA_ROOT, exist_ok=True)

# ---- model naming exactly as your jobs wrote it ----
MODEL_NAME  = "mistralai/Mistral-7B-v0.1"
# your filenames used a SINGLE underscore; be robust to both:
MODEL_SAFES = [
    "mistralai_Mistral-7B-v0.1",   # single underscore (your files)
    "mistralai__Mistral-7B-v0.1",  # double underscore (fallback)
]

# ---- languages to include ----
lang_codes = ["deu", "eng", "fra", "por", "spa", "swe", "ita"]
lang_labels = {
    "cat": "catalan", "ces": "czech", "deu": "german", "eng": "english",
    "fin": "finnish", "fra": "french", "hbs": "serbo-croatian", "hun": "hungarian",
    "ita": "italian", "mon": "mongolian", "pol": "polish", "por": "portuguese",
    "rus": "russian", "spa": "spanish", "swe": "swedish"
}

# ---- pairs & helpers ----
PAIRS_HYPHEN = ["1sg-2sg"]  # start with what you actually ran; add more later if needed

short2long = {
    "1sg": "first_singular",  "2sg": "second_singular", "3sg": "third_singular",
    "1pl": "first_plural",    "2pl": "second_plural",   "3pl": "third_plural",
}

def pair_to_directions(pair_hyphen: str):
    a, b = [p.strip() for p in pair_hyphen.split("-")]
    return [f"{a}to{b}", f"{b}to{a}"]

def pair_to_pair_safe(pair_hyphen: str):
    a, b = [p.strip() for p in pair_hyphen.split("-")]
    return f"{short2long[a]}_to_{short2long[b]}"

# ---- io + plotting ----
def plot_rsa(matrix: torch.Tensor, labels, title, filename, cmap="coolwarm", center=None):
    plt.figure(figsize=(1.2 * len(labels), 1 + 0.8 * len(labels)))
    sns.heatmap(matrix.numpy(), xticklabels=labels, yticklabels=labels,
                cmap=cmap, center=center, annot=True, fmt=".2f")
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
    langs = list(heatmaps.keys())
    n = len(langs)
    if n < 2:
        return langs, None, None, None
    cos = torch.zeros((n, n))
    for i, li in enumerate(langs):
        hi = heatmaps[li].flatten().float()
        for j, lj in enumerate(langs):
            hj = heatmaps[lj].flatten().float()
            cos[i, j] = torch.nn.functional.cosine_similarity(hi, hj, dim=0)
    return langs, None, None, cos

# robust finder: look for directory by exact name first; if missing, glob
def find_lang_dir(pair_safe: str, iso3: str) -> str | None:
    for ms in MODEL_SAFES:
        candidate = os.path.join(PATCHES_ROOT, f"{ms}-{pair_safe}-{iso3}")
        if os.path.isdir(candidate):
            return candidate
    # fallback: glob anything that matches *-pair-iso3
    pattern = os.path.join(PATCHES_ROOT, f"*-{pair_safe}-{iso3}")
    hits = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    return hits[0] if hits else None

def main():
    model_rsa_dir = os.path.join(RSA_ROOT, MODEL_SAFES[0])
    os.makedirs(model_rsa_dir, exist_ok=True)
    print(f"\n================= MODEL: {MODEL_NAME} =================")
    print(f"PATCHES_ROOT: {PATCHES_ROOT}")

    for pair_hyphen in PAIRS_HYPHEN:
        pair_safe = pair_to_pair_safe(pair_hyphen)
        directions = pair_to_directions(pair_hyphen)

        for direction_tag in directions:
            heatmaps: dict[str, torch.Tensor] = {}
            for code in lang_codes:
                lang_full = lang_labels[code]

                lang_dir = find_lang_dir(pair_safe, code)
                if not lang_dir:
                    print(f"⚠️ Missing dir for {code} ({lang_full}): {pair_safe}")
                    continue

                fname = f"attn_head_out_all_pos_patch_results_{direction_tag}_{lang_full}.pt"
                fpath = os.path.join(lang_dir, fname)
                if os.path.exists(fpath):
                    tens = load_patch_tensor(fpath)
                    if tens is not None:
                        heatmaps[lang_full] = tens
                        print(f"✅ Loaded {lang_full} ({direction_tag}) from {fpath}")
                else:
                    print(f"⚠️ Missing file: {fpath}")

            labels, _, _, cos = compute_rsa(heatmaps)
            if cos is None:
                print(f"↪️  Not enough languages for RSA: pair={pair_hyphen}, dir={direction_tag}")
                continue

            outdir = os.path.join(model_rsa_dir, pair_hyphen)
            os.makedirs(outdir, exist_ok=True)

            bundle = {
                "languages": labels,
                "cos": cos,
                "model_name": MODEL_NAME,
                "pair": pair_hyphen,
                "direction": direction_tag,
            }
            pt_out = os.path.join(outdir, f"rsa_matrices_{direction_tag}.pt")
            torch.save(bundle, pt_out)
            print(f"💾 Saved {pt_out}")

            plot_rsa(
                cos, labels,
                f"{MODEL_NAME} • {pair_hyphen} • {direction_tag} • RSA (Cosine)",
                os.path.join(outdir, f"rsa_cosine_{direction_tag}.png"),
                cmap="coolwarm", center=0.0
            )

if __name__ == "__main__":
    main()
