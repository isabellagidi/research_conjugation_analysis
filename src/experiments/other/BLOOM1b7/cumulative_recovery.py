import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# -------- Settings you may tweak --------
BASE_DIR = Path("automation_results")   # folder with your .pt files
PLOTS_DIR = Path("plots")
OVERLAY_PATH = Path("plots2") / "cumulative_overlay.png"
# ---------------------------------------

def load_and_flatten_sorted_patch(pt_file: Path):
    tensor = torch.load(pt_file, map_location="cpu")
    flat = tensor.flatten().cpu().numpy()
    sorted_flat = np.sort(flat)[::-1]
    return sorted_flat

def plot_cumulative_by_language(lang_name, sorted_flat, save_dir=PLOTS_DIR):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cum_recovery = np.cumsum(sorted_flat)
    total_recovery = cum_recovery[-1] if len(cum_recovery) > 0 else 0.0
    normed_cum_recovery = cum_recovery / (total_recovery + 1e-12)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(normed_cum_recovery) + 1), normed_cum_recovery, marker='o')
    plt.title(f"Cumulative Normalized Recovery — {lang_name}")
    plt.xlabel("Number of Heads (sorted by positive importance)")
    plt.ylabel("Cumulative Normalized Recovery")
    plt.grid(True)
    plt.tight_layout()
    out_path = save_dir / f"cumulative_head_recovery_{lang_name.lower()}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")

def plot_overlay_all(languages_sorted_dict, save_path=OVERLAY_PATH):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for lang, sorted_flat in languages_sorted_dict.items():
        cum_recovery = np.cumsum(sorted_flat)
        total_recovery = cum_recovery[-1] if len(cum_recovery) > 0 else 0.0
        normed = cum_recovery / (total_recovery + 1e-12)
        plt.plot(range(1, len(normed) + 1), normed, label=lang)

    plt.title("Cumulative Normalized Recovery by Top Attention Heads")
    plt.xlabel("Number of Heads (sorted by positive importance)")
    plt.ylabel("Cumulative Normalized Recovery")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved overlay plot: {save_path}")

# Map from filename substring → Pretty language name
language_keywords = {
    "spanish": "Spanish",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "catalan": "Catalan",
    "english": "English",
    "russian": "Russian",
    "french": "French",
    "finnish": "Finnish",
    "hungarian": "Hungarian",
}

def find_language_files(base_dir: Path):
    """
    Walk automation_results/ and collect the most recent .pt file per language keyword.
    """
    lang_to_path = {}
    lang_to_mtime = {}

    if not base_dir.exists():
        print(f"⚠️ Base directory not found: {base_dir.resolve()}")
        return lang_to_path

    for path in base_dir.rglob("*.pt"):
        lower = path.name.lower()
        for key, lang in language_keywords.items():
            if key in lower:
                mtime = path.stat().st_mtime
                if (lang not in lang_to_mtime) or (mtime > lang_to_mtime[lang]):
                    lang_to_mtime[lang] = mtime
                    lang_to_path[lang] = path
                break

    return lang_to_path

if __name__ == "__main__":
    LANGUAGES = find_language_files(BASE_DIR)
    print("✅ Loaded languages:", {k: str(v) for k, v in LANGUAGES.items()})

    sorted_data_per_lang = {}
    for lang, pt_path in LANGUAGES.items():
        if pt_path.exists():
            sorted_flat = load_and_flatten_sorted_patch(pt_path)
            sorted_data_per_lang[lang] = sorted_flat
            plot_cumulative_by_language(lang, sorted_flat)
        else:
            print(f"⚠️ Missing file for {lang}: {pt_path}")

    if sorted_data_per_lang:
        plot_overlay_all(sorted_data_per_lang)
    else:
        print("⚠️ No data found to plot.")
