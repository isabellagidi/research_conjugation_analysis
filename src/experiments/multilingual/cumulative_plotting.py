import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Languages and corresponding file paths
LANGUAGES = {
    "Spanish": "automation_results/spanish/attn_head_out_all_pos_patch_results_second2first_spanish.pt",
    "Italian": "automation_results/italian/attn_head_out_all_pos_patch_results_second2first_italian.pt",
    "Portuguese": "automation_results/portuguese/attn_head_out_all_pos_patch_results_second2first_portuguese.pt",
    "Catalan": "automation_results/catalan/attn_head_out_all_pos_patch_results_second2first_catalan.pt",
    "English": "automation_results/english/attn_head_out_all_pos_patch_results_second2first_english.pt",
    "Russian": "automation_results/russian/attn_head_out_all_pos_patch_results_second2first_russian.pt",
    "French": "automation_results/french/attn_head_out_all_pos_patch_results_second2first_french.pt",
    "Finnish": "automation_results/finnish/attn_head_out_all_pos_patch_results_second2first_finnish.pt",
    "Hungarian": "automation_results/hungarian/attn_head_out_all_pos_patch_results_second2first_hungarian.pt",
    
}

def load_and_flatten_sorted_patch(pt_file):
    """
    Loads the patching tensor and returns the sorted 1D array of recovery scores (descending).
    """
    tensor = torch.load(pt_file)  # shape: [n_layers, n_heads]
    flat = tensor.flatten().cpu().numpy()
    sorted_flat = np.sort(flat)[::-1]  # descending order
    return sorted_flat

def plot_cumulative_by_language(lang_name, sorted_flat, save_dir="plots"):
    """
    Plot cumulative recovery vs number of heads for one language.
    """
    os.makedirs(save_dir, exist_ok=True)
    cum_recovery = np.cumsum(sorted_flat)
    total_recovery = cum_recovery[-1]
    normed_cum_recovery = cum_recovery / (total_recovery + 1e-12)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(normed_cum_recovery) + 1), normed_cum_recovery, marker='o')
    plt.title(f"Cumulative Normalized Recovery — {lang_name}")
    plt.xlabel("Number of Heads (sorted by importance)")
    plt.ylabel("Cumulative Normalized Recovery")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cumulative_head_recovery_{lang_name.lower()}.png"))
    plt.close()
    print(f"✅ Saved: {save_dir}/cumulative_head_recovery_{lang_name.lower()}.png")

def plot_overlay_all(languages_sorted_dict, save_path="plots/cumulative_overlay.png"):
    """
    Overlaid cumulative recovery plot for all languages.
    """
    plt.figure(figsize=(10, 6))
    for lang, sorted_flat in languages_sorted_dict.items():
        cum_recovery = np.cumsum(sorted_flat)
        total_recovery = cum_recovery[-1]
        normed = cum_recovery / (total_recovery + 1e-12)
        plt.plot(range(1, len(normed) + 1), normed, label=lang)

    plt.title("Cumulative Normalized Recovery by Top Attention Heads")
    plt.xlabel("Number of Heads (sorted by importance)")
    plt.ylabel("Cumulative Normalized Recovery")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved overlay plot: {save_path}")

# === Run for all languages ===
sorted_data_per_lang = {}
for lang, pt_path in LANGUAGES.items():
    if os.path.exists(pt_path):
        sorted_flat = load_and_flatten_sorted_patch(pt_path)
        sorted_data_per_lang[lang] = sorted_flat
        plot_cumulative_by_language(lang, sorted_flat)
    else:
        print(f"⚠️ Missing file for {lang}: {pt_path}")

# Overlay plot
plot_overlay_all(sorted_data_per_lang)
