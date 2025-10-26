import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Languages and corresponding file paths
LANGUAGES = {
    "Spanish":    "automation_results/spanish/attn_head_out_all_pos_patch_results_second2first_spanish.pt",
    "Italian":    "automation_results/italian/attn_head_out_all_pos_patch_results_second2first_italian.pt",
    "Portuguese": "automation_results/portuguese/attn_head_out_all_pos_patch_results_second2first_portuguese.pt",
    "Catalan":    "automation_results/catalan/attn_head_out_all_pos_patch_results_second2first_catalan.pt",
    "English":    "automation_results/english/attn_head_out_all_pos_patch_results_second2first_english.pt",
    "Russian":    "automation_results/russian/attn_head_out_all_pos_patch_results_second2first_russian.pt",
    "French":     "automation_results/french/attn_head_out_all_pos_patch_results_second2first_french.pt",
    "Finnish":    "automation_results/finnish/attn_head_out_all_pos_patch_results_second2first_finnish.pt",
    "Hungarian":  "automation_results/hungarian/attn_head_out_all_pos_patch_results_second2first_hungarian.pt",
}

# ------------ helpers -------------------------------------------------

def load_flatten_sorted_desc(pt_file):
    """
    Load patch tensor (any shape), flatten to 1D, sort by value (descending).
    """
    t = torch.load(pt_file)
    flat = t.detach().float().flatten().cpu().numpy()
    return np.sort(flat)[::-1]  # big positives first, big negatives last

def split_pos_neg(sorted_desc_vals):
    """
    Split into positive and negative arrays.
    - positives remain in descending order (largest help -> smallest help)
    - negatives are re-sorted so we accumulate the MOST harmful first
      (ascending by value, i.e., most negative -> least negative)
    """
    pos = sorted_desc_vals[sorted_desc_vals > 0]
    neg = sorted_desc_vals[sorted_desc_vals < 0]
    neg_sorted_most_harmful_first = np.sort(neg)  # ascending: most negative first
    return pos, neg_sorted_most_harmful_first

def cumulative_normalized_positive(pos_vals):
    """
    Cumulative fraction of total positive recovery.
    Ends at 1.0 if there is any positive mass; empty otherwise.
    """
    if pos_vals.size == 0:
        return np.array([])  # nothing positive
    cum = np.cumsum(pos_vals)
    return cum / (cum[-1] + 1e-12)

def cumulative_normalized_negative_harm(neg_vals):
    """
    Treat negatives as harm: use absolute values and accumulate from most harmful.
    Ends at 1.0 if there is any negative mass; empty otherwise.
    """
    if neg_vals.size == 0:
        return np.array([])  # nothing negative
    harm = np.abs(neg_vals)  # convert to magnitude of harm
    cum = np.cumsum(harm)
    return cum / (cum[-1] + 1e-12)

# ------------ plotting ------------------------------------------------

def plot_cum_positive(lang_name, curve, save_dir="plots"):
    """
    Plot cumulative normalized positive recovery (only positives).
    """
    if curve.size == 0:
        print(f"ℹ️  No positive heads for {lang_name}; skipping positive plot.")
        return
    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(1, len(curve) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, curve, marker='o')
    plt.title(f"Cumulative Normalized Positive Recovery — {lang_name}")
    plt.xlabel("Number of Heads (sorted by positive contribution)")
    plt.ylabel("Cumulative Normalized Positive Recovery")
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(save_dir, f"cumulative_positive_{lang_name.lower()}.png")
    plt.savefig(out)
    plt.close()
    print(f"✅ Saved: {out}")

def plot_cum_negative(lang_name, curve, save_dir="plots"):
    """
    Plot cumulative normalized negative harm (only negatives).
    """
    if curve.size == 0:
        print(f"ℹ️  No negative heads for {lang_name}; skipping negative plot.")
        return
    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(1, len(curve) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, curve, marker='o')
    plt.title(f"Cumulative Normalized Negative Harm — {lang_name}")
    plt.xlabel("Number of Heads (sorted by harm magnitude)")
    plt.ylabel("Cumulative Normalized Negative Harm")
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(save_dir, f"cumulative_negative_{lang_name.lower()}.png")
    plt.savefig(out)
    plt.close()
    print(f"✅ Saved: {out}")

def plot_overlay(curves_dict, kind="positive", save_path=None):
    """
    Overlay for multiple languages.
    kind: "positive" or "negative"
    """
    if save_path is None:
        save_path = f"plots/cumulative_overlay_{kind}.png"

    # Keep only languages that actually have a curve
    curves_dict = {k: v for k, v in curves_dict.items() if v.size > 0}
    if not curves_dict:
        print(f"ℹ️  No {kind} curves to overlay; skipping.")
        return

    plt.figure(figsize=(10, 6))
    for lang, curve in curves_dict.items():
        x = np.arange(1, len(curve) + 1)
        plt.plot(x, curve, label=lang)
    title_part = "Positive Recovery" if kind == "positive" else "Negative Harm"
    xlab_part = ("sorted by positive contribution"
                 if kind == "positive" else
                 "sorted by harm magnitude")
    plt.title(f"Cumulative Normalized {title_part} by Top Attention Heads")
    plt.xlabel(f"Number of Heads ({xlab_part})")
    ylabel = ("Cumulative Normalized Positive Recovery"
              if kind == "positive" else
              "Cumulative Normalized Negative Harm")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved overlay plot: {save_path}")

# ------------ main ----------------------------------------------------

pos_curves = {}
neg_curves = {}

for lang, pt_path in LANGUAGES.items():
    if not os.path.exists(pt_path):
        print(f"⚠️ Missing file for {lang}: {pt_path}")
        continue

    sorted_vals = load_flatten_sorted_desc(pt_path)
    pos, neg = split_pos_neg(sorted_vals)

    pos_curve = cumulative_normalized_positive(pos)
    neg_curve = cumulative_normalized_negative_harm(neg)

    pos_curves[lang] = pos_curve
    neg_curves[lang] = neg_curve

    plot_cum_positive(lang, pos_curve)
    plot_cum_negative(lang, neg_curve)

# Overlays
plot_overlay(pos_curves, kind="positive", save_path="plots/cumulative_overlay_positive.png")
plot_overlay(neg_curves, kind="negative", save_path="plots/cumulative_overlay_negative.png")
