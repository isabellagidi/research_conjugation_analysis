import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Define language names and file paths ===
langs = ["Spanish", "Italian", "English", "German"]
lang_to_path = {
    "Spanish": "src/experiments/spanish/morphy_net/activation_patching/attn_head_out_all_pos_patch_results_spanish.pt",
    "Italian": "src/experiments/italian/morphy_net/activation_patching/attn_head_out_all_pos_patch_results_italian.pt",
    "English": "src/experiments/english/morphy_net/activation_patching/attn_head_out_all_pos_patch_results_english.pt",
    "German":  "src/experiments/german/morphy_net/activation_patching/attn_head_out_all_pos_patch_results_german.pt",
}

# === Load tensors from disk ===
heatmaps = {}
for lang, path in lang_to_path.items():
    if os.path.exists(path):
        heatmaps[lang] = torch.load(path)
        print(f"✅ Loaded {lang} from {path}")
    else:
        raise FileNotFoundError(f"❌ Could not find file for {lang}: {path}")

# === Initialize similarity matrices ===
n = len(langs)
mse_matrix = torch.zeros((n, n))
corr_matrix = torch.zeros((n, n))
cos_matrix = torch.zeros((n, n))

# === Compute metrics for all language pairs ===
for i, lang_i in enumerate(langs):
    for j, lang_j in enumerate(langs):
        hi = heatmaps[lang_i].flatten()
        hj = heatmaps[lang_j].flatten()

        mse_matrix[i, j] = torch.nn.functional.mse_loss(hi, hj)
        corr_matrix[i, j] = torch.corrcoef(torch.stack([hi, hj]))[0, 1]
        cos_matrix[i, j] = torch.nn.functional.cosine_similarity(hi, hj, dim=0)

# === Plotting helper ===
def plot_rsa(matrix, labels, title, filename, cmap="coolwarm", center=None):
    plt.figure(figsize=(8, 6))
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
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📈 Saved {filename}")
    plt.close()

# === Save heatmaps ===
plot_rsa(mse_matrix, langs, "RSA (Mean Squared Error)", "rsa_mse.png", cmap="viridis")
plot_rsa(corr_matrix, langs, "RSA (Pearson Correlation)", "rsa_corr.png", cmap="coolwarm", center=0.0)
plot_rsa(cos_matrix, langs, "RSA (Cosine Similarity)", "rsa_cosine.png", cmap="coolwarm", center=0.0)
