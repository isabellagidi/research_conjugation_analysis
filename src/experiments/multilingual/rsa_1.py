import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Define language codes and pretty labels ===
lang_codes = [
    "cat", "ces", "deu", "eng", "fin", "fra", "hbs", "hun",
    "ita", "mon", "pol", "por", "rus", "spa", "swe"
]

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

# === Construct paths (folder uses full language name; file uses code) ===
#attn_head_out_all_pos_patch_results_first2second_catalan
#attn_head_out_all_pos_patch_results_first2second_english
#lang_to_path = {
#    lang_labels[code]: f"semi_automation_results/{lang_labels[code]}/attn_head_out_all_pos_patch_results_first2second_{lang_labels[code]}.pt"
#    for code in lang_codes
#}

lang_to_path = {
    lang_labels[code]: f"automation_results/{lang_labels[code]}/attn_head_out_all_pos_patch_results_second2first_{lang_labels[code]}.pt"
    for code in lang_codes
}

# === Load tensors from disk ===
heatmaps = {}
for lang, path in lang_to_path.items():
    if os.path.exists(path):
        heatmaps[lang] = torch.load(path)
        print(f"✅ Loaded {lang} from {path}")
    else:
        print(f"⚠️ Skipping {lang}: File not found at {path}")

langs = list(heatmaps.keys())  # Only keep successfully loaded languages
n = len(langs)

# === Initialize similarity matrices ===
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
    plt.savefig(filename)
    print(f"📈 Saved {filename}")
    plt.close()

# === Save heatmaps ===
plot_rsa(mse_matrix, langs, "RSA (Mean Squared Error)", "rsa_mse.png", cmap="viridis")
plot_rsa(corr_matrix, langs, "RSA (Pearson Correlation)", "rsa_corr.png", cmap="coolwarm", center=0.0)
plot_rsa(cos_matrix, langs, "RSA (Cosine Similarity)", "rsa_cosine.png", cmap="coolwarm", center=0.0)
