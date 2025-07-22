# === Imports ===
import torch
import einops
from transformer_lens import HookedTransformer
import transformer_lens.patching as patching
import sys
from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import gc

from src.utils.evaluation import evaluate_prompts 
from src.utils.dla import compute_direct_logit_attribution
from src.utils.logit_lens import compute_logit_lens, plot_logit_lens, plot_logit_lens_heatmap
from src.utils.layerwise_logit_attribution import compute_layerwise_gold_logit_contributions, compute_average_layerwise_gold_logit_contributions
from src.utils.headwise_logit_attribution import compute_headwise_gold_logit_contributions, compute_average_headwise_gold_logit_contributions
from src.utils.activation_analysis import visualize_average_top_heads_attention

from src.utils.dataset_preparation import (
    load_json_data,
    filter_conjugations,
    build_conjugation_prompts,
    accuracy_filter,
    group_by_token_lengths,
    generate_first_singular_dataset_english,
    generate_second_singular_dataset_english,
)

# === Model and Tokenizer Setup ===
model_name = "bigscience/bloom-1b1"
tl_model = HookedTransformer.from_pretrained(model_name)
device = tl_model.cfg.device

with open(os.path.expanduser("~/.huggingface/token")) as f:
    hf_token = f.read().strip()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def save_heatmap(data, x_labels, y_labels, title, filename, center=0.0, cmap="coolwarm"):
    plt.figure(figsize=(min(len(x_labels) * 0.5, 18), min(len(y_labels) * 0.5, 12)))
    sns.heatmap(
        data.cpu().numpy(),
        xticklabels=x_labels,
        yticklabels=y_labels,
        center=center,
        cmap=cmap,
        cbar_kws={"label": "Normalized Recovery"}
    )
    plt.xlabel("Position" if "Position" in title else "Head")
    plt.ylabel("Layer" if "Layer" in title else "Head")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Saved: {filename}")
    plt.close()

print("model:", tl_model)

# === Load and Prepare Dataset ===
file_path = "/home/lis.isabella.gidi/jsalt2025/src/datasets/morphy_net/MorphyNet_all_conjugations.json"
all_verbs = load_json_data(file_path)
english_conjugations = filter_conjugations(all_verbs, tokenizer, "spa")
english_conjugations = english_conjugations[:100]
print("length of dataset (usually 500 but in this case):", len(english_conjugations))

english_first_sing = generate_first_singular_dataset_english(english_conjugations)
english_second_sing = generate_second_singular_dataset_english(english_conjugations)

# Build prompts
p1, a1, e1 = build_conjugation_prompts(tokenizer, english_first_sing, english_conjugations)
p2, a2, e2 = build_conjugation_prompts(tokenizer, english_second_sing, english_conjugations)

texts_1 = [tokenizer.decode(p, skip_special_tokens=True) for p in p1]
texts_2 = [tokenizer.decode(p, skip_special_tokens=True) for p in p2]

grouped_first = group_by_token_lengths(texts_1, a1, e1, tokenizer)
grouped_second = group_by_token_lengths(texts_2, a2, e2, tokenizer)

top_5 = sorted(grouped_first.items(), key=lambda x: len(x[1]), reverse=True)[:5]

# === Patch 5 groups: resid_pre ===
for (inf_len, conj_len), group in top_5:
    clean_prompts = [ex[0] for ex in group]
    answer_ids = torch.tensor([ex[1] for ex in group], device=device)

    corrupted_group = grouped_second.get((inf_len, conj_len))
    if corrupted_group is None or len(corrupted_group) < len(group):
        print(f"❌ Skipping group {(inf_len, conj_len)} — missing second-person match")
        continue

    corrupted_prompts = [ex[0] for ex in corrupted_group]
    n = min(len(clean_prompts), len(corrupted_prompts))
    clean_prompts = clean_prompts[:n]
    corrupted_prompts = corrupted_prompts[:n]
    answer_ids = answer_ids[:n]

    clean_tokens = tl_model.to_tokens(clean_prompts)
    corrupted_tokens = tl_model.to_tokens(corrupted_prompts)

    clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = tl_model.run_with_cache(corrupted_tokens)

    def get_logit_diff(logits, answer_ids=answer_ids):
        logits = logits[:, -1, :]
        return logits.gather(1, answer_ids.unsqueeze(1)).mean()

    CLEAN_BASELINE = get_logit_diff(clean_logits, answer_ids).item()
    CORRUPTED_BASELINE = get_logit_diff(corrupted_logits, answer_ids).item()

    def conjugation_metric(logits, answer_ids=answer_ids):
        return (get_logit_diff(logits, answer_ids) - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE + 1e-12)

    resid_pre_act_patch_results = patching.get_act_patch_resid_pre(
        tl_model, corrupted_tokens, clean_cache, patching_metric=conjugation_metric
    )

    tokens = tl_model.to_str_tokens(clean_tokens[0])
    save_heatmap(
        data=resid_pre_act_patch_results,
        x_labels=[f"{tok} {i}" for i, tok in enumerate(tokens)],
        y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
        title=f"resid_pre Activation Patching english (inf={inf_len}, conj={conj_len})",
        filename=f"resid_pre_activation_patching_inf{inf_len}_conj{conj_len}_english.png"
    )

# === Additional First→Second Patching: attn_head_out_all_pos ===
# Use the first MAX_PROMPTS prompts for 1st→2nd person comparison
MAX_PROMPTS = 50
clean_prompts = texts_1[:MAX_PROMPTS]
corrupted_prompts = texts_2[:MAX_PROMPTS]
answer_ids = torch.tensor(a1[:MAX_PROMPTS], device=device)

clean_tokens = tl_model.to_tokens(clean_prompts)
corrupted_tokens = tl_model.to_tokens(corrupted_prompts)
clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
corrupted_logits, _ = tl_model.run_with_cache(corrupted_tokens)

def get_logit_diff(logits, answer_ids=answer_ids):
    logits = logits[:, -1, :]
    return logits.gather(1, answer_ids.unsqueeze(1)).mean()

CLEAN_BASELINE = get_logit_diff(clean_logits).item()
CORRUPTED_BASELINE = get_logit_diff(corrupted_logits).item()

def conjugation_metric(logits, answer_ids=answer_ids):
    return (get_logit_diff(logits, answer_ids) - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE + 1e-12)

attn_head_out_all_pos_patch_results = patching.get_act_patch_attn_head_out_all_pos(
    tl_model, corrupted_tokens, clean_cache, patching_metric=conjugation_metric
)

save_heatmap(
    data=attn_head_out_all_pos_patch_results,
    x_labels=[f"H{h}" for h in range(tl_model.cfg.n_heads)],
    y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
    title="attn_head_out Activation Patching (All Positions) english",
    filename="attn_head_out_all_pos_first2second_english.png"
)