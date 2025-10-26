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
    generate_first_singular_dataset_spanish,
    generate_second_singular_dataset_spanish,
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
#NOTE THIS WAS WRONG BEFORE
spanish_conjugations = filter_conjugations(all_verbs, tokenizer, "spa")
spanish_conjugations = spanish_conjugations[:200]
print("length of dataset (usually 500 but in this case):", len(spanish_conjugations))

spanish_first_sing = generate_first_singular_dataset_spanish(spanish_conjugations)
spanish_second_sing = generate_second_singular_dataset_spanish(spanish_conjugations)

# Build prompts
p1, a1, e1 = build_conjugation_prompts(tokenizer, spanish_first_sing, spanish_conjugations)
p2, a2, e2 = build_conjugation_prompts(tokenizer, spanish_second_sing, spanish_conjugations)

# --- Filter for accuracy ---
p1, a1, e1, _, _, _ = accuracy_filter(p1, a1, e1, model_name, batch_size=8)
p2, a2, e2, _, _, _ = accuracy_filter(p2, a2, e2, model_name, batch_size=8)

texts_1 = [tokenizer.decode(p, skip_special_tokens=True) for p in p1]
texts_2 = [tokenizer.decode(p, skip_special_tokens=True) for p in p2]

#FIRST CLEAN SECOND CORRUPT
#grouped_clean = group_by_token_lengths(texts_1, a1, e1, tokenizer)
#grouped_corrupted = group_by_token_lengths(texts_2, a2, e2, tokenizer)
#top_5 = sorted(grouped_first.items(), key=lambda x: len(x[1]), reverse=True)[:5]

# SWAPPED!!!! Second → First: now group "second person" as the clean set
#grouped_clean = group_by_token_lengths(texts_2, a2, e2, tokenizer)
#grouped_corrupted = group_by_token_lengths(texts_1, a1, e1, tokenizer)
#top_5 = sorted(grouped_clean.items(), key=lambda x: len(x[1]), reverse=True)[:5]


# --- helper to run resid_pre patching for one direction ------------------------
def resid_pre_direction(clean_texts, clean_ans, clean_entries,
                        corrupted_texts, corrupted_ans, corrupted_entries, direction_label):
    # ❶  group both sets *separately*
    grouped_clean     = group_by_token_lengths(clean_texts, clean_ans, clean_entries, tokenizer)
    grouped_corrupted = group_by_token_lengths(corrupted_texts, corrupted_ans, corrupted_entries, tokenizer)

    # top k groups **from the clean side**
    top_groups = sorted(grouped_clean.items(),
                        key=lambda kv: len(kv[1]),
                        reverse=True)[:5]

    for (inf_len, conj_len), clean_group in top_groups:
        corrupted_group = grouped_corrupted.get((inf_len, conj_len))
        if corrupted_group is None:
            print(f"↪️  {direction_label}: skipping {(inf_len, conj_len)} – no match")
            continue

        # clip to common length
        m = min(len(clean_group), len(corrupted_group))
        clean_prompts     = [ex[0] for ex in clean_group][:m]
        corrupted_prompts = [ex[0] for ex in corrupted_group][:m]
        answer_ids        = torch.tensor([ex[1] for ex in clean_group][:m], device=device)

        # run patching
        clean_toks     = tl_model.to_tokens(clean_prompts)
        corrupted_toks = tl_model.to_tokens(corrupted_prompts)
        clean_logits, clean_cache = tl_model.run_with_cache(clean_toks)
        corrupted_logits, _       = tl_model.run_with_cache(corrupted_toks)

        def logit_diff(logits):
            return logits[:, -1, :].gather(1, answer_ids.unsqueeze(1)).mean()

        clean_base     = logit_diff(clean_logits).item()
        corrupted_base = logit_diff(corrupted_logits).item()

        def metric(logits):
            return (logit_diff(logits) - corrupted_base) / (clean_base - corrupted_base + 1e-12)

        patch = patching.get_act_patch_resid_pre(
            tl_model, corrupted_toks, clean_cache, patching_metric=metric
        )

        save_heatmap(
            data=patch,
            x_labels=[f"{tok} {i}" for i, tok in enumerate(
                tl_model.to_str_tokens(clean_toks[0]))],
            y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
            title=f"resid_pre ‑ {direction_label} (inf={inf_len},conj={conj_len})",
            filename=f"resid_pre_{direction_label}_inf{inf_len}_conj{conj_len}.png"
        )
        torch.save(patch, f"resid_pre_{direction_label}_inf{inf_len}_conj{conj_len}.pt")


# --- run both directions ------------------------------------------------------
MAX_PROMPTS = 200   # or whatever subset you like

second_texts = texts_2[:MAX_PROMPTS]
first_texts  = texts_1[:MAX_PROMPTS]

resid_pre_direction(second_texts, a2[:MAX_PROMPTS], e2[:MAX_PROMPTS],
                    first_texts, a1[:MAX_PROMPTS], e1[:MAX_PROMPTS], "second2first")

resid_pre_direction(first_texts,  a1[:MAX_PROMPTS], e1[:MAX_PROMPTS],
                    second_texts, a2[:MAX_PROMPTS], e2[:MAX_PROMPTS], "first2second")


# === Run Both Second→First and First→Second Activation Patching ===

MAX_PROMPTS = 50

for direction, clean_texts, corrupted_texts, clean_answers, direction_label in [
    ("second2first", texts_2[:MAX_PROMPTS], texts_1[:MAX_PROMPTS], a2[:MAX_PROMPTS], "second2first"),
    ("first2second", texts_1[:MAX_PROMPTS], texts_2[:MAX_PROMPTS], a1[:MAX_PROMPTS], "first2second"),
]:
    print(f"\n🔁 Running direction: {direction_label}")

    n = min(len(clean_texts), len(corrupted_texts), len(clean_answers))
    if n == 0:
        print("↪️  Skipping", direction_label, "-- zero aligned prompts")
        continue

    clean_prompts      = clean_texts[:n]
    corrupted_prompts  = corrupted_texts[:n]
    answer_ids         = torch.tensor(clean_answers[:n], device=device)

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

    patch_results = patching.get_act_patch_attn_head_out_all_pos(
        tl_model, corrupted_tokens, clean_cache, patching_metric=conjugation_metric
    )

    save_heatmap(
        data=patch_results,
        x_labels=[f"H{h}" for h in range(tl_model.cfg.n_heads)],
        y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
        title=f"attn_head_out Activation Patching (All Positions) spanish ({direction_label})",
        filename=f"attn_head_out_all_pos_{direction_label}_spanish.png"
    )

    torch.save(patch_results, f"attn_head_out_all_pos_patch_results_{direction_label}_spanish.pt")