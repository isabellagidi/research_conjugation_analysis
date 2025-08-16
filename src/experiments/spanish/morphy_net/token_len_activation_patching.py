#This code does the resid_pre token-specific activation patching!!
# 
#  Imports
import torch
import einops
from transformer_lens import HookedTransformer
import transformer_lens.patching as patching
import sys
import torch
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

from src.utils.dataset_preparation import load_json_data, filter_conjugations, build_conjugation_prompts, accuracy_filter, group_by_token_lengths
from src.utils.dataset_preparation import generate_first_singular_dataset_spanish, generate_second_singular_dataset_spanish

# Load BLOOM in TransformerLens
model_name = "bigscience/bloom-1b1"
tl_model = HookedTransformer.from_pretrained(model_name)
device = tl_model.cfg.device

with open(os.path.expanduser("~/.huggingface/token")) as f:
    hf_token = f.read().strip()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # required for BLOOM

def save_heatmap(
    data: torch.Tensor,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    filename: str,
    center: float = 0.0,
    cmap: str = "coolwarm"
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

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

# Load your JSON data
file_path = "/home/lis.isabella.gidi/jsalt2025/src/datasets/morphy_net/MorphyNet_all_conjugations.json"
all_verbs = load_json_data(file_path)
# Prompt sets

spanish_conjugations = filter_conjugations(all_verbs, tokenizer, "spa")

#DOING THIS TO SAVE RAM!!!!!!!
spanish_conjugations = spanish_conjugations[:100]
print("length of dataset (usually 500 but in this case):" , len(spanish_conjugations))

spanish_first_sing = generate_first_singular_dataset_spanish(spanish_conjugations)
spanish_second_sing = generate_second_singular_dataset_spanish(spanish_conjugations)


# Build tokenized prompts and gold answer IDs
prompts_trimmed_first_sing_spanish, answers_ids_first_sing_spanish, entries_first_sing_spanish = build_conjugation_prompts(tokenizer, spanish_first_sing, spanish_conjugations)
prompts_trimmed_second_sing_spanish, answers_ids_second_sing_spanish, entries_second_sing_spanish = build_conjugation_prompts(tokenizer, spanish_second_sing, spanish_conjugations)

# Decode back to strings
prompt_texts_first_sing_spanish = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_first_sing_spanish]
prompt_texts_second_sing_spanish = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_second_sing_spanish]

#GROUPING BY TOKEN LENGTH

#FIRST SING
grouped_examples_first_sing_spanish = group_by_token_lengths(
    prompt_texts_first_sing_spanish,
    answers_ids_first_sing_spanish,
    entries_first_sing_spanish,  # <-- from accuracy_filter
    tokenizer
)

#SECOND SING

grouped_examples_second_sing_spanish = group_by_token_lengths(
    prompt_texts_second_sing_spanish,
    answers_ids_second_sing_spanish,
    entries_second_sing_spanish,  # <-- from accuracy_filter
    tokenizer
)

# Print how many examples in each group
for k, v in sorted(grouped_examples_second_sing_spanish.items()):
    print(f"Infinitive tokens: {k[0]}, Conjugated tokens: {k[1]} --> {len(v)} examples")


#CREATING ARRAYS OF TOP 5 (or whatever you want)
top_5_tokenizations_second_sing_spanish = sorted(grouped_examples_second_sing_spanish.items(), key=lambda x: len(x[1]), reverse=True)[:5]


print("\n🔢 Top 5 tokenization groups by frequency for second sing Spanish:")
for (inf_len, conj_len), group in top_5_tokenizations_second_sing_spanish:
    print(f" - (inf={inf_len}, conj={conj_len}): {len(group)} examples")

# Print how many examples in each group
for k, v in sorted(grouped_examples_first_sing_spanish.items()):
    print(f"Infinitive tokens: {k[0]}, Conjugated tokens: {k[1]} --> {len(v)} examples")


#CREATING ARRAYS OF TOP 5 (or whatever you want)
top_5_tokenizations_first_sing_spanish = sorted(grouped_examples_first_sing_spanish.items(), key=lambda x: len(x[1]), reverse=True)[:5]


print("\n🔢 Top 5 tokenization groups by frequency for first sing Spanish:")
for (inf_len, conj_len), group in top_5_tokenizations_first_sing_spanish:
    print(f" - (inf={inf_len}, conj={conj_len}): {len(group)} examples")


# === Scoring function ===

def get_logit_diff(logits, answer_ids=answers_ids_first_sing_spanish):
    # Select last logits
    logits = logits[:, -1, :]
    return logits.gather(1, answer_ids.unsqueeze(1)).mean()


# Normalized metric
def conjugation_metric(logits, answer_ids=answers_ids_first_sing_spanish):
    return (get_logit_diff(logits, answer_ids) - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE + 1e-12)

for (inf_len, conj_len), group in top_5_tokenizations_first_sing_spanish:
    clean_prompts = [ex[0] for ex in group]
    answer_ids = torch.tensor([ex[1] for ex in group], device=device)

    # Match with second-person prompts from same group
    corrupted_group = grouped_examples_second_sing_spanish.get((inf_len, conj_len))
    if corrupted_group is None or len(corrupted_group) < len(group):
        print(f"❌ Skipping group {(inf_len, conj_len)} — missing second-person match")
        continue

    corrupted_prompts = [ex[0] for ex in corrupted_group]

    # Truncate to equal length
    n = min(len(clean_prompts), len(corrupted_prompts))
    clean_prompts = clean_prompts[:n]
    corrupted_prompts = corrupted_prompts[:n]
    answer_ids = answer_ids[:n]

    # Tokenize
    clean_tokens = tl_model.to_tokens(clean_prompts)
    corrupted_tokens = tl_model.to_tokens(corrupted_prompts)

    # Run with cache
    clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = tl_model.run_with_cache(corrupted_tokens)

    # Metric
    CLEAN_BASELINE = get_logit_diff(clean_logits, answer_ids).item()
    CORRUPTED_BASELINE = get_logit_diff(corrupted_logits, answer_ids).item()


    def conjugation_metric(logits, answer_ids=answer_ids):
        return (get_logit_diff(logits, answer_ids) - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE + 1e-12)

    # Patching
    resid_pre_act_patch_results = patching.get_act_patch_resid_pre(
        tl_model, corrupted_tokens, clean_cache, patching_metric=conjugation_metric
    )

    tokens = tl_model.to_str_tokens(clean_tokens[0])
    save_heatmap(
        data=resid_pre_act_patch_results,
        x_labels=[f"{tok} {i}" for i, tok in enumerate(tokens)],
        y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
        title=f"resid_pre Activation Patching (inf={inf_len}, conj={conj_len})",
        filename=f"resid_pre_activation_patching_inf{inf_len}_conj{conj_len}.png"
    )
