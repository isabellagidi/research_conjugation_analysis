# Imports
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


from src.utils.dataset_preparation import load_json_data, filter_conjugations, build_conjugation_prompts, accuracy_filter
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


# === Your Data Setup ===

# Assume these are already defined earlier:
# - spanish_first_sing
# - spanish_second_sing
# - tokenizer
# - build_conjugation_prompts()

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
prompts_trimmed_first_sing, answers_ids_first_sing, _ = build_conjugation_prompts(tokenizer, spanish_first_sing, spanish_conjugations)
prompts_trimmed_second_sing, answers_ids_second_sing, _ = build_conjugation_prompts(tokenizer, spanish_second_sing, spanish_conjugations)

# Decode back to strings
prompt_texts_first_sing = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_first_sing]
prompt_texts_second_sing = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_second_sing]

# Construct patching sets: clean is first_sing, corrupt is second_sing
clean_prompts = prompt_texts_first_sing
corrupted_prompts = prompt_texts_second_sing
answer_ids = answers_ids_first_sing  # still predicting the first_sing form

MAX_PROMPTS = 50
clean_prompts = clean_prompts[:MAX_PROMPTS]
corrupted_prompts = corrupted_prompts[:MAX_PROMPTS]
answer_ids = answer_ids[:MAX_PROMPTS]

# Tokenize
clean_tokens = tl_model.to_tokens(clean_prompts)
corrupted_tokens = tl_model.to_tokens(corrupted_prompts)
answer_ids = torch.tensor(answer_ids, device=device)

# === Scoring function ===

def get_logit_diff(logits, answer_ids=answer_ids):
    # Select last logits
    logits = logits[:, -1, :]
    return logits.gather(1, answer_ids.unsqueeze(1)).mean()

# === Run with cache ===
clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = tl_model.run_with_cache(corrupted_tokens)

# Baselines
CLEAN_BASELINE = get_logit_diff(clean_logits).item()
CORRUPTED_BASELINE = get_logit_diff(corrupted_logits).item()

print(f"Clean baseline logit diff: {CLEAN_BASELINE:.4f}")
print(f"Corrupted baseline logit diff: {CORRUPTED_BASELINE:.4f}")

# Normalized metric
def conjugation_metric(logits, answer_ids=answer_ids):
    return (get_logit_diff(logits, answer_ids) - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE + 1e-12)

# === Activation patching: resid_pre ===

resid_pre_act_patch_results = patching.get_act_patch_resid_pre(
    tl_model, corrupted_tokens, clean_cache, patching_metric=conjugation_metric
)

tokens = tl_model.to_str_tokens(clean_tokens[0])
save_heatmap(
    data=resid_pre_act_patch_results,
    x_labels=[f"{tok} {i}" for i, tok in enumerate(tokens)],
    y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
    title="resid_pre Activation Patching (first_sing → second_sing)",
    filename="resid_pre_activation_patching_first2second.png"
)


# === Activation patching: attention head out (all pos) ===

attn_head_out_all_pos_patch_results = patching.get_act_patch_attn_head_out_all_pos(
    tl_model, corrupted_tokens, clean_cache, patching_metric=conjugation_metric
)

save_heatmap(
    data=attn_head_out_all_pos_patch_results,
    x_labels=[f"H{h}" for h in range(tl_model.cfg.n_heads)],
    y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
    title="attn_head_out Activation Patching (All Positions)",
    filename="attn_head_out_all_pos_first2second.png"
)

# === Selective head patching by position ===

# Your top heads: (layer, head)
top_heads = [(23, 14), (23, 1), (22, 1), (21, 1)]
head_labels = [f"L{l}H{h}" for l, h in top_heads]

selective_patching = False #WANT TO SKIP THIS
#SKIPPED
if selective_patching:
    # Initialize recovery tensor: [num_heads, seq_len]
    selective_patch_results = torch.zeros((len(top_heads), clean_tokens.shape[1]), device=device)

    for idx, (layer, head) in enumerate(top_heads):
        hook_name = f"blocks.{layer}.attn.hook_result"

        def patch_entire_head(module, input, output):
            patched_output = output.clone()
            patched_output[:, :, head, :] = clean_cache[hook_name][:, :, head, :]
            return patched_output

        with torch.no_grad():
            with tl_model.hooks([(hook_name, patch_entire_head)]):
                patched_logits = tl_model(corrupted_tokens, return_type="logits")

        score = conjugation_metric(patched_logits, answer_ids)
        selective_patch_results[idx, :] = score  # Fill entire row with one score

        # Free memory after each head
        torch.cuda.empty_cache()
        gc.collect()

        #for pos in range(clean_tokens.shape[1]):
            #def patch_head_at_pos(module, input, output):
                #patched_output = output.clone()
                # Replace only head `h` at position `pos`
                #patched_output[:, pos, head, :] = clean_cache[hook_name][:, pos, head, :]
                #eturn patched_output

            #with torch.no_grad():
                #with tl_model.hooks([(hook_name, patch_head_at_pos)]):
                    #patched_logits = tl_model(corrupted_tokens, return_type="logits")

            #score = conjugation_metric(patched_logits, answer_ids)
            #selective_patch_results[idx, pos] = score


    # Save as heatmap
    save_heatmap(
        data=selective_patch_results,
        x_labels=[f"{tok} {i}" for i, tok in enumerate(tokens)],
        y_labels=head_labels,
        title="Selective Head Patching (Top Heads)",
        filename="selective_head_patching_top_heads.png"
    )


# === Activation patching: head out by position (slower) ===
#made it false so doesn't run!!!
DO_SLOW_RUNS = True
if DO_SLOW_RUNS:
    attn_patch_by_pos = patching.get_act_patch_attn_head_out_by_pos(
        tl_model, corrupted_tokens, clean_cache, patching_metric=conjugation_metric
    )
    attn_patch_by_pos = einops.rearrange(attn_patch_by_pos, "layer pos head -> (layer head) pos")
    head_labels = [f"L{i}H{j}" for i in range(tl_model.cfg.n_layers) for j in range(tl_model.cfg.n_heads)]
    save_heatmap(
        data=attn_patch_by_pos,
        x_labels=[f"{tok} {i}" for i, tok in enumerate(tokens)],
        y_labels=[f"L{i}H{j}" for i in range(tl_model.cfg.n_layers) for j in range(tl_model.cfg.n_heads)],
        title="attn_head_out Activation Patching By Position",
        filename="attn_head_out_by_pos_first2second.png"
    )
