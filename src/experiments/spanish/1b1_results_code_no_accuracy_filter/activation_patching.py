import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# maybe cut some of these
sys.path.append('../../')
from src.datasets.spanish.spanish_verbs import spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs
from src.utils.spanish_dataset_generation import create_spanish_verbs, filter_spanish_conjugations
from src.utils.spanish_build_prompts import generate_yo_dataset, generate_tu_dataset, build_conjugation_prompts
from src.utils.evaluation import evaluate_prompts 
from src.utils.dla import compute_direct_logit_attribution
from src.utils.logit_lens import compute_logit_lens, plot_logit_lens, plot_logit_lens_heatmap
from src.utils.layerwise_logit_attribution import compute_layerwise_gold_logit_contributions, compute_average_layerwise_gold_logit_contributions
from src.utils.headwise_logit_attribution import compute_headwise_gold_logit_contributions, compute_average_headwise_gold_logit_contributions
from src.utils.activation_analysis import visualize_average_top_heads_attention
from src.utils.layer_activation_patching import run_mlp_attention_patching, plot_layerwise_activation_patching_heatmap
from src.utils.head_activation_patching import run_head_patching, plot_head_patching_heatmap

#NEW!!!
from src.utils.activation_patching import patch_residual_component, logits_to_avg_logit_diff, normalize_patch_diff, run_activation_patching, plot_activation_patching_heatmap


model_name = "bigscience/bloom-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
model.eval()

print("Starting activation patching…")
sys.stdout.flush()


# Build dataset
spanish_verbs = create_spanish_verbs(spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs)
spanish_conjugations = filter_spanish_conjugations(spanish_verbs, tokenizer)

# Prompt generators

# Prompt sets
spanish_yo = generate_yo_dataset(spanish_conjugations)
spanish_tu = generate_tu_dataset(spanish_conjugations)


#Build Conjugation Prompts

# Build prompts and answer token IDs
prompts_trimmed_yo, answers_ids_yo, _ = build_conjugation_prompts(tokenizer, spanish_yo)
prompts_trimmed_tu, answers_ids_tu, _ = build_conjugation_prompts(tokenizer, spanish_tu)

# Decode prompts back to full text for BLOOM
prompt_texts_yo = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_yo]
prompt_texts_tu = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_tu]

# Activation patching from YO → TÚ direction
clean_prompts_yo2tu    = prompt_texts_yo         # e.g., "Yo hablo"
corrupted_prompts_yo2tu = prompt_texts_tu        # e.g., "Tú hablo"
gold_token_ids_yo2tu   = answers_ids_yo          # still the correct "hablo" token

# Activation patching from TÚ → YO direction
#clean_prompts_tu2yo    = prompt_texts_tu
#corrupted_prompts_tu2yo = prompt_texts_yo
#gold_token_ids_tu2yo   = answers_ids_tu

#patched_diff_yo2tu, token_labels_yo2tu = run_activation_patching(model=model, tokenizer=tokenizer, clean_prompts=clean_prompts_yo2tu, corrupted_prompts=corrupted_prompts_yo2tu, gold_token_ids=gold_token_ids_yo2tu)

#plot_activation_patching_heatmap(patched_diff_yo2tu, token_labels_yo2tu, label="yo2tu")


#recovery_yo2tu, toks_yo2tu = run_activation_patching(model, tokenizer, clean_prompts_yo2tu, corrupted_prompts_yo2tu, gold_token_ids_yo2tu, max_seq_len=32)

#plot_activation_patching_heatmap(recovery_yo2tu, toks_yo2tu, label="YO→TÚ")

#the code for attention and MLP
#attn_diff_yo2tu, mlp_diff_yo2tu, token_labels_yo2tu = run_mlp_attention_patching( model=model, tokenizer=tokenizer, clean_prompts=clean_prompts_yo2tu, corrupted_prompts=corrupted_prompts_yo2tu, gold_token_ids=gold_token_ids_yo2tu)

#plot_layerwise_activation_patching_heatmap(attn_diff_yo2tu, token_labels_yo2tu, label="YO→TÚ_attn")
#plot_layerwise_activation_patching_heatmap(mlp_diff_yo2tu, token_labels_yo2tu, label="YO→TÚ_mlp")

recovery = run_head_patching(
    model=model,
    tokenizer=tokenizer,
    clean_prompts=clean_prompts_yo2tu,
    corrupted_prompts=corrupted_prompts_yo2tu,
    gold_token_ids=gold_token_ids_yo2tu,
)

plot_head_patching_heatmap(recovery, label="YO→TÚ")

print("Done.")
sys.stdout.flush()
