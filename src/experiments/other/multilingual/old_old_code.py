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

#from src.utils.evaluation import evaluate_prompts 
#from src.utils.dla import compute_direct_logit_attribution
#from src.utils.logit_lens import compute_logit_lens, plot_logit_lens, plot_logit_lens_heatmap
#from src.utils.layerwise_logit_attribution import compute_layerwise_gold_logit_contributions, compute_average_layerwise_gold_logit_contributions
#from src.utils.headwise_logit_attribution import compute_headwise_gold_logit_contributions, compute_average_headwise_gold_logit_contributions
#from src.utils.activation_analysis import visualize_average_top_heads_attention

from src.utils.First_2nd_automation import (
    filter_conjugations,
    accuracy_filter,
    group_by_token_lengths,
    prepare_language_dataset,
    save_heatmap,
    resid_pre_direction,
    run_attn_head_out_patching,
    load_json_data
   
)

    
# === Model and Tokenizer Setup ===
model_name = "bigscience/bloom-1b1"
tl_model = HookedTransformer.from_pretrained(model_name)
device = tl_model.cfg.device

print("model:", tl_model)

with open(os.path.expanduser("~/.huggingface/token")) as f:
    hf_token = f.read().strip()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# --- load MorphyNet once ------------------------------------------------
file_path = "/home/lis.isabella.gidi/jsalt2025/src/datasets/morphy_net/MorphyNet_all_conjugations.json"
all_verbs = load_json_data(file_path)

# --- build raw data for Spanish ----------------------------------------
(p1_tok, a1, e1,
 p2_tok, a2, e2,
 texts_1, texts_2,
 spa_conj) = prepare_language_dataset(
     lang_iso3 = "spa",
     lang_name = "spanish",
     all_verbs = all_verbs,
     tokenizer = tokenizer,
     max_verbs = 200,
 )

# --- now run accuracy filtering *outside* ------------------------------
p1_tok, a1, e1, *_ = accuracy_filter(p1_tok, a1, e1, model_name, batch_size=8)
p2_tok, a2, e2, *_ = accuracy_filter(p2_tok, a2, e2, model_name, batch_size=8)

texts_1 = [tokenizer.decode(p, skip_special_tokens=True) for p in p1_tok]
texts_2 = [tokenizer.decode(p, skip_special_tokens=True) for p in p2_tok]



# --- run both directions ------------------------------------------------------
MAX_PROMPTS = 200   # or whatever subset you like

second_texts = texts_2[:MAX_PROMPTS]
first_texts  = texts_1[:MAX_PROMPTS]

resid_pre_direction(first_texts,  a1[:MAX_PROMPTS], e1[:MAX_PROMPTS],
                    second_texts, a2[:MAX_PROMPTS], e2[:MAX_PROMPTS], "first2second", lang_tag = "spanish", tokenizer = tokenizer, tl_model = tl_model, clean_index = 1, corrupt_index=2)

resid_pre_direction(second_texts, a2[:MAX_PROMPTS], e2[:MAX_PROMPTS],
                    first_texts, a1[:MAX_PROMPTS], e1[:MAX_PROMPTS], "second2first", lang_tag = "spanish", tokenizer = tokenizer, tl_model = tl_model, clean_index = 2, corrupt_index=1)



MAX_PROMPTS_HEAD = 50

directions = [
    ("second2first", texts_2[:MAX_PROMPTS_HEAD], texts_1[:MAX_PROMPTS_HEAD], a2[:MAX_PROMPTS_HEAD]),
    ("first2second", texts_1[:MAX_PROMPTS_HEAD], texts_2[:MAX_PROMPTS_HEAD], a1[:MAX_PROMPTS_HEAD]),
]

for label, clean_txts, corrupt_txts, answers in directions:
    run_attn_head_out_patching(
        tl_model,
        clean_txts,
        corrupt_txts,
        answers,
        direction_label=label,
        lang_tag="spanish",
        device=device,
    )
