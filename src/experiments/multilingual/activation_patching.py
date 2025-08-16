# this version took up too much memory


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

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
import os
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

def activation_patching(
    *,                       # keyword‑only for clarity
    lang_iso3: str,          # "spa", "ita", …
    lang_name: str,          # "spanish", "italian", …
    max_verbs: int = 500,
    max_prompts_resid: int = 200,
    max_prompts_head: int = 50,
):
    # -----------  set up output folder ---------------------------------
    out_dir = os.path.join("automation_results", lang_name)
    os.makedirs(out_dir, exist_ok=True)
    orig_cwd = os.getcwd()          # remember current dir
    os.chdir(out_dir)               # everything saved here

    try:
        # === Model and Tokenizer Setup ===
        model_name = "bigscience/bloom-1b1"
        tl_model   = HookedTransformer.from_pretrained(model_name)
        device     = tl_model.cfg.device
        print("model:", tl_model)

        with open(os.path.expanduser("~/.huggingface/token")) as f:
            hf_token = f.read().strip()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # --- load MorphyNet once ---------------------------------------
        file_path = "/home/lis.isabella.gidi/jsalt2025/src/datasets/morphy_net/MorphyNet_all_present_conjugations.json"
        all_verbs = load_json_data(file_path)

        # --- build raw data for the language ---------------------------
        (p1_unfiltered, a1, e1,
         p2_unfiltered, a2, e2,
         texts_1, texts_2,
         _conj) = prepare_language_dataset(
            lang_iso3 = lang_iso3,
            lang_name = lang_name,
            all_verbs = all_verbs,
            tokenizer = tokenizer,
            max_verbs = max_verbs,
        )

        print(
            f"[{lang_name}] after prepare_language_dataset  "
            f"first_prompts={len(texts_1)}  second_prompts={len(texts_2)}"
)
        # --- accuracy filtering ----------------------------------------
        p1_tok, a1, e1, *_ = accuracy_filter(p1_unfiltered, a1, e1, model_name, batch_size=8, device="cpu")
        p2_tok, a2, e2, *_ = accuracy_filter(p2_unfiltered, a2, e2, model_name, batch_size=8, device="cpu")


        print("after accuracy_filter, number of sentences saved:", len(p1_tok), "/", len(p1_unfiltered), ";", len(p2_tok), "/", len(p2_unfiltered))


        texts_1 = [tokenizer.decode(p, skip_special_tokens=True) for p in p1_tok]
        texts_2 = [tokenizer.decode(p, skip_special_tokens=True) for p in p2_tok]

        # --- resid_pre patching ----------------------------------------
        first_texts  = texts_1[:max_prompts_resid]
        second_texts = texts_2[:max_prompts_resid]

        resid_pre_direction(
            first_texts,  a1[:max_prompts_resid], e1[:max_prompts_resid],
            second_texts, a2[:max_prompts_resid], e2[:max_prompts_resid],
            "first2second", lang_tag = lang_name,
            tokenizer = tokenizer, tl_model = tl_model,
            clean_index = 1, corrupt_index = 2
        )

        resid_pre_direction(
            second_texts, a2[:max_prompts_resid], e2[:max_prompts_resid],
            first_texts,  a1[:max_prompts_resid], e1[:max_prompts_resid],
            "second2first", lang_tag = lang_name,
            tokenizer = tokenizer, tl_model = tl_model,
            clean_index = 2, corrupt_index = 1
        )

        # --- attn_head_out patching ------------------------------------
        directions = [
            ("second2first", texts_2[:max_prompts_head], texts_1[:max_prompts_head], a2[:max_prompts_head]),
            ("first2second", texts_1[:max_prompts_head], texts_2[:max_prompts_head], a1[:max_prompts_head]),
        ]

        for label, clean_txts, corrupt_txts, answers in directions:
            run_attn_head_out_patching(
                tl_model,
                clean_txts,
                corrupt_txts,
                answers,
                direction_label = label,
                lang_tag        = lang_name,
                device          = device,
            )

    finally:
        os.chdir(orig_cwd)   # restore working dir
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------

#activation_patching(lang_iso3="ita",
#    lang_name="italian",
#    max_verbs=200,            # verbs to keep from MorphyNet
#    max_prompts_resid=200,    # prefixes passed to resid_pre patching
#    max_prompts_head=50       # prefixes passed to attn_head_out patching
#)


# ---------------------------------------------------------------
# map the ISO‑3 codes you listed to the lang_name strings used in
# generate_dataset() / prepare_language_dataset()

LANG_MAP = {
    "cat": ("catalan",       "cat"),
    "ces": ("czech",         "ces"),
    "deu": ("german",        "deu"),
    "eng": ("english",       "eng"),
    "fin": ("finnish",       "fin"),
    "fra": ("french",        "fra"),
    "hbs": ("serbo-croatian","hbs"),
    "hun": ("hungarian",     "hun"),
    "ita": ("italian",       "ita"),
    "mon": ("mongolian",     "mon"),
    "pol": ("polish",        "pol"),
    "por": ("portuguese",    "por"),
    "rus": ("russian",       "rus"),
    "spa": ("spanish",       "spa"),
    "swe": ("swedish",       "swe"),
}

# size knobs
MAX_VERBS          = 1000    # per language
MAX_PROMPTS_RESID  = 200
MAX_PROMPTS_HEAD   = 50

# ---------------------------------------------------------------
#for iso3 in [
#    "cat", "ces", "deu", "eng", "fin", "fra", "hun",
#    "ita", "mon", "por", "rus", "spa", "swe"
#]:
for iso3 in [
    "ces", "deu",
    "mon", "rus", "spa", "swe"
]:
    lang_name, iso_code = LANG_MAP[iso3]   # iso_code == iso3 here; kept for clarity
    print(f"\n=== Running activation patching for {lang_name} ({iso3}) ===")
    activation_patching(
        lang_iso3          = iso_code,
        lang_name          = lang_name,
        max_verbs          = MAX_VERBS,
        max_prompts_resid  = MAX_PROMPTS_RESID,
        max_prompts_head   = MAX_PROMPTS_HEAD,
    )


