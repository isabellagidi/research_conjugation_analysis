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

from utils_mega_automation import (
    filter_conjugations,
    accuracy_filter,
    group_by_token_lengths,
    prepare_language_dataset,
    save_heatmap,
    resid_pre_direction,
    run_attn_head_out_patching,
    load_json_data,
    PERSON_TO_TUPLE_INDEX, 
    PERSON_SHORT_TAG,
    PERSON_TO_JSON_KEY,
    make_confidence_bins_from_logp,
    compute_gold_logprobs,
    make_bins_from_probs,              # <-- correct helper name
    compute_gold_logprobs,             # <-- already referenced later
    align_corrupt_indices_by_lemma,    # <-- you use this below
)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
import os
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# -----------------------------------------------------

def activation_patching(
        *,                    # keyword-only
        lang_iso3: str,
        lang_name: str,
        tl_model,             # <- shared model (no reload)
        tokenizer,            # <- shared tokenizer
        model_name: str,      # just the string for accuracy_filter
        person_a="first singular",   # must be exactly one of the six
        person_b="second singular",
        max_verbs: int = 500,
        max_prompts_resid: int = 200,
        max_prompts_head: int = 50,
):
     # --- build results/<org>/<model>/<pair>/... (bin will be inserted later) ---
    tag_a   = PERSON_SHORT_TAG[person_a]
    tag_b   = PERSON_SHORT_TAG[person_b]
    pair_id = f"{tag_a}-{tag_b}"                  # e.g., "1sg-2sg"
    model_parts = model_name.split("/")           # ["bigscience","bloom-1b1"]
    base_out_dir = os.path.join("results", *model_parts)

    orig_cwd = os.getcwd()

    try:
        device = tl_model.cfg.device      # uses the shared model

        # --- load MorphyNet once --------------------------------------
        file_path = ("/home/lis.isabella.gidi/jsalt2025/src/datasets/"
                     "morphy_net/MorphyNet_all_present_conjugations.json")
        all_verbs = load_json_data(file_path)

        # --- dataset prep ---------------------------------------------
        (p1_raw, a1, e1,
         p2_raw, a2, e2,
         texts_1, texts_2, _) = prepare_language_dataset(
            lang_iso3  = lang_iso3,
            lang_name  = lang_name,
            all_verbs  = all_verbs,
            tokenizer  = tokenizer,
            max_verbs  = max_verbs,
            person_a   = person_a,
            person_b   = person_b,
        )

        # --- accuracy filter on *CPU* ---------------------------------
        # Keep only prompts where the model predicts its *own* gold token correctly.
        p1_tok, a1, e1, *_ = accuracy_filter(p1_raw, a1, e1, model_name, batch_size=8, device="cpu")
        p2_tok, a2, e2, *_ = accuracy_filter(p2_raw, a2, e2, model_name, batch_size=8, device="cpu")

        # convert back to text (for grouping / patching)
        texts_1 = [tokenizer.decode(t, skip_special_tokens=True) for t in p1_tok]  # person A
        texts_2 = [tokenizer.decode(t, skip_special_tokens=True) for t in p2_tok]  # person B

        # --- CLEAN-SIDE CONFIDENCE BINS (with lemma alignment for recovery) ----
        # Direction B->A: clean = person B (texts_2/a2/e2), corrupted = person A (texts_1/e1)
        clean_BtoA_logps = compute_gold_logprobs(tl_model, texts_2, a2, device)
        clean_BtoA_probs = [math.exp(lp) for lp in clean_BtoA_logps]
        bins_BtoA = make_bins_from_probs(clean_BtoA_probs)

        # Direction A->B: clean = person A (texts_1/a1/e1), corrupted = person B (texts_2/e2)
        clean_AtoB_logps = compute_gold_logprobs(tl_model, texts_1, a1, device)
        clean_AtoB_probs = [math.exp(lp) for lp in clean_AtoB_logps]
        bins_AtoB = make_bins_from_probs(clean_AtoB_probs)
        # ------------------------------------------------------------------------

        # --- head-out patching per BIN and DIRECTION ----------------------------
        # (label, clean_texts, clean_answers, clean_entries, corrupt_texts, corrupt_entries, bins)
        directions = [
            (f"{tag_b}to{tag_a}", texts_2, a2, e2, texts_1, e1, bins_BtoA),  # B->A
            (f"{tag_a}to{tag_b}", texts_1, a1, e1, texts_2, e2, bins_AtoB),  # A->B
        ]

        for label, clean_txts_src, clean_ans_src, clean_ent_src, corrupt_txts_src, corrupt_ent_src, bin_map in directions:
            for bin_name, clean_idxs in bin_map.items():
                if not clean_idxs:
                    print(f"↪️  {label}/{bin_name}: skipping - zero items")
                    continue

                # Align clean and corrupt by lemma (restrict to pairs present on both sides)
                clean_idxs_aligned, corrupt_idxs_aligned = align_corrupt_indices_by_lemma(
                    clean_idxs, clean_ent_src, corrupt_ent_src
                )
                if not clean_idxs_aligned:
                    print(f"↪️  {label}/{bin_name}: skipping - no aligned pairs")
                    continue

                # Cap per your knob
                k = min(len(clean_idxs_aligned), max_prompts_head)
                clean_idxs_aligned   = clean_idxs_aligned[:k]
                corrupt_idxs_aligned = corrupt_idxs_aligned[:k]

                # Build per-bin slices
                clean_txts = [clean_txts_src[i] for i in clean_idxs_aligned]
                answers    = [clean_ans_src[i]   for i in clean_idxs_aligned]
                corrupt_txts = [corrupt_txts_src[j] for j in corrupt_idxs_aligned]

                # Bin-specific output dir: results/<org>/<model>/<pair>/<bin>/<language>/
                out_dir = os.path.join(base_out_dir, bin_name, pair_id, lang_name)
                os.makedirs(out_dir, exist_ok=True)
                os.chdir(out_dir)
                try:
                    print(f"[{lang_name}] {label} | bin={bin_name} | n={len(clean_txts)} (aligned)")
                    run_attn_head_out_patching(
                        tl_model,
                        clean_txts,
                        corrupt_txts,
                        answers,
                        direction_label=f"{label}_{bin_name}",
                        lang_tag=lang_name,
                        device=device
                    )
                finally:
                    os.chdir(orig_cwd)

        torch.cuda.empty_cache()
        gc.collect()

    finally:
        os.chdir(orig_cwd)
