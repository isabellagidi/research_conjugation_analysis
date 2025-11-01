#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-lingual activation patching runner.

- Builds clean (source lang/slot) and corrupted (target lang/opposite-slot) batches.
- Intersects groups by (inf_len, conj_len) across languages for the source slot.
- For each top-K shared group, pairs items and patches attention-head outputs
  from the clean (source) cache into the corrupted (target) forward pass.
- Evaluates lift on the target language's *target-slot* gold token IDs.

Output tree:
  results/<org>/<model>/xling/<LANG_A>__from__<LANG_B>/<slotSrc>to<slotTgt>/shape_inf{u}_conj{v}/
    - xling_attn_head_out_all_pos_<label>.png
    - xling_attn_head_out_patch_results_<label>.pt
    - meta.json
"""

import os
import json
import math
import random
import argparse
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import transformer_lens.patching as patching

# ---------- import your existing utilities ----------
# Make sure all of these names are exported by utils_mega_automation.py
from utils_mega_automation import (
    generate_dataset,
    accuracy_filter,
    group_by_token_lengths,
    save_heatmap,
    load_json_data,
    PERSON_TO_TUPLE_INDEX,
    PERSON_SHORT_TAG,
    # Optional convenience loader; if you don't export it, the fallback below is used.
    # load_tl_for_bloom_or_bloomz,
)

# ------------------ language maps -------------------
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

# ------------------ small helpers -------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_pad_token(tokenizer: AutoTokenizer):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

def model_and_tokenizer_from_name(model_name: str, device: str = "cuda"):
    """
    Try plain TL load; if it's a BLOOMZ variant and fails, fallback to a BLOOMZ→BLOOM bridge.
    """
    try:
        tl_model = HookedTransformer.from_pretrained(model_name).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        safe_pad_token(tokenizer)
        return tl_model, tokenizer
    except Exception:
        # Fallback for BLOOMZ (maps to BLOOM converter with HF model weights)
        if "/bloomz-" in model_name:
            base_name = model_name.replace("bloomz-", "bloom-")
            from transformers import AutoModelForCausalLM
            hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            safe_pad_token(tokenizer)
            tl_model = HookedTransformer.from_pretrained(base_name, hf_model=hf_model).to(device).eval()
            return tl_model, tokenizer
        raise

def tuple_index(slot: str) -> int:
    """Map 'first singular' → 1, etc."""
    return PERSON_TO_TUPLE_INDEX[slot]

def short_tag(slot: str) -> str:
    """Map 'first singular' → '1sg', etc."""
    return PERSON_SHORT_TAG[slot]

def build_conjugation_tuples(all_verbs: dict, lang_iso3: str, max_verbs: Optional[int] = None) -> List[Tuple]:
    """
    Build tuples (lemma, 1sg, 2sg, 3sg, 1pl, 2pl, 3pl) for a language.
    Keep verbs even if some forms are missing; downstream steps will skip missing.
    """
    out = []
    lang_data = all_verbs.get(lang_iso3, {})
    for lemma, forms in lang_data.items():
        t = (
            lemma,
            forms.get("1st_person_singular"),
            forms.get("2nd_person_singular"),
            forms.get("3rd_person_singular"),
            forms.get("1st_person_plural"),
            forms.get("2nd_person_plural"),
            forms.get("3rd_person_plural"),
        )
        out.append(t)
        if max_verbs is not None and len(out) >= max_verbs:
            break
    return out

def build_lang_slot_set_no_filter(
    *,
    all_verbs: dict,
    lang_iso3: str,
    lang_name: str,
    person: str,
    tokenizer: AutoTokenizer,
    max_verbs: int = 1300,
) -> Tuple[List[str], List[int], List[Tuple]]:
    """
    Build prompts and (prefix, gold) for a language/slot WITHOUT the earlier tokenization filter.
    Returns:
      texts:  list[str]  (decoded prefixes, i.e., prompt minus final token)
      golds:  list[int]  (ID of final token for this slot)
      metas:  list[tuple] (MorphyNet tuple for the lemma)
    """
    conj_list = build_conjugation_tuples(all_verbs, lang_iso3, max_verbs=max_verbs)
    # Generate full prompts for requested slot
    prompts = generate_dataset(person=person, language=lang_name, conjugations=conj_list)

    texts, golds, metas = [], [], []
    for meta, prompt in zip(conj_list, prompts):
        # Skip if requested slot missing in tuple
        idx = tuple_index(person)
        conj_form = meta[idx]
        if not conj_form:
            continue
        tid = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(tid) < 1:
            continue
        prefix_ids = tid[:-1]
        gold_id    = tid[-1]
        texts.append(tokenizer.decode(prefix_ids, skip_special_tokens=True))
        golds.append(gold_id)
        metas.append(meta)
    return texts, golds, metas

def gold_last_token_id_for_tuple_entry(
    entry: Tuple,
    target_slot: str,
    tokenizer: AutoTokenizer
) -> Optional[int]:
    """
    From an entry (lemma, 1sg,...,3pl), get the *last-subtoken ID* for target_slot conjugation.
    Returns None if missing or tokenization fails.
    """
    idx = tuple_index(target_slot)
    form = entry[idx]
    if not form:
        return None
    ids = tokenizer(" " + form, add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    return ids[-1]

def intersect_topk_shared_shapes(
    groups_clean: Dict[Tuple[int,int], List[Tuple[str,int,Tuple]]],
    groups_corrupt: Dict[Tuple[int,int], List[Tuple[str,int,Tuple]]],
    topk: int = 3
) -> List[Tuple[int,int]]:
    """
    groups_* map (inf_len, conj_len) → list[(prompt, answer_id, entry)]
    Return top-k shared keys by min(lengths), descending.
    """
    shared = []
    for key in groups_clean.keys() & groups_corrupt.keys():
        n = min(len(groups_clean[key]), len(groups_corrupt[key]))
        if n > 0:
            shared.append((key, n))
    shared.sort(key=lambda x: -x[1])
    return [k for (k, _) in shared[:topk]]

def pair_batches_for_shape_key_xling(
    *,
    key: Tuple[int,int],
    clean_group: List[Tuple[str,int,Tuple]],
    corrupt_group: List[Tuple[str,int,Tuple]],
    target_slot: str,
    tokenizer: AutoTokenizer,
    max_n: int
) -> Tuple[List[str], List[str], List[int]]:
    """
    For a shared shape key, pair clean (source lang/slot) and corrupt (target lang/opposite-slot) examples.
    Build A-target gold IDs per *corrupt* example using its own tuple (same lemma), but the target_slot form.
    Returns (clean_prompts, corrupt_prompts, target_answer_ids), length m <= max_n.
    """
    m = min(len(clean_group), len(corrupt_group), max_n)
    clean_prompts, corrupt_prompts, target_ans_ids = [], [], []
    i = j = 0
    # Greedy pair up to m, skipping items with missing target-slot gold
    while len(target_ans_ids) < m and i < len(clean_group) and j < len(corrupt_group):
        clean_item  = clean_group[i]   # (prompt, gold_id, entry) but gold_id here is for the source slot, we won't use it
        corrupt_item= corrupt_group[j] # (prompt, gold_id, entry) gold_id corresponds to corrupt slot (opposite)
        i += 1
        j += 1

        tgt_id = gold_last_token_id_for_tuple_entry(corrupt_item[2], target_slot, tokenizer)
        if tgt_id is None:
            continue

        clean_prompts.append(clean_item[0])
        corrupt_prompts.append(corrupt_item[0])
        target_ans_ids.append(tgt_id)

    return clean_prompts, corrupt_prompts, target_ans_ids

def run_attn_head_out_patching_xling(
    *,
    tl_model: HookedTransformer,
    clean_prompts: List[str],
    corrupt_prompts: List[str],
    target_answer_ids: List[int],
    direction_label: str,
    save_dir: str,
    lang_tag: str,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    save_prefix: str = "xling_attn_head_out_all_pos"
):
    """
    Cross-lingual head-out patching:
      - cache from CLEAN (source),
      - patch into CORRUPT (target),
      - metric uses TARGET language *target-slot* gold token IDs.
    Saves heatmap PNG + tensor PT + meta.json in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    n = min(len(clean_prompts), len(corrupt_prompts), len(target_answer_ids))
    if n == 0:
        print(f"↪️  Skipping {direction_label} - zero aligned prompts")
        return None

    clean_prompts   = clean_prompts[:n]
    corrupt_prompts = corrupt_prompts[:n]
    ans_tensor      = torch.tensor(target_answer_ids[:n], device=device)

    # Tokenize ONCE for shared padding
    all_prompts      = clean_prompts + corrupt_prompts
    all_tok          = tl_model.to_tokens(all_prompts)
    clean_tokens     = all_tok[:n]
    corrupted_tokens = all_tok[n:]

    clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
    corrupted_logits, _       = tl_model.run_with_cache(corrupted_tokens)

    def logit_diff_on_targets(logits: torch.Tensor) -> torch.Tensor:
        last = logits[:, -1, :]
        log_probs = last.log_softmax(dim=-1)
        return log_probs.gather(1, ans_tensor.unsqueeze(1)).mean()

    # Bases (note: both measured on A-target gold IDs)
    clean_base     = logit_diff_on_targets(clean_logits).item()
    corrupted_base = logit_diff_on_targets(corrupted_logits).item()

    def metric(logits: torch.Tensor) -> torch.Tensor:
        # Normalized recovery against A-target gold IDs
        return (logit_diff_on_targets(logits) - corrupted_base) / (clean_base - corrupted_base + 1e-12)

    # Patch!
    patch = patching.get_act_patch_attn_head_out_all_pos(
        tl_model,
        corrupted_tokens,
        clean_cache,
        patching_metric=metric,
    )

    # Save artifacts
    heatmap_title = f"xling attn_head_out ({lang_tag}, {direction_label})"
    png_name = os.path.join(save_dir, f"{save_prefix}_{direction_label}_{lang_tag}.png")
    pt_name  = os.path.join(save_dir, f"{save_prefix}_patch_results_{direction_label}_{lang_tag}.pt")

    save_heatmap(
        data=patch,
        x_labels=[f"H{h}" for h in range(tl_model.cfg.n_heads)],
        y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
        title=heatmap_title,
        filename=png_name,
    )
    torch.save(patch, pt_name)
    meta = {
        "direction": direction_label,
        "lang_tag": lang_tag,
        "n_examples": n,
        "clean_base_logp_on_target_ids": clean_base,
        "corrupted_base_logp_on_target_ids": corrupted_base,
        "save_prefix": save_prefix,
        "png": os.path.basename(png_name),
        "tensor": os.path.basename(pt_name),
    }
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved: {png_name}")
    print(f"✅ Saved: {pt_name}")
    return patch

def xling_direction_runner(
    *,
    tl_model: HookedTransformer,
    tokenizer: AutoTokenizer,
    all_verbs: dict,
    # language A, B
    langA_name: str, langA_iso3: str,
    langB_name: str, langB_iso3: str,
    slot_src: str,   # X (source)
    slot_tgt: str,   # Y (target to recover in target language)
    model_name: str,
    max_verbs: int,
    max_prompts_head: int,
    topk_groups: int,
    out_root: str = "results"
):
    """
    Run one cross-lingual direction:
      target language (A) tries to recover slot_tgt using:
        - clean cache from source language (B, slot_src),
        - corrupted prompts from A (slot_src; opposite slot),
        - gold IDs from A (slot_tgt), computed per corrupt example.
    """
    device = tl_model.cfg.device if hasattr(tl_model, "cfg") else next(tl_model.parameters()).device
    tag_src = short_tag(slot_src)
    tag_tgt = short_tag(slot_tgt)

    # 1) Build per-lang/slot sets (no tokenization filter), then accuracy-filter
    print(f"\n[build] {langB_name} clean (slot={slot_src})")
    B_src_texts, B_src_golds, B_src_meta = build_lang_slot_set_no_filter(
        all_verbs=all_verbs, lang_iso3=langB_iso3, lang_name=langB_name,
        person=slot_src, tokenizer=tokenizer, max_verbs=max_verbs
    )
    print(f"[acc-filter] {langB_name} clean (slot={slot_src})")
    B_src_tok, B_src_gids, B_src_meta, *_ = accuracy_filter(
        # accuracy_filter expects token-ids prefixes: rebuild from texts
        prompts_trimmed=[tokenizer(t, add_special_tokens=False)["input_ids"] for t in B_src_texts],
        answer_ids=B_src_golds,
        matched_entries=B_src_meta,
        model_name=model_name,
        batch_size=16,
        device="cpu",
    )
    B_src_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in B_src_tok]

    print(f"\n[build] {langA_name} corrupt (slot={slot_src})")
    A_src_texts, A_src_golds, A_src_meta = build_lang_slot_set_no_filter(
        all_verbs=all_verbs, lang_iso3=langA_iso3, lang_name=langA_name,
        person=slot_src, tokenizer=tokenizer, max_verbs=max_verbs
    )
    print(f"[acc-filter] {langA_name} corrupt (slot={slot_src})")
    A_src_tok, A_src_gids, A_src_meta, *_ = accuracy_filter(
        prompts_trimmed=[tokenizer(t, add_special_tokens=False)["input_ids"] for t in A_src_texts],
        answer_ids=A_src_golds,
        matched_entries=A_src_meta,
        model_name=model_name,
        batch_size=16,
        device="cpu",
    )
    A_src_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in A_src_tok]

    # 2) Group by (inf_len, conj_len) within each language for the *source slot*
    print(f"\n[group] source slot={slot_src} (inf_len, conj_len)")
    idx_src = tuple_index(slot_src)
    groups_B_src = group_by_token_lengths(B_src_texts, B_src_gids, B_src_meta, tokenizer, idx_src)
    groups_A_src = group_by_token_lengths(A_src_texts, A_src_gids, A_src_meta, tokenizer, idx_src)

    # 3) Intersect and choose top-K shared shapes
    shared_keys = intersect_topk_shared_shapes(groups_B_src, groups_A_src, topk=topk_groups)
    if not shared_keys:
        print(f"↪️  No shared shapes between {langB_name}(src={slot_src}) and {langA_name}(src={slot_src}). Skipping.")
        return

    # 4) For each shape, pair + patch
    model_parts = model_name.split("/")
    base_dir = os.path.join(
        out_root, *model_parts, "xling", f"{langA_name}__from__{langB_name}", f"{tag_src}to{tag_tgt}"
    )
    for (inf_len, conj_len) in shared_keys:
        clean_group   = groups_B_src[(inf_len, conj_len)]
        corrupt_group = groups_A_src[(inf_len, conj_len)]

        clean_prompts, corrupt_prompts, target_ans_ids = pair_batches_for_shape_key_xling(
            key=(inf_len, conj_len),
            clean_group=clean_group,
            corrupt_group=corrupt_group,
            target_slot=slot_tgt,
            tokenizer=tokenizer,
            max_n=max_prompts_head
        )

        if len(target_ans_ids) == 0:
            print(f"↪️  {langA_name}←{langB_name} {tag_src}→{tag_tgt} | shape={(inf_len,conj_len)}: no valid pairs.")
            continue

        save_dir = os.path.join(base_dir, f"shape_inf{inf_len}_conj{conj_len}")
        label = f"{langB_name}_to_{langA_name}__{tag_src}to{tag_tgt}__inf{inf_len}_conj{conj_len}"
        print(f"[patch] {label} | n={len(target_ans_ids)}")

        run_attn_head_out_patching_xling(
            tl_model=tl_model,
            clean_prompts=clean_prompts,
            corrupt_prompts=corrupt_prompts,
            target_answer_ids=target_ans_ids,
            direction_label=label,
            save_dir=save_dir,
            lang_tag=f"{langA_name}",
            device=tl_model.cfg.device,
            save_prefix="xling_attn_head_out_all_pos"
        )

        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="HF model id (e.g., bigscience/bloomz-1b1)")
    parser.add_argument("--lang_a", required=True, choices=LANG_MAP.keys(), help="ISO3 of target language (A)")
    parser.add_argument("--lang_b", required=True, choices=LANG_MAP.keys(), help="ISO3 of source language (B)")
    parser.add_argument("--slot_x", required=True, help="e.g., 'first singular'")
    parser.add_argument("--slot_y", required=True, help="e.g., 'second singular'")
    parser.add_argument("--morphy_path", default="/home/lis.isabella.gidi/jsalt2025/src/datasets/morphy_net/MorphyNet_all_present_conjugations.json")
    parser.add_argument("--max_verbs", type=int, default=1300)
    parser.add_argument("--max_prompts_head", type=int, default=50)
    parser.add_argument("--topk_groups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load model/tokenizer
    tl_model, tokenizer = model_and_tokenizer_from_name(args.model_name, device="cuda")
    safe_pad_token(tokenizer)

    # Resolve language names/iso3
    langA_name, langA_iso3 = LANG_MAP[args.lang_a]
    langB_name, langB_iso3 = LANG_MAP[args.lang_b]
    slot_X = args.slot_x.strip().lower()
    slot_Y = args.slot_y.strip().lower()

    # Load MorphyNet once
    all_verbs = load_json_data(args.morphy_path)

    print("=== Cross-lingual Activation Patching ===")
    print(f"Model: {args.model_name}")
    print(f"A (target): {args.lang_a} -> {langA_name} | B (source): {args.lang_b} -> {langB_name}")
    print(f"Slots: X={slot_X}  Y={slot_Y}")
    print(f"Max verbs: {args.max_verbs} | Max prompts/head: {args.max_prompts_head} | Top-K groups: {args.topk_groups}")
    print("-------------------------------------------------------")

    # Four runs:
    # 1) A ← B : X → Y
    xling_direction_runner(
        tl_model=tl_model, tokenizer=tokenizer, all_verbs=all_verbs,
        langA_name=langA_name, langA_iso3=langA_iso3,  # target language
        langB_name=langB_name, langB_iso3=langB_iso3,  # source language
        slot_src=slot_X, slot_tgt=slot_Y,
        model_name=args.model_name,
        max_verbs=args.max_verbs,
        max_prompts_head=args.max_prompts_head,
        topk_groups=args.topk_groups,
    )

    # 2) A ← B : Y → X
    xling_direction_runner(
        tl_model=tl_model, tokenizer=tokenizer, all_verbs=all_verbs,
        langA_name=langA_name, langA_iso3=langA_iso3,
        langB_name=langB_name, langB_iso3=langB_iso3,
        slot_src=slot_Y, slot_tgt=slot_X,
        model_name=args.model_name,
        max_verbs=args.max_verbs,
        max_prompts_head=args.max_prompts_head,
        topk_groups=args.topk_groups,
    )

    # 3) B ← A : X → Y
    xling_direction_runner(
        tl_model=tl_model, tokenizer=tokenizer, all_verbs=all_verbs,
        langA_name=langB_name, langA_iso3=langB_iso3,  # now target=B
        langB_name=langA_name, langB_iso3=langA_iso3,  # source=A
        slot_src=slot_X, slot_tgt=slot_Y,
        model_name=args.model_name,
        max_verbs=args.max_verbs,
        max_prompts_head=args.max_prompts_head,
        topk_groups=args.topk_groups,
    )

    # 4) B ← A : Y → X
    xling_direction_runner(
        tl_model=tl_model, tokenizer=tokenizer, all_verbs=all_verbs,
        langA_name=langB_name, langA_iso3=langB_iso3,
        langB_name=langA_name, langB_iso3=langA_iso3,
        slot_src=slot_Y, slot_tgt=slot_X,
        model_name=args.model_name,
        max_verbs=args.max_verbs,
        max_prompts_head=args.max_prompts_head,
        topk_groups=args.topk_groups,
    )

    print("✅ All cross-lingual runs finished.")

if __name__ == "__main__":
    main()
