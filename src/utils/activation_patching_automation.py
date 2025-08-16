# === Imports ===
import os
import torch
from transformer_lens import HookedTransformer
from transformer_lens import patching
from src.utils.automation import save_heatmap, get_logit_diff, conjugation_metric
from transformers import AutoTokenizer

from src.utils.automation import (
    filter_conjugations,
    build_conjugation_prompts,
    accuracy_filter,
    group_by_token_lengths,
    generate_dataset
)


def OLD_activation_patching_automation_OLD(language, conjugation_pair, all_verbs, output_root="automation_results", model_name = "bigscience/bloom-1b1", MAX_PROMPTS = 100):
    tl_model = HookedTransformer.from_pretrained(model_name)
    device = tl_model.cfg.device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lang = language.lower()
    a, b = conjugation_pair

    print(f"\n=== 🔍 Running patching for {lang.upper()} — {a} ↔ {b} ===")

    filtered = filter_conjugations(all_verbs, tokenizer, lang, conjugation_pair)
    filtered = filtered[:MAX_PROMPTS]

    prompts_a = generate_dataset(person=a, language=lang, conjugations=filtered)
    prompts_b = generate_dataset(person=b, language=lang, conjugations=filtered)

    p_a, a_ids, entries_a = build_conjugation_prompts(tokenizer, prompts_a, filtered)
    p_b, b_ids, entries_b = build_conjugation_prompts(tokenizer, prompts_b, filtered)

    correct_prompts_a, correct_ids_a, correct_entries_a, *_ = accuracy_filter(
        prompts_trimmed=p_a, 
        answer_ids=a_ids, 
        matched_entries=entries_a, 
        model_name=model_name
    )
    correct_prompts_b, correct_ids_b, correct_entries_b, *_ = accuracy_filter(
        prompts_trimmed=p_b, 
        answer_ids=b_ids, 
        matched_entries=entries_b, 
        model_name=model_name
    )

    min_len = min(len(correct_prompts_a), len(correct_prompts_b))
    correct_prompts_a = correct_prompts_a[:min_len]
    correct_ids_a = correct_ids_a[:min_len]
    correct_entries_a = correct_entries_a[:min_len]

    correct_prompts_b = correct_prompts_b[:min_len]
    correct_ids_b = correct_ids_b[:min_len]
    correct_entries_b = correct_entries_b[:min_len]

    texts_a = correct_prompts_a
    texts_b = correct_prompts_b

    subdir = f"{lang}/{a.replace(' ', '_')}_to_{b.replace(' ', '_')}"
    out_dir = os.path.join(output_root, subdir)
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    for direction, clean_texts, corrupt_texts, answer_ids, dir_tag in [
        (f"{a}_to_{b}", texts_a, texts_b, correct_ids_a, f"{a.replace(' ', '_')}_to_{b.replace(' ', '_')}"),
        (f"{b}_to_{a}", texts_b, texts_a, correct_ids_b, f"{b.replace(' ', '_')}_to_{a.replace(' ', '_')}")
    ]:
        min_len = min(len(clean_texts), len(corrupt_texts), len(answer_ids))
        clean_texts = clean_texts[:min_len]
        corrupt_texts = corrupt_texts[:min_len]
        answer_ids_tensor = torch.tensor(answer_ids[:min_len], device=device)

        clean_tokens = tl_model.to_tokens(clean_texts)
        corrupted_tokens = tl_model.to_tokens(corrupt_texts)

        print(f"Clean shape: {clean_tokens.shape}")
        print(f"Corrupt shape: {corrupted_tokens.shape}")

        clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
        corrupted_logits, _ = tl_model.run_with_cache(corrupted_tokens)

        clean_score = get_logit_diff(clean_logits, answer_ids_tensor).item()
        corrupted_score = get_logit_diff(corrupted_logits, answer_ids_tensor).item()

        def patch_metric(logits):
            return conjugation_metric(logits, answer_ids_tensor, clean_score, corrupted_score)

        attn_patch = patching.get_act_patch_attn_head_out_all_pos(
            tl_model, corrupted_tokens, clean_cache, patching_metric=patch_metric
        )

        fname = f"attn_head_out_all_pos_patch_results_{lang}_{dir_tag}.pt"
        torch.save(attn_patch, os.path.join(out_dir, fname))
        save_heatmap(
            attn_patch,
            x_labels=[f"H{h}" for h in range(tl_model.cfg.n_heads)],
            y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
            title=f"attn_head_out Activation Patching ({direction})",
            filename=os.path.join(out_dir, f"attn_patch_{lang}_{dir_tag}.png")
        )

        results[f"attn_allpos_{dir_tag}"] = attn_patch

        grouped_clean = group_by_token_lengths(clean_texts, answer_ids, correct_entries_a if "a_to" in dir_tag else correct_entries_b, tokenizer)
        grouped_corrupt = group_by_token_lengths(corrupt_texts, answer_ids, correct_entries_b if "a_to" in dir_tag else correct_entries_a, tokenizer)

        top_5 = sorted(grouped_clean.items(), key=lambda x: len(x[1]), reverse=True)[:5]

        for (inf_len, conj_len), group in top_5:
            c_prompts = [ex[0] for ex in group]
            patch_answer_ids = torch.tensor([ex[1] for ex in group], device=device)

            corrupt_group = grouped_corrupt.get((inf_len, conj_len))
            if corrupt_group is None or len(corrupt_group) < len(group):
                continue

            p_corrupt = [ex[0] for ex in corrupt_group][:len(c_prompts)]
            patch_answer_ids = patch_answer_ids[:len(p_corrupt)]
            c_prompts = c_prompts[:len(p_corrupt)]

            # Tokenize individually to avoid padding
            clean_tok_list = [tl_model.to_tokens(p, prepend_bos=False, append_eos=False)[0] for p in c_prompts]
            corrupt_tok_list = [tl_model.to_tokens(p, prepend_bos=False, append_eos=False)[0] for p in p_corrupt]

            clean_lengths = set(len(t) for t in clean_tok_list)
            corrupt_lengths = set(len(t) for t in corrupt_tok_list)

            if len(clean_lengths) > 1 or len(corrupt_lengths) > 1:
                print(f"⚠️ Skipping group (inf={inf_len}, conj={conj_len}) due to unequal token lengths.")
                continue

            clean_tokens = torch.stack(clean_tok_list).to(device)
            corrupt_tokens = torch.stack(corrupt_tok_list).to(device)

            c_logits, c_cache = tl_model.run_with_cache(clean_tokens)
            x_logits, _ = tl_model.run_with_cache(corrupt_tokens)

            clean_score = get_logit_diff(c_logits, patch_answer_ids).item()
            corrupt_score = get_logit_diff(x_logits, patch_answer_ids).item()

            def patch_metric_resid(logits):
                return conjugation_metric(logits, patch_answer_ids, clean_score, corrupt_score)

            resid_patch = patching.get_act_patch_resid_pre(
                tl_model, corrupt_tokens, c_cache, patching_metric=patch_metric_resid
            )

            tokens = tl_model.to_str_tokens(clean_tokens[0])
            tokens = [tok for tok in tokens if tok != tokenizer.pad_token]
            save_heatmap(
                resid_patch,
                x_labels=[f"{tok} {i}" for i, tok in enumerate(tokens)],
                y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
                title=f"resid_pre Activation Patching {lang} ({direction})",
                filename=os.path.join(out_dir, f"resid_pre_inf{inf_len}_conj{conj_len}_{lang}_{dir_tag}.png")
            )

            results[f"resid_pre_{dir_tag}_inf{inf_len}_conj{conj_len}"] = resid_patch

    return results


# === Imports ===
import os
import torch
from transformer_lens import HookedTransformer
from transformer_lens import patching
from src.utils.automation import save_heatmap, get_logit_diff, conjugation_metric
from transformers import AutoTokenizer

from src.utils.automation import (
    filter_conjugations,
    build_conjugation_prompts,
    accuracy_filter,
    group_by_token_lengths,
    generate_dataset
)


def activation_patching_automation(language, conjugation_pair, all_verbs, output_root="automation_results", model_name = "bigscience/bloom-1b1", MAX_PROMPTS = 100):
    """
    Run activation patching in both directions (A->B and B->A) for a given language and conjugation pair.

    Args:
        language (str): e.g., "spanish"
        conjugation_pair (tuple): e.g., ("first singular", "second singular")
        all_verbs (dict): loaded JSON MorphyNet data
        output_root (str): where to save results

    Returns:
        dict with keys:
            "resid_pre_a_to_b", "resid_pre_b_to_a",
            "attn_allpos_a_to_b", "attn_allpos_b_to_a"
    """
    tl_model = HookedTransformer.from_pretrained(model_name)
    device = tl_model.cfg.device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lang = language.lower()
    a, b = conjugation_pair

    print(f"\n=== 🔍 Running patching for {lang.upper()} — {a} ↔ {b} ===")

    # 1. Filter dataset
    filtered = filter_conjugations(all_verbs, tokenizer, lang, conjugation_pair)
    filtered = filtered[:MAX_PROMPTS]  # RAM savings

    # 2. Generate datasets in both directions
    prompts_a = generate_dataset(person=a, language=lang, conjugations=filtered)
    prompts_b = generate_dataset(person=b, language=lang, conjugations=filtered)

    # 3. Build tokens
    p_a, a_ids, entries_a = build_conjugation_prompts(tokenizer, prompts_a, filtered)
    p_b, b_ids, entries_b = build_conjugation_prompts(tokenizer, prompts_b, filtered)

    # Apply accuracy filter to get correctly predicted prompts
    correct_prompts_a, correct_ids_a, correct_entries_a, *_ = accuracy_filter(
        prompts_trimmed=p_a, 
        answer_ids=a_ids, 
        matched_entries=entries_a, 
        model_name=model_name
    )
    correct_prompts_b, correct_ids_b, correct_entries_b, *_ = accuracy_filter(
        prompts_trimmed=p_b, 
        answer_ids=b_ids, 
        matched_entries=entries_b, 
        model_name=model_name
    )

        # Align both sets to the same number of examples to prevent shape mismatch
    min_len = min(len(correct_prompts_a), len(correct_prompts_b))
    correct_prompts_a = correct_prompts_a[:min_len]
    correct_ids_a = correct_ids_a[:min_len]
    correct_entries_a = correct_entries_a[:min_len]

    correct_prompts_b = correct_prompts_b[:min_len]
    correct_ids_b = correct_ids_b[:min_len]
    correct_entries_b = correct_entries_b[:min_len]


    # Decode to strings for grouping and visualization
    texts_a = correct_prompts_a
    texts_b = correct_prompts_b


    # 4. Output folder
    subdir = f"{lang}/{a.replace(' ', '_')}_to_{b.replace(' ', '_')}"
    out_dir = os.path.join(output_root, subdir)
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    for direction, clean_texts, corrupt_texts, answer_ids, dir_tag in [
        (f"{a}_to_{b}", texts_a, texts_b, correct_ids_a, f"{a.replace(' ', '_')}_to_{b.replace(' ', '_')}"),
        (f"{b}_to_{a}", texts_b, texts_a, correct_ids_b, f"{b.replace(' ', '_')}_to_{a.replace(' ', '_')}")
    ]:
        min_len = min(len(clean_texts), len(corrupt_texts), len(answer_ids))
        clean_texts = clean_texts[:min_len]
        corrupt_texts = corrupt_texts[:min_len]
        answer_ids_tensor = torch.tensor(answer_ids[:min_len], device=device)

        if "a_to_b" in dir_tag:
            ids_clean = correct_prompts_a[:min_len]
            ids_corr  = correct_prompts_b[:min_len]
        else:
            ids_clean = correct_prompts_b[:min_len]
            ids_corr  = correct_prompts_a[:min_len]

        # now hand those straight to to_tokens:
        clean_tokens     = tl_model.to_tokens(ids_clean)
        corrupted_tokens = tl_model.to_tokens(ids_corr)

        print(f"Clean shape: {tl_model.to_tokens(clean_texts).shape}")
        print(f"Corrupt shape: {tl_model.to_tokens(corrupt_texts).shape}")

        clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
        corrupted_logits, _ = tl_model.run_with_cache(corrupted_tokens)

        clean_score = get_logit_diff(clean_logits, answer_ids_tensor).item()
        corrupted_score = get_logit_diff(corrupted_logits, answer_ids_tensor).item()

        def patch_metric(logits):
            return conjugation_metric(logits, answer_ids_tensor, clean_score, corrupted_score)

        # === attn_head_out_all_pos ===
        attn_patch = patching.get_act_patch_attn_head_out_all_pos(
            tl_model, corrupted_tokens, clean_cache, patching_metric=patch_metric
        )

        fname = f"attn_head_out_all_pos_patch_results_{lang}_{dir_tag}.pt"
        torch.save(attn_patch, os.path.join(out_dir, fname))
        save_heatmap(
            attn_patch,
            x_labels=[f"H{h}" for h in range(tl_model.cfg.n_heads)],
            y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
            title=f"attn_head_out Activation Patching ({direction})",
            filename=os.path.join(out_dir, f"attn_patch_{lang}_{dir_tag}.png")
        )

        results[f"attn_allpos_{dir_tag}"] = attn_patch

        # === resid_pre ===
        grouped_clean = group_by_token_lengths(clean_texts, answer_ids, correct_entries_a if "a_to" in dir_tag else correct_entries_b, tokenizer)
        grouped_corrupt = group_by_token_lengths(corrupt_texts, answer_ids, correct_entries_b if "a_to" in dir_tag else correct_entries_a, tokenizer)

        top_5 = sorted(grouped_clean.items(), key=lambda x: len(x[1]), reverse=True)[:5]

        for (inf_len, conj_len), group in top_5:
            c_prompts = [ex[0] for ex in group]
            patch_answer_ids = torch.tensor([ex[1] for ex in group], device=device)

            corrupt_group = grouped_corrupt.get((inf_len, conj_len))
            if corrupt_group is None or len(corrupt_group) < len(group):
                continue

            p_corrupt = [ex[0] for ex in corrupt_group][:len(c_prompts)]
            patch_answer_ids = patch_answer_ids[:len(p_corrupt)]
            c_prompts = c_prompts[:len(p_corrupt)]

            clean_toks = tl_model.to_tokens(c_prompts)
            corrupt_toks = tl_model.to_tokens(p_corrupt)

            c_logits, c_cache = tl_model.run_with_cache(clean_toks)
            x_logits, _ = tl_model.run_with_cache(corrupt_toks)

            clean_score = get_logit_diff(c_logits, patch_answer_ids).item()
            corrupt_score = get_logit_diff(x_logits, patch_answer_ids).item()

            def patch_metric_resid(logits):
                return conjugation_metric(logits, patch_answer_ids, clean_score, corrupt_score)

            resid_patch = patching.get_act_patch_resid_pre(
                tl_model, corrupt_toks, c_cache, patching_metric=patch_metric_resid
            )

            tokens = tl_model.to_str_tokens(clean_toks[0])
            save_heatmap(
                resid_patch,
                x_labels=[f"{tok} {i}" for i, tok in enumerate(tokens)],
                y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
                title=f"resid_pre Activation Patching {lang} ({direction})",
                filename=os.path.join(out_dir, f"resid_pre_inf{inf_len}_conj{conj_len}_{lang}_{dir_tag}.png")
            )

            results[f"resid_pre_{dir_tag}_inf{inf_len}_conj{conj_len}"] = resid_patch

    return results
