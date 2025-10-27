def run_attn_head_out_patching(
    tl_model,
    clean_prompts: list[str],
    corrupted_prompts: list[str],
    clean_answer_ids: list[int],
    direction_label: str,
    lang_tag: str = "spanish",
    save_prefix: str = "attn_head_out_all_pos",
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Run TransformerLens get_act_patch_attn_head_out_all_pos for one direction
    (e.g. second→first or first→second) and save the heat-map + .pt tensor.

    Parameters
    ----------
    tl_model : HookedTransformer
    clean_prompts / corrupted_prompts : list[str]
        Plain-text prompts in the “clean” and “corrupted” roles.
    answer_ids : list[int]
        Gold token IDs that correspond to clean_prompts.
    direction_label : str
        Suffix for filenames, e.g. "second2first".
    lang_tag : str
        Only used in plot / file names ("spanish" in your case).
    save_prefix : str
        Prefix for plot / tensor filenames.
    device : torch.device | str
    """
    n = min(len(clean_prompts), len(corrupted_prompts), len(clean_answer_ids))
    if n == 0:
        print(f"↪️  Skipping {direction_label} - zero aligned prompts")
        return None

    clean_prompts     = clean_prompts[:n]
    corrupted_prompts = corrupted_prompts[:n]
    answer_tensor     = torch.tensor(clean_answer_ids[:n], device=device)

    # --- forward passes -------------------------------------------------
    all_prompts      = clean_prompts + corrupted_prompts
    all_tok          = tl_model.to_tokens(all_prompts)   # single call = common pad length
    clean_tokens     = all_tok[:n]
    corrupted_tokens = all_tok[n:]

    clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
    corrupted_logits, _       = tl_model.run_with_cache(corrupted_tokens)

    def logit_diff(logits):
        logits = logits[:, -1, :]
        return logits.gather(1, answer_tensor.unsqueeze(1)).mean()

    clean_base     = logit_diff(clean_logits).item()
    corrupted_base = logit_diff(corrupted_logits).item()

    def conjugation_metric(logits):
        return (logit_diff(logits) - corrupted_base) / (clean_base - corrupted_base + 1e-12)

    # --- patching -------------------------------------------------------
    patch = patching.get_act_patch_attn_head_out_all_pos(
        tl_model,
        corrupted_tokens,
        clean_cache,
        patching_metric=conjugation_metric,
    )

    # --- save results ---------------------------------------------------
    heatmap_title = f"attn_head_out Activation Patching ({lang_tag}, {direction_label})"
    png_name = f"{save_prefix}_{direction_label}_{lang_tag}.png"
    pt_name  = f"{save_prefix}_patch_results_{direction_label}_{lang_tag}.pt"

    save_heatmap(
        data=patch,
        x_labels=[f"H{h}" for h in range(tl_model.cfg.n_heads)],
        y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
        title=heatmap_title,
        filename=png_name,
    )
    torch.save(patch, pt_name)
    print(f"✅ Saved tensor: {pt_name}")
    return patch
