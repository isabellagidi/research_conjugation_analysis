import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

# ── Setup ─────────────────────────────────────────────────────────────────────

model_name = "bigscience/bloom-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # BLOOM needs *some* pad token
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda").eval()
device = next(model.parameters()).device


# ── STEP 1: Helper for patching a single layer & position ────────────────────

def patch_residual_component(corrupted_resid, hook, pos, clean_cache):
    """
    hook.name will be e.g. "resid_pre_7" for layer 7;
    clean_cache[hook.name] is the clean activations for that layer.
    We overwrite only the slice at the target position.
    """
    corrupted_resid[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_resid


# ── STEP 2 & 3: Scoring & normalization ──────────────────────────────────────

def logits_to_avg_logit_diff(logits, gold_ids):
    # P(gold) at final position, averaged
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    return probs[torch.arange(len(gold_ids)), gold_ids].mean()

def normalize_patch_diff(patched, clean, corrupted):
    return (patched - corrupted) / (clean - corrupted + 1e-12)


# ── STEP 4: Activation-patching loop via PyTorch hooks ───────────────────────

def run_activation_patching(
    model,
    tokenizer,
    clean_prompts: List[str],
    corrupted_prompts: List[str],
    gold_token_ids: List[int],
    max_seq_len: int = 32,
):
    device = next(model.parameters()).device

    # --- Tokenize (same padding for both) ---
    clean = tokenizer(
        clean_prompts,
        return_tensors="pt",
        padding=True, truncation=True, max_length=max_seq_len
    ).to(device)
    corrupt = tokenizer(
        corrupted_prompts,
        return_tensors="pt",
        padding=True, truncation=True, max_length=max_seq_len
    ).to(device)
    gold = torch.tensor(gold_token_ids, device=device)

    n_layers = model.config.num_hidden_layers
    seq_len  = clean["input_ids"].size(1)

    # --- 1) Clean run: capture resid_pre activations ---
    clean_cache = {}
    handles = []
    for layer in range(n_layers):
        name = f"resid_pre_{layer}"
        def make_capture_hook(name):
            def hook(module, inp, outp):
                # inp[0] is the residual *before* block i.e. resid_pre
                clean_cache[name] = inp[0].detach().clone()
            return hook

        block = model.transformer.h[layer].self_attention
        handles.append(block.register_forward_hook(make_capture_hook(name)))

    with torch.no_grad():
        clean_logits = model(**clean).logits

    # remove capture hooks
    for h in handles: h.remove()

    # baseline scores
    clean_score     = logits_to_avg_logit_diff(clean_logits, gold)

    # --- 2) Corrupted run (unpatched) ---
    with torch.no_grad():
        corrupt_logits = model(**corrupt).logits
    corrupted_score = logits_to_avg_logit_diff(corrupt_logits, gold)
    print(f"Clean P(gold)   = {clean_score:.4f}")
    sys.stdout.flush()
    print(f"Corrupt P(gold) = {corrupted_score:.4f}")
    sys.stdout.flush()

    # --- 3) For each layer & position: patch in clean_cache and re-run corrupted ---
    recovery = torch.zeros((n_layers, seq_len), device=device)
    for layer in range(n_layers):
        name = f"resid_pre_{layer}"
        block = model.transformer.h[layer].self_attention

        for pos in range(seq_len):
            # define hook that replaces the corrupted resid_pre with the clean one at this pos
            def make_patch_hook(name, pos):
                def hook(module, inp, outp):
                    inp0 = inp[0]  # the residual stream tensor (batch, seq, d_model)
                    inp0[:, pos, :] = clean_cache[name][:, pos, :]
                    return (inp0, *outp[1:])  # repackage if necessary
                return hook

            h = block.register_forward_hook(make_patch_hook(name, pos))

            with torch.no_grad():
                patched_logits = model(**corrupt).logits

            score = logits_to_avg_logit_diff(patched_logits, gold)
            recovery[layer, pos] = normalize_patch_diff(score, clean_score, corrupted_score)
            h.remove()

    # token labels for plotting
    tokens = tokenizer.convert_ids_to_tokens(clean["input_ids"][1].tolist()) #CHANGE HERE TO CHANGE PROMPT ON AXIS
    return recovery.cpu(), tokens



# ── STEP 5: Plotting ──────────────────────────────────────────────────────────

def plot_activation_patching_heatmap(recovery, tokens, label):
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        recovery.detach().cpu().numpy(),
        xticklabels=[f"{tok}_{i}" for i,tok in enumerate(tokens)],
        yticklabels=[f"L{l}" for l in range(recovery.shape[0])],
        cmap="coolwarm", center=0.0,
        cbar_kws={"label": "Normalized Recovery"}
    )
    plt.xlabel("Token position")
    plt.ylabel("Layer")
    plt.title(f"Activation Patching ({label})")
    plt.tight_layout()
    plt.savefig(f"patching_{label}.png")
    print(f"✅ Saved patching_{label}.png")
    sys.stdout.flush()
    plt.close()
