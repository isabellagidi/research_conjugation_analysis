import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io
import numpy as np

# ── Setup ─────────────────────────────────────────────────────────────────────

model_name = "bigscience/bloom-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # BLOOM needs *some* pad token
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda").eval()
device = next(model.parameters()).device

#helper functions
def logits_to_avg_logit_diff(logits, gold_ids):
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    return probs[torch.arange(len(gold_ids)), gold_ids].mean()

def normalize_patch_diff(patched, clean, corrupted):
    return (patched - corrupted) / (clean - corrupted + 1e-12)

def run_mlp_attention_patching(
    model,
    tokenizer,
    clean_prompts: List[str],
    corrupted_prompts: List[str],
    gold_token_ids: List[int],
    max_seq_len: int = 32,
):
    device = next(model.parameters()).device

    # Tokenize inputs
    clean = tokenizer(clean_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(device)
    corrupt = tokenizer(corrupted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(device)
    gold = torch.tensor(gold_token_ids, device=device)

    n_layers = model.config.num_hidden_layers
    seq_len = clean["input_ids"].shape[1]

    attn_cache = {}
    mlp_cache = {}

    def make_hook(cache, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            cache[name] = output.detach().clone()
        return hook

    # Run clean forward pass with hooks to capture clean attention and MLP outputs
    handles = []
    for l in range(n_layers):
        handles.append(model.transformer.h[l].self_attention.register_forward_hook(make_hook(attn_cache, f"attn_{l}")))
        handles.append(model.transformer.h[l].mlp.register_forward_hook(make_hook(mlp_cache, f"mlp_{l}")))

    with torch.no_grad():
        clean_logits = model(**clean).logits

    for h in handles:
        h.remove()

    torch.cuda.empty_cache()

    clean_score = logits_to_avg_logit_diff(clean_logits, gold)

    with torch.no_grad():
        corrupted_logits = model(**corrupt).logits
    corrupted_score = logits_to_avg_logit_diff(corrupted_logits, gold)

    attn_rec = torch.zeros((n_layers, seq_len), device=device)
    mlp_rec = torch.zeros((n_layers, seq_len), device=device)

    # Patch attention and MLP outputs independently
    for layer in range(n_layers):
        for pos in range(seq_len):
            # Attention patching
            def patch_attn(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    out[:, pos, :] = attn_cache[f"attn_{layer}"][:, pos, :]
                    return (out, *output[1:])
                else:
                    out = output.clone()
                    out[:, pos, :] = attn_cache[f"attn_{layer}"][:, pos, :]
                    return out

            attn_hook = model.transformer.h[layer].self_attention.register_forward_hook(patch_attn)
            with torch.no_grad():
                patched_logits = model(**corrupt).logits
            attn_score = logits_to_avg_logit_diff(patched_logits, gold)
            attn_rec[layer, pos] = normalize_patch_diff(attn_score, clean_score, corrupted_score)
            attn_hook.remove()

            # MLP patching
            def patch_mlp(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    out[:, pos, :] = mlp_cache[f"mlp_{layer}"][:, pos, :]
                    return (out, *output[1:])
                else:
                    out = output.clone()
                    out[:, pos, :] = mlp_cache[f"mlp_{layer}"][:, pos, :]
                    return out

            mlp_hook = model.transformer.h[layer].mlp.register_forward_hook(patch_mlp)
            with torch.no_grad():
                patched_logits = model(**corrupt).logits
            mlp_score = logits_to_avg_logit_diff(patched_logits, gold)
            mlp_rec[layer, pos] = normalize_patch_diff(mlp_score, clean_score, corrupted_score)
            mlp_hook.remove()

        torch.cuda.empty_cache()

    tokens = tokenizer.convert_ids_to_tokens(clean["input_ids"][0])
    return attn_rec.cpu(), mlp_rec.cpu(), tokens

def plot_layerwise_activation_patching_heatmap(recovery, tokens, label, save=True, top_k=5):
    """
    Plot a heatmap of layer by token position recovery values and print top-k patches.

    Args:
        recovery: Tensor of shape (n_layers, seq_len) with normalized recovery scores.
        tokens: List of tokens (length = seq_len) for x-axis labeling.
        label: String to label the plot (used in title and filename).
        save: If True, saves the plot to a PNG file.
        top_k: Number of top patches to print.
    """
    recovery_np = recovery.detach().cpu().numpy()
    n_layers, seq_len = recovery_np.shape

    # Print top-k (layer, position) patches by recovery value
    flat_indices = np.argpartition(recovery_np.flatten(), -top_k)[-top_k:]
    top_coords = [np.unravel_index(idx, recovery_np.shape) for idx in flat_indices]
    top_coords_sorted = sorted(top_coords, key=lambda x: recovery_np[x], reverse=True)

    print(f"\n🔝 Top {top_k} patches for {label}:")
    for layer, pos in top_coords_sorted:
        print(f"  Layer {layer}, Pos {pos} ({tokens[pos]!r}): Recovery = {recovery_np[layer, pos]:.4f}")

    # Plot heatmap
    plt.figure(figsize=(max(12, len(tokens) * 0.6), 6))
    sns.heatmap(
        recovery_np,
        xticklabels=[f"{tok}_{i}" for i, tok in enumerate(tokens)],
        yticklabels=[f"L{l}" for l in range(n_layers)],
        cmap="coolwarm", center=0.0,
        vmin=0.0, vmax=1.0,
        cbar_kws={"label": "Normalized Recovery"}
    )
    plt.xlabel("Token position (with token)")
    plt.ylabel("Transformer Layer")
    plt.title(f"Activation Patching — {label}")
    plt.tight_layout()
    
    if save:
        filename = f"patching_{label}.png"
        plt.savefig(filename)
        print(f"✅ Saved {filename}")
    else:
        plt.show()

    plt.close()