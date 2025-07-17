import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model_name = "bigscience/bloom-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # required for BLOOM
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def compute_headwise_gold_logit_contributions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    gold_token_id: int,
    save_path: str = "headwise_gold_logit.png"
):
    """
    Compute and visualize the contribution of each attention head to the gold token logit.
    Output is a (num_layers, num_heads) heatmap.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    pos = input_ids.shape[1] - 1  # final token position
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.n_head
    d_model = model.config.hidden_size
    head_dim = d_model // n_heads

    # Storage for per-head outputs
    head_outputs = torch.zeros((n_layers, n_heads, head_dim), device=device)

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            attn_output = output[0]  # (batch, seq, d_model)
            attn_output = attn_output[0, pos, :]  # (d_model,)
            # Reshape into per-head outputs
            heads = attn_output.view(n_heads, head_dim)
            head_outputs[layer_idx] = heads.detach()
        return hook_fn

    # Register hooks for all layers
    handles = []
    for layer_idx in range(n_layers):
        handles.append(model.transformer.h[layer_idx].self_attention.register_forward_hook(make_hook(layer_idx)))

    # Run forward pass
    with torch.no_grad():
        _ = model(input_ids)

    # Remove hooks
    for h in handles:
        h.remove()

    # Project each head's output to the gold token logit
    W_U = model.lm_head.weight  # (vocab, d_model)
    gold_embed = W_U[gold_token_id]  # (d_model,)
    # Expand for broadcasting
    gold_embed_heads = gold_embed.view(n_heads, head_dim)

    # Compute contributions: (layer, head) = dot(head_output, gold_embed part)
    logits_by_head = torch.einsum("lhd,hd->lh", head_outputs, gold_embed_heads)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        logits_by_head.detach().cpu().numpy(),
        xticklabels=[f"H{i}" for i in range(n_heads)],
        yticklabels=[f"L{i}" for i in range(n_layers)],
        cmap="coolwarm",
        cbar_kws={"label": "Gold Token Logit Contribution"},
        annot=True,
        fmt=".1f"
    )
    plt.title(f"Headwise Gold Token Logit Attribution\nPrompt: {prompt}")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved headwise attribution heatmap to {save_path}")
    plt.close()


def compute_average_headwise_gold_logit_contributions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    gold_token_ids: List[int],
    save_path: str = "average_headwise_gold_logit.png",
    label: str = "YO",
    top_n: int = 3
):
    """
    Compute and visualize the average contribution of each attention head to the gold token logit
    across a list of prompts. Output is a (num_layers, num_heads) heatmap.
    Also prints the average contribution per head.

    Returns:
        avg_logits_by_head: ndarray of shape (n_layers, n_heads)
        top_heads: list of (layer, head_index, value)
    """
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.n_head
    d_model = model.config.hidden_size
    head_dim = d_model // n_heads

    all_head_logits = []

    for prompt, gold_token_id in zip(prompts, gold_token_ids):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        pos = input_ids.shape[1] - 1  # final token position

        head_outputs = torch.zeros((n_layers, n_heads, head_dim), device=device)

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                attn_output = output[0]  # (batch, seq, d_model)
                attn_output = attn_output[0, pos, :]  # (d_model,)
                heads = attn_output.view(n_heads, head_dim)
                head_outputs[layer_idx] = heads.detach()
            return hook_fn

        handles = []
        for layer_idx in range(n_layers):
            handles.append(model.transformer.h[layer_idx].self_attention.register_forward_hook(make_hook(layer_idx)))

        with torch.no_grad():
            _ = model(input_ids)

        for h in handles:
            h.remove()

        W_U = model.lm_head.weight  # (vocab, d_model)
        gold_embed = W_U[gold_token_id]  # (d_model,)
        gold_embed_heads = gold_embed.view(n_heads, head_dim)

        logits_by_head = torch.einsum("lhd,hd->lh", head_outputs, gold_embed_heads)
        all_head_logits.append(logits_by_head.detach().cpu().numpy())

    # Average over all prompts
    avg_logits_by_head = np.mean(np.stack(all_head_logits), axis=0)

    # Print average contribution per head
    avg_contributions = avg_logits_by_head.mean(axis=0)
    print(f"\n📊 Average Contribution per Head (across prompts and layers) — {label}:")
    for i, val in enumerate(avg_contributions.tolist()):
        print(f"Head H{i}: {val:.3f}")

    # Find top N heads by absolute value
    flat_vals = avg_logits_by_head.flatten()
    top_indices = np.argsort(np.abs(flat_vals))[-top_n:][::-1]
    top_heads = [(i // n_heads, i % n_heads, flat_vals[i]) for i in top_indices]
    print(f"\n🔥 Top {top_n} most involved heads:")
    for layer, head, val in top_heads:
        print(f"L{layer}H{head} → {val:.3f}")

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        avg_logits_by_head,
        xticklabels=[f"H{i}" for i in range(n_heads)],
        yticklabels=[f"L{i}" for i in range(n_layers)],
        cmap="coolwarm",
        cbar_kws={"label": "Average Gold Token Logit Contribution"},
        annot=True,
        fmt=".1f"
    )
    plt.title(f"Average Headwise Gold Token Logit Attribution ({label} Dataset)")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved average headwise attribution heatmap to {save_path}")
    plt.close()

    return avg_logits_by_head, top_heads

