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

#THESE starter fucntions are repeated!!

def logits_to_avg_logit_diff(logits, gold_ids):
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    return probs[torch.arange(len(gold_ids)), gold_ids].mean()

def normalize_patch_diff(patched, clean, corrupted):
    return (patched - corrupted) / (clean - corrupted + 1e-12)

def patch_single_head_z(
    corrupted_z: torch.Tensor,
    hook,
    layer: int,
    head_index: int,
    clean_cache: dict,
):
    """
    Replace the corrupted z[:, :, head_index, :] with clean version.
    z shape: [batch, seq, n_heads, d_head]
    """
    corrupted_z[:, :, head_index, :] = clean_cache[f"z_{layer}"][:, :, head_index, :]
    return corrupted_z


def run_head_patching(
    model,
    tokenizer,
    clean_prompts: List[str],
    corrupted_prompts: List[str],
    gold_token_ids: List[int],
    max_seq_len: int = 32,
):
    device = next(model.parameters()).device

    # Tokenize
    clean = tokenizer(clean_prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_len).to(device)
    corrupt = tokenizer(corrupted_prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_len).to(device)

    gold = torch.tensor(gold_token_ids, device=device)

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.n_head
    d_model = model.config.hidden_size
    d_head = d_model // n_heads

    clean_cache = {}

    # 1. Capture true z = A @ V before W_O for each layer
    def make_z_hook(layer):
        def hook(module, input, output):
            hidden_states = input[0]  # [batch, seq, d_model]

            # project Q, K, V
            qkv = module.query_key_value(hidden_states)
            query, key, value = qkv.split(d_model, dim=-1)

            # reshape: [batch, seq, n_heads, d_head]
            query = query.reshape(hidden_states.shape[0], -1, n_heads, d_head).transpose(1, 2)
            key = key.reshape_as(query).transpose(1, 2)    # [batch, n_heads, seq, d_head]
            value = value.reshape_as(query).transpose(1, 2)

            # attention scores: [batch, n_heads, seq, seq]
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (d_head ** 0.5)
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

            # z = A @ V: [batch, n_heads, seq, d_head]
            z = torch.matmul(attn_weights, value)  # shape: [batch, n_heads, seq, d_head]
            z = z.transpose(1, 2)  # [batch, seq, n_heads, d_head]

            clean_cache[f"z_{layer}"] = z.detach().clone()
        return hook

    # Run clean forward pass to save z
    handles = []
    for layer in range(n_layers):
        handles.append(model.transformer.h[layer].self_attention.register_forward_hook(make_z_hook(layer)))

    with torch.no_grad():
        clean_logits = model(**clean).logits

    for h in handles:
        h.remove()

    clean_score = logits_to_avg_logit_diff(clean_logits, gold)

    with torch.no_grad():
        corrupted_logits = model(**corrupt).logits
    corrupted_score = logits_to_avg_logit_diff(corrupted_logits, gold)

    print(f"Clean P(gold)   = {clean_score:.4f}")
    print(f"Corrupt P(gold) = {corrupted_score:.4f}")

    # 2. Patch individual heads by replacing z before W_O
    recovery = torch.zeros((n_layers, n_heads), device=device)

    for layer in range(n_layers):
        for head_index in range(n_heads):

            def make_patch_hook(layer, head_index):
                def hook(module, input, output):
                    hidden_states = input[0]  # [batch, seq, d_model]

                    # project Q, K, V
                    qkv = module.query_key_value(hidden_states)
                    query, key, value = qkv.split(d_model, dim=-1)

                    query = query.reshape(hidden_states.shape[0], -1, n_heads, d_head).transpose(1, 2)
                    key = key.reshape_as(query).transpose(1, 2)
                    value = value.reshape_as(query).transpose(1, 2)

                    # attention pattern
                    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (d_head ** 0.5)
                    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

                    # z = A @ V
                    z = torch.matmul(attn_weights, value)  # [batch, n_heads, seq, d_head]
                    z = z.transpose(1, 2)  # [batch, seq, n_heads, d_head]

                    # patch one head
                    z[:, :, head_index, :] = clean_cache[f"z_{layer}"][:, :, head_index, :]

                    # reshape & reproject
                    z = z.transpose(1, 2).contiguous().view(hidden_states.shape[0], -1, d_model)
                    attn_output = module.dense(z)  # apply W_O
                    return attn_output
                return hook

            handle = model.transformer.h[layer].self_attention.register_forward_hook(make_patch_hook(layer, head_index))

            with torch.no_grad():
                patched_logits = model(**corrupt).logits
            patched_score = logits_to_avg_logit_diff(patched_logits, gold)
            recovery[layer, head_index] = normalize_patch_diff(patched_score, clean_score, corrupted_score)

            handle.remove()

    return recovery.cpu()



def plot_head_patching_heatmap(recovery, label):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        recovery.numpy(),
        xticklabels=[f"H{i}" for i in range(recovery.shape[1])],
        yticklabels=[f"L{l}" for l in range(recovery.shape[0])],
        cmap="coolwarm", center=0.0,
        vmin=0.0, vmax=1.0,
        cbar_kws={"label": "Normalized Recovery"}
    )
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"Head-Level Activation Patching — {label}")
    plt.tight_layout()
    plt.savefig(f"head_patching_{label}.png")
    print(f"✅ Saved head_patching_{label}.png")
    plt.close()


