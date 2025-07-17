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

def compute_logit_lens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    gold_token_ids: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Apply the logit lens to BLOOM: project residual streams after each layer to logits
    and extract the logit for the correct answer token at each layer.
    """
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    activation_cache = []

    # Define hook
    def get_hook(idx):
        def hook_fn(module, input, output):
            activation_cache.append(output[0].detach() if isinstance(output, tuple) else output.detach())
        return hook_fn

    # Register hooks at all layers
    handles = []
    for i in range(n_layers):
        handles.append(model.transformer.h[i].register_forward_hook(get_hook(i)))

    # Run forward pass
    with torch.no_grad():
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**encoded)
        final_logits = outputs.logits

    # Remove hooks
    for h in handles:
        h.remove()

    # Apply final LayerNorm to each cached residual
    ln_f = model.transformer.ln_f
    normed_resids = [ln_f(resid[:, -1, :]) for resid in activation_cache]  # (batch, d_model)

    # Project to logits
    W_U = model.lm_head.weight  # (vocab, d_model)
    logits_per_layer = [resid @ W_U.T for resid in normed_resids]  # list of (batch, vocab)

    # Convert gold token IDs to tensor
    gold_ids = torch.tensor(gold_token_ids).to(device)

    # Extract logits of gold token at each layer
    gold_logits = torch.stack([
        logits.gather(dim=-1, index=gold_ids.unsqueeze(-1)).squeeze(-1) for logits in logits_per_layer
    ])  # shape: (n_layers, batch)

    avg_logit = gold_logits.mean(dim=1)  # (n_layers,)

    return {
        "gold_logits": gold_logits,       # per layer, per example
        "avg_per_layer": avg_logit,       # per layer
        "labels": [f"Layer {i}" for i in range(n_layers)]
    }


def plot_logit_lens(logits_tensor, labels, title, color_prefix):
    """
    logits_tensor: shape (n_layers, batch)
    labels: list of layer labels
    color_prefix: string ("yo" or "tu")
    """
    num_examples = logits_tensor.shape[1]

    plt.figure(figsize=(10, 6))
    for i in range(num_examples):
        values = logits_tensor[:, i].detach().cpu().numpy()
        plt.plot(range(len(labels)), values, label=f"{color_prefix.upper()} example {i+1}")

    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.xlabel("Layer")
    plt.ylabel("Logit for Gold Token")
    plt.title(f"Logit Lens View — {color_prefix.upper()} examples")
    plt.legend()
    plt.tight_layout()
    
    # Save plot to file instead of showing it
    output_path = f"logit_lens_{color_prefix}.png"
    plt.savefig(output_path)
    print(f"✅ Saved plot to {output_path}")
    sys.stdout.flush()
    plt.close()


def plot_logit_lens_heatmap(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    save_path: str = "logit_lens_heatmap.png",
):
    """
    Plots a heatmap of top-1 token predictions at each layer and position for a single prompt.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    n_layers = model.config.num_hidden_layers
    seq_len = input_ids.shape[1]

    activation_cache = []

    def get_hook(idx):
        def hook_fn(module, input, output):
            activation_cache.append(output[0].detach() if isinstance(output, tuple) else output.detach())
        return hook_fn

    handles = []
    for i in range(n_layers):
        handles.append(model.transformer.h[i].register_forward_hook(get_hook(i)))

    with torch.no_grad():
        _ = model(input_ids)

    for h in handles:
        h.remove()

    ln_f = model.transformer.ln_f
    W_U = model.lm_head.weight

    token_ids = input_ids[0].tolist()
    token_labels = tokenizer.convert_ids_to_tokens(token_ids)

    top_token_ids = []
    top_logits = []
    top_tokens = []

    for resid in activation_cache:
        normed_resid = ln_f(resid)  # (1, seq, d_model)
        logits = normed_resid @ W_U.T  # (1, seq, vocab)
        logits = logits[0]  # (seq, vocab)
        max_vals, max_ids = torch.max(logits, dim=-1)
        top_token_ids.append(max_ids.detach().cpu().numpy())
        top_logits.append(max_vals.detach().cpu().numpy())
        top_tokens.append(tokenizer.convert_ids_to_tokens(max_ids))

    # Convert to arrays for heatmap
    logit_array = np.array(top_logits)  # shape (layers, seq)
    token_array = np.array(top_tokens)  # shape (layers, seq), strings

    fig, ax = plt.subplots(figsize=(1.2 * seq_len, 0.4 * n_layers))
    cmap = sns.color_palette("viridis", as_cmap=True)
    sns.heatmap(
        logit_array,
        xticklabels=token_labels,
        yticklabels=[f"h{2*i}_out" for i in range(n_layers)],
        cmap=cmap,
        cbar_kws={"label": "Logit"},
        annot=token_array,
        fmt="",
        ax=ax
    )
    ax.set_xlabel("Input Token")
    ax.set_ylabel("Layer Output")
    ax.set_title("Model's Top Token Prediction and Logit at Each Layer")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved heatmap to {save_path}")
    plt.close()