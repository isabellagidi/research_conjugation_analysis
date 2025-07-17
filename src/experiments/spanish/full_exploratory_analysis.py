import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- Setup ---
sys.path.append('../../')
from src.datasets.spanish.spanish_verbs import spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs
from jsalt2025.src.utils.spanish_dataset_generation import create_spanish_verbs, filter_spanish_conjugations

# Load tokenizer and model
model_name = "bigscience/bloom-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # required for BLOOM
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Build dataset
spanish_verbs = create_spanish_verbs(spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs)
spanish_conjugations = filter_spanish_conjugations(spanish_verbs, tokenizer)

# Prompt generators
def generate_yo_dataset(data): return [f"Conjugación del verbo {verb} en presente: Yo {yo_form}" for verb, yo_form, *_ in data]
def generate_tu_dataset(data): return [f"Conjugación del verbo {verb} en presente: Tú {tu_form}" for verb, yo_form, tu_form, *_ in data]

# Add your other generators here (b–j) if needed

# Prompt sets
spanish_yo = generate_yo_dataset(spanish_conjugations)
spanish_tu = generate_tu_dataset(spanish_conjugations)


#Build Conjugation Prompts
def build_conjugation_prompts(tokenizer, spanish_conjugations):
    prompts_trimmed = []
    answers_ids = []
    valid_indices = []

    for idx, prompt in enumerate(spanish_conjugations):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) < 2:
            continue

        prompt_trimmed = token_ids[:-1]
        answer_id = token_ids[-1]

        prompts_trimmed.append(prompt_trimmed)
        answers_ids.append(answer_id)
        valid_indices.append(idx)

    return prompts_trimmed, answers_ids, valid_indices


# Build prompts and answer token IDs
prompts_trimmed_yo, answers_ids_yo, _ = build_conjugation_prompts(tokenizer, spanish_yo)
prompts_trimmed_tu, answers_ids_tu, _ = build_conjugation_prompts(tokenizer, spanish_tu)

# Decode prompts back to full text for BLOOM
prompt_texts_yo = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_yo]
prompt_texts_tu = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_tu]




# --- Evaluation Function ---
def evaluate_prompts(text_prompts, answer_ids, label="Dataset"):
    encoded = tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    gold_ids = torch.tensor(answer_ids).to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    final_logits = logits[:, -1, :]  # next-token prediction

    probs = torch.softmax(final_logits, dim=-1)
    gold_probs = probs.gather(dim=-1, index=gold_ids.unsqueeze(-1)).squeeze(-1)

    top1_preds = torch.argmax(final_logits, dim=-1)
    top1_correct = (top1_preds == gold_ids).int()
    top1_acc = top1_correct.float().mean().item()

    sorted_logits, sorted_indices = torch.sort(final_logits, descending=True)
    gold_ranks = (sorted_indices == gold_ids.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1

    return {
        "label": label,
        "accuracy": top1_acc,
        "avg_gold_prob": gold_probs.mean().item(),
        "avg_gold_rank": gold_ranks.float().mean().item(),
        "num_total": len(answer_ids),
        "num_correct": int(top1_correct.sum().item())
    }

print("Evaluation complete — printing results next...")
sys.stdout.flush()

# --- Run Evaluation ---
results_yo = evaluate_prompts(prompt_texts_yo, answers_ids_yo, label="Format YO")
results_tu = evaluate_prompts(prompt_texts_tu, answers_ids_tu, label="Format TU")

print("📎 results_yo =", results_yo)
sys.stdout.flush()

# --- Print Summary ---
print(f"\n📊 Evaluation results for {results_yo['label']}")
print(f"Top-1 Accuracy: {results_yo['accuracy']:.2%} ({results_yo['num_correct']}/{results_yo['num_total']})")
print(f"Average P(gold): {results_yo['avg_gold_prob']:.2%}")
print(f"Average Rank of Gold Token: {results_yo['avg_gold_rank']:.2f}")
sys.stdout.flush()

print(f"\n📊 Evaluation results for {results_tu['label']}")
print(f"Top-1 Accuracy: {results_tu['accuracy']:.2%} ({results_tu['num_correct']}/{results_tu['num_total']})")
print(f"Average P(gold): {results_tu['avg_gold_prob']:.2%}")
print(f"Average Rank of Gold Token: {results_tu['avg_gold_rank']:.2f}")
sys.stdout.flush()




# --- Direct Logit Attribution (DLA) for BLOOM via hooks ---

def compute_direct_logit_attribution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_texts: List[str],
    answer_ids: List[int],
    label: str = "Dataset"
) -> Dict:
    """Compute Direct Logit Attribution (DLA) for a list of prompts and gold token IDs."""
    
    model.eval()
    device = next(model.parameters()).device

    activation_cache = {}

    # Hook function to capture residual stream before unembedding
    def save_residual_stream_hook(module, input, output):
        activation_cache["final_resid"] = output[0].detach() if isinstance(output, tuple) else output.detach()

    # Register hook at final layer norm (BLOOM: .transformer.ln_f)
    hook_handle = model.transformer.ln_f.register_forward_hook(save_residual_stream_hook)

    # Run forward pass with hook
    with torch.no_grad():
        encoded = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        output = model(**encoded)

    # Remove hook
    hook_handle.remove()

    # Prepare inputs
    gold_ids = torch.tensor(answer_ids).to(device)
    residual_stream = activation_cache["final_resid"]       # (batch, seq, d_model)
    final_resid = residual_stream[:, -1, :]                 # (batch, d_model)

    # Apply LayerNorm scaling using model weights
    ln_weight = model.transformer.ln_f.weight.data
    normed_resid = torch.nn.functional.layer_norm(
        final_resid, final_resid.shape[-1:], weight=ln_weight
    )

    # Get unembedding vectors for gold tokens
    W_U = model.lm_head.weight.data                         # (vocab, d_model)
    W_U_gold = W_U[gold_ids]                                # (batch, d_model)

    # Compute dot product = direct logit contribution
    direct_logits = torch.einsum("bd,bd->b", normed_resid, W_U_gold)  # (batch,)

    return {
        "label": label,
        "dla_values": direct_logits.tolist(),
        "avg_dla": direct_logits.mean().item()
    }


results_dla_yo = compute_direct_logit_attribution(
    model=model,
    tokenizer=tokenizer,
    prompt_texts=prompt_texts_yo,     # decoded prompts
    answer_ids=answers_ids_yo,        # gold token IDs
    label="Format YO"
)

# Print summary
print(f"\n🔍 Direct Logit Attribution for {results_dla_yo['label']}")
sys.stdout.flush()
print(f"Per-prompt DLA values (first 10): {results_dla_yo['dla_values'][:10]}")
sys.stdout.flush()
print(f"Average DLA: {results_dla_yo['avg_dla']:.3f}")
sys.stdout.flush()


results_dla_tu = compute_direct_logit_attribution(
    model=model,
    tokenizer=tokenizer,
    prompt_texts=prompt_texts_tu,     # decoded prompts
    answer_ids=answers_ids_tu,        # gold token IDs
    label="Format TU"
)

# Print summary
print(f"\n🔍 Direct Logit Attribution for {results_dla_tu['label']}")
sys.stdout.flush()
print(f"Per-prompt DLA values (first 10): {results_dla_tu['dla_values'][:10]}")
sys.stdout.flush()
print(f"Average DLA: {results_dla_tu['avg_dla']:.3f}")
sys.stdout.flush()


import torch
import numpy as np

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

logit_lens_results_yo = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_yo,
    gold_token_ids=answers_ids_yo
)

print("\n 📈 Logit Lens (YO):")
for label, logit in zip(logit_lens_results_yo["labels"], logit_lens_results_yo["avg_per_layer"]):
    print(f"{label}: {logit.item():.3f}")
sys.stdout.flush()


logit_lens_results_tu = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_tu,
    gold_token_ids=answers_ids_tu
)

print("\n 📈 Logit Lens (TU):")
for label, logit in zip(logit_lens_results_tu["labels"], logit_lens_results_tu["avg_per_layer"]):
    print(f"{label}: {logit.item():.3f}")
sys.stdout.flush()


#PLOT LOGIT LENS
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


logit_lens_results_yo_examples = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_yo[:3],
    gold_token_ids=answers_ids_yo[:3]
)

logit_lens_results_tu_examples = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_tu[:3],
    gold_token_ids=answers_ids_tu[:3]
)

# Plot both YO and TU
#plot_logit_lens(logits_tensor=logit_lens_results_yo_examples["gold_logits"], labels=logit_lens_results_yo_examples["labels"], title="YO Logit Lens", color_prefix="yo")

#plot_logit_lens(logits_tensor=logit_lens_results_tu_examples["gold_logits"], labels=logit_lens_results_tu_examples["labels"], title="TU Logit Lens", color_prefix="tu")

#LOGIT LENS heatmap

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

# Example usage:
#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_yo[4], save_path="logit_lens_heatmap_yo4.png")
#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_tu[4], save_path="logit_lens_heatmap_tu4.png")

#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_yo[5], save_path="logit_lens_heatmap_yo5.png")
#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_tu[5], save_path="logit_lens_heatmap_tu5.png")

#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_yo[6], save_path="logit_lens_heatmap_yo6.png")
#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_tu[6], save_path="logit_lens_heatmap_tu6.png")


#LAYERWISE LOGIT ATTRIBUTION
def compute_layerwise_gold_logit(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    gold_token_id: int,
    save_path: str = "layerwise_gold_logit.png"
):
    """
    Compute and plot the contribution of each layer to the logit for the gold token.
    This is similar to logit lens, but shows only the gold token's score at the final position
    after each layer's residual stream.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    pos = input_ids.shape[1] - 1  # final position

    n_layers = model.config.num_hidden_layers
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

    gold_logits = []
    for resid in activation_cache:
        normed_resid = ln_f(resid)  # (1, seq, d_model)
        final_vector = normed_resid[0, pos, :]  # (d_model,)
        logits = final_vector @ W_U.T  # (vocab,)
        gold_logit = logits[gold_token_id].item()
        gold_logits.append(gold_logit)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(n_layers), gold_logits, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Logit for Gold Token")
    plt.title(f"Layerwise Gold Logit Attribution for Prompt:\n{prompt}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved layerwise gold logit plot to {save_path}")
    plt.close()

# Example usage:
#compute_layerwise_gold_logit(model, tokenizer, prompt=prompt_texts_tu[0], gold_token_id=answers_ids_tu[0], save_path="layerwise_gold_logit_yo0.png")

#compute_layerwise_gold_logit(model, tokenizer, prompt=prompt_texts_tu[1], gold_token_id=answers_ids_tu[1], save_path="layerwise_gold_logit_yo1.png")

#compute_layerwise_gold_logit(model, tokenizer, prompt=prompt_texts_tu[2], gold_token_id=answers_ids_tu[2], save_path="layerwise_gold_logit_yo2.png")


#AVERAGE LAYERWISE LOGIT ATTRIBUTION
def compute_average_layerwise_gold_logit(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    gold_token_ids: List[int],
    save_path: str = "avg_layerwise_gold_logit.png",
    label: str = "YO"
):
    """
    Compute and plot the average gold token logit at each layer over a list of prompts.
    """
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    all_logits = []

    for prompt, gold_token_id in zip(prompts, gold_token_ids):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        pos = input_ids.shape[1] - 1

        activation_cache = []

        def get_hook(idx):
            def hook_fn(module, input, output):
                activation_cache.append(output[0].detach())
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

        gold_logits = []
        for resid in activation_cache:
            normed_resid = ln_f(resid)
            final_vector = normed_resid[0, pos, :]
            logits = final_vector @ W_U.T
            gold_logit = logits[gold_token_id].item()
            gold_logits.append(gold_logit)

        all_logits.append(gold_logits)

    # Average across prompts
    avg_logits = np.mean(all_logits, axis=0)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(n_layers), avg_logits, marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Average Logit for Gold Token")
    plt.title(f"Average Layerwise Gold Token Logit ({label.upper()} Dataset)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved average layerwise gold logit plot to {save_path}")
    plt.close()

#compute_average_layerwise_gold_logit(model, tokenizer, prompts=prompt_texts_yo, gold_token_ids=answers_ids_yo, save_path="avg_layerwise_gold_logit_yo.png", label = "YO")

#ompute_average_layerwise_gold_logit(model, tokenizer, prompts=prompt_texts_tu, gold_token_ids=answers_ids_tu, save_path="avg_layerwise_gold_logit_tu.png", label="TÚ")


#HEADWISE LOGIT ATTRIBUTION


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

# Example usage:
compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_yo[0], gold_token_id=answers_ids_yo[0], save_path="headwise_gold_logit_yo0.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_tu[0], gold_token_id=answers_ids_tu[0], save_path="headwise_gold_logit_tu0.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_yo[1], gold_token_id=answers_ids_yo[1], save_path="headwise_gold_logit_yo1.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_tu[1], gold_token_id=answers_ids_tu[1], save_path="headwise_gold_logit_tu1.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_yo[2], gold_token_id=answers_ids_yo[2], save_path="headwise_gold_logit_yo2.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_tu[2], gold_token_id=answers_ids_tu[2], save_path="headwise_gold_logit_tu2.png")

#AVERAGE
def compute_average_headwise_gold_logit_contributions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    gold_token_ids: List[int],
    save_path: str = "average_headwise_gold_logit.png",
    label: str = "YO"
):
    """
    Compute and visualize the average contribution of each attention head to the gold token logit
    across a list of prompts. Output is a (num_layers, num_heads) heatmap.
    Also prints the average contribution per head.
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

#compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=prompt_texts_yo, gold_token_ids=answers_ids_yo, save_path="avg_headwise_gold_logit_yo.png", label="YO")

#compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=prompt_texts_tu, gold_token_ids=answers_ids_tu, save_path="avg_headwise_gold_logit_tu.png", label="TÚ")

