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

def compute_layerwise_gold_logit_contributions(
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

def compute_average_layerwise_gold_logit_contributions(
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