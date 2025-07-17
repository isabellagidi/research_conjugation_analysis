import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
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

# Visualization of top heads (after computing average contribution)
def visualize_average_top_heads_attention(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    top_heads: Optional[List[Tuple[int, int, float]]] = None,
    manual_head_labels: Optional[List[str]] = None,
    save_path: str = "top_heads_attention_avg.html",
    label: str = "YO"
):
    """
    Computes average attention patterns across all prompts (padded to the same length)
    and visualizes the top attention heads. Saves the result as an HTML file via CircuitsVis.
    """
    import torch
    import circuitsvis.attention as cv_attn

    device = next(model.parameters()).device

    # 1) Find the max token length over all prompts
    lens = [ len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts ]
    max_len = max(lens)

    # 2) Tell the model to return attentions
    model.config.output_attentions = True

    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.n_head

    # initialize accumulator
    attn_acc = {
        layer: torch.zeros((n_heads, max_len, max_len), device=device)
        for layer in range(n_layers)
    }
    total = 0
    str_tokens = None

    for prompt in prompts:
        # 3) Re-tokenize *with* padding up to max_len
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        ).to(device)

        # 4) Run forward and grab attentions
        outputs = model(**inputs, output_attentions=True)
        atts: Tuple[torch.Tensor,...] = outputs.attentions
        # atts[layer] is [batch, n_heads, seq, seq]

        # setup token labels once
        if str_tokens is None:
            second_prompt_inputs = tokenizer(prompts[1], return_tensors="pt", padding="max_length", truncation=True, max_length=max_len).to(device)
            str_tokens = tokenizer.convert_ids_to_tokens(second_prompt_inputs["input_ids"][0])
        # accumulate the first (and only) batch item
        for layer, att in enumerate(atts):
            attn_acc[layer] += att[0].detach()

        total += 1

    # 5) average
    for layer in attn_acc:
        attn_acc[layer] /= total

    # 6) Select heads to visualize
    if manual_head_labels is not None:
        selected_heads = parse_head_labels(manual_head_labels)
        labels = manual_head_labels
    elif top_heads is not None:
        selected_heads = [(layer, head) for (layer, head, _) in top_heads]
        labels = [f"L{layer}H{head}" for (layer, head) in selected_heads]
    else:
        raise ValueError("Must provide either `top_heads` or `manual_head_labels`")

    patterns = [attn_acc[layer][head].cpu() for (layer, head) in selected_heads]
    stacked = torch.stack(patterns, dim=0)


    # 7) render with circuitsvis
    html_str = cv_attn.attention_heads(
        attention=stacked,
        tokens=str_tokens,
        attention_head_names=labels
    ).show_code()

    with open(save_path, "w") as f:
        f.write(html_str)

    print(f"✅ Saved average attention visualization for {label} to {save_path}")
    print("📂 Open it in your browser to explore interactively.")


def parse_head_labels(head_labels: List[str]) -> List[Tuple[int, int]]:
    """
    Converts strings like "L23H1" into tuples like (23, 1)
    """
    parsed = []
    for label in head_labels:
        if label.startswith("L") and "H" in label:
            layer_str, head_str = label[1:].split("H")
            parsed.append((int(layer_str), int(head_str)))
    return parsed
