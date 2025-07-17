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
from src.utils.spanish_dataset_generation import create_spanish_verbs, filter_spanish_conjugations
from jsalt2025.src.utils.spanish_build_prompts import generate_yo_dataset, generate_tu_dataset, build_conjugation_prompts
from src.utils.evaluation import evaluate_prompts 
from src.utils.dla import compute_direct_logit_attribution
from src.utils.logit_lens import compute_logit_lens, plot_logit_lens, plot_logit_lens_heatmap
from src.utils.layerwise_logit_attribution import compute_layerwise_gold_logit_contributions, compute_average_layerwise_gold_logit_contributions
from src.utils.headwise_logit_attribution import compute_headwise_gold_logit_contributions, compute_average_headwise_gold_logit_contributions
from src.utils.activation_analysis import visualize_average_top_heads_attention

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

# Prompt sets
spanish_yo = generate_yo_dataset(spanish_conjugations)
spanish_tu = generate_tu_dataset(spanish_conjugations)


#Build Conjugation Prompts

# Build prompts and answer token IDs
prompts_trimmed_yo, answers_ids_yo, _ = build_conjugation_prompts(tokenizer, spanish_yo)
prompts_trimmed_tu, answers_ids_tu, _ = build_conjugation_prompts(tokenizer, spanish_tu)

# Decode prompts back to full text for BLOOM
prompt_texts_yo = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_yo]
prompt_texts_tu = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_tu]



# --- Evaluation Function ---

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


# Example usage:
#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_yo[4], save_path="logit_lens_heatmap_yo4.png")
#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_tu[4], save_path="logit_lens_heatmap_tu4.png")

#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_yo[5], save_path="logit_lens_heatmap_yo5.png")
#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_tu[5], save_path="logit_lens_heatmap_tu5.png")

#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_yo[6], save_path="logit_lens_heatmap_yo6.png")
#plot_logit_lens_heatmap(model, tokenizer, prompt_texts_tu[6], save_path="logit_lens_heatmap_tu6.png")


#LAYERWISE LOGIT ATTRIBUTION

# Example usage:
#compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_tu[0], gold_token_id=answers_ids_tu[0], save_path="layerwise_gold_logit_yo0.png")

#compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_tu[1], gold_token_id=answers_ids_tu[1], save_path="layerwise_gold_logit_yo1.png")

#compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_tu[2], gold_token_id=answers_ids_tu[2], save_path="layerwise_gold_logit_yo2.png")


#AVERAGE LAYERWISE LOGIT ATTRIBUTION

#compute_average_layerwise_gold_logit_contributions(model, tokenizer, prompts=prompt_texts_yo, gold_token_ids=answers_ids_yo, save_path="avg_layerwise_gold_logit_yo.png", label = "YO")

#ompute_average_layerwise_gold_logit_contributions(model, tokenizer, prompts=prompt_texts_tu, gold_token_ids=answers_ids_tu, save_path="avg_layerwise_gold_logit_tu.png", label="TÚ")


#HEADWISE LOGIT ATTRIBUTION


# Example usage:
#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_yo[0], gold_token_id=answers_ids_yo[0], save_path="headwise_gold_logit_yo0.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_tu[0], gold_token_id=answers_ids_tu[0], save_path="headwise_gold_logit_tu0.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_yo[1], gold_token_id=answers_ids_yo[1], save_path="headwise_gold_logit_yo1.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_tu[1], gold_token_id=answers_ids_tu[1], save_path="headwise_gold_logit_tu1.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_yo[2], gold_token_id=answers_ids_yo[2], save_path="headwise_gold_logit_yo2.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_tu[2], gold_token_id=answers_ids_tu[2], save_path="headwise_gold_logit_tu2.png")



#avg_logits_yo, top_heads_yo = compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=prompt_texts_yo, gold_token_ids=answers_ids_yo, save_path="avg_headwise_gold_logit_yo.png", label="YO")

#avg_logits_tu, top_heads_tu = compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=prompt_texts_tu, gold_token_ids=answers_ids_tu, save_path="avg_headwise_gold_logit_tu.png", label="TÚ")

avg_yo, top_heads_yo = compute_average_headwise_gold_logit_contributions(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_yo,
    gold_token_ids=answers_ids_yo,
    save_path="avg_headwise_gold_logit_yo.png",
    label="YO",
    top_n=3
)

avg_tu, top_heads_tu = compute_average_headwise_gold_logit_contributions(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_tu,
    gold_token_ids=answers_ids_tu,
    save_path="avg_headwise_gold_logit_tu.png",
    label="TÚ",
    top_n=3
)

# Step 2: Visualize average attention patterns
#visualize_average_top_heads_attention(model=model, tokenizer=tokenizer, prompts=prompt_texts_yo, top_heads=top_heads_yo, save_path="top_heads_attention_avg_yo.html", label="YO")

#visualize_average_top_heads_attention(model=model, tokenizer=tokenizer, prompts=prompt_texts_tu, top_heads=top_heads_tu, save_path="top_heads_attention_avg_tu.html", label="TÚ")

visualize_average_top_heads_attention(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_yo,
    manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
    save_path="manual_heads_attention_avg_yo.html",
    label="YO"
)

visualize_average_top_heads_attention(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_tu,
    manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
    save_path="manual_heads_attention_avg_tu.html",
    label="YO"
)



