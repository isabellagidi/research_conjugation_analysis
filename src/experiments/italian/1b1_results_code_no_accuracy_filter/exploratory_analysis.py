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

#language-specific
from src.datasets.italian.italian_verbs import italian_verbs_one, italian_verbs_two, italian_verbs_three
from src.utils.dataset_generation import combine_verbs, filter_conjugations, generate_first_singular_dataset_italian, generate_second_singular_dataset_italian, build_conjugation_prompts

#general
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
italian_verbs = combine_verbs(italian_verbs_one, italian_verbs_two, italian_verbs_three)
italian_conjugations = filter_conjugations(italian_verbs, tokenizer)

# Prompt generators

# Prompt sets
italian_first_sing = generate_first_singular_dataset_italian(italian_conjugations)
italian_second_sing = generate_second_singular_dataset_italian(italian_conjugations)


#Build Conjugation Prompts

# Build prompts and answer token IDs
prompts_trimmed_first_sing_italian, answers_ids_first_sing_italian, _ = build_conjugation_prompts(tokenizer, italian_first_sing)
prompts_trimmed_second_sing_italian, answers_ids_second_sing_italian, _ = build_conjugation_prompts(tokenizer, italian_second_sing)

# Decode prompts back to full text for BLOOM
prompt_texts_first_sing_italian = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_first_sing_italian]
prompt_texts_second_sing_italian = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_second_sing_italian]



# --- Evaluation Function ---

print("Evaluation complete — printing results next...")
sys.stdout.flush()

# --- Run Evaluation ---
results_first_sing_italian = evaluate_prompts(prompt_texts_first_sing_italian, answers_ids_first_sing_italian, label="Format FIRST PERSON SING")
results_second_sing_italian = evaluate_prompts(prompt_texts_second_sing_italian, answers_ids_second_sing_italian, label="Format SECOND PERSON SING")

print("📎 results_yo =", results_first_sing_italian)
sys.stdout.flush()

# --- Print Summary ---
print(f"\n📊 Evaluation results for {results_first_sing_italian['label']}")
print(f"Top-1 Accuracy: {results_first_sing_italian['accuracy']:.2%} ({results_first_sing_italian['num_correct']}/{results_first_sing_italian['num_total']})")
print(f"Average P(gold): {results_first_sing_italian['avg_gold_prob']:.2%}")
print(f"Average Rank of Gold Token: {results_first_sing_italian['avg_gold_rank']:.2f}")
sys.stdout.flush()

print(f"\n📊 Evaluation results for {results_second_sing_italian['label']}")
print(f"Top-1 Accuracy: {results_second_sing_italian['accuracy']:.2%} ({results_second_sing_italian['num_correct']}/{results_second_sing_italian['num_total']})")
print(f"Average P(gold): {results_second_sing_italian['avg_gold_prob']:.2%}")
print(f"Average Rank of Gold Token: {results_second_sing_italian['avg_gold_rank']:.2f}")
sys.stdout.flush()




# --- Direct Logit Attribution (DLA) for BLOOM via hooks ---


results_dla_first_sing_italian = compute_direct_logit_attribution(
    model=model,
    tokenizer=tokenizer,
    prompt_texts=prompt_texts_first_sing_italian,     # decoded prompts
    answer_ids=answers_ids_first_sing_italian,        # gold token IDs
    label="Format YO"
)

# Print summary
print(f"\n🔍 Direct Logit Attribution for {results_dla_first_sing_italian['label']}")
sys.stdout.flush()
print(f"Per-prompt DLA values (first 10): {results_dla_first_sing_italian['dla_values'][:10]}")
sys.stdout.flush()
print(f"Average DLA: {results_dla_first_sing_italian['avg_dla']:.3f}")
sys.stdout.flush()


results_dla_second_sing_italian = compute_direct_logit_attribution(
    model=model,
    tokenizer=tokenizer,
    prompt_texts=prompt_texts_second_sing_italian,     # decoded prompts
    answer_ids=answers_ids_second_sing_italian,        # gold token IDs
    label="Format TU"
)

# Print summary
print(f"\n🔍 Direct Logit Attribution for {results_dla_second_sing_italian['label']}")
sys.stdout.flush()
print(f"Per-prompt DLA values (first 10): {results_dla_second_sing_italian['dla_values'][:10]}")
sys.stdout.flush()
print(f"Average DLA: {results_dla_second_sing_italian['avg_dla']:.3f}")
sys.stdout.flush()



logit_lens_results_first_sing_italian = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_first_sing_italian,
    gold_token_ids=answers_ids_first_sing_italian
)

print("\n 📈 Logit Lens (YO):")
for label, logit in zip(logit_lens_results_first_sing_italian["labels"], logit_lens_results_first_sing_italian["avg_per_layer"]):
    print(f"{label}: {logit.item():.3f}")
sys.stdout.flush()


logit_lens_results_second_sing_italian = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_second_sing_italian,
    gold_token_ids=answers_ids_second_sing_italian
)

print("\n 📈 Logit Lens (TU):")
for label, logit in zip(logit_lens_results_second_sing_italian["labels"], logit_lens_results_second_sing_italian["avg_per_layer"]):
    print(f"{label}: {logit.item():.3f}")
sys.stdout.flush()


#PLOT LOGIT LENS



logit_lens_results_examples_first_sing_italian = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_first_sing_italian[:3],
    gold_token_ids=answers_ids_first_sing_italian[:3]
)

logit_lens_results_examples_second_sing_italian = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_second_sing_italian[:3],
    gold_token_ids=answers_ids_second_sing_italian[:3]
)
# Plot both YO and TU
plot_logit_lens(logits_tensor=logit_lens_results_examples_first_sing_italian["gold_logits"], labels=logit_lens_results_examples_first_sing_italian["labels"], title="FIRST SING Logit Lens", color_prefix="first sing")

plot_logit_lens(logits_tensor=logit_lens_results_examples_second_sing_italian["gold_logits"], labels=logit_lens_results_examples_second_sing_italian["labels"], title="SECOND SING Logit Lens", color_prefix="second sing")

#LOGIT LENS heatmap

plot_logit_lens_heatmap(model, tokenizer, prompt_texts_first_sing_italian[0], save_path="logit_lens_heatmap_first_sing_italian0.png")
plot_logit_lens_heatmap(model, tokenizer, prompt_texts_second_sing_italian[0], save_path="logit_lens_heatmap_second_sing_italian0.png")

plot_logit_lens_heatmap(model, tokenizer, prompt_texts_first_sing_italian[1], save_path="logit_lens_heatmap_first_sing_italian1.png")
plot_logit_lens_heatmap(model, tokenizer, prompt_texts_second_sing_italian[1], save_path="logit_lens_heatmap_second_sing_italian1.png")

plot_logit_lens_heatmap(model, tokenizer, prompt_texts_first_sing_italian[2], save_path="logit_lens_heatmap_first_sing_italian2.png")
plot_logit_lens_heatmap(model, tokenizer, prompt_texts_second_sing_italian[2], save_path="logit_lens_heatmap_second_sing_italian2.png")

plot_logit_lens_heatmap(model, tokenizer, prompt_texts_first_sing_italian[3], save_path="logit_lens_heatmap_first_sing_italian3.png")
plot_logit_lens_heatmap(model, tokenizer, prompt_texts_second_sing_italian[3], save_path="logit_lens_heatmap_second_sing_italian3.png")

plot_logit_lens_heatmap(model, tokenizer, prompt_texts_first_sing_italian[4], save_path="logit_lens_heatmap_first_sing_italian4.png")
plot_logit_lens_heatmap(model, tokenizer, prompt_texts_second_sing_italian[4], save_path="logit_lens_heatmap_second_sing_italian4.png")


#LAYERWISE LOGIT ATTRIBUTION
compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_italian[0], gold_token_id=answers_ids_first_sing_italian[0], save_path="layerwise_gold_logit_first_sing_italian0.png")
compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_italian[1], gold_token_id=answers_ids_first_sing_italian[1], save_path="layerwise_gold_logit_first_sing_italian1.png")
compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_italian[2], gold_token_id=answers_ids_first_sing_italian[2], save_path="layerwise_gold_logit_first_sing_italian2.png")


compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_italian[0], gold_token_id=answers_ids_second_sing_italian[0], save_path="layerwise_gold_logit_second_sing_italian0.png")
compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_italian[1], gold_token_id=answers_ids_second_sing_italian[1], save_path="layerwise_gold_logit_second_sing_italian1.png")
compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_italian[2], gold_token_id=answers_ids_second_sing_italian[2], save_path="layerwise_gold_logit_second_sing_italian2.png")


#AVERAGE LAYERWISE LOGIT ATTRIBUTION

compute_average_layerwise_gold_logit_contributions(model, tokenizer, prompts=prompt_texts_first_sing_italian, gold_token_ids=answers_ids_first_sing_italian, save_path="avg_layerwise_gold_logit_first_sing_italian.png", label = "FIRST SING")

compute_average_layerwise_gold_logit_contributions(model, tokenizer, prompts=prompt_texts_second_sing_italian, gold_token_ids=answers_ids_second_sing_italian, save_path="avg_layerwise_gold_logit_second_sing_italian.png", label="SECOND SING")


#HEADWISE LOGIT ATTRIBUTION


# Example usage:
compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_italian[0], gold_token_id=answers_ids_first_sing_italian[0], save_path="headwise_gold_logit_first_sing_italian0.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_italian[0], gold_token_id=answers_ids_second_sing_italian[0], save_path="headwise_gold_logit_second_sing_italian0.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_italian[1], gold_token_id=answers_ids_first_sing_italian[1], save_path="headwise_gold_logit__first_sing_italian1.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_italian[1], gold_token_id=answers_ids_second_sing_italian[1], save_path="headwise_gold_logit_second_sing_italian1.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_italian[2], gold_token_id=answers_ids_first_sing_italian[2], save_path="headwise_gold_logit_first_sing_italian2.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_italian[2], gold_token_id=answers_ids_second_sing_italian[2], save_path="headwise_gold_logit_second_sing_italian2.png")


avg_logits_first_sing_italian, top_heads_first_sing_italian = compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=prompt_texts_first_sing_italian, gold_token_ids=answers_ids_first_sing_italian, save_path="avg_headwise_gold_logit_first_sing_italian.png", label="FIRST SING")

avg_logits_second_sing_italian, top_heads_second_sing_italian = compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=prompt_texts_second_sing_italian, gold_token_ids=answers_ids_second_sing_italian, save_path="avg_headwise_gold_logit_second_sing_italian.png", label="SECOND SING")

avg_logits_first_sing_italian, top_heads_first_sing_italian = compute_average_headwise_gold_logit_contributions(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_first_sing_italian,
    gold_token_ids=answers_ids_first_sing_italian,
    save_path="avg_headwise_gold_logit_first_sing_italian.png",
    label="FIRST SING",
    top_n=3
)

avg_second_sing_italian, top_heads_second_sing_italian = compute_average_headwise_gold_logit_contributions(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_second_sing_italian,
    gold_token_ids=answers_ids_second_sing_italian,
    save_path="avg_headwise_gold_logit_second_sing_italian.png",
    label="SECOND SING",
    top_n=3
)

# Step 2: Visualize average attention patterns
visualize_average_top_heads_attention(model=model, tokenizer=tokenizer, prompts=prompt_texts_first_sing_italian, top_heads=top_heads_first_sing_italian, save_path="top_heads_attention_avg_first_sing_italian.html", label="FIRST SING")

visualize_average_top_heads_attention(model=model, tokenizer=tokenizer, prompts=prompt_texts_second_sing_italian, top_heads=top_heads_second_sing_italian, save_path="top_heads_attention_avg_second_sing_italian.html", label="SECOND SING")

visualize_average_top_heads_attention(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_first_sing_italian,
    manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
    save_path="manual_heads_attention_avg_first_sing_italian.html",
    label="FIRST SING"
)

visualize_average_top_heads_attention(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_second_sing_italian,
    manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
    save_path="manual_heads_attention_avg_second_sing_italian.html",
    label="SECOND SING"
)