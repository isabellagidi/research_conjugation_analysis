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
from src.datasets.english.english_verbs import english_verbs_one, english_verbs_two, english_verbs_three
from src.utils.dataset_generation import combine_verbs, filter_conjugations, generate_first_singular_dataset_english, generate_second_singular_dataset_english, build_conjugation_prompts

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
english_verbs = combine_verbs(english_verbs_one, english_verbs_two, english_verbs_three)
english_conjugations = filter_conjugations(english_verbs, tokenizer)

# Prompt generators

# Prompt sets
english_first_sing = generate_first_singular_dataset_english(english_conjugations)
english_second_sing = generate_second_singular_dataset_english(english_conjugations)


#Build Conjugation Prompts

# Build prompts and answer token IDs
prompts_trimmed_first_sing_english, answers_ids_first_sing_english, _ = build_conjugation_prompts(tokenizer, english_first_sing)
prompts_trimmed_second_sing_english, answers_ids_second_sing_english, _ = build_conjugation_prompts(tokenizer, english_second_sing)

# Decode prompts back to full text for BLOOM
prompt_texts_first_sing_english = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_first_sing_english]
prompt_texts_second_sing_english = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_second_sing_english]



# --- Evaluation Function ---

print("Evaluation complete — printing results next...")
sys.stdout.flush()

# --- Run Evaluation ---
results_first_sing_english = evaluate_prompts(prompt_texts_first_sing_english, answers_ids_first_sing_english, label="Format FIRST PERSON SING")
results_second_sing_english = evaluate_prompts(prompt_texts_second_sing_english, answers_ids_second_sing_english, label="Format SECOND PERSON SING")

print("📎 results_yo =", results_first_sing_english)
sys.stdout.flush()

# --- Print Summary ---
print(f"\n📊 Evaluation results for {results_first_sing_english['label']}")
print(f"Top-1 Accuracy: {results_first_sing_english['accuracy']:.2%} ({results_first_sing_english['num_correct']}/{results_first_sing_english['num_total']})")
print(f"Average P(gold): {results_first_sing_english['avg_gold_prob']:.2%}")
print(f"Average Rank of Gold Token: {results_first_sing_english['avg_gold_rank']:.2f}")
sys.stdout.flush()

print(f"\n📊 Evaluation results for {results_second_sing_english['label']}")
print(f"Top-1 Accuracy: {results_second_sing_english['accuracy']:.2%} ({results_second_sing_english['num_correct']}/{results_second_sing_english['num_total']})")
print(f"Average P(gold): {results_second_sing_english['avg_gold_prob']:.2%}")
print(f"Average Rank of Gold Token: {results_second_sing_english['avg_gold_rank']:.2f}")
sys.stdout.flush()




# --- Direct Logit Attribution (DLA) for BLOOM via hooks ---


results_dla_first_sing_english = compute_direct_logit_attribution(
    model=model,
    tokenizer=tokenizer,
    prompt_texts=prompt_texts_first_sing_english,     # decoded prompts
    answer_ids=answers_ids_first_sing_english,        # gold token IDs
    label="Format YO"
)

# Print summary
print(f"\n🔍 Direct Logit Attribution for {results_dla_first_sing_english['label']}")
sys.stdout.flush()
print(f"Per-prompt DLA values (first 10): {results_dla_first_sing_english['dla_values'][:10]}")
sys.stdout.flush()
print(f"Average DLA: {results_dla_first_sing_english['avg_dla']:.3f}")
sys.stdout.flush()


results_dla_second_sing_english = compute_direct_logit_attribution(
    model=model,
    tokenizer=tokenizer,
    prompt_texts=prompt_texts_second_sing_english,     # decoded prompts
    answer_ids=answers_ids_second_sing_english,        # gold token IDs
    label="Format TU"
)

# Print summary
print(f"\n🔍 Direct Logit Attribution for {results_dla_second_sing_english['label']}")
sys.stdout.flush()
print(f"Per-prompt DLA values (first 10): {results_dla_second_sing_english['dla_values'][:10]}")
sys.stdout.flush()
print(f"Average DLA: {results_dla_second_sing_english['avg_dla']:.3f}")
sys.stdout.flush()



logit_lens_results_first_sing_english = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_first_sing_english,
    gold_token_ids=answers_ids_first_sing_english
)

print("\n 📈 Logit Lens (YO):")
for label, logit in zip(logit_lens_results_first_sing_english["labels"], logit_lens_results_first_sing_english["avg_per_layer"]):
    print(f"{label}: {logit.item():.3f}")
sys.stdout.flush()


logit_lens_results_second_sing_english = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_second_sing_english,
    gold_token_ids=answers_ids_second_sing_english
)

print("\n 📈 Logit Lens (TU):")
for label, logit in zip(logit_lens_results_second_sing_english["labels"], logit_lens_results_second_sing_english["avg_per_layer"]):
    print(f"{label}: {logit.item():.3f}")
sys.stdout.flush()


#PLOT LOGIT LENS



logit_lens_results_examples_first_sing_english = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_first_sing_english[:3],
    gold_token_ids=answers_ids_first_sing_english[:3]
)

logit_lens_results_examples_second_sing_english = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_second_sing_english[:3],
    gold_token_ids=answers_ids_second_sing_english[:3]
)
# Plot both YO and TU
plot_logit_lens(logits_tensor=logit_lens_results_examples_first_sing_english["gold_logits"], labels=logit_lens_results_examples_first_sing_english["labels"], title="FIRST SING Logit Lens", color_prefix="first sing")

plot_logit_lens(logits_tensor=logit_lens_results_examples_second_sing_english["gold_logits"], labels=logit_lens_results_examples_second_sing_english["labels"], title="SECOND SING Logit Lens", color_prefix="second sing")

#LOGIT LENS heatmap

plot_logit_lens_heatmap(model, tokenizer, prompt_texts_first_sing_english[0], save_path="logit_lens_heatmap_first_sing_english0.png")
plot_logit_lens_heatmap(model, tokenizer, prompt_texts_second_sing_english[0], save_path="logit_lens_heatmap_second_sing_english0.png")

plot_logit_lens_heatmap(model, tokenizer, prompt_texts_first_sing_english[1], save_path="logit_lens_heatmap_first_sing_english1.png")
plot_logit_lens_heatmap(model, tokenizer, prompt_texts_second_sing_english[1], save_path="logit_lens_heatmap_second_sing_english1.png")

plot_logit_lens_heatmap(model, tokenizer, prompt_texts_first_sing_english[2], save_path="logit_lens_heatmap_first_sing_english2.png")
plot_logit_lens_heatmap(model, tokenizer, prompt_texts_second_sing_english[2], save_path="logit_lens_heatmap_second_sing_english2.png")

plot_logit_lens_heatmap(model, tokenizer, prompt_texts_first_sing_english[3], save_path="logit_lens_heatmap_first_sing_english3.png")
plot_logit_lens_heatmap(model, tokenizer, prompt_texts_second_sing_english[3], save_path="logit_lens_heatmap_second_sing_english3.png")

plot_logit_lens_heatmap(model, tokenizer, prompt_texts_first_sing_english[4], save_path="logit_lens_heatmap_first_sing_english4.png")
plot_logit_lens_heatmap(model, tokenizer, prompt_texts_second_sing_english[4], save_path="logit_lens_heatmap_second_sing_english4.png")


#LAYERWISE LOGIT ATTRIBUTION
compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_english[0], gold_token_id=answers_ids_first_sing_english[0], save_path="layerwise_gold_logit_first_sing_english0.png")
compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_english[1], gold_token_id=answers_ids_first_sing_english[1], save_path="layerwise_gold_logit_first_sing_english1.png")
compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_english[2], gold_token_id=answers_ids_first_sing_english[2], save_path="layerwise_gold_logit_first_sing_english2.png")


compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_english[0], gold_token_id=answers_ids_second_sing_english[0], save_path="layerwise_gold_logit_second_sing_english0.png")
compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_english[1], gold_token_id=answers_ids_second_sing_english[1], save_path="layerwise_gold_logit_second_sing_english1.png")
compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_english[2], gold_token_id=answers_ids_second_sing_english[2], save_path="layerwise_gold_logit_second_sing_english2.png")


#AVERAGE LAYERWISE LOGIT ATTRIBUTION

compute_average_layerwise_gold_logit_contributions(model, tokenizer, prompts=prompt_texts_first_sing_english, gold_token_ids=answers_ids_first_sing_english, save_path="avg_layerwise_gold_logit_first_sing_english.png", label = "FIRST SING")

compute_average_layerwise_gold_logit_contributions(model, tokenizer, prompts=prompt_texts_second_sing_english, gold_token_ids=answers_ids_second_sing_english, save_path="avg_layerwise_gold_logit_second_sing_english.png", label="SECOND SING")


#HEADWISE LOGIT ATTRIBUTION


# Example usage:
compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_english[0], gold_token_id=answers_ids_first_sing_english[0], save_path="headwise_gold_logit_first_sing_english0.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_english[0], gold_token_id=answers_ids_second_sing_english[0], save_path="headwise_gold_logit_second_sing_english0.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_english[1], gold_token_id=answers_ids_first_sing_english[1], save_path="headwise_gold_logit__first_sing_english1.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_english[1], gold_token_id=answers_ids_second_sing_english[1], save_path="headwise_gold_logit_second_sing_english1.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_first_sing_english[2], gold_token_id=answers_ids_first_sing_english[2], save_path="headwise_gold_logit_first_sing_english2.png")

compute_headwise_gold_logit_contributions(model, tokenizer, prompt=prompt_texts_second_sing_english[2], gold_token_id=answers_ids_second_sing_english[2], save_path="headwise_gold_logit_second_sing_english2.png")

#delete the naming and use the wrong function if you get an error
avg_logits_first_sing_english, top_heads_first_sing_english = compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=prompt_texts_first_sing_english, gold_token_ids=answers_ids_first_sing_english, save_path="avg_headwise_gold_logit_first_sing_english.png", label="FIRST SING")

avg_logits_second_sing_english, top_heads_second_sing_english = compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=prompt_texts_second_sing_english, gold_token_ids=answers_ids_second_sing_english, save_path="avg_headwise_gold_logit_second_sing_english.png", label="SECOND SING")

avg_first_sing_english, top_heads_first_sing_english = compute_average_headwise_gold_logit_contributions(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_first_sing_english,
    gold_token_ids=answers_ids_first_sing_english,
    save_path="avg_headwise_gold_logit_first_sing_english.png",
    label="FIRST SING",
    top_n=3
)

avg_second_sing_english, top_heads_second_sing_english = compute_average_headwise_gold_logit_contributions(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_second_sing_english,
    gold_token_ids=answers_ids_second_sing_english,
    save_path="avg_headwise_gold_logit_second_sing_english.png",
    label="SECOND SING",
    top_n=3
)

# Step 2: Visualize average attention patterns
visualize_average_top_heads_attention(model=model, tokenizer=tokenizer, prompts=prompt_texts_first_sing_english, top_heads=top_heads_first_sing_english, save_path="top_heads_attention_avg_first_sing_english.html", label="FIRS SING")

visualize_average_top_heads_attention(model=model, tokenizer=tokenizer, prompts=prompt_texts_second_sing_english, top_heads=top_heads_second_sing_english, save_path="top_heads_attention_avg_second_sing_english.html", label="SECOND SING")

visualize_average_top_heads_attention(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_first_sing_english,
    manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
    save_path="manual_heads_attention_avg_first_sing_english.html",
    label="FIRST SING"
)

visualize_average_top_heads_attention(
    model=model,
    tokenizer=tokenizer,
    prompts=prompt_texts_second_sing_english,
    manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
    save_path="manual_heads_attention_avg_second_sing_english.html",
    label="SECOND SING"
)
