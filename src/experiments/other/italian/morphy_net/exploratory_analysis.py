import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# --- Setup ---
sys.path.append('../../')

#general
from src.utils.evaluation import evaluate_prompts 
from src.utils.dla import compute_direct_logit_attribution
from src.utils.logit_lens import compute_logit_lens, plot_logit_lens, plot_logit_lens_heatmap
from src.utils.layerwise_logit_attribution import compute_layerwise_gold_logit_contributions, compute_average_layerwise_gold_logit_contributions
from src.utils.headwise_logit_attribution import compute_headwise_gold_logit_contributions, compute_average_headwise_gold_logit_contributions
from src.utils.activation_analysis import visualize_average_top_heads_attention


from src.utils.dataset_preparation import load_json_data, filter_conjugations, build_conjugation_prompts, accuracy_filter
from src.utils.dataset_preparation import generate_first_singular_dataset_italian, generate_second_singular_dataset_italian



with open(os.path.expanduser("~/.huggingface/token")) as f:
    hf_token = f.read().strip()

# Load tokenizer and model
model_name = "bigscience/bloom-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # required for BLOOM
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

print(model_name)

# Load your JSON data
file_path = "/home/lis.isabella.gidi/jsalt2025/src/datasets/morphy_net/MorphyNet_all_conjugations.json"
all_verbs = load_json_data(file_path)
# Prompt sets

italian_conjugations = filter_conjugations(all_verbs, tokenizer, "ita")

#DOING THIS TO SAVE RAM!!!!!!!
italian_conjugations = italian_conjugations[:500]
print("length of dataset (usually 500 but in this case):" , len(italian_conjugations))

italian_first_singular = generate_first_singular_dataset_italian(italian_conjugations)
italian_second_singular = generate_second_singular_dataset_italian(italian_conjugations)

print("Example prompt input:", italian_first_singular[0])
print("Type:", type(italian_first_singular[0]))

print("Example prompt input:", italian_second_singular[0])
print("Type:", type(italian_second_singular[0]))


prompts_trimmed_first_sing_italian, answers_ids_first_sing_italian, verb_entries_first_sing_italian = build_conjugation_prompts(tokenizer, italian_first_singular, italian_conjugations)
prompts_trimmed_second_sing_italian, answers_ids_second_sing_italian, verb_entries_second_sing_italian = build_conjugation_prompts(tokenizer, italian_second_singular, italian_conjugations)


prompt_texts_first_sing_italian = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_first_sing_italian]
prompt_texts_second_sing_italian = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_second_sing_italian]


correct_prompt_texts_first_sing_italian, correct_answers_ids_first_sing_italian, _, incorrect_prompt_texts_first_sing_italian, incorrect_answers_ids_first_sing_italian, _ = accuracy_filter(
    prompts_trimmed_first_sing_italian,
    answers_ids_first_sing_italian,
    verb_entries_first_sing_italian,
    model_name
)

# Truncate to first 100 correct examples for testing
#correct_prompt_texts_first_sing_italian = correct_prompt_texts_first_sing_italian[:100]
#correct_answers_ids_first_sing_italian = correct_answers_ids_first_sing_italian[:100]


print(f"Correct: {len(correct_prompt_texts_first_sing_italian)} / {len(verb_entries_first_sing_italian)}")
print(f"Incorrect: {len(incorrect_prompt_texts_first_sing_italian)}")

print("✅ Correct prompts:")
for prompt in correct_prompt_texts_first_sing_italian[:3]:
    print(prompt)

print("\n❌ Incorrect prompts:")
for prompt in incorrect_prompt_texts_first_sing_italian[:3]:
    print(prompt)

correct_prompt_texts_second_sing_italian, correct_answers_ids_second_sing_italian, _, incorrect_prompt_texts_second_sing_italian, incorrect_answers_ids_second_sing_italian, _ = accuracy_filter(
    prompts_trimmed_second_sing_italian,
    answers_ids_second_sing_italian,
    verb_entries_second_sing_italian,
    model_name
)

print("number of prompts saved second sing:", len(correct_prompt_texts_second_sing_italian))

# Truncate to first few hundred correct examples for testing
#correct_prompt_texts_second_sing_italian = correct_prompt_texts_second_sing_italian[:200]
#correct_answers_ids_second_sing_italian = correct_answers_ids_second_sing_italian[:200]

print(f"Correct: {len(correct_prompt_texts_second_sing_italian)} / {len(verb_entries_second_sing_italian)}")
print(f"Incorrect: {len(incorrect_prompt_texts_second_sing_italian)}")

print("✅ Correct prompts:")
for prompt in correct_prompt_texts_second_sing_italian[:3]:
    print(prompt)

print("\n❌ Incorrect prompts:")
for prompt in incorrect_prompt_texts_second_sing_italian[:3]:
    print(prompt)


# --- Direct Logit Attribution (DLA) for BLOOM via hooks ---


results_dla_first_sing_italian = compute_direct_logit_attribution(
    model=model,
    tokenizer=tokenizer,
    prompt_texts=correct_prompt_texts_first_sing_italian,     # decoded prompts
    answer_ids=correct_answers_ids_first_sing_italian,        # gold token IDs
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
    prompt_texts=correct_prompt_texts_second_sing_italian,     # decoded prompts
    answer_ids=correct_answers_ids_second_sing_italian,        # gold token IDs
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
    prompts=correct_prompt_texts_first_sing_italian,
    gold_token_ids=correct_answers_ids_first_sing_italian
)

print("\n 📈 Logit Lens (YO):")
for label, logit in zip(logit_lens_results_first_sing_italian["labels"], logit_lens_results_first_sing_italian["avg_per_layer"]):
    print(f"{label}: {logit.item():.3f}")
sys.stdout.flush()


logit_lens_results_second_sing_italian = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=correct_prompt_texts_second_sing_italian,
    gold_token_ids=correct_answers_ids_second_sing_italian
)

print("\n 📈 Logit Lens (TU):")
for label, logit in zip(logit_lens_results_second_sing_italian["labels"], logit_lens_results_second_sing_italian["avg_per_layer"]):
    print(f"{label}: {logit.item():.3f}")
sys.stdout.flush()


#PLOT LOGIT LENS



logit_lens_results_examples_first_sing_italian = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=correct_prompt_texts_first_sing_italian[:3],
    gold_token_ids=correct_answers_ids_first_sing_italian[:3]
)

logit_lens_results_examples_second_sing_italian = compute_logit_lens(
    model=model,
    tokenizer=tokenizer,
    prompts=correct_prompt_texts_second_sing_italian[:3],
    gold_token_ids=correct_answers_ids_second_sing_italian[:3]
)
# Plot both YO and TU
plot_logit_lens(logits_tensor=logit_lens_results_examples_first_sing_italian["gold_logits"], labels=logit_lens_results_examples_first_sing_italian["labels"], title="FIRST SING Logit Lens", color_prefix="first sing")

plot_logit_lens(logits_tensor=logit_lens_results_examples_second_sing_italian["gold_logits"], labels=logit_lens_results_examples_second_sing_italian["labels"], title="SECOND SING Logit Lens", color_prefix="second sing")

#LOGIT LENS heatmap

plot_logit_lens_heatmap(model, tokenizer, correct_prompt_texts_first_sing_italian[0], save_path="logit_lens_heatmap_first_sing_italian0.png")
plot_logit_lens_heatmap(model, tokenizer, correct_prompt_texts_second_sing_italian[0], save_path="logit_lens_heatmap_second_sing_italian0.png")

plot_logit_lens_heatmap(model, tokenizer, correct_prompt_texts_first_sing_italian[1], save_path="logit_lens_heatmap_first_sing_italian1.png")
plot_logit_lens_heatmap(model, tokenizer, correct_prompt_texts_second_sing_italian[1], save_path="logit_lens_heatmap_second_sing_italian1.png")

#plot_logit_lens_heatmap(model, tokenizer, correct_prompt_texts_first_sing_italian[2], save_path="logit_lens_heatmap_first_sing_italian2.png")
#plot_logit_lens_heatmap(model, tokenizer, correct_prompt_texts_second_sing_italian[2], save_path="logit_lens_heatmap_second_sing_italian2.png")

#plot_logit_lens_heatmap(model, tokenizer, correct_prompt_texts_first_sing_italian[3], save_path="logit_lens_heatmap_first_sing_italian3.png")
#plot_logit_lens_heatmap(model, tokenizer, correct_prompt_texts_second_sing_italian[3], save_path="logit_lens_heatmap_second_sing_italian3.png")

#plot_logit_lens_heatmap(model, tokenizer, correct_prompt_texts_first_sing_italian[4], save_path="logit_lens_heatmap_first_sing_italian4.png")
#plot_logit_lens_heatmap(model, tokenizer, correct_prompt_texts_second_sing_italian[4], save_path="logit_lens_heatmap_second_sing_italian4.png")


#EXAMPLE LAYERWISE LOGIT ATTRIBUTION
#compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_first_sing_italian[0], gold_token_id=correct_answers_ids_first_sing_italian[0], save_path="layerwise_gold_logit_first_sing_italian0.png")
#compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_first_sing_italian[1], gold_token_id=correct_answers_ids_first_sing_italian[1], save_path="layerwise_gold_logit_first_sing_italian1.png")
#compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_first_sing_italian[2], gold_token_id=correct_answers_ids_first_sing_italian[2], save_path="layerwise_gold_logit_first_sing_italian2.png")


#compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_second_sing_italian[0], gold_token_id=correct_answers_ids_second_sing_italian[0], save_path="layerwise_gold_logit_second_sing_italian0.png")
#compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_second_sing_italian[1], gold_token_id=correct_answers_ids_second_sing_italian[1], save_path="layerwise_gold_logit_second_sing_italian1.png")
#compute_layerwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_second_sing_italian[2], gold_token_id=correct_answers_ids_second_sing_italian[2], save_path="layerwise_gold_logit_second_sing_italian2.png")


#AVERAGE LAYERWISE LOGIT ATTRIBUTION

compute_average_layerwise_gold_logit_contributions(model, tokenizer, prompts=correct_prompt_texts_first_sing_italian, gold_token_ids=correct_answers_ids_first_sing_italian, save_path="avg_layerwise_gold_logit_first_sing_italian.png", label = "FIRST SING")

compute_average_layerwise_gold_logit_contributions(model, tokenizer, prompts=correct_prompt_texts_second_sing_italian, gold_token_ids=correct_answers_ids_second_sing_italian, save_path="avg_layerwise_gold_logit_second_sing_italian.png", label="SECOND SING")


#HEADWISE LOGIT ATTRIBUTION


# EXAMPLE HEADWISE LOGIT ATTRIBUTION
#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_first_sing_italian[0], gold_token_id=correct_answers_ids_first_sing_italian[0], save_path="headwise_gold_logit_first_sing_italian0.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_second_sing_italian[0], gold_token_id=correct_answers_ids_second_sing_italian[0], save_path="headwise_gold_logit_second_sing_italian0.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_first_sing_italian[1], gold_token_id=correct_answers_ids_first_sing_italian[1], save_path="headwise_gold_logit__first_sing_italian1.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_second_sing_italian[1], gold_token_id=correct_answers_ids_second_sing_italian[1], save_path="headwise_gold_logit_second_sing_italian1.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_first_sing_italian[2], gold_token_id=correct_answers_ids_first_sing_italian[2], save_path="headwise_gold_logit_first_sing_italian2.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_second_sing_italian[2], gold_token_id=correct_answers_ids_second_sing_italian[2], save_path="headwise_gold_logit_second_sing_italian2.png")

# AVERAGE HEADWISE LOGIT ATTRIBUTION
avg_logits_first_sing_italian, top_heads_first_sing_italian = compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=correct_prompt_texts_first_sing_italian, gold_token_ids=correct_answers_ids_first_sing_italian, save_path="avg_headwise_gold_logit_first_sing_italian.png", label="FIRST SING")

avg_logits_second_sing_italian, top_heads_second_sing_italian = compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=correct_prompt_texts_second_sing_italian, gold_token_ids=correct_answers_ids_second_sing_italian, save_path="avg_headwise_gold_logit_second_sing_italian.png", label="SECOND SING")

avg_logits_first_sing_italian, top_heads_first_sing_italian = compute_average_headwise_gold_logit_contributions(
    model=model,
    tokenizer=tokenizer,
    prompts=correct_prompt_texts_first_sing_italian,
    gold_token_ids=correct_answers_ids_first_sing_italian,
    save_path="avg_headwise_gold_logit_first_sing_italian.png",
    label="FIRST SING",
    top_n=3
)

avg_second_sing_italian, top_heads_second_sing_italian = compute_average_headwise_gold_logit_contributions(
    model=model,
    tokenizer=tokenizer,
    prompts=correct_prompt_texts_second_sing_italian,
    gold_token_ids=correct_answers_ids_second_sing_italian,
    save_path="avg_headwise_gold_logit_second_sing_italian.png",
    label="SECOND SING",
    top_n=3
)


#AVERAGE ATTENTION PATTERNS HTML

# Step 2: Visualize average attention patterns
#visualize_average_top_heads_attention(model=model, tokenizer=tokenizer, prompts=prompt_texts_first_sing_italian, top_heads=top_heads_first_sing_italian, save_path="top_heads_attention_avg_first_sing_italian.html", label="FIRST SING")

#visualize_average_top_heads_attention(model=model, tokenizer=tokenizer, prompts=prompt_texts_second_sing_italian, top_heads=top_heads_second_sing_italian, save_path="top_heads_attention_avg_second_sing_italian.html", label="SECOND SING")

#visualize_average_top_heads_attention(
#    model=model,
#    tokenizer=tokenizer,
#    prompts=correct_prompt_texts_first_sing_italian,
#    manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
#    save_path="manual_heads_attention_avg_first_sing_italian.html",
#    label="FIRST SING"
#)

#visualize_average_top_heads_attention(
#    model=model,
#    tokenizer=tokenizer,
#    prompts=correct_prompt_texts_second_sing_italian,
#    manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
#    save_path="manual_heads_attention_avg_second_sing_italian.html",
#    label="SECOND SING"
#)