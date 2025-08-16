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


from src.utils.dataset_preparation import load_json_data, filter_conjugations, build_conjugation_prompts, accuracy_filter, group_by_token_lengths
from src.utils.dataset_preparation import generate_first_singular_dataset_spanish, generate_second_singular_dataset_spanish



with open(os.path.expanduser("~/.huggingface/token")) as f:
    hf_token = f.read().strip()

# Load tokenizer and model
model_name = "bigscience/bloom-1b1"
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

spanish_conjugations = filter_conjugations(all_verbs, tokenizer, "spa")

#DOING THIS TO SAVE RAM!!!!!!!
spanish_conjugations = spanish_conjugations[:500]
print("length of dataset (usually 500 but in this case):" , len(spanish_conjugations))

spanish_first_singular = generate_first_singular_dataset_spanish(spanish_conjugations)
spanish_second_singular = generate_second_singular_dataset_spanish(spanish_conjugations)

print("Example prompt input:", spanish_first_singular[0])
print("Type:", type(spanish_first_singular[0]))

print("Example prompt input:", spanish_second_singular[0])
print("Type:", type(spanish_second_singular[0]))


prompts_trimmed_first_sing_spanish, answers_ids_first_sing_spanish, verb_entries_first_sing_spanish = build_conjugation_prompts(tokenizer, spanish_first_singular, spanish_conjugations)
prompts_trimmed_second_sing_spanish, answers_ids_second_sing_spanish, verb_entries_second_sing_spanish = build_conjugation_prompts(tokenizer, spanish_second_singular, spanish_conjugations)


prompt_texts_first_sing_spanish = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_first_sing_spanish]
prompt_texts_second_sing_spanish = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed_second_sing_spanish]


correct_prompt_texts_first_sing_spanish, correct_answers_ids_first_sing_spanish, _, incorrect_prompt_texts_first_sing_spanish, incorrect_answers_ids_first_sing_spanish, _ = accuracy_filter(
    prompts_trimmed_first_sing_spanish,
    answers_ids_first_sing_spanish,
    verb_entries_first_sing_spanish,
    model_name
)



print(f"Correct: {len(correct_prompt_texts_first_sing_spanish)} / {len(verb_entries_first_sing_spanish)}")
print(f"Incorrect: {len(incorrect_prompt_texts_first_sing_spanish)}")

print("✅ Correct prompts:")
for prompt in correct_prompt_texts_first_sing_spanish[:3]:
    print(prompt)

print("\n❌ Incorrect prompts:")
for prompt in incorrect_prompt_texts_first_sing_spanish[:3]:
    print(prompt)

correct_prompt_texts_second_sing_spanish, correct_answers_ids_second_sing_spanish, correct_entries_first_sing_spanish, incorrect_prompt_texts_second_sing_spanish, incorrect_answers_ids_second_sing_spanish, correct_entries_second_sing_spanish = accuracy_filter(
    prompts_trimmed_second_sing_spanish,
    answers_ids_second_sing_spanish,
    verb_entries_second_sing_spanish,
    model_name
)

print("number of prompts saved:", len(correct_prompt_texts_second_sing_spanish))
# Truncate to first few hundred correct examples for testing
#correct_prompt_texts_second_sing_spanish = correct_prompt_texts_second_sing_spanish[:200]
#correct_answers_ids_second_sing_spanish = correct_answers_ids_second_sing_spanish[:200]

print(f"Correct: {len(correct_prompt_texts_second_sing_spanish)} / {len(verb_entries_second_sing_spanish)}")
print(f"Incorrect: {len(incorrect_prompt_texts_second_sing_spanish)}")

print("✅ Correct prompts:")
for prompt in correct_prompt_texts_second_sing_spanish[:3]:
    print(prompt)

print("\n❌ Incorrect prompts:")
for prompt in incorrect_prompt_texts_second_sing_spanish[:3]:
    print(prompt)


#GROUPING BY TOKEN LENGTH

#FIRST SING
grouped_examples_first_sing_spanish = group_by_token_lengths(
    correct_prompt_texts_first_sing_spanish,
    correct_answers_ids_first_sing_spanish,
    correct_entries_first_sing_spanish,  # <-- from accuracy_filter
    tokenizer
)

# Print how many examples in each group
for k, v in sorted(grouped_examples_first_sing_spanish.items()):
    print(f"Infinitive tokens: {k[0]}, Conjugated tokens: {k[1]} --> {len(v)} examples")


#CREATING ARRAYS OF TOP 5 (or whatever you want)
top_5_tokenizations_first_sing_spanish = sorted(grouped_examples_first_sing_spanish.items(), key=lambda x: len(x[1]), reverse=True)[:5]


print("\n🔢 Top 5 tokenization groups by frequency for first sing Spanish:")
for (inf_len, conj_len), group in top_5_tokenizations_first_sing_spanish:
    print(f" - (inf={inf_len}, conj={conj_len}): {len(group)} examples")

# Run attribution + attention visualizations
for (inf_len, conj_len), examples in top_5_tokenizations_first_sing_spanish:
    # Extract example components
    prompt_texts = [ex[0] for ex in examples]
    answer_ids = [ex[1] for ex in examples]
    verb_entries = [ex[2] for ex in examples]

    # Define unique labels and filenames
    tokenization_label = f"inf{inf_len}_conj{conj_len}"
    base_label = f"FIRST SING ({tokenization_label})"
    logit_save_path = f"avg_headwise_gold_logit_first_sing_spanish_{tokenization_label}.png"
    attention_save_path = f"top_heads_attention_avg_first_sing_spanish_{tokenization_label}.html"

    # === AVERAGE HEADWISE LOGIT ATTRIBUTION ===
    #print(f"\n🧠 Computing headwise logit attribution for {base_label}")
    #avg_logits, top_heads = compute_average_headwise_gold_logit_contributions(
    #    model=model,
    #    tokenizer=tokenizer,
    #    prompts=prompt_texts,
    #    gold_token_ids=answer_ids,
    #    save_path=logit_save_path,
    #    label=base_label,
    #    top_n=3
    #)

    # === AVERAGE ATTENTION PATTERNS (HTML) ===
    print(f"📊 Visualizing top-head attention patterns for {base_label}")
    visualize_average_top_heads_attention(
        model=model,
        tokenizer=tokenizer,
        prompts=prompt_texts,
        #manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
        manual_head_labels=["L20H5", "L17H5", "L19H3", "L19H4"],
        save_path=attention_save_path,
        label=base_label
    )


#SECOND SING

grouped_examples_second_sing_spanish = group_by_token_lengths(
    correct_prompt_texts_second_sing_spanish,
    correct_answers_ids_second_sing_spanish,
    correct_entries_second_sing_spanish,  # <-- from accuracy_filter
    tokenizer
)

# Print how many examples in each group
for k, v in sorted(grouped_examples_second_sing_spanish.items()):
    print(f"Infinitive tokens: {k[0]}, Conjugated tokens: {k[1]} --> {len(v)} examples")


#CREATING ARRAYS OF TOP 5 (or whatever you want)
top_5_tokenizations_second_sing_spanish = sorted(grouped_examples_second_sing_spanish.items(), key=lambda x: len(x[1]), reverse=True)[:5]


print("\n🔢 Top 5 tokenization groups by frequency for second sing Spanish:")
for (inf_len, conj_len), group in top_5_tokenizations_second_sing_spanish:
    print(f" - (inf={inf_len}, conj={conj_len}): {len(group)} examples")

# Run attribution + attention visualizations
for (inf_len, conj_len), examples in top_5_tokenizations_second_sing_spanish:
    # Extract example components
    prompt_texts = [ex[0] for ex in examples]
    answer_ids = [ex[1] for ex in examples]
    verb_entries = [ex[2] for ex in examples]

    # Define unique labels and filenames
    tokenization_label = f"inf{inf_len}_conj{conj_len}"
    base_label = f"SECOND SING ({tokenization_label})"
    logit_save_path = f"avg_headwise_gold_logit_second_sing_spanish_{tokenization_label}.png"
    attention_save_path = f"manual_heads_attention_avg_second_sing_spanish_{tokenization_label}.html"

    # === AVERAGE HEADWISE LOGIT ATTRIBUTION ===
    #print(f"\n🧠 Computing headwise logit attribution for {base_label}")
    #avg_logits, top_heads = compute_average_headwise_gold_logit_contributions(
    #    model=model,
    #    tokenizer=tokenizer,
    #    prompts=prompt_texts,
    #    gold_token_ids=answer_ids,
    #    save_path=logit_save_path,
    #    label=base_label,
    #    top_n=3
    #)

    # === AVERAGE ATTENTION PATTERNS (HTML) ===
    print(f"📊 Visualizing manual top-head attention patterns for {base_label}")
    visualize_average_top_heads_attention(
        model=model,
        tokenizer=tokenizer,
        prompts=prompt_texts,
        #manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
        manual_head_labels=["L20H5", "L17H5", "L19H3", "L19H4"],
        save_path=attention_save_path,
        label=base_label
    )
















#HEADWISE LOGIT ATTRIBUTION


# EXAMPLE HEADWISE LOGIT ATTRIBUTION
#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_first_sing_spanish[0], gold_token_id=correct_answers_ids_first_sing_spanish[0], save_path="headwise_gold_logit_first_sing_spanish0.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_second_sing_spanish[0], gold_token_id=correct_answers_ids_second_sing_spanish[0], save_path="headwise_gold_logit_second_sing_spanish0.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_first_sing_spanish[1], gold_token_id=correct_answers_ids_first_sing_spanish[1], save_path="headwise_gold_logit__first_sing_spanish1.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_second_sing_spanish[1], gold_token_id=correct_answers_ids_second_sing_spanish[1], save_path="headwise_gold_logit_second_sing_spanish1.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_first_sing_spanish[2], gold_token_id=correct_answers_ids_first_sing_spanish[2], save_path="headwise_gold_logit_first_sing_spanish2.png")

#compute_headwise_gold_logit_contributions(model, tokenizer, prompt=correct_prompt_texts_second_sing_spanish[2], gold_token_id=correct_answers_ids_second_sing_spanish[2], save_path="headwise_gold_logit_second_sing_spanish2.png")

# AVERAGE HEADWISE LOGIT ATTRIBUTION
#automatic
#avg_logits_first_sing_spanish, top_heads_first_sing_spanish = compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=correct_prompt_texts_first_sing_spanish, gold_token_ids=correct_answers_ids_first_sing_spanish, save_path="avg_headwise_gold_logit_first_sing_spanish.png", label="FIRST SING")

#avg_logits_second_sing_spanish, top_heads_second_sing_spanish = compute_average_headwise_gold_logit_contributions(model=model, tokenizer=tokenizer, prompts=correct_prompt_texts_second_sing_spanish, gold_token_ids=correct_answers_ids_second_sing_spanish, save_path="avg_headwise_gold_logit_second_sing_spanish.png", label="SECOND SING")

#top k
#avg_logits_first_sing_spanish, top_heads_first_sing_spanish = compute_average_headwise_gold_logit_contributions( model=model, tokenizer=tokenizer, prompts=correct_prompt_texts_first_sing_spanish, gold_token_ids=correct_answers_ids_first_sing_spanish, save_path="avg_headwise_gold_logit_first_sing_spanish.png", label="FIRST SING", top_n=3)

#avg_second_sing_spanish, top_heads_second_sing_spanish = compute_average_headwise_gold_logit_contributions( model=model, tokenizer=tokenizer, prompts=correct_prompt_texts_second_sing_spanish, gold_token_ids=correct_answers_ids_second_sing_spanish, save_path="avg_headwise_gold_logit_second_sing_spanish.png", label="SECOND SING", top_n=3)


#AVERAGE ATTENTION PATTERNS HTML

# Step 2: Visualize average attention patterns

#automatic (top k)

#visualize_average_top_heads_attention(model=model, tokenizer=tokenizer, prompts=prompt_texts_first_sing_spanish, top_heads=top_heads_first_sing_spanish, save_path="top_heads_attention_avg_first_sing_spanish.html", label="FIRST SING")

#visualize_average_top_heads_attention(model=model, tokenizer=tokenizer, prompts=prompt_texts_second_sing_spanish, top_heads=top_heads_second_sing_spanish, save_path="top_heads_attention_avg_second_sing_spanish.html", label="SECOND SING")

#manual

#visualize_average_top_heads_attention(
#    model=model,
#    tokenizer=tokenizer,
#    prompts=correct_prompt_texts_first_sing_spanish,
#    manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
#    save_path="manual_heads_attention_avg_first_sing_spanish.html",
#    label="FIRST SING"
#)

#visualize_average_top_heads_attention(
#    model=model,
#    tokenizer=tokenizer,
#    prompts=correct_prompt_texts_second_sing_spanish,
#    manual_head_labels=["L23H14", "L23H1", "L22H1", "L21H1", "L8H14", "L9H14", "L10H14"],
#    save_path="manual_heads_attention_avg_second_sing_spanish.html",
#    label="SECOND SING"
#)