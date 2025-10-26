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

from src.utils.activation_analysis import visualize_average_top_heads_attention
from src.utils.dataset_preparation import load_json_data, filter_conjugations, build_conjugation_prompts, accuracy_filter_text
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



correct_prompt_texts_first_sing_spanish, correct_answers_ids_first_sing_spanish, _, \
incorrect_prompt_texts_first_sing_spanish, incorrect_answers_ids_first_sing_spanish, _ = accuracy_filter_text(
    prompts_trimmed_first_sing_spanish,
    answers_ids_first_sing_spanish,
    verb_entries_first_sing_spanish,
    model_name=None,           # not needed because we pass tokenizer/model
    tokenizer=tokenizer,
    model=model,
)

correct_prompt_texts_second_sing_spanish, correct_answers_ids_second_sing_spanish, _, \
incorrect_prompt_texts_second_sing_spanish, incorrect_answers_ids_second_sing_spanish, _ = accuracy_filter_text(
    prompts_trimmed_second_sing_spanish,
    answers_ids_second_sing_spanish,
    verb_entries_second_sing_spanish,
    model_name=None,
    tokenizer=tokenizer,
    model=model,
)

# Truncate to first 100 correct examples for testing
#correct_prompt_texts_first_sing_spanish = correct_prompt_texts_first_sing_spanish[:100]
#correct_answers_ids_first_sing_spanish = correct_answers_ids_first_sing_spanish[:100]


print(f"Correct: {len(correct_prompt_texts_first_sing_spanish)} / {len(verb_entries_first_sing_spanish)}")
print(f"Incorrect: {len(incorrect_prompt_texts_first_sing_spanish)}")

print("✅ Correct prompts:")
for prompt in correct_prompt_texts_first_sing_spanish[:3]:
    print(prompt)

print("\n❌ Incorrect prompts:")
for prompt in incorrect_prompt_texts_first_sing_spanish[:3]:
    print(prompt)

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


visualize_average_top_heads_attention(
    model=model,
    tokenizer=tokenizer,
    prompts=correct_prompt_texts_first_sing_spanish,
    manual_head_labels=["L17H5", "L19H3", "L19H4", "L20H5", "L21H2", "L22H1", "L23H0", "L23H2"],
    save_path="manual_heads_attention_avg_first_sing_spanish_red.html",
    label="FIRST SING"
)

visualize_average_top_heads_attention(
    model=model,
    tokenizer=tokenizer,
    prompts=correct_prompt_texts_second_sing_spanish,
    manual_head_labels=["L17H5", "L19H3", "L19H4", "L20H5", "L21H2", "L22H1", "L23H0", "L23H2"],
    save_path="manual_heads_attention_avg_second_sing_spanish_red.html",
    label="SECOND SING"
)

visualize_average_top_heads_attention(
    model=model,
    tokenizer=tokenizer,
    prompts=correct_prompt_texts_first_sing_spanish,
    manual_head_labels=["L18H10", "L18H11", "L19H10", "L19H11", "L20H11"],
    save_path="manual_heads_attention_avg_first_sing_spanish_burgundy.html",
    label="FIRST SING"
)

visualize_average_top_heads_attention(
    model=model,
    tokenizer=tokenizer,
    prompts=correct_prompt_texts_second_sing_spanish,
    manual_head_labels=["L18H10", "L18H11", "L19H10", "L19H11", "L20H11"],
    save_path="manual_heads_attention_avg_second_sing_spanish_burgundy.html",
    label="SECOND SING"
)



