import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.dataset_preparation import (
    load_json_data,
    filter_conjugations,
    build_conjugation_prompts,
    generate_first_singular_dataset_spanish,
    generate_first_singular_dataset_german,
)
from src.utils.evaluation import accuracy_filter

# Load model and tokenizer
model_name = "bigscience/bloom-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Load JSON data
file_path = "/home/lis.isabella.gidi/jsalt2025/src/datasets/morphy_net/MorphyNet_all_conjugations.json"
all_verbs = load_json_data(file_path)

# Prepare Spanish data
spanish_conjugations = filter_conjugations(all_verbs, tokenizer, "spa")[:500]
spanish_first_sing = generate_first_singular_dataset_spanish(spanish_conjugations)
prompts_spa, ids_spa, entries_spa = build_conjugation_prompts(tokenizer, spanish_first_sing, spanish_conjugations)
_, correct_ids_spa, _, _, _, _ = accuracy_filter(prompts_spa, ids_spa, entries_spa, model_name)

# Prepare German data
german_conjugations = filter_conjugations(all_verbs, tokenizer, "deu")[:500]
german_first_sing = generate_first_singular_dataset_german(german_conjugations)
prompts_deu, ids_deu, entries_deu = build_conjugation_prompts(tokenizer, german_first_sing, german_conjugations)
_, correct_ids_deu, _, _, _, _ = accuracy_filter(prompts_deu, ids_deu, entries_deu, model_name)

# Get embedding vectors
w_U = model.lm_head.weight  # (vocab_size, d_model)
spa_embed = w_U[correct_ids_spa[0]]
deu_embed = w_U[correct_ids_deu[0]]
cos_sim = torch.nn.functional.cosine_similarity(spa_embed, deu_embed, dim=0)

# Output
print(cos_sim.item())

