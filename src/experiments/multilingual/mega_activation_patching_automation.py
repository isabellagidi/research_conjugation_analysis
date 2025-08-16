# === Imports ===
import os
import torch
from transformer_lens import HookedTransformer
from transformer_lens import patching
from src.utils.automation import save_heatmap, get_logit_diff, conjugation_metric
from src.utils.activation_patching_automation import activation_patching_automation
from transformers import AutoTokenizer

from src.utils.automation import (
    load_json_data,
    filter_conjugations,
    build_conjugation_prompts,
    accuracy_filter,
    group_by_token_lengths,
    generate_dataset
)

# === Model and Tokenizer Setup ===
model_name = "bigscience/bloom-1b1"
tl_model = HookedTransformer.from_pretrained(model_name)
device = tl_model.cfg.device

with open(os.path.expanduser("~/.huggingface/token")) as f:
    hf_token = f.read().strip()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("model:", tl_model)

# === Load and Prepare Dataset ===
file_path = "/home/lis.isabella.gidi/jsalt2025/src/datasets/morphy_net/MorphyNet_all_conjugations.json"
all_verbs = load_json_data(file_path)


#HERE IS THE BIG MAMA
activation_patching_automation(language = "spanish", conjugation_pair = ("first singular", "second singular"), all_verbs=all_verbs, output_root="automation_results", model_name = model_name, MAX_PROMPTS = 100)