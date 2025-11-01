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
from src.utils.spanish_dataset_generation import create_spanish_verbs


def filter_spanish_conjugations(spanish_verbs, tokenizer):
    spanish_conjugations = []
    print("Kept verb forms:\n")

    for verb, yo, tu, vtype, regularity in spanish_verbs:
        yo_tok = tokenizer.tokenize(" " + yo)
        tu_tok = tokenizer.tokenize(" " + tu)

        condition_1 = len(yo_tok) == 1 and len(tu_tok) == 1
        condition_2 = (
            len(yo_tok) == len(tu_tok) and
            yo_tok[:-1] == tu_tok[:-1]
        )

        if condition_1 or condition_2:
            spanish_conjugations.append((verb, yo, tu, vtype, regularity, len(yo_tok), len(tu_tok)))
            print(f"Infinitive: {verb}")
            print(f"  Yo: {yo} -> {yo_tok}")
            print(f"  Tú: {tu} -> {tu_tok}")
            print("-" * 40)

    return spanish_conjugations


model_name = "bigscience/bloom-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # required for BLOOM
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Build dataset
spanish_verbs = create_spanish_verbs(spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs)
spanish_conjugations = filter_spanish_conjugations(spanish_verbs, tokenizer)
