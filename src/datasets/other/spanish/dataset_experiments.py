
from transformers import AutoTokenizer

import sys
sys.path.append('../../')  

from src.datasets.spanish.spanish_verbs import spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs
from src.utils.spanish_build_prompts import build_conjugation_prompts
from src.utils.spanish_dataset_generation import create_spanish_verbs, filter_spanish_conjugations

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")


spanish_verbs = create_spanish_verbs(spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs)

spanish_conjugations = filter_spanish_conjugations(spanish_verbs, tokenizer)


def generate_yo_dataset_a(spanish_conjugations):
    spanish_yo = [f"Conjugación del verbo {verb} en presente: Yo {yo_form}" for verb, yo_form, _, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset_a(spanish_conjugations):
    spanish_tu = [f"Conjugación del verbo {verb} en presente: Tú {tu_form}" for verb, _, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu


def generate_yo_dataset_b(spanish_conjugations):
    spanish_yo = [f"Conjugación del verbo {verb} en presente en Español: Yo {yo_form}" for verb, yo_form, _, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset_b(spanish_conjugations):
    spanish_tu = [f"Conjugación del verbo {verb} en presente en Español: Tú {tu_form}" for verb, _, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu

def generate_yo_dataset_c(spanish_conjugations):
    spanish_yo = [f"Conjugación del verbo {verb} en presente en Español: {verb}: Yo {yo_form}" for verb, yo_form, _, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset_c(spanish_conjugations):
    spanish_tu = [f"Conjugación del verbo {verb} en presente en Español: {verb}: Tú {tu_form}" for verb, _, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu

def generate_yo_dataset_d(spanish_conjugations):
    spanish_yo = [f"Conjugación del verbo en presente en Español: {verb}: Yo {yo_form}, Yo {yo_form}" for verb, yo_form, _, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset_d(spanish_conjugations):
    spanish_tu = [f"Conjugación del verbo en presente en Español: {verb}: Tú {tu_form}, Tú {tu_form}" for verb, _, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu


def generate_yo_dataset_e(spanish_conjugations):
    spanish_yo = [f"Conjugación del verbo en presente en Español: {verb}: Yo {yo_form}, Tú {tu_form}, Yo {yo_form}" for verb, yo_form, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset_e(spanish_conjugations):
    spanish_tu = [f"Conjugación del verbo en presente en Español: {verb}: Tú {tu_form}, Yo {yo_form}, Tú {tu_form}" for verb, yo_form, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu


def generate_yo_dataset_f(spanish_conjugations):
    spanish_yo = [f"Conjugación del verbo {verb} en presente en Español: {verb}: Tú {tu_form}, Yo {yo_form}" for verb, yo_form, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset_f(spanish_conjugations):
    spanish_tu = [f"Conjugación del verbo {verb} en presente en Español: {verb}: Yo {yo_form}, Tú {tu_form}" for verb, yo_form, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu


def generate_yo_dataset_g(spanish_conjugations):
    spanish_yo = [f"Conjugación del verbo en presente en Español: {verb}: Tú {tu_form}, Yo {yo_form}" for verb, yo_form, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset_g(spanish_conjugations):
    spanish_tu = [f"Conjugación del verbo en presente en Español: {verb}: Yo {yo_form}, Tú {tu_form}" for verb, yo_form, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu

def generate_yo_dataset_h(spanish_conjugations):
    spanish_yo = [f"Conjugación del verbo {verb} en presente: {verb}: Yo siempre {yo_form}" for verb, yo_form, _, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset_h(spanish_conjugations):
    spanish_tu = [f"Conjugación del verbo {verb} en presente: {verb}: Tú siempre {tu_form}" for verb, _, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu

def generate_yo_dataset_i(spanish_conjugations):
    spanish_yo = [f"Conjugación del verbo {verb} en presente: Yo {yo_form}" for verb, yo_form, _, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset_i(spanish_conjugations):
    spanish_tu = [f"Conjugación del verbo {verb}: Tú {tu_form}" for verb, _, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu

def generate_yo_dataset_j(spanish_conjugations):
    spanish_yo = [f"Conjugación de {verb}: Yo {yo_form}" for verb, yo_form, _, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset_j(spanish_conjugations):
    spanish_tu = [f"Conjugación de {verb}: Tú {tu_form}" for verb, _, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu



#a
spanish_yo_a = generate_yo_dataset_a(spanish_conjugations)

spanish_tu_a = generate_tu_dataset_a(spanish_conjugations)


#b
spanish_yo_b = generate_yo_dataset_b(spanish_conjugations)

spanish_tu_b = generate_tu_dataset_b(spanish_conjugations)


#c
spanish_yo_c = generate_yo_dataset_c(spanish_conjugations)

spanish_tu_c = generate_tu_dataset_c(spanish_conjugations)


#d
spanish_yo_d = generate_yo_dataset_d(spanish_conjugations)

spanish_tu_d = generate_tu_dataset_d(spanish_conjugations)


#e
spanish_yo_e = generate_yo_dataset_e(spanish_conjugations)

spanish_tu_e = generate_tu_dataset_e(spanish_conjugations)


#f
spanish_yo_f = generate_yo_dataset_f(spanish_conjugations)

spanish_tu_f = generate_tu_dataset_f(spanish_conjugations)

#g
spanish_yo_g = generate_yo_dataset_g(spanish_conjugations)

spanish_tu_g = generate_tu_dataset_g(spanish_conjugations)

#h
spanish_yo_h = generate_yo_dataset_h(spanish_conjugations)

spanish_tu_h = generate_tu_dataset_h(spanish_conjugations)

#i
spanish_yo_i = generate_yo_dataset_i(spanish_conjugations)

spanish_tu_i = generate_tu_dataset_i(spanish_conjugations)

#j
spanish_yo_j = generate_yo_dataset_j(spanish_conjugations)

spanish_tu_j = generate_tu_dataset_j(spanish_conjugations)


import os, sys, json, random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pathlib import Path
import sys

with open(os.path.expanduser("~/.huggingface/token")) as f:
    hf_token = f.read().strip()

model_name = "bigscience/bloom-1b1"
#model_name = "HPLT/sft-fpft-es-bloom-7b1"
#model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

#a
prompts_trimmed_a, answers_ids_a, _ = build_conjugation_prompts(tokenizer, spanish_yo_a, spanish_tu_a)

#b
prompts_trimmed_b, answers_ids_b, _ = build_conjugation_prompts(tokenizer, spanish_yo_b, spanish_tu_b)

#c
prompts_trimmed_c, answers_ids_c, _ = build_conjugation_prompts(tokenizer, spanish_yo_c, spanish_tu_c)

#d
prompts_trimmed_d, answers_ids_d, _ = build_conjugation_prompts(tokenizer, spanish_yo_d, spanish_tu_d)

#e
prompts_trimmed_e, answers_ids_e, _ = build_conjugation_prompts(tokenizer, spanish_yo_e, spanish_tu_e)

#f
prompts_trimmed_f, answers_ids_f, _ = build_conjugation_prompts(tokenizer, spanish_yo_f, spanish_tu_f)

#g
prompts_trimmed_g, answers_ids_g, _ = build_conjugation_prompts(tokenizer, spanish_yo_g, spanish_tu_g)

#h
prompts_trimmed_h, answers_ids_h, _ = build_conjugation_prompts(tokenizer, spanish_yo_h, spanish_tu_h)

#i
prompts_trimmed_i, answers_ids_i, _ = build_conjugation_prompts(tokenizer, spanish_yo_i, spanish_tu_i)

#j
prompts_trimmed_j, answers_ids_j, _ = build_conjugation_prompts(tokenizer, spanish_yo_j, spanish_tu_j)


PROJECT_ROOT = Path(__file__).resolve().parents[2]   # …/jsalt2025
sys.path.append(str(PROJECT_ROOT))                   

# ---------------------------------------------------------------
# Evaluate the model on every prompt → collect accuracy & average P(gold)
# ---------------------------------------------------------------
import torch

suffixes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
results = {}

total_prompts = sum(len(globals()[f"prompts_trimmed_{s}"]) for s in suffixes)
print(f"🔍 Loaded {total_prompts} prompts for evaluation across all formats")


for suffix in suffixes:
    prompts = globals()[f"prompts_trimmed_{suffix}"]
    answers = globals()[f"answers_ids_{suffix}"]

    total = len(prompts)
    correct_top1 = 0
    sum_gold_probs = 0.0

    for prompt_ids, gold_id in zip(prompts, answers):
        input_ids = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            logits = model(input_ids).logits

        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        gold_prob = probs[gold_id].item()
        sum_gold_probs += gold_prob

        top_id = int(torch.argmax(probs))
        if top_id == gold_id:
            correct_top1 += 1

    results[suffix] = {
        "total": total,
        "correct_top1": correct_top1,
        "accuracy": correct_top1 / total if total > 0 else 0.0,
        "avg_gold_prob": sum_gold_probs / total if total > 0 else 0.0
    }

for s in suffixes:
    prompts = globals()[f"prompts_trimmed_{s}"]
    answers = globals()[f"answers_ids_{s}"]
    decoded_prompt = tokenizer.decode(prompts[0], skip_special_tokens=True)
    gold_token = tokenizer.decode([answers[0]])

    print(f"\n📝 Example for dataset {s.upper()}:")
    print(f"Prompt: {decoded_prompt}")
    print(f"Expected next token: '{gold_token}'")

    r = results[s]
    print(f"📊 Results for format {s.upper()}:")
    print(f"🎯 Accuracy: {r['accuracy']:.2%} ({r['correct_top1']}/{r['total']})")
    print(f"📈 Avg P(gold): {r['avg_gold_prob']:.2%}")