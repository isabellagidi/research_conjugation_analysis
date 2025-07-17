from transformers import AutoTokenizer

import sys
sys.path.append('../../')  

from src.datasets.spanish.spanish_verbs import spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs
from jsalt2025.src.utils.spanish_dataset_generation import create_spanish_verbs, filter_spanish_conjugations


tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")

spanish_verbs = create_spanish_verbs(spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs)

spanish_conjugations = filter_spanish_conjugations(spanish_verbs, tokenizer)

def build_conjugation_prompts(tokenizer, spanish_yo, spanish_tu, spanish_conjugations):
    prompts_trimmed = []
    answers_ids = []
    matched_entries = []

    for prompt, entry in zip(spanish_yo + spanish_tu, spanish_conjugations):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) < 2:
            continue

        prompt_trimmed = token_ids[:-1]
        answer_id = token_ids[-1]

        prompts_trimmed.append(prompt_trimmed)
        answers_ids.append(answer_id)
        matched_entries.append(entry)

    return prompts_trimmed, answers_ids, matched_entries


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
prompts_trimmed_a, answers_ids_a, verb_entries_a = build_conjugation_prompts(tokenizer, spanish_yo_a, spanish_tu_a, spanish_conjugations)

#b
prompts_trimmed_b, answers_ids_b, verb_entries_b = build_conjugation_prompts(tokenizer, spanish_yo_b, spanish_tu_b, spanish_conjugations)

#c
prompts_trimmed_c, answers_ids_c, verb_entries_c = build_conjugation_prompts(tokenizer, spanish_yo_c, spanish_tu_c, spanish_conjugations)

#d
prompts_trimmed_d, answers_ids_d, verb_entries_d = build_conjugation_prompts(tokenizer, spanish_yo_d, spanish_tu_d, spanish_conjugations)

#e
prompts_trimmed_e, answers_ids_e, verb_entries_e = build_conjugation_prompts(tokenizer, spanish_yo_e, spanish_tu_e, spanish_conjugations)

#f
prompts_trimmed_f, answers_ids_f, verb_entries_f = build_conjugation_prompts(tokenizer, spanish_yo_f, spanish_tu_f, spanish_conjugations)

#g
prompts_trimmed_g, answers_ids_g, verb_entries_g = build_conjugation_prompts(tokenizer, spanish_yo_g, spanish_tu_g, spanish_conjugations)

#h
prompts_trimmed_h, answers_ids_h, verb_entries_h = build_conjugation_prompts(tokenizer, spanish_yo_h, spanish_tu_h, spanish_conjugations)


PROJECT_ROOT = Path(__file__).resolve().parents[2]   # …/jsalt2025
sys.path.append(str(PROJECT_ROOT))                   

# ---------------------------------------------------------------
# Evaluate the model on every prompt → collect accuracy & average P(gold)
# ---------------------------------------------------------------

import torch
from collections import defaultdict
import csv

suffixes = ["a", "b", "c", "d", "e", "f", "g", "h"]
results = {}

total_prompts = sum(len(globals()[f"prompts_trimmed_{s}"]) for s in suffixes)
print(f"🔍 Loaded {total_prompts} prompts for evaluation across all formats")

# Detailed stats per suffix → verb type → token length
detailed_stats = {
    s: {
        'ar': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'er': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'ir': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'regularity': {
            'reg': {'total': 0, 'correct': 0},
            'irreg': {'total': 0, 'correct': 0}
        }
    }
    for s in suffixes
}

for suffix in suffixes:
    prompts = globals()[f"prompts_trimmed_{suffix}"]
    answers = globals()[f"answers_ids_{suffix}"]
    verb_entries = globals()[f"verb_entries_{suffix}"]

    total = len(prompts)
    correct_top1 = 0
    sum_gold_probs = 0.0

    yo_total = 0
    yo_correct = 0
    yo_sum_probs = 0.0

    tu_total = 0
    tu_correct = 0
    tu_sum_probs = 0.0

    for idx, (prompt_ids, gold_id) in enumerate(zip(prompts, answers)):
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
        
        # Determine whether the prompt came from a 'yo' or 'tú' sentence
        is_yo = idx < len(prompts) // 2

        if is_yo:
            yo_total += 1
            yo_sum_probs += gold_prob
            if top_id == gold_id:
                yo_correct += 1
        else:
            tu_total += 1
            tu_sum_probs += gold_prob
            if top_id == gold_id:
                tu_correct += 1


        verb_entry = verb_entries[idx]
        vtype = verb_entry[3]         # 'ar', 'er', 'ir'
        yo_tok_len = verb_entry[5]    # number of tokens in 'yo' form
        regularity = verb_entry[4]    # 'reg' or 'irreg'


        if top_id == gold_id:
            detailed_stats[suffix][vtype][yo_tok_len]['correct'] += 1
            detailed_stats[suffix]['regularity'][regularity]['correct'] += 1
        
        detailed_stats[suffix][vtype][yo_tok_len]['total'] += 1
        detailed_stats[suffix]['regularity'][regularity]['total'] += 1

    results[suffix] = {
        "total": total,
        "correct_top1": correct_top1,
        "accuracy": correct_top1 / total if total > 0 else 0.0,
        "avg_gold_prob": sum_gold_probs / total if total > 0 else 0.0
    }

# Summary print + one example prompt per format
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

    print("\n📊 Accuracy by verb type and token length:")
    for vtype in ["ar", "er", "ir"]:
        for tok_len, stats in sorted(detailed_stats[s][vtype].items()):
            total = stats['total']
            correct = stats['correct']
            acc = correct / total if total > 0 else 0.0
            print(f"  {vtype.upper()} verbs (tokens: {tok_len}): {acc:.2%} ({correct}/{total})")
    

    # 📊 Average by verb type (all token lengths)
    print("\n📊 Average accuracy by verb type (all token lengths):")
    for vtype in ["ar", "er", "ir"]:
        total = sum(stats["total"] for stats in detailed_stats[s][vtype].values())
        correct = sum(stats["correct"] for stats in detailed_stats[s][vtype].values())
        acc = correct / total if total > 0 else 0.0
        print(f"  {vtype.upper()} verbs: {acc:.2%} ({correct}/{total})")

    # 📊 Average by token length (aggregated across all verb types)
    print("\n📊 Average accuracy by token length (aggregated across all verb types):")
    token_lengths = set()
    for vtype in ["ar", "er", "ir"]:
        token_lengths.update(detailed_stats[s][vtype].keys())

    for tok_len in sorted(token_lengths):
        total = 0
        correct = 0
        for vtype in ["ar", "er", "ir"]:
            total += detailed_stats[s][vtype][tok_len]["total"]
            correct += detailed_stats[s][vtype][tok_len]["correct"]
        acc = correct / total if total > 0 else 0.0
        print(f"  Token length {tok_len}: {acc:.2%} ({correct}/{total})")
    
    print(f"📈 Avg P(gold): {r['avg_gold_prob']:.2%}")

    print("\n📊 Accuracy by regularity:")
    for reg_type in ["reg", "irreg"]:
        total = detailed_stats[s]['regularity'][reg_type]['total']
        correct = detailed_stats[s]['regularity'][reg_type]['correct']
        acc = correct / total if total > 0 else 0.0
        print(f"  {reg_type.capitalize()} verbs: {acc:.2%} ({correct}/{total})")


    if yo_total > 0:
        print(f"👤 YO Accuracy: {yo_correct / yo_total:.2%} ({yo_correct}/{yo_total})")
        print(f"👤 YO Avg P(gold): {yo_sum_probs / yo_total:.2%}")
    if tu_total > 0:
        print(f"🧑‍🦰 TÚ Accuracy: {tu_correct / tu_total:.2%} ({tu_correct}/{tu_total})")
        print(f"🧑‍🦰 TÚ Avg P(gold): {tu_sum_probs / tu_total:.2%}")
