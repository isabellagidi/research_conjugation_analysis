
import os, sys, json, random
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pathlib import Path
import sys

from src.datasets.morphy_net.utils_dataset_preparation import load_json_data, filter_conjugations, build_conjugation_prompts
from src.datasets.morphy_net.utils_dataset_preparation import (
    generate_first_singular_dataset_spanish,
    generate_second_singular_dataset_spanish,
    generate_first_singular_dataset_italian,
    generate_second_singular_dataset_italian,
    generate_first_singular_dataset_german,
    generate_second_singular_dataset_german,
    generate_first_singular_dataset_english,
    generate_second_singular_dataset_english,
    generate_first_singular_dataset_spanish_b,
    generate_second_singular_dataset_spanish_b,
    generate_first_singular_dataset_italian_b,
    generate_second_singular_dataset_italian_b,
    generate_first_singular_dataset_german_b,
    generate_second_singular_dataset_german_b,
    generate_first_singular_dataset_english_b,
    generate_second_singular_dataset_english_b,
    generate_first_singular_dataset_spanish_d,
    generate_second_singular_dataset_spanish_d,
    generate_first_singular_dataset_italian_d,
    generate_second_singular_dataset_italian_d,
    generate_first_singular_dataset_german_d,
    generate_second_singular_dataset_german_d,
    generate_first_singular_dataset_english_d,
    generate_second_singular_dataset_english_d,
    generate_first_singular_dataset_spanish_e,
    generate_second_singular_dataset_spanish_e,
    generate_first_singular_dataset_italian_e,
    generate_second_singular_dataset_italian_e,
    generate_first_singular_dataset_german_e,
    generate_second_singular_dataset_german_e,
    generate_first_singular_dataset_english_e,
    generate_second_singular_dataset_english_e
)


with open(os.path.expanduser("~/.huggingface/token")) as f:
    hf_token = f.read().strip()

#model_name = "bigscience/bloom-1b1"
#model_name = "bigscience/bloom-1b7"
#model_name = "HPLT/sft-fpft-es-bloom-7b1"
#model_name = "bigscience/bloom-7b1"
#model_name = "bigscience/bloom-560m"
model_name = "bigscience/bloom-3b"

print(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Load your JSON data
file_path = "MorphyNet_all_conjugations.json"
all_verbs = load_json_data(file_path)

#to-do: filter conjugations, generate dataset, build conjugation prompts, then run evaluation (need to automate)
# then choose best dataset, etc and write code to filter out the wrong ones!!
# also need to make all functions compatible with new json format!

# Filter conjugations for the specified language (e.g., "cat" for Catalan)
english_conjugations = filter_conjugations(all_verbs, tokenizer, "eng")

spanish_conjugations = filter_conjugations(all_verbs, tokenizer, "spa")

german_conjugations = filter_conjugations(all_verbs, tokenizer, "deu")

italian_conjugations = filter_conjugations(all_verbs, tokenizer, "ita")


# Spanish
spanish_first_singular = generate_first_singular_dataset_spanish(spanish_conjugations)
spanish_second_singular = generate_second_singular_dataset_spanish(spanish_conjugations)

# Italian
italian_first_singular = generate_first_singular_dataset_italian(italian_conjugations)
italian_second_singular = generate_second_singular_dataset_italian(italian_conjugations)

# German
german_first_singular = generate_first_singular_dataset_german(german_conjugations)
german_second_singular = generate_second_singular_dataset_german(german_conjugations)

# English
english_first_singular = generate_first_singular_dataset_english(english_conjugations)
english_second_singular = generate_second_singular_dataset_english(english_conjugations)

# Spanish Dataset B
spanish_first_singular_b = generate_first_singular_dataset_spanish_b(spanish_conjugations)
spanish_second_singular_b = generate_second_singular_dataset_spanish_b(spanish_conjugations)

# Italian Dataset B
italian_first_singular_b = generate_first_singular_dataset_italian_b(italian_conjugations)
italian_second_singular_b = generate_second_singular_dataset_italian_b(italian_conjugations)

# German Dataset B
german_first_singular_b = generate_first_singular_dataset_german_b(german_conjugations)
german_second_singular_b = generate_second_singular_dataset_german_b(german_conjugations)

# English Dataset B
english_first_singular_b = generate_first_singular_dataset_english_b(english_conjugations)
english_second_singular_b = generate_second_singular_dataset_english_b(english_conjugations)

# Spanish Dataset D
spanish_first_singular_d = generate_first_singular_dataset_spanish_d(spanish_conjugations)
spanish_second_singular_d = generate_second_singular_dataset_spanish_d(spanish_conjugations)

# Italian Dataset D
italian_first_singular_d = generate_first_singular_dataset_italian_d(italian_conjugations)
italian_second_singular_d = generate_second_singular_dataset_italian_d(italian_conjugations)

# German Dataset D
german_first_singular_d = generate_first_singular_dataset_german_d(german_conjugations)
german_second_singular_d = generate_second_singular_dataset_german_d(german_conjugations)

# English Dataset D
english_first_singular_d = generate_first_singular_dataset_english_d(english_conjugations)
english_second_singular_d = generate_second_singular_dataset_english_d(english_conjugations)

# Spanish Dataset E
spanish_first_singular_e = generate_first_singular_dataset_spanish_e(spanish_conjugations)
spanish_second_singular_e = generate_second_singular_dataset_spanish_e(spanish_conjugations)

# Italian Dataset E
italian_first_singular_e = generate_first_singular_dataset_italian_e(italian_conjugations)
italian_second_singular_e = generate_second_singular_dataset_italian_e(italian_conjugations)

# German Dataset E
german_first_singular_e = generate_first_singular_dataset_german_e(german_conjugations)
german_second_singular_e = generate_second_singular_dataset_german_e(german_conjugations)

# English Dataset E
english_first_singular_e = generate_first_singular_dataset_english_e(english_conjugations)
english_second_singular_e = generate_second_singular_dataset_english_e(english_conjugations)

#TEST
print("Example prompt input:", english_first_singular[0])
print("Type:", type(english_first_singular[0]))

print("Example prompt input:", english_second_singular[0])
print("Type:", type(english_second_singular[0]))

# Call the function for each dataset type and language

# Spanish (First Person Singular)
prompts_trimmed_a_spanish_first_sing, answers_ids_a_spanish_first_sing, verb_entries_a_spanish_first_sing = build_conjugation_prompts(tokenizer, spanish_first_singular, spanish_conjugations)
prompts_trimmed_b_spanish_first_sing, answers_ids_b_spanish_first_sing, verb_entries_b_spanish_first_sing = build_conjugation_prompts(tokenizer, spanish_first_singular_b, spanish_conjugations)
prompts_trimmed_d_spanish_first_sing, answers_ids_d_spanish_first_sing, verb_entries_d_spanish_first_sing = build_conjugation_prompts(tokenizer, spanish_first_singular_d, spanish_conjugations)
prompts_trimmed_e_spanish_first_sing, answers_ids_e_spanish_first_sing, verb_entries_e_spanish_first_sing = build_conjugation_prompts(tokenizer, spanish_first_singular_e, spanish_conjugations)

# Spanish (Second Person Singular)
prompts_trimmed_a_spanish_second_sing, answers_ids_a_spanish_second_sing, verb_entries_a_spanish_second_sing = build_conjugation_prompts(tokenizer, spanish_second_singular, spanish_conjugations)
prompts_trimmed_b_spanish_second_sing, answers_ids_b_spanish_second_sing, verb_entries_b_spanish_second_sing = build_conjugation_prompts(tokenizer, spanish_second_singular_b, spanish_conjugations)
prompts_trimmed_d_spanish_second_sing, answers_ids_d_spanish_second_sing, verb_entries_d_spanish_second_sing = build_conjugation_prompts(tokenizer, spanish_second_singular_d, spanish_conjugations)
prompts_trimmed_e_spanish_second_sing, answers_ids_e_spanish_second_sing, verb_entries_e_spanish_second_sing = build_conjugation_prompts(tokenizer, spanish_second_singular_e, spanish_conjugations)

# Italian (First Person Singular)
prompts_trimmed_a_italian_first_sing, answers_ids_a_italian_first_sing, verb_entries_a_italian_first_sing = build_conjugation_prompts(tokenizer, italian_first_singular, italian_conjugations)
prompts_trimmed_b_italian_first_sing, answers_ids_b_italian_first_sing, verb_entries_b_italian_first_sing = build_conjugation_prompts(tokenizer, italian_first_singular_b, italian_conjugations)
prompts_trimmed_d_italian_first_sing, answers_ids_d_italian_first_sing, verb_entries_d_italian_first_sing = build_conjugation_prompts(tokenizer, italian_first_singular_d, italian_conjugations)
prompts_trimmed_e_italian_first_sing, answers_ids_e_italian_first_sing, verb_entries_e_italian_first_sing = build_conjugation_prompts(tokenizer, italian_first_singular_e, italian_conjugations)

# Italian (Second Person Singular)
prompts_trimmed_a_italian_second_sing, answers_ids_a_italian_second_sing, verb_entries_a_italian_second_sing = build_conjugation_prompts(tokenizer, italian_second_singular, italian_conjugations)
prompts_trimmed_b_italian_second_sing, answers_ids_b_italian_second_sing, verb_entries_b_italian_second_sing = build_conjugation_prompts(tokenizer, italian_second_singular_b, italian_conjugations)
prompts_trimmed_d_italian_second_sing, answers_ids_d_italian_second_sing, verb_entries_d_italian_second_sing = build_conjugation_prompts(tokenizer, italian_second_singular_d, italian_conjugations)
prompts_trimmed_e_italian_second_sing, answers_ids_e_italian_second_sing, verb_entries_e_italian_second_sing = build_conjugation_prompts(tokenizer, italian_second_singular_e, italian_conjugations)

# German (First Person Singular)
prompts_trimmed_a_german_first_sing, answers_ids_a_german_first_sing, verb_entries_a_german_first_sing = build_conjugation_prompts(tokenizer, german_first_singular, german_conjugations)
prompts_trimmed_b_german_first_sing, answers_ids_b_german_first_sing, verb_entries_b_german_first_sing = build_conjugation_prompts(tokenizer, german_first_singular_b, german_conjugations)
prompts_trimmed_d_german_first_sing, answers_ids_d_german_first_sing, verb_entries_d_german_first_sing = build_conjugation_prompts(tokenizer, german_first_singular_d, german_conjugations)
prompts_trimmed_e_german_first_sing, answers_ids_e_german_first_sing, verb_entries_e_german_first_sing = build_conjugation_prompts(tokenizer, german_first_singular_e, german_conjugations)

# German (Second Person Singular)
prompts_trimmed_a_german_second_sing, answers_ids_a_german_second_sing, verb_entries_a_german_second_sing = build_conjugation_prompts(tokenizer, german_second_singular, german_conjugations)
prompts_trimmed_b_german_second_sing, answers_ids_b_german_second_sing, verb_entries_b_german_second_sing = build_conjugation_prompts(tokenizer, german_second_singular_b, german_conjugations)
prompts_trimmed_d_german_second_sing, answers_ids_d_german_second_sing, verb_entries_d_german_second_sing = build_conjugation_prompts(tokenizer, german_second_singular_d, german_conjugations)
prompts_trimmed_e_german_second_sing, answers_ids_e_german_second_sing, verb_entries_e_german_second_sing = build_conjugation_prompts(tokenizer, german_second_singular_e, german_conjugations)

# English (First Person Singular)
prompts_trimmed_a_english_first_sing, answers_ids_a_english_first_sing, verb_entries_a_english_first_sing = build_conjugation_prompts(tokenizer, english_first_singular, english_conjugations)
prompts_trimmed_b_english_first_sing, answers_ids_b_english_first_sing, verb_entries_b_english_first_sing = build_conjugation_prompts(tokenizer, english_first_singular_b, english_conjugations)
prompts_trimmed_d_english_first_sing, answers_ids_d_english_first_sing, verb_entries_d_english_first_sing = build_conjugation_prompts(tokenizer, english_first_singular_d, english_conjugations)
prompts_trimmed_e_english_first_sing, answers_ids_e_english_first_sing, verb_entries_e_english_first_sing = build_conjugation_prompts(tokenizer, english_first_singular_e, english_conjugations)

# English (Second Person Singular)
prompts_trimmed_a_english_second_sing, answers_ids_a_english_second_sing, verb_entries_a_english_second_sing = build_conjugation_prompts(tokenizer, english_second_singular, english_conjugations)
prompts_trimmed_b_english_second_sing, answers_ids_b_english_second_sing, verb_entries_b_english_second_sing = build_conjugation_prompts(tokenizer, english_second_singular_b, english_conjugations)
prompts_trimmed_d_english_second_sing, answers_ids_d_english_second_sing, verb_entries_d_english_second_sing = build_conjugation_prompts(tokenizer, english_second_singular_d, english_conjugations)
prompts_trimmed_e_english_second_sing, answers_ids_e_english_second_sing, verb_entries_e_english_second_sing = build_conjugation_prompts(tokenizer, english_second_singular_e, english_conjugations)


from collections import defaultdict

languages = ["spanish", "italian", "german"] #"english"] CUT OUT ENGLISH BC TOO BIG BUT IS IN OTHER ONE!!!
suffixes = ["a", "b", "d", "e"] #skipped c by accident above
results = {}

# Total number of prompts across all languages and suffixes
total_prompts = 0
for lang in languages:
    for suffix in suffixes:
        total_prompts += len(globals().get(f"prompts_trimmed_{suffix}_{lang}_first_sing", []))
        total_prompts += len(globals().get(f"prompts_trimmed_{suffix}_{lang}_second_sing", []))
print(f"🔍 Loaded {total_prompts} prompts for evaluation across all formats and languages")

# Detailed stats per (language, suffix) → token length
detailed_stats = {
    (lang, suffix): defaultdict(lambda: {'total': 0, 'correct': 0})
    for lang in languages for suffix in suffixes
}

print("\n🧪 Dataset availability check:")
for lang in languages:
    for suffix in suffixes:
        for person in ["first", "second"]:
            name_prefix = f"{suffix}_{lang}_{person}_sing"
            prompts = globals().get(f"prompts_trimmed_{name_prefix}")
            answers = globals().get(f"answers_ids_{name_prefix}")
            verb_entries = globals().get(f"verb_entries_{name_prefix}")

            print(f"{name_prefix}: "
                  f"{'✅' if prompts else '❌'} prompts, "
                  f"{'✅' if answers else '❌'} answers, "
                  f"{'✅' if verb_entries else '❌'} entries, "
                  f"({len(prompts) if prompts else 0} items)")

# Evaluation loop
for suffix in suffixes:
    for lang in languages:
        for person in ["first", "second"]:
            name_prefix = f"{suffix}_{lang}_{person}_sing"
            prompts = globals().get(f"prompts_trimmed_{name_prefix}", [])
            answers = globals().get(f"answers_ids_{name_prefix}", [])
            verb_entries = globals().get(f"verb_entries_{name_prefix}", [])

            if not prompts:
                continue

            total = len(prompts)
            correct_top1 = 0
            sum_gold_probs = 0.0

            yo_total = 0
            yo_correct = 0
            yo_sum_probs = 0.0

            tu_total = 0
            tu_correct = 0
            tu_sum_probs = 0.0

            token_length_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

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

                #is_yo = idx < len(prompts) // 2
                #if is_yo:
                    #yo_total += 1
                    #yo_sum_probs += gold_prob
                    #if top_id == gold_id:
                        #yo_correct += 1
                #else:
                    #tu_total += 1
                    #tu_sum_probs += gold_prob
                    #if top_id == gold_id:
                        #tu_correct += 1

                verb_entry = verb_entries[idx]
                tok_len = len(tokenizer.encode(tokenizer.decode([gold_id]), add_special_tokens=False))

                # Track token length accuracy
                if top_id == gold_id:
                    token_length_stats[tok_len]['correct'] += 1
                token_length_stats[tok_len]['total'] += 1

            results[(lang, suffix, person)] = {
                "total": total,
                "correct_top1": correct_top1,
                "accuracy": correct_top1 / total if total > 0 else 0.0,
                "avg_gold_prob": sum_gold_probs / total if total > 0 else 0.0
            }

            # Print example
            decoded_prompt = tokenizer.decode(prompts[0], skip_special_tokens=True)
            gold_token = tokenizer.decode([answers[0]])
            print(f"\n📝 Example for {lang.upper()} dataset {suffix.upper()} ({person.upper()} Singular):")
            print(f"Prompt: {decoded_prompt}")
            print(f"Expected next token: '{gold_token}'")

            # Print results
            r = results[(lang, suffix, person)]
            print(f"📊 Results:")
            print(f"🎯 Accuracy: {r['accuracy']:.2%} ({r['correct_top1']}/{r['total']})")
            print(f"📈 Avg P(gold): {r['avg_gold_prob']:.2%}")

            #print("\n📊 Token Length Accuracy:")
            #for tok_len, stats in sorted(token_length_stats.items()):
                #total = stats['total']
                #correct = stats['correct']
                #acc = correct / total if total > 0 else 0.0
                #print(f"  Token length {tok_len}: {acc:.2%} ({correct}/{total})")

            #if yo_total > 0:
                #print(f"👤 YO Accuracy: {yo_correct / yo_total:.2%} ({yo_correct}/{yo_total})")
                #print(f"👤 YO Avg P(gold): {yo_sum_probs / yo_total:.2%}")
            #if tu_total > 0:
                #print(f"🧑‍🦰 TÚ Accuracy: {tu_correct / tu_total:.2%} ({tu_correct}/{tu_total})")
                #print(f"🧑‍🦰 TÚ Avg P(gold): {tu_sum_probs / tu_total:.2%}")

# ----------------------------------------------------
# 📊 Summary: Averages by language (across all suffixes)
# ----------------------------------------------------
print("\n📚 AVERAGE METRICS BY LANGUAGE")
for lang in languages:
    total = 0
    correct = 0
    sum_gold = 0.0

    for suffix in suffixes:
        for person in ["first", "second"]:
            key = (lang, suffix, person)
            if key not in results:
                continue
            r = results[key]
            total += r["total"]
            correct += r["correct_top1"]
            sum_gold += r["avg_gold_prob"] * r["total"]

    if total > 0:
        acc = correct / total
        avg_prob = sum_gold / total
        print(f"🌍 {lang.upper()}: Accuracy = {acc:.2%}, Avg P(gold) = {avg_prob:.2%}")
    else:
        print(f"🌍 {lang.upper()}: No data")

# ----------------------------------------------------
# 📊 Summary: Averages by suffix (dataset type)
# ----------------------------------------------------
print("\n📚 AVERAGE METRICS BY DATASET TYPE (SUFFIX)")
for suffix in suffixes:
    total = 0
    correct = 0
    sum_gold = 0.0

    for lang in languages:
        for person in ["first", "second"]:
            key = (lang, suffix, person)
            if key not in results:
                continue
            r = results[key]
            total += r["total"]
            correct += r["correct_top1"]
            sum_gold += r["avg_gold_prob"] * r["total"]

    if total > 0:
        acc = correct / total
        avg_prob = sum_gold / total
        print(f"📄 Dataset {suffix.upper()}: Accuracy = {acc:.2%}, Avg P(gold) = {avg_prob:.2%}")
    else:
        print(f"📄 Dataset {suffix.upper()}: No data")








