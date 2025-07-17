from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def generate_yo_dataset(data): 
    return [f"Conjugación del verbo {verb} en presente: Yo {yo_form}" for verb, yo_form, *_ in data]

def generate_tu_dataset(data): 
    return [f"Conjugación del verbo {verb} en presente: Tú {tu_form}" for verb, yo_form, tu_form, *_ in data]

def build_conjugation_prompts(tokenizer, spanish_conjugations):
    prompts_trimmed = []
    answers_ids = []
    valid_indices = []

    for idx, prompt in enumerate(spanish_conjugations):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) < 2:
            continue

        prompt_trimmed = token_ids[:-1]
        answer_id = token_ids[-1]

        prompts_trimmed.append(prompt_trimmed)
        answers_ids.append(answer_id)
        valid_indices.append(idx)

    return prompts_trimmed, answers_ids, valid_indices

