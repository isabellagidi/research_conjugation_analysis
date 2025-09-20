import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from typing import List, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the JSON dataset
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_conjugations(all_verbs, tokenizer, language):
    verb_conjugations = []
    print(f"Kept verb forms for language: {language}\n")

    # Get the verbs for the specified language
    language_data = all_verbs.get(language, {})
    
    if not language_data:
        print(f"No data found for language: {language}")
        return verb_conjugations

    # Iterate over each verb in the specified language
    for verb, forms in language_data.items():
        # Ensure both first and second person singular forms exist
        first_sing = forms.get("1st_person_singular", None)
        second_sing = forms.get("2nd_person_singular", None)
        third_sing = forms.get("3rd_person_singular", None)
        first_pl = forms.get("1st_person_plural", None)
        second_pl = forms.get("2nd_person_plural", None)
        third_pl = forms.get("3rd_person_plural", None)

        if first_sing is None or second_sing is None:
            continue  # Skip if either form is missing

        # Tokenize with an additional space for proper tokenization
        first_sing_tok = tokenizer.tokenize(" " + first_sing)  # Add a space for correct tokenization
        second_sing_tok = tokenizer.tokenize(" " + second_sing)  # Add a space for correct tokenization

        # Condition 1: Single tokens for both forms
        condition_1 = len(first_sing_tok) == 1 and len(second_sing_tok) == 1

        # Condition 2: The first (n-1) tokens are identical
        condition_2 = (
            len(first_sing_tok) == len(second_sing_tok) and
            first_sing_tok[:-1] == second_sing_tok[:-1]
        )

        # If either condition is satisfied, append the verb conjugation to the list
        if condition_1 or condition_2:
            verb_conjugations.append((verb, first_sing, second_sing, third_sing, first_pl, second_pl, third_pl))
            #print(f"Infinitive: {verb}")
            #print(f"  First Person Singular: {first_sing} -> {first_sing_tok}")
            #print(f"  Second Person Singular: {second_sing} -> {second_sing_tok}")
            #print("-" * 40)

    return verb_conjugations


def build_conjugation_prompts(tokenizer, language_prompts, spanish_conjugations):
    prompts_trimmed = []
    answers_ids = []
    matched_entries = []

    for prompt, entry in zip(language_prompts, spanish_conjugations):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) < 2:
            continue

        prompt_trimmed = token_ids[:-1]
        answer_id = token_ids[-1]

        prompts_trimmed.append(prompt_trimmed)
        answers_ids.append(answer_id)
        matched_entries.append(entry)

    return prompts_trimmed, answers_ids, matched_entries


from transformers import AutoModelForCausalLM          # <-- import
import torch

def accuracy_filter(
    prompts_trimmed,           # list[list[int]]
    answer_ids,                # list[int]
    matched_entries,           # list[Tuple[…]]
    model_name,                # str
    batch_size=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Keep only the (prompt, answer, entry) triples for which the model’s
    next‑token prediction equals the gold token.

    Returned `correct_prompts` is **still a list of token‑ID lists**,
    identical to what you passed in (except incorrect items are gone).
    No extra padding tokens are introduced.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    correct_p, correct_a, correct_e = [], [], []
    incorrect_p, incorrect_a, incorrect_e = [], [], []

    # Work in minibatches
    for i in range(0, len(prompts_trimmed), batch_size):
        tok_batch  = prompts_trimmed[i : i + batch_size]     # original token IDs
        gold_batch = answer_ids[i : i + batch_size]
        meta_batch = matched_entries[i : i + batch_size]

        # Convert *tokens*→text so the HF tokenizer can repack them;
        # this does NOT alter tok_batch itself.
        text_batch = [tokenizer.decode(toks, skip_special_tokens=True) for toks in tok_batch]

        enc = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        gold_ids       = torch.tensor(gold_batch, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # For each sequence, take logits at its own last **non‑pad** position
        seq_ends = attention_mask.sum(dim=1) - 1        # shape (B,)
        preds = torch.stack([
            logits[b, seq_ends[b], :].argmax(dim=-1)
            for b in range(logits.size(0))
        ])

        correct_mask = (preds == gold_ids)

        # Split into correct / incorrect
        for j, is_ok in enumerate(correct_mask.tolist()):
            if is_ok:
                correct_p.append(tok_batch[j])
                correct_a.append(gold_batch[j])
                correct_e.append(meta_batch[j])
            else:
                incorrect_p.append(tok_batch[j])
                incorrect_a.append(gold_batch[j])
                incorrect_e.append(meta_batch[j])

        torch.cuda.empty_cache()

    return (
        correct_p,   correct_a,   correct_e,      # <- what you usually keep
        incorrect_p, incorrect_a, incorrect_e
    )


def accuracy_filter_old(prompts_trimmed, answer_ids, matched_entries, model_name, batch_size=8):
    """
    Filters prompts based on whether the model correctly predicts the gold token.

    Args:
        prompts_trimmed: List of tokenized prompt prefixes (list of list of token ids).
        answer_ids: List of token ids (int), the correct conjugated form.
        matched_entries: List of verb metadata tuples aligned with prompts.
        model_name: HuggingFace model name to load model/tokenizer.
        batch_size: Number of prompts to evaluate per batch (default: 8).

    Returns:
        correct_prompts, correct_answer_ids, correct_matched_entries,
        incorrect_prompts, incorrect_answer_ids, incorrect_matched_entries
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    # Decode prompt tokens back into text
    text_prompts = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed]

    # Init output containers
    correct_prompts, correct_answer_ids, correct_entries = [], [], []
    incorrect_prompts, incorrect_answer_ids, incorrect_entries = [], [], []

    # Loop through prompts in batches
    for i in range(0, len(text_prompts), batch_size):
        batch_prompts = text_prompts[i:i+batch_size]
        batch_gold_ids = answer_ids[i:i+batch_size]
        batch_entries = matched_entries[i:i+batch_size]

        # Tokenize batch and move to device
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        gold_ids = torch.tensor(batch_gold_ids, device=device)

        # Forward pass
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        final_logits = logits[:, -1, :]  # next-token predictions

        # Compare predictions to gold
        preds = torch.argmax(final_logits, dim=-1)
        correct_mask = (preds == gold_ids).tolist()

        # Sort batch into correct / incorrect
        for j, is_correct in enumerate(correct_mask):
            if is_correct:
                correct_prompts.append(batch_prompts[j])
                correct_answer_ids.append(batch_gold_ids[j])
                correct_entries.append(batch_entries[j])
            else:
                incorrect_prompts.append(batch_prompts[j])
                incorrect_answer_ids.append(batch_gold_ids[j])
                incorrect_entries.append(batch_entries[j])

        # Free memory between batches
        torch.cuda.empty_cache()

    return (
        correct_prompts,
        correct_answer_ids,
        correct_entries,
        incorrect_prompts,
        incorrect_answer_ids,
        incorrect_entries
    )


def accuracy_filter_old(prompts_trimmed, answer_ids, matched_entries, model_name):
    """
    Filters prompts based on whether the model correctly predicts the gold token.

    Args:
        prompts_trimmed: List of tokenized prompt prefixes (list of list of token ids).
        answer_ids: List of token ids (int), the correct conjugated form.
        matched_entries: List of verb metadata tuples aligned with prompts.
        model_name: HuggingFace model name to load model/tokenizer.

    Returns:
        correct_prompts: List of decoded prompt strings for correct predictions.
        correct_answer_ids: List of gold token ids that were correctly predicted.
        correct_matched_entries: List of verb metadata for correct predictions.

        incorrect_prompts: List of decoded prompt strings for incorrect predictions.
        incorrect_answer_ids: List of gold token ids that were incorrectly predicted.
        incorrect_matched_entries: List of verb metadata for incorrect predictions.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # required for BLOOM
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    # Decode tokenized prompts into text
    text_prompts = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed]

    # Tokenize entire prompts for model input
    encoded = tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    gold_ids = torch.tensor(answer_ids).to(device)

    # Evaluate predictions
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    final_logits = logits[:, -1, :]  # Next-token prediction

    # Compare predictions to gold token IDs
    preds = torch.argmax(final_logits, dim=-1)
    correct_mask = (preds == gold_ids).tolist()

    # Filter by correctness
    correct_prompts = [p for p, c in zip(text_prompts, correct_mask) if c]
    correct_answer_ids = [a for a, c in zip(answer_ids, correct_mask) if c]
    correct_matched_entries = [m for m, c in zip(matched_entries, correct_mask) if c]

    incorrect_prompts = [p for p, c in zip(text_prompts, correct_mask) if not c]
    incorrect_answer_ids = [a for a, c in zip(answer_ids, correct_mask) if not c]
    incorrect_matched_entries = [m for m, c in zip(matched_entries, correct_mask) if not c]

    return (
        correct_prompts,
        correct_answer_ids,
        correct_matched_entries,
        incorrect_prompts,
        incorrect_answer_ids,
        incorrect_matched_entries
    )

from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def accuracy_filter_text(
    prompts_trimmed: List[List[int]],     # list of token-ID prefixes
    answer_ids: List[int],                # gold next-token ids
    matched_entries: List[Tuple],         # aligned metadata
    model_name: Optional[str] = None,     # only needed if tokenizer/model not passed
    batch_size: int = 8,
    device: Optional[str] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    model: Optional[AutoModelForCausalLM] = None,
    max_len: Optional[int] = None,        # if None, use model.config.max_position_embeddings or 2048
):
    """
    Returns decoded prompt strings for correct/incorrect groups so downstream viz expecting strings will work.
    Fixes: per-seq last non-pad position; optional reuse of tokenizer/model; no truncation warnings.
    """
    # --- setup ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tokenizer is None or model is None:
        assert model_name is not None, "Provide model_name if not passing tokenizer/model"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    else:
        # ensure pad_token set (BLOOM needs this)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if max_len is None:
        max_len = getattr(getattr(model, "config", object()), "max_position_embeddings", 2048) or 2048

    # decode prefixes to text (keeps your original prefixes intact)
    text_prompts = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts_trimmed]

    correct_prompts, correct_answer_ids, correct_entries = [], [], []
    incorrect_prompts, incorrect_answer_ids, incorrect_entries = [], [], []

    for i in range(0, len(text_prompts), batch_size):
        batch_prompts = text_prompts[i:i+batch_size]
        batch_gold_ids = answer_ids[i:i+batch_size]
        batch_entries  = matched_entries[i:i+batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,      # <-- no warning now
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        gold_ids = torch.tensor(batch_gold_ids, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, S, V]

        # last non-pad positions per sequence
        seq_ends = attention_mask.sum(dim=1) - 1                        # [B]
        next_logits = logits[torch.arange(logits.size(0), device=device), seq_ends, :]  # [B, V]
        preds = next_logits.argmax(dim=-1)                               # [B]

        correct_mask = (preds == gold_ids).tolist()

        for j, ok in enumerate(correct_mask):
            if ok:
                correct_prompts.append(batch_prompts[j])
                correct_answer_ids.append(batch_gold_ids[j])
                correct_entries.append(batch_entries[j])
            else:
                incorrect_prompts.append(batch_prompts[j])
                incorrect_answer_ids.append(batch_gold_ids[j])
                incorrect_entries.append(batch_entries[j])

        torch.cuda.empty_cache()

    return (
        correct_prompts, correct_answer_ids, correct_entries,
        incorrect_prompts, incorrect_answer_ids, incorrect_entries,
    )


def group_by_token_lengths(
    correct_prompts: List[str],
    correct_answer_ids: List[int],
    matched_entries: List[tuple],
    tokenizer
):
    """
    Groups (prompt, answer_id, entry) examples by:
      (infinitive_token_len, conjugated_token_len)

    Returns:
        A dict where keys are (inf_len, conj_len) and values are lists of examples.
    """
    grouped = defaultdict(list)

    for prompt, answer_id, entry in zip(correct_prompts, correct_answer_ids, matched_entries):
        infinitive = entry[0]      # e.g. "hablar"
        conjugated = entry[1]      # e.g. "hablo" (1st person singular)

        # Tokenize with a space to ensure correct word boundaries (BLOOM-specific)
        inf_tokens = tokenizer.tokenize(" " + infinitive)
        conj_tokens = tokenizer.tokenize(" " + conjugated)

        inf_len = len(inf_tokens)
        conj_len = len(conj_tokens)

        grouped[(inf_len, conj_len)].append((prompt, answer_id, entry))

    return grouped


#spanish
def generate_first_singular_dataset_spanish(spanish_conjugations):
    spanish_first_sing = [f"Conjugación del verbo {verb} en presente: Yo {first_sing}" for verb, first_sing, _, _, _, *__ in spanish_conjugations]
    return spanish_first_sing

def generate_second_singular_dataset_spanish(spanish_conjugations):
    spanish_second_sing = [f"Conjugación del verbo {verb} en presente: Tú {second_sing}" for verb, _, second_sing, _, _, *__ in spanish_conjugations]
    return spanish_second_sing

#italian
def generate_first_singular_dataset_italian(italian_conjugations):
    italian_first_sing = [f"Coniugazione del verbo {verb} al presente: Io {first_sing}" for verb, first_sing, _, _, _, *__ in italian_conjugations]
    return italian_first_sing

def generate_second_singular_dataset_italian(italian_conjugations):
    italian_second_sing = [f"Coniugazione del verbo {verb} al presente: Tu {second_sing}" for verb, _, second_sing, _, _, *__ in italian_conjugations]
    return italian_second_sing

#german
def generate_first_singular_dataset_german(german_conjugations):
    german_first_sing = [f"Konjugation des Verbs {verb} im Präsens: Ich {first_sing}" for verb, first_sing, _, _, _, *__ in german_conjugations]
    return german_first_sing

def generate_second_singular_dataset_german(german_conjugations):
    german_second_sing = [f"Konjugation des Verbs {verb} im Präsens: Du {second_sing}" for verb, _, second_sing, _, _, *__ in german_conjugations]
    return german_second_sing

#english
def generate_first_singular_dataset_english(english_conjugations):
    english_first_sing = [f"Conjugation of the verb {verb} in present tense: I {first_sing}" for verb, first_sing, _, _, _, *__ in english_conjugations]
    return english_first_sing

def generate_second_singular_dataset_english(english_conjugations):
    english_second_sing = [f"Conjugation of the verb {verb} in present tense: You {second_sing}" for verb, _, second_sing, _, _, *__ in english_conjugations]
    return english_second_sing
