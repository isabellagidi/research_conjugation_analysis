import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the JSON dataset
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_conjugations(all_verbs, tokenizer, language, conjugation_pair=("first singular", "second singular")):
    """
    Filters conjugation pairs based on tokenization overlap conditions.

    Args:
        all_verbs (dict): Full MorphyNet dataset.
        tokenizer: HuggingFace tokenizer.
        language (str): Full language name, e.g., "spanish", "french".
        conjugation_pair (tuple): Two grammatical persons to compare, e.g. ("first singular", "second plural").

    Returns:
        List of verb tuples: (verb, 1st_sing, 2nd_sing, 3rd_sing, 1st_pl, 2nd_pl, 3rd_pl)
    """

    # Map full language names to MorphyNet ISO codes
    lang_map = {
        "catalan": "cat", "czech": "ces", "german": "deu", "english": "eng",
        "finnish": "fin", "french": "fra", "serbo-croatian": "hbs", "hungarian": "hun",
        "italian": "ita", "mongolian": "mon", "polish": "pol", "portuguese": "por",
        "russian": "rus", "spanish": "spa", "swedish": "swe"
    }

    morphy_keys = {
        "first singular": "1st_person_singular",
        "second singular": "2nd_person_singular",
        "third singular": "3rd_person_singular",
        "first plural": "1st_person_plural",
        "second plural": "2nd_person_plural",
        "third plural": "3rd_person_plural",
    }

    # Validate conjugation_pair
    if not all(p in morphy_keys for p in conjugation_pair):
        raise ValueError(f"❌ Invalid conjugation_pair: {conjugation_pair}")

    # Map language
    language_iso = lang_map.get(language.lower())
    if not language_iso:
        print(f"❌ Unsupported language: {language}")
        print(f"🧭 Supported: {list(lang_map.keys())}")
        return []

    verb_conjugations = []
    print(f"✅ Keeping verb forms for language: {language} ({language_iso})\n")

    language_data = all_verbs.get(language_iso, {})
    if not language_data:
        print(f"⚠️ No data found for language code: {language_iso}")
        return []

    # Which conjugation keys to compare
    key_a = morphy_keys[conjugation_pair[0]]
    key_b = morphy_keys[conjugation_pair[1]]

    for verb, forms in language_data.items():
        form_a = forms.get(key_a, None)
        form_b = forms.get(key_b, None)

        if form_a is None or form_b is None:
            continue

        tok_a = tokenizer.tokenize(" " + form_a)
        tok_b = tokenizer.tokenize(" " + form_b)

        condition_1 = len(tok_a) == 1 and len(tok_b) == 1
        condition_2 = (
            len(tok_a) == len(tok_b) and
            tok_a[:-1] == tok_b[:-1]
        )

        if condition_1 or condition_2:
            # Always return full conjugation tuple
            full_tuple = (
                verb,
                forms.get("1st_person_singular", None),
                forms.get("2nd_person_singular", None),
                forms.get("3rd_person_singular", None),
                forms.get("1st_person_plural", None),
                forms.get("2nd_person_plural", None),
                forms.get("3rd_person_plural", None),
            )
            verb_conjugations.append(full_tuple)

    return verb_conjugations



def build_conjugation_prompts(tokenizer, language_prompts, conjugations):
    prompts_trimmed = []
    answers_ids = []
    matched_entries = []

    for prompt, entry in zip(language_prompts, conjugations):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) < 2:
            continue

        prompt_trimmed = token_ids[:-1]
        answer_id = token_ids[-1]

        prompts_trimmed.append(prompt_trimmed)
        answers_ids.append(answer_id)
        matched_entries.append(entry)

    return prompts_trimmed, answers_ids, matched_entries


def accuracy_filter(prompts_trimmed, answer_ids, matched_entries, model_name, batch_size=8):
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


def generate_dataset(person="first singular", language="spanish", conjugations=None, templates=None):
    """
    Generate a conjugation prompt dataset based on language and grammatical person.

    Args:
        person (str): e.g., "first singular", "third plural", etc.
        language (str): e.g., "spanish", "french", "portuguese", etc.
        conjugations (List[Tuple]): list of (verb, 1st_sing, 2nd_sing, ..., 3rd_pl)

    Returns:
        List[str]: dataset of formatted prompts
    """
    if conjugations is None:
        raise ValueError("You must provide a conjugations list.")

    person = person.lower()
    language = language.lower()

    # Index of the conjugation in the tuple
    person_to_index = {
        "first singular": 1,
        "second singular": 2,
        "third singular": 3,
        "first plural": 4,
        "second plural": 5,
        "third plural": 6,
    }

    if templates is None:
        # Templates and pronouns per language
        templates = {
            "catalan": (
                "Conjugació del verb {verb} en present: {pronoun} {conj}",
                {
                    "first singular": "Jo",
                    "second singular": "Tu",
                    "third singular": "Ell",
                    "first plural": "Nosaltres",
                    "second plural": "Vosaltres",
                    "third plural": "Ells",
                }
            ),

            "czech": (
                "Časování slovesa {verb} v přítomném čase: {pronoun} {conj}",
                {
                    "first singular": "Já",
                    "second singular": "Ty",
                    "third singular": "On",
                    "first plural": "My",
                    "second plural": "Vy",
                    "third plural": "Oni",
                }
            ),
            "spanish": (
                "Conjugación del verbo {verb} en presente: {pronoun} {conj}",
                {
                    "first singular": "Yo",
                    "second singular": "Tú",
                    "third singular": "Él",
                    "first plural": "Nosotros",
                    "second plural": "Vosotros",
                    "third plural": "Ellos",
                }
            ),
            "french": (
                "Conjugaison du verbe {verb} au présent : {pronoun} {conj}",
                {
                    "first singular": "Je",
                    "second singular": "Tu",
                    "third singular": "Il",
                    "first plural": "Nous",
                    "second plural": "Vous",
                    "third plural": "Ils",
                }
            ),
            "portuguese": (
                "Conjugação do verbo {verb} no presente: {pronoun} {conj}",
                {
                    "first singular": "Eu",
                    "second singular": "Tu",
                    "third singular": "Ele",
                    "first plural": "Nós",
                    "second plural": "Vós",
                    "third plural": "Eles",
                }
            ),
            "german": (
                "Konjugation des Verbs {verb} im Präsens: {pronoun} {conj}",
                {
                    "first singular": "Ich",
                    "second singular": "Du",
                    "third singular": "Er",
                    "first plural": "Wir",
                    "second plural": "Ihr",
                    "third plural": "Sie",
                }
            ),
            "english": (
                "Conjugation of the verb {verb} in present tense: {pronoun} {conj}",
                {
                    "first singular": "I",
                    "second singular": "You",
                    "third singular": "He",
                    "first plural": "We",
                    "second plural": "You",
                    "third plural": "They",
                }
            ),
            "finnish": (
                "Verbien taivutus preesensissä: {pronoun} {conj}",
                {
                    "first singular": "Minä",
                    "second singular": "Sinä",
                    "third singular": "Hän",
                    "first plural": "Me",
                    "second plural": "Te",
                    "third plural": "He",
                }
            ),

            "serbo-croatian": (
                "Konjugacija glagola {verb} u prezentu: {pronoun} {conj}",
                {
                    "first singular": "Ja",
                    "second singular": "Ti",
                    "third singular": "On",
                    "first plural": "Mi",
                    "second plural": "Vi",
                    "third plural": "Oni",
                }
            ),
            "hungarian": (
                "A(z) {verb} ige ragozása jelen időben: {pronoun} {conj}",
                {
                    "first singular": "Én",
                    "second singular": "Te",
                    "third singular": "Ő",
                    "first plural": "Mi",
                    "second plural": "Ti",
                    "third plural": "Ők",
                }
            ),

            "mongolian": (
                "{verb} үйл үгийн одоо цагийн хувилбар: {pronoun} {conj}",
                {
                    "first singular": "Би",
                    "second singular": "Чи",
                    "third singular": "Тэр",
                    "first plural": "Бид",
                    "second plural": "Та нар",
                    "third plural": "Тэд",
                }
            ),

            "italian": (
                "Coniugazione del verbo {verb} al presente: {pronoun} {conj}",
                {
                    "first singular": "Io",
                    "second singular": "Tu",
                    "third singular": "Lui",
                    "first plural": "Noi",
                    "second plural": "Voi",
                    "third plural": "Loro",
                }
            ),

            "polish": (
                "Odmiana czasownika {verb} w czasie teraźniejszym: {pronoun} {conj}",
                {
                    "first singular": "Ja",
                    "second singular": "Ty",
                    "third singular": "On",
                    "first plural": "My",
                    "second plural": "Wy",
                    "third plural": "Oni",
                }
            ),

            "russian": (
                "Спряжение глагола {verb} в настоящем времени: {pronoun} {conj}",
                {
                    "first singular": "Я",
                    "second singular": "Ты",
                    "third singular": "Он",
                    "first plural": "Мы",
                    "second plural": "Вы",
                    "third plural": "Они",
                }
            ),

            "swedish": (
                "Böjning av verbet {verb} i presens: {pronoun} {conj}",
                {
                    "first singular": "Jag",
                    "second singular": "Du",
                    "third singular": "Han",
                    "first plural": "Vi",
                    "second plural": "Ni",
                    "third plural": "De",
                }
            )

        }

    if language not in templates:
        raise ValueError(f"Unsupported language: {language}")
    if person not in person_to_index:
        raise ValueError(f"Unsupported person: {person}")

    template_str, pronoun_map = templates[language]
    idx = person_to_index[person]
    pronoun = pronoun_map[person]

    dataset = []
    for conj in conjugations:
        verb = conj[0]
        conj_form = conj[idx]
        if conj_form:  # Skip if None or missing
            prompt = template_str.format(verb=verb, pronoun=pronoun, conj=conj_form)
            dataset.append(prompt)

    return dataset

def save_heatmap(data, x_labels, y_labels, title, filename, center=0.0, cmap="coolwarm"):
    plt.figure(figsize=(min(len(x_labels) * 0.5, 18), min(len(y_labels) * 0.5, 12)))
    sns.heatmap(
        data.cpu().numpy(),
        xticklabels=x_labels,
        yticklabels=y_labels,
        center=center,
        cmap=cmap,
        cbar_kws={"label": "Normalized Recovery"}
    )
    plt.xlabel("Position" if "Position" in title else "Head")
    plt.ylabel("Layer" if "Layer" in title else "Head")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Saved: {filename}")
    plt.close()

def get_logit_diff(logits, answer_ids):
    logits = logits[:, -1, :]
    return logits.gather(1, answer_ids.unsqueeze(1)).mean()

def conjugation_metric(logits, answer_ids, CLEAN_BASELINE, CORRUPTED_BASELINE):
    return (get_logit_diff(logits, answer_ids) - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE + 1e-12)