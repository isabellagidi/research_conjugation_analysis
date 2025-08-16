import json
from transformers import AutoTokenizer

#model_name = "bigscience/bloom-1b1"
#tokenizer = AutoTokenizer.from_pretrained(model_name)

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

def build_conjugation_prompts(tokenizer, language_prompts, language_conjugations):
    prompts_trimmed = []
    answers_ids = []
    matched_entries = []

    for prompt, entry in zip(language_prompts, language_conjugations):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) < 2:
            continue

        prompt_trimmed = token_ids[:-1]
        answer_id = token_ids[-1]

        prompts_trimmed.append(prompt_trimmed)
        answers_ids.append(answer_id)
        matched_entries.append(entry)

    return prompts_trimmed, answers_ids, matched_entries

# DATASET A
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


# DATASET B (add the language)
#spanish
def generate_first_singular_dataset_spanish_b(spanish_conjugations):
    spanish_first_sing = [f"Conjugación del verbo {verb} en presente en español: Yo {first_sing}" for verb, first_sing, _, _, _, *__ in spanish_conjugations]
    return spanish_first_sing

def generate_second_singular_dataset_spanish_b(spanish_conjugations):
    spanish_second_sing = [f"Conjugación del verbo {verb} en presente en español: Tú {second_sing}" for verb, _, second_sing, _, _, *__ in spanish_conjugations]
    return spanish_second_sing

#italian
def generate_first_singular_dataset_italian_b(italian_conjugations):
    italian_first_sing = [f"Coniugazione del verbo {verb} al presente in italiano: Io {first_sing}" for verb, first_sing, _, _, _, *__ in italian_conjugations]
    return italian_first_sing

def generate_second_singular_dataset_italian_b(italian_conjugations):
    italian_second_sing = [f"Coniugazione del verbo {verb} al presente in italiano: Tu {second_sing}" for verb, _, second_sing, _, _, *__ in italian_conjugations]
    return italian_second_sing

#german
def generate_first_singular_dataset_german_b(german_conjugations):
    german_first_sing = [f"Konjugation des Verbs {verb} im Präsens auf Deutsch: Ich {first_sing}" for verb, first_sing, _, _, _, *__ in german_conjugations]
    return german_first_sing

def generate_second_singular_dataset_german_b(german_conjugations):
    german_second_sing = [f"Konjugation des Verbs {verb} im Präsens auf Deutsch: Du {second_sing}" for verb, _, second_sing, _, _, *__ in german_conjugations]
    return german_second_sing

#english
def generate_first_singular_dataset_english_b(english_conjugations):
    english_first_sing = [f"Conjugation of the verb {verb} in present tense in English: I {first_sing}" for verb, first_sing, _, _, _, *__ in english_conjugations]
    return english_first_sing

def generate_second_singular_dataset_english_b(english_conjugations):
    english_second_sing = [f"Conjugation of the verb {verb} in present tense in English: You {second_sing}" for verb, _, second_sing, _, _, *__ in english_conjugations]
    return english_second_sing



# DATASET C (add the previous conjugation)
#spanish
def generate_first_singular_dataset_spanish_d(spanish_conjugations):
    spanish_first_sing = [f"Conjugación del verbo {verb} en presente en español: Tú {second_sing}, Yo {first_sing}" for verb, first_sing, second_sing, _, _, *__ in spanish_conjugations]
    return spanish_first_sing

def generate_second_singular_dataset_spanish_d(spanish_conjugations):
    spanish_second_sing = [f"Conjugación del verbo {verb} en presente en español: Yo {first_sing}, Tú {second_sing}" for verb, first_sing, second_sing, _, _, *__ in spanish_conjugations]
    return spanish_second_sing

#italian
def generate_first_singular_dataset_italian_d(italian_conjugations):
    italian_first_sing = [f"Coniugazione del verbo {verb} al presente in italiano: Tu {second_sing}, Io {first_sing}" for verb, first_sing, second_sing, _, _, *__ in italian_conjugations]
    return italian_first_sing

def generate_second_singular_dataset_italian_d(italian_conjugations):
    italian_second_sing = [f"Coniugazione del verbo {verb} al presente in italiano: Io {first_sing}, Tu {second_sing}" for verb, first_sing, second_sing, _, _, *__ in italian_conjugations]
    return italian_second_sing

#german
def generate_first_singular_dataset_german_d(german_conjugations):
    german_first_sing = [f"Konjugation des Verbs {verb} im Präsens auf Deutsch: Du {second_sing}, Ich {first_sing}" for verb, first_sing, second_sing, _, _, *__ in german_conjugations]
    return german_first_sing

def generate_second_singular_dataset_german_d(german_conjugations):
    german_second_sing = [f"Konjugation des Verbs {verb} im Präsens auf Deutsch: Ich {first_sing}, Du {second_sing}" for verb, first_sing, second_sing, _, _, *__ in german_conjugations]
    return german_second_sing

#english
def generate_first_singular_dataset_english_d(english_conjugations):
    english_first_sing = [f"Conjugation of the verb {verb} in present tense in English: You {second_sing}, I {first_sing}" for verb, first_sing, second_sing, _, _, *__ in english_conjugations]
    return english_first_sing

def generate_second_singular_dataset_english_d(english_conjugations):
    english_second_sing = [f"Conjugation of the verb {verb} in present tense in English: I {first_sing}, You {second_sing}" for verb, first_sing, second_sing, _, _, *__ in english_conjugations]
    return english_second_sing

# DATASET D (add nosotros -- Hablar: Nosotros hablamos. Tu hablas, yo hablo. 
#spanish
def generate_first_singular_dataset_spanish_d(spanish_conjugations):
    spanish_first_sing = [f"Conjugación del verbo {verb} en presente en español: Nosotros {first_plu}. Tú {second_sing}, Yo {first_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in spanish_conjugations]
    return spanish_first_sing

def generate_second_singular_dataset_spanish_d(spanish_conjugations):
    spanish_second_sing = [f"Conjugación del verbo {verb} en presente en español: Nosotros {first_plu}. Yo {first_sing}, Tú {second_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in spanish_conjugations]
    return spanish_second_sing

#italian
def generate_first_singular_dataset_italian_d(italian_conjugations):
    italian_first_sing = [f"Coniugazione del verbo {verb} al presente in italiano: Noi {first_plu}. Tu {second_sing}, Io {first_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in italian_conjugations]
    return italian_first_sing

def generate_second_singular_dataset_italian_d(italian_conjugations):
    italian_second_sing = [f"Coniugazione del verbo {verb} al presente in italiano: Noi {first_plu}. Io {first_sing}, Tu {second_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in italian_conjugations]
    return italian_second_sing

#german
def generate_first_singular_dataset_german_d(german_conjugations):
    german_first_sing = [f"Konjugation des Verbs {verb} im Präsens auf Deutsch: Wir {first_plu}. Du {second_sing}, Ich {first_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in german_conjugations]
    return german_first_sing

def generate_second_singular_dataset_german_d(german_conjugations):
    german_second_sing = [f"Konjugation des Verbs {verb} im Präsens auf Deutsch: Wir {first_plu}. Ich {first_sing}, Du {second_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in german_conjugations]
    return german_second_sing

#english
def generate_first_singular_dataset_english_d(english_conjugations):
    english_first_sing = [f"Conjugation of the verb {verb} in present tense in English: We {first_plu}. You {second_sing}, I {first_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in english_conjugations]
    return english_first_sing

def generate_second_singular_dataset_english_d(english_conjugations):
    english_second_sing = [f"Conjugation of the verb {verb} in present tense in English: We {first_plu}. I {first_sing}, You {second_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in english_conjugations]
    return english_second_sing


#DATASET E (even simpler. Hablar: Nosotros hablamos. Tu hablas, yo hablo.)
#spanish
def generate_first_singular_dataset_spanish_e(spanish_conjugations):
    spanish_first_sing = [f"{verb}: Nosotros {first_plu}. Tú {second_sing}, Yo {first_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in spanish_conjugations]
    return spanish_first_sing

def generate_second_singular_dataset_spanish_e(spanish_conjugations):
    spanish_second_sing = [f"{verb}: Nosotros {first_plu}. Yo {first_sing}, Tú {second_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in spanish_conjugations]
    return spanish_second_sing

#italian
def generate_first_singular_dataset_italian_e(italian_conjugations):
    italian_first_sing = [f"{verb}: Noi {first_plu}. Tu {second_sing}, Io {first_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in italian_conjugations]
    return italian_first_sing

def generate_second_singular_dataset_italian_e(italian_conjugations):
    italian_second_sing = [f"{verb}: Noi {first_plu}. Io {first_sing}, Tu {second_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in italian_conjugations]
    return italian_second_sing

#german
def generate_first_singular_dataset_german_e(german_conjugations):
    german_first_sing = [f"{verb}: Wir {first_plu}. Du {second_sing}, Ich {first_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in german_conjugations]
    return german_first_sing

def generate_second_singular_dataset_german_e(german_conjugations):
    german_second_sing = [f"{verb}: Wir {first_plu}. Ich {first_sing}, Du {second_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in german_conjugations]
    return german_second_sing

#english
def generate_first_singular_dataset_english_e(english_conjugations):
    english_first_sing = [f"{verb}: We {first_plu}. You {second_sing}, I {first_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in english_conjugations]
    return english_first_sing

def generate_second_singular_dataset_english_e(english_conjugations):
    english_second_sing = [f"{verb}: We {first_plu}. I {first_sing}, You {second_sing}" for verb, first_sing, second_sing, _, first_plu, *__ in english_conjugations]
    return english_second_sing


