from transformers import AutoTokenizer

import sys
sys.path.append('../../')  

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")

#combine ar, er, ir verbs
def combine_verbs(verbs_one, verbs_two, verbs_three):
    return verbs_one + verbs_two + verbs_three

#filter by tokens
def filter_conjugations(all_verbs, tokenizer):
    verb_conjugations = []
    print("Kept verb forms:\n")

    for verb, first_sing, second_sing, vtype, regularity in all_verbs:
        first_sing_tok = tokenizer.tokenize(" " + first_sing) #NEED TO ADD A SPACE SO TOKENIZES CORRECTLY
        second_sing_tok = tokenizer.tokenize(" " + second_sing) #NEED TO ADD A SPACE SO TOKENIZES CORRECTLY

        condition_1 = len(first_sing_tok) == 1 and len(second_sing_tok) == 1
        condition_2 = (
            len(first_sing_tok) == len(second_sing_tok) and
            first_sing_tok[:-1] == second_sing_tok[:-1]
        )

        if condition_1 or condition_2:
            verb_conjugations.append((verb, first_sing, second_sing, vtype, regularity, len(first_sing_tok), len(second_sing_tok)))
            print(f"Infinitive: {verb}")
            print(f"  First Person Singular: {first_sing} -> {first_sing_tok}")
            print(f"  Second Person Singular: {second_sing} -> {second_sing_tok}")
            print("-" * 40)

    return verb_conjugations

def build_conjugation_prompts(tokenizer, language_conjugations):
    prompts_trimmed = []
    answers_ids = []
    valid_indices = []

    for idx, prompt in enumerate(language_conjugations):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) < 2:
            continue

        prompt_trimmed = token_ids[:-1]
        answer_id = token_ids[-1]

        prompts_trimmed.append(prompt_trimmed)
        answers_ids.append(answer_id)
        valid_indices.append(idx)

    return prompts_trimmed, answers_ids, valid_indices


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
    spanish_first_sing = [f"Conjugación del verbo {verb} en presente en Español: Yo {first_sing}" for verb, first_sing, _, _, _, *__ in spanish_conjugations]
    return spanish_first_sing

def generate_second_singular_dataset_spanish_b(spanish_conjugations):
    spanish_second_sing = [f"Conjugación del verbo {verb} en presente en Español: Tú {second_sing}" for verb, _, second_sing, _, _, *__ in spanish_conjugations]
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
def generate_first_singular_dataset_spanish_b(spanish_conjugations):
    spanish_first_sing = [f"Conjugación del verbo {verb} en presente en Español: Tú {second_sing}, Yo {first_sing}" for verb, first_sing, second_sing, _, _, *__ in spanish_conjugations]
    return spanish_first_sing

def generate_second_singular_dataset_spanish_b(spanish_conjugations):
    spanish_second_sing = [f"Conjugación del verbo {verb} en presente en Español: Yo {first_sing}, Tú {second_sing}" for verb, first_sing, second_sing, _, _, *__ in spanish_conjugations]
    return spanish_second_sing

#italian
def generate_first_singular_dataset_italian_b(italian_conjugations):
    italian_first_sing = [f"Coniugazione del verbo {verb} al presente in italiano: Tu {second_sing}, Io {first_sing}" for verb, first_sing, second_sing, _, _, *__ in italian_conjugations]
    return italian_first_sing

def generate_second_singular_dataset_italian_b(italian_conjugations):
    italian_second_sing = [f"Coniugazione del verbo {verb} al presente in italiano: Io {first_sing}, Tu {second_sing}" for verb, first_sing, second_sing, _, _, *__ in italian_conjugations]
    return italian_second_sing

#german
def generate_first_singular_dataset_german_b(german_conjugations):
    german_first_sing = [f"Konjugation des Verbs {verb} im Präsens auf Deutsch: Du {second_sing}, Ich {first_sing}" for verb, first_sing, second_sing, _, _, *__ in german_conjugations]
    return german_first_sing

def generate_second_singular_dataset_german_b(german_conjugations):
    german_second_sing = [f"Konjugation des Verbs {verb} im Präsens auf Deutsch: Ich {first_sing}, Du {second_sing}" for verb, first_sing, second_sing, _, _, *__ in german_conjugations]
    return german_second_sing

#english
def generate_first_singular_dataset_english_b(english_conjugations):
    english_first_sing = [f"Conjugation of the verb {verb} in present tense in English: You {second_sing}, I {first_sing}" for verb, first_sing, second_sing, _, _, *__ in english_conjugations]
    return english_first_sing

def generate_second_singular_dataset_english_b(english_conjugations):
    english_second_sing = [f"Conjugation of the verb {verb} in present tense in English: I {first_sing}, You {second_sing}" for verb, first_sing, second_sing, _, _, *__ in english_conjugations]
    return english_second_sing

# DATASET D (simply just a few conjugations before) ex: Hablar: Nosotros hablamos. Tu hablas, yo hablo. 