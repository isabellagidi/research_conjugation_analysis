

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
def generate_first_singular_dataset_spanish_c(spanish_conjugations):
    spanish_first_sing = [f"Conjugación del verbo {verb} en presente en Español: Tú {second_sing}, Yo {first_sing}" for verb, first_sing, second_sing, _, _, *__ in spanish_conjugations]
    return spanish_first_sing

def generate_second_singular_dataset_spanish_c(spanish_conjugations):
    spanish_second_sing = [f"Conjugación del verbo {verb} en presente en Español: Yo {first_sing}, Tú {second_sing}" for verb, first_sing, second_sing, _, _, *__ in spanish_conjugations]
    return spanish_second_sing

#italian
def generate_first_singular_dataset_italian_c(italian_conjugations):
    italian_first_sing = [f"Coniugazione del verbo {verb} al presente in italiano: Tu {second_sing}, Io {first_sing}" for verb, first_sing, second_sing, _, _, *__ in italian_conjugations]
    return italian_first_sing

def generate_second_singular_dataset_italian_c(italian_conjugations):
    italian_second_sing = [f"Coniugazione del verbo {verb} al presente in italiano: Io {first_sing}, Tu {second_sing}" for verb, first_sing, second_sing, _, _, *__ in italian_conjugations]
    return italian_second_sing

#german
def generate_first_singular_dataset_german_c(german_conjugations):
    german_first_sing = [f"Konjugation des Verbs {verb} im Präsens auf Deutsch: Du {second_sing}, Ich {first_sing}" for verb, first_sing, second_sing, _, _, *__ in german_conjugations]
    return german_first_sing

def generate_second_singular_dataset_german_c(german_conjugations):
    german_second_sing = [f"Konjugation des Verbs {verb} im Präsens auf Deutsch: Ich {first_sing}, Du {second_sing}" for verb, first_sing, second_sing, _, _, *__ in german_conjugations]
    return german_second_sing

#english
def generate_first_singular_dataset_english_c(english_conjugations):
    english_first_sing = [f"Conjugation of the verb {verb} in present tense in English: You {second_sing}, I {first_sing}" for verb, first_sing, second_sing, _, _, *__ in english_conjugations]
    return english_first_sing

def generate_second_singular_dataset_english_c(english_conjugations):
    english_second_sing = [f"Conjugation of the verb {verb} in present tense in English: I {first_sing}, You {second_sing}" for verb, first_sing, second_sing, _, _, *__ in english_conjugations]
    return english_second_sing

# DATASET D (simply just a few conjugations before) ex: Hablar: Nosotros hablamos. Tu hablas, yo hablo. 