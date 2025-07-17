from transformers import AutoTokenizer

import sys
sys.path.append('../../')  

from src.datasets.spanish.spanish_verbs import spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")

#combine ar, er, ir verbs
def create_spanish_verbs(spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs):
    return spanish_ar_verbs + spanish_er_verbs + spanish_ir_verbs

#filter by tokens
def filter_spanish_conjugations(spanish_verbs, tokenizer):
    spanish_conjugations = []
    print("Kept verb forms:\n")

    for verb, yo, tu, vtype, regularity in spanish_verbs:
        yo_tok = tokenizer.tokenize(" " + yo) #NEED TO ADD A SPACE SO TOKENIZES CORRECTLY
        tu_tok = tokenizer.tokenize(" " + tu) #NEED TO ADD A SPACE SO TOKENIZES CORRECTLY

        condition_1 = len(yo_tok) == 1 and len(tu_tok) == 1
        condition_2 = (
            len(yo_tok) == len(tu_tok) and
            yo_tok[:-1] == tu_tok[:-1]
        )

        if condition_1 or condition_2:
            spanish_conjugations.append((verb, yo, tu, vtype, regularity, len(yo_tok), len(tu_tok)))
            #print(f"Infinitive: {verb}")
            #print(f"  Yo: {yo} -> {yo_tok}")
            #print(f"  Tú: {tu} -> {tu_tok}")
            #print("-" * 40)

    return spanish_conjugations

#turn into actual dataset
def generate_yo_dataset(spanish_conjugations):
    spanish_yo = [f"Conjugación del verbo {verb} en presente: Yo {yo_form}" for verb, yo_form, _, _, _, *__ in spanish_conjugations]
    return spanish_yo

# Generate sentences for 'tú' form
def generate_tu_dataset(spanish_conjugations):
    spanish_tu = [f"Conjugación del verbo {verb} en presente: Tú {tu_form}" for verb, _, tu_form, _, _, *__ in spanish_conjugations]
    return spanish_tu

#initiate verb dataset
spanish_verbs = create_spanish_verbs(spanish_ar_verbs, spanish_er_verbs, spanish_ir_verbs)

#filter verbs
spanish_conjugations = filter_spanish_conjugations(spanish_verbs, tokenizer)
print(f"\n✅ Kept {len(spanish_conjugations)} out of {len(spanish_verbs)} ({len(spanish_conjugations)/len(spanish_verbs):.2%})")

#turn into actual dataset
spanish_yo = generate_yo_dataset(spanish_conjugations)
spanish_tu = generate_tu_dataset(spanish_conjugations)

# Display sentences
#print("spanish_yo = [")
#for sentence in spanish_yo:
    #print(f'    "{sentence}",')
#print("]")

#print("\nspanish_tu = [")
#for sentence in spanish_tu:
    #print(f'    "{sentence}",')
#print("]")

