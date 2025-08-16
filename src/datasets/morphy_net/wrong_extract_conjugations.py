import os
import glob
import pandas as pd
import json
from zipfile import ZipFile
from io import TextIOWrapper

MORPHYNET_DIR = "MorphyNet"  # Update this if needed
OUTPUT_FILE = "MorphyNet_all_conjugations.json"
result = {}

def is_verb(features_str):
    if not isinstance(features_str, str) or pd.isna(features_str):
        return False
    feats = features_str.split(';')
    first_part = feats[0]
    return first_part.startswith('V')

def process_df(df, lang):
    for _, row in df.iterrows():
        feats = row['features']
        if not isinstance(feats, str) or pd.isna(feats):
            continue
        if not is_verb(feats):
            continue
        if 'V.PTCP' in feats:
            continue  # Skip participles

        tags = feats.split(';')
        lemma = row['lemma']
        inflected = row['inflected_form']

        if 'PRS' not in feats:
            continue  # Ensure we're processing present tense only
        if 'SG' not in feats and 'PL' not in feats:
            continue  # Ensure we process singular or plural forms

        persons = []
        if '1' in tags:  # 1st person
            if 'SG' in tags:
                persons.append('1st_person_singular')
            elif 'PL' in tags:
                persons.append('1st_person_plural')
        if '2' in tags:  # 2nd person
            if 'SG' in tags:
                persons.append('2nd_person_singular')
            elif 'PL' in tags:
                persons.append('2nd_person_plural')
        if '3' in tags:  # 3rd person
            if 'SG' in tags:
                persons.append('3rd_person_singular')
            elif 'PL' in tags:
                persons.append('3rd_person_plural')

        if lang not in result:
            result[lang] = {}
        if lemma not in result[lang]:
            result[lang][lemma] = {'1st_person_singular': None, '2nd_person_singular': None, 
                                    '3rd_person_singular': None, '1st_person_plural': None,
                                    '2nd_person_plural': None, '3rd_person_plural': None}

        for person in persons:
            if result[lang][lemma][person] is None:
                result[lang][lemma][person] = inflected

        # For English: manually copy the lemma (infinitive) to 1st, 2nd, and 3rd person singular/plural forms if empty
        if lang == "eng":
            # Copy lemma (infinitive) to missing singular forms
            if result[lang][lemma]["1st_person_singular"] is None:
                result[lang][lemma]["1st_person_singular"] = lemma
            if result[lang][lemma]["2nd_person_singular"] is None:
                result[lang][lemma]["2nd_person_singular"] = lemma
            if result[lang][lemma]["3rd_person_singular"] is None:
                result[lang][lemma]["3rd_person_singular"] = lemma

            # Copy lemma (infinitive) to missing plural forms
            if result[lang][lemma]["1st_person_plural"] is None:
                result[lang][lemma]["1st_person_plural"] = lemma
            if result[lang][lemma]["2nd_person_plural"] is None:
                result[lang][lemma]["2nd_person_plural"] = lemma
            if result[lang][lemma]["3rd_person_plural"] is None:
                result[lang][lemma]["3rd_person_plural"] = lemma

def process_tsv(filepath, lang):
    try:
        df = pd.read_csv(
            filepath, sep='\t', header=None,
            names=['lemma', 'inflected_form', 'features', 'morpheme_segmentation'],
            dtype=str)
        process_df(df, lang)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def process_zip(filepath, lang):
    try:
        with ZipFile(filepath) as z:
            for file_name in z.namelist():
                if file_name.endswith('.tsv'):
                    with z.open(file_name) as f:
                        df = pd.read_csv(
                            TextIOWrapper(f, encoding='utf-8'), sep='\t', header=None,
                            names=['lemma', 'inflected_form', 'features', 'morpheme_segmentation'],
                            dtype=str)
                        process_df(df, lang)
    except Exception as e:
        print(f"Error processing zip {filepath}: {e}")

def main():
    for lang_dir in os.listdir(MORPHYNET_DIR):
        lang_path = os.path.join(MORPHYNET_DIR, lang_dir)
        if not os.path.isdir(lang_path):
            continue
        # Handle all possible .tsv inflectional files (including multi-part)
        tsv_files = glob.glob(os.path.join(lang_path, "*.inflectional*.tsv"))
        for tsv_file in tsv_files:
            lang_code = lang_dir  # Use directory as language code
            process_tsv(tsv_file, lang_code)
        # Handle any .zip inflectional files
        zip_files = glob.glob(os.path.join(lang_path, "*.inflectional*.zip"))
        for zip_file in zip_files:
            lang_code = lang_dir
            process_zip(zip_file, lang_code)
    
    # Save result to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Done! JSON saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
