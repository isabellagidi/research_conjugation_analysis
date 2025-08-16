import os
import glob
import pandas as pd
import json
from zipfile import ZipFile
from io import TextIOWrapper

MORPHYNET_DIR = "MorphyNet"  # Update this if needed
OUTPUT_FILE = "MorphyNet_1st_2nd_person_sg_present.json"
result = {}

def is_verb(features_str):
    if not isinstance(features_str, str) or pd.isna(features_str):
        return False
    feats = features_str.split(';')
    first_part = feats[0]
    return first_part.startswith('V')

def process_df(df, lang):
    # Store candidates for fallback base forms (English only)
    base_candidates = {}

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

        # Collect base form candidates for English
        if lang == "eng" and 'PRS' in feats and '3' in tags and 'SG' in tags:
            # e.g. "eats" => "eat"
            base_candidates[lemma] = lemma

        # Handle standard explicitly labeled forms
        if 'PRS' not in feats:
            continue
        if 'SG' not in feats and '3' in tags:
            continue

        if lang == "eng" and '3' not in tags:
            # Assume English base forms are valid for both 1SG and 2SG
            persons = ['1st_person_singular', '2nd_person_singular']
        else:
            if '1' in tags:
                persons = ['1st_person_singular']
            elif '2' in tags:
                persons = ['2nd_person_singular']
            else:
                continue

        if lang not in result:
            result[lang] = {}
        if lemma not in result[lang]:
            result[lang][lemma] = {'1st_person_singular': None, '2nd_person_singular': None}

        for person in persons:
            if result[lang][lemma][person] is None:
                result[lang][lemma][person] = inflected

    # After normal processing, fill in base forms for English
    if lang == "eng":
        for lemma in base_candidates:
            if lang not in result:
                result[lang] = {}
            if lemma not in result[lang]:
                result[lang][lemma] = {'1st_person_singular': None, '2nd_person_singular': None}
            for person in ['1st_person_singular', '2nd_person_singular']:
                if result[lang][lemma][person] is None:
                    result[lang][lemma][person] = lemma

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
