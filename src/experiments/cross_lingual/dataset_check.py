# crosslingual_shapes.py
import re
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

# -----------------------------
# Configuration (no CLI needed)
# -----------------------------
MODEL_NAME = "bigscience/bloom-1b1"  

# Languages included (keys must match TEMPLATES below)
LANGS = [
    "catalan",
    "spanish",
    "french",
    "portuguese",
    "german",
    "english",
    "italian",
    "russian",
    "swedish",
]

PERSONS = [
    "first singular",
    "second singular",
    "third singular",
    "first plural",
    "second plural",
    "third plural",
]

# -----------------------------
# Templates & pronouns
# -----------------------------
# Note: these mirror your generate_dataset templates.
TEMPLATES: Dict[str, Tuple[str, Dict[str, str]]] = {
    "catalan": (
        "Conjugació del verb {verb} en present: {pronoun} {conj}",
        {
            "first singular": "Jo",
            "second singular": "Tu",
            "third singular": "Ell",
            "first plural": "Nosaltres",
            "second plural": "Vosaltres",
            "third plural": "Ells",
        },
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
        },
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
        },
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
        },
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
        },
    ),
    "english": (
        "Conjugation of verb {verb} in present tense: {pronoun} {conj}",
        {
            "first singular": "I",
            "second singular": "You",
            "third singular": "He",
            "first plural": "We",
            "second plural": "You",
            "third plural": "They",
        },
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
        },
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
        },
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
        },
    ),
}

# ---------------------------------
# Tokenization shape helper methods
# ---------------------------------
def _word_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, token) for whitespace-delimited tokens."""
    spans = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.start(), m.end(), text[m.start():m.end()]))
    return spans

def _has_offsets(tokenizer) -> bool:
    try:
        enc = tokenizer("x", add_special_tokens=False, return_offsets_mapping=True)
        return "offset_mapping" in enc
    except Exception:
        return False

def _tokens_per_word_with_offsets(sentence: str, tokenizer) -> Tuple[Tuple[int, ...], int]:
    spans = _word_spans(sentence)
    if not spans:
        return tuple(), 0

    enc = tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    counts = [0] * len(spans)

    for (tok_start, tok_end) in offsets:
        if tok_end <= tok_start:
            continue
        center = (tok_start + tok_end) / 2.0
        for j, (w_start, w_end, _) in enumerate(spans):
            if w_start <= center < w_end:
                counts[j] += 1
                break

    return tuple(counts), sum(counts)

def _tokens_per_word_fallback(sentence: str, tokenizer) -> Tuple[Tuple[int, ...], int]:
    words = re.findall(r"\S+", sentence)
    if not words:
        return tuple(), 0

    counts = []
    total = 0
    for k, w in enumerate(words):
        piece = w if k == 0 else (" " + w)
        n = len(tokenizer.tokenize(piece))
        counts.append(n)
        total += n
    return tuple(counts), total

def tokenization_shape(sentence: str, tokenizer) -> Tuple[Tuple[int, ...], int]:
    """Return per-word subtoken counts and total subtoken count."""
    if _has_offsets(tokenizer):
        return _tokens_per_word_with_offsets(sentence, tokenizer)
    else:
        return _tokens_per_word_fallback(sentence, tokenizer)

def crosslingual_shape_index(
    lang_to_sentences: Dict[str, List[str]],
    tokenizer
) -> Dict[Tuple[int, ...], Dict[str, List[int]]]:
    """
    shape -> { language_name : [indices into lang_to_sentences[language_name]] }
    """
    index: Dict[Tuple[int, ...], Dict[str, List[int]]] = {}
    for lang, sents in lang_to_sentences.items():
        for i, s in enumerate(sents):
            shape, _ = tokenization_shape(s, tokenizer)
            index.setdefault(shape, {}).setdefault(lang, []).append(i)
    return index

def describe_shape(shape: Tuple[int, ...]) -> str:
    return f"words={len(shape)} | per-word={list(shape)} | total_subtokens={sum(shape)}"

# -----------------------------
# Barebones sentence generator
# -----------------------------
def barebones_sentence(language: str, person: str) -> str:
    """
    Produce the template-only sentence for (language, person) with
    verb/conjugation removed, pronoun kept. Collapses extra whitespace.
    """
    if language not in TEMPLATES:
        raise ValueError(f"Unsupported language: {language}")
    template, pronoun_map = TEMPLATES[language]
    if person not in pronoun_map:
        raise ValueError(f"Unsupported person: {person}")

    pronoun = pronoun_map[person]
    s = template.format(verb="", conj="", pronoun=pronoun)

    # Clean up extra spaces produced by empty {verb}/{conj}
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([:;,])", r"\1", s)   # no space before punctuation
    s = s.strip()
    return s

# -----------------------------
# Main: compute and print sets
# -----------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== Cross-lingual template-only tokenization shape matches ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Languages: {LANGS}")
    print("Note: sentences use templates with verb/conjugation removed; pronoun kept.\n")

    for person in PERSONS:
        # One barebones sentence per language for this person
        lang_to_sentences: Dict[str, List[str]] = {}
        for lang in LANGS:
            try:
                sent = barebones_sentence(lang, person)
                lang_to_sentences[lang] = [sent]
            except ValueError:
                # skip unsupported combos
                continue

        # Build cross-lingual index and keep shapes shared by >= 2 languages
        idx = crosslingual_shape_index(lang_to_sentences, tokenizer)
        shared = [(shape, bucket) for shape, bucket in idx.items() if len(bucket) >= 2]
        if not shared:
            print(f"[{person}] No shared shapes across languages.\n")
            continue

        # Sort by: descending #langs, then total subtokens, then #words
        def sort_key(item):
            shape, bucket = item
            return (-len(bucket), sum(shape), len(shape))
        shared.sort(key=sort_key)

        print(f"### PERSON: {person}")
        for shape, bucket in shared:
            langs_sorted = sorted(bucket.keys())
            shape_desc = describe_shape(shape)
            # Print languages sharing the shape
            langs_str = ", ".join(langs_sorted)
            print(f"- {shape_desc}  |  languages: {langs_str}")
        print()

if __name__ == "__main__":
    main()
