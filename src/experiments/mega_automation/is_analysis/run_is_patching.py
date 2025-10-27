# === Imports ===
import os
import gc
import torch
from typing import Dict, List, Tuple
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# Keep just what we need from your utilities
from utils_mega_automation import (
    run_attn_head_out_patching,
    PERSON_SHORT_TAG,
)

# =========================
# Mini "is" dataset helpers
# =========================

PERSONS = ["first singular", "second singular", "third singular"]

EN_PRONOUNS: Dict[str, str] = {
    "first singular":  "I",
    "second singular": "You",
    "third singular":  "He",   # could be 'They (sg.)' if you prefer
}

EN_CONJ: Dict[str, str] = {
    "first singular":  "am",
    "second singular": "are",
    "third singular":  "is",
}

# Prompts match your requested wording:
#   Conjugation of the verb "is" in <person>: <Pronoun> <form>
TEMPLATES: Dict[str, str] = {
    "first singular":  'Conjugation of the verb "is" in first person singular: {pronoun} {conj}',
    "second singular": 'Conjugation of the verb "is" in second person singular: {pronoun} {conj}',
    "third singular":  'Conjugation of the verb "is" in third person singular: {pronoun} {conj}',
}

def build_is_prompts_and_answers(tokenizer) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Build one prompt per person (1sg/2sg/3sg) and compute the gold answer id
    as the *last* token of each prompt (the conjugated form).
    """
    prompts: Dict[str, str] = {}
    gold_ids: Dict[str, int] = {}

    for person in PERSONS:
        prompt = TEMPLATES[person].format(
            pronoun=EN_PRONOUNS[person],
            conj=EN_CONJ[person],
        )
        toks = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompts[person] = prompt
        gold_ids[person] = toks[-1]  # last token is the conjugated form

    return prompts, gold_ids


# ===========================================
# NEW: Specialized activation patching runner
# ===========================================
def activation_patching(
    *,
    tl_model: HookedTransformer,
    tokenizer,
    out_dir: str = "is"
):
    """
    Mini-runner that:
      - builds a tiny English dataset for the verb "is": am/are/is (1sg/2sg/3sg),
      - skips accuracy filtering entirely,
      - runs attn_head_out patching for all 6 ordered pairs,
      - writes results to ./is/.
    """
    os.makedirs(out_dir, exist_ok=True)
    orig_cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        device = tl_model.cfg.device

        # 1) Build prompts and gold answer IDs
        prompts, gold_ids = build_is_prompts_and_answers(tokenizer)

        # 2) All 6 ordered pairs among {1sg,2sg,3sg}
        pairs = [
            ("first singular",  "second singular"),
            ("first singular",  "third singular"),
            ("second singular", "first singular"),
            ("second singular", "third singular"),
            ("third singular",  "first singular"),
            ("third singular",  "second singular"),
        ]

        # 3) Run attention head-out patching per direction
        for src, tgt in pairs:
            label = f'{PERSON_SHORT_TAG[src]}to{PERSON_SHORT_TAG[tgt]}'
            run_attn_head_out_patching(
                tl_model=tl_model,
                clean_prompts=[prompts[src]],          # list[str], length 1
                corrupted_prompts=[prompts[tgt]],      # list[str], length 1
                clean_answer_ids=[gold_ids[src]],      # list[int], length 1
                direction_label=label,
                lang_tag="is",                         # will appear in filenames
                device=device,
            )

        torch.cuda.empty_cache()
        gc.collect()

    finally:
        os.chdir(orig_cwd)


# ===== Config you can edit =====
MODEL_NAME = "bigscience/bloom-1b7"          # <— BLOOM-1b1
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR    = "is"
# =================================

if __name__ == "__main__":
    # Load TL model + HF tokenizer
    tl_model  = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Run the mini "is" experiment
    activation_patching(
        tl_model=tl_model,
        tokenizer=tokenizer,
        out_dir=OUT_DIR,
    )
