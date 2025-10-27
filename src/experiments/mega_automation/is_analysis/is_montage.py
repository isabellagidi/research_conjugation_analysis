# run_is_patching.py
# Mini "is" experiment: 1sg/2sg/3sg attention head-out patching (no accuracy filter),
# saves individual heatmaps + tensors into ./is/ and builds a 6-panel montage.

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# Reuse your existing helper (saves heatmaps/tensors with expected names)
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
    "third singular":  "He",   # swap to "They" if you prefer
}

EN_CONJ: Dict[str, str] = {
    "first singular":  "am",
    "second singular": "are",
    "third singular":  "is",
}

# Prompts exactly per request:
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
    out_dir: str = "is",
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
                lang_tag="is",                         # shows up in filenames
                device=device,
            )

        torch.cuda.empty_cache()
        gc.collect()

    finally:
        os.chdir(orig_cwd)

# ==========================
# Montage of the 6 heatmaps
# ==========================
def make_is_montage(out_dir: str = "is",
                    save_name: str = "is_montage.png",
                    cols: int = 3,
                    figsize=(18, 10)) -> str:
    """
    Collect the six attention head-out heatmaps for:
      1sg→2sg, 1sg→3sg, 2sg→1sg, 2sg→3sg, 3sg→1sg, 3sg→2sg
    and arrange them in a single figure.

    Returns the path to the saved montage image.
    """
    labels = ["1sgto2sg", "1sgto3sg", "2sgto1sg", "2sgto3sg", "3sgto1sg", "3sgto2sg"]
    files = [os.path.join(out_dir, f"attn_head_out_all_pos_{lab}_is.png") for lab in labels]

    # Load only existing images
    images, present = [], []
    for lab, f in zip(labels, files):
        if os.path.exists(f):
            images.append(plt.imread(f))
            present.append(lab)
        else:
            print(f"⚠️ Missing heatmap (skipping): {f}")

    if not images:
        print("No heatmaps found; montage not created.")
        return ""

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(rows, cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if idx < len(images):
                ax.imshow(images[idx])
                # Pretty title, e.g., "1sg → 2sg"
                ax.set_title(present[idx].replace("to", " → "), fontsize=12)
                idx += 1

    plt.tight_layout()
    out_path = os.path.join(out_dir, save_name)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved montage: {out_path}")
    return out_path

# ============================
# Hard-coded main (no argparse)
# ============================
MODEL_NAME = "bigscience/bloom-1b7"      # ← BLOOM-1b1 (hard-coded)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR    = "is"

if __name__ == "__main__":
    # Load TL model + HF tokenizer
    tl_model  = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 1) Run the mini "is" experiment
    activation_patching(
        tl_model=tl_model,
        tokenizer=tokenizer,
        out_dir=OUT_DIR,
    )

    # 2) Build the 6-panel montage
    make_is_montage(out_dir=OUT_DIR, save_name="is_montage.png", cols=3)
