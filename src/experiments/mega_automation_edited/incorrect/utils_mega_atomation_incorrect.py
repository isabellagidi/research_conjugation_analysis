import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from typing import List, Dict
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import transformer_lens.patching as patching

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PERSON_TO_TUPLE_INDEX = {
    "first singular": 1,
    "second singular": 2,
    "third singular": 3,
    "first plural": 4,
    "second plural": 5,
    "third plural": 6,
}

PERSON_TO_JSON_KEY = {
    "first singular":  "1st_person_singular",
    "second singular": "2nd_person_singular",
    "third singular":  "3rd_person_singular",
    "first plural":    "1st_person_plural",
    "second plural":   "2nd_person_plural",
    "third plural":    "3rd_person_plural",
}

PERSON_SHORT_TAG = {
    "first singular":  "1sg",
    "second singular": "2sg",
    "third singular":  "3sg",
    "first plural":    "1pl",
    "second plural":   "2pl",
    "third plural":    "3pl",
}

# Load the JSON dataset
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

## to make plotting labels bigger   
mpl.rcParams.update({
    "axes.titlesize": 18,   # heatmap title
    "axes.labelsize": 14,   # x/y axis labels
    "xtick.labelsize": 11,  # x tick labels
    "ytick.labelsize": 11,  # y tick labels
    "figure.titlesize": 12, # plt.suptitle if you ever use it
})

def save_heatmap(data, x_labels, y_labels, title, filename, center=0.0, cmap="coolwarm"):
    plt.figure(figsize=(min(len(x_labels) * 0.5, 18), min(len(y_labels) * 0.5, 12)))

    # --- Flip so later layers (higher index) are drawn at the TOP ---
    arr = data.detach().cpu().numpy()   # safe if it's a Tensor with grad
    arr = arr[::-1, :]                  # reverse rows
    y_labels_rev = list(y_labels)[::-1] # reverse layer labels
    # ----------------------------------------------------------------

    sns.heatmap(
        arr,                             # <-- use reversed array
        xticklabels=x_labels,
        yticklabels=y_labels_rev,        # <-- use reversed labels
        center=center,
        cmap=cmap,
        cbar_kws={"label": "Normalized Recovery"}
    )
    plt.xlabel("Position" if "resid" in title else "Head")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"✅ Saved: {filename}")
    plt.close()


def filter_conjugations(all_verbs, tokenizer, language, person_a, person_b):
    """
    Keep verbs where both requested person forms exist and either:
      (i) each is a single token, or
      (ii) they share all but the last subtoken.
    Returns tuples in the same layout:
      (verb, 1sg, 2sg, 3sg, 1pl, 2pl, 3pl)
    """

    # resolve which two forms to compare
    try:
        key_a = PERSON_TO_JSON_KEY[person_a]
        key_b = PERSON_TO_JSON_KEY[person_b]
    except KeyError as e:
        raise ValueError(
            f"person_a/person_b must be one of: {list(PERSON_TO_JSON_KEY.keys())}"
        ) from e

    kept_conjugations = []

    # Get the verbs for the specified language
    language_data = all_verbs.get(language, {})
    print(f"total number of verbs for language {language}: ", len(language_data))
    print(f"Filtering pair: {person_a} vs {person_b}  (language={language})\n")

    if not language_data:
        print(f"No data found for language: {language}")
        return kept_conjugations

    # Iterate over each verb in the specified language
    for lemma, forms in language_data.items():
        # Pull all six canonical present forms to preserve tuple layout
        first_sg  = forms.get("1st_person_singular", None)
        second_sg = forms.get("2nd_person_singular", None)
        third_sg  = forms.get("3rd_person_singular", None)
        first_pl  = forms.get("1st_person_plural", None)
        second_pl = forms.get("2nd_person_plural", None)
        third_pl  = forms.get("3rd_person_plural", None)

        # Required pair for filtering
        form_a = forms.get(key_a, None)
        form_b = forms.get(key_b, None)
        if form_a is None or form_b is None:
            continue  # Skip if either requested form is missing

        # Tokenize with an additional space for proper tokenization (BPE-friendly)
        toks_a = tokenizer.tokenize(" " + form_a)
        toks_b = tokenizer.tokenize(" " + form_b)

        # Condition 1: Single tokens for both forms
        cond_single = (len(toks_a) == 1 and len(toks_b) == 1)

        # Condition 2: The first (n-1) tokens are identical
        cond_shared_prefix = (
            len(toks_a) == len(toks_b) and
            toks_a[:-1] == toks_b[:-1]
        )

        # If either condition is satisfied, keep this verb
        if cond_single or cond_shared_prefix:
            kept_conjugations.append(
                (lemma, first_sg, second_sg, third_sg, first_pl, second_pl, third_pl)
            )
            # Debug prints left commented for parity with the original
            # print(f"Infinitive: {lemma}")
            # print(f"  {person_a}: {form_a} -> {toks_a}")
            # print(f"  {person_b}: {form_b} -> {toks_b}")
            # print("-" * 40)

    print("after filter_conjugations, number of saved verbs:", len(kept_conjugations))
    return kept_conjugations

from transformers import AutoModelForCausalLM          # <-- import
import torch

def accuracy_filter(
    prompts_trimmed,           # list[list[int]]
    answer_ids,                # list[int]
    matched_entries,           # list[Tuple[…]]
    model_name,                # str
    batch_size=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Keep only the (prompt, answer, entry) triples for which the model's
    next-token prediction equals the gold token.

    Returned `correct_prompts` is **still a list of token-ID lists**,
    identical to what you passed in (except incorrect items are gone).
    No extra padding tokens are introduced.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    correct_p, correct_a, correct_e = [], [], []
    incorrect_p, incorrect_a, incorrect_e = [], [], []

    # Work in minibatches
    for i in range(0, len(prompts_trimmed), batch_size):
        tok_batch  = prompts_trimmed[i : i + batch_size]     # original token IDs
        gold_batch = answer_ids[i : i + batch_size]
        meta_batch = matched_entries[i : i + batch_size]

        # Convert *tokens*→text so the HF tokenizer can repack them;
        # this does NOT alter tok_batch itself.
        text_batch = [tokenizer.decode(toks, skip_special_tokens=True) for toks in tok_batch]

        enc = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        gold_ids       = torch.tensor(gold_batch, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # For each sequence, take logits at its own last **non‑pad** position
        seq_ends = attention_mask.sum(dim=1) - 1        # shape (B,)
        preds = torch.stack([
            logits[b, seq_ends[b], :].argmax(dim=-1)
            for b in range(logits.size(0))
        ])

        correct_mask = (preds == gold_ids)

        # Split into correct / incorrect
        for j, is_ok in enumerate(correct_mask.tolist()):
            if is_ok:
                correct_p.append(tok_batch[j])
                correct_a.append(gold_batch[j])
                correct_e.append(meta_batch[j])
            else:
                incorrect_p.append(tok_batch[j])
                incorrect_a.append(gold_batch[j])
                incorrect_e.append(meta_batch[j])

        torch.cuda.empty_cache()
        # --- summary printout ---------------------------------------------
    
    total = len(prompts_trimmed)
    num_correct = len(correct_p)
    print(f"number of verbs conjugated correctly: {num_correct}/{total}")

    return (      # <- what you usually keep
        incorrect_p, incorrect_a, incorrect_e,
        correct_p,   correct_a,   correct_e
    )

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

# -------------------------------------------------------------------------
def prepare_language_dataset(
    lang_iso3: str,          # "spa", "ita", "fra", …
    lang_name: str,          # "spanish", "italian", …
    all_verbs: list,
    tokenizer,
    max_verbs: int = 200,
    person_a = "first singular",
    person_b = "second singular"
):
    """
    Build 1st- and 2nd-person prompt datasets for any language supported
    by `generate_dataset`.  *No* accuracy filtering here.

    Returns
    -------
    p1_tok, a1, e1 : lists   # 1st-person prefix-tokens, gold IDs, metadata
    p2_tok, a2, e2 : lists   # 2nd-person prefix-tokens, gold IDs, metadata
    texts_1, texts_2        # decoded prefix strings (handy for grouping)
    conj_list               # the filtered MorphyNet tuples
    """
    # 1. keep only verbs of this language --------------------------------
    conj_list = filter_conjugations(all_verbs, tokenizer, lang_iso3, person_a, person_b)[:max_verbs]
    print(f"[{lang_name}] verbs kept for {person_a} vs {person_b}: {len(conj_list)}")

    # 2. build plain‑text prompts with generate_dataset ------------------
    first_prompts  = generate_dataset(person_a,  language=lang_name,
                                      conjugations=conj_list)
    second_prompts = generate_dataset(person_b, language=lang_name,
                                      conjugations=conj_list)

    # 3. tokenise → (prefix‑tokens, answer_id) ---------------------------
    def to_prefix_and_gold(prompts):
        prefixes, gold_ids, metas = [], [], []
        for meta, prompt in zip(conj_list, prompts):
            tid = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prefixes.append(tid[:-1])      # prefix‑only tokens
            gold_ids.append(tid[-1])       # last token = conjugated form
            metas.append(meta)             # keep the whole tuple
        return prefixes, gold_ids, metas

    p1_tok, a1, e1 = to_prefix_and_gold(first_prompts)
    p2_tok, a2, e2 = to_prefix_and_gold(second_prompts)

    # 4. human‑readable prefixes (useful for grouping/plots) -------------
    texts_1 = [tokenizer.decode(t, skip_special_tokens=True) for t in p1_tok]
    texts_2 = [tokenizer.decode(t, skip_special_tokens=True) for t in p2_tok]

    return p1_tok, a1, e1, p2_tok, a2, e2, texts_1, texts_2, conj_list
# -------------------------------------------------------------------------

def group_by_token_lengths(
    correct_prompts: List[str],
    correct_answer_ids: List[int],
    matched_entries: List[tuple],
    tokenizer,
    conjugated_index
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
        conjugated = entry[conjugated_index]      # need to edit for (1st person singular) or (2nd person singular)

        # Tokenize with a space to ensure correct word boundaries (BLOOM-specific)
        inf_tokens = tokenizer.tokenize(" " + infinitive)
        conj_tokens = tokenizer.tokenize(" " + conjugated)

        inf_len = len(inf_tokens)
        conj_len = len(conj_tokens)

        grouped[(inf_len, conj_len)].append((prompt, answer_id, entry))

    return grouped

# --- helper to run resid_pre patching for one direction ------------------------
def resid_pre_direction(clean_texts, clean_ans, clean_entries,
                        corrupted_texts, corrupted_ans, corrupted_entries,
                        direction_label, lang_tag, tokenizer, tl_model, clean_index, corrupt_index):
    # ❶  group both sets *separately*

    grouped_clean     = group_by_token_lengths(clean_texts, clean_ans, clean_entries, tokenizer, clean_index)
    grouped_corrupted = group_by_token_lengths(corrupted_texts, corrupted_ans, corrupted_entries, tokenizer, corrupt_index)

    # top k groups **from the clean side**
    top_groups = sorted(grouped_clean.items(),
                        key=lambda kv: len(kv[1]),
                        reverse=True)[:5]

    for (inf_len, conj_len), clean_group in top_groups:
        corrupted_group = grouped_corrupted.get((inf_len, conj_len))
        if corrupted_group is None:
            print(f"↪️  {direction_label}: skipping {(inf_len, conj_len)} - no match")
            continue

        # clip to common length
        m = min(len(clean_group), len(corrupted_group))
        clean_prompts     = [ex[0] for ex in clean_group][:m]
        corrupted_prompts = [ex[0] for ex in corrupted_group][:m]
        model_device = tl_model.cfg.device if hasattr(tl_model, "cfg") else next(tl_model.parameters()).device
        answer_ids   = torch.tensor([ex[1] for ex in clean_group][:m], device=model_device)


        # run patching
        # tokenise ONCE so both halves share the same pad length
        all_prompts      = clean_prompts + corrupted_prompts
        all_tok          = tl_model.to_tokens(all_prompts)
        clean_toks       = all_tok[:m]
        corrupted_toks   = all_tok[m:]
        clean_logits, clean_cache = tl_model.run_with_cache(clean_toks)
        corrupted_logits, _       = tl_model.run_with_cache(corrupted_toks)

        def logit_diff(logits):
            return logits[:, -1, :].gather(1, answer_ids.unsqueeze(1)).mean()

        clean_base     = logit_diff(clean_logits).item()
        corrupted_base = logit_diff(corrupted_logits).item()

        def metric(logits):
            return (logit_diff(logits) - corrupted_base) / (clean_base - corrupted_base + 1e-12)

        patch = patching.get_act_patch_resid_pre(
            tl_model, corrupted_toks, clean_cache, patching_metric=metric
        )

        save_heatmap(
            data=patch,
            x_labels=[f"{tok} {i}" for i, tok in enumerate(
                tl_model.to_str_tokens(clean_toks[0]))],
            y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
            title=f"resid_pre - {direction_label} ({lang_tag}, inf={inf_len}, conj={conj_len})",
            filename=f"resid_pre_{direction_label}_{lang_tag}_inf{inf_len}_conj{conj_len}.png"
        )
        torch.save(patch, f"resid_pre_{direction_label}_{lang_tag}_inf{inf_len}_conj{conj_len}.pt")


# === Run Both Second→First and First→Second Activation Patching ===
# ---------------------------------------------------------------------
def run_attn_head_out_patching(
    tl_model,
    clean_prompts: list[str],
    corrupted_prompts: list[str],
    clean_answer_ids: list[int],
    direction_label: str,
    lang_tag: str = "spanish",
    save_prefix: str = "attn_head_out_all_pos",
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Run TransformerLens get_act_patch_attn_head_out_all_pos for one direction
    (e.g. second→first or first→second) and save the heat-map + .pt tensor.

    Parameters
    ----------
    tl_model : HookedTransformer
    clean_prompts / corrupted_prompts : list[str]
        Plain-text prompts in the “clean” and “corrupted” roles.
    answer_ids : list[int]
        Gold token IDs that correspond to clean_prompts.
    direction_label : str
        Suffix for filenames, e.g. "second2first".
    lang_tag : str
        Only used in plot / file names ("spanish" in your case).
    save_prefix : str
        Prefix for plot / tensor filenames.
    device : torch.device | str
    """
    n = min(len(clean_prompts), len(corrupted_prompts), len(clean_answer_ids))
    if n == 0:
        print(f"↪️  Skipping {direction_label} - zero aligned prompts")
        return None

    clean_prompts     = clean_prompts[:n]
    corrupted_prompts = corrupted_prompts[:n]
    answer_tensor     = torch.tensor(clean_answer_ids[:n], device=device)

    # --- forward passes -------------------------------------------------
    all_prompts      = clean_prompts + corrupted_prompts   # single call = common pad length
    all_tok          = tl_model.to_tokens(all_prompts)
    clean_tokens     = all_tok[:n]
    corrupted_tokens = all_tok[n:]

    clean_logits, clean_cache = tl_model.run_with_cache(clean_tokens)
    corrupted_logits, _       = tl_model.run_with_cache(corrupted_tokens)

    def logit_diff(logits):
        # Use gold log-probability at the final position (shift-invariant),
        # then average across the batch.
        last = logits[:, -1, :]
        log_probs = last.log_softmax(dim=-1)
        return log_probs.gather(1, answer_tensor.unsqueeze(1)).mean()

    clean_base     = logit_diff(clean_logits).item()
    corrupted_base = logit_diff(corrupted_logits).item()

    def conjugation_metric(logits):
        return (logit_diff(logits) - corrupted_base) / (clean_base - corrupted_base + 1e-12)

    # --- patching -------------------------------------------------------
    patch = patching.get_act_patch_attn_head_out_all_pos(
        tl_model,
        corrupted_tokens,
        clean_cache,
        patching_metric=conjugation_metric,
    )

    # --- save results ---------------------------------------------------
    heatmap_title = f"attn_head_out Activation Patching ({lang_tag}, {direction_label})"
    png_name = f"{save_prefix}_{direction_label}_{lang_tag}.png"
    pt_name  = f"{save_prefix}_patch_results_{direction_label}_{lang_tag}.pt"

    save_heatmap(
        data=patch,
        x_labels=[f"H{h}" for h in range(tl_model.cfg.n_heads)],
        y_labels=[f"L{l}" for l in range(tl_model.cfg.n_layers)],
        title=heatmap_title,
        filename=png_name,
    )
    torch.save(patch, pt_name)
    print(f"✅ Saved tensor: {pt_name}")
    return patch
# ---------------------------------------------------------------------



from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

#SO that you can treat BLOOMz as a BLOOM model in transformerLens but still use the fine-tuned weights!!!
def load_tl_for_bloom_or_bloomz(hf_name: str, device="cuda"):
    """
    hf_name: e.g. "bigscience/bloom-1b1" or "bigscience/bloomz-1b1"
    Returns (tl_model, tokenizer)
    """
    if "/bloomz-" in hf_name:
        # Map bloomz-* -> bloom-* for the TL loader
        base_name = hf_name.replace("bloomz-", "bloom-")
        # Load HF *weights/tokenizer* from BLOOMZ
        hf_model = AutoModelForCausalLM.from_pretrained(hf_name).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        tokenizer.pad_token = tokenizer.eos_token
        # Ask TL to use the BLOOM converter, but with BLOOMZ weights
        tl_model = HookedTransformer.from_pretrained(
            base_name,           # tells TL which converter to use
            hf_model=hf_model,   # actual weights come from BLOOMZ
        ).to(device).eval()
    else:
        # Plain BLOOM (or any officially supported TL name)
        tl_model = HookedTransformer.from_pretrained(hf_name).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        tokenizer.pad_token = tokenizer.eos_token
    return tl_model, tokenizer
