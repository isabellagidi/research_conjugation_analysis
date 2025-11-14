#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan saved activation-patching results for bigscience/bloom-1b1,
pick top-10 heads per (language, pair, direction), build datasets,
select wrong/correct subsets (aligned on the SAME verbs),
patch those heads into corrupted runs, and report accuracy improvements.
Also save bar plots per pair/direction.

Usage (example):
  python repair_with_top_heads.py \
    --model_name bigscience/bloom-1b1 \
    --results_root results \
    --morphynet_path /path/to/MorphyNet_all_present_conjugations.json \
    --max_verbs 1300 \
    --max_prompts_head 50 \
    --batch_size 16 \
    --device auto
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

# --- your utilities (from earlier code) ---
from utils_mega_automation import (
    filter_conjugations,
    accuracy_filter,                # used conceptually; we reimplement faster check below
    group_by_token_lengths,
    prepare_language_dataset,
    save_heatmap,
    resid_pre_direction,
    run_attn_head_out_patching,
    load_json_data,
    PERSON_TO_TUPLE_INDEX,
    PERSON_SHORT_TAG,
    PERSON_TO_JSON_KEY,
)

# ------------------------- Config helpers -------------------------

LANG_MAP = {
    "cat": ("catalan",       "cat"),
    "ces": ("czech",         "ces"),
    "deu": ("german",        "deu"),
    "eng": ("english",       "eng"),
    "fin": ("finnish",       "fin"),
    "fra": ("french",        "fra"),
    "hbs": ("serbo-croatian","hbs"),
    "hun": ("hungarian",     "hun"),
    "ita": ("italian",       "ita"),
    "mon": ("mongolian",     "mon"),
    "pol": ("polish",        "pol"),
    "por": ("portuguese",    "por"),
    "rus": ("russian",       "rus"),
    "spa": ("spanish",       "spa"),
    "swe": ("swedish",       "swe"),
}

# Reverse mapping from short tags ('1sg') back to person strings
SHORT_TO_PERSON = {v: k for k, v in PERSON_SHORT_TAG.items()}  # e.g., '1sg' -> 'first singular'

def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device(arg)

# -------------------- File discovery & parsing --------------------

def discover_head_pt_files(results_root: Path, model_name: str) -> List[Path]:
    """
    Find all attn_head_out_all_pos .pt files for a given model under results_root.

    Expected layout:
      results/<org>/<model>/<pair>/<language>/
        attn_head_out_all_pos_patch_results_<direction>_<language>.pt
    """
    parts = model_name.split("/")  # e.g., ["bigscience","bloom-1b1"]
    base = results_root.joinpath(*parts)
    return list(base.rglob("attn_head_out_all_pos_patch_results_*_*.pt"))

PAIR_RE = re.compile(r"(?P<a>\d(?:sg|pl))-(?P<b>\d(?:sg|pl))")  # e.g., "1sg-2sg"
PT_NAME_RE = re.compile(
    r"attn_head_out_all_pos_patch_results_(?P<direction>[0-9](?:sg|pl)to[0-9](?:sg|pl))_(?P<lang>[a-z\-]+)\.pt"
)

def parse_context_from_path(pt_path: Path) -> Tuple[str, str, str]:
    """
    From a PT file path, extract:
      pair_id (like "1sg-2sg"),
      language folder (e.g., "spanish"),
      direction label (e.g., "2sgto1sg").
    """
    lang = pt_path.parent.name
    pair_id = pt_path.parent.parent.name
    m = PT_NAME_RE.match(pt_path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {pt_path.name}")
    direction = m.group("direction")
    return pair_id, lang, direction

def pair_to_persons(pair_id: str) -> Tuple[str, str]:
    """
    "1sg-2sg" -> ("first singular", "second singular") using SHORT_TO_PERSON
    """
    m = PAIR_RE.match(pair_id)
    if not m:
        raise ValueError(f"Bad pair folder name: {pair_id}")
    a_short, b_short = m.group("a"), m.group("b")
    return SHORT_TO_PERSON[a_short], SHORT_TO_PERSON[b_short]

def direction_to_person_shorts(direction: str) -> Tuple[str, str]:
    """
    "2sgto1sg" -> ("2sg", "1sg")
    """
    src, dst = direction.split("to")
    return src, dst

# -------------------- Top-10 head extraction ----------------------

def load_top_k_heads(pt_file: Path, k: int = 10) -> List[Tuple[int, int, float]]:
    """
    Load tensor (layers x heads) and return top-k by value:
    List of (layer, head, value) sorted descending.
    """
    t = torch.load(pt_file, map_location="cpu")
    if t.ndim != 2:
        raise ValueError(f"Expected 2D tensor (layers x heads) in {pt_file}, got {t.shape}")
    arr = t.detach().cpu().numpy()
    L, H = arr.shape
    flat_idx = np.argsort(arr.reshape(-1))[::-1][:k]
    out = []
    for idx in flat_idx:
        l = idx // H
        h = idx % H
        out.append((int(l), int(h), float(arr[l, h])))
    return out

# ------------------------- Accuracy utils -------------------------

def compute_accuracy_on_texts(
    model_name: str,
    texts: List[str],
    gold_ids: List[int],
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Return a boolean array of shape (len(texts),) indicating whether
    the model's next-token argmax equals gold_ids for each text.
    Uses HuggingFace model for speed and padding convenience.
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    gold = torch.tensor(gold_ids, device=device)
    preds_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = hf(input_ids=input_ids, attention_mask=attn).logits
        seq_ends = attn.sum(dim=1) - 1
        batch_preds = torch.stack(
            [logits[b, seq_ends[b], :].argmax(dim=-1) for b in range(logits.size(0))]
        )
        preds_all.append(batch_preds.detach().cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()
    preds = torch.cat(preds_all, dim=0).to(device)
    correct = (preds == gold[: preds.shape[0]])
    return correct.detach().cpu().numpy().astype(bool)

# ------------------- Targeted head patching -----------------------

def collect_z_hook_names(cache_keys: List[str]) -> Dict[int, str]:
    """
    Given cache keys from a run_with_cache on clean tokens, map layer -> hook_z name.
    We search names ending with 'hook_z'.
    """
    z_names = [k for k in cache_keys if k.endswith("hook_z")]
    out = {}
    for name in z_names:
        m = re.search(r"blocks\.(\d+)\.", name)
        if m:
            out[int(m.group(1))] = name
    if not out:
        raise RuntimeError("Could not locate any 'hook_z' names in cache keys.")
    return out

def patch_selected_heads_and_measure(
    tl_model: HookedTransformer,
    clean_texts: List[str],
    corrupted_texts: List[str],
    gold_ids: List[int],
    heads: List[Tuple[int, int]],  # list of (layer, head)
    device: torch.device,
) -> Tuple[float, float]:
    """
    Returns (baseline_accuracy, patched_accuracy) over the corrupted_texts.

    Strategy:
      - Build a shared pad length by concatenating clean+corrupt, then slice back.
      - Run clean with cache(names_filter=hook_z) to get per-head z.
      - Run corrupted baseline to get accuracy.
      - Run corrupted with fwd hooks that overwrite (layer, head) z with clean z.
    """
    n = min(len(clean_texts), len(corrupted_texts), len(gold_ids))
    if n == 0:
        return 0.0, 0.0

    clean_texts = clean_texts[:n]
    corrupted_texts = corrupted_texts[:n]
    gold_tensor = torch.tensor(gold_ids[:n], device=device)

    # tokens with shared pad length
    all_prompts = clean_texts + corrupted_texts
    all_tok = tl_model.to_tokens(all_prompts).to(device)
    clean_tok = all_tok[:n]
    corrupt_tok = all_tok[n:]

    # --- get clean cache (only need hook_z) ---
    clean_logits, clean_cache = tl_model.run_with_cache(
        clean_tok, names_filter=lambda s: s.endswith("hook_z")
    )
    cache_keys = list(clean_cache.keys())
    z_name_by_layer = collect_z_hook_names(cache_keys)

    # --- baseline accuracy on corrupted ---
    with torch.no_grad():
        base_logits = tl_model(corrupt_tok)
    base_last = base_logits[:, -1, :]
    base_preds = base_last.argmax(dim=-1)
    baseline_acc = (base_preds == gold_tensor).float().mean().item()

    # --- build forward hooks to patch selected heads ---
    layer_to_heads: Dict[int, List[int]] = defaultdict(list)
    for (l, h) in heads:
        layer_to_heads[l].append(h)

    fwd_hooks = []
    for layer, head_list in layer_to_heads.items():
        hook_name = z_name_by_layer.get(layer, None)
        if hook_name is None:
            # Layer not present? Skip gracefully.
            continue

        clean_z = clean_cache[hook_name]  # [n, pos, n_heads, d_head]

        def make_hook(clean_z_ref, head_indices: List[int]):
            def _hook(z, hook):
                # z: [n, pos, n_heads, d_head]; replace selected heads in-place
                z = z.clone()  # be safe: some TL versions want a returned tensor
                for h in head_indices:
                    z[:, :, h, :] = clean_z_ref[:, :, h, :].to(z.device)
                return z
            return _hook

        fwd_hooks.append((hook_name, make_hook(clean_z, head_list)))

    # --- run with patches ---
    with torch.no_grad():
        patched_logits = tl_model.run_with_hooks(corrupt_tok, fwd_hooks=fwd_hooks)
    patched_last = patched_logits[:, -1, :]
    patched_preds = patched_last.argmax(dim=-1)
    patched_acc = (patched_preds == gold_tensor).float().mean().item()

    return baseline_acc, patched_acc

# ---------------------- Main orchestration ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="e.g., bigscience/bloom-1b1")
    parser.add_argument("--results_root", default="results", help="Root where previous .pt files live")
    parser.add_argument("--morphynet_path", required=True, help="Path to MorphyNet_all_present_conjugations.json")
    parser.add_argument("--max_verbs", type=int, default=1300)
    parser.add_argument("--max_prompts_head", type=int, default=50)
    parser.add_argument("--top_k_heads", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--out_dir", default="results_summary")
    args = parser.parse_args()

    device = resolve_device(args.device)

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) / args.model_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all head-out .pt files for this model
    pt_files = discover_head_pt_files(results_root, args.model_name)
    if not pt_files:
        raise SystemExit(f"No head-out .pt files found under {results_root} for {args.model_name}")

    # Load model(s)
    tl_model = HookedTransformer.from_pretrained(args.model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load verb data
    all_verbs = load_json_data(args.morphynet_path)

    # Track improvements for plotting: {(pair, direction): {language: improvement}}
    improvements: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)

    # Cache top heads for reuse
    top_heads_index: Dict[Tuple[str, str, str], List[Tuple[int,int,float]]] = {}

    # Pre-group .pt files by (pair, lang, direction)
    for pt in pt_files:
        try:
            pair_id, lang_name, direction = parse_context_from_path(pt)
        except ValueError:
            continue
        key = (pair_id, lang_name, direction)
        top10 = load_top_k_heads(pt, k=args.top_k_heads)
        top_heads_index[key] = top10

    # Iterate over each (pair, lang, direction)
    for (pair_id, lang_name, direction), heads_lhv in sorted(top_heads_index.items()):
        # Convert to (layer, head)
        top10_heads = [(l, h) for (l, h, _val) in heads_lhv]

        # Resolve the two persons in the PAIR (order matches folder naming A-B)
        person_a, person_b = pair_to_persons(pair_id)

        # Resolve direction "XtoY" -> which is the clean and corrupted person
        src_short, dst_short = direction_to_person_shorts(direction)
        person_src = SHORT_TO_PERSON[src_short]  # clean
        person_dst = SHORT_TO_PERSON[dst_short]  # corrupted

        # Resolve iso3 for this language name
        name_to_iso = {v[0]: k for k, v in LANG_MAP.items()}
        if lang_name not in name_to_iso:
            print(f"⚠️  Skipping unknown language folder '{lang_name}' (not in LANG_MAP).")
            continue
        iso3_key = LANG_MAP[name_to_iso[lang_name]][1]  # e.g., "spa"

        # ---- Build datasets (like before) ----
        (pA_raw, aA, eA,
         pB_raw, aB, eB,
         texts_A, texts_B, _) = prepare_language_dataset(
             lang_iso3  = iso3_key,
             lang_name  = lang_name,
             all_verbs  = all_verbs,
             tokenizer  = tokenizer,
             max_verbs  = args.max_verbs,
             person_a   = person_a,
             person_b   = person_b,
        )

        # Map persons to blobs
        person_to_blobs = {
            person_a: dict(tokens=pA_raw, gold=aA, entries=eA, texts=texts_A),
            person_b: dict(tokens=pB_raw, gold=aB, entries=eB, texts=texts_B),
        }

        # CLEAN (src) and CORRUPTED (dst) sets for this direction
        clean_blobs   = person_to_blobs[person_src]
        corrupt_blobs = person_to_blobs[person_dst]

        # Decode prefixes back to text for HF accuracy check
        clean_texts_all   = [tokenizer.decode(t, skip_special_tokens=True) for t in clean_blobs["tokens"]]
        corrupt_texts_all = [tokenizer.decode(t, skip_special_tokens=True) for t in corrupt_blobs["tokens"]]

        # Gold IDs always taken from the CLEAN side (same verb order)
        gold_ids_all = clean_blobs["gold"]

        # --- Select SAME-VERB indices: CLEAN=correct AND CORRUPT=wrong ---
        clean_correct_mask = compute_accuracy_on_texts(
            args.model_name, clean_texts_all, gold_ids_all, device=torch.device("cpu"), batch_size=args.batch_size
        )
        corrupt_correct_mask = compute_accuracy_on_texts(
            args.model_name, corrupt_texts_all, gold_ids_all, device=torch.device("cpu"), batch_size=args.batch_size
        )

        joint_idx = [i for i in range(len(gold_ids_all)) if clean_correct_mask[i] and not corrupt_correct_mask[i]]

        if len(joint_idx) == 0:
            print(f"↪️  [{pair_id} | {lang_name} | {direction}] no verbs where clean is correct AND corrupted is wrong. Skipping.")
            improvements[(pair_id, direction)][lang_name] = 0.0
            continue

        # Cap by max_prompts_head
        joint_idx = joint_idx[:args.max_prompts_head]

        clean_texts = [clean_texts_all[i] for i in joint_idx]
        corrupt_texts = [corrupt_texts_all[i] for i in joint_idx]
        gold_ids = [gold_ids_all[i] for i in joint_idx]

        # --- Patch selected heads and measure improvement ---
        base_acc, patched_acc = patch_selected_heads_and_measure(
            tl_model=tl_model,
            clean_texts=clean_texts,
            corrupted_texts=corrupt_texts,
            gold_ids=gold_ids,
            heads=[(l, h) for (l, h) in top10_heads],
            device=device,
        )
        improvement = patched_acc - base_acc
        improvements[(pair_id, direction)][lang_name] = improvement

        # Save a small JSON drop per case
        drop = {
            "model": args.model_name,
            "pair": pair_id,
            "language": lang_name,
            "direction": direction,
            "top_heads": [{"layer": int(l), "head": int(h), "score": float(v)} for (l, h, v) in heads_lhv],
            "baseline_acc": base_acc,
            "patched_acc": patched_acc,
            "improvement": improvement,
            "n_eval": len(gold_ids),
        }
        out_case = out_dir / f"improvement_{pair_id}_{direction}_{lang_name}.json"
        with open(out_case, "w", encoding="utf-8") as f:
            json.dump(drop, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {out_case}  (Δ={improvement:.3f}, n={len(gold_ids)})")

    # ---------------------- Aggregate & plots ----------------------

    # Master CSV
    csv_lines = ["pair,direction,language,improvement"]
    for (pair_id, direction), lang_dict in sorted(improvements.items()):
        for lang_name, imp in sorted(lang_dict.items()):
            csv_lines.append(f"{pair_id},{direction},{lang_name},{imp:.6f}")
    csv_path = out_dir / "improvements_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
    print(f"📄 Wrote {csv_path}")

    # Bar plots: one per (pair, direction), bars = languages
    for (pair_id, direction), lang_dict in sorted(improvements.items()):
        if not lang_dict:
            continue
        langs = list(lang_dict.keys())
        vals = [lang_dict[l] for l in langs]
        plt.figure(figsize=(max(6, 0.6*len(langs)), 4.5))
        sns.barplot(x=langs, y=vals)
        plt.title(f"Accuracy Improvement: {pair_id} / {direction}")
        plt.ylabel("Δ Accuracy (patched - baseline)")
        plt.xlabel("Language")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        png_path = out_dir / f"bar_{pair_id}_{direction}.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"🖼️ Saved {png_path}")

if __name__ == "__main__":
    main()
