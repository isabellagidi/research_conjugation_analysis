# select_examples.py
import json, os, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils_mega_automation import (
    load_json_data, prepare_language_dataset, accuracy_filter
)
from utils_mega_automation import PERSON_TO_TUPLE_INDEX, PERSON_SHORT_TAG

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--lang_iso3", required=True)
    p.add_argument("--lang_name", required=True)
    p.add_argument("--person_a", required=True)
    p.add_argument("--person_b", required=True)
    p.add_argument("--max_verbs", type=int, default=1300)
    p.add_argument("--keep_k", type=int, default=50)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--out_path", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer + small forward-only HF model (same model as patching if you prefer)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # keep dtype bf16 if available for memory
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    ).to(device).eval()

    all_verbs = load_json_data(args.dataset_path)

    # Build two sides (but DO NOT load TransformerLens here)
    (p1_raw, a1, e1,
     p2_raw, a2, e2,
     texts_1, texts_2, _) = prepare_language_dataset(
        lang_iso3=args.lang_iso3, lang_name=args.lang_name,
        all_verbs=all_verbs, tokenizer=tokenizer, max_verbs=args.max_verbs,
        person_a=args.person_a, person_b=args.person_b
    )

    # Accuracy filter ON GPU (fast and memory-safe here, TL not loaded)
    p1_tok, a1c, e1c, *_ = accuracy_filter(p1_raw, a1, e1, args.model_name, batch_size=16, device=device)
    p2_tok, a2c, e2c, *_ = accuracy_filter(p2_raw, a2, e2, args.model_name, batch_size=16, device=device)

    # Convert surviving prefixes back to text
    texts_1 = [tokenizer.decode(t, skip_special_tokens=True) for t in p1_tok]
    texts_2 = [tokenizer.decode(t, skip_special_tokens=True) for t in p2_tok]

    # Keep exactly K (min across both sides just in case)
    k = min(args.keep_k, len(texts_1), len(texts_2), len(a1c), len(a2c))
    payload = {
        "model_name": args.model_name,
        "lang_iso3": args.lang_iso3,
        "lang_name": args.lang_name,
        "person_a": args.person_a,
        "person_b": args.person_b,
        "texts_1": texts_1[:k],
        "texts_2": texts_2[:k],
        "a1": a1c[:k],
        "a2": a2c[:k]
    }
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {k} selected examples to {args.out_path}")

if __name__ == "__main__":
    main()
