# patch_selected.py (only the label bits changed)
import os, json, argparse, torch
from transformer_lens import HookedTransformer
from utils_mega_automation import run_attn_head_out_patching, PERSON_SHORT_TAG

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tl_model_name", required=True)
    p.add_argument("--selections_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=64)
    args = p.parse_args()

    with open(args.selections_path, "r", encoding="utf-8") as f:
        sel = json.load(f)

    lang_name = sel["lang_name"]
    texts_1, texts_2 = sel["texts_1"], sel["texts_2"]
    a1, a2           = sel["a1"], sel["a2"]
    pa, pb           = sel.get("person_a", "first singular"), sel.get("person_b", "second singular")
    tag_a, tag_b     = PERSON_SHORT_TAG[pa], PERSON_SHORT_TAG[pb]

    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(args.out_dir)

    tl_model = HookedTransformer.from_pretrained(
        args.tl_model_name, device="cuda", dtype=torch.bfloat16
    ).eval()
    device = tl_model.cfg.device

    run_attn_head_out_patching(
        tl_model, texts_2, texts_1, a2,
        direction_label=f"{tag_b}to{tag_a}", lang_tag=lang_name, device=device,
        chunk_size=args.chunk_size, max_seq_len=args.max_seq_len
    )
    run_attn_head_out_patching(
        tl_model, texts_1, texts_2, a1,
        direction_label=f"{tag_a}to{tag_b}", lang_tag=lang_name, device=device,
        chunk_size=args.chunk_size, max_seq_len=args.max_seq_len
    )

    torch.cuda.empty_cache()
    print("✅ Patching finished.")

if __name__ == "__main__":
    main()
