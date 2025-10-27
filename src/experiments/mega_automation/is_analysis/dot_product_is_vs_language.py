# dot_product_is_vs_languages_barplots.py
# Build 4 bar charts of dot products between the "is" (BLOOM-1b1) attention
# head-out patching tensors and BLOOM-1b7 tensors across 6 languages,
# for the directions: 1sg→2sg, 2sg→1sg, 1sg→3sg, 3sg→1sg.
# Saves to: is/bar graphs/bar_<direction>.png

import os
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

# -------------------
# Config (edit here)
# -------------------
IS_DIR = "is/bloom-1b7"  # where the "is" tensors live
BLOOM1B7_ROOT = os.path.join("results", "bigscience", "bloom-1b7")

LANGUAGES = ["catalan", "spanish", "english", "french", "portuguese", "russian"]

# Only directions covered by bloom-1b7 pairs
DIRECTIONS = ["1sgto2sg", "2sgto1sg", "1sgto3sg", "3sgto1sg"]

# direction -> pair folder
PAIR_FOR_DIRECTION = {
    "1sgto2sg": "1sg-2sg",
    "2sgto1sg": "1sg-2sg",
    "1sgto3sg": "1sg-3sg",
    "3sgto1sg": "1sg-3sg",
}

# Output directory for bar charts (note the space, per request)
BAR_DIR = os.path.join(IS_DIR, "bar graphs")


def load_patch_tensor(path: str) -> torch.Tensor:
    """Load a saved .pt tensor as float32 on CPU."""
    t = torch.load(path, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {path}, got {type(t)}.")
    return t.to(dtype=torch.float32, device="cpu")


def align_tensors(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """
    If shapes differ, crop to overlapping top-left region along each dimension:
    used_shape[d] = min(a.shape[d], b.shape[d]).
    If ranks differ, flatten and crop to min length.
    """
    if a.shape == b.shape:
        return a, b, tuple(a.shape)

    if len(a.shape) != len(b.shape):
        a_flat = a.flatten()
        b_flat = b.flatten()
        n = min(a_flat.numel(), b_flat.numel())
        return a_flat[:n], b_flat[:n], (n,)

    used_shape = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
    slices = tuple(slice(0, m) for m in used_shape)
    return a[slices], b[slices], used_shape


def dot_product(a: torch.Tensor, b: torch.Tensor) -> float:
    """Raw dot product of flattened tensors."""
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    return float(torch.dot(a_flat, b_flat).item())


def collect_direction_values(direction: str) -> Dict[str, float]:
    """
    For a given direction (e.g., '1sgto2sg'), load the "is" tensor and
    each BLOOM-1b7 language tensor, align shapes, and compute dot products.
    Returns a dict {language: dot}.
    """
    values: Dict[str, float] = {}

    # "is" path
    is_path = os.path.join(IS_DIR, f"attn_head_out_all_pos_patch_results_{direction}_is.pt")
    if not os.path.exists(is_path):
        print(f"⚠️ Missing 'is' tensor for {direction}: {is_path}")
        return values

    try:
        t_is = load_patch_tensor(is_path)
    except Exception as e:
        print(f"⚠️ Failed loading {is_path}: {e}")
        return values

    pair_dir = PAIR_FOR_DIRECTION[direction]

    for lang in LANGUAGES:
        other_path = os.path.join(
            BLOOM1B7_ROOT, pair_dir, lang,
            f"attn_head_out_all_pos_patch_results_{direction}_{lang}.pt"
        )
        if not os.path.exists(other_path):
            print(f"⚠️ Missing other-language tensor ({direction}, {lang}): {other_path}")
            continue

        try:
            t_lang = load_patch_tensor(other_path)
        except Exception as e:
            print(f"⚠️ Failed loading {other_path}: {e}")
            continue

        t_is_used, t_lang_used, used_shape = align_tensors(t_is, t_lang)
        dp = dot_product(t_is_used, t_lang_used)
        values[lang] = dp
        print(f"[{direction} | {lang}] dot={dp:.6f} (is={tuple(t_is.shape)} other={tuple(t_lang.shape)} used={used_shape})")

    return values


def plot_bar(direction: str, lang_to_val: Dict[str, float], out_path: str) -> None:
    """
    Make a bar chart for one direction:
      x-axis = languages, y-axis = dot product.
    """
    if not lang_to_val:
        print(f"⚠️ No data to plot for {direction}. Skipping chart.")
        return

    langs = list(lang_to_val.keys())
    vals  = [lang_to_val[l] for l in langs]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(langs, vals)
    plt.title(f"Dot products: is for BLOOM-1b7 — {direction}")
    plt.xlabel("Language")
    plt.ylabel("Dot product")

    # Annotate bar values
    for rect, val in zip(bars, vals):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height,
                 f"{val:.3g}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved bar chart: {out_path}")


def main() -> None:
    os.makedirs(BAR_DIR, exist_ok=True)

    for direction in DIRECTIONS:
        values = collect_direction_values(direction)
        out_file = os.path.join(BAR_DIR, f"bar_{direction}.png")
        plot_bar(direction, values, out_file)


if __name__ == "__main__":
    main()
