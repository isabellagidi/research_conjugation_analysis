import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

model_name = "bigscience/bloom-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # required for BLOOM
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def compute_direct_logit_attribution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_texts: List[str],
    answer_ids: List[int],
    label: str = "Dataset"
) -> Dict:
    """Compute Direct Logit Attribution (DLA) for a list of prompts and gold token IDs."""
    
    model.eval()
    device = next(model.parameters()).device

    activation_cache = {}

    # Hook function to capture residual stream before unembedding
    def save_residual_stream_hook(module, input, output):
        activation_cache["final_resid"] = output[0].detach() if isinstance(output, tuple) else output.detach()

    # Register hook at final layer norm (BLOOM: .transformer.ln_f)
    hook_handle = model.transformer.ln_f.register_forward_hook(save_residual_stream_hook)

    # Run forward pass with hook
    with torch.no_grad():
        encoded = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        output = model(**encoded)

    # Remove hook
    hook_handle.remove()

    # Prepare inputs
    gold_ids = torch.tensor(answer_ids).to(device)
    residual_stream = activation_cache["final_resid"]       # (batch, seq, d_model)
    final_resid = residual_stream[:, -1, :]                 # (batch, d_model)

    # Apply LayerNorm scaling using model weights
    ln_weight = model.transformer.ln_f.weight.data
    normed_resid = torch.nn.functional.layer_norm(
        final_resid, final_resid.shape[-1:], weight=ln_weight
    )

    # Get unembedding vectors for gold tokens
    W_U = model.lm_head.weight.data                         # (vocab, d_model)
    W_U_gold = W_U[gold_ids]                                # (batch, d_model)

    # Compute dot product = direct logit contribution
    direct_logits = torch.einsum("bd,bd->b", normed_resid, W_U_gold)  # (batch,)

    return {
        "label": label,
        "dla_values": direct_logits.tolist(),
        "avg_dla": direct_logits.mean().item()
    }