import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "bigscience/bloom-1b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # required for BLOOM
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def evaluate_prompts(text_prompts, answer_ids, label="Dataset"):
    encoded = tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    gold_ids = torch.tensor(answer_ids).to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    final_logits = logits[:, -1, :]  # next-token prediction

    probs = torch.softmax(final_logits, dim=-1)
    gold_probs = probs.gather(dim=-1, index=gold_ids.unsqueeze(-1)).squeeze(-1)

    top1_preds = torch.argmax(final_logits, dim=-1)
    top1_correct = (top1_preds == gold_ids).int()
    top1_acc = top1_correct.float().mean().item()

    sorted_logits, sorted_indices = torch.sort(final_logits, descending=True)
    gold_ranks = (sorted_indices == gold_ids.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1

    return {
        "label": label,
        "accuracy": top1_acc,
        "avg_gold_prob": gold_probs.mean().item(),
        "avg_gold_rank": gold_ranks.float().mean().item(),
        "num_total": len(answer_ids),
        "num_correct": int(top1_correct.sum().item())
    }