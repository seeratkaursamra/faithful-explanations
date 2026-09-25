"""Integrated Gradients over BERT's word embeddings, aggregated to my sentences."""
import torch

from .metrics import sentence_scores
from .sentences import token_sentences


def baseline_ids(input_ids, tokenizer):
    """The input with every token replaced by [PAD], except [CLS] and [SEP], which are kept.

    Keeping them means the path only changes the words, so the specials get exactly zero attribution.
    [UNK] is not kept: it stands for a real word in the post.
    """
    keep = (input_ids == tokenizer.cls_token_id) | (input_ids == tokenizer.sep_token_id)
    return torch.where(keep, input_ids, torch.full_like(input_ids, tokenizer.pad_token_id))


def integrated_gradients(model, input_ids, attention_mask, baseline, target, steps=50, batch=10):
    """Per-token attribution to one output logit, and the completeness gap.

    Walks a straight line from the baseline's embeddings to the input's in `steps` equal steps, takes
    the gradient of the target logit at the midpoint of each step, and multiplies their average by
    how far each embedding moved. The attributions should sum to f(input) - f(baseline); the second
    return value is how far they miss, which shrinks as steps grow. Report it.
    """
    model.eval()
    embed = model.get_input_embeddings()
    with torch.no_grad():
        x, x0 = embed(input_ids), embed(baseline)                  # (1, tokens, hidden)
    alphas = (torch.arange(steps, dtype=x.dtype, device=x.device) + 0.5) / steps

    total = torch.zeros_like(x)
    for start in range(0, steps, batch):
        a = alphas[start:start + batch].view(-1, 1, 1)
        path = (x0 + a * (x - x0)).requires_grad_(True)            # (batch, tokens, hidden)
        mask = attention_mask.expand(len(a), -1)
        logits = model(inputs_embeds=path, attention_mask=mask).logits[:, target]
        total += torch.autograd.grad(logits.sum(), path)[0].sum(dim=0, keepdim=True)

    attributions = ((x - x0) * total / steps).sum(dim=-1)[0]      # one number per token
    with torch.no_grad():
        f_x = model(inputs_embeds=x, attention_mask=attention_mask).logits[0, target]
        f_x0 = model(inputs_embeds=x0, attention_mask=attention_mask).logits[0, target]
    delta = float(attributions.sum() - (f_x - f_x0))
    return attributions.detach().cpu().tolist(), delta


def explain(model, tokenizer, text, spans, target, max_len=512, steps=50, batch=10):
    """Sentence scores for one (post, symptom) on M2, plus the completeness gap.

    Tokenised exactly as in training (truncated at max_len), so sentences past the cut get no score.
    """
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids, attention_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    tokens, delta = integrated_gradients(model, input_ids, attention_mask,
                                         baseline_ids(input_ids, tokenizer), target, steps, batch)
    return sentence_scores(tokens, token_sentences(text, spans, tokenizer, max_len)), delta
