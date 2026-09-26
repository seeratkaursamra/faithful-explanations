"""M2 on the test split: IG and LIME over sentences, the decoupled control's rationale for BERT's labels,
and erasure on BERT of the sentences that rationale cites, written as harness records."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import BertForSequenceClassification, BertTokenizer

from . import lime_sentences
from .attribution import explain as integrated_gradients
from .attribution import randomised_head
from .data import NO_SYMPTOMS, binarizer, check_against_paper, load, split
from .llm import ask
from .metrics import erasure, rank_agreement
from .run_m3 import numbered
from .sentences import split as split_sentences
from .sentences import texts, truncated
from .train_bert import pick_device


def bert_predict(model, tokenizer, device, max_len=512, batch=8):
    """Posts (each a list of sentences) to a probability row per post, over every label.

    Sentences are joined with spaces; BERT's tokenizer treats a space and the post's original line
    break alike, so the full post gets the same tokens it was trained on (tested).
    """
    model.eval()

    def predict(posts):
        rows = []
        for i in range(0, len(posts), batch):
            enc = tokenizer([" ".join(p) for p in posts[i:i + batch]], padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
            with torch.no_grad():
                rows.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
        return np.concatenate(rows)
    return predict


def sentences_seen(text, spans, tokenizer, max_len=512):
    """How many sentences BERT sees at least part of. Truncation keeps the start of a post, so these
    are always the first ones, and their indices are the same as in the whole post."""
    cut = set(truncated(text, spans, tokenizer, max_len))
    return next((i for i in range(len(spans)) if i in cut), len(spans))


def remembered(predict):
    """The same predict, but each distinct post is scored once. LIME asks about the same 100
    perturbations for every label of a post, and BERT scores all labels in one pass."""
    seen = {}

    def cached(posts):
        new = [p for p in dict.fromkeys(map(tuple, posts)) if p not in seen]
        if new:
            seen.update(zip(new, predict([list(p) for p in new])))
        return np.array([seen[tuple(p)] for p in posts])
    return cached


def decoupled(sentences, labels, prompts, ask_fn):
    """The LLM explains BERT's labels, and is held to exactly those labels. {label: (evidence, rationale)},
    empty when the reply never parsed."""
    prompt = prompts["decoupled"].format(numbered_post=numbered(sentences), labels=", ".join(labels))
    result = ask_fn(prompt, prompts["system"], n_sentences=len(sentences), expected=labels)
    if not result["ok"]:
        return {}, result
    return {s["label"]: (s["evidence"], s["rationale"] or None) for s in result["symptoms"]}, result


def post_records(post_id, text, gold_labels, classes, predict, ig, explain_why, lime_cfg, threshold=0.5,
                 seen=lambda text, spans: len(spans), ig_random=None):
    """Harness records for one post. `predict` scores lists of sentences, `ig(text, spans, j)` gives IG
    sentence scores and the completeness gap for label j, `explain_why(sentences, labels)` is the
    decoupled control, `seen(text, spans)` is how many sentences fit in BERT's window, and `ig_random`,
    when given, is IG on the randomised-head copy, for the sanity check. Also returns the decoupled
    result, or None when BERT predicted nothing.

    For M2 the post is the sentences BERT sees. Past the cut, a sentence the control cited would
    erase to exactly 0 and IG could never rank it, which would score as unfaithfulness when BERT
    simply never read it. So the control, LIME and erasure all get only the visible sentences.
    """
    spans = split_sentences(text)
    n_seen = seen(text, spans)
    sentences = texts(text, spans)[:n_seen]
    predict = remembered(predict)
    probs = predict([sentences])[0]
    labels = [c for c, p in zip(classes, probs) if p > threshold and c != NO_SYMPTOMS]

    # the control has nothing to explain when BERT found no symptom
    stated, result = explain_why(sentences, labels) if labels else ({}, None)

    records = []
    for label in sorted(set(labels) | (set(gold_labels) - {NO_SYMPTOMS})):
        j = classes.index(label)
        record = {"model": "M2", "post_id": post_id, "symptom": label, "predicted": label in labels,
                  "gold": label in gold_labels, "stated_evidence": None, "stated_rationale": None,
                  "attribution": {}, "comprehensiveness": None, "sufficiency": None, "confidence": float(probs[j]),
                  "sentences_seen": n_seen}
        if label in labels:
            ig_scores, record["ig_delta"] = ig(text, spans, j)
            if ig_random is not None:
                record["ig_random_spearman"] = rank_agreement(ig_scores, ig_random(text, spans, j)[0])
            lime_scores, record["lime_r2"] = lime_sentences.explain(sentences, lambda ps, j=j: predict(ps)[:, j],
                                                                    **lime_cfg)
            record["attribution"] = {"IG": {str(i): s for i, s in ig_scores.items()},
                                     "LIME": {str(i): s for i, s in lime_scores.items()}}
            if label in stated:
                evidence, record["stated_rationale"] = stated[label]
                record["stated_evidence"] = evidence
                record["comprehensiveness"], record["sufficiency"] = erasure(
                    sentences, set(evidence), float(probs[j]), lambda kept, j=j: float(predict([kept])[0, j]))
        records.append(record)
    return records, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/bert_baseline/seed0/checkpoint",
                    help="seed 0 per the prereg; rerun with another seed's checkpoint for the check")
    ap.add_argument("--data", default="data")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--config", default="configs/attribution.yaml")
    ap.add_argument("--out", default="outputs/records")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="first N test posts only, for a trial run")
    ap.add_argument("--allow-mismatch", action="store_true", help="for the public sample")
    ap.add_argument("--sanity", action="store_true", help="also IG on a randomised head (doubles IG time)")
    args = ap.parse_args()

    posts, _ = load(args.data)
    if not args.allow_mismatch and not check_against_paper(posts)["matches_paper"]:
        raise SystemExit("data does not match the paper, refusing to run")
    _, test = split(posts)
    test = test.head(args.limit) if args.limit else test
    classes = list(binarizer(posts).classes_)
    cfg = yaml.safe_load(Path(args.config).read_text())
    prompts = yaml.safe_load(Path(args.prompts).read_text())

    device = pick_device()
    tokenizer = BertTokenizer.from_pretrained(args.checkpoint)
    model = BertForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    predict = bert_predict(model, tokenizer, device)

    def ig(text, spans, j):
        return integrated_gradients(model, tokenizer, text, spans, j, **cfg["ig"])

    def explain_why(sentences, labels):
        return decoupled(sentences, labels, prompts, ask)

    def seen(text, spans):
        return sentences_seen(text, spans, tokenizer)

    ig_random = None
    if args.sanity:
        random = randomised_head(model, cfg["sanity"]["seed"])

        def ig_random(text, spans, j):
            return integrated_gradients(random, tokenizer, text, spans, j, **cfg["ig"])

    records, failed, asked = [], [], 0
    for row in test.itertuples():
        recs, result = post_records(row.post_id, row.text, row.labels, classes, predict, ig, explain_why,
                                    cfg["lime"], args.threshold, seen, ig_random)
        records += recs
        if result is not None:
            asked += 1
            failed += [] if result["ok"] else [row.post_id]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    deltas = [abs(r["ig_delta"]) for r in records if "ig_delta" in r]
    log = {"posts": len(test), "decoupled_asked": asked, "decoupled_failures": len(failed), "failed_posts": failed,
           "ig_delta_max": max(deltas, default=None), "ig_delta_median": float(np.median(deltas)) if deltas else None,
           "checkpoint": args.checkpoint}
    (out / "m2_log.json").write_text(json.dumps(log, indent=2))
    print(f"{len(records)} records from {len(test)} posts; decoupled failures {len(failed)} of {asked}; "
          f"largest IG completeness gap {log['ig_delta_max']}")


if __name__ == "__main__":
    main()
