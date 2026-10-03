"""M2 on the test split: IG and LIME over sentences, the decoupled control's rationale for BERT's labels,
and erasure on BERT of the sentences that rationale cites, written as harness records."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from . import lime_sentences
from .attribution import explain as integrated_gradients
from .attribution import randomised_head
from .data import NO_SYMPTOMS, binarizer, check_against_paper, load, split
from .llm import ask
from .metrics import agreement, erasure, jaccard, rank_agreement, top_k
from .run_m3 import numbered
from .sentences import split as split_sentences
from .sentences import texts, truncated
from .train_bert import pick_device


MAIN_OUT = "outputs/records"
MAIN_CHECKPOINT = "outputs/bert_baseline/seed0/checkpoint"


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


def finite(x):
    """NaN as None, so the records stay valid JSON (a one-sentence post has no Spearman and no LIME fit)."""
    return None if x is None or x != x else x


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
                record["ig_random_spearman"] = finite(rank_agreement(ig_scores, ig_random(text, spans, j)[0]))
            lime_scores, lime_r2 = lime_sentences.explain(sentences, lambda ps, j=j: predict(ps)[:, j],
                                                                    **lime_cfg)
            record["lime_r2"] = finite(lime_r2)
            record["attribution"] = {"IG": {str(i): s for i, s in ig_scores.items()},
                                     "LIME": {str(i): s for i, s in lime_scores.items()}}
            if label in stated:
                evidence, record["stated_rationale"] = stated[label]
                record["stated_evidence"] = evidence
                record["comprehensiveness"], record["sufficiency"] = erasure(
                    sentences, set(evidence), float(probs[j]), lambda kept, j=j: float(predict([kept])[0, j]))
        records.append(record)
    return records, result


def seed_check(run_a, run_b, post_ids):
    """The prereg's check of seed 0 against another seed, on the same test posts. Label agreement per
    post, then, for the (post, symptom) items both seeds predicted, how far IG and LIME agree: median
    Spearman of the sentence scores and mean overlap of the top two sentences."""
    predicted = [{}, {}]
    labels = [{p: set() for p in post_ids} for _ in range(2)]
    for i, records in enumerate([run_a, run_b]):
        for r in records:
            if r["predicted"]:
                labels[i][r["post_id"]].add(r["symptom"])
                predicted[i][(r["post_id"], r["symptom"])] = r["attribution"]
    shared = predicted[0].keys() & predicted[1].keys()
    out = {"n_posts": len(post_ids), "n_shared_predictions": len(shared),
           "labels": float(np.mean([agreement(labels[0][p], labels[1][p]) for p in post_ids]))}
    for method in ("IG", "LIME"):
        pairs = [({int(k): v for k, v in predicted[0][key][method].items()},
                  {int(k): v for k, v in predicted[1][key][method].items()})
                 for key in shared if predicted[0][key].get(method) and predicted[1][key].get(method)]
        rho = [r for r in (rank_agreement(x, y) for x, y in pairs) if not np.isnan(r)]
        out[f"{method}_median_spearman"] = float(np.median(rho)) if rho else None
        out[f"{method}_top2_jaccard"] = float(np.mean([jaccard(top_k(x, 2), top_k(y, 2)) for x, y in pairs])) \
            if pairs else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=MAIN_CHECKPOINT,
                    help="seed 0 per the prereg; rerun with another seed's checkpoint for the check")
    ap.add_argument("--data", default="data")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--config", default="configs/attribution.yaml")
    ap.add_argument("--out", default=MAIN_OUT)
    ap.add_argument("--bert-config", default="configs/bert_baseline.yaml",
                    help="max_len and threshold come from the recipe the checkpoint was trained with")
    ap.add_argument("--threshold", type=float, default=None, help="overrides the recipe's threshold")
    ap.add_argument("--limit", type=int, default=None, help="first N test posts only, for a trial run")
    ap.add_argument("--allow-mismatch", action="store_true", help="for the public sample")
    ap.add_argument("--sanity", action="store_true", help="also IG on a randomised head (doubles IG time)")
    ap.add_argument("--seed-check", nargs=2, metavar="JSONL", help="compare two finished runs (seed 0, seed 1)")
    args = ap.parse_args()
    # the seed check or a trial written into the main folder would replace the records the harness reads
    if args.out == MAIN_OUT and not args.seed_check and (args.checkpoint != MAIN_CHECKPOINT or args.limit):
        raise SystemExit(f"another checkpoint or a limit must not write to {MAIN_OUT}; give it its own --out")

    posts, _ = load(args.data)
    if not args.allow_mismatch and not check_against_paper(posts)["matches_paper"]:
        raise SystemExit("data does not match the paper, refusing to run")
    _, test = split(posts)
    test = test.head(args.limit) if args.limit else test

    if args.seed_check:
        # posts from the split, not the records: a post both seeds leave empty writes no records
        runs = [[json.loads(l) for l in Path(f).read_text().splitlines() if l] for f in args.seed_check]
        print(json.dumps(seed_check(runs[0], runs[1], list(test.post_id)), indent=2))
        return

    classes = list(binarizer(posts).classes_)
    cfg = yaml.safe_load(Path(args.config).read_text())
    prompts = yaml.safe_load(Path(args.prompts).read_text())

    recipe = yaml.safe_load(Path(args.bert_config).read_text())
    max_len = recipe["max_len"]
    threshold = recipe["threshold"] if args.threshold is None else args.threshold

    device = pick_device()
    tokenizer = BertTokenizer.from_pretrained(args.checkpoint)
    model = BertForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    predict = bert_predict(model, tokenizer, device, max_len)

    # the checkpoint must be the model training scored: its test predictions should come back unchanged
    print("checking the checkpoint against training's saved predictions...")
    checkpoint_diff = None
    scored = Path(args.checkpoint).parent / "test_probs.csv"
    if scored.exists():
        saved = pd.read_csv(scored).set_index("post_id")[classes]
        again = predict([texts(t, split_sentences(t)) for t in test["text"]])
        checkpoint_diff = float(np.abs(again - saved.loc[list(test["post_id"])].to_numpy()).max())
        if checkpoint_diff > 1e-3:
            print(f"warning: this checkpoint's predictions differ from {scored} by up to {checkpoint_diff:.4f}")

    def ig(text, spans, j):
        return integrated_gradients(model, tokenizer, text, spans, j, max_len=max_len, **cfg["ig"])

    def explain_why(sentences, labels):
        return decoupled(sentences, labels, prompts, ask)

    def seen(text, spans):
        return sentences_seen(text, spans, tokenizer, max_len)

    ig_random = None
    if args.sanity:
        random = randomised_head(model, cfg["sanity"]["seed"])

        def ig_random(text, spans, j):
            return integrated_gradients(random, tokenizer, text, spans, j, max_len=max_len, **cfg["ig"])

    records, failed, asked = [], [], 0
    start = time.time()
    for row in tqdm(test.itertuples(), total=len(test), desc="M2 posts", unit="post"):
        recs, result = post_records(row.post_id, row.text, row.labels, classes, predict, ig, explain_why,
                                    cfg["lime"], threshold, seen, ig_random)
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
           "checkpoint": args.checkpoint, "checkpoint_check_max_diff": checkpoint_diff, "max_len": max_len,
           "threshold": threshold, "device": str(device), "sanity": args.sanity,
           "seconds": round(time.time() - start, 1), "seconds_per_post": round((time.time() - start) / len(test), 2)}
    (out / "m2_log.json").write_text(json.dumps(log, indent=2))
    print(f"{len(records)} records from {len(test)} posts; decoupled failures {len(failed)} of {asked}; "
          f"largest IG completeness gap {log['ig_delta_max']}; {log['seconds']}s on {device}")


if __name__ == "__main__":
    main()
