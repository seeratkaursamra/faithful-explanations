"""M3 on the test split: the joint answer, then erasure with the same prompt, written as harness records."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import combinations
from pathlib import Path

import yaml

from . import lime_sentences
from .data import NO_SYMPTOMS, check_against_paper, load, split
from .llm import ask
from .metrics import erasure, jaccard
from .sentences import split as split_sentences
from .sentences import texts


def numbered(sentences):
    return "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))


def answer(sentences, prompts, ask_fn):
    """The joint prompt on these sentences. Returns ask's result dict."""
    return ask_fn(prompts["joint"].format(numbered_post=numbered(sentences)), prompts["system"],
                  n_sentences=len(sentences))


def confidence(result, label):
    """The model's output for one label on 0-1: its confidence if it names the label, 0 if it does not.
    None when the reply failed to parse or named the label without a confidence."""
    if not result["ok"]:
        return None
    for s in result["symptoms"]:
        if s["label"] == label:
            return None if s["confidence"] is None else s["confidence"] / 100
    return 0.0


def post_records(post_id, text, gold_labels, prompts, ask_fn, lime_cfg=None):
    """Harness records for one post: every symptom predicted or gold. Also returns the joint result,
    so the caller can count parse failures and retries.

    With lime_cfg, each predicted symptom also gets LIME over sentences through the joint prompt.
    Every reply is kept per distinct set of sentences: one reply carries a confidence for every
    label, so a post's perturbations are asked once and serve all its symptoms and its erasure.
    """
    sentences = texts(text, split_sentences(text))
    replies = {}

    def reply(kept):
        key = tuple(kept)
        if key not in replies:
            replies[key] = answer(list(kept), prompts, ask_fn)
        return replies[key]

    result = reply(sentences)
    if not result["ok"]:
        return [], result
    said = {s["label"]: s for s in result["symptoms"] if s["label"] != NO_SYMPTOMS}
    records = []
    for label in sorted(set(said) | (set(gold_labels) - {NO_SYMPTOMS})):
        s = said.get(label)
        record = {"model": "M3", "post_id": post_id, "symptom": label, "predicted": s is not None,
                  "gold": label in gold_labels, "stated_evidence": None, "stated_rationale": None,
                  "attribution": {}, "comprehensiveness": None, "sufficiency": None}
        if s is not None:
            p_full = None if s["confidence"] is None else s["confidence"] / 100
            record["comprehensiveness"], record["sufficiency"] = erasure(
                sentences, set(s["evidence"]), p_full, lambda kept, label=label: confidence(reply(kept), label))
            record.update(stated_evidence=s["evidence"], stated_rationale=s["rationale"] or None,
                          confidence=p_full)
            if lime_cfg is not None:
                unscored = []

                def predict(samples, label=label):
                    scores = [confidence(reply(kept), label) for kept in samples]
                    unscored.extend(x for x in scores if x is None)
                    return scores

                weights, record["lime_r2"] = lime_sentences.explain(sentences, predict, **lime_cfg)
                record["attribution"] = {"LIME": {str(i): w for i, w in weights.items()}}
                record["lime_unscored"] = len(unscored)
        records.append(record)
    return records, result


def _agree(a, b):
    """Jaccard, except that two runs both saying "nothing" agree completely."""
    return 1.0 if not a and not b else jaccard(a, b)


def stability(runs, post_ids):
    """Mean pairwise agreement between runs of the same posts: of the predicted label sets per post,
    and of the cited evidence per (post, symptom) that both runs predicted."""
    labels = [{p: set() for p in post_ids} for _ in runs]
    evidence = [{} for _ in runs]
    for i, records in enumerate(runs):
        for r in records:
            if r["predicted"]:
                labels[i][r["post_id"]].add(r["symptom"])
                evidence[i][(r["post_id"], r["symptom"])] = set(r["stated_evidence"] or [])
    label_scores, evidence_scores = [], []
    for a, b in combinations(range(len(runs)), 2):
        label_scores += [_agree(labels[a][p], labels[b][p]) for p in post_ids]
        evidence_scores += [_agree(evidence[a][k], evidence[b][k]) for k in evidence[a].keys() & evidence[b].keys()]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {"labels": mean(label_scores), "evidence": mean(evidence_scores),
            "n_posts": len(post_ids), "n_shared_predictions": len(evidence_scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--out", default="outputs/records")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests; set OLLAMA_NUM_PARALLEL to match")
    ap.add_argument("--limit", type=int, default=None, help="first N test posts only, for a trial run")
    ap.add_argument("--allow-mismatch", action="store_true", help="for the public sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 for the main run; 0.3 for stability runs")
    ap.add_argument("--stability", nargs="+", metavar="JSONL", help="compare finished runs instead of running")
    ap.add_argument("--lime", action="store_true", help="also LIME on every predicted symptom: the long run")
    ap.add_argument("--config", default="configs/attribution.yaml")
    args = ap.parse_args()
    lime_cfg = yaml.safe_load(Path(args.config).read_text())["lime"] if args.lime else None

    posts, _ = load(args.data)
    if not args.allow_mismatch and not check_against_paper(posts)["matches_paper"]:
        raise SystemExit("data does not match the paper, refusing to run")
    _, test = split(posts)
    test = test.head(args.limit) if args.limit else test

    if args.stability:
        # the post list comes from the split, not the records: a post every run calls empty writes
        # no records, and it is full agreement that must still be counted
        runs = [[json.loads(l) for l in Path(f).read_text().splitlines() if l] for f in args.stability]
        print(json.dumps(stability(runs, list(test.post_id)), indent=2))
        return
    prompts = yaml.safe_load(Path(args.prompts).read_text())

    def one(row):
        return post_records(row.post_id, row.text, row.labels, prompts,
                            partial(ask, seed=args.seed, temperature=args.temperature), lime_cfg)

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, test.itertuples()))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "m3.jsonl", "w") as f:
        for records, _ in results:
            for r in records:
                f.write(json.dumps(r) + "\n")
    failed = [pid for pid, (_, r) in zip(test.post_id, results) if not r["ok"]]
    log = {"posts": len(test), "parse_failures": len(failed), "failure_rate": len(failed) / len(test),
           "failed_posts": failed, "retried": sum(r["attempts"] > 1 for _, r in results)}
    if args.lime:
        lime = [r for records, _ in results for r in records if "lime_unscored" in r]
        log["lime_explained"] = len(lime)
        log["lime_unscored_perturbations"] = sum(r["lime_unscored"] for r in lime)
    (out / "m3_log.json").write_text(json.dumps(log, indent=2))
    print(f"{sum(len(r) for r, _ in results)} records from {len(test)} posts; "
          f"parse failures {len(failed)} ({log['failure_rate']:.1%}, prereg limit 10%)")


if __name__ == "__main__":
    main()
