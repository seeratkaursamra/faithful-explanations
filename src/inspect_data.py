"""Check the data against the paper, dump stats, and write a batch of posts to read by hand."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import NO_SYMPTOMS, SYMPTOMS, check_against_paper, load, split, symptom_counts
from .sentences import alignment_report

# posts per symptom in the hand-reading batch, weighted toward the classes I care most about
READING_BATCH = {
    "DEPRESSED_MOOD": 10, "WORTHLESSNESS": 5, "SUICIDAL_THOUGHTS": 5, "ANHEDONIA": 5,
    "PSYCHOMOTOR": 2, "APPETITE_CHANGE": 2, "COGNITIVE_ISSUES": 1,
}


def describe(posts, ann):
    train, test = split(posts)
    positives = posts[~posts["is_negative"]]
    stats = {
        "posts": len(posts),
        "negatives": int(posts["is_negative"].sum()),
        "symptoms_per_post_mean": float(posts["n_symptoms"].mean()),
        "symptoms_per_positive_post_mean": float(positives["n_symptoms"].mean()) if len(positives) else None,
        "words_per_post": {k: float(v) for k, v in posts["n_words"].describe()[["mean", "50%", "std", "min", "max"]].items()},
        "annotation_rows": len(ann),
        "annotation_rows_status1": int((ann["status"] == 1).sum()),
        "annotation_rows_status0": int((ann["status"] == 0).sum()),
        "symptom_counts": symptom_counts(posts),
        "split": {"train": len(train), "test": len(test), "test_symptom_counts": symptom_counts(test)},
        # rough: 400 words is usually past 512 wordpieces, so these posts get truncated by BERT
        "posts_over_400_words": int((posts["n_words"] > 400).sum()),
    }
    co = pd.DataFrame(0, index=SYMPTOMS, columns=SYMPTOMS)
    for labs in posts["labels"]:
        labs = [lab for lab in labs if lab != NO_SYMPTOMS]
        for i in labs:
            for j in labs:
                co.loc[i, j] += 1
    stats["cooccurrence"] = co.to_dict()
    return stats


def reading_batch(posts, ann, seed=0):
    rng = np.random.default_rng(seed)
    chosen = []
    for sym, n in READING_BATCH.items():
        pool = [pid for pid, labs in zip(posts["post_id"], posts["labels"]) if sym in labs and pid not in chosen]
        chosen += list(rng.choice(pool, size=min(n, len(pool)), replace=False))
    negs = posts.loc[posts["is_negative"], "post_id"].tolist()
    chosen += list(rng.choice(negs, size=min(3, len(negs)), replace=False))

    lines = ["# Hand-reading batch", "",
             "Read the post, then the evidence sentences and rationales. Notes go in docs/observations.md.", ""]
    for pid in chosen:
        row = posts[posts["post_id"] == pid].iloc[0]
        lines += [f"## {pid}  labels: {', '.join(row['labels'])}  ({row['n_words']} words)", "",
                  row["text"].replace("\n", " ").strip()[:4000], ""]
        for _, r in ann[ann["post_id"] == pid].sort_values("sentence_index").iterrows():
            tag = "EVIDENCE" if r["status"] == 1 else "REJECTED"
            lines += [f"- **[{tag}] {r['DSM5_symptom']}** (sent {r['sentence_index']}): {r['sentence_text']}",
                      f"  - rationale: {r['explanation']}"]
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/data_stats.json")
    ap.add_argument("--reading-sample", default=None)
    ap.add_argument("--allow-mismatch", action="store_true", help="for the public sample")
    args = ap.parse_args()

    posts, ann = load(args.data)
    check = check_against_paper(posts)
    stats = describe(posts, ann)
    alignment, unaligned = alignment_report(posts, ann)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"paper_check": check, "stats": stats, "alignment": alignment,
                                          "unaligned": unaligned}, indent=2))

    print(f"posts={stats['posts']} negatives={stats['negatives']} annotation_rows={stats['annotation_rows']} "
          f"(status1={stats['annotation_rows_status1']}, status0={stats['annotation_rows_status0']})")
    print(f"train/test = {stats['split']['train']}/{stats['split']['test']}")
    print("symptom counts (ours vs paper):")
    for s, (ours, paper) in check["symptom_counts"].items():
        print(f"  {s:<18} {ours:>5} vs {paper:>5}{'' if ours == paper else '  <-- MISMATCH'}")
    print("matches paper:", check["matches_paper"])
    # every sentence-level metric rests on gold sentences landing on mine, so say how often they do
    total, aligned = alignment["total"], alignment["total"] - alignment["none"]
    print(f"gold sentences aligned to my split: {aligned}/{total} ({aligned / total:.0%}) - exact "
          f"{alignment['exact']}, contains {alignment['contains']}, overlap {alignment['overlap']}, none "
          f"{alignment['none']}; sentence_id equals my index for {alignment['id_equals_index']} of the {aligned}")
    print(f"stats written to {args.out}")

    if args.reading_sample:
        Path(args.reading_sample).parent.mkdir(parents=True, exist_ok=True)
        Path(args.reading_sample).write_text(reading_batch(posts, ann))
        print(f"hand-reading batch written to {args.reading_sample}")

    if not check["matches_paper"] and not args.allow_mismatch:
        print("\nstats do not match the paper; sort that out before training anything", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
