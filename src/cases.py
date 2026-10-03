"""The qualitative cases: plausible but unfaithful explanations, picked by a fixed rule, for reading by hand.

Two outputs. cases.md holds the post, the model's rationale and the psychologist's, for reading only; it
lives in outputs/ and is never committed or shared. cases.csv holds no post text, just the scores and an
empty `reading` column to fill in; Table 3 is built from that alone, so no post can be quoted in the paper.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .data import gold_text, load
from .sentences import split as split_sentences
from .sentences import texts

# Faithfulness at or below this quantile of the model's own items counts as low.
LOW_FAITHFULNESS = 0.25


def pick(items, per_model=10, faithfulness="comprehensiveness"):
    """For each model with both measures: its low-faithfulness predictions, the most plausible first.
    Ties break by post and symptom, so the same run always gives the same cases."""
    have = items[items["correctness"].isin(["TP", "FP"])]
    have = have[have["plausibility"].notna() & have[faithfulness].notna()] if {"plausibility", faithfulness} <= set(
        items.columns) else have.iloc[:0]
    chosen = []
    for _, m in have.groupby("model"):
        low = m[m[faithfulness] <= m[faithfulness].quantile(LOW_FAITHFULNESS)]
        chosen.append(low.sort_values(["plausibility", faithfulness, "post_id", "symptom"],
                                      ascending=[False, True, True, True]).head(per_model))
    columns = ["model", "post_id", "symptom", "correctness", "plausibility", faithfulness]
    return pd.concat(chosen)[columns].reset_index(drop=True) if chosen else pd.DataFrame(columns=columns)


def reading_file(cases, records, posts, ann, faithfulness="comprehensiveness"):
    """Markdown for reading the cases: the post split into numbered sentences, the model's rationale and
    cited sentences, and the psychologist's rationale. Private: it quotes posts."""
    by_key = {(r["model"], r["post_id"], r["symptom"]): r for r in records}
    text_by_id = dict(zip(posts["post_id"], posts["text"]))
    lines = ["# Qualitative cases (private: quotes posts, never commit or share)", ""]
    for i, c in enumerate(cases.itertuples(), 1):
        r = by_key.get((c.model, c.post_id, c.symptom), {})
        sentences = texts(text_by_id[c.post_id], split_sentences(text_by_id[c.post_id]))
        cited = set(r.get("stated_evidence") or [])
        lines += [f"## {i}. {c.model} {c.symptom} ({c.correctness}), plausibility {c.plausibility:.2f}, "
                  f"faithfulness {getattr(c, faithfulness):.2f}", ""]
        lines += [f"{'**' if j in cited else ''}{j}: {s}{'**' if j in cited else ''}" for j, s in enumerate(sentences)]
        lines += ["", f"- model's rationale (cited in bold): {r.get('stated_rationale')}",
                  f"- psychologist's rationale: {gold_text(ann, c.post_id, c.symptom) or '(none: not a gold symptom)'}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="the same JSONL files the harness read")
    ap.add_argument("--metrics", default="outputs/metrics")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/qualitative")
    ap.add_argument("--per-model", type=int, default=10)
    args = ap.parse_args()

    items = pd.read_csv(Path(args.metrics) / "items.csv")
    records = [json.loads(line) for path in args.records for line in Path(path).read_text().splitlines() if line]
    posts, ann = load(args.data)
    cases = pick(items, args.per_model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.md").write_text(reading_file(cases, records, posts, ann))
    table = out / "cases.csv"
    if table.exists():
        print(f"{table} exists and may hold your readings; not overwritten")
    else:
        cases.assign(reading="").to_csv(table, index=False)
    print(f"{len(cases)} cases; read {out / 'cases.md'}, write your reading into {table}")


if __name__ == "__main__":
    main()
