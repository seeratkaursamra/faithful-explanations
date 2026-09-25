"""The harness: every explanation metric per (model, post, symptom), then TP against FP overall and
per symptom, then H1 under the pre-registered decision rules.

Input is JSONL, one record per (model, post, symptom) that the model or the gold labels call positive:
  model, post_id, symptom
  predicted, gold               booleans
  stated_evidence               sentence indices the model cited; null when it states nothing (M1)
  stated_rationale              its rationale text, or null
  attribution                   {"IG": {sentence: score}, "LIME": {...}}; any subset, may be empty
  comprehensiveness             probability drop in 0-1 when stated_evidence is removed, or null
  sufficiency                   probability drop in 0-1 when only stated_evidence is kept, or null
Sentence indices refer to src.sentences.split of the post. M3's 0-100 confidence is divided by 100
before it gets here, so every model's erasure numbers share one scale.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import SYMPTOMS, gold, gold_text, load, references, split
from .metrics import (MIN_SUPPORT, bootstrap_ci, cliffs_delta, cosine, did_ci, holm, jaccard, mann_whitney_p,
                      mean_similarity, precision_at_k, precision_recall_f1, rank_agreement, symptom_null, top_k,
                      unpaired_diff_ci)
from .sentences import align
from .sentences import split as split_sentences

KIND = {(True, True): "TP", (True, False): "FP", (False, True): "FN", (False, False): "TN"}

# Prereg: H1 needs the plausibility gap to be small, and is inconclusive below 30 false positives.
SMALL_PLAUSIBILITY_GAP = 0.05
MIN_FALSE_POSITIVES = 30

# Bootstrap settings; the real values come from configs/evaluate.yaml, these match it.
BOOT = {"n_resamples": 1000, "seed": 0}


def gold_evidence(ann, post_id, symptom, text, spans):
    """G_evid as indices into my split. Gold sentences that do not align are left out."""
    return {align(s, text, spans)[0] for s in gold(ann, post_id, symptom)["sentence_text"]} - {-1}


def score(record, g_evid, g_text, vectors, refs):
    """Every per-item metric the pre-registration names. A metric that is undefined for this item
    (no rationale, no gold, nothing cited) is left out of the row, never scored as 0."""
    kind = KIND[(bool(record["predicted"]), bool(record["gold"]))]
    stated = record.get("stated_evidence")
    s_evid = set(stated or [])
    row = {"model": record["model"], "post_id": record["post_id"], "symptom": record["symptom"],
           "correctness": kind, "n_gold": len(g_evid), "n_stated": len(s_evid) if stated is not None else np.nan,
           "comprehensiveness": record.get("comprehensiveness"), "sufficiency": record.get("sufficiency")}

    rationale = record.get("stated_rationale")
    if rationale:
        v = vectors[rationale]
        row["plausibility"] = mean_similarity(v, refs.get(record["symptom"], []))
        row["null"] = symptom_null(v, refs, record["symptom"])
        if kind == "TP" and g_text:
            row["own_similarity"] = cosine(v, vectors[g_text])

    if kind == "TP" and stated is not None and g_evid:
        row["evidence_jaccard"] = jaccard(s_evid, g_evid)
        row["evidence_p"], row["evidence_r"], row["evidence_f1"] = precision_recall_f1(s_evid, g_evid)

    attributions = {m: {int(i): v for i, v in s.items()} for m, s in (record.get("attribution") or {}).items()}
    for method, scores in attributions.items():
        if s_evid:
            row[f"{method}_stated_jaccard"] = jaccard(top_k(scores, len(s_evid)), s_evid)
            row[f"{method}_stated_p@k"] = precision_at_k(scores, s_evid, len(s_evid))
            row[f"{method}_stated_jaccard_k2"] = jaccard(top_k(scores, 2), s_evid)
        if kind == "TP" and g_evid:
            row[f"{method}_gold_jaccard"] = jaccard(top_k(scores, len(g_evid)), g_evid)
    if "IG" in attributions and "LIME" in attributions:
        row["ig_lime_jaccard"] = jaccard(top_k(attributions["IG"], 2), top_k(attributions["LIME"], 2))
        row["ig_lime_spearman"] = rank_agreement(attributions["IG"], attributions["LIME"])
    return row


def compare(items, metric, boot=BOOT):
    """TP against FP on one metric: both means with intervals, the gap, its test and effect size."""
    have = items[items[metric].notna()]
    tp, fp = have[have["correctness"] == "TP"], have[have["correctness"] == "FP"]
    row = {"metric": metric, "n_tp": len(tp), "n_fp": len(fp)}
    for name, group in (("tp", tp), ("fp", fp)):
        if len(group):
            row[f"{name}_mean"], row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(group[metric], posts=group["post_id"], **boot)
    if len(tp) and len(fp):
        row["gap"], row["gap_lo"], row["gap_hi"] = unpaired_diff_ci(tp[metric], fp[metric], posts_a=tp["post_id"],
                                                                    posts_b=fp["post_id"], **boot)
        row["p"] = mann_whitney_p(tp[metric], fp[metric])
        row["cliffs_delta"] = cliffs_delta(tp[metric], fp[metric])
    return row


def metric_columns(items):
    fixed = {"model", "post_id", "symptom", "correctness", "n_gold", "n_stated"}
    return [c for c in items.columns if c not in fixed and items[c].notna().any()]


def summarise(items, boot=BOOT):
    """One row per (model, symptom or ALL, metric). Per-symptom p-values are Holm-corrected across
    the symptoms, and a symptom with fewer than MIN_SUPPORT gold positives is flagged, not dropped."""
    rows = []
    for model, m_items in items.groupby("model"):
        for metric in metric_columns(m_items):
            rows.append({"model": model, "symptom": "ALL", **compare(m_items, metric, boot)})
            per_symptom = []
            for symptom in SYMPTOMS:
                s_items = m_items[m_items["symptom"] == symptom]
                if len(s_items):
                    support = int(s_items["correctness"].isin(["TP", "FN"]).sum())
                    per_symptom.append({"model": model, "symptom": symptom, **compare(s_items, metric, boot),
                                        "support": support, "low_support": support < MIN_SUPPORT})
            tested = [r for r in per_symptom if "p" in r and not np.isnan(r["p"])]
            for r, adjusted in zip(tested, holm([r["p"] for r in tested])):
                r["p_holm"] = adjusted
            rows += per_symptom
    return pd.DataFrame(rows)


def h1(items, model, faithfulness="comprehensiveness", boot=BOOT):
    """The primary statistic and the pre-registered verdict for one model."""
    have = items[(items["model"] == model) & items["plausibility"].notna() & items[faithfulness].notna()]
    tp, fp = have[have["correctness"] == "TP"], have[have["correctness"] == "FP"]
    out = {"model": model, "faithfulness": faithfulness, "n_tp": len(tp), "n_fp": len(fp)}
    if len(fp) < MIN_FALSE_POSITIVES or not len(tp):
        out["verdict"] = (f"inconclusive: {len(fp)} false positives with both plausibility and "
                          f"{faithfulness}, fewer than {MIN_FALSE_POSITIVES}")
        return out

    posts = {"posts_correct": tp["post_id"], "posts_wrong": fp["post_id"], **boot}
    out["did"] = did_ci(tp["plausibility"], tp[faithfulness], fp["plausibility"], fp[faithfulness], **posts)
    out["plausibility_gap"] = unpaired_diff_ci(tp["plausibility"], fp["plausibility"],
                                               posts_a=tp["post_id"], posts_b=fp["post_id"], **boot)
    out["faithfulness_gap"] = unpaired_diff_ci(tp[faithfulness], fp[faithfulness],
                                               posts_a=tp["post_id"], posts_b=fp["post_id"], **boot)
    out["verdict"] = verdict(out["did"], out["plausibility_gap"], out["faithfulness_gap"])
    return out


def verdict(did, plausibility_gap, faithfulness_gap):
    """The decision rules in docs/prereg.md, applied mechanically. Each argument is (estimate, lo, hi)."""
    d, d_lo, d_hi = did
    p, p_lo, p_hi = plausibility_gap
    if d_lo <= 0 <= d_hi or p >= faithfulness_gap[0]:
        return "refuted"
    if d > 0 and (p_lo <= 0 <= p_hi or p < SMALL_PLAUSIBILITY_GAP):
        return "supported"
    return "partial support: both degrade, faithfulness faster"


def run(records, posts, ann, train_ids, embed, boot=BOOT):
    """Score every record, then summarise. `embed` maps a list of texts to a list of vectors."""
    text_by_id = dict(zip(posts["post_id"], posts["text"]))
    ref_texts = references(ann, train_ids)
    rows, needed, prepared = [], set(), []
    for record in records:
        text = text_by_id[record["post_id"]]
        spans = split_sentences(text)
        g_text = gold_text(ann, record["post_id"], record["symptom"])
        prepared.append((record, gold_evidence(ann, record["post_id"], record["symptom"], text, spans), g_text))
        needed |= {t for t in (record.get("stated_rationale"), g_text) if t}

    needed = sorted(needed)
    vectors = dict(zip(needed, embed(needed))) if needed else {}
    refs = {s: embed(ts) for s, ts in ref_texts.items()}
    for record, g_evid, g_text in prepared:
        rows.append(score(record, g_evid, g_text, vectors, refs))
    items = pd.DataFrame(rows)
    models = sorted(items["model"].unique())
    return items, summarise(items, boot), [h1(items, m, boot=boot) for m in models if "plausibility" in items]


def main():
    from .llm import embed

    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="JSONL files in the format above")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/metrics")
    ap.add_argument("--config", default="configs/evaluate.yaml")
    args = ap.parse_args()
    boot = yaml.safe_load(Path(args.config).read_text())["bootstrap"]

    posts, ann = load(args.data)
    train, _ = split(posts)
    records = [json.loads(line) for path in args.records for line in Path(path).read_text().splitlines() if line]
    items, summary, hypothesis = run(records, posts, ann, train["post_id"], embed, boot)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "items.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "h1.json").write_text(json.dumps(hypothesis, indent=2, default=float))
    for h in hypothesis:
        print(f"{h['model']}: {h['verdict']}")


if __name__ == "__main__":
    main()
