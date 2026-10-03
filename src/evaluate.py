"""The harness: every explanation metric per (model, post, symptom), then TP against FP overall and
per symptom, then H1 under the pre-registered decision rules.

Input is JSONL, one record per (model, post, symptom) that the model or the gold labels call positive:
  model, post_id, symptom
  predicted, gold               booleans
  stated_evidence               sentence indices the model cited; null when it states nothing (M1)
  stated_rationale              its rationale text, or null
  attribution                   {"IG": {sentence: score}, "LIME": {...}}; any subset, may be empty
  comprehensiveness             what removing stated_evidence does, or null: a probability drop in 0-1 for M2;
                                for M3, 1 if the label disappears and 0 if not (the prereg's decision flip)
  sufficiency                   the same with only stated_evidence kept
  ig_delta, lime_r2             optional: IG's completeness gap and LIME's local R-squared, reported as-is
  sentences_seen                optional, M2: how many leading sentences fit in BERT's window
  comprehensiveness_confidence, sufficiency_confidence
                                optional, M3: the confidence drops, kept beside the decision flips
  ig_random_spearman            optional, M2: IG on the trained model against IG on a randomised head
Sentence indices refer to src.sentences.split of the post. M3's 0-100 confidence is divided by 100
before it gets here, so every model's erasure numbers share one scale.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from .data import SYMPTOMS, gold, gold_text, load, references, rejection_text, split, symptom_counts
from .lexicon import load_nrc, markers
from .metrics import (MIN_SUPPORT, _interval, bootstrap_ci, cliffs_delta, cosine, delta_ci, delta_did_ci, did_ci, holm, median_ci,
                      paired_rank_biserial,
                      jaccard, mann_whitney_p,
                      paired_diff_ci, wilcoxon_p,
                      mean_similarity, precision_at_k, precision_recall_f1, rank_agreement, resample_posts,
                      symptom_null, top_k, unpaired_diff_ci)
from .sentences import align, texts
from .sentences import split as split_sentences

KIND = {(True, True): "TP", (True, False): "FP", (False, True): "FN", (False, False): "TN"}

# Amendment 2: H1's plausibility drop must be negligible as a Cliff's delta, below 0.147 (the
# conventional cut-off, Romano et al. 2006). The model-level floor of 30 FPs is the prereg's own.
NEGLIGIBLE_DELTA = 0.147
MIN_FALSE_POSITIVES = 30
# Amendment 2: a symptom's H3 difference rests on at least this many false positives.
MIN_FP_PER_SYMPTOM = 5
# Amendment 2: H2 is supported only if every model's interval rules out a correlation this strong.
H2_MAX_UPPER = 0.5

# Prereg H3: the gap should be wider for these than for suicidal thoughts.
H3_TARGETS = ["PSYCHOMOTOR", "APPETITE_CHANGE", "COGNITIVE_ISSUES"]
H3_REFERENCE = "SUICIDAL_THOUGHTS"

# Prereg H4: M1 may trail M2 by at most this much micro-F1.
H4_MAX_F1_GAP = 0.10

# Prereg conditions for calling the whole study inconclusive.
MIN_IG_LIME_AGREEMENT = 0.2      # median Spearman between IG and LIME on M2
MAX_PARSE_FAILURE = 0.10         # LLM replies still unparsable after retries

# Randomised-model check: IG has degraded if trained and randomised-head sentence rankings agree
# less than this, as a median of |Spearman|. The prereg names the check but not a number; see the log.
RANDOMISED_MAX_AGREEMENT = 0.5

# Bootstrap settings; the real values come from configs/evaluate.yaml, these match it.
BOOT = {"n_resamples": 1000, "seed": 0}


def gold_evidence(ann, post_id, symptom, text, spans):
    """G_evid as indices into my split. Gold sentences that do not align are left out."""
    return {align(s, text, spans)[0] for s in gold(ann, post_id, symptom)["sentence_text"]} - {-1}


def density(sentences, top, negative_words):
    """Marker rates in the top sentences minus the rest. Each group's words are pooled, so a
    one-word sentence cannot swing it. Empty when either group is empty."""
    top = {i for i in top if i < len(sentences)}
    rest = [s for i, s in enumerate(sentences) if i not in top]
    if not top or not rest:
        return {}
    a = markers(" ".join(sentences[i] for i in sorted(top)), negative_words)
    b = markers(" ".join(rest), negative_words)
    return {k: a[k] - b[k] for k in a if not (np.isnan(a[k]) or np.isnan(b[k]))}


def score(record, g_evid, g_text, vectors, refs, sentences=None, negative_words=None, rejection=""):
    """Every per-item metric the pre-registration names. A metric that is undefined for this item
    (no rationale, no gold, nothing cited) is left out of the row, never scored as 0.
    Marker density (H4) needs the post's sentences and the NRC words; without them it is skipped.
    `rejection` is the annotator's status-0 note for this symptom in this post, if any (exploratory)."""
    kind = KIND[(bool(record["predicted"]), bool(record["gold"]))]
    stated = record.get("stated_evidence")
    s_evid = set(stated or [])
    row = {"model": record["model"], "post_id": record["post_id"], "symptom": record["symptom"],
           "correctness": kind, "n_gold": len(g_evid), "n_stated": len(s_evid) if stated is not None else np.nan,
           "comprehensiveness": record.get("comprehensiveness"), "sufficiency": record.get("sufficiency")}
    # post length, so a TP-FP gap in attribution overlap can be checked against a gap in length
    if sentences is not None:
        row["n_sentences"] = len(sentences)
    if record.get("sentences_seen") is not None:
        row["sentences_seen"] = record["sentences_seen"]
    # how far to trust the attributions themselves: IG's completeness gap, LIME's local fit
    for quality in ("ig_delta", "lime_r2", "ig_random_spearman"):
        if record.get(quality) is not None:
            row[quality] = record[quality]

    rationale = record.get("stated_rationale")
    if rationale:
        v = vectors[rationale]
        row["plausibility"] = mean_similarity(v, refs.get(record["symptom"], []))
        row["null_similarity"] = symptom_null(v, refs, record["symptom"])
        if kind == "TP" and g_text:
            row["own_similarity"] = cosine(v, vectors[g_text])
        # exploratory (Amendment 2): on a false positive, the rationale against the annotator's note
        # rejecting this symptom in this post, the one reference grounded in the post itself
        if kind == "FP" and rejection:
            row["rejection_similarity"] = cosine(v, vectors[rejection])

    # M2 only: how much of the psychologist's evidence lies past BERT's 512-token cut, out of its reach
    if record.get("sentences_seen") is not None and g_evid:
        row["gold_past_cut"] = sum(i >= record["sentences_seen"] for i in g_evid) / len(g_evid)

    # Amendment 1 says no-citation items are counted and reported: as a share, TP against FP
    if stated is not None and kind in ("TP", "FP"):
        row["no_citation"] = float(not s_evid)

    if kind == "TP" and s_evid and g_evid:
        row["evidence_jaccard"] = jaccard(s_evid, g_evid)
        row["evidence_p"], row["evidence_r"], row["evidence_f1"] = precision_recall_f1(s_evid, g_evid)

    attributions = {m: {int(i): v for i, v in s.items()} for m, s in (record.get("attribution") or {}).items()}
    # an attribution over fewer than two sentences ranks nothing: an empty or failed fit, or a
    # one-sentence post whose single sentence would match any citation trivially
    attributions = {m: s for m, s in attributions.items() if len(s) >= 2}
    for method, scores in attributions.items():
        if s_evid:
            row[f"{method}_stated_jaccard"] = jaccard(top_k(scores, len(s_evid)), s_evid)
            row[f"{method}_stated_p@k"] = precision_at_k(scores, s_evid, len(s_evid))
            row[f"{method}_stated_jaccard_k2"] = jaccard(top_k(scores, 2), s_evid)
        if kind == "TP" and g_evid:
            row[f"{method}_gold_jaccard"] = jaccard(top_k(scores, len(g_evid)), g_evid)
            row[f"{method}_gold_jaccard_k2"] = jaccard(top_k(scores, 2), g_evid)
        if sentences is not None and negative_words is not None:
            for marker, diff in density(sentences, top_k(scores, 2), negative_words).items():
                row[f"{method}_density_{marker}"] = diff
    if "IG" in attributions and "LIME" in attributions:
        row["ig_lime_jaccard"] = jaccard(top_k(attributions["IG"], 2), top_k(attributions["LIME"], 2))
        row["ig_lime_spearman"] = rank_agreement(attributions["IG"], attributions["LIME"])
    return row


def compare(items, metric, boot=BOOT):
    """TP against FP on one metric: both means with intervals, the gap, its test and effect size.
    Also the mean over TP and FP together, which is what H4's "higher than zero" question needs."""
    have = items[items[metric].notna()]
    tp, fp = have[have["correctness"] == "TP"], have[have["correctness"] == "FP"]
    row = {"metric": metric, "n_tp": len(tp), "n_fp": len(fp)}
    for name, group in (("all", pd.concat([tp, fp])), ("tp", tp), ("fp", fp)):
        if len(group):
            row[f"{name}_mean"], row[f"{name}_lo"], row[f"{name}_hi"] = bootstrap_ci(group[metric], posts=group["post_id"], **boot)
    if len(tp) and len(fp):
        row["gap"], row["gap_lo"], row["gap_hi"] = unpaired_diff_ci(tp[metric], fp[metric], posts_a=tp["post_id"],
                                                                    posts_b=fp["post_id"], **boot)
        row["p"] = mann_whitney_p(tp[metric], fp[metric])
        row["cliffs_delta"] = cliffs_delta(tp[metric], fp[metric])
    return row


def metric_columns(items):
    fixed = {"model", "post_id", "symptom", "correctness", "n_gold", "n_stated", "sentences_seen"}
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


def explaining_models(items):
    """Models that state a rationale, so H1, H3 and the false-positive floor apply. M1 states none."""
    return sorted(items.loc[items["plausibility"].notna(), "model"].unique()) if "plausibility" in items else []


def h1(items, model, faithfulness="comprehensiveness", boot=BOOT):
    """The primary statistic (Amendment 2: Cliff's deltas, not raw gaps) and the verdict for one model.
    The raw difference of gaps is kept as did_raw, a secondary."""
    m = items[items["model"] == model]
    have = m[m["plausibility"].notna() & m[faithfulness].notna()]
    tp, fp = have[have["correctness"] == "TP"], have[have["correctness"] == "FP"]
    # items without both measures (no rationale, nothing cited, a failed erasure) drop out; say how many
    out = {"model": model, "faithfulness": faithfulness, "primary": model == "M3", "n_tp": len(tp), "n_fp": len(fp),
           "left_out": {k: int((m["correctness"] == k).sum()) - len(g) for k, g in (("TP", tp), ("FP", fp))}}
    if len(fp) < MIN_FALSE_POSITIVES or not len(tp):
        out["verdict"] = (f"inconclusive: {len(fp)} false positives with both plausibility and "
                          f"{faithfulness}, fewer than {MIN_FALSE_POSITIVES}")
        return out

    posts = {"posts_correct": tp["post_id"], "posts_wrong": fp["post_id"], **boot}
    groups = {"posts_a": tp["post_id"], "posts_b": fp["post_id"], **boot}
    out["did"] = delta_did_ci(tp["plausibility"], tp[faithfulness], fp["plausibility"], fp[faithfulness], **posts)
    out["plausibility_delta"] = delta_ci(tp["plausibility"], fp["plausibility"], **groups)
    out["faithfulness_delta"] = delta_ci(tp[faithfulness], fp[faithfulness], **groups)
    out["did_raw"] = did_ci(tp["plausibility"], tp[faithfulness], fp["plausibility"], fp[faithfulness], **posts)
    out["verdict"] = verdict(out["did"], out["plausibility_delta"], out["faithfulness_delta"])
    return out


def verdict(did, plausibility_delta, faithfulness_delta):
    """The decision rules in docs/prereg.md (Amendment 2), applied mechanically. Each argument is
    (estimate, lo, hi) on the Cliff's delta scale. A plausibility rise on wrong predictions counts as
    holding, as in the original rule; only a drop of a non-negligible size breaks it."""
    d, d_lo, d_hi = did
    p, p_lo, p_hi = plausibility_delta
    if d_lo <= 0 <= d_hi or p >= faithfulness_delta[0]:
        return "refuted"
    if d > 0 and (p_lo <= 0 <= p_hi or p < NEGLIGIBLE_DELTA):
        return "supported"
    return "partial support: both degrade, faithfulness faster"


def h3(items, model, train_counts=None, faithfulness="comprehensiveness", boot=BOOT):
    """The H1 difference in differences within each symptom, ordered by training frequency, and the
    pre-registered H3 verdict. Low-support symptoms are reported but not argued from."""
    m = items[items["model"] == model]
    measured = {"plausibility", faithfulness} <= set(m.columns)
    rows = []
    for symptom in SYMPTOMS:
        s = m[m["symptom"] == symptom]
        support = int(s["correctness"].isin(["TP", "FN"]).sum())
        have = s[s["plausibility"].notna() & s[faithfulness].notna()] if measured else s.iloc[:0]
        tp, fp = have[have["correctness"] == "TP"], have[have["correctness"] == "FP"]
        row = {"symptom": symptom, "train_count": (train_counts or {}).get(symptom, 0), "support": support,
               "low_support": support < MIN_SUPPORT, "n_tp": len(tp), "n_fp": len(fp),
               "few_fp": len(fp) < MIN_FP_PER_SYMPTOM}
        if len(tp) and len(fp):
            row["did"] = delta_did_ci(tp["plausibility"], tp[faithfulness], fp["plausibility"], fp[faithfulness],
                                      posts_correct=tp["post_id"], posts_wrong=fp["post_id"], **boot)
        rows.append(row)
    rows.sort(key=lambda r: -r["train_count"])
    return {"model": model, "faithfulness": faithfulness, "by_symptom": rows, "verdict": h3_verdict(rows)}


def h3_verdict(rows):
    """Supported if every usable target's DiD exceeds suicidal thoughts'. Usable (Amendment 2): a DiD
    exists, the symptom is not low-support, and it rests on at least MIN_FP_PER_SYMPTOM false positives."""
    by = {r["symptom"]: r for r in rows}
    usable = [s for s in H3_TARGETS + [H3_REFERENCE] if "did" in by.get(s, {}) and not by[s]["low_support"]
              and not by[s].get("few_fp", False)]
    if H3_REFERENCE not in usable:
        return "inconclusive: suicidal thoughts has no difference in differences, too little support, or too few FPs"
    targets = [s for s in H3_TARGETS if s in usable]
    if not targets:
        return "inconclusive: none of psychomotor, appetite change, cognitive issues is usable"
    reference = by[H3_REFERENCE]["did"][0]
    if all(by[s]["did"][0] > reference for s in targets):
        return f"supported, on {', '.join(targets)}"
    return f"refuted, on {', '.join(targets)}"


def randomised_check(items, boot=BOOT):
    """Did IG fall apart when the classifier head was randomised? Median |Spearman| between the trained
    and randomised rankings across posts (each post once, at its mean), with an interval. None when the
    check was not run."""
    if "ig_random_spearman" not in items:
        return None
    rho = items.assign(a=items["ig_random_spearman"].abs()).groupby("post_id")["a"].mean().dropna()
    if not len(rho):
        return None
    median, lo, hi = median_ci(rho, **boot)
    return {"n_posts": len(rho), "median_abs_spearman": median, "interval": (lo, hi),
            "degraded": median < RANDOMISED_MAX_AGREEMENT}


def per_symptom(items, faithfulness="comprehensiveness"):
    """One row per (model, symptom): F1 from the TP/FP/FN counts (F1 needs no true negatives), gold
    support, and mean faithfulness over the predicted items that have it."""
    rows = []
    for (model, symptom), g in items.groupby(["model", "symptom"]):
        tp, fp, fn = (int((g["correctness"] == k).sum()) for k in ("TP", "FP", "FN"))
        predicted = g[g["correctness"].isin(["TP", "FP"])]
        faith = predicted[faithfulness].dropna() if faithfulness in g else predicted.iloc[:0]
        rows.append({"model": model, "symptom": symptom, "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
                     "support": tp + fn, "faithfulness": float(faith.mean()) if len(faith) else np.nan})
    return pd.DataFrame(rows)


def _f1_faith_rho(items, keep, faithfulness, within_models=True):
    """Spearman between per-symptom F1 and faithfulness. With within_models, each model's symptoms are
    ranked on their own before pooling, so a model that is better overall but less faithful overall
    cannot reverse the sign (Simpson's paradox), while every model's points still count."""
    points = per_symptom(items, faithfulness)
    points = points[points["symptom"].isin(keep)].dropna(subset=["f1", "faithfulness"])
    if within_models:
        points = points.assign(f1=points.groupby("model")["f1"].rank(),
                               faithfulness=points.groupby("model")["faithfulness"].rank())
    if len(points) < 3 or points["f1"].nunique() < 2 or points["faithfulness"].nunique() < 2:
        return np.nan, len(points)
    return float(spearmanr(points["f1"], points["faithfulness"])[0]), len(points)


def _rho_ci(items, keep, faithfulness, boot):
    rho, n = _f1_faith_rho(items, keep, faithfulness)
    if np.isnan(rho):
        return None, n
    draws = [_f1_faith_rho(items.iloc[rows], keep, faithfulness)[0]
             for rows in resample_posts(items["post_id"], boot["n_resamples"], boot["seed"])]
    return _interval(rho, draws), n


def h2(items, faithfulness="comprehensiveness", boot=BOOT):
    """H2 (Amendment 2): per-symptom F1 against per-symptom faithfulness, Spearman over ranks taken within
    each model and then pooled, with a post-level interval. Pooling raw values could reverse the sign
    (Simpson's paradox); a single model's 7 or so symptoms cannot reach a verdict at all (a perfect
    correlation gave an interval of -0.03 to 0.96). Refuted if the interval is above zero; supported
    only if its upper bound is below H2_MAX_UPPER, so thin data cannot count as support; otherwise
    inconclusive. Each model's own correlation is reported alongside. Low-support symptoms are left out."""
    support = per_symptom(items, faithfulness).groupby("symptom")["support"].max()
    keep = set(support[support >= MIN_SUPPORT].index)
    out = {"faithfulness": faithfulness, "symptoms": sorted(keep), "by_model": {}}
    for model, m in items.groupby("model"):
        spearman, n = _rho_ci(m, keep, faithfulness, boot)
        if spearman:
            out["by_model"][model] = {"points": n, "spearman": spearman}
    spearman, n = _rho_ci(items, keep, faithfulness, boot)
    out["points"] = n
    if spearman is None:
        out["verdict"] = f"inconclusive: {n} (model, symptom) points with both F1 and {faithfulness}, need 3"
        return out
    out["spearman"] = spearman
    out["verdict"] = h2_verdict(spearman)
    return out


def h2_verdict(spearman):
    """Refuted if the interval is above zero; supported if its upper bound is below H2_MAX_UPPER."""
    if spearman[1] > 0:
        return "refuted: accuracy and faithfulness rise together"
    if spearman[2] < H2_MAX_UPPER:
        return "supported"
    return f"inconclusive: the interval cannot rule out a correlation of {H2_MAX_UPPER} or more"


def h4(items, micro_f1, boot=BOOT):
    """Prereg H4: M1's micro-F1 within H4_MAX_F1_GAP of M2's, and M1's top-attributed sentences richer
    in marker words than the rest, the mean difference's interval above zero. micro_f1 maps "M1" and
    "M2" to their test micro-F1; M2's is the five-seed mean, the reproduced baseline."""
    out = {"micro_f1": micro_f1}
    column = "M1_density_marker_words"
    m1 = items[(items["model"] == "M1") & items["correctness"].isin(["TP", "FP"])] if column in items else items.iloc[:0]
    m1 = m1[m1[column].notna()] if len(m1) else m1
    if not {"M1", "M2"} <= set(micro_f1) or not len(m1):
        out["verdict"] = "inconclusive: needs both micro-F1s and M1 marker density"
        return out
    out["f1_gap"] = micro_f1["M2"] - micro_f1["M1"]
    # M2's micro-F1 is a five-seed mean; its spread is the honest error bar on the gap
    if "M2_std" in micro_f1:
        out["m2_seed_std"] = micro_f1["M2_std"]
    out["marker_density"] = bootstrap_ci(m1[column], posts=m1["post_id"], **boot)
    out["verdict"] = h4_verdict(out["f1_gap"], out["marker_density"])
    return out


def h4_verdict(f1_gap, marker_density):
    """Supported if M1 trails M2 by at most H4_MAX_F1_GAP and the marker-density interval is above zero."""
    cheap = f1_gap <= H4_MAX_F1_GAP
    on_markers = marker_density[1] > 0
    if cheap and on_markers:
        return "supported"
    return "refuted: " + " and ".join(reason for reason, failed in [
        (f"M1 trails M2 by {f1_gap:.3f} micro-F1", not cheap),
        ("M1's top sentences are not richer in marker words", not on_markers)] if failed)


def study_checks(items, baseline=None, logs=None, boot=BOOT):
    """Every condition under which the prereg calls the whole study inconclusive, each marked triggered,
    clear, or not run. A missing input is "not run", never a pass. `baseline` is the five-seed
    summary.json; `logs` maps "M2" and "M3" to their run logs."""
    logs = logs or {}
    checks = []

    def check(condition, value, triggered, interval=None):
        checks.append({"condition": condition, "value": value, "interval": interval,
                       "status": "not run" if triggered is None else "triggered" if triggered else "clear"})

    check("BERT baseline outside 0.51 +/- 0.02 micro-F1",
          None if baseline is None else baseline["micro_f1"]["mean"],
          None if baseline is None else not baseline["within_tolerance"])

    # across test posts, as the prereg says: a post with several symptoms counts once, at its mean
    agreement = (items[items["model"] == "M2"].groupby("post_id")["ig_lime_spearman"].mean().dropna()
                 if "ig_lime_spearman" in items else [])
    median, lo, hi = median_ci(agreement, **boot) if len(agreement) else (None, None, None)
    check(f"IG and LIME on M2: median Spearman below {MIN_IG_LIME_AGREEMENT}", median,
          None if median is None else median < MIN_IG_LIME_AGREEMENT, None if median is None else (lo, hi))

    randomised = randomised_check(items, boot)
    check(f"IG does not degrade on a randomised head: median |Spearman| {RANDOMISED_MAX_AGREEMENT} or above",
          None if randomised is None else randomised["median_abs_spearman"],
          None if randomised is None else not randomised["degraded"],
          None if randomised is None else randomised["interval"])

    rates = {}
    if "M3" in logs:
        rates["M3 joint"] = logs["M3"]["failure_rate"]
        if "call_failure_rate" in logs["M3"]:
            rates["M3 every call"] = logs["M3"]["call_failure_rate"]
    if "M2" in logs and logs["M2"]["decoupled_asked"]:
        rates["M2 decoupled control"] = logs["M2"]["decoupled_failures"] / logs["M2"]["decoupled_asked"]
    check(f"LLM parse failures above {MAX_PARSE_FAILURE:.0%} after retries", rates or None,
          None if not rates else max(rates.values()) > MAX_PARSE_FAILURE)

    # Amendment 2: only models that state a rationale; M1's correctness split tests nothing
    explaining = explaining_models(items)
    # Amendment 3: the false positives H1 can use, with both plausibility and comprehensiveness, so this
    # check and H1 cannot disagree
    for model in explaining:
        m = items[(items["model"] == model) & (items["correctness"] == "FP")]
        n_fp = int((m["plausibility"].notna() & m["comprehensiveness"].notna()).sum()) if "comprehensiveness" in m else 0
        check(f"{model}: fewer than {MIN_FALSE_POSITIVES} usable false positives", n_fp, n_fp < MIN_FALSE_POSITIVES)
    if not explaining:
        check(f"fewer than {MIN_FALSE_POSITIVES} false positives in a model that explains itself", None, None)

    statuses = {c["status"] for c in checks}
    overall = "inconclusive" if "triggered" in statuses else "incomplete" if "not run" in statuses else "clear"
    failed = items.groupby("model")["parse_failed"].sum().astype(int).to_dict() if "parse_failed" in items else {}
    return {"status": overall, "checks": checks, "parse_failed_items": failed}


# The prereg's secondary outcomes run through H1 in place of comprehensiveness: M3's confidence drop,
# attribution against the stated evidence by IG and by LIME, and the same at k = 2.
SECONDARY_FAITHFULNESS = ["comprehensiveness_confidence", "IG_stated_jaccard", "LIME_stated_jaccard",
                          "IG_stated_jaccard_k2", "LIME_stated_jaccard_k2"]

# Metrics that mean the same thing whichever model produced them, so two models can be compared on them.
# Not comprehensiveness or sufficiency: graded probability drops on BERT, near 0/1 flips on the LLM.
SHARED_METRICS = ["plausibility", "own_similarity", "evidence_jaccard", "evidence_f1", "LIME_stated_jaccard"]


def _paired(a, b, posts, boot):
    mean, lo, hi = paired_diff_ci(a, b, posts=posts, **boot)
    return {"n": len(a), "diff": mean, "lo": lo, "hi": hi, "p": wilcoxon_p(a, b), "rank_biserial": paired_rank_biserial(a, b)}


def method_contrasts(items, boot=BOOT):
    """IG minus LIME on the same M2 items, for every metric both methods have (paired, as the prereg
    reserves paired tests for contrasts on the same items)."""
    m2 = items[items["model"] == "M2"]
    rows = []
    for column in m2.columns:
        other = "LIME_" + column[3:]
        if column.startswith("IG_") and other in m2:
            both = m2[m2[column].notna() & m2[other].notna()]
            if len(both) >= 2:
                rows.append({"contrast": "IG - LIME", "metric": column[3:],
                             **_paired(both[column], both[other], both["post_id"], boot)})
    return rows


def model_contrasts(items, boot=BOOT):
    """Each pair of models on the (post, symptom) items that both have a value for, metric by metric.
    Plausibility against faithfulness is not here: they are on different scales, and their paired
    contrast is the difference in differences. Posts longer than BERT's window are left out: M2's
    control only saw the part BERT saw, so on those posts the two models read different text."""
    rows = []
    cut = set()
    if {"sentences_seen", "n_sentences"} <= set(items.columns):
        cut = set(items.loc[items["sentences_seen"] < items["n_sentences"], "post_id"])
    items = items[~items["post_id"].isin(cut)]
    for a, b in combinations(sorted(items["model"].unique()), 2):
        shared = items[items["model"] == a].merge(items[items["model"] == b], on=["post_id", "symptom"],
                                                  suffixes=("_a", "_b"))
        for metric in SHARED_METRICS:
            if f"{metric}_a" in shared and f"{metric}_b" in shared:
                both = shared[shared[f"{metric}_a"].notna() & shared[f"{metric}_b"].notna()]
                if len(both) >= 2:
                    rows.append({"contrast": f"{a} - {b}", "metric": metric, "posts_past_window_left_out": len(cut),
                                 **_paired(both[f"{metric}_a"], both[f"{metric}_b"], both["post_id"], boot)})
    return rows


def run(records, posts, ann, train_ids, embed, boot=BOOT, negative_words=None, micro_f1=None, baseline=None,
        logs=None):
    """Score every record, then summarise. `embed` maps a list of texts to a list of vectors.
    Returns the items, the summary table, and the H1 and H3 results per model."""
    text_by_id = dict(zip(posts["post_id"], posts["text"]))
    ref_texts = references(ann, train_ids)
    rows, needed, prepared = [], set(), []
    for record in records:
        text = text_by_id[record["post_id"]]
        spans = split_sentences(text)
        g_text = gold_text(ann, record["post_id"], record["symptom"])
        wrong = record["predicted"] and not record["gold"]
        rejection = rejection_text(ann, record["post_id"], record["symptom"]) if wrong else ""
        prepared.append((record, gold_evidence(ann, record["post_id"], record["symptom"], text, spans), g_text,
                         texts(text, spans), rejection))
        needed |= {t for t in (record.get("stated_rationale"), g_text, rejection) if t}

    needed = sorted(needed)
    vectors = dict(zip(needed, embed(needed))) if needed else {}
    refs = {s: embed(ts) for s, ts in ref_texts.items()}
    for record, g_evid, g_text, sentences, rejection in prepared:
        rows.append(score(record, g_evid, g_text, vectors, refs, sentences, negative_words, rejection))
    items = pd.DataFrame(rows)
    train_counts = symptom_counts(posts[posts["post_id"].isin(set(train_ids))]) if "labels" in posts else None
    explaining = explaining_models(items)
    hypotheses = {"study": study_checks(items, baseline, logs, boot),
                  "H1": [h1(items, m, boot=boot) for m in explaining],
                  "H2": [h2(items, boot=boot)],
                  "H3": [h3(items, m, train_counts, boot=boot) for m in explaining],
                  "H4": [h4(items, micro_f1 or {}, boot)],
                  # the prereg's secondary outcomes: the same analysis with other faithfulness measures
                  "H1_secondary": [h1(items, m, faithfulness=f, boot=boot) for m in explaining
                                   for f in SECONDARY_FAITHFULNESS
                                   if f in items and items.loc[items["model"] == m, f].notna().any()],
                  "paired": method_contrasts(items, boot) + model_contrasts(items, boot)}
    return items, summarise(items, boot), hypotheses


def main():
    from functools import partial

    from .llm import embed

    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="JSONL files in the format above")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/metrics")
    ap.add_argument("--config", default="configs/evaluate.yaml")
    ap.add_argument("--nrc", default="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
    ap.add_argument("--m1-metrics", default="outputs/m1/metrics.json")
    ap.add_argument("--m2-summary", default="outputs/bert_baseline/summary.json", help="the five-seed mean")
    ap.add_argument("--logs", default="outputs/records", help="where m2_log.json and m3_log.json are")
    ap.add_argument("--embed-model", default="nomic-embed-text",
                    help="all-minilm reruns everything with the prereg's secondary embedding model")
    args = ap.parse_args()
    micro_f1 = {}
    if Path(args.m1_metrics).exists():
        micro_f1["M1"] = json.loads(Path(args.m1_metrics).read_text())["micro_f1"]
    baseline = json.loads(Path(args.m2_summary).read_text()) if Path(args.m2_summary).exists() else None
    if baseline:
        micro_f1["M2"], micro_f1["M2_std"] = baseline["micro_f1"]["mean"], baseline["micro_f1"]["std"]
    # only H4 and marker density need the lexicon; without it they are skipped, not the whole harness
    negative_words = load_nrc(args.nrc) if Path(args.nrc).exists() else None
    if negative_words is None:
        print(f"no NRC file at {args.nrc}: marker density and H4 are skipped")
    logs = {m: json.loads((Path(args.logs) / f"{m.lower()}_log.json").read_text())
            for m in ("M2", "M3") if (Path(args.logs) / f"{m.lower()}_log.json").exists()}
    boot = yaml.safe_load(Path(args.config).read_text())["bootstrap"]

    posts, ann = load(args.data)
    train, _ = split(posts)
    records = [json.loads(line) for path in args.records for line in Path(path).read_text().splitlines() if line]
    items, summary, hypotheses = run(records, posts, ann, train["post_id"], partial(embed, model=args.embed_model),
                                     boot, negative_words, micro_f1, baseline, logs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "items.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    hypotheses["embed_model"] = args.embed_model
    (out / "hypotheses.json").write_text(json.dumps(hypotheses, indent=2, default=float))
    study = hypotheses["study"]
    print(f"study: {study['status']}")
    for c in study["checks"]:
        print(f"  [{c['status']}] {c['condition']}: {c['value']}")
    for name, results in hypotheses.items():
        if name.startswith("H"):
            for h in results:
                print(f"{name} {h.get('model', 'all models')}: {h['verdict']}")


if __name__ == "__main__":
    main()
