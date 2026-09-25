import math

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compare, gold_evidence, h1, run, score, summarise, verdict
from src.sentences import split

TEXT = "I cannot sleep. The weather is fine. I lie awake all night."


def record(**kw):
    base = {"model": "M3", "post_id": "p1", "symptom": "SLEEP_ISSUES", "predicted": True, "gold": True,
            "stated_evidence": [0, 2], "stated_rationale": None, "attribution": {},
            "comprehensiveness": 0.4, "sufficiency": 0.1}
    return {**base, **kw}


# --- the pre-registered decision rules --------------------------------------------------------------

def test_verdict_supported_when_did_excludes_zero_and_plausibility_barely_moves():
    assert verdict((0.3, 0.1, 0.5), (0.01, -0.02, 0.04), (0.31, 0.1, 0.5)) == "supported"
    # the plausibility interval excludes zero, but the gap is under 0.05: still small enough
    assert verdict((0.3, 0.1, 0.5), (0.04, 0.02, 0.06), (0.34, 0.1, 0.5)) == "supported"


def test_verdict_refuted_when_did_interval_contains_zero():
    assert verdict((0.1, -0.05, 0.25), (0.0, -0.02, 0.02), (0.1, 0.0, 0.2)) == "refuted"


def test_verdict_refuted_when_plausibility_falls_as_far_as_faithfulness():
    assert verdict((-0.1, -0.2, -0.05), (0.3, 0.2, 0.4), (0.2, 0.1, 0.3)) == "refuted"


def test_verdict_partial_when_both_fall_but_faithfulness_faster():
    assert verdict((0.2, 0.1, 0.3), (0.15, 0.1, 0.2), (0.35, 0.2, 0.5)).startswith("partial")


# --- per item ------------------------------------------------------------------------------------------

def test_gold_evidence_maps_gold_sentences_onto_my_split():
    ann = pd.DataFrame({"post_id": ["p1", "p1", "p1"], "DSM5_symptom": ["SLEEP_ISSUES"] * 3,
                        "status": [1, 1, 0],
                        "sentence_text": ["I cannot sleep.", "Nothing like this is in the post at all", "I lie awake all night."]})
    # the second gold sentence does not align and is left out; the status 0 row is not evidence
    assert gold_evidence(ann, "p1", "SLEEP_ISSUES", TEXT, split(TEXT)) == {0}


def test_score_a_true_positive():
    rec = record(attribution={"IG": {"0": 0.9, "1": 0.1, "2": 0.5}})
    row = score(rec, g_evid={0}, g_text="", vectors={}, refs={})
    assert row["correctness"] == "TP"
    assert row["evidence_jaccard"] == pytest.approx(1 / 2)           # stated {0, 2} vs gold {0}
    assert row["IG_stated_jaccard"] == 1.0                            # top-2 of IG is {0, 2}
    assert row["IG_gold_jaccard"] == 1.0                              # top-1 of IG is {0}
    assert row["comprehensiveness"] == 0.4


def test_score_a_false_positive_leaves_gold_metrics_undefined_not_zero():
    rec = record(gold=False, attribution={"IG": {"0": 0.9, "1": 0.1, "2": 0.5}})
    row = score(rec, g_evid=set(), g_text="", vectors={}, refs={})
    assert row["correctness"] == "FP"
    assert "evidence_jaccard" not in row and "IG_gold_jaccard" not in row
    # k = |S_evid| = 2, not |G_evid| = 0, so A-vs-S is measured on false positives too
    assert row["IG_stated_jaccard"] == 1.0


def test_score_nothing_cited_leaves_attribution_overlap_undefined():
    row = score(record(stated_evidence=[], attribution={"IG": {"0": 1.0}}), {0}, "", {}, {})
    assert "IG_stated_jaccard" not in row
    assert row["n_stated"] == 0
    assert math.isnan(score(record(stated_evidence=None), {0}, "", {}, {})["n_stated"])   # M1 states nothing


def test_score_plausibility_uses_other_posts_and_own_similarity_only_on_tp():
    vectors = {"why": [1.0, 0.0], "own gold": [1.0, 0.0]}
    refs = {"SLEEP_ISSUES": [[1.0, 0.0], [0.0, 1.0]], "FATIGUE": [[0.0, 1.0]]}
    row = score(record(stated_rationale="why"), {0}, "own gold", vectors, refs)
    assert row["plausibility"] == pytest.approx(0.5)
    assert row["null"] == pytest.approx(0.0)
    assert row["own_similarity"] == pytest.approx(1.0)
    row = score(record(stated_rationale="why", gold=False), set(), "", vectors, refs)
    assert "own_similarity" not in row


def test_ig_and_lime_agreement_needs_both():
    both = {"IG": {"0": 0.9, "1": 0.5, "2": 0.1}, "LIME": {"0": 0.8, "1": 0.6, "2": 0.0}}
    row = score(record(attribution=both), {0}, "", {}, {})
    assert row["ig_lime_jaccard"] == 1.0 and row["ig_lime_spearman"] == pytest.approx(1.0)
    assert "ig_lime_jaccard" not in score(record(attribution={"IG": both["IG"]}), {0}, "", {}, {})


# --- aggregation ---------------------------------------------------------------------------------------

def items_frame(n_tp, n_fp, tp_value, fp_value, symptom="SLEEP_ISSUES", model="M3", n_fn=0):
    rows = [{"model": model, "post_id": f"t{i}", "symptom": symptom, "correctness": "TP", "x": tp_value}
            for i in range(n_tp)]
    rows += [{"model": model, "post_id": f"f{i}", "symptom": symptom, "correctness": "FP", "x": fp_value}
             for i in range(n_fp)]
    rows += [{"model": model, "post_id": f"n{i}", "symptom": symptom, "correctness": "FN", "x": np.nan}
             for i in range(n_fn)]
    return pd.DataFrame(rows)


def test_compare_constant_groups():
    row = compare(items_frame(5, 4, 0.8, 0.5), "x")
    assert (row["n_tp"], row["n_fp"]) == (5, 4)
    assert row["gap"] == pytest.approx(0.3) and row["gap_lo"] == pytest.approx(0.3)
    assert row["cliffs_delta"] == 1.0


def test_summarise_flags_low_support_and_holm_corrects_per_symptom():
    items = pd.concat([items_frame(12, 12, 0.9, 0.1, "SLEEP_ISSUES", n_fn=1),
                       items_frame(3, 3, 0.9, 0.1, "PSYCHOMOTOR")])
    s = summarise(items)
    sleep = s[(s.symptom == "SLEEP_ISSUES") & (s.metric == "x")].iloc[0]
    psycho = s[(s.symptom == "PSYCHOMOTOR") & (s.metric == "x")].iloc[0]
    assert sleep["support"] == 13 and not sleep["low_support"]       # TP + FN
    assert psycho["low_support"]
    assert sleep["p_holm"] == pytest.approx(min(1.0, 2 * sleep["p"]))  # the smaller of two p-values, times 2
    assert "ALL" in set(s.symptom)


def test_h1_is_inconclusive_under_thirty_false_positives():
    items = items_frame(40, 29, 0.0, 0.0).assign(plausibility=0.8, comprehensiveness=0.5)
    assert h1(items, "M3")["verdict"].startswith("inconclusive")


def test_h1_supported_on_its_own_shape():
    # plausibility flat at 0.8, comprehensiveness 0.6 on TP and 0.1 on FP, with some spread
    rng = np.random.default_rng(0)
    tp = items_frame(60, 0, 0, 0).assign(plausibility=0.8 + rng.normal(0, 0.01, 60),
                                        comprehensiveness=0.6 + rng.normal(0, 0.05, 60))
    fp = items_frame(0, 40, 0, 0).assign(plausibility=0.8 + rng.normal(0, 0.01, 40),
                                        comprehensiveness=0.1 + rng.normal(0, 0.05, 40))
    out = h1(pd.concat([tp, fp]), "M3")
    assert out["verdict"] == "supported"
    assert out["did"][0] == pytest.approx(0.5, abs=0.05)


# --- end to end ----------------------------------------------------------------------------------------

def test_run_end_to_end_on_a_toy_dataset():
    posts = pd.DataFrame({"post_id": ["p1", "p2", "tr"], "text": [TEXT, "The weather is fine.", "I never sleep."]})
    ann = pd.DataFrame({"post_id": ["p1", "tr"], "sentence_text": ["I cannot sleep.", "I never sleep."],
                        "DSM5_symptom": ["SLEEP_ISSUES"] * 2, "status": [1, 1], "sentence_index": [0, 0],
                        "explanation": ["Says they cannot sleep.", "Reports never sleeping."]})
    records = [record(stated_rationale="sleep is poor", attribution={"IG": {"0": 0.9, "1": 0.0, "2": 0.4}}),
               record(post_id="p2", gold=False, stated_evidence=[0], stated_rationale="sleep is poor"),
               record(post_id="p1", symptom="FATIGUE", predicted=False, stated_evidence=None)]
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[1.0, float(len(t) % 3)] for t in texts]

    items, summary, hypothesis = run(records, posts, ann, train_ids=["tr"], embed=embed)
    assert list(items.correctness) == ["TP", "FP", "FN"]
    assert items.loc[0, "evidence_jaccard"] == pytest.approx(1 / 2)   # stated {0, 2}, gold {0}
    assert not np.isnan(items.loc[0, "plausibility"])                 # compared with the training post
    assert "sleep is poor" in calls[0] and "Reports never sleeping." not in calls[0]
    # the reference set is the training post's rationale only; p1's own gold must not be in it
    assert calls[1:] == [["Reports never sleeping."]]
    assert hypothesis[0]["verdict"].startswith("inconclusive")          # 1 false positive
