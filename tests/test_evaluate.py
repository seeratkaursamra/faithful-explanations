import math

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compare, density, gold_evidence, h1, h3, h3_verdict, run, score, summarise, verdict
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
    assert row["null_similarity"] == pytest.approx(0.0)
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

    items, summary, hypotheses = run(records, posts, ann, train_ids=["tr"], embed=embed)
    assert list(items.correctness) == ["TP", "FP", "FN"]
    assert items.loc[0, "evidence_jaccard"] == pytest.approx(1 / 2)   # stated {0, 2}, gold {0}
    assert not np.isnan(items.loc[0, "plausibility"])                 # compared with the training post
    assert "sleep is poor" in calls[0] and "Reports never sleeping." not in calls[0]
    # the reference set is the training post's rationale only; p1's own gold must not be in it
    assert calls[1:] == [["Reports never sleeping."]]
    assert hypotheses["H1"][0]["verdict"].startswith("inconclusive")    # 1 false positive
    assert hypotheses["H3"][0]["verdict"].startswith("inconclusive")


def test_the_bootstrap_seed_reaches_every_interval():
    items = pd.concat([items_frame(8, 0, 0, 0), items_frame(0, 6, 0, 0)])
    items["x"] = np.random.default_rng(1).random(len(items))
    first = compare(items, "x", {"n_resamples": 200, "seed": 0})
    again = compare(items, "x", {"n_resamples": 200, "seed": 0})
    other = compare(items, "x", {"n_resamples": 200, "seed": 5})
    assert first == again
    assert (first["gap_lo"], first["tp_lo"]) != (other["gap_lo"], other["tp_lo"])


def test_the_config_matches_the_code_defaults():
    import yaml
    from src.evaluate import BOOT
    assert yaml.safe_load(open("configs/evaluate.yaml"))["bootstrap"] == BOOT



# --- H4: marker density ----------------------------------------------------------------------------

def test_density_is_top_minus_rest_with_words_pooled():
    sentences = ["I hate myself.", "The weather is nice today.", "Dinner was fine."]
    d = density(sentences, {0}, negative_words={"hate"})
    # top: I, hate, myself -> first person 2/3, negative 1/3; rest: 8 words, none of either
    assert d["first_person"] == pytest.approx(2 / 3)
    assert d["negative_affect"] == pytest.approx(1 / 3)
    assert density(sentences, {0, 1, 2}, {"hate"}) == {}          # nothing left to compare with
    assert density(sentences, {7}, {"hate"}) == {}                 # an index past the post is ignored


def test_score_adds_density_only_when_it_has_the_sentences_and_words():
    sentences = ["I cannot sleep.", "The weather is fine.", "I lie awake all night."]
    rec = record(attribution={"M1": {"0": 0.9, "1": 0.0, "2": 0.5}})
    row = score(rec, {0}, "", {}, {}, sentences=sentences, negative_words=set())
    # top 2 by attribution is {0, 2}: 9 words (NLTK splits "cannot" into "can not") with two "I" = 2/9,
    # against 4 words with none. Top 1 would give 1/4 - 1/9 instead.
    assert row["M1_density_first_person"] == pytest.approx(2 / 9)
    assert "M1_density_absolutist" in row
    assert not any(k.startswith("M1_density") for k in score(rec, {0}, "", {}, {}))


def test_compare_reports_the_mean_over_all_items():
    row = compare(items_frame(3, 1, 0.8, 0.4), "x")
    assert row["all_mean"] == pytest.approx((3 * 0.8 + 0.4) / 4)


# --- H3 --------------------------------------------------------------------------------------------

def h3_rows(**did):
    """Rows as h3 builds them: every named symptom gets that DiD and enough support."""
    return [{"symptom": s, "low_support": False, "did": (d, d - 0.1, d + 0.1)} for s, d in did.items()]


def test_h3_verdict_rules():
    assert h3_verdict(h3_rows(COGNITIVE_ISSUES=0.4, PSYCHOMOTOR=0.3, SUICIDAL_THOUGHTS=0.1)).startswith("supported")
    assert h3_verdict(h3_rows(COGNITIVE_ISSUES=0.4, PSYCHOMOTOR=0.05, SUICIDAL_THOUGHTS=0.1)).startswith("refuted")
    # the reference has to be usable
    rows = h3_rows(COGNITIVE_ISSUES=0.4, SUICIDAL_THOUGHTS=0.1)
    rows[1]["low_support"] = True
    assert h3_verdict(rows).startswith("inconclusive")
    # a low-support target is left out rather than argued from
    rows = h3_rows(COGNITIVE_ISSUES=0.4, PSYCHOMOTOR=-1.0, SUICIDAL_THOUGHTS=0.1)
    rows[1]["low_support"] = True
    assert h3_verdict(rows) == "supported, on COGNITIVE_ISSUES"
    assert h3_verdict(h3_rows(SUICIDAL_THOUGHTS=0.1)).startswith("inconclusive")


def test_h3_computes_per_symptom_and_orders_by_training_frequency():
    rng = np.random.default_rng(0)
    frames = []
    for symptom, faith_fp in [("COGNITIVE_ISSUES", 0.1), ("SUICIDAL_THOUGHTS", 0.5)]:
        tp = items_frame(12, 0, 0, 0, symptom).assign(plausibility=0.8, comprehensiveness=0.6 + rng.normal(0, 0.01, 12))
        fp = items_frame(0, 12, 0, 0, symptom).assign(plausibility=0.8, comprehensiveness=faith_fp + rng.normal(0, 0.01, 12))
        frames += [tp.assign(post_id=tp.post_id + symptom), fp.assign(post_id=fp.post_id + symptom)]
    out = h3(pd.concat(frames), "M3", train_counts={"SUICIDAL_THOUGHTS": 130, "COGNITIVE_ISSUES": 47})
    by = {r["symptom"]: r for r in out["by_symptom"]}
    assert by["COGNITIVE_ISSUES"]["did"][0] == pytest.approx(0.5, abs=0.02)
    assert by["SUICIDAL_THOUGHTS"]["did"][0] == pytest.approx(0.1, abs=0.02)
    assert [r["symptom"] for r in out["by_symptom"][:2]] == ["SUICIDAL_THOUGHTS", "COGNITIVE_ISSUES"]
    assert out["verdict"] == "supported, on COGNITIVE_ISSUES"


def test_attribution_quality_is_carried_through():
    row = score(record(ig_delta=0.003, lime_r2=0.91), {0}, "", {}, {})
    assert (row["ig_delta"], row["lime_r2"]) == (0.003, 0.91)
    assert "ig_delta" not in score(record(), {0}, "", {}, {})


def test_no_metric_name_turns_into_a_missing_value_in_the_csv(tmp_path):
    # pandas reads "null", "NA", "nan" and friends in a CSV as missing, which would erase a metric's name
    rec = record(stated_rationale="why", attribution={"IG": {"0": 0.9, "1": 0.1, "2": 0.5}})
    row = score(rec, {0}, "own", {"why": [1.0, 0.0], "own": [1.0, 0.0]}, {"SLEEP_ISSUES": [[1.0, 0.0]], "FATIGUE": [[0.0, 1.0]]})
    path = tmp_path / "names.csv"
    pd.DataFrame({"metric": list(row)}).to_csv(path, index=False)
    assert pd.read_csv(path)["metric"].notna().all()


def test_gold_past_the_cut_is_reported_for_m2():
    # 2 seen means sentences 0 and 1; sentence 2 is the first past the cut
    row = score(record(model="M2", sentences_seen=2), g_evid={1, 2}, g_text="", vectors={}, refs={})
    assert row["gold_past_cut"] == pytest.approx(0.5)
    assert "gold_past_cut" not in score(record(), {0, 3}, "", {}, {})


def test_randomised_check_uses_the_median_absolute_agreement():
    from src.evaluate import randomised_check
    # a reversed ranking (-0.9) is as much "not degraded" as a kept one: it is still the text's structure
    kept = pd.DataFrame({"ig_random_spearman": [0.9, -0.9, 0.8, np.nan]})
    assert randomised_check(kept) == {"n": 3, "median_abs_spearman": pytest.approx(0.9), "degraded": False}
    fell_apart = pd.DataFrame({"ig_random_spearman": [0.1, -0.2, 0.3]})
    assert randomised_check(fell_apart)["degraded"]
    assert randomised_check(pd.DataFrame({"x": [1]})) is None


# --- H2 and H4 ---------------------------------------------------------------------------------------

def symptom_items(model, symptom, tp, fp, fn, faith):
    rows = [(k, faith) for k in ["TP"] * tp + ["FP"] * fp] + [("FN", np.nan)] * fn
    return pd.DataFrame([{"model": model, "post_id": f"{model}{symptom}{i}", "symptom": symptom, "correctness": k,
                          "comprehensiveness": c} for i, (k, c) in enumerate(rows)])


def test_per_symptom_f1_needs_no_true_negatives():
    from src.evaluate import per_symptom
    row = per_symptom(symptom_items("M2", "FATIGUE", tp=3, fp=1, fn=2, faith=0.4)).iloc[0]
    assert row["f1"] == pytest.approx(6 / 9) and row["support"] == 5 and row["faithfulness"] == pytest.approx(0.4)


SYMPTOM_SET = ["DEPRESSED_MOOD", "WORTHLESSNESS", "FATIGUE", "SLEEP_ISSUES"]


def h2_items(faith_rises_with_f1):
    frames = []
    for model in ("M2", "M3"):
        for k, symptom in enumerate(SYMPTOM_SET):
            tp, fn = 6 + 2 * k, 6 - k                        # F1 climbs with k; support always >= 10
            faith = 0.2 + 0.1 * k if faith_rises_with_f1 else 0.6 - 0.1 * k
            frames.append(symptom_items(model, symptom, tp=tp, fp=3, fn=fn, faith=faith))
    return pd.concat(frames, ignore_index=True)


def test_h2_refuted_when_faithfulness_rises_with_accuracy():
    from src.evaluate import h2
    out = h2(h2_items(True), boot={"n_resamples": 200, "seed": 0})
    assert out["spearman"][0] > 0.8 and out["verdict"].startswith("refuted") and out["points"] == 8


def test_h2_supported_when_they_do_not():
    from src.evaluate import h2
    out = h2(h2_items(False), boot={"n_resamples": 200, "seed": 0})
    assert out["spearman"][0] < 0 and out["verdict"] == "supported"


def test_h2_leaves_out_low_support_symptoms_and_needs_three_points():
    from src.evaluate import h2
    items = pd.concat([h2_items(True), symptom_items("M2", "PSYCHOMOTOR", tp=2, fp=1, fn=1, faith=0.9)])
    assert "PSYCHOMOTOR" not in h2(items, boot={"n_resamples": 50, "seed": 0})["symptoms"]
    assert h2(symptom_items("M2", "FATIGUE", 8, 2, 4, 0.3))["verdict"].startswith("inconclusive")


def m1_items(density):
    return pd.DataFrame([{"model": "M1", "post_id": f"p{i}", "symptom": "FATIGUE", "correctness": "TP",
                          "M1_density_marker_words": density + 0.01 * (i % 3)} for i in range(20)])


def test_h4_verdicts():
    from src.evaluate import h4
    boot = {"n_resamples": 200, "seed": 0}
    assert h4(m1_items(0.05), {"M1": 0.45, "M2": 0.51}, boot)["verdict"] == "supported"
    assert "trails M2" in h4(m1_items(0.05), {"M1": 0.35, "M2": 0.51}, boot)["verdict"]
    assert "not richer" in h4(m1_items(-0.05), {"M1": 0.45, "M2": 0.51}, boot)["verdict"]
    assert h4(m1_items(0.05), {"M1": 0.45}, boot)["verdict"].startswith("inconclusive")


def noisy_items(rates, seed):
    """Comprehensiveness as 0/1 flips, as M3's are, drawn per item at a rate per symptom."""
    rng = np.random.default_rng(seed)
    frames = []
    for model in ("M2", "M3"):
        for k, symptom in enumerate(SYMPTOM_SET):
            tp, fn = 6 + 2 * k, 6 - k
            flips = (rng.random(tp + 3) < rates[k]).astype(float)
            frames.append(pd.DataFrame(
                [{"model": model, "post_id": f"{model}{symptom}{i}", "symptom": symptom, "correctness": c,
                  "comprehensiveness": f} for i, (c, f) in enumerate(zip(["TP"] * tp + ["FP"] * 3, flips))]
                + [{"model": model, "post_id": f"{model}{symptom}n{i}", "symptom": symptom, "correctness": "FN",
                    "comprehensiveness": np.nan} for i in range(fn)]))
    return pd.concat(frames, ignore_index=True)


def test_h2_a_positive_correlation_whose_interval_reaches_zero_is_still_supported():
    # the prereg's rule is the interval, not the sign of the estimate
    from src.evaluate import h2
    out = h2(noisy_items([0.45, 0.5, 0.5, 0.55], seed=4), boot={"n_resamples": 300, "seed": 0})
    rho, lo, hi = out["spearman"]
    assert rho > 0 and lo < 0 < hi
    assert out["verdict"] == "supported"


def test_h4_a_positive_density_whose_interval_crosses_zero_is_not_enough():
    from src.evaluate import h4
    items = pd.DataFrame([{"model": "M1", "post_id": f"p{i}", "symptom": "FATIGUE", "correctness": "TP",
                           "M1_density_marker_words": 0.10 if i % 2 else -0.09} for i in range(20)])
    out = h4(items, {"M1": 0.45, "M2": 0.51}, {"n_resamples": 300, "seed": 0})
    mean, lo, hi = out["marker_density"]
    assert mean > 0 and lo < 0
    assert "not richer" in out["verdict"]


# --- the whole-study conditions ------------------------------------------------------------------------

def study_items(n_fp=40, ig_lime=0.5, random_rho=0.2):
    rows = [{"model": "M2", "post_id": f"t{i}", "symptom": "FATIGUE", "correctness": "TP",
             "ig_lime_spearman": ig_lime, "ig_random_spearman": random_rho} for i in range(10)]
    rows += [{"model": "M2", "post_id": f"f{i}", "symptom": "FATIGUE", "correctness": "FP",
              "ig_lime_spearman": ig_lime, "ig_random_spearman": random_rho} for i in range(n_fp)]
    return pd.DataFrame(rows)


BASELINE_OK = {"micro_f1": {"mean": 0.505}, "within_tolerance": True}
LOGS_OK = {"M3": {"failure_rate": 0.02}, "M2": {"decoupled_asked": 100, "decoupled_failures": 3}}


def status_of(result, fragment):
    return next(c["status"] for c in result["checks"] if fragment in c["condition"])


def test_study_is_clear_only_when_every_condition_was_run_and_passed():
    from src.evaluate import study_checks
    assert study_checks(study_items(), BASELINE_OK, LOGS_OK)["status"] == "clear"


def test_each_condition_can_trigger_on_its_own():
    from src.evaluate import study_checks
    cases = [
        (study_checks(study_items(), {"micro_f1": {"mean": 0.45}, "within_tolerance": False}, LOGS_OK), "baseline"),
        (study_checks(study_items(ig_lime=0.1), BASELINE_OK, LOGS_OK), "IG and LIME"),
        (study_checks(study_items(random_rho=0.8), BASELINE_OK, LOGS_OK), "randomised head"),
        (study_checks(study_items(), BASELINE_OK, {**LOGS_OK, "M3": {"failure_rate": 0.12}}), "parse failures"),
        (study_checks(study_items(), BASELINE_OK, {**LOGS_OK, "M2": {"decoupled_asked": 50, "decoupled_failures": 6}}),
         "parse failures"),
        (study_checks(study_items(n_fp=29), BASELINE_OK, LOGS_OK), "false positives"),
    ]
    for result, fragment in cases:
        assert result["status"] == "inconclusive", fragment
        assert status_of(result, fragment) == "triggered", fragment


def test_a_missing_input_is_not_run_never_a_pass():
    from src.evaluate import study_checks
    result = study_checks(study_items().drop(columns=["ig_random_spearman"]))
    assert result["status"] == "incomplete"
    assert status_of(result, "baseline") == "not run"
    assert status_of(result, "randomised head") == "not run"
    assert status_of(result, "parse failures") == "not run"
    assert status_of(result, "IG and LIME") == "clear"


def test_ig_lime_agreement_is_judged_on_m2_alone():
    # the prereg names M2; another model's rows, however they disagree, must not move the check
    from src.evaluate import study_checks
    other = study_items(ig_lime=-0.9).assign(model="MX")
    result = study_checks(pd.concat([study_items(), other, other]), BASELINE_OK, LOGS_OK)
    assert status_of(result, "IG and LIME") == "clear"


# --- paired contrasts ------------------------------------------------------------------------------------

def test_ig_minus_lime_is_paired_on_the_same_m2_items():
    from src.evaluate import method_contrasts
    items = pd.DataFrame([{"model": "M2", "post_id": f"p{i}", "symptom": "FATIGUE", "correctness": "TP",
                           "IG_stated_jaccard": 0.5 + 0.05 * i, "LIME_stated_jaccard": 0.3 + 0.05 * i,
                           "IG_gold_jaccard": 0.4} for i in range(12)])
    rows = {r["metric"]: r for r in method_contrasts(items, {"n_resamples": 200, "seed": 0})}
    # every item differs by exactly 0.2, so a paired interval collapses on it even though levels vary
    assert rows["stated_jaccard"]["diff"] == pytest.approx(0.2)
    assert rows["stated_jaccard"]["lo"] == pytest.approx(0.2) and rows["stated_jaccard"]["hi"] == pytest.approx(0.2)
    assert rows["stated_jaccard"]["p"] < 0.01
    assert "gold_jaccard" not in rows                  # no LIME counterpart, nothing to pair


def test_models_are_compared_only_on_items_both_have():
    from src.evaluate import model_contrasts
    m2 = pd.DataFrame([{"model": "M2", "post_id": f"p{i}", "symptom": "FATIGUE", "correctness": "TP",
                        "plausibility": 0.6 + 0.01 * i, "comprehensiveness": 0.5} for i in range(10)])
    m3 = m2.assign(model="M3", plausibility=m2["plausibility"] + 0.1)
    only_m3 = pd.DataFrame([{"model": "M3", "post_id": "extra", "symptom": "FATIGUE", "correctness": "FP",
                             "plausibility": 0.0, "comprehensiveness": 0.0}])
    # a second symptom on the same posts, so pairing on the post alone would cross FATIGUE with SLEEP
    m2_sleep = m2.assign(symptom="SLEEP_ISSUES", plausibility=0.9)
    m3_sleep = m2_sleep.assign(model="M3", plausibility=1.0)
    items = pd.concat([m2, m3, only_m3, m2_sleep, m3_sleep])
    rows = {r["metric"]: r for r in model_contrasts(items, {"n_resamples": 200, "seed": 0})}
    assert rows["plausibility"]["n"] == 20            # 10 posts x 2 symptoms; the unmatched M3 item is left out
    assert rows["plausibility"]["contrast"] == "M2 - M3"
    assert rows["plausibility"]["diff"] == pytest.approx(-0.1)
    assert rows["plausibility"]["lo"] == pytest.approx(-0.1)     # every pair differs by 0.1: a paired interval collapses
    assert math.isnan(rows["comprehensiveness"]["p"])                  # identical on every item: nothing to test
