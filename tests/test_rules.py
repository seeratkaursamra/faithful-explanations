"""Every pre-registered threshold, tested just below, at, and just above it.

A rule written the wrong way round (< for <=, a flipped comparison) still returns a plausible verdict, so
each boundary is pinned to the prereg's wording: "below 0.147" means 0.147 itself is not negligible,
"fewer than 30" means 30 is enough, "above 10%" means 10% is still allowed.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (H2_MAX_UPPER, H4_MAX_F1_GAP, MAX_PARSE_FAILURE, MIN_FALSE_POSITIVES, MIN_FP_PER_SYMPTOM,
                          MIN_IG_LIME_AGREEMENT, NEGLIGIBLE_DELTA, RANDOMISED_MAX_AGREEMENT, h1, h2_verdict, h3,
                          h3_verdict, h4_verdict, study_checks, verdict)
from src.metrics import MIN_SUPPORT

EPS = 1e-9
BOOT = {"n_resamples": 30, "seed": 0}


def test_the_thresholds_are_the_registered_ones():
    assert (NEGLIGIBLE_DELTA, MIN_FALSE_POSITIVES, MIN_FP_PER_SYMPTOM, MIN_SUPPORT) == (0.147, 30, 5, 10)
    assert (H2_MAX_UPPER, H4_MAX_F1_GAP, MIN_IG_LIME_AGREEMENT, RANDOMISED_MAX_AGREEMENT, MAX_PARSE_FAILURE) == \
        (0.5, 0.10, 0.2, 0.5, 0.10)


# --- H1 -------------------------------------------------------------------------------------------------

FAITH = (0.9, 0.8, 1.0)


def test_h1_a_did_interval_touching_zero_counts_as_containing_it():
    assert verdict((0.5, 0.0, 0.8), (0.0, -0.1, 0.1), FAITH) == "refuted"
    assert verdict((0.5, EPS, 0.8), (0.0, -0.1, 0.1), FAITH) == "supported"


def test_h1_plausibility_equal_to_faithfulness_is_refuted():
    assert verdict((0.5, 0.2, 0.8), (0.9, 0.8, 1.0), FAITH) == "refuted"
    assert verdict((0.5, 0.2, 0.8), (0.9 - EPS, 0.8, 1.0), FAITH).startswith("partial")


def test_h1_negligible_means_strictly_below_the_cut_off():
    # the plausibility interval excludes zero, so only the cut-off decides
    assert verdict((0.5, 0.2, 0.8), (NEGLIGIBLE_DELTA, 0.05, 0.3), FAITH).startswith("partial")
    assert verdict((0.5, 0.2, 0.8), (NEGLIGIBLE_DELTA - EPS, 0.05, 0.3), FAITH) == "supported"


def test_h1_a_plausibility_interval_touching_zero_counts_as_holding():
    assert verdict((0.5, 0.2, 0.8), (0.3, 0.0, 0.6), FAITH) == "supported"
    assert verdict((0.5, 0.2, 0.8), (0.3, EPS, 0.6), FAITH).startswith("partial")


def items(n_tp, n_fp, symptom="FATIGUE", model="M3", n_fn=0, prefix=""):
    rows = [{"model": model, "post_id": f"{prefix}t{i}", "symptom": symptom, "correctness": "TP",
             "plausibility": 0.8, "comprehensiveness": 1.0} for i in range(n_tp)]
    rows += [{"model": model, "post_id": f"{prefix}f{i}", "symptom": symptom, "correctness": "FP",
              "plausibility": 0.8, "comprehensiveness": 0.0} for i in range(n_fp)]
    rows += [{"model": model, "post_id": f"{prefix}n{i}", "symptom": symptom, "correctness": "FN"} for i in range(n_fn)]
    return pd.DataFrame(rows)


def test_h1_thirty_false_positives_are_enough_and_twenty_nine_are_not():
    assert not h1(items(40, MIN_FALSE_POSITIVES), "M3", boot=BOOT)["verdict"].startswith("inconclusive")
    assert h1(items(40, MIN_FALSE_POSITIVES - 1), "M3", boot=BOOT)["verdict"].startswith("inconclusive")


# --- H2 and H4 ------------------------------------------------------------------------------------------

def test_h2_refutation_needs_the_interval_strictly_above_zero():
    assert h2_verdict((0.6, EPS, 0.9)).startswith("refuted")
    assert not h2_verdict((0.6, 0.0, 0.9)).startswith("refuted")


def test_h2_support_needs_the_upper_bound_strictly_below_the_limit():
    assert h2_verdict((0.0, -0.4, H2_MAX_UPPER - EPS)) == "supported"
    assert h2_verdict((0.0, -0.4, H2_MAX_UPPER)).startswith("inconclusive")


def test_h4_a_gap_of_exactly_the_margin_is_still_modest():
    assert h4_verdict(H4_MAX_F1_GAP, (0.05, 0.01, 0.09)) == "supported"
    assert "trails M2" in h4_verdict(H4_MAX_F1_GAP + EPS, (0.05, 0.01, 0.09))


def test_h4_marker_density_needs_its_interval_strictly_above_zero():
    assert h4_verdict(0.05, (0.05, EPS, 0.09)) == "supported"
    assert "not richer" in h4_verdict(0.05, (0.05, 0.0, 0.09))


# --- H3 -------------------------------------------------------------------------------------------------

def rows(**did):
    return [{"symptom": s, "low_support": False, "few_fp": False, "did": (d, d, d)} for s, d in did.items()]


def test_h3_a_target_equal_to_the_reference_is_not_larger():
    assert h3_verdict(rows(COGNITIVE_ISSUES=0.4, SUICIDAL_THOUGHTS=0.4)).startswith("refuted")
    assert h3_verdict(rows(COGNITIVE_ISSUES=0.4 + EPS, SUICIDAL_THOUGHTS=0.4)).startswith("supported")


def test_h3_ten_examples_and_five_false_positives_are_enough():
    def symptom_rows(support, n_fp):
        frames = [items(support, n_fp, "COGNITIVE_ISSUES", prefix="c"), items(12, 12, "SUICIDAL_THOUGHTS", prefix="s")]
        return {r["symptom"]: r for r in h3(pd.concat(frames), "M3", boot=BOOT)["by_symptom"]}
    at = symptom_rows(MIN_SUPPORT, MIN_FP_PER_SYMPTOM)["COGNITIVE_ISSUES"]
    assert not at["low_support"] and not at["few_fp"]
    below = symptom_rows(MIN_SUPPORT - 1, MIN_FP_PER_SYMPTOM - 1)["COGNITIVE_ISSUES"]
    assert below["low_support"] and below["few_fp"]


# --- the whole-study conditions -------------------------------------------------------------------------

BASELINE = {"micro_f1": {"mean": 0.51}, "within_tolerance": True}


def study(ig_lime=0.5, random_rho=0.1, rate=0.0, n_fp=40):
    frame = items(10, n_fp, model="M2").assign(ig_lime_spearman=ig_lime, ig_random_spearman=random_rho)
    result = study_checks(frame, BASELINE, {"M3": {"failure_rate": rate}}, BOOT)
    return {c["condition"].split(":")[0]: c["status"] for c in result["checks"]}


def status(checks, start):
    return next(s for c, s in checks.items() if c.startswith(start))


def test_ig_lime_agreement_exactly_at_the_limit_is_not_below_it():
    assert status(study(ig_lime=MIN_IG_LIME_AGREEMENT), "IG and LIME") == "clear"
    assert status(study(ig_lime=MIN_IG_LIME_AGREEMENT - EPS), "IG and LIME") == "triggered"


def test_a_randomised_head_agreeing_at_exactly_the_limit_has_not_degraded():
    assert status(study(random_rho=RANDOMISED_MAX_AGREEMENT), "IG does not degrade") == "triggered"
    assert status(study(random_rho=RANDOMISED_MAX_AGREEMENT - EPS), "IG does not degrade") == "clear"


def test_a_parse_failure_rate_of_exactly_ten_percent_is_allowed():
    assert status(study(rate=MAX_PARSE_FAILURE), "LLM parse failures") == "clear"
    assert status(study(rate=MAX_PARSE_FAILURE + EPS), "LLM parse failures") == "triggered"


def test_the_study_floor_is_thirty_usable_false_positives():
    assert status(study(n_fp=MIN_FALSE_POSITIVES), "M2") == "clear"
    assert status(study(n_fp=MIN_FALSE_POSITIVES - 1), "M2") == "triggered"
