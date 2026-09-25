import re

import pytest

from src.metrics import erasure
from src.run_m3 import answer, confidence, post_records, stability

PROMPTS = {"system": "s", "joint": "Post:\n{numbered_post}\nAnswer."}


def fake_ask(seen=None, confidence_value=90):
    """A stand-in M3: SLEEP_ISSUES whenever a sentence mentions sleep, citing those sentences.
    It reads the numbered lines out of the prompt, as the real model has to."""
    def ask(prompt, system, n_sentences=None):
        lines = re.findall(r"^(\d+): (.*)$", prompt, flags=re.M)
        if seen is not None:
            seen.append([s for _, s in lines])
        assert [int(i) for i, _ in lines] == list(range(len(lines)))       # renumbered from 0
        cited = [int(i) for i, s in lines if "sleep" in s.lower()]
        if not cited:
            return {"ok": True, "attempts": 1, "symptoms": [
                {"label": "NO_SYMPTOMS", "evidence": [], "rationale": "", "confidence": 80}]}
        return {"ok": True, "attempts": 1, "symptoms": [
            {"label": "SLEEP_ISSUES", "evidence": cited, "rationale": "mentions sleep", "confidence": confidence_value}]}
    return ask


def sleep_confidence(ask_fn):
    """The M3 predict function erasure is given: the joint prompt's confidence in SLEEP_ISSUES."""
    return lambda kept: confidence(answer(kept, PROMPTS, ask_fn), "SLEEP_ISSUES")


def test_confidence_reads_the_label_or_zero():
    result = {"ok": True, "symptoms": [{"label": "FATIGUE", "confidence": 70}]}
    assert confidence(result, "FATIGUE") == pytest.approx(0.7)
    assert confidence(result, "SLEEP_ISSUES") == 0.0
    assert confidence({"ok": False, "symptoms": None}, "FATIGUE") is None
    assert confidence({"ok": True, "symptoms": [{"label": "FATIGUE", "confidence": None}]}, "FATIGUE") is None


def test_erasure_removes_then_keeps_only_the_evidence():
    seen = []
    sentences = ["I cannot sleep.", "The weather is fine.", "Dinner was nice."]
    comp, suff = erasure(sentences, {0}, 0.9, sleep_confidence(fake_ask(seen)))
    assert seen == [["The weather is fine.", "Dinner was nice."], ["I cannot sleep."]]
    assert comp == pytest.approx(0.9)        # without the evidence the label disappears: 0.9 - 0
    assert suff == pytest.approx(0.0)        # the evidence alone keeps it at 0.9


def test_erasure_is_undefined_when_nothing_is_cited_or_nothing_would_remain():
    sentences = ["I cannot sleep.", "Weather."]
    assert erasure(sentences, set(), 0.9, sleep_confidence(fake_ask())) == (None, None)
    comp, suff = erasure(["I cannot sleep."], {0}, 0.9, sleep_confidence(fake_ask()))
    assert comp is None and suff == pytest.approx(0.0)


def test_post_records_writes_predicted_and_missed_symptoms():
    records, result = post_records("p1", "I cannot sleep. The weather is fine.", ["SLEEP_ISSUES", "FATIGUE"],
                                   PROMPTS, fake_ask())
    assert result["ok"]
    by = {r["symptom"]: r for r in records}
    assert set(by) == {"SLEEP_ISSUES", "FATIGUE"}
    tp, fn = by["SLEEP_ISSUES"], by["FATIGUE"]
    assert (tp["predicted"], tp["gold"], fn["predicted"], fn["gold"]) == (True, True, False, True)
    assert tp["stated_evidence"] == [0] and tp["stated_rationale"] == "mentions sleep"
    assert tp["comprehensiveness"] == pytest.approx(0.9) and tp["sufficiency"] == pytest.approx(0.0)
    assert fn["stated_evidence"] is None and fn["comprehensiveness"] is None


def test_a_false_positive_and_a_no_symptom_post():
    records, _ = post_records("p2", "I sleep fine these days.", ["NO_SYMPTOMS"], PROMPTS, fake_ask())
    assert [(r["symptom"], r["predicted"], r["gold"]) for r in records] == [("SLEEP_ISSUES", True, False)]
    records, _ = post_records("p3", "The weather is fine.", ["NO_SYMPTOMS"], PROMPTS, fake_ask())
    assert records == []                     # nothing predicted, nothing gold: no (post, symptom) to score


def test_a_parse_failure_gives_no_records_and_is_reported():
    failing = lambda *a, **k: {"ok": False, "attempts": 3, "symptoms": None, "reason": "not json"}
    records, result = post_records("p4", "I cannot sleep.", ["SLEEP_ISSUES"], PROMPTS, failing)
    assert records == [] and not result["ok"]


def rec(post, symptom, evidence):
    return {"post_id": post, "symptom": symptom, "predicted": True, "stated_evidence": evidence}


def test_stability_counts_agreeing_on_nothing_as_agreement():
    run_a = [rec("p1", "SLEEP_ISSUES", [0, 1])]
    run_b = [rec("p1", "SLEEP_ISSUES", [0]), rec("p1", "FATIGUE", [2])]
    s = stability([run_a, run_b], ["p1", "p2"])
    # p1: {SLEEP} vs {SLEEP, FATIGUE} = 0.5; p2: both predict nothing = 1.0
    assert s["labels"] == pytest.approx(0.75)
    # only SLEEP_ISSUES on p1 was predicted by both: {0, 1} vs {0} = 0.5
    assert s["evidence"] == pytest.approx(0.5) and s["n_shared_predictions"] == 1


def test_stability_of_identical_runs_is_one():
    run = [rec("p1", "SLEEP_ISSUES", [0])]
    assert stability([run, run, run], ["p1", "p2"])["labels"] == 1.0
