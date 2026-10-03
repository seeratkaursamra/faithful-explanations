import pandas as pd

from src.cases import pick, reading_file


def scored(model, rows):
    return pd.DataFrame([{"model": model, "post_id": p, "symptom": "FATIGUE", "correctness": k,
                          "plausibility": pl, "comprehensiveness": c} for p, k, pl, c in rows])


def test_pick_takes_the_most_plausible_of_each_models_least_faithful_quarter():
    m3 = scored("M3", [("a", "TP", 0.9, 0.9), ("b", "FP", 0.8, 0.0), ("c", "TP", 0.7, 0.0),
                       ("d", "FP", 0.95, 0.5), ("e", "FN", 0.99, 0.0)] +
                [(f"x{i}", "TP", 0.5, 0.9) for i in range(4)])
    cases = pick(m3, per_model=2)
    # the lowest quarter is the 0.0s; e is a false negative, never an explanation to judge;
    # d is the most plausible overall but not unfaithful, so it is not a case
    assert list(cases["post_id"]) == ["b", "c"]


def test_pick_is_per_model_and_deterministic():
    both = pd.concat([scored("M2", [(f"m{i}", "TP", 0.5 + 0.01 * i, 0.0) for i in range(8)]),
                      scored("M3", [(f"n{i}", "FP", 0.6, 0.0) for i in range(8)])])
    cases = pick(both, per_model=3)
    assert list(cases["model"]) == ["M2"] * 3 + ["M3"] * 3
    assert list(cases["post_id"][3:]) == ["n0", "n1", "n2"]          # a plausibility tie breaks by post id
    assert pick(both, per_model=3).equals(cases)


def test_the_reading_file_bolds_what_the_model_cited_and_shows_both_rationales():
    posts = pd.DataFrame({"post_id": ["p1"], "text": ["I cannot sleep. The weather is fine."]})
    ann = pd.DataFrame({"post_id": ["p1"], "DSM5_symptom": ["SLEEP_ISSUES"], "status": [1], "sentence_index": [0],
                        "explanation": ["Reports insomnia."]})
    cases = scored("M3", [("p1", "TP", 0.8, 0.0)]).assign(symptom="SLEEP_ISSUES")
    records = [{"model": "M3", "post_id": "p1", "symptom": "SLEEP_ISSUES", "stated_evidence": [1],
                "stated_rationale": "talks about sleep"}]
    text = reading_file(cases, records, posts, ann)
    assert "0: I cannot sleep." in text and "**1: The weather is fine.**" in text
    assert "talks about sleep" in text and "Reports insomnia." in text and "never commit" in text


def test_low_faithfulness_is_judged_within_each_model():
    # M2 is faithful across the board, M3 is not: a pooled cut would give every case to M3
    both = pd.concat([scored("M2", [(f"m{i}", "TP", 0.7, 0.5 + 0.05 * i) for i in range(8)]),
                      scored("M3", [(f"n{i}", "FP", 0.7, 0.05 * i) for i in range(8)])])
    cases = pick(both, per_model=2)
    assert list(cases["model"]) == ["M2", "M2", "M3", "M3"]
    assert set(cases[cases["model"] == "M2"]["post_id"]) == {"m0", "m1"}     # M2's own least faithful


def test_on_a_flip_low_means_the_label_survived():
    # one item in ten did not flip: the 25th percentile of the 0/1 values is 1, so a percentile cut would take
    # every item and return the most plausible faithful ones as cases
    m3 = scored("M3", [("kept", "FP", 0.6, 0.0)] + [(f"flip{i}", "FP", 0.9, 1.0) for i in range(9)])
    assert list(pick(m3, per_model=3)["post_id"]) == ["kept"]
    # a graded measure still uses the percentile
    m2 = scored("M2", [(f"g{i}", "FP", 0.9, 0.1 * i) for i in range(8)])
    assert list(pick(m2, per_model=8)["post_id"]) == ["g0", "g1"]
