import numpy as np
import pytest
from scipy.special import logit

from src.m1 import explain, fit, predict_proba
from src.sentences import split

CLASSES = ["FATIGUE", "NO_SYMPTOMS", "SLEEP_ISSUES"]
TEXTS = [
    "I cannot sleep at night. The weather is fine.",
    "I never sleep anymore. I like dogs.",
    "I lie awake and cannot sleep. We went out.",
    "I feel tired all the time. Work is ok.",
    "Always tired and exhausted. Nice day.",
    "So tired again today. The bus was late.",
    "The film was good. We went out.",
    "Dinner was nice. The weather is fine.",
]
LABELS = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
NEGATIVE = {"tired", "exhausted"}


def model():
    return fit(TEXTS, LABELS, CLASSES, NEGATIVE, min_df=1)


def test_sentences_markers_crossing_and_bias_add_up_to_the_logit():
    m = model()
    text = "I cannot sleep at all. The weather is fine. I am tired."
    for label in CLASSES:
        scores, parts = explain(m, text, split(text), label)
        p = predict_proba(m, [text])[0, CLASSES.index(label)]
        assert sum(scores.values()) + parts["markers"] + parts["crossing"] + parts["bias"] == pytest.approx(logit(p))


def test_a_bigram_across_a_sentence_boundary_belongs_to_no_sentence():
    m = model()
    # "night the" straddles the full stop, in training (TEXTS[0]) and here
    text = "I cannot sleep at night. The weather is fine."
    assert "night the" in m["vectorizer"].vocabulary_
    _, parts = explain(m, text, split(text), "SLEEP_ISSUES")
    assert parts["crossing"] != 0.0
    # with no boundary to cross there is nothing left over
    _, parts = explain(m, "I cannot sleep at night.", split("I cannot sleep at night."), "SLEEP_ISSUES")
    assert parts["crossing"] == pytest.approx(0.0)


def test_the_sentence_carrying_the_symptom_ranks_first():
    m = model()
    text = "The weather is fine. I cannot sleep at night. Dinner was nice."
    scores, _ = explain(m, text, split(text), "SLEEP_ISSUES")
    assert max(scores, key=scores.get) == 1
    # (the neutral sentence must not come from a fatigue post in TEXTS, or the model rightly learns it)
    text = "The film was good. So tired and exhausted."
    scores, _ = explain(m, text, split(text), "FATIGUE")
    assert max(scores, key=scores.get) == 1


def test_i_is_a_feature():
    assert "i" in model()["vectorizer"].vocabulary_


def test_a_label_that_never_varies_predicts_a_constant_and_explains_nothing():
    labels = np.column_stack([LABELS, np.zeros(len(TEXTS), dtype=int)])
    m = fit(TEXTS, labels, CLASSES + ["PSYCHOMOTOR"], NEGATIVE, min_df=1)
    assert (predict_proba(m, ["I cannot sleep."])[:, 3] == 0.0).all()
    scores, parts = explain(m, "I cannot sleep.", split("I cannot sleep."), "PSYCHOMOTOR")
    assert scores == {0: 0.0} and parts == {"markers": 0.0, "crossing": 0.0, "bias": 0.0}


def test_a_post_with_no_finite_verb_still_gets_a_prediction():
    # present tense is nan here; it must become the training mean, not a crash or a nan probability
    p = predict_proba(model(), ["Tired. So tired."])
    assert np.isfinite(p).all()
