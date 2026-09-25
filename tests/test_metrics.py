import math

import pytest

from src.metrics import (bootstrap_ci, comprehensiveness, cosine, jaccard, precision_at_k,
                         precision_recall_f1, rank_agreement, sentence_scores, sufficiency, top_k)


def test_jaccard():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)
    assert jaccard({1}, {1}) == 1.0
    assert jaccard({1}, {2}) == 0.0
    assert jaccard(set(), {1}) == 0.0
    assert jaccard(set(), set()) == 0.0


def test_precision_recall_f1():
    # gold is {1, 2, 3}; model cited {2, 3, 4, 5}: 2 hits out of 4 cited, 2 of 3 gold found
    p, r, f1 = precision_recall_f1({2, 3, 4, 5}, {1, 2, 3})
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert precision_recall_f1(set(), {1}) == (0.0, 0.0, 0.0)


def test_top_k_breaks_ties_by_id():
    scores = {"s0": 0.1, "s1": 0.9, "s2": 0.9, "s3": 0.5}
    assert top_k(scores, 1) == {"s1"}
    assert top_k(scores, 2) == {"s1", "s2"}
    assert top_k(scores, 3) == {"s1", "s2", "s3"}
    assert top_k(scores, 10) == set(scores)


def test_precision_at_k():
    scores = {"s0": 0.1, "s1": 0.9, "s2": 0.7, "s3": 0.5}
    assert precision_at_k(scores, {"s1", "s3"}, 2) == pytest.approx(0.5)   # top-2 is s1, s2
    assert precision_at_k(scores, {"s1", "s2"}, 2) == 1.0
    assert precision_at_k(scores, {"s0"}, 2) == 0.0
    assert precision_at_k(scores, {"s1"}, 0) == 0.0


def test_comprehensiveness_and_sufficiency_signs():
    # rationale mattered: removing it drops 0.9 -> 0.3
    assert comprehensiveness(0.9, 0.3) == pytest.approx(0.6)
    # rationale alone almost reproduces the prediction: 0.9 -> 0.85
    assert sufficiency(0.9, 0.85) == pytest.approx(0.05)
    # rationale irrelevant: removing it changes nothing, keeping only it kills the prediction
    assert comprehensiveness(0.9, 0.9) == 0.0
    assert sufficiency(0.9, 0.1) == pytest.approx(0.8)


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine([0, 0], [1, 0]) == 0.0


def test_rank_agreement():
    a = {"s0": 0.1, "s1": 0.5, "s2": 0.9}
    same = {"s0": 1, "s1": 2, "s2": 3}
    reversed_ = {"s0": 3, "s1": 2, "s2": 1}
    assert rank_agreement(a, same) == pytest.approx(1.0)
    assert rank_agreement(a, reversed_) == pytest.approx(-1.0)
    assert math.isnan(rank_agreement(a, {"s0": 1}))
    # only shared ids count
    assert rank_agreement(a, {**same, "s9": 100}) == pytest.approx(1.0)


def test_sentence_scores_sums_positives_and_normalises():
    # sentence 0 has 3 tokens (0.5, -1.0, 0.5), sentence 1 has 2 tokens (0.2, 0.2)
    tokens = [0.5, -1.0, 0.5, 0.2, 0.2]
    sents = [0, 0, 0, 1, 1]
    assert sentence_scores(tokens, sents, normalise=False) == {0: pytest.approx(1.0), 1: pytest.approx(0.4)}
    assert sentence_scores(tokens, sents, normalise=True) == {0: pytest.approx(1 / 3), 1: pytest.approx(0.2)}


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_ci([1, 2, 3, 4, 5], n_resamples=500, seed=0)
    assert mean == 3.0
    assert lo <= mean <= hi
    assert lo > 1.0 and hi < 5.0
    # constant data has a zero-width interval
    assert bootstrap_ci([2, 2, 2], seed=0) == (2.0, 2.0, 2.0)
