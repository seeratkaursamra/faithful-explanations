import numpy as np
import pytest

from src.lime_sentences import explain, masks
from src.metrics import top_k

SENTENCES = ["I slept badly.", "Work was busy.", "I have no appetite.", "It rained.", "I saw a film."]


def driven_by(index, base=0.1, lift=0.8):
    """A stand-in model whose probability rises only when one known sentence survives."""
    def predict(samples):
        return [base + lift * (SENTENCES[index] in kept) for kept in samples]
    return predict


def test_masks_keep_the_original_and_never_empty_the_post():
    m = masks(5, 50, np.random.default_rng(0))
    assert m[0].tolist() == [1, 1, 1, 1, 1]
    assert m.min() == 0 and m.max() == 1
    assert (m.sum(axis=1) >= 1).all()


def test_the_driving_sentence_gets_the_largest_weight():
    weights, r2 = explain(SENTENCES, driven_by(2), n_samples=200)
    assert max(weights, key=weights.get) == 2
    assert top_k(weights, 1) == {2}
    # the true lift is 0.8; ridge shrinks it, so expect most of it back and near-nothing elsewhere
    assert 0.4 < weights[2] < 0.8
    assert all(abs(weights[i]) < 0.1 for i in [0, 1, 3, 4])
    assert r2 > 0.95


def test_two_drivers_come_out_roughly_equal():
    def predict(samples):
        return [0.1 + 0.4 * (SENTENCES[0] in kept) + 0.4 * (SENTENCES[3] in kept) for kept in samples]

    weights, _ = explain(SENTENCES, predict, n_samples=300)
    assert top_k(weights, 2) == {0, 3}
    assert weights[0] == pytest.approx(weights[3], abs=0.05)


def test_a_sentence_that_lowers_the_output_gets_a_negative_weight():
    def predict(samples):
        return [0.9 - 0.6 * (SENTENCES[1] in kept) for kept in samples]

    weights, _ = explain(SENTENCES, predict, n_samples=200)
    assert weights[1] < -0.3
    assert min(weights, key=weights.get) == 1


def test_a_model_that_ignores_the_post_gets_flat_weights():
    weights, _ = explain(SENTENCES, lambda samples: [0.5] * len(samples), n_samples=100)
    assert all(abs(w) < 1e-6 for w in weights.values())


def test_predict_receives_exactly_the_surviving_sentences():
    seen = []

    def predict(samples):
        seen.extend(samples)
        return [0.5] * len(samples)

    explain(SENTENCES, predict, n_samples=20)
    assert seen[0] == SENTENCES
    for kept in seen:
        assert kept == [s for s in SENTENCES if s in kept]  # order preserved
        assert 1 <= len(kept) <= len(SENTENCES)


def test_same_seed_gives_the_same_answer():
    a, _ = explain(SENTENCES, driven_by(2), n_samples=100, seed=7)
    b, _ = explain(SENTENCES, driven_by(2), n_samples=100, seed=7)
    c, _ = explain(SENTENCES, driven_by(2), n_samples=100, seed=8)
    assert a == b
    assert a != c


def test_edge_cases():
    assert explain([], driven_by(0))[0] == {}
    assert explain(["only one"], lambda s: [0.5] * len(s))[0] == {0: 1.0}


def test_unscored_perturbations_are_dropped_not_zeroed():
    # every third reply fails; the fit should still find sentence 2, and a 0 would have dragged it
    base = driven_by(2)

    def flaky(samples):
        return [None if i % 3 == 1 else p for i, p in enumerate(base(samples))]

    weights, r2 = explain(SENTENCES, flaky, n_samples=200)
    assert max(weights, key=weights.get) == 2 and r2 > 0.95
    assert explain(SENTENCES, lambda s: [None] * len(s), n_samples=20) == ({}, pytest.approx(float("nan"), nan_ok=True))
