import math

import pytest

from src.metrics import (bootstrap_ci, cliffs_delta, comprehensiveness, cosine, did_ci, erasure, holm, mean_similarity,
                         jaccard, mann_whitney_p, paired_diff_ci, precision_at_k,
                         precision_recall_f1, rank_agreement, sentence_scores, sufficiency, top_k,
                         symptom_null, unpaired_diff_ci, wilcoxon_p)


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


def test_precision_at_k_on_a_post_shorter_than_k():
    # two sentences, both cited: k=3 can only ever return two, so a perfect answer is still 1.0
    assert precision_at_k({0: 0.9, 1: 0.8}, {0, 1}, 3) == 1.0
    assert precision_at_k({0: 0.9, 1: 0.8}, {0}, 3) == pytest.approx(0.5)
    assert precision_at_k({}, {0}, 2) == 0.0


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


def test_sentence_scores_default_is_the_signed_sum():
    # sentence 0 has 3 tokens (0.5, -1.0, 0.5), sentence 1 has 2 tokens (0.2, 0.2)
    tokens = [0.5, -1.0, 0.5, 0.2, 0.2]
    sents = [0, 0, 0, 1, 1]
    scores = sentence_scores(tokens, sents)
    assert scores == {0: pytest.approx(0.0), 1: pytest.approx(0.4)}
    assert sum(scores.values()) == pytest.approx(sum(tokens))      # completeness carries over


def test_sentence_scores_options_clip_and_normalise():
    tokens = [0.5, -1.0, 0.5, 0.2, 0.2]
    sents = [0, 0, 0, 1, 1]
    assert sentence_scores(tokens, sents, positive_only=True) == {0: pytest.approx(1.0), 1: pytest.approx(0.4)}
    assert sentence_scores(tokens, sents, normalise=True, positive_only=True) == \
        {0: pytest.approx(1 / 3), 1: pytest.approx(0.2)}


def test_sentence_scores_drops_special_tokens():
    # token_sentences marks [CLS] and [SEP] as -1; a large score there must not become a "sentence"
    scores = sentence_scores([0.9, 0.2, 0.1, 0.5, 0.7], [-1, 0, 0, 1, -1])      # sentence 0 is 0.3, sentence 1 is 0.5
    assert set(scores) == {0, 1}
    assert top_k(scores, 1) == {1}


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_ci([1, 2, 3, 4, 5], n_resamples=500, seed=0)
    assert mean == 3.0
    assert lo <= mean <= hi
    assert lo > 1.0 and hi < 5.0
    # constant data has a zero-width interval
    assert bootstrap_ci([2, 2, 2], seed=0) == (2.0, 2.0, 2.0)


def test_paired_diff_ci_on_a_constant_gap():
    # every item differs by exactly 0.2, so the mean gap is 0.2 and the interval collapses onto it
    a = [0.5, 0.6, 0.7, 0.8]
    b = [0.3, 0.4, 0.5, 0.6]
    mean, lo, hi = paired_diff_ci(a, b, n_resamples=200)
    assert mean == pytest.approx(0.2)
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(0.2)


def test_unpaired_diff_ci_on_two_constant_groups():
    mean, lo, hi = unpaired_diff_ci([0.8, 0.8, 0.8], [0.5, 0.5], n_resamples=200)
    assert mean == pytest.approx(0.3)
    assert lo == pytest.approx(0.3)
    assert hi == pytest.approx(0.3)


def test_did_ci_isolates_the_faithfulness_drop():
    # plausibility is 0.8 whether the prediction is right or wrong; faithfulness falls 0.7 -> 0.2.
    # H1's shape exactly: the plausibility gap is 0, the faithfulness gap is 0.5, so H1 wants +0.5.
    plaus_correct, plaus_wrong = [0.8, 0.8, 0.8], [0.8, 0.8]
    faith_correct, faith_wrong = [0.7, 0.7, 0.7], [0.2, 0.2]
    observed, lo, hi = did_ci(plaus_correct, faith_correct, plaus_wrong, faith_wrong, n_resamples=200)
    assert observed == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_did_ci_is_zero_when_both_metrics_drop_together():
    # both fall by 0.4, so nothing distinguishes plausibility from faithfulness: H1 unsupported
    observed, _, _ = did_ci([0.8, 0.8], [0.7, 0.7], [0.4, 0.4], [0.3, 0.3], n_resamples=200)
    assert observed == pytest.approx(0.0)


def test_bootstrap_over_posts_is_not_fooled_by_repeated_items():
    # each post contributes the same value twice; resampling posts must give exactly the interval of
    # one value per post, while resampling items treats the copies as independent and narrows it
    per_post = [0.2, 0.9, 0.4, 0.7, 0.5, 0.1, 0.8]
    twice = [v for v in per_post for _ in range(2)]
    posts = [p for p in range(len(per_post)) for _ in range(2)]
    assert bootstrap_ci(twice, posts=posts, seed=3) == pytest.approx(bootstrap_ci(per_post, seed=3))
    _, lo_items, hi_items = bootstrap_ci(twice, seed=3)
    _, lo_posts, hi_posts = bootstrap_ci(twice, posts=posts, seed=3)
    assert hi_items - lo_items < hi_posts - lo_posts


# Each post has one correct and one wrong item. The level varies a lot between posts, but within every
# post the wrong item is exactly 0.5 lower. Drawing a post brings both of its items, so every resample
# sees the same 0.5 gap and the interval collapses; drawing the groups independently cannot know that.
LEVELS = [0.9, 0.6, 0.8, 0.55, 0.95, 0.7]
POSTS = list(range(len(LEVELS)))


def test_unpaired_diff_draws_a_posts_items_together():
    right, wrong = LEVELS, [v - 0.5 for v in LEVELS]
    assert unpaired_diff_ci(right, wrong, posts_a=POSTS, posts_b=POSTS, n_resamples=200) == \
        pytest.approx((0.5, 0.5, 0.5))
    _, lo, hi = unpaired_diff_ci(right, wrong, n_resamples=200)
    assert hi - lo > 0.05


def test_did_ci_draws_a_posts_items_together():
    # plausibility flat everywhere, faithfulness 0.5 lower on each post's wrong item: H1 wants +0.5.
    # (If plausibility carried the same per-post levels they would cancel out of the DiD whatever was drawn.)
    plaus = [0.8] * len(LEVELS)
    faith_correct, faith_wrong = LEVELS, [v - 0.5 for v in LEVELS]
    assert did_ci(plaus, faith_correct, plaus, faith_wrong, posts_correct=POSTS, posts_wrong=POSTS,
                  n_resamples=200) == pytest.approx((0.5, 0.5, 0.5))
    _, lo, hi = did_ci(plaus, faith_correct, plaus, faith_wrong, n_resamples=200)
    assert hi - lo > 0.05


def test_wilcoxon_and_mann_whitney():
    assert math.isnan(wilcoxon_p([0.3, 0.4], [0.3, 0.4]))
    shifted = [x + 0.5 for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    assert wilcoxon_p(shifted, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) < 0.05
    assert mann_whitney_p([5, 6, 7, 8], [1, 2, 3, 4]) < 0.05


def test_cliffs_delta_spans_minus_one_to_one():
    assert cliffs_delta([1, 2, 3], [0]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [4]) == pytest.approx(-1.0)
    assert cliffs_delta([1, 2], [1, 2]) == pytest.approx(0.0)
    assert cliffs_delta([2, 4], [1, 5]) == pytest.approx(0.0)


def test_holm_steps_down_and_stays_monotone():
    # n=3: 3*0.01=0.03, then 2*0.03=0.06, then 1*0.04 is smaller so the running max holds at 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.5]) == pytest.approx([1.0, 1.0])


def test_mean_similarity():
    # identical to one reference, orthogonal to the other: mean cosine 0.5
    assert mean_similarity([1, 0], [[1, 0], [0, 1]]) == pytest.approx(0.5)
    assert mean_similarity([1, 0], [[2, 0], [3, 0]]) == pytest.approx(1.0)   # length does not matter
    assert math.isnan(mean_similarity([1, 0], []))


def test_symptom_null_pools_every_other_symptom():
    refs = {"A": [[1, 0]], "B": [[0, 1]], "C": [[0, 1], [1, 0]]}
    # for A, the others pooled are B's one vector and C's two: cosines 0, 0, 1
    assert symptom_null([1, 0], refs, "A") == pytest.approx(1 / 3)
    assert math.isnan(symptom_null([1, 0], {"A": [[1, 0]]}, "A"))


def test_erasure_with_any_predict_function():
    # a model that is sure exactly when the sleep sentence is present
    predict = lambda kept: 0.9 if "I cannot sleep." in kept else 0.2
    sentences = ["I cannot sleep.", "The weather is fine."]
    assert erasure(sentences, {0}, 0.9, predict) == (pytest.approx(0.7), pytest.approx(0.0))
    assert erasure(sentences, set(), 0.9, predict) == (None, None)
    assert erasure(sentences, {0, 1}, 0.9, predict)[0] is None        # nothing would be left
    assert erasure(sentences, {0}, 0.9, lambda kept: None) == (None, None)


def test_resample_posts_keeps_every_row_of_a_drawn_post():
    from src.metrics import resample_posts
    posts = ["a", "a", "b", "c"]
    for rows in resample_posts(posts, n_resamples=50, seed=0):
        drawn = [posts[r] for r in rows]
        assert drawn.count("a") % 2 == 0          # a's two rows always come together


def test_agreement_counts_two_empty_answers_as_full_agreement():
    from src.metrics import agreement
    assert agreement(set(), set()) == 1.0
    assert agreement({"A"}, {"A", "B"}) == pytest.approx(0.5)
    assert agreement({"A"}, set()) == 0.0
