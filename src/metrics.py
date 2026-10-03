"""Explanation metrics per (post, symptom), and the tests that aggregate them across posts."""
import numpy as np
from scipy.stats import mannwhitneyu, rankdata, spearmanr, wilcoxon

# Below this many test examples a per-symptom number is reported but not argued from.
MIN_SUPPORT = 10


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def agreement(a, b):
    """Jaccard for comparing two runs, except that both saying "nothing" is full agreement."""
    return 1.0 if not set(a) and not set(b) else jaccard(a, b)


def precision_recall_f1(retrieved, relevant):
    """Treat the gold set as relevant and the model's cited set as retrieved."""
    retrieved, relevant = set(retrieved), set(relevant)
    hits = len(retrieved & relevant)
    p = hits / len(retrieved) if retrieved else 0.0
    r = hits / len(relevant) if relevant else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def top_k(scores, k):
    """The k highest-scoring sentence ids. Ties break by sentence id so the result is deterministic."""
    ranked = sorted(scores, key=lambda s: (-scores[s], s))
    return set(ranked[:k])


def precision_at_k(scores, relevant, k):
    """Share of the top-k attributed sentences that the model (or the expert) actually cited.

    A post shorter than k can only return that many, so divide by what was returned, not by k.
    """
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    return len(top_k(scores, k) & set(relevant)) / k


def comprehensiveness(p_full, p_without_rationale):
    """How much the prediction drops when the rationale sentences are removed. High = they mattered."""
    return p_full - p_without_rationale


def sufficiency(p_full, p_rationale_only):
    """How much the prediction drops when only the rationale sentences are kept. Low = they suffice."""
    return p_full - p_rationale_only


def erasure(sentences, evidence, p_full, predict):
    """(comprehensiveness, sufficiency) for one label. `predict` takes a list of sentences and returns
    the model's probability for the label, or None when it cannot say (an LLM reply that failed).

    Undefined (None) when nothing was cited, since removing nothing tests nothing, and
    comprehensiveness is undefined when the evidence is the whole post, since nothing would be left.
    """
    if p_full is None or not evidence:
        return None, None
    without = [s for i, s in enumerate(sentences) if i not in evidence]
    only = [s for i, s in enumerate(sentences) if i in evidence]
    p_without = predict(without) if without else None
    p_only = predict(only)
    return (None if p_without is None else comprehensiveness(p_full, p_without),
            None if p_only is None else sufficiency(p_full, p_only))


def cosine(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / denom) if denom else 0.0


def mean_similarity(vector, references):
    """Mean cosine between one vector and a set of vectors. Plausibility, when the references are
    the same symptom's gold rationales from the training posts."""
    if not len(references):
        return float("nan")
    return float(np.mean([cosine(vector, r) for r in references]))


def symptom_null(vector, references_by_symptom, symptom):
    """The same mean similarity against every other symptom's references, pooled. A rationale that
    is no closer to its own symptom than to this is reading generic clinical language."""
    others = [r for s, refs in references_by_symptom.items() if s != symptom for r in refs]
    return mean_similarity(vector, others)


def rank_agreement(scores_a, scores_b):
    """Spearman correlation between two sentence-score dicts over their shared sentence ids."""
    ids = sorted(set(scores_a) & set(scores_b))
    if len(ids) < 2:
        return float("nan")
    rho, _ = spearmanr([scores_a[i] for i in ids], [scores_b[i] for i in ids])
    return float(rho)


def sentence_scores(token_scores, token_sentence, normalise=False, positive_only=False):
    """Aggregate token attributions to sentences: the plain signed sum by default.

    Signed and unnormalised, the sentence scores add up to what the token scores add up to, so IG's
    completeness carries over to sentences, and a sentence is scored like LIME scores it: by its
    whole effect, not its average token. Special tokens carry sentence id -1 and are dropped.
    """
    total, length = {}, {}
    for score, sid in zip(token_scores, token_sentence):
        if sid < 0:
            continue
        total[sid] = total.get(sid, 0.0) + (max(score, 0.0) if positive_only else score)
        length[sid] = length.get(sid, 0) + 1
    return {sid: total[sid] / length[sid] if normalise else total[sid] for sid in total}


def _post_draws(groups, n_resamples, seed):
    """Item positions for each resample: draw posts with replacement and keep every item they have.

    groups holds one list of post ids per group, aligned with that group's values. A post with
    items in two groups (a TP for one symptom, an FP for another) brings both along together.
    With no post ids, each item is its own post and this is the ordinary bootstrap.
    """
    everything = list(dict.fromkeys(p for posts in groups for p in posts))
    where = []
    for posts in groups:
        items = {}
        for i, p in enumerate(posts):
            items.setdefault(p, []).append(i)
        where.append(items)
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        drawn = [everything[d] for d in rng.integers(0, len(everything), len(everything))]
        yield [np.array([i for p in drawn for i in items.get(p, [])], dtype=int) for items in where]


def resample_posts(posts, n_resamples=1000, seed=0):
    """Row positions for each resample of posts: every row of each drawn post, for statistics that
    are not a mean, such as a correlation computed from per-symptom aggregates."""
    for (rows,) in _post_draws([list(posts)], n_resamples, seed):
        yield rows


def _ids(posts, n, group):
    return [(group, i) for i in range(n)] if posts is None else list(posts)


def _interval(observed, draws):
    """Percentile interval, skipping resamples where a group came out empty."""
    draws = [d for d in draws if not np.isnan(d)]
    return float(observed), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _mean(values, idx):
    return values[idx].mean() if len(idx) else np.nan


def bootstrap_ci(values, posts=None, n_resamples=1000, seed=0):
    """Mean with a 95% percentile bootstrap interval over posts. posts gives each value's post id."""
    values = np.asarray(values, dtype=float)
    groups = [_ids(posts, len(values), "a")]
    return _interval(values.mean(), [_mean(values, i) for (i,) in _post_draws(groups, n_resamples, seed)])


def paired_diff_ci(a, b, posts=None, n_resamples=1000, seed=0):
    """Mean of a - b over the same items, with a bootstrap interval. Use when a and b are aligned."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d, posts, n_resamples, seed)


def unpaired_diff_ci(a, b, posts_a=None, posts_b=None, n_resamples=1000, seed=0):
    """mean(a) - mean(b) for two disjoint groups of items, with a bootstrap interval over posts."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    groups = [_ids(posts_a, len(a), "a"), _ids(posts_b, len(b), "b")]
    return _interval(a.mean() - b.mean(),
                     [_mean(a, i) - _mean(b, j) for i, j in _post_draws(groups, n_resamples, seed)])


def did_ci(plaus_correct, faith_correct, plaus_wrong, faith_wrong, posts_correct=None, posts_wrong=None,
           n_resamples=1000, seed=0):
    """H1 as one number: the faithfulness correct-wrong gap minus the plausibility correct-wrong gap.

    Plausibility and faithfulness come from the same items, so one draw of posts is read for both.
    """
    plaus_correct, faith_correct = np.asarray(plaus_correct, float), np.asarray(faith_correct, float)
    plaus_wrong, faith_wrong = np.asarray(plaus_wrong, float), np.asarray(faith_wrong, float)

    def did(i, j):
        return ((_mean(faith_correct, i) - _mean(faith_wrong, j))
                - (_mean(plaus_correct, i) - _mean(plaus_wrong, j)))

    groups = [_ids(posts_correct, len(plaus_correct), "c"), _ids(posts_wrong, len(plaus_wrong), "w")]
    observed = did(np.arange(len(plaus_correct)), np.arange(len(plaus_wrong)))
    return _interval(observed, [did(i, j) for i, j in _post_draws(groups, n_resamples, seed)])


def delta_ci(a, b, posts_a=None, posts_b=None, n_resamples=1000, seed=0):
    """Cliff's delta of a over b with a 95% interval over posts. Rank-based, so it means the same on a
    cosine that barely moves and on a 0-1 probability drop."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    groups = [_ids(posts_a, len(a), "a"), _ids(posts_b, len(b), "b")]
    draws = [cliffs_delta(a[i], b[j]) for i, j in _post_draws(groups, n_resamples, seed)]
    return _interval(cliffs_delta(a, b), draws)


def delta_did_ci(plaus_correct, faith_correct, plaus_wrong, faith_wrong, posts_correct=None, posts_wrong=None,
                 n_resamples=1000, seed=0):
    """H1 on one scale: Cliff's delta of faithfulness, correct over wrong, minus that of plausibility.
    Both deltas come from the same draw of posts. Subtracting raw gaps would set a cosine against a
    probability; subtracting deltas compares how far each measure separates right from wrong."""
    plaus_correct, faith_correct = np.asarray(plaus_correct, float), np.asarray(faith_correct, float)
    plaus_wrong, faith_wrong = np.asarray(plaus_wrong, float), np.asarray(faith_wrong, float)

    def did(i, j):
        return cliffs_delta(faith_correct[i], faith_wrong[j]) - cliffs_delta(plaus_correct[i], plaus_wrong[j])

    groups = [_ids(posts_correct, len(plaus_correct), "c"), _ids(posts_wrong, len(plaus_wrong), "w")]
    observed = did(np.arange(len(plaus_correct)), np.arange(len(plaus_wrong)))
    return _interval(observed, [did(i, j) for i, j in _post_draws(groups, n_resamples, seed)])


def paired_rank_biserial(a, b):
    """Effect size for paired items, the companion of the Wilcoxon test: rank the non-zero differences
    by size, then (sum of ranks where a > b - sum where a < b) / all ranks, in [-1, 1]."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def median_ci(values, n_resamples=1000, seed=0):
    """Median with a 95% percentile bootstrap interval. Pass one value per post, so posts are resampled."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [np.median(rng.choice(values, len(values), replace=True)) for _ in range(n_resamples)]
    return _interval(np.median(values), draws)


def wilcoxon_p(a, b):
    """Two-sided signed-rank p-value for paired items. nan when every difference is zero."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.all(a == b):
        return float("nan")
    return float(wilcoxon(a, b).pvalue)


def mann_whitney_p(a, b):
    """Two-sided rank-sum p-value for two disjoint groups."""
    return float(mannwhitneyu(np.asarray(a, dtype=float), np.asarray(b, dtype=float)).pvalue)


def cliffs_delta(a, b):
    """Effect size for two disjoint groups: P(a > b) - P(a < b), in [-1, 1]."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not len(a) or not len(b):
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / (len(a) * len(b)))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, returned in the order given. For the nine symptoms."""
    p = np.asarray(pvalues, dtype=float)
    # max(running, nan) keeps running, so a NaN would silently become a number; refuse it instead
    if np.isnan(p).any():
        raise ValueError("holm got a NaN p-value; drop untested symptoms before correcting")
    n = len(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return [float(x) for x in adjusted]
