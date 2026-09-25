"""Explanation metrics per (post, symptom), and the tests that aggregate them across posts."""
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

# Below this many test examples a per-symptom number is reported but not argued from.
MIN_SUPPORT = 10


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


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
    """Share of the top-k attributed sentences that the model (or the expert) actually cited."""
    if k == 0:
        return 0.0
    return len(top_k(scores, k) & set(relevant)) / k


def comprehensiveness(p_full, p_without_rationale):
    """How much the prediction drops when the rationale sentences are removed. High = they mattered."""
    return p_full - p_without_rationale


def sufficiency(p_full, p_rationale_only):
    """How much the prediction drops when only the rationale sentences are kept. Low = they suffice."""
    return p_full - p_rationale_only


def cosine(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / denom) if denom else 0.0


def rank_agreement(scores_a, scores_b):
    """Spearman correlation between two sentence-score dicts over their shared sentence ids."""
    ids = sorted(set(scores_a) & set(scores_b))
    if len(ids) < 2:
        return float("nan")
    rho, _ = spearmanr([scores_a[i] for i in ids], [scores_b[i] for i in ids])
    return float(rho)


def sentence_scores(token_scores, token_sentence, normalise=True):
    """Aggregate token attributions to sentences: sum of positive scores, divided by length if normalise."""
    total, length = {}, {}
    for score, sid in zip(token_scores, token_sentence):
        total[sid] = total.get(sid, 0.0) + max(score, 0.0)
        length[sid] = length.get(sid, 0) + 1
    return {sid: total[sid] / length[sid] if normalise else total[sid] for sid in total}


def bootstrap_ci(values, n_resamples=1000, seed=0):
    """Mean with a 95% percentile bootstrap interval over posts."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)]
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(a, b, n_resamples=1000, seed=0):
    """Mean of a - b over the same items, with a bootstrap interval. Use when a and b are aligned."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d, n_resamples, seed)


def unpaired_diff_ci(a, b, n_resamples=1000, seed=0):
    """mean(a) - mean(b) for two disjoint groups, with a bootstrap interval. Each group resampled alone."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
             for _ in range(n_resamples)]
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def did_ci(plaus_correct, faith_correct, plaus_wrong, faith_wrong, n_resamples=1000, seed=0):
    """H1 as one number: the faithfulness correct-wrong gap minus the plausibility correct-wrong gap.

    Both metrics come from the same items, so a resample draws item positions once and reads
    plausibility and faithfulness from the same draw.
    """
    plaus_correct, faith_correct = np.asarray(plaus_correct, float), np.asarray(faith_correct, float)
    plaus_wrong, faith_wrong = np.asarray(plaus_wrong, float), np.asarray(faith_wrong, float)

    def did(i, j):
        return ((faith_correct[i].mean() - faith_wrong[j].mean())
                - (plaus_correct[i].mean() - plaus_wrong[j].mean()))

    rng = np.random.default_rng(seed)
    draws = [did(rng.integers(0, len(plaus_correct), len(plaus_correct)),
                 rng.integers(0, len(plaus_wrong), len(plaus_wrong))) for _ in range(n_resamples)]
    observed = did(np.arange(len(plaus_correct)), np.arange(len(plaus_wrong)))
    return float(observed), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


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
    n = len(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return [float(x) for x in adjusted]
