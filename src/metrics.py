"""Per-(post, symptom) explanation metrics. Evidence sets are sets of sentence ids."""
import numpy as np
from scipy.stats import spearmanr


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
