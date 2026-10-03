"""LIME over sentences: drop random subsets, then fit a weighted linear model to what the output did."""
import numpy as np
from sklearn.linear_model import Ridge


def masks(n_sentences, n_samples, rng):
    """Keep/drop matrix. Row 0 is the untouched post; every other row drops a random subset.

    At least one sentence always survives, so the model is never asked to score an empty post.
    """
    out = np.ones((n_samples, n_sentences), dtype=int)
    for row in range(1, n_samples):
        n_drop = int(rng.integers(1, n_sentences))
        out[row, rng.choice(n_sentences, n_drop, replace=False)] = 0
    return out


def kernel(dropped, width):
    """How much a perturbed post counts in the fit: posts that lost more of their sentences sit further
    from the original and count for less. exp(-d^2 / width^2), d the fraction of sentences dropped."""
    return np.exp(-(np.asarray(dropped, dtype=float) ** 2) / width ** 2)


def explain(sentences, predict, n_samples=200, width=0.25, alpha=1.0, seed=0):
    """Attribution weight per sentence, plus the local model's weighted R-squared.

    `predict` takes a list of perturbed posts, each one the list of sentences that survived, and
    returns one probability per post. Keeping it that general is why the same function explains
    BERT and the LLM: neither the perturbation nor the fit depends on what produced the number.

    The R-squared says how well a linear model describes the neighbourhood. Low values mean the
    weights are not a trustworthy summary and should be reported, not quietly used.
    """
    n = len(sentences)
    if n == 0:
        return {}, float("nan")
    if n == 1:
        return {0: 1.0}, float("nan")

    rng = np.random.default_rng(seed)
    kept = masks(n, n_samples, rng)
    samples = [[s for s, keep in zip(sentences, row) if keep] for row in kept]
    # an LLM reply that failed to parse comes back as None; that perturbation is dropped, not scored 0
    probs = np.array([np.nan if p is None else p for p in predict(samples)], dtype=float)
    scored = np.isfinite(probs)
    if scored.sum() < 2:
        return {}, float("nan")
    kept, probs = kept[scored], probs[scored]

    weights = kernel(1 - kept.mean(axis=1), width)

    local = Ridge(alpha=alpha).fit(kept, probs, sample_weight=weights)
    return ({i: float(c) for i, c in enumerate(local.coef_)},
            float(local.score(kept, probs, sample_weight=weights)))
