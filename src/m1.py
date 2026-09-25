"""M1: one logistic regression per label on TF-IDF n-grams plus the depression markers, explained exactly."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import binarizer, check_against_paper, load, split
from .lexicon import load_nrc, markers
from .train_bert import metrics

MARKERS = ["first_person", "negative_affect", "absolutist", "present_tense"]


def marker_matrix(texts, negative_words):
    return np.array([[markers(t, negative_words)[k] for k in MARKERS] for t in texts], dtype=float)


def _features(model, texts):
    """TF-IDF columns, then the four markers standardised on the training posts. A missing marker
    (no finite verb, say) becomes the training mean, which is 0 after standardising."""
    z = (marker_matrix(texts, model["negative_words"]) - model["mean"]) / model["std"]
    return hstack([model["vectorizer"].transform(texts), csr_matrix(np.nan_to_num(z))]).tocsr()


def fit(texts, y, classes, negative_words, C=1.0, min_df=2):
    """One binary logistic regression per label. A label that never varies in training gets None
    and a constant prediction, since there is nothing to learn from."""
    # sklearn drops one-letter tokens by default, which would drop "I", the most telling word here
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(texts)
    raw = marker_matrix(texts, negative_words)
    std = np.nanstd(raw, axis=0)
    model = {"vectorizer": vectorizer, "negative_words": negative_words, "classes": list(classes),
             "mean": np.nanmean(raw, axis=0), "std": np.where(std > 0, std, 1.0)}
    X = _features(model, texts)
    y = np.asarray(y)
    model["heads"] = [LogisticRegression(C=C, max_iter=2000).fit(X, y[:, j]) if len(set(y[:, j])) > 1
                      else float(y[0, j]) for j in range(y.shape[1])]
    return model


def predict_proba(model, texts):
    X = _features(model, texts)
    return np.column_stack([h.predict_proba(X)[:, 1] if isinstance(h, LogisticRegression)
                            else np.full(X.shape[0], h) for h in model["heads"]])


def explain(model, text, spans, label):
    """Each sentence's exact share of the label's logit, and the parts no sentence owns.

    The logit is the sum over features of weight * value, plus the bias. An n-gram's term is split
    across the sentences it occurs in by how often it occurs there. What is left is the markers
    (post-level rates), n-grams that cross a sentence boundary, and the bias. So
    sum(scores) + markers + crossing + bias == logit, exactly.
    """
    head = model["heads"][model["classes"].index(label)]
    if not isinstance(head, LogisticRegression):
        return {i: 0.0 for i in range(len(spans))}, {"markers": 0.0, "crossing": 0.0, "bias": 0.0}

    x = _features(model, [text])
    w = head.coef_[0]
    vocab = model["vectorizer"].vocabulary_
    n_text = len(vocab)
    term = {j: w[j] * v for j, v in zip(x.indices, x.data) if j < n_text}

    analyze = model["vectorizer"].build_analyzer()
    in_post = Counter(g for g in analyze(text) if g in vocab)
    scores = {}
    for i, (a, b) in enumerate(spans):
        in_sentence = Counter(g for g in analyze(text[a:b]) if g in vocab)
        scores[i] = float(sum(term[vocab[g]] * c / in_post[g] for g, c in in_sentence.items()))

    parts = {"markers": float(sum(w[j] * v for j, v in zip(x.indices, x.data) if j >= n_text)),
             "crossing": float(sum(term.values()) - sum(scores.values())),
             "bias": float(head.intercept_[0])}
    return scores, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--nrc", default="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
    ap.add_argument("--out", default="outputs/m1")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--allow-mismatch", action="store_true", help="for the public sample")
    args = ap.parse_args()

    posts, _ = load(args.data)
    if not args.allow_mismatch and not check_against_paper(posts)["matches_paper"]:
        raise SystemExit("data does not match the paper, refusing to train")
    train_df, test_df = split(posts)
    mlb = binarizer(posts)
    classes = list(mlb.classes_)
    model = fit(train_df["text"].tolist(), mlb.transform(train_df["labels"]), classes, load_nrc(args.nrc))
    probs = predict_proba(model, test_df["text"].tolist())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = metrics(mlb.transform(test_df["labels"]), probs, classes, args.threshold)
    m.update({"n_train": len(train_df), "n_test": len(test_df), "classes": classes})
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    pd.DataFrame(probs, columns=classes).assign(post_id=test_df["post_id"].values) \
        .to_csv(out / "test_probs.csv", index=False)
    print(f"M1 micro-F1 {m['micro_f1']:.3f}  macro-F1 {m['macro_f1']:.3f}  (written to {out})")


if __name__ == "__main__":
    main()
