"""The authors' SVM baseline (examples/svm.ipynb), copied exactly: a second reproduction check.

It is deterministic, so on the authors' split it should give their 0.39 micro / 0.28 macro F1 exactly. If
BERT misses 0.51 but this lands, the BERT run is at fault; if this misses too, the data or split differ.
"""
import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from .data import binarizer, check_split, load, split
from .train_bert import metrics

PAPER = {"micro_f1": 0.39, "macro_f1": 0.28}


def run(posts):
    train, test = split(posts)
    mlb = binarizer(posts)
    classes = list(mlb.classes_)
    # the notebook's settings, word for word; max_features is effectively unlimited
    vectorizer = TfidfVectorizer(max_features=10000000, ngram_range=(1, 1), stop_words="english",
                                 sublinear_tf=True, use_idf=True)
    model = OneVsRestClassifier(LinearSVC(random_state=0, tol=1e-5))
    model.fit(vectorizer.fit_transform(train["text"]), mlb.transform(train["labels"]))
    predictions = model.predict(vectorizer.transform(test["text"]))
    # metrics() thresholds probabilities at 0.5; the SVM's 0/1 predictions pass through unchanged
    m = metrics(mlb.transform(test["labels"]), predictions.astype(float), classes, 0.5)
    # the notebook prints two decimals, so that is the precision a match can be judged at
    m["matches_paper"] = all(round(m[k], 2) == v for k, v in PAPER.items())
    m["paper"] = PAPER
    m["split"] = check_split(test)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs/svm")
    args = ap.parse_args()
    m = run(load(args.data)[0])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"SVM micro-F1 {m['micro_f1']:.3f} (paper 0.39)  macro-F1 {m['macro_f1']:.3f} (paper 0.28)  "
          f"-> matches paper: {m['matches_paper']}; test split matches the authors': {m['split']['matches_authors']}")


if __name__ == "__main__":
    main()
