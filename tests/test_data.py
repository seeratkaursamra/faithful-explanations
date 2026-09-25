import math
import shutil
from pathlib import Path

import pandas as pd

from src.data import (NO_SYMPTOMS, SYMPTOMS, binarizer, check_against_paper, gold, gold_text, load, references,
                      split, symptom_counts)


def test_labels_are_status_1_symptoms_only():
    posts, ann = load("data", sample=True)
    for pid, labs in zip(posts["post_id"], posts["labels"]):
        rows = ann[(ann.post_id == pid) & (ann.status == 1) & ann.DSM5_symptom.isin(SYMPTOMS)]
        expected = sorted(set(rows.DSM5_symptom)) or [NO_SYMPTOMS]
        assert labs == expected, pid


def test_negatives_get_no_symptoms():
    posts, _ = load("data", sample=True)
    negs = posts[posts.is_negative]
    assert len(negs) > 0
    assert all(labs == [NO_SYMPTOMS] for labs in negs.labels)


def test_split_is_deterministic_and_80_20():
    posts, _ = load("data", sample=True)
    train_a, test_a = split(posts)
    train_b, test_b = split(posts)
    assert list(test_a.post_id) == list(test_b.post_id)
    assert len(test_a) == math.ceil(0.2 * len(posts))  # sklearn ceils, so 1484 -> 297
    assert not set(train_a.post_id) & set(test_a.post_id)


def test_binarizer_classes_are_known():
    posts, _ = load("data", sample=True)
    assert set(binarizer(posts).classes_) <= set(SYMPTOMS) | {NO_SYMPTOMS}


def test_gold_returns_only_status_1_for_that_symptom():
    posts, ann = load("data", sample=True)
    row = ann[ann.status == 1].iloc[0]
    g = gold(ann, row.post_id, row.DSM5_symptom)
    assert len(g) >= 1
    assert (g.status == 1).all() and (g.DSM5_symptom == row.DSM5_symptom).all()


def test_symptom_counts_ignore_no_symptoms():
    posts, _ = load("data", sample=True)
    assert NO_SYMPTOMS not in symptom_counts(posts)


def test_paper_check_reports_each_symptom():
    posts, _ = load("data", sample=True)
    report = check_against_paper(posts)
    assert set(report["symptom_counts"]) == set(SYMPTOMS)
    assert isinstance(report["matches_paper"], bool)


def test_sample_true_ignores_the_real_files(tmp_path):
    # a stand-in "real" dataset of one post next to the sample: sample=True must still read the sample
    for name in ["redsm5_posts_sample_paraphrased.csv", "redsm5_annotations_sample_paraphrased.csv"]:
        shutil.copy(Path("data") / name, tmp_path / name)
    (tmp_path / "redsm5_posts.csv").write_text("post_id,text\nfake_1,I cannot sleep.\n")
    (tmp_path / "redsm5_annotations.csv").write_text(
        "post_id,sentence_id,sentence_text,DSM5_symptom,status,explanation\n"
        "fake_1,fake_1_0,I cannot sleep.,SLEEP_ISSUES,1,says so\n")
    assert list(load(tmp_path)[0].post_id) == ["fake_1"]
    assert list(load(tmp_path, sample=True)[0].post_id) == list(load("data", sample=True)[0].post_id)


def _annotations(rows):
    return pd.DataFrame(rows, columns=["post_id", "sentence_index", "DSM5_symptom", "status", "explanation"])


def test_gold_text_joins_evidence_in_sentence_order():
    ann = _annotations([
        ("p1", 4, "SLEEP_ISSUES", 1, "Second."),
        ("p1", 1, "SLEEP_ISSUES", 1, "First."),
        ("p1", 2, "SLEEP_ISSUES", 0, "Rejected, so left out."),
        ("p1", 3, "FATIGUE", 1, "Other symptom, left out."),
    ])
    assert gold_text(ann, "p1", "SLEEP_ISSUES") == "First. Second."
    assert gold_text(ann, "p1", "ANHEDONIA") == ""


def test_references_only_use_the_posts_given():
    ann = _annotations([
        ("train1", 0, "FATIGUE", 1, "Tired."),
        ("train2", 0, "FATIGUE", 1, "Exhausted."),
        ("train2", 1, "SLEEP_ISSUES", 1, "No sleep."),
        ("test1", 0, "FATIGUE", 1, "Test post, must not appear."),
        ("train1", 1, "ANHEDONIA", 0, "Rejected, must not appear."),
    ])
    assert references(ann, ["train1", "train2"]) == {"FATIGUE": ["Tired.", "Exhausted."],
                                                     "SLEEP_ISSUES": ["No sleep."]}
