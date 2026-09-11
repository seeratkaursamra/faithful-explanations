import math

from src.data import NO_SYMPTOMS, SYMPTOMS, binarizer, check_against_paper, gold, load, split, symptom_counts


def test_labels_are_status_1_symptoms_only():
    posts, ann = load("data")
    for pid, labs in zip(posts["post_id"], posts["labels"]):
        rows = ann[(ann.post_id == pid) & (ann.status == 1) & ann.DSM5_symptom.isin(SYMPTOMS)]
        expected = sorted(set(rows.DSM5_symptom)) or [NO_SYMPTOMS]
        assert labs == expected, pid


def test_negatives_get_no_symptoms():
    posts, _ = load("data")
    negs = posts[posts.is_negative]
    assert len(negs) > 0
    assert all(labs == [NO_SYMPTOMS] for labs in negs.labels)


def test_split_is_deterministic_and_80_20():
    posts, _ = load("data")
    train_a, test_a = split(posts)
    train_b, test_b = split(posts)
    assert list(test_a.post_id) == list(test_b.post_id)
    assert len(test_a) == math.ceil(0.2 * len(posts))  # sklearn ceils, so 1484 -> 297
    assert not set(train_a.post_id) & set(test_a.post_id)


def test_binarizer_classes_are_known():
    posts, _ = load("data")
    assert set(binarizer(posts).classes_) <= set(SYMPTOMS) | {NO_SYMPTOMS}


def test_gold_returns_only_status_1_for_that_symptom():
    posts, ann = load("data")
    row = ann[ann.status == 1].iloc[0]
    g = gold(ann, row.post_id, row.DSM5_symptom)
    assert len(g) >= 1
    assert (g.status == 1).all() and (g.DSM5_symptom == row.DSM5_symptom).all()


def test_symptom_counts_ignore_no_symptoms():
    posts, _ = load("data")
    assert NO_SYMPTOMS not in symptom_counts(posts)


def test_paper_check_reports_each_symptom():
    posts, _ = load("data")
    report = check_against_paper(posts)
    assert set(report["symptom_counts"]) == set(SYMPTOMS)
    assert isinstance(report["matches_paper"], bool)
