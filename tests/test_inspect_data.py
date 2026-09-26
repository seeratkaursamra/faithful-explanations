import pytest

from src.data import load
from src.inspect_data import describe


def test_describe_reports_sentences_and_the_marker_comparison():
    posts, ann = load("data", sample=True)
    stats = describe(posts, ann)
    assert stats["sentences_per_post"]["max"] >= stats["sentences_per_post"]["median"] >= 1
    for group in ("with_symptoms", "negatives"):
        assert 0 <= stats["markers"][group]["first_person"] <= 1
        assert 0 <= stats["markers"][group]["present_tense"] <= 1
