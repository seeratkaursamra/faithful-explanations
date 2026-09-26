import math

import pytest

from src.lexicon import load_nrc, markers

NEGATIVE = {"hopeless", "afraid", "angry"}


def test_rates_count_words_not_punctuation():
    # 8 words: I, feel, hopeless, and, I, always, hate, myself -> 3 first person, 1 negative, 1 absolutist
    m = markers("I feel hopeless, and I always hate myself.", NEGATIVE)
    assert m["first_person"] == pytest.approx(3 / 8)
    assert m["negative_affect"] == pytest.approx(1 / 8)
    assert m["absolutist"] == pytest.approx(1 / 8)


def test_first_person_is_case_blind_and_catches_contractions():
    # "I'm" tokenises to "I" + "'m"; only alphabetic tokens count, so I, tired, ME, too: 2 of 4
    assert markers("I'm tired. ME too", set())["first_person"] == pytest.approx(2 / 4)


def test_present_tense_is_a_share_of_finite_verbs():
    # feel (present), felt (past): one of two finite verbs is present
    assert markers("Now I feel empty but last year I felt fine.", set())["present_tense"] == pytest.approx(0.5)
    assert markers("I feel empty and she is tired.", set())["present_tense"] == 1.0
    assert markers("I felt empty.", set())["present_tense"] == 0.0


def test_nothing_to_divide_by_gives_nan():
    assert all(math.isnan(v) for v in markers("", NEGATIVE).values())
    assert math.isnan(markers("Tired.", NEGATIVE)["present_tense"])     # no finite verb


def test_load_nrc_keeps_only_the_chosen_emotions(tmp_path):
    path = tmp_path / "nrc.txt"
    path.write_text("\n".join([
        "afraid\tfear\t1", "afraid\tjoy\t0",
        "sunny\tjoy\t1", "sunny\tsadness\t0",
        "grief\tsadness\t1",
        "awful\tnegative\t1",           # sentiment polarity, not an emotion: excluded by default
        "badline",
    ]))
    assert load_nrc(path) == {"afraid", "grief"}
    assert load_nrc(path, categories=("negative",)) == {"awful"}


def test_load_nrc_refuses_a_file_that_yields_no_words(tmp_path):
    # the VAD lexicon's shape: word and three scores, no emotion column
    vad = tmp_path / "vad.txt"
    vad.write_text("word\tvalence\tarousal\tdominance\nafraid\t0.1\t0.8\t0.2\n")
    with pytest.raises(ValueError, match="word-level EmoLex"):
        load_nrc(vad)


def test_marker_words_counts_a_word_once_even_if_it_is_two_markers():
    # I, always, hate, myself: first person, absolutist, negative, first person -> all 4 of 4
    assert markers("I always hate myself.", {"hate"})["marker_words"] == 1.0
    # "never" is absolutist and, here, also negative: one word, counted once -> 1 of 3
    assert markers("Things never change.", {"never"})["marker_words"] == pytest.approx(1 / 3)
