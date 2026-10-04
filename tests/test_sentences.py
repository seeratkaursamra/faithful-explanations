import re

import pandas as pd

from src.sentences import align, alignment_report, split, texts, token_sentences, truncated


class WordTokenizer:
    """Stand-in for a HF tokenizer: one token per word, with a zero-width special token at each end."""

    def __call__(self, text, truncation=True, max_length=512, return_offsets_mapping=True):
        offsets = [(0, 0)] + [(m.start(), m.end()) for m in re.finditer(r"\S+", text)] + [(0, 0)]
        return {"offset_mapping": offsets[:max_length] if truncation else offsets}


def test_split_finds_three_sentences():
    text = "I cannot sleep. My appetite is gone! What do I do?"
    spans = split(text)
    assert texts(text, spans) == ["I cannot sleep.", "My appetite is gone!", "What do I do?"]


def test_spans_index_back_into_the_original_text():
    text = "One two. Three four."
    for a, b in split(text):
        assert text[a:b].strip() == text[a:b]
    assert split(text) == [(0, 8), (9, 20)]


def test_line_breaks_end_a_sentence():
    # Punkt alone keeps these three as one sentence, which would merge three symptom claims
    text = "I can't sleep anymore\nMy appetite is gone too\nI don't know what to do."
    assert texts(text, split(text)) == ["I can't sleep anymore",
                                        "My appetite is gone too",
                                        "I don't know what to do."]


def test_blank_lines_are_skipped():
    assert texts("First line.\n\n\nSecond line.", split("First line.\n\n\nSecond line.")) == \
        ["First line.", "Second line."]


def test_abbreviations_do_not_split():
    text = "Dr. Smith said it would pass. It did not."
    assert texts(text, split(text)) == ["Dr. Smith said it would pass.", "It did not."]


def test_token_sentences_maps_each_word_to_its_sentence():
    text = "aa bb. cc dd."
    spans = split(text)
    # special, aa, bb., cc, dd., special
    assert token_sentences(text, spans, WordTokenizer()) == [-1, 0, 0, 1, 1, -1]


def test_truncated_lists_sentences_the_model_never_sees():
    text = "aa bb. cc dd. ee ff."
    spans = split(text)
    # 4 tokens of room: the opening special plus aa, bb., cc
    assert truncated(text, spans, WordTokenizer(), max_len=4) == [2]
    assert truncated(text, spans, WordTokenizer(), max_len=512) == []


def test_align_exact_and_contains_and_none():
    text = "I cannot sleep. My appetite is gone."
    spans = split(text)
    assert align("I cannot sleep.", text, spans) == (0, "exact")
    assert align("My appetite is gone", text, spans) == (1, "contains")
    assert align("The weather is nice today in Edmonton", text, spans) == (-1, "none")


def test_align_falls_back_to_word_overlap():
    text = "I have not been sleeping well for weeks. Work is busy."
    spans = split(text)
    # a paraphrase: no exact or substring match, but most words are shared
    index, how = align("I have not been sleeping well for many weeks", text, spans)
    assert (index, how) == (0, "overlap")


def test_a_short_line_inside_the_gold_does_not_beat_the_real_sentence():
    # "why" sits inside the gold sentence, but sentence 1 is the gold one, differing only by apostrophes
    text = "why\nI dont know why I cant sleep anymore."
    spans = split(text)
    assert align("I don't know why I can't sleep anymore.", text, spans) == (1, "overlap")


def test_align_handles_a_missing_gold_sentence():
    text = "I cannot sleep."
    assert align(float("nan"), text, split(text)) == (-1, "none")
    assert align(None, text, split(text)) == (-1, "none")


def test_alignment_report_counts_every_row_once():
    posts = pd.DataFrame({"post_id": ["p1"], "text": ["I cannot sleep. My appetite is gone."]})
    annotations = pd.DataFrame({
        "post_id": ["p1", "p1", "p1"],
        "sentence_id": ["p1_0", "p1_1", "p1_9"],
        "sentence_text": ["I cannot sleep.", "My appetite is gone", "Nothing like this in the post"],
        "sentence_index": [0, 1, 9],
    })
    counts, failures = alignment_report(posts, annotations)
    assert counts["total"] == 3
    assert counts["exact"] == 1 and counts["contains"] == 1 and counts["none"] == 1
    assert counts["id_equals_index"] == 2
    assert failures == [("p1", "p1_9")]


def test_word_overlap_exactly_at_the_threshold_counts():
    # neither string contains the other, so only the overlap rule can match: {a, b, x} vs {a, b, y}
    # share 2 of 4 words, exactly the 0.5 threshold, which counts ("at least", not "more than")
    text = "a b y."
    assert align("a b x", text, split(text), threshold=0.5) == (0, "overlap")
    # {a, b, c, x} vs {a, b, y, z}: 2 of 6, under the threshold
    text = "a b y z."
    assert align("a b c x", text, split(text), threshold=0.5) == (-1, "none")


def test_an_overlap_exactly_at_the_threshold_still_beats_a_containment():
    # sentence 1 shares exactly half the words (2 of 4) with the gold; sentence 0 contains the gold but shares only
    # 3 of 11. At the threshold counts as clearing it, as in the final overlap test, so the better overlap wins
    text = "Most nights I lie awake until the birds start singing outside. I lie still."
    assert align("I lie awake", text, split(text)) == (1, "overlap")
