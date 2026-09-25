"""Linguistic markers of depression, per post or per sentence: the features for M1 and the marker-density check."""
from pathlib import Path

import numpy as np
from nltk import pos_tag, word_tokenize

FIRST_PERSON = {"i", "me", "my", "mine", "myself"}

# Al-Mosaiwi and Johnstone (2018), the absolutist dictionary validated on mental-health forums.
ABSOLUTIST = {"absolutely", "all", "always", "complete", "completely", "constant", "constantly",
              "definitely", "entire", "ever", "every", "everyone", "everything", "full", "must",
              "never", "nothing", "totally", "whole"}

# Tense only shows on finite verbs, so present tense is a share of these, not of all words.
PRESENT, PAST = {"VBP", "VBZ"}, {"VBD"}

# LIWC's negative emotion is anxiety + anger + sadness; these are the matching NRC emotions.
NRC_NEGATIVE = ("fear", "anger", "sadness")


def load_nrc(path="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", categories=NRC_NEGATIVE):
    """Words the NRC Emotion Lexicon links to any of the given emotions. Licensed; never committed."""
    words = set()
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[1] in categories and parts[2] == "1":
            words.add(parts[0])
    return words


def markers(text, negative_words):
    """Rates per word for first person, negative affect and absolutist words, and the present-tense
    share of finite verbs. nan where there is nothing to divide by: no words, or no finite verb."""
    tagged = pos_tag(word_tokenize(text or ""))
    words = [w.lower() for w, _ in tagged if w.isalpha()]
    finite = [t for _, t in tagged if t in PRESENT | PAST]
    n = len(words)
    return {
        "first_person": sum(w in FIRST_PERSON for w in words) / n if n else np.nan,
        "negative_affect": sum(w in negative_words for w in words) / n if n else np.nan,
        "absolutist": sum(w in ABSOLUTIST for w in words) / n if n else np.nan,
        "present_tense": sum(t in PRESENT for t in finite) / len(finite) if finite else np.nan,
    }
