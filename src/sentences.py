"""Split posts into sentences, map tokens onto them, and match the psychologist's evidence to them."""
import re

from nltk.tokenize import sent_tokenize

from .metrics import jaccard


def split(text):
    """Sentence spans as (start, end) character offsets.

    Punkt does not treat a line break as a sentence end, but Reddit posts use them instead of full
    stops, so each line is split on its own and the offsets are shifted back into the whole post.
    """
    spans = []
    for offset, line in _lines(text or ""):
        pos = 0
        for sentence in sent_tokenize(line):
            i = line.find(sentence, pos)
            if i < 0:
                continue
            spans.append((offset + i, offset + i + len(sentence)))
            pos = i + len(sentence)
    return spans


def _lines(text):
    """Non-blank lines with the character offset each one starts at."""
    pos = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            yield pos + line.index(stripped), stripped
        pos += len(line) + 1


def texts(text, spans):
    return [text[a:b] for a, b in spans]


def token_sentences(text, spans, tokenizer, max_len=512):
    """The sentence index each token falls in, aligned with the tokeniser's output. -1 for special tokens."""
    enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    return [_locate(spans, start) if start != end else -1 for start, end in enc["offset_mapping"]]


def _locate(spans, char):
    for i, (a, b) in enumerate(spans):
        if a <= char < b:
            return i
    return -1


def truncated(text, spans, tokenizer, max_len=512):
    """Sentences the model never sees because the post is cut at max_len."""
    seen = set(token_sentences(text, spans, tokenizer, max_len))
    return [i for i in range(len(spans)) if i not in seen]


def align(gold_text, text, spans, threshold=0.5):
    """Match one gold evidence sentence to my split.

    Returns (index, how). 'exact' when the strings are the same, 'contains' when one holds the
    other, 'overlap' when word overlap clears the threshold, 'none' with index -1 when nothing does.

    A containment match loses to a sentence that overlaps the gold better and clears the threshold,
    so a short line like "why" that happens to sit inside the gold sentence cannot take its place.
    """
    gold = gold_text.strip() if isinstance(gold_text, str) else ""
    sentences = [s.strip() for s in texts(text, spans)]
    if not gold or not sentences:
        return -1, "none"
    for i, s in enumerate(sentences):
        if s == gold:
            return i, "exact"
    gold_words = _words(gold)
    overlap = [(jaccard(gold_words, _words(s)), i) for i, s in enumerate(sentences)]
    best, index = max(overlap)
    contained = [(o, i) for o, i in overlap if gold in sentences[i] or sentences[i] in gold]
    if contained:
        c_best, c_index = max(contained)
        if not (best >= threshold and best > c_best):
            return c_index, "contains"
    return (index, "overlap") if best >= threshold else (-1, "none")


def _words(s):
    return set(re.findall(r"[a-z']+", s.lower()))


def alignment_report(posts, annotations, threshold=0.5):
    """How often the psychologist's sentences land on one of mine, and whether sentence_id agrees.

    Returns (counts, failures). Run this before trusting any sentence-level metric: if 'none' is
    common, gold evidence cannot be located and every comparison built on it is unsound.
    """
    text_by_id = dict(zip(posts["post_id"], posts["text"]))
    spans_by_id = {}
    counts = {"exact": 0, "contains": 0, "overlap": 0, "none": 0, "id_equals_index": 0, "total": 0}
    failures = []
    for row in annotations.itertuples():
        text = text_by_id.get(row.post_id)
        if not isinstance(text, str):
            continue
        if row.post_id not in spans_by_id:
            spans_by_id[row.post_id] = split(text)
        index, how = align(row.sentence_text, text, spans_by_id[row.post_id], threshold)
        counts[how] += 1
        counts["total"] += 1
        if index >= 0 and index == getattr(row, "sentence_index", -1):
            counts["id_equals_index"] += 1
        if how == "none":
            failures.append((row.post_id, row.sentence_id))
    return counts, failures
