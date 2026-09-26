import pytest
import torch

from src.attribution import explain as integrated_gradients
from src.run_m2 import bert_predict, decoupled, post_records, remembered
from src.sentences import split, texts
from tests.test_attribution import tiny_bert, tokenizer

CLASSES = ["FATIGUE", "NO_SYMPTOMS", "SLEEP_ISSUES"]
TEXT = "I cannot sleep.\nMy appetite is gone. Tired."
LIME = {"n_samples": 20, "seed": 0}


def pieces(tmp_path):
    tok, model = tokenizer(tmp_path), tiny_bert(len(CLASSES))
    predict = bert_predict(model, tok, torch.device("cpu"))
    ig = lambda text, spans, j: integrated_gradients(model, tok, text, spans, j, steps=5)
    return tok, predict, ig


def explains_everything(calls, evidence=(0,)):
    """A stand-in decoupled control: cites the given sentences for every label it is handed."""
    def explain_why(sentences, labels):
        calls.append(list(labels))
        return {label: (list(evidence), f"why {label}") for label in labels}, {"ok": True}
    return explain_why


def test_joined_sentences_give_bert_the_same_tokens_as_the_post(tmp_path):
    tok = tokenizer(tmp_path)
    joined = " ".join(texts(TEXT, split(TEXT)))
    assert tok(joined)["input_ids"] == tok(TEXT)["input_ids"]


def test_remembered_scores_each_distinct_post_once():
    calls = []

    def predict(posts):
        calls.append(len(posts))
        return [[float(len(p))] for p in posts]

    cached = remembered(predict)
    assert cached([["a"], ["a", "b"], ["a"]]).tolist() == [[1.0], [2.0], [1.0]]
    assert cached([["a", "b"], ["c"]]).tolist() == [[2.0], [1.0]]
    assert calls == [2, 1]                 # ["a"] and ["a","b"] once, then only ["c"] is new


def test_every_predicted_label_gets_ig_lime_a_rationale_and_erasure(tmp_path):
    tok, predict, ig = pieces(tmp_path)
    calls = []
    # threshold 0: a sigmoid is always above it, so BERT "predicts" every label
    records, _ = post_records("p1", TEXT, ["SLEEP_ISSUES"], CLASSES, predict, ig, explains_everything(calls),
                              LIME, threshold=0.0)
    assert calls == [["FATIGUE", "SLEEP_ISSUES"]]                     # NO_SYMPTOMS is never handed over
    by = {r["symptom"]: r for r in records}
    assert set(by) == {"FATIGUE", "SLEEP_ISSUES"}
    sleep = by["SLEEP_ISSUES"]
    assert (sleep["predicted"], sleep["gold"], by["FATIGUE"]["gold"]) == (True, True, False)
    assert set(sleep["attribution"]["IG"]) == set(sleep["attribution"]["LIME"]) == {"0", "1", "2"}
    assert "ig_delta" in sleep and "lime_r2" in sleep
    # LIME must explain this label's column, not another's: rerun it directly with the same seed
    from src import lime_sentences
    sentences = texts(TEXT, split(TEXT))
    direct, _ = lime_sentences.explain(sentences, lambda ps: predict(ps)[:, CLASSES.index("SLEEP_ISSUES")], **LIME)
    assert sleep["attribution"]["LIME"] == pytest.approx({str(i): w for i, w in direct.items()})
    assert sleep["stated_evidence"] == [0] and sleep["stated_rationale"] == "why SLEEP_ISSUES"
    # erasure, recomputed by hand: the post without sentence 0, and sentence 0 alone
    sentences = texts(TEXT, split(TEXT))
    j = CLASSES.index("SLEEP_ISSUES")
    full, without, only = predict([sentences, sentences[1:], sentences[:1]])[:, j]
    assert sleep["confidence"] == pytest.approx(full)
    assert sleep["comprehensiveness"] == pytest.approx(full - without)
    assert sleep["sufficiency"] == pytest.approx(full - only)


def test_no_prediction_means_no_control_and_a_bare_false_negative(tmp_path):
    tok, predict, ig = pieces(tmp_path)
    calls = []
    records, result = post_records("p1", TEXT, ["SLEEP_ISSUES"], CLASSES, predict, ig, explains_everything(calls),
                                   LIME, threshold=1.1)
    assert calls == [] and result is None
    assert [(r["symptom"], r["predicted"], r["gold"], r["attribution"]) for r in records] == \
        [("SLEEP_ISSUES", False, True, {})]


def test_a_failed_control_keeps_the_attributions_and_leaves_erasure_undefined(tmp_path):
    tok, predict, ig = pieces(tmp_path)
    failed = lambda sentences, labels: ({}, {"ok": False})
    records, _ = post_records("p1", TEXT, [], CLASSES, predict, ig, failed, LIME, threshold=0.0)
    for r in records:
        assert r["attribution"]["IG"] and r["stated_evidence"] is None and r["comprehensiveness"] is None


def test_decoupled_tells_the_llm_the_labels_and_holds_it_to_them():
    seen = {}

    def ask_fn(prompt, system, n_sentences=None, expected=None):
        seen.update(prompt=prompt, n=n_sentences, expected=expected)
        return {"ok": True, "symptoms": [{"label": "FATIGUE", "evidence": [1], "rationale": "tired"}]}

    prompts = {"system": "s", "decoupled": "{numbered_post}\nLabels: {labels}"}
    stated, _ = decoupled(["I sleep badly.", "Tired."], ["FATIGUE"], prompts, ask_fn)
    assert stated == {"FATIGUE": ([1], "tired")}
    assert seen["prompt"] == "0: I sleep badly.\n1: Tired.\nLabels: FATIGUE"
    assert (seen["n"], seen["expected"]) == (2, ["FATIGUE"])


LONG = "I cannot sleep. My appetite is gone. I am tired. I cannot sleep."   # 4 + 5 + 4 + 4 word tokens


def test_sentences_seen_counts_the_prefix_inside_the_window(tmp_path):
    from src.run_m2 import sentences_seen
    tok = tokenizer(tmp_path)
    spans = split(LONG)
    assert sentences_seen(LONG, spans, tok, max_len=12) == 3        # 10 word tokens: all of 0, 1, part of 2
    assert sentences_seen(LONG, spans, tok, max_len=512) == 4


def test_m2_never_hands_the_control_or_erasure_a_sentence_bert_cannot_see(tmp_path):
    from src.run_m2 import sentences_seen
    tok, model = tokenizer(tmp_path), tiny_bert(len(CLASSES))
    predict = bert_predict(model, tok, torch.device("cpu"), max_len=12)
    ig = lambda text, spans, j: integrated_gradients(model, tok, text, spans, j, max_len=12, steps=5)
    handed = []

    def explain_why(sentences, labels):
        handed.append(list(sentences))
        return {label: ([len(sentences) - 1], "last one") for label in labels}, {"ok": True}

    records, _ = post_records("p1", LONG, [], CLASSES, predict, ig, explain_why, LIME, threshold=0.0,
                              seen=lambda text, spans: sentences_seen(text, spans, tok, max_len=12))
    assert handed == [texts(LONG, split(LONG))[:3]]                 # the fourth sentence is past the cut
    for r in records:
        assert r["sentences_seen"] == 3 and set(r["attribution"]["LIME"]) == {"0", "1", "2"}
        assert r["stated_evidence"] == [2] and r["comprehensiveness"] is not None
    # the visible prefix is scored exactly as the whole post: BERT never read the rest anyway
    full = predict([texts(LONG, split(LONG))])[0]
    assert records[0]["confidence"] == pytest.approx(full[CLASSES.index(records[0]["symptom"])])


def test_sanity_check_compares_trained_and_randomised_ig_per_item(tmp_path):
    from src.attribution import randomised_head
    tok, predict, ig = pieces(tmp_path)
    random = randomised_head(tiny_bert(len(CLASSES)), seed=3)
    ig_random = lambda text, spans, j: integrated_gradients(random, tok, text, spans, j, steps=5)
    records, _ = post_records("p1", TEXT, [], CLASSES, predict, ig, explains_everything([]), LIME,
                              threshold=0.0, ig_random=ig_random)
    assert all(-1.0 <= r["ig_random_spearman"] <= 1.0 for r in records)
    # the model against itself agrees perfectly: the comparison is wired to the same item and label
    records, _ = post_records("p1", TEXT, [], CLASSES, predict, ig, explains_everything([]), LIME,
                              threshold=0.0, ig_random=ig)
    assert all(r["ig_random_spearman"] == pytest.approx(1.0) for r in records)


def m2_record(post, symptom, ig, lime):
    return {"post_id": post, "symptom": symptom, "predicted": True,
            "attribution": {"IG": {str(i): s for i, s in enumerate(ig)}, "LIME": {str(i): s for i, s in enumerate(lime)}}}


def test_seed_check_compares_labels_per_post_and_attributions_on_shared_items():
    from src.run_m2 import seed_check
    seed0 = [m2_record("p1", "FATIGUE", [0.9, 0.5, 0.1], [0.8, 0.6, 0.0]),
             m2_record("p1", "SLEEP_ISSUES", [0.1, 0.2, 0.3], [0.1, 0.2, 0.3])]
    seed1 = [m2_record("p1", "FATIGUE", [0.1, 0.5, 0.9], [0.7, 0.5, 0.1])]      # IG reversed, LIME kept
    out = seed_check(seed0, seed1, ["p1", "p2"])
    # p1: {FATIGUE, SLEEP} vs {FATIGUE} = 0.5; p2: neither seed predicts anything = 1.0
    assert out["labels"] == pytest.approx(0.75)
    assert out["n_shared_predictions"] == 1                      # only FATIGUE on p1 is in both
    assert out["IG_median_spearman"] == pytest.approx(-1.0) and out["IG_top2_jaccard"] == pytest.approx(1 / 3)
    assert out["LIME_median_spearman"] == pytest.approx(1.0) and out["LIME_top2_jaccard"] == 1.0
