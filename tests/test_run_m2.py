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
