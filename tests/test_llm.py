import json

from src.llm import ask, embed, parse, repair, unwrap

GOOD = json.dumps({"symptoms": [
    {"label": "SLEEP_ISSUES", "evidence": [0, 2], "rationale": "says so", "confidence": 90},
    {"label": "FATIGUE", "evidence": [2], "rationale": "tired", "confidence": 60},
]})


def test_unwrap_survives_fences_and_chatter():
    assert unwrap('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert unwrap('Sure! Here you go: {"a": 1} Hope that helps.') == '{"a": 1}'
    assert unwrap('{"a": 1}') == '{"a": 1}'


def test_a_good_reply_parses():
    symptoms, reason = parse(GOOD, n_sentences=5)
    assert reason == "ok"
    assert [s["label"] for s in symptoms] == ["SLEEP_ISSUES", "FATIGUE"]
    assert symptoms[0]["evidence"] == [0, 2]
    assert symptoms[1]["confidence"] == 60


def test_evidence_is_deduplicated_and_sorted():
    reply = json.dumps({"symptoms": [{"label": "FATIGUE", "evidence": [3, 1, 3]}]})
    symptoms, _ = parse(reply, n_sentences=5)
    assert symptoms[0]["evidence"] == [1, 3]


def test_broken_json_is_rejected():
    assert parse("not json at all")[1] == "not json"
    assert parse('{"symptoms": [')[1] == "not json"
    assert parse("")[1] == "not json"
    assert parse('{"other": []}')[1] == "no symptoms list"


def test_a_label_outside_the_taxonomy_is_rejected():
    reply = json.dumps({"symptoms": [{"label": "Insomnia", "evidence": [0]}]})
    symptoms, reason = parse(reply)
    assert symptoms is None and reason == "unknown label 'Insomnia'"


def test_evidence_must_point_at_a_sentence_in_the_post():
    reply = json.dumps({"symptoms": [{"label": "FATIGUE", "evidence": [9]}]})
    assert parse(reply, n_sentences=5)[1].endswith("outside the post")
    assert parse(reply, n_sentences=10)[1] == "ok"          # same reply, longer post
    bad = json.dumps({"symptoms": [{"label": "FATIGUE", "evidence": ["first"]}]})
    assert parse(bad, n_sentences=5)[1].endswith("not a list of integers")


def test_confidence_must_be_a_number_in_range():
    over = json.dumps({"symptoms": [{"label": "FATIGUE", "evidence": [], "confidence": 140}]})
    assert parse(over)[1].endswith("not a number in 0-100")
    missing = json.dumps({"symptoms": [{"label": "FATIGUE", "evidence": []}]})
    assert parse(missing)[1] == "ok"


def test_contradictory_replies_are_rejected():
    twice = json.dumps({"symptoms": [{"label": "FATIGUE"}, {"label": "FATIGUE"}]})
    assert parse(twice)[1] == "the same symptom appears twice"
    mixed = json.dumps({"symptoms": [{"label": "NO_SYMPTOMS"}, {"label": "FATIGUE"}]})
    assert parse(mixed)[1] == "NO_SYMPTOMS returned alongside symptoms"


def test_ask_returns_on_the_first_good_reply():
    seen = []

    def call(prompt, system, seed=0, temperature=0, **kw):
        seen.append((prompt, temperature))
        return GOOD

    result = ask("p", "s", n_sentences=5, call=call)
    assert result["ok"] and result["attempts"] == 1
    assert seen == [("p", 0)]        # the first pass is always deterministic


def test_a_retry_says_what_was_wrong_and_warms_up():
    seen = []

    def call(prompt, system, seed=0, temperature=0, **kw):
        seen.append((prompt, temperature))
        return "garbage" if len(seen) < 3 else GOOD

    result = ask("p", "s", n_sentences=5, attempts=3, call=call)
    assert result["ok"] and result["attempts"] == 3
    assert seen[0] == ("p", 0)
    # retries carry the reason and a non-zero temperature; at 0 the seed is ignored so nothing
    # would change and the same reply would come back
    assert all(t > 0 and "rejected because not json" in p for p, t in seen[1:])


def test_repair_names_the_failure():
    text = repair("original question", "unknown label 'Insomnia'")
    assert text.startswith("original question")
    assert "unknown label 'Insomnia'" in text


def test_ask_gives_up_and_says_why():
    result = ask("p", "s", attempts=2, call=lambda *a, **k: '{"symptoms": [{"label": "Nope"}]}')
    assert not result["ok"]
    assert result["symptoms"] is None
    assert result["reason"] == "unknown label 'Nope'"
    assert result["attempts"] == 2
    assert result["raw"] is not None


def fake_embedder(calls):
    """Stands in for Ollama: the vector for a text is [its length, 1], and every request is recorded."""
    def post(url, body):
        calls.append(body)
        return {"embeddings": [[len(t), 1] for t in body["input"]]}
    return post


def test_embed_keeps_order_and_duplicates(tmp_path):
    calls = []
    out = embed(["ab", "c", "ab"], cache_dir=tmp_path, post=fake_embedder(calls))
    assert out == [[2, 1], [1, 1], [2, 1]]
    assert calls[0]["input"] == ["ab", "c"]          # the duplicate is only sent once


def test_embed_caches_each_text(tmp_path):
    calls = []
    embed(["ab", "c"], cache_dir=tmp_path, post=fake_embedder(calls))
    assert embed(["c", "ab", "new"], cache_dir=tmp_path, post=fake_embedder(calls)) == [[1, 1], [2, 1], [3, 1]]
    assert [c["input"] for c in calls] == [["ab", "c"], ["new"]]


def test_embed_batches_and_applies_the_prefix(tmp_path):
    calls = []
    embed(["a", "b", "c"], prefix="clustering: ", batch=2, cache_dir=tmp_path, post=fake_embedder(calls))
    assert [c["input"] for c in calls] == [["clustering: a", "clustering: b"], ["clustering: c"]]
    # a different prefix is a different cache entry, not a hit
    embed(["a"], cache_dir=tmp_path, post=fake_embedder(calls))
    assert calls[-1]["input"] == ["a"]


def test_ask_first_pass_can_sample_for_stability_runs():
    seen = []

    def call(prompt, system, seed=0, temperature=0, **kw):
        seen.append(temperature)
        return "garbage" if len(seen) < 2 else GOOD

    ask("p", "s", n_sentences=5, temperature=0.5, retry_temperature=0.3, call=call)
    assert seen == [0.5, 0.5]            # a retry never cools below the first pass


def test_the_decoupled_control_must_explain_exactly_the_labels_it_was_given():
    assert parse(GOOD, expected=["SLEEP_ISSUES", "FATIGUE"])[1] == "ok"
    assert "was asked to explain exactly" in parse(GOOD, expected=["SLEEP_ISSUES"])[1]     # added FATIGUE
    assert "was asked to explain exactly" in parse(GOOD, expected=["SLEEP_ISSUES", "FATIGUE", "ANHEDONIA"])[1]


def test_ask_retries_a_decoupled_reply_with_the_wrong_labels():
    seen = []
    only_sleep = json.dumps({"symptoms": [{"label": "SLEEP_ISSUES", "evidence": [0]}]})

    def call(prompt, system, seed=0, temperature=0, **kw):
        seen.append(prompt)
        return GOOD if len(seen) == 1 else only_sleep

    result = ask("p", "s", n_sentences=5, expected=["SLEEP_ISSUES"], call=call)
    assert result["ok"] and result["attempts"] == 2
    assert "was asked to explain exactly ['SLEEP_ISSUES']" in seen[1]


def test_a_rationale_that_is_not_text_is_rejected():
    reply = json.dumps({"symptoms": [{"label": "FATIGUE", "evidence": [0], "rationale": ["tired", "all day"]}]})
    assert parse(reply, n_sentences=2)[1] == "rationale for FATIGUE is not text"
    assert parse(json.dumps({"symptoms": [{"label": "FATIGUE", "evidence": [0], "rationale": None}]}), 2)[0][0]["rationale"] == ""
