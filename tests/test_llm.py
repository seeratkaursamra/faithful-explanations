import json

from src.llm import ask, parse, unwrap

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
    calls = []

    def call(prompt, system, seed=0, **kw):
        calls.append(seed)
        return GOOD

    result = ask("p", "s", n_sentences=5, call=call)
    assert result["ok"] and result["attempts"] == 1
    assert calls == [0]


def test_ask_retries_under_a_new_seed_and_recovers():
    seeds = []

    def call(prompt, system, seed=0, **kw):
        seeds.append(seed)
        return "garbage" if seed < 2 else GOOD

    result = ask("p", "s", n_sentences=5, attempts=3, call=call)
    assert result["ok"] and result["attempts"] == 3
    assert seeds == [0, 1, 2]        # the seed moves, because temperature 0 repeats itself


def test_ask_gives_up_and_says_why():
    result = ask("p", "s", attempts=2, call=lambda *a, **k: '{"symptoms": [{"label": "Nope"}]}')
    assert not result["ok"]
    assert result["symptoms"] is None
    assert result["reason"] == "unknown label 'Nope'"
    assert result["attempts"] == 2
    assert result["raw"] is not None
