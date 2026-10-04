"""Talk to the local model, and trust nothing it returns until it parses and validates."""
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .data import NO_SYMPTOMS, SYMPTOMS

URL = "http://localhost:11434/api/generate"
EMBED_URL = "http://localhost:11434/api/embed"
VALID_LABELS = set(SYMPTOMS) | {NO_SYMPTOMS}

# Ollama's default window was 2,048 tokens per request, and it cuts a longer prompt from the front without an
# error, system prompt first. The window has to hold the prompt and the whole reply, or a long reply pushes the
# prompt out as it is written: 32,768 (Qwen 2.5's own limit) holds the longest test post (10,746 tokens) plus a reply
# at the cap below. It stays out of the cache key: a prompt that fits gets the same reply in any window (checked),
# so only the replies that did not fit had to be asked again. See the decision log.
NUM_CTX = 32768

# A reply can loop: in JSON mode Qwen sometimes repeats a few characters until something stops it, which a big
# window lets run for many minutes. The longest real reply is 3,638 tokens (721 cited sentence numbers), so this cap
# leaves room for real replies, and a looping one comes back as broken JSON, which ask() retries like any other.
# Out of the cache key for the same reason as the window.
NUM_PREDICT = 8192


def unwrap(text):
    """Pull the JSON object out of a reply, even if the model wrapped it in prose or a code fence."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def parse(text, n_sentences=None, expected=None):
    """Return (symptoms, reason). symptoms is None when the reply fails a check, and reason says which.

    n_sentences, when given, is how many sentences the post was split into, so cited evidence
    indices can be checked against the post the model was actually shown. expected, when given, is
    the label set the decoupled control was told to explain; a reply that adds or drops one fails.
    """
    try:
        payload = json.loads(unwrap(text))
    except (json.JSONDecodeError, TypeError):
        return None, "not json"
    if not isinstance(payload, dict) or not isinstance(payload.get("symptoms"), list):
        return None, "no symptoms list"

    out = []
    for entry in payload["symptoms"]:
        if not isinstance(entry, dict) or "label" not in entry:
            return None, "entry is not an object with a label"
        label = entry["label"]
        if label not in VALID_LABELS:
            return None, f"unknown label {label!r}"

        evidence = entry.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(i, int) and not isinstance(i, bool)
                                                     for i in evidence):
            return None, f"evidence for {label} is not a list of integers"
        if n_sentences is not None and any(i < 0 or i >= n_sentences for i in evidence):
            return None, f"evidence for {label} cites a sentence outside the post"

        rationale = entry.get("rationale", "")
        if rationale is not None and not isinstance(rationale, str):
            return None, f"rationale for {label} is not text"

        confidence = entry.get("confidence")
        if confidence is not None and not (isinstance(confidence, (int, float))
                                           and not isinstance(confidence, bool)
                                           and 0 <= confidence <= 100):
            return None, f"confidence for {label} is not a number in 0-100"

        out.append({"label": label, "evidence": sorted(set(evidence)),
                    "rationale": rationale or "", "confidence": confidence})

    labels = [s["label"] for s in out]
    if len(labels) != len(set(labels)):
        return None, "the same symptom appears twice"
    if NO_SYMPTOMS in labels and len(labels) > 1:
        return None, "NO_SYMPTOMS returned alongside symptoms"
    if expected is not None and set(labels) != set(expected):
        return None, f"it explained {sorted(labels)} but was asked to explain exactly {sorted(expected)}"
    return out, "ok"


def generate(prompt, system, model="qwen2.5:7b", seed=0, temperature=0,
             cache_dir="outputs/llm_cache", timeout=1800, transport_attempts=3):
    """One call to Ollama, cached on disk so an interrupted run resumes instead of starting over.

    Only connection and timeout errors are retried here. A reply that arrives but is malformed is
    the caller's problem, because retrying it unchanged would return the same thing.
    """
    body = {"model": model, "system": system, "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": temperature, "seed": seed}}
    cached = _cache_path(cache_dir, body)
    if cached and cached.exists():
        return cached.read_text()

    sent = {**body, "options": {**body["options"], "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}}
    for attempt in range(transport_attempts):
        try:
            request = urllib.request.Request(URL, data=json.dumps(sent).encode(),
                                             headers={"Content-Type": "application/json"})
            reply = json.loads(urllib.request.urlopen(request, timeout=timeout).read())["response"]
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(reply)
            return reply
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == transport_attempts - 1:
                raise
            time.sleep(2 ** attempt)


def _cache_path(cache_dir, body):
    if not cache_dir:
        return None
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return Path(cache_dir) / f"{key}.json"


def _post(url, body, timeout=300):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request, timeout=timeout).read())


def embed(texts, model="nomic-embed-text", prefix="", batch=64, cache_dir="outputs/embed_cache", post=_post):
    """One vector per text, in order. Each text is cached on its own, so a gold rationale that is
    compared against every test item is embedded once.

    No task prefix by default: nomic-embed-text was trained with prefixes like "clustering: ", but on
    the sample's rationales none of them separated same-symptom from other-symptom pairs as well as
    no prefix did. See the decision log.
    """
    vectors, missing = {}, []
    for text in dict.fromkeys(texts):
        cached = _cache_path(cache_dir, {"model": model, "prefix": prefix, "text": text})
        if cached and cached.exists():
            vectors[text] = json.loads(cached.read_text())
        else:
            missing.append(text)

    for start in range(0, len(missing), batch):
        chunk = missing[start:start + batch]
        reply = post(EMBED_URL, {"model": model, "input": [prefix + t for t in chunk]})
        for text, vector in zip(chunk, reply["embeddings"]):
            vectors[text] = vector
            cached = _cache_path(cache_dir, {"model": model, "prefix": prefix, "text": text})
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(vector))
    return [vectors[t] for t in texts]


def repair(prompt, reason):
    """Re-ask, saying what was wrong with the last reply."""
    return (f"{prompt}\n\nYour previous reply was rejected because {reason}. "
            "Return only the JSON object described above, with nothing before or after it.")


def ask(prompt, system, n_sentences=None, attempts=3, seed=0, temperature=0, retry_temperature=0.3,
        expected=None, call=generate, **kwargs):
    """Call the model, parse the reply, and retry a malformed one.

    The first pass is temperature 0 and reproducible. Retrying is only useful if something
    actually changes: at temperature 0 Ollama ignores the seed, so an unchanged request returns
    an unchanged reply. A retry therefore says what was wrong and raises the temperature, which
    is the only way to draw a different sample.

    `temperature` above 0 is only for the stability runs, where sampling is the point.

    Returns {ok, symptoms, reason, attempts, raw}. ok is False when every attempt failed a check.
    """
    reason, raw, text = "no attempts made", None, prompt
    for i in range(attempts):
        warmth = temperature if i == 0 else max(temperature, retry_temperature)
        raw = call(text, system, seed=seed, temperature=warmth, **kwargs)
        symptoms, reason = parse(raw, n_sentences, expected)
        if symptoms is not None:
            return {"ok": True, "symptoms": symptoms, "reason": "ok", "attempts": i + 1, "raw": raw}
        text = repair(prompt, reason)
    return {"ok": False, "symptoms": None, "reason": reason, "attempts": attempts, "raw": raw}
