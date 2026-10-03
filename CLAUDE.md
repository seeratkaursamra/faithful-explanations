# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
Are model explanations for DSM-5 depression-symptom detection faithful to the model's decision, or
merely plausible? Built on ReDSM5 (1,484 Reddit posts, sentence-level DSM-5 labels, psychologist-written
rationales). Extends STEP-EZ (Farruque et al. 2021), whose explanation metric could not separate
meaningful explanations from short ones.

## Design
Per (post, symptom): G = gold evidence sentences + rationale; S = the model's stated rationale;
A = post-hoc attribution (Integrated Gradients, LIME; token scores aggregated to sentences).
Compare S-vs-G (plausibility), A-vs-S + ERASER comprehensiveness/sufficiency (faithfulness), A-vs-G.
Split every metric by correct vs incorrect prediction and by symptom. Models: M1 interpretable linear,
M2 fine-tuned bert-base-uncased (reproduces the ReDSM5 baseline), M3 self-rationalising LLM
(Qwen 2.5 7B Instruct via Ollama), plus a decoupled-rationale control.

`docs/prereg.md` fixes the primary outcome, decision rules and analysis rules, and is dated before
any real-data result. Treat it as binding: anything that departs from it is a deviation and gets
logged, not a quiet change.

## Phases
0 env + verify data + reproduce BERT baseline (5 seeds)  ->  1 attribution  ->  2 rationale generation
->  3 evaluation harness (pre-register first)  ->  4 analysis + writeup.

Built so far: data layer, sentence segmentation + gold alignment, metrics, BERT baseline, LIME over
sentences, Integrated Gradients, lexicon markers, Ollama client with embeddings, pipeline figure.
M1 linear model, the evaluation harness, and every runner that produces the harness's input.
The data arrived 2026-10-02 (release b81b534); inspect and the SVM have run on it, the models have not.

## Code map
- `src/data.py` - the only place the label scheme and the split live. Everything else imports `load`,
  `split`, `binarizer`, `gold`. `check_against_paper` is the gate before modelling.
- `src/sentences.py` - `split` gives character spans (Punkt, run per line); `token_sentences` maps a
  tokeniser's output onto them for attribution; `align` matches a gold sentence to my split and
  `alignment_report` says how often that fails. Run the report before trusting any sentence metric.
- `src/metrics.py` - every explanation metric and every statistical test, pure functions on plain
  dicts/lists. `delta_did_ci` (Cliff's deltas, prereg Amendment 2) is the H1 primary statistic; the raw
  `did_ci` is a secondary. Unpaired tests for TP-vs-FP, paired only for
  contrasts on the same items.
- `src/lime_sentences.py` - LIME written here, not the `lime` package: the unit is the sentence and
  `predict` is any callable, so the same function explains BERT and an LLM behind HTTP. Returns the
  local weighted R-squared alongside the weights; report it.
- `src/llm.py` - Ollama client. `parse` validates every reply (taxonomy, evidence indices in range,
  no duplicate or contradictory labels); `ask` retries a malformed reply with the reason appended and
  the temperature raised. Calls are cached on disk under `outputs/llm_cache`.
- `src/attribution.py` - Integrated Gradients by hand over the word embeddings, [CLS]/[SEP] kept in the
  baseline, completeness gap returned; `explain` goes straight to sentence scores. Matched to Captum.
- `src/lexicon.py` - first person, negative affect (NRC fear+anger+sadness, file licensed and gitignored),
  absolutist words, present-tense share of finite verbs. Features for M1 and the marker-density check.
- `src/m1.py` - one logistic regression per label on TF-IDF 1-2 grams ("I" kept) + the four markers.
  `explain` gives each sentence its exact share of the logit; markers, boundary bigrams and bias apart.
- `src/evaluate.py` - the harness. Input is one JSONL record per (model, post, symptom), format in its
  docstring. Scores every item, compares TP vs FP overall and per symptom, marker density for H4, and
  applies the prereg's H1 and H3 decision rules as code (`verdict`, `h3_verdict`). Undefined metrics are
  left out, never scored 0.
- `src/run_m3.py` - M3 over the test split: the joint answer, then comprehensiveness and sufficiency through
  the same joint prompt, written as harness records with FNs included and a parse-failure log. `--lime`
  adds LIME per predicted symptom (the overnight run); each perturbation is asked once per post.
- `src/run_m2.py` - M2 over the test split from the seed-0 checkpoint: IG and LIME on predicted symptoms, the
  decoupled control explaining BERT's labels (held to them), and erasure on BERT of what it cites.
- `src/svm_baseline.py` - the authors' SVM copied exactly; a deterministic second reproduction check.
- `src/train_bert.py` - the baseline, hyperparameters from the config, seeds and device handling.
- `src/inspect_data.py` - stats, the gold-alignment rate, the first-person/tense check, and a hand-reading batch.
- `src/figures.py` - Fig 1 (pipeline) always; Figs 2-5 and Tables 1-3 from a finished harness run.
- `src/cases.py` - the qualitative cases by a fixed rule; `cases.md` quotes posts and stays in outputs/.
- `configs/bert_baseline.yaml`, `configs/prompts.yaml`, `configs/evaluate.yaml`, `configs/attribution.yaml` -
  training recipe, the M3 prompts, the bootstrap settings, and LIME / IG / randomised-head settings
  (the joint prompt and the decoupled control).

## Data
- data/redsm5_posts.csv (post_id, text); data/redsm5_annotations.csv (post_id, sentence_id,
  sentence_text, DSM5_symptom, status, explanation). Gated; never committed. `load` falls back to
  the authors' public paraphrased sample (17 posts) in `data/` when the real files are absent.
- Post labels = symptoms with status == 1; none -> NO_SYMPTOMS (10 classes). status == 0 rows are
  sentences the annotator considered and rejected; keep them.
- Split: train_test_split(test_size=0.2, random_state=42), unstratified -> 1187 / 297.
- Baseline recipe: bert-base-uncased, max_len 512, batch 8, AdamW 2e-5, 5 epochs, BCE, sigmoid > 0.5.
  Reported 0.51 micro-F1 / 0.36 macro-F1 (single unseeded run); tolerance +/- 0.02 over 5 seeds.
- `make inspect` must print "matches paper: True" before any modelling. It also checks the test split against
  the authors' per-label support (from their svm notebook); `make svm` should then give 0.39 / 0.28 exactly. `require_paper_match` in the
  config makes training refuse otherwise (set it false for a sample run); `--allow-mismatch` is for
  inspect, m1, run_m2 and run_m3 on the public sample, and `make inspect-sample` uses it.

## Commands
```bash
python3 -m venv venv && source venv/bin/activate && make setup   # packages, NLTK data, the Ollama models
make test                                                  # pytest over tests/
make inspect | inspect-sample                              # full data | public sample
make baseline | baseline5 | svm                            # one seed | five seeds + summary.json | the authors' SVM
make m1                                                    # M1; needs the NRC file in data/
make m2 | m3 | m3-lime                                     # records for M2 | M3 | M3 with LIME (overnight); need ollama
make evaluate | evaluate-minilm                            # the harness | rerun with the secondary embedding model
make cases | figures                                       # qualitative cases (private) | figures and tables
make figures                                               # regenerate figures/ (gitignored)
python -m pytest tests/test_metrics.py::test_jaccard -q     # a single test
```
M3 needs Ollama running locally (`ollama serve`, `ollama pull qwen2.5:7b`); nothing in `src/llm.py`
works without it. Training picks CUDA, then the Apple GPU, then CPU; set `device:` in the config to
force one. Everything generated goes in outputs/ (gitignored). All randomness seeded.

## Conventions
- Reproduce the baseline unchanged first; accuracy is not the contribution.
- Sentence is the unit of explanation. Two attribution methods, always report their agreement.
- Every metric function gets a unit test on a tiny synthetic example with a known answer. Tests use
  synthetic inputs or the public sample only, and never need a network or a GPU.
- Log any design change in docs/decision_log.md, with the alternatives and the reason.
- Plain, readable Python. One-line docstrings or none. Comments only where the why is non-obvious.

## Ethics
Never commit or redistribute the data. Never paste posts into external tools. No verbatim post quotes
in writeups. No re-identification. This is not a screening or diagnostic tool.
