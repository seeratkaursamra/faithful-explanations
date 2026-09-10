# CLAUDE.md

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
M2 fine-tuned bert-base-uncased (reproduces the ReDSM5 baseline), M3 self-rationalising LLM, plus a
decoupled-rationale control.

## Phases
0 env + verify data + reproduce BERT baseline (5 seeds)  ->  1 attribution  ->  2 rationale generation
->  3 evaluation harness (pre-register first)  ->  4 analysis + writeup. Current: Phase 0.

## Data
- data/redsm5_posts.csv (post_id, text); data/redsm5_annotations.csv (post_id, sentence_id,
  sentence_text, DSM5_symptom, status, explanation). Gated; never committed.
- Post labels = symptoms with status == 1; none -> NO_SYMPTOMS (10 classes). status == 0 rows are
  sentences the annotator considered and rejected; keep them.
- Split: train_test_split(test_size=0.2, random_state=42), unstratified -> 1187 / 297.
- Baseline recipe: bert-base-uncased, max_len 512, batch 8, AdamW 2e-5, 5 epochs, BCE, sigmoid > 0.5.
  Reported 0.51 micro-F1 / 0.36 macro-F1 (single unseeded run); tolerance +/- 0.02 over 5 seeds.
- `make inspect` must print "matches paper: True" before any modelling.

## Commands
`make test`, `make inspect`, `make baseline`, `make baseline5`. Configs in configs/*.yaml.
Everything generated goes in outputs/ (gitignored). All randomness seeded.

## Conventions
- Reproduce the baseline unchanged first; accuracy is not the contribution.
- Sentence is the unit of explanation. Two attribution methods, always report their agreement.
- Every metric function gets a unit test on a tiny synthetic example with a known answer.
- Log any design change in docs/decision_log.md.
- Plain, readable Python. One-line docstrings or none. Comments only where the why is non-obvious.

## Ethics
Never commit or redistribute the data. Never paste posts into external tools. No verbatim post quotes
in writeups. No re-identification. This is not a screening or diagnostic tool.
