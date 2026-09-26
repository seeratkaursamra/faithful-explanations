# Faithful or merely plausible?

When a model says a Reddit post shows a symptom of depression and explains why, is that explanation
what actually drove its decision, or does it just sound like something a psychologist would write?

I test this on [ReDSM5](https://huggingface.co/datasets/irlab-udc/redsm5) (Bao et al., 2025): 1,484
Reddit posts labelled at the sentence level for the nine DSM-5 symptoms of a major depressive episode,
each label with a psychologist's written rationale. It picks up a question STEP-EZ (Farruque et al.,
2021) left open: its explanation score could not tell a meaningful explanation from a short one.

**Status:** everything is built and tested (172 tests) but nothing has run on real data yet, because
ReDSM5 access is gated and still pending. The analysis is pre-registered in `docs/prereg.md`, dated
before any result existed.

## The idea

For every (post, symptom) there are three things to compare:

| | What it is | Where it comes from |
|---|---|---|
| **G**, gold | the sentences the psychologist marked as evidence, and their rationale | ReDSM5 |
| **S**, stated | the sentences the model says it relied on, and its own rationale | the model |
| **A**, attributed | the sentences that actually moved the model's output | Integrated Gradients, LIME |

- **Plausibility** is how much S sounds like an expert: the similarity of the model's rationale to the
  psychologists' rationales for that symptom.
- **Faithfulness** is whether S is what the model really used. The main test is erasure: remove the
  sentences the model cited and see whether the prediction falls (comprehensiveness), keep only those
  sentences and see whether it holds (sufficiency). A-vs-S overlap is the second view.

The central comparison is **right predictions against wrong ones**. A wrong prediction has no real
evidence behind it, so a faithful explanation should weaken. If the explanation still reads as
convincing while its faithfulness drops, it is plausible but not faithful, and a clinician reading it
would be misled.

## Models

| | Model | Its explanation |
|---|---|---|
| **M1** | logistic regression on word n-grams plus four depression markers (first person, negative emotion, absolutist words, present tense) | exact by construction: each sentence's share of the model's output |
| **M2** | `bert-base-uncased` fine-tuned with the authors' recipe, reproducing their 0.51 micro-F1 | Integrated Gradients and LIME, plus a *decoupled* rationale: the LLM explains BERT's labels after the fact |
| **M3** | Qwen 2.5 7B Instruct, run locally through Ollama | self-rationalising: it answers with labels, cited sentences, a rationale and a confidence, all at once |

The authors' SVM is reproduced too, only as a check that the data and split match theirs.

## Hypotheses

Decided in `docs/prereg.md` and coded as rules in `src/evaluate.py`, so the verdicts are computed, not
judged after the fact.

- **H1 (primary).** On wrong predictions plausibility holds while faithfulness drops. Measured as a
  difference in differences (faithfulness gap minus plausibility gap, right vs wrong) with a 95%
  bootstrap interval over posts. Supported if that interval is above zero and the plausibility gap is
  small.
- **H2.** Accuracy does not predict faithfulness: per-symptom F1 against per-symptom faithfulness, pooled
  across models. Supported if the correlation's interval reaches zero or below.
- **H3.** The H1 gap is widest for rare, inference-heavy symptoms (psychomotor, appetite change,
  cognitive issues) and narrowest for suicidal thoughts.
- **H4.** The transparent M1 costs little accuracy (within 0.10 micro-F1 of BERT) and its top sentences
  are richer in depression markers than the rest.

The study is called **inconclusive** if the BERT baseline misses 0.51 ± 0.02, IG and LIME disagree (median
Spearman below 0.2), IG does not fall apart on a randomised classifier head, LLM replies fail to parse
more than 10% of the time, or a model has fewer than 30 wrong predictions. The harness checks all of
these first and prints each as triggered, clear, or not run.

## Setup

Built and tested on Python 3.14.7, macOS on an Apple M4 with 16 GB. Every dependency is pinned.

### Ollama first

[Ollama](https://ollama.com) runs the LLM and the embedding models locally, so no post ever leaves the
machine. Built with version 0.33.3.

```bash
brew install ollama                                   # macOS (how it is set up here)
curl -fsSL https://ollama.com/install.sh | sh         # Linux
                                                      # Windows: the installer from ollama.com

ollama serve                                          # leave this running in its own terminal
curl -s localhost:11434/api/version                   # in another terminal: prints {"version":"..."}
```

If you installed the macOS desktop app instead, it starts the server itself from the menu bar, so skip
`ollama serve` (it would only say the address is in use). "Connection refused" from any command here
means the server is not running.

For the overnight LIME run, the server can answer several requests at once. Start it as
`OLLAMA_NUM_PARALLEL=4 ollama serve` and run `python -m src.run_m3 --lime --workers 4` in place of
`make m3-lime`. How much that saves on a 16 GB machine is untested, so time `--limit 10` both ways first.

### The models

`make setup` below pulls all three. By hand, with the server running:

```bash
ollama pull qwen2.5:7b          # M3 and the decoupled control, 4.7 GB
ollama pull nomic-embed-text    # plausibility embeddings, 274 MB
ollama pull all-minilm          # secondary embeddings, 45 MB
ollama run qwen2.5:7b "Reply with the single word: ready"      # should answer "Ready"
```

`qwen2.5:7b` is Qwen 2.5 7B **Instruct** (7.6B parameters, 4-bit Q4_K_M quantisation, 32k context,
Apache-2.0), which runs comfortably on the 16 GB M4 used here; the first call after a pull takes a few
seconds to load it. I use it because a 2026 comparison of eight open models on these same nine DSM-5
symptoms ranked it first and found no gain going to 14B, and because a local model means posts never
leave the machine and the model cannot change under me (see the decision log). The code sends its own
system prompt (`configs/prompts.yaml`), so Ollama's default one is never used.

### Then everything else

```bash
python3 -m venv venv && source venv/bin/activate
make setup
```

`make setup` installs the pinned packages, downloads NLTK's sentence splitter and tagger, and pulls the
three models above. Run it again any time; anything already there is skipped. `ollama list` should then
show all three.

**NRC Emotion Lexicon.** Download it from
[saifmohammad.com](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) (free for research;
accept the terms on the page) and put `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` in `data/`. It is
licensed, so it is gitignored. Check it loaded: `python -c "from src.lexicon import load_nrc; print(len(load_nrc()))"`
should print about 2,400.

**The data.** Request access on Hugging Face, then put `redsm5_posts.csv` and `redsm5_annotations.csv`
in `data/`. They are gitignored and never committed. Until then, `data/` holds the authors' public
paraphrased sample (17 posts), and everything falls back to it.

Training picks CUDA, then the Apple GPU, then the CPU. Set `device:` in `configs/bert_baseline.yaml`
to force one.

## Try it now, without the data

```bash
make test             # 172 tests, about 6 s
make inspect-sample   # the data checks on the 17-post sample
```

`make inspect-sample` should end with mismatches, which are expected on 17 posts:

```
posts=17 negatives=2 annotation_rows=25 (status1=21, status0=4)
train/test = 13/4
...
matches paper: False
test split matches the authors' (per-label support from their svm notebook): False
gold sentences aligned to my split: 5/25 (20%) - ...
```

A small end-to-end run with the real LLM (needs `ollama serve`):

```bash
python -m src.run_m3 --allow-mismatch --out outputs/demo/records
python -m src.evaluate outputs/demo/records/m3.jsonl --logs outputs/demo/records --out outputs/demo/metrics
```

```
4 records from 4 posts; parse failures 0 (0.0%, prereg limit 10%); 8 questions in ...s
study: inconclusive
  [not run] BERT baseline outside 0.51 +/- 0.02 micro-F1: None
  ...
  [triggered] M3: fewer than 30 false positives: 0
H1 M3: inconclusive: 0 false positives with both plausibility and comprehensiveness, fewer than 30
...
```

"Inconclusive" is the correct answer here: 4 test posts cannot answer anything. The run shows the
pipeline works end to end, not a result. The first run takes about half a minute; repeats print `0.0s`,
because every LLM call is cached in `outputs/llm_cache/`.

## The full run

In this order. The first two steps take seconds and catch a wrong dataset or split before anything
long starts.

| # | Command | Time on the M4 | What to check |
|---|---|---|---|
| 1 | `make inspect` | seconds | `matches paper: True`, `test split matches the authors': True`, and a high gold-alignment rate |
| 2 | `make svm` | seconds | `matches paper: True` and the split matching again. The SVM is deterministic, so on the authors' split it should give 0.39 / 0.28 at their two decimals |
| 3 | `make baseline5` | ~1.4 h | `within tolerance: True`: the five-seed mean micro-F1 within 0.51 ± 0.02 |
| 4 | `make m1` | minutes | M1's micro-F1 and its records |
| 5 | `make m2` | ~2-3 h | `decoupled failures` low; IG's completeness gap small |
| 6 | `make m3` | ~2-2.5 h | `parse failures` under 10% |
| 7 | `make m3-lime` | overnight, 15-20 h | rewrites `m3.jsonl` with LIME added; time a `--limit 10` run first |
| 8 | `make evaluate` | minutes | the study status, then the H1-H4 verdicts |
| 9 | `make evaluate-minilm` | minutes | whether the verdicts hold with the second embedding model |
| 10 | `make figures` | ~10 s | Figures 1-5 and Tables 1-2 in `figures/` |
| 11 | `make cases` | seconds | the qualitative cases to read by hand |

If step 1 or 2 fails, stop there: the data or the split is not the authors', and every number after
would be about a different test set.

Three secondary runs, each written outside `outputs/records/` so the harness never reads them twice:

```bash
# the seed check: attributions on BERT seed 1 against seed 0
python -m src.run_m2 --checkpoint outputs/bert_baseline/seed1/checkpoint --out outputs/seedcheck
python -m src.run_m2 --seed-check outputs/records/m2.jsonl outputs/seedcheck/m2.jsonl

# M3 stability: three seeds at the retry temperature on 100 posts, then compared
for s in 1 2 3; do python -m src.run_m3 --seed $s --temperature 0.3 --limit 100 --out outputs/stability/seed$s; done
python -m src.run_m3 --limit 100 --stability outputs/stability/seed{1,2,3}/m3.jsonl
```

Everything LLM-side is cached on disk, so an interrupted run resumes where it stopped.

## What comes out

| Path | What it holds |
|---|---|
| `outputs/data_stats.json` | the paper check, the split check, alignment, descriptive statistics |
| `outputs/bert_baseline/seed*/metrics.json`, `summary.json` | BERT per seed (with the device and time), and the five-seed summary |
| `outputs/svm/metrics.json`, `outputs/m1/metrics.json` | the SVM check and M1 |
| `outputs/records/{m1,m2,m3}.jsonl` | one line per (model, post, symptom): prediction, cited sentences, rationale, attributions, erasure scores |
| `outputs/records/{m2,m3}_log.json` | parse failures, IG completeness gaps, runtime |
| `outputs/metrics/items.csv` | every metric for every item, marked TP / FP / FN |
| `outputs/metrics/summary.csv` | right vs wrong for every metric, overall and per symptom, with intervals, tests and effect sizes |
| `outputs/metrics/hypotheses.json` | the study status, the H1-H4 verdicts and their numbers, and the paired contrasts |
| `figures/` | Fig 1 pipeline, Fig 2 H1, Fig 3 per symptom (H3), Fig 4 accuracy vs faithfulness (H2), Fig 5 sanity checks; Tables 1-3 as CSV and LaTeX |
| `outputs/qualitative/cases.md` | the cases with their posts, **for reading only**: quotes posts, never shared |

A metric that does not apply to an item (no gold evidence on a wrong prediction, nothing cited) is left
blank, never scored as 0, and every mean is reported with how many items it covers.

## Reading the results

`make evaluate` prints the study status first. Only if it is `clear` do the verdicts count as results.
If a condition is `triggered`, the prereg says the study is inconclusive on that ground, and that is
reported as the finding. `not run` means a step was skipped; run it.

Each verdict is `supported`, `refuted`, `partial support` (H1 only: both measures fall, faithfulness
faster) or `inconclusive`, with the reason. Symptoms with fewer than 10 test examples (psychomotor and
appetite change, most likely) are shown but not argued from.

For Table 3, open `outputs/qualitative/cases.md`, read each case, write a short reading into the
`reading` column of `outputs/qualitative/cases.csv`, and run `make figures` again. Table 3 only ever
takes the scores and the reading, never post text.

## Repository

```
src/
  data.py            loading, the authors' split, the paper and split checks
  sentences.py       sentence splitting and matching gold sentences to it
  metrics.py         every metric and statistical test, pure functions
  lexicon.py         the depression markers
  llm.py             the Ollama client: validated replies, retries, embeddings, a disk cache
  lime_sentences.py  LIME with sentences as features, for BERT and the LLM alike
  attribution.py     Integrated Gradients, and the randomised-head copy for the sanity check
  train_bert.py      M2, the authors' recipe with seeds
  svm_baseline.py    the authors' SVM, as a reproduction check
  m1.py              M1 and its exact sentence scores
  run_m2.py          M2's attributions, the decoupled control, erasure
  run_m3.py          M3's answers, erasure, LIME, stability
  evaluate.py        the harness: per-item metrics, right vs wrong, the prereg's rules
  cases.py           the qualitative cases, by a fixed rule
  figures.py         figures and tables
configs/             training recipe, prompts, bootstrap and attribution settings
docs/prereg.md       the pre-registration, with one dated amendment made before any data
docs/decision_log.md every design decision, the alternatives, and why
docs/observations.md notes from reading posts by hand
tests/               one test file per module; synthetic inputs or the public sample, no network or GPU
```

## Decisions worth knowing

All of these are in `docs/decision_log.md` with the evidence.

- **The split.** The authors' notebooks read a prebuilt CSV, so the same seed only gives the same test
  posts if the rows come in the same order. `make inspect` compares our test split with the per-label
  counts printed in their SVM notebook, and `make svm` should then reproduce their scores exactly.
- **Plausibility on wrong predictions.** A wrong prediction has no gold rationale of its own, so every
  explanation is compared with the same symptom's rationales from *other* training posts. The null
  baseline is the other eight symptoms' rationales.
- **BERT only sees 512 tokens,** so M2's explanations are computed on the sentences inside that window.
  Otherwise a sentence BERT never read could be cited, and its erasure would falsely score as
  unfaithful.
- **LIME and IG are written here,** not taken from libraries, because the unit is the sentence and one
  of the models is an LLM behind an HTTP call. IG matches Captum to within 1e-6.
- **M3 is zero-shot,** so its explanations are its own and not steered toward the gold style.

## Ethics

The posts are about people's mental health. The data is never committed or shared, posts are never
pasted into external tools, and the write-up quotes no post: Table 3 is built without post text by
design. The LLM runs locally for the same reason. Nothing is done to identify anyone. This is research on
how models explain themselves; it is not a screening or diagnostic tool.

## References

- Bao, Pérez, Parapar (2025). ReDSM5: A Reddit Dataset for DSM-5 Depression Detection. CIKM.
- Farruque, Goebel, Zaiane, Sivapalan (2021). STEP-EZ, explainable zero-shot depression-symptom detection. ICMLA; arXiv:2106.10928.
- DeYoung et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. ACL.
- Sundararajan, Taly, Yan (2017). Axiomatic Attribution for Deep Networks. ICML.
- Ribeiro, Singh, Guestrin (2016). "Why Should I Trust You?" KDD.
- Adebayo et al. (2018). Sanity Checks for Saliency Maps. NeurIPS.
- Mohammad, Turney (2013). Crowdsourcing a Word-Emotion Association Lexicon. Computational Intelligence.
- Al-Mosaiwi, Johnstone (2018). In an Absolute State: Elevated Use of Absolutist Words Is a Marker
  Specific to Anxiety, Depression, and Suicidal Ideation. Clinical Psychological Science.

---

<sub>Made for love, with love.</sub>
