# Pre-registration

Written 2026-09-25, before any model has been run on ReDSM5. Dated so the analysis cannot be
read as chosen after the results.

## What I have already seen

Access to the full dataset has not been granted yet. Nothing here is informed by a result.
What I have seen: the authors' published paraphrased sample of 17 posts, which I read by hand
(`docs/observations.md`), and the numbers the ReDSM5 and STEP-EZ papers report. No model of mine
has produced a prediction, an attribution or a rationale on real data.

## Question

For DSM-5 depression-symptom detection on ReDSM5, are model explanations faithful to the model's
decision, or only plausible, and does that depend on whether the prediction is correct and on
which symptom it is?

Three quantities per (post, symptom): **G**, the psychologist's evidence sentences and rationale;
**S**, the model's stated rationale and cited sentences; **A**, post-hoc attribution over sentences.
Plausibility is S against G. Faithfulness is A against S, plus erasure metrics.

## Hypotheses

**H1 (primary).** On incorrect predictions, plausibility stays at roughly its level on correct
predictions, while faithfulness drops.

**H2.** Accuracy does not predict faithfulness. Across models and symptoms, per-symptom F1 and
faithfulness are not positively associated.

**H3.** The gap in H1 is widest for rare or inference-heavy symptoms (psychomotor, appetite change,
cognitive issues) and narrowest for suicidal thoughts.

**H4 (secondary).** The interpretable linear model M1 loses modest accuracy against M2 and is
faithful by construction, with attributions concentrated on known linguistic markers.

## Primary outcome

One number, fixed in advance, on **M3 Condition A** (Qwen 2.5 7B Instruct, self-rationalising), over the test split:

- **Plausibility**: cosine similarity between embeddings of S_text and G_text, nomic-embed-text.
  This is deliberately the measure ReDSM5 used to report 0.78, so the test speaks to their metric.
- **Faithfulness**: ERASER comprehensiveness with R = S_evid. The LLM reports confidence as a
  0-100 integer and a probe on 2026-09-25 found it saturated at 0 or 100, so for M3 this is a
  decision-flip rate rather than a probability drop. Stated here in advance, not discovered later.
- **Primary statistic**: the difference in differences, `did_ci` in `src/metrics.py`. The
  faithfulness gap between correct and incorrect predictions, minus the plausibility gap, with a
  95% bootstrap interval over 1,000 resamples.

## Decision rules

Fixed before seeing any result.

**H1 is supported** if both hold:
1. The difference-in-differences interval excludes zero and the point estimate is positive.
2. The plausibility gap on its own is small, its 95% interval containing zero or its point
   estimate under 0.05 on the cosine scale.

**H1 is refuted** if the difference-in-differences interval contains zero, or if the plausibility
gap is as large as the faithfulness gap.

**Partial support**, to be reported as such rather than as a win: the difference in differences
excludes zero but plausibility also drops materially, meaning both degrade and faithfulness merely
degrades faster.

**H2 is supported** if the correlation between per-symptom F1 and per-symptom faithfulness, pooled
across models, has an interval containing zero or is negative.

**H3 is supported** if the H1 difference in differences, computed per symptom, is larger for
psychomotor, appetite change and cognitive issues than for suicidal thoughts, with the ordering
holding under the small-support rule below.

**H4 is supported** if M1's micro-F1 is within 0.10 of M2's and its top-attributed sentences carry
higher marker density than its other sentences.

## Secondary outcomes

Reported, but not used to decide the hypotheses: evidence Jaccard and sentence P/R/F1 for S against
G; attribution-stated Jaccard and precision@k; sufficiency; A against G overlap; the same analysis
on M2 with the decoupled control; the same with LIME in place of Integrated Gradients; all-MiniLM-L6-v2
in place of nomic-embed-text; k = 2 in place of k = |G_evid|.

## The correctness split

Every (post, symptom) is TP, FP, FN or TN. The analysis compares **TP against FP**. These are
disjoint sets, so all TP-versus-FP contrasts use an unpaired bootstrap and Mann-Whitney, with
Cliff's delta as the effect size. Paired tests are reserved for contrasts on the same items:
Integrated Gradients against LIME, plausibility against faithfulness, one model against another.

FN and TN produce no stated rationale for that symptom and are excluded. This exclusion is a
limitation and will be stated as one, not quietly dropped.

## Analysis rules

- 95% percentile bootstrap intervals, 1,000 resamples over posts, on every aggregate.
- Effect sizes reported alongside every p-value, never a p-value alone.
- Holm correction across the nine symptoms wherever a test is repeated per symptom.
- Any symptom with fewer than 10 test examples is reported but not argued from. Psychomotor and
  appetite change are expected to fall under this on a 297-post test split.
- M2 is trained with five seeds; attribution runs on seed 0 with a check on one other seed.
- All randomness seeded; seeds live in configs.

## Conditions for calling the whole thing inconclusive

- The BERT baseline does not reproduce within 0.02 micro-F1, so M2 is not the published model.
- Integrated Gradients and LIME disagree so sharply on M2 that neither can stand for A. Pre-specified
  as median Spearman below 0.2 across test posts.
- Attributions on a randomly re-initialised classifier head do not degrade, meaning the attribution
  is not reflecting anything learned.
- The LLM's JSON parse-failure rate exceeds 10% after retries.
- Fewer than 30 FP items for a given model, which makes the correctness split unstable.

## Deviations

Any departure from this document gets an entry in `docs/decision_log.md` with its date and reason,
and is listed in the write-up. Exploratory analyses not named here are permitted and will be
labelled exploratory.
