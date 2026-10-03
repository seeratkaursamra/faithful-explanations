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

## Amendment 1, 2026-09-25

Made after a review of the analysis code, still before any model has been run on ReDSM5 and before
the dataset has arrived. The text above is left as first written; where it conflicts with this
section, this section holds. Each change has an entry in `docs/decision_log.md`.

**1. What plausibility is measured against.** As written, plausibility compares S_text with the
post's own G_text. On a false positive the symptom has no status == 1 rows, so G_text does not exist
and the plausibility half of the primary statistic could not be computed for incorrect predictions.
Plausibility is now the mean cosine similarity between S_text and every G_text for the same symptom
from the training split, computed identically for correct and incorrect predictions. Test posts
never appear in the reference set. Comparing correct predictions with their own post but incorrect
ones with other posts is ruled out, because own-post similarity is higher by construction and would
build a plausibility gap into the measure. Similarity to the post's own G_text is kept as a secondary
outcome on correct predictions only, since that is the comparison behind ReDSM5's 0.78.

The null baseline changes with it: the same mean similarity, but against the training-split G_texts
of the other eight symptoms. If a stated rationale is no closer to its own symptom's references than
to other symptoms', the measure is reading generic clinical language rather than symptom content.

**2. G_text when there are several evidence sentences.** ReDSM5 gives one explanation per evidence
sentence. G_text for a (post, symptom) is those explanations joined in sentence order.

**3. Resampling unit.** The analysis rules already say resamples are over posts. The implementation
resampled (post, symptom) items, which treats several symptoms from one post as independent and
narrows the intervals. Every bootstrap, `did_ci` included, now draws posts and takes all of a drawn
post's items, so correct and incorrect items from the same post are resampled together.

**4. Which prompt the erasure and LIME calls use on M3.** The joint prompt, the one that produced
S_evid. A probe on the paraphrased sample found the joint and label-only prompts gave different label
sets on 5 of 17 posts at temperature 0, so erasure or LIME run through the label-only prompt would
measure a different decision from the one being explained. The model's output for a symptom is its
joint-prompt confidence for that label, and 0 when the label is absent.

**5. k for attribution overlap.** A_topk against S_evid uses k = |S_evid|, not k = |G_evid|. On a false
positive |G_evid| is 0, so A-vs-S overlap would have been 0 on every incorrect prediction by
construction, producing the faithfulness drop H1 predicts. k = |G_evid| is kept for A against G, on
correct predictions only. Where the model cites no sentence, A-vs-S overlap is undefined: those items
are counted and reported, not scored as 0. The k = 2 robustness check is unchanged.

**6. Evidence overlap between S and G on incorrect predictions.** Evidence Jaccard and sentence
precision, recall and F1 need G_evid, which is empty on a false positive. They are reported on
correct predictions only.

## Amendment 2, 2026-10-02

Made after a sanity check of the whole analysis code, on the day ReDSM5 access was granted and before
the data files were downloaded. No model has been run on the real data and no real-data number exists.
The text above stands where this section does not override it. Each change has a decision-log entry,
and each was found by a probe on synthetic data that the old rule got wrong.

**1. H1 compares the two gaps on one scale.** The difference in differences subtracted a gap in mean
cosine (which barely moves: the sample's expert rationales span about 0.11 between own symptom and the
others) from a gap in a 0-1 probability drop. A simulated plausibility drop of 0.9 SD against a
faithfulness drop of 0.35 SD came out "supported". Each gap is now Cliff's delta, correct over wrong:
P(correct > wrong) - P(correct < wrong). The primary statistic is delta(faithfulness) - delta(plausibility),
both read from the same post-level resamples (`metrics.delta_did_ci`).
- Supported: the interval of that difference excludes zero and is positive, and plausibility's delta is
  negligible: its interval contains zero or its estimate is below 0.147, the conventional cut-off for a
  negligible Cliff's delta (Romano et al., 2006). This replaces the 0.05 cosine threshold.
- Refuted: the interval contains zero, or plausibility's delta is at least faithfulness's.
- Partial support otherwise.
The raw difference in differences is reported as a secondary.

**2. H2 is judged on ranks within each model, pooled.** One correlation over every (model, symptom)
point let a model that is better overall but less faithful overall reverse the sign: a simulation
positive within each model came out "supported". Correlating within each model alone has almost no
power at this size: a perfect correlation over 7 symptoms gave an interval of -0.03 to 0.96. So each
model's per-symptom F1 and mean comprehensiveness are ranked within that model, and Spearman is taken
over the pooled ranks, with a post-level interval.
- Refuted: the interval is above zero.
- Supported: the interval's upper bound is below 0.5. Thin data that cannot rule out a strong positive
  link is no longer counted as support, as the original "interval contains zero" rule allowed.
- Inconclusive otherwise. Each model's own correlation is reported alongside. Symptoms under the
  10-example rule are left out, as before.

**3. H3 needs false positives, not only gold support.** A symptom enters the H3 comparison only with at
least 10 test examples and at least 5 false positives. Before, two symptoms with one false positive each
decided the verdict. The per-symptom statistic is the same Cliff's-delta difference as H1's.

**4. The false-positive floor applies to models that explain themselves.** The condition "fewer than 30
false positives for a given model" exists because it makes the correctness split unstable, and M1 states
no rationale, so its split tests nothing. M1 had made the whole study inconclusive in a probe. The floor,
H1 and H3 now apply to M2 and M3 only.

**5. Rules settled since Amendment 1, recorded here so this document stands alone.**
- H4: M1's micro-F1 at most 0.10 below M2's five-seed mean, and the rate of marker words (first person,
  NRC fear, anger or sadness, absolutist; each word once) in M1's top two sentences minus the rest,
  with a post-level interval above zero.
- The randomised-model check: IG on M2 against IG on a copy with the classifier head re-drawn at
  BERT's initial scale; IG has failed to degrade if the median absolute Spearman is 0.5 or more.
- Parse failures: the worse of the M3 joint run and the M2 decoupled control counts.
- M2 is explained only on the sentences inside BERT's 512-token window; the share of gold evidence
  past the cut is reported.
- M3 is prompted zero-shot.

**6. What the plausibility half of H1 can show.** Since Amendment 1, plausibility compares a rationale
with the same symptom's rationales from other posts. A model writing about the symptom it predicted
will read like that symptom's typical rationale whether or not the post supports it, so plausibility
holding on wrong predictions is close to built into the measure. That is the point about
similarity-based evaluation this study makes, and the write-up will say so: the evidence for H1 is in
the faithfulness half. As an exploratory check grounded in the post, a model's rationale on a false
positive is also compared with the annotator's rejection note for that symptom in that post, where one
exists (status 0 rows).

### Correction to Amendment 2's timing, 2026-10-02

Amendment 2 says it was made "before the data files were downloaded". That is true of the decisions and
not of the commit. The record, from file timestamps:

- 2026-09-25 19:02: the sanity check that found all four problems, with the fixes proposed (kept
  privately), a week before access.
- 2026-10-02, before access was announced: the fixes agreed.
- 2026-10-02 19:12:55 to 19:13:06: the two ReDSM5 files saved into data/, while the fixes were being
  coded. Every Amendment 2 file was saved after that, between 19:13 and 19:20; the commit is 19:21.

Between the download and the commit no process read the data files. Every test reads the public sample
or synthetic inputs (checked: the one `load` without the sample flag reads fake files in a temporary
folder), no output under outputs/, figures/ or docs/ other than this file and the decision log was
written after 19:12, and the newest LLM call in the cache is from 2026-09-25. No real-data number of any
kind, descriptive or otherwise, existed when Amendment 2 was committed or when this note was written.
