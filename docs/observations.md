# Hand-reading observations

Notes from reading `docs/reading_sample.md` (regenerate with `make inspect`).

Batch read: 10 posts, 12 evidence rows, 3 rejected rows. Posts are 18-152 words. The sample is
paraphrased, so anything about wording or sentence ids is provisional until I re-read the real data.

## How the psychologist writes rationales
- One or two sentences, roughly 20-35 words. Almost every one ends by tying the sentence back to
  "major depressive disorder" or "DSM-5".
- Structure is always the same: restate the sentence in clinical vocabulary, then link it to the
  disorder. E.g. crying -> "tearfulness"; guilt -> "worthlessness"; can't focus -> "difficulty
  concentrating". The clinical noun is the bridge between the post and the label.
- Criterion numbers are named about a third of the time ("criterion one", "criterion eight",
  "ninth criterion", "criteria four"). Not consistent; the rest just say "DSM-5 criteria".
- Hedging is the default register. "May reflect", "could point to", "can indicate", "may imply",
  "could suggest" appear in 9 of 12 evidence rationales. The three flat ones are anhedonia,
  guilt/worthlessness and cognitive issues ("is a hallmark", "is closely tied"). Even suicidal
  ideation gets "shows ... closely tied to", not "meets".
- Two rationales carry a conditional: "if these emotions are frequent and overwhelming, they may
  meet...". The psychologist is saying the sentence alone is not sufficient, yet it is still gold.
- The rationales are mostly about the symptom, not about this post. "Frequent crying can indicate
  emotional vulnerability, often seen in depression" would fit any crying sentence in the corpus.
  This matters for S-vs-G: a model rationale could score high on similarity just by using the
  symptom's clinical vocabulary, without pointing at anything specific to the post. I should check
  how similar gold rationales are to each other within a symptom before trusting S-vs-G.
- Rationales sometimes add duration or frequency that is not in the sentence ("for weeks",
  "frequent", "repeated"). The psychologist is importing context or assuming it.

## Explicit vs implicit evidence
- Explicit (the sentence states the symptom in plain words): can't stop crying (x2), can't focus,
  sleep worse plus pills, taking my own life, nothing interests me anymore, feel incredibly guilty.
  That is 7 of 12.
- Implicit (the label needs an inference): "I made so many bad choices" -> worthlessness and
  depressed mood; "I'm socially awkward and people can sense it" -> worthlessness; "I felt
  incredibly energized" -> fatigue (the fatigue is only implied by the contrast with the next
  sentences); lifelong lack of interest in sex -> anhedonia. That is 5 of 12.
- Rough ratio 60/40 explicit, on a tiny sample. The implicit ones cluster in WORTHLESSNESS and
  DEPRESSED_MOOD, the affective symptoms. The somatic/behavioural ones (sleep, concentration,
  suicidality) are literal. Relevant to H3: the subtle symptoms are where gold itself requires
  inference, so token attribution has the least chance of matching gold there.

## Negation, hedging, temporal framing
- The posts hedge less than the rationales do. No literal "I'm not sad, but" in the batch.
- The clearest temporal case is s_993_1060: the evidence for FATIGUE is "I felt incredibly
  energized", describing the opposite state in the past. The fatigue is inferred from "then
  everything fell apart". Any attribution method that highlights "energized" for FATIGUE is
  doing something wrong, yet that sentence is gold. Expect A-vs-G to punish the model here
  unfairly, or the gold to be wrong. Either way it needs a note in the analysis.
- s_205_396: "never been interested in sex... never really bothered me". Lifelong trait, no
  change, explicitly not distressing, still labelled ANHEDONIA. DSM-5 anhedonia is about loss of
  interest. The psychologist applied the "loss" test in s_2748_341 (high sex drive rejected for
  anhedonia) but not here. Inconsistent on temporal framing.
- s_2338_920: acute reaction to a near-accident, "I've calmed down now", and "I didn't feel
  anything". Labelled DEPRESSED_MOOD, and the rationale generalises one event to "frequent
  crying". Single-day distress counted as a symptom.
- s_329_224 is retrospective ("if I ever try again") and s_321_219 minimises ("my life isn't even
  terrible, but"). Neither framing changed the label. The minimising sentence was not chosen as
  evidence; the crying sentence was.
- Takeaway: the annotator does not treat temporal or minimising framing as a reason to withhold a
  label. Negation only shows up as a rejection reason when the symptom direction is reversed (the
  sex drive case).

## Rejected (status == 0) sentences: why were they rejected?
- s_2748_341, high sex drive -> ANHEDONIA rejected. Topic match on sex/libido, wrong direction.
  The cleanest rejection: the psychologist checks the sign, not just the topic.
- s_1955_1069, "I feel extremely weighed down" -> WORTHLESSNESS rejected. Emotional vocabulary in
  an angry post about a racist joke. Context overrides the lexical cue.
- s_1404_102, "I always mess it up" -> WORTHLESSNESS rejected, while "I'm socially awkward and
  people can sense it" in the same post is accepted for the same symptom. I cannot tell these two
  apart. Either the annotator picks one best sentence per symptom and marks the runner-up as
  rejected, or this is noise. Worth checking on the real data whether rejected rows tend to sit in
  posts that already have an accepted row for the same symptom.
- Pattern: all three are lexical or topical matches (sex, weighed down, mess it up) where the
  surrounding context does not support the symptom. Rejected rationales still hedge ("might relate
  to depressive symptoms, but").
- These are the useful hard negatives. A plausible-but-unfaithful explanation would highlight
  them; a NO_SYMPTOMS prediction that is faithful should put no mass on them.
- s_2748_341 also says "I still only sleep in short, weird bursts" and is labelled NO_SYMPTOMS
  with no SLEEP_ISSUES row, not even a rejected one. Possibly because the poster is on medication
  and "trying to fix" it, but it looks like a missed label.

## Symptom conflations to watch
- s_329_224: one sentence (sent 0) is gold for both WORTHLESSNESS and DEPRESSED_MOOD, with
  rationales that differ only in which symptom they name. Identical evidence for two labels means
  sentence-level A-vs-G cannot distinguish the two symptoms for that post; only S-vs-G can.
- WORTHLESSNESS is the bucket for guilt, regret, self-criticism, low self-esteem and social
  awkwardness. Guilt is in the criterion text, so fine. Social awkwardness is a stretch.
- DEPRESSED_MOOD is operationalised as crying in 3 of 4 cases. Tearfulness is in the DSM-5 text,
  but it makes "cry" a lexical shortcut a model can learn without any notion of mood. Check how
  often "cry" appears in DEPRESSED_MOOD evidence on the real data.
- FATIGUE vs ANHEDONIA: the FATIGUE rationale in s_993_1060 itself says "energy loss or lack of
  pleasure". The annotator blends them in the rationale even when the label is one or the other.
- ANHEDONIA vs libido: both anhedonia-related rows in the batch are about sex drive. If that holds
  on the real data, ANHEDONIA is partly a libido detector, which is not what the criterion means.
- SLEEP vs FATIGUE: not conflated in the batch; s_993_1060 has both from different sentences.
- No PSYCHOMOTOR, APPETITE or EATING rows in 10 posts. Rare classes will need a targeted read.

## Does sentence_text match the post text exactly?
- No evidence sentence in the batch matches the post verbatim, as expected from the paraphrase.
  Examples: "made so many bad choices" vs "how many mistakes I made"; "I hate that I'm crying
  right now" vs "I'm crying right now and I hate being so weak".
- s_156_366's evidence is two sentences joined with an ellipsis and an exclamation mark that is
  not in the post. s_1404_102's "Nothing interests me anymore" has no counterpart sentence in the
  paraphrased post at all. The posts and annotations were paraphrased independently.
- Sentence ids do not index into the paraphrased posts: sent 11 in a 7-sentence post, sent 34/51/62
  in a 15-sentence post. Either the ids refer to the original post under a finer splitter (clause
  level?) or they are artefacts of the sample.
- On the real data, check three things: is sentence_text a substring of text; what splitter
  reproduces sentence_id; and whether the multi-sentence evidence spans occur. If gold sentences
  are not exact substrings, I need an alignment step (fuzzy match against my own sentence split)
  and that is a design decision for the decision log, because it changes what "gold sentence"
  means for every metric.

## Surprises
- Gold evidence can describe the opposite of the symptom ("energized" -> FATIGUE).
- A lifelong, untroubling lack of sexual interest counts as anhedonia.
- Accepted and rejected sentences for the same symptom in the same post can look equivalent.
- A NO_SYMPTOMS post contains an unannotated sleep complaint.
- The rationales are more hedged than I expected from a clinician, and read like a template
  applied per symptom. I want to see whether the real rationales look this uniform or whether the
  paraphrase flattened them.
- Posts self-diagnose ("because of my depression" in s_1404_102). The word "depression" in the
  post is a shortcut a classifier could use for any symptom. Count how often it appears in the
  real posts, split by label.
- Sample posts are short (max 152 words). If the real posts are much longer, max_len 512 will
  truncate and the gold sentence could sit past the cutoff. Check the length distribution and
  where gold sentences fall in the post.
