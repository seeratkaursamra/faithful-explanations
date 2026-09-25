# Faithful or merely plausible?

Are model explanations for DSM-5 depression-symptom detection faithful to what the model actually
did, or just plausible? I'm working on ReDSM5 (Bao et al. 2025), extending the explanation
question left open in STEP-EZ (Farruque et al. 2021).

## Data

ReDSM5 is gated. Requested access at https://huggingface.co/datasets/irlab-udc/redsm5.
Until it arrives, `data/` has the authors' public paraphrased sample (17 posts) so `src/data.py` can run.

Things I learned from the authors' code rather than the paper:

- A post's labels are the symptoms with `status == 1`. Posts with none get `NO_SYMPTOMS`, so there are 10 classes.
- `status == 0` rows are sentences the psychologist looked at and rejected for that symptom. I keep them.
- The split is `train_test_split(test_size=0.2, random_state=42)`, not stratified: 1187 train / 297 test.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
make test             # data-layer tests, run on the sample
make inspect          # full data: checks counts against the paper, writes docs/reading_sample.md
make inspect-sample   # same on the public sample; the mismatch with the paper is expected
make baseline         # one seed of the ReDSM5 BERT baseline
make baseline5        # five seeds + summary; target micro-F1 0.51 +/- 0.02
```

Generated files go in `outputs/`. My notes from reading posts by hand are in `docs/observations.md`,
design decisions in `docs/decision_log.md`.
