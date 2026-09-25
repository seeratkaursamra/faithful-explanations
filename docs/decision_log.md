# Decision log

| Date | Decision | Alternatives | Reason |
|---|---|---|---|
| 2026-09-10 | Reproduce the authors' BERT recipe *unchanged* (unstratified split, seed 42, no early stopping, threshold 0.5) | stratified split, tuned threshold | Comparability with the reported 0.51/0.36 comes first; improvements are not the contribution |
| 2026-09-10 | Add torch/numpy seeds and run 5 seeds | single run like the notebook | The notebook is unseeded, so a single number is not reproducible; report mean +/- std and a +/-0.02 tolerance |
| 2026-09-10 | Post label = symptoms with status==1; negatives -> NO_SYMPTOMS (10 classes) | drop negatives; 9 classes | Mirrors analysis/statistics.py and the notebook's 10-class head |
| 2026-09-10 | Keep status==0 rows as "rejected candidate" evidence | discard | They are gold examples of topic-match-without-evidence, useful later for the plausibility analysis |
| 2026-09-10 | Test set = 297 posts (sklearn ceils 0.2 * 1484) | - | Recorded so per-symptom support numbers are interpretable |
