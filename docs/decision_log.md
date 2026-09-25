# Decision log

| Date | Decision | Alternatives | Reason |
|---|---|---|---|
| 2026-09-10 | Reproduce the authors' BERT recipe *unchanged* (unstratified split, seed 42, no early stopping, threshold 0.5) | stratified split, tuned threshold | Comparability with the reported 0.51/0.36 comes first; improvements are not the contribution |
| 2026-09-10 | Add torch/numpy seeds and run 5 seeds | single run like the notebook | The notebook is unseeded, so a single number is not reproducible; report mean +/- std and a +/-0.02 tolerance |
| 2026-09-10 | Post label = symptoms with status==1; negatives -> NO_SYMPTOMS (10 classes) | drop negatives; 9 classes | Mirrors analysis/statistics.py and the notebook's 10-class head |
| 2026-09-10 | Keep status==0 rows as "rejected candidate" evidence | discard | They are gold examples of topic-match-without-evidence, useful later for the plausibility analysis |
| 2026-09-10 | Test set = 297 posts (sklearn ceils 0.2 * 1484) | - | Recorded so per-symptom support numbers are interpretable |
| 2026-09-25 | TP vs FP contrasts use an unpaired bootstrap and Mann-Whitney, not a paired test | paired bootstrap / Wilcoxon, as first written in the plan | A (post, symptom) pair is either a TP or an FP, never both, so the two groups are disjoint and there is nothing to pair. Paired tests stay for contrasts on the same items: IG vs LIME, plausibility vs faithfulness, M2 vs M3 on the same posts |
| 2026-09-25 | H1 reported as a difference in differences with a bootstrap interval | two separate TP-FP gaps compared by eye | The claim is that the faithfulness gap exceeds the plausibility gap, so it should be one number with one interval rather than an informal comparison of two |
| 2026-09-25 | M3 is Qwen 2.5 7B Instruct, run locally through Ollama | Llama 3.x 8B, gemma-3, an API model | The LREC 2026 comparison of eight open models on these same nine DSM-5 symptoms put Qwen 2.5 7B first and found no gain from 7B to 14B, so it is the best operating point at a size that runs here. Local weights also mean no post ever leaves the machine and the model cannot drift under me |
| 2026-09-25 | Call Ollama with format=json rather than relying on the prompt alone | free-form output plus a parser with retries | Constrained decoding removed parse failures in the probe and cut latency by roughly a third. The retry path stays in the code for the cases it cannot save |
