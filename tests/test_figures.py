import json

import numpy as np
import pandas as pd
import pytest

from src.figures import classification_table, explanation_table, latex, results


def items(n=12):
    """Every post has one TP and, on half the posts, an FP, for one symptom and one model."""
    rows = [{"model": "M2", "post_id": f"p{i}", "symptom": "FATIGUE", "correctness": "TP",
             "plausibility": 0.6, "comprehensiveness": 0.5} for i in range(n)]
    rows += [{"model": "M2", "post_id": f"q{i}", "symptom": "FATIGUE", "correctness": "FP",
              "plausibility": 0.6, "comprehensiveness": 0.1} for i in range(n // 2)]
    return pd.DataFrame(rows)


def test_table1_counts_and_an_f1_interval_that_collapses_when_nothing_varies():
    table = classification_table(items(), {"n_resamples": 100, "seed": 0}).iloc[0]
    # 12 TP, 6 FP, 0 FN: precision 12/18, recall 1, F1 = 24/30
    assert (table["support"], table["precision"], table["recall"]) == (12, pytest.approx(2 / 3), 1.0)
    assert table["f1"] == pytest.approx(0.8)
    assert table["f1_lo"] <= 0.8 <= table["f1_hi"]
    all_tp = classification_table(items().query("correctness == 'TP'"), {"n_resamples": 50, "seed": 0}).iloc[0]
    assert (all_tp["f1"], all_tp["f1_lo"], all_tp["f1_hi"]) == (1.0, 1.0, 1.0)


def test_table2_formats_means_intervals_and_gaps():
    summary = pd.DataFrame([{"model": "M3", "symptom": "ALL", "metric": "plausibility", "n_tp": 40, "n_fp": 31,
                             "tp_mean": 0.63, "tp_lo": 0.61, "tp_hi": 0.65, "fp_mean": 0.6, "fp_lo": 0.58,
                             "fp_hi": 0.62, "gap": 0.03, "gap_lo": 0.0, "gap_hi": 0.06},
                            {"model": "M3", "symptom": "FATIGUE", "metric": "plausibility"}])
    row = explanation_table(summary).iloc[0]
    assert (row["correct"], row["wrong"], row["gap"]) == ("0.63 [0.61, 0.65]", "0.60 [0.58, 0.62]", "0.03 [0.00, 0.06]")
    assert row["n TP / FP"] == "40 / 31" and len(explanation_table(summary)) == 1       # per-symptom rows are left out


def test_latex_escapes_underscores_and_percent():
    tex = latex(pd.DataFrame({"measure": ["IG_stated_jaccard"], "value": [0.5]}), "95% intervals", "tab:x")
    assert r"IG\_stated\_jaccard" in tex and r"95\% intervals" in tex and r"\toprule" in tex and "0.50" in tex


def test_every_figure_and_table_is_written_from_a_harness_run(tmp_path):
    from src.evaluate import h1, h2, h3, h4, summarise
    boot = {"n_resamples": 50, "seed": 0}
    df = items()
    df["ig_lime_spearman"] = np.linspace(-0.2, 0.9, len(df))
    hypotheses = {"H1": [h1(df, "M2", boot=boot)], "H2": [h2(df, boot=boot)], "H3": [h3(df, "M2", boot=boot)],
                  "H4": [h4(df, {}, boot)]}
    df.to_csv(tmp_path / "items.csv", index=False)
    summarise(df, boot).to_csv(tmp_path / "summary.csv", index=False)
    (tmp_path / "hypotheses.json").write_text(json.dumps(hypotheses, default=float))
    written = results(tmp_path, tmp_path / "out", boot)
    for name in ["fig2_key", "fig3_symptoms", "fig4_accuracy", "fig5_sanity"]:
        assert (tmp_path / "out" / f"{name}.pdf").stat().st_size > 0
        assert (tmp_path / "out" / f"{name}.png").stat().st_size > 0
    assert (tmp_path / "out" / "table1_classification.csv").exists() and len(written) == 6


def test_table3_keeps_only_annotated_cases_and_never_any_text_column():
    from src.figures import TABLE3_COLUMNS, cases_table
    cases = pd.DataFrame({"model": ["M3", "M3"], "symptom": ["SLEEP_ISSUES", "FATIGUE"], "correctness": ["FP", "TP"],
                          "plausibility": [0.8, 0.7], "comprehensiveness": [0.0, 0.0],
                          "reading": ["topic match without evidence", "  "],
                          "post_text": ["a quoted post that must never reach the paper", "another"]})
    table = cases_table(cases)
    assert list(table.columns) == TABLE3_COLUMNS
    assert len(table) == 1 and table.iloc[0]["symptom"] == "Sleep issues"
