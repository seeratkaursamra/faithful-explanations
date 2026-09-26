"""Figures and tables for the writeup. Run with: python -m src.figures"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .data import SYMPTOMS
from .evaluate import BOOT, MIN_IG_LIME_AGREEMENT, RANDOMISED_MAX_AGREEMENT, per_symptom
from .metrics import bootstrap_ci, resample_posts

# Greys only, so the figure survives a black and white print.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
FILL = "#f2f2f2"
GRID = "#e6e6e6"
FAINT = "#b3b3b3"

MODEL_NAMES = {"M1": "M1 linear", "M2": "M2 BERT, decoupled rationale", "M3": "M3 Qwen 2.5 7B, self-rationalising"}
MARKERS = {"M1": "^", "M2": "s", "M3": "o"}


def _box(ax, x, y, w, h, title, body=None, fill=FILL, edge=INK, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, edgecolor=edge, facecolor=fill,
                                linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h * (0.62 if body else 0.5), title, ha="center", va="center",
            fontsize=9, color=INK, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.27, body, ha="center", va="center", fontsize=7.4, color=MUTED)


def _arrow(ax, start, end, label=None, style="-|>", colour=INK, offset=0.018, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11,
                                 linewidth=1.0, color=colour, shrinkA=2, shrinkB=2,
                                 linestyle="--" if dashed else "-"))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + offset, label,
                ha="center", va="bottom", fontsize=7.2, color=MUTED)


def pipeline(path="figures/fig1_pipeline.pdf"):
    """Post to models to the three quantities, and the comparisons between them."""
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, mid, right, w = 0.025, 0.37, 0.715, 0.26
    cl, cm, cr = left + w / 2, mid + w / 2, right + w / 2

    _box(ax, 0.29, 0.895, 0.42, 0.085, "Reddit post, split into sentences",
         "one (post, symptom) pair at a time")

    # who produces each quantity
    _box(ax, left, 0.695, w, 0.105, "Psychologist", "ReDSM5 annotation")
    _box(ax, mid, 0.695, w, 0.105, "Model", "M1 linear · M2 BERT\nM3 Qwen 2.5 7B")
    _box(ax, right, 0.695, w, 0.105, "Attribution", "Integrated Gradients\nLIME")

    for x in (cl, cm, cr):
        _arrow(ax, (0.50, 0.895), (x, 0.802))
    _arrow(ax, (mid + w, 0.748), (right, 0.748))
    ax.text((mid + w + right) / 2, 0.762, "probes", ha="center", fontsize=6.6, color=MUTED)

    # the three quantities
    _box(ax, left, 0.455, w, 0.135, "G  gold", "evidence sentences\n+ clinical rationale")
    _box(ax, mid, 0.455, w, 0.135, "S  stated", "sentences the model cites\n+ its own rationale")
    _box(ax, right, 0.455, w, 0.135, "A  attributed", "sentences that actually\nmoved the prediction")

    for x in (cl, cm, cr):
        _arrow(ax, (x, 0.695), (x, 0.592))

    # comparisons between neighbouring quantities
    for a, b, name in [(mid, left + w, "plausibility"), (mid + w, right, "faithfulness")]:
        _arrow(ax, (a, 0.505), (b, 0.505), style="<|-|>")
        ax.text((a + b) / 2, 0.532, name, ha="center", fontsize=7.4, color=INK, fontweight="bold")

    ax.text((left + w + mid) / 2, 0.437, "cosine of rationales\nevidence Jaccard, P/R/F1",
            ha="center", va="top", fontsize=6.6, color=MUTED)
    ax.text((mid + w + right) / 2, 0.437, "A-vs-S overlap, precision@k\ncomprehensiveness, sufficiency",
            ha="center", va="top", fontsize=6.6, color=MUTED)

    # gold against attribution, bowed below the metric labels so nothing overlaps
    ax.annotate("", xy=(cl, 0.452), xytext=(cr, 0.452),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, linewidth=1.0,
                                connectionstyle="arc3,rad=-0.32", mutation_scale=11))
    ax.text(0.50, 0.300, "clinical grounding", ha="center", va="top", fontsize=7.4,
            color=INK, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))

    _box(ax, 0.045, 0.045, 0.91, 0.125,
         "Every comparison is split by correct and incorrect prediction",
         "H1: plausibility holds while faithfulness falls\nrepeated per symptom for H3",
         fill="white", dashed=True)
    _arrow(ax, (0.50, 0.250), (0.50, 0.172), dashed=True, colour=MUTED)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(Path(path).with_suffix(".png")), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(Path(path).with_suffix(".png")), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _axes(ax, grid="x"):
    """Hairline axes and a recessive solid grid, so the data is the only loud thing."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.6, length=3, labelcolor=INK)
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _nice(symptom):
    return symptom.replace("_", " ").lower().capitalize()


def _readable(text):
    """A verdict with the dataset's symptom labels written as words."""
    for s in SYMPTOMS:
        text = text.replace(s, _nice(s).lower())
    return text


def _h3_role(symptom):
    """H3 compares its targets (*) with suicidal thoughts (ref.); marked so the verdict can be read off."""
    from .evaluate import H3_REFERENCE, H3_TARGETS
    return " ref." if symptom == H3_REFERENCE else " *" if symptom in H3_TARGETS else ""


def _ci_text(triple):
    return f"{triple[0]:.2f} [{triple[1]:.2f}, {triple[2]:.2f}]"


def _measured(items, faithfulness="comprehensiveness"):
    """Models with both a plausibility and a faithfulness value somewhere: the ones H1 applies to."""
    if not {"plausibility", faithfulness} <= set(items.columns):
        return []
    have = items[items["plausibility"].notna() & items[faithfulness].notna()]
    return sorted(have["model"].unique())


def key_figure(items, hypotheses, path, boot=BOOT):
    """Fig 2, H1: for each model, each measure's mean on correct (filled) and wrong (open) predictions,
    with 95% intervals over posts. H1 predicts the plausibility pair stays together while the
    faithfulness pair comes apart."""
    models = _measured(items)
    fig, axes = plt.subplots(1, max(len(models), 1), figsize=(7.0, 2.1), squeeze=False, sharex=True)
    rows = [("plausibility", "Plausibility\nsimilarity to expert rationales"),
            ("comprehensiveness", "Faithfulness\ndrop when its evidence is removed")]
    for ax, model in zip(axes[0], models):
        for y, (metric, _) in zip((1, 0), rows):
            ends = {}
            for kind, dy, face in (("TP", 0.12, INK), ("FP", -0.12, "white")):
                g = items[(items["model"] == model) & (items["correctness"] == kind) & items[metric].notna()]
                if len(g):
                    mean, lo, hi = bootstrap_ci(g[metric], posts=g["post_id"], **boot)
                    ends[kind] = (mean, y + dy)
                    ax.errorbar(mean, y + dy, xerr=[[mean - lo], [hi - mean]], fmt="o", ms=5, mfc=face, mec=INK,
                                mew=1.0, ecolor=INK, elinewidth=0.8, capsize=0, zorder=3)
            if len(ends) == 2:
                ax.plot(*zip(ends["TP"], ends["FP"]), color=FAINT, linewidth=1.0, zorder=2)
        ax.set_yticks([1, 0], [label for _, label in rows] if ax is axes[0][0] else ["", ""], fontsize=7)
        ax.set_ylim(-0.55, 1.55)
        ax.set_xlim(0, 1)
        _axes(ax)
        h1 = next((h for h in hypotheses.get("H1", []) if h["model"] == model), {})
        note = f"DiD {_ci_text(h1['did'])}" if "did" in h1 else h1.get("verdict", "").split(":")[0]
        ax.set_title(f"{MODEL_NAMES.get(model, model)}\n{note}", fontsize=7.5, color=INK, loc="left")
    handles = [Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=5, label="correct (TP)"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=5, label="wrong (FP)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    return _save(fig, path)


def symptom_figure(hypotheses, path):
    """Fig 3, H3: the difference in differences per symptom, most common first. Hollow grey marks are
    under the 10-example support rule: shown, not argued from."""
    results = hypotheses.get("H3", [])
    # one x scale for every panel, or a small effect in one model looks as big as a large one in another
    fig, axes = plt.subplots(1, max(len(results), 1), figsize=(7.0, 2.6), squeeze=False, sharey=True, sharex=True)
    for ax, h3 in zip(axes[0], results):
        rows = h3["by_symptom"]
        for y, r in enumerate(reversed(rows)):
            if "did" not in r:
                ax.text(0, y, "no FP", fontsize=6, color=MUTED, va="center", ha="center")
                continue
            mean, lo, hi = r["did"]
            colour, face = (FAINT, "white") if r["low_support"] else (INK, INK)
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", ms=4.5, mfc=face, mec=colour,
                        ecolor=colour, elinewidth=0.8, capsize=0, zorder=3)
        ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
        ax.set_yticks(range(len(rows)), [f"{_nice(r['symptom'])} ({r['support']}){_h3_role(r['symptom'])}"
                                         for r in reversed(rows)], fontsize=7)
        _axes(ax)
        ax.set_title(f"{MODEL_NAMES.get(h3['model'], h3['model'])}\n{_readable(h3['verdict'])}", fontsize=7.5,
                     loc="left")
        ax.set_xlabel("faithfulness gap minus plausibility gap", fontsize=7, color=INK)
    return _save(fig, path)


def accuracy_figure(items, hypotheses, path, faithfulness="comprehensiveness"):
    """Fig 4, H2: per-symptom F1 against per-symptom faithfulness, one mark per (model, symptom). Models
    differ by shape, not colour; hollow marks are low-support symptoms, left out of the test."""
    points = per_symptom(items, faithfulness).dropna(subset=["f1", "faithfulness"])
    h2 = (hypotheses.get("H2") or [{}])[0]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for model, g in points.groupby("model"):
        low = g["symptom"].isin(set(g["symptom"]) - set(h2.get("symptoms", g["symptom"])))
        for subset, face in ((g[~low], INK), (g[low], "white")):
            # a white ring under each mark keeps overlapping marks apart
            ax.scatter(subset["f1"], subset["faithfulness"], marker=MARKERS.get(model, "o"), s=70,
                       facecolors="white", edgecolors="none", zorder=2)
            ax.scatter(subset["f1"], subset["faithfulness"], marker=MARKERS.get(model, "o"), s=28,
                       facecolors=face, edgecolors=INK, linewidths=0.9, zorder=3)
    _axes(ax, grid="both")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-symptom F1", fontsize=7)
    ax.set_ylabel(f"mean {faithfulness}", fontsize=7)
    title = f"Spearman {_ci_text(h2['spearman'])}" if "spearman" in h2 else h2.get("verdict", "H2 not run")
    ax.set_title(title, fontsize=7.5, loc="left")
    handles = [Line2D([], [], marker=MARKERS.get(m, "o"), ls="", mfc=INK, mec=INK, ms=5, label=MODEL_NAMES.get(m, m))
               for m in sorted(points["model"].unique())]
    handles.append(Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=5,
                          label="hollow: fewer than 10\nexamples, not tested"))
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, path)


def sanity_figure(items, path):
    """Fig 5: (a) IG against LIME on M2, per item, with the prereg's 0.2 line; (b) IG on the trained
    model against IG on a randomised head, with the 0.5 line. Each marks its median."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    m2 = items[items["model"] == "M2"]
    panels = [("ig_lime_spearman", False, MIN_IG_LIME_AGREEMENT, (-1, 1), "(a) IG vs LIME on M2: Spearman"),
              ("ig_random_spearman", True, RANDOMISED_MAX_AGREEMENT, (0, 1),
               "(b) trained vs randomised head: |Spearman|")]
    for ax, (column, absolute, line, limits, title) in zip(axes, panels):
        values = m2[column].dropna() if column in m2 else pd.Series(dtype=float)
        values = values.abs() if absolute else values
        _axes(ax, grid="y")
        ax.set_xlim(*limits)
        ax.set_title(title, fontsize=7.5, loc="left")
        if not len(values):
            ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
            continue
        counts, _, _ = ax.hist(values, bins=np.linspace(*limits, 21), color=FAINT, rwidth=0.85, zorder=2)
        tallest = counts.max()
        ax.set_ylim(0, tallest * 1.45)                 # headroom, so the labels sit above the bars
        ax.axvline(line, color=INK, linewidth=0.8, zorder=3)
        ax.axvline(values.median(), color=INK, linewidth=1.6, zorder=3)
        ax.text(line, tallest * 1.38, f" threshold {line}", fontsize=6, va="top", color=INK)
        ax.text(values.median(), tallest * 1.2, f" median {values.median():.2f}", fontsize=6, va="top", color=INK)
        ax.set_ylabel("items", fontsize=7)
    return _save(fig, path)


def classification_table(items, boot=BOOT):
    """Table 1: per model and symptom, gold support and precision, recall and F1, F1 with a 95% interval
    over posts. From the harness items, so it describes exactly the predictions being explained."""
    rows = []
    for model, m in items.groupby("model"):
        f1_draws = {s: [] for s in SYMPTOMS}
        for rows_drawn in resample_posts(m["post_id"], boot["n_resamples"], boot["seed"]):
            for r in per_symptom(m.iloc[rows_drawn]).itertuples():
                f1_draws[r.symptom].append(r.f1)
        for symptom in SYMPTOMS:
            s = m[m["symptom"] == symptom]
            if not len(s):
                continue
            tp, fp, fn = (int((s["correctness"] == k).sum()) for k in ("TP", "FP", "FN"))
            draws = [d for d in f1_draws[symptom] if not np.isnan(d)]
            rows.append({"model": model, "symptom": symptom, "support": tp + fn,
                         "precision": tp / (tp + fp) if tp + fp else np.nan,
                         "recall": tp / (tp + fn) if tp + fn else np.nan,
                         "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
                         "f1_lo": float(np.percentile(draws, 2.5)) if draws else np.nan,
                         "f1_hi": float(np.percentile(draws, 97.5)) if draws else np.nan})
    return pd.DataFrame(rows)


TABLE2_METRICS = ["plausibility", "null_similarity", "own_similarity", "evidence_f1", "comprehensiveness",
                  "sufficiency", "IG_stated_jaccard", "LIME_stated_jaccard"]


def explanation_table(summary):
    """Table 2: every model's plausibility and faithfulness measures on correct and wrong predictions,
    and the gap, each with its interval. The overall rows of summary.csv, formatted."""
    s = summary[(summary["symptom"] == "ALL") & summary["metric"].isin(TABLE2_METRICS)]

    def cell(r, name):
        return f"{r[name + '_mean']:.2f} [{r[name + '_lo']:.2f}, {r[name + '_hi']:.2f}]" \
            if pd.notna(r.get(name + "_mean")) else "-"

    rows = [{"model": r["model"], "measure": r["metric"], "n TP / FP": f"{int(r['n_tp'])} / {int(r['n_fp'])}",
             "correct": cell(r, "tp"), "wrong": cell(r, "fp"),
             "gap": f"{r['gap']:.2f} [{r['gap_lo']:.2f}, {r['gap_hi']:.2f}]" if pd.notna(r.get("gap")) else "-"}
            for _, r in s.iterrows()]
    return pd.DataFrame(rows)


TABLE3_COLUMNS = ["model", "symptom", "correctness", "plausibility", "comprehensiveness", "reading"]


def cases_table(cases):
    """Table 3: the qualitative cases with the reading written into cases.csv. Only these columns, so
    nothing quoted from a post can reach the paper even if the CSV grew one."""
    done = cases[cases["reading"].fillna("").str.strip() != ""]
    return done[TABLE3_COLUMNS].assign(symptom=done["symptom"].map(_nice)).reset_index(drop=True)


def latex(table, caption, label):
    """A booktabs table for the paper. Hand-built to keep the dependencies to what is pinned."""
    esc = lambda x: str(x).replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{" + "l" * len(table.columns) + "}",
             r"\toprule", " & ".join(esc(c) for c in table.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(f"{v:.2f}" if isinstance(v, float) else v) for v in row) + r" \\"
              for row in table.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def results(metrics_dir, out, boot=BOOT, cases_csv=None):
    """Figures 2-5 and Tables 1-2 from a finished harness run (items.csv, summary.csv, hypotheses.json),
    and Table 3 once cases.csv has readings in it."""
    metrics_dir, out = Path(metrics_dir), Path(out)
    items = pd.read_csv(metrics_dir / "items.csv")
    summary = pd.read_csv(metrics_dir / "summary.csv")
    hypotheses = json.loads((metrics_dir / "hypotheses.json").read_text())
    written = [key_figure(items, hypotheses, out / "fig2_key.pdf", boot),
               symptom_figure(hypotheses, out / "fig3_symptoms.pdf"),
               accuracy_figure(items, hypotheses, out / "fig4_accuracy.pdf"),
               sanity_figure(items, out / "fig5_sanity.pdf")]
    tables = [("table1_classification", classification_table(items, boot),
               "Classification on the test split, per symptom, with a 95% post-level interval for F1."),
              ("table2_explanations", explanation_table(summary),
               "Plausibility and faithfulness on correct and wrong predictions, with 95% post-level intervals.")]
    if cases_csv and Path(cases_csv).exists():
        table3 = cases_table(pd.read_csv(cases_csv))
        if len(table3):
            tables.append(("table3_cases", table3,
                           "Plausible but unfaithful explanations: the most plausible of each model's lowest-faithfulness "
                           "quarter, with a reading of why the explanation convinces. Exploratory."))
    for name, table, caption in tables:
        table.to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.tex").write_text(latex(table, caption, f"tab:{name.split('_')[0]}"))
        written.append(out / f"{name}.tex")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--metrics", default="outputs/metrics", help="a finished harness run, for figures 2-5")
    ap.add_argument("--config", default="configs/evaluate.yaml")
    ap.add_argument("--cases", default="outputs/qualitative/cases.csv", help="annotated cases, for Table 3")
    args = ap.parse_args()
    print("wrote", pipeline(f"{args.out}/fig1_pipeline.pdf"))
    if (Path(args.metrics) / "hypotheses.json").exists():
        boot = yaml.safe_load(Path(args.config).read_text())["bootstrap"]
        for path in results(args.metrics, args.out, boot, args.cases):
            print("wrote", path)
    else:
        print(f"no harness run in {args.metrics}; figures 2-5 need `make evaluate` first")
