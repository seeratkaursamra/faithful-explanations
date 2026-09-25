"""Figures for the writeup. Run with: python -m src.figures"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Greys only, so the figure survives a black and white print.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
FILL = "#f2f2f2"


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    print("wrote", pipeline(f"{args.out}/fig1_pipeline.pdf"))
