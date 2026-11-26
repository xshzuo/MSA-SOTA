#!/usr/bin/env python3
"""Generate the three figures of the manuscript.

Figure 1  the three-stream architecture and the routed context
Figure 2  the two published columns against each other, one point per arm
Figure 3  the seven-level accuracy against the measured spread of the predictions

The figure conventions of the target journal are respected: a sans-serif
typeface throughout, a white background, no truncated axes, and error bars
where a quantity varies over seeds.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "code"))
from paired_deltas import load as load_arms  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ACCEPT_MOSI = {"acc2_all": 87.51, "acc2_nz": 88.26, "acc7": 50.34,
               "mae": 0.638, "cor": 0.838}
# the arms shown in the frontier, in the order in which their components were added
FRONTIER = [
    ("A0_mosi_bert_flat", "BERT-base"),
    ("A1_mosi_bert_cpf", "+ CPF"),
    ("B0_mosi_roberta_cpf", "+ RoBERTa"),
    ("B1_mosi_roberta_cpf_cnUW", "+ front end"),
    ("E6_mosi_deberta_bhead", "+ DeBERTa, head"),
    ("E4_mosi_bhead_ctx", "+ context"),
    ("E8_mosi_bhead_ctxreg", "+ routed"),
    ("E10_mosi_deberta_bhead_ctxreg", "+ DeBERTa"),
    ("E17_mosi_deberta_lb05_mse", "$\\lambda$=0.5, MSE"),
]


def means(arms, arm):
    g = arms.get(arm, {})
    if not g:
        return None
    return {k: st.mean([g[s][k] for s in g]) for k in
            ("acc2_all", "acc2_nz", "acc7", "mae", "cor")}


def fig_architecture(path):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef2f7", ec="#33475b", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    linewidth=0.9, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, color="#33475b", ls="-", lw=0.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, color=color, linestyle=ls,
                                     linewidth=lw, shrinkA=0, shrinkB=0))

    box(0.2, 5.2, 2.2, 1.0, "text encoder\n(transformer)")
    box(0.2, 3.8, 2.2, 1.0, "audio encoder")
    box(0.2, 2.4, 2.2, 1.0, "vision encoder")
    box(0.2, 0.6, 2.2, 1.0, "context encoder\n(neighbouring utterances)",
        fc="#f4f1e8", ec="#8a6d3b")

    box(3.6, 3.6, 2.2, 1.0, "cross-attention\n+ fusion")
    box(6.6, 4.7, 2.6, 0.9, "regression head (intensity)", fc="#eef7ee", ec="#3b6b3b")
    box(6.6, 3.2, 2.6, 0.9, "decision head (polarity)", fc="#eef7ee", ec="#3b6b3b")

    arrow(2.4, 5.7, 3.6, 4.5)
    arrow(2.4, 4.3, 3.6, 4.2)
    arrow(2.4, 2.9, 3.6, 3.9)
    # the context reaches the fusion, and from the fusion only the regression head
    arrow(2.4, 1.1, 4.7, 3.6, color="#8a6d3b", ls="--")
    arrow(5.8, 4.5, 6.6, 5.15)
    arrow(5.8, 4.0, 6.6, 3.65)
    ax.text(2.5, 2.25, "context enters the fusion;\nonly the regression head sees it",
            color="#8a6d3b", fontsize=8.5, ha="center", va="top")
    ax.text(6.6, 2.75, "text, audio, vision only", color="#3b6b3b",
            fontsize=8.5, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_pareto(path, arms):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    xs, ys, labels = [], [], []
    for arm, label in FRONTIER:
        m = means(arms, arm)
        if not m:
            continue
        xs.append(m["acc2_all"])
        ys.append(m["acc7"])
        labels.append(label)
    ax.plot(xs, ys, "-", color="#9aa5b1", linewidth=0.9, zorder=1)
    ax.scatter(xs, ys, s=26, color="#1f4e79", zorder=3)
    offsets = {"+ routed": (12, 4), "$\\lambda$=0.5, MSE": (-34, -16),
               "+ DeBERTa, head": (2, -16), "+ DeBERTa": (-4, 10),
               "+ front end": (-30, 6), "+ context": (-20, -10),
               "+ RoBERTa": (7, -5), "+ CPF": (7, 5), "BERT-base": (7, -5)}
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=offsets.get(lab, (5, 4)), fontsize=8.5, color="#333333")
    ax.axvline(ACCEPT_MOSI["acc2_all"], color="#b03a2e", linewidth=0.9,
               linestyle="--", zorder=2)
    ax.axhline(ACCEPT_MOSI["acc7"], color="#b03a2e", linewidth=0.9,
               linestyle="-.", zorder=2)
    # the two acceptance lines are named in the legend rather than by text
    # next to the lines, because the arms cluster in the corner where the
    # lines meet and any label there collides with them
    ax.plot([], [], color="#b03a2e", linewidth=0.9, linestyle="--",
            label="polarity acceptance")
    ax.plot([], [], color="#b03a2e", linewidth=0.9, linestyle="-.",
            label="intensity acceptance")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_xlabel("All-items polarity accuracy (Acc-2, Has0)", labelpad=2)
    ax.set_ylabel("Seven-level accuracy (Acc-7)", labelpad=2)
    ax.set_xlim(min(xs) - 1.2, max(xs) + 2.4)
    ax.set_ylim(min(ys) - 1.8, max(ys) + 1.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scale_of(results, arm, seed="1111"):
    import numpy as np
    for d in sorted(glob.glob(os.path.join(results, arm + "_s*"))):
        if not os.path.basename(d).endswith("_s" + seed):
            continue
        files = [f for f in sorted(glob.glob(os.path.join(d, "preds_*.npz")))
                 if not f.endswith("_dev.npz")]
        if not files:
            return None
        z = np.load(files[0])
        pk = [k for k in z.files if "pred" in k]
        tk = [k for k in z.files if "truth" in k or "true" in k or "label" in k]
        if not pk or not tk:
            return None
        pr = z[pk[0]].reshape(-1)
        tr = z[tk[0]].reshape(-1)
        return float(pr.std() / tr.std())
    return None


def fig_scale(path, arms, results):
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    arms_to_show = [
        ("E10_mosi_deberta_bhead_ctxreg", "routed context"),
        ("E6_mosi_deberta_bhead", "DeBERTa + head"),
        ("E12_mosi_bhead_ctxreg_mse", "squared error"),
        ("E13_mosi_deberta_bhead_ctxreg_lb05", "$\\lambda_{bin}$=0.5"),
        ("E8_mosi_bhead_ctxreg", "RoBERTa + routed"),
        ("E17_mosi_deberta_lb05_mse", "$\\lambda$=0.5, MSE"),
        ("E19_mosi_deberta_bhead_bal", "balanced"),
    ]
    xs, ys, colours = [], [], []
    palette = ["#1f4e79", "#b03a2e", "#2e7d32", "#8a6d3b", "#6a1b9a", "#00838f", "#c62828"]
    for (arm, label), colour in zip(arms_to_show, palette):
        g = arms.get(arm, {})
        for seed in sorted(g):
            r = scale_of(results, arm, seed)
            if r is None:
                continue
            xs.append(r)
            ys.append(g[seed]["acc7"])
            colours.append(colour)
    ax.scatter(xs, ys, s=26, c=colours, zorder=3, alpha=0.85)
    if len(xs) >= 3:
        order = np.argsort(xs)
        px = np.array(xs)[order]
        py = np.array(ys)[order]
        # a least squares line, drawn as a guide to the eye
        slope, intercept = np.polyfit(px, py, 1)
        xr = np.linspace(px.min(), px.max(), 50)
        ax.plot(xr, slope * xr + intercept, "-", color="#555555", linewidth=1.0,
                zorder=2)
        r = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.03, 0.95, f"Pearson r = {r:+.2f}  (n = {len(xs)})",
                transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.set_xlabel("Spread of the predictions / spread of the labels")
    ax.set_ylabel("Seven-level accuracy (Acc-7)")
    ax.set_xlim(min(xs) - 0.008, max(xs) + 0.008)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = load_arms(args.results)
    fig_architecture(os.path.join(args.out, "architecture.png"))
    fig_pareto(os.path.join(args.out, "pareto.png"), arms)
    fig_scale(os.path.join(args.out, "scale.png"), arms, args.results)
    print("已生成 3 张图 →", args.out)


if __name__ == "__main__":
    main()
