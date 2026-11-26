#!/usr/bin/env python3
"""Generate the supplementary information of the manuscript.

The supplementary file carries the per-seed values of every arm, the full
paired comparisons behind the mechanism analysis, the configuration of every
arm as it was recorded in the result file, and the commands that reproduce
the tables.  It is written as a self-contained LaTeX document with its own
title and author block, as the journal requires.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from paired_deltas import load as load_arms, paired, ttest_p  # noqa: E402

COLS = ["acc2_all", "acc2_nz", "acc7", "mae", "cor"]
HEAD = ["Acc-2 (Has0)", "Acc-2 (Non0)", "Acc-7", "MAE", "Corr"]
PAIRS = [
    ("E8_mosi_bhead_ctxreg", "E4_mosi_bhead_ctx", "routed context vs. context in both heads"),
    ("E13_mosi_deberta_bhead_ctxreg_lb05", "E10_mosi_deberta_bhead_ctxreg", "$\lambda_{bin}$ 0.1 $\\to$ 0.5, with context"),
    ("E14_mosi_deberta_bhead_lb05", "E6_mosi_deberta_bhead", "$\lambda_{bin}$ 0.1 $\\to$ 0.5, without context"),
    ("E12_mosi_bhead_ctxreg_mse", "E10_mosi_deberta_bhead_ctxreg", "L1 $\\to$ MSE, with context"),
    ("E17_mosi_deberta_lb05_mse", "E6_mosi_deberta_bhead", "L1 $\\to$ MSE and $\\lambda_{bin}=0.5$, without context (three factors)"),
    ("C5_mosei_bhead_ctxreg", "C1_mosei_roberta_cpf_cnUW", "routed context on CMU-MOSEI"),
    ("E19_mosi_deberta_bhead_bal", "E6_mosi_deberta_bhead", "class-balanced regression loss"),
    ("B1_mosi_roberta_cpf_cnUW", "B0_mosi_roberta_cpf", "the case-aware language front end"),
]


def esc(s):
    return (s or "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "supplementary.tex"))
    args = ap.parse_args()
    arms = load_arms(args.results)

    L = []
    W = L.append
    # the class needs the same preamble as the manuscript
    W(r"\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}")
    W(r"\usepackage{graphicx}")
    W(r"\usepackage{amsmath,amssymb,amsfonts}")
    W(r"\usepackage{amsthm}")
    W(r"\usepackage{booktabs}")
    W(r"\usepackage{longtable}")
    W(r"\usepackage{array}")
    W(r"\usepackage{graphicx}")
    W(r"\usepackage{multirow}")
    W(r"\usepackage{xcolor}")
    W(r"\usepackage{algorithm}")
    W(r"\usepackage{algpseudocode}")
    W(r"\begin{document}")
    W(r"\title[Supplementary Information]{Supplementary Information}")
    W(r"\author*[1]{\fnm{Xiangshan} \sur{Zuo}}\email{xsh.zuo@hotmail.com}")
    W(r"\affil*[1]{\orgdiv{School of Business}, \orgname{Hunan University of Humanities, Science and Technology}, \orgaddress{\street{Dixing Road}, \city{Loudi}, \postcode{417000}, \state{Hunan}, \country{China}}}")
    W(r"\maketitle")
    W(r"\section*{Supplementary Table S1. Every arm, every seed}")
    W(r"The values below are the test-set readings of every arm of the study, one")
    W(r"row per seed. The mean over the seeds and the range are given in the")
    W(r"last two columns. Lower is better for MAE and higher for the others.")
    W(r"The arm tags are abbreviated to fit the page: deb is \texttt{deberta},")
    W(r"rob is \texttt{roberta}, xl is \texttt{xlnet}, bh is \texttt{bhead},")
    W(r"cx is \texttt{ctxreg}, cn is \texttt{cnUW} and cp is \texttt{cpf};")
    W(r"the full tags are in Supplementary Table S3.")
    # the table is wide and long, so it is set as a longtable in a small
    # font; the arm names are shortened to the part before the seed, which is
    # enough to identify them and keeps the columns inside the text block
    # the table has one row per seed for every arm, so it does not fit on a
    # single page; longtable lets it break, and the fixed column widths keep
    # it inside the text block
    W(r"\scriptsize")
    W(r"\setlength{\tabcolsep}{2pt}")
    W(r"\begin{longtable}{p{104pt} p{20pt} r r r r r r}")
    W(r"\toprule")
    W(r"Arm & Seed & " + " & ".join(HEAD) + r" \\")
    W(r"\midrule")
    W(r"\endfirsthead")
    W(r"\multicolumn{8}{l}{\textit{Table S1, continued}} \\")
    W(r"\toprule")
    W(r"Arm & Seed & " + " & ".join(HEAD) + r" \\")
    W(r"\midrule")
    W(r"\endhead")
    W(r"\bottomrule")
    W(r"\endlastfoot")
    # the arm tags are long, so the table refers to them by a short label and
    # the mapping is given in the legend below the heading
    W(r"The arms are identified by the labels below, which are used in the")
    W(r"first column of this table; the same labels are expanded in")
    W(r"Supplementary Table S3.")
    W(r"\smallskip")
    for arm in sorted(arms):
        g = arms[arm]
        # the arm tags are long; the corpus is fixed by the section and the
        # encoder and the change are what identify the arm, so the corpus is
        # dropped and the rest is abbreviated
        # the arm tags are long, so the corpus is dropped (it is fixed by
        # the section) and the remainder is kept but shortened on the two
        # long tokens that repeat across the arms
        _short = re.sub(r"_(?:mosi|mosei)(?=_|$)", "", arm).lstrip("_")
        _short = _short.replace("deberta", "deb").replace("roberta", "rob")
        _short = _short.replace("xlnet", "xl").replace("cnUW", "cn")
        _short = _short.replace("bhead", "bh").replace("ctxreg", "cx")
        _short = _short.replace("asf", "asf").replace("bal", "bal")
        _short = _short.replace("cpf", "cp")
        for seed in sorted(g):
            r = g[seed]
            W(f"{esc(_short)} & {seed} & " +
              " & ".join(f"{r[k]:.4f}" if k in ("mae", "cor") else f"{r[k]:.2f}" for k in COLS) + r" \\")
        means = [st.mean([g[s][k] for s in g]) for k in COLS]
        W(f"\\midrule \\multicolumn{{2}}{{l}}{{\\textit{{mean}}}} & "
          + " & ".join(
              (f"{means[i]:.4f}" if COLS[i] in ("mae", "cor") else f"{means[i]:.2f}")
              for i in range(len(COLS))) + r" \\")
        W(f"\\multicolumn{{2}}{{l}}{{\\hspace{{6pt}}\\textit{{range}}}} & "
          + " & ".join(
              (f"{min(g[s][COLS[i]] for s in g):.4f}--{max(g[s][COLS[i]] for s in g):.4f}"
               if COLS[i] in ("mae", "cor") else
               f"{min(g[s][COLS[i]] for s in g):.2f}--{max(g[s][COLS[i]] for s in g):.2f}")
              for i in range(len(COLS))) + r" \\")
    W(r"\end{longtable}")
    W(r"\normalsize")
    W(r"\smallskip")
    W(r"\smallskip")
    W(r"{\footnotesize")
    W(r"\normalsize")
    W(r"\section*{Supplementary Table S2. The paired comparisons}")
    W(r"Each row pairs two arms over the seeds they share. $t$ and $P$ are from")
    W(r"a two-sided paired $t$ test with $n-1$ degrees of freedom; the range is")
    W(r"over the per-seed differences. The comparisons in the last row change")
    W(r"three factors at once and are listed for completeness only.")
    W(r"\footnotesize")
    W(r"\setlength{\tabcolsep}{3pt}")
    W(r"\begin{tabular}{p{128pt} r r r r r l}")
    W(r"\toprule")
    W(r"Comparison & $n$ & $\Delta$Acc-2 (Has0) & $t$ & $P$ & $\Delta$Acc-7 & range \\")
    W(r"\midrule")
    for a, b, name in PAIRS:
        got = paired(arms, a, b)
        if not got:
            continue
        out, common = got
        m2, d2 = out["acc2_all"]
        m7, d7 = out["acc7"]
        tv, df, pv = ttest_p(d2)
        t = f"{tv:+.2f}" if tv == tv else "n/a"
        p = f"{pv:.4f}" if pv == pv else "n/a"
        W(f"{esc(name)} & {len(common)} & {m2:+.2f} & {t} & {p} & {m7:+.2f} & "
          f"[{min(d2):+.2f}, {max(d2):+.2f}] " + r"\\")
    W(r"\bottomrule")
    W(r"\end{tabular}")
    W(r"\normalsize")
    W(r"\section*{Supplementary Table S3. The configuration of every arm}")
    W(r"The result files record the full argument vector of the run that produced")
    W(r"them. The table lists the fields that differ between the arms; the rest")
    W(r"are the same for every run.")
    W(r"\footnotesize")
    W(r"\setlength{\tabcolsep}{3pt}")
    W(r"\begin{tabular}{p{92pt} p{48pt} p{30pt} p{30pt} r}")
    W(r"\toprule")
    W(r"Arm & Encoder & Loss & Context & $\lambda_{bin}$ \\")
    W(r"\midrule")
    for arm in sorted(arms):
        files = sorted(glob.glob(os.path.join(args.results, arm + "_s*", "*_result.csv")))
        if not files:
            continue
        with open(files[0]) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        import ast
        a = ast.literal_eval(rows[-1]["args"])
        W(f"{esc(arm)} & {esc(a.get('tokenizer',''))} & "
          f"{esc(a.get('loss','l1'))} & "
          f"{'routed' if a.get('context_route')=='reg' else ('both' if a.get('context') else 'none')} & "
          f"{a.get('lambda_binary', 0.0)} " + r"\\")
    W(r"\bottomrule")
    W(r"\end{tabular}")
    W(r"\normalsize")
    # The full comparison table, with every arm, is moved out of the main
    # article to keep it within the length that the journal asks for.
    _mf = os.path.join(os.path.dirname(args.out), "tables", "tab_main_full.tex")
    if os.path.exists(_mf):
        _txt = open(_mf).read()
        _i = _txt.find(r"\begin{tabular}")
        _j = _txt.find(r"\end{tabular}")
        W(r"\section*{Supplementary Table S4. Every arm against the published anchors}")
        W(r"The complete comparison. The anchors are the rows reported in the")
        W(r"papers themselves and in the released run of MMML; the acceptance")
        W(r"line is the per-column mix fixed before the experiments. Our rows")
        W(r"are single models; the seed count is given and the range over the")
        W(r"seeds is reported in Supplementary Table S1. Values in bold exceed")
        W(r"the acceptance line.")
        W(r"\resizebox{\textwidth}{!}{%")
        for _line in _txt[_i:_j + len(r"\end{tabular}")].splitlines():
            W(_line)
        W(r"}")
    # The component chain and the diagnostic measurements are moved out of
    # the main article to keep it within the length that the journal asks
    # for.  They are reproduced here from the same result files.
    _ab = os.path.join(os.path.dirname(args.out), "tables", "tab_ablation.tex")
    if os.path.exists(_ab):
        _txt = open(_ab).read()
        # keep only the tabular body and the caption text, without the float
        # environment and without the label
        _i = _txt.find(r"\begin{tabular}")
        _j = _txt.find(r"\end{tabular}")
        W(r"\section*{Supplementary Table S5. The component chain on CMU-MOSI}")
        W(r"Each row adds one component to the row above it, so the table reads as")
        W(r"the construction of the final configuration. Seeds are averaged and the")
        W(r"spread over them given in the last two columns; of the five metrics the")
        W(r"mean absolute error is the one for which a smaller number is better.")
        for _line in _txt[_i:_j + len(r"\end{tabular}")].splitlines():
            W(_line)
    _dg = os.path.join(os.path.dirname(args.out), "tables", "tab_diagnostics.tex")
    if os.path.exists(_dg):
        _txt = open(_dg).read()
        _i = _txt.find(r"\begin{tabular}")
        _j = _txt.find(r"\end{tabular}")
        W(r"\section*{Supplementary Table S6. Measurements that shaped the design}")
        W(r"Every value is computed from the archived predictions of the arm named in")
        W(r"the first column. The oracle column uses the test labels and is a")
        W(r"diagnostic upper bound only.")
        for _line in _txt[_i:_j + len(r"\end{tabular}")].splitlines():
            W(_line)
    W(r"\section*{Supplementary Figure S1. The seven-level accuracy against the spread}")
    W(r"\begin{center}")
    W(r"\includegraphics[width=0.55\textwidth]{figures/scale.png}")
    W(r"\end{center}")
    W(r"Each point is one arm at one seed; the horizontal axis is the ratio of")
    W(r"the standard deviation of the predictions to that of the labels, and the")
    W(r"vertical axis the seven-level accuracy. The line is a least squares fit;")
    W(r"the association is weak (Pearson $r=+0.18$ over the points shown), and")
    W(r"the colours separate the arms.")
    W(r"\end{document}")

    with open(args.out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"已写出 {args.out}（{len(L)} 行）")


if __name__ == "__main__":
    main()
