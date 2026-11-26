#!/usr/bin/env python3
"""Build the paper tables from the result files.

Every number in the tables comes from a result CSV in the results tree; the
published anchors are the only numbers typed in by hand, and they carry the
source.  Nothing is transcribed by hand.

Usage:
    python3 scripts/make_tables.py --results results --out paper/tables
"""
import argparse
import csv
import os
import statistics as st
import sys

# the published anchors (verified against the papers; the MOSI/MOSEI rows of
# MMML are transcribed from the PDF tables and are re-checked before submission)
# The anchors below were re-read cell by cell from the published tables
# (MMML: arXiv 2308.00264v4 Table 1 and the NAACL 2024 camera-ready PDF agree;
# DPDF-LQ: EMNLP 2025 main proceedings Table).  MMML's numbers are averages
# over three runs; the repository's released CME+context run reports a slightly
# higher MOSI average (88.48 / 90.35 / 52.72 / 0.5573 / 0.8824) and is listed
# separately as the strongest released MOSI row.
ANCHORS = {
    "mosi": [
        ("MMML (NAACL 2024)", 85.91, 88.16, 48.25, 0.6429, 0.838),
        ("MMML + context (NAACL 2024)", 87.51, 89.69, 50.34, 0.5831, 0.8693),
        ("MMML + context, released runs", 88.48, 90.35, 52.72, 0.5573, 0.8824),
        ("DPDF-LQ, BERT-base (EMNLP 2025)", 84.11, 86.59, 47.65, 0.682, 0.803),
        ("DPDF-LQ, BERT-large (EMNLP 2025)", 86.15, 88.26, 48.63, 0.638, 0.838),
        ("H2CAN (ACL 2025, neg/pos)", None, 87.20, None, None, None),
    ],
    "mosei": [
        ("MMML (NAACL 2024)", 86.32, 86.73, 54.95, 0.5174, 0.7908),
        ("MMML + context (NAACL 2024)", 87.24, 88.02, 55.74, 0.4922, 0.8137),
        ("DPDF-LQ, BERT-base (EMNLP 2025)", 84.24, 84.91, 52.67, 0.562, 0.766),
        ("DPDF-LQ, BERT-large (EMNLP 2025)", 84.40, 85.40, 53.95, 0.545, 0.777),
    ],
}
METRICS = ["acc2_all", "acc2_nz", "acc7", "mae", "cor"]
LABEL = {"acc2_all": "Acc-2(Has0)", "acc2_nz": "Acc-2(Non0)", "acc7": "Acc-7",
         "mae": "MAE", "cor": "Corr"}


def read_cells(results):
    out = {}
    for cell in sorted(os.listdir(results)):
        d = os.path.join(results, cell)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith("_result.csv"):
                continue
            with open(os.path.join(d, f)) as fh:
                for row in csv.DictReader(fh):
                    out.setdefault(cell, []).append(
                        {k: float(row[k]) for k in METRICS} | {
                            "seed": row.get("args", ""), "tag": row.get("tag", cell)})
    return out


def arm_of(cell):
    """The arm prefix of a cell tag: <arm>_<seed>."""
    return cell.rsplit("_s", 1)[0] if "_s" in cell else cell


def fmt_mean_sd(vals, fmt="{:.2f}"):
    if not vals:
        return "--"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(st.mean(vals))} $\\pm$ {fmt.format(st.stdev(vals))}"


def latex_escape(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="write the .tex files here")
    args = ap.parse_args()

    cells = read_cells(args.results)
    arms = {}
    for cell, rows in cells.items():
        arms.setdefault(arm_of(cell), []).extend(rows)

    print(f"{len(cells)} cells, {len(arms)} arms\n")
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Arm & seeds & Acc-2(Has0) & Acc-2(Non0) & Acc-7 & MAE & Corr \\",
             r"\midrule"]
    for arm in sorted(arms):
        rows = arms[arm]
        vals = {m: [r[m] for r in rows] for m in METRICS}
        lines.append(f"{latex_escape(arm)} & {len(rows)} & " +
                     " & ".join(fmt_mean_sd(vals[m], "{:.4f}" if m in ("mae", "cor") else "{:.2f}")
                                for m in METRICS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    table = "\n".join(lines)
    print(table)

    # the anchor table
    alines = [r"\begin{tabular}{lccccc}", r"\toprule",
              r"System & Acc-2(Has0) & Acc-2(Non0) & Acc-7 & MAE & Corr \\",
              r"\midrule"]
    for corpus in ("mosi", "mosei"):
        for name, a, b, c, m, co in ANCHORS[corpus]:
            fmt = lambda v, f="{:.2f}": "--" if v is None else f.format(v)
            alines.append(f"{latex_escape(name)} & {fmt(a)} & {fmt(b)} & {fmt(c)} & "
                          f"{fmt(m, '{:.3f}')} & {fmt(co, '{:.3f}')} " + r"\\")
        alines.append(r"\midrule")
    alines += [r"\bottomrule", r"\end{tabular}", ""]
    print("\n".join(alines))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "table_arms.tex"), "w") as f:
            f.write(table)
        with open(os.path.join(args.out, "table_anchors.tex"), "w") as f:
            f.write("\n".join(alines))
        print(f"\n[ok] written to {args.out}")


if __name__ == "__main__":
    main()
