#!/usr/bin/env python3
"""How much of the binary accuracy is a threshold problem rather than a ranking problem.

MOSI's test split has a different class prior from the training split (positive
rate 0.5701 in train, 0.4475 in test), so a model trained on the training prior
decides with a threshold that is systematically off on the test split.  This
script separates the two effects: the ranking quality (AUC, which no threshold
can change) and the threshold placement (the oracle threshold is an upper
bound that uses the labels and is therefore a diagnostic, not a method).
"""
import argparse
import csv
import glob
import os

import numpy as np


def auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = labels.sum()
    nneg = (~labels).sum()
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load_cell(results, tag):
    for c in glob.glob(os.path.join(results, tag, "*_result.csv")):
        d = os.path.dirname(c)
        with open(c) as f:
            row = list(csv.DictReader(f))[-1]
        npz = sorted(glob.glob(os.path.join(d, "preds_*.npz")))
        if not npz:
            return None
        return row, np.load(npz[0], allow_pickle=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    for tag in args.tags:
        got = load_cell(args.results, tag)
        if not got:
            print(f"{tag}: 没有可用产物")
            continue
        row, z = got
        p = np.asarray(z["preds"], float)
        y = np.asarray(z["truths"], float)
        pos_true = y >= 0
        pos_pred = p >= 0
        acc_def = (pos_pred == pos_true).mean() * 100
        cand = np.unique(np.concatenate([p, [p.min() - 1, p.max() + 1]]))
        best_acc, best_t = max((((p >= t) == pos_true).mean() * 100, t) for t in cand)
        nz = y != 0
        acc_nz_def = ((p[nz] >= 0) == (y[nz] > 0)).mean() * 100
        cand2 = np.unique(p[nz])
        best_nz, best_t2 = max((((p[nz] >= t) == (y[nz] > 0)).mean() * 100, t) for t in cand2)
        print(f"{tag}")
        print(f"   真实正类率 {pos_true.mean():.4f}   模型判正率 {pos_pred.mean():.4f}"
              f"   （训练集正类率 0.5701）")
        print(f"   Acc-2(Has0) 默认阈值 {acc_def:.2f}  oracle 阈值 {best_t:+.4f} -> {best_acc:.2f}"
              f"   上界增益 {best_acc - acc_def:+.2f} pp")
        print(f"   Acc-2(Non0) 默认阈值 {acc_nz_def:.2f}  oracle 阈值 {best_t2:+.4f} -> {best_nz:.2f}"
              f"   上界增益 {best_nz - acc_nz_def:+.2f} pp")
        print(f"   AUC(正类) {auc(p, pos_true):.4f}   MAE {np.abs(p - y).mean():.4f}")


if __name__ == "__main__":
    main()
