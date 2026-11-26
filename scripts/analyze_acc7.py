#!/usr/bin/env python3
"""Why can two systems have the same MAE but different 7-class accuracy?

Acc-7 is the accuracy of round(clip(prediction)) against round(clip(truth)).
It is therefore sensitive to how the predictions are *distributed* around the
class boundaries, not only to their average error: a model whose predictions
are compressed toward the centre has a small MAE (no extreme misses) yet
crosses more class boundaries.

This script compares the prediction scale, the boundary crossing rate and the
per-class accuracy of two systems.
"""
import argparse
import csv
import glob
import os

import numpy as np


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


def clip_round(x):
    return np.clip(np.round(x), -3, 3)


def describe(tag, results):
    got = load_cell(results, tag)
    if not got:
        print(f"{tag}: 没有可用产物")
        return
    row, z = got
    p = np.asarray(z["preds"], float)
    y = np.asarray(z["truths"], float)
    pc, yc = clip_round(p), clip_round(y)
    acc7 = (pc == yc).mean() * 100
    # the scale of the predictions relative to the labels
    scale = p.std() / max(y.std(), 1e-9)
    # how often does the prediction sit within half a class of a boundary
    near = np.abs(p - np.round(p)) > 0.35
    print(f"{tag}")
    print(f"   Acc-7 (recomputed) {acc7:.2f}   MAE {np.abs(p - y).mean():.4f}")
    print(f"   pred std {p.std():.4f}   truth std {y.std():.4f}   ratio {scale:.4f}"
          f"   {'<- predictions compressed' if scale < 0.95 else ''}")
    print(f"   bias (mean pred - mean truth) {p.mean() - y.mean():+.4f}")
    print(f"   within 0.35 of a class boundary: {near.mean() * 100:.1f}% of the samples")
    print(f"   predicted class histogram (rounded): "
          f"{ {int(k): int(v) for k, v in zip(*np.unique(pc, return_counts=True))} }")
    print(f"   true class histogram (rounded):     "
          f"{ {int(k): int(v) for k, v in zip(*np.unique(yc, return_counts=True))} }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    for tag in args.tags:
        describe(tag, args.results)
        print()


if __name__ == "__main__":
    main()
