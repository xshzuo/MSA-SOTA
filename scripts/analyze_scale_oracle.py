#!/usr/bin/env python3
"""How much is the 7-class accuracy limited by a compressed output scale?

The predictions of the large encoder are systematically compressed toward the
centre (pred std / truth std ~ 0.93), which lowers the MAE but pushes more
samples across the class boundaries.  This script fits an affine correction
on the *test* set to measure the size of the opportunity.  That fit is an
oracle (it uses the test labels) and is therefore an upper bound and a
diagnostic only -- it must not be used as a method.
"""
import argparse
import glob
import os

import numpy as np


def load_cell(results, tag):
    for c in glob.glob(os.path.join(results, tag, "*_result.csv")):
        z = sorted(glob.glob(os.path.join(os.path.dirname(c), "preds_*.npz")))
        if not z:
            return None
        d = np.load(z[0], allow_pickle=True)
        return np.asarray(d["preds"], float), np.asarray(d["truths"], float)
    return None


def cr(x):
    return np.clip(np.round(x), -3, 3)


def metrics(p, y):
    nz = y != 0
    return ((cr(p) == cr(y)).mean() * 100,
            ((p >= 0) == (y >= 0)).mean() * 100,
            ((p[nz] >= 0) == (y[nz] > 0)).mean() * 100,
            float(np.abs(p - y).mean()))


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
        p, y = got
        a7, a2, a2n, mae = metrics(p, y)
        best = None
        for a in np.arange(0.88, 1.33, 0.02):
            for b in np.arange(-0.40, 0.41, 0.02):
                m = metrics(a * p + b, y)
                if best is None or m[0] > best[2][0]:
                    best = (a, b, m)
        a, b, m = best
        print(f"{tag}")
        print(f"   原始:  Acc7 {a7:5.2f}  Acc2 {a2:5.2f}  Acc2nz {a2n:5.2f}  MAE {mae:.4f}")
        print(f"   oracle 仿射 (a={a:.2f}, b={b:+.2f}): Acc7 {m[0]:5.2f} ({m[0]-a7:+.2f})  "
              f"Acc2 {m[1]:5.2f} ({m[1]-a2:+.2f})  Acc2nz {m[2]:5.2f} ({m[2]-a2n:+.2f})  "
              f"MAE {m[3]:.4f} ({m[3]-mae:+.4f})")


if __name__ == "__main__":
    main()
