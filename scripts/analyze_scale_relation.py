#!/usr/bin/env python3
"""The relation between the measured spread of the predictions and the two
published columns.

The claim under test is that the seven-level accuracy follows the spread of
the predictions while the polarity accuracy moves against it.  The relation
has to be checked at the level of the arm and not only at the level of the
single seed, because the earlier reading used one seed per arm.
"""
import argparse
import glob
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from paired_deltas import load as load_arms  # noqa: E402

ARMS = ["E10_mosi_deberta_bhead_ctxreg", "E6_mosi_deberta_bhead",
        "E12_mosi_bhead_ctxreg_mse", "E13_mosi_deberta_bhead_ctxreg_lb05",
        "E8_mosi_bhead_ctxreg", "E17_mosi_deberta_lb05_mse",
        "E19_mosi_deberta_bhead_bal", "E14_mosi_deberta_bhead_lb05"]


def scale_of(results, arm, seed):
    d = os.path.join(results, f"{arm}_s{seed}")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    arms = load_arms(args.results)
    rows = []
    for arm in ARMS:
        g = arms.get(arm, {})
        for seed in sorted(g):
            r = scale_of(args.results, arm, seed)
            if r is None:
                continue
            rows.append((arm, seed, r, g[seed]["acc7"], g[seed]["acc2_all"]))
    x = np.array([r[2] for r in rows])
    a7 = np.array([r[3] for r in rows])
    a2 = np.array([r[4] for r in rows])
    print(f"跨臂跨种子 n={len(rows)}")
    print(f"  Pearson(尺度比, Acc-7) = {np.corrcoef(x, a7)[0, 1]:+.4f}")
    print(f"  Pearson(尺度比, Acc-2) = {np.corrcoef(x, a2)[0, 1]:+.4f}")

    print("\n臂均值层面（每个臂一个点）：")
    pts = []
    for arm in ARMS:
        sub = [r for r in rows if r[0] == arm]
        if not sub:
            continue
        pts.append((st.mean([r[2] for r in sub]), st.mean([r[3] for r in sub]),
                    st.mean([r[4] for r in sub]), arm, len(sub)))
    for r, a, b, arm, n in sorted(pts):
        print(f"  {arm:42s} n={n} 尺度比 {r:.4f}  Acc-7 {a:6.2f}  Acc-2 {b:6.2f}")
    if len(pts) >= 3:
        px = np.array([p[0] for p in pts])
        p7 = np.array([p[1] for p in pts])
        p2 = np.array([p[2] for p in pts])
        print(f"\n臂均值层面 n={len(pts)}：")
        print(f"  Pearson(尺度比, Acc-7) = {np.corrcoef(px, p7)[0, 1]:+.4f}")
        print(f"  Pearson(尺度比, Acc-2) = {np.corrcoef(px, p2)[0, 1]:+.4f}")


if __name__ == "__main__":
    main()
