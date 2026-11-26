#!/usr/bin/env python3
"""Fit the output scale on the development split and apply it to the test split.

The large encoder's predictions are compressed toward the centre: the mean
absolute error gets smaller but more samples land in the wrong 7-class bin.
A single multiplicative factor restores the scale.

Two properties make this a legitimate, minimal correction:
  * only one parameter is fitted, on the *development* split, with the
    development labels -- the test labels are never touched;
  * a multiplicative factor cannot change the sign of a prediction, so the
    binary accuracies are bit-for-bit unchanged.

Usage:
    python3 scripts/calibrate_scale.py --results results --cells E10b_...
"""
import argparse
import csv
import glob
import os

import numpy as np


def cr(x):
    return np.clip(np.round(x), -3, 3)


def metrics(p, y):
    nz = y != 0
    return {"acc7": float((cr(p) == cr(y)).mean() * 100),
            "acc2_all": float(((p >= 0) == (y >= 0)).mean() * 100),
            "acc2_nz": float(((p[nz] >= 0) == (y[nz] > 0)).mean() * 100),
            "mae": float(np.abs(p - y).mean())}


def load_pairs(results, tag):
    """Return (dev_pred, dev_true, test_pred, test_true) or None."""
    dev = test = None
    for c in glob.glob(os.path.join(results, tag, "*_result.csv")):
        d = os.path.dirname(c)
        all_npz = sorted(glob.glob(os.path.join(d, "preds_*.npz")))
        dev_files = [f for f in all_npz if f.endswith("_dev.npz")]
        test_files = [f for f in all_npz if not f.endswith("_dev.npz")]
        if not test_files or not dev_files:
            return None
        dev_f = dev_files[0]
        dv = np.load(dev_f, allow_pickle=True)
        te = np.load(test_files[0], allow_pickle=True)
        dev = (np.asarray(dv["preds"], float), np.asarray(dv["truths"], float))
        test = (np.asarray(te["preds"], float), np.asarray(te["truths"], float))
    return dev, test if dev else None


def fit_scale(pred, true, grid):
    """The scale that minimises the MAE of a * pred on the given split."""
    best = (None, None)
    for a in grid:
        m = float(np.abs(a * pred - true).mean())
        if best[0] is None or m < best[0]:
            best = (m, a)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--cells", required=True, help="common prefix of the cells")
    ap.add_argument("--grid", default="0.80,1.25",
                    help="the range the scale is searched in")
    args = ap.parse_args()
    lo, hi = (float(v) for v in args.grid.split(","))
    grid = np.round(np.arange(lo, hi + 1e-9, 0.01), 3)

    tags = sorted(d for d in os.listdir(args.results)
                  if d.startswith(args.cells)
                  and os.path.isdir(os.path.join(args.results, d)))
    if not tags:
        print(f"没有匹配的格：{args.cells}")
        return 1
    deltas = []
    for tag in tags:
        got = load_pairs(args.results, tag)
        if not got:
            print(f"{tag}: 缺少 dev 预测，无法拟合")
            continue
        (dp, dy), (tp, ty) = got
        a = fit_scale(dp, dy, grid)
        before, after = metrics(tp, ty), metrics(a * tp, ty)
        print(f"{tag}: dev 拟合的尺度 a = {a:.2f}"
              f"（dev 的 std 比 {dp.std()/dy.std():.3f}）")
        print(f"   Acc-7 {before['acc7']:5.2f} -> {after['acc7']:5.2f} "
              f"({after['acc7']-before['acc7']:+.2f})   "
              f"Acc-2 {before['acc2_all']:5.2f} -> {after['acc2_all']:5.2f} "
              f"({after['acc2_all']-before['acc2_all']:+.2f})   "
              f"Acc-2nz {before['acc2_nz']:5.2f} -> {after['acc2_nz']:5.2f} "
              f"({after['acc2_nz']-before['acc2_nz']:+.2f})   "
              f"MAE {before['mae']:.4f} -> {after['mae']:.4f} "
              f"({after['mae']-before['mae']:+.4f})")
        deltas.append(after["acc7"] - before["acc7"])
        # the binary accuracies must be bit-for-bit unchanged
        assert before["acc2_all"] == after["acc2_all"], "乘性校准不应改变 Acc-2"
        assert before["acc2_nz"] == after["acc2_nz"], "乘性校准不应改变 Acc-2(nz)"
    if deltas:
        print(f"\nΔAcc-7 均值 {np.mean(deltas):+.3f}（{len(deltas)} 个种子，"
              f"判据 ≥ +0.5）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
