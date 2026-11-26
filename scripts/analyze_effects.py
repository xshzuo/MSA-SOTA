#!/usr/bin/env python3
"""The mechanism analysis of the attributable effect.

The claim the gate has to support is not "the gate helps" but "the measured
effect is informative": a stream whose removal changes the prediction a lot is
a stream the sample relies on, and a stream whose removal changes nothing is
a stream the sample does not rely on.  This script measures that directly on
the saved artefacts:

  * how much each stream changes the prediction when it is removed
  * whether the measured effect predicts the *damage* the removal does
    (the error grows, the sample flips its class)
  * whether the effect separates the samples the model gets right from the
    ones it gets wrong

Usage:
    python3 scripts/analyze_effects.py --npz results/<cell>/preds_*.npz
"""
import argparse
import glob

import numpy as np


def auc(scores, labels):
    """Rank-based AUC with tie handling."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = labels.sum()
    n_neg = (~labels).sum()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def report(name, npz_path):
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.keys())
    if "eff_a" not in keys:
        print(f"[{name}] 该结果没有效应记录（不是 ASF 臂）")
        return
    preds = z["preds"]
    truths = z["truths"]
    err = np.abs(preds - truths)
    correct = (np.round(np.clip(preds, -3, 3)) == np.round(np.clip(truths, -3, 3)))
    print(f"\n=== {name} ===")
    print(f"   n={len(preds)}  测试 MAE {err.mean():.4f}")
    for stream, eff_key, drop_key in (("acoustic", "eff_a", "pred_no_audio"),
                                      ("visual", "eff_v", "pred_no_vision"),
                                      ("context", "eff_c", "pred_no_context")):
        if eff_key not in keys:
            continue
        eff = np.asarray(z[eff_key], dtype=float)
        line = (f"   {stream:9s} 效应 均值 {eff.mean():.4f} 中位 {np.median(eff):.4f} "
                f"p90 {np.percentile(eff, 90):.4f}")
        if drop_key in keys:
            drop = np.asarray(z[drop_key], dtype=float)
            d_err = np.abs(drop - truths) - err          # the damage of the removal
            flips = (np.round(np.clip(drop, -3, 3)) != np.round(np.clip(preds, -3, 3)))
            line += (f" | 移除后 MAE 变化 {d_err.mean():+.4f}，"
                     f"七分类翻转 {flips.mean() * 100:.2f}%")
            if d_err.std() > 0:
                r = np.corrcoef(eff, d_err)[0, 1]
                line += f" | 效应↔伤害 相关 r={r:+.3f}"
            line += f" | 伤害>0 的样本里 效应均值 {eff[d_err > 0].mean():.4f}" \
                f" vs 其余 {eff[d_err <= 0].mean():.4f}"
            a = auc(eff, d_err > 0)
            line += f" | 用效应预测伤害的 AUC {a:.3f}"
        print(line)
    # does the effect separate the correct from the incorrect samples?
    for stream, eff_key in (("acoustic", "eff_a"), ("visual", "eff_v"),
                            ("context", "eff_c")):
        if eff_key not in keys:
            continue
        eff = np.asarray(z[eff_key], dtype=float)
        a = auc(eff, ~correct)
        print(f"   {stream:9s} 用效应预测「模型是否出错」的 AUC {a:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    args = ap.parse_args()
    for pat in args.npz:
        for path in sorted(glob.glob(pat)):
            report(path.split("/")[-2] if "/" in path else path, path)


if __name__ == "__main__":
    main()
