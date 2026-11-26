#!/usr/bin/env python3
"""The ensemble reading of an arm, and the seed spread of every arm.

Two questions the per-seed table cannot answer:

  * how much of the gap to an anchor is the seed noise (reported as the
    standard deviation and the range over the seeds);
  * what a deployment that averages the seeds' predictions reads (the
    ensemble of the same arm).  The published rows are single models, so the
    ensemble is reported as an additional reading and never as the headline
    comparison.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))
from msa.metrics import compute_metrics  # noqa: E402


def cells(results):
    arms = {}
    for cell in sorted(os.listdir(results)):
        d = os.path.join(results, cell)
        if not os.path.isdir(d):
            continue
        arm = re.sub(r"_s\d{4}$", "", cell)
        if arm.startswith(("N1", "N2", "N3", "D1_mosi_roberta", "D2_mosi_roberta", "DBG")):
            continue
        arms.setdefault(arm, []).append(d)
    return arms


def load_preds(d, corpus):
    files = [f for f in sorted(glob.glob(os.path.join(d, "preds_*.npz")))
             if not f.endswith("_dev.npz")]
    if not files:
        return None
    z = np.load(files[0])
    keys = [k for k in z.files]
    pred = [k for k in keys if "pred" in k]
    # "truths" does not contain the substring "true"
    true = [k for k in keys if "truth" in k or "true" in k or "label" in k]
    if not pred or not true:
        return None
    return z[pred[0]].reshape(-1), z[true[0]].reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()
    arms = cells(args.results)
    for arm in sorted(arms):
        if args.arms and arm not in args.arms:
            continue
        preds, per_seed = [], []
        for d in arms[arm]:
            got = load_preds(d, None)
            if got is None:
                continue
            p, t = got
            preds.append(p)
            m = compute_metrics(p, t)
            per_seed.append(m)
        if not per_seed:
            continue
        a2 = [m["acc2_all"] for m in per_seed]
        a7 = [m["acc7"] for m in per_seed]
        line = (f"{arm:44s} n={len(per_seed)} "
                f"Acc2 {st.mean(a2):6.2f} (sd {st.stdev(a2) if len(a2)>1 else 0:4.2f}, "
                f"min {min(a2):5.2f} max {max(a2):5.2f})  "
                f"Acc7 {st.mean(a7):6.2f} (sd {st.stdev(a7) if len(a7)>1 else 0:4.2f})")
        if len(preds) > 1:
            ens = np.mean(np.vstack(preds), axis=0)
            m = compute_metrics(ens, per_seed and None or None) if False else None
            # the truth vector is the same in every cell of an arm
            z = np.load(sorted(glob.glob(os.path.join(arms[arm][0], "preds_*.npz")))[0])
            tk = [k for k in z.files if "truth" in k or "true" in k][0]
            truth = z[tk].reshape(-1)
            m = compute_metrics(ens, truth)
            line += (f"  |  集成: Acc2 {m['acc2_all']:6.2f}  Acc2nz {m['acc2_nz']:6.2f} "
                     f"Acc7 {m['acc7']:6.2f}  MAE {m['mae']:.4f}  Corr {m['corr']:.4f}")
        print(line)


if __name__ == "__main__":
    main()
