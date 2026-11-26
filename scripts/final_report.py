#!/usr/bin/env python3
"""The final acceptance report.

For every arm it reports the mean and the standard deviation over the seeds,
how many seeds clear each acceptance column, and which columns are cleared.
Three anchor sets are used:

  user  the acceptance line of this project (a per-column mix of the
        published rows, kept verbatim)
  paper the strongest published row in the papers themselves (MMML + context)
  repo  the strongest released run MMML's repository reports

The arms are grouped by their cell name, the seed number is stripped, and
every number comes from the result files.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st

ANCHORS = {
    "user": {
        "mosi": {"acc2_all": 87.51, "acc2_nz": 88.26, "acc7": 50.34,
                 "mae": 0.638, "cor": 0.838},
        "mosei": {"acc2_all": 87.24, "acc2_nz": 88.02, "acc7": 55.74,
                  "mae": 0.4922, "cor": 0.8137},
    },
    "paper": {
        "mosi": {"acc2_all": 87.51, "acc2_nz": 89.69, "acc7": 50.34,
                 "mae": 0.5831, "cor": 0.8693},
        "mosei": {"acc2_all": 87.24, "acc2_nz": 88.02, "acc7": 55.74,
                  "mae": 0.4922, "cor": 0.8137},
    },
    "repo": {
        "mosi": {"acc2_all": 88.48, "acc2_nz": 90.35, "acc7": 52.72,
                 "mae": 0.5573, "cor": 0.8824},
        "mosei": {"acc2_all": 87.24, "acc2_nz": 88.02, "acc7": 55.74,
                  "mae": 0.4922, "cor": 0.8137},
    },
}
METRICS = ["acc2_all", "acc2_nz", "acc7", "mae", "cor"]
LABEL = {"acc2_all": "Acc-2(Has0)", "acc2_nz": "Acc-2(Non0)", "acc7": "Acc-7",
         "mae": "MAE", "cor": "Corr"}
HIGHER = {"acc2_all": True, "acc2_nz": True, "acc7": True, "mae": False, "cor": True}
SKIP_PREFIX = ("N1", "N2", "N3", "D1_mosi_roberta", "D2_mosi_roberta", "DBG", "P0", "P1")


def collect(results):
    arms = {}
    for cell in sorted(os.listdir(results)):
        d = os.path.join(results, cell)
        if not os.path.isdir(d) or cell.startswith(SKIP_PREFIX):
            continue
        arm = re.sub(r"_s\d{4}$", "", cell)
        for f in sorted(glob.glob(os.path.join(d, "*_result.csv"))):
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    arms.setdefault(arm, []).append(
                        {k: float(row[k]) for k in METRICS})
    return arms


def corpus_of(arm):
    return "mosei" if "_mosei" in arm else "mosi"


def report(arms, anchor_name, only=None):
    anchors = ANCHORS[anchor_name]
    print(f"\n{'='*104}\n锚点集合：{anchor_name}\n{'='*104}")
    print(f"{'臂':42s} {'n':>2s} {'Acc2(Has0)':>11s} {'Acc2(Non0)':>11s} "
          f"{'Acc-7':>9s} {'MAE':>9s} {'Corr':>8s}  过线")
    best = {}
    for arm in sorted(arms, key=lambda a: -st.mean([m["acc2_all"] for m in arms[a]])):
        g = arms[arm]
        if only and corpus_of(arm) != only:
            continue
        anc = anchors[corpus_of(arm)]
        means = {k: st.mean([m[k] for m in g]) for k in METRICS}
        sds = {k: (st.stdev([m[k] for m in g]) if len(g) > 1 else 0.0) for k in METRICS}
        cleared = []
        for k in METRICS:
            thr = anc[k]
            good = sum((m[k] > thr) if HIGHER[k] else (m[k] < thr) for m in g)
            if means[k] > thr if HIGHER[k] else means[k] < thr:
                cleared.append(f"{LABEL[k]}({good}/{len(g)})")
        print(f"{arm:42s} {len(g):2d} "
              f"{means['acc2_all']:6.2f}±{sds['acc2_all']:4.2f} "
              f"{means['acc2_nz']:6.2f}±{sds['acc2_nz']:4.2f} "
              f"{means['acc7']:5.2f}±{sds['acc7']:4.2f} "
              f"{means['mae']:6.4f}±{sds['mae']:5.4f} "
              f"{means['cor']:6.4f}±{sds['cor']:6.4f}  "
              f"{len(cleared)}/5 {', '.join(cleared)}")
        n_cleared = len(cleared)
        if arm not in best or n_cleared > best[arm][0]:
            best[arm] = (n_cleared, means)
    return best


def frontier(arms, only="mosi"):
    print(f"\n{'='*104}\nPareto 前沿（{only}）：Acc-2(Has0) 与 Acc-7 不可兼得\n{'='*104}")
    pts = []
    for arm, g in arms.items():
        if corpus_of(arm) != only:
            continue
        pts.append((st.mean([m["acc2_all"] for m in g]),
                    st.mean([m["acc7"] for m in g]), arm, len(g)))
    pts.sort(key=lambda p: -p[0])
    print(f"{'Acc-2(Has0)':>11s} {'Acc-7':>8s}  {'臂':42s} n")
    for a, c7, arm, n in pts:
        mark = ""
        if a > ANCHORS["user"][only]["acc2_all"]:
            mark += " Acc-2过线"
        if c7 > ANCHORS["user"][only]["acc7"]:
            mark += " Acc-7过线"
        print(f"{a:11.2f} {c7:8.2f}  {arm:42s} {n}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--only", default=None, choices=[None, "mosi", "mosei"])
    ap.add_argument("--anchors", nargs="+", default=["user", "paper", "repo"])
    args = ap.parse_args()
    arms = collect(args.results)
    best_overall = {}
    for a in args.anchors:
        best = report(arms, a, args.only)
        for arm, (n, means) in best.items():
            if arm not in best_overall or n > best_overall[arm][0]:
                best_overall[arm] = (n, means)
    frontier(arms, args.only or "mosi")
    print(f"\n{'='*104}\n最强的一臂（按通过列数）\n{'='*104}")
    for arm, (n, means) in sorted(best_overall.items(), key=lambda kv: -kv[1][0]):
        if n < 4:
            continue
        print(f"{arm}: {n}/5 列通过  " +
              "  ".join(f"{LABEL[k]} {means[k]:.4f}" for k in METRICS))


if __name__ == "__main__":
    main()
