#!/usr/bin/env python3
"""Recompute the seed-matched deltas quoted in docs/08_CLAIMS.md.

Every delta in the claims document is a paired difference over the same
seeds, not a difference of two means with different seed sets.  This script
prints the paired numbers so the document can be checked against the data
instead of against memory.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st

COLS = ["acc2_all", "acc2_nz", "acc7", "mae", "cor"]
LABEL = {"acc2_all": "Acc-2(Has0)", "acc2_nz": "Acc-2(Non0)", "acc7": "Acc-7",
         "mae": "MAE", "cor": "Corr"}
HIGHER = {"acc2_all": True, "acc2_nz": True, "acc7": True, "mae": False, "cor": True}



def _betacf(a, b, x):
    """The continued fraction for the incomplete beta function (Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def betai(a, b, x):
    """The regularised incomplete beta function (Numerical Recipes)."""
    import math
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """The two-sided p value of a t statistic, from the regularised beta."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return max(0.0, min(1.0, betai(df / 2.0, 0.5, x)))


def ttest_p(diffs):
    """A two-sided paired t test: returns (t, df, p)."""
    n = len(diffs)
    if n < 2:
        return float("nan"), n - 1, float("nan")
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    sd = var ** 0.5
    if sd == 0:
        return float("nan"), n - 1, float("nan")
    t = m / (sd / n ** 0.5)
    return t, n - 1, t_sf_two_sided(t, n - 1)


def load(results):
    arms = {}
    for cell in sorted(os.listdir(results)):
        d = os.path.join(results, cell)
        if not os.path.isdir(d):
            continue
        m = re.search(r"_s(\d{4})$", cell)
        if not m:
            continue
        arm = cell[:m.start()]
        seed = m.group(1)
        for f in sorted(glob.glob(os.path.join(d, "*_result.csv"))):
            with open(f) as fh:
                rows = list(csv.DictReader(fh))
            if rows:
                r = rows[-1]
                arms.setdefault(arm, {})[seed] = {k: float(r[k]) for k in COLS}
    return arms


def paired(arms, a, b):
    """The paired difference a - b over the seeds both arms have."""
    ga, gb = arms.get(a, {}), arms.get(b, {})
    common = sorted(set(ga) & set(gb))
    if not common:
        return None
    out = {}
    for k in COLS:
        d = [ga[s][k] - gb[s][k] for s in common]
        out[k] = (st.mean(d), d)
    return out, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="A:B pairs, A minus B")
    args = ap.parse_args()
    arms = load(args.results)
    for pair in args.pairs:
        a, b = pair.split(":")
        got = paired(arms, a, b)
        if got is None:
            print(f"{a} - {b}: 没有共同种子")
            continue
        out, common = got
        print(f"\n{a} − {b}   配对种子 {', '.join(common)}  (n={len(common)})")
        for k in COLS:
            m, d = out[k]
            tv, df, pv = ttest_p(d)
            rng = f"[{min(d):+.2f}, {max(d):+.2f}]"
            if pv == pv:
                t = f"  t={tv:+.2f}  df={df}  P={pv:.4f}"
            elif len(d) > 1:
                t = "  P=n/a（逐种子差值相同）"
            else:
                t = ""
            print(f"  Δ{LABEL[k]:12s} {m:+8.4f}{t}  range {rng}")
        print(f"  {a} 均值: " + "  ".join(f"{LABEL[k]} {st.mean([arms[a][s][k] for s in common]):.4f}" for k in COLS))
        print(f"  {b} 均值: " + "  ".join(f"{LABEL[k]} {st.mean([arms[b][s][k] for s in common]):.4f}" for k in COLS))


if __name__ == "__main__":
    main()
