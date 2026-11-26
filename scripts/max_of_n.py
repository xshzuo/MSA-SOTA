#!/usr/bin/env python3
"""How many runs of the released configuration would it take to reach the
published number?

The reproduction gap on the sign-based headline accuracy (ACC-2) is not
explained by any of the candidates tested so far, so the remaining hypothesis
is a *protocol* difference: the paper reports one number, and the release
selects the reported checkpoint by the **test** loss.  A number obtained under
that protocol is the maximum of a run-to-run distribution, and the question is
how far into the tail the published value sits.

This script takes the seed sweep of the released default and reports:

  * the empirical distribution (mean, sd, range, quantiles),
  * the z-score and the percentile of the published number,
  * the max-of-N curve: the running maximum as seeds are added, with the
    expected maximum for a Gaussian with the same mean and sd, so the number
    of runs the published value corresponds to can be read off.

Usage:
    python3 scripts/max_of_n.py --results remote_snapshot/results \
        --jobs jobs/p24_mosi_seed_sweep.txt --published 87.03
"""
import argparse
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from claim_cells import read_csv, parse_job, matches  # noqa: E402


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    """Acklam's inverse normal CDF."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    # one Newton step
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def expected_max(n, mean, sd):
    """E[max of n iid N(mean, sd)] via the standard order-statistic integral."""
    if n <= 0:
        return float("nan")
    if n == 1:
        return mean
    # E[max] = mean + sd * E[max of n standard normals]
    #        = mean + sd * n * int_{-inf}^{inf} z * phi(z) * Phi(z)^{n-1} dz
    # evaluated numerically on a fine grid
    lo, hi, steps = -8.0, 8.0, 200000
    h = (hi - lo) / steps
    total = 0.0
    for i in range(steps + 1):
        z = lo + i * h
        phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        cdf = norm_cdf(z)
        if cdf <= 0:
            continue
        total += z * phi * (cdf ** (n - 1))
    total *= h
    return mean + sd * n * total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="remote_snapshot/results")
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--published", type=float, required=True)
    ap.add_argument("--metric", default="acc2_all")
    args = ap.parse_args()

    specs = {}
    for jf in args.jobs:
        for line in open(jf):
            p = parse_job(line.strip())
            if p:
                specs[p[0]] = p[1]

    rows = []
    for tag, spec in specs.items():
        d = os.path.join(args.results, tag)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith("_result.csv"):
                continue
            got = read_csv(os.path.join(d, f))
            if not got:
                continue
            m, n, cfg = got
            if matches(spec, cfg):
                rows.append((tag, m[args.metric], m.get("mae"), m.get("acc7")))
                break

    if len(rows) < 2:
        print(f"only {len(rows)} cell(s) claimed -- cannot characterise a distribution")
        return

    vals = [r[1] for r in rows]
    vals_sorted = sorted(vals)
    mean = st.mean(vals)
    sd = st.stdev(vals)
    z = (args.published - mean) / sd
    pct = norm_cdf(z) * 100

    print(f"格子数 n = {len(vals)}   指标 = {args.metric}")
    print(f"  mean {mean:.2f}   sd {sd:.2f}   min {min(vals):.2f}   max {max(vals):.2f}")
    print(f"  中位数 {st.median(vals):.2f}")
    q = st.quantiles(vals, n=4)
    print(f"  四分位 {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}")
    print(f"  论文值 {args.published:.2f} → z = {z:+.2f} sd，正态分位 {pct:.2f}%")
    print(f"  与最好种子相差 {args.published - max(vals):+.2f} pp")

    print("\n  max-of-N 曲线（观测最大值 vs 同均值同方差的正态期望最大值）")
    running = []
    print("   N   观测 max   正态期望 max")
    for n in range(1, len(vals) + 1):
        running.append(max(vals[:n]))
        if n in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, len(vals)):
            if n <= len(vals):
                print(f"  {n:3d}   {running[-1]:7.2f}   {expected_max(n, mean, sd):9.2f}")

    # how many runs would the expected maximum need to reach the published value?
    n = 1
    while n < 100000 and expected_max(n, mean, sd) < args.published:
        n += 1
    print(f"\n  按正态期望，发表值对应约 n ≈ {n} 次运行（E[max_n] 首次越过 {args.published}）")
    if max(vals) >= args.published:
        print("  观测到的最大值已经达到发表值 ✓")
    else:
        print(f"  观测最大值 {max(vals):.2f} 仍低于发表值（差 {args.published - max(vals):.2f} pp）")


if __name__ == "__main__":
    main()
