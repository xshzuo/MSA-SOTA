#!/usr/bin/env python3
"""Group every claimed result cell by the configuration recorded inside it.

Attribution by job file is not enough: the same tag can appear in several job
files with different arguments, and several job files can contain cells of the
same arm.  The only reliable key is the configuration the result file itself
records, with the seed removed.

For each group the script reports the seed distribution of the sign-based
headline accuracy (ACC-2(all)) and where the published number sits in it.

Usage:
    python3 scripts/group_by_config.py --results remote_snapshot/results \
        --published-mosi 87.03 --published-mosei 86.03
"""
import argparse
import ast
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from claim_cells import read_csv  # noqa: E402

IGNORE = {"seed", "modelname", "npy_path", "data_dir", "dataset", "data",
          "n_train", "n_valid", "n_test", "args"}


def as_dict(cfg):
    """read_csv already parses the args column into a dict; older files may
    still hold the raw repr."""
    if isinstance(cfg, dict):
        return cfg
    try:
        d = ast.literal_eval(cfg)
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def signature(cfg):
    d = as_dict(cfg)
    if d is None:
        return None
    return tuple(sorted((k, str(v)) for k, v in d.items() if k not in IGNORE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="remote_snapshot/results")
    ap.add_argument("--min-seeds", type=int, default=3)
    args = ap.parse_args()

    groups = {}
    for tag in sorted(os.listdir(args.results)):
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
            sig = signature(cfg)
            if sig is None:
                continue
            _d = as_dict(cfg) or {}
            corpus = str(_d.get("dataset", "mosi"))
            key = (corpus, sig)
            groups.setdefault(key, []).append((tag, m, cfg))

    for (corpus, sig), rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(rows) < args.min_seeds:
            continue
        # Deduplicate by seed: the same configuration and seed may appear in
        # several result directories (the same run launched by different job
        # files).  Those duplicates are a free reproducibility check: they must
        # be bit-identical.
        by_seed = {}
        duplicates = identical = 0
        for tag, m, cfg in rows:
            d = as_dict(cfg) or {}
            seed = d.get("seed")
            if seed in by_seed:
                duplicates += 1
                prev = by_seed[seed][1]
                if all(abs(prev[k] - m[k]) < 1e-12 for k in ("mae", "cor", "acc7", "acc2_all")):
                    identical += 1
                continue
            by_seed[seed] = (tag, m, cfg)
        rows = list(by_seed.values())
        accs = [m["acc2_all"] for _, m, _ in rows]
        seeds = sorted(by_seed)
        mu, sd = st.mean(accs), (st.stdev(accs) if len(accs) > 1 else 0.0)
        pub = 87.03 if corpus == "mosi" else 86.03
        z = (pub - mu) / sd if sd else float("nan")
        pct = 100 * 0.5 * (1 + math.erf(z / math.sqrt(2))) if sd else float("nan")
        interesting = {k: v for k, v in sig if k in
                       ("text_encoder", "npy_selection", "llrd", "warmup_steps", "lr_bert",
                        "alpha", "beta", "eta", "case_norm", "batch_size", "n_tv",
                        "num_layers_cross", "text_kmean_size",
                        "num_layers_counterfactual_attention", "lr_main")}
        print(f"\n=== {corpus} · {len(rows)} 个种子 ===")
        print(f"  配置: {interesting}")
        print(f"  ACC-2: mean {mu:.2f}  sd {sd:.2f}  min {min(accs):.2f}  max {max(accs):.2f}")
        print(f"  种子 {seeds}")
        if duplicates:
            print(f"  重复（同配置同种子，多个结果目录）{duplicates} 次，其中逐位一致 {identical} 次")
        print(f"  论文 {pub} → z={z:+.2f} sd（分位 {pct:.2f}%），比最好种子高 {pub - max(accs):+.2f} pp")
        if sd:
            for n in (5, 30, 300, 3000, 30000):
                print(f"    E[max_{n}] ≈ {mu + sd * _ppf(1 - 1 / n):.2f}")
        print("  逐种子 ACC-2:", " ".join(f"{a:.2f}" for a in sorted(accs)))


def _ppf(p):
    """Inverse normal CDF (Acklam)."""
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
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    e = 0.5 * (1 + math.erf(x / math.sqrt(2))) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


if __name__ == "__main__":
    main()
