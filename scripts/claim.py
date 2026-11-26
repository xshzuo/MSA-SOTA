#!/usr/bin/env python3
"""Claim each cell's result by the configuration recorded inside the CSV.

A cell directory may hold files from earlier runs, so "the newest CSV by
mtime" is not a safe way to pick a result.  The reliable key is the `args`
column, which every result file writes with its own full configuration; this
script parses it and matches it against the expected job specification.

The prediction sidecar is named `preds_<corpus>_<tag>_eam<mae>.npz`, so the
claim also carries the artefact it belongs to.

Usage:
    python3 scripts/claim.py --results results --jobs jobs/s2_mosi.txt jobs/s3_mosei.txt
"""
import argparse
import ast
import csv
import os
import sys

METRICS = ("mae", "cor", "acc7", "acc2_all", "acc2_nz")


def parse_job(line):
    """<tag> <seed> <arg> <arg> ...  ->  (tag, spec)"""
    parts = line.split()
    if len(parts) < 2:
        return None
    tag, seed = parts[0], parts[1]
    spec = {"seed": int(seed)}
    for tok in parts[2:]:
        if tok.startswith("--"):
            key, _, val = tok[2:].partition("=")
            if not val:
                continue
            spec[key] = val
    return tag, spec


def parse_args_cell(raw):
    """The `args` column holds a Python dict repr -> a dict."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        d = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    return d if isinstance(d, dict) else {}


def read_cell(results, tag):
    """The claimed metrics of a cell, or None."""
    d = os.path.join(results, tag)
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if not f.endswith("_result.csv"):
            continue
        with open(os.path.join(d, f)) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        row = rows[-1]
        try:
            m = {k: float(row[k]) for k in METRICS}
        except (KeyError, ValueError):
            continue
        args = parse_args_cell(row.get("args"))
        side = [x for x in os.listdir(d) if x.startswith("preds_") and x.endswith(".npz")]
        return m, args, side
    return None


def matches(spec, args):
    """The recorded configuration has to contain the values of the job spec."""
    for k, v in spec.items():
        if k == "seed":
            if str(args.get("seed")) != str(v):
                return False
            continue
        if k not in args:
            return False
        if str(args[k]) != str(v):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    specs = []
    for jf in args.jobs:
        for line in open(jf):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = parse_job(line)
            if p:
                specs.append(p)

    claimed = missed = 0
    for tag, spec in specs:
        got = read_cell(args.results, tag)
        if not got:
            print(f"  [未找到] {tag}")
            missed += 1
            continue
        m, cfg, side = got
        ok = matches(spec, cfg)
        if ok:
            claimed += 1
            print(f"  [认领] {tag:44s} mae {m['mae']:.4f} cor {m['cor']:.4f} "
                  f"acc7 {m['acc7']:5.2f} acc2all {m['acc2_all']:5.2f} "
                  f"acc2nz {m['acc2_nz']:5.2f}  产物 {side[0] if side else '无'}")
        else:
            print(f"  [配置不符] {tag}  期望 {spec}  实际 "
                  f"{ {k: cfg.get(k) for k in spec} }")
            missed += 1
    print(f"\n认领 {claimed} / 共 {len(specs)} 格（{missed} 格未认领）")
    return 0 if claimed else 1


if __name__ == "__main__":
    sys.exit(main())
