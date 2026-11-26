#!/usr/bin/env python3
"""Verify our metric implementation against the reference implementation.

The reference saves the test-set predictions next to its result file, and the
result row carries the prediction file's stem in its `weight_name` column.
Feeding the *paired* prediction file through *our* metric code must reproduce
the numbers in that row exactly; otherwise every later comparison against a
published table would be meaningless.

The pairing is by the artifact, not by position: a cell holds predictions for
both corpora and for several runs, so a positional pairing would compare a
MOSI prediction against a MOSEI row and fail for the wrong reason.

Usage:
    python3 check_metrics.py --results <results root> [--limit 200]
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msa.metrics import compute_metrics   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    # the prediction files and the result files live in different cells, so the
    # pairing has to be by the file stem (which the result row carries)
    index = {}
    for npz_path in glob.glob(os.path.join(args.results, "**", "preds_*.npz"),
                              recursive=True):
        stem = os.path.basename(npz_path)[len("preds_"):-len(".npz")]
        index.setdefault(stem, npz_path)

    checked = ok = 0
    skipped = []
    mismatches = []
    for cell in sorted(os.listdir(args.results)):
        d = os.path.join(args.results, cell)
        if not os.path.isdir(d):
            continue
        for csv_path in sorted(glob.glob(os.path.join(d, "*_result.csv"))):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                stem = (row.get("weight_name") or "").strip()
                if not stem:
                    continue
                npz_path = index.get(stem)
                if npz_path is None:
                    skipped.append((cell, stem))
                    continue
                z = np.load(npz_path, allow_pickle=True)
                keys = {k.lower(): k for k in z.keys()}
                # the reference saves the validation and the test split
                # separately; the reported row is the test split
                if "test_pred" in keys and "test_true" in keys:
                    pk, tk = keys["test_pred"], keys["test_true"]
                elif "preds" in keys and "truths" in keys:
                    pk, tk = keys["preds"], keys["truths"]
                else:
                    skipped.append((cell, stem))
                    continue
                m = compute_metrics(z[pk], z[tk])
                ref = {"mae": float(row["mae"]), "cor": float(row["cor"]),
                       "acc7": float(row["acc7"]), "acc2_all": float(row["acc2_1"]),
                       "acc2_nz": float(row["acc2_2"])}
                checked += 1
                same = (abs(m["mae"] - ref["mae"]) < 1e-6
                        and abs(m["corr"] - ref["cor"]) < 1e-6
                        and abs(m["acc7"] - ref["acc7"]) < 1e-6
                        and abs(m["acc2_all"] - ref["acc2_all"]) < 1e-2
                        and abs(m["acc2_nz"] - ref["acc2_nz"]) < 1e-2)
                if same:
                    ok += 1
                else:
                    mismatches.append((cell, stem,
                                       {k: (round(m[k], 6), round(ref[k], 6))
                                        for k in ref if abs(m[k] - ref[k]) > 1e-2}))
                if checked >= args.limit:
                    break
            if checked >= args.limit:
                break
        if checked >= args.limit:
            break

    print(f"checked {checked} (result row, prediction file) pairs, {ok} reproduced exactly")
    if skipped:
        print(f"skipped {len(skipped)} row(s) without a matching prediction file")
    for cell, stem, diff in mismatches[:10]:
        print(f"  MISMATCH {cell}/{stem}: {diff}")
    if mismatches:
        raise SystemExit(f"{len(mismatches)} pair(s) did not reproduce")
    if ok == 0:
        raise SystemExit("no pair could be checked")
    print("ALL METRIC CHECKS PASSED")


if __name__ == "__main__":
    main()
