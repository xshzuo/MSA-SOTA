#!/usr/bin/env python3
"""Pick the correct result file for a cell.

The training script writes its result file into the directory named by
``--out`` but with the file name of the tag, so several runs of the same
cell land in the same place and overwrite each other.  The prediction files
carry the validation error of the run in their name, so the run whose
archived predictions belong to the result file can be identified.

The rule used here is the one that matters for the comparison: the result
file whose recorded arguments match the arguments the arm was produced with.
"""
import ast
import csv
import glob
import os
import sys

RESULTS = "results"


def args_of(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    try:
        return ast.literal_eval(rows[-1]["args"])
    except Exception:
        return None


def pick(tag, required):
    """Return the result file whose arguments contain every required item."""
    files = [f for f in glob.glob(f"{RESULTS}/{tag}*_result.csv")
             if "old_flat" not in f]
    good = []
    for f in files:
        a = args_of(f)
        if not a:
            continue
        if all(a.get(k) == v for k, v in required.items()):
            good.append(f)
    if not good:
        return None, files
    return good[0], files


if __name__ == "__main__":
    # the arms were produced with the maximum sequence length 64 and the
    # case rule that folds shouted transcriptions
    req = {"max_len": 64, "case_rule": "upper_words"}
    for tag in sys.argv[1:]:
        f, allf = pick(tag, req)
        print(f"{tag}:")
        for x in allf:
            a = args_of(x) or {}
            ok = all(a.get(k) == v for k, v in req.items())
            print(f"   {'采用' if ok else '弃用'} {x}  max_len={a.get('max_len')} case={a.get('case_rule')}")
        if f:
            r = list(csv.DictReader(open(f)))[-1]
            print(f"   -> acc2={float(r['acc2_all']):.2f} acc7={float(r['acc7']):.2f}")
