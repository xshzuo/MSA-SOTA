"""Claim each grid cell's result by the configuration recorded INSIDE the CSV.

Why this file exists
--------------------
The grid chains copy whatever is in `pre_trained_best_models_mosei/` into the
cell's result directory, and the prediction `.npz` files are never deleted,
so a cell directory can hold many files from earlier runs.  Picking "the
newest CSV by mtime" is not safe: two cells were reported with identical
numbers because they happened to share a file.

The reliable key is the `args` column that the released code writes into every
result CSV: it records the full configuration namespace, including
`dataset`, `seed`, `text_encoder`, `npy_selection`, `lr_bert`, `llrd`,
`warmup_steps`, `eta` and `token_weights`.  This script parses that, matches
it against the expected cell specification, and reports the metrics only for
cells whose configuration was actually found.

Usage:
    python3 claim_cells.py --results <dir> --jobs <jobfile> [...]
"""
import argparse
import ast
import csv
import glob
import os
import re
import sys

METRICS = ("mae", "cor", "acc7", "acc2_all", "acc2_nz")
COLS = (1, 2, 3, 4, 5)

# the keys of the configuration that identify a cell
KEYS = ("dataset", "seed", "text_encoder", "npy_selection", "lr_bert",
        "llrd", "warmup_steps", "eta", "token_weights")


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return None
    header = [h.strip() for h in rows[0]]
    idx = {h: i for i, h in enumerate(header)}
    d = rows[-1]
    out = {}
    for k, i in zip(METRICS, COLS):
        try:
            out[k] = float(d[i])
        except (ValueError, IndexError):
            return None
    name = None
    if "weight_name" in idx and idx["weight_name"] < len(d):
        name = d[idx["weight_name"]].strip().strip("[]'\"")
    args = {}
    if "args" in idx and idx["args"] < len(d):
        raw = d[idx["args"]]
        # the cell holds a LIST containing a Namespace repr: "[Namespace(...)]"
        m = re.search(r"Namespace\((.*)\)", raw, re.S)
        if m:
            # ast.literal_eval cannot evaluate a Namespace(...) call, so the
            # key=value pairs are split by hand (values are literals).
            args = {}
            body = m.group(1)
            depth = 0
            cur = ""
            parts = []
            in_str = None
            for ch in body:
                if in_str:
                    cur += ch
                    if ch == in_str:
                        in_str = None
                    continue
                if ch in "'\"":
                    in_str = ch
                    cur += ch
                    continue
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                if ch == "," and depth == 0:
                    parts.append(cur)
                    cur = ""
                    continue
                cur += ch
            if cur.strip():
                parts.append(cur)
            for part in parts:
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    args[k] = ast.literal_eval(v)
                except Exception:
                    args[k] = v
            # keep every recorded flag: the job specification is compared
            # against all of them (a whitelist would silently drop flags such
            # as --d_vh / --d_ah and then no cell could ever be claimed)
    return out, name, args


def parse_job(line):
    parts = line.split()
    if not parts or parts[0].startswith("#"):
        return None
    tag, seed = parts[0], parts[1]
    cfg = {"seed": int(seed)}
    i = 2
    while i < len(parts):
        tok = parts[i]
        if tok.startswith("--"):
            key = tok[2:]
            if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                val = parts[i + 1]
                cfg[key] = val
                i += 2
                continue
            cfg[key] = True
        i += 1
    return tag, cfg


def norm(v):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return "True"
    s = str(v)
    try:
        f = float(s)
        return f"{f:g}"
    except ValueError:
        return s


def matches(cfg_job, cfg_args):
    """True when the recorded configuration agrees with the job specification."""
    for k, v in cfg_job.items():
        if k == "seed":
            if int(cfg_args.get("seed", -1)) != int(v):
                return False
            continue
        a = norm(cfg_args.get(k))
        j = norm(v)
        if k in ("llrd", "warmup_steps", "lr_bert", "eta", "d_vh", "d_ah",
                 "text_kmean_size", "alpha", "beta", "num_epochs", "patience"):
            if j is None:
                j = {"llrd": "1", "warmup_steps": "0"}.get(k)
            if a is None or j is None:
                return False
            if abs(float(a) - float(j)) > 1e-12:
                return False
            continue
        if a != j:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # Every CSV under the results root, indexed by its recorded configuration.
    # The cell's OWN directory is the authority: a directory can also hold
    # files copied in from earlier runs, and different runs that differ only
    # in a code patch (not in a CLI flag) have identical `args`, so a global
    # search can claim the wrong file.  The directory plus the args together
    # are unambiguous.
    index = []
    for path in glob.glob(os.path.join(args.results, "**", "*_result.csv"), recursive=True):
        got = read_csv(path)
        if got is None:
            continue
        metrics, name, cfg = got
        if not cfg:
            continue
        index.append((path, metrics, name, cfg))
    print(f"indexed {len(index)} result CSV with a readable args column")
    total_ok = total = 0
    for jf in args.jobs:
        print(f"\n=== {os.path.basename(jf)} ===")
        with open(jf) as fh:
            for line in fh:
                parsed = parse_job(line.strip())
                if not parsed:
                    continue
                tag, cfg_job = parsed
                total += 1
                hit = None
                cell_dir = os.path.join(args.results, tag)
                # ONLY the cell's own directory is searched.  A global fallback
                # is unsafe: the job files do not carry --dataset (the chain
                # supplies it), so a global search can match a same-named cell
                # from the OTHER corpus and report its numbers as this cell's.
                # Missing data is reported as missing; a wrong number is not.
                for path, metrics, name, cfg in index:
                    if not path.startswith(cell_dir + os.sep):
                        continue
                    if matches(cfg_job, cfg):
                        hit = (path, metrics, name, cfg)
                        break
                if hit is None:
                    print(f"  {tag:<44} (NO CSV with that configuration)")
                    continue
                total_ok += 1
                path, metrics, name, cfg = hit
                extra = ""
                if "text_encoder" in cfg_job:
                    extra = f" [{cfg_job['text_encoder']}"
                    for k in ("lr_bert", "llrd", "warmup_steps", "eta"):
                        if k in cfg_job:
                            extra += f" {k}={cfg_job[k]}"
                    extra += "]"
                print(f"  {tag:<44} MAE {metrics['mae']:.4f}  Corr {metrics['cor']:.4f}"
                      f"  ACC-7 {metrics['acc7']:.2f}  ACC-2(a) {metrics['acc2_all']:.2f}"
                      f"  ACC-2(nz) {metrics['acc2_nz']:.2f}{extra}")
    print(f"\nclaimed {total_ok}/{total} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
