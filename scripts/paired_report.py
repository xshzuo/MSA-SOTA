"""Paired comparison between a baseline set of cells and a treatment set.

Both sets are specified as job files (tag + seed + flags).  Every cell is
claimed with `claim_cells.py`'s rule: only the cell's OWN directory is
searched, and the recorded `args` of the CSV must match the job
specification.  A cell with no matching CSV is reported as missing, never
substituted.

Usage:
    python3 paired_report.py --results <dir> \
        --base jobs/mosi_c2_coarse.txt --base-prefix p9d_mosi_base_s \
        --treat jobs/mosi_p9b.txt --treat-prefix p9b_mosi_lstm32_s \
        --label "LSTM hidden dim 16 -> 32 (MOSI)"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from claim_cells import matches, parse_job, read_csv   # noqa: E402

METRICS = ("mae", "cor", "acc7", "acc2_all", "acc2_nz")
SEEDS = (1111, 2222, 3333)


def load_jobs(path):
    out = {}
    for line in open(path):
        p = parse_job(line.strip())
        if p:
            out[p[0]] = p[1]
    return out


def claim(results, tag, spec):
    d = os.path.join(results, tag)
    for path in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not path.endswith("_result.csv"):
            continue
        got = read_csv(os.path.join(d, path))
        if not got:
            continue
        metrics, name, cfg = got
        if matches(spec, cfg):
            return metrics
    return None


def paired_t(deltas):
    n = len(deltas)
    if n < 2:
        return float("nan")
    m = sum(deltas) / n
    var = sum((x - m) ** 2 for x in deltas) / (n - 1)
    sd = var ** 0.5
    if sd == 0:
        return float("inf") if m else 0.0
    return m / (sd / n ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-prefix", required=True)
    ap.add_argument("--treat", required=True)
    ap.add_argument("--treat-prefix", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    base_jobs = load_jobs(args.base)
    treat_jobs = load_jobs(args.treat)

    print(f"=== {args.label or args.treat_prefix} ===")
    print(f"{'seed':<7}{'base MAE':>10}{'treat MAE':>11}{'base ACC-7':>12}"
          f"{'treat ACC-7':>13}{'base ACC-2':>12}{'treat ACC-2':>13}")
    rows = []
    for s in SEEDS:
        bt = f"{args.base_prefix}{s}"
        tt = f"{args.treat_prefix}{s}"
        b = claim(args.results, bt, base_jobs.get(bt, {})) if bt in base_jobs else None
        t = claim(args.results, tt, treat_jobs.get(tt, {})) if tt in treat_jobs else None
        if b is None or t is None:
            print(f"{s:<7} (missing: base={'ok' if b else 'MISSING'}"
                  f" / treat={'ok' if t else 'MISSING'})")
            continue
        rows.append((s, b, t))
        print(f"{s:<7}{b['mae']:>10.4f}{t['mae']:>11.4f}{b['acc7']:>12.2f}"
              f"{t['acc7']:>13.2f}{b['acc2_all']:>12.2f}{t['acc2_all']:>13.2f}")

    if not rows:
        print("no paired cells")
        return 1
    print(f"\npaired over {len(rows)} seed(s)")
    print(f"{'metric':<12}{'base':>10}{'treat':>10}{'delta':>10}{'t':>9}")
    for k in METRICS:
        b = [r[1][k] for r in rows]
        t = [r[2][k] for r in rows]
        d = [y - x for x, y in zip(b, t)]
        print(f"{k:<12}{sum(b)/len(b):>10.4f}{sum(t)/len(t):>10.4f}"
              f"{sum(d)/len(d):>+10.4f}{paired_t(d):>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
