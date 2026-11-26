#!/usr/bin/env python3
"""Generate docs/07_ACCEPTANCE.md from the result files.

The document holds three things: the acceptance anchors, the per-arm table
with the seed count and the standard deviation, and the ensemble reading of
each arm.  Every number comes from the result files; nothing is typed in by
hand.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from msa.metrics import compute_metrics  # noqa: E402

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
LABEL = {"acc2_all": "Acc-2(Has0)", "acc2_nz": "Acc-2(Non0)", "acc7": "Acc-7",
         "mae": "MAE", "cor": "Corr"}
HIGHER = {"acc2_all": True, "acc2_nz": True, "acc7": True, "mae": False, "cor": True}
ORDER = ["acc2_all", "acc2_nz", "acc7", "mae", "cor"]
SKIP = ("N1", "N2", "N3", "D1_mosi_roberta", "D2_mosi_roberta", "DBG")


def load(results):
    arms = {}
    for cell in sorted(os.listdir(results)):
        d = os.path.join(results, cell)
        if not os.path.isdir(d) or cell.startswith(SKIP):
            continue
        arm = re.sub(r"_s\d{4}$", "", cell)
        for f in sorted(glob.glob(os.path.join(d, "*_result.csv"))):
            with open(f) as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                continue
            r = rows[-1]
            arms.setdefault(arm, []).append(
                {k: float(r[k]) for k in ORDER})
    return arms


def preds_of(d):
    files = [f for f in sorted(glob.glob(os.path.join(d, "preds_*.npz")))
             if not f.endswith("_dev.npz")]
    if not files:
        return None
    z = np.load(files[0])
    pk = [k for k in z.files if "pred" in k]
    tk = [k for k in z.files if "truth" in k or "true" in k or "label" in k]
    if not pk or not tk:
        return None
    return z[pk[0]].reshape(-1), z[tk[0]].reshape(-1)


def ensemble(results, arm):
    cells = sorted(glob.glob(os.path.join(results, arm + "_s*")))
    ps, truth = [], None
    for d in cells:
        got = preds_of(d)
        if got is None:
            continue
        p, t = got
        ps.append(p)
        truth = t
    if len(ps) < 2:
        return None
    return compute_metrics(np.mean(np.vstack(ps), axis=0), truth)


def corpus_of(arm):
    return "mosei" if "_mosei" in arm else "mosi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "07_ACCEPTANCE.md"))
    args = ap.parse_args()
    arms = load(args.results)

    lines = []
    W = lines.append
    W("# 验收表（自动生成）")
    W("")
    W("本文件由 `scripts/make_results_doc.py` 从结果文件自动生成，数字不经手工录入。")
    W("指标顺序统一为 MAE / Corr / Acc-7 / Acc-2(all) / Acc-2(nz)；")
    W("`Acc-2(all)` 为全样本、阈值 `>= 0`；`Acc-2(nz)` 为标签非零子集、阈值 **严格 `> 0`**。")
    W("")
    W("## 外部锚点")
    W("")
    W("| 锚点 | 语料 | Acc-2(Has0) | Acc-2(Non0) | Acc-7 | MAE | Corr |")
    W("|---|---|---|---|---|---|---|")
    for key, label in (("user", "项目验收线"), ("paper", "MMML + context（论文最强行）"),
                        ("repo", "MMML + context（仓库发布运行）")):
        for corpus in ("mosi", "mosei"):
            a = ANCHORS[key][corpus]
            W(f"| {label} | {corpus.upper()} | {a['acc2_all']} | {a['acc2_nz']} | "
              f"{a['acc7']} | {a['mae']} | {a['cor']} |")
    W("")

    for corpus in ("mosi", "mosei"):
        W(f"## {corpus.upper()}")
        W("")
        for key, label in (("user", "项目验收线"), ("paper", "论文最强行"),
                            ("repo", "仓库最强运行")):
            a = ANCHORS[key][corpus]
            rows = []
            for arm, g in arms.items():
                if corpus_of(arm) != corpus:
                    continue
                means = {k: st.mean([m[k] for m in g]) for k in ORDER}
                cleared = []
                for k in ORDER:
                    thr = a[k]
                    ok = (means[k] > thr) if HIGHER[k] else (means[k] < thr)
                    if ok:
                        cleared.append(LABEL[k])
                rows.append((len(cleared), arm, len(g), means, g, cleared))
            rows.sort(key=lambda r: (-r[0], -r[3]["acc2_all"]))
            W(f"### 对 {label}（{a['acc2_all']} / {a['acc2_nz']} / {a['acc7']} / "
              f"{a['mae']} / {a['cor']}）")
            W("")
            W("| 臂 | n | Acc-2(Has0) | Acc-2(Non0) | Acc-7 | MAE | Corr | 过线 |")
            W("|---|---|---|---|---|---|---|---|")
            for ncleared, arm, n, means, g, cleared in rows:
                sd = {k: (st.stdev([m[k] for m in g]) if n > 1 else 0.0) for k in ORDER}
                W(f"| {arm} | {n} | {means['acc2_all']:.2f}±{sd['acc2_all']:.2f} | "
                  f"{means['acc2_nz']:.2f}±{sd['acc2_nz']:.2f} | "
                  f"{means['acc7']:.2f}±{sd['acc7']:.2f} | "
                  f"{means['mae']:.4f}±{sd['mae']:.4f} | "
                  f"{means['cor']:.4f}±{sd['cor']:.4f} | "
                  f"{ncleared}/5 {', '.join(cleared)} |")
            W("")
            W("### 集成读数（把该臂各种子的预测平均；已发表锚点为单模型，故仅作附加读数）")
            W("")
            W("| 臂 | n | Acc-2(Has0) | Acc-2(Non0) | Acc-7 | MAE | Corr |")
            W("|---|---|---|---|---|---|---|")
            ens_rows = []
            for arm in arms:
                if corpus_of(arm) != corpus:
                    continue
                m = ensemble(args.results, arm)
                if m:
                    ens_rows.append((m["acc2_all"], arm, m))
            ens_rows.sort(key=lambda r: -r[0])
            for _, arm, m in ens_rows:
                W(f"| {arm} | {len(arms[arm])} | {m['acc2_all']:.2f} | "
                  f"{m['acc2_nz']:.2f} | {m['acc7']:.2f} | {m['mae']:.4f} | "
                  f"{m['corr']:.4f} |")
            W("")

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"已写出 {args.out}（{len(lines)} 行）")


if __name__ == "__main__":
    main()
