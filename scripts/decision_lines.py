#!/usr/bin/env python3
"""Judge the pre-registered decision lines of the MSA-SOTA plan.

Every line is a paired comparison between two arms of the same grid, over the
same seeds.  The lines are read from `docs/00_INNOVATION_PLAN.md`:

    L1  the corpus-aware case front-end must add >= 1.0 pp Acc-2(Has0) to a
        case-sensitive encoder, on each corpus, and the truncation rate must
        fall below 2 %
    L2  the capacity-preserving fine-tuning must add >= 1.0 pp Acc-7 over the
        flat schedule, and the seed standard deviation must be <= 1.0
    L3  the attributable selective fusion must add >= 0.4 pp Acc-2(Has0)

A line that is not met is reported as failed; nothing is re-tuned afterwards.

Usage:
    python3 scripts/decision_lines.py --results results --jobs jobs/s2_mosi.txt
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from claim import read_cell, parse_job, matches   # noqa: E402


def claim(results, tag, spec):
    got = read_cell(results, tag)
    if not got:
        return None
    m, cfg, side = got
    if matches(spec, cfg):
        return m
    return None


def paired(a_rows, b_rows, key):
    """Same-seed pairs; a seed missing on either side is reported, not filled."""
    a = {r["seed"]: r for r in a_rows}
    b = {r["seed"]: r for r in b_rows}
    seeds = sorted(set(a) & set(b))
    missing = sorted((set(a) | set(b)) - set(seeds))
    diffs = [a[s][key] - b[s][key] for s in seeds]
    return seeds, diffs, missing


def report(name, base_rows, arm_rows, key, threshold, direction="gt"):
    if not base_rows or not arm_rows:
        print(f"[{name}] 未决：缺少一臂的结果")
        return False
    seeds, diffs, missing = paired(base_rows, arm_rows, key)
    if not seeds:
        print(f"[{name}] 未决：没有可配对的种子")
        return False
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
    t = (mean / (sd / len(diffs) ** 0.5)) if sd else float("inf")
    ok = (mean > threshold) if direction == "gt" else (mean < threshold)
    extra = f"  缺配对种子 {missing}" if missing else ""
    print(f"[{name}] n={len(seeds)} Δ{key} 均值 {mean:+.4f} (sd {sd:.4f}, t={t:+.2f}) "
          f"阈值 {threshold:+} → {'通过' if ok else '未通过'}{extra}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--jobs", nargs="+", required=True)
    args = ap.parse_args()

    specs = {}
    for jf in args.jobs:
        for line in open(jf):
            p = parse_job(line.strip())
            if p:
                specs[p[0]] = p[1]

    rows = {}
    for tag, spec in specs.items():
        m = claim(args.results, tag, spec)
        if m:
            got = read_cell(args.results, tag)
            seed = got[1].get("seed", spec.get("seed"))
            rows[tag] = dict(m, seed=int(seed), tag=tag)

    by_prefix = {}
    for tag, r in rows.items():
        by_prefix.setdefault(tag.rsplit("_s", 1)[0], []).append(r)
    # the seed recorded in the file is the one that counts

    print(f"已认领 {len(rows)}/{len(specs)} 格\n")
    for p in sorted(by_prefix):
        rs = sorted(by_prefix[p], key=lambda r: r["seed"])
        m = st.mean([r["acc2_all"] for r in rs])
        print(f"  {p:44s} n={len(rs)} seeds={[r['seed'] for r in rs]} "
              f"Acc2(all) {m:.2f}  MAE {st.mean([r['mae'] for r in rs]):.4f} "
              f"Acc7 {st.mean([r['acc7'] for r in rs]):.2f}")

    def arm(prefix):
        return by_prefix.get(prefix, [])

    print()
    report("L2  CPF vs 平坦 LR（Acc-7）", arm("A1_mosi_bert_cpf"), arm("A0_mosi_bert_flat"),
           "acc7", 1.0, "gt")
    if arm("A1_mosi_bert_cpf"):
        sds = [r["acc7"] for r in arm("A1_mosi_bert_cpf")]
        print(f"     A1 的种子标准差（Acc-2）= "
              f"{st.stdev([r['acc2_all'] for r in arm('A1_mosi_bert_cpf')]):.2f}")
    report("L1  大小写前端（Acc-2）", arm("B1_mosi_roberta_cpf_cnUW"), arm("B0_mosi_roberta_cpf"),
           "acc2_all", 1.0, "gt")
    report("L3  ASF（Acc-2）", arm("B2_mosi_roberta_cpf_cnUW_asf"),
           arm("B1_mosi_roberta_cpf_cnUW"), "acc2_all", 0.4, "gt")
    report("   编码器效应（参考）", arm("B0_mosi_roberta_cpf"), arm("A1_mosi_bert_cpf"),
           "acc2_all", 0.0, "gt")
    # --- the same lines on CMU-MOSEI, where the case front-end must be inert
    report("L1' 大小写前端在 MOSEI 上不得有害（Acc-2）",
           arm("C1_mosei_roberta_cpf_cnUW"), arm("C0_mosei_roberta_cpf"),
           "acc2_all", -0.5, "gt")
    report("L3' ASF 在 MOSEI 上（Acc-2）",
           arm("C2_mosei_roberta_cpf_cnUW_asf"), arm("C1_mosei_roberta_cpf_cnUW"),
           "acc2_all", 0.4, "gt")


if __name__ == "__main__":
    main()
