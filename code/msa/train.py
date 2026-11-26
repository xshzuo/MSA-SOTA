#!/usr/bin/env python3
"""Train the MSA model.

Protocol
--------
* The reported epoch is selected by the **validation** loss (never by the test
  loss).
* Every run writes one CSV row that carries the full configuration, so a
  result can be claimed by the configuration recorded inside it rather than by
  a file name.
* The parameter groups of the optimiser are **printed** (size and learning
  rate per group): an intervention that selects parameters by name must show
  what it selected, because a pattern that matches nothing fails silently.

Usage:
    python3 -m msa.train --data <dir> --corpus mosi --tokenizer bert-base-uncased \
        --out results/<tag>.csv
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msa.data import Collator, MSADataset          # noqa: E402
from msa.metrics import compute_metrics, fmt        # noqa: E402
from msa.model import MSAModel                      # noqa: E402

LAYER_RE = re.compile(r"(?:^|\.)(?:layers?|blocks?)\.(\d+)\.")


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # the same seed then reproduces the same numbers bit for bit; without
        # it the non-deterministic GPU kernels move the result by roughly
        # 1.7 pp Acc-7 between two runs of the same configuration
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_optimizer(model, args):
    """Layer-wise learning-rate decay over the language encoder.

    The encoder's blocks are found by name; the *pattern* differs between
    architectures (`encoder.layer.N.` for BERT/RoBERTa/DeBERTa, `layer.N.` for
    XLNet), so both are matched and the resulting groups are printed.
    """
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = LAYER_RE.search(name)
        if m and name.startswith("text."):
            idx = int(m.group(1))          # the regex has a single capture group
            key = ("text_layer", idx)
        elif name.startswith("text."):
            key = ("text_other", 0)
        else:
            key = ("main", 0)
        groups.setdefault(key, []).append(p)

    depth = max([k[1] for k in groups if k[0] == "text_layer"], default=-1) + 1
    if depth == 0:
        raise SystemExit(
            "no language-encoder block matched the layer pattern -- the "
            "layer-wise decay would silently do nothing.  Check the "
            "parameter names before training.")
    params = []
    for key in sorted(groups, key=lambda k: (k[0] != "text_layer", k[1])):
        ps = groups[key]
        n = sum(p.numel() for p in ps)
        if key[0] == "text_layer":
            lr = args.lr_bert * (args.llrd ** (depth - 1 - key[1]))
        elif key[0] == "text_other":
            lr = args.lr_bert
        else:
            lr = args.lr_main
        params.append({"params": ps, "lr": lr})
        print(f"  [param group] {key[0]}{key[1]:<3d} lr={lr:.3e}  tensors={len(ps):<4d} "
              f"params={n/1e6:.2f}M")
    opt = torch.optim.AdamW(params, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * max(1, (args.train_size + args.batch_size - 1)
                                           // args.batch_size))

    def lr_lambda(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        if args.lr_schedule == "cosine":
            import math
            t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched, depth


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred, _ = model(batch)
        preds.append(pred.detach().float().cpu().numpy())
        truths.append(batch["label"].cpu().numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return compute_metrics(preds, truths), preds, truths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with train/dev/test npz")
    ap.add_argument("--corpus", default="mosi")
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--case_rule", default="none", choices=["none", "lower", "upper_words"])
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr_main", type=float, default=1e-3)
    ap.add_argument("--lr_bert", type=float, default=5e-5)
    ap.add_argument("--llrd", type=float, default=0.9)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"])
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation: the effective batch is "
                         "batch_size * accum and the number of optimizer steps "
                         "is unchanged.  Needed for the large encoders, whose "
                         "fp32 training set does not fit the accelerator in one "
                         "piece (deberta-v3-large needs > 15.2 GiB at batch 16)")
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--deterministic", type=int, default=0,
                    help="make a run reproducible bit for bit (same seed -> same numbers)")
    ap.add_argument("--loss", default="l1", choices=["l1", "mse"])
    ap.add_argument("--neg_weight", type=float, default=1.0,
                    help="weight of the negative-class samples in the regression "
                         "loss.  The L1 objective is minimised by the *weighted* "
                         "median, so this weight moves the regression output "
                         "along the label axis without touching the metric "
                         "definition.  1.0 = unweighted; n_pos/n_neg = a class "
                         "balanced objective; no test label is involved.")
    ap.add_argument("--context", type=int, default=0,
                    help="add the neighbouring utterance as a third, gated stream")
    ap.add_argument("--context_route", default="both", choices=["both", "reg"],
                    help="reg: the context feeds the regression head only, not the "
                         "decision head (the neighbouring utterance informs the "
                         "intensity but misleads the polarity)")
    ap.add_argument("--asf", type=int, default=0,
                    help="attributable selective fusion: the per-sample modality weights "
                         "come from the measured effect of removing each non-verbal stream")
    ap.add_argument("--asf_lambda", type=float, default=1.0,
                    help="how strongly the measured effect scales the weaker stream "
                         "(0 = the gate is off, 1 = the weaker stream is damped to its "
                         "relative effect size)")
    ap.add_argument("--binary_head", type=int, default=0)
    ap.add_argument("--lambda_binary", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_preds", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    set_seed(args.seed, deterministic=bool(args.deterministic))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir,
                                        use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if hasattr(tok, "eos_token") else tok.unk_token

    def load(split):
        return MSADataset(os.path.join(args.data, f"{split}.npz"),
                          case_rule=args.case_rule, with_context=bool(args.context))

    train_ds, dev_ds, test_ds = load("train"), load("dev"), load("test")
    coll = Collator(tok, max_len=args.max_len, case_rule=args.case_rule,
                    with_context=bool(args.context))
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=coll, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=coll)
    args.train_size = len(train_ds)

    d_audio = int(train_ds.audio.shape[-1])
    d_vision = int(train_ds.vision.shape[-1])
    print(f"[data] audio {train_ds.audio.shape} vision {train_ds.vision.shape} "
          f"-> stream widths {d_audio}/{d_vision}")
    model = MSAModel(args.tokenizer, d_model=args.d_model, n_layers=args.n_layers,
                     n_heads=args.n_heads, dropout=args.dropout,
                     d_audio=d_audio, d_vision=d_vision,
                     cache_dir=args.cache_dir,
                     binary_head=bool(args.binary_head),
                     use_context=bool(args.context),
                     route_context=args.context_route).to(device)
    n_all = sum(p.numel() for p in model.parameters())
    n_text = sum(p.numel() for p in model.text.parameters())
    print(f"[model] total {n_all/1e6:.2f}M  text {n_text/1e6:.2f}M  "
          f"non-text {(n_all-n_text)/1e6:.2f}M")
    opt, sched, depth = build_optimizer(model, args)
    print(f"[optim] encoder depth detected = {depth} layers; llrd={args.llrd}; "
          f"warmup={args.warmup_steps}")

    crit = nn.L1Loss() if args.loss == "l1" else nn.MSELoss()
    neg_weight = float(args.neg_weight)
    eff_accum = [torch.zeros(0) for _ in range(6)]
    best = {"val": 1e9, "state": None, "epoch": -1}
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        accum_count = 0
        for batch in dl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if args.asf:
                out = model.forward_with_effects(batch)
                if args.context:
                    (pred, logits, eff_a, eff_v, eff_c, w_a, w_v, w_c,
                     z_ta, z_tv, z_tc) = out
                    pred_c = model._predict(z_ta, z_tv, torch.zeros_like(z_tc))[0]
                else:
                    (pred, logits, eff_a, eff_v, w_a, w_v, z_ta, z_tv) = out
                    eff_c = w_c = z_tc = None
                zero = torch.zeros_like(z_ta)
                pred_a = model._predict(zero, z_tv, z_tc)[0]
                pred_v = model._predict(z_ta, zero, z_tc)[0]
                # the gate scales every stream by its relative effect size;
                # the streams are reused, so the language encoder runs once
                scale_a = (1.0 - args.asf_lambda * (1.0 - w_a)).unsqueeze(-1)
                scale_v = (1.0 - args.asf_lambda * (1.0 - w_v)).unsqueeze(-1)
                if z_tc is not None:
                    scale_c = (1.0 - args.asf_lambda * (1.0 - w_c)).unsqueeze(-1)
                    pred, _ = model._predict(z_ta * scale_a, z_tv * scale_v,
                                             z_tc * scale_c)
                    eff_accum[2] = torch.cat([eff_accum[2], eff_c.detach().cpu()])
                else:
                    pred, _ = model._predict(z_ta * scale_a, z_tv * scale_v)
                eff_accum[0] = torch.cat([eff_accum[0], eff_a.detach().cpu()])
                eff_accum[1] = torch.cat([eff_accum[1], eff_v.detach().cpu()])
                # the intervention predictions are kept as well: the mechanism
                # claim is that the measured effect predicts the damage the
                # removal does, and that needs both sides
                for slot, arr in ((3, pred_a), (4, pred_v)):
                    eff_accum[slot] = torch.cat([eff_accum[slot], arr.detach().cpu()])
                if z_tc is not None:
                    eff_accum[5] = torch.cat([eff_accum[5], pred_c.detach().cpu()])
            else:
                pred, logits = model(batch)
            if neg_weight == 1.0:
                loss = crit(pred, batch["label"])
            else:
                # the negative class carries a different weight, so the
                # minimiser of the loss is the weighted median
                w = torch.where(batch["label"] < 0,
                                torch.full_like(pred, neg_weight),
                                torch.ones_like(pred))
                if args.loss == "l1":
                    loss = (w * (pred - batch["label"]).abs()).mean()
                else:
                    loss = (w * (pred - batch["label"]).pow(2)).mean()
            if logits is not None and args.lambda_binary > 0:
                tgt = (batch["label"] >= 0).float()
                loss = loss + args.lambda_binary * nn.functional.binary_cross_entropy_with_logits(
                    logits, tgt)
            # gradient accumulation: the micro-batch loss is scaled so that
            # the accumulated gradient equals the gradient of the effective
            # batch (batch_size * accum); the optimizer step happens every
            # accum micro-batches, so the number of steps is unchanged
            (loss / args.accum).backward()
            accum_count += 1
            if accum_count % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            tot += float(loss.item())
            nb += 1
        vm, _, _ = evaluate(model, dev_loader, device, args)
        print(f"[epoch {ep:02d}] train_loss {tot/max(nb,1):.4f}  val_mae {vm['mae']:.4f}  "
              f"val_corr {vm['corr']:.4f}  ({time.time()-t0:.0f}s)")
        if vm["mae"] < best["val"]:
            best = {"val": vm["mae"], "state": {k: v.detach().cpu().clone()
                                                for k, v in model.state_dict().items()},
                    "epoch": ep}
    model.load_state_dict(best["state"])
    tm, preds, truths = evaluate(model, test_loader, device, args)
    print(f"[best epoch {best['epoch']}] TEST  {fmt(tm)}")

    budget = coll.budget_report()
    print(f"[token budget] {json.dumps(budget)}")

    # one directory per cell, holding the result CSV and the prediction
    # sidecar; the sidecar name carries the corpus prefix and the run's own
    # reported MAE, so it can be claimed without sorting
    cell_dir = args.out if os.path.isdir(args.out) else os.path.dirname(os.path.abspath(args.out))
    os.makedirs(cell_dir, exist_ok=True)
    tag = args.tag or os.path.basename(os.path.abspath(args.out)).replace(".csv", "")
    out_csv = os.path.join(cell_dir, f"{tag}_result.csv")
    row = {
        "tag": tag,
        "mae": tm["mae"], "cor": tm["corr"], "acc7": tm["acc7"],
        "acc2_all": tm["acc2_all"], "acc2_nz": tm["acc2_nz"],
        "best_epoch": best["epoch"], "n_test": tm["n"],
        "token_mean": budget.get("token_mean"), "token_p95": budget.get("token_p95"),
        "truncated_rate": budget.get("truncated_rate"),
        "params_M": n_all / 1e6,
        # a Python repr, not JSON: the claimer parses this column with
        # ast.literal_eval, and JSON's true/false/null are not Python literals
        "args": repr(vars(args)),
    }
    import csv
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    side = os.path.join(cell_dir, f"preds_{args.corpus}_{tag}_eam{tm['mae']:.5f}.npz")
    extra = {}
    if args.asf and eff_accum[0].numel():
        extra = {"eff_a": eff_accum[0].numpy(), "eff_v": eff_accum[1].numpy(),
                 "pred_no_audio": eff_accum[3].numpy(),
                 "pred_no_vision": eff_accum[4].numpy()}
        if eff_accum[2].numel():
            extra["eff_c"] = eff_accum[2].numpy()
            extra["pred_no_context"] = eff_accum[5].numpy()
    np.savez(side, preds=preds, truths=truths, **extra)
    print(f"[done] {out_csv}  ({time.time()-t0:.0f}s)")

    # the development split's predictions are kept as well: a scale fitted on
    # the development split is a legitimate way to correct a compressed output
    # (it never touches the test labels)
    dm, dpreds, dtruths = evaluate(model, dev_loader, device, args)
    np.savez(side.replace(".npz", "_dev.npz"), preds=dpreds, truths=dtruths)
    print(f"[done] dev predictions saved  ({side.replace('.npz', '_dev.npz')})")


if __name__ == "__main__":
    main()
