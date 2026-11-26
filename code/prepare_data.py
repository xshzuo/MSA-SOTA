#!/usr/bin/env python3
"""Build the MSA working data from the official CMU-MMSA pickles.

The pickles (`mosi_aligned_50.pkl`, `mosei_aligned_50.pkl`) are the official
release of the CMU-Multimodal-SDK; their SHA-256 values are published on the
MMSA wiki, so the source can be checked before anything is derived from it.

Each split is written as one `.npz` holding

    raw_text    (N,)      the transcript (str)
    audio       (N, 50, 5)   word-aligned acoustic features
    vision      (N, 50, 20)  word-aligned visual features
    bert_ids    (N, 50)      the BERT token ids shipped in the pickle
    label_reg   (N,)      the sentiment intensity in [-3, 3]
    label_cls   (N,)      {0: negative, 1: neutral, 2: positive}

Nothing is pre-tokenised for the model: the text is kept as a string so that a
case-sensitive encoder can be used with the corpus-aware front-end (component
LNF in docs/02_METHOD_DESIGN.md) without re-running this script.

Usage:
    python3 prepare_data.py --pkl <pickle> --out <dir> --corpus mosi \
        --sha256 d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9
"""
import argparse
import hashlib
import os
import pickle

import numpy as np

SPLITS = (("train", "train"), ("valid", "dev"), ("test", "test"))


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", required=True, choices=["mosi", "mosei"])
    ap.add_argument("--sha256", default=None, help="expected SHA-256 of the pickle")
    args = ap.parse_args()

    if args.sha256:
        got = sha256(args.pkl)
        if got != args.sha256:
            raise SystemExit(f"SHA-256 mismatch:\n  expected {args.sha256}\n  got      {got}")
        print(f"[ok] source SHA-256 verified: {got}")

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    os.makedirs(args.out, exist_ok=True)
    for src, dst in SPLITS:
        if src not in data:
            raise SystemExit(f"split {src!r} not in the pickle (keys: {list(data)})")
        s = data[src]
        n = len(s["raw_text"])
        ids = s["text_bert"][:, 0, :].astype(np.int32)
        out = {
            "raw_text": np.asarray(s["raw_text"], dtype=object),
            "audio": s["audio"].astype(np.float32),
            "vision": s["vision"].astype(np.float32),
            "bert_ids": ids,
            "label_reg": s["regression_labels"].astype(np.float32),
            "label_cls": s["classification_labels"].astype(np.int64),
        }
        path = os.path.join(args.out, f"{dst}.npz")
        np.savez_compressed(path, **out)
        reg = out["label_reg"]
        print(f"[ok] {path}  n={n}  audio{out['audio'].shape} vision{out['vision'].shape}"
              f"  label_reg[{reg.min():.1f},{reg.max():.1f}]"
              f"  非零标签占比 {float((reg != 0).mean()):.3f}"
              f"  ≥0 占比 {float((reg >= 0).mean()):.3f}")


if __name__ == "__main__":
    main()
