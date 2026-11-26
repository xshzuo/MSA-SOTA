"""Dataset, case front-end and collation for the MSA model.

The case front-end (component LNF) is implemented here, because it belongs to
the *input* side of the model and must be switchable per corpus:

    none          keep the transcript as it is
    lower         fold the whole transcript
    upper_words   fold only the words written entirely in upper case

`upper_words` is the interesting one: the CMU-MOSI transcripts are upper case
(100 % of the training transcripts contain an upper-case character) and a
case-sensitive tokenizer shreds them into pieces, while the CMU-MOSEI
transcripts are ordinary sentences whose case carries information.  The rule
is therefore applied per corpus, and the decision is recorded in the run
configuration rather than assumed.

The collate function also records the *pre-truncation* token length of every
sample, which is what the token-budget table of the paper is built from.
"""
import re

import numpy as np
import torch
from torch.utils.data import Dataset

_UPPER_WORD = re.compile(r"\b[A-Z]{2,}\b")


def apply_case_rule(text, mode):
    if mode == "lower":
        return text.lower()
    if mode == "upper_words":
        return _UPPER_WORD.sub(lambda m: m.group(0).lower(), text)
    if mode == "none":
        return text
    raise ValueError(f"unknown case rule: {mode}")


def corpus_case_rule(decisions):
    """Pick the case rule from a per-corpus decision record.

    `decisions` maps a corpus name to the fraction of its training transcripts
    that contain an upper-case word.  Above `threshold` the transcripts are
    shouted and the case carries no information, so the case-sensitive
    tokenizer needs the fold; below it the case is meaningful and must be kept.
    """
    frac = decisions.get("upper_word_fraction", 0.0)
    return "upper_words" if frac >= decisions.get("threshold", 0.20) else "none"


class MSADataset(Dataset):
    def __init__(self, path, case_rule="none", with_context=False):
        z = np.load(path, allow_pickle=True)
        self.raw_text = z["raw_text"]
        self.audio = z["audio"]
        self.vision = z["vision"]
        self.label_reg = z["label_reg"]
        self.label_cls = z["label_cls"]
        self.case_rule = case_rule
        self.with_context = with_context
        self.ids = z["bert_ids"]

    def __len__(self):
        return len(self.label_reg)

    def __getitem__(self, i):
        text = apply_case_rule(str(self.raw_text[i]), self.case_rule)
        item = {
            "text": text,
            "audio": self.audio[i],
            "vision": self.vision[i],
            "label": float(self.label_reg[i]),
            "label_cls": int(self.label_cls[i]),
        }
        if self.with_context:
            # the previous utterance of the same split.  The rows of the
            # official aligned pickles keep the original order (verified: the
            # label autocorrelation of adjacent rows is 0.34 on CMU-MOSI and
            # 0.44 on CMU-MOSEI against -0.03 for random pairs), so the
            # neighbouring row is the neighbouring utterance of the same
            # recording.  The first row of a split has no context.
            j = i - 1
            item["ctx_text"] = (apply_case_rule(str(self.raw_text[j]), self.case_rule)
                                if j >= 0 else "")
            item["ctx_audio"] = self.audio[j] if j >= 0 else np.zeros_like(self.audio[0])
        return item


class Collator:
    def __init__(self, tokenizer, max_len=50, case_rule="none", with_context=False):
        self.tok = tokenizer
        self.max_len = max_len
        self.case_rule = case_rule
        self.with_context = with_context
        self.token_lengths = []      # pre-truncation lengths
        self.truncated = 0
        self.n_seen = 0

    def __call__(self, batch):
        texts = [apply_case_rule(b["text"], self.case_rule) for b in batch]
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        # the token budget is measured before truncation, so it needs the raw
        # lengths of the (untruncated) encodings
        raw = self.tok(texts, padding=False, truncation=False)["input_ids"]
        for r in raw:
            self.token_lengths.append(len(r))
            self.n_seen += 1
            if len(r) > self.max_len:
                self.truncated += 1
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "audio": torch.from_numpy(np.stack([b["audio"] for b in batch])),
            "vision": torch.from_numpy(np.stack([b["vision"] for b in batch])),
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
            "label_cls": torch.tensor([b["label_cls"] for b in batch], dtype=torch.long),
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"]
        if self.with_context:
            ctx = [b.get("ctx_text", "") for b in batch]
            cenc = self.tok(ctx, padding=True, truncation=True, max_length=self.max_len,
                            return_tensors="pt")
            out["ctx_input_ids"] = cenc["input_ids"]
            out["ctx_attention_mask"] = cenc["attention_mask"]
            if "token_type_ids" in cenc:
                out["ctx_token_type_ids"] = cenc["token_type_ids"]
            out["ctx_audio"] = torch.from_numpy(
                np.stack([b.get("ctx_audio", np.zeros_like(b["audio"])) for b in batch]))
        return out

    def budget_report(self):
        if not self.token_lengths:
            return {}
        a = np.asarray(self.token_lengths)
        return {
            "token_mean": float(a.mean()),
            "token_p50": float(np.percentile(a, 50)),
            "token_p95": float(np.percentile(a, 95)),
            "token_max": int(a.max()),
            "truncated_rate": self.truncated / max(self.n_seen, 1),
            "n_seen": self.n_seen,
        }
