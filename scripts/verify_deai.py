#!/usr/bin/env python3
"""Verify a de-AI scrub in the rendered PDF, both directions.

Every pair is (new wording, old wording).  The new wording must be present in
the extracted text and the old wording must be absent.  The text layer is
normalised first, because a PDF probe false-fails on ligatures, typographic
apostrophes and en-dashes for reasons that have nothing to do with the edit.

usage:  python3 scripts/verify_deai.py paper/main.pdf
"""
import re
import sys

import fitz

PAIRS = [
    ("routing it to the regression head only, so that it never reaches the decision head, turns it",
     "routing it to the regression head only---so that it never reaches the decision head---turns it"),
    ("We rebuilt the pipeline from scratch (data, model, metrics and training) and asked",
     "We rebuilt the pipeline from scratch---data, model, metrics and training---and asked"),
    ("Compared here are the columns the field reports by convention:",
     "The columns compared here are the ones the field reports by convention---"),
    ("A capacity-preserving fine-tuning recipe, namely layer-wise learning rate decay with warmup, is worth",
     "A capacity-preserving fine-tuning recipe---layer-wise learning rate decay with warmup---is worth"),
    ("the interventions that raise it (a squared error loss, a heavier decision-head weight) make",
     "the interventions that raise it---a squared error loss, a heavier decision-head weight---make"),
    ("Included in the repository cited above are the derived data of this work",
     "The derived data of this work---the prepared splits"),
    ("One adds context to the utterance", "The first adds context to the utterance"),
    ("Another replaces or augments the text encoder", "The second replaces or augments the text encoder"),
    ("A third asks for causal or counterfactual evidence", "The third asks for causal or counterfactual evidence"),
    ("The contributions are these.", "Three contributions follow."),
    ("Two claims this paper does not make.", "Two things this paper does not do."),
    ("The mechanism is mechanical:", "The reason is mechanical:"),
    ("This locates where the competition happens.", "What this locates is where the competition happens."),
    ("Three limitations constrain the result.", "Three limitations should be stated."),
    ("Four rules further govern the protocol.", "Four further rules govern the protocol."),
    ("Two smaller choices belong to the recipe.", "Two smaller choices are part of the recipe."),
    ("The numbers carry two properties worth setting out.", "Two properties of the numbers are worth setting out."),
    ("The comparison carries caveats.", "The comparison carries caveats, which we state next."),
    ("and three of them matter here", "Three are worth stating here."),
    # the last pass broke up the runs of sentences that open with the
    # definite article, which is where most of the cadence tell sat
    ("Polarity columns are noisy:", "The polarity columns are noisy:"),
    ("Seven-level accuracy depends on how the predictions are distributed",
     "The seven-level accuracy depends on how the predictions are distributed"),
    ("Training uses a layer-wise", "The encoder is trained with a layer-wise"),
]


def norm(s):
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("ﬀ", "ff").replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "paper/main.pdf"
    doc = fitz.open(path)
    raw = "".join(p.get_text() for p in doc)
    P = norm(raw)

    fails = 0
    for new, old in PAIRS:
        new_ok = norm(new) in P
        old_gone = norm(old) not in P
        flag = "OK " if (new_ok and old_gone) else "FAIL"
        if not (new_ok and old_gone):
            fails += 1
        print(f"  {flag} new={new_ok} old_gone={old_gone}  {new[:56]}")
    print(f"\n{len(PAIRS) - fails}/{len(PAIRS)} pairs verified")
    print(f"pages: {doc.page_count}")
    print(f"em-dash in PDF: {raw.count(chr(8212))}   en-dash: {raw.count(chr(8211))}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
