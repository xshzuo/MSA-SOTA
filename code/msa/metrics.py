"""The evaluation metrics, matching the conventions of the published tables.

The conventions are the ones the CMU-MMSA line of work uses and that the
reference implementation applies (verified against its source):

    MAE        mean(|pred - truth|)                       (no clipping)
    Corr       pearson(pred, truth)
    Acc-7      mean(round(clip(pred,-3,3)) == round(clip(truth,-3,3)))
    Acc-2(all) mean( (pred >= 0) == (truth >= 0) )         <- the "Has0" column
    Acc-2(nz)  mean( (pred >  0) == (truth >  0) ) over the samples with truth != 0
                                                         <- the "Non0" column

Note the two different thresholds: the all-sample column treats a zero label as
non-negative (>= 0), the non-zero subset uses a strict > 0.  Getting this
wrong changes the headline number by several points, so it is spelled out here
and checked against the reference implementation in `check_metrics.py`.
"""
import numpy as np


def compute_metrics(preds, truths):
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    truths = np.asarray(truths, dtype=np.float64).reshape(-1)
    if preds.shape != truths.shape:
        raise ValueError(f"shape mismatch: {preds.shape} vs {truths.shape}")
    out = {}
    out["mae"] = float(np.mean(np.abs(preds - truths)))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["corr"] = float(np.corrcoef(preds, truths)[0][1])

    p7 = np.clip(preds, -3.0, 3.0)
    t7 = np.clip(truths, -3.0, 3.0)
    out["acc7"] = float(np.sum(np.round(p7) == np.round(t7)) / len(t7) * 100.0)

    nz = truths != 0
    out["acc2_all"] = float(np.mean((preds >= 0) == (truths >= 0)) * 100.0)
    out["acc2_nz"] = float(np.mean((preds[nz] > 0) == (truths[nz] > 0)) * 100.0)
    out["n"] = int(len(truths))
    out["n_nz"] = int(nz.sum())
    return out


def fmt(m):
    return (f"MAE {m['mae']:.4f}  Corr {m['corr']:.4f}  Acc7 {m['acc7']:.2f}  "
            f"Acc2(all) {m['acc2_all']:.2f}  Acc2(nz) {m['acc2_nz']:.2f}")
