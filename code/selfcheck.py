#!/usr/bin/env python3
"""Self-checks for the model code.

The attributable-effect read and the context stream are new code paths, so they
are checked directly rather than only through the training numbers.

Run:  python3 code/selfcheck.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msa.data import apply_case_rule          # noqa: E402
from msa.metrics import compute_metrics       # noqa: E402
from msa.model import MSAModel                # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}  {detail}")
        FAILS.append(name)


def main():
    torch.manual_seed(0)
    # --- the case front-end
    # a single upper-case letter is left alone on purpose: in ordinary text
    # the pronoun "I" is written in upper case, so folding it would destroy a
    # meaningful case difference
    check("全大写单词被折叠（单字母除外）",
          apply_case_rule("I LOVE THIS MOVIE", "upper_words") == "I love this movie",
          apply_case_rule("I LOVE THIS MOVIE", "upper_words"))
    check("混合大小写只折叠全大写词",
          apply_case_rule("I love THIS movie", "upper_words") == "I love this movie",
          apply_case_rule("I love THIS movie", "upper_words"))
    check("两个以上字母的全大写词被折叠",
          apply_case_rule("THIS IS GOOD", "upper_words") == "this is good")
    check("none 模式不改文本", apply_case_rule("I LOVE THIS", "none") == "I LOVE THIS")
    check("lower 模式整体折叠", apply_case_rule("I Love This", "lower") == "i love this")

    # --- the metrics against hand-computed values
    preds = np.array([2.0, -1.0, 0.5, -0.2, 1.5])
    truths = np.array([2.0, -1.0, 0.0, 0.0, 1.0])
    m = compute_metrics(preds, truths)
    check("MAE", abs(m["mae"] - np.mean(np.abs(preds - truths))) < 1e-12)
    check("Acc-2(all) 用 >=0", abs(m["acc2_all"] - 80.0) < 1e-9, m)
    check("Acc-2(nz) 用 >0 且只算非零子集",
          abs(m["acc2_nz"] - 100.0) < 1e-9, m)     # the two non-zero samples agree
    # the rounded pairs are (2,2) (1,1) (0,0) (0,0) (2,1) -> four of five
    check("Acc-7 先 clip 再 round", abs(m["acc7"] - 80.0) < 1e-9, m)

    # --- the model: the effect read must not change the factual prediction
    # the language encoder is replaced by a stub: a self-check must not need
    # the network
    import msa.model as M

    class FakeConfig:
        hidden_size = 12

    class FakeHF(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.emb = torch.nn.Embedding(100, 12)
            self.config = FakeConfig()

        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def forward(self, input_ids, attention_mask=None, **kw):
            out = type("O", (), {})()
            out.last_hidden_state = self.emb(input_ids)
            return out

    M.AutoConfig = type("AC", (), {"from_pretrained": staticmethod(
        lambda *a, **k: FakeConfig())})
    M.AutoModel = FakeHF

    d_model = 16
    model = MSAModel(text_encoder="stub", d_model=d_model, n_layers=1, n_heads=2,
                     d_audio=5, d_vision=20)
    model.binary_head = None

    B, L = 4, 6
    batch = {
        "input_ids": torch.randint(1, 99, (B, L)),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "audio": torch.randn(B, 50, 5),
        "vision": torch.randn(B, 50, 20),
    }
    model.eval()
    with torch.no_grad():
        pred_f, _ = model(batch)
        out = model.forward_with_effects(batch)
    check("效应读取与事实前向给出一致的预测",
          torch.allclose(pred_f, out[0], atol=1e-6),
          f"{pred_f.tolist()} vs {out[0].tolist()}")
    eff_a, eff_v = out[2], out[3]
    check("效应量非负", bool((eff_a >= 0).all() and (eff_v >= 0).all()))
    check("相对权重归一到 [0,1]", bool((out[4] <= 1 + 1e-6).all() and (out[5] <= 1 + 1e-6).all()))

    # --- the context stream (a second model, because the head width depends
    #     on the number of streams)
    model_ctx = MSAModel(text_encoder="stub", d_model=d_model, n_layers=1, n_heads=2,
                         d_audio=5, d_vision=20, use_context=True)
    model_ctx.binary_head = None
    model_ctx.eval()

    cbatch = dict(batch)
    cbatch["ctx_input_ids"] = torch.randint(1, 99, (B, L))
    cbatch["ctx_attention_mask"] = torch.ones(B, L, dtype=torch.long)
    cbatch["ctx_audio"] = torch.randn(B, 50, 5)
    with torch.no_grad():
        pred_c, _ = model_ctx(cbatch)
        out_c = model_ctx.forward_with_effects(cbatch)
    check("上下文流给出形状正确的预测", tuple(pred_c.shape) == (B,))
    check("上下文臂有第三个效应量", len(out_c) == 11 and out_c[4].numel() == B)
    with torch.no_grad():
        z = model_ctx._streams(cbatch)
    check("上下文流被读出", len(z) == 3 and z[2] is not None)
    check("无上下文时不产生第三流",
          len(model._streams(batch)) == 3 and model._streams(batch)[2] is None)
    check("融合头宽度与流数一致（2 流 / 3 流）",
          model.head[0].in_features == 2 * d_model
          and model_ctx.head[0].in_features == 3 * d_model,
          f"{model.head[0].in_features} / {model_ctx.head[0].in_features}")

    # --- reproducibility of the construction: a conditionally used module must
    #     not be built when it is not used, because it would consume random
    #     draws and change the initialisation of every later module
    torch.manual_seed(7)
    m1 = MSAModel(text_encoder="stub", d_model=d_model, n_layers=1, n_heads=2,
                  d_audio=5, d_vision=20)
    torch.manual_seed(7)
    m2 = MSAModel(text_encoder="stub", d_model=d_model, n_layers=1, n_heads=2,
                  d_audio=5, d_vision=20)
    same = all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(),
                                                  m2.state_dict().values()))
    check("同一 seed 构造出完全相同的参数", same)
    check("未启用上下文时不创建 cross_c",
          model.cross_c is None and model_ctx.cross_c is not None)
    # and the parameter set must not depend on whether the context is used
    n_off = sum(p.numel() for p in model.parameters())
    n_on = sum(p.numel() for p in model_ctx.parameters())
    check("参数规模随流数变化（无上下文更小）", n_on > n_off, f"{n_off} / {n_on}")

    # --- the context routing: the decision head may be withheld the context
    M.AutoModel = FakeHF
    torch.manual_seed(11)
    routed = MSAModel(text_encoder="stub", d_model=d_model, n_layers=1, n_heads=2,
                      d_audio=5, d_vision=20, binary_head=True, use_context=True,
                      route_context="reg")
    check("路由后判别头的宽度不含上下文流",
          routed.binary_head.in_features == 2 * d_model,
          routed.binary_head.in_features)
    cb = dict(cbatch)
    with torch.no_grad():
        pred_r, logit_r = routed(cb)
    check("路由后仍能出预测与判别分数",
          tuple(pred_r.shape) == (B,) and tuple(logit_r.shape) == (B,))
    # the regression head must still see the context.  The intervention is
    # *zeroing* a stream, not removing it: the head's width is fixed at
    # construction, so a removed stream would not fit.
    with torch.no_grad():
        z_r = routed._streams(cb)
        pred_full, _ = routed._predict(z_r[0], z_r[1], z_r[2])
        pred_noctx, _ = routed._predict(z_r[0], z_r[1], torch.zeros_like(z_r[2]))
    check("路由不影响回归头读上下文",
          not torch.allclose(pred_full, pred_noctx, atol=1e-6))
    with torch.no_grad():
        # _predict returns (prediction, decision logit)
        _, logit_ctx = routed._predict(z_r[0], z_r[1], z_r[2])
        _, logit_noctx = routed._predict(z_r[0], z_r[1], torch.zeros_like(z_r[2]))
    check("路由后判别头读不到上下文（两次读数相同）",
          torch.allclose(logit_ctx, logit_noctx, atol=1e-6))

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' CHECK(S) FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
