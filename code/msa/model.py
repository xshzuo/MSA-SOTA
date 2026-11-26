"""The MSA model.

A deliberately small, readable architecture:

    text (HF encoder)  ──► H_t (B, L, d)
    audio (B,50,5)     ──► Linear+LN+Transformer ──► H_a (B, 50, d)
    vision (B,50,20)   ──► Linear+LN+Transformer ──► H_v (B, 50, d)

    H_ta = CrossAttn(Q=H_t, K=V=H_a)          # text asks the acoustic stream
    H_tv = CrossAttn(Q=H_t, K=V=H_v)          # text asks the visual stream

    z = [ mean(H_ta) ; mean(H_tv) ]           # (B, 2d)
    ŷ = MLP(z)

The two cross streams are kept separate (not summed) because component ASF
needs to weight them per sample; the weights default to 1 and are supplied by
`forward(..., stream_weights=...)`.

No parameter of the counterfactual / non-verbal branch is ever separate from
the factual one: the model has a single set of weights for each modality, so a
difference between two readings is attributable to the intervention alone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class SeqEncoder(nn.Module):
    """Audio or visual stream: project, normalise, then a small transformer."""

    def __init__(self, d_in, d_model, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.ln = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, norm_first=False)
        self.enc = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x):
        return self.enc(self.ln(self.proj(x)))


class TextEncoder(nn.Module):
    """A pretrained language encoder plus a projection into the shared space."""

    def __init__(self, name, d_model, dropout=0.1, cache_dir=None):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
        # several published checkpoints (deberta-v3-large among them) ship fp16
        # weights; loading them as-is makes the forward pass fail with
        # "expected scalar type Half but found Float"
        self.model = AutoModel.from_pretrained(name, config=cfg, cache_dir=cache_dir,
                                               torch_dtype=torch.float32)
        self.ln = nn.LayerNorm(self.model.config.hidden_size)
        self.proj = nn.Linear(self.model.config.hidden_size, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, **kw):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kw)
        h = out.last_hidden_state                      # (B, L, d_hf)
        return self.drop(self.proj(self.ln(h))), attention_mask


class CrossAttention(nn.Module):
    """The text stream queries one non-verbal stream."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))

    def forward(self, q, kv, kv_mask=None):
        o, _ = self.attn(q, kv, kv, key_padding_mask=kv_mask)
        q = self.ln1(q + o)
        return self.ln2(q + self.ff(q))


class MSAModel(nn.Module):
    def __init__(self, text_encoder="bert-base-uncased", d_model=64, n_layers=2,
                 n_heads=4, dropout=0.1, text_dropout=0.1, d_audio=None, d_vision=None,
                 cache_dir=None, binary_head=False, use_context=False,
                 route_context="both"):
        # the stream widths differ between the corpora (CMU-MOSI: 5 acoustic /
        # 20 visual, CMU-MOSEI: 74 / 35), so they are read from the data
        # instead of being fixed here
        super().__init__()
        self.text = TextEncoder(text_encoder, d_model, text_dropout, cache_dir)
        self.audio = SeqEncoder(d_audio, d_model, n_layers, n_heads, dropout)
        self.vision = SeqEncoder(d_vision, d_model, n_layers, n_heads, dropout)
        self.d_audio, self.d_vision = d_audio, d_vision
        self.cross_a = CrossAttention(d_model, n_heads, dropout)
        self.cross_v = CrossAttention(d_model, n_heads, dropout)
        # the optional context stream: the neighbouring utterance, read with
        # the *same* encoders (no extra parameters) and gated by the same
        # attributable mechanism as the two non-verbal streams
        self.use_context = bool(use_context)
        # the context branch is built only when it is used: an unused module
        # would still consume random draws at construction time, which would
        # change the initialisation of every later module and break the
        # reproducibility of a run
        self.cross_c = (CrossAttention(d_model, n_heads, dropout)
                        if self.use_context else None)
        # the head's width has to match the number of streams that are actually
        # read: two (acoustic, visual) without the context, three with it
        n_streams = 3 if self.use_context else 2
        self.head = nn.Sequential(nn.Linear(n_streams * d_model, d_model), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))
        # the decision head can be routed: the neighbouring utterance informs
        # the intensity but misleads the polarity, so it may be withheld from
        # the decision head while still feeding the regression head
        self.route_context = route_context     # "both" | "reg"
        n_cls = (2 if (self.route_context == "reg" and self.use_context)
                 else n_streams) * d_model
        self.binary_head = nn.Linear(n_cls, 1) if binary_head else None

    def _streams(self, batch):
        """The cross-modal streams and the text mask (shared by every read)."""
        kw = {}
        if "token_type_ids" in batch:
            kw["token_type_ids"] = batch["token_type_ids"]
        h_t, tmask = self.text(batch["input_ids"], batch["attention_mask"], **kw)
        h_a = self.audio(batch["audio"])
        h_v = self.vision(batch["vision"])
        w = tmask.unsqueeze(-1).to(h_t.dtype)
        denom = w.sum(dim=1).clamp(min=1.0)
        z_ta = (self.cross_a(h_t, h_a) * w).sum(dim=1) / denom
        z_tv = (self.cross_v(h_t, h_v) * w).sum(dim=1) / denom
        if self.use_context and self.cross_c is not None:
            ckw = {}
            if "ctx_token_type_ids" in batch:
                ckw["token_type_ids"] = batch["ctx_token_type_ids"]
            h_c, _ = self.text(batch["ctx_input_ids"], batch["ctx_attention_mask"], **ckw)
            h_ca = self.audio(batch["ctx_audio"])
            z_tc = (self.cross_c(h_t, h_ca) * w).sum(dim=1) / denom
        else:
            z_tc = None
        return z_ta, z_tv, z_tc

    def _predict(self, z_ta, z_tv, z_tc=None):
        parts = [z_ta, z_tv] + ([z_tc] if z_tc is not None else [])
        z = torch.cat(parts, dim=-1)
        pred = self.head(z).squeeze(-1)
        logits = None
        if self.binary_head is not None:
            if self.route_context == "reg" and z_tc is not None:
                z_cls = torch.cat([z_ta, z_tv], dim=-1)
            else:
                z_cls = z
            logits = self.binary_head(z_cls).squeeze(-1)
        return pred, logits

    def forward_with_effects(self, batch):
        """Read the model twice per modality and measure the attributable effect.

        The intervention removes one non-verbal stream (its cross-attention
        input is zeroed) and *nothing else changes*: the language encoder, the
        cross-attention weights and the fusion head are the same parameters in
        both readings.  The difference in the prediction is therefore
        attributable to that modality's contribution, not to a parameter
        difference between two branches.
        """
        z_ta, z_tv, z_tc = self._streams(batch)
        pred, logits = self._predict(z_ta, z_tv, z_tc)

        zero = torch.zeros_like(z_ta)
        pred_a, _ = self._predict(zero, z_tv, z_tc)      # the acoustic stream removed
        pred_v, _ = self._predict(z_ta, zero, z_tc)      # the visual stream removed
        eff_a = (pred - pred_a).abs()
        eff_v = (pred - pred_v).abs()
        denom = torch.maximum(eff_a, eff_v).clamp(min=1e-6)
        w_a, w_v = eff_a / denom, eff_v / denom
        if z_tc is not None:
            pred_c, _ = self._predict(z_ta, z_tv, zero)  # the context stream removed
            eff_c = (pred - pred_c).abs()
            denom_c = torch.maximum(torch.maximum(eff_a, eff_v), eff_c).clamp(min=1e-6)
            w_a, w_v, w_c = eff_a / denom_c, eff_v / denom_c, eff_c / denom_c
            return pred, logits, eff_a, eff_v, eff_c, w_a, w_v, w_c, z_ta, z_tv, z_tc
        # the streams are returned so that the gated read reuses them instead
        # of running the language encoder a second time
        return pred, logits, eff_a, eff_v, w_a, w_v, z_ta, z_tv

    def forward(self, batch, stream_weights=None, return_streams=False):
        # the padded text positions still carry non-zero hidden states, so the
        # pooling is mask-aware (inside _streams).
        z_ta, z_tv, z_tc = self._streams(batch)

        if stream_weights is not None:
            wa, wv = stream_weights
            z_ta = z_ta * wa.unsqueeze(-1)
            z_tv = z_tv * wv.unsqueeze(-1)

        # the reading goes through _predict so that the routing of the
        # decision head applies here as well
        pred, logits = self._predict(z_ta, z_tv, z_tc)
        if return_streams:
            return pred, logits, z_ta, z_tv, z_tc
        return pred, logits
