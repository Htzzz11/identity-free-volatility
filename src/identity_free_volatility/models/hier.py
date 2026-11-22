"""Hierarchical transcript model, standard training (no identity removal).

Frozen chunk vectors (models/chunks.py) -> linear projection + sinusoidal
positions -> 2-layer transformer over the chunks of one call -> masked mean
-> call vector -> linear head for the four horizons.

Standard training means the raw log-volatility target, no residual target, no
differencing and no adversary. This is the reference that M1 to M3 are compared
against. Targets are z-scored per horizon with training statistics only.
One model per test quarter, early stopping on val MSE.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data import LABELS, Corpus


class HierModel(nn.Module):
    def __init__(self, dim_in: int, d_model: int = 256, layers: int = 2, heads: int = 4,
                 dropout: float = 0.1, n_out: int = 4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim_in, d_model), nn.LayerNorm(d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_out)

    @staticmethod
    def positions(length: int, dim: int, device) -> torch.Tensor:
        pos = torch.arange(length, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe

    def call_vector(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """x: (B, L, dim_in), pad: (B, L) True where padded."""
        h = self.proj(x)
        h = h + self.positions(h.shape[1], h.shape[2], h.device)
        h = self.norm(self.encoder(h, src_key_padding_mask=pad))
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        return (h * keep).sum(1) / keep.sum(1)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.call_vector(x, pad)))


def pad_chunks(chunks: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    longest = max(len(c) for c in chunks)
    x = torch.zeros(len(chunks), longest, chunks[0].shape[1])
    pad = torch.ones(len(chunks), longest, dtype=torch.bool)
    for i, c in enumerate(chunks):
        x[i, :len(c)] = torch.from_numpy(c)
        pad[i, :len(c)] = False
    return x, pad


def fit(x, pad, y, tr, va, seed: int, *, lr=3e-4, weight_decay=0.01, batch=16,
        epochs=30, patience=5) -> tuple[HierModel, torch.Tensor, torch.Tensor]:
    """Train on rows tr, early-stop on rows va. Returns model and target mean/std."""
    torch.manual_seed(seed)
    mu, sd = y[tr].mean(0), y[tr].std(0).clamp_min(1e-6)
    z = (y - mu) / sd
    model = HierModel(x.shape[2]).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    tr_idx, va_idx = tr.nonzero().squeeze(1), va.nonzero().squeeze(1)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=tr_idx.device)]
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(x[idx], pad[idx]), z[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = _predict(model, x, pad, va_idx, loss_target=z)
        if val < best - 1e-4:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), mu, sd


def _predict(model, x, pad, idx, loss_target=None, vectors=False, batch=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            j = idx[i:i + batch]
            out.append(model.call_vector(x[j], pad[j]) if vectors else model(x[j], pad[j]))
    out = torch.cat(out)
    if loss_target is not None:
        return nn.functional.mse_loss(out, loss_target[idx]).item()
    return out


def run(corpus: Corpus, chunks: list[np.ndarray], seed: int, device: torch.device,
        tests: list[str] | None = None, log=print) -> tuple[pd.DataFrame, np.ndarray]:
    """Rolling-by-quarter predictions, plus the call vectors of every call from
    the model of the last test quarter (for post-hoc probes)."""
    x, pad = (t.to(device) for t in pad_chunks(chunks))
    y = torch.as_tensor(corpus.y, dtype=torch.float32, device=device)
    out, model = [], None
    for test in tests or list(corpus.splits):
        tr, va, te = (torch.as_tensor(m, device=device) for m in corpus.masks(test))
        model, mu, sd = fit(x, pad, y, tr, va, seed)
        pred = (_predict(model, x, pad, te.nonzero().squeeze(1)) * sd + mu).cpu().numpy()
        te = te.cpu().numpy()
        mse = ((pred - corpus.y[te]) ** 2).mean(axis=0)
        log(f"  {test} n_train={int(tr.sum())} mse={np.round(mse, 3)}")
        out.append(pd.DataFrame(pred, columns=LABELS).assign(
            call_id=corpus.frame.loc[te, "call_id"].to_numpy(), test=test))
    everyone = torch.arange(len(y), device=device)
    vectors = _predict(model, x, pad, everyone, vectors=True).cpu().numpy()
    return pd.concat(out, ignore_index=True), vectors
