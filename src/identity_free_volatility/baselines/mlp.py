"""Frozen-embedding MLP, a compact port of Yu et al.'s TMLP.

Model and schedule follow github.com/piqueyd/Same-Company-Same-Signal (MIT):
residual MLP (dim -> 512 -> dim, dropout 0.1) + linear head, one model per
horizon, Adam lr 1e-3 halved every epoch after the first, batch 16, up to 10
epochs, early stopping on val MSE with patience 3 (min delta 1e-4).

Random(Ticker): every call of a ticker shares one random vector (identity only).
Random(All):    every call gets its own random vector (no identity, no text).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data import LABELS, Corpus


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TMLP(nn.Module):
    def __init__(self, dim: int, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.temporal = nn.Sequential(nn.Linear(dim, d_model), nn.ReLU(),
                                      nn.Linear(d_model, dim), nn.Dropout(dropout))
        self.projection = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x + self.temporal(x)).squeeze(-1)


def fit_predict(x_tr, y_tr, x_va, y_va, x_te, seed: int, *, lr=1e-3, batch=16,
                epochs=10, patience=3, min_delta=1e-4) -> np.ndarray:
    torch.manual_seed(seed)
    model = TMLP(x_tr.shape[1]).to(x_tr.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr)).to(x_tr.device)
        for i in range(0, len(x_tr) - batch + 1, batch):  # drop_last, as upstream
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(x_va), y_va).item()
        if best_state is None or val <= best - min_delta:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        for group in opt.param_groups:  # upstream "type1": lr0 * 0.5 ** epoch
            group["lr"] = lr * 0.5 ** epoch
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x_te).cpu().numpy()


def run(corpus: Corpus, emb: np.ndarray, seed: int, log=print) -> pd.DataFrame:
    """Rolling-by-quarter predictions for all horizons from fixed call vectors."""
    device = pick_device()
    x = torch.as_tensor(emb, dtype=torch.float32, device=device)
    y = torch.as_tensor(corpus.y, dtype=torch.float32, device=device)
    out = []
    for test in corpus.splits:
        tr, va, te = (torch.as_tensor(m, device=device) for m in corpus.masks(test))
        pred = np.column_stack([
            fit_predict(x[tr], y[tr, j], x[va], y[va, j], x[te], seed) for j in range(len(LABELS))])
        te = te.cpu().numpy()
        mse = ((pred - corpus.y[te]) ** 2).mean(axis=0)
        log(f"  {test} n_train={int(tr.sum())} mse={np.round(mse, 3)}")
        out.append(pd.DataFrame(pred, columns=LABELS).assign(
            call_id=corpus.frame.loc[te, "call_id"].to_numpy(), test=test))
    return pd.concat(out, ignore_index=True)


def random_ticker(corpus: Corpus, seed: int, dim: int = 3072) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes, uniques = pd.factorize(corpus.frame["ticker"])
    return rng.random((len(uniques), dim), dtype=np.float32)[codes]


def random_all(corpus: Corpus, seed: int, dim: int = 3072) -> np.ndarray:
    return np.random.default_rng(seed).random((len(corpus.frame), dim), dtype=np.float32)
